from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import math

from torchvision import models

class LayerNorm(nn.Module):
   def __init__(self, hidden_size, eps=1e-12):
       """Construct a layernorm module in the TF style (epsilon inside the square root).
       """
       super(LayerNorm, self).__init__()
       self.weight = nn.Parameter(torch.ones(hidden_size))
       self.bias = nn.Parameter(torch.zeros(hidden_size))
       self.variance_epsilon = eps

   def forward(self, x):
       u = x.mean(-1, keepdim=True)
       s = (x - u).pow(2).mean(-1, keepdim=True)
       x = (x - u) / torch.sqrt(s + self.variance_epsilon)
       return self.weight * x + self.bias

class clinNet(nn.Module):
   def __init__(self, in_size ,dropout):
       super(clinNet, self).__init__()
       self.features = nn.Sequential(
           nn.Conv1d(in_channels=1, out_channels=25, kernel_size=8, stride=1, padding=1, padding_mode='zeros'),
           nn.ReLU(inplace=True),
           nn.MaxPool1d(kernel_size=2, stride=2),
           nn.Conv1d(in_channels=25, out_channels=5, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
           nn.ReLU(inplace=True),
           #nn.MaxPool1d(kernel_size=2, stride=2),
           nn.AvgPool1d(kernel_size=2, stride=2),
           nn.Dropout(dropout)
       )
   def forward(self, x):
       y_1 = self.features(x)
       return y_1


class cnaNet(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, dropout):
        super(cnaNet, self).__init__()
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_size, hidden_size)  
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        x = self.dropout(x) 
        x = self.fc(x)  
        return x


class genNet(nn.Module):
   def __init__(self, in_size ,dropout):
       super(genNet, self).__init__()
       self.features = nn.Sequential(
           nn.Conv1d(in_channels=1, out_channels=25, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
           nn.ReLU(inplace=True),
           nn.MaxPool1d(kernel_size=2, stride=2),
           nn.Conv1d(in_channels=25, out_channels=5, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
           nn.ReLU(inplace=True),
           nn.MaxPool1d(kernel_size=2, stride=2),
           nn.Conv1d(in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
           nn.ReLU(inplace=True),
           nn.MaxPool1d(kernel_size=2, stride=2),
           nn.Flatten(),
           nn.Dropout(dropout)
       )
   def forward(self, x):
       y_1 = self.features(x)
       return y_1


class AttentionFusionLayer(nn.Module):
    def __init__(self, s1, s2, s3, so, dm, nh, nel):
        super().__init__()
        self.h = [nn.Linear(s1, dm), nn.Linear(s2, dm), nn.Linear(s3, dm)]
        self.e = [nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nh), nel) for _ in range(3)]
        self.g = [nn.Sequential(nn.Linear(dm, dm), nn.Sigmoid()) for _ in range(3)]
        self.aw = nn.Linear(3 * dm, 3)
        self.sm = nn.Softmax(dim=1)
        self.o = nn.Linear(dm, so)

    def forward(self, a1, a2, a3):

        buf = []
        for i, a in enumerate([a1, a2, a3]):
            t = self.h[i](a*1)
            t = self.e[i](t.unsqueeze(1)).squeeze(1)
            t = self.g[i](t) * t 
            buf.append(t)
      
        atw = self.sm(self.aw(torch.cat(buf, dim=1) + 1e-9))

        z = sum(atw[:, j].unsqueeze(1) * buf[j] for j in range(3))
        return self.o(z)


class OptimizedResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(OptimizedResidualMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout

    def forward(self, x):
        x = self.input_layer(x)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norm_layers)):
            residual = x
            out = layer(x)
            out = torch.relu(out)  
            out = norm(out)
            out = self.dropout(out)
            x = out + residual  
        x = self.output_layer(x)
        return torch.sigmoid(x)

class MLPGenreClassifierModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPGenreClassifierModel, self).__init__()
        self.residual_mlp = OptimizedResidualMLP(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.residual_mlp(x)


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)

class RandomForestClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomForestClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Linear(50, output_dim)
        )

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)


class Ensemble(nn.Module):
    def __init__(self, a, b, c, d=1):
        super().__init__()
        self.m1 = LogisticRegressionClassifier(a, c)
        self.m2 = RandomForestClassifier(a, c)
        self.m3 = OptimizedResidualMLP(a, b, c)
        self.w = nn.Parameter(torch.ones(3))
        self.ff = d
        self.df = nn.Parameter(torch.ones(1))
        self.fc = nn.Linear(c, c)
    def forward(self, x):
        x = torch.stack([self.m1(x)*1 , 
                         self.m2(x) + 1e-8 - 1e-8, 
                         self.m3(x).clone()], 0)
        x = torch.sum(
            ( (self.ff + torch.sigmoid(self.df )) *
                F.softmax(self.w.view(-1, 1).squeeze(), dim=0).view(3, 1, 1)
            ) *
            torch.sort(x, dim=0, descending=True)[0],
            dim=0
        )
        return self.fc(x)

class Net(nn.Module):
    def __init__(self, in_size, hid_size, nhead_att, drop_prob, drop_main, dim_feat, lstm_layers, dm, nh, enc_layers):
        super(Net, self).__init__()
        g, c, cl = dim_feat  
        self._dims = (g, c, cl)
        self._gc_sum = g + c
        self._hid = hid_size
        self.maps = nn.ModuleDict({
            'm_cl': nn.Linear(cl, cl),
            'm_cna': nn.Linear(c, c),
            'm_gen': nn.Linear(g, g)
        })
        
        # 子网络合并，使用模式标记
        self.subnets = nn.ModuleDict({
            's_cl': clinNet(cl, drop_main),
            's_cna': cnaNet(c, drop_main),
            's_gen': genNet(g, drop_main)
        })

        self.fuse = AttentionFusionLayer(hid_size, hid_size, hid_size, hid_size, dm, nh, enc_layers)
        self.ens = Ensemble(input_dim=hid_size, hidden_dim=hid_size, num_classes=2)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):

        f_cl = self.subnets['s_cl'](x[:, self._gc_sum:])
        f_cl = self.act(self.maps['m_cl'](f_cl))
        f_gen = self.subnets['s_gen'](x[:, :self._dims[0]].unsqueeze(1))
        f_gen = self.drop(f_gen)
        f_cna = self.subnets['s_cna'](x[:, self._dims[0]:self._gc_sum].unsqueeze(1))
        f_cna = torch.tanh(self.maps['m_cna'](f_cna))
        u1, u2, u3 = f_gen, f_cl, f_cna
        merged = self.fuse(u1, u3, u2)
        out = torch.sigmoid(self.ens(merged))
        return out

if __name__=='__main__':
   pass
