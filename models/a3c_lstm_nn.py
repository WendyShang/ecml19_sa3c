from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.model import Model
from utils.tools import init_weights, normalized_columns_initializer
from utils.sampler import sampler
from utils.crelu import crelu
from utils.NoisyLinear import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class A3C_LSTM_NN(Model):
    def __init__(self, args):
        super(A3C_LSTM_NN, self).__init__(args)
        self.conv1 = nn.Conv2d(3,  32, 5, stride=1, padding=2)
        self.down1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.down2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.down3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.down4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(1024, self.hidden_dim)
        self.linear_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn_x = nn.BatchNorm1d(self.hidden_dim*2)

        self.policy_5 = NoisyLinear(self.hidden_dim * 2, self.output_dims)
        self.policy_6 = nn.Softmax(dim=1)
        self.value_5  = NoisyLinear(self.hidden_dim * 2, 1)
        self.sampler = sampler(args)

        self._reset()
        self.train()

    def _init_weights(self):
        self.apply(init_weights)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)
        self.linear_mu.bias.data.fill_(0)
        self.linear_encoder.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), 3, self.input_dims[1], self.input_dims[1])
        x = F.leaky_relu(self.down1(F.leaky_relu(self.conv1(x))))
        x = F.leaky_relu(self.down2(F.leaky_relu(self.conv2(x))))
        x = F.leaky_relu(self.down3(F.leaky_relu(self.conv3(x))))
        x = F.leaky_relu(self.down4(F.leaky_relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, lstm_hidden_vb)
        x = self.linear_encoder(hx)
        z = self.linear_mu(hx)
        x = F.leaky_relu(self.bn_x(torch.cat([x,z], dim=1)))
        p = self.policy_5(x)
        p = self.policy_6(p)
        v = self.value_5(x)
        return p, v, (hx, cx)

    def sample_noise(self):
        self.policy_5.sample_noise()
        self.value_5.sample_noise()

    def remove_noise(self):
        self.policy_5.remove_noise()
        self.value_5.remove_noise()


class A3C_LSTM_NN_CRELU(Model):
    def __init__(self, args):
        super(A3C_LSTM_NN_CRELU, self).__init__(args)
        print("USING NEW MODEL with CRELU")
        self.conv1 = nn.Conv2d(3,  16, 5, stride=1, padding=2)
        self.down1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.crelu1 = crelu()

        self.conv2 = nn.Conv2d(32, 16, 5, stride=1, padding=2)
        self.down2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.crelu2 = crelu()

        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.down3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.crelu3 = crelu()

        self.conv4 = nn.Conv2d(64, 32, 3, stride=1, padding=0)
        self.down4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.crelu4 = crelu()

        self.lstm = nn.LSTMCell(1024, self.hidden_dim)
        self.linear_encoder = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.linear_mu = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.crelu = crelu()
        self.crelu_encoder = crelu()

        self.policy_5 = NoisyLinear(self.hidden_dim * 2, self.output_dims)
        self.policy_6 = nn.Softmax()
        self.value_5  = NoisyLinear(self.hidden_dim * 2, 1)
        self.sampler = sampler(args)
        
        self._reset()
        self.train()

    def _init_weights(self):
        self.apply(init_weights)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)
        self.linear_mu.bias.data.fill_(0)
        self.linear_encoder.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), 3, self.input_dims[1], self.input_dims[1])
        x = F.leaky_relu(self.down1(self.crelu1(self.conv1(x))))
        x = F.leaky_relu(self.down2(self.crelu2(self.conv2(x))))
        x = F.leaky_relu(self.down3(self.crelu3(self.conv3(x))))
        x = F.leaky_relu(self.down4(self.crelu4(self.conv4(x))))
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, lstm_hidden_vb)
        x = self.linear_encoder(hx)
        z = self.linear_mu(hx)
        x = self.crelu_encoder(torch.cat([x,z], dim=1))
        p = self.policy_5(x)
        p = self.policy_6(p)
        v = self.value_5(x)
        return p, v, (hx, cx)

    def sample_noise(self):
        self.policy_5.sample_noise()
        self.value_5.sample_noise()

    def remove_noise(self):
        self.policy_5.remove_noise()
        self.value_5.remove_noise()

