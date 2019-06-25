from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from models.model import Model
from utils.cuda import Variable
from utils.tools import init_weights, normalized_columns_initializer
from utils.sampler import sampler
from utils.crelu import crelu
import torch
import torch.nn as nn
import torch.nn.functional as F

class A3C_LSTM_HPA3C(Model):
    def __init__(self, args):
        super(A3C_LSTM_HPA3C, self).__init__(args)
        if args.crelu:
            self.crelu = True
        else:
            self.crelu = False
        self.sig  = args.sig
        self.conv1 = nn.Conv2d(3,  32, 5, stride=1, padding=2)
        self.down1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.down2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.down3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.down4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.linear_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_sigma = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.prior_mu = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.lstm = nn.LSTMCell(1024, self.hidden_dim)
        self.linear_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)


        self.policy_5 = nn.Linear(self.hidden_dim * 2, self.output_dims)
        self.policy_6 = nn.Softmax(dim=1)
        self.value_5  = nn.Linear(self.hidden_dim * 2, 1)       
        self.sampler = sampler(args)

        if self.crelu:
            self.crelu_z = crelu()
            self.crelu_x = crelu()
            self.prior_mu = nn.Linear(self.hidden_dim*2, self.hidden_dim)
            self.policy_5 = nn.Linear(self.hidden_dim * 4, self.output_dims)
            self.value_5  = nn.Linear(self.hidden_dim * 4, 1)

        self._reset()
        self.train()

    def _init_weights(self):
        self.apply(init_weights)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)

        self.linear_mu.bias.data.fill_(0)
        self.linear_sigma.bias.data.fill_(0)
        self.prior_mu.bias.data.fill_(0)
        self.linear_encoder.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, z_prev, lstm_hidden_vb=None, eps=None):
        x = x.view(x.size(0), 3, self.input_dims[1], self.input_dims[1])
        x = F.leaky_relu(self.down1(F.leaky_relu(self.conv1(x))),inplace=True)
        x = F.leaky_relu(self.down2(F.leaky_relu(self.conv2(x))),inplace=True)
        x = F.leaky_relu(self.down3(F.leaky_relu(self.conv3(x))),inplace=True)
        x = F.leaky_relu(self.down4(F.leaky_relu(self.conv4(x))),inplace=True)
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, lstm_hidden_vb)

        if self.crelu:
            z_prev = self.crelu_z(z_prev)
        else:
            z_prev = F.leaky_relu(z_prev,inplace=True)

        mu_prior = self.prior_mu(z_prev)

        sigma_prior = Variable(torch.zeros(mu_prior.size(0), mu_prior.size(1))-self.sig)
        
        x = self.linear_encoder(hx)
        mu = self.linear_mu(hx)
        sigma = self.linear_sigma(hx)
        self.x = Variable(x.data)
        z = self.sampler(mu, sigma, eps=eps)

        if self.crelu:
            x = self.crelu_x(torch.cat([x,z], dim=1))
        else:
            x = F.leaky_relu(torch.cat([x,z], dim=1),inplace=True)

        p = self.policy_5(x)
        p = self.policy_6(p)
        v = self.value_5(x)

        return p, v, z, (hx, cx), (mu, sigma), (mu_prior, sigma_prior)


class A3C_LSTM_HPA3C_CRELU(Model):
    def __init__(self, args):
        super(A3C_LSTM_S7P_CRELU, self).__init__(args)
        self.sig   = args.sig
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

        self.linear_mu = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.linear_sigma = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.prior_mu  = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
        self.lstm = nn.LSTMCell(1024, self.hidden_dim)
        self.linear_encoder = nn.Linear(self.hidden_dim, int(self.hidden_dim/2))
   
        self.crelu_z = crelu()
        self.crelu_encoder = crelu()
        self.sampler = sampler(args)
        
        self.policy_5 = nn.Linear(self.hidden_dim * 2, self.output_dims)
        self.policy_6 = nn.Softmax()
        self.value_5  = nn.Linear(self.hidden_dim * 2, 1)

        self._reset()
        self.train()

    def _init_weights(self):
        self.apply(init_weights)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)

        self.linear_mu.bias.data.fill_(0)
        self.linear_sigma.bias.data.fill_(0)
        self.prior_mu.bias.data.fill_(0)
        self.linear_encoder.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)


    def forward(self, x, z_prev, lstm_hidden_vb=None, eps=None):
        x = x.view(x.size(0), 3, self.input_dims[1], self.input_dims[1])
        x = F.leaky_relu(self.down1(self.crelu1(self.conv1(x))),inplace=True)
        x = F.leaky_relu(self.down2(self.crelu2(self.conv2(x))),inplace=True)
        x = F.leaky_relu(self.down3(self.crelu3(self.conv3(x))),inplace=True)
        x = F.leaky_relu(self.down4(self.crelu4(self.conv4(x))),inplace=True)
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, lstm_hidden_vb)

        x = self.linear_encoder(hx)
        mu = self.linear_mu(hx)
        sigma = self.linear_sigma(hx)
        z = self.sampler(mu, sigma, eps=eps)
        mu_prior = self.prior_mu(z_prev)
        sigma_prior = Variable(torch.zeros(mu.size(0), mu.size(1))-self.sig)
        x = self.crelu_encoder(torch.cat([x,z], dim=1))
        p = self.policy_5(x)
        p = self.policy_6(p)
        v = self.value_5(x)
        z = self.crelu_z(z)
        return p, v, z, (hx, cx), (mu, sigma), (mu_prior, sigma_prior)

