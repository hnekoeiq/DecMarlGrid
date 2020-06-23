import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import torch_ac
from torch.distributions.categorical import Categorical

import random
import math
from functools import partial


def DQN(state_shape, action_dim, args):
    if args.c51:
        if args.dueling:
            model = CategoricalDuelingDQN(state_shape, action_dim, args.noisy, args.sigma_init,
                                          args.Vmin, args.Vmax, args.num_atoms, args.batch_size, args.memory)
        else:
            model = CategoricalDQN(state_shape, action_dim, args.noisy, args.sigma_init,
                                   args.Vmin, args.Vmax, args.num_atoms, args.batch_size, args.memory)
    else:
        if args.dueling:
            model = DuelingDQN(state_shape, action_dim, args.noisy, args.sigma_init, args.memory)
        else:
            model = DQNBase(state_shape, action_dim, args.noisy, args.sigma_init, args.memory)

    return model


def fanin_init(size, fanin=None):
    """
    weight initializer known from https://arxiv.org/abs/1502.01852
    :param size:
    :param fanin:
    :return:
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inputs_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 16, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x.view(-1, self.inputs_shape[0], self.inputs_shape[1], self.inputs_shape[2]))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros((1,) + self.inputs_shape)).view(1, -1).size(1)


class DQNBase(nn.Module):
    """
    Basic DQN + NoisyNet

    Noisy Networks for Exploration
    https://arxiv.org/abs/1706.10295

    parameters
    ---------
    env         environment(openai gym)
    noisy       boolean value for NoisyNet.
                If this is set to True, self.Linear will be NoisyLinear module
    """

    def __init__(self, state_shape, action_dim,  noisy, sigma_init, memory):
        super(DQNBase, self).__init__()

        self.input_shape = state_shape
        self.num_actions = action_dim
        self.noisy = noisy

        if noisy:
            self.Linear = partial(NoisyLinear, sigma_init=sigma_init)
        else:
            self.Linear = nn.Linear
        self.rnn = nn.GRUCell
        self.flatten = Flatten()

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=5, stride=4),
            nn.Conv2d(state_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.Conv2d(16, 64, kernel_size=4, stride=2),
            # nn.ReLU()
        )
        if memory is True:
            self.fc = nn.Sequential(
                self.Linear(self._feature_size(), 128),
                nn.ReLU(),
                self.rnn(128, 128),
                nn.ReLU(),
                self.Linear(128, self.num_actions))
        else:
            self.fc = nn.Sequential(
                self.Linear(self._feature_size(), 64),
                nn.ReLU(),
                self.Linear(64, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def _feature_size(self):
        return self.features(torch.zeros((1,) + self.input_shape)).view(1, -1).size(1)

    def update_noisy_modules(self):
        if self.noisy:
            self.noisy_modules = [module for module in self.modules() if isinstance(module, NoisyLinear)]

    def sample_noise(self):
        for module in self.noisy_modules:
            module.sample_noise()

    def remove_noise(self):
        for module in self.noisy_modules:
            module.remove_noise()


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """

    def __init__(self, state_shape, action_dim,  noisy, sigma_init, memory):
        super(DuelingDQN, self).__init__(state_shape, action_dim, noisy, sigma_init, memory)

        self.advantage = self.fc

        if memory is True:
            self.value = nn.Sequential(
                self.Linear(self._feature_size(), 128),
                nn.ReLU(),
                self.rnn(128, 128),
                nn.ReLU(),
                self.Linear(128, 1)
            )
        else:
            self.value = nn.Sequential(
                self.Linear(self._feature_size(), 128),
                nn.ReLU(),
                self.Linear(128, 1)
            )

    def forward(self, x):
        x = self.features(x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


class CategoricalDQN(DQNBase):
    """
    A Distributional Perspective on Reinforcement Learning
    https://arxiv.org/abs/1707.06887
    """

    def __init__(self, state_shape, action_dim,  noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size, memory):
        super(CategoricalDQN, self).__init__(state_shape, action_dim,  noisy, sigma_init, memory)

        support = torch.linspace(Vmin, Vmax, num_atoms)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, num_atoms)

        self.register_buffer('support', support)
        self.register_buffer('offset', offset)
        self.num_atoms = num_atoms

        if memory is True:
            self.fc = nn.Sequential(
                self.Linear(self._feature_size(), 512),
                nn.ReLU(),
                self.rnn(512, 512),
                nn.ReLU(),
                self.Linear(512, self.num_actions * self.num_atoms),
            )
        else:
            self.fc = nn.Sequential(
                self.Linear(self._feature_size(), 512),
                nn.ReLU(),
                self.Linear(512, self.num_actions * self.num_atoms),
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x


class CategoricalDuelingDQN(CategoricalDQN):

    def __init__(self, state_shape, action_dim,  noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size, memory):
        super(CategoricalDuelingDQN, self).__init__(state_shape, action_dim, noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size, memory)

        self.advantage = self.fc

        if memory is True:
            self.value = nn.Sequential(
                self.Linear(self._feature_size(), 512),
                nn.ReLU(),
                self.rnn(512, 512),
                nn.ReLU(),
                self.Linear(512, num_atoms)
            )
        else:
            self.value = nn.Sequential(
                self.Linear(self._feature_size(), 512),
                nn.ReLU(),
                self.Linear(512, num_atoms)
            )

    def forward(self, x):
        x = self.features(x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        x = self.flatten(x)

        advantage = self.advantage(x).view(-1, self.num_actions, self.num_atoms)
        value = self.value(x).view(-1, 1, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, self.num_actions, self.num_atoms)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.register_buffer('sample_weight_in', torch.FloatTensor(in_features))
        self.register_buffer('sample_weight_out', torch.FloatTensor(out_features))
        self.register_buffer('sample_bias_out', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.sample_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.bias_sigma.size(0)))

    def sample_noise(self):
        self.sample_weight_in = self._scale_noise(self.sample_weight_in)
        self.sample_weight_out = self._scale_noise(self.sample_weight_out)
        self.sample_bias_out = self._scale_noise(self.sample_bias_out)

        self.weight_epsilon.copy_(self.sample_weight_out.ger(self.sample_weight_in))
        self.bias_epsilon.copy_(self.sample_bias_out)

    def _scale_noise(self, x):
        x = x.normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x
