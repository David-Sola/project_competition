import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_space, action_space, out_fcn=nn.Tanh(), fc1_units=400, fc2_units=200, fc3_units=100):
        '''
        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_space)
        self.fcn = out_fcn
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fcn(self.fc4(x))


class Critic(nn.Module):
    def __init__(self, state_space, action_space, fc1_units=400, fc2_units=200, fc3_units=100):
        '''
        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Critic, self).__init__()

        #Q1 architecture
        self.fc1 = nn.Linear(state_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_space, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_space)

        # Q2 architecture
        self.fc5 = nn.Linear(state_space, fc1_units)
        self.fc6 = nn.Linear(fc1_units + action_space, fc2_units)
        self.fc7 = nn.Linear(fc2_units, fc3_units)
        self.fc8 = nn.Linear(fc3_units, action_space)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc6.weight.data.uniform_(*hidden_init(self.fc6))
        self.fc7.weight.data.uniform_(*hidden_init(self.fc7))

    def forward(self, x, action):
        state_q1 = x
        state_q2 = x
        xs = F.relu(self.fc1(state_q1))
        state_q1 = torch.cat((xs, action), dim=1)
        state_q1 = F.relu(self.fc2(state_q1))
        state_q1 = F.relu(self.fc3(state_q1))
        q1 = self.fc4(state_q1)

        xs = F.relu(self.fc5(state_q2))
        state_q2 = torch.cat((xs, action), dim=1)
        state_q2 = F.relu(self.fc6(state_q2))
        state_q2 = F.relu(self.fc7(state_q2))
        q2 = self.fc8(state_q2)

        return q1, q2

