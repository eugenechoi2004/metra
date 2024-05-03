import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, learning_rate,state_size,action_size, hidden_layer_size):
        super(Critic).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )
        self.optimizer = optim.Adam(self.parameters, lr= learning_rate)
        self.to(self.device)
    
    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        return self.critic(input)

class Value(nn.Module):
    def __init__(self, learning_rate,state_size, hidden_layer_size):
        super(Value).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.value = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )
        self.optimizer = optim.Adam(self.parameters, lr= learning_rate)
        self.to(self.device)

    def forward(self, state):
        return self.value(state)


class ActorNetwork(nn.Module):
    def __init__(self, learning_rate,state_size, action_size, hidden_layer_size):
        super(Critic).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
        )
        self.optimizer = optim.Adam(self.parameters, lr= learning_rate)
        self.to(self.device)
