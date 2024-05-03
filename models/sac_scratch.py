import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, learning_rate,state_size,action_size, hidden_layer_size):
        super(CriticNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr= learning_rate)
        self.to(self.device)
    
    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        return self.critic(input)

class ValueNetwork(nn.Module):
    def __init__(self, learning_rate,state_size, hidden_layer_size):
        super(ValueNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.value = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr= learning_rate)
        self.to(self.device)

    def forward(self, state):
        return self.value(state)


class ActorNetwork(nn.Module):
    def __init__(self, learning_rate,state_size, action_size, hidden_layer_size, max_action):
        super(ActorNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_layer_size, 1)
        self.sigma = nn.Linear(hidden_layer_size, 1)
        self.optimizer = optim.Adam(self.actor.parameters(), lr= learning_rate)
        self.to(self.device)
        self.reparam_noise = 1e-6
        self.max_action = max_action

    def forward(self, state):
        prob = self.actor(state)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = torch.clamp(min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probs = Normal(mu, sigma)
        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        action = (torch.tanh(actions) * self.max_action).to(self.device)

        log_probs = probs.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
