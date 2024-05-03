import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.max_size = size
        self.idx = 0

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size):
        batch_indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in batch_indices]

        # Ensure all arrays have consistent shapes
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array([np.array(state) for state in states])
        actions = np.array([np.array(action) for action in actions])
        rewards = np.array(rewards)
        next_states = np.array([np.array(next_state) for next_state in next_states])
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        self.alpha = alpha
        self.gamma = 0.99
        self.tau = 0.005
        self.target_entropy = -action_dim
        
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state):
        # Extract the actual state array if it's a tuple
        if isinstance(state, tuple):
            state = state[0]

        # Ensure state is a numpy array
        state = np.array(state)
        
        # Convert to torch tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _ = self.policy.sample(state)
        return action.squeeze(0).detach().cpu().numpy()
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_target = self.target_q1(next_states, next_actions)
            q2_target = self.target_q2(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_target
        
        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        actions, log_probs = self.policy.sample(states)
        q1_pi = self.q1(states, actions)
        q2_pi = self.q2(states, actions)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_probs - q_pi).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        alpha_loss = -self.log_alpha * (log_probs + self.target_entropy).detach().mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Training Loop
def train(env, agent, num_episodes, batch_size, buffer_size):
    replay_buffer = ReplayBuffer(buffer_size)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            replay_buffer.add((state, action, reward, next_state, done))
                        

            if len(replay_buffer.buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode}, Reward: {episode_reward}")


env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = SACAgent(state_dim, action_dim)

train(env, agent, num_episodes=500, batch_size=256, buffer_size=100000)
