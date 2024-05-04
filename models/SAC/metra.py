from torch.distributions import Uniform
import torch
import gym
from sac_torch import Agent
from networks import Phi
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Metra():
    def __init__(self, **kwargs):
        self.z_dist = Uniform(low=kwargs['latent_low'], high=kwargs['latent_high'])
        self.latent_dim = kwargs['latent_dim']
        self.n_epochs = kwargs['n_epochs']
        self.batch_size = kwargs['batch_size']
        self.env_name = kwargs['env_name']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = gym.make(self.env_name, exclude_current_positions_from_observation=False)
        self.agent = Agent(input_dims=self.env.observation_space.shape, env=self.env,
                           n_actions=self.env.action_space.shape[0])
        self.lr = kwargs['lr']
        self.state_dim = self.env.observation_space.shape[0]
        self.phi = Phi(self.state_dim, self.latent_dim, self.lr).to(self.device)
        self.lamb = torch.tensor(kwargs['lamb'], requires_grad=True).to(self.device)
        self.epsilon = kwargs['epsilon']
        self.grad_steps_per_epoch = kwargs['grad_steps_per_epoch']
        self.minibatch_size = kwargs['minibatch_size']

        self.checkpoint_epoch = kwargs['checkpoint_epoch']
        # Optimizers
        self.lambda_optimizer = optim.Adam([self.lamb], lr=self.lr)

    def sample_skill(self):
        skill_sample = self.z_dist.sample((self.latent_dim,))
        return skill_sample.to(self.device)

    def train(self):
        for epoch in range(self.n_epochs):
            phi_losses = []
            lambda_losses = []
            done = False
            observation = self.env.reset()
            z = self.sample_skill()
            while not done:
                # self.env.render(mode='human')
                action = self.agent.choose_action(observation, z)
                observation_, reward, done, _ = self.env.step(action)
                observation_ = torch.tensor(observation_, dtype=torch.float).to(self.device)

                self.agent.remember(observation, action, reward, observation_, done, z)
            
            for i in range(self.grad_steps_per_epoch):
                if self.agent.memory.mem_cntr < self.batch_size:
                    continue
                states, actions, _, next_states, dones, skills = self.agent.memory.sample_buffer(self.minibatch_size)

                dones = torch.tensor(dones).to(self.device)
                next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
                states = torch.tensor(states, dtype=torch.float).to(self.device)
                actions = torch.tensor(actions, dtype=torch.float).to(self.device)
                skills = torch.tensor(skills, dtype=torch.float).to(self.device)

                phi_loss_value = self.phi_loss(states, next_states, skills, self.epsilon)
                self.phi.phi_optimizer.zero_grad()
                phi_loss_value.backward()
                self.phi.phi_optimizer.step()
                phi_losses.append(phi_loss_value.item())

                lambda_loss_value = self.lambda_loss_fn(states, next_states, self.lamb, self.epsilon)
                self.lambda_optimizer.zero_grad()
                lambda_loss_value.backward()
                self.lambda_optimizer.step()
                lambda_losses.append(lambda_loss_value.item())

                #calculate rewards 
                rewards = self.reward(states, next_states, skills)
                self.agent.learn(rewards, dones, next_states, states, actions, skills)
            

            avg_phi_loss = np.mean(phi_losses)
            avg_lambda_loss = np.mean(lambda_losses)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, "
                    f"Avg Phi Loss: {avg_phi_loss:.4f}, Avg Lambda Loss: {avg_lambda_loss:.4f}")
        
            if epoch % self.checkpoint_epoch == 0:
                self.agent.save_models()
                self.phi.save_checkpoint()
                with open("tmp/sac/lambda_value.txt", "w") as file:
                    file.write(str(self.lamb))

    def run_loaded_model(self, lamb):
        self.agent.load_models()
        self.phi.load_checkpoint()
        self.lamb = lamb
        observation = self.env.reset()
        z = self.sample_skill()
        z[0] = -1
        z[1] = 1
        done = False
        while not done:
            self.env.render(mode='human')
            action = self.agent.choose_action(observation, z)
            observation_, reward, done, info = self.env.step(action)
            
            observation_ = torch.tensor(observation_, dtype=torch.float).to(self.device)

    def reward(self, s, s_prime, z):
        diff = self.phi(s_prime) - self.phi(s)
        return torch.einsum('ij,ij->i', diff, z)

    def phi_loss(self, s, s_prime, z, epsilon):
        diff = self.phi(s_prime) - self.phi(s)
        term1 = torch.einsum('ij,ij->i', diff, z).mean()
        norm_diff = torch.norm(diff, dim=1)
        penalty = torch.clamp(1 - norm_diff ** 2, min=epsilon)
        term2 = self.lamb * penalty.mean()
        return -(term1 + term2)
    
    def lambda_loss_fn(self, s, s_prime, lambda_param, epsilon):
        norm_diff = torch.norm(self.phi(s_prime) - self.phi(s))**2
        penalty_term = min(epsilon, 1 - norm_diff)
        return lambda_param * penalty_term
    
    def locomotion_metric(self, n_skills=48):
        trajectories = []
        self.agent.load_models()
        self.phi.load_checkpoint()
        for i in range(n_skills):
            print(i)
            trajectory = []
            skill = self.sample_skill()
            observation = self.env.reset()
            done = False
            while not done:
                action = self.agent.choose_action(observation, skill)
                observation_, reward, done, info = self.env.step(action)
                if 'x_position' in info and 'y_position' in info:
                    x = torch.clamp(torch.tensor(info['x_position']), min=-50, max=50).item() + 25
                    y = torch.clamp(torch.tensor(info['y_position']), min=-50, max=50).item() + 25
                    trajectory.append((y,x))
                else:
                    print("Info dictionary missing position data")
                    done = True 
            trajectories.append(trajectory)
        return trajectories

    def plot_trajectories(self, trajectories):
        plt.figure(figsize=(10, 8))
        cmap = cm.get_cmap('tab10', len(trajectories))  
        for i, trajectory in enumerate(trajectories):
            if trajectory:
                y_coords, x_coords = zip(*trajectory)
                plt.plot(x_coords, y_coords, color=cmap(i), linestyle='-', linewidth=0.5) 
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Locomotion Trajectories')
        plt.grid(True)
        plt.show()


    def calculate_skill(self, state, goal_state):
        return (self.phi(goal_state) - self.phi(state))/torch.norm(self.phi(goal_state) - self.phi(state), dim=1)





    
