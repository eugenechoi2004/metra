from torch.distributions import Normal, Categorical, Uniform
import torch
import gym
from sac_torch import Agent
from networks import Phi
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from envs.custom_dmc_tasks import dmc
from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper

class Metra():
    def __init__(self, **kwargs):
        self.discrete = kwargs['discrete']
        if self.discrete:
            self.z_dist = Categorical(probs=torch.ones(16)/16)
        else:
            self.z_dist = Normal(loc=0, scale=1)
            # self.z_dist = Uniform(low=kwargs['latent_low'], high=kwargs['latent_high'])
        self.latent_dim = kwargs['latent_dim']
        self.env_name = kwargs['env_name']
        if kwargs['pixel']:
            self.env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=0)
            self.env = RenderWrapper(self.env)
        else:
            self.env = gym.make(self.env_name, exclude_current_positions_from_observation=False)

        example_ob = self.env.reset()
        if kwargs['encoder']:
            def make_encoder(**kwargs):
                return Encoder(pixel_shape=(64,64,3), **kwargs)
            example_encoder = make_encoder()
            self.encoder = make_encoder(spectral_normalization=True)
            self.module_obs_dim = example_encoder(torch.as_tensor(example_ob).float().unsqueeze(0)).shape[-1]
        else:
            self.module_obs_dim = self.env.observation_space.shape
        
        self.n_epochs = kwargs['n_epochs']
        self.batch_size = kwargs['batch_size']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.agent = Agent(input_dims=self.module_obs_dim, env=self.env,
                           n_actions=self.env.action_space.shape[0], layer1_size=256, layer2_size=256, z_dim=self.latent_dim)
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
        if self.discrete:
            skill_sample = torch.nn.functional.one_hot(self.z_dist.sample((1,)), self.latent_dim)
        else:
            skill_sample = self.z_dist.sample((self.latent_dim,))
            skill_sample /= torch.norm(skill_sample, dim=0)
        return skill_sample.to(self.device)

    def train(self):
        self.agent.load_models()
        self.phi.load_checkpoint()
        self.lamb = torch.tensor(21.8182, requires_grad=True, device=self.device)
        for epoch in range(self.n_epochs + 1):
            phi_losses = []
            lambda_losses = []
            done = False
            observation = self.env.reset()
            z = self.sample_skill()
            while not done:
                # self.env.render(mode='human')
                action = self.agent.choose_action(observation, z)
                observation_, reward, done, _ = self.env.step(action)
                self.agent.remember(observation, action, reward, observation_, done, z)
                observation = observation_
            
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
            print(f"Epoch {epoch}/{self.n_epochs}, "
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
    
    # metrics

    def locomotion_metric(self, n_skills=48):
        visited = set()
        visited_states = []
        trajectories = []
        self.agent.load_models()
        self.phi.load_checkpoint()
        self.env._max_episode_steps = 1000
        for i in range(n_skills):
            print(i)
            trajectory = []
            skill = self.sample_skill()
            observation = self.env.reset()
            done = False
            while not done:
                action = self.agent.choose_action(observation, skill)
                observation_, reward, done, info = self.env.step(action)
                visited.add((int(info['x_position']), int(info['y_position'])))
                visited_states.append(len(visited))
                if 'x_position' in info and 'y_position' in info and observation_[2] > 0.3:
                    x = torch.clamp(torch.tensor(info['x_position']), min=-50, max=50).item()
                    y = torch.clamp(torch.tensor(info['y_position']), min=-50, max=50).item()
                    trajectory.append((y,x))
                else:
                    done = True 
            trajectories.append(trajectory)
        return trajectories, visited_states

    def locomotion_metric_discrete(self):
        trajectories = []
        self.agent.load_models()
        self.phi.load_checkpoint()
        self.env._max_episode_steps = 5000
        for i in range(16):
            print(i)
            trajectory = []
            observation = self.env.reset()
            done = False
            while not done:
                action = self.agent.choose_action(observation, torch.nn.functional.one_hot(torch.tensor(i), 16).unsqueeze(0))
                observation_, reward, done, info = self.env.step(action)
                x = torch.clamp(torch.tensor(observation_[0]), min=-50, max=50).item()
                trajectory.append((i,x))
            trajectories.append(trajectory)
        return trajectories

    def plot_trajectories(self, trajectories):
        plt.figure(figsize=(10, 10))
        cmap = cm.get_cmap('tab10', len(trajectories))  
        for i, trajectory in enumerate(trajectories):
            if trajectory:
                y_coords, x_coords = zip(*trajectory)
                plt.plot(x_coords, y_coords, color=cmap(i), linestyle='-', linewidth=0.5) 
        plt.xlabel('X Position')
        plt.ylabel('Skill')
        plt.axis([-25, 25, 0, 16])
        plt.title('HalfCheetah', weight='bold')
        plt.show()

    def calculate_skill(self, state, goal_state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        goal_state = torch.tensor(goal_state, dtype=torch.float).to(self.device)
        diff = self.phi(goal_state) - self.phi(state)
        if len(diff.shape) == 1:
            den = torch.norm(diff, dim=0)  
        else:
            den = torch.norm(diff, dim=1)  
        return diff / den if den != 0 else diff 

    def generate_goal(self, x, y, current_observation):
        goal_x = np.random.uniform(low=x-7.5, high=x+7.5)
        goal_y = np.random.uniform(low=y-7.5, high=y+7.5)
        current_observation[0] = goal_x
        current_observation[1] = goal_y
        return current_observation, goal_x, goal_y
    
    def test_skills(self, episodes):
        self.agent.load_models()
        self.phi.load_checkpoint()
        avg_rewards = []
        for i in range(1000):
            print(i)
            rewards = []
            tot_rewards = 0
            observation = self.env.reset()
            goal_state, goal_x, goal_y = self.generate_goal(0, 0, observation)
            z = self.calculate_skill(observation, goal_state)
            done = False
            for j in range(1000):
                self.env.render(mode='human')
                action = self.agent.choose_action(observation, z)
                observation_, reward, done, info = self.env.step(action)
                observation_ = torch.tensor(observation_, dtype=torch.float).to(self.device)
                goal_pos = np.array([goal_x, goal_y])
                current_pos = np.array([info['x_position'], info['y_position']])
                distance = np.linalg.norm(goal_pos - current_pos)
                if distance <= 3:
                    print("reached goal")
                    tot_rewards += 7.5
                    goal_state, goal_x, goal_y = self.generate_goal(info['x_position'],info['y_position'], observation)
                    z = self.calculate_skill(observation, goal_state)
                rewards.append(tot_rewards)
            avg_rewards.append(rewards)
        avg_rewards = np.array(avg_rewards)
        avg_rewards = avg_rewards.mean(axis=0)
        print(avg_rewards.shape)
        return avg_rewards