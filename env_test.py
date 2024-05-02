import gym
from stable_baselines3 import SAC

# Create the environment
env = gym.make('Ant-v4')

# Create the SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)  # Adjust timesteps as needed

# Save the model
model.save("sac_ant")

# Optionally, load the trained model
# model = SAC.load("sac_ant")

# Test the trained model
state, _ = env.reset()
episode_reward = 0

for _ in range(1000):  # Adjust the number of test steps
    env.render()
    action, _ = model.predict(state)
    state, reward, done, truncated, _ = env.step(action)
    episode_reward += reward

    if done or truncated:
        print(f"Total Reward: {episode_reward}")
        state, _ = env.reset()
        episode_reward = 0

env.close()
