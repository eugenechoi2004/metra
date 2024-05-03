import gym

# Create the environment
env = gym.make('Ant-v4')

# Reset the environment
state, _ = env.reset()
episode_reward = 0

# Test with random actions
for _ in range(1000):  # Adjust the number of test steps
    env.render()
    
    # Sample a random action from the environment's action space
    action = env.action_space.sample()
    
    # Take a step in the environment with the random action
    next_state, reward, done, _, _ = env.step(action)
    print(next_state)
    # Accumulate the reward for the current episode
    episode_reward += reward
    
    # If the episode ends, reset the environment
    if done:
        state, _ = env.reset()
        print(f"Episode finished with reward: {episode_reward}")
        episode_reward = 0

# Close the environment
env.close()
