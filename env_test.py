import gym

# Create the environment
env = gym.make('Ant-v4', render_mode='human')

# Initialize the environment
state = env.reset()

# Set the number of episodes and steps per episode
num_episodes = 10
max_steps_per_episode = 1000

for episode in range(num_episodes):
    # Reset the environment at the beginning of each episode
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Render the environment (if needed)
        env.render()

        # Sample a random action from the action space
        action = env.action_space.sample()

        # Take a step in the environment using the sampled action
        next_state, reward, done, truncated, info = env.step(action)

        # Update the episode reward
        episode_reward += reward

        # Move to the next state
        state = next_state

        # Check if the episode has ended
        if done or truncated:
            break

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# Close the environment
env.close()
