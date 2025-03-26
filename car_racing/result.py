import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

# Create the CarRacing-v3 environment
env = gym.make('CarRacing-v3', render_mode="human")  # Ensure visual rendering

# Load the trained model
model = PPO.load("ppo_car_racing")

# Test the trained agent
obs = env.reset()[0]  # Reset the environment properly
episode_rewards = []
done = False
truncated = False

while not done and not truncated:
    # The agent picks an action using the policy
    action, _states = model.predict(obs, deterministic=True)
    
    # Take the action in the environment
    obs, reward, done, truncated, info = env.step(action)
    
    # Collect the rewards over the episode
    episode_rewards.append(reward)

# Calculate the total reward the agent accumulated
total_reward = np.sum(episode_rewards)
print(f"Total reward after training: {total_reward}")

# Close the environment
env.close()

# Plot the cumulative reward over the episode
plt.plot(np.cumsum(episode_rewards))  # Cumulative rewards for better visualization
plt.title("Cumulative Rewards Over Episode")
plt.xlabel("Time Steps")
plt.ylabel("Cumulative Reward")
plt.show()