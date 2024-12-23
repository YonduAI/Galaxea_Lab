import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import omni.isaac.lab_tasks

# Load the registered environment

env_id = "Isaac-R1-Bin-Pick-Env-v0"

# Vectorize the environment for parallel training

vec_env = make_vec_env(env_id, n_envs=4)  # Adjust n_envs based on your hardware

# Initialize PPO model

model = PPO(
"MlpPolicy",         # Use Multi-layer Perceptron policy
vec_env,
verbose=1,
tensorboard_log="./ppo_binpick_tensorboard/",
learning_rate=3e-4,
n_steps=2048,
batch_size=64,
n_epochs=10,
gamma=0.99,
gae_lambda=0.95,
clip_range=0.2,
)

# Train the model

timesteps = 500000  # Adjust based on training needs
model.learn(total_timesteps=timesteps)

# Save the trained model

model.save("ppo_yondu_binpick_registered")

print("Training complete. Model saved as 'ppo_yondu_binpick_registered.zip'.")