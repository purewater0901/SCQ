from offlinerl.algo.SCQ import SCQ
from offlinerl.algo.SAC import SAC
from offlinerl.utils.replay_buffer import ReplayBuffer

import torch
import numpy as np
import random
import argparse
import yaml
import wandb
import gym
import d4rl
from tqdm import tqdm

# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    avg_step = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = np.array(state).reshape(1,-1)
            action = policy.get_action(state).reshape(-1)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            avg_step += 1

    avg_reward /= eval_episodes
    avg_step /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}, Average Step: {avg_step}")
    print("---------------------------------------")
    return d4rl_score, avg_reward, avg_step

def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser()
	# Experiment
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--log_interval", default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_episodes", default=10, type=int)    # Number of trajectories for evaluation
    parser.add_argument("--option_name", default=None)

    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--actor_learning_rate", default=3e-4, type=float)      # Actor learning rate
    parser.add_argument("--critic_learning_rate", default=3e-4, type=float)     # Critic learning rate
    parser.add_argument("--soft_target_tau", default=0.005)                     # Target network update rate
    parser.add_argument("--actor_num_hidden_layers", default=2, type=int)
    parser.add_argument("--critic_num_hidden_layers", default=2, type=int)
    parser.add_argument("--hidden_layer_dim", default=400, type=int)
    parser.add_argument("--actor_clip_grad_norm", default=None, type=float)
    parser.add_argument("--critic_clip_grad_norm", default=None, type=float)
    parser.add_argument("--data_size_ratio", default=None, type=int)      # data size ratio for experiment

    parser.add_argument("--actor_penalty_coef", default=None, type=float)
    parser.add_argument("--critic_penalty_coef", default=0.0, type=float)
    parser.add_argument("--normalize_reward", default=False)
    parser.add_argument("--use_layernormalization", default=False)
    parser.add_argument("--use_actor_scheduler", default=False)
    parser.add_argument("--log_sig_min", default=-5.0, type=float)
    parser.add_argument("--log_sig_max", default=2.0, type=float)
    parser.add_argument("--vae_hidden_layer_dim", default=750, type=int)
    parser.add_argument("--vae_num_hidden_layers", default=1, type=int)
    parser.add_argument("--vae_learning_rate", default=1e-3, type=float)
    parser.add_argument("--vae_sampling_num", default=10, type=int)

    parser.add_argument("--use_automatic_entropy_tuning", default=True)
    parser.add_argument("--lagrange_tau", default=None, type=float)
    args = parser.parse_args()

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.config:
        config = parse_yaml(args.config)

        for key, value in config.items():
            setattr(args, key, value)

    print(args)

    # setup env
    env = gym.make(args.env)

    ## setup seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Load dataset
    replay_buffer = ReplayBuffer(state_dim, action_dim, args.device)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env),
                               env_name=args.env,
                               normalize_reward=args.normalize_reward,
                               data_size_ratio=args.data_size_ratio)

    # Load Policy
    if args.policy == "SCQ":
        policy = SCQ(state_dim, action_dim, max_action, args)
    elif args.policy == "SAC":
        policy = SAC(state_dim, action_dim, max_action, args)
    else:
        raise NotImplementedError

    # setup wandb
    project_name = args.env
    group_name = args.env
    if args.option_name:
        group_name = group_name + "_" + str(args.option_name)

    wandb.init(project="Strategically Conservative Q Learning", config=args, group=group_name)
    if args.lagrange_tau:
        wandb.run.name = f"{project_name}_lagrange_tau{args.lagrange_tau}_seed{args.seed}"
    else:
        wandb.run.name = f"{project_name}_lam{args.critic_penalty_coef}_seed{args.seed}"

    if args.actor_penalty_coef:
        wandb.run.name = wandb.run.name + "_actor_lam:" + str(args.actor_penalty_coef)

    if args.option_name:
        wandb.run.name = wandb.run.name + "_" + str(args.option_name)

    wandb.mark_preempting()

    # Start Training
    max_time_steps = int(args.max_timesteps)
    for step in tqdm(range(1, max_time_steps+1)):
        state, action, next_state, reward, done = replay_buffer.sample(args.batch_size)
        policy.train(state, action, next_state, reward, done, step)
		# Evaluate episode
        if step % args.eval_freq == 0 or step==1:
            d4rl_score, avg_reward, avg_step = eval_policy(policy, args.env, args.seed, eval_episodes=args.eval_episodes)
            wandb.log({"eval/step": step,
       				   "eval/d4rl_score": d4rl_score,
                       "eval/return": avg_reward,
                       "eval/episode length": avg_step,})
    wandb.finish()

if __name__ == "__main__":
    main()
