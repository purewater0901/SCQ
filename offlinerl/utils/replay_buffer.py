import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset, env_name, normalize_reward=False, data_size_ratio=None):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = self.normalize_reward(dataset['rewards'], env_name) if normalize_reward else dataset['rewards']
		self.reward = self.reward.reshape(-1,1)
		self.done = dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]

		if data_size_ratio is not None:
			data_size = int(self.size * data_size_ratio)
			print("original data size: ", self.size)
			print("shrinked data size: ", data_size)

			perm_idx = np.random.permutation(self.size)
			target_idx = perm_idx[:data_size]

			self.state = self.state[target_idx]
			self.action = self.action[target_idx]
			self.next_state = self.next_state[target_idx]
			self.reward = self.reward[target_idx]
			self.done = self.done[target_idx]
			self.size = data_size

	def normalize_reward(self, rewards, env_name):
		if "antmaze" in env_name:
			return (rewards - 2.0) * 2.0

		raise NotImplementedError


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
