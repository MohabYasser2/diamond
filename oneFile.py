"""
Single-file runnable trainer for BreakoutNoFrameskip-v4.

This script implements a compact but complete training flow similar to the
DIAMOND repo: (1) collect real env data, (2) train a world model that predicts
next observation + reward + done, (3) create an imagined env that uses the
world model, and (4) train an actor-critic inside the imagined env.

This file is self-contained and does not import from the rest of the codebase.
It is intentionally small and tuned for quick iteration; it reproduces the
high-level flow rather than byte-for-byte model architectures from the repo.

Dependencies: torch, gym or gymnasium, pillow (PIL), numpy

Run:
	python oneFile.py

Note: Atari ROMs must be available for Breakout. If you get errors about ROMs,
install them via: `pip install gym[accept-rom-license]` or follow your gym
distribution instructions.
"""

import argparse
import collections
import random
import time
from dataclasses import dataclass
from typing import Deque, List, Tuple
import os
import json
from datetime import datetime
from tqdm import trange, tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import importlib

# Prefer `gym` if available (classic). Fall back to `gymnasium` if needed.
if importlib.util.find_spec("gym") is not None:
	import gym
else:
	import gymnasium as gym

from PIL import Image
# try to read YAML configs from the repo
try:
	import yaml
except Exception:
	yaml = None


def _load_repo_config():
	"""Load config/trainer.yaml (and merge agent/env defaults) if available.

	Returns a dict with nested keys or empty dict if YAML not available.
	"""
	base = {}
	root = os.path.dirname(__file__)
	cfg_paths = [os.path.join(root, "config", "trainer.yaml"), os.path.join(root, "config", "agent", "default.yaml"), os.path.join(root, "config", "env", "atari.yaml")]
	if yaml is None:
		return base
	for p in cfg_paths:
		try:
			with open(p, "r", encoding="utf-8") as f:
				d = yaml.safe_load(f)
				if not d:
					continue
				# shallow merge for top-level keys (trainer.yaml is primary)
				for k, v in d.items():
					if k not in base:
						base[k] = v
					else:
						# if both are dicts, merge nested keys
						if isinstance(base[k], dict) and isinstance(v, dict):
							base[k].update(v)
						else:
							base[k] = v
		except Exception:
			continue
	return base


def _get_cfg_value(cfg, path, default=None):
	"""Get nested config value by dot-separated path."""
	cur = cfg
	for p in path.split("."):
		if not isinstance(cur, dict):
			return default
		if p not in cur:
			return default
		cur = cur[p]
	return cur


def preprocess_frame(frame, size=64):
	# frame: HxWxC (uint8)
	img = Image.fromarray(frame)
	img = img.resize((size, size)).convert("RGB")
	arr = np.array(img, dtype=np.uint8)
	return arr


class SimpleFileLogger:
	def __init__(self, path):
		self.path = path
		try:
			os.makedirs(os.path.dirname(path), exist_ok=True)
		except Exception:
			pass

	def log(self, d: dict):
		# add timestamp
		d = dict(d)
		d["timestamp"] = datetime.utcnow().isoformat()
		s = json.dumps(d, default=str)
		try:
			with open(self.path, "a") as f:
				f.write(s + "\n")
		except Exception:
			pass
		# print concise terminal line
		keys = [k for k in ("phase", "wm/epoch", "ac/iter", "collect/avg_return", "wm/avg_loss", "ac/loss") if k in d]
		if keys:
			kp = ", ".join(f"{k}={d[k]}" for k in keys)
			print(f"[LOG] {kp}")


class ReplayBuffer:
	def __init__(self, capacity_steps: int = 100000):
		self.capacity = capacity_steps
		self.obs = collections.deque(maxlen=capacity_steps)
		self.actions = collections.deque(maxlen=capacity_steps)
		self.rews = collections.deque(maxlen=capacity_steps)
		self.dones = collections.deque(maxlen=capacity_steps)

	def add(self, obs, action, rew, done):
		self.obs.append(obs)
		self.actions.append(action)
		self.rews.append(rew)
		self.dones.append(done)

	def __len__(self):
		return len(self.obs)

	def sample_batch(self, batch_size, seq_len=1):
		# sample random indices where a contiguous seq_len segment exists
		max_start = len(self.obs) - seq_len
		if max_start <= 0:
			raise ValueError("Not enough data to sample")
		idxs = np.random.randint(0, max_start, size=batch_size)
		obs_b, act_b, rew_b, done_b, next_obs_b = [], [], [], [], []
		for i in idxs:
			obs_b.append(self.obs[i])
			act_b.append(self.actions[i])
			rew_b.append(self.rews[i])
			done_b.append(self.dones[i])
			next_obs_b.append(self.obs[i + 1])
		obs_b = np.stack(obs_b)
		act_b = np.array(act_b)
		rew_b = np.array(rew_b, dtype=np.float32)
		done_b = np.array(done_b, dtype=np.float32)
		next_obs_b = np.stack(next_obs_b)
		return obs_b, act_b, rew_b, done_b, next_obs_b


class SmallWorldModel(nn.Module):
	"""Predicts next RGB frame (64x64x3), reward and done flag from (obs, action).

	Obs input is uint8 image; model expects float [0,1].
	"""

	def __init__(self, img_channels=3, action_dim=4, hidden=128):
		super().__init__()
		self.enc = nn.Sequential(
			nn.Conv2d(img_channels, 32, 4, 2, 1),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, 2, 1),
			nn.ReLU(),
			nn.Conv2d(64, 128, 4, 2, 1),
			nn.ReLU(),
		)
		self.flatten = nn.Flatten()
		self.fc = nn.Linear(128 * 8 * 8 + action_dim, hidden)
		# predict next frame (decoder)
		self.fc_decode = nn.Linear(hidden, 128 * 8 * 8)
		self.dec = nn.Sequential(
			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 4, 2, 1),
			nn.ReLU(),
			nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
			nn.Sigmoid(),
		)
		# reward and done heads
		self.reward_head = nn.Linear(hidden, 1)
		self.done_head = nn.Linear(hidden, 1)

	def forward(self, obs, action):
		# obs: B x H x W x C (uint8 or float), convert to BxCxHxW float
		if obs.dtype == torch.uint8:
			obs = obs.float() / 255.0
		obs = obs.permute(0, 3, 1, 2)
		h = self.enc(obs)
		hflat = self.flatten(h)
		x = torch.cat([hflat, action.float()], dim=1)
		z = F.relu(self.fc(x))
		dec = self.fc_decode(z).reshape(-1, 128, 8, 8)
		next_frame = self.dec(dec)
		reward = self.reward_head(z).squeeze(1)
		done_logit = self.done_head(z).squeeze(1)
		done = torch.sigmoid(done_logit)
		return next_frame, reward, done


class SmallActorCritic(nn.Module):
	def __init__(self, action_dim):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, 8, 4, 2),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, 2, 1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
		)
		self.fc = nn.Linear(64 * 8 * 8, 256)
		self.policy = nn.Linear(256, action_dim)
		self.value = nn.Linear(256, 1)

	def forward(self, obs):
		# obs: BxHxWxC uint8 or float
		if obs.dtype == torch.uint8:
			obs = obs.float() / 255.0
		obs = obs.permute(0, 3, 1, 2)
		h = self.conv(obs)
		h = h.reshape(h.size(0), -1)
		h = F.relu(self.fc(h))
		logits = self.policy(h)
		value = self.value(h).squeeze(1)
		return logits, value


class WorldModelEnv:
	"""A simple imagined env that uses the world model to step.

	It stores the current imagined observation, and stepping uses the world
	model to predict the next frame, reward and done.
	"""

	def __init__(self, world_model: SmallWorldModel, device: torch.device, action_space):
		self.wm = world_model
		self.device = device
		self.action_space = action_space
		self.current_obs = None

	def reset_from(self, obs):
		# obs: HxWxC uint8
		self.current_obs = obs
		return self.current_obs

	def step(self, action):
		# action: scalar int
		a = torch.zeros(1, self.action_space.n, device=self.device)
		a[0, action] = 1.0
		obs_t = torch.from_numpy(self.current_obs[None]).to(self.device)
		with torch.no_grad():
			next_frame, reward, done = self.wm(obs_t, a)
		# convert next_frame to uint8 image
		nf = (next_frame[0].cpu().numpy() * 255.0).astype(np.uint8)
		nf = np.transpose(nf, (1, 2, 0))
		self.current_obs = nf
		r = float(reward.item())
		d = bool(done.item() > 0.5)
		info = {}
		return nf, r, d, info


def _env_reset(env):
	res = env.reset()
	# gym: obs OR gymnasium: (obs, info)
	if isinstance(res, tuple):
		return res[0]
	return res


def _env_step(env, action):
	res = env.step(action)
	# gymnasium: obs, reward, terminated, truncated, info
	if isinstance(res, tuple) and len(res) == 5:
		obs, rew, terminated, truncated, info = res
		done = bool(terminated or truncated)
		return obs, rew, done, info
	# gym (older): obs, reward, done, info
	if isinstance(res, tuple) and len(res) == 4:
		obs, rew, done, info = res
		return obs, rew, bool(done), info
	# fallback
	raise RuntimeError("Unknown env.step() return signature")


def collect_initial_data(env_name: str, num_steps=5000, size=64, save_dir=None, logger=None) -> Tuple[ReplayBuffer, List[float]]:
	env = gym.make(env_name)
	buf = ReplayBuffer(capacity_steps=max(10000, num_steps + 1000))
	obs = _env_reset(env)
	obs = preprocess_frame(obs, size=size)
	episode_returns = []
	cur_ret = 0.0
	pbar = trange(num_steps, desc="Collecting")
	for step in pbar:
		action = env.action_space.sample()
		next_obs, rew, done, info = _env_step(env, action)
		next_obs_proc = preprocess_frame(next_obs, size=size)
		buf.add(obs, action, float(rew), bool(done))
		cur_ret += float(rew)
		obs = next_obs_proc
		if done:
			episode_returns.append(cur_ret)
			cur_ret = 0.0
			obs = _env_reset(env)
			obs = preprocess_frame(obs, size=size)
		# update progress bar with stats
		avg_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
		pbar.set_postfix({"buf": len(buf), "avg_return": f"{avg_ret:.2f}", "eps": len(episode_returns)})
	env.close()
	# persist collect summary
	if save_dir:
		logpath = os.path.join(save_dir, "collect_stats.jsonl")
		try:
			with open(logpath, "a") as f:
				f.write(json.dumps({"phase": "collect", "num_steps": num_steps, "num_episodes": len(episode_returns), "avg_return": float(np.mean(episode_returns) if episode_returns else 0.0)}) + "\n")
		except Exception:
			pass
	if logger:
		logger.log({"phase": "collect", "collect/avg_return": float(np.mean(episode_returns) if episode_returns else 0.0), "collect/num_episodes": len(episode_returns)})
	return buf, episode_returns


def train_world_model(wm: SmallWorldModel, buf: ReplayBuffer, device, epochs=3, batch_size=32, save_dir=None, save_every_epochs=1, logger=None):
	opt = optim.Adam(wm.parameters(), lr=1e-4)
	wm.to(device)
	for ep in range(1, epochs + 1):
		iters = max(1, len(buf) // batch_size)
		epoch_loss = 0.0
		for it in trange(iters, desc=f"WM epoch {ep}"):
			obs_b, act_b, rew_b, done_b, next_obs_b = buf.sample_batch(batch_size)
			obs_t = torch.from_numpy(obs_b).to(device)
			# next_obs_b: B x H x W x C -> convert to B x C x H x W normalized
			next_t = torch.from_numpy(next_obs_b).permute(0, 3, 1, 2).float().to(device) / 255.0
			# one-hot actions
			acts = np.zeros((batch_size, env_action_dim), dtype=np.float32)
			acts[np.arange(batch_size), act_b] = 1.0
			acts_t = torch.from_numpy(acts).to(device)
			pred_next, pred_rew, pred_done = wm(obs_t, acts_t)
			# losses
			loss_frame = F.mse_loss(pred_next, next_t)
			loss_rew = F.mse_loss(pred_rew, torch.from_numpy(rew_b).to(device))
			loss_done = F.binary_cross_entropy(pred_done, torch.from_numpy(done_b).to(device))
			loss = loss_frame + 1.0 * loss_rew + 1.0 * loss_done
			opt.zero_grad()
			loss.backward()
			opt.step()
			epoch_loss += loss.item()

		avg_loss = epoch_loss / iters
		if logger:
			try:
				logger.log({"phase": "wm", "wm/epoch": ep, "wm/avg_loss": avg_loss})
			except Exception:
				pass
		else:
			print(f"WM Epoch {ep}/{epochs} avg_loss={avg_loss:.6f}")

		# checkpoint
		if save_dir and (ep % save_every_epochs == 0):
			path = os.path.join(save_dir, f"wm_epoch_{ep:05d}.pt")
			torch.save({"wm": wm.state_dict(), "opt": opt.state_dict(), "epoch": ep}, path)
			if logger:
				try:
					logger.log({"wm/checkpoint": path})
				except Exception:
					pass
			else:
				print(f"Saved world model checkpoint: {path}")


def train_actor_critic_on_imagination(actor: SmallActorCritic, wm: SmallWorldModel, buf: ReplayBuffer, device, env_name, steps=1000, save_dir=None, save_every_steps=200, logger=None):
	# Simple A2C-like training inside imagination
	actor.to(device)
	wm.to(device)
	opt = optim.Adam(actor.parameters(), lr=1e-4)
	wm_env = WorldModelEnv(wm, device, gym.make(env_name).action_space)

	# start imagined episodes from random real observations
	recent_returns: Deque[float] = collections.deque(maxlen=100)
	pbar = trange(steps, desc="ActorCritic")
	for it in pbar:
		# sample random starting state
		idx = random.randint(0, len(buf) - 2)
		start_obs = buf.obs[idx]
		obs = wm_env.reset_from(start_obs)
		traj_obs = []
		traj_actions = []
		traj_rewards = []
		for t in range(50):
			ob_t = torch.from_numpy(obs[None]).to(device)
			logits, value = actor(ob_t)
			probs = F.softmax(logits, dim=1)
			m = torch.distributions.Categorical(probs)
			a = m.sample().item()
			next_obs, r, d, _ = wm_env.step(a)
			traj_obs.append(ob_t)
			traj_actions.append(a)
			traj_rewards.append(r)
			obs = next_obs
			if d:
				break

		# compute returns and advantages (simple)
		R = 0.0
		returns = []
		for r in reversed(traj_rewards):
			R = r + 0.99 * R
			returns.insert(0, R)
		if len(returns) == 0:
			continue
		returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
		actions_t = torch.tensor(traj_actions, dtype=torch.long, device=device)
		obs_t = torch.cat(traj_obs, dim=0)
		logits, values = actor(obs_t)
		probs = F.softmax(logits, dim=1)
		m = torch.distributions.Categorical(probs)
		logp = m.log_prob(actions_t)
		advantage = returns_t - values
		loss_policy = -(logp * advantage.detach()).mean()
		loss_value = F.mse_loss(values, returns_t)
		loss = loss_policy + 0.5 * loss_value
		opt.zero_grad()
		loss.backward()
		opt.step()

		# log return and loss
		# Use raw predicted rewards sum for episode return (imagined env),
		# not the sum of discounted returns tensor. This better reflects
		# whether predicted rewards are positive/negative.
		ep_ret = float(sum(traj_rewards)) if len(traj_rewards) > 0 else 0.0
		recent_returns.append(ep_ret)
		avg_recent = float(np.mean(recent_returns)) if recent_returns else 0.0
		if logger and (it % 10 == 0):
			try:
				logger.log({"phase": "ac", "ac/iter": it, "ac/loss": loss.item(), "ac/ep_return": ep_ret, "ac/avg_recent_return": avg_recent})
			except Exception:
				pass
		if it % 10 == 0:
			pbar.set_postfix({"loss": f"{loss.item():.4f}", "ep_ret": f"{ep_ret:.2f}", "avg_ret": f"{avg_recent:.2f}"})

		# checkpoint
		if save_dir and ((it + 1) % save_every_steps == 0):
			path = os.path.join(save_dir, f"actor_step_{it:06d}.pt")
			torch.save({"actor": actor.state_dict(), "opt": opt.state_dict(), "step": it}, path)
			if logger:
				try:
					logger.log({"ac/checkpoint": path})
				except Exception:
					pass
			else:
				print(f"Saved actor checkpoint: {path}")


if __name__ == "__main__":
	# load repo YAML config (if available) and use values as defaults
	repo_cfg = _load_repo_config()

	# defaults mapping
	default_collect = _get_cfg_value(repo_cfg, "collection.train.num_steps_total", 10000)
	default_wm_epochs = _get_cfg_value(repo_cfg, "training.num_final_epochs", 50)
	# estimate total actor steps from actor_critic.training: steps_first_epoch + steps_per_epoch*(num_epochs-1)
	ac_first = _get_cfg_value(repo_cfg, "actor_critic.training.steps_first_epoch", 5000)
	ac_per = _get_cfg_value(repo_cfg, "actor_critic.training.steps_per_epoch", 400)
	num_epochs = default_wm_epochs if default_wm_epochs is not None else 50
	default_ac_steps = int(ac_first + ac_per * max(0, int(num_epochs) - 1))

	# device default mapping from common.devices
	common_devices = _get_cfg_value(repo_cfg, "common.devices", None)
	if isinstance(common_devices, int) and torch.cuda.is_available():
		default_device = f"cuda:{common_devices}"
	else:
		default_device = "cuda" if torch.cuda.is_available() else "cpu"

	default_wandb = bool(_get_cfg_value(repo_cfg, "wandb.enable", False))

	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="BreakoutNoFrameskip-v4")
	parser.add_argument("--device", default=default_device)
	parser.add_argument("--collect", type=int, default=int(default_collect), help="number of real env steps to collect")
	parser.add_argument("--wm_epochs", type=int, default=int(default_wm_epochs), help="world model training epochs")
	parser.add_argument("--ac_steps", type=int, default=int(default_ac_steps), help="actor-critic imagined iterations")
	parser.add_argument("--save-dir", type=str, default="checkpoints", help="directory to store checkpoints")
	parser.add_argument("--save-every-wm", type=int, default=1, help="save world model every N epochs")
	parser.add_argument("--save-every-ac", type=int, default=200, help="save actor every N iterations")
	parser.add_argument("--resume", action="store_true", help="resume from latest checkpoints if available")
	# default wandb flag reflects config; allow overriding on CLI
	if default_wandb:
		parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="disable wandb logging")
		parser.set_defaults(wandb=True)
	else:
		parser.add_argument("--wandb", dest="wandb", action="store_true", help="enable wandb logging if installed")
	args = parser.parse_args()

	device = torch.device(args.device)
	env_name = args.env

	print(f"Collecting initial data from {env_name} ({args.collect} steps)")
	# prepare logger and save dir
	os.makedirs(args.save_dir, exist_ok=True)
	file_logger = SimpleFileLogger(os.path.join(args.save_dir, "train_log.jsonl"))
	buf, ep_returns = collect_initial_data(env_name, num_steps=args.collect, save_dir=args.save_dir, logger=file_logger)

	global env_action_dim
	env_action_dim = gym.make(env_name).action_space.n

	# prepare save dir
	os.makedirs(args.save_dir, exist_ok=True)

	# optional wandb
	wandb_logger = None
	if args.wandb:
		try:
			import wandb

			wandb.init(project="diamond-onefile", name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
			wandb_logger = wandb
		except Exception as e:
			print("wandb not available or failed to init:", e)

	print("Building models")
	wm = SmallWorldModel(img_channels=3, action_dim=env_action_dim)
	actor = SmallActorCritic(action_dim=env_action_dim)

	# resume logic: attempt to load latest checkpoints
	if args.resume:
		# find latest wm checkpoint
		wm_files = [f for f in os.listdir(args.save_dir) if f.startswith("wm_epoch_") and f.endswith('.pt')]
		if wm_files:
			latest = sorted(wm_files)[-1]
			path = os.path.join(args.save_dir, latest)
			print("Loading WM checkpoint", path)
			d = torch.load(path, map_location=device)
			wm.load_state_dict(d['wm'])
    
		actor_files = [f for f in os.listdir(args.save_dir) if f.startswith("actor_step_") and f.endswith('.pt')]
		if actor_files:
			latest = sorted(actor_files)[-1]
			path = os.path.join(args.save_dir, latest)
			print("Loading actor checkpoint", path)
			d = torch.load(path, map_location=device)
			actor.load_state_dict(d['actor'])

	print("Training world model")
	train_world_model(wm, buf, device, epochs=args.wm_epochs, batch_size=32, save_dir=args.save_dir, save_every_epochs=args.save_every_wm, logger=wandb_logger)

	print("Training actor-critic in imagination")
	train_actor_critic_on_imagination(actor, wm, buf, device, env_name, steps=args.ac_steps, save_dir=args.save_dir, save_every_steps=args.save_every_ac, logger=wandb_logger)

	print("Done (single-file flow completed)")

