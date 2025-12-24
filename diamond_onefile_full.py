#!/usr/bin/env python3
"""
DIAMOND-style one-file trainer (Atari BreakoutNoFrameskip-v4 compatible).

Includes:
- Atari preprocessing (64x64, grayscale, frame stack)
- Diffusion world model (EDM-style sigma schedule + conditional denoiser UNet)
- Reward/termination model
- Imagination rollouts (world-model env)
- PPO actor-critic training in imagination
- Replay buffer + dataset sampling
- outputing (best eval return)
- W&B logging (optional)

Recommended runtime: GPU (cuda). CPU runs only for debugging.
"""

from __future__ import annotations
import os
import time
import math
import json
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

# -----------------------------
# Gym import: prefer legacy gym for BreakoutNoFrameskip-v4.
# (Your working Kaggle stack uses gym==0.26.2 and ale-py==0.8.0)
# -----------------------------
try:
    import gym
except Exception as e:
    raise RuntimeError("Legacy gym is required for BreakoutNoFrameskip-v4. Install gym==0.26.2.") from e

# Optional: W&B
try:
    import wandb
    _WANDB_OK = True
except Exception:
    _WANDB_OK = False

# Optional: cv2 for resize
try:
    import cv2
except Exception as e:
    raise RuntimeError("opencv-python is required (cv2). Please install opencv-python.") from e


# ============================================================
# Config
# ============================================================

@dataclass
class Cfg:
    # Env
    env_id: str = "BreakoutNoFrameskip-v4"
    seed: int = 0
    img_size: int = 64
    frame_stack: int = 4
    noop_max: int = 30
    frame_skip: int = 4  # env already NoFrameskip; we do action repeat ourselves in wrapper

    # Device
    device: str = "cuda"

    # Training schedule (DIAMOND-style)
    epochs: int = 30
    collect_steps_initial: int = 20_000
    collect_steps_per_epoch: int = 5_000

    # Replay buffer
    replay_size: int = 200_000

    # World model (diffusion) training
    wm_batch: int = 32
    wm_steps: int = 300
    wm_lr: float = 3e-4

    # Reward/termination training
    re_batch: int = 64
    re_steps: int = 200
    re_lr: float = 3e-4

    # PPO training (in imagination)
    ppo_steps: int = 200
    ppo_lr: float = 3e-4
    ppo_clip: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Imagination rollouts
    imagine_horizon: int = 32
    imagine_batch_envs: int = 32

    # Diffusion sampler / EDM-style noise
    sigma_min: float = 0.002
    sigma_max: float = 5.0
    rho: float = 7.0
    sampler_steps: int = 15

    # Logging / eval
    eval_episodes: int = 5
    log_every: int = 50
    save_dir: str = "outputs_onefile"
    use_wandb: bool = False
    wandb_project: str = "diamond-onefile"
    wandb_run_name: Optional[str] = None


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_tag():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def wandb_log(d: Dict, step: int):
    if _WANDB_OK:
        wandb.log(d, step=step)

# ============================================================
# Atari preprocessing wrapper
# ============================================================

class AtariPreprocess:
    """
    Minimal, robust Atari preprocessing:
    - action repeat (frame_skip)
    - max-pooling over last two frames
    - grayscale
    - resize to img_size x img_size
    - frame stacking
    """

    def __init__(self, env, img_size=64, frame_skip=4, frame_stack=4, noop_max=30, seed=0):
        self.env = env
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.noop_max = noop_max
        self.rng = np.random.RandomState(seed)

        self.obs_buf = np.zeros((2, 210, 160), dtype=np.uint8)  # raw grayscale frames
        self.frames = FrameDeque(frame_stack)
        self.action_space = env.action_space

        # build observation space: (C,H,W) float32 in [0,1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(frame_stack, img_size, img_size),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # random no-ops
        noops = self.rng.randint(0, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            obs, r, done, info = self._step_raw(0)
            if done:
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

        frame = self._process(obs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._get_obs()

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info = {}
        # action repeat, max over last two
        last_obs = None
        for t in range(self.frame_skip):
            obs, r, done, info = self._step_raw(action)
            total_reward += float(r)
            if t == self.frame_skip - 2:
                self.obs_buf[0] = self._to_gray(obs)
            if t == self.frame_skip - 1:
                self.obs_buf[1] = self._to_gray(obs)
            if done:
                break
            last_obs = obs

        max_frame = np.maximum(self.obs_buf[0], self.obs_buf[1])
        frame = self._resize(max_frame)
        self.frames.append(frame)
        return self._get_obs(), total_reward, done, info

    def _step_raw(self, action: int):
        out = self.env.step(action)
        # gym 0.26: step returns (obs, reward, terminated, truncated, info)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        obs, reward, done, info = out
        return obs, reward, done, info

    def _to_gray(self, obs):
        # obs is RGB 210x160x3
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

    def _resize(self, gray_210_160):
        resized = cv2.resize(gray_210_160, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)

    def _process(self, obs):
        return self._resize(self._to_gray(obs))

    def _get_obs(self):
        # (C,H,W)
        return np.stack(list(self.frames), axis=0).astype(np.float32)

class FrameDeque:
    """Simple wrapper around collections.deque with fixed maxlen.

    Implements append, clear, __iter__, __len__, __getitem__, and __repr__.
    """
    def __init__(self, maxlen: int):
        from collections import deque
        self._d = deque(maxlen=maxlen)
        self._maxlen = maxlen

    @property
    def maxlen(self):
        return self._maxlen

    def append(self, x):
        self._d.append(x)

    def clear(self):
        self._d.clear()

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, i):
        return list(self._d)[i]

    def __repr__(self):
        return repr(self._d)


def make_env(cfg: Cfg):
    env = gym.make(cfg.env_id)
    env = AtariPreprocess(
        env,
        img_size=cfg.img_size,
        frame_skip=cfg.frame_skip,
        frame_stack=cfg.frame_stack,
        noop_max=cfg.noop_max,
        seed=cfg.seed
    )
    return env


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int,int,int]):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, size=batch)
        return dict(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
        )


# ============================================================
# Conditioning utilities (actions)
# ============================================================

class ActionEmbed(nn.Module):
    def __init__(self, n_actions: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_actions, dim)
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.emb(a)


# ============================================================
# Diffusion world model (EDM-style)
# - Denoiser predicts x0 (clean next frame-stack) from noisy x and sigma, conditioned on prev_obs & action
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor):
        # x: [B]
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(tdim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class DenoiserUNet(nn.Module):
    """
    Inputs:
      - x: noisy next_obs  [B,C,H,W]
      - sigma: noise level [B]
      - prev_obs: prev obs [B,C,H,W]
      - action: [B]
    Output:
      - x0_pred [B,C,H,W]
    """
    def __init__(self, obs_ch: int, n_actions: int, base_ch: int = 64, tdim: int = 256, act_dim: int = 64):
        super().__init__()
        self.t_emb = SinusoidalPosEmb(tdim)
        self.t_mlp = nn.Sequential(
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.a_emb = ActionEmbed(n_actions, act_dim)

        in_ch = obs_ch * 2 + act_dim  # x + prev_obs + action_map
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.rb1 = ResBlock(base_ch, base_ch, tdim)
        self.d1 = Down(base_ch)
        self.rb2 = ResBlock(base_ch, base_ch*2, tdim)
        self.d2 = Down(base_ch*2)
        self.rb3 = ResBlock(base_ch*2, base_ch*4, tdim)

        self.u2 = Up(base_ch*4)
        self.rb4 = ResBlock(base_ch*4 + base_ch*2, base_ch*2, tdim)
        self.u1 = Up(base_ch*2)
        self.rb5 = ResBlock(base_ch*2 + base_ch, base_ch, tdim)

        self.out = nn.Conv2d(base_ch, obs_ch, 3, padding=1)

    def forward(self, x, sigma, prev_obs, action):
        # sigma to embedding: log sigma stabilizes scale
        t = torch.log(sigma + 1e-8)
        t = self.t_mlp(self.t_emb(t))

        a = self.a_emb(action)  # [B, act_dim]
        # broadcast action embedding to spatial map
        a_map = a[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])

        h = torch.cat([x, prev_obs, a_map], dim=1)
        h = self.in_conv(h)

        h1 = self.rb1(h, t)
        h2 = self.d1(h1)
        h2 = self.rb2(h2, t)
        h3 = self.d2(h2)
        h3 = self.rb3(h3, t)

        u2 = self.u2(h3)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.rb4(u2, t)
        u1 = self.u1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.rb5(u1, t)

        return self.out(u1)

class DiffusionWorldModel(nn.Module):
    def __init__(self, obs_ch: int, n_actions: int, cfg: Cfg):
        super().__init__()
        self.denoiser = DenoiserUNet(obs_ch=obs_ch, n_actions=n_actions)
        self.cfg = cfg

    @staticmethod
    def edm_sigma_schedule(num_steps: int, sigma_min: float, sigma_max: float, rho: float, device):
        # Karras/EDM schedule
        i = torch.arange(num_steps, device=device, dtype=torch.float32)
        t = i / (num_steps - 1)
        sigmas = (sigma_max ** (1 / rho) + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        return torch.cat([sigmas, torch.zeros(1, device=device)], dim=0)  # last is 0

    def sample_sigma(self, batch: int, device):
        # log-uniform sampling in [sigma_min, sigma_max]
        u = torch.rand(batch, device=device)
        return self.cfg.sigma_min * (self.cfg.sigma_max / self.cfg.sigma_min) ** u

    def loss(self, prev_obs, action, next_obs):
        """
        EDM-style denoising loss: predict x0 from noisy x.
        """
        B = prev_obs.shape[0]
        device = prev_obs.device
        sigma = self.sample_sigma(B, device)  # [B]
        noise = torch.randn_like(next_obs)
        x_noisy = next_obs + noise * sigma[:, None, None, None]

        x0_pred = self.denoiser(x_noisy, sigma, prev_obs, action)
        # weight: 1/sigma^2 (common choice); stable enough for this scale
        w = 1.0 / (sigma**2 + 1e-8)
        loss = (w[:, None, None, None] * (x0_pred - next_obs) ** 2).mean()
        return loss

    @torch.no_grad()
    def imagine_next(self, prev_obs, action):
        """
        Generate next obs via iterative denoising (EDM Euler).
        prev_obs: [B,C,H,W]
        action: [B]
        """
        device = prev_obs.device
        B, C, H, W = prev_obs.shape

        sigmas = self.edm_sigma_schedule(self.cfg.sampler_steps, self.cfg.sigma_min, self.cfg.sigma_max, self.cfg.rho, device)
        x = torch.randn((B, C, H, W), device=device) * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_b = torch.full((B,), sigma, device=device)

            x0 = self.denoiser(x, sigma_b, prev_obs, action)
            d = (x - x0) / (sigma + 1e-8)
            dt = sigma_next - sigma
            x = x + d * dt

        return x.clamp(0.0, 1.0)


# ============================================================
# Reward / termination model
# ============================================================

class RewEndModel(nn.Module):
    def __init__(self, obs_ch: int, n_actions: int):
        super().__init__()
        self.a_emb = ActionEmbed(n_actions, 32)
        in_ch = obs_ch + 32

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU()
        )
        self.rew = nn.Linear(128, 1)
        self.done = nn.Linear(128, 1)

    def forward(self, obs, action):
        B, C, H, W = obs.shape
        a = self.a_emb(action)[:, :, None, None].expand(-1, -1, H, W)
        x = torch.cat([obs, a], dim=1)
        h = self.net(x).view(B, -1)
        h = self.head(h)
        r = self.rew(h).squeeze(-1)
        d = torch.sigmoid(self.done(h)).squeeze(-1)
        return r, d

    def loss(self, obs, action, reward, done):
        r_pred, d_pred = self.forward(obs, action)
        # reward regression + done BCE
        loss_r = F.mse_loss(r_pred, reward)
        loss_d = F.binary_cross_entropy(d_pred, done)
        return loss_r + loss_d, loss_r.detach(), loss_d.detach()


# ============================================================
# Actor-Critic (PPO)
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_ch: int, n_actions: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(obs_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        # for 64x64: compute feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, obs_ch, 64, 64)
            feat_dim = self.enc(dummy).shape[-1]
        self.pi = nn.Linear(feat_dim, n_actions)
        self.v = nn.Linear(feat_dim, 1)

    def forward(self, obs):
        f = self.enc(obs)
        logits = self.pi(f)
        value = self.v(f).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs):
        logits, v = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v

    def logp_value_entropy(self, obs, action):
        logits, v = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(action)
        ent = dist.entropy()
        return logp, v, ent


# ============================================================
# Imagination buffer
# ============================================================

@dataclass
class RolloutBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    ret: torch.Tensor
    adv: torch.Tensor
    val: torch.Tensor

def compute_gae(rewards, dones, values, gamma, lam):
    """
    rewards, dones: [T,B]
    values: [T+1,B]
    """
    T, B = rewards.shape
    adv = torch.zeros((T, B), device=rewards.device)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


# ============================================================
# Training loop
# ============================================================

def evaluate_real_env(env, ac: ActorCritic, device: str, episodes: int):
    total = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_t = torch.tensor(obs, device=device).unsqueeze(0)
            a, _, _ = ac.act(obs_t)
            obs, r, done, _ = env.step(int(a.item()))
            ep_r += float(r)
        total.append(ep_r)
    return float(np.mean(total)), float(np.std(total))

def collect_real(env, rb: ReplayBuffer, ac: ActorCritic, device: str, steps: int, eps: float):
    obs = env.reset()
    pbar = tqdm(total=steps, desc="Collect")
    for _ in range(steps):
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, device=device).unsqueeze(0)
                a_t, _, _ = ac.act(obs_t)
                a = int(a_t.item())
        next_obs, r, done, _ = env.step(a)
        rb.add(obs, a, r, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()
        pbar.update(1)
    pbar.close()

def train_world_model(wm: DiffusionWorldModel, opt: torch.optim.Optimizer, rb: ReplayBuffer, cfg: Cfg):
    wm.train()
    losses = []
    for _ in trange(cfg.wm_steps, desc="WM"):
        batch = rb.sample(cfg.wm_batch)
        prev_obs = to_torch(batch["obs"], cfg.device)
        act = to_torch(batch["action"], cfg.device)
        next_obs = to_torch(batch["next_obs"], cfg.device)

        loss = wm.loss(prev_obs, act, next_obs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))

def train_rew_end(model: RewEndModel, opt: torch.optim.Optimizer, rb: ReplayBuffer, cfg: Cfg):
    model.train()
    losses, lr, ld = [], [], []
    for _ in trange(cfg.re_steps, desc="RewEnd"):
        batch = rb.sample(cfg.re_batch)
        obs = to_torch(batch["obs"], cfg.device)
        act = to_torch(batch["action"], cfg.device)
        rew = to_torch(batch["reward"], cfg.device)
        done = to_torch(batch["done"], cfg.device)

        loss, loss_r, loss_d = model.loss(obs, act, rew, done)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item()); lr.append(loss_r.item()); ld.append(loss_d.item())
    return float(np.mean(losses)), float(np.mean(lr)), float(np.mean(ld))

@torch.no_grad()
def imagine_rollout(wm: DiffusionWorldModel, re: RewEndModel, ac: ActorCritic, start_obs: torch.Tensor, cfg: Cfg):
    """
    start_obs: [B,C,H,W]
    Returns PPO training batch computed from imagined trajectories.
    """
    T = cfg.imagine_horizon
    B = start_obs.shape[0]

    obs = start_obs
    obs_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []

    for t in range(T):
        a, logp, v = ac.act(obs)
        next_obs = wm.imagine_next(obs, a)
        r, d = re(next_obs, a)  # reward/done for transition outcome
        # Clip rewards for Atari stability (common)
        r = torch.clamp(r, -1.0, 1.0)

        obs_list.append(obs)
        act_list.append(a)
        logp_list.append(logp)
        rew_list.append(r)
        done_list.append(d)
        val_list.append(v)

        # episode termination in imagination
        obs = next_obs
        # optionally reset done states by masking; we keep them for GAE

    # last value
    _, v_last = ac.forward(obs)
    val_list.append(v_last)

    obs_t = torch.stack(obs_list, dim=0)         # [T,B,C,H,W]
    act_t = torch.stack(act_list, dim=0)         # [T,B]
    logp_t = torch.stack(logp_list, dim=0)       # [T,B]
    rew_t = torch.stack(rew_list, dim=0)         # [T,B]
    done_t = torch.stack(done_list, dim=0)       # [T,B]
    val_t = torch.stack(val_list, dim=0)         # [T+1,B]

    adv, ret = compute_gae(rew_t, done_t, val_t, cfg.gamma, cfg.gae_lambda)

    # flatten to [T*B,...]
    obs_f = obs_t.reshape(T*B, *obs_t.shape[2:])
    act_f = act_t.reshape(T*B)
    logp_f = logp_t.reshape(T*B)
    adv_f = adv.reshape(T*B)
    ret_f = ret.reshape(T*B)
    val_f = val_t[:-1].reshape(T*B)

    # normalize advantage
    adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

    return RolloutBatch(obs=obs_f, act=act_f, logp=logp_f, ret=ret_f, adv=adv_f, val=val_f)

def ppo_update(ac: ActorCritic, opt: torch.optim.Optimizer, batch: RolloutBatch, cfg: Cfg):
    ac.train()
    logp, v, ent = ac.logp_value_entropy(batch.obs, batch.act)

    ratio = torch.exp(logp - batch.logp)
    surr1 = ratio * batch.adv
    surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * batch.adv
    pi_loss = -torch.min(surr1, surr2).mean()

    v_loss = F.mse_loss(v, batch.ret)
    ent_loss = -ent.mean()

    loss = pi_loss + cfg.ppo_vf_coef * v_loss + cfg.ppo_ent_coef * ent_loss

    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), cfg.ppo_max_grad_norm)
    opt.step()

    return float(loss.item()), float(pi_loss.item()), float(v_loss.item()), float(ent.mean().item())

def save_output(save_path: Path, cfg: Cfg, wm, re, ac, best_return: float, epoch: int):
    payload = {
        "cfg": asdict(cfg),
        "epoch": epoch,
        "best_return": best_return,
        "wm": wm.state_dict(),
        "re": re.state_dict(),
        "ac": ac.state_dict(),
    }
    torch.save(payload, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train"])
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="diamond-onefile")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = Cfg()
    cfg.env_id = args.env
    cfg.device = args.device
    cfg.epochs = args.epochs
    cfg.use_wandb = bool(args.wandb)
    cfg.wandb_project = args.wandb_project
    cfg.wandb_run_name = args.wandb_run_name

    set_seed(cfg.seed)
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cfg.device = device

    # Output dirs
    root = Path(cfg.save_dir) / now_tag()
    ckpt_dir = root / "outputs"
    ensure_dir(ckpt_dir)
    meta_path = root / "run_meta.json"
    meta_path.write_text(json.dumps(asdict(cfg), indent=2))

    # W&B
    if cfg.use_wandb:
        if not _WANDB_OK:
            raise RuntimeError("wandb requested but not installed.")
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    env = make_env(cfg)
    obs_shape = env.observation_space.shape  # (C,H,W)
    n_actions = env.action_space.n

    # Models
    wm = DiffusionWorldModel(obs_ch=obs_shape[0], n_actions=n_actions, cfg=cfg).to(cfg.device)
    re = RewEndModel(obs_ch=obs_shape[0], n_actions=n_actions).to(cfg.device)
    ac = ActorCritic(obs_ch=obs_shape[0], n_actions=n_actions).to(cfg.device)

    # Opts
    wm_opt = torch.optim.AdamW(wm.parameters(), lr=cfg.wm_lr)
    re_opt = torch.optim.AdamW(re.parameters(), lr=cfg.re_lr)
    ac_opt = torch.optim.AdamW(ac.parameters(), lr=cfg.ppo_lr)

    # Buffer
    rb = ReplayBuffer(cfg.replay_size, obs_shape)

    # Initial collection with random-ish policy
    print(f"Device: {cfg.device}")
    print(f"Env: {cfg.env_id}")
    print("Collecting initial experience...")
    collect_real(env, rb, ac, cfg.device, cfg.collect_steps_initial, eps=1.0)

    best_ret = -1e9
    global_step = 0

    for epoch in trange(cfg.epochs, desc="Epochs"):
        t0 = time.time()

        # Anneal exploration (simple)
        eps = max(0.05, 1.0 - epoch / max(1, cfg.epochs * 0.7))

        # Collect real data
        collect_real(env, rb, ac, cfg.device, cfg.collect_steps_per_epoch, eps=eps)
        global_step += cfg.collect_steps_per_epoch

        # Train WM
        wm_loss = train_world_model(wm, wm_opt, rb, cfg)

        # Train reward/end
        re_loss, re_r, re_d = train_rew_end(re, re_opt, rb, cfg)

        # Imagination rollouts + PPO
        # Sample start states from replay
        batch = rb.sample(cfg.imagine_batch_envs)
        start_obs = to_torch(batch["obs"], cfg.device)
        rollout = imagine_rollout(wm, re, ac, start_obs, cfg)

        ppo_losses = []
        for _ in trange(cfg.ppo_steps, desc="PPO"):
            l = ppo_update(ac, ac_opt, rollout, cfg)
            ppo_losses.append(l[0])

        # Evaluate in real env
        mean_ret, std_ret = evaluate_real_env(env, ac, cfg.device, cfg.eval_episodes)

        # output
        if mean_ret > best_ret:
            best_ret = mean_ret
            save_output(ckpt_dir / "best.pt", cfg, wm, re, ac, best_ret, epoch)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.epochs} | steps={global_step} | wm={wm_loss:.4f} | re={re_loss:.4f} "
              f"| ppo={float(np.mean(ppo_losses)):.4f} | eval={mean_ret:.2f}±{std_ret:.2f} | best={best_ret:.2f} | {dt:.1f}s")

        if cfg.use_wandb:
            wandb_log({
                "epoch": epoch,
                "env_steps": global_step,
                "wm_loss": wm_loss,
                "re_loss": re_loss,
                "re_loss_reward": re_r,
                "re_loss_done": re_d,
                "ppo_loss": float(np.mean(ppo_losses)),
                "eval_return_mean": mean_ret,
                "eval_return_std": std_ret,
                "best_return": best_ret,
                "eps_explore": eps,
                "time_sec": dt,
            }, step=global_step)

    print("Training complete.")
    print(f"Best output: {ckpt_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
