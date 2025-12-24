# diamond_onefile.py
# Single-file DIAMOND-style training (diffusion world model + reward/end model + actor-critic in world model env)
# Notes:
# - Preserves DIAMOND pipeline structure in one file.
# - Avoids torcheval/torchaudio pitfalls (no confusion-matrix logging).
# - Uses Gymnasium Atari. Requires AutoROM --accept-license.
#
# This is DIAMOND-style, not byte-identical to the original repo (Hydra/DDP/dataset plumbing differs).

import os
import math
import time
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from gym.spaces import Box


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 255).to(torch.uint8)

def normalize_obs_uint8(x_u8: torch.Tensor) -> torch.Tensor:
    # uint8 -> [-1, 1]
    return (x_u8.float() / 255.0) * 2.0 - 1.0

def denormalize_obs(x: torch.Tensor) -> torch.Tensor:
    # [-1, 1] -> uint8
    return to_uint8(((x + 1.0) * 0.5) * 255.0)

def clip_reward_sign(r: torch.Tensor) -> torch.Tensor:
    return r.sign().clamp(-1, 1)

def save_ckpt(path: Path, payload: Dict[str, Any]):
    ensure_dir(path.parent)
    torch.save(payload, path)

def load_ckpt(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


# -------------------------
# Atari Preprocessing (derived from Gym wrapper; simplified)
# -------------------------

class AtariPreprocessing(gym.Wrapper):
    """
    Minimal Atari preprocessing:
    - Noop reset
    - Frame skip
    - Max pool over last two frames
    - Resize to screen_size
    - Return RGB uint8
    """
    def __init__(self, env: gym.Env, noop_max: int = 30, frame_skip: int = 4, screen_size: int = 64):
        super().__init__(env)
        assert frame_skip > 0 and screen_size > 0 and noop_max >= 0
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        assert isinstance(env.observation_space, Box)
        self.obs_buffer = [
            np.empty(env.observation_space.shape, dtype=np.uint8),
            np.empty(env.observation_space.shape, dtype=np.uint8),
        ]
        self.lives = 0
        self.game_over = False
        self.observation_space = Box(low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8)

    @property
    def ale(self):
        return self.env.unwrapped.ale

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.lives = self.ale.lives()
        # Noop reset
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            obs, _, terminated, truncated, info2 = self.env.step(0)
            info.update(info2)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)

        self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs(), info

    def step(self, action):
        import cv2
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for t in range(self.frame_skip):
            _, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            # collect last two frames for maxpool
            if t == self.frame_skip - 2:
                self.ale.getScreenRGB(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                self.ale.getScreenRGB(self.obs_buffer[0])

        # max-pool
        np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        frame = self.obs_buffer[0]
        frame = cv2.resize(frame, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        frame = np.asarray(frame, dtype=np.uint8)
        return frame, float(total_reward), bool(terminated), bool(truncated), info

    def _get_obs(self):
        import cv2
        frame = self.obs_buffer[0]
        frame = cv2.resize(frame, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        return np.asarray(frame, dtype=np.uint8)


class FrameStack(gym.Wrapper):
    """Stack last N frames along time dimension: obs shape (N,H,W,3) uint8."""
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), rew, term, trunc, info

    def _get_obs(self):
        return np.array(self.frames, dtype=np.uint8)  # (N,H,W,3)


def make_atari_env(env_name: str, screen_size=64, frame_skip=4, noop_max=30, frame_stack=4, render_mode=None, seed=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = AtariPreprocessing(env, noop_max=noop_max, frame_skip=frame_skip, screen_size=screen_size)
    env = FrameStack(env, num_stack=frame_stack)
    if seed is not None:
        env.reset(seed=seed)
    return env


# -------------------------
# Dataset / Replay (episode-aware)
# -------------------------

@dataclass
class Batch:
    obs: torch.Tensor         # [B, T, FS, H, W, C] uint8
    act: torch.Tensor         # [B, T] long
    rew: torch.Tensor         # [B, T] float
    end: torch.Tensor         # [B, T] long (0/1)
    trunc: torch.Tensor       # [B, T] long (0/1)


class EpisodeBuffer:
    """Stores full episodes, samples fixed-length segments."""
    def __init__(self, capacity_steps: int, obs_shape: Tuple[int, ...], num_actions: int, device: torch.device):
        self.capacity_steps = int(capacity_steps)
        self.obs_shape = obs_shape  # (FS,H,W,C)
        self.num_actions = num_actions
        self.device = device

        self.episodes: List[Dict[str, Any]] = []
        self.total_steps = 0

    def add_episode(self, obs_seq: np.ndarray, act_seq: np.ndarray, rew_seq: np.ndarray, end_seq: np.ndarray, trunc_seq: np.ndarray):
        # obs_seq: (L,FS,H,W,C) uint8
        ep = dict(obs=obs_seq, act=act_seq, rew=rew_seq, end=end_seq, trunc=trunc_seq)
        self.episodes.append(ep)
        self.total_steps += len(obs_seq)

        # crude eviction by steps
        while self.total_steps > self.capacity_steps and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed["obs"])

    def __len__(self):
        return self.total_steps

    def sample(self, batch_size: int, seq_len: int) -> Optional[Batch]:
        if len(self.episodes) == 0:
            return None
        # choose episodes with enough length
        valid = [ep for ep in self.episodes if len(ep["obs"]) >= seq_len]
        if not valid:
            return None

        obs_b, act_b, rew_b, end_b, trunc_b = [], [], [], [], []
        for _ in range(batch_size):
            ep = valid[np.random.randint(len(valid))]
            L = len(ep["obs"])
            start = np.random.randint(0, L - seq_len + 1)
            sl = slice(start, start + seq_len)

            obs_b.append(ep["obs"][sl])
            act_b.append(ep["act"][sl])
            rew_b.append(ep["rew"][sl])
            end_b.append(ep["end"][sl])
            trunc_b.append(ep["trunc"][sl])

        obs = torch.from_numpy(np.stack(obs_b)).to(self.device)     # uint8
        act = torch.from_numpy(np.stack(act_b)).long().to(self.device)
        rew = torch.from_numpy(np.stack(rew_b)).float().to(self.device)
        end = torch.from_numpy(np.stack(end_b)).long().to(self.device)
        trunc = torch.from_numpy(np.stack(trunc_b)).long().to(self.device)
        return Batch(obs=obs, act=act, rew=rew, end=end, trunc=trunc)


# -------------------------
# Building blocks (DIAMOND-style UNet)
# -------------------------

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        y = self.norm(x)
        qkv = self.qkv(y).view(n, self.n_head, 3, c // self.n_head, h * w)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [n, head, ch, hw]
        q = q.transpose(-2, -1)  # [n, head, hw, ch]
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(-2, -1).reshape(n, c, h, w)
        return x + self.out(out)

class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: [B]
        f = 2 * math.pi * sigma.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

class Downsample(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x): return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, cond_c: int, attn: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.norm1 = AdaGroupNorm(in_c, cond_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_c, cond_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.attn = SelfAttention2d(out_c) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x, cond):
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        return self.attn(x)

class ResBlocks(nn.Module):
    def __init__(self, list_in: List[int], list_out: List[int], cond_c: int, attn: bool) -> None:
        super().__init__()
        assert len(list_in) == len(list_out)
        self.blocks = nn.ModuleList([ResBlock(i, o, cond_c, attn) for i, o in zip(list_in, list_out)])

    def forward(self, x, cond, to_cat: Optional[List[torch.Tensor]] = None):
        outs = []
        for i, b in enumerate(self.blocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = b(x, cond)
            outs.append(x)
        return x, outs

class UNet(nn.Module):
    def __init__(self, cond_c: int, depths: List[int], channels: List[int], attn_depths: List[int]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.num_down = len(channels) - 1

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(ResBlocks([c1] + [c2] * (n - 1), [c2] * n, cond_c, bool(attn_depths[i])))
            u_blocks.append(ResBlocks([2 * c2] * n + [c1 + c2], [c2] * n + [c1], cond_c, bool(attn_depths[i])))

        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(list(reversed(u_blocks)))

        self.mid = ResBlocks([channels[-1]] * 2, [channels[-1]] * 2, cond_c, attn=True)

        self.downs = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]])
        self.ups = nn.ModuleList([nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])])

    def forward(self, x, cond):
        *_, h, w = x.size()
        n = self.num_down
        pad_h = math.ceil(h / (2 ** n)) * (2 ** n) - h
        pad_w = math.ceil(w / (2 ** n)) * (2 ** n) - w
        x = F.pad(x, (0, pad_w, 0, pad_h))

        d_outs = []
        for block, down in zip(self.d_blocks, self.downs):
            x_down = down(x)
            x, block_outs = block(x_down, cond)
            d_outs.append((x_down, *block_outs))

        x, _ = self.mid(x, cond)

        for block, up, skip in zip(self.u_blocks, self.ups, reversed(d_outs)):
            x_up = up(x)
            x, _ = block(x_up, cond, list(skip[::-1]))

        x = x[..., :h, :w]
        return x


# -------------------------
# Denoiser (diffusion world model)
# -------------------------

@dataclass
class DenoiserConfig:
    img_channels: int         # channels per frame (3)
    frame_stack: int          # FS (4)
    cond_channels: int        # conditioning embed dim
    depths: List[int]
    channels: List[int]
    attn_depths: List[int]
    num_actions: int

class Denoiser(nn.Module):
    """
    EDM-style denoiser:
    - inputs: noisy next frame x (B,C,H,W), sigma (B), prev_obs (B,FS*C,H,W), prev_act (B)
    - output: predicted denoised x0 (B,C,H,W)
    """
    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        self.img_c = cfg.img_channels * cfg.frame_stack  # conditioning obs channels (stacked)
        self.out_c = cfg.img_channels * cfg.frame_stack  # predict next stacked obs
        self.act_emb = nn.Embedding(cfg.num_actions, cfg.cond_channels)
        self.sigma_emb = nn.Sequential(
            FourierFeatures(cfg.cond_channels),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        # UNet operates on concatenation of [noisy_next, prev_obs] along channels
        in_c = self.out_c + self.img_c
        self.conv_in = nn.Conv2d(in_c, cfg.channels[0], 3, padding=1)
        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)
        self.conv_out = nn.Conv2d(cfg.channels[0], self.out_c, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    @property
    def device(self): return next(self.parameters()).device

    def denoise(self, x_noisy: torch.Tensor, sigma: torch.Tensor, prev_obs: torch.Tensor, prev_act: torch.Tensor) -> torch.Tensor:
        # cond vector
        cond = self.sigma_emb(sigma) + self.act_emb(prev_act)
        x = torch.cat([x_noisy, prev_obs], dim=1)
        x = self.conv_in(x)
        x = self.unet(x, cond)
        x = self.conv_out(x)
        return x

    def forward(self, batch: Batch, sigma_min=2e-3, sigma_max=5.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # batch.obs: [B,T,FS,H,W,C] uint8
        # We train on one-step transitions inside sequences:
        # prev_obs = obs[:,t] , next_obs = obs[:,t+1], act=act[:,t]
        obs_u8 = batch.obs
        act = batch.act
        end = batch.end

        B, T, FS, H, W, C = obs_u8.shape
        assert FS * C == self.img_c

        # pick random t (avoid last)
        t_idx = torch.randint(0, T - 1, (B,), device=self.device)
        prev = obs_u8[torch.arange(B, device=self.device), t_idx]      # [B,FS,H,W,C]
        nxt  = obs_u8[torch.arange(B, device=self.device), t_idx + 1]  # [B,FS,H,W,C]
        a    = act[torch.arange(B, device=self.device), t_idx]

        # reshape to channel-first stacked
        prev = prev.permute(0, 1, 4, 2, 3).reshape(B, FS * C, H, W)  # [B,FS*C,H,W]
        nxt  = nxt.permute(0, 1, 4, 2, 3).reshape(B, FS * C, H, W)

        x0 = normalize_obs_uint8(nxt)

        # sample sigma (log-uniform)
        u = torch.rand(B, device=self.device)
        sigma = sigma_min * (sigma_max / sigma_min) ** u  # [B]
        noise = torch.randn_like(x0)
        x_noisy = x0 + noise * sigma[:, None, None, None]

        x_pred = self.denoise(x_noisy, sigma, normalize_obs_uint8(prev), a)

        # weighted MSE (EDM-ish)
        weight = 1.0 / (sigma ** 2 + 1e-8)
        loss = ((x_pred - x0) ** 2).mean(dim=(1,2,3)) * weight
        loss = loss.mean()

        return loss, {"loss_total": loss.detach()}


# -------------------------
# Reward + End model (learned)
# -------------------------

@dataclass
class RewEndConfig:
    img_channels: int
    frame_stack: int
    cond_channels: int
    lstm_dim: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[int]
    num_actions: int

class RewEndEncoder(nn.Module):
    def __init__(self, in_c: int, cond_c: int, depths: List[int], channels: List[int], attn_depths: List[int]) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_c, channels[0], 3, padding=1)
        blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            blocks.append(ResBlocks([c1] + [c2] * (n - 1), [c2] * n, cond_c, bool(attn_depths[i])))
        blocks.append(ResBlocks([channels[-1]] * 2, [channels[-1]] * 2, cond_c, attn=True))
        self.blocks = nn.ModuleList(blocks)
        self.downs = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]] + [nn.Identity()])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for b, d in zip(self.blocks, self.downs):
            x = d(x)
            x, _ = b(x, cond)
        return x

class RewEndModel(nn.Module):
    """
    Predict clipped reward (-1,0,+1) and end (0/1) from (obs, act, next_obs) sequences.
    """
    def __init__(self, cfg: RewEndConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.img_c = cfg.img_channels * cfg.frame_stack
        self.act_emb = nn.Embedding(cfg.num_actions, cfg.cond_channels)
        self.encoder = RewEndEncoder(2 * self.img_c, cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)
        enc_out_dim = cfg.channels[-1] * (64 // (2 ** (len(cfg.depths) - 1))) ** 2
        self.lstm = nn.LSTM(enc_out_dim, cfg.lstm_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
            nn.SiLU(),
            nn.Linear(cfg.lstm_dim, 3 + 2, bias=False),
        )

    @property
    def device(self): return next(self.parameters()).device

    def predict(self, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor, hx_cx=None):
        # obs,next_obs: [B,T,C,H,W] normalized
        B, T, C, H, W = obs.shape
        x = torch.cat([obs, next_obs], dim=2).reshape(B*T, 2*C, H, W)
        a = act.reshape(B*T)
        cond = self.act_emb(a)
        z = self.encoder(x, cond).reshape(B, T, -1)
        z, hx_cx = self.lstm(z, hx_cx)
        logits = self.head(z)
        logits_rew = logits[:, :, :3]
        logits_end = logits[:, :, 3:]
        return logits_rew, logits_end, hx_cx

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs_u8 = batch.obs
        act = batch.act
        rew = batch.rew
        end = batch.end

        B, T, FS, H, W, C = obs_u8.shape
        # build obs[t], next_obs[t] for t=0..T-2
        obs0 = obs_u8[:, :-1].permute(0,1,2,5,3,4).reshape(B, T-1, FS*C, H, W)
        obs1 = obs_u8[:, 1:].permute(0,1,2,5,3,4).reshape(B, T-1, FS*C, H, W)
        a = act[:, :-1]

        obs0 = normalize_obs_uint8(obs0)
        obs1 = normalize_obs_uint8(obs1)

        logits_rew, logits_end, _ = self.predict(obs0, a, obs1)

        # targets
        target_rew = clip_reward_sign(rew[:, :-1]).long().add(1)  # {-1,0,1}->{0,1,2}
        target_end = end[:, :-1].long()                           # {0,1}

        loss_rew = F.cross_entropy(logits_rew.reshape(-1,3), target_rew.reshape(-1))
        loss_end = F.cross_entropy(logits_end.reshape(-1,2), target_end.reshape(-1))
        loss = loss_rew + loss_end
        return loss, {"loss_total": loss.detach(), "loss_rew": loss_rew.detach(), "loss_end": loss_end.detach()}


# -------------------------
# Diffusion sampler (Karras-ish)
# -------------------------

@dataclass
class SamplerCfg:
    num_steps: int = 10
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: int = 7

def build_sigmas(cfg: SamplerCfg, device: torch.device) -> torch.Tensor:
    min_inv = cfg.sigma_min ** (1 / cfg.rho)
    max_inv = cfg.sigma_max ** (1 / cfg.rho)
    t = torch.linspace(0, 1, cfg.num_steps, device=device)
    sigmas = (max_inv + t * (min_inv - max_inv)) ** cfg.rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])

@torch.no_grad()
def sample_next(denoiser: Denoiser, prev_obs: torch.Tensor, prev_act: torch.Tensor, cfg: SamplerCfg) -> torch.Tensor:
    """
    prev_obs: [B, FS*C, H, W] normalized
    prev_act: [B] long
    returns next_obs: [B, FS*C, H, W] normalized
    """
    device = prev_obs.device
    B, C, H, W = prev_obs.shape
    sigmas = build_sigmas(cfg, device=device)
    x = torch.randn(B, C, H, W, device=device) * sigmas[0]
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i].expand(B)
        sigma_next = sigmas[i+1].expand(B)
        x0 = denoiser.denoise(x, sigma, prev_obs, prev_act)
        # Euler step on ODE: dx/dsigma ~ (x - x0)/sigma
        d = (x - x0) / (sigma[:, None, None, None] + 1e-8)
        x = x + (sigma_next - sigma)[:, None, None, None] * d
    return x


# -------------------------
# Actor-Critic (CNN + LSTM) trained inside WorldModelEnv
# -------------------------

@dataclass
class ACfg:
    img_channels: int
    frame_stack: int
    img_size: int
    channels: List[int]
    down: List[int]
    lstm_dim: int
    num_actions: int
    gamma: float = 0.995
    lambda_: float = 0.95
    w_value: float = 0.5
    w_entropy: float = 0.01

class SmallResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.f = nn.Sequential(GroupNorm(in_c), nn.SiLU(inplace=True), nn.Conv2d(in_c, out_c, 3, padding=1))
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)

    def forward(self, x): return self.skip(x) + self.f(x)

class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ACfg):
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        layers = [nn.Conv2d(cfg.img_channels * cfg.frame_stack, cfg.channels[0], 3, padding=1)]
        for i in range(len(cfg.channels)):
            layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, cfg: ACfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = ActorCriticEncoder(cfg)
        # compute feature dim
        down_pow = sum(cfg.down)
        feat_hw = cfg.img_size // (2 ** down_pow)
        feat_dim = cfg.channels[-1] * feat_hw * feat_hw
        self.lstm = nn.LSTMCell(feat_dim, cfg.lstm_dim)
        self.pi = nn.Linear(cfg.lstm_dim, cfg.num_actions)
        self.v  = nn.Linear(cfg.lstm_dim, 1)
        nn.init.zeros_(self.pi.weight); nn.init.zeros_(self.pi.bias)
        nn.init.zeros_(self.v.weight);  nn.init.zeros_(self.v.bias)

    @property
    def device(self): return next(self.parameters()).device

    def step(self, obs: torch.Tensor, hx_cx: Tuple[torch.Tensor, torch.Tensor]):
        # obs: [B,C,H,W] normalized
        z = self.encoder(obs).flatten(1)
        hx, cx = self.lstm(z, hx_cx)
        logits = self.pi(hx)
        val = self.v(hx).squeeze(1)
        return logits, val, (hx, cx)

def lambda_returns(rew, end, trunc, v_boot, gamma, lam):
    # rew,end,trunc,v_boot: [B,T]
    rew = clip_reward_sign(rew)
    end_or_trunc = (end + trunc).clamp(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc
    ret = rew + not_end * gamma * (not_trunc * (1 - lam) + trunc) * v_boot
    if lam == 0: return ret
    last = v_boot[:, -1]
    for t in reversed(range(rew.size(1))):
        ret[:, t] += (end_or_trunc[:, t] == 0).float() * gamma * lam * last
        last = ret[:, t]
    return ret


# -------------------------
# WorldModelEnv
# -------------------------

class WorldModelEnv:
    """
    Vectorized env driven by:
      - denoiser (predict next obs)
      - rew_end_model (predict reward/end)
    Reset samples random conditioning slices from dataset buffer.
    """
    def __init__(self, denoiser: Denoiser, rew_end: RewEndModel, data: EpisodeBuffer, sampler_cfg: SamplerCfg, batch_size: int, img_size: int, frame_stack: int, img_channels: int, device: torch.device):
        self.denoiser = denoiser
        self.rew_end = rew_end
        self.data = data
        self.sampler_cfg = sampler_cfg
        self.B = batch_size
        self.device = device
        self.C = frame_stack * img_channels
        self.H = img_size
        self.W = img_size
        self.reset()

    def reset(self):
        # sample a batch of single states from dataset
        # we use seq_len=2 to get obs_t and obs_{t+1} as potential "state" seed
        b = self.data.sample(batch_size=self.B, seq_len=2)
        if b is None:
            # cold start: random noise state
            self.obs = torch.zeros(self.B, self.C, self.H, self.W, device=self.device)
        else:
            o = b.obs[:, 0].permute(0,1,4,2,3).reshape(self.B, self.C, self.H, self.W)
            self.obs = normalize_obs_uint8(o)
        self.hx = torch.zeros(self.B, acfg.lstm_dim, device=self.device)
        self.cx = torch.zeros(self.B, acfg.lstm_dim, device=self.device)
        return self.obs

    @torch.no_grad()
    def step(self, act: torch.Tensor):
        # act: [B]
        prev = self.obs
        nxt = sample_next(self.denoiser, prev, act, self.sampler_cfg)  # normalized

        # reward/end prediction expects sequence; we pass T=1
        obs1 = prev.unsqueeze(1)
        obs2 = nxt.unsqueeze(1)
        a = act.unsqueeze(1)
        logits_rew, logits_end, _ = self.rew_end.predict(obs1, a, obs2)
        # sample discrete outcomes
        rew_cls = torch.distributions.Categorical(logits=logits_rew.squeeze(1)).sample()  # {0,1,2}
        end_cls = torch.distributions.Categorical(logits=logits_end.squeeze(1)).sample()  # {0,1}
        rew = (rew_cls - 1).float()
        end = end_cls.float()
        trunc = torch.zeros_like(end)

        self.obs = nxt
        return nxt, rew, end, trunc


# -------------------------
# Training loops
# -------------------------

@dataclass
class TrainCfg:
    env_name: str = "BreakoutNoFrameskip-v4"
    img_size: int = 64
    frame_stack: int = 4
    img_channels: int = 3
    num_actions: int = 4

    seed: int = 42
    device: str = "cuda"

    # data collection
    collect_steps_initial: int = 20_000
    collect_steps_per_epoch: int = 5_000
    epsilon: float = 0.1

    # dataset / sequences
    capacity_steps: int = 200_000
    seq_len_wm: int = 16
    batch_wm: int = 32

    # training schedule
    epochs: int = 50
    steps_wm: int = 300
    steps_rew_end: int = 200
    steps_ac: int = 200
    ac_batch_envs: int = 32
    ac_rollout: int = 32

    # optimization
    lr_wm: float = 1e-4
    lr_rew_end: float = 1e-4
    lr_ac: float = 3e-4
    wd: float = 1e-6
    grad_clip: float = 10.0

    # logging/ckpt
    out_dir: str = "outputs_onefile"
    wandb: bool = False
    project: str = "diamond-onefile"
    run_name: str = None


def collect_real_data(env, dataset: EpisodeBuffer, num_steps: int, epsilon: float, device: torch.device, policy: Optional[ActorCritic] = None):
    obs, _ = env.reset()
    ep_obs, ep_act, ep_rew, ep_end, ep_trunc = [], [], [], [], []
    steps = 0
    episodes = 0

    # AC hidden state is not used in data collection; we use epsilon-random unless policy provided
    hx = cx = None

    while steps < num_steps:
        if np.random.rand() < epsilon or policy is None:
            act = env.action_space.sample()
        else:
            # policy expects [B,C,H,W] with channels=FS*C
            o = torch.from_numpy(obs).to(device)                       # [FS,H,W,3]
            o = o.permute(0,3,1,2).reshape(1, -1, o.size(1), o.size(2)) # [1,FS*C,H,W]
            o = normalize_obs_uint8(o)
            if hx is None:
                hx = torch.zeros(1, policy.cfg.lstm_dim, device=device)
                cx = torch.zeros(1, policy.cfg.lstm_dim, device=device)
            with torch.no_grad():
                logits, val, (hx, cx) = policy.step(o, (hx, cx))
                dist = torch.distributions.Categorical(logits=logits)
                act = int(dist.sample().item())

        next_obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        # store transition at time t (we store obs stack)
        ep_obs.append(obs.copy())
        ep_act.append(act)
        ep_rew.append(float(reward))
        ep_end.append(1 if terminated else 0)
        ep_trunc.append(1 if truncated else 0)

        obs = next_obs
        steps += 1

        if done:
            # last observation (optional): we do not append terminal next frame; consistent with many datasets
            dataset.add_episode(
                obs_seq=np.asarray(ep_obs, dtype=np.uint8),
                act_seq=np.asarray(ep_act, dtype=np.int64),
                rew_seq=np.asarray(ep_rew, dtype=np.float32),
                end_seq=np.asarray(ep_end, dtype=np.int64),
                trunc_seq=np.asarray(ep_trunc, dtype=np.int64),
            )
            episodes += 1
            obs, _ = env.reset()
            ep_obs, ep_act, ep_rew, ep_end, ep_trunc = [], [], [], [], []
            hx = cx = None

    # flush partial episode
    if len(ep_obs) > 5:
        dataset.add_episode(
            obs_seq=np.asarray(ep_obs, dtype=np.uint8),
            act_seq=np.asarray(ep_act, dtype=np.int64),
            rew_seq=np.asarray(ep_rew, dtype=np.float32),
            end_seq=np.asarray(ep_end, dtype=np.int64),
            trunc_seq=np.asarray(ep_trunc, dtype=np.int64),
        )
        episodes += 1

    return {"steps": steps, "episodes": episodes, "dataset_steps": len(dataset)}


def train_onefile(cfg: TrainCfg):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    out_root = Path(cfg.out_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    ensure_dir(out_root)
    ckpt_path = out_root / "best.pt"
    meta_path = out_root / "run_meta.json"

    # (Optional) wandb
    if cfg.wandb:
        import wandb
        wandb.init(project=cfg.project, name=cfg.run_name, config=cfg.__dict__)

    env = make_atari_env(cfg.env_name, screen_size=cfg.img_size, frame_stack=cfg.frame_stack, seed=cfg.seed)

    dataset = EpisodeBuffer(
        capacity_steps=cfg.capacity_steps,
        obs_shape=(cfg.frame_stack, cfg.img_size, cfg.img_size, cfg.img_channels),
        num_actions=cfg.num_actions,
        device=device
    )

    # Models
    den_cfg = DenoiserConfig(
        img_channels=cfg.img_channels,
        frame_stack=cfg.frame_stack,
        cond_channels=256,
        depths=[2,2,2],
        channels=[128,256,512],
        attn_depths=[0,0,1],
        num_actions=cfg.num_actions,
    )
    rew_cfg = RewEndConfig(
        img_channels=cfg.img_channels,
        frame_stack=cfg.frame_stack,
        cond_channels=256,
        lstm_dim=256,
        depths=[2,2,2],
        channels=[128,256,512],
        attn_depths=[0,0,1],
        num_actions=cfg.num_actions,
    )
    global acfg
    acfg = ACfg(
        img_channels=cfg.img_channels,
        frame_stack=cfg.frame_stack,
        img_size=cfg.img_size,
        channels=[64,128,256],
        down=[1,1,1],
        lstm_dim=256,
        num_actions=cfg.num_actions
    )

    denoiser = Denoiser(den_cfg).to(device)
    rew_end  = RewEndModel(rew_cfg).to(device)
    actor    = ActorCritic(acfg).to(device)

    opt_wm  = torch.optim.AdamW(denoiser.parameters(), lr=cfg.lr_wm, weight_decay=cfg.wd)
    opt_rew = torch.optim.AdamW(rew_end.parameters(),  lr=cfg.lr_rew_end, weight_decay=cfg.wd)
    opt_ac  = torch.optim.AdamW(actor.parameters(),    lr=cfg.lr_ac, weight_decay=cfg.wd)

    sampler_cfg = SamplerCfg(num_steps=10, sigma_min=2e-3, sigma_max=5.0, rho=7)

    # Initial collection
    print(f"[Collect] initial steps={cfg.collect_steps_initial}")
    collect_real_data(env, dataset, cfg.collect_steps_initial, cfg.epsilon, device, policy=None)
    print(f"[Dataset] steps={len(dataset)} episodes={len(dataset.episodes)}")

    best_return = -1e9

    # Training epochs
    for epoch in range(cfg.epochs):
        t0 = time.time()

        # more collection using current actor (optional)
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        col = collect_real_data(env, dataset, cfg.collect_steps_per_epoch, cfg.epsilon, device, policy=actor)
        print(f"  Collected: {col}")

        # Train denoiser
        denoiser.train()
        wm_losses = []
        for _ in range(cfg.steps_wm):
            b = dataset.sample(cfg.batch_wm, cfg.seq_len_wm)
            if b is None:
                continue
            loss, metrics = denoiser(b)
            opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), cfg.grad_clip)
            opt_wm.step()
            wm_losses.append(metrics["loss_total"].item())
        wm_loss = float(np.mean(wm_losses)) if wm_losses else 0.0

        # Train reward/end
        rew_end.train()
        re_losses = []
        for _ in range(cfg.steps_rew_end):
            b = dataset.sample(cfg.batch_wm, cfg.seq_len_wm)
            if b is None:
                continue
            loss, metrics = rew_end(b)
            opt_rew.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rew_end.parameters(), cfg.grad_clip)
            opt_rew.step()
            re_losses.append(metrics["loss_total"].item())
        re_loss = float(np.mean(re_losses)) if re_losses else 0.0

        # Train actor-critic in WorldModelEnv
        actor.train()
        wm_env = WorldModelEnv(denoiser, rew_end, dataset, sampler_cfg, cfg.ac_batch_envs, cfg.img_size, cfg.frame_stack, cfg.img_channels, device)
        ac_losses = []
        for _ in range(cfg.steps_ac):
            # rollout inside world model env
            B = cfg.ac_batch_envs
            hx = torch.zeros(B, acfg.lstm_dim, device=device)
            cx = torch.zeros(B, acfg.lstm_dim, device=device)
            obs = wm_env.reset()  # [B,C,H,W] normalized

            obs_seq, act_seq, rew_seq, end_seq, trunc_seq, val_seq, logp_seq, ent_seq = [], [], [], [], [], [], [], []
            for t in range(cfg.ac_rollout):
                logits, val, (hx, cx) = actor.step(obs, (hx, cx))
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
                ent = dist.entropy()

                next_obs, rew, end, trunc = wm_env.step(act)

                obs_seq.append(obs)
                act_seq.append(act)
                rew_seq.append(rew)
                end_seq.append(end)
                trunc_seq.append(trunc)
                val_seq.append(val)
                logp_seq.append(logp)
                ent_seq.append(ent)

                obs = next_obs

            # bootstrap value
            with torch.no_grad():
                logits_b, v_boot, _ = actor.step(obs, (hx, cx))
            v_boot = v_boot.detach()

            # stack
            act_t = torch.stack(act_seq, dim=1)       # [B,T]
            rew_t = torch.stack(rew_seq, dim=1)       # [B,T]
            end_t = torch.stack(end_seq, dim=1)       # [B,T]
            trunc_t = torch.stack(trunc_seq, dim=1)   # [B,T]
            val_t = torch.stack(val_seq, dim=1)       # [B,T]
            logp_t = torch.stack(logp_seq, dim=1)     # [B,T]
            ent_t = torch.stack(ent_seq, dim=1)       # [B,T]

            v_boot_all = torch.cat([val_t[:, 1:], v_boot.unsqueeze(1)], dim=1)  # [B,T]
            lamret = lambda_returns(rew_t, end_t, trunc_t, v_boot_all, acfg.gamma, acfg.lambda_)

            adv = (lamret - val_t).detach()
            loss_pi = -(logp_t * adv).mean()
            loss_v = F.mse_loss(val_t, lamret.detach())
            loss_ent = -ent_t.mean()
            loss = loss_pi + acfg.w_value * loss_v + acfg.w_entropy * loss_ent

            opt_ac.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
            opt_ac.step()
            ac_losses.append(loss.item())

        ac_loss = float(np.mean(ac_losses)) if ac_losses else 0.0

        # Quick eval in REAL env (policy only)
        mean_ret = evaluate_real(env, actor, device, episodes=3)
        elapsed = time.time() - t0
        print(f"  WM loss: {wm_loss:.4f} | RewEnd loss: {re_loss:.4f} | AC loss: {ac_loss:.4f} | Eval return: {mean_ret:.2f} | {elapsed:.1f}s")

        if cfg.wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "wm_loss": wm_loss,
                "rew_end_loss": re_loss,
                "ac_loss": ac_loss,
                "eval_return": mean_ret,
                "dataset_steps": len(dataset),
                "dataset_episodes": len(dataset.episodes),
                "time_sec": elapsed
            })

        # Save best
        if mean_ret > best_return:
            best_return = mean_ret
            save_ckpt(ckpt_path, {
                "epoch": epoch,
                "cfg": cfg.__dict__,
                "best_return": best_return,
                "denoiser": denoiser.state_dict(),
                "rew_end": rew_end.state_dict(),
                "actor": actor.state_dict(),
                "den_cfg": den_cfg.__dict__,
                "rew_cfg": rew_cfg.__dict__,
                "acfg": acfg.__dict__,
            })
            print(f"  Saved best -> {ckpt_path}")

        # save metadata
        meta = {
            "best_return": best_return,
            "ckpt": str(ckpt_path),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDone. Best return: {best_return:.2f}")
    if cfg.wandb:
        import wandb
        wandb.finish()

    env.close()
    return str(ckpt_path)


@torch.no_grad()
def evaluate_real(env, actor: ActorCritic, device: torch.device, episodes: int = 5):
    actor.eval()
    rets = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        hx = torch.zeros(1, actor.cfg.lstm_dim, device=device)
        cx = torch.zeros(1, actor.cfg.lstm_dim, device=device)
        while not done:
            o = torch.from_numpy(obs).to(device)                         # [FS,H,W,3]
            o = o.permute(0,3,1,2).reshape(1, -1, o.size(1), o.size(2))
            o = normalize_obs_uint8(o)
            logits, val, (hx, cx) = actor.step(o, (hx, cx))
            act = logits.argmax(dim=-1).item()
            obs, r, term, trunc, _ = env.step(act)
            ep_ret += r
            done = term or trunc
        rets.append(ep_ret)
    actor.train()
    return float(np.mean(rets))


def play(cfg_path: str, render: bool):
    ckpt = load_ckpt(Path(cfg_path), map_location="cpu")
    cfgd = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_atari_env(cfgd["env_name"], screen_size=cfgd["img_size"], frame_stack=cfgd["frame_stack"],
                         render_mode="human" if render else None, seed=cfgd["seed"])
    global acfg
    acfg = ACfg(**ckpt["acfg"])
    actor = ActorCritic(acfg).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs, _ = env.reset()
    hx = torch.zeros(1, actor.cfg.lstm_dim, device=device)
    cx = torch.zeros(1, actor.cfg.lstm_dim, device=device)
    done = False
    ep_ret = 0.0
    while not done:
        o = torch.from_numpy(obs).to(device)
        o = o.permute(0,3,1,2).reshape(1, -1, o.size(1), o.size(2))
        o = normalize_obs_uint8(o)
        with torch.no_grad():
            logits, val, (hx, cx) = actor.step(o, (hx, cx))
        act = logits.argmax(dim=-1).item()
        obs, r, term, trunc, _ = env.step(act)
        ep_ret += r
        done = term or trunc
    print(f"Episode return: {ep_ret}")
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "play"], default="train")
    p.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    p.add_argument("--device", type=str, default=None, help="cuda/cpu")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--render", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    if args.mode == "train":
        cfg = TrainCfg(env_name=args.env, wandb=args.wandb, epochs=args.epochs)
        if args.device is not None:
            cfg.device = args.device
        train_onefile(cfg)
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt is required for play mode")
        play(args.ckpt, args.render)

if __name__ == "__main__":
    main()
