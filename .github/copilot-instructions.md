<!-- Copilot / AI agent instructions for the DIAMOND codebase -->

# Quick orientation for AI coding assistants

This file contains concise, project-specific instructions to help AI coding assistants be immediately productive in this repository.

1. Big picture

- **What**: DIAMOND is an RL agent trained inside a diffusion-based world model (visual world models + actor-critic). See [README.md](README.md) for paper/overview.
- **Major components**: model code under [src/models](src/models), agent composition in [src/agent.py](src/agent.py), training loop in [src/trainer.py](src/trainer.py), experiment entrypoint [src/main.py](src/main.py), and interactive visualizer [src/play.py](src/play.py).

2. How to run (developer workflows)

- Play pretrained world models: `python src/play.py --pretrained` (downloads from HF). See [src/play.py](src/play.py).
- Launch a training run (Hydra config overrides):

```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.devices=0
```

- Resume crashed run from run folder: `./scripts/resume.sh` (run folder = `outputs/YYYY-MM-DD/hh-mm-ss/`). See `scripts/resume.sh`.

3. Config & patterns to follow

- Hydra is used for configuration. The root config is [config/trainer.yaml](config/trainer.yaml). Entry points initialize Hydra with `config_path="../config"` (see [src/main.py](src/main.py)).
- Typical overrides use dot notation (e.g. `training.model_free=true` to disable world model).
- Runtime working dir: Hydra changes the working directory to an `outputs/...` run folder; `Trainer` moves `.hydra/config.yaml` into `config/trainer.yaml` the first time a run is created.

4. Important project-specific conventions

- Checkpoints: `checkpoints/state.pt` (full state) and `checkpoints/agent_epoch_XXXXX.pt` (agent weights). `Trainer` uses `checkpoints` relative to run folder. See [src/trainer.py](src/trainer.py).
- Dataset layout: `dataset/<run_name>/{train,test}` or `static_dataset.path` to point to a static dataset. Play/recorded episodes saved under `dataset/rec_*` (see [README.md](README.md)).
- World model vs model-free: toggle via `training.model_free` in `config/trainer.yaml`.
- Diffusion sampling tuning: edit `world_model_env.diffusion_sampler` section in [config/trainer.yaml](config/trainer.yaml).

5. Key code patterns to mirror in edits

- Models are composed into an `Agent` (see [src/agent.py](src/agent.py)). Loading a checkpoint uses `Agent.load(path)` which expects the checkpoint format produced by `Trainer`.
- Data loaders use custom `BatchSampler` and `Dataset` under `src/data` — prefer using those utilities for compatibility with training loops.
- WorldModelEnv is used as an RL environment wrapper around denoiser + rew_end_model; see creation in [src/play.py](src/play.py) and [src/trainer.py](src/trainer.py).

6. Integration & external dependencies

- Hugging Face Hub: `src/play.py` downloads pretrained configs & checkpoints when `--pretrained` is used.
- W&B is optional; configured in `config/trainer.yaml` (default disabled). `Trainer` calls `wandb.init` when rank 0.
- Multi-GPU: DDP is supported via `mp.spawn` in [src/main.py](src/main.py); `common.devices` controls `CUDA_VISIBLE_DEVICES`.

7. When making code changes

- Preserve Hydra-config keys and types; prefer adding new config entries in `config/*` and keep default values in `config/trainer.yaml` or `config/agent/default.yaml`.
- When editing training behavior, update both `src/trainer.py` and `config/trainer.yaml` examples.
- For UI/visualizer changes, prefer modifying [src/play.py](src/play.py) and `game` helpers in [src/game](src/game).

8. Quick references

- Entrypoints: [src/main.py](src/main.py) (train), [src/play.py](src/play.py) (visualize/play).
- Core trainer: [src/trainer.py](src/trainer.py).
- Agent composition: [src/agent.py](src/agent.py) and [src/models](src/models).
- Configs: [config/trainer.yaml](config/trainer.yaml), [config/agent/default.yaml](config/agent/default.yaml), [config/env/atari.yaml](config/env/atari.yaml).

If anything here is unclear or you'd like additional examples (e.g., a small patch showing how to change the diffusion sampler), tell me which area to expand and I'll iterate.
