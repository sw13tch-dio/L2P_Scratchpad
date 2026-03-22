# L2P Replication Project — Handoff Document

**For:** Another AI model or team member  
**Goal:** Reproduce L2P (Learning to Prompt for Continual Learning) on Split-CIFAR-100 at 10% data, target ~80% accuracy (Shakarian's result)  
**Date:** March 2025  
**GitHub:** https://github.com/sw13tch-dio/L2P_Scratchpad

---

## 1. Executive Summary

We are attempting to replicate the L2P continual-learning paper using the **JAX/Flax** implementation from [google-research/l2p](https://github.com/google-research/l2p). The original repo uses outdated dependencies (`jaxlib==0.1.68`, `flax.optim`) that are no longer installable on current systems (Colab, Python 3.12). **A full migration from `flax.optim` to `optax` has been completed locally.** The code runs locally but has not yet been successfully pushed to GitHub due to a large push size (311 MB) causing HTTP 408 timeouts.

---

## 2. Project Layout

```
L2P_Code/
├── l2p/                    # JAX/Flax implementation (MAIN FOCUS)
│   ├── train_continual.py  # Core training loop — migrated to optax
│   ├── libml/              # Utils, losses, checkpoint loading — updated
│   │   ├── utils.py
│   │   ├── utils_vit.py
│   │   └── ...
│   ├── models/             # ViT, ResNet definitions
│   ├── configs/            # e.g., cifar100_l2p.py
│   └── requirements.txt    # Updated for optax, modern JAX
├── l2p-pytorch/            # Alternative PyTorch port — NOT primary path
├── HANDOFF_DOCUMENT.md     # This file
└── ... (other docs, fix scripts)
```

---

## 3. What We’re Doing: The JAX Migration Explained

### 3.1 Why JAX?

- The paper’s reference implementation is in JAX/Flax.
- JAX is designed for high-performance ML (XLA compilation, easy parallelism).
- Colab provides free GPU/TPU support for JAX.

### 3.2 The Original Problem

The upstream `l2p` repo pinned:

- `jax==0.2.17`
- `jaxlib==0.1.68` ← **No longer on PyPI**
- `flax==0.3.5` (which used `flax.optim`)

`flax.optim` was removed in later Flax. The original code used:

- `flax.optim.Optimizer`, `flax.optim.Adam`, `flax.optim.Momentum`
- `flax.optim.MultiOptimizer` for freezing parts of the model
- `optimizer.target` for params
- `optimizer.apply_gradient(grad, learning_rate=lr)`

### 3.3 The Solution: Optax Migration

We switched to **optax**, the standard optimizer library for JAX. Changes:

| Old (Flax)                 | New (Optax)                                   |
|----------------------------|-----------------------------------------------|
| `TrainState(optimizer=...)`| `TrainState(params=..., opt_state=..., optimizer=...)` |
| `state.optimizer.target`    | `state.params`                                |
| `optimizer.apply_gradient(grad, lr)` | `opt.update(grad, opt_state, params)` + `optax.apply_updates(params, updates)` |
| `optimizer.replace(target=x)` | `state.replace(params=x)` + `opt.init(x)`     |
| `MultiOptimizer` for freeze | `optax.transforms.freeze(mask)` or `optax.masked(optax.set_to_zero(), mask)` |

### 3.4 What Changed, Technically

**TrainState** (`train_continual.py`):

```python
@flax.struct.dataclass
class TrainState:
  step: int
  params: Any          # model parameters
  opt_state: Any       # optax optimizer state (momentum, etc.)
  optimizer: Any        # optax chain ( GradientTransformation )
  model_state: Any      # batch norm stats, etc.
```

**Optimizer creation**:

- Returns `(optimizer, opt_state)` instead of a single Flax optimizer.
- Uses `optax.adamw` / `optax.sgd`, with a per-step learning rate.

**Training step**:

1. Compute gradients w.r.t. `state.params`.
2. `updates, new_opt_state = state.optimizer.update(grad, state.opt_state, state.params)`.
3. Scale updates by current LR (schedule computed outside optax).
4. `new_params = optax.apply_updates(state.params, updates)`.
5. Update `state` with `params`, `opt_state`, `model_state`.

**Freezing**:

- Build a mask from param paths (`freeze_part = ["encoder", "embedding", "cls"]`).
- `optax.transforms.freeze(mask)` or `optax.masked(optax.set_to_zero(), mask)` zeroes updates for frozen params.

**Checkpoint loading**:

- Support both old (`state["optimizer"]["target"]`) and new (`state["params"]`) formats.
- Reinitialize `opt_state` after loading pretrained params.

---

## 4. Files Modified

| File | Summary |
|------|---------|
| `l2p/train_continual.py` | TrainState, `create_optimizer`, `train_step`, all `state.optimizer.target` → `state.params` |
| `l2p/libml/utils.py` | `state_with_new_param`, `load_and_custom_init_checkpoint`, `_load_and_custom_init_checkpoint` |
| `l2p/requirements.txt` | `jax>=0.4.0`, `jaxlib>=0.4.0`, `flax>=0.7.0`, `optax>=0.2.0`, remove `tensorflow_addons` |

### Other Known Fixes (may exist elsewhere)

- **tensorflow_addons**: Not on Python 3.12. There may be a `fix_tfa_for_colab.py` or similar that replaces TFA with `scipy` for augmentations.

---

## 5. How to Run (After Push)

**Locally:**

```bash
cd l2p
pip install -r requirements.txt
# Download ViT checkpoint to a path, then:
python main.py --my_config configs/cifar100_l2p.py --workdir=./output --my_config.init_checkpoint=<path-to-ViT-B_16.npz>
```

**Colab:**

1. Clone from `https://github.com/sw13tch-dio/L2P_Scratchpad` (after push succeeds).
2. `pip install -r l2p/requirements.txt`
3. Download ViT-B_16.npz (e.g., from Google Storage) and pass its path as `init_checkpoint`.
4. Run the same `main.py` command.

---

## 6. Git Push Issue — What You Need to Fix

The last `git push` failed with:

```
Writing objects: 100% (15/15), 311.04 MiB
error: RPC failed; HTTP 408
fatal: the remote end hung up unexpectedly
```

The push was ~311 MB, which is too large for a normal GitHub push. Likely causes:

- Virtualenvs: `venv/`, `venv_l2p/`, `venv_l2p_pytorch/`
- Model checkpoints: `*.npz`, `checkpoints/`
- Datasets or caches

### Steps to Fix the Push

1. **Add or update `.gitignore`** at the repo root (`L2P_Code/`):

```
# Virtual environments
venv/
.venv/
venv_*/
env/
__pycache__/
*.pyc

# Models & checkpoints
*.npz
*.ckpt
*.pt
*.pth
checkpoints/
checkpoint*/

# Data
*.tfrecord
data/
datasets/
output/

# Other
*.out
*.err
.DS_Store
```

2. **Stop tracking large files:**

```powershell
cd "c:\Users\sw13t\OneDrive - Syracuse University\SU_Spring2026\Leibnitz Lab\L2P_Code"
git rm -r --cached .
git add .
git status   # Verify no large files
git commit -m "Exclude large files; JAX/optax migration"
```

3. **Optional: increase buffer:**

```powershell
git config http.postBuffer 524288000
```

4. **Push:**

```powershell
git push -u origin main --force
```

If history is already polluted:

- Use `git filter-branch` or BFG Repo-Cleaner to remove large files from history, or
- Start clean with `git init`, add `.gitignore`, add all files, and `git push --force`.

---

## 7. Current State Checklist

- [x] Optax migration completed in `l2p/`
- [x] `requirements.txt` updated
- [x] Checkpoint loading supports old and new formats
- [ ] `.gitignore` updated to avoid large files
- [ ] Successful push to GitHub
- [ ] Training run validated on Colab or locally
- [ ] Target accuracy (~80%) on Split-CIFAR-100 at 10% data confirmed

---

## 8. Key Config for CIFAR-100 L2P

From `configs/cifar100_l2p.py`:

- Model: `ViT-B_16`
- `freeze_part = ["encoder", "embedding", "cls"]`
- `optim = "adam"`, `learning_rate = 0.03`
- `pull_constraint_coeff = 1.0`
- 10 tasks, 10 classes per task
- Prompt pool: size 10, length 10, top_k 4

---

## 9. References

- L2P paper: [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654)
- Upstream JAX repo: https://github.com/google-research/l2p
- Optax: https://github.com/google-deepmind/optax
- JAX: https://github.com/google/jax

---

## 10. Contact / Next Steps

1. Ensure `.gitignore` excludes venvs and large artifacts.
2. Remove large files from the git index and history if needed.
3. Push to `https://github.com/sw13tch-dio/L2P_Scratchpad`.
4. Run a full training on Colab to confirm the migration works end-to-end.
