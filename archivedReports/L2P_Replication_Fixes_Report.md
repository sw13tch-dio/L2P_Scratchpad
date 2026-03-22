# L2P Replication Fixes — Technical Report

**For:** Dr. Paulo Shakarian  
**From:** Dio Brown  
**Date:** March 2026  
**Subject:** Systematic fixes applied to PyTorch L2P reimplementation to match Wang et al. (CVPR 2022) paper

---

## 0. Why Patches? Can’t We Just Follow the Code Exactly?

**Short answer:** No. The patches fix two different kinds of problems:

| Type | Why it’s needed |
|------|------------------|
| **Colab environment** | The original repo assumes a local server with direct access to Google Storage. On Colab, those URLs often return 403 or fail. The repo also expects `.npz` weight files from Google; Colab is better served by HuggingFace Hub. So we patch weight loading to work in Colab. |
| **Paper alignment** | The PyTorch reimplementation (JH-LEE-KR/l2p-pytorch) has defaults that don’t match the paper: λ=0.1 vs 0.5, batch 16 vs 128. Those would be wrong even if we ran locally. We fix them via config args and gradient accumulation. |

**“Follow the code exactly”** would keep those wrong defaults and Colab would still fail on weight loading. The patches are necessary for both correctness and Colab compatibility.

---

## 1. Executive Summary

Our initial runs achieved ~47% accuracy at 50% data, while you reported ~80% at 10% data. That gap indicated we were not implementing the paper correctly. After auditing the PyTorch reimplementation against the L2P paper (Wang et al., CVPR 2022), we identified several mismatches between the paper’s specification and the JH-LEE-KR/l2p-pytorch codebase. This document describes the issues and the fixes applied.

---

## 2. Paper Context — What L2P Specifies

From Wang et al., “Learning to Prompt for Continual Learning” (CVPR 2022):

- **Equation 5 ( Optimization objective):**  
  The loss has two terms:  
  (1) Cross-entropy for classification  
  (2) **Pull constraint:** \( \mathcal{L} = \mathcal{L}_{CE} - \lambda \sum \gamma(q(x), k_{s_i}) \)  
  where \( \gamma \) is cosine similarity between the query \( q(x) \) and the selected keys \( k_{s_i} \).  
  The paper sets **λ = 0.5**, so half the training signal comes from organizing the prompt pool.

- **Section 5.2 (Training details):**  
  Batch size **128** (8 GPUs × 16 per GPU). Adam optimizer, lr = 0.03, 5 epochs per task. Label smoothing 0.1. For Split-CIFAR-100: M=10 prompts, N=5 selected per input, L_p=5 tokens per prompt.

- **Prompt position averaging:**  
  The classification head uses \( f^{avg}_r = \text{AvgPool}(f_r(x_p)[0:N·L_p, :]) \) — i.e., the output at **prompt token positions** is averaged (not the CLS token). The PyTorch code already does this correctly.

---

## 3. Issues Identified in the PyTorch Reimplementation

### Issue 1: Pull Constraint Coefficient (λ) Incorrect

**Paper:** λ = 0.5  
**Code:** `configs/cifar100_l2p.py` line 92 sets `pull_constraint_coeff` default = **0.1**

**Impact:** The pull constraint drives keys toward the queries that select them. With λ = 0.1 instead of 0.5, only ~20% of the intended pull signal is used. Keys barely specialize, and prompt selection remains poorly aligned.

**Location:** `configs/cifar100_l2p.py`

---

### Issue 2: Effective Batch Size Too Small

**Paper:** Batch size 128 total  
**Code:** Default `batch-size` = 16 per device; single-GPU setups use 16 total

**Impact:** The pull constraint averages cosine similarity over the batch (`reduce_sim = torch.sum(sim) / x_embed.shape[0]` in `prompt.py`). With 16 samples, this average is noisy. Stable key–query alignment requires larger batches. The paper’s 128 samples give much cleaner gradients.

**Location:** `engine.py` training loop, `configs/cifar100_l2p.py`

---

### Issue 3: Pretrained Weight Loading Broken on Colab

**Problem:** `vision_transformer.py` sets `pretrained_custom_load='npz' in pretrained_cfg['url']`, which routes loading through timm’s legacy Google Storage .npz path. On Colab this path is unreliable (403s or unavailable URLs).

**Impact:** Weights fail to load or fall back to random init, which collapses performance.

**Location:** `vision_transformer.py` line ~723

---

### Issue 4: models.py Weight Source Incompatible

**Problem:** The original `models.py` uses timm’s `_create_vision_transformer`, which fetches weights from Google Storage. Those fetches often fail on Colab.

**Impact:** Same as Issue 3 — failed or incorrect weight loading.

**Location:** `models.py`

---

### Issue 5: Positional Embedding Shape Mismatch

**Problem:** Pretrained ViT uses 197 positions (1 CLS + 196 patches). L2P adds 25 prompt tokens (5 prompts × 5 tokens), so it needs 222 positions. The original models do not handle this shape change.

**Impact:** Either loading fails or positional embeddings are wrong, hurting performance.

**Location:** `models.py` (our custom version handles this)

---

### Issue 6: Gradient Accumulation Patch Crashes Evaluation

**Problem:** When adding gradient accumulation, the patch used `str.replace()` to inject a “flush remaining gradients” block before `metric_logger.synchronize_between_processes()`. That block appears both in `train_one_epoch` and in `evaluate`. Replacing all occurrences also modified `evaluate`, inserting `optimizer.zero_grad()` where `optimizer` does not exist.

**Impact:** Training finishes, but evaluation crashes with `NameError: name 'optimizer' is not defined`.

**Location:** `engine.py` patch logic — the replacement matched both `train_one_epoch` and `evaluate`

---

### Issue 7: `--subsample` Not in Upstream Config

**Problem:** The upstream repo on GitHub (`JH-LEE-KR/l2p-pytorch`) does **not** define `--subsample` in `configs/cifar100_l2p.py`. The local fork may have it, but a fresh `git clone` on Colab does not. Passing `--subsample 0.1` then causes: `error: unrecognized arguments: --subsample 0.1`.

**Impact:** Cannot run on 10% data without modifying the config.

**Location:** `configs/cifar100_l2p.py` — argument must be added via patch.

---

## 4. Fixes Applied (in Order)

### Fix 1: Set Pull Constraint Coefficient to 0.5

**What:** Pass `--pull_constraint_coeff 0.5` at training time (or change the config default).

**Where:** Command line / `configs/cifar100_l2p.py`

**Result:** Matches paper; pull constraint now gets the intended weight.

---

### Fix 2: Gradient Accumulation to Simulate Batch Size 128

**What:** Keep `batch-size 16` but add `--gradient_accumulation_steps 8`. Gradients are accumulated over 8 steps before each optimizer update, giving an effective batch size of 16 × 8 = 128.

**Code changes in `engine.py`:**

- Replace per-step `optimizer.zero_grad() → loss.backward() → clip_grad → optimizer.step()` with:
  - Scale loss by `1/8`
  - Call `loss_scaled.backward()` each step
  - Every 8th step: `clip_grad_norm`, `optimizer.step()`, `optimizer.zero_grad()`
- Add `_ga_counter` and initial `optimizer.zero_grad()` before the training loop
- Add flush logic at end of epoch for leftover gradients when total steps is not divisible by 8

**Where:** `engine.py`, `configs/cifar100_l2p.py` (new `--gradient_accumulation_steps` argument)

**Result:** Effective batch size 128, as in the paper.

---

### Fix 3: Remove Broken .npz Loader from vision_transformer.py

**What:** Remove the line `pretrained_custom_load='npz' in pretrained_cfg['url']` via regex so timm’s custom .npz path is never used.

**Where:** `vision_transformer.py`

**Result:** Prevents failed .npz loading on Colab.

---

### Fix 4: Replace models.py with HuggingFace Hub Loading

**What:** Replace `models.py` with a custom `vit_base_patch16_224` that:

- Loads weights from HuggingFace Hub (`timm/vit_base_patch16_224.augreg_in21k`) via `hf_hub_download` and `safetensors` or `torch.load`
- Strips `head` weights (classification head is retrained)
- Handles `pos_embed` shape mismatch: copy the first 197 positions into a 222-position tensor and leave the extra positions as initialized

**Where:** `models.py`

**Result:** Reliable weight loading on Colab and correct positional embeddings.

---

### Fix 5: Limit Gradient-Flush Patch to train_one_epoch

**What:** Change the replacement from:

```python
engine_code.replace(gather_block, flush_block)
```

to:

```python
engine_code.replace(gather_block, flush_block, 1)
```

so only the **first** occurrence (in `train_one_epoch`) is replaced. The `evaluate` function keeps its original `synchronize_between_processes` block.

**Where:** Patch script that modifies `engine.py`

**Result:** Evaluation no longer crashes; test metrics are reported.

---

### Fix 6 & 7: Add `--subsample` and Use 10% Data

**What:** The upstream config doesn’t define `--subsample`. The patch must add it to `configs/cifar100_l2p.py`, then we pass `--subsample 0.1` at training time. The repo’s `datasets.py` already supports subsampling when `args.subsample < 1.0`.

**Where:** Patch adds `subparsers.add_argument('--subsample', default=1.0, type=float, ...)` to the config; training command passes `--subsample 0.1`

**Result:** Matches your experimental setup of 10% data per task.

---

## 5. Final Training Command (10% Data, Paper-Aligned)

```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py cifar100_l2p \
  --model vit_base_patch16_224 \
  --batch-size 16 \
  --data-path ./data/ \
  --output_dir ./output_10pct \
  --pull_constraint_coeff 0.5 \
  --gradient_accumulation_steps 8 \
  --subsample 0.1
```

---

## 6. Summary Table

| Issue | Paper Spec | Original Code | Fix |
|-------|-----------|---------------|-----|
| Pull constraint λ | 0.5 | 0.1 | `--pull_constraint_coeff 0.5` |
| Batch size | 128 | 16 | Gradient accumulation × 8 |
| Weight loading | — | Google .npz (fails) | HuggingFace Hub + pos_embed handling |
| Eval crash | — | Patch modified wrong function | `replace(..., 1)` |
| Data fraction | 10% (our target) | 100% default | `--subsample 0.1` |

---

## 7. References

- Wang, Z., Zhang, Z., Lee, C.-Y., Zhang, H., Sun, R., Ren, X., Su, G., Perot, V., Dy, J., & Pfister, T. (2022). Learning to Prompt for Continual Learning. *CVPR 2022*. arXiv:2112.08654
- JH-LEE-KR/l2p-pytorch: https://github.com/JH-LEE-KR/l2p-pytorch
- google-research/l2p (JAX): https://github.com/google-research/l2p

---

## Appendix A: Changes Along the Way (Changelog)

| Date/Session | What happened | Fix |
|--------------|---------------|-----|
| Initial runs | 47% at 50% data vs target 80% at 10% | Identified paper mismatches (λ, batch size) |
| Colab setup | Weight loading failed (403, .npz) | Patches A & B: vision_transformer, models.py |
| First fix run | Training completed, eval crashed | Patch C bug: `replace(..., 1)` to avoid modifying `evaluate` |
| Run with `--subsample 0.1` | `unrecognized arguments: --subsample 0.1` | Upstream config lacks `--subsample`; add it in patch |

---

## Appendix B: Complete Config Patch (adds both args)

The patch Cell 2 must add **both** `--gradient_accumulation_steps` and `--subsample` to the config, since the upstream repo has neither:

```python
config_code = open("configs/cifar100_l2p.py").read()
additions = []
if 'gradient_accumulation_steps' not in config_code:
    additions.append("    subparsers.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps')")
if 'subsample' not in config_code:
    additions.append("    subparsers.add_argument('--subsample', default=1.0, type=float, help='fraction of training data per task (e.g. 0.1 for 10%%)')")
if additions:
    config_code = config_code.rstrip() + "\n" + "\n".join(additions) + "\n"
    open("configs/cifar100_l2p.py", "w").write(config_code)
```

See `Colab_Cells_Complete.py` in this repo for the full copy-paste Colab cells (Cell 1: clone, Cell 2: patches, Cell 3: run).

---

*Document generated from code audit and paper review. March 2026. Updated with subsample fix.*
