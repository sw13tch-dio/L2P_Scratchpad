# L2P Replication Scratchpad

Replicating **Learning to Prompt for Continual Learning** (Wang et al., CVPR 2022) — a prompt-based continual learning method that achieves state-of-the-art performance on split CIFAR-100 without replay buffers or task identifiers at test time.

## The Assignment

Replicate the L2P paper's results on split CIFAR-100 (10 tasks, 10 classes each) using a pretrained ViT-B/16. The benchmark: **~80% average accuracy** across all tasks, achieved by our lab supervisor using only 10% of the training data.

## The Journey

### Phase 1: PyTorch Port (Weeks 1-2) — Dead End

Started with a community PyTorch reimplementation (`l2p-pytorch/`). After two weeks of debugging:

- **Best result: ~47% accuracy** using 50% of data
- Found two critical bugs via forensic code audit:
  1. `pull_constraint_coeff` was set to `0.1` when the paper specifies `1.0`
  2. Effective batch size was 16 instead of the paper's 128
- Attempted coherence-guided sampling (Run 5): **~26% accuracy** — proved that removing atypical training examples *hurts* L2P's prompt pool specialization
- A Colab notebook with all fixes was prepared but we pivoted before running it

### Phase 2: Original JAX Code (March 21-22) — Breakthrough

**Key decision**: Stop reimplementing. Run the authors' original code.

**Problem**: The original repo (early 2022) uses deprecated APIs that don't exist in modern JAX/Flax/Python. Specifically:

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| `flax.optim` removed | Flax > 0.5.3 dropped its optimizer module | Built a ~100-line shim backed by `optax` |
| `tensorflow_addons` deprecated | Package fully removed from PyPI | Stubbed 3 image transform functions with native TF ops |
| `collections.Mapping` gone | Moved to `collections.abc` in Python 3.10+ | Find-and-replace |
| `jax.tree_map` removed | JAX 0.6+ moved to `jax.tree.map` | Global find-and-replace |
| Flax `FrozenDict` vs `dict` | Modern Flax wraps params in FrozenDict, optax expects plain dict | Added `unfreeze()` calls at param boundaries |
| Checkpoint serialization fails | `flax.serialization` can't serialize custom optimizer shim | Disabled checkpointing (acceptable for single run) |
| Jupyter import caching | Failed imports get cached in `sys.modules` | Kernel restart between fix attempts |

**Result**: The original JAX L2P running on Google Colab T4 GPU with modern dependencies (JAX 0.7.2, Flax 0.12.5, Optax 0.2.6, Python 3.12).

### Current Results (Pending Final)

- **Task 1 accuracy: 97.8%** (27.4 minutes)
- Full 10-task run in progress (~3-4 hours total)
- Final average accuracy pending — expecting to match the ~80% benchmark

## Project Structure

```
L2P_Code/
│
├── README.md                    ← You are here
├── dearSoraya_L2P.md            ← Session handoff document (full context for continuity)
├── Colab_JAX_L2P_Cells.md       ← Copy-paste ready Colab notebook cells
├── run_l2p_patched.py           ← Self-contained setup + run script (WIP)
├── .gitignore                   ← Excludes weights, venvs, checkpoints, data
│
├── l2p/                         ← Original Google Research JAX repo (the one that works)
│   ├── main.py                  ← Entry point (absl flags)
│   ├── train_continual.py       ← Core training loop, optimizer setup, eval
│   ├── run_l2p.py               ← Local run script (for RTX GPU)
│   ├── setup_and_run.bat        ← Windows batch launcher
│   ├── configs/
│   │   ├── cifar100_l2p.py      ← THE config (pull_constraint_coeff=1.0 ✅)
│   │   ├── cifar100_dualprompt.py
│   │   └── ...
│   ├── models/
│   │   ├── vit.py               ← ViT-B/16 model definition
│   │   ├── prompt.py            ← Prompt pool implementation (the core of L2P)
│   │   └── prefix_attention.py  ← Prefix-tuning attention variant
│   ├── libml/
│   │   ├── input_pipeline.py    ← CIFAR-100 split dataset creation
│   │   ├── utils.py             ← Checkpoint loading, weight transfer
│   │   ├── utils_vit.py         ← ViT weight loading from .npz
│   │   ├── losses.py            ← Cross-entropy + pull constraint loss
│   │   └── preprocess.py        ← Image preprocessing
│   └── augment/                 ← Data augmentation (uses tensorflow_addons)
│
├── l2p-pytorch/                 ← Community PyTorch port (abandoned — has bugs)
│   ├── main.py                  ← PyTorch training entry point
│   ├── engine.py                ← Training loop (had batch size bug)
│   ├── models.py                ← PyTorch ViT + prompt pool
│   ├── prompt.py                ← PyTorch prompt implementation
│   ├── configs/
│   │   └── cifar100_l2p.py      ← Config (pull_constraint_coeff=0.1 ⚠️ WRONG)
│   └── data/                    ← Downloaded CIFAR-100 data
│
├── researchPapers/              ← Reference papers
│   ├── Wang_Learning_To_Prompt_...CVPR_2022_paper.pdf  ← The L2P paper
│   ├── BR_for_Continual_Learning.pdf                   ← Belief revision paper
│   └── core50 2017 1705.03550v1.pdf                    ← CORe50 dataset paper
│
├── archivedCode/                ← Earlier attempts (preserved for reference)
│   ├── Colab_Cells_Complete.py  ← PyTorch-era Colab notebook
│   ├── diagnose.py              ← Diagnostic scripts from debugging phase
│   └── fix_tfa_for_colab.py     ← Early tensorflow_addons fix attempt
│
├── archivedReports/             ← Earlier analysis documents
│   ├── HANDOFF_DOCUMENT.md      ← Original PyTorch-era handoff
│   ├── L2P_Replication_Fixes_Report.md  ← Forensic bug analysis
│   └── L2P_ResearchReport.md    ← Initial research summary
│
└── ViT-B_16.npz                ← Pretrained weights (gitignored, ~330MB)
                                    Download: storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```

## How to Run

### Option 1: Google Colab (recommended)
1. Open a Colab notebook with T4 GPU + High RAM
2. Copy Cell 1 and Cell 2 from `Colab_JAX_L2P_Cells.md`
3. Run Cell 1, wait for ✅
4. Restart kernel, run Cell 2
5. Watch task-by-task progress (~3-4 hours total)

### Option 2: Local (RTX GPU)
```bash
cd l2p/
python -m venv venv_l2p
venv_l2p\Scripts\activate        # Windows
pip install -r requirements.txt
# Download ViT-B_16.npz to project root
python run_l2p.py
```

## Key Config Values

| Parameter | JAX (correct) | PyTorch port (wrong) |
|-----------|:---:|:---:|
| `pull_constraint_coeff` | **1.0** | 0.1 |
| `pool_size` | 10 | 10 |
| `prompt_length` | 10 | 5 |
| `top_k` | 4 | 5 |
| `batch_size` | 16 | 16 |
| `epochs` | 5 | 5 |
| `optimizer` | adam | adam |
| `learning_rate` | 0.03 | 0.03 |

## Key Takeaways

1. **The PyTorch port was the problem, not the method** — the original JAX code with correct hyperparameters works as expected
2. **`pull_constraint_coeff=1.0`** is critical — running it at 0.1 (as the PyTorch port does) prevents prompt key specialization
3. **Visual diversity helps L2P** — coherence-guided sampling that removes atypical examples actively hurts prompt pool specialization
4. **Modern compatibility shims > downgrading** — monkey-patching `flax.optim` with optax is cleaner than fighting the entire dependency stack

## References

- Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022
- Original repo: https://github.com/google-research/l2p
- PyTorch port: https://github.com/JH-LEE-KR/l2p-pytorch
