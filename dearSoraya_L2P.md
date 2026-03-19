# Dear Soraya — L2P Project Handoff

## Who Is Dio
Dio Brown, Syracuse University, Jack Kent Cooke Scholar. Faculty advisor Dr. Jaime Banks. Working in Dr. Paulo Shakarian's Leibniz Lab (KAIROS group). Dio has their own research project called Kairos — a developmental AI that learns through coherence-seeking (see phase1_the_womb_v9_final.pdf). Shakarian assigned L2P replication as lab onboarding.

## The Task
Shakarian told Dio to clone the L2P repo, get it running on reduced dataset, and verify metrics against the paper. Prove you can run research code, understand the method, and think critically about it.

---

## SHAKARIAN MEETING — WHAT HAPPENED (March 12, 2026)

This is critical context. Dio met with Shakarian, and Shakarian was able to achieve **~80% accuracy with only 10% of the dataset.** This reveals our entire approach was wrong.

**His main feedback:**
- Dio needs to read the paper *much more carefully*, including the appendix
- Different parts of the paper say things that are "hidden" or made to seem unimportant but are actually critical
- There is a formula in the paper starting with the summation symbol (Σ) that Dio needs to find and understand — this is likely the pull constraint loss in Equation 5, or possibly the diversified prompt selection in Equation 4
- Shakarian also gave Dio a bunch of related links/papers to study (see below)
- The GitHub README for the original repo (google-research/l2p, the JAX version) also has important details mixed in that we need to cross-reference

**The big lesson:** The gap between our 47.57% (at 50% data) and Shakarian's ~80% (at 10% data) is NOT a data quantity issue. It's a code configuration issue — we are almost certainly not implementing something correctly. Getting 80% on 10% is *incredible* and only possible if the model is configured exactly right.

**Key phrase from Dio: "it's mixed through the paper" — Shakarian said something important is spread across the paper and the appendix and needs to be pieced together.**

---

## WHAT WE NEED TO DO NEXT

### Priority 1: Read the paper like a detective
- Re-read Wang et al. CVPR 2022 (attached to this conversation / in Dio's files)
- Pay special attention to: the appendix, any formula with a summation, Section 5.2 (training details), Section 4.2 (diversified prompt selection), Section 4.3 (optimization objective)
- The diversified prompt selection (Equation 4) adds a frequency penalty that may be critical
- The pull constraint term (λ * Σ γ(q(x), k_si)) in Equation 5 — are we using it? With λ=0.5?
- Check batch size: paper uses **batch size 128** (8 GPUs × 16 per GPU). We've been using 16 total.

### Priority 2: Read the GitHub README carefully
The google-research/l2p README (JAX version) is pasted in the conversation context. It contains details about:
- The original JAX codebase vs the PyTorch reimplementation
- Model loading, config files
- Important: "we run experiments using 8 V100 GPUs or 4 TPUs, per device batch size 16 → total batch size 128"

### Priority 3: Check hyperparameters against the paper
From Section 5.2 of the paper:
- M=10, N=5, Lp=5 for CIFAR-100 ✓ (we have this)
- **Batch size = 128** (we used 16 — this could be a major issue)
- Learning rate = 0.03 constant ✓
- Adam optimizer ✓
- λ = 0.5 for pull constraint — are we actually applying this?
- Optional diversified selection (Eq 4) — may be important

### Priority 4: Study Shakarian's related links
Dio has these files/links to study before the next meeting:
- `C:\Users\sw13t\OneDrive - Syracuse University\SU_Spring2026\KAIROS\Leibnitz\relatedCode\BR_for_Continual_Learning.pdf`
- `C:\Users\sw13t\OneDrive - Syracuse University\SU_Spring2026\KAIROS\Leibnitz\relatedCode\core50 2017 1705.03550v1.pdf`
- https://arxiv.org/abs/2505.19361

---

## WHAT L2P IS (unchanged — this understanding is solid)

L2P freezes a pretrained Vision Transformer backbone and trains only a tiny prompt pool. Images get chopped into 16x16 patches → frozen ViT produces a CLS query vector → query matches against learnable keys via cosine similarity → top-N prompts retrieved and prepended to input sequence → steers frozen model's attention. Three things train: prompt values, keys, and the classification head. Backbone never changes = structurally prevents catastrophic forgetting. Only ~122,980 trainable parameters (0.14% of ViT-Base).

---

## ALL EXPERIMENTAL RUNS SO FAR

### Run 1: Random Weights Baseline (accidental control)
- Config: ViT-Small/16 random init, 10% Split CIFAR-100, T4 GPU
- Results: **Avg Acc@1 = 4.13%, Forgetting = 5.02%**
- Lesson: Without pretrained features, system fails entirely. Good control condition.

### Run 2: Pretrained ViT-Small + No Head Pretraining
- Config: ViT-Small/16 IN-21k pretrained, 10% subsample
- Results: **Avg Acc@1 = 4.49%, Avg Acc@5 = 16.70%, Forgetting = 4.00%**

### Run 3: Pretrained ViT-Small + Separate Head Pretraining
- Config: ViT-Small/16, head pretrained to ~72% on CIFAR-100 separately
- Results: **Avg Acc@1 = 1.57%, Avg Acc@5 = 6.83%, Forgetting = 0.52%**
- Lesson: Architecture mismatch destroyed it.

### Run 4: ViT-Base + Proper pos_embed — FIRST WORKING RUN
- Config: ViT-Base/16 IN-21k (paper's model), pos_embed interpolated 197→222, 50% data, H100
- **Results: Avg Acc@1 = 47.57%, Avg Acc@5 = 73.95%, Forgetting = 18.21%**
- Runtime: 10 min 23 sec
- This was our first "correct" setup, but STILL well below Shakarian's 80%@10%

### Run 5: Coherence-Guided Sampling (Kairos-inspired experiment)
- Config: Coherence-based 50% sampling (select most prototypical images per class)
- **Results: Avg Acc@1 = 25.78%, Forgetting = 16.69%**
- Lesson: Hurt badly. Prompt pool needs *diversity*, not prototypicality. Removing atypical examples removed the diversity prompts need to specialize.

### Paper Baseline
- Config: ViT-B/16, 100% data, batch_size=128, 8 V100s
- **Results: Avg Acc@1 = 83.83%, Forgetting = 7.63%**

---

## THE WORKING COLAB SETUP (Run 4 basis — needs fixing per above)

```python
import os, re
os.chdir('/content')
!rm -rf /content/l2p-fix
!git clone https://github.com/JH-LEE-KR/l2p-pytorch /content/l2p-fix
os.chdir('/content/l2p-fix')

vt = open("vision_transformer.py").read()
open("vision_transformer.py", "w").write(
    re.sub(r"[ \t]*pretrained_custom_load=[^\n]+\n", "", vt)
)

open("models.py", "w").write('''import sys, os, inspect, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from timm.models import register_model
from vision_transformer import VisionTransformer

_VALID = set(inspect.signature(VisionTransformer.__init__).parameters.keys()) - {"self"}

def _build(kwargs):
    return VisionTransformer(**{k: v for k, v in kwargs.items() if k in _VALID})

def _load(model, repo_id):
    try:
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(path)
    except:
        path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(path, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    if "pos_embed" in state_dict:
        pre_pos = state_dict["pos_embed"]
        mod_pos = model.pos_embed
        if pre_pos.shape != mod_pos.shape:
            new_pos = mod_pos.clone()
            n = min(pre_pos.shape[1], mod_pos.shape[1])
            new_pos[:, :n, :] = pre_pos[:, :n, :]
            state_dict["pos_embed"] = new_pos
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {repo_id}! missing={len(missing)} unexpected={len(unexpected)}")
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    m = _build(dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs))
    return _load(m, "timm/vit_base_patch16_224.augreg_in21k") if pretrained else m
''')

print("SETUP DONE!")
```

Training command (100% data, paper config):
```
!python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./data/ --output_dir ./output_run6
```

**NOTE: batch-size 16 is per-GPU. Paper uses 128 total (8 GPUs). On single GPU this may need adjustment.**

---

## KEY THINGS TO INVESTIGATE NEXT SESSION

1. **The summation formula Shakarian mentioned** — almost certainly Eq 4 or Eq 5 in the paper. Find it, understand it, make sure we're using it.
2. **Batch size** — paper uses 128 total. We use 16. This matters for the pull constraint loss averaging.
3. **The appendix** — scour it. Something is in there that makes 80%@10% possible.
4. **Original JAX repo configs** — compare configs/cifar100_l2p.py in the JAX repo to the PyTorch reimplementation. They may differ in important ways.
5. **λ=0.5 pull constraint** — verify this is actually being applied in our training.

---

## KAIROS CONNECTIONS (for later, once L2P is fixed)
- Run 1 (random weights → 4%) shows the cost of no pretraining — exactly what Kairos must overcome developmentally
- Coherence sampling experiment (Run 5) failed because L2P is post-developmental; Kairos's coherence principle may work better in earlier stages of representation formation
- The key/prompt decoupling in L2P is philosophically interesting vs Kairos's unified cluster architecture

---

*Last updated: March 12, 2026, after meeting with Shakarian. Shakarian achieved ~80%@10% data. We need to find what we're missing in the paper.*
