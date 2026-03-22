# =============================================================================
# L2P Colab Setup — Complete Cells (copy each block into a separate Colab cell)
# =============================================================================
# These patches fix: (1) Colab weight-loading failures, (2) Paper mismatches
# See L2P_Replication_Fixes_Report.md for full explanation.
# =============================================================================

# -----------------------------------------------------------------------------
# CELL 1: Clone repo
# -----------------------------------------------------------------------------
"""
import os
os.chdir('/content')
!rm -rf /content/l2p-fix
!git clone https://github.com/JH-LEE-KR/l2p-pytorch /content/l2p-fix
os.chdir('/content/l2p-fix')
print("✅ Cell 1 done — repo cloned")
"""

# -----------------------------------------------------------------------------
# CELL 2: Apply all patches
# -----------------------------------------------------------------------------
"""
import os, re
os.chdir('/content/l2p-fix')

# ── PATCH A: vision_transformer.py (Colab: remove broken .npz loader) ─────────
vt = open("vision_transformer.py").read()
vt = re.sub(r"[ \t]*pretrained_custom_load=[^\n]+\n", "", vt)
open("vision_transformer.py", "w").write(vt)
print("✅ Patch A: vision_transformer.py")

# ── PATCH B: models.py (Colab: HuggingFace weights + pos_embed fix) ───────────
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
    except Exception:
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
    print(f"Loaded {repo_id}  missing={len(missing)} unexpected={len(unexpected)}")
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    m = _build(dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs))
    return _load(m, "timm/vit_base_patch16_224.augreg_in21k") if pretrained else m
''')
print("✅ Patch B: models.py")

# ── PATCH C: engine.py (Paper: gradient accumulation for batch 128) ──────────
engine_code = open("engine.py").read()

old_block = \"\"\"        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()\"\"\"

new_block = \"\"\"        accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
        loss_scaled = loss / accum_steps
        loss_scaled.backward()
        _ga_counter += 1
        if _ga_counter % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()\"\"\"

gather_block = '    # gather the stats from all processes\\n    metric_logger.synchronize_between_processes()'
flush_block = '''    accum_steps = getattr(args, "gradient_accumulation_steps", 1)
    if _ga_counter % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()'''

if old_block in engine_code:
    engine_code = engine_code.replace(
        '    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):',
        '    optimizer.zero_grad()\\n    _ga_counter = 0\\n    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):'
    )
    engine_code = engine_code.replace(old_block, new_block)
    engine_code = engine_code.replace(gather_block, flush_block, 1)  # CRITICAL: 1 = only train_one_epoch
    open("engine.py", "w").write(engine_code)
    print("✅ Patch C: engine.py — gradient accumulation")
else:
    print("❌ ERROR: Could not find optimizer block in engine.py")

# ── PATCH D: config — add gradient_accumulation_steps AND subsample ──────────
config_code = open("configs/cifar100_l2p.py").read()
additions = []
if 'gradient_accumulation_steps' not in config_code:
    additions.append("    subparsers.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps')")
if 'subsample' not in config_code:
    additions.append("    subparsers.add_argument('--subsample', default=1.0, type=float, help='fraction of training data per task (e.g. 0.1 for 10%%)')")
if additions:
    config_code = config_code.rstrip() + "\n" + "\n".join(additions) + "\n"
    open("configs/cifar100_l2p.py", "w").write(config_code)
    print("✅ Patch D: config — added gradient_accumulation_steps, subsample (if missing)")

print("\\n✅ Cell 2 done — all patches applied")
"""

# -----------------------------------------------------------------------------
# CELL 3: Run training (10% data, paper-aligned)
# -----------------------------------------------------------------------------
"""
!python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py cifar100_l2p \\
  --model vit_base_patch16_224 \\
  --batch-size 16 \\
  --data-path ./data/ \\
  --output_dir ./output_10pct \\
  --pull_constraint_coeff 0.5 \\
  --gradient_accumulation_steps 8 \\
  --subsample 0.1
"""
