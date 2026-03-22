#!/usr/bin/env python3
"""
L2P JAX Setup & Run Script
===========================
Self-contained script that clones the L2P repo, applies all compatibility
patches for modern JAX/Flax/Python, and runs training with verbose output.

Usage (in Colab or any Jupyter notebook with GPU):
    !pip install -q ml_collections clu tensorflow-datasets importlib_resources
    %run /path/to/run_l2p_patched.py

Or from command line:
    pip install ml_collections clu tensorflow-datasets importlib_resources
    python run_l2p_patched.py
"""

import os, sys, subprocess, time, types, functools
import numpy as np

# ============================================================
# STEP 1: Clone repo and download weights
# ============================================================

L2P_DIR = "/tmp/l2p-jax"
WEIGHTS_PATH = f"{L2P_DIR}/ViT-B_16.npz"
WEIGHTS_URL = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"

if not os.path.exists(L2P_DIR):
    print("📥 Cloning L2P repo...")
    subprocess.run(["git", "clone", "https://github.com/google-research/l2p", L2P_DIR],
                   check=True, capture_output=True)

if not os.path.exists(WEIGHTS_PATH):
    print("📥 Downloading ViT-B/16 weights (~330MB)...")
    subprocess.run(["wget", "-q", "--show-progress", "-O", WEIGHTS_PATH, WEIGHTS_URL],
                   check=True)

print(f"✅ Repo at {L2P_DIR}, weights at {WEIGHTS_PATH}")

# ============================================================
# STEP 2: Apply source patches (idempotent)
# ============================================================

import re

def patch_file(filepath, replacements):
    """Apply a list of (old, new) replacements to a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)

# --- Patch all .py files for removed jax.tree_* functions ---
print("🔧 Patching jax.tree_* deprecations...")
for root, dirs, files in os.walk(L2P_DIR):
    for fname in files:
        if fname.endswith('.py'):
            fpath = os.path.join(root, fname)
            patch_file(fpath, [
                ('jax.tree_map', 'jax.tree.map'),
                ('jax.tree_leaves', 'jax.tree.leaves'),
                ('jax.tree_flatten', 'jax.tree.flatten'),
            ])

# --- Patch utils_vit.py: collections.Mapping ---
print("🔧 Patching collections.Mapping...")
patch_file(f"{L2P_DIR}/libml/utils_vit.py", [
    ('collections.Mapping', 'collections.abc.Mapping'),
])

# --- Patch train_continual.py: unfreeze + disable checkpointing ---
print("🔧 Patching train_continual.py...")
tc_path = f"{L2P_DIR}/train_continual.py"
with open(tc_path, 'r') as f:
    tc = f.read()

# Add unfreeze import (only if not already present)
if 'from flax.core import unfreeze' not in tc:
    tc = tc.replace('import flax\n', 'import flax\nfrom flax.core import unfreeze\n')

# Unfreeze model state and params
tc = tc.replace(
    'model_state = dict(variables)',
    'model_state = unfreeze(dict(variables))')
tc = tc.replace(
    'params = model_state.pop("params")',
    'params = unfreeze(model_state.pop("params"))')

# Disable checkpointing (our shim can't serialize)
tc = re.sub(
    r'    ckpt = checkpoint\.MultihostCheckpoint\(.*?\)',
    '    pass  # checkpointing disabled', tc)
tc = re.sub(
    r'    state = ckpt\.restore_or_initialize\(state\)',
    '    # ckpt restore disabled', tc)
tc = re.sub(
    r'              ckpt\.save\(.*?\)',
    '              pass  # ckpt save disabled', tc)

with open(tc_path, 'w') as f:
    f.write(tc)

# --- Patch utils.py: unfreeze ---
print("🔧 Patching utils.py...")
utils_path = f"{L2P_DIR}/libml/utils.py"
with open(utils_path, 'r') as f:
    utils_content = f.read()

if 'from flax.core import unfreeze' not in utils_content:
    utils_content = 'from flax.core import unfreeze\n' + utils_content

utils_content = utils_content.replace(
    'optimizer = init_state.optimizer.replace(target=restored_params)',
    'optimizer = init_state.optimizer.replace(target=unfreeze(restored_params))')

with open(utils_path, 'w') as f:
    f.write(utils_content)

print("✅ All patches applied")

# ============================================================
# STEP 3: Install runtime shims
# ============================================================

sys.path.insert(0, L2P_DIR)

import optax, jax, jax.numpy as jnp, flax

# --- Shim: flax.optim (removed in Flax > 0.5.3) ---

class _HP:
    def __init__(self, lr=0.0): self.learning_rate = lr
    def replace(self, **kw):
        hp = _HP(self.learning_rate)
        for k, v in kw.items(): setattr(hp, k, v)
        return hp

class _OD:
    def __init__(self, hp=None): self.hyper_params = hp or [_HP()]

def _pts(path):
    parts = []
    for k in path:
        if hasattr(k, 'key'): parts.append(str(k.key))
        elif hasattr(k, 'idx'): parts.append(str(k.idx))
        else: parts.append(str(k))
    return '/'.join(parts)

class OptaxOptimizer:
    """Drop-in replacement for flax.optim.Optimizer, backed by optax."""
    def __init__(self, tx, target, opt_state, optimizer_def=None, param_labels=None):
        self.tx, self.target, self.opt_state = tx, target, opt_state
        self.optimizer_def = optimizer_def or _OD()
        self.param_labels = param_labels

    def apply_gradient(self, grad, learning_rate=None, hyper_params=None):
        updates, new_os = self.tx.update(grad, self.opt_state, self.target)
        if hyper_params is not None and self.param_labels is not None:
            lr_tree = jax.tree.map(lambda l: hyper_params[l].learning_rate, self.param_labels)
            updates = jax.tree.map(lambda u, lr: u * lr, updates, lr_tree)
        elif learning_rate is not None:
            updates = jax.tree.map(lambda u: u * learning_rate, updates)
        return OptaxOptimizer(self.tx, optax.apply_updates(self.target, updates),
            new_os, self.optimizer_def, self.param_labels)

    def replace(self, target=None):
        return OptaxOptimizer(self.tx, target if target is not None else self.target,
            self.opt_state, self.optimizer_def, self.param_labels)

jax.tree_util.register_pytree_node(OptaxOptimizer,
    lambda o: ((o.target, o.opt_state), (o.tx, o.optimizer_def, o.param_labels)),
    lambda aux, ch: OptaxOptimizer(aux[0], ch[0], ch[1], aux[1], aux[2]))

class _Adam:
    def __init__(self, weight_decay=0, learning_rate=None): self.wd = weight_decay
    def _make_tx(self):
        return optax.chain(optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay=self.wd), optax.scale(-1.0))
    def create(self, p): tx = self._make_tx(); return OptaxOptimizer(tx, p, tx.init(p))

class _Mom:
    def __init__(self, beta=0.9, nesterov=False, learning_rate=None):
        self.beta, self.nesterov = beta, nesterov
    def _make_tx(self):
        if self.beta == 0 and not self.nesterov: return optax.chain(optax.scale(-1.0))
        return optax.chain(optax.trace(decay=self.beta, nesterov=self.nesterov),
            optax.scale(-1.0))
    def create(self, p): tx = self._make_tx(); return OptaxOptimizer(tx, p, tx.init(p))

class _MPT:
    def __init__(self, fn): self.fn = fn

class _MO:
    def __init__(self, *groups): self.groups = groups
    def create(self, params):
        transforms = {i: od._make_tx() for i, (_, od) in enumerate(self.groups)}
        def lf(path, leaf):
            ps = _pts(path)
            for i, (t, _) in enumerate(self.groups):
                if t.fn(ps, leaf): return i
            return 0
        labels = jax.tree_util.tree_map_with_path(lf, params)
        tx = optax.multi_transform(transforms, labels)
        return OptaxOptimizer(tx, params, tx.init(params),
            _OD([_HP() for _ in self.groups]), labels)

m = types.ModuleType('flax.optim')
m.Optimizer, m.Adam, m.Momentum = OptaxOptimizer, _Adam, _Mom
m.ModelParamTraversal, m.MultiOptimizer = _MPT, _MO
flax.optim = m; sys.modules['flax.optim'] = m
print("✅ flax.optim shim installed")

# --- Shim: tensorflow_addons (fully deprecated) ---
import tensorflow as tf

def _rot(img, rad):
    c, s = tf.math.cos(rad), tf.math.sin(rad)
    return tf.raw_ops.ImageProjectiveTransformV3(images=tf.expand_dims(img,0),
        transforms=tf.expand_dims(tf.cast([c,-s,0,s,c,0,0,0],tf.float32),0),
        output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)[0]

def _trans(img, t):
    return tf.raw_ops.ImageProjectiveTransformV3(images=tf.expand_dims(img,0),
        transforms=tf.expand_dims(tf.cast([1,0,-t[0],0,1,-t[1],0,0],tf.float32),0),
        output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)[0]

def _xform(img, t, **kw):
    return tf.raw_ops.ImageProjectiveTransformV3(images=tf.expand_dims(img,0),
        transforms=tf.expand_dims(tf.cast(t,tf.float32),0),
        output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR", fill_mode="NEAREST", fill_value=0.0)[0]

tfa = types.ModuleType('tensorflow_addons')
tfa_i = types.ModuleType('tensorflow_addons.image')
tfa_i.rotate, tfa_i.translate, tfa_i.transform = _rot, _trans, _xform
tfa.image = tfa_i
sys.modules['tensorflow_addons'] = tfa
sys.modules['tensorflow_addons.image'] = tfa_i
print("✅ tensorflow_addons shim installed")

tf.config.experimental.set_visible_devices([], "GPU")
print(f"✅ JAX {jax.__version__} | {jax.devices()}")

# ============================================================
# STEP 4: Import L2P and configure
# ============================================================

import train_continual
from libml import utils, utils_vit
from models import vit
from clu import metric_writers
from flax.core import unfreeze
from configs.cifar100_l2p import get_config

config = get_config()
config.init_checkpoint = WEIGHTS_PATH
config.save_last_ckpt_only = False

print(f"⚙️  coeff={config.pull_constraint_coeff} "
      f"bs={config.per_device_batch_size} epochs={config.num_epochs}")

# ============================================================
# STEP 5: Run training with verbose task-by-task feedback
# ============================================================

_orig = train_continual.train_and_evaluate_per_task

def run():
    workdir = f"{L2P_DIR}/workdir"
    if not tf.io.gfile.exists(workdir):
        tf.io.gfile.makedirs(workdir)

    rng = jax.random.PRNGKey(config.seed)
    rng, trl, evl, csl, cml, tds, ntc = \
        train_continual.get_train_eval_components(config, rng)
    print(f"📊 {ntc} classes, {config.continual.num_tasks} tasks")

    rng, mrng = jax.random.split(rng)
    model, state = train_continual.create_train_state(config, mrng,
        input_shape=tds.element_spec["image"].shape[1:],
        num_classes=ntc)
    print(f"🧠 {config.model_name} created")

    if config.get("init_checkpoint"):
        state = utils.load_and_custom_init_checkpoint(
            config=config, init_state=state)
        print("📦 Weights loaded")

    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0)
    am = np.zeros((config.continual.num_tasks,
                    config.continual.num_tasks))

    ovm, ovp = None, None
    if (config.get("prompt_pool_param") and
            config.prompt_pool_param.embedding_key == "cls"):
        omc, omc2 = vit.create_original_vit(config.model_name)
        ovm = functools.partial(omc, num_classes=ntc)
        rng, mrng = jax.random.split(rng)
        oip = ovm(train=False).init(
            mrng, jnp.ones(tds.element_spec["image"].shape[1:]))
        ovp = utils_vit.load_pretrained(
            pretrained_path=config.init_checkpoint,
            init_params=oip["params"], model_config=omc2)
        print("🔑 CLS extractor loaded")

    print(f"\n{'='*60}")
    print(f"🏁 GO: {config.continual.num_tasks} tasks x "
          f"{config.num_epochs} epochs")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    for tid in range(config.continual.num_tasks):
        ts = time.time()
        print(f"🚀 Task {tid+1}/{config.continual.num_tasks}...",
              flush=True)

        state, rng = _orig(tid, config, workdir,
            model=model, state=state,
            original_vit_model=ovm, original_vit_params=ovp,
            num_total_class=ntc,
            train_ds_list=trl, eval_ds_list=evl,
            class_stats_list=csl, class_mask_list=cml,
            acc_matrix=am, writer=writer,
            replay_buffer=None, rng=rng)

        te = time.time() - ts
        tt = time.time() - t0
        d = np.diag(am)
        avg = np.mean(d[:tid+1])
        print(f"✅ Task {tid+1}: {te/60:.1f}min | "
              f"avg_acc={avg*100:.1f}% | total={tt/60:.1f}min")
        if tid > 0:
            f = np.mean((np.max(am, axis=1) -
                         am[:, tid])[:tid])
            print(f"   forgetting={f*100:.1f}%")
        print(flush=True)

    final = np.mean(np.diag(am))
    total = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"🏆 FINAL AVG ACCURACY: {final*100:.1f}% in {total:.1f}min")
    print(f"{'='*60}")
    print(f"\nAccuracy matrix:\n{am}")


if __name__ == "__main__":
    run()
