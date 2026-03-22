# L2P JAX — Working Colab Cells (March 22, 2026)

These two cells run the original Google Research L2P code on modern JAX/Flax.
Tested on Colab T4 GPU with High RAM. Task 1: 97.8% accuracy.

---

## Cell 1 — Setup (run once, wait for ✅)

```python
!pip install -q ml_collections clu tensorflow-datasets importlib_resources

# Fresh clone
!rm -rf /tmp/l2p-jax
!git clone https://github.com/google-research/l2p /tmp/l2p-jax
!wget -q --show-progress -O /tmp/l2p-jax/ViT-B_16.npz \
  "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"

# Patch 1: collections.Mapping -> collections.abc.Mapping (Python 3.10+)
!sed -i 's/collections.Mapping/collections.abc.Mapping/g' \
  /tmp/l2p-jax/libml/utils_vit.py

# Patch 2: removed jax.tree_* functions (JAX 0.6+)
!grep -rl "jax\.tree_map" /tmp/l2p-jax/ --include="*.py" | \
  xargs sed -i 's/jax\.tree_map/jax.tree.map/g'
!grep -rl "jax\.tree_leaves" /tmp/l2p-jax/ --include="*.py" | \
  xargs sed -i 's/jax\.tree_leaves/jax.tree.leaves/g'
!grep -rl "jax\.tree_flatten" /tmp/l2p-jax/ --include="*.py" | \
  xargs sed -i 's/jax\.tree_flatten/jax.tree.flatten/g'

```python
# Patch 3: add unfreeze to train_continual.py (FrozenDict vs optax)
!sed -i 's/^import flax$/import flax\nfrom flax.core import unfreeze/' \
  /tmp/l2p-jax/train_continual.py
!sed -i 's/model_state = dict(variables)/model_state = unfreeze(dict(variables))/' \
  /tmp/l2p-jax/train_continual.py
!sed -i 's/params = model_state.pop("params")/params = unfreeze(model_state.pop("params"))/' \
  /tmp/l2p-jax/train_continual.py

# Patch 4: add unfreeze to utils.py
!sed -i '1s/^/from flax.core import unfreeze\n/' /tmp/l2p-jax/libml/utils.py
!sed -i 's/optimizer = init_state.optimizer.replace(target=restored_params)/optimizer = init_state.optimizer.replace(target=unfreeze(restored_params))/' \
  /tmp/l2p-jax/libml/utils.py

# Patch 5: disable checkpointing (shim can't serialize)
!sed -i '/ckpt = checkpoint.MultihostCheckpoint/c\    pass  # checkpointing disabled' \
  /tmp/l2p-jax/train_continual.py
!sed -i '/state = ckpt.restore_or_initialize/c\    # ckpt restore disabled' \
  /tmp/l2p-jax/train_continual.py
!sed -i '/ckpt.save(/c\              pass  # ckpt save disabled' \
  /tmp/l2p-jax/train_continual.py

print("✅ All patches applied")
!ls -lh /tmp/l2p-jax/ViT-B_16.npz
```

---

## Cell 2 — Run (restart kernel first if Cell 1 was re-run)

```python
import sys, types, time, functools
import numpy as np

sys.path.insert(0, '/tmp/l2p-jax')

# === Shim 1: flax.optim (removed in modern flax, backed by optax) ===
import optax, jax, jax.numpy as jnp, flax

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
    def create(self, p):
        tx = self._make_tx(); return OptaxOptimizer(tx, p, tx.init(p))

class _Mom:
    def __init__(self, beta=0.9, nesterov=False, learning_rate=None):
        self.beta, self.nesterov = beta, nesterov
    def _make_tx(self):
        if self.beta == 0 and not self.nesterov:
            return optax.chain(optax.scale(-1.0))
        return optax.chain(
            optax.trace(decay=self.beta, nesterov=self.nesterov), optax.scale(-1.0))
    def create(self, p):
        tx = self._make_tx(); return OptaxOptimizer(tx, p, tx.init(p))

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
print("✅ flax.optim shim")

# === Shim 2: tensorflow_addons (deprecated, stub 3 functions) ===
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
print("✅ tfa shim")

tf.config.experimental.set_visible_devices([], "GPU")
print(f"✅ JAX {jax.__version__} | {jax.devices()}")

# === Import L2P ===
import train_continual
from libml import utils, utils_vit
from models import vit
from clu import metric_writers
from flax.core import unfreeze
from configs.cifar100_l2p import get_config

config = get_config()
config.init_checkpoint = "/tmp/l2p-jax/ViT-B_16.npz"
config.save_last_ckpt_only = False

print(f"⚙️  coeff={config.pull_constraint_coeff} bs={config.per_device_batch_size} epochs={config.num_epochs}")

# === Verbose training with task-by-task feedback ===
_orig = train_continual.train_and_evaluate_per_task

def run():
    import os
    workdir = "/tmp/l2p-jax/workdir"
    if not tf.io.gfile.exists(workdir): tf.io.gfile.makedirs(workdir)

    rng = jax.random.PRNGKey(config.seed)
    rng, trl, evl, csl, cml, tds, ntc = \
        train_continual.get_train_eval_components(config, rng)
    print(f"📊 {ntc} classes, {config.continual.num_tasks} tasks")

    rng, mrng = jax.random.split(rng)
    model, state = train_continual.create_train_state(config, mrng,
        input_shape=tds.element_spec["image"].shape[1:], num_classes=ntc)
    print(f"🧠 {config.model_name} created")

    if config.get("init_checkpoint"):
        state = utils.load_and_custom_init_checkpoint(config=config, init_state=state)
        print("📦 Weights loaded")

    writer = metric_writers.create_default_writer(workdir,
        just_logging=jax.process_index() > 0)
    am = np.zeros((config.continual.num_tasks, config.continual.num_tasks))

    ovm, ovp = None, None
    if config.get("prompt_pool_param") and \
       config.prompt_pool_param.embedding_key == "cls":
        omc, omc2 = vit.create_original_vit(config.model_name)
        ovm = functools.partial(omc, num_classes=ntc)
        rng, mrng = jax.random.split(rng)
        oip = ovm(train=False).init(mrng,
            jnp.ones(tds.element_spec["image"].shape[1:]))
        ovp = utils_vit.load_pretrained(
            pretrained_path=config.init_checkpoint,
            init_params=oip["params"], model_config=omc2)
        print("🔑 CLS extractor loaded")

    print(f"\n{'='*60}")
    print(f"🏁 GO: {config.continual.num_tasks} tasks x {config.num_epochs} epochs")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    for tid in range(config.continual.num_tasks):
        ts = time.time()
        print(f"🚀 Task {tid+1}/{config.continual.num_tasks}...", flush=True)

        state, rng = _orig(tid, config, workdir,
            model=model, state=state,
            original_vit_model=ovm, original_vit_params=ovp,
            num_total_class=ntc, train_ds_list=trl, eval_ds_list=evl,
            class_stats_list=csl, class_mask_list=cml, acc_matrix=am,
            writer=writer, replay_buffer=None, rng=rng)

        te = time.time() - ts; tt = time.time() - t0
        d = np.diag(am); avg = np.mean(d[:tid+1])
        print(f"✅ Task {tid+1}: {te/60:.1f}min | avg_acc={avg*100:.1f}% | total={tt/60:.1f}min")
        if tid > 0:
            f = np.mean((np.max(am, axis=1) - am[:, tid])[:tid])
            print(f"   forgetting={f*100:.1f}%")
        print(flush=True)

    final = np.mean(np.diag(am))
    print(f"\n{'='*60}")
    print(f"🏆 FINAL AVG ACCURACY: {final*100:.1f}% in {(time.time()-t0)/60:.1f}min")
    print(f"{'='*60}")
    print(f"\n{am}")

run()
```

---

## Notes

- **Task 1 takes ~27min** (JIT compilation overhead). Tasks 2-10 are faster.
- **Total runtime**: ~2-4 hours on T4 GPU
- **Must restart kernel** between Cell 1 and Cell 2 if Cell 1 was re-run
- Checkpointing is disabled — if runtime disconnects, restart from scratch
- All deprecation warnings from tensorflow_datasets are harmless
