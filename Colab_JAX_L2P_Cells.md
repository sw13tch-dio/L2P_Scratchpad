# JAX L2P (Original) — Colab Cells

**Use this if PyTorch keeps giving ~40%.** The authors' code with paper-correct defaults.

---

## Cell 1: Clone + install

```python
import os
os.chdir('/content')
!rm -rf /content/l2p-jax
!git clone https://github.com/google-research/l2p /content/l2p-jax
os.chdir('/content/l2p-jax')

# Colab has JAX/Flax. Install extra deps (avoid downgrading JAX - use Colab's version)
!pip install -q ml_collections clu tensorflow-datasets tensorflow-addons
print("✅ Done")
```

---

## Cell 2: Patch tensorflow_addons (required — tfa doesn't support Python 3.12)

```python
# Replace tensorflow_addons with scipy fallback (tfa not available on Colab/Python 3.12)
path = "/content/l2p-jax/augment/augment_ops.py"
with open(path) as f:
    content = f.read()

old = "from tensorflow_addons import image as image_transform"
new = '''try:
  from tensorflow_addons import image as image_transform
except ImportError:
  import numpy as np
  import scipy.ndimage as ndimage
  class _ImageTransform:
    @staticmethod
    def rotate(images, angle):
      img = images[..., :3]
      deg = -np.degrees(float(angle))
      out = tf.numpy_function(lambda x: ndimage.rotate(x, deg, reshape=False, order=1, mode="nearest").astype(np.uint8), [img], tf.uint8)
      out.set_shape(img.shape)
      return tf.concat([out, tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)], -1)
    @staticmethod
    def translate(images, shift):
      img = images[..., :3]
      dx, dy = float(shift[0]), float(shift[1])
      out = tf.numpy_function(lambda x: ndimage.shift(x, (dy, dx, 0), order=1, mode="nearest").astype(np.uint8), [img], tf.uint8)
      out.set_shape(img.shape)
      return tf.concat([out, tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)], -1)
    @staticmethod
    def transform(images, transforms):
      img = images[..., :3]
      t = np.array(transforms, dtype=np.float32)
      m = np.array([[t[0],t[1],t[2]],[t[3],t[4],t[5]],[t[6],t[7],1.]])
      inv = np.linalg.inv(m)
      out = tf.numpy_function(lambda x: ndimage.affine_transform(x, inv[:2,:2], inv[:2,2], order=1, mode="nearest").astype(np.uint8), [img], tf.uint8)
      out.set_shape(img.shape)
      return tf.concat([out, tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)], -1)
  image_transform = _ImageTransform()'''

content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print("✅ Patched augment_ops.py (tensorflow_addons -> scipy)")
```

---

## Cell 3: Download ViT weights

```python
# ViT-B/16 pretrained on ImageNet-21k (paper's model)
!wget -q https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -O /content/l2p-jax/ViT-B_16.npz
assert os.path.exists('/content/l2p-jax/ViT-B_16.npz'), "Download failed - try manual download"
print("✅ ViT weights downloaded")
```

**If wget fails (403):** Download manually from a browser:  
https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz  
Upload to Colab at `/content/l2p-jax/ViT-B_16.npz`

---

## Cell 4: Run training (10% data via flag)

```python
!cd /content/l2p-jax && python main.py \
  --my_config=configs/cifar100_l2p.py \
  --workdir=./output_jax_10pct \
  --my_config.init_checkpoint=/content/l2p-jax/ViT-B_16.npz \
  --my_config.subsample_rate=10
```

The `--my_config.subsample_rate=10` override gives 10% of the data per task.

---

## What's different from PyTorch

| | PyTorch (JH-LEE-KR) | JAX (Original) |
|---|-------------------|----------------|
| **Author** | Community | Paper authors |
| **λ (pull)** | 0.1 default → we set 0.5 | 1.0 in config |
| **Prompt length** | 5 | 10 |
| **top_k** | 5 | 4 |
| **Weights** | HuggingFace (patched) | Google .npz |

The JAX config already uses stronger pull (`1.0`) and the authors’ setup. If ViT download works, this should give results closer to the paper.

---

## If ViT download fails

1. Open in browser: https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz  
2. Download to your machine  
3. In Colab: **Files** → **Upload** → upload `ViT-B_16.npz`  
4. Move it: `!mv /content/ViT-B_16.npz /content/l2p-jax/`  
5. Re-run Cell 3

---

## TensorBoard (optional)

```python
%load_ext tensorboard
%tensorboard --logdir /content/l2p-jax/output_jax_10pct
```
