"""
Patch for augment_ops.py to remove tensorflow_addons dependency.
Run this in Colab after cloning l2p: exec(open('fix_tfa_for_colab.py').read())
Or copy the patch block into a Colab cell.
"""

PATCH = '''
# --- PATCH: Replace tensorflow_addons with scipy (no extra deps) ---
import math
from absl import flags
from absl import logging
from augment import color_util
from libml.losses import apply_label_smoothing
import tensorflow.compat.v2 as tf
import numpy as np

def _tf_rotate(image, radians):
  """Rotation using scipy via tf.numpy_function."""
  def _rotate_np(img):
    import scipy.ndimage as ndimage
    img_np = img.numpy() if hasattr(img, 'numpy') else img
    return ndimage.rotate(img_np, -np.degrees(radians), reshape=False, order=1, mode='nearest')
  return tf.numpy_function(lambda x: np.array(_rotate_np(x), dtype=np.uint8), [image], tf.uint8)

def _tf_translate(image, dx, dy):
  """Translation using scipy via tf.numpy_function."""
  def _translate_np(img, shift_x, shift_y):
    import scipy.ndimage as ndimage
    img_np = img.numpy() if hasattr(img, 'numpy') else img
    return ndimage.shift(img_np, (shift_y, shift_x, 0), order=1, mode='nearest')
  return tf.numpy_function(lambda x: np.array(_translate_np(x, dx, dy), dtype=np.uint8), [image], tf.uint8)

def _tf_transform(image, transforms):
  """Projective transform - 8 params [a0,a1,a2,b0,b1,b2,c0,c1]."""
  def _transform_np(img, tr):
    import scipy.ndimage as ndimage
    img_np = img.numpy() if hasattr(img, 'numpy') else img
    h, w = img_np.shape[:2]
    tr = tr.numpy() if hasattr(tr, 'numpy') else tr
    # Build 3x3 matrix from 8 params (c2=1)
    m = np.array([[tr[0], tr[1], tr[2]], [tr[3], tr[4], tr[5]], [tr[6], tr[7], 1.]], dtype=np.float32)
    inv = np.linalg.inv(m)
    return ndimage.affine_transform(img_np, inv[:2,:2], inv[:2,2], order=1, mode='nearest')
  return tf.numpy_function(lambda x,t: np.array(_transform_np(x,t), dtype=np.uint8), [image, transforms], tf.uint8)

class _ImageTransform:
  @staticmethod
  def rotate(images, angle):
    # images: [h,w,4] from wrap(), angle in radians
    img_3ch = images[..., :3]
    rotated = _tf_rotate(img_3ch, angle)
    ones = tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)
    return tf.concat([rotated, ones], axis=-1)
  
  @staticmethod  
  def translate(images, shift):
    dx, dy = float(shift[0]), float(shift[1])
    img_3ch = images[..., :3]
    translated = _tf_translate(img_3ch, dx, dy)
    ones = tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)
    return tf.concat([translated, ones], axis=-1)
  
  @staticmethod
  def transform(images, transforms):
    # transforms: [8] array
    img_3ch = images[..., :3]
    tr = tf.constant(transforms, dtype=tf.float32)
    transformed = _tf_transform(img_3ch, tr)
    ones = tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)
    return tf.concat([transformed, ones], axis=-1)

image_transform = _ImageTransform()
# --- END PATCH ---
'''

# Simpler: just replace the import line and add a fallback module
IMPORT_REPLACEMENT = '''try:
  from tensorflow_addons import image as image_transform
except ImportError:
  # Fallback: use scipy (tensorflow_addons unavailable on Python 3.12)
  import numpy as np
  import scipy.ndimage as ndimage
  class _ImageTransform:
    @staticmethod
    def rotate(images, angle):
      img = images[..., :3]
      deg = -np.degrees(float(angle))
      def _rot(x):
        return ndimage.rotate(x, deg, reshape=False, order=1, mode="nearest").astype(np.uint8)
      out = tf.numpy_function(_rot, [img], tf.uint8)
      out.set_shape(img.shape)
      return tf.concat([out, tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)], -1)
    @staticmethod
    def translate(images, shift):
      img = images[..., :3]
      dx, dy = float(shift[0]), float(shift[1])
      def _tr(x):
        return ndimage.shift(x, (dy, dx, 0), order=1, mode="nearest").astype(np.uint8)
      out = tf.numpy_function(_tr, [img], tf.uint8)
      out.set_shape(img.shape)
      return tf.concat([out, tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)], -1)
    @staticmethod
    def transform(images, transforms):
      img = images[..., :3]
      t = np.array(transforms, dtype=np.float32)
      m = np.array([[t[0],t[1],t[2]],[t[3],t[4],t[5]],[t[6],t[7],1.]])
      inv = np.linalg.inv(m)
      def _tr(x):
        return ndimage.affine_transform(x, inv[:2,:2], inv[:2,2], order=1, mode="nearest").astype(np.uint8)
      out = tf.numpy_function(_tr, [img], tf.uint8)
      out.set_shape(img.shape)
      return tf.concat([out, tf.ones([tf.shape(images)[0], tf.shape(images)[1], 1], dtype=images.dtype)], -1)
  image_transform = _ImageTransform()
'''

def apply_patch():
  path = "/content/l2p-jax/augment/augment_ops.py"
  try:
    with open(path) as f:
      content = f.read()
  except FileNotFoundError:
    path = "augment/augment_ops.py"
    with open(path) as f:
      content = f.read()
  
  old = "from tensorflow_addons import image as image_transform"
  if old not in content:
    print("Line not found - file may already be patched")
    return False
  
  # Replace with try/except block
  new = IMPORT_REPLACEMENT.strip()
  content = content.replace(old, new)
  with open(path, 'w') as f:
    f.write(content)
  print("Patched augment_ops.py - tensorflow_addons replaced with scipy fallback")
  return True

if __name__ == "__main__":
  apply_patch()
