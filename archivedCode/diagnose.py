"""Diagnostic script - paste this into a single Colab cell."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir('/content/l2p/l2p')

# Hide GPU from TF before it initializes
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

# Suppress the massive parameter dump
import logging
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)

import jax
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

from absl import flags
from ml_collections import config_flags
import train_continual

FLAGS = flags.FLAGS
try:
    config_flags.DEFINE_config_file('my_config', None, '', lock_config=True)
    flags.DEFINE_string('workdir', None, '')
    flags.DEFINE_string('exp_id', None, '')
except Exception:
    pass  # flags already defined from previous run

FLAGS(['main',
    '--my_config=configs/cifar100_l2p.py',
    '--workdir=./output_diag',
    '--my_config.init_checkpoint=./ViT-B_16.npz',
])

print("=== STARTING TRAINING ===", flush=True)
try:
    train_continual.train_and_evaluate(FLAGS.my_config, FLAGS.workdir)
    print("=== TRAINING COMPLETE ===")
except Exception as e:
    print(f"\n=== CRASH ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()
