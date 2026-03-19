"""Clean launcher for L2P training. Bypasses absl flags entirely."""
import os
import sys

# CRITICAL: Hide GPU from TensorFlow BEFORE importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

# Suppress the massive parameter dump
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
absl_logger = logging.getLogger('absl')
absl_logger.setLevel(logging.INFO)

import jax
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# Load config directly as Python object - no flags needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.cifar100_l2p import get_config
import train_continual

config = get_config()
config.init_checkpoint = './ViT-B_16.npz'
workdir = './output_run6'

print(f"\n{'='*60}")
print(f"L2P Training Configuration:")
print(f"  pull_constraint_coeff: {config.pull_constraint_coeff}")
print(f"  prompt pool size: {config.prompt_pool_param.pool_size}")
print(f"  prompt length: {config.prompt_pool_param.length}")
print(f"  top_k: {config.prompt_pool_param.top_k}")
print(f"  batch_size: {config.per_device_batch_size}")
print(f"  epochs: {config.num_epochs}")
print(f"  learning_rate: {config.learning_rate}")
print(f"  checkpoint: {config.init_checkpoint}")
print(f"  workdir: {workdir}")
print(f"{'='*60}\n")

print("=== STARTING TRAINING ===", flush=True)
try:
    train_continual.train_and_evaluate(config, workdir)
    print("\n=== TRAINING COMPLETE ===")
except Exception as e:
    print(f"\n{'='*60}")
    print(f"=== CRASH ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"{'='*60}")
    import traceback
    traceback.print_exc()
