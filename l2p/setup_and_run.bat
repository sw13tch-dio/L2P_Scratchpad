@echo off
echo ========================================
echo  L2P JAX Setup and Run
echo ========================================

set PYTHON=C:\Users\sw13t\AppData\Local\Programs\Python\Python312\python.exe
set L2P_DIR=C:\Users\sw13t\OneDrive - Syracuse University\SU_Spring2026\Leibnitz Lab\L2P_Code\l2p
set VENV=%L2P_DIR%\venv_new

cd /d "%L2P_DIR%"

echo [1/4] Creating virtual environment...
"%PYTHON%" -m venv "%VENV%"

echo [2/4] Installing JAX with CUDA support...
"%VENV%\Scripts\pip.exe" install --upgrade pip
"%VENV%\Scripts\pip.exe" install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo [3/4] Installing other dependencies...
"%VENV%\Scripts\pip.exe" install flax optax ml_collections clu tensorflow-cpu tensorflow-datasets numpy absl-py

echo [4/4] Starting training...
"%VENV%\Scripts\python.exe" run_l2p.py

pause
