# Intel Arc A370M GPU installation for PhysisML

*Read this in: [Italiano](../../it/setup/gpu_intel_arc.md)*

Hardware: Intel Arc A370M (DG2), 4GB GDDR6
```
03:00.0 Display controller: Intel Corporation DG2 [Arc A370M] (rev 05)
```
System: KDE Neon 24.04 (Ubuntu-based), kernel 6.17.x

---

## Check this first: which torch actually gets imported

**Measured 23 August 2026:** `build.sh` reports `Device: cpu (conda
physisml_gpu)` — the conda env's python imports torch from
`~/.local/lib/python3.12/site-packages`, version `2.12.1+cu130` (a CUDA build),
whose `torch.xpu.is_available()` is False. The whole L0->L10 curriculum was
trained on CPU.

The cause is that `~/.local/lib/python3.12/site-packages` is on `sys.path` for
any python 3.12, conda envs included, and it **wins** over the env's own
packages. A `pip install torch` run without the env active therefore shadows
the XPU installation silently.

Check:

```bash
$HOME/miniforge3/envs/physisml_gpu/bin/python -c \
  "import torch; print(torch.__file__, torch.__version__, torch.xpu.is_available())"
```

If the printed path contains `.local`, the env is not the one running. Fix with
`PYTHONNOUSERSITE=1`, or by removing the torch in `~/.local`.

---

## Installation status (2026-04-14)

| Component | Version | Status |
|---|---|---|
| PyTorch | 2.8.0+cpu | ✅ Working (CPU) |
| IPEX CPU | 2.8.0+cpu | ⚠️ Residual conflicts |
| intel-level-zero-gpu | 1.3.29735 | ✅ |
| intel-opencl-icd | 24.39.31294 | ✅ |
| Intel oneAPI Base Toolkit | 2025.3.2 | ✅ (~10GB installed) |
| Intel PTI | 0.16.0 | ✅ (symlink 0.10→0.16 created) |
| **XPU active** | — | ❌ Versions not aligned |

**Main problem**: IPEX XPU requires an exact ecosystem of versions
that are hard to align on a system with mixed packages.

**Recommended solution**: a dedicated conda environment (see section below).

---

## Verified dependencies (first-hand experience)

```
torch 2.5.1+cxx11.abi  ←→  IPEX 2.5.10+xpu  ←→  torchvision 0.20.1+cxx11.abi
torch 2.8.0 (xpu)      ←→  IPEX 2.8.10+xpu  ←→  torchvision 0.23.0+xpu
```

Every component must belong to the same family. Mixing them (e.g. torch CUDA
with IPEX XPU) causes `undefined symbol` errors.

## Required system libraries

All the following components must be present BEFORE installing IPEX XPU:

```bash
# 1. Intel GPU driver
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu noble client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu.list
sudo apt update && sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero

# 2. Intel oneAPI Base Toolkit (~10GB)
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo apt-key add -
echo "deb https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update && sudo apt install -y intel-oneapi-base-toolkit intel-oneapi-mkl

# 3. Intel PTI (Profiling Tools Interface)
sudo apt install -y intel-pti

# 4. Libraries in the system path
echo "/opt/intel/oneapi/mkl/latest/lib" | sudo tee /etc/ld.so.conf.d/intel-oneapi.conf
echo "/opt/intel/oneapi/compiler/latest/lib" | sudo tee -a /etc/ld.so.conf.d/intel-oneapi.conf
echo "/opt/intel/oneapi/pti/latest/lib" | sudo tee -a /etc/ld.so.conf.d/intel-oneapi.conf
sudo ldconfig

# 5. PTI symlink (version 0.16 → 0.10 required by torch 2.5.1)
sudo ln -sf /opt/intel/oneapi/pti/0.16/lib/libpti_view.so.0.16.0 \
  /opt/intel/oneapi/pti/0.16/lib/libpti_view.so.0.10
sudo ldconfig
```

---

## Recommended installation: conda environment

This approach avoids all version conflicts:

```bash
# Install miniforge/conda if not present
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate

# Create a dedicated environment with the exact versions
conda create -n physisml_gpu python=3.12 -y
conda activate physisml_gpu

# Install torch + IPEX XPU from the Intel source
pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi \
  --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/eu/

pip install intel-extension-for-pytorch==2.5.10+xpu \
  --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/eu/

# Enable oneAPI in the environment
source /opt/intel/oneapi/setvars.sh
```

Check:
```bash
python3 -c "
import torch, intel_extension_for_pytorch as ipex
print(torch.xpu.is_available())        # must be True
print(torch.xpu.get_device_name(0))    # Intel Arc A370M
"
```

---

## Direct download with wget (bypasses pip 403)

The Intel server often returns 403 to pip but accepts wget:

```bash
# Torch 2.5.1+cxx11.abi (860MB)
wget --user-agent "pip/24.0" \
  "https://download.pytorch-extension.intel.com/ipex_stable/xpu/torch-2.5.1%2Bcxx11.abi-cp312-cp312-linux_x86_64.whl" \
  -O torch-2.5.1+cxx11.abi.whl

# IPEX 2.8.10+xpu (1GB)
wget --user-agent "pip/24.0" \
  "https://download.pytorch-extension.intel.com/ipex_stable/xpu/intel_extension_for_pytorch-2.8.10%2Bxpu-cp312-cp312-linux_x86_64.whl" \
  -O ipex-2.8.10+xpu.whl

# torchvision 0.20.1+cxx11.abi
wget --user-agent "pip/24.0" \
  "https://download.pytorch-extension.intel.com/ipex_stable/xpu/torchvision-0.20.1%2Bcxx11.abi-cp312-cp312-linux_x86_64.whl" \
  -O torchvision-0.20.1+cxx11.abi.whl

# Install from local files
pip install torch-2.5.1+cxx11.abi.whl torchvision-0.20.1+cxx11.abi.whl
pip install ipex-2.8.10+xpu.whl
```

---

## Expected speedup (Arc A370M vs i7-1360P CPU)

| Task | CPU (16T) | Arc A370M XPU | Speedup |
|---|---|---|---|
| Training step (8K vocab, B=8) | ~10 seq/s | ~200-400 seq/s | **20-40×** |
| L3 phase 0 (21MB corpus) | ~13 hours | ~20-40 min | **20-40×** |
| Full L0→L2 build | ~2 hours | ~5-10 min | **20-40×** |
| Inference (generate) | ~220 seq/s | ~2000+ seq/s | ~10× |

---

## PhysisML code — no changes needed

The code detects the device automatically (updated 2026-04-14):

```python
# tests/test_1/splx/torch_model.py
DEVICE = get_device()  # auto: cuda > xpu > mps > cpu

# TrainerB automatically moves model and tensors to DEVICE
self.model = model.to(self.device)
ids = torch.from_numpy(ids_np).long().to(self.device)
```

It is enough for `torch.xpu.is_available()` to return `True` and all
training will use the GPU automatically.

---

## Notes on the system environment (2026-04-14)

After several attempts the system has these extra files in ld.so.conf.d:
- `/etc/ld.so.conf.d/intel-mkl.conf` — MKL + compiler + PTI paths
- `/etc/ld.so.conf.d/torch.conf` — PyTorch lib path

To clean up:
```bash
sudo rm /etc/ld.so.conf.d/torch.conf
sudo ldconfig
```
