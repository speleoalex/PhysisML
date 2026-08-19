# Installazione GPU Intel Arc A370M per PhysisML

*Leggi in: [English](../../en/setup/gpu_intel_arc.md)*

Hardware: Intel Arc A370M (DG2), 4GB GDDR6
```
03:00.0 Display controller: Intel Corporation DG2 [Arc A370M] (rev 05)
```
Sistema: KDE Neon 24.04 (Ubuntu-based), kernel 6.17.x

---

## Stato installazione (2026-04-14)

| Componente | Versione | Stato |
|---|---|---|
| PyTorch | 2.8.0+cpu | ✅ Funzionante (CPU) |
| IPEX CPU | 2.8.0+cpu | ⚠️ Conflitti residui |
| intel-level-zero-gpu | 1.3.29735 | ✅ |
| intel-opencl-icd | 24.39.31294 | ✅ |
| Intel oneAPI Base Toolkit | 2025.3.2 | ✅ (~10GB installato) |
| Intel PTI | 0.16.0 | ✅ (symlink 0.10→0.16 creato) |
| **XPU attivo** | — | ❌ Versioni non allineate |

**Problema principale**: IPEX XPU richiede un ecosistema esatto di versioni
che sono difficili da allineare su un sistema con pacchetti misti.

**Soluzione raccomandata**: ambiente conda dedicato (vedi sezione sotto).

---

## Dipendenze accertate (esperienza diretta)

```
torch 2.5.1+cxx11.abi  ←→  IPEX 2.5.10+xpu  ←→  torchvision 0.20.1+cxx11.abi
torch 2.8.0 (xpu)      ←→  IPEX 2.8.10+xpu  ←→  torchvision 0.23.0+xpu
```

Ogni componente deve essere della stessa famiglia. Mescolare (es. torch CUDA
con IPEX XPU) causa `undefined symbol` errors.

## Librerie di sistema necessarie

Tutti i componenti seguenti devono essere presenti PRIMA di installare IPEX XPU:

```bash
# 1. Driver GPU Intel
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

# 4. Librerie nel path di sistema
echo "/opt/intel/oneapi/mkl/latest/lib" | sudo tee /etc/ld.so.conf.d/intel-oneapi.conf
echo "/opt/intel/oneapi/compiler/latest/lib" | sudo tee -a /etc/ld.so.conf.d/intel-oneapi.conf
echo "/opt/intel/oneapi/pti/latest/lib" | sudo tee -a /etc/ld.so.conf.d/intel-oneapi.conf
sudo ldconfig

# 5. Symlink PTI (versione 0.16 → richiesta 0.10 da torch 2.5.1)
sudo ln -sf /opt/intel/oneapi/pti/0.16/lib/libpti_view.so.0.16.0 \
  /opt/intel/oneapi/pti/0.16/lib/libpti_view.so.0.10
sudo ldconfig
```

---

## Installazione raccomandata: ambiente conda

Questo approccio evita tutti i conflitti di versione:

```bash
# Installa miniforge/conda se non presente
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate

# Crea ambiente dedicato con le versioni esatte
conda create -n physisml_gpu python=3.12 -y
conda activate physisml_gpu

# Installa torch + IPEX XPU dalla fonte Intel
pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi \
  --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/eu/

pip install intel-extension-for-pytorch==2.5.10+xpu \
  --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/eu/

# Attiva oneAPI nell'ambiente
source /opt/intel/oneapi/setvars.sh
```

Verifica:
```bash
python3 -c "
import torch, intel_extension_for_pytorch as ipex
print(torch.xpu.is_available())        # deve essere True
print(torch.xpu.get_device_name(0))    # Intel Arc A370M
"
```

---

## Download diretto con wget (bypassa 403 di pip)

Il server Intel spesso restituisce 403 a pip ma accetta wget:

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

# Installa da file locali
pip install torch-2.5.1+cxx11.abi.whl torchvision-0.20.1+cxx11.abi.whl
pip install ipex-2.8.10+xpu.whl
```

---

## Speedup atteso (Arc A370M vs i7-1360P CPU)

| Task | CPU (16T) | Arc A370M XPU | Speedup |
|---|---|---|---|
| Training step (8K vocab, B=8) | ~10 seq/s | ~200-400 seq/s | **20-40×** |
| L3 phase 0 (21MB corpus) | ~13 ore | ~20-40 min | **20-40×** |
| Build L0→L2 completo | ~2 ore | ~5-10 min | **20-40×** |
| Inference (generate) | ~220 seq/s | ~2000+ seq/s | ~10× |

---

## Codice PhysisML — nessuna modifica necessaria

Il codice rileva automaticamente il device (aggiornato 2026-04-14):

```python
# tests/test_1/splx/torch_model.py
DEVICE = get_device()  # auto: cuda > xpu > mps > cpu

# TrainerB sposta automaticamente modello e tensori su DEVICE
self.model = model.to(self.device)
ids = torch.from_numpy(ids_np).long().to(self.device)
```

Basta che `torch.xpu.is_available()` restituisca `True` e tutto il
training userà la GPU automaticamente.

---

## Note sull'ambiente di sistema (2026-04-14)

Dopo vari tentativi il sistema ha questi file aggiuntivi in ld.so.conf.d:
- `/etc/ld.so.conf.d/intel-mkl.conf` — MKL + compiler + PTI paths
- `/etc/ld.so.conf.d/torch.conf` — PyTorch lib path

Se si vuole pulire:
```bash
sudo rm /etc/ld.so.conf.d/torch.conf
sudo ldconfig
```
