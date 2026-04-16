# Experiment Documentation: Mamba-Wavelet Enhanced EGE-UNet

## 1. Project Overview

**Base Model**: EGE-UNet (MICCAI 2023) — Efficient Group Enhanced UNet for Skin Lesion Segmentation  
**Dataset**: ISIC 2017 / ISIC 2018  
**Task**: Binary skin lesion segmentation (256x256 input)  
**Baseline Performance (ISIC2018)**: mIoU = 80.94 +- 0.11, DSC = 89.46 +- 0.07

---

## 2. Proposed Improvements

### Module 1: Mamba SS2D Bottleneck Enhancement

| Item | Details |
|------|---------|
| **Reference** | VM-UNet (2024), U-Mamba (2024), Swin-UMamba (2024) |
| **Position** | After encoder6, before decoder1 (bottleneck, 64ch @ 8x8) |
| **Mechanism** | 4-directional selective state space scanning for global context |
| **Key Params** | d_model=64, d_state=16, expand=1.5 |
| **Added Params** | 24,128 |
| **Target Bottleneck** | GHPA lacks global sequential dependency modeling |

### Module 2: Wavelet Frequency-Domain Feature Enhancement

| Item | Details |
|------|---------|
| **Reference** | WaveSNet (2024), Wavelet-enhanced UNet series (2024-2025) |
| **Position** | Before GAB1 (t1: 8ch @ 128x128) and GAB2 (t2: 16ch @ 64x64) |
| **Mechanism** | Haar DWT decomposition -> separate low/high-freq processing -> IDWT reconstruction |
| **Key Feature** | Learnable alpha scaling for high-frequency (boundary) enhancement |
| **Added Params** | 3,914 (wave1: 921, wave2: 2,993) |
| **Target Bottleneck** | GAB skip connections lack frequency-domain awareness |

---

## 3. Model Configurations (Ablation)

| Config | use_ssm | use_wavelet | Total Params | Delta |
|--------|---------|-------------|-------------|-------|
| **Baseline** | False | False | 53,374 | - |
| **+SSM** | True | False | 77,502 | +24,128 |
| **+Wavelet** | False | True | 57,288 | +3,914 |
| **+Both (Ours)** | True | True | 81,416 | +28,042 |

To set ablation configs, modify `configs/config_setting.py`:
```python
model_config = {
    'num_classes': 1,
    'input_channels': 3,
    'c_list': [8,16,24,32,48,64],
    'bridge': True,
    'gt_ds': True,
    'use_ssm': True,       # Toggle for Mamba SS2D
    'use_wavelet': True,   # Toggle for Wavelet enhancement
}
```

---

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 256 x 256 |
| Batch Size | 8 |
| Epochs | 300 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| LR Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-5) |
| Loss | GT_BceDiceLoss (BCE + Dice + Deep Supervision) |
| Seed | 42 |
| GPU | Single GPU |
| Augmentation | HFlip(0.5), VFlip(0.5), Rotation(0-360, 0.5) |

---

## 5. Experiment Plan

### Experiment 1: Ablation Study (Core)

Run 4 configurations on ISIC2018 (3 seeds each for error bars):

| # | Model | use_ssm | use_wavelet | Seeds |
|---|-------|---------|-------------|-------|
| 1 | Baseline | False | False | 42, 123, 456 |
| 2 | +SSM | True | False | 42, 123, 456 |
| 3 | +Wavelet | False | True | 42, 123, 456 |
| 4 | +Both (Ours) | True | True | 42, 123, 456 |

**Metrics**: mIoU, DSC (F1), Accuracy, Sensitivity, Specificity, HD95, Params, FLOPs

### Experiment 2: SOTA Comparison (ISIC2018)

Compare against:
- U-Net (2015)
- U-Net++ (2018)
- Att-UNet (2018)
- TransUNet (2021)
- SwinUNet (2021)
- MALUNet (2022)
- EGE-UNet (2023, baseline)
- VM-UNet (2024)
- **Ours**

### Experiment 3: Frequency-Domain Visualization

- Visualize Haar DWT sub-bands (LL, LH, HL, HH) before and after WaveletFeatureEnhance
- Show high-frequency enhancement effect on boundary features
- Compare boundary predictions with/without wavelet module

### Experiment 4: SSM Scan Direction Ablation

| Config | Directions | Expected |
|--------|-----------|----------|
| 1-dir | -> only | Lower bound |
| 2-dir | -> <- | Moderate |
| 4-dir | -> <- (down) (up) | Full (default) |

### Experiment 5: Wavelet Enhancement Position Ablation

| Config | Position | Expected |
|--------|----------|----------|
| GAB1 only | t1 (128x128) | HD95 improvement |
| GAB2 only | t2 (64x64) | Moderate |
| GAB1+GAB2 | t1 + t2 (default) | Best |

### Experiment 6: Cross-Dataset Generalization

- Train on ISIC2018 -> Test on ISIC2017
- Demonstrates robustness of proposed modules

### Experiment 7: Efficiency Comparison

Report for all methods:
- Parameter count (K)
- FLOPs (G)
- Inference time (ms) on single GPU

---

## 6. Results Template

### 6.1 Ablation Results (ISIC2018)

| Model | mIoU (%) | DSC (%) | Acc (%) | Sen (%) | Spe (%) | HD95 | Params (K) |
|-------|----------|---------|---------|---------|---------|------|------------|
| Baseline | 80.94+-0.11 | 89.46+-0.07 | - | - | - | - | 53.4 |
| +SSM | | | | | | | 77.5 |
| +Wavelet | | | | | | | 57.3 |
| +Both (Ours) | | | | | | | 81.4 |

### 6.2 SOTA Comparison (ISIC2018)

| Method | Year | mIoU (%) | DSC (%) | Params (K) | FLOPs (G) |
|--------|------|----------|---------|------------|-----------|
| U-Net | 2015 | | | | |
| U-Net++ | 2018 | | | | |
| Att-UNet | 2018 | | | | |
| TransUNet | 2021 | | | | |
| EGE-UNet | 2023 | 80.94 | 89.46 | 53.4 | |
| **Ours** | 2025 | | | 81.4 | |

---

## 7. Code Changes Summary

### Modified Files

| File | Changes |
|------|---------|
| `models/egeunet.py` | Added SS2D, MambaBottleneck, HaarWavelet2D, WaveletFeatureEnhance classes; Modified EGEUNet init/forward with use_ssm/use_wavelet flags |
| `configs/config_setting.py` | Added use_ssm and use_wavelet to model_config |

### New Module Architecture

```
EGE-UNet + Mamba + Wavelet:

Encoder:
  x -> [Conv+GN+GELU+Pool] -> t1 (8ch, 128x128)
  -> [Conv+GN+GELU+Pool] -> t2 (16ch, 64x64)
  -> [Conv+GN+GELU+Pool] -> t3 (24ch, 32x32)
  -> [GHPA+GN+GELU+Pool] -> t4 (32ch, 16x16)
  -> [GHPA+GN+GELU+Pool] -> t5 (48ch, 8x8)
  -> [GHPA+GELU] -> t6 (64ch, 8x8)
  -> [MambaBottleneck] -> t6' (64ch, 8x8)  *** NEW ***

Decoder:
  t6' -> [GHPA+GN+GELU] -> out5 + GAB5(t6,t5) -> out5
  -> [GHPA+GN+GELU+Up] -> out4 + GAB4(t5,t4) -> out4
  -> [GHPA+GN+GELU+Up] -> out3 + GAB3(t4,t3) -> out3
  -> [Conv+GN+GELU+Up] -> out2
     t2' = WaveletEnhance(t2) *** NEW ***
     + GAB2(t3,t2') -> out2
  -> [Conv+GN+GELU+Up] -> out1
     t1' = WaveletEnhance(t1) *** NEW ***
     + GAB1(t2,t1') -> out1
  -> [Conv+Up] -> out0 (1ch, 256x256)
```

---

## 8. Running Experiments

### Quick Test (verify setup)
```bash
cd /hy-tmp/EGE-UNet
python3 -c "
import torch
from models.egeunet import EGEUNet
model = EGEUNet(**{'num_classes':1,'input_channels':3,'c_list':[8,16,24,32,48,64],
                   'bridge':True,'gt_ds':True,'use_ssm':True,'use_wavelet':True})
x = torch.randn(2,3,256,256)
gt_pre, out = model(x)
print(f'Output: {out.shape}, Params: {sum(p.numel() for p in model.parameters())}')
"
```

### Training
```bash
cd /hy-tmp/EGE-UNet
python3 train.py
```

### Ablation Configs
Edit `configs/config_setting.py` model_config dict:
- Baseline: `use_ssm=False, use_wavelet=False`
- +SSM: `use_ssm=True, use_wavelet=False`
- +Wavelet: `use_ssm=False, use_wavelet=True`
- +Both: `use_ssm=True, use_wavelet=True`

---

## 9. Paper Narrative

**Title suggestion**: "Mamba-Wavelet Enhanced Lightweight UNet for Accurate Skin Lesion Segmentation"

**Key story**:
1. EGE-UNet has attention asymmetry (shallow encoder lacks attention) and spatial-only feature processing
2. Mamba SS2D at bottleneck captures global morphological context with O(N) complexity
3. Wavelet enhancement at skip connections explicitly separates and enhances boundary (high-frequency) features
4. The two modules work synergistically: SSM improves "what to segment", wavelet improves "where to segment"
5. Both modules maintain the lightweight advantage (<82K params total)

**Target venues**: Biomedical Signal Processing and Control, Computers in Biology and Medicine, Applied Sciences, Diagnostics
