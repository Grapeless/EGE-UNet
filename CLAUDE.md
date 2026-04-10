# EGE-UNet 改进项目 — 工作接续文档

> 最后更新：2026-04-09

## 项目目标

基于 EGE-UNet (MICCAI 2023) 复现，添加两个创新模块（CA + BAS），产出足以支撑 **SCI 四区论文** 的实验结果。任务是皮肤病变分割（ISIC2017 / ISIC2018）。

---

## 代码架构

```
/hy-tmp/EGE-UNet/
├── models/
│   ├── egeunet.py                  # 主模型：6级编解码器 + GAB跳连 + 深监督
│   └── modules/
│       ├── coordinate_attention.py # 模块A：坐标注意力 (CA, CVPR 2021)
│       ├── boundary_aware.py       # 模块B：边界感知监督 (BAS, 自设计)
│       └── __init__.py
├── datasets/dataset.py             # NPY_datasets, 加载PNG图像+mask
├── configs/config_setting.py       # 所有超参、增强pipeline、模块开关
├── engine.py                       # train/val/test循环 + HD95 + TTA
├── train.py                        # 主训练脚本 + warmup调度器
├── utils.py                        # 损失函数 + 数据增强类 + 工具函数
├── EXPERIMENT_LOG.md               # 实验日志（结果表格+变更记录）
└── data/
    ├── isic2017/  (train/val, images+masks)
    └── isic2018/  (train/val, images+masks)
```

## 模型配置开关

在 `configs/config_setting.py` 的 `model_config` 字典中：

```python
'use_ca': True/False,        # 坐标注意力 (encoder 1-3)
'use_boundary': True/False,  # 边界感知监督 (多尺度)
```

对应 4 组消融：Exp-0(OFF/OFF), Exp-1(ON/OFF), Exp-2(OFF/ON), Exp-3(ON/ON)

损失函数自动跟随 `use_boundary` 开关切换 `GT_BceDiceLoss` / `GT_BceDiceLoss_WithBoundary`。

---

## 已完成的工作（按时间线）

### 2026-04-07: v1 模块实现
- 实现 CA（坐标注意力）插入 encoder 1-3，+1,200 参数
- 实现 BAS（边界感知监督）：Sobel 边界GT + 单尺度边界头 + BCE损失
- 4 组消融前向/反向验证通过
- Exp-3 训练结果：DSC=0.8886, mIoU=0.7995，提升微弱(+0.6%)

### 2026-04-07: v2 模块改进
- **CA 改进**：加残差连接 + 可学习 scale (init=0, 渐进贡献)
- **BAS 改进**：多尺度边界头(H/8+H/2) + 膨胀边界GT(radius=3) + Dice边界损失 + 边界反馈注入
- 损失权重：boundary_weight 1.0→0.5，移除 pos_weight 改用 BCE+Dice
- 验证通过，Exp-3 参数量 57,855 (+8.4%)

### 2026-04-09: v3 训练策略优化（本次会话）

**修改了以下文件：**

1. **`utils.py`** — 新增 3 个数据增强类：
   - `myColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)` — 仅影响图像
   - `myRandomResizedCrop(256, 256, scale=(0.8,1.0))` — 替代 myResize，同步裁剪图像+mask
   - `myGaussianBlur(p=0.3, kernel_size=3, sigma=(0.1,1.0))` — 仅影响图像

2. **`configs/config_setting.py`** —
   - 训练 pipeline：加入 ColorJitter → HFlip → VFlip → Rotation → RandomResizedCrop → GaussianBlur
   - `T_max`: 50 → 290（单次 cosine 衰减，配合 warmup）
   - 新增 `use_warmup=True`, `warmup_epochs=10`, `warmup_start_factor=0.01`

3. **`train.py`** —
   - 用 `SequentialLR(LinearLR + CosineAnnealingLR, milestones=[10])` 实现 warmup
   - 训练结束后自动跑 TTA 测试（普通 + TTA 两组结果）

4. **`engine.py`** —
   - 新增 `compute_hd95()` 函数（基于 scipy.ndimage.distance_transform_edt）
   - 新增 `_tta_predict()` 函数（original + HFlip + VFlip + HVFlip 四次预测取平均）
   - `test_one_epoch()` 新增 `use_tta` 参数 + 逐图像 HD95 计算 + 日志输出 HD95

5. **`EXPERIMENT_LOG.md`** — 合并表格，备注加版本号 [v1]/[v2]/[v3]

**所有改动已通过快速前向测试验证（增强类、HD95、TTA、warmup 调度器均正常）。**

---

## 当前实验结果

### ISIC2018

| 实验 | mIoU | DSC | Acc | Sen | Spe | 备注 |
|------|------|-----|-----|-----|-----|------|
| Exp-0 | 0.7994 | 0.8885 | 0.9458 | 0.8870 | 0.9647 | [v1, lr=0.003, bs=24] |
| Exp-0 | 0.8056 | 0.8923 | 0.9480 | 0.8852 | 0.9682 | [v1, lr=0.001, bs=8] |
| Exp-3 | 0.8137 | 0.8973 | 0.9492 | 0.9117 | 0.9612 | [v2, lr=0.003, bs=24] |
| Exp-3 | 0.8111 | 0.8957 | 0.9493 | 0.8940 | 0.9671 | [v2, lr=0.003, bs=24] |

**v3 recipe 的 8 组实验（ISIC2018 × 4 + ISIC2017 × 4）尚未开始。**

---

## 下一步计划（按优先级）

详细计划见 `/root/.claude/plans/elegant-singing-fox.md`

| 优先级 | 任务 | 状态 |
|:------:|------|:----:|
| **1** | 用 v3 recipe 跑 ISIC2018 Exp-0 和 Exp-3，看差距 | 待执行 |
| **2** | 补全 ISIC2018 消融 (Exp-1, Exp-2) | 待执行 |
| **3** | ISIC2017 全部 4 组实验 | 待执行 |
| **4** | 分析结果：如差距 ≥1% mIoU 则继续，否则进入模块微调 | 待决策 |
| **5** | SOTA 对比表（引用原论文数据） | 待执行 |
| **6** | 定性可视化（challenging cases 对比图） | 待执行 |
| **7** | 多 seed 运行 + Wilcoxon 显著性检验 | 待执行 |
| **备选** | BAS 损失权重搜索 / CA 位置调整 / 第三微模块 | 仅在差距不够时 |

## 跑实验方法

```bash
cd /hy-tmp/EGE-UNet

# 1. 编辑 configs/config_setting.py 设置:
#    - datasets = 'isic18' 或 'isic17'
#    - use_ca = True/False
#    - use_boundary = True/False (criterion 会自动切换)

# 2. 启动训练
python train.py
# 训练结束后自动跑 test + TTA test，日志输出 mIoU/DSC/Acc/Sen/Spe/HD95
```