# EGE-UNet 模型优化实验记录

## 基线模型

| 项目 | 内容 |
|------|------|
| 模型 | EGE-UNet (MICCAI 2023) |
| 任务 | 皮肤病变分割 (Skin Lesion Segmentation) |
| 数据集 | ISIC2017 / ISIC2018 (7:3划分) |
| 输入尺寸 | 256 x 256 |
| 通道配置 | [8, 16, 24, 32, 48, 64] |
| 优化器 | AdamW (lr=0.001, weight_decay=0.01) |
| 调度器 | CosineAnnealingLR (T_max=50, eta_min=1e-5) |
| 损失函数 | GT_BceDiceLoss (BCE + Dice + 多尺度GT监督) |
| 训练轮数 | 300 epochs |
| 其他参数 | num_workers = 0，amp = False，batch_size = 8|

---

## 创新模块列表

### (Deprecated)模块A：坐标注意力 (Coordinate Attention, CA)

| 项目 | 内容 |
|------|------|
| 来源论文 | Coordinate Attention for Efficient Mobile Network Design (CVPR 2021) |
| 插入位置 | Encoder 1-3 (浅层编码器，Conv2d之后、GroupNorm之前) |
| 解决问题 | 浅层编码器仅用裸卷积，缺乏通道与空间注意力 |
| 核心思路 | 将通道注意力分解为水平方向和垂直方向两个1D编码，保留位置信息的同时完成通道重标定 |
| 控制开关 | `model_config: use_ca=True/False` |
| 新增参数量 | +1,200 (53,374 → 54,574) |
| 实现文件 | `models/modules/coordinate_attention.py` |

**状态**: [x] 待实现 → [x] 已实现 → [x] 已验证 → [x] 已弃用

---

### (Deprecated)模块B：边界感知监督 (Boundary-Aware Supervision, BAS)

| 项目 | 内容 |
|------|------|
| 设计思路 | 新增边界检测头 + 从GT自动生成边界标签 + 边界监督损失 |
| 插入位置 | 解码器中间层 (decoder3 输出, 即 H/8 尺度) |
| 解决问题 | 模型缺乏显式边界感知，病灶边缘分割模糊 |
| 核心思路 | 用Sobel算子从GT mask提取边界，加入Boundary BCE Loss监督边界预测 |
| 控制开关 | `model_config: use_boundary=True/False` |
| 新增参数量 | +2,629 (53,374 → 56,003) |
| 实现文件 | `models/modules/boundary_aware.py`, `utils.py` (损失函数) |

**状态**: [x] 待实现 → [x] 已实现 → [x] 已验证 → [x] 已弃用

---

### 模块A：多尺度深度增强 (MSDE)

| 项目 | 内容 |
|------|------|
| 解决问题 | 浅层编码器单一感受野，不同尺度病灶特征捕获不足 → 欠分割(低Sensitivity) |
| 核心思路 | 双尺度depthwise conv(dilation=1+3) + 逐点融合 + 通道门控 |
| 插入位置 | Encoder 1-3，Conv2d之后、GroupNorm之前（与CA位置相同） |
| 控制开关 | `model_config: use_msde=True/False` |
| 新增参数量 | +2,803 (53,374 → 56,177) |
| 实现文件 | `models/modules/msde.py` |

**状态**: [x] 已实现 → [x] 已验证 →  [ ] 已测试

### 模块B：语义引导解码精炼 (SGDR)

| 项目 | 内容 |
|------|------|
| 解决问题 | 解码器最终特征缺乏精炼和边界感知 → FN/FP均可改善 |
| 核心思路 | 残差特征精炼 + 边界引导注意力(不detach,梯度畅通) |
| 插入位置 | Decoder5 输出(out1, H/2)之后、final Conv之前 |
| 控制开关 | `model_config: use_sgdr=True/False` |
| 新增参数量 | +241 (53,374 → 53,615) |
| 实现文件 | `models/modules/sgdr.py` |

**状态**: [x] 已实现 → [x] 已验证 → [ ] 已测试

---

## 消融实验设计

| 实验编号 | 模块A | 模块B | 说明 |
|----------|:--:|:---:|------|
| Exp-0 | OFF | OFF | 原始 EGE-UNet 基线 |
| Exp-1 | ON  | OFF | 仅加模块A |
| Exp-2 | OFF | ON  | 模块B |
| Exp-3 | ON  | ON  | 两模块同时启用 |

### 评价指标

- mIoU
- F1 / DSC (Dice Similarity Coefficient)
- Accuracy
- Sensitivity
- Specificity
- HD95


## 实验结果记录

### ISIC2018

| 实验 | mIoU | DSC | Acc | Sen | Spe | HD95 | 参数量 | 备注 |
|------|------|-----|-----|-----|-----|------|--------|------|
| Exp-0 | 0.7994 | 0.8885 | 0.9458 | 0.8870 | 0.9647 | - | 53,374 | baseline [v1, lr=0.003, bs=24] |
|       | 0.8056 | 0.8923 | 0.9480 | 0.8852 | 0.9682 | - | 53,374 | baseline [v1, lr=0.001, bs=8] |
|       | 0.8125 | 0.8965 | 0.9499 | 0.8920 | 0.9685 | 14.34 | 53,374 | baseline [v3, lr=0.001] |
|       | 0.8117 | 0.8961 | 0.9498 | 0.8884 | 0.9696 | 13.72 | 53,374 | baseline [v3, TTA] |
| Exp-1 | - | - | - | - | - | - | 54,574 | +CA [v3] |
| Exp-2 | - | - | - | - | - | - | 56,003 | +BAS [v3] |
| Exp-3 | 0.8137 | 0.8973 | 0.9492 | 0.9117 | 0.9612 | - | 57,855 | +CA+BAS [v2, lr=0.003, bs=24] |
|       | 0.8111 | 0.8957 | 0.9493 | 0.8940 | 0.9671 | - | 57,855 | +CA+BAS [v2, lr=0.003, bs=24] |
|       | 0.8176 | 0.8997 | 0.9518 | 0.8881 | 0.9723 | 14.29 | 57,855 | +CA+BAS [v3, lr=0.001] |
|       | 0.8185 | 0.9002 | 0.9522 | 0.8857 | 0.9736 | 13.95 | 57,855 | +CA+BAS [v3, TTA] |

### ISIC2017 

| 实验 | mIoU | DSC | Acc | Sen | Spe | HD95 | 参数量 | 备注 |
|------|------|-----|-----|-----|-----|------|--------|------|
| Exp-0 | - | - | - | - | - | - | 53,374 | 基线A |
| Exp-1 | - | - | - | - | - | - | 54,574 | +模块B |
| Exp-3 | - | - | - | - | - | - | 57,855 | +模块A+模块B |



## 变更日志

| 日期 | 操作 | 详情 |
|------|------|------|
| 2026-04-07 | 创建文档 | 确定两个创新模块方案：CA + BAS。
|      |     |  修改lr=0.003，num_workers = 0，amp = False，batch_size = 8 |
| 2026-04-07 | 模块实现v1 | CA模块(+1,200参数) + BAS模块(+2,629参数) 实现完成 |
| 2026-04-07 | 验证通过v1 | 4组消融配置前向/反向传播、损失函数全部验证通过 |
| 2026-04-07 | Exp-3训练v1 | DSC=0.8886, mIoU=0.7995, 提升微弱(+0.6%) |
| 2026-04-07 | 模块改进v2 | 问题诊断: CA无残差/边界单尺度太粗/损失权重过大 |
|  |  | CA: 加残差连接+可学习scale(init=0,渐进贡献) |
|  |  | BAS: 多尺度边界头(H/8+H/2)+膨胀边界GT+Dice边界损失+边界反馈 |
|  |  | 损失: boundary_weight 1.0→0.5, 移除pos_weight改用BCE+Dice |
| 2026-04-07 | 验证通过v2 | 4组配置全部通过, Exp-3参数量57,855(+8.4%) |
| 2026-04-09 | 训练策略优化v3 | **Phase 1: 训练Recipe升级** |
|  |  | 数据增强：+ColorJitter(p=0.5) +RandomResizedCrop(scale=0.8-1.0) +GaussianBlur(p=0.3) |
|  |  | LR调度：T_max 50→290（单次cosine衰减），+10 epoch LinearLR warmup(start=0.01) |
|  |  | 评价指标：+HD95(边界质量) +TTA推理(4-flip平均) |
|  |  | 目标：统一新recipe重训所有消融实验，拉大改进差距 |
| 2026-04-10 | HD95修复 | 原 compute_hd95 边界提取bug：distance_transform_edt→bool 导致边界永远为空，fallback 用全部前景像素 |
|  |  | 修复：改用 binary_erosion + XOR 提取真正的形态学边界 |
|  |  | HD95 重测结果：Exp-0=14.34, Exp-3=14.29 (仅差0.05, BAS边界改善不明显) |
| 2026-04-10 | **v4 模块重设计** | **诊断**：DSC瓶颈在于欠分割(Sensitivity=0.892)，CA+BAS反而降低Sensitivity(0.892→0.888) |
|  |  | **Bug修复**：myRandomRotation角度在__init__中固定，整个训练用同一角度，已修复为每次__call__随机 |
|  |  | **CA缺陷**：BatchNorm与模型GroupNorm不一致；8/16/24窄通道bottleneck退化；无多尺度感受野 |
|  |  | **BAS缺陷**：H/8尺度边界亚像素不可学；detach()切断梯度；参数多效果差 |
|  |  | **新模块A: MSDE**(Multi-Scale Depthwise Enhancement)：双尺度dilation(1+3)+通道门控+GroupNorm，+2,803参数 |
|  |  | **新模块B: SGDR**(Semantic-Guided Decoder Refinement)：H/2特征精炼+边界引导注意力(不detach)，+241参数 |
|  |  | **新损失**：TverskyDiceLoss(α=0.3,β=0.7) 偏向惩罚FN提升Sensitivity |
|  |  | **新增强**：Mixup(alpha=0.2, p=0.3) |
|  |  | 5组消融配置(Exp-0~4)前向/反向传播全部验证通过 |
| | | |
