# RSTT-S 渐进压缩实验

本轮从 `checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth` 开始，逐步得到可由现有 `RSTTTiny` 独立加载的两帧模型。正式实验目录为 `experiments/progressive_rstt_s_140k_b16`；最终测量和结论见该目录的 `REPORT.md`。

## 结构与训练阶段

| 阶段 | 通道 | 输入→输出帧数 | Encoder / Decoder 每级深度 | 训练步数 | 参数量 |
|---|---:|---:|---|---:|---:|
| 原始 RSTT-S | 96 | 4→7 | 4 / 4 | 已有权重 | 4,485,275 |
| 01 直接重建头 | 96 | 4→7 | 4 / 4 | 10,000 | 3,786,984 |
| 02 两帧输入适配 | 96 | 2→3 | 4 / 4 | 20,000 | 3,669,384 |
| 03 编码器减深 | 96 | 2→3 | 2 / 4 | 10,000 | 3,062,292 |
| 04 解码器减深 | 96 | 2→3 | 2 / 2 | 10,000 | 2,133,564 |
| 05 通道裁剪 | 64 | 2→3 | 2 / 2 | 20,000 | 984,572 |
| 06 通道裁剪 | 48 | 2→3 | 2 / 2 | 20,000 | 579,036 |
| 07 通道裁剪 | 32 | 2→3 | 2 / 2 | 30,000 | 286,140 |
| 08 纯 GT 微调 | 32 | 2→3 | 2 / 2 | 20,000 | 286,140 |

四级 U-Net 保留。深度 2 表示一组普通窗口 block 和移位窗口 block。每个 decoder block 包含 self-attention、cross-attention 和 MLP。

输入端保留 RGB→C 的 3×3 卷积。已有教师的 `back_RBs=0`。原模型还注册了一个未参与前向计算的第四下采样卷积，共 147,552 个参数；渐进版本移除此注册项，加载教师后的前向输出已验证一致。

直接重建头为 `Conv2d(C, 48, 3, padding=1) → PixelShuffle(4)`。第一阶段保留原始 RGB 三线性基线；第二阶段切换为时间线性插值 `align_corners=True`，再逐帧双线性空间插值 `align_corners=False`。两帧版本输出两个端点及中间帧。

## 权重迁移与训练

- 第一阶段前 2,000 步仅训练新重建头，之后解冻主干。
- 时间注意力偏置按有符号时间偏移居中截取，并重新生成索引。
- 每次深度裁剪比较保留 `[0,1]` 或 `[2,3]` 的验证表现，均保留普通/移位窗口顺序。
- 通道重要性来自输入投影及八个编解码阶段的校准激活 RMS，并结合输入卷积权重 RMS。每个最细注意力分组保留相同数量的通道；卷积、Q/K/V、投影、LayerNorm 和跳接使用一致的通道映射。MLP 隐藏维另行筛选。
- 第 01–07 阶段损失为 `GT Charbonnier + 0.3 × 前阶段输出蒸馏 + 0.1 × 原始教师输出蒸馏 + 0.05 × 特征蒸馏`。特征先对应时间片和裁剪通道，再按教师每通道 RMS 归一化。
- 第 08 阶段仅用 GT。最终从第 07、08 阶段的最佳权重中选择验证分数更高的一份。

默认 batch 16，Vimeo/REDS 每 batch 各 8 个样本，HR patch 128×128、LR patch 32×32，BF16 训练。头部学习率 `1e-4`、主干 `2e-5`；通道裁剪阶段统一 `5e-5`，各阶段余弦衰减至 `1e-6`。每 2,000 步以固定 Vimeo 128 + REDS 128 验证片段评估 FP32 PSNR-Y，按两个数据集均值选择最佳权重。阶段衔接使用前阶段最佳权重。

## 运行指令

在 SRCHIP 环境运行完整实验：

```bash
cd /home/tyzheng/RSTT
conda run --no-capture-output -n SRCHIP python -B train_progressive.py \
  --output experiments/progressive_rstt_s_140k_b16
```

从已保存进度恢复：

```bash
conda run --no-capture-output -n SRCHIP python -B train_progressive.py \
  --output experiments/progressive_rstt_s_140k_b16 --resume
```

`--batch-size 8` 可调整为每数据集 4 个样本；`--stage-steps` 接受八个逗号分隔的步数。新实验使用新的 `--output` 目录。

最终完整评测：

```bash
conda run --no-capture-output -n SRCHIP python -B evaluate_progressive.py \
  --run-dir experiments/progressive_rstt_s_140k_b16 --workers 2 --fps
```

评测同时加载本轮 Tiny、此前 20 万步续训的 Tiny 和原始 RSTT-S，输出逐帧 CSV、分组 PSNR/SSIM 和同设备 FPS。Vimeo 测试每片段七帧；Vid4 每个全局帧计分一次，四段序列等权；REDS 使用原有浮点 LR 的中间三帧协议。Vimeo/Vid4 的 LR 使用 PyTorch 抗锯齿 bicubic 生成并量化为 RGB8。

## 文件与加载

每阶段目录保存 `initial.pth`、`best.pth`、`last.pth`、`migration.json`、`train.jsonl`、修改前后验证结果和 `summary.json`。`last.pth` 包括优化器、调度器及 RNG 状态。总目录保存阶段汇总、数据划分、最终权重和校验值。

最终部署只加载 `final.pth`：

```python
import torch
from models.RSTT_tiny import RSTTTiny

checkpoint = torch.load(
    'experiments/progressive_rstt_s_140k_b16/final.pth',
    map_location='cpu', weights_only=True,
)
model = RSTTTiny(**checkpoint['model_config'])
model.load_state_dict(checkpoint['state_dict'], strict=True)
model = model.cuda().eval()

# pair: [B, 2, 3, H, W]，RGB 浮点值范围 [0, 1]
with torch.inference_mode():
    output = model(pair.cuda())  # [B, 3, 3, 4H, 4W]
```

模型实现：`models/RSTT_progressive.py`；训练：`train_progressive.py`；评测：`evaluate_progressive.py`。独立部署沿用 `models/RSTT_tiny.py`、`benchmark_tiny_4k.py` 和 `infer_tiny_video.py`。
