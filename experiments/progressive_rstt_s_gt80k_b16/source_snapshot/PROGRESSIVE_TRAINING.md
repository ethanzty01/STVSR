# RSTT-S 渐进压缩实验

本轮从 `checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth` 开始，逐步得到可由现有 `RSTTTiny` 独立加载的两帧模型。前八阶段训练 140,000 步，目录为 `experiments/progressive_rstt_s_140k_b16`；随后第九阶段继续纯 GT 训练 80,000 步，目录为 `experiments/progressive_rstt_s_gt80k_b16`，两部分合计 220,000 步。追加训练用于检验充分恢复后与原有 Tiny 的性能差距。

截至 2026-11-17，九个阶段共 **220,000 步训练、四模型完整测试和独立 4K 部署测量均已完成**。最终权重为 `experiments/progressive_rstt_s_gt80k_b16/final.pth`，来自第九阶段局部 74,000 步、累计 214,000 步；固定验证均值 PSNR-Y 为 **29.329027 dB**，较原有 Tiny 提高 **0.121986 dB**。Vid4、Vimeo 三个运动分组及 REDS 的测试 PSNR-Y / SSIM-Y 均提高。训练与评测审计通过，两份 4K120 视频已生成并完整解码核验。详细结果见[实验报告](experiments/progressive_rstt_s_140k_b16/REPORT.md)。

下文保留本轮运行与恢复命令，重新开展实验时使用新的输出目录。

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
| 09 追加纯 GT 恢复 | 32 | 2→3 | 2 / 2 | 80,000 | 286,140 |

四级 U-Net 保留。深度 2 表示一组普通窗口 block 和移位窗口 block。每个 decoder block 包含 self-attention、cross-attention 和 MLP。

输入端保留 RGB→C 的 3×3 卷积。已有教师的 `back_RBs=0`。原模型还注册了一个未参与前向计算的第四下采样卷积，共 147,552 个参数；渐进版本移除此注册项，加载教师后的前向输出已验证一致。

直接重建头为 `Conv2d(C, 48, 3, padding=1) → PixelShuffle(4)`。第一阶段保留原始 RGB 三线性基线；第二阶段切换为时间线性插值 `align_corners=True`，再逐帧双线性空间插值 `align_corners=False`。两帧版本输出两个端点及中间帧。

## 权重迁移与训练

- 第一阶段前 2,000 步仅训练新重建头，之后解冻主干。
- 时间注意力偏置按有符号时间偏移居中截取，并重新生成索引。
- 每次深度裁剪比较保留 `[0,1]` 或 `[2,3]` 的验证表现，均保留普通/移位窗口顺序。
- 通道重要性来自输入投影及八个编解码阶段的校准激活 RMS，并结合输入卷积权重 RMS。每个最细注意力分组保留相同数量的通道；卷积、Q/K/V、投影、LayerNorm 和跳接使用一致的通道映射。MLP 隐藏维另行筛选。
- 第 01–07 阶段损失为 `GT Charbonnier + 0.3 × 前阶段输出蒸馏 + 0.1 × 原始教师输出蒸馏 + 0.05 × 特征蒸馏`。特征先对应时间片和裁剪通道，再按教师每通道 RMS 归一化。
- 第 08、09 阶段仅用 GT。前八阶段从第 07、08 阶段的最佳权重中选择验证分数更高的一份，保存为第一部分的 `final.pth`，再以此初始化第 09 阶段。
- 第 09 阶段的最佳权重候选包含初始化权重和后续验证点，因此按固定验证集选择的最终分数至少保留初始化时的水平。

默认 batch 16，Vimeo/REDS 每 batch 各 8 个样本，HR patch 128×128、LR patch 32×32，BF16 训练。头部学习率 `1e-4`、主干 `2e-5`；通道裁剪阶段统一 `5e-5`，各阶段余弦衰减至 `1e-6`。每 2,000 步以固定 Vimeo 128 + REDS 128 验证片段评估 FP32 PSNR-Y，按两个数据集均值选择最佳权重。阶段衔接使用前阶段最佳权重。

第 09 阶段沿用同一数据划分、增强、batch 和验证协议，全模型学习率从 `5e-5` 余弦衰减至 `1e-6`，仅计算 GT Charbonnier 损失。

## 运行指令

在 SRCHIP 环境运行前八阶段，共 140,000 步：

```bash
cd /home/tyzheng/RSTT
conda run --no-capture-output -n SRCHIP python -B train_progressive.py \
  --output experiments/progressive_rstt_s_140k_b16
```

恢复前八阶段的已保存进度：

```bash
conda run --no-capture-output -n SRCHIP python -B train_progressive.py \
  --output experiments/progressive_rstt_s_140k_b16 --resume
```

`--batch-size 8` 可调整为每数据集 4 个样本；`--stage-steps` 接受八个逗号分隔的步数。新实验使用新的 `--output` 目录。

前八阶段完成并生成 `final.pth` 后，启动第九阶段的 80,000 步纯 GT 训练：

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny_mixed.py \
  --init experiments/progressive_rstt_s_140k_b16/final.pth \
  --output experiments/progressive_rstt_s_gt80k_b16 \
  --steps 80000 --mode supervised --lr 5e-5 --batch-size 16 \
  --compile --compile-mode reduce-overhead --skip-test
```

恢复第九阶段的已保存进度：

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny_mixed.py \
  --output experiments/progressive_rstt_s_gt80k_b16 --resume --skip-test
```

第九阶段完成后，将其验证最佳权重保存为最终部署与评测入口：

```bash
cp experiments/progressive_rstt_s_gt80k_b16/distilled_best.pth \
  experiments/progressive_rstt_s_gt80k_b16/final.pth
```

随后执行最终完整评测，同时比较追加训练前后的模型：

```bash
conda run --no-capture-output -n SRCHIP python -B evaluate_progressive.py \
  --run-dir experiments/progressive_rstt_s_gt80k_b16 \
  --intermediate-checkpoint experiments/progressive_rstt_s_140k_b16/final.pth \
  --workers 2 --fps
```

评测同时加载追加 GT 训练后的 Tiny、前八阶段的 Tiny、此前 20 万步续训的 Tiny 和原始 RSTT-S，输出逐帧 CSV、分组 PSNR/SSIM 和同设备 FPS。Vimeo 测试每片段七帧；Vid4 每个全局帧计分一次，四段序列等权；REDS 使用原有浮点 LR 的中间三帧协议。Vimeo/Vid4 的 LR 使用 PyTorch 抗锯齿 bicubic 生成并量化为 RGB8。

## 文件与加载

前八阶段的每阶段目录保存 `initial.pth`、`best.pth`、`last.pth`、`migration.json`、`train.jsonl`、修改前后验证结果和 `summary.json`。`last.pth` 包括优化器、调度器及 RNG 状态。第一部分总目录保存阶段汇总、数据划分、对应的最终权重和校验值。

第九阶段目录保存 `initial.pth`、`distilled_best.pth`、`distilled_last.pth`、`train.jsonl`、验证结果及 `training_summary.json`。其中 `distilled_last.pth` 提供完整续训状态；训练完成后复制的 `final.pth` 与该阶段的 `distilled_best.pth` 内容一致。

最终部署只加载 `final.pth`：

```python
import torch
from models.RSTT_tiny import RSTTTiny

checkpoint = torch.load(
    'experiments/progressive_rstt_s_gt80k_b16/final.pth',
    map_location='cpu', weights_only=True,
)
model = RSTTTiny(**checkpoint['model_config'])
model.load_state_dict(checkpoint['state_dict'], strict=True)
model = model.cuda().eval()

# pair: [B, 2, 3, H, W]，RGB 浮点值范围 [0, 1]
with torch.inference_mode():
    output = model(pair.cuda())  # [B, 3, 3, 4H, 4W]
```

模型实现：`models/RSTT_progressive.py`；前八阶段训练：`train_progressive.py`；第九阶段训练：`train_tiny_mixed.py`；评测：`evaluate_progressive.py`。独立部署沿用 `models/RSTT_tiny.py`、`benchmark_tiny_4k.py` 和 `infer_tiny_video.py`。

## 独立部署与视频处理

在 GPU 空闲时，以独立进程测量两帧 960×540 → 三帧 3840×2160 的峰值显存与速度：

```bash
conda run --no-capture-output -n SRCHIP python -B benchmark_tiny_4k.py \
  --checkpoint experiments/progressive_rstt_s_gt80k_b16/final.pth \
  --output experiments/tiny_4k_new/benchmark_fp32.json \
  --precision fp32

conda run --no-capture-output -n SRCHIP python -B benchmark_tiny_4k.py \
  --checkpoint experiments/progressive_rstt_s_gt80k_b16/final.pth \
  --output experiments/tiny_4k_new/benchmark_amp_fp16.json \
  --precision amp-fp16
```

逐帧解码视频、按相邻两帧调用模型、及时将结果传回 CPU 并编码：

```bash
conda run --no-capture-output -n SRCHIP python -B infer_tiny_video.py \
  --input experiments/tiny_mixed_200k_b16/demo_input_960x540_60fps.mp4 \
  --checkpoint experiments/progressive_rstt_s_gt80k_b16/final.pth \
  --output experiments/tiny_4k_new/output_4k120_amp_fp16.mp4 \
  --precision amp-fp16
```

输入帧率自动读取，输出空间尺寸扩大四倍、播放帧率翻倍。视频脚本最后保持一帧以保留时长，并重新解码核对输出帧数、分辨率和帧率；测量结果写入同名 JSON。重复实验时选择新的输出文件名。

本轮 RTX 5090 上实测：FP32 / AMP 的最大进程显存采样为 **5.85 / 4.48 GiB**，纯模型去重处理速度为 **8.08 / 9.69 fps**。60 帧输入生成的 4K120 一秒视频，完整转换耗时分别为 **28.26 / 25.71 秒**。播放帧率和处理速度分别记录在[部署汇总](experiments/progressive_rstt_s_gt80k_b16/deployment_summary.json)。

已生成的演示视频：[FP32](experiments/progressive_rstt_s_gt80k_b16/demo_4k120_fp32.mp4)、[AMP-FP16](experiments/progressive_rstt_s_gt80k_b16/demo_4k120_amp_fp16.mp4)。
