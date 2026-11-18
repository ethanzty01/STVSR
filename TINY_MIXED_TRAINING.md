# Vimeo + REDS 继续训练

`train_tiny.py mixed` 从现有 tiny 权重开始一段新的训练计划。默认模型保持
32 通道、每阶段两个窗口块、`32 → 48 → PixelShuffle(4)` 重建头，参数量 286,140。
输入两帧 LR，输出三帧 HR；空间放大四倍，中间补一帧。

## 继续训练 200,000 步

在项目目录执行：

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny.py mixed \
  --init experiments/tiny_direct_d2_20k_b8/distilled_best.pth \
  --output experiments/tiny_mixed_200k_b16 \
  --steps 200000 --batch-size 16 --precision bf16 --workers 8 \
  --compile --compile-mode reduce-overhead
```

`--steps 200000` 表示新增 200,000 次更新；起点为原来第 20,000 步时，最终累计
220,000 步。每批包含 8 个 Vimeo 样本和 8 个 REDS 样本，两个数据源均有放回采样。
HR 裁剪尺寸为 128×128，对应 LR 为 32×32。REDS 每个样本取同一序列内连续七帧，
老师读取其中四帧，学生读取相邻两个老师输入帧；监督目标为对应的三帧 HR。
使用共同的水平/垂直翻转、转置和时间反转增强。

默认继续蒸馏：

- 初始化：当前 `distilled_best.pth`；老师：已保存的 RSTT-S 权重。
- 损失：GT Charbonnier + 0.5 × 老师输出 Charbonnier。
- AdamW 初始学习率 `5e-5`，余弦下降至 `1e-6`。
- BF16 自动混合精度执行老师和学生前向；权重、优化器状态及损失归约保留 FP32。
- 每 2,000 步验证；Vimeo 和 REDS 各使用 128 个固定验证样本。
- 用两个数据集的平均 PSNR-Y 的等权均值选择 `distilled_best.pth`。
- 验证及最终测试使用 FP32、取整到 uint8 的 Y-PSNR/Y-SSIM；验证采用固定中心裁剪，最终测试采用整帧。

将 `--batch-size` 改为 `8` 时，每批各含四个样本。使用 `--mode supervised` 可只用
GT 监督训练。可通过 `--lr`、`--kd-weight`、`--val-every` 调整训练计划。

## 恢复中断的运行

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny.py mixed \
  --output experiments/tiny_mixed_200k_b16 --resume --workers 8
```

恢复会读取运行目录中的 `config.json` 与 `distilled_last.pth`，继续原定的 200,000 步
计划。断点包含权重、优化器、学习率调度器、梯度缩放器和随机状态。数据样本及增强
由绝对训练步数确定，预取和 worker 数量变化仍保持相同的后续样本序列。
保存采用临时文件完成后替换目标文件的方式。

`--init` 用于从权重开始新一轮计划，`--resume` 用于接续同一次运行。

## 测试与产物

训练结束后自动调用 `evaluate_tiny_mixed.py`，分别测试完整 Vimeo 和 REDS 测试集，
比较起点 tiny、最佳 tiny、最后一步 tiny、RSTT-S 老师及双线性基线。
使用 `--skip-test` 可让训练脚本在保存训练结果后结束，随后手动执行：

```bash
conda run --no-capture-output -n SRCHIP python -B evaluate_tiny_mixed.py \
  --run-dir experiments/tiny_mixed_200k_b16 \
  --data-root ../Datasets_pt \
  --teacher checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth \
  --vimeo-batch-size 4 --reds-batch-size 1
```

主要训练文件：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 模型、数据比例、超参数、权重哈希及精度设置 |
| `split_ids.json` | 训练与验证样本 ID |
| `initial.pth` | 本轮起点权重 |
| `distilled_best.pth` | 验证均值最高的权重及其步数 |
| `distilled_last.pth` | 最近一次验证后完整可恢复断点 |
| `train.jsonl` | 每一步的损失、学习率、累计步数及验证记录 |
| `training_summary.json` | 完成步数、样本数、时间和最佳验证分数 |

## 选择运行配置

以下命令实际加载混合数据并执行训练，统计预热后的数据加载与训练总吞吐：

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny.py mixed \
  --output experiments/profile_mixed_b16_bf16 \
  --batch-size 16 --precision bf16 --workers 4 \
  --profile-warmup 10 --profile-steps 30
```

结果保存在 `profile.json`。增加 `--compile` 可测量 `torch.compile`，
编译成本发生在预热阶段；采用正式运行前应检查预热后速度和数值运行情况。
每次新的训练或测速使用新的输出目录。

本轮 RTX 5090 实测包含数据读取的 batch 16 吞吐：BF16 eager 约 67.10 ms/步，
默认编译约 38.24 ms/步，`reduce-overhead` 约 25.23 ms/步。因此正式运行使用
8 个数据 worker 和 `reduce-overhead`。测速结果分别保存在
`experiments/profile_mixed_b16_bf16_w4/profile.json`、
`experiments/profile_mixed_b16_bf16_compile_w8/profile.json` 与
`experiments/profile_mixed_b16_bf16_graph_w8/profile.json`。
