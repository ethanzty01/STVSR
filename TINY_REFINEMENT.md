# Tiny 裁块与中间帧监督实验

本轮从 `experiments/progressive_rstt_s_gt80k_b16/final.pth` 开始，模型保持两帧输入、三帧输出、空间 ×4、32 通道、每级深度 2，共 286,140 参数。

| 组别 | HR 训练裁块 | HR 验证裁块 | GT 三帧权重 | 更新步数 |
|---|---:|---:|---|---:|
| A | 128 | 128 | 1:1:1 | 20,000 |
| B | 256 | 128 | 1:1:1 | 20,000 |
| C | 256 | 128 | 1:2:1 | 20,000 |
| D | 沿用筛选最佳组 | 128 | 沿用筛选最佳组 | 60,000 |

A/B/C 从同一份权重开始，batch 16、BF16、Vimeo/REDS 各占一半，纯 GT Charbonnier。筛选阶段学习率为 2e-5→1e-6，D 阶段为 1e-5→1e-6，使用余弦衰减。中间帧加权的损失为 `(L0 + w * Lmid + L1) / (2 + w)`。

验证固定为原有 Vimeo 128 + REDS 128 片段及 HR128 中心裁块，FP32、RGB8 转 Y 后计分。每 2,000 步验证，按两个数据集等权均值选择权重；初始权重也参与最佳值选择。验证同时记录端点与中间帧的 PSNR/SSIM。

数据划分种子固定为 2026。A/B/C 的训练抽样与增强种子为 2027；D 使用 2028，数据集成员和验证样本保持一致。

## 完整运行

以下命令执行三组筛选、自动选择、60k 精调和最终完整测试。重新实验时使用新的输出目录。

```bash
cd /home/tyzheng/RSTT
conda run --no-capture-output -n SRCHIP python -B run_tiny_refinement.py \
  --output experiments/tiny_patch_center_new \
  --init experiments/progressive_rstt_s_gt80k_b16/final.pth
```

恢复已有进度时指定原输出目录：

```bash
conda run --no-capture-output -n SRCHIP python -B run_tiny_refinement.py \
  --output experiments/tiny_patch_center_20261117 --resume
```

各组的 `distilled_last.pth` 包含优化器、调度器与随机状态。已完成的组会先通过审计，后续继续执行未完成阶段。实验计划保存于 `experiment_plan.json`，进度保存于 `experiment_status.json`。

## 单独运行一组

例如运行 C 组：

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny.py mixed \
  --init experiments/progressive_rstt_s_gt80k_b16/final.pth \
  --output experiments/tiny_patch256_center2_new \
  --steps 20000 --batch-size 16 \
  --patch-size 256 --validation-patch-size 128 --center-weight 2 \
  --seed 2026 --sampling-seed 2027 \
  --mode supervised --lr 2e-5 \
  --compile --compile-mode reduce-overhead --skip-test
```

新增参数：

- `--validation-patch-size`：独立指定验证裁块；省略时沿用训练裁块。
- `--center-weight`：GT 中间帧损失权重，默认 1，按三帧总权重归一化。
- `--sampling-seed`：独立控制训练抽样与增强；`--seed` 继续决定固定数据划分。

恢复单组训练使用原输出目录和 `--resume`。脚本从已保存配置恢复裁块、损失权重及抽样种子。

## 最终评测与记录

完整流水线在精调结束后运行下列对比：

```bash
conda run --no-capture-output -n SRCHIP python -B evaluate_progressive.py \
  --run-dir experiments/tiny_patch_center_20261117 \
  --previous experiments/progressive_rstt_s_gt80k_b16/final.pth \
  --intermediate-checkpoint experiments/tiny_mixed_200k_b16/distilled_best.pth \
  --workers 2 --fps
```

评测 JSON 中的 `progressive` 表示本轮最终权重，`previous_tiny` 表示本轮起点权重，`intermediate` 表示早期蒸馏 Tiny，`teacher` 表示 RSTT-S。数据协议沿用前一轮完整测试。

关键文件：`selection.json` 记录筛选结果；`training_audit.json` 核对更新次数与权重衔接；`final.pth` 为最终部署入口；`final_manifest.json` 记录选中步数、配置和 SHA-256；`evaluation/` 保存完整逐帧结果。真实训练步骤计数与最终权重选中步数分别记录。

本轮 2026-11-17 实验已经完成全部 120,000 步及四模型完整测试。胜出配置为 A：HR128、三帧等权；最终选择精调第 52,000 步，固定验证均值为 29.361608 dB。完整结果见 [实验报告](experiments/tiny_patch_center_20261117/REPORT.md)，交付权重为 [final.pth](experiments/tiny_patch_center_20261117/final.pth)。
