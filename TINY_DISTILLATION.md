# 双帧 RSTT 小模型与蒸馏

`models/RSTT_tiny.py` 提供 `RSTTTiny`：输入 `[B,2,3,H,W]`，输出
`[B,3,3,4H,4W]`。默认 32 通道、四级编码/解码、每阶段两个单 block、
原始注意力头数 `[2,4,8,16,16,8,4,2]`。当前默认重建头为
`Conv3x3(32,48) → PixelShuffle(4)`：`depth=1` 共 **181,270** 个参数，
`depth=2` 共 **286,140** 个参数，后者保留普通窗口和移位窗口组合。
已完成的 `tiny_distill_10k` 实验使用 `depth=1, head='legacy'`，参数量
317,033，末端64通道；评估脚本会根据保存的配置加载相应结构。

时间插值使用 `linear, align_corners=True`，空间使用
`bilinear, align_corners=False`。高分辨率残差与插值分帧生成；
`forward_sequence()` 接收逐帧迭代器并逐帧返回结果，N 帧输入生成 2N−1 帧。
共享端点采用后一对输入的预测（最后一帧使用最后一对的末端预测）。
训练反向传播会保存所需激活；分帧重建的显存优势主要体现在推理。

## 训练

入口 `train_tiny.py direct` 支持 `--mode supervised`（单独纯监督）、
`--mode distill`（单独蒸馏）、`--mode paired`（默认，同时训练两组对照）。
`--init PATH` 加载学生权重后开始新的训练阶段，重新建立 AdamW 和余弦学习率；
`--steps` 表示这一阶段新增的步数。继续训练时建议从 `--lr 5e-5` 开始。
脚本先验证加载的初始权重，将其纳入最佳权重候选，记录在 `initial_validation.json`。
纯监督模式的训练与评估均仅加载学生。所有模式沿用下文的数据增强。

```bash
cd /home/tyzheng/RSTT
conda activate SRCHIP

# 从随机初始化开始纯监督训练
python -B train_tiny.py direct --mode supervised --output experiments/direct_d2_supervised_100k --steps 100000 --batch-size 8 --test-batch-size 4 --device cuda

# 加载本轮纯监督权重，继续训练50000步
python -B train_tiny.py direct --mode supervised --init experiments/tiny_direct_d2_20k_b8/supervised_best.pth --output experiments/direct_d2_finetune_50k --steps 50000 --batch-size 8 --lr 5e-5 --test-batch-size 4 --device cuda

# 从随机初始化开始蒸馏训练
python -B train_tiny.py direct --mode distill --output experiments/direct_d2_distill_100k --steps 100000 --batch-size 8 --kd-weight 0.5 --test-batch-size 4 --device cuda
```

默认模型为 `--depth 2 --head direct`，教师路径为已保存的 RSTT-S 权重。
蒸馏模式同样支持 `--init`，可加载现有 `distilled_best.pth` 或同结构纯监督权重。
每个新阶段使用新的输出目录；阶段完成后自动评估完整默认测试集。

在 `RSTT/` 目录执行，使用已有 `SRCHIP` 环境：

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny.py direct \
  --data-root ../Datasets_pt \
  --teacher checkpoints/RSTT-S/f96d4w4h24816Ushape0res_1690000.pth \
  --output experiments/tiny_direct_d2_20k_b8 --depth 2 --head direct \
  --steps 20000 --batch-size 8 --val-every 1000 \
  --validation-limit 128 --test-batch-size 4 --device cuda
```

每次新训练使用一个新的输出目录。数据原文件按需读取，HR 张量转换为
float32 [0,1]，从四帧生成 bicubic 四倍降采样输入（`antialias=True`）。
先全图下采样再对齐裁剪为 HR128/LR32，并共用翻转、转置及时间反转。

教师为完整 RSTT-S，严格加载已有权重并冻结。教师读取七帧片段中的
第 1、3、5、7 帧 LR；学生随机选教师输入中相邻两帧，监督为对应的三帧 HR。
教师输出按相同时间索引选取三帧。

蒸馏损失：

`L = Charbonnier(student, GT) + 0.5 * Charbonnier(student, teacher)`

Charbonnier epsilon 为 0.001。配对运行的纯监督学生使用完全相同的初始化、
输入样本和训练步数，仅使用 GT 损失。AdamW，初始学习率 2e-4，
betas=(0.9,0.99)，weight_decay=1e-4，余弦衰减至 1e-6，梯度裁剪 1.0，
FP32，学生 drop_path_rate=0。

Vimeo 训练集按种子 2026 留出 128 个片段用于验证，剩余 64,484 个用于训练。
每个模型独立按验证 Y-PSNR 选最佳权重。测试集始终仅用于最终评分。
当前两 block 实验为 20,000 步、batch 8，每个学生 160,000 次样本读取。
此前 `tiny_distill_10k` 为 10,000 步、batch 4，每个学生 40,000 次样本读取。

## 测试与结果

默认测试 7,815 个 Vimeo 片段，包含 slow 1,613、medium 4,977、fast 1,225。
每个片段固定评估中间三帧（第 3、4、5 帧），共 23,445 帧。
教师输入四帧，学生输入两帧；目标帧、LR 降质、裁边和评分规则相同。
这项比较同时体现模型容量和时间上下文的变化。

完整帧尺寸 HR256×448 / LR64×112。预测与 GT 转换为舍入后的 uint8；
RGB/Y PSNR 的 data_range=255，Y 使用 BT.601 系数；Y SSIM 使用
11×11 高斯窗口、sigma=1.5、valid 区域。外层裁边为 0。
总体指标为全部帧的平均值；另按速度组、原时刻帧和插入帧分别汇总。

产物包含：

- `distilled_best.pth` / `supervised_best.pth`：验证选出的最佳权重。
- `*_last.pth`：最终权重与优化器状态；`initial.pth`：共同初始化。
- `config.json` / `split_ids.json` / `test_ids.json`：配置、教师 SHA256、样本划分。
- `train.jsonl` / `training_summary.json`：逐步损失、验证分数和耗时。
- `test_per_frame.csv` / `results.json`：逐帧测试与汇总、参数量和速度/显存测量。

可重新执行评估：

```bash
conda run --no-capture-output -n SRCHIP python -B train_tiny.py direct \
  --output experiments/tiny_direct_d2_20k_b8 --evaluate-only --test-batch-size 4 --device cuda
```

推理基准为 batch 1、LR64×112、FP32，先预热三次再计时二十次。
教师每次输出七帧，学生每次输出三帧，同时记录每调用和每输出帧耗时。
GPU 峰值字段为本次前向在已分配模型/张量之上的额外 allocated 峰值。

## 检查

```bash
conda run --no-capture-output -n SRCHIP python -B -m pytest -q tests
```

覆盖时间对齐、训练/验证划分、指标参考实现、双样本前向反向、插值、
奇数尺寸和流式输出。原教师结构仅将 `timm` 的两个工具替换为
PyTorch 实现，权重名称与张量尺寸保持兼容。
