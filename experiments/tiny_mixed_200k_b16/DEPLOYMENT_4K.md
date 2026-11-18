# 最终 tiny 部署实测：4K、120 fps 视频

已完成 **200,000 个新增训练步**后的独立部署验证。使用验证集选出的 `distilled_best.pth`：本轮第 **198,000 步**，累计第 **218,000 步**；训练本身运行到新增第 200,000 步。模型 C=32、depth=2、direct PixelShuffle head，**286,140 个参数**。权重 SHA256 为 `882203dc99011eea7773d53abceb9c6e9707fc005430031f6a08323f01d9dbc8`。定量画质测试见 [RESULTS.md](RESULTS.md)。

环境为 RTX 5090、PyTorch 2.7.1+cu128、OpenCV 4.13.0、SRCHIP。四个实验各自启动新的 Python 进程，GPU 空闲时顺序运行；每个部署进程只构建 tiny。输入 `[1,2,3,540,960]`，输出 `[1,3,3,2160,3840]`。

## 峰值显存

真实 MP4 流水线得到的活跃显存峰值：**FP32 3.55 GiB，AMP FP16 2.88 GiB**。把短时基准和真实视频的全部测量合并，进程最高观测占用分别为 **5.85 GiB** 和 **4.48 GiB**，可据此理解实际部署占用。

以下单位均为 MiB；1 GiB = 1024 MiB。`allocated` 为 PyTorch 活跃张量及分配器管理的工作空间；`reserved` 为其缓存分配器保留内存；PID 为 `nvidia-smi` 对本进程的最高采样值，包含 CUDA 上下文及库内存。

| 实验 | allocated 峰值 | reserved 峰值 | PID 最高采样值 |
|---|---:|---:|---:|
| 合成输入基准，FP32 | 3637.47 | 5216 | 5830 |
| 真实视频完整流水线，FP32 | 3637.77 | 5374 | 5988 |
| 合成输入基准，AMP FP16 | 2944.67 | 3970 | 4586 |
| 真实视频完整流水线，AMP FP16 | 2946.12 | 3876 | 4492 |

同种精度各实验的总体最大值：

| 精度 | allocated 最大值 | reserved 最大值 | PID 最高采样值 |
|---|---:|---:|---:|
| FP32 | 3637.77 | 5374 | 5988 |
| AMP FP16 | 2946.12 | 3970 | 4586 |

这里报告的是包含模型和输入的绝对峰值；扣除测量前驻留分配的增量另写在 JSON 中。分配器直接累计 allocated/reserved 峰值；PID 约每 0.1 秒采样一次，因此标记为最高观测值。不同分配顺序、缓存复用和上下文会影响 reserved 与 PID 数字。

## 测量覆盖范围

`benchmark_tiny_4k.py` 使用 `eval()`、`inference_mode()`、batch=1，开启 `cudnn.benchmark=True`。它分别记录模型/输入初始化、冷启动首个前向、输出有限值检查与两次预热、十次稳定前向、6 帧输入→11 帧输出的即时 CPU RGB8 消费。新增 `initialization_and_warmup` 覆盖阶段之间的分配；`overall_measured_peak` 汇总所有阶段最高值。冷启动的 cuDNN 算法搜索也纳入统计。两种精度的输出有限值检查均通过。

`infer_tiny_video.py` 实际打开 960×540、60 fps 的 MP4，顺序完成 CPU 解码、逐对 GPU 推理、即时 CPU RGB8 转换、BGR 转换、OpenCV FFmpeg `mp4v` CPU 编码。峰值从第一次前向前重置，持续记录至编码器 flush 完成，覆盖真实视频的冷启动与所有帧。视频处理计时覆盖这一流式处理阶段，启动时的权重载入、打开文件及先读首帧在计时准备阶段；输出文件的全量重新解码验证在处理计时之后完成。

AMP FP16 使用 `torch.autocast`。参数、原始 RGB 输入、最终 RGB 输出保留 FP32；卷积与矩阵乘法使用 FP16。现有相对位置偏置及 shifted-window mask 使 attention 概率保留 FP32，stage trace 记录了实际 dtype。

## 已生成并验证的视频

输入来自授权的 `Datasets_pt/test/REDS/000` 第 0–59 帧，每帧由 1280×720 缩小到 960×540，再封装为 60 fps、1 秒视频。来源记录见 [demo_input_960x540_60fps.source.json](demo_input_960x540_60fps.source.json)。

- [输入：960×540、60 fps、60 帧](demo_input_960x540_60fps.mp4)
- [FP32 输出：3840×2160、120 fps、120 帧](demo_output_4k_120fps_fp32.mp4)
- [AMP FP16 输出：3840×2160、120 fps、120 帧](demo_output_4k_120fps_amp_fp16.mp4)

两个输出文件均重新打开，并完整解码确认：**3840×2160，120 fps，120 帧，1.000 秒**。模型逐对生成 119 帧，片尾将最后一张 CPU 输出帧保持一次，得到 120 个编码帧，保持输入的一秒时长。这一片尾保持步骤无需新增 GPU 推理。

## 处理速度

| 精度 | 稳定 GPU 前向，ms/对 | 持续新增 HR 帧/秒，仅前向 | 完整视频处理耗时 | 编码帧/秒，含 CPU 编解码 |
|---|---:|---:|---:|---:|
| FP32 | 248.06 | 8.06 | 28.15 秒 | 4.26 |
| AMP FP16 | 206.96 | 9.66 | 25.74 秒 | 4.66 |

相邻推理共享端点，每对输入在持续视频中新增 2 帧，因此前向持续产帧速度按 `2 / 每对耗时` 计算。60 fps 输入对应 120 fps 输出时间轴；实时处理预算为每对 **16.67 ms**。当前执行以上表速度离线生成 120 fps 视频文件。

## 要存储的内容

以下是各项逻辑大小；张量释放和工作空间复用决定最终峰值，表中各项的生命周期不同。

| 内容 | 尺寸 | FP32 MiB |
|---|---|---:|
| 模型参数 | 286,140 个数 | 1.092 |
| 相对位置索引等注册 buffer | — | 0.297 |
| 单对前向的输入 tensor | `[1,2,3,540,960]` | 11.865 |
| 流式路径同时保留的 previous/current 原帧 | 两个 `[1,3,540,960]` | 额外 11.865 |
| Encoder 0，保留给 cross-attention | `[1,2,32,544,960]` | 127.500 |
| Encoder 1 | `[1,2,32,272,480]` | 31.875 |
| Encoder 2 | `[1,2,32,136,240]` | 7.969 |
| Encoder 3 | `[1,2,32,68,120]` | 1.992 |
| Decoder 0 | `[1,3,32,68,120]` | 2.988 |
| Decoder 1 | `[1,3,32,136,240]` | 11.953 |
| Decoder 2 | `[1,3,32,272,480]` | 47.813 |
| Decoder 3 | `[1,3,32,544,960]` | 191.250 |
| 单帧重建头输出 | `[1,48,540,960]` | 94.922 |
| 时间插值后的低分辨率 baseline | `[1,3,3,540,960]` | 17.798 |
| 单张 4K RGB 浮点帧 | `[1,3,2160,3840]` | 94.922 |
| 返回的三张 4K 帧 | `[1,3,3,2160,3840]` | 284.766 |

显存大头来自高分辨率 attention 及其临时张量。例如最后一级 decoder 有 32,640 个空间窗口，每个窗口 48 个时空 token、2 个 head，self-attention 矩阵 `[32640,2,48,48]` 单份达到 **573.75 MiB**；对应 shifted-window self mask 为 **286.875 MiB**，cross mask 为 **191.25 MiB**。此外还有 Q/K/V、softmax 前后张量、MLP 中间特征、残差、卷积工作空间等。最高峰出现在最后一级 decoder。

四级 encoder 特征会一起保留，合计 **169.336 MiB**；decoder 逐级处理。重建头逐帧执行，随后 `torch.stack` 会为三张输出分配连续存储，短时同时保留原输出列表与 stack 结果。

流式路径在前向期间同时保留 `previous`、`current` 两张原帧（合计 **11.865 MiB**）和 `torch.stack((previous,current), dim=1)` 生成的连续输入副本（另 **11.865 MiB**），因此这一阶段的输入相关存储合计 **23.730 MiB**。直接调用单对前向时只预先分配一个输入 tensor，对应 **11.865 MiB**。上述流式实测已包含这份 stack 临时副本。

已修正前一对输出的 view 跨下一次前向存活的问题。消费者在取下一帧前释放已消费的 GPU view；该行为有弱引用生命周期测试覆盖。视频长度增加时，按此消费方式保持有限缓存。

本次消费路径先复制一张 FP32 输出到 CPU（**94.922 MiB**），然后原地 clamp/缩放/取整，并分配单张 RGB8 图像（**23.730 MiB**）；转换时两份 CPU 存储短时共存，消费后立即释放。8-bit NV12 单个 4K surface 为 **11.865 MiB**，若编码器保留 K 个 surface，对应基础图像存储为 `K × 11.865 MiB`，另有编码器自己的状态和工作空间。部署脚本仅加载 tiny 权重；其 GPU 常驻内容为模型参数、注册 buffer 和正在处理的输入。


真实视频消费路径还将当前 CPU RGB8 转成 BGR 交给编码器，并保留最近一张 CPU BGR 帧用于片尾保持。BGR8 单张同样为 **23.730 MiB**。本次 `mp4v` 编解码器在 CPU 上运行，其内部参考帧与编码工作空间位于主存；若改用 GPU 编解码，应按所选 surface 数和编码器状态追加显存。

## 运行指令与原始记录

在 `/home/tyzheng/RSTT` 中运行，复测视频时为 `--output` 选择新文件名：

```bash
conda run --no-capture-output -n SRCHIP python -B benchmark_tiny_4k.py \
  --checkpoint experiments/tiny_mixed_200k_b16/distilled_best.pth \
  --output experiments/tiny_mixed_200k_b16/deployment_4k_fp32.json \
  --precision fp32

conda run --no-capture-output -n SRCHIP python -B infer_tiny_video.py \
  --input experiments/tiny_mixed_200k_b16/demo_input_960x540_60fps.mp4 \
  --checkpoint experiments/tiny_mixed_200k_b16/distilled_best.pth \
  --output experiments/tiny_mixed_200k_b16/rerun_output_4k_120fps_fp32.mp4 \
  --precision fp32
```

将 `--precision` 改为 `amp-fp16` 并换一个输出文件名即可复测 AMP。原始统计包括：

- [FP32 基准](deployment_4k_fp32.json)、[AMP 基准](deployment_4k_amp_fp16.json)：全部阶段峰值、真实特征与 attention 形状、dtype。
- [FP32 视频流水线](demo_output_4k_120fps_fp32.json)、[AMP 视频流水线](demo_output_4k_120fps_amp_fp16.json)：完整处理峰值、编码结果、重新解码验证的尺寸、帧率和帧数。

四份记录的模型 SHA256 与参数量一致，测试进程均已正常退出。
