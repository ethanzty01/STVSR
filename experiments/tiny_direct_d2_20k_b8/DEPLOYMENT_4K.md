# Tiny 单独部署：4K、时间轴 120 fps

本次加载本目录 `distilled_best.pth`（20,000 步），模型为 C=32、depth=2、direct head，参数量 **286,140**。输入 `[1,2,3,540,960]`，输出 `[1,3,3,2160,3840]`。GPU 为 RTX 5090，PyTorch 2.7.1+cu128；每种精度使用独立 Python 进程，进程仅加载 tiny。新增训练完成后可用同一命令复测最终权重。

## 实测显存

单位为 MiB，1 GiB = 1024 MiB。`allocated` 是 PyTorch 活跃张量及其分配的工作空间；`reserved` 是 PyTorch 缓存分配器保留的 GPU 内存；PID 显存由 `nvidia-smi` 每 100 ms 采样，覆盖 CUDA 上下文及库内存。

| 推理方式 | 精度 | allocated 峰值 | reserved 峰值 | PID 采样峰值 |
|---|---|---:|---:|---:|
| 冷启动首个前向 | FP32 | 3625.73 | 4954 | 5568 |
| 单对输入，输出立即释放 | FP32 | 3624.61 | 5002 | 5616 |
| 流式输入，结果立即转 CPU RGB8 | FP32 | **3637.47** | **5216** | **5830** |
| 冷启动首个前向 | AMP FP16 | 2932.94 | 3560 | 4174 |
| 单对输入，输出立即释放 | AMP FP16 | 2933.57 | 3970 | 4586 |
| 流式输入，结果立即转 CPU RGB8 | AMP FP16 | **2944.67** | **3784** | **4400** |

部署容量取冷启动、稳定前向和流式三个测量阶段中的最高值：

| 精度 | allocated 最大值 | reserved 最大值 | PID 最高采样值 |
|---|---:|---:|---:|
| FP32 | **3637.47 MiB（3.55 GiB）** | **5216 MiB（5.09 GiB）** | **5830 MiB（5.69 GiB）** |
| AMP FP16 | **2944.67 MiB（2.88 GiB）** | **3970 MiB（3.88 GiB）** | **4586 MiB（4.48 GiB）** |

开启了 `cudnn.benchmark=True`，首次遇到输入形状时会搜索卷积算法，因此冷启动也纳入统计。运行分配顺序会影响 reserved 和进程值，部署时应为缓存和上下文预留空间。PyTorch 的 allocated/reserved 峰值由分配器直接累计；PID 数值是每 100 ms 采样得到的最高观测值。

这些数字来自真实前向执行：`eval()`、`inference_mode()`、batch=1；每种精度冷启动一次、预热两次、计时十次，再测试 6 帧输入→11 帧输出的流式过程。流式测试使用固定的合成 CPU 帧源，每次生成新的 GPU 输入副本，按顺序执行输入传输、推理、输出转 CPU 和 uint8 量化。这验证了指定形状与数据类型下的分配和缓存行为；真实视频的解码与编码器 surface pool 需按照选用的编解码配置单独计入。

AMP FP16 使用 `torch.autocast`。权重、原始输入与最终输出保留 FP32，卷积和矩阵乘法使用 FP16；现有实现的相对位置偏置与 mask 会使 attention 概率保留 FP32。JSON 的 stage trace 记录了实际 dtype。

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

本次消费路径先复制一张 FP32 输出到 CPU（**94.922 MiB**），然后原地 clamp/缩放/取整，并分配单张 RGB8 图像（**23.730 MiB**）；转换时两份 CPU 存储短时共存，消费后立即释放。8-bit NV12 单个 4K surface 为 **11.865 MiB**，若编码器保留 K 个 surface，对应基础图像存储为 `K × 11.865 MiB`，另有编码器自己的状态和工作空间。GPU 上训练所用的教师、梯度和优化器状态均由单独的训练进程负责，部署脚本只构建 tiny。

## 处理速度与 120 fps 时间轴

相邻两帧生成端点、中间帧和后端点。流式相邻调用共享端点，因此每处理一对输入，持续新增 **2 张**输出帧。

| 精度 | GPU 每对输入耗时 | 仅前向持续产帧速度 | 含传输与 CPU RGB8 量化 |
|---|---:|---:|---:|
| FP32 | 245.38 ms | 8.15 帧/秒 | 5.27 帧/秒 |
| AMP FP16 | 205.93 ms | 9.71 帧/秒 | 5.86 帧/秒 |

60 fps 输入对应 120 fps 输出时间轴；实时运行对应每对输入 **16.67 ms** 的预算。当前实现适合按以上速度离线生成 120 fps 视频，实时吞吐仍有优化空间。N 帧输入实际输出 `2N−1` 帧，文件末端可根据封装要求设置最后一帧时长。

## 复测指令

```bash
conda run --no-capture-output -n SRCHIP python -B benchmark_tiny_4k.py \
  --checkpoint experiments/tiny_direct_d2_20k_b8/distilled_best.pth \
  --output experiments/tiny_direct_d2_20k_b8/deployment_4k_fp32.json \
  --precision fp32

conda run --no-capture-output -n SRCHIP python -B benchmark_tiny_4k.py \
  --checkpoint experiments/tiny_direct_d2_20k_b8/distilled_best.pth \
  --output experiments/tiny_direct_d2_20k_b8/deployment_4k_amp_fp16.json \
  --precision amp-fp16
```

原始结果：`deployment_4k_fp32.json`、`deployment_4k_amp_fp16.json`。脚本记录权重 SHA256、环境、冷启动/稳定阶段统计、真实特征形状及 attention 矩阵 dtype，适合用最终权重重复运行。
