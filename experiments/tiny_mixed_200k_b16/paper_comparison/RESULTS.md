# Vid4 与 Vimeo 七帧测试

Seven unique frames per Vimeo clip; Vid4 each frame scored once and sequences equally weighted; Y-channel on rounded RGB8, no crop; PyTorch antialiased bicubic x4 then rounded LR8; first occurrence of duplicate endpoint; FP32. Official code uses saved bicubic PNG; generating LR with PyTorch can differ from official Matlab resize.

| 模型 | Vid4 PSNR / SSIM | Fast PSNR / SSIM | Medium PSNR / SSIM | Slow PSNR / SSIM | 参数量(M) |
|---|---:|---:|---:|---:|---:|
| tiny 验证最优 | 24.5324 / 0.6989 | 33.0977 / 0.8906 | 32.3005 / 0.8926 | 30.8042 / 0.8708 | 0.286140 |
| tiny 最终 | 24.5312 / 0.6986 | 33.0973 / 0.8906 | 32.2991 / 0.8926 | 30.8033 / 0.8708 | 0.286140 |
| RSTT-S 本机 | 26.2791 / 0.7942 | 36.5250 / 0.9376 | 35.3211 / 0.9348 | 33.2685 / 0.9126 | 4.485275 |

RTX 5090 FP32：输入 [1,2,3,144,180]，预热10次，计时100次，每对 14.240 ms，连续流每对新增2帧为 140.45 fps；3帧/次统计为 210.68 fps。

Vid4 四个视频等权平均，每个视频中的帧各记一次：calendar 41、city 34、foliage 49、walk 47。Vimeo fast 1225、medium 4977、slow 1613 段，每段7帧。输出采用 first occurrence 重复端点。教师为4帧输入，tiny为2帧输入。

官方低清图像为预先保存的bicubic PNG；本次由本地HR使用PyTorch antialiased bicubic生成并四舍五入为8bit，故与论文低清数据处理存在差异。论文FPS硬件为Quadro RTX 6000，与本次RTX 5090和统计口径不同。

[完整结果](results.json) · [Vimeo逐帧](vimeo_frames.csv) · [Vid4逐帧](vid4_frames.csv) · [FPS](fps.json) · [脚本](../../../eval_paper_tiny.py)
