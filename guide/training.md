# CIFAR-10 ResNet-50 训练指南

本项目使用 CIFAR-sized ResNet-50 在 CIFAR-10 上从头训练。训练过程始终记录 loss、accuracy、
entropy、normalized entropy、confidence、ECE 和 class-wise 指标。`--entropy-experiment`
只控制训练结束时是否自动生成 entropy 图片，不控制指标记录。

所有新训练都会创建带时间戳的独立目录。建议每个实验使用不同的 `--output-dir`。

## 环境检查

```bash
python -m pip install -e 'CV_Corruption[plot]'
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

配置文件是 `configs/training/cifar10_resnet50.yaml`。其中 `gpu_ids` 表示使用的物理 GPU，
`batch_size` 表示每张 GPU 的 batch size。

## CPU Smoke Test

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --cpu --smoke --batch-size 2 \
  --train-batches 1 --test-batches 1 --workers 0 \
  --output-dir runs/cifar10_cpu_smoke
```

## 单卡训练

### 不自动生成 entropy 图片

指标仍会完整记录，但训练结束时只自动生成普通训练曲线：

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 \
  --no-entropy-experiment \
  --epochs 200 \
  --output-dir runs/cifar10_single_gpu_no_entropy
```

### 自动生成 entropy 图片

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 \
  --entropy-experiment \
  --epochs 200 \
  --output-dir runs/cifar10_single_gpu_entropy
```

## 四卡训练

四卡 DDP 必须使用 `torchrun`。`--nproc_per_node`、`--gpu-ids` 和 YAML 中的 GPU 数量必须一致。
例如配置文件中使用：

```yaml
gpu_ids: [0, 1, 2, 3]
```

### 不自动生成 entropy 图片

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m cv_corruption.cli.train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 1 2 3 \
  --no-entropy-experiment \
  --epochs 200 \
  --output-dir runs/cifar10_4gpu_no_entropy
```

### 自动生成 entropy 图片

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m cv_corruption.cli.train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 1 2 3 \
  --entropy-experiment \
  --epochs 200 \
  --output-dir runs/cifar10_4gpu_entropy
```

不要在 `torchrun` 命令中加入 `--seeds`。`--seeds` 是下面介绍的外层多 seed 编排模式。

## 多 seed 训练

一次运行多个 seed 时，外层使用普通 `cv-train`。程序会为每个 seed 创建独立子 run，训练完成后
在父 run 中生成 mean ± standard deviation across seeds 曲线。

### 单卡多 seed，无 entropy 图片

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 \
  --seeds 0 1 2 \
  --no-entropy-experiment \
  --epochs 200 \
  --output-dir runs/cifar10_multiseed_single_gpu
```

### 单卡多 seed，自动生成 entropy 图片

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 \
  --seeds 0 1 2 \
  --entropy-experiment \
  --epochs 200 \
  --output-dir runs/cifar10_multiseed_single_gpu_entropy
```

### 四卡多 seed，无 entropy 图片

外层仍然使用普通 `cv-train`；程序会自动为每个 seed 启动一个四卡 `torchrun` 子任务：

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 1 3 \
  --seeds 42 43 44 45 46 123 \
  --no-entropy-experiment \
  --epochs 200 \
  --output-dir runs/cifar10_multiseed_4gpu
```

### 四卡多 seed，自动生成 entropy 图片

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 1 3 \
  --seeds 42 43 44 45 46 123 \
  --entropy-experiment \
  --epochs 300 \
  --output-dir runs/cifar10_multiseed_4gpu_entropy_500_epochs
```

多 seed 父目录结构类似：

```text
cifar10_multiseed_4gpu_entropy_<timestamp>/
├── seed_0_<timestamp>/
├── seed_1_<timestamp>/
├── seed_2_<timestamp>/
├── metrics.jsonl                 # mean 和 *_std
├── metrics/epoch_metrics.jsonl   # entropy 等指标的 mean 和 *_std
├── outputs/                      # 父 run 的 mean ± std 图
└── seeds.json
```

子 run 可以独立分析；父 run 的实线是 seed 均值，阴影是均值上下一个标准差。

## 输出指标

每个 run 都会写入：

```text
metrics.jsonl
metrics/epoch_metrics.jsonl
metrics/batch_metrics.jsonl
config.json
resolved_config.json
checkpoints/last.pt
checkpoints/best.pt
```

即使关闭 entropy 图片，`epoch_metrics.jsonl` 和 `batch_metrics.jsonl` 仍然存在。

## 训练完成后重绘

修改绘图配置后，不需要重新训练。普通训练曲线：

```bash
cv-plot-training --run-dir runs/<run-directory>
```

训练曲线默认使用窗口为 5 的 moving average，并绘制每条曲线最多 10 个 marker。重绘时
可以临时覆盖这些选项：

```bash
cv-plot-training --run-dir runs/<run-directory> \
  --no-smooth --no-markers
cv-plot-training --run-dir runs/<run-directory> \
  --smooth-window 15 --markers 20
cv-plot-training --run-dir runs/<run-directory> --no-original
```

marker 数量支持 `5`、`10`、`15`、`20` 和 `MAX`；`MAX` 表示所有记录点。长期使用的设置
请修改 `configs/visualization/training_curves.yaml` 中的 `plot.smooth_enabled`、
`plot.smooth`、`plot.markers.enabled` 和 `plot.markers.count`。
原始曲线显示由 `plot.show_original` 控制，也可以使用 `--no-original` 临时关闭。

Entropy 曲线：

```bash
cv-plot-entropy \
  --run-dir runs/<run-directory> \
  --config configs/visualization/entropy_curves.yaml
```

Entropy 曲线默认同样使用窗口为 5 的 moving average、原始曲线参考线和最多 10 个 marker。
可用 `--smooth-window`、`--no-smooth`、`--no-original`、`--no-markers` 以及
`--markers 5|10|15|20|MAX` 覆盖默认设置；长期配置请修改
`configs/visualization/entropy_curves.yaml` 中的 `plot` 和 `figure` 字段。
每个 epoch metric 和 relationships 图都会同时输出 train/test 合并图、单独 train 图和单独
test 图；合并图使用原文件名，单独图使用 `_train` 和 `_test` 后缀。例如
`epoch_entropy.png`、`epoch_entropy_train.png`、`epoch_entropy_test.png`。

如果 run 中存在 `metrics/batch_metrics.jsonl`，还会生成 `outputs/*/entropy/batch/` 下的
batch 图。`Before` 表示一次 optimizer 更新前模型对该 batch 的指标，`After` 表示更新后
模型再次处理同一个 batch 的指标；`batch_transfer_*` 绘制 `After - Before`。

class-wise 图位于 `outputs/*/entropy/classwise/`，reliability 图位于
`outputs/*/entropy/reliability/`。这些图同样遵循 `entropy_curves.yaml` 中的 figsize、grid、
smooth、marker 和 series 设置。

多 seed 父 run 的绘图命令和单 seed 相同；父 run 会继续绘制 mean ± std 阴影带。训练曲线的
样式配置为 `configs/visualization/training_curves.yaml`，entropy 图的样式配置为
`configs/visualization/entropy_curves.yaml`。

## 恢复训练

单卡恢复：

```bash
cv-train \
  --resume runs/cifar10_single_gpu_entropy_<timestamp>/checkpoints/last.pt \
  --gpu-ids 0 \
  --epochs 220
```

四卡恢复：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m cv_corruption.cli.train \
  --resume runs/cifar10_4gpu_entropy_<timestamp>/checkpoints/last.pt \
  --gpu-ids 0 1 2 3 \
  --epochs 220
```

恢复时必须保持 GPU 数量、关键优化器参数和 AMP 设置一致。多 seed 父 run 不使用 `--resume`；
如需恢复某个 seed，请对相应的 `seed_<id>_<timestamp>` 子 run 单独恢复。

## 指标定义

Entropy 为 `-sum(p * log(p))`，normalized entropy 为 entropy 除以 `log(10)`；confidence 是
最大 softmax 概率；ECE 默认使用 10 个等宽 confidence bins。数据集指标按样本数加权，class-wise
指标按类别聚合。
