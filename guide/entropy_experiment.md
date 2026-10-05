# Experiment 1：预训练阶段 Entropy 监测

本实验使用 CIFAR-10 上从头训练的 CIFAR-sized ResNet-50，记录训练过程中模型的
entropy、confidence、accuracy、cross-entropy 和 ECE。实验默认训练 200 个 epoch；不同
随机种子应使用独立 run 目录保存。

## 指标定义

对一个 batch 的 softmax 概率 `p`：

- `entropy`：`-sum(p * log(p))`，CIFAR-10 的最大值为 `log(10)`。
- `normalized_entropy`：`entropy / log(10)`，范围为 `[0, 1]`。
- `confidence`：最大 softmax 概率。
- `accuracy`：预测类别与标签相等的样本比例。
- `loss`：交叉熵损失。
- `ece`：默认 10 个等宽 confidence bins 的 Expected Calibration Error。

数据集级指标按样本数加权，不能对 batch 指标做简单平均。class-wise 指标只对该类
样本聚合；一个 batch 或数据集不包含某类时，该类的 `samples` 为 0。

## 运行

先确认 CIFAR-10 pickles 位于 `data/raw/cifar10/cifar-10-batches-py/`，然后执行：

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --seed 0 \
  --epochs 200 \
  --ece-bins 10 \
  --output-dir runs/entropy_seed0
```

推荐对多个 seed 分别运行，例如 `0, 1, 2, 3, 4`。不要复用已有 output 目录；训练程序会
自动创建带时间戳的目录并保存 resolved configuration。

CPU smoke test：

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --cpu --smoke --batch-size 2 \
  --train-batches 1 --test-batches 1 --workers 0 \
  --output-dir runs/entropy_smoke
```

smoke 模式只处理少量 batch，适合验证代码和输出格式，不用于报告实验结果。

## 四张 GPU 运行

先确认四张卡均可见：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
```

四卡 DDP 使用 `torchrun` 启动四个进程；`--gpu-ids` 必须与
`--nproc_per_node` 一一对应。下面命令运行 seed 0 的完整 200 epoch entropy 实验：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m cv_corruption.cli.train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 1 2 3 \
  --seed 0 \
  --epochs 200 \
  --batch-size 256 \
  --ece-bins 10 \
  --output-dir runs/entropy_seed0_4gpu
```

这里的 `batch-size` 是每张 GPU 的 batch size，全局 batch size 为
`batch_size * 4`。学习率会按全局 batch size 相对于 128 自动缩放。不同随机种子必须使用
不同的 `--output-dir`，例如：

```bash
for seed in 0 1 2 3 4; do
  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
    -m cv_corruption.cli.train \
    --config configs/training/cifar10_resnet50.yaml \
    --gpu-ids 0 1 2 3 --seed "$seed" --epochs 200 \
    --output-dir "runs/entropy_seed${seed}_4gpu"
done
```

四卡运行时 rank 0 负责写日志、checkpoint 和指标文件；训练和测试数据由 DDP sampler
分片，epoch-level entropy 评估使用固定顺序的完整 loader。启动后应看到
`rank=0/4`、`gpus=[0, 1, 2, 3]`，否则说明进程数或 GPU 参数不匹配。

### 四卡 smoke test

正式运行前可以用四卡做快速检查：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m cv_corruption.cli.train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 1 2 3 --seed 0 --smoke \
  --batch-size 2 --train-batches 1 --test-batches 1 --workers 0 \
  --output-dir runs/entropy_smoke_4gpu
```

### 断点恢复

四卡训练必须使用相同数量的 GPU、相同关键优化器参数和相同 AMP 设置恢复：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m cv_corruption.cli.train \
  --resume runs/entropy_seed0_4gpu_YYYYMMDD-HHMMSS/checkpoints/last.pt \
  --gpu-ids 0 1 2 3 --epochs 220
```

常见错误：`world_size` 与 `--gpu-ids` 数量不一致时，请确认
`--nproc_per_node=4 --gpu-ids 0 1 2 3`；显存不足时优先降低每卡 `--batch-size`，不要改变
GPU 数量后直接恢复旧 checkpoint。

## 输出文件

每个 run 目录包含：

```text
metrics/epoch_metrics.jsonl   # epoch 0 和每个完成 epoch 的 train/test 指标
metrics/batch_metrics.jsonl   # 每个训练 batch 的 before/after/delta 指标
config.json                   # 实际训练配置
resolved_config.json          # CLI/YAML 解析后的配置
checkpoints/                  # last.pt 和 best.pt
```

`epoch_metrics.jsonl` 的每行包含 `epoch`、`train` 和 `test`。其中每个 split 具有
`samples`、`entropy`、`normalized_entropy`、`confidence`、`accuracy`、`loss`、`ece`
和 `classwise` 字段。epoch 0 使用未训练模型，并且评估 loader 不使用数据增强。

`batch_metrics.jsonl` 的每行包含 `epoch`、`batch`、`before`、`after` 和 `delta`。其中
`delta[q] = after[q] - before[q]`，表示同一 batch 在一次参数更新前后的 transfer effect。
before/after 的 class-wise 指标也会保留。

## 绘图

训练成功结束后，程序会自动调用 entropy 绘图器，图片保存到：

```text
outputs/png/entropy/*.png
outputs/pdf/entropy/*.pdf

原有 loss/accuracy/LR 图位于：

```text
outputs/png/training_curves/*.png
outputs/pdf/training_curves/*.pdf
```
```

当前自动生成的图包括：

- train/test entropy、normalized entropy、confidence、accuracy、loss、ECE 曲线；
- entropy-confidence-accuracy 的绝对值、epoch 差分和变化率；
- epoch 0/10/50/100（若存在）的 class-wise entropy、accuracy、confidence；
- 每个类别单独的 entropy/accuracy 随 epoch 图；
- 按配置分组的 class-wise entropy/accuracy 图；
- epoch 0、早期、中期和最终 checkpoint 的 reliability summary；
- batch entropy/loss 以及相邻 batch transfer effect。

batch 图分别输出为：

```text
batch_entropy.pdf/png
batch_loss.pdf/png
batch_transfer_entropy.pdf/png
batch_transfer_loss.pdf/png
```

已有 run 不需要重新训练，可以单独重绘：

```bash
cv-plot-entropy \
  --run-dir runs/entropy_seed0_4gpu_20261005-123039
```

class-wise 绘图由 `configs/visualization/entropy_curves.yaml` 控制。配置中的 class 编号
默认使用人类习惯的 1-based 编号（1 到 10），例如：

```yaml
class_index_base: 1
class_groups:
  - [1, 10]
  - [2, 3, 4]
include_aggregate_classwise: false
```

这会生成每个 class 独立的 entropy/accuracy 图（共 20 张），以及分开的分组 entropy/accuracy
图（classes 1/10 和 classes 2/3/4）。不同 class 在分组图中使用不同颜色。修改配置后
重跑同一条 `cv-plot-entropy` 命令即可；也可以指定其他配置文件：

输出文件示例：

```text
class_01_entropy_over_epoch.pdf
class_01_accuracy_over_epoch.pdf
classes_01_10_entropy_over_epoch.pdf
classes_01_10_accuracy_over_epoch.pdf
```

```bash
cv-plot-entropy \
  --run-dir runs/entropy_seed0_4gpu_20261005-123039 \
  --config configs/visualization/entropy_curves.yaml
```

如果尚未执行 `pip install -e .`，也可以直接使用模块入口：

```bash
PYTHONPATH=src python -m cv_corruption.cli.plot_entropy \
  --run-dir runs/entropy_seed0_4gpu_20261005-123039
```

绘图器只读取 `metrics/epoch_metrics.jsonl` 和 `metrics/batch_metrics.jsonl`，不会重新加载
模型或数据集。当前 run 保存的是 aggregate confidence/accuracy/ECE，因此 reliability 图
以 mean confidence-accuracy 点和 ECE 标注呈现；后续若保存逐 bin calibration counts，可
扩展为完整的 bin reliability diagram。

## 检查结果

运行回归测试：

```bash
python -m unittest discover -s tests -v
```

当前实现已覆盖指标计算、样本加权、ECE bin 配置和 batch transfer effect 的数据记录。
entropy-confidence-accuracy 曲线、指定 epoch 的 class-wise 图和 reliability summary 可
使用上述 CLI 离线生成。

## 与 Experiment 2 的边界

本阶段不执行 CIFAR-10-C 上的 TTA，也不实现 EM-only 更新。Experiment 2 应使用冻结的
预训练 checkpoint，单独记录每个 corruption/severity 和 EM-only 训练轮数的 entropy 与
真实 accuracy，避免将测试集适应过程混入本实验的预训练指标。
