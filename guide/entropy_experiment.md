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

## 检查结果

运行回归测试：

```bash
python -m unittest discover -s tests -v
```

当前实现已覆盖指标计算、样本加权、ECE bin 配置和 batch transfer effect 的数据记录。
entropy-confidence-accuracy 曲线、指定 epoch 的 class-wise 图和 reliability diagram 使用
上述 JSONL 数据离线生成；绘图 CLI 将在后续实验迭代中补充。

## 与 Experiment 2 的边界

本阶段不执行 CIFAR-10-C 上的 TTA，也不实现 EM-only 更新。Experiment 2 应使用冻结的
预训练 checkpoint，单独记录每个 corruption/severity 和 EM-only 训练轮数的 entropy 与
真实 accuracy，避免将测试集适应过程混入本实验的预训练指标。
