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
