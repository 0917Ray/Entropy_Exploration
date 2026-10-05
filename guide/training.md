# 从头训练 CIFAR-10 ResNet-50

配置文件：`configs/training/cifar10_resnet50.yaml`。

训练使用随机初始化的 CIFAR-sized ResNet-50，不加载 ImageNet 权重。每次新训练都会创建带本地时间戳的独立目录，例如
`runs/cifar10_resnet50_20261002-153000`。只有 `--resume` 会复用原目录。

## 环境检查

```bash
python -m pip install -e 'CV_Corruption[plot]'
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

## CPU Smoke Test

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --cpu --smoke --batch-size 2 \
  --train-batches 1 --test-batches 1 --workers 0 \
  --output-dir runs/cifar10_resnet50_cpu_smoke
```

命令会实际创建类似 `cifar10_resnet50_cpu_smoke_YYYYMMDD-HHMMSS` 的目录。

## 单卡训练

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 --epochs 200 \
  --output-dir runs/cifar10_resnet50
```

## 多卡 DDP

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
  -m cv_corruption.cli.train \
  --config configs/training/cifar10_resnet50.yaml \
  --gpu-ids 0 1 2 3 --epochs 500 \
  --output-dir runs/cifar10_resnet50_4gpu_500epochs
```

Entropy 实验的四卡完整教程（包括 GPU 检查、四卡 smoke test、seed 批量运行和断点恢复）见
[Experiment 1：Entropy 监测](entropy_experiment.md#四张-gpu-运行)。

## 恢复训练

```bash
cv-train \
  --resume runs/cifar10_resnet50_YYYYMMDD-HHMMSS/checkpoints/last.pt \
  --gpu-ids 0 --epochs 220
```

恢复时 GPU 数量、关键优化器参数和 AMP 设置必须与 checkpoint 一致；可以增加
`epochs`，但不能把完整训练切换成 smoke。训练结果包含 `resolved_config.json`、
`config.json`、`metrics.jsonl`、日志、`checkpoints/` 和曲线 bundle。

## 训练曲线

项目级绘图模板位于 `configs/visualization/training_curves.yaml`。训练结束时，程序读取该
YAML，并将本次运行实际使用的配置保存为 run 目录内的
`configs/training_curves.json`；修改 YAML 不会改变已经完成的 run。

### 配置绘图样式

编辑以下文件可以控制后续 run 的默认绘图结果：

```text
CV_Corruption/configs/visualization/training_curves.yaml
```

常用字段示例：

```yaml
plot:
  mode: all-separate
  smooth: 5
figure:
  dpi: 300
  transparent: true
axis:
  xlabel: Epoch
  grid: true
output:
  filename: curves.png
  pdf: true
```

其中 `smooth` 是训练 loss 的显示平滑窗口；原始值仍完整保存在
`data/training_curves.csv`，不会被修改。YAML 只作为新生成 run 的模板，不会覆盖已经
完成实验的结果。

训练正常结束后会自动生成：

- `data/training_curves.csv`：完整精度的 epoch、train/validation loss 和 learning rate；
- `configs/training_curves.json`：可编辑的绘图配置；
- `scripts/accuracy_curves.py`：accuracy 曲线的独立重绘脚本；
- `outputs/png/`、`outputs/pdf/`：曲线图片，文件名只保留曲线主体；
- `logs/training_curves.log`：绘图日志；
- 四张 loss/LR PNG 和对应 PDF（训练损失、验证损失、两者比较、学习率）；
- 三张 accuracy PNG 和对应 PDF（训练准确率、验证准确率、两者比较）；
- `scripts/accuracy_curves.py`：accuracy 曲线的独立重绘脚本。

也可以在已有 run 上单独重绘：

```bash
cv-plot-training --run-dir CV_Corruption/runs/cifar10_resnet50_4gpu_300_epochs_20261002-204823 --smooth 1
```

对于已经完成的 run，修改其目录内的 `configs/training_curves.json` 后，执行
`cv-plot-training --run-dir <run-dir>` 即可离线重绘，
不需要重新训练。绘图使用 `plot-training-curves` skill 的标准配置和可视化样式；accuracy
曲线由 `scripts/accuracy_curves.py` 生成。

训练终端只显示 rank 0 的结构化日志，例如 `[INFO]`、`[WARNING]` 和
`[SUCCESS]`。绘图脚本的详细输出（包括 Matplotlib warning）保存在
`logs/training_curves.log`；完整训练日志保存在 `train.log`。设置
`OMP_NUM_THREADS=1` 可避免 `torchrun` 自身打印线程数提示。

配置文件中的 `log_every_epochs` 控制详细 epoch 日志打印频率，默认每 5 个 epoch 打印一次；
rank 0 同时显示 tqdm 进度条，其中包含完成比例、速度和预计剩余时间。训练默认使用 5 个
epoch 的 learning-rate warmup，以降低多卡大 batch 从头训练时出现 NaN 的风险。

## 常用覆盖

```bash
cv-train \
  --config configs/training/cifar10_resnet50.yaml \
  --epochs 20 --batch-size 256 --set seed=7 \
  --output-dir runs/cifar10_seed7
```
