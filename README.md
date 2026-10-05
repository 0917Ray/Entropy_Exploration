# CV Corruption

项目架构说明见 [`guide/architecture.md`](guide/architecture.md)。新的代码包位于
`src/cv_corruption/`，建议先执行：

```bash
python -m pip install -e .
cv-train --help
cv-evaluate --help
```

源码统一位于 `src/cv_corruption/`，原有旧目录已移除。原始数据位于 `data/raw/`，实验产物位于 `runs/`。

This task owns image-corruption datasets, CV model definitions, training,
evaluation, visualization, and their outputs.

从 [guide](guide/README.md) 开始运行 smoke test、完整训练、DDP、恢复训练、
CIFAR-10-C 评测和数据可视化。详细数据报告保留在 `docs/CV_Corruption/`。

```text
src/cv_corruption/           唯一源码包
configs/training/             training experiment parameters
configs/evaluation/           evaluation experiment parameters
configs/visualization/        visualization experiment parameters
data/raw/                     local archives and extracted data
guide/                        split usage guides
runs/<experiment>_<timestamp>/ checkpoints, metrics, logs, and figures
tests/                        configuration and CLI regression tests
```

The CIFAR-10 ResNet-50 baseline uses normalized `3 x 32 x 32` tensors and
stores its clean checkpoints under `CV_Corruption/runs/cifar10_resnet50_*`.
CIFAR-10-C evaluation is a separate frozen-model workflow.

Experiment entry points accept `--config`; the three galleries load their
default YAML automatically.
CLI values override file values; `--set key=value` provides final overrides.
Resolved configurations are saved next to results. See the Guide for field
definitions, path rules, CPU smoke tests and DDP commands.
