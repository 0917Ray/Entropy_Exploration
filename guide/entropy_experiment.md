# Entropy 监控（统一训练 pipeline）

本页保留用于兼容旧链接。Entropy 监控已经并入统一的 CIFAR-10 ResNet-50 训练流程，完整说明、
命令、输出和离线重绘请参阅 [训练与监控指南](training.md)。

训练时使用 `--entropy-experiment` 只表示训练结束后自动渲染 entropy 图片；即使不使用该参数，
`metrics/epoch_metrics.jsonl` 和 `metrics/batch_metrics.jsonl` 仍会完整记录 entropy、
confidence、accuracy、loss、ECE 以及 class-wise 指标。

已有 run 可以直接重绘：

```bash
cv-plot-entropy \
  --run-dir runs/cifar10_resnet50_YYYYMMDD-HHMMSS \
  --config configs/visualization/entropy_curves.yaml
```

Entropy 的每个 epoch metric 和 relationships 图都会生成 train/test 合并图、仅 train 图和
仅 test 图。例如：

```text
outputs/png/entropy/epoch_metrics/epoch_entropy.png
outputs/png/entropy/epoch_metrics/epoch_entropy_train.png
outputs/png/entropy/epoch_metrics/epoch_entropy_test.png
```

对应 PDF 文件位于 `outputs/pdf/entropy/`，并使用相同的命名规则。绘图样式、smooth、原始
曲线、marker 和各图的 `series` 覆盖项统一配置在 `configs/visualization/entropy_curves.yaml`。
