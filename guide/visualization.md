# 可视化

## CIFAR-10 clean 样例

配置文件：`configs/visualization/cifar10_examples.yaml`。

```bash
cv-visualize-examples \
  --config configs/visualization/cifar10_examples.yaml
```

## CIFAR-10-C corruption 对照图

配置文件：`configs/visualization/cifar10_c.yaml`。

```bash
cv-visualize \
  --config configs/visualization/cifar10_c.yaml
```

只画一个 corruption 和部分 severity：

```bash
cv-visualize \
  --config configs/visualization/cifar10_c.yaml \
  --set 'groups={noise: [gaussian_noise]}' \
  --set 'severities=[1, 3, 5]' --no-save-pdf
```

## Tiny ImageNet-C

```bash
cv-visualize-tiny-imagenet \
  --config configs/visualization/tiny_imagenet_c.yaml
```

## Corruption 统计柱状图

配置文件：`configs/visualization/corruption_counts.json`。

```bash
cv-plot-corruption-counts \
  --config configs/visualization/corruption_counts.json
```

## CIFAR-10 ResNet-50 训练曲线

训练完成后，可以从已有 run 离线生成或重绘 training loss、validation loss、learning rate
以及 train/validation accuracy 曲线：

```bash
cv-plot-training --run-dir runs/<run-directory>
```

默认配置位于 `configs/visualization/training_curves.yaml`。默认开启窗口为 5 的移动平均，
并且每条曲线最多绘制 10 个 marker。配置中的绘图开关如下：

```yaml
plot:
  smooth_enabled: true
  smooth: 5
  show_original: true
  markers:
    enabled: true
    count: 10       # 5、10、15、20 或 MAX
```

也可以在命令行中临时覆盖这些设置：

```bash
# 关闭 smooth 和 marker，只保留原始曲线
cv-plot-training --run-dir runs/<run-directory> \
  --no-smooth --no-markers

# 保留 smooth，但隐藏低透明度的原始曲线
cv-plot-training --run-dir runs/<run-directory> --no-original

# 使用窗口 15，并且每条曲线最多绘制 20 个 marker
cv-plot-training --run-dir runs/<run-directory> \
  --smooth-window 15 --markers 20

# 绘制全部 marker
cv-plot-training --run-dir runs/<run-directory> --markers MAX
```

`count: MAX` 表示每个记录点都绘制 marker；关闭 smooth 时显示原始曲线，开启 smooth 时
默认保留低透明度点线样式的原始曲线作为参考，并以平滑曲线作为主曲线。`show_original: false`
或 `--no-original` 可以隐藏这条参考曲线。`b_val_loss` 和
`c_train_vs_val_loss` 中的 validation 曲线也使用同一个 smooth 设置。accuracy 图与 loss
图使用相同的边框宽度和半透明 marker 样式。

## Entropy 监控曲线

Entropy 图使用 `configs/visualization/entropy_curves.yaml`。epoch、关系图、class-wise
曲线和 batch 曲线统一使用以下绘图配置：

```yaml
plot:
  smooth_enabled: true
  smooth: 5
  show_original: true
  markers:
    enabled: true
    count: 10       # 5、10、15、20 或 MAX
figure:
  spine_width: 2.5
  transparent: true

series:
  train:
    color: '#5E887E'
    linewidth: 2.0
    line_alpha: 0.92
    marker: o
    marker_size: 5.0
    marker_face_alpha: 0.60
    marker_edge_alpha: 0.95
    marker_edge_width: 1.35
  test:
    color: '#BA6580'
    linewidth: 2.0
    line_alpha: 0.92
    marker: o
    marker_size: 5.2
    marker_face_alpha: 0.60
    marker_edge_alpha: 0.95
    marker_edge_width: 1.35
```

已有 run 可以通过 CLI 临时覆盖：

```bash
cv-plot-entropy --run-dir runs/<run-directory> \
  --smooth-window 15 --markers 20
cv-plot-entropy --run-dir runs/<run-directory> \
  --no-smooth --no-markers --no-original
```

Entropy 的原始曲线使用低透明度点线，平滑曲线使用实线；关闭 smooth 时只绘制原始曲线。
Entropy 的 `figure`、`axis`、`legend` 和 `series` 参数与 training curves 使用相同的数值约定；
其中 train/test 的线宽、透明度、marker 形状和大小分别对应 training curves 的 train loss
和 validation loss 样式。
`series.train` 和 `series.test` 可以分别控制颜色、线宽、线透明度、marker 形状、marker 大小、
marker 填充透明度、marker 边缘透明度和 marker 边缘宽度。classwise 曲线使用 `series.test`
的线型参数，但颜色固定为 classwise 红色。
还可以按具体图覆盖默认样式，例如 `series.epoch_entropy`、`series.epoch_accuracy`、
`series.epoch_confidence`、`series.epoch_ece`、`series.epoch_loss`、
`series.epoch_normalized_entropy`、`series.classwise`、`series.batch_entropy` 和
`series.batch_loss`。覆盖项会与对应的 `train`/`test` 默认样式合并，因此只需写需要修改的字段：

```yaml
series:
  epoch_accuracy:
    train:
      color: '#516480'
      linewidth: 2.4
      marker_size: 6.0
    test:
      marker: '^'
      marker_face_alpha: 0.35
```
每个 epoch metric（entropy、confidence、accuracy、loss、ECE 等）都会生成三张图：
train/test 合并图、仅 train 图和仅 test 图。relationships 的普通、delta、rate 三组图也
遵循相同的三图结构；合并图保留原文件名，单 split 图分别添加 `_train` 和 `_test` 后缀。
例如 entropy 的文件名为 `epoch_entropy.png`、`epoch_entropy_train.png` 和
`epoch_entropy_test.png`；PNG 与 PDF 目录下都遵循同一命名规则。

其他 entropy 输出位于同一目录下：`classwise/` 保存 class-wise 图，`reliability/` 保存
可靠性摘要图，`batch/` 保存 batch entropy/loss 以及 `After - Before` transfer 图。batch
图中的 `Before`/`After` 分别表示一次 optimizer 更新前后对同一个 batch 的指标。

所有可视化产物默认写入 `runs/dataset_visualizations/`。各 YAML/JSON 文件保存了数据、
输出、图像尺寸和样式参数；修改配置即可复用同一入口。

## 输出目录

新的输出按产物类型归类：

```text
<run>/
├── outputs/png/      PNG 图片
├── outputs/pdf/      PDF 图片
├── data/             CSV 等数据
├── configs/          实际生效配置
├── metadata/         样本清单和审计元数据
├── reports/          Markdown 报告
└── logs/             运行日志
```
