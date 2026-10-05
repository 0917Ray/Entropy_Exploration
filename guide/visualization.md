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

所有可视化产物默认写入 `runs/visualizations/`。各 YAML/JSON 文件保存了数据、
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
