# CIFAR-10-C 评测

配置文件：`configs/evaluation/cifar10_c.yaml`。

先准备 clean CIFAR-10 checkpoint，再运行：

```bash
cv-evaluate \
  --config configs/evaluation/cifar10_c.yaml \
  --gpu-id 0
```

快速检查单个 corruption：

```bash
cv-evaluate \
  --config configs/evaluation/cifar10_c.yaml \
  --corruptions gaussian_noise \
  --severities 1 --limit 16 --batch-size 16 --gpu-id 0
```

默认评测标准 15 种 corruption、5 个 severity。结果写入配置中的 `output`，
并额外保存 `<output>_config.json`。

主要参数：

- `checkpoint`：clean 模型 checkpoint
- `data_dir`：CIFAR-10-C 数据目录
- `clean_data_dir`：用于校验标签顺序的 clean CIFAR-10 目录
- `corruptions`、`severities`：评测子集
- `limit`：每个条件最多评测的样本数，范围为 1 到 10,000
- `cpu`、`gpu_id`、`batch_size`、`workers`：执行设备和 DataLoader 参数
