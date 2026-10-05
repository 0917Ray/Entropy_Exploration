# 环境与安装

## 安装

推荐使用 Python 3.11 的独立环境：

```bash
conda create -n cv_corruption python=3.11 -y
conda activate cv_corruption
python -m pip install --upgrade pip
python -m pip install -e CV_Corruption
```

绘图功能需要 Matplotlib；项目依赖中已包含常用依赖，也可以显式安装：

```bash
python -m pip install -e 'CV_Corruption[plot]'
```

## GPU 检查

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
```

单卡可用 `CUDA_VISIBLE_DEVICES=0` 限制可见设备。DDP 多卡示例见
[训练](training.md)。

## 数据位置

原始数据统一位于：

```text
data/raw/cifar10/cifar-10-batches-py/
data/raw/cifar10_c/CIFAR-10-C/
data/raw/tiny_imagenet_c/tiny-imagenet-200/
data/raw/tiny_imagenet_c/Tiny-ImageNet-C/
```

## 路径规则

YAML 中的相对路径相对于 `CV_Corruption/` 根目录解析：

```yaml
data: data/raw/cifar10_c/CIFAR-10-C
output: runs/visualizations/cifar10_c
```

因此从仓库上级目录和 `CV_Corruption/` 目录启动都有效。绝对路径也支持。
