# CV_Corruption 项目架构

## 目标

本项目采用“模块化单体 + 轻量分层”的结构，服务于训练、分布偏移评测和数据可视化。所有实验仍在一个 Python 包中运行，但数据、模型、训练、评测和 CLI 之间具有明确边界，便于复现实验和逐步扩展新的数据集或模型。

## 目录

```text
CV_Corruption/
├── pyproject.toml                  # 可编辑安装和 CLI 入口
├── src/cv_corruption/
│   ├── cli/                        # 参数解析和命令编排
│   ├── config/                     # 配置加载、默认值和轻量 schema
│   ├── data/                       # CIFAR-10、CIFAR-10-C 等数据集
│   ├── models/                     # 模型定义和工厂
│   ├── training/                   # 训练循环、checkpoint、分布式辅助
│   ├── evaluation/                 # 评测、指标和 corruption 定义
│   ├── visualization/              # 图像 gallery 和绘图公共逻辑
│   └── common/                     # 路径、随机种子、日志、环境辅助
├── configs/                        # 按实验类型组织的 YAML/JSON 配置
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── data/raw/                       # 原始数据（不提交 Git）
├── data/processed/                 # 可复用的预处理数据
├── runs/                           # checkpoint、日志、指标和图片
├── tests/                          # 快速回归测试
└── docs/                           # 长篇实验说明
```

源码唯一保留在 `src/cv_corruption/`；旧的 `algorithms/`、`datasets/`、`models/` 和 `utils/` 实现已移除。原始数据统一放在 `data/raw/`，实验产物统一放在 `runs/`。

## 依赖方向

```text
cli -> training -> data/models/common
cli -> evaluation -> data/models/common
cli -> visualization -> data/common
```

底层模块不依赖 CLI；可视化不参与训练；评测不修改模型参数。跨模块逻辑应放到明确的 `common` 子模块，而不是继续扩张一个全能 `utils.py`。

## 运行方式

在 `CV_Corruption/` 目录执行：

```bash
python -m pip install -e .
cv-train --config configs/training/cifar10_resnet50.yaml --cpu --smoke
cv-evaluate --config configs/evaluation/cifar10_c.yaml --cpu
cv-visualize --config configs/visualization/cifar10_c.yaml
```

也可以直接调用模块：

```bash
python -m cv_corruption.cli.train --help
```

配置优先级保持为：代码默认值 < YAML/JSON 文件 < CLI 参数 < `--set key=value`。每个实验的解析配置应和结果一起保存到 `runs/`。

## 新增功能的约定

1. 新数据集放在 `data/`，只负责读取、校验和 transform。
2. 新模型放在 `models/`，通过 `models/factory.py` 暴露构造入口。
3. 新训练 recipe 放在 `training/`，CLI 只负责编排。
4. 新指标放在 `evaluation/metrics.py`，不要写在绘图脚本中。
5. 原始数据和大型实验结果不进入源码包，也不应提交到 Git。
