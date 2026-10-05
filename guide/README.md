# CV_Corruption Guide

本目录按工作流拆分使用说明。所有命令都可以从仓库上级目录或
`CV_Corruption/` 目录运行；推荐先安装项目包：

```bash
python -m pip install -e CV_Corruption
```

文档导航：

- [环境与安装](setup.md)
- [项目架构](architecture.md)
- [训练](training.md)
- [Experiment 1：Entropy 监测](entropy_experiment.md)
- [CIFAR-10-C 评测](evaluation.md)
- [可视化](visualization.md)
- 训练完成后可使用 `cv-plot-training --run-dir <run>` 生成或重绘 loss/LR 曲线。

训练曲线默认配置：`configs/visualization/training_curves.yaml`；训练指南中包含字段说明和
已有 run 的离线重绘方法。

配置文件按用途组织在 `configs/`：

```text
configs/
├── training/
├── evaluation/
└── visualization/
```

所有实验产物写入 `runs/`，原始数据位于 `data/raw/`。
