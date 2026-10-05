# 项目架构

```text
CV_Corruption/
├── src/cv_corruption/       唯一源码包
│   ├── cli/                 命令入口
│   ├── config/              配置加载与校验代码
│   ├── data/                数据集与 transform
│   ├── models/              模型定义和工厂
│   ├── training/            训练循环、checkpoint、DDP 辅助
│   ├── evaluation/          评测和指标
│   ├── visualization/       绘图公共逻辑
│   └── common/              路径、日志、随机数、环境辅助
├── configs/                 实验参数文件
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── data/raw/                原始数据
├── data/processed/          可复用的预处理数据
├── runs/                    checkpoint、日志、指标、图像
├── tests/                   快速回归测试
└── docs/                    长篇数据和实验报告
```

依赖方向：

```text
cli -> training -> data/models/common
cli -> evaluation -> data/models/common
cli -> visualization -> data/common
```

`src/cv_corruption/config/` 放的是配置处理代码，不放 YAML。YAML 属于具体实验，
放在 `configs/` 中并按用途分目录。

## 配置优先级

```text
代码默认值 < YAML/JSON 配置 < CLI 参数 < --set
```

运行时最终配置会保存到对应的 `runs/` 目录，便于复现实验。
