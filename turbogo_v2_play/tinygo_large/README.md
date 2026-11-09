<div align="center">

# TinyGo
### 基于 ResNet 的超轻量级高性能围棋 AI

[English](READMEs/README_EN.md) | [简体中文](README.md) | [日本語](READMEs/README_JP.md)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2FMPS-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---
基于 ResNet 神经网络的超轻量级围棋 AI，通过学习 KataGo 的自我对弈数据，以极少的参数量实现了高水平的对弈能力。该项目还提供了胜率预测功能以及可选的蒙特卡洛树搜索 (MCTS) 增强模式。



TinyGo 旨在打造能够在消费级硬件上流畅运行的围棋AI，同时是一个学习和了解ResNet神经网络构建和训练过程的完美项目。



下面是项目主要特点：

- **轻量级 ResNet 架构**: 采用专为围棋优化的自定义深度残差网络（ResNet）实现。

- 两个版本的模型：

  *   普通模型 (/src)：6 个残差块 ，64 个通道。
  *   大型模型 (/src_large) : 10 个残差块，128 个通道。

  *   超大模型 (/extra_large): 20 个残差块，256 个通道。

*   系统使用**监督学习 (Supervised Learning)**。 使用高质量的超过3000万个KataGo 自我对弈落子进行训练，确保策略网络的落子直觉接近顶尖 AI。
*   **胜率预测模块 (/win_rate)**: 独立的价值网络 (Value Network)，本质为均方误差回归任务，能够评估当前局面的胜率 (0-100%)。同样适用katago自我对弈数据进行训练。
    注意，胜率预测模块设计存在问题。KataGo的自我对弈数据中，双方都在极力避免犯错。因此，训练集中极度缺乏“一方下出恶手，导致胜率瞬间暴跌”的负样本。模型从未见过“开局第一手下在1-1位置”这种局面，因此它会根据惯性预测为50%。并且，均方误差回归任务似乎不合适，模型缺乏对于细微目数变化的敏感性。目前还未找到有效解决方案。
*   **MCTS 蒙特卡洛树搜索**: 结合策略网络（落子建议）和价值网络（局面评估）的完整搜索算法，大幅提升中盘战斗和死活计算能力。
*   内置基于 Tkinter 的图形界面，支持人机对弈、形势判断显示、悔棋等功能。
*   **跨平台支持**: 可运行于 Linux, macOS (支持 Apple Silicon MPS 加速), Windows (支持 NVIDIA CUDA 加速)。



## 成果

在未进行超大规模训练的情况下：

large模型使用4%数据（120万个moves样本）训练142轮，Top1准曲率在38.7%，Top5准确率70.4%。

large模型 + MCTSx100，打败了KataGo “直觉权重（1级位）”模式水平，且提升空间极大。

![KataGo-winrate](imgs/KataGo-winrate.png)

*图片中黑棋为TinyGo-large-MCTSx100，白棋为KataGo“直觉权重（1级位）”模式。



不使用MCTS的情况下，单独使用large模型推理，在RTX5070Ti GPU上单步推理速度达到了0.04秒的惊人速度，与KataGo “直觉权重（1级位）”模式水平大致相当。

win_rate的small模型，在使用400万样本训练18轮后，MSE为0.042，MAE达到了0.112，已经基本可用于MCTS分析。


2025年12月11日，best_model_extra_large.pth首次出山。Top1准确率达到44.6%，Top5准确率达到90.4%。模型文件大小273MB，在RTX5070Ti上的推理速度依旧高达<0.09s。
该模型实际水平约为准2段（业余）。


你可以在http://server.i-o.autos:9999上体验与该模型对弈。





### 数据表示

模型输入为 `(Batch_Size, 3, 19, 19)` 的张量，包含 3 个特征平面：

1.  **我方棋子**: 当前执子一方的棋子位置 (1 为有子，0 为无子)。
2.  **对手棋子**: 对手一方的棋子位置。
3.  **偏置平面 (Bias)**: 全 1 矩阵，用于辅助网络捕捉空地信息和边界特征。

### 网络架构

TinyGo 采用了经典的 AlphaZero 风格的 ResNet 架构：

*   **卷积块 (Initial Block)**: 3x3 卷积核，Stride=1，Padding=1，接 Batch Normalization 和 ReLU 激活。
*   **残差塔 (Residual Tower)**: 堆叠 $N$ 个残差块。
    *   每个残差块包含：`Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add Skip Connection -> ReLU`。
*   **策略头 (Policy Head)**:
    *   1x1 卷积，输出通道数为 2。
    *   Batch Normalization + ReLU。
    *   全连接层 (Linear): 输入 `2 * 19 * 19`，输出 `361` (对应棋盘 19x19 个落子点)。
*   **价值头 (Value Head)** (仅限胜率模型):
    *   1x1 卷积，输出通道数为 1。
    *   全连接层 -> ReLU -> 全连接层 -> Sigmoid，输出 0~1 之间的标量。

### 训练配置

*   **优化器**: Adam
*   **学习率调度**: `ReduceLROnPlateau` (当验证集准确率不再提升时，学习率衰减 0.5 倍)。
*   **数据增强 (Data Augmentation)**: 为了充分利用数据，训练过程中在 GPU 上实时进行 8 种对称变换（旋转 0/90/180/270度 $\times$ 是否翻转）。
*   **损失函数**: CrossEntropyLoss (策略网络), MSE/BCE (价值网络)。

### MCTS 算法

*   **PUCT 公式**: 用于平衡“利用”(Exploitation, 选择高胜率点) 和 “探索”(Exploration, 访问较少的点)。
    *   $U(s, a) = C_{puct} \times P(s, a) \times \frac{\sqrt{N(s)}}{1 + N(s, a)}$
*   **模拟次数**: 默认 100 次/步 (可配置)。
*   **多线程**: 支持在 Python 层面通过批处理评估来加速搜索。







## 安装指南

1. **克隆项目代码**

   ```bash
   git clone https://github.com/LaoChouPro/TinyGo.git
   cd TinyGo
   ```

2. **安装依赖**
   请确保您的 Python 版本 $\ge$ 3.8。

   ```bash
   pip install -r requirements.txt
   ```

   *   **GPU 加速提示**:
       *   **NVIDIA 用户**: 请安装 CUDA 版本的 PyTorch (例如: `pip install torch --index-url https://download.pytorch.org/whl/cu118`)。
       *   **Mac 用户**: PyTorch 默认支持 MPS (Metal Performance Shaders)，无需额外操作。







## 使用TinyGo进行人机对弈、继续训练

### 1. 人机对弈 (Play)

**标准模式 (极速响应)**

仅使用策略网络进行落子，速度极快，适合快速测试布局和定式。

注意，该脚本使用large模型。你可以在源码中更改模型路径。

```bash
python play_gui.py
```

**MCTS 增强模式 (最强棋力)**
开启蒙特卡洛树搜索，结合胜率估值，具备深度计算能力。

```bash
python play_gui_mcts.py
```



### 2. 模型训练 (Train)

如果您想要使用现有数据继续训练（当前模型未进行超大规模训练），或希望复现训练结果，可以使用以下脚本。

**训练策略网络 (Policy Network)**

```bash
python src/train.py #同理，使用python src_large/train.py可以训练large模型。
```

根据交互式提示设置：

*   `Epochs`: 训练轮数，注意，该数应该继之前训练到的轮数继续，比如best模型训练到了30轮，则输入35轮以继续训练5轮。
*   `Batch Size`: 批大小 (推荐 64-1024，视显存而定)。
*   `Learning Rate`: 学习率 (默认 0.001，会使用优化器自动调度)。

**训练价值网络 (Value Network)**

```bash
cd win_rate
python process_data.py  # 第一步：预处理数据
python src_small/train.py  # 第二步：开始训练
```



## 项目结构详解

*   `src/`: **标准模型源码 (Standard Model)**
    *   `model.py`: 定义 TinyGoNet (6 blocks) 结构。
    *   `train.py`: 主训练循环，包含数据加载和验证逻辑。
    *   `dataset.py`: 处理 SGF 数据，生成特征张量。
*   `src_large/`: **大型模型源码 (Large Model)**
    *   包含 10 blocks, 128 channels 的模型定义。
*   `src_extra_large/`: **超大模型源码 (Extra Large Model)**
    *   包含 20 blocks, 256 channels 的模型定义。
*   `src_mcts/`: **蒙特卡洛树搜索核心**
    *   `mcts.py`: MCTS 主循环逻辑。
    *   `node.py`: 树节点定义及 PUCT 选择公式。
*   `win_rate/`: **胜率预测模块**
    *   包含独立的数据处理和价值网络训练代码。
*   `play_gui.py`: 标准模式启动入口。
*   `play_gui_mcts.py`: MCTS 模式启动入口。





## 致谢
*   训练数据来源于开源的 **KataGo** 自我对弈棋谱,来自https://katagoarchive.org。



## 许可证

Apache License Version 2.0, January 2004
