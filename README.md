# Transformer Ablation Study

手工搭建的Transformer模型，在小规模文本建模任务上进行训练和消融实验。

## 项目结构
```text
transformer-ablation-study/
├── src/                    # 源代码
│   ├── models/            # 模型定义
│   ├── data/              # 数据加载
│   ├── training/          # 训练逻辑
│   └── utils/             # 工具函数
├── scripts/               # 运行脚本
├── results/               # 实验结果
├── requirements.txt       # 依赖包
└── README.md              # 说明文档
```




## 环境要求

- Python 3.7+

- PyTorch 1.9+

- 其他依赖见 requirements.txt

  

## 硬件要求

- **最低配置**: 4GB RAM, 无需GPU

- **推荐配置**: 8GB RAM, NVIDIA GPU (支持CUDA)

- **预计运行时间**: 2.8小时 (取决于硬件)

  

## 安装和运行

### 1. 克隆仓库
git clone <repository-url>
cd transformer-ablation-study

### 2. 安装依赖
pip install -r requirements.txt

### 3. 运行实验
\#使用脚本运行

chmod +x scripts/run.sh
./scripts/run.sh

\#或直接运行

python main.py

### 4. 使用特定随机种子
\#修改 main.py 中的 set_seed() 函数调用

python main.py  # 默认使用种子42



## 数据集

使用IWSLT2017英德翻译数据集的小规模版本，包含约200个训练样本。如果数据文件不存在，代码会自动生成演示数据。

## 实验设置

消融实验包含以下变体：
- baseline: 基准模型 (4层, 8头, 256维)
- small_layers: 减少层数 (2层)
- small_heads: 减少注意力头数 (4头)
- small_dim: 减小模型维度 (128维)
- no_pos_encoding: 移除位置编码
- tiny_model: 超小模型 (2层, 4头, 128维)

## 预期结果

实验将生成：

1. 训练和验证损失曲线
2. 各模型最终困惑度比较
3. 参数量统计
4. 模型性能对比分析



## 复现说明

要完全复现实验结果：

1. 确保使用相同的随机种子 (42)

2. 使用相同的硬件配置

3. 按照提供的命令顺序执行

   

## 许可证

MIT License



## 关键特性

1. **模块化设计**: 代码结构清晰，易于扩展
2. **训练稳定性**: 包含梯度裁剪、学习率调度、AdamW优化器
3. **完整的实验流程**: 从数据加载到结果可视化
4. **消融研究**: 系统比较不同架构变体
5. **易于复现**: 设置随机种子，提供详细说明



## 运行说明


### 1. 克隆项目
git clone <your-repo-url>
cd transformer-ablation-study

### 2. 创建虚拟环境 (可选)
python -m venv venv
source venv/bin/activate  # Linux/Mac

\#venv\Scripts\activate  # Windows

### 3. 安装依赖
pip install -r requirements.txt

### 4. 运行实验
python main.py

### 5. 查看结果
\#结果将保存在 results/ 目录中
