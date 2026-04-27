# BERT-DualTower-Rec: 基于双塔语义召回的电影推荐系统

这是一个基于深度学习的推荐系统，利用 **BERT** 模型作为特征提取器，构建 **双塔模型 (Dual-Tower Model)** 实现向量化召回。

## 🚀 项目架构
项目采用语义匹配思路：
- **用户塔 (User Tower)**：对用户历史观看的电影标题进行建模，捕捉用户兴趣偏好。
- **物品塔 (Item Tower)**：对电影标题和题材进行建模，生成电影的语义向量。
- **计算**：通过计算用户向量与电影向量的余弦相似度（Cosine Similarity），从海量库中快速检索用户感兴趣的内容。

## 📂 目录结构
- `src/`: 核心源代码（模型、数据加载、训练、评估）
- `configs/`: 配置文件，包含超参数和路径设置
- `data/`: 存放处理后的数据文件
- `checkpoints/`: 存放训练好的模型权重和 Faiss 索引

## 🛠️ 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python src/train.py --config configs/default.yaml
```

### 3. 模型评估与索引生成
```bash
python src/evaluate.py --config configs/default.yaml
```

### 4. 实时推荐 (Inference)
```bash
python src/inference.py --text "我想看科幻大片，类似于星际穿越的那种"
```
