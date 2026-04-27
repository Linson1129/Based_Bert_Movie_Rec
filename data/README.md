
# 数据来源与预处理说明 (Data Processing)

本文件夹包含用于模型训练和验证的电影推荐数据集。

## 📊 数据来源
数据源自 **MovieLens 最新公开数据集**，我们从中提取了 2014 年至 2024 年之间的现代电影数据，以确保 BERT 模型能够更好地处理具有当代语境的电影标题和描述。

## 🧹 数据清洗与处理流程

为了适配双塔模型的训练需求，我们对原始数据进行了以下处理：

### 1. 字段映射与标准化
我们将原始字段统一重命名为项目要求的标准格式：
- `movie_id` -> `item_id`: 电影唯一标识
- `genres` -> 拆分为 `category` (首选类型) 和 `description` (完整描述)
- `timestamp`: 统一转化为秒级时间戳

### 2. 采样与精简 (Downsampling)
为了平衡训练效率与模型性能，我们执行了**按用户采样**：
- 仅保留了活跃用户及其完整的行为链。
- 将总交互规模控制在约 **6,000 条**，大幅提升了 BERT 在初始化阶段的编码速度，降低了显存占用。

### 3. 数据集切分 (Train/Val Split)
采用了 **时间序切分 (Time-series Split)** 而非随机切分：
- **interactions_train.csv**: 每个用户前 80% 的历史行为，用于模型学习兴趣。
- **interactions_val.csv**: 每个用户最近 20% 的行为，模拟“未来预测”进行测试。
- **items.csv**: 电影元数据，作为模型查找语义信息的字典。

## 📋 文件规范
| 文件名 | 用途 | 关键列 |
| :--- | :--- | :--- |
| `items.csv` | 电影库 | item_id, title, description, category |
| `interactions_train.csv` | 训练集 | user_id, item_id, rating, timestamp |
| `interactions_val.csv` | 验证集 | user_id, item_id, rating, timestamp |
| `interactions.csv` | 全量数据 | 用于评估时过滤已看记录 |
