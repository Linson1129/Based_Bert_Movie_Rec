import pandas as pd
import os

# 1. 读取原始提取的数据
# 假设你的文件中有：movie_id, user_id, rating, title, genres, timestamp
df = pd.read_csv("data/extracted_movies.csv")

# 2. 严格按照规格书重命名和筛选列
# 规格书要求：user_id, item_id, rating, timestamp
df = df.rename(columns={'movie_id': 'item_id'})

# 3. 补齐 items.csv 所需的字段
# 规格书要求：item_id, title, description, category
print("正在生成标准 items.csv...")
items_df = df[['item_id', 'title', 'genres']].drop_duplicates('item_id')

items_final = pd.DataFrame({
    'item_id': items_df['item_id'],
    'title': items_df['title'],
    # 核心：将 genres 放入 description 供 BERT 学习语义
    'description': items_df['genres'].apply(lambda x: f"电影类型: {str(x).replace('|', ' ')}"),
    # 辅助：取第一个标签作为 category
    'category': items_df['genres'].apply(lambda x: str(x).split('|')[0] if pd.notna(x) else "Unknown")
})

# 4. 准备 interactions 数据
# 只保留规格书要求的四列
interactions_all = df[['user_id', 'item_id', 'rating', 'timestamp']].copy()
interactions_all = interactions_all.sort_values(by=['user_id', 'timestamp'])

# 5. 按照 8:2 拆分训练集和验证集
print("正在按 8:2 拆分交互数据...")
train_list = []
val_list = []

for uid, group in interactions_all.groupby('user_id'):
    n = len(group)
    if n < 5:
        train_list.append(group)
    else:
        split_idx = int(n * 0.8)
        train_list.append(group.iloc[:split_idx])
        val_list.append(group.iloc[split_idx:])

interactions_train = pd.concat(train_list)
interactions_val = pd.concat(val_list)

# 6. 统一保存到 data 目录
os.makedirs("data", exist_ok=True)

# 文件 1: 词典 (不拆分)
items_final.to_csv("data/items.csv", index=False)

# 文件 2: 训练集 (80%)
interactions_train.to_csv("data/interactions_train.csv", index=False)

# 文件 3: 验证集 (20%)
interactions_val.to_csv("data/interactions_val.csv", index=False)

# 文件 4: 全量表 (100%，用于评估时过滤已看内容)
interactions_all.to_csv("data/interactions.csv", index=False)

print("\n✅ 标准化处理完成！")
print(f"- items.csv: {len(items_final)} 行")
print(f"- interactions_train.csv: {len(interactions_train)} 行")
print(f"- interactions_val.csv: {len(interactions_val)} 行")