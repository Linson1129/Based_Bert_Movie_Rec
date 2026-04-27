import pandas as pd
import os


def downsample_data(data_dir, target_rows=6000):
    print(f"开始精简数据，目标行数：约 {target_rows} 条...")

    # 1. 加载原始全量数据
    # 这里读取你之前生成的那个 100% 的总表
    if not os.path.exists(os.path.join(data_dir, "interactions.csv")):
        print("错误：找不到 interactions.csv，请确保路径正确")
        return

    df = pd.read_csv(os.path.join(data_dir, "interactions.csv"))

    # 2. 计算采样比例
    # 我们按用户进行采样，这样能保证保留下来的用户拥有完整的行为链
    current_rows = len(df)
    sample_ratio = target_rows / current_rows

    unique_users = df['user_id'].unique()
    sample_user_count = int(len(unique_users) * sample_ratio)

    # 随机选出一部分用户
    import numpy as np
    np.random.seed(42)  # 固定随机种子，方便复现
    sampled_users = np.random.choice(unique_users, size=sample_user_count, replace=False)

    # 3. 提取这些用户的交互记录
    df_small = df[df['user_id'].isin(sampled_users)].copy()
    print(f"采样后实际行数: {len(df_small)}")

    # 4. 重新进行 8:2 拆分
    train_list = []
    val_list = []
    for _, group in df_small.groupby('user_id'):
        n = len(group)
        if n < 2:
            train_list.append(group)
        else:
            split_idx = int(n * 0.8)
            train_list.append(group.iloc[:split_idx])
            val_list.append(group.iloc[split_idx:])

    df_train = pd.concat(train_list)
    df_val = pd.concat(val_list)

    # 5. 同步更新 items.csv (只保留这些交互中出现的电影，减小 BERT 编码压力)
    items_full = pd.read_csv(os.path.join(data_dir, "items.csv"))
    active_item_ids = df_small['item_id'].unique()
    items_small = items_full[items_full['item_id'].isin(active_item_ids)]

    # 6. 保存覆盖原文件
    df_train.to_csv(os.path.join(data_dir, "interactions_train.csv"), index=False)
    df_val.to_csv(os.path.join(data_dir, "interactions_val.csv"), index=False)
    df_small.to_csv(os.path.join(data_dir, "interactions.csv"), index=False)
    items_small.to_csv(os.path.join(data_dir, "items.csv"), index=False)

    print(f"✅ 数据已精简！")
    print(f"- 训练集: {len(df_train)} 行")
    print(f"- 验证集: {len(df_val)} 行")
    print(f"- 电影库: {len(items_small)} 部 (这将大幅加快 BERT 初始化速度)")


if __name__ == "__main__":
    downsample_data("D:\Working\VScode\Rec\data")