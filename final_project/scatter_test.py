# 安裝需要的庫
# pip install numpy pandas matplotlib scikit-learn umap-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# 假設 data 是你的原始數據框
train_df = pd.read_csv("html-2024-fall-final-project-stage-1/training_df_aug.csv")

def date_one_hot_process(df):
    df['date'] = pd.to_datetime(df['date'])

    # 提取日期特徵
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

    # 移除原始的 date 欄位
    df = df.drop(columns=['date', 'home_team_season', 'away_team_season','home_pitcher', 'away_pitcher', "id", "year"])

    df_encoded = pd.get_dummies(df, columns=["home_team_abbr", "away_team_abbr"])
    
    return df_encoded

train_df_encoded = date_one_hot_process(train_df)

X = train_df_encoded.drop(columns=['home_team_win'])
y = train_df_encoded['home_team_win']

# 1. 數據標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

# 使用 t-SNE 進行降維
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(data_scaled)

# 使用 UMAP 進行降維
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(data_scaled)

# 繪製 t-SNE 結果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, alpha=0.5, s=10, cmap='viridis')
plt.title('t-SNE Result')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# 繪製 UMAP 結果
plt.subplot(1, 2, 2)
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=y, alpha=0.5, s=10, cmap='viridis')
plt.title('UMAP Result')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.tight_layout()
plt.show()


