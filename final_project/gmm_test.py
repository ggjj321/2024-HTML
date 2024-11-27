import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, accuracy_score

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

# 1. 使用 Pandas 讀取數據
df = pd.read_csv('html-2024-fall-final-project-stage-1/training_df_aug.csv')

train_df_encoded = date_one_hot_process(df)

# 查看數據
print(df.head())

X = train_df_encoded.drop(columns=['home_team_win'])
y_true = train_df_encoded['home_team_win']

# 2. 使用 GMM 模型進行分類
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

# 預測類別
y_pred = gmm.predict(X)

# 3. 評估結果
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Accuracy Score:", accuracy_score(y_true, y_pred))

# 可視化分類結果
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.title("GMM Classification Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
