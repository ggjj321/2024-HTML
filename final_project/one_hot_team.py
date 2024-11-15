#%%
import pandas as pd

# 讀取 CSV 檔案

train_df = pd.read_csv("html-2024-fall-final-project-stage-1/training_df_aug.csv")
test_df = pd.read_csv("html-2024-fall-final-project-stage-1/test_df_aug.csv")

print(test_df.shape)
print(train_df.shape)
# %%
# 隊名 one hot
# only_team_name_df = df[["home_team_abbr", "away_team_abbr", "home_team_win"]]
# 將 date 欄位轉換為 datetime 型別
# 將 date 欄位轉換為 datetime 型別
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
test_df_encoded = date_one_hot_process(test_df)

print(train_df_encoded.shape)
print(test_df_encoded.shape)

# %%
# logstic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# team_name_name_columns_list = list(df_encoded)[1:]

# X_train = train_df_encoded.drop(columns=['home_team_win'])
# y_train = train_df_encoded['home_team_win']
# X_test = test_df_encoded.drop(columns=['home_team_win'])
# y_test = test_df_encoded['home_team_win']
X = train_df_encoded.drop(columns=['home_team_win'])
y = train_df_encoded['home_team_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# 預測
# accuracy_list = []
for c in range(1000):
    print(c)
    c+=10000
    c = c / 100000
    if c == 0:
        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # 計算準確率
        accuracy = accuracy_score(y_test, y_pred)
    else:
        model = LogisticRegression(penalty='l2', C=c)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # 計算準確率
        accuracy = accuracy_score(y_test, y_pred)
    
    accuracy_list.append(accuracy)
    
    print("Accuracy:", accuracy)

print(f"best accuracy : {max(accuracy_list)}")

# %%
import matplotlib.pyplot as plt
import numpy as np

print(len(accuracy_list))

print(max(accuracy_list))
# x = np.arange(0.00001, 0.1 + 0.00001, 0.00001)  

# # 繪製圖表
# plt.plot(x ,accuracy_list)
# plt.xlim(0.00001, 0.1)

# # 添加標題和標籤
# plt.title("Sample Plot")
# plt.xlabel("constraint")
# plt.ylabel("accuracy")

# # 顯示圖表
# plt.show()
# %%
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

accuracy_list = []

for c in range(1, 3000):
    print(c)
    c = c / 100000  # 計算 C 值
    
    model = SGDClassifier(loss='log_loss', penalty='l2', alpha=c, random_state=42)
    
    # 訓練模型
    model.fit(X_train, y_train)
    
    # 預測
    y_pred = model.predict(X_test)
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    print("Accuracy:", accuracy)
print(f"best accuracy : {max(accuracy_list)}")
# %%
# try feature aug
import pandas as pd
import numpy as np
from feature_aug import feature_aug
df = pd.read_csv("html-2024-fall-final-project-stage-1/train_data.csv")
aug_df = pd.read_csv("html-2024-fall-final-project-stage-1/training_df_aug.csv")

print(df.shape[1])
print(aug_df.shape[1])

# %%
