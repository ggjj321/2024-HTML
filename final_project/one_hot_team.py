#%%
import pandas as pd

# 讀取 CSV 檔案
df = pd.read_csv("html-2024-fall-final-project-stage-1/train_data.csv")
all_home_team = set(df["home_team_abbr"].values)
all_away_tem = set(df["away_team_abbr"].values)
print(f"home team nums : {len(all_home_team)}")
print(f"away team nums : {len(all_away_tem)}")
# %%
# 隊名 one hot
only_team_name_df = df[["home_team_abbr", "away_team_abbr", "home_team_win"]]
df_encoded = pd.get_dummies(only_team_name_df, columns=["home_team_abbr", "away_team_abbr"])
print(df_encoded)
# %%
# logstic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

team_name_name_columns_list = list(df_encoded)[1:]
X = df_encoded[team_name_name_columns_list]
y = df_encoded['home_team_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# 預測
accuracy_list = []
for c in range(2000):
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
x = np.arange(0.00001, 0.02 + 0.00001, 0.00001)  

# 繪製圖表
plt.plot(x ,accuracy_list)
plt.xlim(0.00001, 0.02)

# 添加標題和標籤
plt.title("Sample Plot")
plt.xlabel("constraint")
plt.ylabel("accuracy")

# 顯示圖表
plt.show()
# %%
