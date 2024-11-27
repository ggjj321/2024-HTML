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

# train_df_encoded = train_df[["home_team_abbr", "away_team_abbr", "home_team_win"]]
# train_df_encoded = pd.get_dummies(train_df_encoded, columns=["home_team_abbr", "away_team_abbr"])

print(train_df_encoded.shape)
# print(test_df_encoded.shape)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# 分離特徵和標籤
X = train_df_encoded.drop(columns=['home_team_win'])
y = train_df_encoded['home_team_win']

# 構建 L1 正則化的邏輯回歸模型
model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model.fit(X, y)

# 使用 SelectFromModel 選擇特徵
selector = SelectFromModel(model, prefit=True)
X_new = selector.transform(X)

# 獲取選中的特徵列名
selected_columns = X.columns[selector.get_support()]

# 將 NumPy 數組轉回 DataFrame
X_new_df = pd.DataFrame(X_new, columns=selected_columns)

# 查看結果
print(X_new_df.head())

# %%
# logstic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.metrics import accuracy_score, classification_report

# team_name_name_columns_list = list(df_encoded)[1:]

# X_train = train_df_encoded.drop(columns=['home_team_win'])
# y_train = train_df_encoded['home_team_win']
# X_test = test_df_encoded.drop(columns=['home_team_win'])
# y_test = test_df_encoded['home_team_win']
# X = train_df_encoded.drop(columns=['home_team_win'])
# y = train_df_encoded['home_team_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# 預測
from sklearn.model_selection import cross_val_score
import numpy as np

accuracy_list = []
max_accuracy = 0
max_c = -1
for c in range(200):
    print(c)
    c = c / 100000
    if c == 0:
        model = LogisticRegression()
    else:
        model = LogisticRegression(C=c)

    # 計算準確率
    # Using cross-validation with 5 folds
    scores = cross_val_score(model, X_new_df, y, cv=5, scoring='accuracy')
    accuracy = np.mean(scores)
    
    accuracy_list.append(accuracy)
    
    if accuracy > max_accuracy:
        max_c = c
        max_accuracy = accuracy
    
    print("Accuracy:", accuracy)


# %%
import matplotlib.pyplot as plt

print(f"best accuracy : {max_accuracy} best c : {max_c}")

print(len(accuracy_list))

print(max(accuracy_list))
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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

accuracy_list = []

for c in range(1, 200):
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
# generate data
test_df = pd.read_csv("/Users/wukeyang/ntu_course/2024-HTML/final_project/html-2024-fall-final-project-stage-1/test_df_aug.csv")
c = 0.00199
model = LogisticRegression(C=c)
model.fit(X_new_df, y)
# %%
# predict
# test_df_encoded = test_df[["home_team_abbr", "away_team_abbr"]]
# test_df_encoded = pd.get_dummies(test_df_encoded, columns=["home_team_abbr", "away_team_abbr"])

# test_df_encoded = date_one_hot_process(test_df)

# 獲取選中的特徵列名
selected_columns = X.columns[selector.get_support()]

# 將 NumPy 數組轉回 DataFrame
test_df_new_df = pd.DataFrame(test_df_encoded, columns=selected_columns)
# print(test_df_encoded.shape)
# y_submmit = model.predict(test_df_encoded)

# %%
import csv

y_submmit = model.predict(test_df_new_df)
# 假設有一個列表
data_list = y_submmit

# 指定輸出的 CSV 文件名稱
output_file = 'output/logstic feature selection.csv'

# 打開文件並寫入內容
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 寫入標題行（可選）
    writer.writerow(['id', 'home_team_win'])
    # 寫入數據
    for idx, value in enumerate(data_list):
        writer.writerow([idx, value])

print(f"CSV file '{output_file}' generated successfully.")

# %%
