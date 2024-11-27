# %%
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
# %%
train_df = pd.read_csv("html-2024-fall-final-project-stage-1/training_df_aug.csv")
print(train_df.shape)
# %%
# def date_one_hot_process(df):
#     df['date'] = pd.to_datetime(df['date'])

#     # 提取日期特徵
#     df['year'] = df['date'].dt.year
#     df['month'] = df['date'].dt.month
#     df['day'] = df['date'].dt.day
#     df['weekday'] = df['date'].dt.weekday

#     # 移除原始的 date 欄位
#     df = df.drop(columns=['date', 'home_team_season', 'away_team_season','home_pitcher', 'away_pitcher', "id", "year"])

#     df_encoded = pd.get_dummies(df, columns=["home_team_abbr", "away_team_abbr"])
    
#     return df_encoded

# train_df_encoded = date_one_hot_process(train_df)
# X = train_df_encoded.drop(columns=['home_team_win'])
# y = train_df_encoded['home_team_win']

train_df_encoded = train_df[["home_team_abbr", "away_team_abbr", "home_team_win"]]
train_df_encoded = pd.get_dummies(train_df_encoded, columns=["home_team_abbr", "away_team_abbr"])

X = train_df_encoded.drop(columns=['home_team_win'])
y = train_df_encoded['home_team_win']
# %%
# 3. Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %%
# 4. Train an SVM classifier with cross-validation
c_list = [10, 1, 0.1, 0.01]
best_c = -1
best_acc = 0

for c in c_list:
    svm_classifier = SVC(kernel='linear', C=c, random_state=42)
    scores = cross_val_score(svm_classifier, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.2f}")
    
    if np.mean(scores) > best_acc:
        best_acc = np.mean(scores)
        best_c = c
print(f"best acc : {best_acc}, best c : {best_c}")

# %%
svm_classifier = SVC(kernel='linear', C=0.01, random_state=42)
svm_classifier.fit(X, y)

#%%
from sklearn.metrics import accuracy_score, classification_report

y_pred = svm_classifier.predict(X)
train_accuracy = accuracy_score(y, y_pred)
print(f"Training Set Accuracy (Ein): {train_accuracy:.2f}")
#%%
test_df = pd.read_csv("/Users/wukeyang/ntu_course/2024-HTML/final_project/html-2024-fall-final-project-stage-1/test_df_aug.csv")
x_test = scaler.fit_transform(date_one_hot_process(test_df))
y_submmit = svm_classifier.predict(x_test)
import csv

# 假設有一個列表
data_list = y_submmit

# 指定輸出的 CSV 文件名稱
output_file = 'output/svm stage1 cross val with aug.csv'

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
