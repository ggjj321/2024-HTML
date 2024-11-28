import csv
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

def date_one_hot_process(df):
    df['date'] = pd.to_datetime(df['date'])

    # 提取日期特徵
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    
    label_encoders = {}
    
    for column in ["home_team_abbr", "away_team_abbr", 'home_pitcher', 'away_pitcher', 'home_team_season', 'away_team_season']:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
        
    df = df.drop(columns=['date', "year"])
    
    return df

train_df = pd.read_csv("html-2024-fall-final-project-stage-1/training_df_aug.csv")
train_df_encoded = date_one_hot_process(train_df)

X = train_df_encoded.drop(columns=['home_team_win'])
y = train_df_encoded['home_team_win']

# selector = SelectKBest(score_func=f_classif, k=180)
# X_selected = selector.fit_transform(X, y)

d_train = lgb.Dataset(X, label=y)

# 設定參數
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 訓練模型
model = lgb.train(params, d_train, num_boost_round=100)

test_df = pd.read_csv("/Users/wukeyang/ntu_course/2024-HTML/final_project/html-2024-fall-final-project-stage-1/test_df_aug.csv")
test_df_encoded = date_one_hot_process(test_df)

# test_df_select = selector.transform(test_df_encoded)

y_pred_prob = model.predict(test_df_encoded, num_iteration=model.best_iteration)
y_submmit = (y_pred_prob >= 0.5).astype(bool) 

# 假設有一個列表
data_list = y_submmit

# 指定輸出的 CSV 文件名稱
output_file = 'output/lightgbm new encode.csv'

# 打開文件並寫入內容
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 寫入標題行（可選）
    writer.writerow(['id', 'home_team_win'])
    # 寫入數據
    for idx, value in enumerate(data_list):
        writer.writerow([idx, value])

print(f"CSV file '{output_file}' generated successfully.")

