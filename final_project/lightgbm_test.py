import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# 特徵選擇 - 選擇最佳的 5 個特徵
# selector = SelectKBest(score_func=f_classif, k=180)
# X_selected = selector.fit_transform(X, y)

# print(X_selected)

# # 查看每個特徵對應的分數
# feature_scores = selector.scores_
# feature_pvalues = selector.pvalues_
# feature_names = X.columns

# # for name, score, pval in zip(feature_names, feature_scores, feature_pvalues):
# #     print(f'Feature: {name}, Score: {score:.2f}, P-value: {pval:.4f}')

# # 查看分數最高的前 10 個特徵
# top_features = sorted(zip(feature_names, feature_scores), key=lambda x: x[1], reverse=True)
# print("\nTop 5 Features by Score:")
# for name, score in top_features[:10]:
#     print(f'Feature: {name}, Score: {score:.2f}')

# # 額外排序包含 "batting" 的特徵
# batting_features = [f for f in top_features if 'batting' in f[0]]
# batting_features_sorted = sorted(batting_features, key=lambda x: x[1], reverse=True)
# print("\nTop Features with 'batting':")
# for name, score in batting_features_sorted[:10]:
#     print(f'Feature: {name}, Score: {score:.2f}')

# # 額外排序包含 "pitching" 的特徵
# pitching_features = [f for f in top_features if 'pitching' in f[0]]
# pitching_features_sorted = sorted(pitching_features, key=lambda x: x[1], reverse=True)
# print("\nTop Features with 'pitching':")
# for name, score in batting_features_sorted[:10]:
#     print(f'Feature: {name}, Score: {score:.2f}')
    
# pca = PCA(n_components=1)

# batt_selected_feature_columns = [name for name, score in batting_features_sorted[:10]]
# batt_df = pd.DataFrame(X[batt_selected_feature_columns])

# pitch_selected_feature_columns = [name for name, score in pitching_features_sorted[:10]]
# pitch_df = pd.DataFrame(X[pitch_selected_feature_columns])

# batt_pca = pca.fit_transform(batt_df)
# pitch_pca = pca.fit_transform(pitch_df)

# plt.scatter(batt_pca, pitch_pca, c=y, cmap='bwr', alpha=0.5, s=10)
# plt.xlabel('batt')
# plt.ylabel('pitch')
# plt.title('Scatter Plot of batt_pca vs pitch_pca (Colored by home_team_win)')
# plt.grid(True)
# plt.show()

# selected_feature_columns = [name for name, score in top_features]
# X_selected = pd.DataFrame(X[selected_feature_columns])

# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# accuracies = []
# roc_aucs = []

# for train_index, test_index in kf.split(X_selected, y):
#     X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#     # 建立 LightGBM 資料集
#     d_train = lgb.Dataset(X_train, label=y_train)
#     d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)

#     # 設定參數
#     params = {
#         'objective': 'binary',  # 二元分類
#         'metric': 'binary_error',  # 評估指標: 二元分類錯誤率
#         'boosting_type': 'gbdt',
#         'num_leaves': 31,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.9
#     }
    
#     # 訓練模型
#     num_round = 100
#     bst = lgb.train(params, d_train, num_round, valid_sets=[d_test])

#     # 使用測試集進行預測
#     y_pred_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
#     y_pred = (y_pred_prob >= 0.5).astype(int)  # 將預測機率轉為 0 或 1

#     # 評估模型效果
#     accuracy = accuracy_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred_prob)

#     accuracies.append(accuracy)
#     roc_aucs.append(roc_auc)

# # 計算平均準確率和 ROC AUC
# mean_accuracy = np.mean(accuracies)
# mean_roc_auc = np.mean(roc_aucs)

# print(f'Mean Accuracy: {mean_accuracy}')
# print(f'Mean ROC AUC: {mean_roc_auc}')
    

# for feacture_id in range(4):
#     for next_feacture_id in range(feacture_id + 1, 5):
#         plt.scatter(train_df[top_features[feacture_id][0]], train_df[top_features[next_feacture_id][0]], c=y, cmap='bwr', alpha=0.5)
#         plt.xlabel('pitching_SO_advantage')
#         plt.ylabel('pitching_ERA_advantage')
#         plt.title('Scatter Plot of pitching_SO_advantage vs pitching_ERA_advantage (Colored by home_team_win)')
#         plt.grid(True)
#         plt.show()

accuracy_list = []

for feature_num in range(5, 275, 5):
    # 特徵選擇 - 選擇最佳的 5 個特徵
    selector = SelectKBest(score_func=f_classif, k=feature_num)
    X_selected = selector.fit_transform(X, y)

    # 使用 StratifiedKFold 進行交叉驗證
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    roc_aucs = []

    for train_index, test_index in kf.split(X_selected, y):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 建立 LightGBM 資料集
        d_train = lgb.Dataset(X_train, label=y_train)
        d_test = lgb.Dataset(X_test, label=y_test, reference=d_train)

        # 設定參數
        params = {
            'objective': 'binary',  # 二元分類
            'metric': 'binary_error',  # 評估指標: 二元分類錯誤率
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        # 訓練模型
        num_round = 100
        bst = lgb.train(params, d_train, num_round, valid_sets=[d_test])

        # 使用測試集進行預測
        y_pred_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
        y_pred = (y_pred_prob >= 0.5).astype(int)  # 將預測機率轉為 0 或 1

        # 評估模型效果
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)

    # 計算平均準確率和 ROC AUC
    mean_accuracy = np.mean(accuracies)
    mean_roc_auc = np.mean(roc_aucs)

    print(f'Mean Accuracy: {mean_accuracy:.2f}')
    print(f'Mean ROC AUC: {mean_roc_auc:.2f}')
    
    accuracy_list.append(mean_accuracy)

# 繪製 accuracy list 對應 range(5, 200, 5)
k_values = list(range(5, 275, 5))

print(max(accuracy_list))

plt.plot(k_values, accuracy_list, marker='o')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K Value')
plt.grid(True)
plt.show()

