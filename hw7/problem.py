#%%
from libsvm.svmutil import svm_train, svm_predict, svm_read_problem
import pandas as pd
import numpy as np

train_y, train_x = svm_read_problem('madelon')

x_df = pd.DataFrame(train_x)
x_arr = x_df.values
y_arr = np.array(train_y)
# %%
import numpy as np

def decision_stump_multi_feature(X, y, weights):
    """
    多維度的 Decision Stump（決策樹樁）實作，利用動態規劃（DP）找出
    最適的單一特徵、方向 (sign)、閥值 (threshold)，以及對應的
    0/1 錯誤率 (E_in) 與加權錯誤率 (epsilon_t)。
    
    參數：
    X : ndarray, shape = (N, d)
        所有樣本的特徵。N 筆資料、d 個特徵維度。
    y : ndarray, shape = (N,)
        標籤（+1 或 -1）。
    weights : ndarray, shape = (N,)
        每筆樣本的權重，用於計算加權錯誤率。
    
    回傳：
    best_stump : dict
        包含最好的 (feature, s, theta, E_in, epsilon_t)
        feature   : 最佳特徵維度索引
        s         : 最佳方向 (sign) 為 +1 或 -1
        theta     : 最佳的閥值
        E_in      : 該閥值下的 0/1 錯誤率
        epsilon_t : 該閥值下的加權錯誤率
    """
    N, d = X.shape
    best_stump = {
        'feature': None,
        's': None,
        'theta': None,
        'E_in': float('inf'),      # 0/1 錯誤率
        'weighted_error': float('inf')  # 加權錯誤率
    }
    
    # 遍歷所有特徵
    for feature in range(d):
        feature_values = X[:, feature]
        
        # 依照該特徵值進行排序（方便動態調整閥值）
        sorted_indices = np.argsort(feature_values)
        X_sorted = feature_values[sorted_indices]
        y_sorted = y[sorted_indices]
        weights_sorted = weights[sorted_indices]
        
        # 計算候選閥值：在連續兩個不同特徵值的中點插入一個 threshold
        thresholds = []
        for i in range(N - 1):
            if X_sorted[i] != X_sorted[i + 1]:
                thresholds.append((X_sorted[i] + X_sorted[i + 1]) / 2)
        # 一般也包含 -∞ 與 +∞ 作為邊界
        thresholds = [-float('inf')] + thresholds + [float('inf')]
        
        # 對每個方向 (sign)，分別找出最佳閥值
        for s in [-1, 1]:
            # 初始化：先用 thresholds[0] 計算一次整體的加權錯誤和 0/1 錯誤
            predictions = s * np.sign(X_sorted - thresholds[0])
            
            # np.sign(0) == 0，為避免歸類成 0，需要自行指定成 s
            predictions[predictions == 0] = s
            
            weighted_error = np.sum(weights_sorted[y_sorted != predictions]) 
            unweighted_error = np.sum(y_sorted != predictions)  # 計數方式，後面再除以N
            
            min_error = weighted_error
            min_unweighted_error = unweighted_error
            min_theta = thresholds[0]
            
            # 動態規劃：假設閥值依序往後移，逐筆更新錯誤率
            for i in range(1, len(thresholds)):
                # 當閥值從 thresholds[i-1] 移動到 thresholds[i] 時
                # 只有一個樣本 (i-1) 會改變預測結果
                x_i = X_sorted[i - 1]
                y_i = y_sorted[i - 1]
                
                old_prediction = s * np.sign(x_i - thresholds[i - 1])
                new_prediction = s * np.sign(x_i - thresholds[i])
                
                # 只有在“分界改變後”該筆樣本的預測會被翻轉
                if old_prediction != new_prediction:
                    if new_prediction != y_i:
                        # 本來是正確，現在變錯誤
                        weighted_error += weights_sorted[i - 1]
                        unweighted_error += 1
                    else:
                        # 本來是錯誤，現在變正確
                        weighted_error -= weights_sorted[i - 1]
                        unweighted_error -= 1
                
                # 若現在的 weighted_error 更小，就更新最小值
                if weighted_error  < min_error:
                    min_error = weighted_error 
                    min_unweighted_error = unweighted_error
                    min_theta = thresholds[i]
            
            # 檢查這個 feature、這個 sign 是否比當前的全域最佳更好
            if min_error < best_stump['weighted_error']:
                best_stump = {
                    'feature': feature,
                    's': s,
                    'theta': min_theta,
                    'E_in': min_unweighted_error / N,  # 0/1 錯誤率記得除以 N
                    'weighted_error': min_error   # 加權錯誤率
                }
    
    return best_stump

# %%
import numpy as np

def decision_stump_multi_feature_no_dp(X, y, weights):
    """
    多維度的 Decision Stump（決策樹樁）實作，不用 DP，
    而是直接對每個特徵的所有候選閥值逐一檢查。
    
    參數：
    X : ndarray, shape = (N, d)
        所有樣本的特徵。N 筆資料、d 個特徵維度。
    y : ndarray, shape = (N,)
        標籤（+1 或 -1）。
    weights : ndarray, shape = (N,)
        每筆樣本的權重，用於計算加權錯誤率。
    
    回傳：
    best_stump : dict
        包含最佳 (feature, s, theta, E_in, epsilon_t)
        feature   : 最佳特徵維度索引
        s         : 最佳方向 (sign) 為 +1 或 -1
        theta     : 最佳的閥值
        E_in      : 該閥值下的 0/1 錯誤率
        epsilon_t : 該閥值下的加權錯誤率
    """
    N, d = X.shape
    best_stump = {
        'feature': None,
        's': None,
        'theta': None,
        'E_in': float('inf'),      # 0/1 錯誤率
        'epsilon_t': float('inf')  # 加權錯誤率
    }
    
    # 逐一遍歷每個特徵
    for feature in range(d):
        # 取出該特徵的所有值，排序是為了取得candidate thresholds
        feature_values = X[:, feature]
        sorted_indices = np.argsort(feature_values)
        X_sorted = feature_values[sorted_indices]
        y_sorted = y[sorted_indices]
        weights_sorted = weights[sorted_indices]
        
        # 計算候選閥值：在連續兩個不同特徵值的中點插入一個 threshold
        thresholds = []
        for i in range(N - 1):
            if X_sorted[i] != X_sorted[i + 1]:
                thresholds.append((X_sorted[i] + X_sorted[i + 1]) / 2)
        # 通常也會考慮 -∞ 與 +∞ 作為邊界候選
        thresholds = [-float('inf')] + thresholds + [float('inf')]
        
        # 對每個 sign (方向) 做搜尋
        for s in [-1, 1]:
            # 用簡單迴圈直接檢查所有 thresholds
            min_error = float('inf')
            min_unweighted_error = float('inf')
            min_theta = None
            
            for thresh in thresholds:
                # 根據 thresh 做分類預測
                predictions = s * np.sign(X_sorted - thresh)
                # 處理邊界情況：np.sign(0) == 0，需要指定為 s
                predictions[predictions == 0] = s
                
                # 計算該 thresh 的加權錯誤率和 0/1 錯誤率
                weighted_error = np.sum(weights_sorted[y_sorted != predictions])
                unweighted_error = np.sum(y_sorted != predictions)
                
                # 若該 threshold 的加權錯誤更小，就更新當前最優
                if weighted_error < min_error:
                    min_error = weighted_error
                    min_unweighted_error = unweighted_error
                    min_theta = thresh
            
            # 比對全域最佳
            if min_error < best_stump['epsilon_t']:
                best_stump = {
                    'feature': feature,
                    's': s,
                    'theta': min_theta,
                    'E_in': min_unweighted_error / N,  # 0/1 錯誤率
                    'epsilon_t': min_error
                }
    
    return best_stump

# %%
def adaboost(X, y, T):
    """
    AdaBoost implementation with multi-dimensional Decision Stump as weak classifier.
    
    Parameters:
    X : ndarray of shape (N, d), feature values for all samples
    y : ndarray of shape (N,), labels for all samples (+1 or -1)
    T : int, number of boosting rounds
    
    Returns:
    classifiers : list of (feature, s, theta, alpha) for each weak classifier
    """
    N, d = X.shape
    weights = np.ones(N) / N  # Initialize uniform weights
    U_list = []
    classifiers = []
    
    ein_list = []
    epsilon_list = []

    for t in range(T):
        print(t)
        # Find the best decision stump across all features
        stump = decision_stump_multi_feature_no_dp(X, y, weights)
        e_in = stump['E_in']
        weighted_error= stump['epsilon_t']
        
        epsilon = weighted_error / np.sum(weights)
        
        ein_list.append(e_in)
        epsilon_list.append(epsilon)
        
        # Compute alpha for the weak classifier
        if epsilon == 0:  # Avoid division by zero
            alpha = float('inf')
        else:
            alpha = np.log(((1 - epsilon) / epsilon) ** 0.5)
        
        # Update sample weights
        feature_values = X[:, stump['feature']]
        predictions = stump['s'] * np.sign(feature_values - stump['theta'])
        predictions[predictions == 0] = stump['s']
        weights *= np.exp(-alpha * y * predictions)
        
        U_list.append(np.sum(weights))
        
        # weights_acc.append(np.sum(weights_acc[-1]) + np.sum(weights))
        
        # Store the weak classifier
        classifiers.append({
            'feature': stump['feature'],
            's': stump['s'],
            'theta': stump['theta'],
            'alpha': alpha
        })
    
    return classifiers, ein_list, epsilon_list, U_list

# %%
classifiers, ein_list, epsilon_list, weights_acc = adaboost(x_arr, y_arr, 500)
# %%
import matplotlib.pyplot as plt
T = 500

plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), ein_list, label="Ein 0/1 error", color="blue")
plt.plot(range(1, T + 1), epsilon_list, label="ϵt", color="red", linestyle="--")
plt.xlabel("Iteration (t)")
plt.ylabel("Error")
plt.title("E_in(g_t) and Normalized Error (ε_t) vs. Iterations")
plt.legend()
plt.grid()
plt.show()
# %%
for classifier in classifiers:
    print(classifier)
# %%
def compute_strong_classifier(X, y, classifiers):
    """
    Compute the strong classifier G(x) and its Ein.
    
    Parameters:
    X : ndarray of shape (N, d), feature values for all samples
    y : ndarray of shape (N,), labels for all samples (+1 or -1)
    classifiers : list of dict, each containing the parameters of a weak classifier
                  [{'feature': ..., 's': ..., 'theta': ..., 'alpha': ...}, ...]
    
    Returns:
    G_predictions : ndarray of shape (N,), the predictions of the strong classifier (+1 or -1)
    Ein_G : float, the 0/1 error of the strong classifier
    """
    N = X.shape[0]
    # 初始化強分類器的加權和
    strong_sum = np.zeros(N)
    
    e_list = []
    
    # 遍歷每個弱分類器
    for classifier in classifiers:
        feature = classifier['feature']
        s = classifier['s']
        theta = classifier['theta']
        alpha = classifier['alpha']
        
        # 計算弱分類器的預測
        predictions = s * np.sign(X[:, feature] - theta)
        predictions[predictions == 0] = s  # 處理邊界
        
        # 加權累加
        strong_sum += alpha * predictions
    
        # 計算強分類器的最終預測
        G_predictions = np.sign(strong_sum)
        
        # 計算 Ein(G)
        E = np.sum(G_predictions != y) / N  # 0/1 錯誤率
        
        e_list.append(E)
    
    return G_predictions, e_list

ein_predictions, ein_list = compute_strong_classifier(x_arr, y_arr, classifiers)

test_y, test_x = svm_read_problem('madelon.t')

test_x_df = pd.DataFrame(test_x)
test_x_arr = test_x_df.values
test_y_arr = np.array(test_y)
eout_predictions, eout_list = compute_strong_classifier(test_x_arr, test_y_arr, classifiers)
# %%
plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), ein_list, label="Ein of G(t)", color="blue")
plt.plot(range(1, T + 1), eout_list, label="Eout of G(t)", color="red", linestyle="--")
plt.xlabel("Iteration (t)")
plt.ylabel("Error")
plt.title("Ein of G(t) and Eout of G(t) vs. Iterations")
plt.legend()
plt.grid()
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), ein_list, label="Ein of G(t)", color="blue")
plt.plot(range(1, T + 1), weights_acc, label="Ut", color="red", linestyle="--")
plt.xlabel("Iteration (t)")
plt.ylabel("Error and Sum of weight")
plt.title("Ein of G(t) and Ut vs. Iterations")
plt.legend()
plt.grid()
plt.show()
# %%
