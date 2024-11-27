# %%
import numpy as np
import matplotlib.pyplot as plt
import random

def process_binary_file(extract_path, original_file_path):
    with open(extract_path, "w") as extract_file:
        with open(original_file_path, 'rb') as original_file:
            while True:
                line = original_file.readline()
                if not line:
                    break
                
                row_data = line.split()
                
                if int(row_data[0].decode('utf-8')) == 3 or int(row_data[0].decode('utf-8')) == 7:
                    print(str(line)[2:-3], file=extract_file)

process_binary_file("train_extract.txt", "mnist.scale")
# %%
from libsvm.svmutil import svm_train, svm_predict, svm_read_problem

train_y, train_x = svm_read_problem('mnist.scale')

c_list = [0.1, 1, 10]        # 正則化參數 C
Q_list = [2, 3, 4]        # 多項式的次數（這裡假設 Q=3）
coef0 = 1      # 偏置項，對應公式中的 "+1"
gamma = 1      # gamma 設為 1，對應於公式中的內積縮放因子

c_q_num = []

for c in c_list:
    for Q in Q_list:
        model = svm_train(train_y, train_x, f'-c {c} -s 0 -t 1 -d {Q} -r {coef0} -g {gamma}')

        # 獲取支持向量的數量
        num_support_vectors = len(model.get_SV())
        print(f'Number of support vectors: {num_support_vectors}')
        
        c_q_num.append([c, Q, num_support_vectors])
# %%
# 參數設定
from libsvm.svmutil import svm_train
c_list = [0.1, 1, 10]          # 正則化參數 C
gamma_list = [0.1, 1, 10]      # 高斯核的 gamma 值

# 用於存儲每組 (C, gamma) 的支持向量數和邊界大小
results = []

for c in c_list:
    for gamma in gamma_list:
        # 訓練模型
        model = svm_train(train_y, train_x, f'-c {c} -s 0 -t 2 -g {gamma}')
        
        # 獲取支持向量的數量
        num_support_vectors = len(model.get_SV())

        # 獲取支持向量的係數
        sv_coefs = model.get_sv_coef()

        # 計算 margin 大小，1 / ||w|| 使用支持向量的係數
        margin_size = 1 / sum(abs(coef[0]) for coef in sv_coefs)

        # 儲存結果 (C, gamma, 支持向量數, 邊界大小)
        results.append((c, gamma, num_support_vectors, margin_size))

        # 打印每組參數的結果
        print(f'C: {c}, gamma: {gamma}, 支持向量數量: {num_support_vectors}, 邊界大小: {margin_size}')

# 打印最終的表格
print("\n結果表格 (C, gamma, 支持向量數量, 邊界大小):")
for result in results:
    print(result)
    print(result)

# %%
