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

train_y, train_x = svm_read_problem('train_extract.txt')

c_list = [0.1, 1, 10]        # 正則化參數 C
Q_list = [2, 3, 4]        # 多項式的次數（這裡假設 Q=3）
coef0 = 1      # 偏置項，對應公式中的 "+1"
gamma = 1      # gamma 設為 1，對應於公式中的內積縮放因子

c_q_num = []

file_path = 'p10_num_support_vectors.txt'

with open(file_path, 'w', encoding='utf-8') as file:
    for c in c_list:
        for Q in Q_list:
            model = svm_train(train_y, train_x, f'-c {c} -s 0 -t 1 -d {Q} -r {coef0} -g {gamma}')

            # 獲取支持向量的數量
            num_support_vectors = len(model.get_SV())
            print(f'Number of support vectors: {num_support_vectors}')
            
            c_q_num.append([c, Q, num_support_vectors])

            output_text = f"c : {c} Q : {Q} num_support_vectors : {num_support_vectors}"

            file.write(output_text + '\n')
# %%
# 參數設定
import concurrent.futures
from libsvm.svmutil import svm_train, svm_read_problem
import numpy as np

c_list = [1, 10]          # 正则化参数 C
gamma_list = [0.1, 1, 10]      # 高斯核的 gamma 值

results = []
file_path = 'p11_fix_right.txt'

def train_and_evaluate(c, gamma):
    # 训练模型
    model = svm_train(train_y, train_x, f'-c {c} -s 0 -t 2 -g {gamma}')
    
    sv = model.get_SV()
    # sv_indices = model.get_sv_indices()  
    sv_coefs = model.get_sv_coef()  # alpha * y
    num_support_vectors = len(sv)

    max_index = max([max(sv_i.keys()) for sv_i in sv])
    N = num_support_vectors
    D = max_index
    sv_array = np.zeros((N, D))
    for i, sv_i in enumerate(sv):
        for idx, value in sv_i.items():
            sv_array[i, idx - 1] = value
    # labels = [train_y[i - 1] for i in sv_indices]

    def gaussian_kernel(x1, x2, gamma):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel(sv_array[i], sv_array[j], gamma)
    
    alpha = np.array([coef[0] for coef in sv_coefs])  # alpha_i y_i 
    # alpha_y = alpha * np.array(labels)                # caculate alpha_i * y_i
    w_norm_squared = np.sum(alpha[:, None] * alpha[None, :] * K)

    w_norm = np.sqrt(w_norm_squared)

    print(f"norm {w_norm}")

    if w_norm == 0:
        margin_size = float('inf')
    else:
        margin_size = 1 / w_norm

    output_text = f'C: {c}, gamma: {gamma}, 支持向量数量: {num_support_vectors}, 边界大小: {margin_size}'
    results.append(output_text)
    print(output_text)
    return c, gamma, num_support_vectors, margin_size, output_text

# 主程序
if __name__ == "__main__":
    # 读取数据
    train_y, train_x = svm_read_problem('train_extract.txt')

    # 组合参数
    param_combinations = [(c, gamma) for c in c_list for gamma in gamma_list]

    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        # 提交任务
        futures = [executor.submit(train_and_evaluate, c, gamma) for c, gamma in param_combinations]

        # 收集结果
        with open(file_path, 'w', encoding='utf-8') as file:
            for future in concurrent.futures.as_completed(futures):
                c, gamma, num_support_vectors, margin_size, output_text = future.result()
                results.append((c, gamma, num_support_vectors, margin_size))
                file.write(output_text + '\n')
                file.flush()

    # 打印最终的表格
    print("\n结果表格 (C, gamma, 支持向量数量, 边界大小):")
    for result in results:
        print(result)

# %%
from libsvm.svmutil import svm_train, svm_predict, svm_read_problem
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_y, train_x = svm_read_problem('train_extract.txt')

gamma_list = [0.01, 0.1, 1, 10, 100]  # 高斯核的 gamma 值
C = 1  # 固定 C = 1
num_iterations = 128  # 重複 128 次
selection_counts = {gamma: 0 for gamma in gamma_list}  # 記錄每個 gamma 被選擇的次數

N = len(train_x)  # 總資料數量

for iteration in range(num_iterations):
    split_train_x, val_x, split_train_y, val_y = train_test_split(
        train_x, train_y, test_size=200, random_state=iteration
    )
    print(f"iteration : {iteration}")
    gamma_errors = {}  # 儲存每個 gamma 的驗證錯誤率
    
     # 定義訓練和評估的函數
    def train_and_evaluate(gamma):
        # 訓練模型
        param_str = f'-c {C} -s 0 -t 2 -g {gamma} -q'  # -q 靜音模式
        model = svm_train(split_train_y, split_train_x, param_str)
        
        # 在驗證集上評估
        p_label, p_acc, p_val = svm_predict(val_y, val_x, model, '-q')
        # 計算錯誤率
        error_rate = 100 - p_acc[0]
        print(gamma, error_rate)
        return gamma, error_rate

    for gamma in gamma_list:
        gamma_errors[gamma] = train_and_evaluate(gamma)
    
    # 選擇具有最小驗證錯誤率的 gamma
    min_error = min(gamma_errors.values())
    # 找到所有錯誤率最小的 gamma
    candidate_gammas = [gamma for gamma in gamma_list if gamma_errors[gamma] == min_error]
    # 在平局的情況下選擇最小的 gamma
    selected_gamma = min(candidate_gammas)
    # 更新選擇次數
    selection_counts[selected_gamma] += 1
# %%
gammas = sorted(selection_counts.keys())
frequencies = [selection_counts[gamma] for gamma in gammas]

plt.bar([str(gamma) for gamma in gammas], frequencies)
plt.xlabel('Gamma value')
plt.ylabel('selected num')
plt.title('selected num ')
plt.show()

# 輸出結果
print("Gamma 值與其被選擇次數:")
for gamma in gammas:
    print(f'Gamma = {gamma}, 被選擇次數 = {selection_counts[gamma]}')
# %%
