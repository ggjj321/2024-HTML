#%%
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
                
                if int(row_data[0].decode('utf-8')) == 2 or int(row_data[0].decode('utf-8')) == 6:
                    print(str(line)[2:-3], file=extract_file)

process_binary_file("train_extract.txt", "mnist.scale")
process_binary_file("test_extract.txt", "mnist.scale.t")
# %%
import threading
from concurrent.futures import ThreadPoolExecutor
from liblinear.liblinearutil import *
λ_list = [-2, -1, 0, 1, 2, 3]

train_y, train_x = svm_read_problem("train_extract.txt")
test_y, test_x = svm_read_problem("test_extract.txt")

eout = []
non_zero_num = []

eout_lock = threading.Lock()
non_zero_num_lock = threading.Lock()

def run_experiment(_):
    each_model = []
    ein_list = []

    for λ in λ_list:
        c = 1 / (10 ** λ)
        m = train(train_y, train_x, f'-c {c} -s 6 -B 1')
        p_label, p_acc, p_val = predict(train_y, train_x, m)

        ein = 1 - p_acc[0] / 100
        ein_list.append(ein)
        each_model.append(m)

    min_index = ein_list.index(min(ein_list))
    best_model = each_model[min_index]

    weights = best_model.get_decfun()[0]
    non_zero_weights_count = sum(1 for w in weights if w != 0)

    p_out_label, p_out_acc, p_out_val = predict(test_y, test_x, best_model)
    e_out_value = 1 - p_out_acc[0] / 100

    with non_zero_num_lock:
        non_zero_num.append(non_zero_weights_count)

    with eout_lock:
        eout.append(e_out_value)
        
num_threads = 8  
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(run_experiment, _) for _ in range(1126)]

for future in futures:
    future.result()
    

# %%
import matplotlib.pyplot as plt
plt.hist(eout, bins=5, edgecolor='black') 


plt.title("Histogram of Eout")
plt.xlabel("Value")
plt.ylabel("Frequency")


plt.show()
# %%
plt.hist(non_zero_num, bins=5, edgecolor='black') 

plt.title("Histogram of non zero number")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()
# %%
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from liblinear.liblinearutil import *

val_eout = []

eout_val_lock = threading.Lock()

def run_val_experiment(train_y, train_x, test_y, test_x):
    train_data = list(zip(train_y, train_x))
    np.random.shuffle(train_data)
    
    shuffled_y, shuffled_x = zip(*train_data)
    
    train_y = list(shuffled_y[:8000])
    train_x = list(shuffled_x[:8000])
    val_y = list(shuffled_y[8000:])
    val_x = list(shuffled_x[8000:])
    
    each_model = []
    eval_list = []

    # 遍历每个 λ 值
    for λ in λ_list:
        c = 1 / (10 ** λ)
        m = train(train_y, train_x, f'-c {c} -s 6 -B 1')
        p_label, p_acc, p_val = predict(val_y, val_x, m)

        eval = 1 - p_acc[0] / 100
        eval_list.append(eval)
        each_model.append(m)

    min_index = eval_list.index(min(eval_list))
    best_model = each_model[min_index]

    p_out_label, p_out_acc, p_out_val = predict(test_y, test_x, best_model)
    e_out_value = 1 - p_out_acc[0] / 100

    with eout_val_lock:
        val_eout.append(e_out_value)

num_threads = 8  
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    train_y, train_x = svm_read_problem("train_extract.txt")
    test_y, test_x = svm_read_problem("test_extract.txt")
    futures = [executor.submit(run_val_experiment, train_y, train_x, test_y, test_x) for _ in range(1126)]


for future in futures:
    future.result()
# %%
plt.hist(val_eout, bins=5, edgecolor='black')  # 设置边界颜色为黑色

# 添加标题和标签
plt.title("Histogram of validation Eout")
plt.xlabel("Value")
plt.ylabel("Frequency")

# 显示图形
plt.show()
# %%
import numpy as np
from liblinear.liblinearutil import *
from concurrent.futures import ThreadPoolExecutor
import threading

train_y, train_x = svm_read_problem("train_extract.txt")
test_y, test_x = svm_read_problem("test_extract.txt")

eout_val_lock = threading.Lock()
three_fold_val_eout = []  
λ_list = [-2, -1, 0, 1, 2, 3] 

def run_three_fold_val_experiment(train_y, train_x, test_y, test_x):
    train_data = list(zip(train_y, train_x))
    np.random.shuffle(train_data)
    shuffled_y, shuffled_x = zip(*train_data)
    
    fold_size = len(shuffled_y) // 3
    folds_y = [list(shuffled_y[i * fold_size:(i + 1) * fold_size]) for i in range(3)]
    folds_x = [list(shuffled_x[i * fold_size:(i + 1) * fold_size]) for i in range(3)]
    
    each_model = []
    eval_list = []

    for λ in λ_list:
        c = 1 / (10 ** λ)
        fold_evals = []
        
        for i in range(3):
            val_y = folds_y[i]
            val_x = folds_x[i]
            train_y_fold = sum(folds_y[:i] + folds_y[i+1:], [])
            train_x_fold = sum(folds_x[:i] + folds_x[i+1:], [])

            m = train(train_y_fold, train_x_fold, f'-c {c} -s 6 -B 1')
            
            p_label, p_acc, p_val = predict(val_y, val_x, m)
            eval = 1 - p_acc[0] / 100 
            fold_evals.append(eval)

        avg_eval = np.mean(fold_evals)
        eval_list.append(avg_eval)
        each_model.append(m)

    min_index = eval_list.index(min(eval_list))
    best_model = each_model[min_index]

    p_out_label, p_out_acc, p_out_val = predict(test_y, test_x, best_model)
    e_out_value = 1 - p_out_acc[0] / 100

    with eout_val_lock:
        three_fold_val_eout.append(e_out_value)
        
num_threads = 8 
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    train_y, train_x = svm_read_problem("train_extract.txt")
    test_y, test_x = svm_read_problem("test_extract.txt")
    futures = [executor.submit(run_three_fold_val_experiment, train_y, train_x, test_y, test_x) for _ in range(1126)]

for future in futures:
    future.result()
# %%
import matplotlib.pyplot as plt
plt.hist(three_fold_val_eout, bins=5, edgecolor='black')  # 设置边界颜色为黑色

# 添加标题和标签
plt.title("Histogram of 3 fold validation Eout")
plt.xlabel("Value")
plt.ylabel("Frequency")

# 显示图形
plt.show()
# %%
