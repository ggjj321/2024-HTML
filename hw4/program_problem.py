#%%
import numpy as np
import matplotlib.pyplot as plt
import random

exp_times = 1126
q = 3

def process_binary_file(file_path, n = 13):
    datas = []
    with open(file_path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            row_data = line.split()
            insert_data = [int(row_data[0].decode('utf-8')), 1.0]
            
            
            for weight in row_data[1:n]:
                insert_data.append(float(weight.decode('utf-8').split(":")[1]))
            datas.append(insert_data)
    return datas

def random_n_select(n, max_number, datas):
    random_numbers = random.sample(range(1, max_number), n)
    
    y_values = []
    x_values = []
    
    for random_number in random_numbers:
        y_values.append(datas[random_number][0])
        x_values.append(datas[random_number][1:])
    
    y_matrix = np.array(y_values)
    x_matrix = np.array(x_values)
    
    return y_matrix, x_matrix, random_numbers

def wlin_exp(y_matrix, x_matrix):
    x_pseudo_inverse = np.linalg.pinv(x_matrix)
    wlin = x_pseudo_inverse @ y_matrix
    return wlin

def ein(wlin, y_matrix, x_matrix):
    y_hat = x_matrix @ wlin
    ein = np.mean((y_matrix - y_hat) ** 2)
    # print("Ein : " + str(ein))
    return ein

def eout(wlin, random_numbers, datas, is_poly):
    out_y_values = []
    out_x_values = []
    
    for data_index in range(len(datas)):
        if data_index not in random_numbers:
            out_y_values.append(datas[data_index][0])
            out_x_values.append(datas[data_index][1:])
    
    out_y_matrix = np.array(out_y_values)
    out_x_matrix = np.array(out_x_values)
    
    if is_poly:
        out_x_matrix = z_transform(out_x_matrix)
    
    y_out_hat = out_x_matrix @ wlin
    eout = np.mean((out_y_matrix - y_out_hat) ** 2) 
    
    # print("Eout : " + str(eout))
    return eout

def z_transform(x_matrix):
    transform_x_matrix = []
    
    for xn in x_matrix:
        transform_xn = [1]
        for order in range(1, q + 1):
            for x in xn[1:]:
                transform_xn.append(x ** order)
        transform_x_matrix.append(transform_xn)
    
    return np.array(transform_x_matrix)
   
   
#%%
def sgd_linear(sgd_w, x_matrix, y_matrix, η = 0.01):
    random_sample_index = random.randint(0, len(y_matrix) - 1)
    
    gradient = 2 * (y_matrix[random_sample_index] - sgd_w @ x_matrix[random_sample_index]) * x_matrix[random_sample_index]
    sgd_w = sgd_w + η * gradient
    
    return sgd_w
#%%
# p10
exp_times = 1126
datas = process_binary_file("../hw3/cpusmall_scale")

wlin_eins = []
wlin_eouts = []
total_sgd_ein = [0] * int(100000 / 200)
total_sgd_eout = [0] * int(100000 / 200)

for _ in range(exp_times):
    print(_)
    y_matrix, x_matrix, random_numbers = random_n_select(64, 8192, datas)
    wlin = wlin_exp(y_matrix, x_matrix)
    wlin_eins.append(ein(wlin, y_matrix, x_matrix))
    wlin_eouts.append(eout(wlin, random_numbers, datas))
        
    sgd_w = np.zeros(len(wlin))

    for i in range(1, 100000 + 1):
        sgd_w = sgd_linear(sgd_w, x_matrix, y_matrix)
        if i % 200 == 0:
            total_sgd_ein[int(i / 200) - 1] += ein(sgd_w, y_matrix, x_matrix)
            total_sgd_eout[int(i / 200) - 1] += eout(sgd_w, random_numbers, datas)

total_sgd_ein = [x / exp_times for x in total_sgd_ein]
total_sgd_eout = [x / exp_times for x in total_sgd_eout] 


# %%
print(np.average(wlin_eins))
print(np.average(wlin_eouts))
print(total_sgd_ein[-1])
print(total_sgd_eout[-1])

# 36.5608795559208
# 1231.9823902340313
# 55.95245155789427
# 198.87250438080366
# %%
plt.axhline(y=np.average(wlin_eins), color='r', linestyle='--', label='Wlin Ein')
plt.axhline(y=np.average(wlin_eouts), color='b', linestyle='--', label='Wlin Eout')

plt.plot(range(0, 100000, 200), total_sgd_ein, label='SGD Ein', marker='o')
plt.plot(range(0, 100000, 200), total_sgd_eout, label='SGD Eout', marker='x')

plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Error')

plt.legend()

plt.grid(True)

plt.show()
# %%
# p11_p12        
datas = process_binary_file("../hw3/cpusmall_scale")

wlin_eins = []
wlin_eouts = []

wpoly_eins = []
wpoly_eouts = []

# %%
for times in range(exp_times):
    print(times)
    y_matrix, x_matrix, random_numbers = random_n_select(64, 8192, datas)
    wlin = wlin_exp(y_matrix, x_matrix)
    wlin_eins.append(ein(wlin, y_matrix, x_matrix))
    wlin_eouts.append(eout(wlin, random_numbers, datas, False))
    
    transform_x_matrix = z_transform(x_matrix)
    
    wpoly = wlin_exp(y_matrix, transform_x_matrix)
    wpoly_eins.append(ein(wpoly, y_matrix, transform_x_matrix))
    wpoly_eouts.append(eout(wpoly, random_numbers, datas, True))

# %%
ein_diff = [a - b for a, b in zip(wlin_eins, wpoly_eins)]
eout_diff = [a - b for a, b in zip(wlin_eouts, wpoly_eouts)]

# %%
plt.hist(ein_diff, edgecolor='black') 
plt.title("ein_diff")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.hist(eout_diff, edgecolor='black')  
plt.title("eout_diff")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
# %%
print(np.average(ein_diff))
print(np.average(eout_diff))

print(np.average(wlin_eins))
print(np.average(wlin_eouts))

print(np.average(wpoly_eins))
print(np.average(wpoly_eouts))
# %%

