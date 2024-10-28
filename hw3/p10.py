import numpy as np
import matplotlib.pyplot as plt
import random

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
    print("Ein : " + str(ein))
    return ein

def eout(wlin, random_numbers, datas):
    out_y_values = []
    out_x_values = []
    
    for data_index in range(len(datas)):
        if data_index not in random_numbers:
            out_y_values.append(datas[data_index][0])
            out_x_values.append(datas[data_index][1:])
    
    out_y_matrix = np.array(out_y_values)
    out_x_matrix = np.array(out_x_values)
    
    y_out_hat = out_x_matrix @ wlin
    eout = np.mean((out_y_matrix - y_out_hat) ** 2) 
    
    print("Eout : " + str(eout))
    return eout
    

def p10_exp():   
    datas = process_binary_file("cpusmall_scale")
    
    eins = []
    eouts = []
    
    for _ in range(1126):
        y_matrix, x_matrix, random_numbers = random_n_select(32, 8192, datas)
        wlin = wlin_exp(y_matrix, x_matrix)
        eins.append(ein(wlin, y_matrix, x_matrix))
        eouts.append(eout(wlin, random_numbers, datas))
        
    plt.scatter(eins, eouts, color='blue', marker='o', s=50)  # s 參數控制點的大小

    plt.title('Ein vs Eout Scatter Plot')
    plt.xlabel('Ein')
    plt.ylabel('Eout')

    plt.grid(True)

    plt.show()

def p11_and_12exp(p_number):
    if p_number == 11:
        datas = process_binary_file("cpusmall_scale")
    elif p_number == 12:
        datas = process_binary_file("cpusmall_scale", n = 3)
    
    eins = []
    eouts = []
    
    for n_numbers in range(25, 2000, 25):
        each_eins = []
        each_eout = []
        
        for _ in range(16):
            y_matrix, x_matrix, random_numbers = random_n_select(n_numbers, 8192, datas)
            wlin = wlin_exp(y_matrix, x_matrix)
            each_eins.append(ein(wlin, y_matrix, x_matrix))
            each_eout.append(eout(wlin, random_numbers, datas))
            
        eins.append(np.mean(each_eins))
        eouts.append(np.mean(each_eout))
        
        
    plt.plot(range(25, 2000, 25), eins, label='Train Error (Ein)', marker='o')
    plt.plot(range(25, 2000, 25), eouts, label='Test Error (Eout)', marker='x')

    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Error')

    plt.legend()

    plt.grid(True)

    plt.show()      

p11_and_12exp(11)