import numpy as np 
import statistics
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import random

array_size = 47206
read_line_num = 200
    
def process_binary_to_yn_and_xn(file_path):
    all_yn = np.zeros(read_line_num)
    all_xn = np.zeros((read_line_num, array_size), dtype=np.float32)
    with open(file_path, 'rb') as f:
        for data_index in range(read_line_num):
            data_list = []
            
            line = f.readline()
            data_list = line.split()
            all_yn[data_index] = int(data_list[0])
            
            for x_data_str in data_list[1:]:
                x_data = str(x_data_str).split(":")
                all_xn[data_index][0] = 1
                all_xn[data_index][int(x_data[0][2:])] = float(x_data[1][:-1])
                
    return all_yn, all_xn

def random_PLA(all_yn , all_xn):
    correct_judge_num = 0
    pla_num = 0
    w = np.zeros(array_size)
    w_len = [0]
    
    while correct_judge_num < 5 * read_line_num:
        random_vector_index = random.randint(0, 199)
        predict_yn = np.sign(np.dot(w, all_xn[random_vector_index]))
        
        if predict_yn == 0:
            predict_yn = -1
            
        if predict_yn == all_yn[random_vector_index]:
            correct_judge_num += 1
        else:
            correct_judge_num = 0
            w = w + all_yn[random_vector_index] * all_xn[random_vector_index]
            w_len.append(np.linalg.norm(w))
            pla_num += 1
    return pla_num, w_len

def q12_PLA_update(all_yn , all_xn):
    correct_judge_num = 0
    pla_num = 0
    w = np.zeros(array_size)
    
    while correct_judge_num < 5 * read_line_num:
        random_vector_index = random.randint(0, 199)
        while True:
            predict_yn = np.sign(np.dot(w, all_xn[random_vector_index]))
            
            if predict_yn == 0:
                predict_yn = -1
                
            if predict_yn == all_yn[random_vector_index]:
                correct_judge_num += 1
                break
            else:
                correct_judge_num = 0
                w = w + all_yn[random_vector_index] * all_xn[random_vector_index]
                pla_num += 1
    return pla_num

def plot_histogram(all_yn , all_xn):
    all_pla_times = []
    
    # q10 ver 102
    for _ in range(1000):
        pla_num, w_len = random_PLA(all_yn , all_xn)
        all_pla_times.append(pla_num)
    
    # q12 ver q12 median : 103.0
    # for _ in range(1000):
    #     pla_num = q12_PLA_update(all_yn , all_xn)
    #     print(pla_num)
    #     all_pla_times.append(pla_num)
    
    print(statistics.median(all_pla_times))
    
    df = pd.DataFrame({
        'data_column': all_pla_times
    })
    
    fig = px.histogram(df, x="data_column", nbins=5, title="q10 Histogram")
    
    fig.update_layout(
        xaxis_title="PLA implement times",
        yaxis_title="Implement times appear Frequency"
    )

    fig.show()

def plot_line_chart(all_yn , all_xn):
    min_times = 10000
    all_pla_w_len = []
    
    for _ in range(1000):
        pla_num, w_len = random_PLA(all_yn , all_xn)
        if pla_num < min_times:
            min_times = pla_num
        all_pla_w_len.append(w_len)
    
    x_values = np.arange(1, min_times)
    data = []

    for i in range(1000):
        y_values = all_pla_w_len[i][:min_times]
        
        trace = go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            name=f'exp {i+1}',
            line=dict(width=0.5),  
            opacity=0.5           
        )
        
        data.append(trace)

    layout = go.Layout(
        title='1000 times ||wt||',
        xaxis=dict(title='t'),
        yaxis=dict(title='||wt||'),
        showlegend=False 
    )

    fig = go.Figure(data=data, layout=layout)

    pio.show(fig)
    

def main():
    file_path = "rcv1_train.binary"
    all_yn , all_xn = process_binary_to_yn_and_xn(file_path)
    plot_histogram(all_yn , all_xn)

main()