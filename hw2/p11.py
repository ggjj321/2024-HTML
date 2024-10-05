import numpy as np
import matplotlib.pyplot as plt
import random

def p11_ein(xs, ys, thetas, sign):
    e_in_number = 0
    min_theta = thetas[0]
    
    predictions = sign * np.sign(xs - thetas[0])
    predictions[predictions == 0] = -1
    
    errors = predictions != ys
    e_in_number = np.sum(errors)
    
    min_e_in_number = e_in_number

    for theta_index in range(1, len(thetas)):
        x_i = xs[theta_index - 1]
        y_i = ys[theta_index - 1]
        
        old_prediction = sign * np.sign(x_i - thetas[theta_index - 1])
        if x_i - thetas[theta_index - 1] == 0:
            old_prediction = -sign
        
        new_prediction = sign * np.sign(x_i - thetas[theta_index])
        if x_i - thetas[theta_index] == 0:
            new_prediction = -sign
        
        if old_prediction != new_prediction:
            if new_prediction != y_i:
                e_in_number += 1
            else:
                e_in_number -= 1
        
        if e_in_number < min_e_in_number:
            min_e_in_number = e_in_number
            min_theta = thetas[theta_index]
    
    return min_theta, min_e_in_number / len(xs)

def p12_e_in(xs, ys, thetas):
    e_ins = []
    signs = [1, -1]
    
    for sign in signs:
        for theta in thetas:
            e_in = 0
            for x_index in range(len(xs)):
                if sign * np.sign(xs[x_index] - theta) != ys[x_index]:
                    e_in += 1
            e_ins.append([sign, theta, e_in])
            
    random_e_in = random.choice(e_ins)
    return random_e_in[0], random_e_in[1], random_e_in[2] / len(xs)

def ein_eout_experiment():
    xs = np.random.uniform(-1, 1, 12)
    ys = []
    thetas = []
    p = 0.15

    xs.sort()
    
    for x in xs:
        sign_x = np.sign(x)
        if random.random() < p:
            sign_x = -sign_x
        ys.append(sign_x)

    thetas = [-1] + [(xs[i] + xs[i+1])/2 for i in range(len(xs)-1)]
    
    sign, theta, e_in = p12_e_in(xs, ys, thetas)
    
    print(sign, theta, e_in)
    
    # postive_min_theta, postive_e_in = ein(xs, ys, thetas, 1)
    # negative_min_theta, negative_e_in = ein(xs, ys, thetas, -1)
    
    # if postive_e_in < negative_e_in:
    #     e_in = postive_e_in
    #     min_theta = postive_min_theta
    #     sign = 1
    # else:
    #     e_in = negative_e_in
    #     min_theta = negative_min_theta
    #     sign = -1
    
    v = sign * (0.5 - p)
    u = 0.5 - v
    
    e_out = u + v * abs(theta)
    
    return e_in, e_out
    
eins = []
eouts = []
for _ in range(2000):
    e_in, e_out = ein_eout_experiment()
    eins.append(e_in)
    eouts.append(e_out)
    
diffs = np.array(eouts) - np.array(eins)
median_diff = np.median(diffs)
print(f"The median of E_out(g) - E_in(g) is {median_diff}")

plt.scatter(eins, eouts)

plt.title('Scatter Plot of Ein(g) vs Eout(g)')
plt.xlabel('Ein(g)')
plt.ylabel('Eout(g)')

plt.show()