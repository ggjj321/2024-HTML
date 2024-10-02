import numpy as np
import matplotlib.pyplot as plt
import random

def ein(xs, ys, thetas):
    min_e_in = float('inf')
    best_s = None
    best_theta = None

    for theta in thetas:
        for s in [1, -1]:
            predictions = s * np.sign(xs - theta)
            predictions[predictions == 0] = -1  # 处理 sign(0) = -1
            errors = predictions != ys
            e_in = np.mean(errors)
            if e_in < min_e_in or (e_in == min_e_in and s * theta < best_s * best_theta):
                min_e_in = e_in
                best_s = s
                best_theta = theta
    return best_s, best_theta, min_e_in

def p11_ein_eout_experiment():
    xs = np.random.uniform(-1, 1, 12)
    xs.sort()
    ys = []
    p = 0.15

    for x in xs:
        sign_x = np.sign(x)
        if random.random() < p:
            sign_x = -sign_x
        ys.append(sign_x)

    thetas = [-1] + [(xs[i] + xs[i+1])/2 for i in range(len(xs)-1)]
    best_s, best_theta, best_ein = ein(xs, ys, thetas)

    v = best_s * (0.5 - p)
    u = 0.5 - v

    eout = u + v * abs(best_theta)

    return best_ein, eout

eins = []
eouts = []
for _ in range(2000):
    best_ein, eout = p11_ein_eout_experiment()
    eins.append(best_ein)
    eouts.append(eout)

plt.scatter(eins, eouts)
plt.title('Scatter Plot of Ein(g) vs Eout(g)')
plt.xlabel('Ein(g)')
plt.ylabel('Eout(g)')
plt.show()

# 计算 E_out(g) - E_in(g) 的中位数
diffs = np.array(eouts) - np.array(eins)
median_diff = np.median(diffs)
print(f"The median of E_out(g) - E_in(g) is {median_diff}")
