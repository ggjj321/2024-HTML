# 10

由於每個 gt 都是弱分類器，Ein(gt) 和 ϵt 都是比亂猜好一點點的效果，符合講議中 Adaptive Boosting 中第一個元素：weak base learning algorithm A (Student)。
# 11

發現到經過越多 gt 的聚合，可以將 Ein 做得非常小，但是 Eout 並沒有顯著的下降，推論隨著 T 增加，模型複雜度增加，VC bound 也會增加，讓 Eout 沒有有效減少，符合此公式，因此也造成了 overfitting。

# 12

隨著 t 的增加，Ein 會跟著減少，訓練集中樣本數正確的數量增加，正確樣本的權重根據更新公式會除上 ϵt，權重總和就會跟著降低，圖中趨勢符合這個推論。

# 10

Since each gt​ is a weak classifier, both Ein(gt) and ϵt​ show performance that is only slightly better than random guessing. This aligns with the first element in the Adaptive Boosting theory: the weak base learning algorithm A (Student).

# 11

It is observed that as more gt​'s are aggregated, Ein​ can be reduced to a very small value. However, Eout does not decrease significantly. This suggests that as T increases, the model complexity grows, and the VC bound also increases, preventing Eout​ from effectively decreasing. This aligns with the formula, which explains the phenomenon of overfitting.

# 12

As t increases, Ein decreases, meaning that the number of correctly classified samples in the training set increases. According to the update formula, the weights of correctly classified samples are divided by ♦t​, causing the total weight to decrease. The trend in the graph aligns with this reasoning.
