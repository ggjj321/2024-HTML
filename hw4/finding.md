# 10

大約 0-20000 iterations Ein 下降 Eout 也跟著下降，這一段有 learning happen 的效果，但是在 20000 iteration 以後，Ein 會繼續下降，但 Eout 則會上升，則會有 over fitting 的效果。推測是因為在 linear regrresion 使用 SGD 時，每次都會更新 w 讓 w 的 model compelxity 變得更高，就會造成講義上 model compelxity、Ein 及 Eout 的關係，而直接使用 Wlin 來計算 linear regreesion 的 Ein 及 Eout 是直接考慮 w model compelxity 最高的情況，因此會有 Ein 極小，而 Eout 極大的狀況。

From approximately 0 to 20,000 iterations, Ein​ decreases and Eout​ also decreases, indicating an effective learning phase where learning is happening. However, after 20,000 iterations, Ein​ continues to decrease while Eout​ starts to increase, demonstrating an overfitting effect. This is likely because, when using SGD for linear regression, the weights w are continuously updated, causing the model's complexity to increase. This aligns with the relationship between model complexity, Ein, and Eout​ as discussed in the lecture. Using wlin directly to calculate Ein​ and Eout​ in linear regression considers the case where the model complexity of w is at its highest, resulting in an extremely small Ein​ but a very large Eout​.


# 11 
average:33.437899868590144
在訓練資料取樣數是 64，對於整筆資料是 8192 的大小來說是小的取樣數，因此 wpoly 和 wlin 在 Ein 的表現上並不會差太多，最多只會到 10^2 等級。

average: 33.437899868590144
With a training sample size of 64, which is relatively small compared to the total data size of 8192, the performance of wpoly​ and wlin​ on Ein​ will not differ significantly, at most reaching a difference of around 10^2 in magnitude.

# 12 
average:-2134261045262086.0
相對於整筆資料，測試資料取樣數算是小的，但是由於整體 x 經過 Φ(x) transform, wpoly 的target function 的 complexity 會提高，因此在測試 Eout 時就會比較偏向圖中左上角的紅色區域，Eout 差距會很大。另外實際查看 wlin 的 平均 Ein 是 36.58453011875178，平均 Eout 是 1141.530538343793，wpoly 的平均 Ein 是 3.1466302501616386，平均 Eout 是 3.1466302501616386，wpoly 也較 wlin 有 bad generalization

average: -2134261045262086.0
Compared to the entire dataset, the test sample size is relatively small. However, due to the Φ(x) transform applied to x, the target function complexity for wpoly​ increases. As a result, when evaluating Eout, the results tend to fall in the red region in the upper left corner of the graph, indicating a large difference in Eout. Additionally, when examining wlin​, the average Ein is 36.58453011875178, and the average Eout​ is 1141.530538343793. For wpoly​, the average Ein is 3.1466302501616386, and the average Eout​ is 2134261045263228.2. This shows that wpoly​ has worse generalization compared to wlin​.