# 11
相較於第 10 題，使用 random split validation 選出的 λ 所算出的 Eout 分佈會比起使用 Ein 選出的 λ 所算出的 Eout 還要小一些，發現到由於使用 Eval 評估的模型會比較接近真實 Eout，符合講義上的這張圖。

Compared to question 10, the distribution of Eout calculated using the λ selected through random split validation is slightly smaller than that obtained using λ selected based on Ein​. It was observed that models evaluated with Eva​ are closer to the true Eout​, consistent with the figure in the lecture notes.

# 12

使用 cross validation 總體分佈和 random split validation 差不多，但是 0.019~0.02 等較高數值的錯誤分佈比起 random split validation 還少，發現到使用 cross validation 來評估模型還是會比用 random split validation 還要好。

The overall distribution of cross-validation is similar to that of random split validation, but the distribution of higher error values around 0.019~0.02 is lower than in random split validation. This observation suggests that using cross-validation to evaluate models is still better than using random split validation.