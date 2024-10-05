## Question 6

$$
\text{P(one specific number is green)} = \frac{1}{2} \\
$$

$$
\text{P(one specific number all green on 5 cards)} = \left(\frac{1}{2}\right)^5 = \frac{1}{32} \\
$$

$$
\text{P(one specific number not all green on 5 cards)} = 1 - \frac{1}{32} = \frac{31}{32} \\
$$

$$
\text{P(16 numbers are not all green on 5 cards)} = \left( \frac{31}{32} \right)^{16} \\
$$
$$
\text{P(there are some numbers that are green on 5 cards)} = 1 - \text{P(16 numbers are not all green on 5 cards)} \\ = 1 - \left( \frac{31}{32} \right)^{16} \\
$$

## Question 7

$$
\text{P(pick one green 5's)} = \frac{1}{2} \\
$$

$$
\text{P(pick 5 green 5's)} = \left(\frac{1}{2}\right)^5 = \frac{1}{32}
$$

## Question 8

$$
\text{Since there are M machines and a total of t pulls, the size of the hypothesis set number is } Mt.  \text{And } E_{\text{out}} = \mu_m, \, E_{\text{in}} = \frac{c_m}{N_m}. \\
$$

$$
\text{Let } P(\text{bad at t}) = P\left(\mu_m > \frac{c_m}{N_m} + \sqrt{\frac{\ln t - \frac{1}{2} \ln \delta}{N_m}}\right). \\
$$

$$
\sum_{t=1}^{\infty} \delta t^{-2} \geq \sum_{t=1}^{\infty} P(\text{bad at t}) \geq P(\text{bad at t = 1 or bad at t = 2} \dots \text{bad at t = } \infty) \\
$$

$$
\text{because} \quad \sum_{t=1}^{\infty} \delta t^{-2} = \frac{\pi^2}{6} \delta > \sum_{t=1}^{\infty} P(\text{bad at t}). \\
$$

$$
\text{Thus, } P(\text{bad at t}) \text{ will be constrained to a certain value and will not approach infinity}. \\
$$

$$
\sum_{t=1}^{\infty} P(\text{bad at t}) \geq P(\text{bad at t = 1} \, \text{or} \, \dots \, \text{bad at } \infty). \\
$$

$$
\text{We can get:} \\
$$

$$
P\left(\mu_m > \frac{c_m}{N_m} + \epsilon\right) \leq 2 \cdot \text{hypothesis number} \cdot \exp(-2 \epsilon^2 N_m). \\
$$

$$
\Rightarrow P\left(\mu_m > \frac{c_m}{N_m} + \epsilon\right) \leq 2 \cdot Mt \cdot \exp(-2 \epsilon^2 N_m). \\
$$

$$
\text{Let} \quad \delta = 2 \cdot Mt \cdot \exp(-2 \epsilon^2 N_m). \\
\frac{\delta}{2Mt} = \exp(-2 \epsilon^2 N_m). \\
$$

$$
\Rightarrow 2 \ln M + 2 \ln t - \ln \delta = 2 \epsilon^2 N_m. \\
$$

$$
\Rightarrow \ln t + \ln M - \frac{1}{2} \ln \delta = \epsilon^2. \\
$$

$$
\Rightarrow \sqrt{\frac{\ln t + \ln M - \frac{1}{2} \ln \delta}{N_m}} = \epsilon. \\
$$

$$
\text{With } 1 - \delta, \text{ good case, } E_{\text{out}} \leq E_{\text{in}} + \epsilon. \\
$$

$$
\Rightarrow \mu_m \leq \frac{c_m}{N_m} + \sqrt{\frac{\ln t + \ln M - \frac{1}{2} \ln \delta}{N_m}} \quad \text{is proved}.
$$

## Question 9

We can find k different positive numbers of ones in $\text{} \{-1, 1\}^k,$and all zeros can get 0, so $\{-1, 1\}^k$ can shatter the hypothesis set of k+1, so $d_{vc} \geq k+1 $
And we can't find any k+2 different numbers offered by $\{-1, 1\}^k$, so $d_{vc} < k+2$
with $d_{vc} \geq k+1 $ and $d_{vc} < k+2$, we can get $d_{vc} = k+1$

## Question 10

$$
\text{With } x \text{ in uniform distribution } [-1, 1], \text{ the pdf is } \frac{1}{2}. \\
$$

$$
\text{Hence } E_{\text{out}}(h_{s, \theta}) = \frac{1}{2} \int_{-1}^{1} P\left(y \neq h_{s, \theta}(x)\right) dx. \\
$$

$$
\text{And with } E_{\text{out}}(h_{s, \theta}) = u + v \cdot | \theta |, \text{ we have:}
$$
$$
s = \frac{1}{2} - v + v | \theta | \\
$$

$$
u = \frac{1}{2} + v \cdot | \theta | (1 - 1) \\
$$

$$
E_{\text{out}}(h_{s, \theta}) = \frac{1}{2} + s \left( \frac{1}{2} - p \right) \cdot (| \theta | - 1). \\
$$

$$
\text{We can divide } s, \theta \text{ into four cases:} \\
$$
$$
(s = +1, \theta \geq 0), (s = +1, \theta < 0), (s = -1, \theta \geq 0), (s = -1, \theta < 0). \\
$$

$$
\text{This leads to:} \\
$$

$$
E_{\text{out}}(h_{s, \theta})_{s=+1, \theta \geq 0}, \quad E_{\text{out}}(h_{s, \theta})_{s=+1, \theta < 0}, \\
E_{\text{out}}(h_{s, \theta})_{s=-1, \theta \geq 0}, \quad E_{\text{out}}(h_{s, \theta})_{s=-1, \theta < 0}. \\
$$

$$
\text{We can then calculate } E_{\text{out}}(h_{s, \theta}) \text{ by conquering these cases.}
$$

$$
\textbf{Case} \ \text{sign} = +1, \text{ has  2 } \theta \text{ cases:} \\
$$

$$
\theta \geq 0, \theta < 0 \\
$$

$$
\textbf{In} \ \theta \geq 0 \textbf{ :} \\
$$

$$
\text{We have 3 cases:}\ x \in [-1, 0], \ x \in (0, \theta],  x \in (\theta, 1]\\
 \\
$$

$$
x \in [-1, 0],  \text{sign}(x) = -1, \ P(y = 1) = p, \ P(y = -1) = 1 - p, \ h_{s,\theta} = -1, \ P(E_{\text{out of this case}}) = p, \\
\text{The range of this case is } -1 \sim 0. \\
$$

$$
x \in (0, \theta],  \text{sign}(x) = +1, \ P(y = 1) = 1 - p, \ P(y = -1) = p, \ h_{s,\theta} = -1, \ P(E_{\text{out of this case}}) = 1 - p, \\
\text{The range of this case is } 0 \sim \theta.
$$

$$
x \in [\theta, 1],  \text{sign}(x) = 1, \ P(y = 1) = 1 - p, \ P(y = -1) = p, \ h_{s,\theta} = 1, \ P(E_{\text{out of this case}}) = p \\
\text{The range of this case is } \theta \sim 1. \\
$$

$$
\textbf{In} \ \text{sign} = +1, \theta \geq 0: \\
$$

$$
E_{\text{out}}(h_{s,\theta})_{s=+1, \theta>0} = \frac{1}{2} \left[ \int_{-1}^{0} p \, dx + \int_{0}^{\theta} (1 - p) \, dx + \int_{\theta}^{1} p \, dx \right] \\
$$

$$
= \frac{1}{2} \left[ p + (1 - p) \theta + p(1 - \theta) \right] \\
$$

$$
= \frac{1}{2} \left[ p + (1 - p) \theta + p - p\theta \right] \\
$$

$$
= p + \frac{1}{2} (1 - 2p) \theta \\
$$

$$
\text{With } v, u \text{ in case s =} +1, \theta > 0: \\
$$

$$
E_{\text{out}}(h_{s,\theta}) = \frac{1}{2} + s \left( \frac{1}{2} - p \right) (| \theta | - 1) \\
$$

$$
= \frac{1}{2} + \frac{\theta}{2} - \frac{1}{2} + p \theta + p \\
$$

$$
= p + \frac{1}{2} (1 - 2p) \theta, \quad E_{\text{out}}(h_{s,\theta})_{s=+1, \theta>0} \ \text{is proved}.
$$

$$
\textbf{In} \ \theta < 0 \textbf{ case:} \\
$$

$$
\text{We have 3 cases:}\ x \in [-1, \theta], \ x \in (\theta, 0],  x \in (0, 1]\\
 \\
$$

$$
x \in [-1, \theta], \text{sign}(x) = -1, \ P(y = 1) = p, \ P(y = -1) = 1 - p, \ h_{s,\theta} = -1, \ P(E_{\text{out of this case}}) = p \\
\text{The range of this case is } -1 \sim \theta. \\
$$

$$
x \in (\theta, 0], \text{sign}(x) = -1, \ P(y = 1) = p, \ P(y = -1) = 1 - p, \ h_{s,\theta} = 1, \ P(E_{\text{out of this case}}) = 1 - p, \\
\text{The range of this case is } \theta \sim 0. \\
$$

$$
x \in (0, 1], \text{sign}(x) = +1, \ P(y = 1) = 1 - p, \ P(y = -1) = p, \ h_{s,\theta} = 1, \ P(E_{\text{out of this case}}) = p, \\
\text{The range of this case is } 0 \sim 1.
$$

$$
\textbf{In } \ \text{sign} = +1, \ \theta < 0 \text{ :}\\
$$

$$
E_{\text{out}}(h_{s,\theta})_{s=+1, \theta<0} = \frac{1}{2} \left[ \int_{-1}^{\theta} p \, dx + \int_{\theta}^{0} (1 - p) \, dx + \int_{0}^{1} p \, dx \right] \\
$$

$$
= \frac{1}{2} \left[ p(\theta + 1) + (1 - p)\theta + p \right] \\
$$

$$
= \frac{1}{2} \left[ p(\theta + 1) - (1 - p)\theta + p \right] \\
$$

$$
= \frac{1}{2} \left[ 2p - (1 - 2p)\theta \right] = p - \frac{1}{2}(1 - 2p)\theta \\
$$

$$
\text{With } v, u, \text{ in case } +1, \theta < 0: \\
$$

$$
E_{\text{out}}(h_{s,\theta}) = \frac{1}{2} + s\left(\frac{1}{2} - p\right)(| \theta | - 1) \\
$$

$$
= \frac{1}{2} - \left(\frac{1}{2} - p\right)(\theta + 1) \\
$$

$$
= \frac{1}{2} - \frac{\theta}{2} - \frac{1}{2} + p\theta + p \\
$$

$$
= p - \frac{1}{2}(1 - 2p)\theta \quad E_{\text{out}}(h_{s,\theta})_{s=+1, \theta<0} \text{ is proved}.
$$

$$
\textbf{Case} \ \text{sign} = -1, \text{ has 2 } \theta \text{ cases:} \\
$$

$$
\theta \geq 0, \theta < 0 \\
$$

$$
\text{In }\theta \geq 0
$$

$$
\text{We have 3 cases:}\ x \in [-1, 0), \ x \in [0, \theta),  x \in [\theta, 1]\\
 \\
$$

$$
x \in [-1, 0],\text{sign}(x) = -1, \ P(y = 1) = p, \ P(y = -1) = 1 - p, \ h_{s,\theta} = 1, \ P(E_{\text{out of this case}}) = 1 - p, \\
\text{The range of this case is } -1 \sim 0. \\
$$

$$
x \in [0, \theta), \text{sign}(x) = 1, \ P(y = 1) = 1 - p, \ P(y = -1) = p, \ h_{s,\theta} = 1, \ P(E_{\text{out of this case}}) = p, \\
\text{The range of this case is } 0 \sim \theta.
$$

$$
x \in [\theta, 1], \text{sign}(x) = 1, \ P(y = 1) = 1 - p, \ P(y = -1) = p, \ h_{s,\theta} = -1, \ P(E_{\text{out of this case}}) = p \\
\text{The range of this case is } \theta \sim 1. \\
$$
$$
\textbf{In} \ \text{sign} = -1, \ \theta \geq 0 \ \textbf{:} \\
$$

$$
E_{\text{out}}(h_{s,\theta})_{s=-1, \theta \geq 0} = \frac{1}{2} \left[ \int_{-1}^{0} (1 - p) \, dx + \int_{0}^{\theta} p \, dx + \int_{\theta}^{1} (1 - p) \, dx \right] \\
$$

$$
= \frac{1}{2} \left[ (1 - p) \cdot (0 + 1) + p \cdot \theta + (1 - p) \cdot (1 - \theta) \right] \\
$$

$$
= \frac{1}{2} \left[ 1 - p + p \theta + (1 - p)(1 - \theta) \right] \\
$$

$$
= \frac{1}{2} \left[ 1 - p + p \theta + (2 p - 1)(1 - \theta) \right] \\
$$

$$
= 1 - p + \frac{1}{2}(2 p - 1) \theta \\
$$

$$
\text{With } v, u, \text{ in case } s = -1, \theta \geq 0: \\
$$

$$
E_{\text{out}}(h_{s,\theta}) = \frac{1}{2} - \left( \frac{1}{2} - p \right)(|\theta| - 1) \\
$$

$$
= \frac{1}{2} - \left(\frac{1}{2} - p \right)(\theta - 1) \\
$$

$$
= 1 - p + \frac{1}{2}(2 p - 1) \theta \quad E_{\text{out}}(h_{s,\theta})_{s=-1, \theta \geq 0} \text{ is proved.}
$$

$$
\textbf{In} \ \theta < 0 \ \textbf{:}
$$

$$
\text{We have 3 cases:}\ x \in [-1, \theta], \ x \in (\theta, 0],  x \in (0, 1]\\
 \\
$$

$$
x \in [-1, \theta], \text{sign}(x) = -1, \ P(y = 1) = p, \ P(y = -1) = 1 - p, \ h_{s,\theta} = 1, \ P(E_{\text{out of this case}}) = 1 - p \\
\text{The range of this case is } -1 \sim \theta. \\
$$

$$
x \in (\theta, 0], \text{sign}(x) = -1, \ P(y = 1) = p, \ P(y = -1) = 1 - p, \ h_{s,\theta} = -1, \ P(E_{\text{out of this case}}) = p \\
\text{The range of this case is } \theta \sim 0. \\
$$

$$
x \in (0, 1], \text{sign}(x) = 1, \ P(y = 1) = 1 - p, \ P(y = -1) = p, \ h_{s,\theta} = -1, \ P(E_{\text{out of this case}}) = 1-p \\
\text{The range of this case is } 0 \sim 1.
$$

$$
\text{In } \text{sign} = -1, \ \theta < 0
$$

$$
E_{\text{out}}(h_{s,\theta}) = \frac{1}{2} \left[ \int_{-1}^{\theta} p(1 - p) \, dx + \int_{\theta}^{0} p \, dx + \int_{0}^{1} (1 - p) \, dx \right]
$$

$$
= \frac{1}{2} \left[ \theta(1 - p) + p + (1 - p) \right]
$$

$$
= \frac{1}{2} \left[ \theta - p\theta + 1 - p + 1 - p \right] = 1 - p - p\theta + \frac{\theta}{2}
$$

$$
= 1 - p + \frac{1}{2}(1 - 2p)\theta
$$
$$
\text{with } v, u, \text{ in case } s = -1, \theta < 0
$$

$$
E_{\text{out}}(h_{s,\theta}) = \frac{1}{2} - \left( \frac{1}{2} - p \right)(1 - \lvert \theta \rvert)
$$

$$
= \frac{1}{2} + \left( \frac{1}{2} - p \right)(\lvert \theta \rvert + 1) = \frac{1}{2} + \frac{\theta}{2} - p\theta + 1 - p
$$

$$
= 1 - p + \frac{1}{2}(1 - 2p)\theta, \ E_{\text{out}}(h_{s,\theta})_{\{s = -1, \theta < 0\}} \text{ is proved.}
$$

$$
\text{Hence, } E_{\text{out}}(h_{s,\theta})_{s = \pm 1, \theta \geq 0}, E_{\text{out}}(h_{s,\theta})_{s = \pm 1, \theta < 0} \text{ are proved.}
$$

$$
E_{\text{out}}(h_{s,\theta}) = u + v \cdot \lvert \theta \rvert, \ \text{where } u = \frac{1}{2} - v \text{ is proved.}
$$
