# Kalman Filter

The Kalman Filter is a popular filtering technique used in engineering and econometrics and was first introduced by R.E. Kalman in [1960](https://www.cs.unc.edu/~welch/kalman/kalmanPaper.html). This post contains the equations to implement the filter and estimate the parameters. 

---
## Dynamics

The underlying stochastic process is a time-invarient dynamical system in discrete time. The hidden states $Z_t$ follow an AR(1) process with Gaussian noise. The observations $y_t$ are a linear transformation of the states with added Gaussian noise.  

$$
\begin{aligned}
Z_{t+1} & = A Z_t + \epsilon_s \\
Y_t & = C Z_t + B X_t + \epsilon_n \\
\epsilon_n & \sim N(0, R) \\
\epsilon_s & \sim N(0, Q) \\
Z_0 & \sim N(\mu, \Sigma)
\end{aligned}
$$

## Variables

- $Y$ : observations (n x T)
- $Z$ : hidden states (m x (T+1))
- $\epsilon_n$ : observation noise (n x T)
- $\epsilon_s$ : ??? noise (m x T)

## Parameters

- T : number of time observations
- m : number of states
- n : number of variables per observations
- $A$ : (m x m)
- $C$ : observation matrix (n x m)
- $Q$ : (m x m)
- $R$ : covariance matrix of observation noise (n x n)
- $\Sigma$ : prior covariance matrix (m x m)
- $\mu$ : prior mean (m x 1)

---
## Estimating the hidden states

Based on the parameters A, C, Q & R, one can compute the conditional and predictive distributions of the hidden states using the following formulas. 

$$
\begin{aligned}
    p(Z_t|Y_{0...t}, \theta) & = N(Z_{t|t}, P_{t|t}) \\
    p(Z_{t+1}|Y_{0...t}, \theta) & = N(Z_{t+1|t}, P_{t+1|t}) \\
    p(Z_t|Y_{0...T}, \theta) & = N(Z_{t|T}, P_{t|T})
\end{aligned}
$$

The conditional distributions can be calculated with a forward pass and backward pass. This entire procedure is called Kalman Smoothing. 

- ## Forward pass

Initialisation
$$
\begin{aligned}
    Z_{0|0} & = \mu \\
    P_{0|0} & = \Sigma
\end{aligned}
$$

For t = 1,2,...,T-1
$$
\begin{aligned}
Z_{t+1|t} & = A Z_{t|t} \\
P_{t+1|t} & = A P_{t|t} A' + Q \\
K_{t+1} & = P_{t+1|t} C' (C P_{t+1|t} C' + R)^{-1} \\
Z_{t+1|t+1} & = Z_{t+1|t} + K_{t+1} (Y_{t+1} - B X_{t+1} - C Z_{t+1|t}) \\
P_{t+1|t+1} & = P_{t+1|t} - K_{t+1} C P_{t+1|t}
\end{aligned}
$$

- ## Backward pass

Initialisation

$$
\begin{aligned}
    Z_{T|T} & = Z_{T|T} \\
    P_{T|T} & = P_{T|T} \\
    P_{T,T-1|T} & = (I-K_T C)A P_{T-1|T-1}
\end{aligned}
$$

For t = T-1,T-2,...,0
$$
\begin{aligned}
L_t & = P_{t|t} A' P_{t+1|t}^{-1} \\
z_{t|T} & = z_{t|t} + L_t (z_{t+1|T}-z_{t+1|t}) \\
P_{t|T} & = P_{t|t} + L_t (P_{t+1|T}-P_{t+1|t}) L_t' \\
P_{t+1,t|T} & = P_{t+1|t+1} L_t' + L_{t+1} (P_{t+2,t+1|T}-AP_{t+1|t+1}) L_t' \\
\end{aligned}
$$

---
## Parameter Estimation

There does not exist a closed form solution of the parameters of the previously defined stochastic process. The most popular approach is to use the Expectation Maximisation (EM) algorithm to iteratively estimate the weights. The EM algorithm is described below. 


## EM Algorithm

The EM Algorithm estimates the parameters by iteratively optimizing the lower bound of the marginal likelihood $p(Y|\theta)$. The evidence lower bound (ELBO) is defined as 

$$
\begin{aligned}
    \log p(Y|\theta) & = \log \int p(Y,Z|\theta) dZ \\
    & = \log \int q(Z) \frac{p(Y,Z|\theta)}{q(Z)} dZ \\
    & \geq \int q(Z) \log \frac{p(Y,Z|\theta)}{q(Z)} dZ \\
    & = E_{q(Z)}[\log p(Y,Z|\theta)] - E_{q(Z)}[\log q(Z)] \\
    & = ELBO(q(z), \theta) 
\end{aligned}  
$$

There are two unknowns in this equation: the parameters $\theta$ and the distribution $q(Z)$. The E-step optimizes the ELBO by changing $q(Z)$ whilst keeping the parameters $\theta$ fixed and the M-Step optimizes the ELBO by changing $\theta$ and keeping the distribution $q(Z)$ fixed. 

## E-step

The ELBO can be rewritten as
$$
\begin{aligned}
    ELBO(q(z), \theta) & = E_{q(Z)}[\log p(Y,Z|\theta)] - E_{q(Z)}[\log q(Z)] \\ 
    & = E_{q(Z)}[\log p(Y|\theta)] + E_{q(Z)}[\log p(Z|Y,\theta)] - E_{q(Z)}[\log q(Z)] \\
    & = \log p(Y|\theta) - KL(q(z)||p(Z|Y,\theta)) 
\end{aligned}  
$$

The ELBO is equal to the marginal likelihood if the KL divergence is zero. This only occurs if the distribution $q(Z)$ is set to
$$
\begin{aligned}
    q(z) & = p(Z|Y,\theta)
\end{aligned}
$$

## M-step

In the M-step, the parameters $\theta$ are optimized by keeping $q(Z)$ fixed. As $q(Z)$ is independent of $\theta$, the objective can be formulated as 

$$
\begin{aligned}
    & Objective : \argmax_{\theta} ELBO(q(z), \theta) = \argmax_{\theta} E_{q(Z)}[\log p(Y,Z|\theta)]
\end{aligned}  
$$

because $E_{q(Z)}[\log q(Z)]$
is independent of $\theta$. 

---
## Marginal Likelihood

After the E-step, the ELBO is equal to the marginal likelihood. Therefore, the convergence can be checked with the marginal likelihood which can be calculated with

$$
\begin{aligned}
\log p(Y|\theta) & = \sum_{t=1}^T \log p(Y_t|Y_{1...t-1}, \theta) \\
& = \sum_{t=1}^T \log N(Y_t|C Z_{t|t-1}, C P_{t|t-1}C' + R) \\
& \propto \sum_{t=1}^T -\frac{1}{2} \log |C P_{t|t-1}C' + R| - \frac{1}{2}(Y_t- B X_t - C Z_{t|t-1})(CP_{t|t-1}C' + R)^{-1}(Y_t - B X_t - C Z_{t|t-1})
\end{aligned}
$$

---
## Kalman filter : E-Step

The conditional expected values of the states are 

$$
\begin{aligned}
    E_{p(Z|Y,\theta^{(k)})}[z_t|Y_{1...T}] & = z_{t|T} \\
    E_{p(Z|Y,\theta^{(k)})}[z_t z_t'|Y_{1...T}] & = P_{t|T} + z_{t|T} z_{t|T}' \\
    E_{p(Z|Y,\theta^{(k)})}[z_t z_{t+1}'|Y_{1...T}] & = z_{t|t} z_{t+1|T}' + L_t (P_{t+1|T} + (z_{t+1|T}-z_{t+1|t}) z_{t+1|T}')
\end{aligned}
$$


---
## Kalman filter : M-Step

In the M-step of the EM procedure, the objective is to maximize the parameters of the expected joined likelihood of the observations and the states. 

$$
\begin{aligned}
    \textrm{Objective : } \argmax_{\theta^{(k+1)}} E_{p(Z|Y,\theta^{(k)})}[\log p(Y,Z|\theta^{(k+1)})]
\end{aligned}
$$

The joined likelihood of the observations and the states is 

$$
\begin{aligned}
\log p(Y,Z|\theta) & \propto -\frac{1}{2} \log{|\Sigma|} -\frac{1}{2} (z_0 - \mu) \Sigma^{-1} (z_0 - \mu) \\
& \ \ \ \ -\frac{T}{2} \log{|Q|} 
- \frac{1}{2}\sum_{t=0}^{T-1} (z_{t+1} - A z_t)' Q^{-1} (z_{t+1} - A z_t) \\
& \ \ \ \ -\frac{T}{2} \log{|R|} - \frac{1}{2}\sum_{t=1}^{T} (Y_t - B X_t - C z_t)' R^{-1} (Y_t - B X_t - C z_t) \\
\end{aligned}
$$

The following matrices will be used to reduce the length of the expressions later

$$
\begin{aligned}
    D & = \sum_{t=0}^{T-1} P_{t|T} + z_{t|T} z_{t|T}' \\
    E & = \sum_{t=0}^{T-1} P_{t+1,t|T} + z_{t+1|T} z_{t|T}'\\
    F & = \sum_{t=0}^{T-1} P_{t+1|T} + z_{t+1|T} z_{t+1|T}' \\
    H & = \sum_{t=1} ^{T} Y_{t} Y_t' \\
    M & = \sum_{t=1}^T Y_t Z_{t|T}'\\
    N & = \sum_{t=1}^T Y_t X_t' \\
    P & = \sum_{t=1}^T X_t X_t' \\
    V & = \sum_{t=1}^T X_t Z_{t|T}'
\end{aligned}
$$

Without $B$ and $X_t$
$$
\begin{aligned}
E[L] & \propto -\frac{1}{2} \log{|\Sigma|} - \frac{1}{2} tr \Big( \Sigma^{-1} (P_{0|T} + (z_{0|T} - \mu) (z_{0|T} - \mu)' \Big)  \\
& \ \ \ \ -\frac{T}{2} \log{|Q|} 
- \frac{1}{2} tr(Q^{-1} (F-EA'-AE'+ADA') )  \\
& \ \ \ \ -\frac{T}{2} \log{|R|} - \frac{1}{2} tr\Big(R^{-1}\big( H  - M C'  - C M' + C F C'\big) \Big) 
\end{aligned}
$$

With $B$ and $X_t$
$$
\begin{aligned}
E_{p(Z|Y,\theta^{(k)})}[Z|Y_{1...T}] & \propto -\frac{1}{2} \log{|\Sigma|} - \frac{1}{2} tr \Big( \Sigma^{-1} (P_{0|T} + (z_{0|T} - \mu) (z_{0|T} - \mu)' \Big)  \\
& \ \ \ \ -\frac{T}{2} \log{|Q|} 
- \frac{1}{2} tr(Q^{-1} (F-EA'-AE'+ADA') )  \\
& \ \ \ \ -\frac{T}{2} \log{|R|} - \frac{1}{2} tr\Big(R^{-1}\big( H - N B' - M C' - B N' + B P B' + B V C' - C M' + C V' B' + C F C'\big) \Big) 
\end{aligned}
$$

<!-- $$
\begin{aligned}
& - (M-B V) C' - C (M' - V' B') + C F C' \\
&  B (V C'-N') + B P B' + (C V'-N) B'
\end{aligned}
$$ -->

<!-- $$
\begin{aligned}
    Y_t Y_T' - Y_t X_t' B' - Y_t Z_t' C' - B X_t Y_t' + B X_t X_t' B' + B X_t Z_t' C' - C Z_t Y_t' + C Z_t X_t' B' + C Z_t Z_t' C'
\end{aligned}
$$ -->

### Case I : Optimize A, C, Q and R
Without B
$$
\begin{aligned}
A & = E D^{-1} \\
Q & = \frac{1}{T} (F - E D^{-1} E') \\
C & = M F^{-1}\\
R & = \frac{1}{T} (H - M F^{-1} M')
\end{aligned}
$$
With B
$$
\begin{aligned}
A & = E D^{-1} \\
Q & = \frac{1}{T} (F - E D^{-1} E') \\
C & = (M - BV) F^{-1}\\
B & = (N - C V') P^{-1}\\
\end{aligned}
$$

### Case II : Optimize Q and R

$$
\begin{aligned}
Q & = \frac{1}{T} (F-EA'-AE'+ADA') \\
R & = \frac{1}{T} (H - M C' - C M'+ C F C')
\end{aligned}
$$
With B
$$
\begin{aligned}
R & = \frac{1}{T} (H - N B' - M C' - B N' + B P B' + B V C' - C M' + C V' B' + C F C')
\end{aligned}
$$

## Constraint optimization

- ### Case I : B weights sum to zero

Set
$$
\begin{aligned}
    S_B & = \iota_I \\
    W_B & = 0 * \iota_n
\end{aligned}
$$

Optimize
$$
\begin{aligned}
\argmin_B &\ tr(R^{-1}(B (V C'-N') + B P B' + (C V'-N) B')\ s.t.\ S_B' B'  = W_B' \\
L & = tr(R^{-1}(B (V C'-N') + B P B' + (C V'-N) B') + (S_B' B'-W_B') \lambda \\
\frac{\partial L}{\partial B'} & = 2(VC' - N')R^{-1} + 2 P B' R^{-1} + S_B \lambda' = 0 \\
& = 2(VC' - N') + 2 P B' + S_B \lambda' R = 0\\
B' & = \frac{1}{2} P^{-1}( - 2(V C' - N') - S_B \lambda' R)\\
S_B' B' & = \frac{1}{2} S_B' P^{-1}( - 2(V C' - N') - S_B \lambda' R) = W_B'\\
S_B' P^{-1} S_B \lambda' R & = -2 S_B' P^{-1}(V C' - N') - 2 W_B' \\
\lambda' & = -\frac{2}{S_B' P^{-1} S_B} \big(S_B' P^{-1}(V C' - N') + W_B\big)R^{-1} 
\end{aligned}
$$

- ### Case II : weights C sum to 1

Set
$$
\begin{aligned}
    S_C & = \iota_Z \\
    W_C & = \iota_n
\end{aligned}
$$

Optimize
$$
\begin{aligned}
\argmin_B &\ tr(R^{-1}(C (V' B' - M') + C F C' + (B V - M) C')\ s.t.\ S_C' C'  = W_C' \\ 
L & = tr(R^{-1}(C (V' B' - M') + C F C' + (B V - M) C') + (S_C' C'-W_C') \lambda \\
\frac{\partial L}{\partial C'} & = 2(V'B' - M')R^{-1} + 2 F C' R^{-1} + S_C \lambda' = 0 \\
& = 2(V'B' - M') + 2 F C' + S_C \lambda' R = 0\\
C' & = \frac{1}{2} F^{-1}( - 2(V'B' - M') - S_C \lambda' R)\\
S_C' C' & = \frac{1}{2} S_C' F^{-1}( - 2(V'B' - M') - S_C \lambda' R) = W_C'\\
S_C' F^{-1} S_C \lambda' R & = -2 S_C' F^{-1}(V' B' - M') - 2 W_C'\\
\lambda' & = -\frac{2}{S_C' P^{-1} S_C} \big(S_C' F^{-1}(V' B' - M') + W_C'\big) R^{-1}
\end{aligned}
$$

### References
- [kalman smoother](http://arl.cs.utah.edu/resources/EM%20Algorithm.pdf)
- [EM Algorithm](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=7585AADC9216B1BB702D0A88E4E7A2A8?doi=10.1.1.644.1894&rep=rep1&type=pdf)
- [Full EM Algorithm](https://aura.abdn.ac.uk/bitstream/handle/2164/4320/emfitAMCSubmission.pdf;jsessionid=B88F649E8D52A36FD2B9CAC54E689AE3?sequence=1)
- [1960 Kalman Filter](https://www.cs.unc.edu/~welch/kalman/kalmanPaper.html)
- [Derivation Smoothing](https://users.ece.cmu.edu/~byronyu/papers/derive_ks.pdf)
- [KL Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
- [ELBO](https://minhuanli.github.io/notes/stat/elbo/)
- [q distribution](https://zhiyzuo.github.io/EM/#derivation)
- [lagrange inverse](https://people.duke.edu/~hpgavin/cee201/LagrangeMultipliers.pdf)
- [KalmanFilter and Neural Networks](http://booksbw.com/books/mathematical/hayking-s/2001/files/kalmanfilteringneuralnetworks2001.pdf)