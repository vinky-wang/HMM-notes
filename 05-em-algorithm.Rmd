# Expectation-Maximization Algorithm (Baum-Welch) {#em}

The EM algorithm is an alternative to [direct numerical maximization of the likelihood](#numerical) to obtain MLEs. In the context of HMMs, the EM algorithm is known as the Baum-Welch algorithm, which uses [forward and backward probabilities](#fbalg). See Section 4.4 of the textbook for a discussion of the advantages and disadvantages of the EM algorithm versus direct numerical maximization. 

## EM Algorithm (General)

The EM algorithm is an iterative method for performing maximum likelihood estimation when there are missing data. It repeats the following steps until some convergence criterion has been satisfied:

- **E step**: Compute the **conditional expectations** of the "missing" data given the observations and given the current estimate of the parameters of interest $\boldsymbol{\theta}$

- **M step**: Replace functions of the missing data with their conditional expectations in the complete-data log-likelihood (CDLL) (i.e. log-likelihood of the observations and the missing data). **Maximize** the CDLL with respect to the parameters of interest $\boldsymbol{\theta}$. 


## EM Algorithm (for HMMs)

For HMMs, the sequence of states $c_1, c_2, \dots, c_T$ occupied by the MC is treated as the missing data. For convenience, we represent this by the zero-one random variables defined as follows:

$$u_j (t) = 1 \qquad{\text{if and only if }} \qquad{} c_t = j \qquad{t=1, 2, \dots, T}$$

$$v_{jk} (t) = 1 \qquad{\text{if and only if }} \qquad{} c_{t-1} = j \text{ and } c_t=k\qquad{t=2, 3, \dots, T}$$


Then the CDLL of an HMM is given by 

\begin{align}
\log \left(\Pr(\boldsymbol{x}^{(T)}, \boldsymbol{c}^{(T)}) \right)
&= \log \left(\delta_{c_1} \prod_{t=2}^T \gamma_{c_{t-1}, c_t} \prod_{t=1}^T p_{c_t} (x_t) \right)\\
&= \log \delta_{c_1} + \sum_{t=2}^T \log \gamma_{c_{t-1}, c_t} + \sum_{t=1}^T \log p_{c_t} (x_t)\\
&= \sum_{j=1}^m u_j (1) \log \delta_j + \sum_{j=1}^m \sum_{k=1}^m \left(\sum_{t=2}^T v_{jk} (t) \right) \log \gamma_{jk} + \sum_{j=1}^m \sum_{t=1}^T u_j (t) \log p_j (x_t) (\#eq:cdll) \\
&= (1)  \qquad{} \qquad{} \qquad{} + (2)  \qquad{}  \qquad{} \qquad{} \qquad{} + 3
\end{align}

**Note:** 

- Term 1 follows from $u_j(1) = \begin{cases} 1 & c_1 = j\\  0 & c_1 \neq j\end{cases}$, so $\sum_j u_j (1) = 1$ and it follows that $\log \delta_{c_1} = \sum_j u_j(1) \log \delta_j$

- Term 2 follows from $v_{jk} = \begin{cases} 1 & c_{t-1} = j, c_t=k\\ 0 & c_{t-1} \neq j, c_t \neq k  \end{cases}$, so $\sum_j \sum_k v_{jk} (t) = 1$ and it follows that $\sum_{t=2}^T \log \gamma_{c_{t-1}, c_t} = \sum_j \sum_k v_{jk} (t) \sum_{t=2}^T \log \gamma_{jk}$

- Term 3 follows from $u_j(1) = \begin{cases} 1 & c_1 = j\\  0 & c_1 \neq j\end{cases}$, so $\sum_j u_j (1) = 1$, so $\sum_j u_j (t) = 1$ and it follows that $\sum_{t=1}^T \log p_{c_t} (x_t) = \sum_j u_j(t) \sum_{t=1}^T \log p_j (x_t)$

**E-Step**:

Compute the **conditional expectations** of the hidden states $c_1, c_2, \dots, c_T$ given the observations $x_1, x_2, \dots, x_T$. That is, replace all quantities $v_{jk} (t)$ and $u_j (t)$ by their conditional expectations given the observations $\boldsymbol{x}^{(T)}$ and current estimate of $\boldsymbol{\theta} = (\boldsymbol{\delta}, \boldsymbol{\Gamma}, \boldsymbol{\lambda})$. 


\begin{equation}
\hat{u}_j (t) = \Pr(C_t = j|\boldsymbol{x}^{(T)}) = \frac{1}{L_T} \alpha_t (j) \beta_t (j) 
(\#eq:uj)
\end{equation}

\begin{equation}
\hat{v}_{jk} (t) = \Pr(C_{t-1} = j, C_t = k|\boldsymbol{x}^{(T)}) = \frac{1}{L_T} \alpha_{t-1} (j) \gamma_{jk} p_k (x_t)  \beta_t (k) 
(\#eq:vjk)
\end{equation}

**Note:** See Equations \@ref(eq:emprepone) and \@ref(eq:empreptwo) . 

**M-Step**:

Replace $v_{jk} (t)$ and $u_j (t)$ by $\hat{v}_{jk} (t)$ and $\hat{u}_j (t)$, then maximize Equation \@ref(eq:cdll) with respect to the parameters of interest $\boldsymbol{\theta}$. 

That is, maximize 

\begin{align}
\log \left( \Pr(\boldsymbol{x}^{(T)}, \boldsymbol{c}^{(T)}) \right)
&= \sum_{j=1}^m \hat{u}_j (1) \log \delta_j + \sum_{j=1}^m \sum_{k=1}^m \left( \sum_{t=2}^T \hat{v}_{jk} (t) \right) \log \gamma_{jk} + \sum_{j=1}^m \sum_{t=1}^T \hat{u}_j (t) \log p_j (x_t)
\end{align}

with respect to $\boldsymbol{\delta}$, $\boldsymbol{\lambda}$, and $\boldsymbol{\Gamma}$.

Notice, term $(1)$, $(2)$, and $(3)$ depend only on $\boldsymbol{\delta}$, $\boldsymbol{\Gamma}$, and $\boldsymbol{\lambda}$, respectively. Effectively, we are performing the three maximizations

1. $\sum_{j=1}^m \hat{u}_j (1) \log \delta_j \qquad{\text{wrt} \qquad{}} \boldsymbol{\delta}$ 

2. $\sum_{j=1}^m \sum_{k=1}^m \left(\sum_{t=2}^T \hat{v}_{jk} (t) \right) \log \gamma_{jk} \qquad{\text{wrt} \qquad{}} \boldsymbol{\Gamma}$

3. $\sum_{j=1}^m \sum_{t=1}^T \hat{u}_j (t) \log p_j (x_t) \qquad{\text{wrt} \qquad{}} \boldsymbol{\lambda}$

The solutions are

1. $\delta_j = \frac{\hat{u}_j (1)}{\sum_{j=1}^m \hat{u}_j (1)} = \hat{u}_j (1)$

2. $\gamma_{jk} = \frac{f_{jk}}{\sum_{k=1}^m f_{jk}}$ where $f_{jk} = \sum_{t=2}^T \hat{v}_{jk} (t)$

3. The maximization may be easy or difficult, depending on the nature of the state-dependent distributions assumed. Closed form solutions exist for the Poisson and Normal distributions, but not for the Gamma and Negative Binomial distributions. When no closed-form solutions exist, numerical maximization will be necessary.


**Note**: The solutions can be derived using Lagrange multipliers. 

For Term 1, the new objective function is 

$$F(\delta_i) = \sum_{j=1}^m \hat{u_j} (1) \log \delta_j - \lambda \left(\sum_{j=1}^m \delta_j - 1\right)$$

Then differentiating with respect to $\delta_i$ and setting to zero

$$\frac{\hat{u_j} (1)}{\delta_j} - \lambda = 0$$

it follows that 

$\frac{\hat{u}_j (1)}{\delta_j} = \lambda$ and $\delta_j = \frac{\hat{u}_j (1)}{\lambda}$

so $\sum_{j=1}^m \frac{\hat{u}_j (1)}{\lambda = 1} \Rightarrow \lambda = \sum_{j=1}^m \hat{u}_j (1)$.

Thus, $\delta_j = \frac{\hat{u}_j (1)}{\sum_{j=1}^m \hat{u}_j (1)} = \hat{u}_j (1)$

For Term 2, the new objective function is 

$$F (\gamma_{ij}) = \sum_{j=1}^m \sum_{k=1}^m \sum_{t=2}^T \hat{v}_{jk} (t) \log \gamma_{jk} - \sum_{k=1}^m \lambda_k \left(\sum_{k=1}^m \gamma_{jk} - 1\right)$$

Then differentiating with respect to $\gamma_{jk}$ and setting to zero

$$\frac{\sum_{t=2}^T \hat{v}_{jk} (t)}{\gamma_{jk}} - \lambda_k = 0$$

it follows that $\frac{\sum_{t=2}^T \hat{v}_{jk} (t)}{\gamma_{jk}} = \lambda_k$ and $\gamma_{jk} = \frac{\sum_{t=2}^T \hat{v}_{jk} (t)}{\lambda_k}$

and since $\sum_{k=1}^m \gamma_{jk} = 1$, it follows that $\sum_{k=1}^m \frac{\sum_{t=2}^T \hat{v}_{jk} (t)}{\lambda_k} = 1 \Rightarrow \lambda_k = \sum_{k=1}^m \sum_{t=2}^T \hat{v}_{jk} (t)$

Thus, $\gamma_{jk} = \frac{\sum_{t=2}^T \hat{v}_{jk}(t)}{\sum_{k=1}^m \sum_{t=2}^T \hat{v}_{jk}(t)} = \frac{f_{jk}}{\sum_{k=1}^m f_{jk}}$ where $f_{jk} = \sum_{t=2}^T \hat{v}_{jk} (t)$. 


For Term 3, 

if the state-dependent distribution is Poisson then $p_j(x_t) = \frac{e^{- \lambda_j} \lambda_j^{x_t}}{x!}$

so,

$$\sum_{j=1}^m \sum_{t=1}^T \hat{u}_j (t) \log \left(\frac{e^{- \lambda_j} \lambda_j^{x_t}}{x!}\right)$$ where 

Then differentiating with respect to $\lambda_j$ and setting to zero, 

\begin{align}
0 
&= \sum_{t=1}^T \hat{u}_j (t) (-1 + \frac{x_t}{\lambda_j})\\
\Rightarrow \sum_{t=1}^T \hat{u}_j (t) 
&= \sum_{t=1}^T \hat{u}_j (t) x_t \frac{1}{\lambda_j}\\
\Rightarrow \hat{\lambda}_j &= \frac{\sum_{t=1}^T \hat{u}_j (t) x_t}{\sum_{t=1}^t \hat{u}_j (t)}
\end{align}

### Stationary Markov Chain

If we assume in addition that the MC is stationary with stationary distribution $\boldsymbol{\delta}$, then the M-Step simplifies to maximizing 

\begin{align}
\sum_{j=1}^m \hat{u}_j (1) \log \delta_j + \sum_{j=1}^m \sum_{k=1}^m \left(\sum_{t=2}^T \hat{v}_{jk} (t) \right) \log \gamma_{jk} + \sum_{j=1}^m \sum_{t=1}^T \hat{u}_j (t) \log p_j (x_t)
\end{align}


**Note**: Even in the case of only two states, analytic maximization would require the solution of a pair of equations quadratic in two variables (See Exercise 2). Often, numerical optimization is needed for this part of the M step if stationarity is assumed, or else some modification of the EM designed to circumvent such numerical optimization. 


## Exercises 

1. a. Suppose $L_i > 0$ for $i = 1, 2, \dots, m$. Maximize $L = \sum_{i=1}^m a_i L_i$ over the region $a_i \geq 0$, $\sum_{i=1}^m a_i = 1$. 

    b. Consider an HMM with initial distribution $\boldsymbol{\delta}$, and consider the (observed-data) likelihood as a function of $\boldsymbol{\delta}$. Show that, at a maximum of the likelihood, $\boldsymbol{\delta}$ is a unit vector. 
  

2. Consider the example of Visser, Raijmakers and Molenaar (2002, pp. 186–187). There a series of length 1000 is simulated from an HMM with states S1 and S2 and the three observation symbols 1, 2 and 3. The transition probability matrix is $\boldsymbol{A} =  \begin{pmatrix} 0.9 & 0.1\\ 0.3 & 0.7 \end{pmatrix}$, the initial probabilities are $\boldsymbol{\pi} = (0.5, 0.5)$, and the state-dependent distribution in state $i$ is row $i$ of the matrix $\boldsymbol{B} = \begin{pmatrix} 0.7 & 0.0 & 0.3\\ 0.0 & 0.4 & 0.6 \end{pmatrix}$

The parameters $\boldsymbol{A}$, $\boldsymbol{B}$, and $\boldsymbol{\pi}$ are then estimated by EM; the estimates of $\boldsymbol{A}$ and $\boldsymbol{B}$ are close to $\boldsymbol{A}$ and $\boldsymbol{B}$, but that of $\boldsymbol{\pi}$ is $(1, 0)$. This estimate of $\boldsymbol{\pi}$ is explained as follows: ‘The reason for this is that the sequence of symbols that was generated actually starts with the symbol 1 which can only be produced from state S1.’

Do you agree with the above statement? What if the probability of symbol 1 in state S2 had been (say) 0.1 rather than 0.0?

3. Consider the fitting by EM of a two-state HMM based on a stationary Markov chain. In the M step, the sum of terms 1 and 2 must be maximized with respect to $\boldsymbol{\Gamma}$. Write term 1 + term 2 as a function of $\gamma_{12}$ and $\gamma_{21}$, the off-diagonal transition probabilities, and differentiate to find the equations satisfied by these probabilities at a stationary point. 

4. Let $\{X_t\}$ be an HMM on $m$ states. Suppose the state-dependent distributions are binomial. More precisely, assume that $$\Pr(X_t = x|C_t = j) = \binom{n_t}{x} p_j^x (1 - p_j)^{n_t - x}$$

Find the value for $p_j$ that will maximize the third term of equation \@ref(eq:cdll). 


## Solutions

**Question 1**

**Part a**

Let $L_j = \max \{L_1, \dots, L_m\}$. 

We will show that $L_j$ is the maximum of $L$. 

Let $a_i \in \{a_1, \dots, a_m\}$. Since $L_j$ is defined as the maximum of $\{L_1, \dots, L_m\}$, $L_i \leq L_j$ for any $i$.

Then $a_i L_i \leq a_i L_j$ since $a_i \geq 0$, and summing over $i$,
$L = \sum_{i=1}^m a_i L_i \leq \sum_{i=}^m a_i L_j = L_j$ since $\sum_{i=1}^m a_i = 1$. 

Thus $L$ is bounded by $L_j$.

Now we will show that there exists a choice of $a_i$ such that $L = L_j$. 

Take $a_i$ to be the $j$-th unit vector. Then $a_i \geq 0$ and $\sum_{i=1}^m a_i = 1$, and $L = \sum_{i=1}^m a_i L_j = L_j$.

Thus, $L = L_j$ when $a_i$ is the $j$-th unit vector.

Therefore, $L_j$ is the maximum of $L$. 


**Part b**

Maximize $\sum_{j=1}^m u_j (1) \log \delta_j$ subject to $\delta_i \geq 0$ and $\sum_j \delta_j = 1$.

Using Lagrange multipliers, the new objective function is 

$$F (\delta_j) = \sum_{j=1}^m u_j (1) \log \delta_j - \lambda \left(\sum_{j=1}^m \delta_j - 1 \right)$$

Then differentiating with respect to $\delta_j$ and setting to zero,

$$\frac{u_j (1)}{\delta_j} - \lambda = 0$$

$\Rightarrow \frac{u_j (1)}{\hat{\delta}_j} = \lambda$ and $\hat{\delta}_j = \frac{u_j (1)}{\lambda}$

so $\sum_{j=1}^m \frac{u_j(1)}{\lambda} \Rightarrow \lambda = \sum_{j=1}^m \hat{u}_j (1)$

Thus, $\delta_j = \frac{u_j (1)}{\sum_{j=1}^m u_j (1)} = u_j (1)$ and $\boldsymbol{\delta}$ is the $j$-th unit vector corresponding to $c_1 = j$. 

**Question 2**

The statement is correct since $\Pr(X_t = 1|C_t = S_1) = 0.7$ whereas $\Pr(X_t = 1|C_t = S_2) = 0.0$.

If $\Pr(X_t = 1|C_t = S_2) = 0.1$, then we would still expect $\boldsymbol{\pi} = (1, 0)$ since it is more likely that at $t=1$, $\Pr(X_t=1|C_t = S_1)$ and for a large number of simulations, $\boldsymbol{\pi} \rightarrow (1, 0)$. That is, the initial distribution converges to the $j$-th unit vector corresponding to $c_1 = j$. 

**Question 3**

\begin{align} 
& \log \left(\Pr\left(\boldsymbol{x}^{(T)}, \boldsymbol{c}^{(T)}\right)\right) \\
& = \hat{u}_1(1) \log \left(\frac{\gamma_{21}}{\gamma_{12}+\gamma_{21}}\right)+\hat{u}_{2} (1) \log \left(\frac{\gamma_{21}}{\gamma_{12}+\gamma_{21}}\right) \\ 
& +\hat{v}_{11}(1) \log \left(1-\gamma_{12}\right)+\hat{v}_{12}(1) \log \left(\gamma_{12}\right) \\ 
& +\hat{v}_{21}(1) \log \left(\gamma_{21}\right)+\hat{v}_{22}(1) \log \left(1-\gamma_{21}\right) \\ 
& +\hat{v}_{11}(2) \log \left(1-\gamma_{12}\right)+\hat{v}_{12}(2) \log \left(\gamma_{12}\right) \\ 
& +\hat{v}_{21}(2) \log \left(\gamma_{21}\right)+\hat{v}_{22}(2) \log \left(1-\gamma_{21}\right) \\
& = \hat{u}_1(1)\left[\log \left(\gamma_{21}\right)-\log \left(\gamma_{12}+\gamma_{21}\right)\right] +\hat{u}_2(1)\left[\log \left(\gamma_{21}\right)-\log \left(\gamma_{12}+\gamma_{21}\right)\right] \\
& +\hat{v}_{11}(1) \log \left(1-\gamma_{12}\right)+\hat{v}_{12}(1) \log \left(\gamma_{12}\right) \\ 
& +\hat{v}_{21}(1) \log \left(\gamma_{21}\right)+\hat{v}_{22}(1) \log \left(1-\gamma_{21}\right) \\ 
& +\hat{v}_{11}(2) \log \left(1-\gamma_{12}\right)+\hat{v}_{12}(2) \log \left(\gamma_{12}\right) \\ 
& +\hat{v}_{21}(2) \log \left(\gamma_{21}\right)+\hat{v}_{22}(2) \log \left(1-\gamma_{21}\right) \\
(\#eq:q3)
\end{align}


Then differentiating Equation \@ref(eq:q3) with respect to $\gamma_{12}$ and setting to zero, it follows that 

$$\frac{- \hat{u}_1 (1)}{\gamma_{12} + \gamma_{21}} - \frac{\hat{u}_2 (1)}{\gamma_{12} + \gamma_{21}} - \frac{\hat{v}_{11} (1)}{1 - \gamma_{12}} + \frac{\hat{v}_{12} (1)}{\gamma_{12}} - \frac{\hat{v}_{11} (2)}{1 - \gamma_{12}} + \frac{\hat{v}_{12} (2)}{\gamma_{12}} = 0$$

and multiplying each term by $(\gamma_{12} + \gamma_{21})(1-\gamma_{12})(\gamma_{12})$, 

\begin{align}
& -\hat{u}_1 (1) (1 - \gamma_{12}) (\gamma_{12}) - \hat{u}_2 (1) (1 - \gamma_{12}) (\gamma_{12}) \\
& - \hat{v}_{11} (1) (\gamma_{12} + \gamma_{21}) (\gamma_{12}) + \hat{v}_{12} (1) (\gamma_{12} + \gamma_{21}) (1 - \gamma_{12}) \\
& - \hat{v}_{11} (2) (\gamma_{12} + \gamma_{21}) (\gamma_{12}) + \hat{v}_{12} (2) (\gamma_{12} + \gamma_{21}) (1 - \gamma_{12}) \\
& = 0 \\
\end{align}

\begin{align}
& \Rightarrow - \hat{u}_1 (1) \gamma_{12} + \hat{u}_1 (1) \gamma_{12}^2 - \hat{u}_2 (1) \gamma_{12} + \hat{u}_2 (1) \gamma_{12}^2 \\
& - \hat{v}_{11} (1) \gamma_{12}^2 - \hat{v}_{11} (1) \gamma_{12} \gamma_{21} + \hat{v}_{12} (1) \gamma_{12} + \hat{v}_{12} (1) \gamma_{21} - \hat{v}_{12} (1) \gamma_{12}^2 - \hat{v}_{12} (1) \gamma_{12}^2 \\
& - \hat{v}_{11} (2) \gamma_{12}^2 - \hat{v}_{11} (2) \gamma_{12} \gamma_{21} + \hat{v}_{12} (2) \gamma_{12} + \hat{v}_{12} (2) \gamma_{21} - \hat{v}_{12} (2) \gamma_{12}^2 - \hat{v}_{12} (2) \gamma_{12}^2\\
&= 0\\
\end{align}

\begin{align}
& \Rightarrow \gamma_{12}^2 [\hat{u}_1 (1) + \hat{u}_2 (1) - \hat{v}_{11} (1) - \hat{v}_{12} (1) - \hat{v}_{12} (1) - \hat{v}_{11} (2) - \hat{v}_{12} (2) - \hat{v}_{12} (2)]\\
& + \gamma_{12} [- \hat{u}_{1} (1) - \hat{u}_2 (1) - \hat{v}_{11} (1) \gamma_{21} + \hat{v}_{12} (1) - \hat{v}_{11} (2) \gamma_{21} + \hat{v}_{12} (2)]\\
& + \hat{v}_{12} (1) \gamma_{21} + \hat{v}_{12} (2) \gamma_{21}\\
&= 0
\end{align}

\begin{align}
& \Rightarrow \gamma_{12}^2 [\hat{u}_1 (1) + \hat{u}_2 (1) - \hat{v}_{11} (1) - 2 \hat{v}_{12} (1) - \hat{v}_{11} (2) - 2 \hat{v}_{12} (2)]\\
& + \gamma_{12} [- \hat{u}_{1} (1) - \hat{u}_2 (1) - \hat{v}_{11} (1) \gamma_{21} + \hat{v}_{12} (1) - \hat{v}_{11} (2) \gamma_{21} + \hat{v}_{12} (2)]\\
& + \hat{v}_{12} (1) \gamma_{21} + \hat{v}_{12} (2) \gamma_{21}\\
&= 0
\end{align}

And differentiating Equation \@ref(eq:q3) with respect to $\gamma_{21}$ and setting to zero, it follows that 

$$\frac{\hat{u}_1 (1)}{\gamma_{21}} - \frac{\hat{u}_1 (1)}{\gamma_{12} + \gamma_{21}} + \frac{\hat{u}_2 (1)}{\gamma_{21}} - \frac{\hat{u}_2 (1)}{\gamma_{12} + \gamma_{21}} + \frac{\hat{v}_{21} (1)}{\gamma_{21}} - \frac{\hat{v}_{22} (1)}{1 - \gamma_{21}} + \frac{\hat{v}_{21} (2)}{\gamma_{21}} - \frac{\hat{v}_{22} (2)}{1 - \gamma_{21}} =0$$

and multiplying each term by $(\gamma_{12} + \gamma_{21})(1 - \gamma_{21})(\gamma_{21})$


\begin{align}
& \hat{u}_1 (1) (\gamma_{12} + \gamma_{21}) (1 - \gamma_{21}) - \hat{u}_1 (1) (1 - \gamma_{21}) (\gamma_{21}) + \hat{u}_2 (1) (\gamma_{12} + \gamma_{21}) (1 - \gamma_{21})\\
& - \hat{u}_2 (1) (1 - \gamma_{21}) (\gamma_{21}) + \hat{v}_{21} (1) (\gamma_{12} + \gamma_{21}) (1 - \gamma_{21})\\
& - \hat{v}_{22} (1) (\gamma_{12} + \gamma_{21}) (\gamma_{21}) + \hat{v}_{21} (2) (\gamma_{12} + \gamma_{21}) (1 - \gamma_{21})\\
& - \hat{v}_{22} (2) (\gamma_{12} + \gamma_{21}) (\gamma_{21})\\
&=0
\end{align}

\begin{align}
& \Rightarrow \hat{u}_1 (1) \gamma_{12} + \hat{u}_1 (1) \gamma_{21} - \hat{u}_1 (1) \gamma_{12} \gamma_{21} - \hat{u}_1 (1) \gamma_{21}^2\\
&- \hat{u}_1 (1) \gamma_{21} + \hat{u}_1 (1) \gamma_{21}^2 \\
&+ \hat{u}_2 (1) \gamma_{12} + \hat{u}_2 (1) \gamma_{21} - \hat{u}_2 (1) \gamma_{12} \gamma_{21} - \hat{u}_2 (1) \gamma_{21}^2\\
&- \hat{u}_2 (1) \gamma_{21} + \hat{u}_2 (1) \gamma_{21}^2\\
&+ \hat{v}_{21} (1) \gamma_{12} + \hat{v}_{21} (1) \gamma_{21} - \hat{v}_{21} (1) \gamma_{12} \gamma_{21} - \hat{v}_{21} (1) \gamma_{21}^2\\
&= - \hat{v}_{22} (2) \gamma_{12} \gamma_{21} - \hat{v}_{22}(2) \gamma_{21}^2\\
&=0\\
\end{align}

\begin{align}
&\Rightarrow \gamma_{21}^2 [-\hat{u}_1 (1) + \hat{u}_1 (1) - \hat{u}_2 (1) + \hat{u}_2 (1) - \hat{v}_{21} (1) - \hat{v}_{22} (1) - \hat{v}_{21} (1) - \hat{v}_{22} (2)]\\
& + \gamma_{21} [\hat{u}_1 (1) - \hat{u}_1 (1) \gamma_{21} - \hat{u}_1 (1) + \hat{u}_2 (1) - \hat{u}_2 (1) \gamma_{12} - \hat{u}_2 (1)\\
&+ \hat{v}_{21} (1) - \hat{v}_{21} (1) \gamma_{12} - \hat{v}_{22} (1) \gamma_{12} + \hat{v}_{21} (2) - \hat{v}_{21} (1) \gamma_{12} - \hat{v}_{22} (2) \gamma_{12}]\\
&+ \hat{u}_1 (1) \gamma_{12} + \hat{u}_2 (1) \gamma_{12} + \hat{v}_{21} \gamma_{12} + \hat{v}_{21} (2) \gamma_{12}\\
&=0\\
\end{align}

\begin{align}
& \Rightarrow \gamma_{21}^2 [-2 \hat{v}_{21} (1) - \hat{v}_22 (1) - \hat{v}_{22} (2)]\\
&+ \gamma_{21} [- \hat{u}_1 (1) \gamma_{12} - \hat{u}_2 (1) \gamma_{12} + \hat{v}_{21} (1) - \hat{v}_{21} (1) \gamma_{12} - \hat{v}_{22} (1) \gamma_{12} + \hat{v}_{21} (2) - \hat{v}_{21} (1) \gamma_{12} - \hat{v}_{22} (2) \gamma_{12}]\\
&+ \hat{u}_1 (1) \gamma_{12} + \hat{u}_2 (1) \gamma_{12} + \hat{v}_{21} \gamma_{12} + \hat{v}_{21} (2) \gamma_{12}\\
&=0\\
\end{align}

**Question 4**

\begin{align}
&\left[\sum_{j=1}^m \sum_{t=1}^T \hat{u}_j (t) \log \left( \binom{n_t}{x_t} p_j^{x_t} (1 - p_j)^{n_t - x_t} \right)\right] \frac{d}{d p_j}\\
&= \left[\sum_{j=1}^m \sum_{t=1}^T \hat{u}_j (t) [\log \binom{n_t}{x_t} + x_t \log p_j + (n_t - x_t) \log (1 - p_j)\right] \frac{d}{d p_j}\\
&= \sum_{t=1}^T \hat{u}_j (t) \left[\frac{x_t}{p_j} - \frac{n_t - x_t}{1 - p_j}\right]\\
\end{align}

Setting to zero and solving for $p_j$

\begin{align}
& 0=\sum_{t=1}^T \hat{u}_j(t)\left[\frac{x_t}{p_j}-\frac{n_t-x_t}{1-p_j}\right] \\
& =\sum_{t=1}^T \hat{u}_j(t) \frac{x_t}{p_j}-\sum_{t=1}^{T} \hat{u}_j(t) \frac{n_t-x_t}{1-p_j} \\
& \sum_{t=1}^{T} \hat{u}_j(t) \frac{n_t-x_t}{1-p_j}=\sum_{t=1}^T \hat{u}_j(t) \frac{x_t}{p_j} \\
& \frac{\sum_{t=1}^T \hat{u}_j(t)\left(n_t-x_t\right)}{\sum_{t=1}^T \hat{u}_j(t) x_t}=\frac{1}{p_j}-1 \\
& \frac{\sum_{t=1}^T \hat{u}_j(t)\left(n_t-x_i\right)}{\sum_{i=1}^T \hat{u}_j(t) x_t}+1=\frac{1}{p_j} \\
& \frac{\sum_{t=1}^{T} \hat{u}_j(t)\left(n_t-x_t\right)+\sum_{t=1}^{T} \hat{u}_j(t) x_t}{\sum_{t=1}^{T} \hat{u}_j(t) x_t}=\frac{1}{p_j} \\
& \frac{\sum_{t=1}^T \hat{u}_j(t) x_t}{\sum_{t=1}^T \hat{u}_j(t)\left(n_t-x_i\right)+\sum_{t=1}^T \hat{u}_j(t) x_t}=p_j \\
\end{align}









