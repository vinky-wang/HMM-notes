# Preliminaries {#prelim}

We cover 

1. [(Finite) Mixture distributions](#mix), which is the structure of the marginal distribution of an HMM

2. [Markov chains](#mc), which is the structure of the underlying 'parameter process' of an HMM 

## Independent Mixture Models {#mix}

A (finite) **independent mixture distribution** consists of $m$ finitely many component distributions with *probability mass* (pmf) or *density* (pdf) functions $p_1, \dots, p_m$ with probabilities $\delta_1, \dots, \delta_m$ of being “active” that is determined by a discrete random variable $C$ which performs the “mixing”.

That is,

\begin{equation} 
  C = \left\{\begin{aligned}
  &1 && \delta_1\\
  &2 && \delta_2\\
  &\vdots && \vdots\\
  &m && \delta_m = 1 - \delta_1 - \delta_2 - \cdots - \delta_m\\
\end{aligned}
\right.
\end{equation} 

determines which component is active, then the observations are generated from the active component density.

**Note:** See Figure 1.3 of the textbook as an example. 


### Properties

Let $X$ denote the discrete random variable which has the mixture distribution. Let $Y_i$ denote the random variable with pmf $p_i$. $X$ has the following properties

**probability mass function**

\begin{align}
  p(x) 
  &= \sum_{i=1}^m \Pr(X = x | C=i) \Pr(C=i)\\
  &= \sum_{i=1}^m \delta_i p_i(x)
  (\#eq:pmf)
\end{align}

**expectation**

\begin{align} 
  E(X) 
  &= \sum_{i=1}^m \Pr(C=i) E(X| C = i)\\
  &= \sum_{i=1}^m \delta_i E (Y_i)
  (\#eq:exp)
\end{align}

**k-th moment about the origin**

\begin{align} 
  E(X^k) 
  &= \sum_{i=1}^m \delta_i E (Y_i^k)
  \qquad{k = 1, 2, \dots} (\#eq:moment)
\end{align}

Notice, the above are weighings of the $i$-th component with weights $\delta_i$.


**Note:** The continuous case is analagous. 


### Parameter Estimation

The **likelihood** is given by 

\begin{equation}
L(\boldsymbol{\theta_1} , \dots, \boldsymbol{\theta_m} , \delta_1, \dots , \delta_m | x_1, \dots , x_n) = \prod_{j=1}^n \sum_{i=1}^m \delta_i p_i (x_j, \boldsymbol{\theta_i} )
(\#eq:likelihood)
\end{equation}

where $\boldsymbol{\theta_1} , \dots, \boldsymbol{\theta_m}$ are the parameter vectors of the component distributions, $\delta_1, \dots , \delta_m$ are the mixing parameters, and $x_1, \dots , x_n$ are the $n$ observations.

It is often not possible to analytically maximize the likelihood (i.e. no closed form solution), so numerical maximization or the EM algorithm is used instead. 

**Note:** See Section 1.2.3 of the textbook for a discussion of unbounded likelihood in mixtures of continuous distributions. 



## Markov Chains {#mc}

A **Markov Chain** (MC) is a sequence of random variables $\{C_t: t \in \mathbb{N}\}$ that satisfy the *Markov property* for all $t \in \mathbb{N}$

$$\Pr(C_{t+1}|C_t, \dots, C_1) = \Pr(C_{t+1}|C_t)$$

or for compactness, 

$$\Pr(C_{t+1}| \boldsymbol{C}^{(t)} ) = \Pr(C_{t+1}|C_t) \qquad{\text{where } \boldsymbol{C}^{(t)} = (C_1, C_2, \dots, C_t)}$$

That is, conditioning on the 'history' of the process up to time $t$ is equivalent to conditioning only on the most recent value $C_t$; so the past and future are dependent only through the present.

### Probabilities

The **transition probabilities** are the probabilities of the MC transitioning to state $j$ in $t$ steps given that it was in state $i$

$$\gamma_{ij} = \Pr (C_{s+t}=j|C_s =i)$$

The MC is **homogenous** when $\gamma_{ij} (t)$ depend only on the step size and not on $s$. 

The **unconditional probabilities** $\Pr (C_t = j)$ of a MC being in a given state at a given time $t$ are often denoted by the row vector 

$$\boldsymbol{u} (t) = (\Pr(C_t = 1), \dots, \Pr(C_t = m)) \qquad{\text{t} \in \mathbb{N}}$$

where $u(1)$ is the **initial distribution** of the MC.

### Stationary Distribution

A **stationary distribution** $\boldsymbol{\delta}$ of the MC satisfy 
$$\boldsymbol{\delta \Gamma} = \boldsymbol{\delta}$$ and $$\boldsymbol{\delta 1'} = 1$$

An ***irreducible***^[The textbook does not formally introduce irreducibility, so we refer to @Lalley], *homogenous, discrete-time, finite state-space* MC has a unique, strictly positive, stationary distribution. If the MC is also ***aperiodic*** (returns back to the state at irregular time steps), then the MC has a unique limiting distribution which is precisely the stationary distribution. 

**Note:** A MC that started from its stationary distribution will continue to have that stationary distribution at all subsequent time points. This process is referred to as a "stationary Markov chain" in the textbook, but others may refer to "stationary" as how we defined homogenous here. 

The stationary distribution can be computed by solving 
\begin{equation}
\boldsymbol{\delta} = \boldsymbol{1} (\boldsymbol{I}_m - \boldsymbol{\Gamma} + \boldsymbol{U})^{-1}
(\#eq:stationary)
\end{equation}


where $\boldsymbol{U}$ is a square matrix of ones and $\boldsymbol{I}_m$ is an $(m \times m)$ identity matrix.

**Note:** We will see in later sections how the assumption of stationary simplifies computation. 


### Transition Probabilities Estimation {#tp}

The transition probabilities can be estimated using the relative frequencies of transitions between states. 

Let $f_{ij}$ denote the number of transitions observed from state $i$ to state $j$. Let $c_1, c_2, \dots, c_T$ be a realization of an $m-$state MC $\{C_t\}$. 


#### Non-stationary case

The likelihood of the transition probabilities conditioned on the first observation is 

$$L(\gamma_{ij}|c_1) = \prod_{i=1}^m \prod_{j=1}^m \gamma_{ij}^{f_{ij}}$$

and the log-likelihood is 

$$l (\gamma_{ij}|c_1) = \sum_{i=1}^m \left( \sum_{j=1}^m f_{ij} \log \gamma_{ij} \right)$$

Let $l_i = \sum_{j=1}^m f_{ij} \log \gamma_{ij}$. Let $\gamma_{ii} = 1 - \sum_{k \neq i} \gamma_{ik}$. 

Then we can maximize $l$ by maximizing each $l_i$ separately with respect to the transition probabilities, in terms of the off-diagonals. 

That is, 

substitute the off-diagonal transition probabilities

$$l(\gamma_{ij}|c_1) = \sum_{i=1}^m f_{ii} \log (1 - \sum_{k \neq i} \gamma_{ik}) + \sum_{i \neq j} f_{ij} \log \gamma_{ij}$$

differentiate with respect to $\gamma_{ij}$

\begin{align*}
\frac{dl}{d \gamma_{ij}} 
&= \frac{-f_{ii}}{1 - \sum_{k \neq i} \gamma_{ik}} + \frac{f_{ij}}{\gamma_{ij}}\\
&= \frac{- f_{ii}}{\gamma_{ii}} + \frac{f_{ij}}{\gamma_{ij}}
\end{align*}

set to zero

\begin{equation*}
0 = - \frac{f_{ii}}{\gamma_{ii}} + \frac{f_{ij}}{\gamma_{ij}}\\
\frac{f_{ii}}{\gamma_{ii}} = \frac{f_{ij}}{\gamma_{ij}}\\
f_{ii} \gamma_{ij} = f_{ij} \gamma_{ii} \qquad{\text{for } \gamma_{ii}, \gamma_{ij}  \neq 0}
\end{equation*}

Then summing over $j$,

\begin{equation*}
\sum_{j=1}^m f_{ij}  \gamma_{ii} =  \sum_{j=1}^m f_{ii}  \gamma_{ij}\\
\gamma_{ii} \sum_{j=1}^m f_{ij} = f_{ii} \qquad{\text{since } \sum_{j} \gamma_{ij} = 1}
\end{equation*}

it follows that 

$\gamma_{ii} = \frac{f_{ii}}{\sum_{j=1}^m f_{ij}}$ and $\gamma_{ij} = \frac{f_{ij} \gamma_{ii}}{f_{ii}} = \frac{f_{ij} f_{ii}}{f_{ii} \sum_{j=1}^m f_{ij}} = \frac{f_{ij}}{\sum_{j=1}^m f_{ij}}$

Therefore, the conditional maximum likelihood estimator of the transition probabilities, assuming non-stationarity, is given by $\hat{\gamma_{ij}} = \frac{f_{ij}}{\sum_{k=1}^m f_{ik}} \qquad{(i, j = 1, \dots, m)}$. 


#### Stationary case

Let $\delta_{c_1}$ be the stationary distribution.

The likelihood of the transition probabilities is 

$$L(\gamma_{ij}) = \delta_{c_1} \prod_{i=1}^m \prod_{j=1}^m \gamma_{ij}^{f_{ij}}$$

and the log-likelihood is

$$l (\gamma_{ij}) = \log \delta_{c_1} + \sum_{i=1}^m \sum_{j=1}^m f_{ij} \log \gamma_{ij}$$

We will maximize $l$ subject to $\gamma_{ij}$ using Lagrange multipliers this time with the $m$ constraints $\sum_{j} \gamma_{ij} = 1$. 

That is,

write the new objective function as

$$F(\gamma_{ij}) = \log \delta_{c_1} + \sum_{i=1}^m \sum_{j=1}^m f_{ij} \log \gamma_{ij} - \sum_{i = 1}^m \lambda_i \left( \sum_{j=1}^m \gamma_{ij}  -1 \right)$$

differentiate with respect to $\gamma_{ij}$

$$\frac{dF}{d \gamma_{ij}} = \frac{f_{ij}}{\gamma_{ij}} - \lambda_i$$

set to zero

\begin{equation*}
0 = \frac{f_{ij}}{\gamma_{ij}} - \lambda_i\\
\end{equation*}

It follows that $\lambda_i = \frac{f_{ij}}{\gamma_{ij}}$ and $f_{ij} = \frac{\gamma_{ij}}{\lambda_i}$

Solving for $\lambda_i$

\begin{equation*}
\sum_{j=1}^m \gamma_{ij} = \sum_{j=1}^m \frac{f_{ij}}{\lambda_i} = 1\\
\sum_{j=1}^m f_{ij} = \lambda_i
\end{equation*}

it follows that $\hat{\gamma}_{ij} = \frac{f_{ij}}{\lambda_i} = \frac{f_{ij}}{\sum_{j=1}^m f_{ij}}$.

Therefore, the unconditional maximum likelihood estimator of the transition probabilities, assuming stationarity, is given by $\hat{\gamma}_{ij} =  \frac{f_{ij}}{\sum_{j=1}^m f_{ij}}$. 


## Exercises

Prove 1-4.

1. All finite state-space homogenous Markov chains satisfy the **Chapman-Kolmogorov equations** 

\begin{align}
\boldsymbol{\Gamma}(t+u) = \boldsymbol{\Gamma}(t) \boldsymbol{\Gamma}(u)
(\#eq:ck)
\end{align}

2. It follows from \@ref(eq:ck) that $\boldsymbol{\Gamma} (t) = \boldsymbol{\Gamma} (1)^t$. That is, the matrix of $t-$step transition probabilities is the $t$-th power of the matrix of one-step transition probabilities. 

3. To deduce the distribution at time $t+1$ from that at $t$, we postmultiply by the transition probability matrix $\boldsymbol{\Gamma}$. That is,

\begin{equation}
  \boldsymbol{u} (t+1) = \boldsymbol{u} (t) \boldsymbol{\Gamma}
  (\#eq:post)
\end{equation}

4. The vector $\boldsymbol{\delta}$ with non-negative elements is a stationary distribution of the MC with tpm $\boldsymbol{\Gamma}$ if and only if $\boldsymbol{\delta} ( \boldsymbol{I_m} - \boldsymbol{\Gamma} + \boldsymbol{U}) = \boldsymbol{1}$ where $\boldsymbol{1}$ is a row vector of ones, $\boldsymbol{I_m}$ is the $m \times m$ identity matrix, and $\boldsymbol{U}$ is the $m \times m$ matrix of ones. 


Solve 5-7.

5. Suppose that the sequence of rainy and sunny days is such that each day's weather depends only on the previous day's, and the transition probabilities are given by

|           | day t+1 |       |
|-----------|---------|-------|
| **day t** | rainy   | sunny |
| rainy     | 0.9     | 0.1   |
| sunny     | 0.6     | 0.4   |


Suppose that today's weather (t=1) is sunny. What is the distribution of today's weather and tomorrow's weather? What is the stationary distribution?

6. Let $X$ be a random variable which is distributed as a $(\delta_1, \delta_2)-$ mixture of two distributions with expectations $\mu_1, \mu_2$ and variances $\sigma_1, \sigma_2$, respectively, where $\delta_1 + \delta_2 = 1$. Show that $Var(X) = \delta_1 \sigma_1^2 + \delta_2 \sigma_2^2 + \delta_1 \delta_2 (\mu_1 + \mu_2)^2$
    
7. Consider a stationary two-state Markov chain with transition probability matrix given by 
  $$\boldsymbol{\Gamma} = \left(\begin{array}{cc}
  \gamma_{11} & \gamma_{12}\\
  \gamma_{21} & \gamma_{22}
  \end{array}\right)
  $$
  
    a. Show that the stationary distribution is $$(\delta_1, \delta_2) = \frac{1}{\gamma_{12} + \gamma_{21}} (\gamma_{21}, \gamma_{12})$$
  
    b. Consider the case   
  
    $$\boldsymbol{\Gamma} = 
    \left(\begin{array}{cc}
    \gamma_{11} & \gamma_{12}\\
    \gamma_{21} & \gamma_{22}
    \end{array}\right)
    $$
  
    and the following two sequences of observations that are assumed to be generated by the above Markov chain. 
  
    $$\text{Sequence 1}: 1 1 1 2 2 1$$
    $$\text{Sequence 2}: 2 1 1 2 1 1$$
  
    Compute the probability of each of the sequences. 
  
Code 8-9.

8.

  a. Write an **R** function `statdist(gamma)` that computes the stationary distribution of the Markov chain with tpm `gamma`. 

  b. Use the function to find the stationary distribution of the following transition probabilitiy matrices. Identify the irreducible tpm(s).
  
  (i)  
 \begin{align}
 \begin{pmatrix}
    0.7 & 0.2 & 0.1\\
    0 & 0.6 & 0.4\\
    0.5 & 0 & 0.5
  \end{pmatrix}
  \end{align}
  
  (ii)
 \begin{align}
 \begin{pmatrix}
    0.25 & 0.25 & 0.25 & 0.25\\
    0.25 & 0.25 & 0.5 & 0\\
    0 & 0 & 0.25 & 0.75\\
    0 & 0 & 0.5 & 0.5\\
  \end{pmatrix}
  \end{align}
  
  (iii)
 \begin{align}
 \begin{pmatrix}
    1 & 0 & 0 & 0\\
    0.5 & 0 & 0.5 & 0\\
    0 & 0.75 & 0 & 0.75\\
    0 & 0 & 0 & 1\\
  \end{pmatrix}
  \end{align}

9. Write an **R** function `rMC(n, m, gamma, delta=NULL)` that generates a series of length $n$ from an $m$-state Markov chain with tpm `gamma`. Use the initial state distribution if it is provided and the stationary distribution as the initial distribution otherwise. 


## Solutions

**Question 1**

Let $i, j \in \mathbb{N}$. 

Then 

\begin{align*}
\left( \boldsymbol{\Gamma} (t + u) \right)_{ij} 
&= \Pr(C_{t + u + s} = j| C_s = i)\\
&= \sum_{k=1}^m \Pr(C_{t+u+s} = j| C_{t+s} = k, C_s = i) \Pr(C_{t+s} = k|C_s = i) 
\qquad{\text{by LOTP}}\\
&= \sum_{k=1}^m \Pr(C_{t+u+s} = j|C_{t+s} = k) \Pr(C_{t+s} = k|C_s = i)
\qquad{\text{by Markov property}}\\
&= \sum_{k=1}^m \Pr(C_{u+s} = j|C_s = k) \Pr(C_{t+s} = k|C_s = i)
\qquad{\text{by homogeneity of Markov chains}}\\
&= \sum_{k=1}^m \Pr(C_{t+s} = k|C_s = i) \Pr(C_{u+s} = j|C_s = k)\\
&= \sum_{k=1}^m \left( \boldsymbol{\Gamma} (t) \right)_{ik} \left( \boldsymbol{\Gamma} (u) \right)_{kj}
\end{align*}

Since the $i, j$-th entry of $\boldsymbol{\Gamma} (t+u)$ is equal to the $i, j$-th entry of $\boldsymbol{\Gamma} (t) \boldsymbol{\Gamma} (u)$ for arbitrary $i, j \in \mathbb{N}$, we conclude that $\boldsymbol{\Gamma} (t + u) = \boldsymbol{\Gamma} (t) \boldsymbol{\Gamma} (u)$. 


**Question 2**

\begin{align*}
\boldsymbol{\Gamma} (1) 
&= \boldsymbol{\Gamma} (1) \cdot \boldsymbol{\Gamma} (1) \cdots \boldsymbol{\Gamma} (1) 
\qquad{\text{t-times}}\\
&= \boldsymbol{\Gamma} (1 + 1 + \cdots + 1)
\qquad{\text{by Chapman-Kolomogorov equations}}\\
&= \boldsymbol{\Gamma} (t)
\end{align*}


**Question 3**

Let $j \in \mathbb{N}$. Then 

\begin{align*}
\left( \boldsymbol{u} (t+1) \right)_{1j} 
&= \Pr(C_{t+1} = j|C_t = i) \Pr(C_t = i)
\qquad{\text{by Chain rule}}\\
&= \boldsymbol{u} (t) \gamma_{ij} (t)\\
&= \left(  \boldsymbol{u} (t) \boldsymbol{\Gamma}\right)_{1j}
\end{align*}

Since the $j$-th entry of $\boldsymbol{u} (t+1)$ is equal to the $j$-th entry of $\boldsymbol{u} (t) \boldsymbol{\Gamma}$, we conclude that $\boldsymbol{u} (t+1) = \boldsymbol{u} (t) \boldsymbol{\Gamma}$. 

**Question 4**

$(\Rightarrow)$ Suppose $\boldsymbol{\delta}$ is a stationary distribution of a discrete-time homogenous Markov chain on $m$ states. That is, $\boldsymbol{\delta \Gamma} = \boldsymbol{\delta}$ and $\boldsymbol{\delta 1} = \boldsymbol{1}'$.

Then 

\begin{align*}
\boldsymbol{\delta} (\boldsymbol{I}_m - \boldsymbol{\Gamma} + \boldsymbol{U})
&= \boldsymbol{\delta} \boldsymbol{I}_m - \boldsymbol{\delta \Gamma} + \boldsymbol{\delta U}\\
&= \boldsymbol{\delta U}\\
&= (\delta_1, \delta_2, \dots, \delta_m) 
\begin{pmatrix}
1 & 1 & \cdots & 1\\
1 & 1 & \cdots & 1\\
\vdots & \vdots & \cdots & \vdots\\
1 & 1 & \cdots & 1
\end{pmatrix}\\
&= (\delta_1 + \delta_2 + \cdots + \delta_m, \delta_1 + \delta_2 + \cdots + \delta_m, \dots, \delta_1 + \delta_2 + \cdots + \delta_m)\\
&= (1, 1, \dots, 1)\\
&= \boldsymbol{1}
\end{align*}


$(\Leftarrow)$ Suppose $\boldsymbol{\delta} (\boldsymbol{I}_m - \boldsymbol{\Gamma} + \boldsymbol{U})$. 

Then 

\begin{equation*}
\boldsymbol{\delta} - \boldsymbol{\delta} \boldsymbol{\Gamma} + \boldsymbol{\delta} \boldsymbol{U} = \boldsymbol{1}\\
\boldsymbol{\delta} \boldsymbol{1}' - \boldsymbol{\delta \Gamma} \boldsymbol{1}' + \boldsymbol{\delta U}  \boldsymbol{1}' =   \boldsymbol{1} \boldsymbol{1}'\\
\end{equation*}


And $\boldsymbol{\Gamma 1}' = \boldsymbol{1}'$ since $\sum_{j} \gamma_{ij} = 1$, $\boldsymbol{U 1}' = \boldsymbol{m} = (m, \dots, m)' = \boldsymbol{m 1}'$, and $\boldsymbol{1 1}' = m$

So 
\begin{equation*}
\boldsymbol{\delta} \boldsymbol{1}' - \boldsymbol{\delta} \boldsymbol{1}' + \boldsymbol{\delta m 1'} =   m\\
\boldsymbol{\delta m 1'} =   m\\
\boldsymbol{\delta 1'} =   1
\end{equation*}

Therefore, $\boldsymbol{\delta}$ is a stationary distribution if and only if $\boldsymbol{\delta} ( \boldsymbol{I}_m - \boldsymbol{\Gamma} + \boldsymbol{U}) = \boldsymbol{1}$


**Question 5**

The distribution of today's weather is $\boldsymbol{u} (1) = (\Pr(C_1 = 1, \Pr(C_1 = 2)) = (0, 1)$

We can obtain the distribution of tomorrow's weather by postmultiplying $\boldsymbol{u} (1)$ by $\boldsymbol{\Gamma}$ (Equation \@ref(eq:post)), which is

$\boldsymbol{u} (2) = (\Pr(C_2 = 1), \Pr (C_2 = 2)) = \boldsymbol{u} (1) \boldsymbol{\Gamma} = (0.6, 0.4)$.

The stationary distribution can be obtained by solving $\boldsymbol{\delta \Gamma} = \boldsymbol{\delta}$ and $\boldsymbol{\delta 1'} = \boldsymbol{1}$. 

So 

\begin{equation*}
(\delta_1, \delta_2) \begin{pmatrix}
0.9 & 0.1\\
0.6 & 0.4
\end{pmatrix}
= \begin{pmatrix}
\delta_1\\
\delta_2
\end{pmatrix}\\
0.9 \delta_1 + 0.6 \delta_2 = \delta_1\\
0.1 \delta_1 + 0.4 \delta_2 = \delta_2
\end{equation*}

and we have the additional constraint that $\delta_1 + \delta_2 = 1$. Solving the above systems of equation, we get that $\boldsymbol{\delta} = (\delta_1, \delta_2) = (\frac{6}{7}, \frac{1}{7})$.

**Question 6**
Let $Y_1, Y_2$ be the component random variables of the mixture distributions with expectations $\mu_1, \mu_2$, and variances $\sigma_1^2, \sigma_2^2$, respectively. Then $E(Y_1^2) - (E(Y_1))^2 = \sigma_1^2 \Rightarrow E(Y_1^2) = \sigma_1^2 + (E(Y_1))^2$. Similarly, $E(Y_2^2) = \sigma_2^2 + (E(Y_2))^2$. 


Then

\begin{align}
Var(X) 
&= E(X^2) - (E(X))^2\\
&= \delta_1 E(Y_1^2) + \delta_2 E(Y_2^2) - (\delta_1 E(Y_1) + \delta_2 E(Y_2))^2
\qquad{\text{by Equation 2.2 and 2.3}}\\
&= \delta_1 (\sigma_1^2 + (E(Y_1))^2) + \delta_2 (\sigma_2^2 + (E(Y_2))^2) - (\delta_1 E(Y_1) + \delta_2 E(Y_2))^2
\qquad{\text{by the above}}\\
&= \delta_1 \sigma_1^2 + \delta_1 (E(Y_1))^2 + \delta_2 \sigma_2^2 + \delta_2 (E(Y_2))^2 - \delta_1^2 (E(Y_1))^2 - 2 \delta_1 \delta_2 E(Y_1) E(Y_2) - \delta_2^2 (E(Y_2))^2\\
&= \delta_1 \sigma_1^2 + \delta_1 \mu_1^2 + \delta_2 \sigma_2^2 + \delta_2 \mu_2^2 - \delta_1^2 \mu_1^2 - 2 \delta_1 \delta_2 \mu_1 \mu_2 - \delta_2^2 \mu_2^2\\
&= \delta_1 \sigma_1^2 + \delta_2 \sigma_2^2 + \delta_1 \mu_1^2 (1 - \delta_1) + \delta_2 \mu_2^2 (1 - \delta_2) - 2 \delta_1 \delta_2 \mu_1 \mu_2\\
&= \delta_1 \sigma_1^2 + \delta_2 \sigma_2^2 + \delta_1 \delta_2 \mu_1^2 - 2 \delta_1 \delta_2 \mu_1 \mu_2 + \delta_1 \delta_2 \mu_2^2
\qquad{\text{since } \delta_1 + \delta_2 = 1}\\
&= \delta_1 \sigma_1^2 + \delta_2 \sigma_2^2 + \delta_1 \delta_2 (\mu_1 - \mu_2)^2
\end{align}


**Question 7**

**Part a**

We need to solve $\boldsymbol{\delta \Gamma} = \boldsymbol{\delta}$ and $\boldsymbol{\delta 1'} = 1$. 

Hence,

\begin{equation*}
(\delta_1, \delta_2) \begin{pmatrix}
\gamma_{11} & \gamma_{12}\\
\gamma_{21} & \gamma_{22}
\end{pmatrix}
= \begin{pmatrix}
\delta_1\\
\delta_2
\end{pmatrix}\\
\gamma_{11} \delta_1 + \gamma_{21} \delta_2 = \delta_1\\
\gamma_{12} \delta_1 + \gamma_{22} \delta_2 = \delta_2
\end{equation*}

Since $\gamma_{11} + \gamma_{12} = 1$ and $\gamma_{21} + \gamma_{22} = 1$, and we have the additional constraint that $\delta_1 + \delta_2 = 1$, this is equivalent to solving the systems of equations

\begin{equation*}
(1 - \gamma_{12}) \delta_1  + \gamma_{21} \delta_2 = \delta_1\\
\gamma_{12} \delta_1 + (1 - \gamma_{21}) \delta_2 = \delta_2\\
\delta_1 + \delta_2 = 1
\end{equation*}

Then solving system of equations above, we get $\frac{1}{\gamma_{12} + \gamma_{21}} (\gamma_{21}, \gamma_{12})$

**Part b**

By [Transition Probabilities Estimation](#tp), 

$\Pr(\text{sequence 1}) = \gamma_{11}^{f_{11}} \gamma_{12}^{f_{12}}  \gamma_{21}^{f_{21}} \gamma_{22}^{f_{22}} = (0.9)^2 (0.1)^1 (0.2)^1 (0.8)^1 = 0.01296$

and

$\Pr(\text{sequence 2}) = \gamma_{11}^{f_{11}} \gamma_{12}^{f_{12}}  \gamma_{21}^{f_{21}} \gamma_{22}^{f_{22}} = (0.9)^2 (0.1)^1 (0.2)^2 (0.8)^0 = 0.00324$


**Question 8**


```r
statdist = function(Gamma){
  # number of components
  m = nrow(Gamma)
  
  # identity matrix (mxm)
  I = diag(m)
  # ones matrix (mxm)
  U = matrix(rep(1,m*m), nrow=m,ncol=m)
  # vector of ones (1xm)
  one_vec = rep(1, m)
  
  # I - Gamma + U
  X = I - Gamma + U
  
  # delta(Im - Gamma + U) = 1
  delta = solve(t(X), one_vec)
  return(delta)
}
```


```r
matrix_i = t(matrix(c(0.7, 0.2, 0.1, 0, 0.6, 0.4, 0.5, 0, 0.5), nrow=3, ncol=3))
matrix_ii = t(matrix(c(0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0, 0, 0, 0.25, 0.75, 0, 0, 0.5, 0.5), nrow=4, ncol=4))
matrix_iii = t(matrix(c(1, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0.75, 0, 0.25, 0, 0, 0, 1), nrow=4, ncol=4))
```


```r
statdist(matrix_i)
```

```
## [1] 0.4761905 0.2380952 0.2857143
```


```r
statdist(matrix_ii)
```

```
## [1] 2.434560e-17 1.720846e-17 4.000000e-01 6.000000e-01
```


```r
statdist(matrix_iii)
```

Matrix (iii) causes an error because there is no unique stationary distribution. Once the Markov chain is in state 1, it will stay in state 1 at all subsequent time points and once it is in state 4, it will stay in state 4 at all subsequent time points. It is also reducible because the transitions to the other states cannot be reached.  

Matrix (i) is irreducible because all states can be reached. Matrix (ii) is reducible because the transitions to states 3 and 4 cannot be reached.

**Question 9**


```r
rMC = function(n, m, gamma, delta=NULL){
  # Solve for stationary distribution if none provided
  if(is.null(delta)) delta = statdist(gamma)
  
  # Simulate sequence of states
  state_seq = c()
  set.seed(548)
  
  ## For t=1
  state_seq[1] = sample(m, 1, prob=delta) # the initial distribution

  ## For t > 1
  for(i in 2:n){
    state_seq <- c(state_seq, sample(m, 1, prob=gamma[state_seq[i-1], ]))
  }
  return(state_seq)
}
```






























