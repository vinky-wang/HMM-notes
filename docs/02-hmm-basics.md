# Introduction to Hidden Markov Models {#introhmm}

A **hidden Markov model** $\{X_t: t \in \mathbb{N}\}$ can be summarized by 

\begin{align}
\Pr(C_t|\boldsymbol{C}^{(t-1)}) 
&= \Pr(C_t|C_{t-1}) \qquad{t = 2, 3, \dots}
(\#eq:hmmh)
\end{align}

\begin{align}
\Pr(X_t|\boldsymbol{X}^{(t-1)}, \boldsymbol{C}^{(t)}) 
&= \Pr(X_t|C_t) \qquad{t \in \mathbb{N}}
(\#eq:hmmo)
\end{align}

where

\@ref(eq:hmmh) is the unobserved parameter process, which satisfies the Markov property 

\@ref(eq:hmmo) is the observed state-dependent process, which are conditionally independent on past and future observations and states


## State-Dependent Distributions {#sdd}

The **state-dependent distributions** of observation $X_t$ given that the Markov chain is in state $i$ at time $t$ are the probability mass (pmf) or density (pdf) functions 

$$p_i (x) = \Pr(X_t = x|C_t = i) \qquad{\text{for } i = 1, \dots, m}$$ 

or in matrix notation,

$$
\boldsymbol{P} (x) = 
\begin{pmatrix}
p_1 (x) & & 0\\
& \ddots & \\
0 & & p_m(x)
\end{pmatrix}
$$

**Note:** The remaining sections assume discrete observations $X_t$, however the continuous case is analagous. 


## Marginal Distributions

### Univariate Case

Define $u_i (t) = \Pr (C_t = i)$ for $t = 1, \dots, T$. 

The marginal distribution of $X_t$ where the MC is homogenous, but not necessarily stationary is given by

\begin{align}
\Pr (X_t = x) 
&= \sum_{i=1}^m \Pr (C_t = i) \Pr(X_t = x|C_t = i)\\
&= \sum_{i=1}^m u_i (t) p_i (x)
\end{align}

or in matrix notation,

\begin{align}
\Pr(X_t = x) 
&= (u_1(t), \dots, u_m(t)) 
\begin{pmatrix}
p_1 (x) & & 0\\
& \ddots & \\
0 & & p_m(x)
\end{pmatrix}
\begin{pmatrix}
1\\
\vdots\\
1
\end{pmatrix}\\
&= \boldsymbol{u} (t) \boldsymbol{P} (x) \boldsymbol{1'}
\end{align}

Since $\boldsymbol{u} (t+1) = \boldsymbol{u} (t) \boldsymbol{\Gamma}$ (Equation \@ref(eq:post)), it follows that $\boldsymbol{u} (t) = \boldsymbol{u} (1) \boldsymbol{\Gamma}^{(t-1)}$, so

\begin{equation}
\Pr(X_t = x) = \boldsymbol{u} (1) \boldsymbol{\Gamma}^{t-1} \boldsymbol{P} (x) \boldsymbol{1'}
\end{equation}

The marginal distribution of $X_t$ if the MC is stationary is given by 

\begin{equation}
\Pr(X_t = x) = \boldsymbol{\delta P} (x) \boldsymbol{1'}
\end{equation}


**Note:** Since a MC with stationary distribution $\boldsymbol{\delta}$ such that $\boldsymbol{\delta \Gamma} = \boldsymbol{\delta} \text{ and } \boldsymbol{\delta 1'} = 1$ implies that $\boldsymbol{\delta \Gamma}^{t-1} = \boldsymbol{\delta}$ for all $t \in \mathbb{N}$.


### Bivariate Case {#bivariate}

The bivariate distribution of $X_t$ and $X_{t+k}$ where the MC is homogenous, but not necessarily stationary is given by 

\begin{align}
\Pr(X_t = v, X_{t+k} = w)
&= \sum_{i=1}^m \sum_{j=1}^m \Pr (X_t = v, X_{t+k} = w, C_t = i, C_{t+k} = j)\\
&= \sum_{i=1}^m \sum_{j=1}^m \Pr(C_t = i) \Pr(X_t = v|C_t = i) \Pr(C_{t+k} = j|C_t = i) \Pr(X_{t+k} = w|C_{t+k} = j)\\
&= \sum_{i=1}^m \sum_{j=1}^m u_i (t) p_i (v) \gamma_{ij} (k) p_j (w)
\end{align}

or in matrix notation

\begin{align}
\Pr(X_t = v, X_{t+k} = w) &= \boldsymbol{u} (t) \boldsymbol{P} (v) \boldsymbol{\Gamma}^k \boldsymbol{P} (w) \boldsymbol{1'}
\qquad{\text{where } \gamma_{ij} (k) \text{ denotes the (i, j) element of } \boldsymbol{\Gamma}^k}
\end{align}

The marginal distribution of $X_t$ and $X_{t+k}$ if the MC is stationary is given by 
\begin{align}
\Pr(X_t = v, X_{t+k} = w) &= \boldsymbol{\delta P} (v) \boldsymbol{\Gamma}^k \boldsymbol{P} (w) \boldsymbol{1'}
\end{align}

**Note:** The above follows from Equation \@ref(eq:directed) in which $\Pr(X_t, X_{t+k}, C_{t}, C_{t+k}) = \Pr (C_t) \Pr(X_t|C_t) \Pr(C_{t+k}|C_t) \Pr(X_{t+k}|C_{t+k})$.





## Moments

### Univariate Case

Let $g$ be any functions for which the relevant state-dependent expectations exist. 

The state-dependent expectation of $X_t$ where the MC is homogenous, but not necessarily stationary is given by

\begin{align}
E(g(X_t))
&= \sum_{i=1}^m E(g(X_t)|C_t = i) \Pr(C_t = i)\\
&= \sum_{i=1}^m u_i (t) E(g(X_t)|C_t = i)
\end{align}

and in the stationary case,

\begin{align}
E(X_t) 
&= \sum_{i=1}^m E(X_t|C_t = i) \Pr(C_t = i)\\
&= \sum_{i=1}^m u_i (t) E(X_t|C_t = i)
\end{align}

and in the stationary case,

\begin{align}
E(g(X_t)) = \sum_{i=1}^m \delta_i E(g(X_t)|C_t = i)
\end{align}


### Bivariate Case

The state-dependent expectation of $X_t$ and $X_{t+k}$ where the MC is homogenous, but not necessarily stationary is given by

\begin{align}
E(g(X_t, X_{t+k})) 
&= \sum_{i, j=1}^m E(g(X_t, X_{t+k})|C_t = i, C_{t+k} = j) \Pr(C_t = i) \Pr(C_{t+k}=j|C_t = i)\\
&= \sum_{i, j=1}^m E(g(X_t, X_{t+k})|C_t = i, C_{t+k} = j) u_i(t)\gamma_{ij} (k)
\end{align}

and in the stationary case,

\begin{align}
E(g(X_t, X_{t+k})) = \sum_{i, j=1}^m E(g(X_t, X_{t+k})|C_t = i, C_{t+k} = j) \delta_i \gamma_{ij} (k)
\end{align}

If $g$ is factorizable as $g(X_t, X_{t+k}) = g_1 (X_t) g_2 (X_{t+k})$, then the above becomes

$$E(g(X_t, X_{t+k})) = \sum_{i, j=1}^m E(g_1(X_t)|C_t = i)E(g_2(X_{t+k})|C_{t+k}=j) \delta_i \gamma_{ij} (k)$$






## Likelihood of Hidden Markov Models {#lik}

::: {.proposition #likelihood}
The likelihood is given by 
:::

\begin{equation}
L_T = \boldsymbol{\delta P} (x_1) \boldsymbol{\Gamma P} (x_2) \boldsymbol{\Gamma} \cdots \boldsymbol{\Gamma P} (x_T)
\boldsymbol{1'}
(\#eq:hmmlik)
\end{equation}

*If $\boldsymbol{\delta}$, the distribution of $C_1$, is the stationary distribution of the Markov chain, then in addition*

\begin{equation}
L_T = \boldsymbol{\delta \Gamma P} (x_1) \boldsymbol{\Gamma P} (x_2) \boldsymbol{\Gamma} \cdots \boldsymbol{\Gamma P} (x_T)
\boldsymbol{1'}
(\#eq:hmmlikstat)
\end{equation}



*Proof.* 

\begin{align}
L_T 
&= \Pr(\boldsymbol{X}^{(T)} = \boldsymbol{x}^{(T)})\\
&= \sum_{c_1, c_2, \dots, c_T = 1}^m \Pr(\boldsymbol{X}^{(T)} = \boldsymbol{x}^{(T)}, \boldsymbol{C}^{(T)} = \boldsymbol{c}^{(T)})
\qquad{\text{by LOTP}}\\
&= \sum_{c_1, c_2, \dots, c_T = 1}^m \Pr(C_1 = c_1) \prod_{k=2}^T \Pr(C_k = c_k|C_{k-1}=c_{k-1}) \prod_{k=1}^T \Pr(X_k = x_k|C_k = c_k) 
\qquad{\text{by Equation (10.1)}}\\
&= \sum_{c_1, c_2, \dots, c_T = 1}^m \left(\delta_{c_1} \gamma_{c_1, c_2} \gamma_{c_2, c_3} \cdots \gamma_{c_{T-1}, c_T} \right) \left( p_{c_1} (x_1) p_{c_2} (x_2) \cdots p_{c_T} (x_T) \right)\\
&= \sum_{c_1, \dots, c_T = 1}^m \delta_{c_1} p_{c_1} (x_1) \gamma_{c_1, c_2} p_{c_2} (x_2) \gamma_{c_2, c_3} \cdots \gamma_{c_{T-1}, c_T} p_{c_T} (x_T)\\
&= \boldsymbol{\delta P} (x_1) \boldsymbol{\Gamma P} (x_2) \boldsymbol{\Gamma} \cdots \boldsymbol{\Gamma P} (x_T) \boldsymbol{1'}
\end{align}

If $\boldsymbol{\delta}$ is the stationary distribution, then $\boldsymbol{\delta P} (x_1) = \boldsymbol{\delta \Gamma P} (x_1)$.

**Note:** The likelihood consists of a sum of $m^T$ terms, each of which is a product of $2T$ factors, which would require $\mathcal{O} (Tm^T)$ operations. For example, the likelihood from the [Bivariate Distributions section](#bivariate) consists of $m^2$ terms, each of which is a product of $2 \times 2$ factors. The recursive nature of the likelihood drastically reduces the model complexity and can be evaluated by the **forward algorithm** which performs $\mathcal{O} (Tm^2)$ operations. This makes numerical maximization feasible as shown in the next chapter. 

### Forward Algorithm {#forsec}

For $t = 1, 2, \dots, T$

\begin{align}
\boldsymbol{\alpha}_t 
&= \boldsymbol{\delta P} (x_1) \boldsymbol{\Gamma P} (x_2) \boldsymbol{\Gamma} \cdots \boldsymbol{\Gamma P} (x_T)\\
&= \boldsymbol{\delta P} (x_1) \prod_{s=2}^t \boldsymbol{\Gamma P} (x_s)
(\#eq:forward)
\end{align}

with the convention that an empty product is the identity matrix. 

It follows from this definition that

\begin{equation}
L_T = \boldsymbol{\alpha}_T \boldsymbol{1'}
\qquad{ \text{ and }}
\qquad{}
\boldsymbol{\alpha}_t = \boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P} (x_t) \text{ for } t \geq 2
\end{equation}

For a homogenous but not necessarily stationary MC, the likelihood can be computed as $L_T = \boldsymbol{\alpha}_T \boldsymbol{1'}$ where

$$\boldsymbol{\alpha}_1 = \boldsymbol{\delta P} (x_1)$$

$$\boldsymbol{\alpha}_t = \boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P} (x_t) \qquad{\text{for } t = 2, 3, \dots, T}$$


and in the stationary case,

$$\boldsymbol{\alpha}_0 = \boldsymbol{\delta}$$

$$\boldsymbol{\alpha}_t = \boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P} (x_t) \qquad{\text{for } t = 1, 2, \dots, T}$$





## Exercises

1. Consider a stationary two-state Poisson-HMM with parameters 

  $$\boldsymbol{\Gamma} = 
  \begin{pmatrix}
  0.1 & 0.9\\
  0.4 & 0.6
  \end{pmatrix} \qquad{} \boldsymbol{\lambda} = (1, 3)$$ 

    
  $\qquad{}$ In each of the following ways, compute the probability that the first three observations from this 
  $\qquad{}$ model are 0, 2, 1.

  a. Consider all possible sequences of the states of the Markov chain that could have occured. Compute the probability of each sequence, and the probability of the observations given each sequence.
  
  b. Apply the formula $\Pr(X_1 = 0, X_2 = 2, X_3 = 1) = \boldsymbol{\delta P} (0) \boldsymbol{\Gamma P} (2) \boldsymbol{\Gamma P} (1) \boldsymbol{1'}$, where 
  
  \begin{align}
  \boldsymbol{P} (x)
  &= 
  \begin{pmatrix}
  \frac{\lambda_1^s e^{- \lambda_1}}{s!} & 0\\
  0 & \frac{\lambda_2^s e^{- \lambda_2}}{s!}
  \end{pmatrix}\\
  &= 
  \begin{pmatrix}
  \frac{1^s e^{-1}}{s!} & 0\\
  0 & \frac{3^s e^{-3}}{s!}
  \end{pmatrix}
  \end{align}
  
2.

  a. Consider the vector $\boldsymbol{\alpha}_t = (\alpha_1(1), \dots, \alpha_t (m))$ defined by 
  $$\alpha_t(j) = \Pr(\boldsymbol{X}^{(t)} = \boldsymbol{x}^{(t)}, C_t = j), \qquad{j = 1, 2, \dots, m}$$
  
  $\qquad{}$ Use conditional probability and the conditional independence assumptions to show that 
  
  \begin{equation}
    \alpha_t(j) = \sum_{i=1}^m \alpha_{t-1} (i) \gamma_{ij} p_j (x_t)
    (\#eq:ascalar)
  \end{equation}

  
  b. Verify that the result from (a), written in matrix notation, yields the forward recursion 
  $$\boldsymbol{\alpha}_t = \boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P} (x_t) \qquad{t = 2, \dots, T}$$
  
  c. Hence derive the matrix expression for the likelihood.
  
  
## Solutions

**Question 1**

**Part a**


```r
# Set up -----------------------------------------------------------------------
## Load package
library(tidyverse)

## Recall statdist function in chapter 1
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


# Question 1 --------------------------------------------------------------------

# Parameters 
Gamma_pois = matrix(c(0.1, 0.4, 0.9, 0.6), nrow=2, ncol=2)
lambda_pois = c(1, 3)
delta_pois = statdist(Gamma_pois) ## assuming stationary


# All possible sequences
i = c(rep(1, 4), rep(2, 4))
j = c(rep(c(1, 1, 2, 2), 2))
k = c(rep(c(1, 2), 4))

# values for X
x1 = 0
x2 = 2
x3 = 1



# Function that computes Pr(X=s|C=c)
p = function(lambda1, lambda2, index, s){
  if(index==1){
      ps = ((lambda1^s)*exp(-lambda1))/(factorial(s))
      return(ps)
  }
  else{
      ps = ((lambda2^s)*exp(-lambda2))/(factorial(s))
      return(ps)
  }
}


# Compute Pr(X=s|C=c)
pi0 = c()
pj2 = c()
pk1 = c()
for(ind in 1:length(i)){
  pi0[ind] = p(lambda_pois[1], lambda_pois[2], i[ind], x1)
  pj2[ind] = p(lambda_pois[1], lambda_pois[2], j[ind], x2)
  pk1[ind] = p(lambda_pois[1], lambda_pois[2], k[ind], x3)
}

# delta_i
delta_i = c(rep(delta_pois[1], 4), rep(delta_pois[2], 4))

# gamma_ij
gamma_ij = c()

for(ind in 1:length(i)){
  gamma_ij[ind] = Gamma_pois[i[ind], j[ind]]
}

# gamma_jk
gamma_jk = c()

for(ind in 1:length(i)){
  gamma_jk[ind] = Gamma_pois[j[ind], k[ind]]
}

# Table for likelihood computation
lktab = as.data.frame(cbind(i, j, k, pi0, pj2, pk1, delta_i, gamma_ij, gamma_jk))

lktab <- lktab %>%
  mutate(product = pi0*pj2*pk1*delta_i*gamma_ij*gamma_jk) %>%
  mutate_if(is.numeric, round, 4)

# Likelihood
sum(lktab$product)
```

```
## [1] 0.0073
```

```r
# Global decoding
lktab[which(lktab$product == max(lktab$product)),] 
```

```
##   i j k    pi0   pj2    pk1 delta_i gamma_ij gamma_jk product
## 3 1 2 1 0.3679 0.224 0.3679  0.3077      0.9      0.4  0.0034
```
The sequence that maximizes the conditional probability is $i = 1, j = 2, k=1$

**Part b**


```r
# P(s) matrices
Ps = function(lambda1, lambda2, s){
  Ps_matrix = matrix(c(((lambda1^s)*exp(-lambda1))/(factorial(s)),
                       0,
                       0,
                       ((lambda2^s)*exp(-lambda2))/(factorial(s))), nrow=2, ncol=2)
  return(Ps_matrix)
}

P0 = Ps(lambda_pois[1], lambda_pois[2], 0)
P2 = Ps(lambda_pois[1], lambda_pois[2], 2)
P1 = Ps(lambda_pois[1], lambda_pois[2], 1)

# Row vector of ones
one_vec = rep(1, 2)

# Apply the formula
delta_pois%*%P0%*%Gamma_pois%*%P2%*%Gamma_pois%*%P1%*%one_vec
```

```
##            [,1]
## [1,] 0.00729174
```


**Question 2**

**Part a**

\begin{align}
\alpha_t (j) 
&= \Pr(\boldsymbol{X}^{(t)} = \boldsymbol{x}^{(t)}, C_t = j)\\
&= \Pr (X_t = x_t | \boldsymbol{X}^{(t-1)} = \boldsymbol{x}^{(t-1)}, C_t = j) \Pr(\boldsymbol{X}^{(t-1)} = \boldsymbol{x}^{(t-1)}, C_t = j)
\qquad{\text{by Chain rule}}\\
&= \Pr(X_t = x_t |C_t = j) \Pr(\boldsymbol{X}^{(t-1)} = \boldsymbol{x}^{(t-1)}, C_t = j)
\qquad{\text{by conditional independence}}\\
&= \Pr(X_t = x_t |C_t = j) \sum_{i=1}^m \Pr(C_t = j|\boldsymbol{X}^{(t-1)} = \boldsymbol{x}^{(t-1)}, C_{t-1} = i) \Pr(\boldsymbol{X}^{(t-1)} = \boldsymbol{x}^{(t-1)}, C_{t-1} = i)
\qquad{\text{by Law of Total Probability}}\\
&= \Pr(X_t = x_t |C_t = j) \sum_{i=1}^m \Pr(C_t = j|C_{t-1} = i) \Pr(\boldsymbol{X}^{(t-1)} = \boldsymbol{x}^{(t-1)}, C_{t-1} = i)
\qquad{\text{by conditional independence}}\\
&= \sum_{i=1}^m \Pr(\boldsymbol{X}^{(t-1)} = \boldsymbol{x}^{(t-1)}, C_{t-1} = i) \Pr(C_t = j|C_{t-1} = i) \Pr(X_t = x_t|C_t=j)\\
&= \sum_{i=1}^m \alpha_{t-1} (i) \gamma_{ij} p_j (x_t)
\end{align}

**Part b**

Let $j \in \mathbb{N}$. Then

\begin{align}
\left(\boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P} (x_t)\right)_{1j}\\
&= \sum_{k=1}^m \left(\boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P}\right)_{1k} \left(\boldsymbol{\Gamma P} (x_t)\right)_{kj}\\
&= \sum_{k=1}^m \sum_{i=1}^m \left(\boldsymbol{\alpha}_{t-1} \right)_{1i} \left(\boldsymbol{\Gamma}\right)_{ik} \left(\boldsymbol{P} (x_t)\right)_{kj}
\qquad{\text{since } \boldsymbol{P} (x_t) \text{ is a diagonal matrix so} \left(\boldsymbol{P} (x_t) \right)_{kj} = 0 for all k \neq j}\\
&= \sum_{i=1}^m \boldsymbol{\alpha}_{t-1} (i) \gamma_{ij} p_j (x_t)
\end{align}

Since the $j$-th entry of $\boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P} (x_t)$ is equal to the $j$-th entry of $\boldsymbol{\alpha}_t$ for an arbitrary $j \in \mathbb{N}$, we conclude that $\boldsymbol{\alpha}_t = \boldsymbol{\alpha}_{t-1} \boldsymbol{\Gamma P} (x_t)$.


**Part c**

We will prove for $t = 1, \dots, T$ that $\boldsymbol{\alpha}_t = \boldsymbol{\delta P} (x_1) \boldsymbol{\Gamma} \cdots  \boldsymbol{\Gamma P} (x_t)$ by induction on $t$.

Base case:

If $t=1$, then

\begin{align*}
\boldsymbol{\alpha}_1 
&= (\alpha_1 (1), \dots, \alpha_1 (m))\\
&= (\Pr(X_1 = x_1, C_1 = 1), \dots, \Pr(X_1 = x_1, C_1 = m))\\
&= (\Pr(C_1 = 1) \Pr(X_1 = x_1|C_1 = 1), \dots, \Pr(C_1 = m) \Pr(X_1 = x_1|C_1 = m))\\
&= (\delta_{c_1} p_1(x_1), \dots, \delta_{c_1} p_m (x_1))\\
&= \boldsymbol{\delta P} (x_1)
\end{align*}

Inductive step:

Let $t \in \{1, \dots, T\}$. Suppose $\boldsymbol{\alpha}_t = \boldsymbol{\delta P} (x_1) \boldsymbol{\Gamma} \cdots  \boldsymbol{\Gamma P} (x_t)$. Then

\begin{align*}
\boldsymbol{\alpha}_{t+1} 
&= (\alpha_{t+1} (1), \dots, \alpha_{t+1} (m))\\
&= (\Pr(\boldsymbol{X}^{(t+1)} = \boldsymbol{x}^{(t+1)}, C_{t+1} = 1), \dots, \Pr(\boldsymbol{X}^{(t+1)} = \boldsymbol{x}^{(t+1)}, C_{t+1} = m))\\
&= (\sum_{i=1}^m \alpha_t (i) \gamma_{i1} p_1 (x_{t+1}), \dots, \sum_{i=1}^m \alpha_t (i) \gamma_{im} p_m (x_{t+1}))
\qquad{\text{by part a}}\\
&= \boldsymbol{\alpha}_t \boldsymbol{\Gamma P} (x_{t+1})
\qquad{\qquad{\text{by part b}}}\\
&= \boldsymbol{\delta P} (x_1) \boldsymbol{\Gamma} \cdots  \boldsymbol{\Gamma P} (x_t) \boldsymbol{\Gamma P} (x_{t+1})
\qquad{\text{by the inductive hypothesis}}
\end{align*}
  







