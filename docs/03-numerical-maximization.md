# Numerical Maximization of the Likelihood {#numerical}

We cover solutions to several potential issues when performing direct numerical maximization of the likelihood (using the [forward algorithm](#forsec))

1. [Scaling the likelihood](#likscale) to avoid numerical under/over-flow

2. [Reparameterizing](#reparam) in order to use an unconstrained optimizer

3. [Choosing a range of starting values](#startval) to avoid multiple maxima when finding the global maximum


## Scaling the Likelihood Computation {#likscale}

The forward probabilities of $\boldsymbol{\alpha_t}$ become progressively smaller as $t$ increases (see [Exercise 1b](#introhmm)), which may lead to numerical underflow. Since the likelihood is a product of matrices, not of scalars, it is not possible to circumvent numerical underflow by simply computing the log of the likelihood. Instead, the vector of forward probabilities $\boldsymbol{\alpha_t}$ can be scaled at each time $t$, so that the log likelihood is a sum of the logs of the scale factors.

Define, for $t = 0, 1, \dots, T$, the vector

$$\boldsymbol{\phi_t} = \frac{\boldsymbol{\alpha_t}}{w_t}$$

where $w_t = \sum_i \alpha_t(i) = \boldsymbol{\alpha_t 1'}$

By the definitions of $\boldsymbol{\phi_t}$ and $w_t$, the immediate consequences are the following:

\begin{align}
&\tag{1} \qquad &w_0 &= \boldsymbol{\alpha_0 1'} = \boldsymbol{\delta 1'} = 1\\
&\tag{2} \qquad &\boldsymbol{\phi_0} &= \boldsymbol{\delta}\\
&\tag{3} & w_t \boldsymbol{\phi_t} &= w_{t-1} \boldsymbol{\phi_{t-1} \Gamma P} (x_t)\\
&\tag{4} & L_T = \boldsymbol{\alpha_T 1'} &= w_T (\boldsymbol{\phi_T 1'}) = w_T
\end{align}


Thus, the log-likelihood is

\begin{equation}\log L_T = \sum_{t=1}^T \log(\frac{w_t}{w_{t-1}}) = \sum_{t=1}^T \log(\boldsymbol{\phi_{t-1} \Gamma P} (x_t) \boldsymbol{1'})
(\#eq:scalelik)
\end{equation}


**Note:** 

$(1)$ follows from $w_0 = \boldsymbol{\alpha_0 1'} = \boldsymbol{\delta 1'} = \delta_1 + \cdots \delta_m = 1$

$(2)$ follows from $\boldsymbol{\phi_0} = \frac{\boldsymbol{\alpha_0}}{w_0} = \boldsymbol{\delta} \frac{1}{1} = \boldsymbol{\delta}$

$(3)$ follows from $\boldsymbol{\phi_t} = \frac{\boldsymbol{\alpha_t}}{w_t} \Rightarrow w_t \boldsymbol{\phi_t} = \boldsymbol{\alpha_t} = \boldsymbol{\alpha_{t-1} \Gamma P} (x_t) = w_{t-1} \boldsymbol{\phi_{t-1} \Gamma P} (x_t)$

$(4)$ follows from $L_T = \boldsymbol{\alpha_T 1'} = w_T(\boldsymbol{\phi_T 1'}) = w_T (\sum_T \phi_T) = w_T (\sum_T \frac{\alpha_T}{w_T} ) = w_T$

Equation \@ref(eq:scalelik) follows from $L_T = w_T = \frac{w_1}{w_0} \frac{w_2}{w_1} \cdots \frac{w_T}{w_{T-1}} = \prod_{t=1}^T \frac{w_t}{w_{t-1}} = (\boldsymbol{\phi_{t-1} \Gamma P} (x_t) \boldsymbol{1'}$, which follows from $(3)$ $w_t = w_{t-1} (\boldsymbol{\phi_{t-1} \Gamma P} (x_t) \boldsymbol{1'})$. 





## Reparameterization to Avoid Constraints {#reparam}

The parameters of the Markov chain, state-dependent distributions, and initial distributions must be reparameterized in order to use an unconstrained optimizer, such as `nlm`. 

### Transition Probabilities

For the transition probabilities $\gamma_{ij}$, to ensure that $\sum_{j} \gamma_{ij} = 1$ and $\gamma_{ij} \geq 0$, apply the following reparameterization:

1. Define a new matrix $\boldsymbol{T}$ with entries $\tau_{ij}$ for $i \neq j$

2. Choose a strictly increasing function $g: \mathbb{R} \rightarrow \mathbb{R}^+$

3. Define new entries $\nu_{ij}$ where 

\begin{equation} 
  \nu_{ij} = \left\{\begin{aligned}
  & g(\tau_{ij}) && \text{ for } i \neq j\\
  &1 && \text{ for } i = j\\
\end{aligned}
\right.
\end{equation} 

4. Maximize $L_T$ with respect to $\boldsymbol{T}$ (and state-dependent parameters and initial probabilities)

5. Transform back 

$$\hat{\gamma_{ij}} = \frac{\hat{\nu}_{ij}}{\sum_{k=1}^m \hat{\nu}_{ik}} = \frac{g^{-1} (\hat{\tau}_{ik})}{\sum_{k=1, k \neq i}^m g^{-1} (\hat{\tau}_{ik})} \qquad{\text{ for }} i, j = 1, 2, \dots, m$$. 


### Parameters of State-Dependent Distributions

For parameters of the state-dependent distribution, 

1. Define new parameters 

$\qquad{\text{i.}}$ If $X_t \sim \text{Poisson} (\lambda_i)$, then define $\eta_i = \log \lambda_i$ for $i = 1, \dots, m$

$\qquad{\text{ii.}}$ If $X_t \sim \text{Binomial} (p_i)$, then define $\eta_i = \log {\frac{p_i}{1 - p_i}}$ for $i = 1, \dots, m$

2. Maximize $L_T$ with respect to $\eta_i$ (and transition and initial probabilities)

3. Transform back 

$\qquad{\text{i. }} \hat{\boldsymbol{\lambda}} = \exp(\boldsymbol{\hat{\eta}})$

$\qquad{\text{ii. }} \hat{\boldsymbol{p}} = \frac{\exp(\boldsymbol{\hat{\eta}})}{1 + \exp{\boldsymbol{\hat{\eta}}}}$


### Initial Probabilities

For the initial probabilities $\delta_i$, to ensure that $\delta_i \geq 0$ and $\sum_i \delta_i = 1$,

1. Define new parameters

$$\pi_i = \log(\delta_i)$$

2. Maximize $L_T$ with respect to $\eta_i$ (and state-dependent parameters and transition probabilities)

3. Transform back 

$$\hat{\boldsymbol{\delta}} = \exp(\hat{\boldsymbol{\pi}})$$


## Strategies for Choosing Starting Values {#startval}

There may be multiple maxima (i.e. local maxima) in the likelihood. There is no simple method of determining in general whether a numerical maximization has reached the global maximum. A strategy is to use a range of starting values for the maximization, and see whether the same maximum is identified in each case.

Possible starting values for the state-dependent distribution could be to take values close to measures of location.

- E.g. For a two-state Poisson-HMM, use values slightly smaller and slightly larger than the mean as starting values.

- E.g. For a three-state Poisson-HMM, use the lower quartile, median, and upper quartile of the observed counts as starting values. 

Possible starting values for the transition probabilities could be to assign a common starting value to all the off-diagonal transition probabilities. 

Possible starting values for the stationary distribution could be to assign uniform probability to all the states. 


## Obtaining Standard Errors and Confidence Intervals {#boot}

**Parametric bootstrapping** uses fitted parameters $\hat{\boldsymbol{\Theta}}$ to generate **bootstrap** samples of observations, which can be used to evaluate properties of the model and obtain standard errors and confidence intervals. 

To obtain the variance-covariance matrix of $\boldsymbol{\Theta}$:

1. Fit the model (i.e. compute $\boldsymbol{\Theta}$)

2. a. Generate a sample of observations from the fitted model of the same length as the original number of observations (i.e. generate bootstrap samples from the model with parameters $\boldsymbol{\hat{\Theta}}$)
  
  b. Estimate the parameters $\boldsymbol{\Theta}$ by $\boldsymbol{\hat{\Theta}^*}$ for the bootstrap sample
  
  c. Repeat steps $(a)$ and $(b)$ for (a large number of) *B* times and record the values $\boldsymbol{\hat{\Theta}}^*$
  
3. Estimate the variance-covariance matrix of $\hat{\boldsymbol{\Theta}}$ by the sample variance-covariance matrix of the boostrap estimates $\boldsymbol{\hat{\Theta}}^* (b)$ for $b= 1, 2, \dots, B$

$$\widehat{\text{Var-Cov}} (\hat{\boldsymbol{\Theta}}) = \frac{1}{B-1} \sum_{b=1}^B (\hat{\boldsymbol{\Theta}}^* (b) - \hat{\boldsymbol{\Theta}}^*(\cdot))'(\hat{\boldsymbol{\Theta}}^* (b) - \hat{\boldsymbol{\Theta}}^*(\cdot))$$

where $\hat{\boldsymbol{\Theta}}^*(\cdot) = B^{-1} \sum_{b=1}^B \hat{\boldsymbol{\Theta}}^*(b)$

The bootstrap confidence intervals can be obtained by the 'percentile method' where the $100 - \alpha \%$ confidence interval has lower and upper limits of the $\frac{\alpha}{2}$- and $100 - \frac{\alpha}{2}-$ th percentile.  

**Note:** See Section 3.6.1 of the textbook for a discussion of obtaining standard errors and confidence intervals using the Hessian.


## Exercises

1. Consider the following parameterization of the tpm of an $m$-state Markov chain. Let $\tau_{ij} \in \mathbb{R} (i, j = 1, 2, \dots, m,; i \neq j)$ be $m(m-1)$ arbitrary real numbers. Let $g: \mathbb{R} \rightarrow \mathbb{R}^+$ be some strictly increasing function, e.g. $g(x) = e^x$. Define $\nu_{ij}$ and $\gamma_{ij}$ as from [the above](#reparam)

    a. Show that the matrix $\boldsymbol{\Gamma}$ with entries $\gamma_{ij}$ that are constructed in this way is a tpm.

    b. Given $m \times m$ tpm $\boldsymbol{\Gamma} = (\gamma_{ij})$, derive an expression for the parameters $\tau_{ij}$, for $i, j = 1, 2, \dots, m$; $i \neq j$. 


## Solutions

**Part a**
We will first show that the rows of $\boldsymbol{\Gamma}$ sum to 1. 

Let $i = 1, \dots, m$. Then

\begin{align*}
\sum_{j=1}^m \gamma_{ij}
&= \sum_{j=1}^m \frac{\nu_{ij}}{\sum_{k=1}^m \nu_{ik}}\\
&= \frac{1}{1 + \sum_{k=1, k \neq i}^m g(\tau_{ik})} + \sum_{k=1, k \neq i}^m \frac{g (\tau_{ik})}{1 + \sum_{k=1, k \neq i}^m g(\tau_{ik})}\\
&= \frac{1 + \sum_{k=1, k \neq i}^m g(\tau_{ik})}{1 + \sum_{k=1, k \neq i}^m g(\tau_{ik})}\\
&= 1
\end{align*}

Thus $\sum_{j=1}^m \gamma_{ij} = 1$.

Now we will show that the entries of $\boldsymbol{\Gamma}$ are between 0 and 1.

Since $g: \mathbb{R} \rightarrow \mathbb{R}^+$, it follows from our definition of $\gamma_{ij}$ that

\begin{align} 
  \gamma_{ij} &= \frac{\nu_{ij}}{\sum_{k=1}^m \nu_{ik}} = 
  \left\{\begin{aligned}
  &\frac{1}{1 + \sum_{k=1, k \neq i}^m g(\tau_{ik})} \geq \frac{1}{1 + 0} = 1 && \text{ if } i = j\\
  & \frac{g (\tau_{ij})}{1 + \sum_{k=1, k \neq i}^m g(\tau_{ik})} \geq \frac{0}{1 + 0} = 0 && \text{ if } i \neq j\\
\end{aligned}
\right.
\end{align} 

Thus, $\gamma_{ij} \geq 0$. 

Then 

$1 = \sum_{j=1}^m \gamma_{ij} \geq \gamma_{ij}$ since $\gamma_{ij}$ are non-negative. 

Thus,  $\gamma_{ij} \leq 1$.

Therefore, the constructed $\boldsymbol{\Gamma}$ is a tpm. 


**Part b**

Let $i, j = 1, 2, \dots, m$. Define $g^{-1}$ to be the inverse function of $g$, $g^{-1}: \mathbb{R}^+ \rightarrow \mathbb{R}$.

Then

\begin{align}
g^{-1} \left(\frac{\gamma_{ij}}{\gamma_{ii}}\right)
&= g^{-1} \left(\gamma_{ij} (\frac{1}{\gamma_{ii}}) \right)\\
&= g^{-1} \left(\gamma_{ij} (\frac{1 + \sum_{k=1, k \neq i}^m \nu_{ik}}{\nu_{ii}}) \right)
\qquad{\text{since } \gamma_{ii} = \frac{\nu_{ii}}{\sum_{k=1}^m \nu_{ik}}}\\
&= g^{-1} \left(\gamma_{ij} (\frac{1 + \sum_{k=1, k \neq i}^m g(\tau_{ik})}{1}) \right)
\qquad{\text{by definition of } \nu_{ii}}\\
&= g^{-1} \left(\gamma_{ij} (1 + \sum_{k=1, k \neq i}^m g (\tau_{ik})) \right)\\
&= g^{-1} \left(g (\tau_{ij})\right)
\qquad{\text{since } \gamma_{ij} = \frac{\nu_{ij}}{\sum_{k=1}^m \nu_{ik}} = \frac{g(\tau_{ij})}{1 + \sum_{k = 1, k \neq i}^m \nu_{ik}}, i \neq j}\\
&= \tau_{ij}
\end{align}














