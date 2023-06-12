# Bayesian Inference for HMMs

We apply Bayesian inference in the context of Poisson-HMMs with Gibbs sampling. 

**Note:** We use STAN for Bayesian inference in the [Earthquake Analysis](#eq) which uses Hamiltonian Monte Carlo sampling. 

## Reparameterization to Avoid Label Switching

Consider each observed count $x_t$ as the sum $\sum_j x_{jt}$ of contributions from up to $m$ regimes. That is, if the MC is in state $i$ at a given time, then regimes $1, \dots, i$ are all "active" and regimes $i+1, \dots, m$ are all inactive at that time. 

Then parameterizing the model in terms of the non-negative increments $\boldsymbol{\tau} = (\tau_1, \dots, \tau_m)$ where $\tau_j = \lambda_j - \lambda_{j-1}$ (with $\tau_0 \equiv 0$), or equivalently, $\lambda_i = \sum_{j=1}^i \tau_j$, the random variable $\tau_j$ can be thought of as the mean contribution of regime $j$ to the count observed at a give time, if regime $j$ is active. 

This parameterization has the effect of placing the $\lambda_j$ in increasing order, which is useful to prevent label switching. 


## Gibbs Sampling Procedure

1. Given the observed counts $\boldsymbol{x}^{(T)}$ and current values of the parameters $\boldsymbol{\Gamma}, \boldsymbol{\lambda}$, generate a sample path of the MC.

2. Use this sample path to decompose the observed counts into (simulated) regime contributions.

3. With the MC sample path available and regime contributions, update $\boldsymbol{\Gamma}, \boldsymbol{\tau}$. 

4. Repeat steps 1-3 for a large number of times, called a "burn-in period". Then repeat steps 1-3 to obtain the posterior estimates.


### Generating Sample Paths of the MC {#gen}

Given the observations $\boldsymbol{x}^{(T)}$ and current values of the parameters $\boldsymbol{\theta} = (\boldsymbol{\Gamma}, \boldsymbol{\lambda})$, simulate a sample path $\boldsymbol{C}^{(T)}$ of the MC. 

First, draw the state of the MC at the last time point $C_T$ from $\Pr(C_T|x^{(T)}, \boldsymbol{\theta}) \propto \alpha_T (C_T)$. 

This follows from

$$\Pr(C_t|\boldsymbol{x}^{(t)}, \boldsymbol{\theta}) = \frac{\Pr(C_t, \boldsymbol{x}^{(t)}|\boldsymbol{\theta})}{\Pr(\boldsymbol{x}^{(t)}|\boldsymbol{\theta})} = \frac{\alpha_t(C_t)}{L_t} \propto \alpha_t(C_t) \qquad{\text{for } t=1, \dots, T}$$ 

Next, draw the remaining states of MC from descending time points $C_{T-1}, C_{T-2}, \dots, C_{1}$, by the following 

\begin{align}
\Pr(C_t|\boldsymbol{x}^{(T)}, \boldsymbol{C}_{t+1}^T, \boldsymbol{\theta}) 
& \propto \Pr(C_t|\boldsymbol{x}^{(T)}, \boldsymbol{\theta}) \Pr(\boldsymbol{x}_{t+1}^{(T)}, \boldsymbol{C}_{t+1}^{(T)}|\boldsymbol{x}^{(t)}, C_t, \boldsymbol{\theta}) \\
& \propto \Pr(C_t|\boldsymbol{x}^{(t)}, \boldsymbol{\theta}) \Pr(C_{t+1}|C_t, \boldsymbol{\theta}) \Pr(\boldsymbol{x}_{t+1}^{(T)}, \boldsymbol{C}_{t+2}^{(T)}|\boldsymbol{x}^{(t)}, C_t, C_{t+1}, \boldsymbol{\theta}) \\
& \propto \alpha(C_t) \Pr(C_{t+1}|C_t, \boldsymbol{\theta})
\end{align}


### Decomposing the Observed Counts into Regime Contributions {#decom}

Given the sample path $\boldsymbol{C}^{(T)}$ of the MC (from step 1), suppose that $C_t = i$ so that regimes $1, \dots, i$ are active at time $t$. Decompose each observation $x_t (t=1, 2, \dots, T)$ into regime contributions $x_{1t}, \dots, x_{it}$ such that $\sum_{j=1}^i x_{jt} = x_t$ by 

$$f(x_{1t}, \dots, x_{it}|C_t = i, X_t = x_t) = \frac{x_t}{x_{1t}! \cdots x_{it}!} \tau_1^{x_{1t}} \cdots \tau_i^{x_{it}}$$

### Updating the Parameters

Suppose we assign the following priors,

$$\boldsymbol{\Gamma}_r \sim Dirichlet(\boldsymbol{\nu_r})$$

$$\tau_j = \lambda_j - \lambda_{j-1} \sim Gamma(\text{shape} = a_j, \text{rate} = b_j)$$

That is, 

the rows $\boldsymbol{\Gamma}_1, \boldsymbol{\Gamma}_2, \dots, \boldsymbol{\Gamma}_m$ have a Dirichlet distribution with parameters $\nu_1, \nu_2, \dots, \nu_m$ respectively. Hence, $f(\boldsymbol{\Gamma}_1, \boldsymbol{\Gamma}_2, \dots, \boldsymbol{\Gamma}_m) \propto \boldsymbol{\Gamma}_1^{\nu_1 -1} \cdots \boldsymbol{\Gamma}_m^{\nu_m -1}$ where $\sum_{i=1}^m \boldsymbol{\Gamma}_i = 1, \boldsymbol{\Gamma}_i \geq 0$. 

and the increment $\tau_j$ is such that $f(\tau_j) = \frac{b^a}{\Gamma(a)} x^{a-1} e^{-bx}$. Hence, $\tau_j$ have mean $\frac{a}{b}$, variance $\frac{a}{b^2}$, and coefficient of variation $\frac{1}{\sqrt{a}}$. 

Update $\boldsymbol{\Gamma}$ and $\boldsymbol{\tau}$ using the [MC path](#gen) and [regime contributions](#decom) by drawing 

$$\boldsymbol{\Gamma}_r \sim Dirichlet(\boldsymbol{\nu_r} + \boldsymbol{T}_r)$$ where $\boldsymbol{T}_r$ is the $r$-th row of the (simulated) matrix of transition counts

and 

$$\tau_j \sim Gamma(a_j + \sum_{t=1}^T x_{jt}, b_j + N_j)$$ where $N_j$ denotes the number of times regime $j$ was active in the simulated sample path of the MC and $x_{jt}$ is the contribution of regime $j$ to $x_t$.

**Note:** The posterior distributions for the above follow from the fact that observations of regime contributions $x_{i1}, \dots, x_{it}$ where the variables $x_{jt} \sim Poisson(\tau_j)$ and $x=\sum_j x_{jt}$, $f(x_{1t}, \dots, x_{it}|C_t = i, X_t = x_t) = \frac{x_t}{x_{1t}! \cdots x_{it}!} \tau_1^{x_{1t}} \cdots \tau_i^{x_{it}}$, $\boldsymbol{T}_r \sim Dirichlet(\boldsymbol{\nu}_r)$, and $\tau_j \sim Gamma(a, b)$. 


### Repeat the Above

Repeat steps 1 to 3 for a large number of samples, called the "burn-in period". Now repeat steps 1 to 3 for the posterior estimates. 


## Exercises

1. Consider $u$ defined by $u=\sum_{j=1}^i u_j$, where the variables $u_j$ are independent Poisson random variables with means $\tau_j$. Show that, conditional on $u$, the joint distribution $u_1, u_2, \dots, u_i$ is a multinomial with total $u$ and probability vector $\frac{(\tau_1, \dots, \tau_i)}{\sum_{j=1}^i \tau_j}$. 

2. Let $\boldsymbol{w}=(w_1, \dots, w_m)$ be an observation from a multinomial distribution with probability vector $\boldsymbol{y}$, which has a Dirichlet distribution with parameter vector $\boldsymbol{d} = (d_1, \dots, d_m)$. Show that the posterior distribution $\boldsymbol{y}$ is the Dirichlet distribution with parameters $\boldsymbol{d} + \boldsymbol{w}$.

3. Let $y_1, y_2, \dots, y_n$ be a random sample from the Poisson distribution with mean $\tau$, which is gamma distributed with parameters $a$ and $b$. Show that the posterior distribution of $\tau$, is the gamma distribution with parameters $a+\sum_{i=1}^n y_i$ and $b+n$. 


**Question 1**

Since $u= \sum_{j=1}^i u_j$, $u_j \sim Poisson(\tau_j)$, and the $u_j$'s are independent, it follows that $u \sim Poisson(\sum_{j=1}^i \tau_j)$, hence $f(u) = \frac{\sum_{j=1}^i \tau_j^u e^{- \sum_{j=1}^i \tau_j}}{u!}$. 

Then

\begin{align}
f(u_1, \dots, u_i|u)
&= \frac{f(u_1, \dots, u_i, u)}{f(u)}\\
&= \frac{\left(\frac{\tau_1^{u_1} e^{- \tau_1}}{u_1 !}\right) \cdots \left(\frac{\tau_j^{u_i} e^{- \tau_j}}{u_i !}\right)}{\frac{\sum_{j=1}^i \tau_j^u e^{- \sum_{j=1}^i \tau_j}}{u!}}\\
&= \frac{1}{u_1! \cdots u_i!} \left(\tau_1^{u_1}\right) \cdots \left(\tau_j^{u_i}\right) \left(e^{-\sum_{j=1}^i \tau_j}\right) \left(\frac{u!}{\sum_{j=1}^i \tau_j^u e^{- \sum_{j=1}^i \tau_j}}\right)\\
&= \frac{u!}{u_1! \cdots u_i!} \left( \tau_1^{u_1} \right) \cdots \left( \tau_j^{u_i} \right) \frac{1}{\sum_{j=1}^i \tau_j^{u_1 + \dots + u_i}}\\
&= \frac{u!}{u_1! \cdots u_i!} \left( \frac{\tau_1}{\sum_{j=1}^i \tau_j} \right)^{u_1} \cdots \left( \frac{\tau_j}{\sum_{j=1}^i \tau_j} \right)^{u_i}
\end{align}

Thus, $u_1, \dots, u_i|u \sim Multinomial \left(\frac{\tau_1, \dots, \tau_i}{\sum_{j=1}^i \tau_j}; u \right)$.


**Question 2**

The posterior distribution of $\boldsymbol{y}$ is 

\begin{align}
\Pr(\boldsymbol{y}|\boldsymbol{w}) 
&= \frac{\Pr(\boldsymbol{y}, \boldsymbol{w})}{\Pr(\boldsymbol{w})}\\
&= \frac{\Pr(\boldsymbol{w}|\boldsymbol{y})\Pr(\boldsymbol{y})}{\Pr(\boldsymbol{w})}\\
& \propto \Pr(\boldsymbol{w}|\boldsymbol{y}) \Pr(\boldsymbol{y})\\
& \propto \frac{n!}{w_1! \cdots w_m!} y_1^{w_1} \cdots y_2^{w_2} \cdots y_m^{w_m} \cdot y_1^{d_1 -1} \cdot y_2^{d_2-1} \cdots y_m^{d_m-1}\\
& \propto y_1^{(w_1 + d_1)-1} y_2^{(w_2+d_2)-1} \cdots y_m^{(w_m + d_m)-1}
\end{align}

Thus, $\boldsymbol{y}|\boldsymbol{w} \sim Dir(\boldsymbol{w}+\boldsymbol{d})$. 


**Question 3**

The posterior distribution of $\tau$ is 

\begin{align}
\Pr(\tau|y_1, \dots, y_n)
&= \frac{\Pr(\tau_1, y_1, \dots, y_n)}{\Pr(y_1, \dots, y_n)}\\
&= \frac{\Pr(y_1, \dots, y_n|\tau) \Pr(\tau)}{\Pr(y_1, \dots, y_n)}\\
& \propto \Pr(y_1, \dots, y_n|\tau) \Pr(\tau)\\
&= \prod_{i=1}^n \frac{\tau^{y_i} e^{-\tau}}{y_i !} \frac{b^a}{\Gamma(a)} \tau^{a-1} e^{-b \tau}\\
& \propto \prod_{i=1}^n \tau^{y_i} e^{- \tau} \tau^{a-1} e^{-b \tau}\\
&= \tau^{\sum_{i=1}^n y_i} e^{-n \tau} \tau^{a-1} e^{-b \tau}\\
&= \tau^{a+\sum_{i=1}^n y_i - 1} e^{-(b+n)\tau}
\end{align}

Thus, $\tau|y_1, \dots, y_n \sim Gamma(a+\sum_{i=1}^n y_i, b+n)$. 




