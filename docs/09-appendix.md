# Appendix

An intuitive and less laborious way to establish properties of the HMM is to invoke properties of directed graphical models.

In a directed graphical model, the probability of a set of random variables $\{V_1, V_2, \dots, V_n\}$ can be factored into a product of conditional probabilities, one for each parent (node) $\text{pa} (V_i)$

\begin{equation}
\Pr(V_1, V_2, \dots, V_n) = \prod_{i=1}^n \Pr(V_i| \text{pa} (V_i))
(\#eq:directed)
\end{equation}

In the case of HMMs, 

- $C_1$ has no parents

- $\text{pa} (X_k) = C_k \qquad{\text{for } k = 2, 3, \dots}$

- $\text{pa} (C_k) = C_{k-1} \qquad{\text{for } k = 2, 3, \dots}$

Then the joint distribution of $\boldsymbol{X}^{(t)}$ and $\boldsymbol{C}^{(t)}$  is given by 

\begin{equation}
\Pr(\boldsymbol{X}^{(t)}, \boldsymbol{C}^{(t)}) = \Pr(C_1) \prod_{k=2}^t \Pr(C_k|C_{k-1}) \prod_{k=1}^t \Pr(X_k|C_k)
(\#eq:joint)
\end{equation}


