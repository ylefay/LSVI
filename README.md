# What is LSVI?

LSVI stands for Least Square Variational Inference.
It is a package for minimizing the KL divergence of an (unnormalized) distribution to a distribution within a chosen
exponential family. Such procedures are typically used to approximate a posterior distribution in a Bayesian setting by
an easy-to-sample from distribution, to estimate posterior-based quantities.

# How is it done in this package?

The first-order optimality condition $\nabla_{\eta}KL(q_\eta\mid \pi)$ with $\pi \propto \exp(f(x))$ and $q_{\eta}(x) = \exp(\eta^{\top}s(x))$ suggests the
following scheme

```math
\eta_{t+1} = (1-\varepsilon_t) \eta_t+\varepsilon_t \arg\min_{\eta\in \mathbb{R}^m} \mathbb{E}_{\eta_t}(\eta^{\top}s(X) - f(X))^2,
```

which we iterate until convergence.
All the involved expectations are approximated by Monte Carlo sampling.

## Fast inference for Gaussian distributions in $O(n^3)$
When the variational family is the set of full-rank Gaussian distributions, the previous iteration requires inversion of $n^2 \times n^2$ matrices.
A reparametrization trick allows us to avoid this computational burden, making exact inference possible with optimal complexity, i.e., $O(n^3)$ operations.

# How to use it?

See `LSVI/experiments` for some examples, including the variational approximation of the posterior of a logistic
regression on a real dataset, the variable selection problem in a linear regression using the variational family of
product of Bernoullis, and a stochastic movement model with intractable likelihood.

# References
Yvann Le Fay, Nicolas Chopin, Simon Barthelmé. Least squares variational inference. 2025. [⟨hal-04963327⟩](https://hal.science/hal-04963327)
