# Feature Learning for the High Dimensional Stationary Sch\"odinger Equation with Deep Ritz Method

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
This paper investigates feature learning within the framework of the deep Ritz method for solving the stationary Schr\"odinger equation with Neumann boundary conditions. We first analyze the convergence of Riemannian gradient descent in an agnostic setting, where the hypothesis function is restricted to a single-index model while the PDE solution is arbitrary. We prove that gradient descent reaches an approximate global minimum: after $T = O(\log(1/\epsilon))$ iterations, the loss is within $\epsilon$ of a constant multiple of the optimal loss. We then examine the loss landscape when the source term of the PDE itself follows a single-index model, considering hypothesis functions given by either a single-index model or a two-neuron multi-index model. In the single-index case, we show that the minimum Ritz energy is attained at the feature vector aligned with that of the source term. In the two-neuron case, we study the landscape of regularized Ritz losses and characterize how a second feature emerges, given that the first feature is aligned with the source, as the regularization parameter varies. Finally, numerical experiments are presented to validate the feature emergence theory in the two-neuron setting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper theoretically analyzes feature learning in simple neural networks when solving high-dimensional stationary Schrödinger equations via the Deep Ritz method. For a single-index (single-neuron) model, they derive convergence rates for Riemannian gradient descent to an approximate global minimum in a general agnostic setting. In the realizable setting, they prove that the minimum Ritz energy is attained when the feature aligns with the feature of the source term. For a two-neuron model (with one feature fixed), the study reveals that the emergence of a new, distinct feature depends critically on the $\ell_2$-regularization strategy: distinct features emerge when only penalizing the outer-layer weight of the second neuron, whereas strong penalization of both outer weights leads to feature collapse. Finally, numerical experiments are provided to validate the results on the loss landscapes and phase transitions for the two-neuron case.

### Strengths
- The paper tackles a gap in the theory of deep learning for PDEs. Previous work either only covers generalization & approximation results, or optimization in the "lazy" (NTK) or infinite-width (mean-field) regimes, which often fail to capture how finite-width networks learn relevant features.
- In particular, the authors provide a convergence result in a fully agnostic setting. In the simpler, realizable setting, they are able to derive a clear and interpretable result on how regularization strategies govern feature learning, i.e., encourage/discourage the network from finding new features.
- The theoretical findings for the two-neuron model are well-supported by numerical experiments, validating the predicted loss landscapes, phase transitions, and locations of the minimizers

### Weaknesses
- The analysis relies on very simple models (one/two-neuron networks) with a squared ReLU activation. While such a simplification is necessary for a tractable theoretical analysis, it is very far from the architectures used in practice. Moreover, the results do not cover other differentiable activation functions, such as tanh, sigmoid, and the more commonly used GeLU, ELU, and SiLU activations.
- The related works on neural PDE solvers seem to be missing a series of stochastic/SDE-based neural PDE solvers for elliptic equations; see https://arxiv.org/pdf/2112.03749, https://arxiv.org/abs/2001.06145, https://arxiv.org/pdf/2406.03494.
- It should be clearly distinguished which results are derived from (or minor adaptations of) existing results for single/multi-index models (as in Wu (2022); Awasthi et al. (2023); Wang et al. (2023); Gollakota et al. (2023); Zarifis et al. (2024); Diakonikolas et al. (2024)).
- The results only hold for a single PDE (stationary Schrödinger equation) with specific (Neumann) boundary conditions, since the analysis relies on the variational (energy minimization) structure of this elliptic problem. In particular, it is unclear which findings would generalize to other types of PDEs or methods (e.g., PINNs or SDE-based).
- The two-neuron analysis hinges on the assumption that the first feature is *already fixed* to be perfectly aligned with the source feature. This sidesteps the much harder question of the *optimization dynamics* that would lead to this state. Thus, the analysis is purely of the landscape *given* this optimal first feature, not of the full training process to find both features simultaneously.
- The guarantees in the agnostic case only provide a converge results w.r.t. the *best possible loss* achievable by the single-index class. While one could employ approximation results for neural network hypothesis classes, a single-index model is typically a very poor approximation for practically relevant PDEs, making the convergence guarantee practically weak.

Minor: Some readers might be unfamiliar with the term “single- or multi-index models” and it should be explained in the introduction.

### Questions
See weaknesses above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates feature learning in small neural networks trained with the deep Ritz method (DRM) to solve a stationary Schr\"odinger-type PDE on the unit ball with Neumann boundary conditions. DRM minimizes an energy whose gradient depends only on the known source $f$, not the unknown solution, and the authors study two tiny hypothesis classes: a single-index model $u(x)=\sigma^2(w\cdot x)$ and a two-neuron model $u(x)=a_1\sigma^2(w_1\cdot x)+a_2\sigma^2(w_2\cdot x)$ with squared-ReLU. Directions live on the sphere and are trained by simple Riemannian gradient descent. \\
Result 1: even if the single-index class is misspecified, this training converges quickly to nearly the best value achievable within that class (fast, finite-width optimization). \\
Result 2: if the source is single-index $f(x)=\sigma^2(w^\star\cdot x)$, the learned feature aligns with $w^\star$ (identifiability). \\
Result 3: in the two-neuron case, a high-dimensional analysis reduces the regularized DRM objective to a landscape in the angle $\theta$ between $w_2$ and $w^\star$: joint ridge on $(a_1,a_2)$ induces feature collapse ($\theta=0$) beyond a threshold, whereas penalizing only $a_2$ always yields a nonzero minimizer, so a genuinely new feature emerges. Overall, the work shows that DRM can learn and control finite-width features, with regularization acting as a knob between reuse and diversity.

### Strengths
A key strength is that the paper tackles finite-width feature learning head-on: by working with tiny models and the deep Ritz objective, it avoids the usual infinite-width or NTK linearization and still proves concrete optimization guarantees, identifiability under a single-index source, and a crisp, angle-based account of when a second feature emerges or collapses. The energy-based training is appealing in practice because gradients depend only on the known forcing, and the use of Riemannian gradient descent on the sphere keeps the algorithm simple and transparent. The analysis leverage of squared-ReLU and closed-form angular kernels is elegant, giving an interpretable one-dimensional landscape in high dimension and a clean picture of how regularization choice acts as a knob between reuse and diversity; the theory is paired with numerics that qualitatively match the predicted regimes.

### Weaknesses
In my opinion the weakness of this approach is that the scope is quite narrow: the PDE is a specific Schr\"odinger-type model on the unit ball with Neumann boundaries, activations are fixed to squared-ReLU, sampling is uniform, and most sharp statements are for one or two neurons with some results relying on a high-dimensional limit, so it is unclear how thresholds and guarantees translate to more realistic geometries, boundary conditions, activations, or wider networks. The identifiability result leans on a single-index structure in the source, which is strong and may fail in multi-directional or noisy settings; the convergence bound in the agnostic case is constant-factor to the best-in-class rather than globally optimal, and practical performance will still depend on step-size, sample budget, and regularization tuning. Finally, while the paper clarifies the mechanisms inside DRM, it offers limited empirical breadth and lacks head-to-head comparisons against alternative PDE training objectives (e.g., residual-based PINNs) on more challenging benchmarks, so the external validity and scalability of the insights remain to be established.

### Questions
1. Which activation characteristics—such as polynomial degree, Cho-Saul integral moment structure, and smoothness at 0—are absolutely required for the two-neuron phase behavior and identifiability to hold? Do smooth sigmoids, GELU, or ReLU maintain the same angle landscape?

2. Can you quantify (theoretically or empirically) when the deep Ritz objective is better/worse than residual-based PINNs for the same PDE and model size?

3. How are the optimization landscapes and conditioning different for matched architectures and sampling, and in which regimes does DRM avoid collapse or produce better feature alignment than residual minimization, as demonstrated (or empirically, in controlled tests)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies feature learning for PDEs within the framework of the deep Ritz method applied to the stationary Schrödinger equation. By restricting the hypothesis space to a single-index model, the authors prove that Riemannian gradient descent converges to a loss within  $\epsilon$ of a constant multiple of the optimal loss. They also analyze the minimizers of the energy functional under two models, assuming the source term is itself a single-index function.

### Strengths
- The paper focuses on the optimization aspect of neural PDE solvers, a fundamental but comparatively less understood topic.
- To my knowledge, using single-index models to study neural PDE solvers is novel in the literature.
- The presentation is well organized, and the writing is clear.

### Weaknesses
Overall, the paper analyzes optimization in rather restrictive settings. These assumptions seem chosen primarily to make the analysis feasible, but they limit the relevance of the results to practical PDE solving. If I could rate on a continuous scale, my score would be a 3, between the criteria for 2 and 4.

Specific concerns:
- The link function is restricted to squared ReLU, with no justification. This is a strong and arguably artificial assumption. Additionally, the proof appears challenging to extend directly to other link functions.

- The convergence result only guarantees reaching a minimum within a factor $\gamma>1$ of the global minimum. But within such a narrow hypothesis space, the global minimum itself may be large. In that case, the result is of limited significance.

- Boundary conditions are essential in PDEs, yet they are not incorporated into the algorithmic framework at all.

### Questions
- Related to the second weakness: do the authors have any approximation results showing that the global minimum in the single-index setting is actually small? This would help justify the meaningfulness of the convergence guarantee.

- Can anything be said if a boundary penalty is added to the loss? Alternatively, are there settings in which the hypothesis class inherently satisfies the boundary conditions?

- In Section 2.3, the assumption $w_1 = w^*$ seems to imply a staged training process: first learn one index fully, then optimize the second. If all parameters are trained jointly (as in practice), can the authors comment anything on the resulting landscape?

- The numerical experiments rely on simplifications that $d \to \infty $. Can the authors provide experiments for the original finite-dimensional problem to give any insight into practical relevance?

### Soundness
3

### Presentation
3

### Contribution
2
