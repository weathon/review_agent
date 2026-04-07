## Summary
The paper proposes adaptive SWIM (a-SWIM), which integrates trainable rational activation functions into the sampling-based neural network training framework SWIM. The key idea is to determine activation parameters through localized sub-optimization problems rather than global backpropagation, preserving sampling efficiency while improving approximation accuracy. Experiments on six PDE-related function approximation tasks demonstrate that a-SWIM often outperforms fixed-activation SWIM variants and achieves training times roughly 20× faster than backpropagation-based networks, though with some accuracy trade-offs.

## Strengths
- **Clear methodological contribution**: The paper addresses a genuine gap—SWIM and related sampling-based methods have only employed fixed activation functions. Combining adaptive activations (rational functions) with SWIM's sampling framework is a natural extension with demonstrated empirical benefit. Section 4 clearly explains the xu-point set construction, local parameter optimization, and neuron selection pipeline.
- **Empirical improvement over SWIM**: Across six objective functions (KdV sine, Advection, Euler-Bernoulli, Burgers, Discontinuous Trivial/Complicated), a-SWIM achieves the best MSE on 3 tasks and remains competitive (never worst) on the others (Tables 1a-1f). For KdV sine and Euler-Bernoulli, the improvement is substantial—orders of magnitude lower MSE at wider network configurations.
- **Training efficiency preserved**: a-SWIM retains the key advantage of sampling-based methods: training completes in seconds (~12-14s) compared to minutes (~800s) for BP-NNs (Table 2b). The 2× slowdown relative to standard SWIM is a reasonable trade-off given the accuracy gains.
- **Honest limitation disclosure**: The authors acknowledge the single-layer restriction, pole instability concerns, poor performance on high-frequency/discontinuous functions, and increased parameter count relative to SWIM (Section 6.1).

## Weaknesses
- **Misleading "no gradient-based optimization" claim**: The abstract states the method enables learning "without gradient-based optimization." However, Section 4.3 explicitly employs Adam optimizer for each neuron's adaptive parameters. The distinction between network-wide backpropagation and local per-neuron gradient descent is meaningful but should be stated accurately. The abstract overstates the contribution.
- **Title oversells PDE solving capability**: The title claims "Application to the Solution of PDEs," but the method performs supervised function approximation on pre-simulated solution data. No PDE residual loss or boundary condition enforcement is incorporated during training. The method cannot solve PDEs without ground truth data—this is a significant limitation given the title's framing.
- **No ablation of design choices**: The paper introduces three probability strategies (variance, cosine, coefficient), two loss functions (MSE, cosine), and multiple initialization methods, yet only reports results using the default configuration (variance probability, MSE loss, ReLU-like initialization). Without systematic ablation, it is unclear which components contribute to performance.
- **Limited architectural scope**: The entire method is restricted to single-hidden-layer networks. While multi-layer SWIM exists (Bolager et al., 2023), the adaptive version is unexplored. This limits expressivity compared to deep BP-NNs and makes the parameter-count comparison somewhat asymmetric.
- **Heuristic probability derivations rely on acknowledged false assumptions**: Section 4.4 derives probability strategies under assumptions ($\mathbf{F}^T\mathbf{F} = \mathbf{\Lambda}$ or $\mathbf{I}$) that the paper admits do not hold. While the authors correctly note these serve as "rough guidance," no analysis validates whether these heuristics outperform simpler alternatives (e.g., uniform sampling).
- **Mixed empirical results**: On Burgers and discontinuous functions, a-SWIM underperforms ReLU-SWIM. The claim that a-SWIM is "never the worst" is accurate but modest—the method is not uniformly superior to the simplest baseline.

## Nice-to-Haves
- **Physics-informed formulation**: Extending the method to minimize PDE residuals directly (without pre-simulated ground truth) would align the contribution with the title and significantly broaden applicability.
- **Ablation study**: Systematic comparison of the three probability strategies, two loss functions, and initialization methods would clarify which design choices matter.
- **Statistical reporting**: Reporting mean and standard deviation over multiple random seeds would strengthen empirical reliability claims.
- **Training time breakdown**: Analyzing where computation time is spent (sub-optimization vs. output weight solving vs. data processing) would help readers understand scaling behavior.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Missing comparison with PINNs/neural operators**: The paper positions itself at the intersection of sampling-based methods and adaptive activations. Demanding comparison with PINNs or neural operators would be scope creep—the authors chose SWIM and BP-NNs as their competitive set, which is reasonable for a methods paper focused on training mechanism innovation.
- **High-dimensional experiments absent**: The introduction references the curse of dimensionality as motivation (citing Grohs et al.), but this is background context. The paper's contribution is the adaptive activation integration; testing high-dimensional scaling is a separate research question.
- **Cross-reference organization**: The harsh critic claims section numbering is inconsistent. Checking the paper, Section 7.3 (experiments), Section 6 (conclusions), and Section 6.2 (outlooks) all exist as referenced. The appendix placement of detailed experiments is unusual but does not impede understanding.
- **BP-NN activation function inconsistency**: Experiment 3 uses rational activations for BP-NN (accuracy comparison) while Experiment 4 uses adaptive Tanh (time comparison). This is noted but is a minor point—different experiments may legitimately use different configurations.

## Novel Insights
The paper demonstrates that adaptive activation functions—previously validated only in backpropagation-trained networks—can be integrated into sampling-based training with measurable accuracy gains. The localized sub-optimization approach (per-neuron Adam optimization on small point sets) provides a principled way to determine adaptive parameters without network-wide gradient computation. The finding that error distributions differ qualitatively (a-SWIM produces smoother, more uniform error maps while fixed-activation SWIM shows localized high-error spots) suggests the adaptive approach captures different representational characteristics.

## Suggestions
- Revise the abstract to accurately describe the local optimization process: the method avoids global backpropagation but uses per-neuron gradient-based optimization for adaptive parameters.
- Retitle or qualify the scope: "Function Approximation for PDE Solutions" would be more accurate than "Solution of PDEs."
- Add a brief ablation comparing at least the three probability strategies on one or two representative functions to justify the default choice.
- Include an explicit statement about the computational overhead of sub-optimization (K=5 points per neuron, Adam iterations) and how it scales with pool size N and network width M.