## Summary

This paper presents a theoretical framework explaining the dynamical simplicity bias in neural networks—where gradient descent learns solutions of increasing complexity over time—as a consequence of saddle-to-saddle dynamics. The authors prove the existence of a nested hierarchy of embedded fixed points (Theorem 1) and invariant manifolds (Theorem 3) for a broad class of architectures (fully-connected, convolutional, attention-based) defined by a unified layer equation. They then analyze the learning dynamics for two-layer linear and quadratic networks, identifying two distinct timescale separation mechanisms: data-induced separation (yielding low-rank weights) and initialization-induced separation (yielding sparse weights), and validate predictions about how width, data distribution, and initialization affect plateau dynamics.

## Strengths

- **Unified architectural treatment of fixed points and invariant manifolds.** Theorems 1 and 3, along with Corollary 2, apply to deep networks with FC, convolutional, and attention layers under a single framework (Equation 1). This is a genuine generalization of prior work (e.g., Fukumizu & Amari, 2000) that was restricted to fully-connected networks. The extension to new embedded fixed point constructions (Equations 6, 7) beyond the classical ones (Equations 4, 5) is both novel and essential, as the authors show the saddles visited during training fall under these new categories.

- **Disentanglement of two distinct timescale separation mechanisms.** The paper identifies that linear networks exhibit data-induced timescale separation (between directions across units, yielding low-rank weights) while quadratic/attention networks exhibit initialization-induced timescale separation (between units, yielding sparse weights). This distinction is genuinely novel—prior literature on saddle-to-saddle dynamics did not separate these mechanisms—and produces qualitatively different predictions (e.g., width scaling affects one but not the other, as shown in Figure 2A).

- **Predictive power validated through controlled experiments.** The framework makes specific, testable predictions: (i) increasing width shortens plateaus in self-attention but not linear FC nets (Figure 2A); (ii) flattening the data spectrum eliminates plateaus in linear nets but only shortens them in self-attention (Figure 2B); (iii) large low-rank initialization induces saddle-to-saddle dynamics without an initial plateau (Figure 2C). These are non-trivial predictions that go beyond post-hoc explanation.

## Weaknesses

### Major:

- **Gap between broad title/abstract claims and the scope of rigorous dynamical analysis.** The title promises an explanation "across neural network architectures," and the abstract discusses deep networks, but the core dynamical analysis (Section 5) is rigorously developed only for two-layer networks with homogeneous polynomial activations (linear and quadratic). The extension to deep networks is a conjecture (Section 7, "we conjecture that the order of the activation function...continues to predict learning behaviors"), and the paper explicitly acknowledges that general nonlinear activations like tanh do not satisfy the invariant manifold conditions needed for saddle-to-saddle dynamics (Section 7: "rank-one weights do not correspond to an invariant manifold with effective width one. Consequently, tanh networks are not guided to approach the saddle with one effective unit, and probably do not have saddle-to-saddle dynamics in general"). This excludes widely-used smooth activations (GELU, Swish, GLU variants) from the rigorous theory. While the authors are transparent about these limits in the body, the abstract and title do not reflect them—claiming the framework explains simplicity bias "across neural network architectures" without qualifying that the dynamical mechanism is proven only for a restricted class is misleading for a venue like ICLR where precision of claims matters.

- **The theory requires small initialization (ε→0), which is atypical in modern practice.** Both Theorem 4 and Proposition 5 rely on asymptotically small initialization for the timescale separation to emerge. Figure 2D shows that increasing initialization scale gradually weakens plateaus, and the paper acknowledges that "neural networks with large random initialization generally do not exhibit saddle-to-saddle dynamics." Standard initialization schemes (Xavier, He) are tuned for variance propagation, not for being small in the asymptotic sense. While the paper's contribution as a theoretical framework is clear, the practical relevance of the mechanism depends on whether the small-initialization regime is actually operative in real training pipelines. The paper could strengthen its case by discussing which practical settings (e.g., specific learning rate schedules, weight decay, or layer-wise initialization choices) might place training in a regime where this mechanism is relevant, even if approximately.

- **Empirical validation is limited to controlled synthetic settings and small-scale tasks.** The experiments use 2D synthetic data (Figures 1, 2, 4, 5) or binary MNIST classification with two-layer networks (Figure 3). While these are appropriate for validating theoretical predictions in controlled conditions, they leave open the question of whether saddle-to-saddle dynamics and the associated simplicity bias mechanism operate in the training of modern architectures at scale. The MNIST experiments (Figure 3) show the phenomenon persists with real data but with significant noise, and the paper does not demonstrate that the specific predictions (e.g., about width scaling or plateau duration) hold beyond the synthetic setting.

### Minor:

- **Theorem 3 establishes invariant sets, not attractors.** The theorem proves that if weights start on a manifold satisfying certain constraints, they remain there. It does not prove that gradient flow converges to these manifolds from generic initialization. The paper attempts to fill this gap via timescale separation arguments in Section 5, which is reasonable, but the distinction between "invariant" and "attracting" could be sharper. The text uses phrasing like "steers dynamics toward invariant manifolds" (Section 4), which implies attraction without proof.

- **The softmax self-attention experiment (Figure 4A) lacks theoretical backing.** The paper's dynamical theory in Section 5.2 covers linear (quadratic) attention, and the framework's invariant manifold conditions rely on homogeneity. Softmax breaks this homogeneity. Figure 4A shows stage-like dynamics in a softmax attention model, but this is presented without analysis of why or whether the same mechanism applies. This is a notable gap given that softmax attention is the dominant architecture.

### Trivial:

- The dimensionality of $v_i$ in Equation (1) changes between scalar (FC layers) and vector/matrix (attention), which requires careful cross-referencing but does not affect correctness given the consistent notation in Appendix D.

## Nice-to-Haves

- Analysis of how batch normalization or layer normalization interacts with the invariant manifold structure, since these layers break weight homogeneity required for Theorem 3(iii)-(iv).

- A discussion connecting the effective-width notion of simplicity to other common definitions (e.g., Kolmogorov complexity, description length), since the simplicity bias literature invokes multiple notions.

- Experiments with stochastic gradient descent (rather than full-batch gradient flow) to demonstrate that the plateaus and saddle-to-saddle transitions survive optimization noise, which is more relevant to practice.

- A quantification of the minimum singular value gap required for observable plateaus in the linear case, which would delineate the theory's domain of applicability for real data spectra.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Demand for CIFAR-10/ImageNet experiments with deep ResNets/ViTs.** This is scope creep. The paper is a theory paper with controlled experiments designed to validate specific theoretical predictions. Demanding large-scale benchmarks shifts the contribution type entirely.

- **Missing comparison against NTK spectral bias baselines.** The paper explicitly discusses the difference from NTK/kernel regime in Appendix A.2, noting that NTK dynamics exhibit smooth exponential decay rather than plateaus. A formal experimental comparison is unnecessary given this clear theoretical distinction.

- **Error bars / statistical significance for Figure 2.** For controlled synthetic experiments validating asymptotic theoretical predictions, single-run demonstrations are standard. The concern about stochasticity is more relevant for noisy real-data settings.

- **Broader impact statement.** Not required for ICLR and outside the paper's scope.

- **Missing related work citations (e.g., Chen et al. 2023 on symmetry-induced saddles).** Per the hard rules, I cannot confirm the existence of specific uncited works and should not flag missing related work.

- **Formatting/notation density complaints.** Per hard rules, formatting nitpicks are removed.

- **Reproducibility concerns about undisclosed hyperparameters.** Per hard rules, trivial implementation details are removed. The paper provides explicit hyperparameters in Appendix I.

- **Gradient flow vs. discrete SGD as a "limitation."** The paper explicitly states it analyzes gradient flow and this is standard practice in the theoretical deep learning literature. Flagging this as a weakness rather than a known modeling choice would be applying standards not standard in the field.

## Novel Insights

The disentanglement of data-induced versus initialization-induced timescale separation is the paper's most insightful contribution. It reveals that the *source* of the timescale separation—whether it arises from the data spectrum (producing distributed, low-rank representations) or from the randomness of initialization (producing sparse, localized representations)—fundamentally determines the *type* of simplicity bias a network exhibits. This has a concrete architectural implication: linear self-attention, being quadratic in the weights, inherits initialization-induced separation and thus sparse features, while linear fully-connected networks inherit data-induced separation and thus low-rank features. This predicts that scaling width should accelerate learning in attention architectures but not in linear FC networks—a non-obvious architectural distinction with potential practical consequences. The observation that large low-rank initialization can produce saddle-to-saddle dynamics *without* an initial plateau (Figure 2C) is also novel and nuances the common view equating exponential loss curves with lazy learning.

## Suggestions

- Qualify the title and abstract to reflect that the *dynamical mechanism* is rigorously established for two-layer networks with homogeneous/linear activations, while the fixed point and invariant manifold results apply more broadly. For example, "Saddle-to-Saddle Dynamics Explains a Simplicity Bias Across Neural Network Architectures: A Framework with Proofs for Two-Layer Homogeneous Networks."

- Add a brief table or paragraph categorizing common activations (ReLU, LeakyReLU, GELU, Swish, tanh, quadratic) by which conditions in Theorems 1 and 3 they satisfy, so practitioners can immediately assess applicability.

- Discuss explicitly which practical training settings (e.g., small learning rate with weight decay driving effective weights toward zero, or specific layer-wise initialization schemes) might place modern training in a regime where the small-initialization approximation is relevant, even if approximately.

- For the softmax attention experiment (Figure 4A), either add a brief analysis of why stage-like dynamics might persist despite broken homogeneity (e.g., near-zero weights make softmax approximately linear), or explicitly frame it as an empirical observation beyond the current theory's scope.

---

**Axis Evaluations:**

- **Novelty:** Strong. The unified treatment of embedded fixed points and invariant manifolds across FC/conv/attention architectures, the new fixed point constructions (Equations 6, 7), and the disentanglement of data- vs. initialization-induced timescale separation are all genuine contributions beyond prior work.

- **Technical soundness:** Good within its stated assumptions. Theorems 1 and 3 are rigorously proven for the general architecture class; the dynamical analysis is sound for two-layer homogeneous networks. The gap is that the most interesting claims (deep networks, general nonlinearities) rest on conjecture and empirical observation rather than proof.

- **Empirical support:** Adequate for a theory paper. The controlled experiments in Section 6 validate specific, non-trivial predictions. However, the gap between synthetic 2D experiments and practical training regimes is significant, and the softmax attention experiment lacks theoretical explanation.

- **Significance:** High for the theoretical deep learning community. Providing a unified mechanism for simplicity bias across architectures, with concrete predictions, advances the field's understanding of implicit regularization and learning dynamics. The practical impact depends on whether the mechanism operates in realistic training regimes.

- **Clarity:** Good. The paper is well-structured and the unified notation (Equation 1) is effective. The distinction between proven results and conjectures is maintained in the body but could be sharper in the abstract and title.