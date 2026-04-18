## Summary

This paper proposes bilinear MLPs—Gated Linear Units without element-wise nonlinearities—as an interpretable alternative to standard MLPs. Because bilinear layers can be expressed as third-order tensors via interaction matrices, their weight structure admits eigendecomposition that reveals low-rank, often interpretable structure without requiring input data. The authors demonstrate this framework across toy tasks, image classification (MNIST, Fashion-MNIST), and language modeling (TinyStories, FineWeb), including applications such as identifying overfitting from weights alone, constructing adversarial masks, and extracting a sentiment negation circuit from a small transformer.

## Strengths

- **Elegant and principled mathematical framework**: The observation that the anti-symmetric part of an interaction matrix vanishes under quadratic evaluation (Section 2), enabling real-valued eigendecompositions of symmetric interaction matrices, is clean and correct. The unified treatment connecting bilinear tensors, interaction matrices, and eigenvector decompositions under three regimes (with input/output features, with only output features, without features) is well-organized and rigorous.

- **Ground-truth validation via mechanistic interpretability challenge**: Section 4.3 demonstrates recovery of a known algorithmic computation (cosine similarity to target ± complement) from weights alone, succeeding where prior methods required dataset access and hints. This is a convincing proof-of-concept for weight-based reverse engineering.

- **Constructive adversarial examples from weights alone**: The pseudoinverse-based adversarial masks (Section 4.4) show that directions extracted from weight decompositions have genuine causal relevance—targeted misclassification without any forward passes or optimization. The distinction between robust and non-robust feature directions in regularized vs. unregularized models is particularly insightful.

- **Overfitting detection from weights alone**: The observation that eigenvectors of unregularized models focus on outlier pixels (Figure 4), and that increasing noise regularization yields more digit-like eigenvectors and lower-rank spectra, provides a practical diagnostic tool that doesn't require input data.

- **Low-rank approximation results on language models**: The finding that many SAE output features' activations are well-approximated by a small number of eigenvectors (Figure 9), with most features achieving correlation >0.75 with just two eigenvectors, is a promising quantitative finding about the structure of bilinear layer computation.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed scope relative to demonstrated scale**: The abstract states bilinear layers serve as an "interpretable drop-in replacement" and that "weight-based interpretability is viable for understanding deep-learning models." The discussion further claims viability "even for large language models." However, all language model experiments use a 6-layer TinyStories model and small FineWeb models (12–16 layers). No experiments on models approaching realistic scale (hundreds of millions or billions of parameters) are presented. The performance comparison with SwiGLU also only appears in Appendix I for small models. Extending these claims to "large language models" is speculative—the scalability of both the performance gap and the eigendecomposition approach (which requires $O(d_{\text{input}}^2)$ interaction matrices) is untested at meaningful scale. The discussion's suggestion that one could "plausibly prove bounds on a layer's outputs" has no concrete demonstration. These overclaims matter because they frame the contribution as solving a problem (interpretability of LLMs) that the evidence does not support.

- **No comparison to interpretability methods for standard MLPs**: The paper's central motivation is that bilinear MLPs enable weight-based mechanistic interpretability that standard MLPs with element-wise nonlinearities cannot easily provide. Yet there is no comparison to what can be extracted from standard MLPs under similarly sophisticated analysis (e.g., second-order Taylor approximation around a data distribution, Jacobian/Hessian-based approaches, or transcoders). Without such baselines, the paper establishes that bilinear MLPs are *amenable* to eigendecomposition, but not that they are *better* for interpretability than alternatives applied to standard architectures. This gap matters because the architectural recommendation to replace standard MLPs requires demonstrating an interpretability advantage, not just tractability.

- **Cherry-picked circuit evidence without causal validation**: The sentiment negation circuit (Section 5.1) is the paper's flagship demonstration of mechanistic circuit discovery in language models, yet it is explicitly "cherry-picked" with no systematic evaluation of how representative it is. The quantitative fit is moderate (correlation ~0.66 overall, 0.76 conditioned on large activations), and no causal intervention (e.g., ablating or steering specific eigenvectors and measuring behavioral change) is performed to confirm the circuit drives the behavior. For a paper claiming to enable "mechanistic interpretability," the absence of causal validation for the single detailed circuit example is a significant evidential gap. No statistics are given on what fraction of features admit interpretable, low-rank decompositions.

### Minor

- **"Weight-based interpretability" framing does not fully account for SAE dependence**: The title promises "weight-based mechanistic interpretability," but for language models (Section 5), the method relies on SAE-derived feature directions obtained from activation data. Only the HOSVD approach (Section 3.3) is fully weight-based, and it is demonstrated only in Appendix D without rigorous evaluation. The limitations section acknowledges this partially, but the framing throughout the paper (including the title) implies a stronger weight-only method than is actually delivered for the most important application domain.

- **Dependence on SAE quality and the "hidden transition"**: The low-rank approximation quality depends on SAE training time, with Appendix H revealing a "hidden transition near convergence" where eigenvector-activation correlation improves dramatically. This raises the question of how much of the observed low-rank structure is genuine versus an artifact of the SAE basis, which is not disentangled in the analysis.

- **Moderate approximation quality for low-rank structure**: The distribution in Figure 9B shows a substantial tail of features with correlation well below 0.5 when using two eigenvectors. While many features are well-approximated, the paper does not characterize which types of features are poorly approximated or whether interpretability degrades systematically for higher-rank spectra, as the limitations section suggests.

### Trivial

- The paper mentions "no guarantees the eigenvectors will be monosemantic" and suggests sparse dictionary learning on the tensor as a solution, but this direction is entirely unexplored.

## Nice-to-Haves

- Systematic quantification of what fraction of eigenvectors/SAE features are interpretable across layers and model sizes, rather than relying on cherry-picked examples.
- Causal intervention experiments (ablation/steering of specific eigenvectors) to validate that identified circuits are mechanistically causal, not just correlational.
- Multi-layer circuit composition analysis, since all experiments operate within a single bilinear layer, and the motivation explicitly calls for understanding end-to-end computation.
- Comparison with transcoders or gradient-based attribution on the same bilinear models to clarify whether the weight-based approach provides genuine advantages beyond architectural convenience.
- Discussion of computational cost for eigendecomposing interaction matrices at scale (the $O(d_{\text{input}}^2)$ matrix size), which would affect practical viability for large models.

## Removed Points

- **Harsh critic's point about "no comparison to quadratic or polynomial networks"**: This asks for comparison with a class of alternative architectures that are not standard baselines in the interpretability literature. The paper's contribution is showing that a particular existing architecture (GLU variant) has useful interpretability properties; demanding comparison with every related architecture is scope creep.

- **Harsh critic's point about the ground-truth task being "the easiest possible setting" and therefore not providing evidence about typical behavior**: All proof-of-concept demonstrations use simple settings; this is standard practice. The task is valid specifically because it has a known ground truth, enabling verification that the method recovers the correct algorithm.

- **Harsh critic's point about "performance comparisons relegated to Appendix I"**: Conference papers routinely place detailed benchmarking in appendices. The main text does state the key finding ("equal loss when keeping training time constant and marginally worse loss when keeping data constant"). While having more details in the main text would strengthen the "drop-in replacement" claim, the information is present and accessible.

- **Harsh critic's point about numerical conditioning of large interaction matrices**: This is a valid engineering concern but is not addressed because the paper does not attempt eigendecomposition at large scale; it would be speculative to discuss numerical issues at scales not tested.

- **Neutral reviewer's suggestion about human evaluation of eigenvector interpretability**: While this would strengthen claims, human evaluation of feature interpretability is not standard in the mechanistic interpretability literature (SAE papers rely on visual inspection too), making this a nice-to-have rather than a weakness.

- **Spark's suggestion to explore hybrid approaches (bilinear SAEs, etc.)**: This is an interesting future direction but falls outside the paper's stated scope.

## Novel Insights

The recognition that the anti-symmetric part of interaction matrices vanishes under quadratic evaluation—and therefore that bilinear layers admit a complete symmetric eigendecomposition with real eigenvalues—is a clean theoretical insight that makes the entire framework tractable. The adversarial mask construction via pseudoinverses of eigenvector matrices is also a novel application that leverages the weight-based decomposition in a genuinely causal way, connecting to the broader literature on non-robust features (Ilyas et al., 2019). The finding that SAE training quality strongly affects eigenvector-activation correlation (the "hidden transition") suggests an important interaction between dictionary learning and weight-based analysis that future work should disentangle.

## Suggestions

- Narrow the headline claims: replace "drop-in replacement" with "competitive alternative" and remove or heavily qualify claims about "large language models" pending experiments at non-trivial scale.
- Add causal intervention experiments for the sentiment negation circuit: ablate the top eigenvectors and measure the change in not-good/not-bad feature activations and downstream token predictions.
- Report systematic statistics on the fraction of SAE features that admit low-rank, interpretable decompositions across all layers, not just the cherry-picked circuit.
- Present the key performance comparison results (Table/Figure from Appendix I) in the main text, since the "competitive performance" claim is central to the architectural argument.

## Score and Decision

**Calibration**: I compared against several papers with similar strengths/weaknesses:
- **CRATE/white-box language models** (6X7HaOEpZS, scores 3-5): Similar "interpretable by design" architecture with overclaimed benefits relative to evidence and limited scale. Weaker technical contribution (interpretability was less rigorous). This paper is stronger.
- **Modular addition without black-boxes** (yBhSORdXqq, scores 3-8, avg ~5): Similar pattern of novel methodology on toy/small models with overclaimed generalization. This paper has broader empirical coverage but similar overclaiming issues.
- **Monet: Mixture of Monosemantic Experts** (1Ogw1SHY3p, scores 6-8): More complete evidence at larger scale, clearer competitive performance. This paper is weaker due to limited scale and missing baselines.
- **Monosemanticity and Robustness** (g6Qc3p7JH5, scores 5-6): Borderline accept with similar issues around limited scope but genuine contribution. Comparable to this paper.

This paper makes a genuine, novel technical contribution (eigendecomposition of bilinear interaction matrices for interpretability) with elegant theory and compelling small-scale demonstrations. However, the claims are overextended relative to the evidence—particularly regarding "large language models" and "drop-in replacement"—and the absence of comparison to interpretability methods on standard architectures, plus the lack of causal validation for the single circuit demonstration, are significant gaps. The contribution is real but narrower than claimed. A score of 5.5 reflects genuine novelty undermined by overclaiming and missing empirical comparisons.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>