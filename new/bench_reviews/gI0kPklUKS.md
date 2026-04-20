After thoroughly reading the paper and verifying the critics' claims against the text, here is my consolidated review.

## Summary

This paper proposes a novel framework for weight-based mechanistic interpretability of bilinear MLPs (GLU variants without element-wise nonlinearities). By reformulating bilinear layers via symmetric interaction tensors and applying eigendecomposition, the method extracts interpretable input directions that explain layer outputs directly from weights. The paper demonstrates this across MNIST/FMNIST (with ground-truth algorithm recovery and adversarial mask construction) and language models (sentiment negation circuits and low-rank SAE feature approximation), positioning bilinear MLPs as an interpretable drop-in replacement for standard architectures.

## Strengths

- **Ground-truth algorithm recovery from weights alone (Section 4.3, Figure 6):** The eigendecomposition of a model trained on a known similarity-based classification task recovers the exact target/complement similarity algorithm, extracting the ground-truth computation from weights without dataset access or manual hints. This is the paper's strongest piece of evidence and directly validates the method.
- **Adversarial masks constructed purely from weights (Section 4.4, Figure 7):** The pseudoinverse-based mask construction causes significant accuracy drops and targeted misclassification compared to random baselines, providing causal evidence that the extracted eigenvectors capture functionally important computational structure without any forward passes or optimization.
- **Low-rank structure in language models is compelling (Section 5.2, Figure 9):** Output SAE features in bilinear transformers are well-approximated by very few eigenvectors—average correlation starts at ~0.65 for one eigenvector and most features exceed 0.75 with just two. This demonstrates the method's promise at LM scale.
- **Consistency and regularity (Section 4.2, Figures 4–5):** Top eigenvectors show cosine similarities of 0.8–0.9 across independent runs, improving with model size. The method also detects overfitting via scattered eigenvector artifacts (Figure 4), demonstrating practical diagnostic utility.
- **Clean mathematical formulation (Section 3):** The derivation of the bilinear tensor and its symmetric interaction matrix is mathematically precise, and the framework offers multiple analysis paths (direct interactions, eigendecomposition, HOSVD) depending on available information.

## Weaknesses

### Fatal
None

### Major

- **Circuit discovery claims rely on correlation, not causal intervention.** Section 5.1 identifies a "sentiment negation circuit" as an AND-gate between negation and sentiment features, but the evidence is limited to an interaction heatmap and correlation (0.66 overall, 0.76 conditioned on activation) between the true feature and a two-eigenvector approximation. The paper does not perform causal ablation—zeroing or scaling the top eigenvectors—to verify necessity and sufficiency. High conditional correlation measures approximation quality over the active tail, not mechanistic causality. The paper's use of terms like "circuit" and "mechanistic" in the abstract implies causal structure that is not demonstrated. This gap between correlational evidence and mechanistic claims weakens the paper's contribution to the mechanistic interpretability literature, where causal scrubbing and ablation are the standard.

- **Performance baselines for the "competitive drop-in replacement" claim are largely external.** The paper's central positioning—that bilinear MLPs are performance-competitive with SwiGLU/ReLU—relies on Shazeer (2020) as the primary evidence, with Appendix I providing corroboration. The main text contains no validation perplexity, loss curves, or standard NLP metrics on non-trivial benchmarks (e.g., TinyStories, FineWeb) against matched ReLU/SwiGLU baselines under identical parameter/FLOP budgets. While the Shazeer reference is established, the paper's novel interpretability framework would benefit from demonstrating performance parity under the same experimental conditions as the interpretability claims.

### Minor

- **Interpretability of eigenvectors beyond the top few is subjective.** Section 4.1 relies on visual inspection of top eigenvectors (Figure 2B, Figure 3). The paper acknowledges that mid-spectrum eigenvectors can be scattered or uninterpretable (Section 6: "for high-rank spectra, the orthogonality between eigenvectors may limit their interpretability"). However, no systematic quantitative metric (e.g., automated feature clustering, human coherence ratings, or alignment with known feature bases) is provided to evaluate interpretability across the full spectrum. This makes the interpretability claim partially qualitative and dependent on cherry-picking the most visually coherent components.

### Trivial
None

## Nice-to-Haves

- Causal ablation experiments intervening on top eigenvectors (zeroing or scaling $\lambda_i v_i$ terms) and measuring downstream feature activation changes would strengthen the mechanistic claims substantially.
- A computational and scaling analysis of eigendecomposing $(d_{\text{output}}, d_{\text{input}}^2)$ interaction matrices for modern LM hidden dimensions would help assess practical feasibility at larger scales.
- Including validation perplexity curves against matched SwiGLU/ReLU baselines on language model training would make the "drop-in replacement" claim self-contained rather than citation-dependent.

## Removed Points

- **Orthogonality vs. polysemanticity invalidating the core premise.** The harsh critic claims the orthogonality constraint "fundamentally misaligns with neural feature structure" and "invalidates the core premise." However, the paper explicitly acknowledges this in Section 6 ("We expect that for high-rank spectra, the orthogonality between eigenvectors may limit their interpretability") and empirically demonstrates that many features *are* low-rank and interpretable. The paper does not claim monosemanticity; it claims low-rank structure is common. This is a recognized limitation, not an invalidation. The ground-truth recovery (Section 4.3) and adversarial mask experiments work regardless.
- **"SDL oversimplification" in the introduction's framing.** The critic argues the statement that SDL "only describes which features are present, not how they are formed" misrepresents activation-based interpretability. This is a debatable framing choice, not an error—the paper's point is that activation-based methods don't give weight-level structure, which is genuinely true by definition.
- **Conditional correlation "cannot support any mechanistic claims."** The critic overstates this. While the paper's LM circuit analysis is correlational (not causal), the correlation values (0.66–0.76) honestly represent approximation quality, and the paper does appropriately show the low-rane structure captures active feature behavior. The weakness is the *degree* of mechanistic claim, not that correlation is useless.
- **PCA equivalence of eigendecomposition is not addressed.** This isn't a weakness—it's the entire design of the method. Section 3.2 explicitly defines eigendecomposition of the interaction matrix Q; the fact that eigendecomposition of a symmetric matrix is equivalent to PCA is mathematically true and intended.
- **Eigenvector sensitivity to input distribution/normalization.** The critic raises that eigenvectors of Q are sensitive to input covariance. This is addressed in the paper: Section 4.1 shows that Gaussian noise regularization improves eigenvector quality (Figure 4), and the method works robustly across different settings. This is an implementation consideration that does not undermine the method.

## Novel Insights

The paper's most genuinely novel contribution is the demonstration that bilinear MLPs—a practically competitive GLU variant—admit a fully weight-based interpretability pipeline. The ground-truth experiment (Section 4.3) is particularly notable: it recovers a known classification algorithm from weights alone, without any dataset access, outperforming prior manual reverse-engineering that required knowledge of the task and dataset. This establishes that weight-based methods can, in principle, match or exceed activation-based approaches for certain tasks. The pseudoinverse-based adversarial mask construction is also a clever closed-form technique that demonstrates the causal relevance of eigenvectors without optimization or forward passes. These results suggest a promising direction: architecture choice (removing element-wise nonlinearities) can unlock interpretability without sacrificing performance.

## Suggestions

1. Add a section or experiment that performs causal ablation on the sentiment negation circuit (Section 5.1)—for instance, zeroing the top two eigenvectors and measuring the change in the not-good feature activation. This would move the circuit claim from correlational to causal.
2. Include a small table in the main text with perplexity or validation loss comparisons between bilinear and ReLU/SwiGLU MLPs on a standard LM benchmark (even TinyStories) under matched budgets to make the performance claim self-contained.
3. Provide a quantitative measure of eigenvector interpretability (e.g., similarity scores to known visual templates for MNIST, or automated coherence metrics) to complement the qualitative visualizations.

## Score and Decision

**Calibration anchors consulted:**
- **High-scoring:** I4e82CIDxv.md (Sparse Feature Circuits, avg 8) — gold-standard mechanistic interpretability with causal verification; this paper is below this bar since its LM evidence is correlational. jKTUlxo5zy.md (Submodular attribution, avg ~7.5) — strong experiments with extensive validation; this paper has slightly less comprehensive baselines.
- **Medium-scoring:** rWQDzq3O5c.md (Graph Transformers, avg ~5.75) — solid theoretical and empirical work; this paper is more novel in its practical applications. L7jtdGhWzT.md (FEI attribution, avg ~4.7) — similar faithfulness concerns but weaker empirical grounding; this paper is stronger with its ground-truth recovery.
- **Low-scoring:** 04RLVxDvig.md (NanoMoE, avg 3) — weak baselines and toy-only experiments; this paper is substantially stronger with genuine results on multiple tasks.

This paper sits between the medium and high-scoring anchors. Its ground-truth recovery and adversarial mask experiments are genuinely impressive (matching or exceeding some high-scoring papers in empirical elegance), but its LM circuit claims lack causal verification and its performance baselines rely heavily on an external citation. Compared to the 6-range anchors, it's competitive or slightly above due to novelty; compared to the 8-range anchors, it's below due to weaker causal evidence and baselines. I place it at **6.0** — a solid paper with meaningful contributions and addressable gaps.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>