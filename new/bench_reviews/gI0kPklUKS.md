Now let me check the paper's appendix references for the performance comparison claims.

## Summary
This paper proposes bilinear MLPs—Gated Linear Unit variants without element-wise nonlinearities—as interpretable replacements for standard MLPs. Because bilinear layers can be expressed as third-order tensors, their weights decompose into interpretable eigenvector-structured interaction matrices that are *exactly equivalent* to the layer's computation. The authors demonstrate this framework across MNIST/Fashion-MNIST (revealing digit-specific edge detectors, detecting overfitting, constructing adversarial masks from weights alone), a ground-truth mechanistic task, and in language models where they extract a sentiment-negation circuit and show low-rank approximation of SAE features.

## Strengths

- **Clean, principled mathematical framework.** The bilinear tensor formulation, symmetry argument, and eigendecomposition approach (Sections 2–3) are elegant and well-motivated. The key insight—that removing the element-wise nonlinearity from GLU layers yields a layer whose entire computation can be decomposed via linear algebra on the weights—is sound and enables analysis fully equivalent to the original computation, unlike gradient-based approximations.

- **Compelling ground-truth evaluation.** Section 4.3 recovers a known labeling function (similarity to a target digit) from weights alone without dataset knowledge. This is a rare instance of verifiable interpretability where the decomposition recovers the exact algorithm, providing a strong proof-of-concept.

- **Stability and truncation results.** Figure 5 shows that top eigenvectors are consistent across training runs (cosine similarity 0.8–0.9) and that truncating to a small number of eigenvectors preserves classification accuracy. These are informative quantitative results about bilinear layers' internal structure.

- **The sentiment negation circuit (Section 5.1) is a strong qualitative demonstration.** The AND-gate interaction between negation and sentiment features, explainable via two dominant eigenvectors, shows how bilinear MLPs enable interpretable nonlinear circuit discovery—something that would require gradient approximation in standard MLPs.

- **Adversarial masks from weights alone (Section 4.4)** demonstrate causal, not just descriptive, utility of the decomposition. This is a practical demonstration that the extracted structure is functionally relevant.

## Weaknesses

### Major:

- **Overclaimed generality of "drop-in replacement" and "viability for large language models."** The paper repeatedly claims bilinear MLPs are "interpretable drop-in replacements" (Abstract) and that "weight-based interpretability is viable for understanding deep-learning models" (Discussion). What is actually demonstrated is: (i) qualitative interpretability in single-layer MNIST/Fashion-MNIST models; (ii) one cherry-picked circuit in a 6-layer TinyStories model; (iii) aggregate low-rank correlation statistics in small transformers. The performance comparison is relegated to Appendix I with no main-text quantitative data, and no experiments at modern LLM scale. The paper's own language-model results depend on SAEs (not purely weight-based) and the largest model tested is ~150M parameters. This gap between framing and evidence is the paper's most significant issue. *The claim of "drop-in replacement" is not substantiated at the scale where it would matter most.*

- **Heavy dependence on SAEs for language-model results without adequate controls.** Section 5's language-model results are entirely mediated by SAE-derived features. The paper does not disambiguate whether the observed low-rank structure reflects genuine bilinear MLP computation versus SAE inductive biases (e.g., ReLU + L1 encouraging features that align with low-rank quadratic forms). No controls are provided—e.g., applying the same SAE pipeline to non-bilinear MLPs, or comparing with alternative unsupervised features (PCA, non-sparse autoencoders). Given that the paper's most ambitious claims rest on these LM results, this is a major evidential gap.

- **Cherry-picked circuit without systematic evaluation.** Section 5.1 explicitly states "We cherry-pick and discuss one such circuit." There is no quantitative audit of how many features yield interpretable low-rank structure versus how many eigenvectors remain polysemantic or uninterpretable. Figure 9B shows a distribution of correlations including a tail of low-correlation features, but the paper does not analyze these failure cases. Without this, it is unclear whether the sentiment-negation circuit is representative or exceptional.

### Minor:

- **Computational scalability of eigendecomposition is unaddressed.** Decomposing the interaction matrix Q (d_input × d_input) requires O(d³) computation. For d_input ≥ 2048, this becomes expensive. The paper does not discuss approximate methods (e.g., Lanczos) or analyze where this becomes intractable. This is relevant to the "drop-in replacement" claim at scale.

- **No direct comparison with interpretability baselines on the same models.** The paper argues for superiority over gradient-based or transcoder-based methods but never benchmarks these alternatives on the same bilinear model. Such a comparison would clarify whether bilinear architecture *plus* its native decomposition truly outperforms standard architecture *plus* standard interpretability methods.

- **The "full equivalence" framing is somewhat misleading relative to practical usage.** The decomposition is mathematically exact, but all interpretable outputs (eigenvectors, circuits, adversarial masks) rely on truncating to a small number of components. The paper provides no quantitative bounds on truncation error in language-model settings, leaving unclear how much of the computation is genuinely captured by the "interpretable" portion.

## Nice-to-Haves

- Systematic quantitative evaluation of eigenvector interpretability across randomly sampled features (not just top eigenvectors of selected classes) with human or automated ratings.
- Head-to-head comparison with transcoders or SAE-based circuit discovery on the same bilinear model.
- Performance comparison tables (SwiGLU vs. bilinear) for the actual models used in interpretability experiments, in the main text.
- Analysis of failure modes: what fraction of features have poor low-rank approximation, and what characterizes them?
- Experiments at 1B+ parameter scale, or explicit discussion of expected computational requirements.

## Removed Points

These points are flagged to be removed; treat them with caution.

- *Harsh Critic point about "lack of rigorous baselines for adversarial masks."* The adversarial mask construction is a demonstration of the framework's utility, not a core claim. The paper frames it as "demonstrating the utility of weight-based decomposition" and constructs masks "from weights alone, without training or any forward passes," which is the novel aspect. Standard gradient-based baselines (PGD, FGSM) operate on a fundamentally different principle (input optimization vs. weight decomposition), making the comparison not apples-to-apples. That said, comparing effectiveness is interesting but not essential.

- *Harsh Critic point about "overfitting detection anecdotal."* The overfitting visualization (Figure 4) is clearly qualitative, and the paper frames it as such ("we can identify overfitting in image models by visualizing the top eigenvectors and searching for spatial artifacts"). This is a reasonable proof-of-concept, not a rigorous detection method. The claim is appropriately qualified.

- *Demand for user studies or human evaluation of interpretability.* Systematic human evaluation is not standard in the mechanistic interpretability community for papers introducing new analysis frameworks, and would be an unreasonable burden for an initial proof-of-concept. The ground-truth task in Section 4.3 serves as an objective evaluation.

- *Harsh Critic point about "overfitting detection lacks comparison with ReLU/SwiGLU."* The paper's contribution is about what bilinear MLPs enable, not about comparing interpretability across architectures. Requiring a cross-architecture comparison is scope creep beyond the paper's stated contribution.

- *Spark's suggestion to "test at larger scale" (1–3B).* This is a reasonable future direction but not a fair requirement for a paper that demonstrably operates within its computational means and clearly acknowledges the scale limitation in its Discussion.

## Novel Insights

The observation that removing element-wise nonlinearities from GLU layers yields an architecture whose computation can be *exactly* decomposed into interpretable components via standard linear algebra—without sacrificing much performance—is a genuinely useful insight for mechanistic interpretability. The sentiment-negation circuit, where two eigenvectors capture an AND-gate-like interaction between negation tokens and sentiment features, illustrates how quadratic interactions can be made transparent in a way that is structurally impossible with elementwise nonlinearities. The paper's most novel contribution is demonstrating that this mathematical fact translates to qualitatively interpretable decompositions in practice, even if the scale and generality of this translation remain open questions.

## Suggestions

- **Temper the framing**: Replace "drop-in replacement" and "viable for understanding deep-learning models" with more measured claims like "promising direction for interpretable architectures" and "demonstrated viability in small-scale language models."
- **Add SAE controls**: Apply the same SAE + eigendecomposition pipeline to a standard SwiGLU transformer on TinyStories to establish whether the low-rank structure is specific to bilinear architectures or an artifact of SAE inductive biases.
- **Systematic circuit audit**: Report what fraction of output features have correlation >0.5 with their 2-eigenvector approximation, and show representative failure cases (low-correlation features). This would clarify the method's scope without requiring massive additional experiments.
- **Move performance comparison to main text**: Even 2–3 rows of a table comparing bilinear vs. SwiGLU on the TinyStories/FineWeb models used for interpretability would substantiate the "near-SwiGLU performance" claim.

## Evaluation on Key Axes

- **Originality**: High. The bilinear MLP interpretability framework is novel and well-motivated; the eigendecomposition approach for interaction matrices is natural but had not been systematically developed.
- **Importance of research question**: High. Making MLP computation interpretable is a central challenge in mechanistic interpretability.
- **Claims support**: Moderate. The mathematical framework is solid and toy-scale demonstrations are convincing, but the strongest claims (drop-in replacement, viability for large LMs) outpace the evidence.
- **Experimental soundness**: Moderate. Well-designed for MNIST and the ground-truth task, but the LM experiments rely on SAEs without controls, and the circuit analysis is cherry-picked.
- **Clarity**: Good. The paper is well-written; the mathematical exposition in Sections 2–3 is clear; figures are informative.
- **Value to community**: Moderate-to-high. If the approach scales, it could provide a principled alternative to gradient-based circuit discovery. The current evidence is promising but preliminary.

## Score Calibration

Comparing against calibration papers:
- **CRATE white-box language models** (scores: 5, 5, 3): Similar pattern of architectural interpretability claims with limited scale and performance trade-offs. This paper has stronger mathematical grounding and more diverse experiments, but similarly overclaimed.
- **Spectral Dynamics of Weights** (scores: 8, 6, 6, 5): Also uses spectral methods for understanding neural networks. This paper has a more focused and novel contribution (bilinear architecture + eigendecomposition) but less systematic evaluation.
- **Sparse Feature Circuits** (scores: 8, 8, 8, 8): Much more systematic evaluation, larger scale, direct comparisons. This paper is less mature in its empirical evaluation.
- **Not All Features Are Linear** (scores: 8, 6, 8, 6): Accepted poster. Novel insight with strong evidence but some limitations. This paper has comparable novelty but weaker empirical validation at scale.

The paper sits in the 5–6 range: a novel and principled framework with convincing toy-scale demonstrations, but with important evidential gaps for its strongest claims (especially regarding language models and the "drop-in replacement" framing). The core contribution is real and valuable, but the overclaiming is significant enough to warrant a score below acceptance threshold for a top venue, though not so low as to dismiss the work entirely.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>