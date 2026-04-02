# ICLR Benchmark Results

Date: 2026-04-01 21:00
Critic/Merger: minimax/minimax-m2.7 (OpenRouter)
Neutral: minimax/minimax-m2.7, Related Work: minimax/minimax-m2.7:online (OpenRouter)

## cJd1BgZ9CS

- GT: Accept (Poster) (avg 5.0)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
The paper introduces Distributed Speculative Inference (DSI), which leverages "speculation parallelism" to parallelize the verification process in speculative inference (SI) by overlapping verification with drafting using multiple target and drafter model instances. The authors prove that DSI is lossless, never slower than non-SI, and strictly faster than both SI and non-SI in expectation—even with drafters for which standard SI fails. Simulations on simulated multi-GPU setups demonstrate 1.29-1.92x speedups over SI across various model pairs.

## Strengths
- **Novel parallelism concept**: Introducing speculation parallelism (SP) as a new type of task parallelism that orchestrates target and drafter instances to overlap in time is a genuine conceptual contribution that addresses a fundamental limitation of sequential SI.
- **Theoretically sound**: The paper provides rigorous proofs (Theorems 1-2, Proposition 1) establishing that DSI is lossless, at least as fast as non-SI, and faster than SI in expectation under reasonable assumptions. The analysis correctly identifies when SI fails (slow/inaccurate drafters) and proves DSI remedies this.
- **Addresses a real gap**: The paper correctly identifies that SI can be slower than non-SI when drafters are insufficient, and provides a principled solution that expands the applicability of speculative inference to a wider range of LMs.
- **Reproducibility**: Code is open-sourced with extensive testing, and latency estimates use real model measurements from actual GPU experiments (TTFT, TPOT, acceptance rates).

## Weaknesses
- **Simulated multi-GPU execution**: The main empirical validation replaced forward passes with wait commands on a single GPU rather than running on actual 8-GPU hardware. This leaves open questions about whether theoretical speedups materialize with real concurrent execution, thread management overhead, communication, and resource contention. The paper acknowledges this limitation but it substantially weakens the empirical contribution.
- **Modest empirical speedups**: The reported 1.29-1.92x speedups are considerably lower than the theoretical maximums (4x-50x) discussed in the paper, suggesting significant overhead or that the theoretical assumptions don't fully translate to practice.
- **Incomplete comparison in main results**: Table 2 only compares DSI to SI, not to non-SI, despite the paper's central claim that DSI is faster than both. While Section 4.1's offline simulation addresses this, a direct comparison in the main table would strengthen the paper.

## Nice-to-Haves
- **Lookahead ablation**: The paper tests only lookahead ∈ {1, 5, 10} without systematic validation of lookahead selection strategies in real wall-clock time.
- **SP degree scaling experiments**: The paper claims DSI scales with available hardware but doesn't show speedup curves as SP degree varies from 1 to 8 on real hardware.
- **Case study on "failing" drafters**: A concrete example showing DSI recovering speedup where SI fails would validate the paper's core motivational claim with real measurements.
- **Production implementation guidance**: More detail on integrating DSI with existing serving systems (vLLM, TGI) and efficient KV cache management for token trees would aid practical adoption.

## Novel Insights
The paper's core insight is that SI's draft-then-verify cycle is fundamentally sequential and blocking—the verification of one iteration must complete before drafting begins for the next. By recognizing that verification is not inherently sequential, DSI introduces speculation parallelism where verification tasks run concurrently with drafting on additional servers. This transforms SI into a non-blocking algorithm where verification contributes to latency only upon rejection. The key theoretical contribution is proving this parallelism preserves correctness while achieving strict speedup over both SI and non-SI in expectation, even when drafters are too slow or inaccurate for standard SI to be beneficial. This insight unlocks acceleration for LMs where SI has previously failed, representing a meaningful advance in inference optimization.

## Potentially Missed Related Work
- **Medusa/Cascade decoding (Cai et al., 2024)**: Multiple independently-trained draft heads that could serve as drafters for DSI, potentially increasing acceptance rates and speedups.
- **EAGLE (Li et al., 2024)**: An alternative speculative decoding approach that uses feature uncertainty rather than token matching for acceptance.
- **Context parallelism for LLM inference (Yang et al., 2024)**: Relevant for combining with DSI in multi-node setups with long context lengths.

## Suggestions
- **Validate on real multi-GPU hardware**: The most impactful next step is implementing DSI using proper distributed execution frameworks (e.g., Ray, MPI) to run actual model instances across multiple GPUs and measure real wall-clock speedups, communication overhead, and resource contention effects.
- **Add non-SI comparison to Table 2**: Include measured DSI vs non-SI speedups in the main results table alongside DSI vs SI for direct comparison.

---

## 7Cx05z4pUc

- GT: Reject (avg 5.0)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces "Decomposed Learning," a novel SVD-based training method that parameterizes weight matrices as UₖΣₖVₖᵀ products to mitigate or eliminate grokking in neural networks. Through extensive experiments on modular arithmetic division (mod 97), the authors demonstrate that their method can significantly reduce or eliminate the delayed generalization characteristic of grokking, establish an important data-rank tradeoff relationship, and enable parameter-efficient training with practical compression benefits.

## Strengths
- **Comprehensive empirical evaluation**: The paper systematically explores decomposed learning across all major transformer layers (token embedding, position embedding, multi-head attention, feed-forward blocks, output layer), multiple ranks (12.5% to 100%), and multiple data fractions (50%, 65%, 80%), providing thorough coverage of when and how the method helps.
- **Valuable mechanistic insights**: The spectral analysis connecting decomposed learning to accelerated transition from high to low stable rank, and the finding that "more data requires fewer ranks" to mitigate grokking, provide genuinely useful understanding that connects to lazy-to-rich learning theory (Kumar et al., 2024).
- **Practical compression benefits**: The demonstration that models can be trained in decomposed form with fewer parameters and then recomposed for inference at original size with maintained performance is a compelling practical contribution.
- **Strong reproducibility**: Experimental setup closely follows Power et al. (2022) with clearly specified hyperparameters, architecture details, and training procedures.
- **Extension beyond synthetic tasks**: Appendix experiments on MNIST, TinyShakespeare, and CIFAR-10 demonstrate the method's applicability beyond the primary modular arithmetic task.

## Weaknesses
- **No comparison to established grokking mitigation methods**: The paper compares only against a naive baseline without Power et al. (2022) or Liu et al. (2023) with their optimized hyperparameters. Given that prior work showed grokking can be avoided through hyperparameter tuning, direct comparison is essential to demonstrate the method's relative value.
- **Mechanism incompletely characterized**: While Appendix C shows Uₖ and Vₖ become non-orthogonal and Σₖ becomes non-diagonal during training, the paper does not adequately explain what structural property of the method actually drives the benefit. The implicit regularization framing is interpretive rather than rigorously established.
- **Incomplete ablation**: The paper does not isolate whether benefits come from the SVD initialization, the factorization structure, or the effective rank constraint. Comparison with alternative initializations (e.g., random orthogonal) or standard low-rank training methods (e.g., LoRA) would strengthen the contribution.
- **Limited primary validation scope**: The main claims rest heavily on modular division mod 97. While Appendix F and G extend to other datasets, these results show more modest benefits and lack the same experimental rigor.
- **Training-time computational overhead not addressed**: SVD decomposition adds upfront cost, and the paper does not discuss whether total compute is reduced or just steps-to-generalization.

## Nice-to-Haves
- Formal analysis of why decomposed learning accelerates the stable rank transition observed in Figure 20
- Comparison with LoRA (Hu et al., 2022) as the standard low-rank adaptation baseline
- Additional runs with error bars systematically reported in tables, not just visible in figures
- Investigation into why position embedding decomposition increases steps to grok in some settings

## Novel Insights
The paper's most valuable contribution is the empirical discovery that there exists a fundamental tradeoff between training data representativeness and the minimum rank needed to avoid grokking: with less representative data, higher ranks are required, but as data becomes more representative, lower ranks suffice. This insight has direct implications for parameter-efficient fine-tuning methods like LoRA, suggesting that rank selection should be adapted based on how well the fine-tuning data covers the target distribution. The spectral analysis further provides a plausible mechanistic explanation by showing that decomposed learning speeds up the transition from lazy (high stable rank) to rich (low stable rank) learning dynamics.

## Potentially Missed Related Work
- **Yunis et al. (2024)**: "Approaching deep learning through the spectral dynamics of weights" — directly relevant to the paper's spectral analysis discussion and stable rank findings in Appendix D
- **GaLore (Zhao et al., 2024)**: Gradient low-rank projection for memory-efficient LLM training — similar SVD-based approach applied during training that warrants comparison
- **Q-GaLore (Zhang et al., 2024)**: Quantized GaLore with INT4 projection — related gradient decomposition method

## Suggestions
1. Add direct comparison with Liu et al. (2023)'s optimized hyperparameters to demonstrate that decomposed learning provides benefits beyond what hyperparameter tuning alone can achieve.
2. Include an ablation comparing SVD initialization against random orthogonal initialization in factorized form to isolate whether the benefit comes from the initialization or the structure.
3. Report training time and total compute overhead alongside steps-to-generalization to provide a complete picture of practical utility.

---

## N0i0d27RTW

- GT: Reject (avg 4.5)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper develops statistical risk bounds for approximate stationary points of shallow ℓ₁-regularized neural networks, demonstrating that stationary points achieve essentially the same generalization rates as global optima (up to logarithmic factors). The core contribution addresses a fundamental gap between theory (which typically analyzes global optima) and practice (which yields only approximate stationary points from gradient-based optimization).

## Strengths
- **Addresses a fundamental gap**: The paper directly tackles why neural networks trained with gradient descent generalize despite finding only stationary points, not global optima—a core question in deep learning theory.
- **Technically rigorous**: The proofs employ Taylor expansions around stationary points combined with empirical process theory, introducing novel analysis techniques for this setting (scaling tricks, rescaled network arguments).
- **Comprehensive coverage**: Four main theorems cover stationary points (Theorem 1), approximate stationary points (Theorem 2), ReLU networks (Theorem 3), and heavy-tailed noise (Theorem 4), providing a cohesive theoretical framework.
- **Novel statistical insight**: The result that all "reasonable" stationary points achieve near-optimal rates is genuinely novel and provides mathematical justification for the empirical observation that local optima suffice in practice.
- **Numerical validation**: Experiments on synthetic data confirm that approximate stationary points achieve near-identical test performance to potential global optima.

## Weaknesses
- **Near-identity assumption limits ReLU contribution**: Theorem 3 requires Θ ≈ I_w for both the stationary point and target network. This assumption essentially restricts the result to networks operating in their linear regime. While the paper cites prior work supporting this (Hardt & Ma, 2016; Li & Yuan, 2017), the practical applicability to general ReLU networks is unclear.
- **Shallow networks only**: No theoretical framework extends to multi-layer networks despite the paper's claim of making "considerable progress" toward deep learning. The deep network experiments (Table 4, Table 5) are labeled as "conjecture" and appear only in the appendix.
- **Highly restrictive "reasonability" bound**: The condition ||γ||₁ ≤ √log n ≈ 3 for n=1000 is extremely tight. While the paper argues this is natural given typical initialization (near-zero weights) and large tuning parameters, practical deep networks may not satisfy this bound.
- **Limited empirical evaluation**: Experiments use only synthetic data with small networks (d=w=10 or d=100, w=20). No experiments on real datasets or systematic scaling analysis are provided.
- **No lower bounds**: The paper claims optimal rates but provides no matching lower bounds to support this claim.

## Nice-to-Haves
- Ablation study showing how the gap between stationary points and global optima evolves with regularization parameter r would strengthen the practical claims.
- Systematic experiments varying network width w would demonstrate scaling behavior and validate theory beyond the small-network regime.
- Explicit characterization of "bad" stationary points (if they exist under the assumptions) would clarify the scope of the positive result.
- Discussion of what happens when the target is approximately but not exactly representable as a neural network.

## Novel Insights
The paper's central insight is that for ℓ₁-regularized shallow neural networks, **every** reasonable stationary point achieves generalization bounds matching those of global optimizers up to logarithmic factors. This provides the first rigorous statistical theory explaining why gradient-based optimization "suffices" in practice—the local geometry near any stationary point is well-curved enough that the statistical risk remains controlled. The technical machinery of scaling tricks and rescaled networks (exploiting the neural network's reparametrization invariance) combined with empirical process bounds offers a new analytical toolkit for neural network theory.

## Potentially Missed Related Work
- **Hardt & Ma (2016) / Bartlett et al. (2018a,b)**: Already cited, but their results on near-identity representations could be more explicitly connected to explain the practical relevance of the Θ ≈ I_w assumption.
- **Li & Yuan (2017)**: Already cited regarding convergence of SGD with near-identity initialization for ReLU networks; their observation that "networks with small average spectral norm already have good performance" directly supports the paper's framework.
- **Drori & Shamir (2020)**: "The complexity of finding stationary points with stochastic gradient descent" provides complementary complexity-theoretic perspective on the tractability of finding stationary points, which relates to the paper's τ-approximation analysis.
- **Zhou & Liang (2018)**: Already cited for landscape properties of linear networks; their characterization of saddle points complements this paper's statistical analysis.

## Suggestions
The paper would benefit from significantly strengthening the ReLU analysis: either (1) removing or substantially motivating the Θ ≈ I_w assumption with empirical evidence that practical stationary points satisfy this, or (2) explicitly characterizing the regime where this assumption holds and quantifying the gap to general networks. Additionally, presenting the deep network experiments as preliminary evidence (rather than conjecture) with appropriate uncertainty framing would set clearer expectations about the paper's scope.

---

## WVLBWiKxjM

- GT: Reject (avg 3.0)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary

This paper proposes a keypoint-based detection approach for micro-scale crack identification from numerical wave propagation data (synthetic dLEM simulations), where traditional visual inspection is impossible. Instead of pixel-wise segmentation, the model predicts four corner coordinates defining a crack bounding region, reframing the problem as coordinate regression to mitigate class imbalance inherent in crack detection. The model achieves an average IoU of 0.511 across all cracks (0.631 for larger cracks) while requiring significantly fewer computational resources than a prior 1D-DenseNet200E baseline.

## Strengths

- **Novel problem framing**: The application of keypoint detection concepts to numerical, non-visual wave propagation data represents a genuinely innovative contribution—this is the first work to apply object detection approaches to crack detection from synthetic wave propagation data, opening new research directions for structural health monitoring.
- **Addresses class imbalance through problem reformulation**: By reframing crack detection as 4-point coordinate regression rather than pixel-wise classification, the approach conceptually sidesteps the dominant-class bias problem where cracks occupy small regions relative to background.
- **Computational efficiency gains**: The model achieves comparable detection capability with significantly fewer layers (90 vs. 444) and faster training time (17.03s vs. 89.14s per epoch), representing meaningful efficiency improvements over prior work.
- **Thoughtful architectural design**: The Inception-style multi-scale architecture (1×1, 3×3, 5×5 branches) with self-attention is well-motivated for capturing both fine-grained wave patterns and broader spatial relationships from the 9×9 sensor grid.
- **Comprehensive loss function analysis**: The paper provides clear explanations of MSE, MAE, and Huber loss trade-offs for this regression task.

## Weaknesses

- **Fundamentally invalid baseline comparison**: Table 2 compares MicroCrackPointNet's bounding-box IoU with 1D-DenseNet200E's pixel-level segmentation IoU. These metrics are incommensurable—the DenseNet's metric includes true negatives through binarized confusion matrix calculation, while the proposed method measures bounding box overlap. The paper acknowledges this but presents the comparison anyway, and notably the DenseNet achieves *higher* IoU by this comparison, suggesting the proposed method is inferior on this metric, not superior.

- **Missing critical experimental details for reproducibility**: The dataset size (number of samples) is never disclosed. No information is provided on train/validation/test splits, random seeds, or hyperparameter search details beyond vague references to "extensive evaluations." This is essential for reproducibility.

- **No ablation studies**: The paper combines multiple architectural innovations (Inception blocks, self-attention, keypoint formulation) without isolating the contribution of each component. It is impossible to determine whether the attention mechanism, multi-scale branches, or the keypoint formulation itself drives the observed performance.

- **Modest absolute performance**: An average IoU of 0.511 is relatively weak for detection tasks. Without clear justification for why this level of performance is meaningful for the application domain, and without comparison to properly evaluated baselines, the results appear preliminary.

- **Inconsistent terminology**: The model is called "New Model," "Proposed Model," "MicroCrackPointNet," and "the model" interchangeably, creating confusion. More critically, the conclusion mentions a "Squeeze and Excite mechanism" that never appears in the architecture description (Section 3.1), indicating an inconsistency between the method and conclusion sections.

- **Single-crack limitation**: The model can only detect one crack per sample with a fixed four-keypoint structure. This severely limits real-world applicability, as structures typically contain multiple cracks with complex geometries.

- **Synthetic data only**: All experiments use dLEM synthetic data without validation that findings transfer to real ultrasonic, acoustic emission, or Lamb wave measurements from physical experiments.

## Nice-to-Haves

- **Statistical rigor**: Report confidence intervals, error bars, or statistical significance tests across multiple runs rather than single-point estimates.
- **Failure mode analysis**: Systematically characterize when the model fails (e.g., by crack size, position, or geometry) to help practitioners understand when to trust the method.
- **Feature visualization**: Show attention weights or intermediate activations to validate that the model focuses on crack-relevant wave patterns rather than spurious correlations.
- **Hyperparameter sensitivity**: Document the max-pooling factor selection process and sensitivity to other hyperparameters rather than claiming "extensive evaluations" without details.

## Novel Insights

The paper's core insight—reframing crack detection from pixel-wise segmentation to keypoint coordinate regression for numerical wave data—is genuinely novel and addresses a real gap in structural health monitoring. The observation that object detection concepts can be applied to non-visual numerical data (wave propagation measurements) where human interpretation is impossible is valuable, as it opens computer vision techniques to new domains beyond image analysis. The conceptual connection between class imbalance mitigation and problem reformulation (from per-pixel classification to coordinate regression) is sound and worth exploring further.

## Potentially Missed Related Work

- **Lamb wave-based damage detection methods**: While Azuara et al. (2020) on RAPID algorithm is cited, more comprehensive coverage of Lamb wave tomography and migration-based imaging methods for crack localization in composite materials could strengthen the related work.
- **Point detection approaches in computer vision**: Heatmap-based keypoint detection methods (e.g., Hourglass networks, stacked hourglass for human pose estimation) represent a well-established paradigm that could provide both baselines and architectural insights.
- **None identified**: The search for potentially missed related work was skipped in the provided analysis, so no specific omissions can be confirmed.

## Suggestions

1. **Fix the comparison methodology**: Either recalculate DenseNet's IoU using the same bounding-box approach, or remove Table 2 entirely. A valid comparison could be made against other keypoint-based detection baselines on the same data.

2. **Add comprehensive ablation studies**: Systematically remove (a) the attention mechanism, (b) Inception-style multi-scale branches, (c) the keypoint formulation itself, and compare against a simple regression baseline to isolate what components are necessary.

3. **Report dataset statistics**: Document total samples, split ratios, and ideally release code/data for reproducibility. Without this, the scientific contribution is incomplete.

4. **Resolve terminology inconsistency**: Remove the "Squeeze and Excite" reference from the conclusion if it is not part of the architecture, or add it to the method section with appropriate justification.

5. **Extend to multi-crack detection**: Given the paper explicitly identifies this as future work, consider either limiting the paper's scope explicitly to single-crack scenarios with clear justification, or implement a proof-of-concept multi-crack extension to demonstrate scalability.

---

## hKMPz3wkPV

- GT: Reject (avg 6.8)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes "representational compositionality" — a formal, quantitative definition of compositionality grounded in algorithmic information theory — claiming that compositional representations are those expressible as a simple function of discrete constituent parts. The authors validate their definition on synthetic data with known ground truth and demonstrate its advantages over existing heuristics like topological similarity.

## Strengths
- **Addresses a genuine theoretical gap**: The paper tackles the long-standing problem that compositionality, despite being central to cognitive science and AI, has lacked a formal, measurable definition. This is a valuable contribution to both fields.
- **Strong theoretical grounding**: The use of Kolmogorov complexity provides a principled foundation that naturally captures intuitions about simplicity, structure, and expressivity. The decomposition K(Z) ≈ K(pw) + K(W|pw) + K(f) + K(Z|W,f) is well-motivated.
- **Excellent synthetic validation**: The lookup table and context-free grammar experiments (Section 4.1) rigorously validate that C(Z) aligns with intuitions across multiple axes — sentence length, vocabulary size, dimensionality, disentanglement, grammar width/depth — using ground-truth complexity terms.
- **Clear superiority over topological similarity**: The paper convincingly demonstrates that the proposed definition produces more intuitive orderings than existing heuristics, particularly in the grammar experiments where topological similarity shows "far more noise."
- **High-quality intuition-building**: The examples in Appendix D (lookup tables, modular functions, equivariances) are particularly helpful for building understanding, and the connections to cognitive science concepts (systematicity, modularity) are well-executed.

## Weaknesses
- **The primary definition C(Z) is never validated on real neural representations**: All experiments use either synthetic data with known structure (where K(·) terms are computed from the generative program) or language system compositionality C_L(Z) with externally-fixed W. The paper's central claim — that representational compositionality can measure compositionality in DNNs or brain data — is never demonstrated. This is a significant gap between theory and empirical validation.
- **Estimation pipeline is unimplemented and unvalidated**: The abstract claims C(Z) "can be readily estimated using standard deep learning tools," but Appendix B only sketches an approach using discrete autoencoders with autoregressive priors. The paper acknowledges severe challenges (VQ-VAE training instability, simplicial embedding temperature scheduling, GFlowNet exploration problems) but provides no working implementation or empirical demonstration. The claim of "readily estimated" is aspirational.
- **Paper conflates Definition 2 (C(Z)) and Definition 3 (C_L(Z))**: Section 4.2 and 4.3 experiments measure language system compositionality where W is externally defined, not the full representational compositionality with intrinsic parts. The paper presents these as validating the main definition, but they validate a substantially easier quantity. The main claim about measuring representations from pretrained models remains untested.
- **No evidence for the generalization hypothesis**: The paper repeatedly claims (Appendix E) that high C(Z) should enable compositional generalization, but provides zero empirical evidence. This is the primary motivation for AI applications and is entirely speculative.
- **The connection between C(Z) and existing intuitions is asserted, not derived**: The paper claims C(Z) generalizes systematicity, structure-preservation, and modularity, but provides only informal analogies. Formal derivations or constructive counterexamples would strengthen these claims.

## Nice-to-Haves
- Implement and release code demonstrating the discrete autoencoder estimation pipeline on at least one real representation space (e.g., CLIP embeddings, transformer hidden states)
- Test whether C(Z) correlates with out-of-distribution generalization on benchmarks like SCAN or COGS
- Compare C(Z) head-to-head with existing compositionality metrics (Andreas 2019, Lewis 2022, etc.) on identical data
- Provide theoretical bounds on estimation error when the actual shortest program deviates from the assumed form
- Address natural language experiment limitations more thoroughly (translation quality variation, embedding model biases across languages)

## Novel Insights
The paper's core insight — that compositional representations are those compressible as simple functions of discrete parts, captured by the ratio K(Z)/K(Z|W) — is genuinely original and theoretically well-grounded. The decomposition of Kolmogorov complexity into language complexity, sentence complexity, semantics complexity, and reconstruction error provides a unified lens that connects disparate concepts (disentanglement, modularity, equivariances, systematicity) under a single framework. The observation that C(Z) naturally handles graded compositionality (rather than binary) and excludes arbitrary mappings (like idioms) through the complexity of the semantics function K(f) is particularly insightful. The distinction between intrinsic parts (derived from the representation itself) versus externally-defined parts is also valuable, as it avoids the arbitrariness of prior approaches.

## Potentially Missed Related Work
- Andreas, J. (2019). Measuring compositionality in representation learning — directly compares and contrasts with this paper's approach
- Lewis, M. et al. (2022). Does CLIP bind concepts? Probing compositionality in large image models — provides a benchmark for comparing C(Z) against existing methods
- Lippl & Stachenfeld (2024). When does compositional structure yield compositional generalization? — offers a different theoretical perspective on the compositionality-generalization connection that could be contrasted with the paper's framework

## Suggestions
The most impactful next step is to implement the discrete autoencoder pipeline (Appendix B) and apply C(Z) to representations from at least one pretrained model (e.g., measure C(Z) for CLIP image embeddings or transformer hidden states, visualize the learned tokens W, and show whether interpretable parts emerge). This single experiment would bridge the gap between the paper's theoretical contribution and its practical utility claims.

---

## ih3BJmIZbC

- GT: Accept (Poster) (avg 6.8)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
RSVC (Representational Similarity via Interpretable Visual Concepts) introduces a method for comparing deep neural network representations by leveraging concept-based explainability. The approach extracts visual concepts from each model via dictionary learning, then uses regression to measure how well one model can predict another's concept coefficients. This provides interpretable insights into what visual concepts are shared or unique between models, addressing a key limitation of existing representational similarity methods that produce only a single numerical score.

## Strengths
- **Compelling controlled validation**: The toy concept experiment (Section 4.1) provides strong evidence for RSVC's effectiveness—training one model to use a pink square concept and another to ignore it yields a near-zero similarity score (0.02) for that concept while maintaining high similarity for unrelated concepts. This demonstrates the method's specificity.
- **Extensive architectural coverage**: Experiments span ResNets, Vision Transformers, and self-supervised models (DINO, MAE) across ImageNet and NABirds, showing the approach's generality.
- **Meaningful validation linking representations to behavior**: The replacement test (Section 3.3) effectively demonstrates that concept similarity differences correspond to measurable changes in KL-divergence of logits and prediction accuracy, validating that the method captures functionally relevant differences.
- **Novel visualization strategy**: The over/under-predicted coefficient visualization for interpreting dissimilar concepts provides actionable insights (e.g., identifying that ViT-S lacks the "hands lifting barbells" concept present in ResNet50).

## Weaknesses
- **Missing comparison to established representational similarity methods**: The paper never compares RSVC's coarse-grained similarity rankings to CKA or CCA. This is a significant omission—if existing methods already capture the same relative similarities, RSVC's contribution reduces primarily to visualization without demonstrating quantitative agreement.
- **Reconstruction error variability confounds interpretation**: Table A2 shows reconstruction errors ranging from 176 (ResNet-18) to 1650.8 (ViT-L)—nearly a 10x difference. The paper acknowledges this limitation but doesn't quantify how differing reconstruction quality affects similarity scores. Low similarity between ViT models may partially be an artifact of higher reconstruction error rather than genuine conceptual differences.
- **High computational cost limits practical utility**: Table A3 shows approximately 20 hours for a single ResNet-18 vs ResNet-50 comparison on ImageNet, making large-scale comparisons infeasible.
- **Arbitrary hyperparameter selection**: k=10 concepts and λ=0.1 regularization are chosen with limited validation (swept on a subset, not across all model pairs). Figure A14 shows k significantly affects similarity distributions, but no guidance is provided for selecting k optimally for different architectures.
- **Asymmetric similarity**: The method produces different similarity scores in each direction (M1→M2 ≠ M2→M1). The paper doesn't analyze when/why asymmetry occurs or propose a symmetric variant.

## Nice-to-Haves
- **Human evaluation of interpretability**: As an XAI-focused paper, conducting user studies to verify that the concept visualizations genuinely help humans understand model differences would strengthen the interpretability claims.
- **Extension to NLP models**: The paper acknowledges the method could apply to language models but doesn't demonstrate this. Even preliminary results comparing BERT variants would strengthen claims of generality.
- **Theoretical analysis**: Providing bounds or conditions under which CMCS reliably measures similarity would address the current reliance on purely empirical validation.
- **Comparison to Rosetta Neurons**: Dravid et al. (2023) mines shared features across networks; comparing RSVC findings to this baseline would strengthen positioning.

## Novel Insights
RSVC reveals that model differences are largely driven by medium-similarity, medium-importance concepts rather than low-similarity concepts, and that architecture families (ResNets vs. ViTs) show qualitatively different structures in similarity-importance plots—ResNets exhibit a longer tail of low-similarity, low-importance concepts. The layerwise analysis confirms expected patterns (highest similarity in early layers, gradual decay, slight increase near final layers) while providing architectural insights (e.g., ViT-S layers 4-9 correspond to ViT-L layers 5-20, and ResNet early layers map to even earlier ViT layers). These findings suggest the method can serve as a diagnostic tool for understanding how different architectures organize visual information, not merely a descriptive visualization technique.

## Potentially Missed Related Work
- **Rosetta Neurons (Dravid et al., 2023)** — mines common neurons across models; a direct comparison would strengthen the paper's positioning relative to alternative approaches for finding shared features across networks.

## Suggestions
- Add a quantitative comparison showing that RSVC's coarse-grained similarity rankings correlate with CKA/CCA scores across model pairs, while providing the additional interpretability benefits. This would demonstrate that RSVC doesn't sacrifice quantitative accuracy for interpretability.
- Conduct ablation experiments holding reconstruction error constant (e.g., by adjusting k per model) to isolate whether similarity differences between architectures are genuine or artifacts of extraction quality.

---

## o3V7OuPxu4

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary

StarCraft II Arena proposes a benchmark for evaluating LLMs' strategic planning, real-time decision-making, and adaptability using the StarCraft II game environment. The paper introduces fine-grained capability metrics (RPM, EER, SUR, APM, EPM, WinRateTrend, ErrorRateTrend) and a decision-tracking mechanism to move beyond simple win rates in LLM evaluation. Seven LLMs are evaluated across these dimensions, revealing that proprietary models excel in strategic planning while smaller open-source models outperform in real-time decision-making.

## Strengths

- **Well-motivated problem**: The paper correctly identifies a critical limitation in existing LLM benchmarks—that over-reliance on final success rates neglects intermediate decision-making processes in dynamic environments. This motivation is clearly articulated and addresses a genuine gap in evaluation methodology.

- **Comprehensive metric framework**: The fine-grained capability metrics (defined in Table 5) provide quantifiable measures for distinct aspects of LLM performance: resource management (RPM), supply utilization (SUR), action efficiency (APM/EPM), and adaptability (WinRateTrend, ErrorRateTrend). This enables nuanced model comparison beyond binary win/loss outcomes.

- **Practical experimental setup**: Testing models in both synchronous (real-time) and asynchronous (planning) modes reveals important trade-offs between reasoning quality and response speed. This dual-mode evaluation provides actionable insights for different deployment scenarios.

- **Valuable counterintuitive finding**: The observation that smaller models (Llama 3.1-8B with 47.05) outperform larger models like GPT-4o (21.12) in real-time decision-making provides genuinely useful guidance for real-time application design.

## Weaknesses

- **Win-rate inconsistency undermines metric validity**: GPT-4o achieves the highest strategic planning score (62.01) yet wins only 2/10 games. This discrepancy is never adequately explained and raises questions about whether the metrics capture meaningful behavior or conflate unrelated capabilities. The paper needs to either reconcile this or acknowledge it as a limitation.

- **Insufficient reproducibility details**: Critical implementation specifics are missing: how LLM outputs are mapped to valid StarCraft actions, the complete action space specification, game length statistics, opponent difficulty level, and the environmental setup (pySC2 vs. custom implementation). The async/sync modes are referenced throughout but never formally defined. Without these details, the benchmark cannot be independently reproduced or extended.

- **Limited evaluation rigor**: Testing only 10 games per model provides insufficient statistical power, and no confidence intervals or error bars are reported. Win rates of 2/10 versus 5/10 have substantial variance that could affect conclusions about model differences. The paper lacks statistical significance testing entirely.

- **Table 4 caption-content mismatch**: The table caption mentions dimensions ("social reasoning," "teamwork," "learning ability") that do not appear in the table itself or anywhere else in the paper. This inconsistency undermines credibility and suggests copy-editing errors.

- **Missing comparative baselines**: The paper does not compare LLM performance against built-in StarCraft II agents (at varying difficulty levels) or prior learning-based approaches (e.g., AlphaStar). Win rates of 2-5/10 are uninterpretable without knowing whether these represent strong or weak performance relative to established baselines.

- **Single-race evaluation scope**: All experiments appear to use Protoss exclusively. Since Terran and Zerg have fundamentally different mechanics and strategies, this creates a confounding variable that limits the generalizability of findings.

## Nice-to-Haves

- **Validate fine-grained metrics**: Show that higher scores on the proposed metrics correlate with better actual outcomes (higher win rates, human evaluation of decision quality) to establish predictive validity.

- **Ablation studies**: Test what happens without the fine-grained metrics framework—does standard win-rate evaluation produce different conclusions? This would strengthen the paper's core claim.

- **Failure mode analysis**: Systematically categorize where LLMs fail (economy management, unit composition, micro-control, strategy adaptation) to make the benchmark more diagnostic and useful.

- **More diverse model coverage**: Include additional model families (Claude, Mistral, Qwen) and test different races to demonstrate broader applicability.

## Novel Insights

The paper makes a genuinely useful contribution by demonstrating that evaluating LLMs solely on win rates obscures important capability differences. The finding that GPT-4o's superior strategic planning (62.01) coexists with poor win rates (2/10) and low real-time decision-making (21.12) reveals that these are separable capabilities that current benchmarks fail to distinguish. Similarly, the trade-off between synchronous and asynchronous evaluation modes—where models excel in different conditions—suggests that deployment context fundamentally affects which model "wins." These observations have practical implications for choosing LLMs for real-time applications versus planning-intensive tasks.

## Potentially Missed Related Work

- **Ma et al. (2023) "Large Language Models Play StarCraft II"**: The paper references this work but does not adequately position its contributions against it. This concurrent work on LLM evaluation in SC2 should be discussed in Related Work with clear differentiation of what StarCraft II Arena adds.

## Suggestions

- **Resolve the GPT-4o paradox**: Investigate and explain why the highest-scoring model on strategic planning loses most games. This could reveal important insights about the relationship between planning quality and execution, or expose flaws in the metric definitions. A dedicated analysis section addressing this inconsistency would significantly strengthen the paper.

- **Release implementation details**: Provide the prompt templates, action mapping code, environmental interface specifications, and complete metric definitions in the main text (or at minimum, a clearly referenced supplementary source). Reproducibility is essential for a benchmark paper to have impact.

---

## ZHTYtXijEn

- GT: Reject (avg 2.3)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes DIRAD, a structural adaptation method that grows networks by resolving "statistical conflicts" (net-zero gradients despite nonzero per-sample requirements) through Edge-Node Conversion that creates modulatory nodes capable of inverting opposing gradients. It then extends this to PREVAL, a continual learning framework that detects new tasks and assigns samples to appropriate models without requiring task labels or data replay. The methods are evaluated on 14×14 MNIST and FashionMNIST with 2-class tasks, where PREVAL achieves 73-84% accuracy on 6-class problems compared to 98-99% for baselines that lack task detection capabilities.

## Strengths
- **Novel problem identification**: The concept of "statistical conflicts" causing net-zero adaptive pressure despite nonzero per-sample requirements is well-motivated and addresses a fundamental limitation of gradient descent in fixed topologies.
- **Theoretically grounded core mechanism**: The ENC design with modulatory nodes has formal justification through the neutrality condition, and the design choice of σ₁ ranging from (-1, 3) enabling sign inversion is intuitive and well-explained.
- **Clean motivation and gap identification**: The paper clearly explains why existing continual learning methods fail—either requiring external task signals, storing/replaying data, or lacking autonomous new task detection—and positions PREVAL as addressing all three limitations simultaneously.
- **Practical complexity reduction for L0**: The demonstration that DIRAD can solve 2-class MNIST with ~6 nodes and ~15 edges (versus 3296 for a minimal FC NN) provides strong evidence for the minimal complexity claim in the task network.
- **Unified framework**: The combination of DIRAD for network growth, L0/L1 architecture for prediction validation, and multi-model management into PREVAL addresses continual learning comprehensively rather than piecemeal.

## Weaknesses
- **Significant performance gap undermines practical competitiveness**: PREVAL achieves 73% accuracy on 6-class problems while PNN+FAE achieves 98% and MAS achieves 99% on MNIST (Table 1). Even accounting for PREVAL's additional task detection capabilities, this gap is substantial and raises questions about whether the framework is competitive for practical use. The paper acknowledges baselines "should be seen as upper limits" but does not adequately address this gap.
- **L1 complexity explosion contradicts minimal complexity claims**: The paper states L1 requires ">10-fold increase in edges" compared to L0. The "minimal network complexity" claim only holds for L0; the combined DIRAD+PREVAL system is not minimal. This undermines the scalability argument and creates tension with the stated design philosophy.
- **Limited experimental scope**: Only 2-class classification tasks with 3 tasks total on downscaled 14×14 images is a minimal proof-of-concept. No evidence the approach scales to larger benchmarks (CIFAR, ImageNet) or more complex task structures.
- **Theoretical convergence claim unverified**: Equation 21 provides a condition under which adaptation proceeds, but the paper explicitly states: "We did not check whether this condition would guarantee a global optimum theoretically." The claim of "strong guarantees" in the abstract is therefore overstated.
- **Missing ablation studies**: The paper states "all components of the algorithms are required for their basic operation" without ablation. Components like priority ordering restrictions, the specific modulatory node design, and threshold parameters could be justified or simplified.

## Nice-to-Haves
- **Scale-up experiments on harder benchmarks**: CIFAR-10/100, SVHN, or full 28×28 MNIST with multi-class tasks would strengthen practical viability claims.
- **Formal proof of convergence properties**: Either verify Equation 21 guarantees global optimality or remove/reframe the related claims.
- **Ablation of DIRAD components**: Isolate contributions of ENC vs. edge generation vs. priority ordering to justify design choices.
- **Threshold sensitivity analysis**: TCP, Tconf, TSV, εIS, δmin, R1, R2 are all hand-picked. Systematic analysis would clarify robustness.

## Novel Insights
The paper's most significant insight is the identification of "statistical conflicts"—situations where gradient descent fails not due to local minima but due to opposing gradient pressures that average to zero across samples—along with a novel structural solution. The Edge-Node Conversion mechanism elegantly resolves this by transferring weight gradients (which cannot be collectively optimized) to node deltas (which can be vectorially adapted), mediated through a modulatory node design that can invert signs. This transforms a problem that appears fundamental to gradient-based learning into a structural adaptation problem. The PREVAL framework further extends this by using prediction validation across hierarchical levels (L0 and L1 networks) as a signal for task novelty, providing a principled approach to the task detection problem that has been largely ignored in continual learning literature.

## Potentially Missed Related Work
- **Net2Net (Chen et al., 2016)**: Function-preserving network transformations for architecture growth—directly relevant to the structural adaptation claim.
- **SET (摘) and related dynamic network methods**: Weight-based network growing/pruning methods that operate without explicit architecture search.
- **Predictive coding networks (Rao & Ballard, 1999; Millidge et al., 2021)**: The paper notes PREVAL "can possibly be interpreted within the predictive coding framework" but doesn't deeply engage with this literature for novelty detection.
- **(task-agnostic continual learning methods without replay)**: Methods like "Functional Regularization" approaches that store minimal information rather than samples.

## Suggestions
1. **Address the performance gap**: Either demonstrate PREVAL on harder tasks where baselines fail even with task labels, or show the accuracy gap can be closed through parameter tuning or hybrid approaches (the paper briefly mentions combining PREVAL with familiarity autoencoders—this hybrid deserves exploration).
2. **Revise the theoretical claims**: Remove or clearly qualify the "strong guarantees" language in the abstract given the unverified convergence condition.
3. **Clarify the minimal complexity claim**: Explicitly state that PREVAL's L1 networks are not minimal and discuss tradeoffs between L0 complexity and L1 prediction capability.

---

## QBiFoWQp3n

- GT: Reject (avg 4.6)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper conducts architecturally aligned comparisons between Vision Mambas (VMamba) and newly constructed Gated ConvNets (GConvNets) to determine whether SSM-based token mixers provide genuine advantages over convolutional alternatives in vision tasks. By replacing VMamba's SS2D modules with gated 7×7 depth-wise convolutions while controlling all other architectural variables, the authors demonstrate VMamba's superiority on ImageNet-1K and COCO, reveal that the MLP classifier (not the block structure) explains MambaOut's prior success, and discover that hybrid architectures with self-attention benefit both models through opposite mechanisms.

## Strengths
- **Methodologically rigorous comparison framework**: The paper carefully controls macro architecture, depth-width trade-offs, and computation budgets when comparing token mixers, addressing a genuine gap in the literature where prior comparisons had uncontrolled variables (e.g., MambaOut vs VMamba).
- **Significant empirical finding**: The ablation revealing that the MLP classifier explains MambaOut's superiority over VMamba overturns a potentially misleading conclusion in prior work and provides valuable practical guidance.
- **Novel insight on hybrid models**: The discovery that self-attention reduces overfitting in VMambas while enhancing fitting ability in GConvNets is an interesting and practically useful finding with clear implications for architecture design.
- **Comprehensive evaluation**: Experiments span classification (ImageNet-1K), detection/instance segmentation (COCO), and robustness benchmarks (ImageNet-A/R/C), providing a holistic performance picture across three model sizes.

## Weaknesses
- **Missing statistical rigor**: No error bars, standard deviations, or confidence intervals are reported despite small accuracy differences (0.4-0.6% on ImageNet-1K). Without multiple runs or statistical validation, it is unclear whether observed differences are reliable.
- **Potential confound from EMA inconsistency**: VMamba uses EMA (following original paper) while GConvNet does not ("removing the EMA as it does not improve performance"). This creates a potential confound—if EMA provides any benefit, this could partially explain VMamba's apparent advantage on validation metrics.
- **Limited generalizability**: The paper focuses exclusively on VMamba as the sole Vision Mamba variant and gated 7×7 depth-wise convolutions as the sole convolutional alternative. Conclusions may not extend to other Vision Mamba architectures (LocalMamba, PlainMamba) or other convolution variants.
- **Shallow mechanistic analysis**: The attribution of VMamba's superiority to "stronger expressivity" based on training loss curves is circumstantial. The root cause of VMamba's observed overfitting issue is not investigated.
- **Incomplete ablation on gated branch**: The paper does not isolate whether the advantage comes from the SSM mechanism itself or other architectural factors. A GConvNet variant without the gated branch would clarify this.

## Nice-to-Haves
- **Semantic segmentation evaluation**: The paper claims to provide evidence for "vision" broadly but only reports ImageNet-1K and COCO. ADE20K segmentation results would strengthen the scope, particularly given that prior work showed VMamba's advantage is more pronounced in segmentation.
- **Throughput/latency benchmarks**: The paper touts SSM's linear complexity advantage but presents only accuracy and FLOPs. Actual runtime measurements would substantiate efficiency claims.
- **Ablation of SS2D scan directions**: Not isolating which component (selective mechanism vs. bidirectional scanning) contributes to VMamba's advantage limits mechanistic understanding.
- **Comparison with ConvNeXt-V2**: GConvNet is a novel baseline not compared against existing gated-CNN state-of-the-art, leaving uncertainty about whether results generalize beyond the specific comparison.

## Novel Insights
The paper's most valuable contribution is demonstrating that prior claims about Mamba being unnecessary for vision (MambaOut) were based on an unfair comparison—the MLP classifier, not the token mixer, was responsible for MambaOut's advantage. Perhaps more interestingly, the finding that self-attention plays opposite roles in VMambas (reducing overfitting) versus GConvNets (improving fitting ability) reveals fundamental differences in how these architectures learn and suggests that hybrid Mamba-Transformer models may inherit complementary benefits from both paradigms. The empirical evidence that SSM token mixers provide genuine advantages over well-tuned convolutional alternatives—especially for downstream tasks like detection—provides important guidance for future architecture design.

## Potentially Missed Related Work
- PlainMamba (Yang et al., 2024): Non-hierarchical Mamba variant that could provide additional context for hierarchical architecture effects.
- Mamba-2 (Dao & Gu, 2024): The paper references this work on hybrid Mamba-attention models but could more deeply compare findings on hybrid architecture design principles.

## Suggestions
- Run experiments with at least 2-3 different random seeds and report mean ± std to validate whether observed differences are statistically significant rather than noise.
- Perform a controlled ablation with and without EMA for both VMamba and GConvNet to isolate whether EMA usage explains any portion of the performance gap.
- Consider isolating the contribution of SS2D's bidirectional scanning versus the selective mechanism itself through an ablation with reduced scan directions.

---

## kN6MFmKUSK

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
PolaFormer introduces a polarity-aware linear attention mechanism that explicitly models both same-signed (positive-positive, negative-negative) and opposite-signed (positive-negative, negative-positive) query-key interactions, addressing information loss inherent in traditional linear attention's non-negative feature maps. The method also incorporates a learnable power function with theoretical guarantees for entropy reduction, using depthwise convolutions to mitigate low-rank issues in attention matrices. Extensive experiments across ImageNet classification, COCO detection/segmentation, ADE20K semantic segmentation, and Long Range Arena benchmarks demonstrate consistent improvements of 0.3% to 4.6% over baseline transformers while maintaining linear complexity.

## Strengths
- **Well-motivated problem diagnosis**: The paper clearly identifies two fundamental limitations of linear attention—loss of negative query-key interactions and loss of attention "spikiness" (high entropy)—with concrete empirical evidence (Figure 1) showing how prior methods produce overly uniform attention distributions
- **Sound theoretical contribution**: Theorem 1 provides conditions under which element-wise functions with positive first and second derivatives can reduce entropy in attention distributions, with complete proofs in the appendix (Lemmas 1 and 2). The insight connecting convex functions to entropy reduction is valuable for understanding why softmax produces sharper attention
- **Comprehensive experiments**: Evaluations span four diverse benchmarks (ImageNet-1K, COCO, ADE20K, LRA) with multiple backbone integrations (DeiT, PVT, PVTv2, Swin), demonstrating consistent improvements and good compatibility across architectures
- **Effective ablation study**: Table 3 systematically isolates contributions of polarity-aware attention (+1.8%), polarity-aware coefficients (+6.5%), and depthwise convolutions (+12.7%), validating each component's value
- **Modular design**: The approach is architecture-agnostic, successfully integrated with major vision transformers with minimal parameter overhead

## Weaknesses
- **Missing ablation on learnable power function**: The power function is a key theoretical contribution but its isolated effect is not demonstrated in the ablation study. Table 3 does not show what happens without the learnable power exponent, only confirming the combined approach reaches 74.6%
- **No comparison with state-space models**: Recent methods like VMamba (Liu et al., 2024) and Vision Mamba (Zhu et al., 2024) are mentioned in related work but not included in experiments. Given that SSM-based methods represent a major alternative for efficient vision transformers, direct comparisons would strengthen the paper's positioning
- **Scalability concerns**: Improvements diminish substantially on larger models (0.3-0.6% on Swin-B vs 2.1-3.7% on PVT-T). The paper does not analyze why this occurs or whether the approach is best suited for resource-constrained scenarios
- **Insufficient analysis of initialization sensitivity**: Table A.4 in the appendix shows substantial accuracy variation (70.70% to 74.40%) depending on initialization of G matrices. This practical concern is not discussed in the main paper or acknowledged as a limitation

## Nice-to-Haves
- **Longer sequence scaling experiments**: LRA results only go to 4k sequence length. Experiments at 8k, 16k, 32k tokens demonstrating maintained linear scaling would more convincingly validate the efficiency claims
- **Cross-head diversity analysis**: The method adds per-head learnable exponents but does not analyze whether different heads learn different patterns, making it unclear if per-head specialization justifies the added complexity
- **Quantitative reconstruction analysis**: The paper claims the method "faithfully reconstructs" original softmax attention but provides no quantitative measure of reconstruction error ||SM(q,k) - PolaAttn(q,k)||

## Novel Insights
PolaFormer makes a conceptually elegant observation: existing linear attention methods discard negative query-key interactions entirely (by applying ReLU or ELU+1 to features), which loses information that may be semantically meaningful. The paper's insight to decompose query-key dot products element-wise into same-signed and opposite-signed interactions, then process them through separate value streams with learnable scaling coefficients, represents a genuinely new approach to bridging the gap between linear and softmax attention. The theoretical connection between convex element-wise functions and entropy reduction provides useful analytical insight, even if the specific application (power function with ReLU pre-processing) is relatively straightforward. The complementary learning between G_s and G_o matrices (shown in Figure 3) suggests the model can effectively balance the two interaction types, though deeper analysis of what semantic patterns these matrices capture would strengthen this claim.

## Potentially Missed Related Work
- **Hedgehog (Zhang et al., 2024a)**: A concurrent work on expressive linear attentions with softmax mimicry that achieves strong LRA results (59.66% average). Comparison would contextualize PolaFormer's 60.72% improvement
- **VMamba (Liu et al., 2024)** and **Vision Mamba (Zhu et al., 2024)**: State-space model approaches to efficient vision that are discussed in related work but not compared experimentally
- **Agent Attention (Han et al., 2023b)**: Another concurrent approach to improving linear attention using agent tokens, which the paper critiques but does not benchmark against

## Suggestions
- Add an ablation specifically isolating the learnable power function contribution to validate it independently of the polarity-aware mechanism
- Include at least one comparison against an SSM-based baseline (VMamba or Vision Mamba) to situate the work among current efficient vision alternatives
- Discuss the initialization sensitivity observed in Table A.4 and provide guidance on recommended initialization strategies for practitioners

---

## VpWki1v2P8

- GT: Accept (Oral) (avg 8.7)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces LoRA-RITE, a novel adaptive matrix preconditioning optimizer for LoRA fine-tuning that achieves transformation invariance—the property that equivalent LoRA factorizations should produce identical updates to the underlying weight matrix. The method uses unmagnified gradients and one-sided matrix preconditioning on the low-rank side, achieving transformation invariance with only O(r) overhead compared to Adam. Experiments across Gemma-2B, Gemma-7B, and mT5-XXL on Super-Natural Instructions and multiple LLM benchmarks show consistent 2-5% improvements over Adam and substantial margins over other optimizers.

## Strengths
- **Novel and important problem identification**: The paper clearly demonstrates that all existing optimizers (Adam, Adagrad, RMSProp, Shampoo, LAMB) fail scalar scale invariance, and provides formal proofs showing diagonal preconditioners mathematically cannot achieve transformation invariance. This is a fundamental inconsistency in LoRA optimization that the community has overlooked.

- **Elegant theoretical solution**: The unmagnified gradient concept (∇̄A = ∇A RB†) and the resulting transformation invariance property are theoretically sound. Theorem 2 provides a complete proof that Algorithm 1 achieves invariance across all equivalent LoRA pairs.

- **Significant and consistent empirical gains**: The method shows 2-5% improvements over Adam across diverse tasks (classification, generation, reasoning) and model scales (2B to 7B parameters), with particularly strong gains on reasoning tasks like GSM8K (55.50% vs 48.37% for Adam on Gemma-7B).

- **Practical efficiency**: Time complexity O(mr² + nr² + r³) is only r times slower than Adam (typically r=16, negligible compared to backpropagation), with minimal memory overhead. This is a substantial improvement over full matrix preconditioners like Shampoo.

- **Strong baseline comparison**: The paper includes Adam, LoRA+, ScaledAdam, Shampoo, and Lamb—covering both first-order, second-order, and recent LoRA-specific methods. The consistent superiority of LoRA-RITE across all baselines is compelling.

## Weaknesses
- **Missing statistical significance**: All main results in Tables 1 and 2 report point estimates without error bars or statistical significance tests. For a paper claiming substantial improvements (4-5% over strong baselines), the variance across random seeds is unknown. This is a significant empirical rigor concern for ICLR.

- **Escaped mass term not ablated**: The ρ_A^t term (escaped mass) is central to the algorithm's handling of basis changes in accumulated moments, but no experiment isolates its individual contribution. It's unclear whether the empirical gains come from transformation invariance or from the ρ regularization effect.

- **Theory-practice gap in convergence analysis**: Theorem 3 and 4 analyze a simplified setting where first moment is omitted and second moment uses direct summation. However, the actual Algorithm 1 uses EMA for both moments (lines 6, 9), creating a gap between the stated theory and the implemented algorithm.

- **Assumption 1 not discussed**: The stronger convergence result (Theorem 4) relies on Assumption 1, which bounds spectral properties of the basis projection matrices. The paper provides no discussion of when this assumption holds or how restrictive it is in practice—a significant gap for a theory-focused paper.

- **Limited model architecture diversity**: Experiments are confined to Gemma and mT5-XXL. Evaluating on LLaMA, Mistral, or Phi models would strengthen claims about generalization across architectures.

- **No full fine-tuning comparison**: The paper never compares against full fine-tuning, which is the true performance ceiling. Without this comparison, the practical significance of LoRA-RITE's gains is unclear relative to simply using full fine-tuning.

## Nice-to-Haves
- Ablate the polar decomposition requirement: Replacing it with alternatives (SVD, no basis extraction) would validate whether this computational overhead is justified by the invariance property.
- Direct empirical validation of efficient feature learning: Theorem 1 claims transformation invariance guarantees efficient feature learning, but the paper doesn't show that δZx maintains O(1) magnitude empirically.
- Extended training dynamics analysis: Plots showing gradient magnitudes, preconditioner conditioning, or escaped mass growth over training would strengthen the "why it works" narrative.
- Apply to LoRA variants: Demonstrating LoRA-RITE on DoRA or AdaLoRA would broaden the contribution and show the invariance property transfers.

## Novel Insights
The paper's most valuable contribution is identifying transformation invariance as a fundamental desideratum for LoRA optimization that all existing optimizers violate. The insight that equivalent factorizations (A,B) and (AR, BR^{-T}) should produce identical updates is both mathematically principled and practically important, as empirically demonstrated by the unbalanced training dynamics where factor B dominates updates while factor A barely changes. The unmagnified gradient concept—extracting the column space information independent of magnitude through polar decomposition—is a clever technical solution that enables transformation invariance while maintaining the efficiency of one-sided preconditioning. The escaped mass mechanism for handling accumulated moments under basis changes is also novel and addresses a subtlety in extending adaptive methods to non-Euclidean parameterizations.

## Potentially Missed Related Work
None identified based on the related work section provided. The paper appropriately discusses LoRA+, ScaledAdam, Shampoo, LAMB, and Riemannian gradient descent as relevant prior work.

## Suggestions
1. **Add error bars to all experiments**: Run each experiment with at least 3 random seeds and report mean ± std. This is essential for assessing the reliability of claimed improvements.

2. **Ablate the escaped mass term**: Include an experiment removing ρ_A^t to determine whether it contributes positively to performance or is merely a corrective mechanism for the theory.

3. **Close the theory-practice gap**: Either extend the convergence analysis to the EMA setting used in Algorithm 1, or implement the simplified version analyzed in the theory for a fair comparison.

4. **Add full fine-tuning baselines**: Include at least one experiment comparing against full fine-tuning to contextualize LoRA-RITE's practical utility.

---

## ywFOSIT9ik

- GT: Accept (Spotlight) (avg 6.8)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
This paper investigates what random perturbation distribution minimizes the variance of two-point zeroth-order gradient estimators, formulating a constrained functional optimization problem over perturbation distributions. The key theoretical contribution identifies two classes of minimum-variance perturbations: the well-known fixed-length perturbations and a novel class called directionally aligned perturbations (DAPs). The paper provides convergence analysis for SGD using δ-unbiased perturbations and demonstrates DAPs empirically on synthetic optimization problems and LLM fine-tuning.

## Strengths
- **Novel theoretical insight**: The identification of directionally aligned perturbations (DAPs) as a distinct minimum-variance class is genuinely novel and underexplored in existing literature. The functional optimization framework provides a principled way to derive minimum-variance conditions.
- **Technically sound core**: Theorem 2.2 is correctly derived using Cauchy-Schwarz inequality, and the equality conditions are properly analyzed. Proposition 4.1's proof for the sampling algorithm is mathematically sound.
- **Unifying framework**: Corollary 3.2 successfully extends convergence bounds O(d/ε²) for non-convex and O(d/ε) for strongly convex settings to a broader class of minimum-variance perturbations, unifying uniform smoothing, Rademacher, coordinate descent, and other methods.
- **Informative visualizations**: Figures 1 and 2 effectively illustrate the directional alignment property and anisotropic behavior that distinguishes DAPs from uniform perturbations.

## Weaknesses
- **Incomplete necessity characterization**: The paper acknowledges (page 4-5) that mixed distributions (DAP with probability p, uniform with probability 1-p) achieve minimum variance without satisfying either condition in Theorem 2.2, yet the theorem is presented as characterizing all minimum-variance perturbations. The claim is only sufficient, not necessary as stated.
- **Limited empirical scope**: Experiments are confined to d=16 synthetic functions and a single LLM task (OPT-1.3b on SST-2). This is insufficient to validate claims about scalability to high dimensions and generalizability to diverse applications mentioned in the Introduction (adversarial attacks, RL).
- **Unfair computational comparison**: Algorithm 2 requires 2b function evaluations per step (b/2 for initial gradient estimate + b/2 for DAPs), yet comparisons use the same batch size b rather than the same total query budget. It's unclear whether DAP's gains come from directional alignment or simply more function evaluations.
- **Missing strong baselines**: The paper never compares against variance-reduced ZO methods like ZO-SVRG-Coord-Rand or ZO-SPIDER-Coord (Ji et al., 2019), which are standard SOTA baselines mentioned in the related work section.
- **Asymptotic-only theory**: All theoretical results assume μ → 0, but practical ZO optimization uses finite perturbation steps. Non-asymptotic analysis or characterization of performance degradation as μ increases is absent.

## Nice-to-Haves
- Error bars or confidence intervals in experimental figures to demonstrate statistical significance
- Ablation on gradient estimation quality in Algorithm 2 (how DAP degrades with imperfect gradient estimates)
- Explicit characterization of when DAPs outperform fixed-length perturbations (e.g., variance reduction ratio as a function of gradient anisotropy)
- Comparison on additional LLM tasks beyond SST-2 (e.g., instruction tuning, harder NLP tasks)

## Novel Insights
The paper's central insight—that minimum-variance perturbations need not have fixed length but can instead align directionally with the gradient—is both theoretically elegant and practically relevant. By reformulating the variance minimization as a functional optimization over distributions with a δ-unbiasedness constraint, the authors reveal that the widely-used Gaussian perturbation does not achieve minimum variance due to excess kurtosis, a finding that challenges common practice. The DAP scheme's anisotropic behavior (providing lower variance along dimensions with larger gradient components) is particularly compelling for sparse gradient scenarios, representing a genuinely new design principle for zeroth-order gradient estimators.

## Potentially Missed Related Work
- **ZO-SVRG-Coord-Rand and ZO-SPIDER-Coord** (Ji et al., 2019): Mentioned in related work Section A.1 but never compared in experiments, despite being standard SOTA baselines for ZO optimization
- **Variance-reduced ZO methods** generally: The experimental section would benefit from comparisons against these methods to contextualize DAP's practical utility

## Suggestions
1. **Revise Theorem 2.2 framing**: Clarify explicitly that conditions are sufficient (not necessary) for minimum variance, or prove necessity under additional assumptions
2. **Match query budgets in experiments**: Compare DAP against uniform/Gaussian using the same total number of function evaluations, not the same batch size
3. **Add higher-dimensional experiments**: Include synthetic experiments at d ∈ {128, 512, 1024} to validate theoretical claims about DAP's utility in high-dimensional sparse settings
4. **Include variance-reduced baselines**: Compare against ZO-SVRG, SPIDER, or other modern variance-reduced ZO methods

---

## pL8ws91RW2

- GT: Reject (avg 2.6)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces a hierarchical self-supervised graph contrastive learning framework that performs contrastive learning at node, subgraph, and graph levels to capture multi-scale structural information. The method generates augmented graph views and maximizes agreement between embeddings at multiple hierarchical scales. Experiments on citation networks (Cora, Citeseer, PubMed) and a social network (Reddit) demonstrate state-of-the-art performance in node classification, particularly in low-label regimes.

## Strengths
- **Comprehensive experimental evaluation**: The paper evaluates on multiple benchmark datasets in both transductive and inductive settings, includes ablation studies, low-label regime experiments, and t-SNE visualizations demonstrating discriminative embedding quality.
- **Good reproducibility practices**: Detailed hyperparameter settings per dataset, hardware specifications, random seed settings (42), mention of code availability, and extensive appendix implementation details.
- **Consistent performance gains**: The method achieves highest accuracy across all four datasets (Cora: 86.2%, Citeseer: 74.6%, PubMed: 81.5%, Reddit: 96.1%), improving ~1-2% over previous state-of-the-art.
- **Clear motivation**: Effectively motivates the need for self-supervised learning on graphs and explains the rationale for capturing multi-scale structural information through hierarchical contrast.

## Weaknesses
- **Missing graph-level contrastive loss equation**: The paper references equations for graph-level contrast (Section 3.10) but the actual equation is presented as "picture intentionally omitted." This is a fundamental methodological component—a third of the proposed "hierarchical" framework cannot be evaluated. Without this equation, the method is incomplete and reproducibility is compromised.
- **"Adaptive augmentation" claim unsubstantiated**: Section 3.1 and the abstract claim an "adaptive" augmentation strategy, but the paper describes fixed augmentation rates (e.g., edge drop: 0.2, feature mask: 0.3 from Table 7) without any adaptation mechanism. This is a misleading characterization; the augmentations are standard and fixed-rate, not adaptive.
- **Missing standard deviations in main results**: While Appendix C.5 mentions "average of 10 runs," the main paper (Table 4) reports only point estimates without standard deviations. Given the ~1-2% improvement margins, ICLR standards require statistical rigor to assess whether gains are meaningful or within noise.
- **Novelty claim imprecise**: The paper claims to be "the first" with hierarchical contrastive learning and Table 1 presents this as a distinguishing feature. However, MVGRL (Hassani et al., 2020) already performs multi-view contrastive learning combining node-level and graph-level representations. The paper should more precisely characterize what distinguishes their hierarchical approach from prior multi-view methods.
- **Incomplete subgraph extraction details**: Section 3.5 describes subgraph sampling but never specifies: (a) the k-hop radius used, (b) the pooling function for obtaining subgraph embeddings (Eq. 11 shows a generic "Pool" notation without implementation details), or (c) whether random walks or k-hop neighborhoods are used. This information is essential for reproducibility.
- **Negative sampling strategy unspecified**: Contrastive learning performance critically depends on negative sampling. The paper does not clearly specify how negative pairs are constructed for subgraph-level and graph-level contrasts. Using subgraphs from different central nodes as negatives assumes they represent different semantic groups, which may not hold in dense graphs with overlapping neighborhoods.

## Nice-to-Haves
- Add BGRL, CCA-SSG, and GraphMAE (Table 8 in appendix) to main paper Table 4 for complete SOTA comparison, not just appendix.
- Report ablation results on subgraph extraction radius (k) to justify why subgraph-level contrast provides value beyond node-level alone.
- Add statistical significance testing (e.g., paired t-tests) for key comparisons against MVGRL and other top baselines.
- Provide analysis of when hierarchical contrast helps vs. hurts (graph types, homophily levels) to establish the method's true applicability.

## Novel Insights
The paper's core insight—combining contrastive objectives at node, subgraph, and graph levels within a unified framework—is conceptually sound and well-motivated. The ablation studies provide evidence that each hierarchical level contributes positively to performance, with node-level contrast having the largest impact. The consistent improvements across datasets and settings (transductive, inductive, low-label regimes) suggest the approach successfully captures complementary structural information at different scales. However, the paper does not provide qualitative analysis of *what* each level learns, making it difficult to assess whether the three-level hierarchy captures fundamentally different structural patterns or partially overlapping information.

## Potentially Missed Related Work
- **BGRL (Bootstrapped Graph Latents)**: While mentioned in Appendix Table 8, it could be more prominently compared given it represents a bootstrapping-based alternative to contrastive methods.
- **CCA-SSG (Canonical Correlation Analysis for Self-Supervised Graph Learning)**: Provides a different self-supervised objective (CCA-based) that could serve as an informative comparison point.
- **HIGHLIGHTS / GRACE residual**: Zhu et al. (2020) GRACE is cited, but the paper does not discuss whether there are baseline implementation concerns (the paper's DGI results appear slightly lower than published numbers).

## Suggestions
- **Complete the methodology**: Restore the graph-level contrastive loss equation (Eq. 12) and provide mathematical details for subgraph extraction and pooling operations.
- **Either implement actual adaptive augmentation or rephrase**: If the augmentation strategy does not adapt based on training dynamics, remove the "adaptive" claim and describe it as a multi-augmentation strategy instead.
- **Report variance in all main results**: Add standard deviations to Table 4 and consider adding confidence intervals or significance test results to ablation tables.
- **Clarify inductive setting**: The 90%/10% split on Reddit tests unseen nodes within the same graph, not generalization to entirely new graphs. Rename or clarify this evaluation setting.

---

## XQED8Nk9mu

- GT: Reject (avg 5.0)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces the first method for generating Counterfactual Explanations (CEs) for 3D point cloud classifiers using a diffusion model with score distillation loss, addressing a clear gap in XAI research. The approach optimizes perturbations in a latent space learned by the LION diffusion model, employing novel losses including a K-nearest-neighbor structure prior and diversity regularization. While the paper provides a solid technical foundation and comprehensive evaluation metrics, several methodological concerns and limited evaluation scope temper the contribution's strength.

## Strengths
- **First work on 3D point cloud counterfactuals**: This genuinely opens a new research direction in XAI, with clear motivation for why 3D explainability matters (autonomous systems, healthcare). The paper correctly identifies this as a gap (Section 1: "to the best of knowledge, no such effort has yet been made for 3D processing ones").
- **Novel adaptation of score distillation**: The application of score distillation loss from DreamFusion to counterfactual generation is technically sound. The ablation study (Table 2) demonstrates substantial improvements: L_sds reduces FID from 0.836 to 0.154 and increases NTFR from 0.506 to 0.850.
- **Theoretically-grounded structure loss**: The K-nearest-neighbor distribution prior (Eqs. 4-5) derives from principled statistical assumptions about point distributions, with a proper mathematical derivation in Appendix A.1.
- **Comprehensive evaluation metrics**: The paper introduces a tailored metric suite for 3D CEs (FID, NTFR, MNO, CD, LPIPS-SimpleView, MNAC, Diversity), providing a foundation for future research in this domain.
- **Thorough ablation study**: Table 2 systematically validates each loss component's contribution, showing that all components provide meaningful improvements.

## Weaknesses
- **Unfair baseline comparison**: The AE baseline is trained per-category while Diffusion uses all 55 categories of ShapeNetCore (confirmed in Appendix Section "Pre-trained Generators"). This conflates architecture differences with training data scale differences. A fair comparison would require matching training conditions.
- **Limited evaluation scope**: The method is tested only on DGCNN classifiers with no evaluation on other architectures (PointNet, PointNet++). Generalization to other classifier types remains undemonstrated.
- **Real-world data absent**: Evaluation uses only ShapeNetCore (clean synthetic CAD data). Despite motivation about autonomous systems and healthcare, no experiments on real sensor data (KITTI, ScanNet, ModelNet) are provided.
- **Multi-class claim is inaccurate**: The paper states being "the first to study CEs for multi-class classifiers; even in the image domain, previous studies focused on binary cases only." This is demonstrably false—multi-class CEs have been explored in prior work (Wachter et al. 2017, Mothilal et al. 2020). The actual contribution is limited to targeting the second-highest probability class.
- **Missing implementation details**: K for K-NN in structure loss is never specified. The diversity loss weight λ_div is absent from Table 3a. These omissions affect reproducibility.
- **Metric trade-offs unaddressed**: The ablation shows L_sds dramatically improves FID (0.836→0.154) but substantially increases CD (4.153→10.364) and LPIPS (0.811→1.003). The paper does not discuss which metric should be prioritized or analyze this trade-off.
- **No statistical rigor**: No error bars, confidence intervals, or variance estimates are reported. Results appear deterministic despite using random initializations.

## Nice-to-Haves
- **Human evaluation study**: The paper claims CEs should be "understandable and meaningful to humans" but provides no human validation. Even a small study would strengthen interpretability claims.
- **Runtime analysis**: No computational cost comparisons are provided. While score distillation reduces backprop through diffusion, total optimization cost versus AAE baseline is unclear.
- **Classifier architecture ablation**: Testing on PointNet or PointNet++ would demonstrate whether the method's effectiveness is classifier-specific or generalizable.
- **Cross-category CE demonstration**: Showing CEs that intentionally target diverse class transitions would better illustrate semantic control.

## Novel Insights
This work's primary value lies in demonstrating feasibility: counterfactual explanations can be generated for 3D point cloud classifiers through diffusion-based latent space optimization. The score distillation loss adaptation eliminates the need for backpropagation through multiple diffusion steps, making the approach computationally tractable. The empirical observation that "breaking symmetry" changes classifier outputs (suggesting reliance on symmetric structures) represents a potentially valuable model debugging insight, though it lacks statistical validation in the current work.

## Potentially Missed Related Work
- Wachter et al. (2017) "Counterfactual explanations without opening the black box" — showed multi-class counterfactuals in image domain, contradicting the paper's claim of being "first for multi-class"
- Mothilal et al. (2020) "Explaining machine learning classifiers through diverse counterfactual explanations" — multi-class CE work in 2D image domain
- Point cloud counterfactual work building on Leclerc et al. (2021) "3DB" framework for vision model debugging
- Point diffusion models beyond LION: Luo & Hu (2021) energy-based approaches for point cloud generation

## Suggestions
1. **Revise the multi-class claim**: Conduct a proper literature review and either remove this as a main contribution or clarify it as "first multi-class CE specifically for point cloud classifiers targeting second-highest probability class."
2. **Match training conditions for baselines**: Retrain AE on all 55 categories to enable fair architectural comparison, or retrain Diffusion on single-category to isolate method differences.
3. **Add at least one additional classifier**: PointNet is a natural choice and would substantially strengthen generalization claims.
4. **Specify missing hyperparameters**: Report K for K-NN and provide λ_div value in the hyperparameter table.
5. **Report variance metrics**: At minimum, run CE generation with multiple random seeds and report standard deviations for key metrics (FID, NTFR).

---

## 1NYhrZynvC

- GT: Reject (avg 2.5)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper establishes a theoretical framework for adaptive stepsize selection in gradient descent, deriving an optimal stepsize αk* = ⟨x⋆ − x[k], −∇f(x[k])⟩ / ∥∇f(x[k])∥² that achieves exact linear convergence with factor sin²(ηk), where ηk measures the angle between the direction to the optimum and the negative gradient. The feasible stepsize range (0, 2αk*) is shown to contain the classical (0, 2/L) range. Practical approximations avoiding knowledge of x⋆ are proposed, with empirical validation on MNIST classification.

## Strengths
- **Elegant theoretical derivation**: The paper provides complete and correct proofs for all major results. The key insight that minimizing the error decrease term yields αk* with exact linear rate sin²(ηk) is novel and theoretically significant. The derivations via error characterization (Equation A.3) are mathematically rigorous.

- **Meaningful theoretical connections**: Proposition 2.2 correctly establishes αk* ≥ 1/L, proving the adaptive range (0, 2αk*) strictly contains the classical (0, 2/L) range, validating consistency with established L-smoothness theory while strictly generalizing it.

- **Addressing ill-conditioning**: The R² quadratic example (Section 4.1) demonstrates a strictly better convergence rate than exact line search, with the error factor bounded by 1/2ᵏ regardless of the condition number γ—a meaningful improvement over existing approaches for this well-known bottleneck.

- **Principled practical framework**: Algorithm 1 provides a coherent auto-correction mechanism with theoretical convergence guarantees when f̄₀ is a valid lower bound, and the tune-free case αk = f(x[k])/∥∇f(x[k])∥² is elegantly derived for the f(x⋆) = 0 case.

## Weaknesses
- **Fundamental theory-practice gap**: Both the optimal stepsize αk* and the practical approximation's convergence analysis depend on x⋆ through ⟨x⋆ − x[k], −∇f(x[k])⟩ and σₖ. The paper acknowledges this limitation but does not adequately address it. The claim in the abstract that the result "is universally applicable" is misleading—the theoretical result requires knowledge of x⋆, which is unavailable in practice.

- **Contradictory default parameter**: Corollary 3.4 states "we should choose γ₀ > 1" while Remarks 3.3 states "our default choice is therefore conservatively set to γ₀ = 1." No justification is provided for this inconsistency, and no experiments test γ₀ > 1.

- **Algorithm 1 correction breaks theory**: The correction mechanism in Algorithm 1 (line 5) modifies γ₀ and resets x[k+1] ← x[k] when certain conditions are violated. This heuristic intervention invalidates the linear rate proof in Theorem 3.1, which assumes the stepsize formula holds without intervention. Either convergence with corrections must be proven, or the corrections should be removed from the algorithm description.

- **Unsubstantiated non-convex global convergence claim**: Theorem 2.1 and the conclusion claim convergence to "the global optimal point" for non-convex functions. This is false in general—non-convex functions have multiple local minima, and the paper provides no mechanism to guarantee global rather than local convergence. This claim must be retracted or substantially qualified to stationary point convergence.

- **Limited empirical validation**: MNIST experiments use only a small 2-layer network (200 hidden units). No comparison with backtracking line search (the standard textbook approach) or modern neural network optimizers (LAMB, SAM) is provided. The claim of "better performance than state-of-the-art" is not supported by experiments on modern benchmarks.

- **No stochastic gradient analysis**: The paper proposes a fixed-stepsize GD framework but applies it to neural networks with mini-batches (Section 4.3) without theoretical justification. No analysis of how the stepsize choice behaves under stochastic gradients is provided.

## Nice-to-Haves
- Comparison with classical line search methods (Armijo backtracking) to contextualize practical utility
- Systematic hyperparameter sensitivity analysis for γ₀, τ₁, τ₂, and T beyond "very roughly picked"
- Experiments on CIFAR-10 or ImageNet to support claims about state-of-the-art performance
- Characterization of when the tune-free case (f(x⋆) = 0) applies, with guidance for general cases

## Novel Insights
The paper's core insight—deriving the optimal adaptive stepsize by directly minimizing the error decrease in one iteration—is genuinely novel. The characterization of the feasible stepsize range as (0, 2αk*) with αk* = ⟨x⋆ − x[k], −∇f(x[k])⟩ / ∥∇f(x[k])∥² and the resulting exact linear rate sin²(ηk) provides an elegant alternative to classical L-smoothness-based bounds. The insight that the optimal stepsize depends on the angle ηk between the gradient and the direction to the optimum connects geometric intuition to algorithmic design in a principled way.

## Potentially Missed Related Work
- **Line search methods**: The paper does not compare against classical backtracking line search (e.g., Armijo-Wolfe conditions), which is the standard approach for stepsize selection and has solid theoretical guarantees. This comparison is essential for validating practical utility.

## Suggestions
1. **Prove convergence of Algorithm 1 with the correction step**: The current theory assumes the stepsize formula holds without modification. The authors should either (a) prove convergence when corrections occur, or (b) remove the corrections and rely purely on the theory.

2. **Resolve the γ₀ contradiction**: Either justify why γ₀ = 1 is used as default despite Corollary 3.4 recommending γ₀ > 1, or conduct experiments to determine the optimal default value.

3. **Correct the non-convex claim**: Replace "global optimal point" with a qualified statement about stationary point convergence, or clarify the specific non-convex function classes for which global convergence holds.

---

## UkGrcekmSZ

- GT: Reject (avg 2.0)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary
This paper presents a transfer learning framework using pre-trained ResNet-50 and EfficientNetV2 to classify renal diseases from CT images across four binary classification tasks (Normal vs Cyst, Cyst vs Stone, Cyst vs Tumor, and Stone vs Tumor). The authors report testing accuracies approaching 100% and claim EfficientNetV2 performs marginally better than ResNet-50. While the clinical motivation is well-articulated and the evaluation metrics are comprehensive, the reported perfect accuracy raises serious methodological concerns, and the paper lacks sufficient technical details for reproducibility.

## Strengths
- **Clear clinical motivation**: The paper effectively establishes the global burden of kidney diseases and the critical shortage of specialists (1 per million in Asia vs. 25.3 in Europe), providing a compelling rationale for automated diagnostic tools.

- **Comprehensive evaluation metrics**: The authors report accuracy, precision, recall, AUC-ROC, and confusion matrices across all four binary classification tasks, providing a multi-faceted view of model performance.

- **Proper validation methodology**: The paper employs a checkpointing mechanism to select the best model weights based on validation performance, preventing the use of overfitted models.

- **Detailed architectural descriptions**: Both ResNet-50 and EfficientNetV2 are thoroughly explained, including key components like skip connections, MBConv blocks, and the adapted binary classification heads.

## Weaknesses

- **Critical data leakage concern**: The paper claims 100% testing accuracy, which is highly suspicious for medical imaging tasks. The Kaggle CT Kidney Dataset used in this study is known to contain near-duplicate images and multiple slices per patient. The paper provides no evidence that images from the same patient were excluded from both training and test sets. Patient-level splits are standard practice in medical imaging and essential for reliable evaluation—without this verification, the perfect accuracy likely reflects memorization rather than genuine generalization.

- **Missing reproducibility details**: The paper lacks critical hyperparameters including learning rate, optimizer choice (only "gradient descent" is mentioned), exact batch size, learning rate schedule, and data augmentation strategy. The statement "10 epochs with 256 iterations per epoch" describes iterations per epoch but not batch size. Without these details, the experiments cannot be reproduced.

- **No class imbalance handling**: The dataset exhibits significant class imbalance (e.g., Cyst=3709 vs Stone=1377 in the Cyst vs Stone task). The paper does not discuss any strategy to address this imbalance, such as weighted loss functions or oversampling, making it unclear how performance was maintained across imbalanced tasks.

- **No external validation or generalization testing**: Results are reported only on a single dataset from one hospital in Bangladesh. No experiments validate the models on data from different institutions, scanners, or populations. Claims of "clinical application" and "widespread clinical application" are unsubstantiated without demonstrating generalization.

- **Lack of interpretability analysis**: For medical diagnostic applications, understanding model predictions is critical. The paper provides no attention maps, GradCAM visualizations, or any analysis showing what features the models use to make predictions. This is essential for clinical trust and is standard practice in medical imaging venues.

- **No ablation studies**: The paper does not justify architectural choices or isolate the contribution of transfer learning. It is unclear whether fine-tuning provides benefit over frozen feature extraction, or what impact data augmentation would have.

## Nice-to-Haves
- A comparison with a unified multi-class model instead of four separate binary classifiers would be more clinically practical and allow for direct class comparison.
- Statistical significance testing for claimed "marginal" and "slight" advantages between models, with mean and standard deviation across multiple random seeds.
- Error analysis showing concrete examples of misclassified samples and discussion of failure modes.

## Novel Insights
This paper does not present novel methodological or theoretical insights. The approach—applying pre-trained ResNet-50 and EfficientNetV2 to medical image classification—is well-established in the literature. The contribution is an empirical comparative evaluation across four binary renal disease classification tasks. While such comparative studies have value, the claimed 100% accuracy undermines confidence in the results and prevents any meaningful insights about when one architecture might be preferred over another for renal CT classification.

## Potentially Missed Related Work
None identified.

## Suggestions
1. **Verify and report patient-level data splits**: Before any other changes, the authors should verify whether any patients appear in both training and test sets. If duplicates exist, re-run experiments with strict patient-level separation and report results transparently. This is the single most critical issue affecting result validity.

2. **Add interpretability analysis**: Include GradCAM or similar attention visualizations showing which regions the model focuses on, and validate whether these align with clinically relevant anatomical features rather than spurious artifacts.

3. **Report complete hyperparameters**: Specify learning rate, optimizer (Adam, SGD, etc.), batch size, and any learning rate scheduling. Include data augmentation strategies used during training.

---

## SOd07Qxkw4

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Reject (5.0/10)
- Match: No

### Final Review

## Summary
This paper establishes an improved convergence rate of O(d^(1/3) ε^(-2/3)) for score-based diffusion models under L-smooth score function assumptions, improving upon the prior best complexity of O(L^(5/3) d^(5/12) ε^(-1)). The key conceptual contribution is a new analysis framework that reduces the sampling problem to discretizing the probability flow ODE, combined with a randomized midpoint discretization method adapted from prior work.

## Strengths
- **Meaningful theoretical improvement**: The paper achieves a non-trivial dimension dependency improvement from d^(5/12) to d^(1/3), representing an O(d^(1/12)) factor improvement over Gupta et al. (2024) and Chen et al. (2024b). The comparison with non-smooth results (Benton et al., 2023; Li et al., 2024c) is carefully qualified with explicit conditions on L.
- **Elegant analysis framework**: Lemma 2's reduction of the sampling problem to ODE discretization with injected noise is a clean conceptual contribution that could be useful for analyzing other sampling algorithms beyond this specific application.
- **Well-structured presentation**: The paper clearly distinguishes assumptions, states main theorems with detailed implications, and organizes the proof into logical steps (introduce auxiliary sequence, control divergence, combine).
- **Competitive parallelization results**: Theorem 2 provides parallel sampling guarantees with O(log²(d/ε)) parallel rounds while requiring only O(d^(1/3) ε^(-2/3)) processors, improving upon prior work's processor requirements.

## Weaknesses
- **No empirical validation**: Despite claiming results "align more closely with practical requirements," the paper provides zero experimental validation. This is a significant gap even for a theory-heavy submission at ICLR. The paper motivates with practical observations (50 steps for CIFAR-10/ImageNet) but never tests whether the theoretical improvement translates to practical gains.
- **Algorithmic novelty is limited**: The randomized midpoint method is directly adapted from Shen & Lee (2019) and extended to diffusion models by Gupta et al. (2024). The main algorithmic contribution is incremental rather than fundamentally new.
- **Unverifiable proofs**: Critical lemmas (1, 2, 3, 5) reference detailed proofs in "Li & Jiao (2024)" which is not included in the submission. For a theoretical paper where correctness of the analysis is the core contribution, this makes independent verification impossible within the submission.
- **Conditional improvement over non-smooth results**: The O(d^(1/3) ε^(-2/3)) bound is only better than O(d/ε) when L < d^(2/3) ε^(-4/3) (for ε > d^(-1/2)), and the paper provides minimal guidance on whether this condition is satisfied by realistic neural network score estimators. The examples (L < 3381 for ImageNet at ε ≈ 1) don't clarify what L values neural networks actually achieve.

## Nice-to-Haves
- **Numerical demonstration of improvement**: Add a table comparing iteration requirements for CIFAR-10 (d=3072) and ImageNet 256×256 across ε ∈ {0.1, 0.01, 0.001} to show whether the asymptotic improvement is practically meaningful at realistic problem sizes.
- **Measurement of Lipschitz constants**: Estimate or bound the smoothness parameter L for real neural network score estimators to assess whether the theoretical conditions hold in practice.
- **Tightness analysis**: The paper does not discuss whether the O(d^(1/3) ε^(-2/3)) bound is tight. Lower bounds establishing optimality would strengthen the contribution significantly.
- **Parallelization discussion**: The parallel algorithm assumes GPU memory for batch score estimation, but additional discussion of multi-GPU communication overhead and practical implementation challenges would be valuable.

## Novel Insights
The paper's primary novel contribution is the analysis framework rather than the algorithm itself. By explicitly showing how injecting noise into the probability flow ODE reduces the sampling task to ODE discretization with controlled error (Lemma 2), the paper provides a clean conceptual tool that could be applied to analyze other sampling variants. The randomized midpoint discretization, while adapted from prior work, is applied in a new context where the ODE analysis reveals why randomization improves convergence—the key insight being that tracking error propagation across multiple steps simultaneously enables tighter bounds than stepwise analysis alone. This connection between the deterministic ODE structure and the benefit of randomized discretization is the paper's most interesting theoretical observation.

## Potentially Missed Related Work
- None identified (related work search was skipped by the reviewer).

## Suggestions
- **Include a proof sketch for Lemma 3 in the appendix**: Even if full proofs are deferred to a technical report, a sketch of the key steps for Lemma 3 would allow reviewers to verify correctness without external documents. This is particularly important since Lemma 3 is the core analytical innovation.
- **Add a single empirical validation**: Even one experiment on a simple synthetic distribution (e.g., Gaussian mixture) comparing the number of steps required for target TV error against theory would significantly strengthen the paper's claim of practical relevance.

---

## j4gzziSUr0

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes optimizing importance weights for sub-population shifts via bi-level optimization with implicit differentiation, arguing that standard likelihood ratio weights neglect finite-sample variance. The authors demonstrate consistent improvements across five existing methods (GW-ERM, SUBG, DFR, GDRO, JTT) on three benchmark datasets, with the key practical insight that benefits are largest when minority groups are small or total sample size is limited.

## Strengths
- **Sound theoretical foundation**: The bias-variance decomposition for linear regression (Section 4) provides genuine insight into why standard weights are suboptimal—Equation 9 explicitly characterizes how optimal weights trade off between bias (minimized at p_te) and variance (minimized at p_tr), with the optimal choice depending on sample size and dimensionality.
- **Comprehensive empirical validation**: Testing across 5 methods × 3 datasets with paired statistical tests provides strong evidence; 12/15 cases show statistically significant improvements at the 10% level, with gains up to 5.6%.
- **Practical value**: The finding that optimized weights are most beneficial for small training sets and small minority groups provides actionable guidance for practitioners facing data scarcity—a common real-world scenario.
- **Technical rigor**: The implicit function theorem derivation for hypergradients (Appendix A) is correctly applied, and the optimization procedure is clearly described and reproducible.

## Weaknesses
- **Unfair comparison methodology**: The same validation set is used for both hyperparameter tuning (L1 penalty, etc.) and weight optimization. Standard methods only access validation data for hyperparameter selection, while optimized weights directly fit to it. This asymmetry could explain part of the observed improvement. Nested cross-validation or separate validation sets would provide a fairer comparison.
- **Insufficient analysis of failure cases**: The -4.265% worst-group accuracy degradation for SUBG on CelebA and JTT failures are barely discussed. Understanding why optimization fails in these cases is critical for practitioners, and the paper should investigate whether adaptive procedures or constraints could prevent such degradation.
- **Limited theoretical scope**: The complete bias-variance analysis is restricted to linear regression with a specific DGP (intercept differences only). While the intuition likely generalizes, the paper provides no analysis of the neural network setting where the optimization is applied, leaving a gap between theory and practice.

## Nice-to-Haves
- **Visualization of learned weights**: Showing the optimized group weights across methods/datasets would reveal whether optimization produces intuitive results (e.g., less extreme upweighting of minorities than likelihood ratios) or surprising patterns.
- **Ablation on optimization hyperparameters**: The paper uses T=100-200 steps with fixed learning rates and momentum without justification; an ablation would clarify robustness.
- **Comparison to simpler alternatives**: A grid search over group weights on the validation set would provide a computationally cheaper baseline to compare against bi-level optimization.
- **Optimization trajectory analysis**: Plotting validation loss over optimization steps would reveal whether convergence is reliable and whether oscillations occur.

## Novel Insights
The paper's core insight reframes importance weighting from an unbiased estimation problem to a bias-variance trade-off: likelihood ratio weights minimize bias (asymptotically) but introduce variance by overweighting minority groups, especially problematic with finite samples. The analytical solution for linear regression (Equation 9) elegantly shows that optimal weights interpolate between training and test distribution proportions depending on sample size and feature dimension. Empirically, the surprising finding that optimizing for weighted average accuracy also improves worst-group accuracy (and vice versa) suggests these objectives are less opposed under properly calibrated weights than previously understood—this warrants deeper investigation.

## Potentially Missed Related Work
- Ren et al. (2018) "Learning to Reweight Examples": Directly optimizes sample weights via meta-learning with a validation set, sharing the goal of learning optimal importance weights—should be more prominently contrasted.

## Suggestions
The failure cases on CelebA (SUBG, JTT) require investigation. The paper should analyze whether these failures stem from: (1) validation set size being too small for reliable weight optimization, (2) non-convex optimization finding poor local minima, or (3) inherent unsuitability of weight optimization for certain method/dataset combinations. Adding constraints (e.g., maximum deviation from standard weights) or early stopping based on validation loss could prevent catastrophic degradation while preserving benefits in successful cases.

---

## nwDRD4AMoN

- GT: Accept (Oral) (avg 9.0)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces Artificial Kuramoto Oscillatory Neurons (AKOrN), a dynamical alternative to conventional neural network units that encodes information in the directions of N-dimensional unit vectors on a hypersphere. The Kuramoto update rule enables synchronization-based binding between neurons, providing a physics-grounded mechanism for competitive learning. The authors demonstrate AKOrN's effectiveness across unsupervised object discovery (competitive with slot-based models on natural images), adversarial robustness without adversarial training, calibrated uncertainty, and reasoning tasks like Sudoku solving.

## Strengths
- **Compelling empirical breadth**: The paper demonstrates consistent improvements across four distinct domains—object discovery, adversarial robustness, calibration, and reasoning—using the same underlying neuronal mechanism. This versatility is rare and valuable.
- **First synchrony-based model scaled to natural images**: As claimed, AKOrN is the first synchrony-based model (using oscillatory dynamics) that successfully scales to natural images without relying on pretrained feature extractors (Table 2: 52.0 MBO_i on PascalVOC, outperforming DINO at 47.2).
- **Strong theoretical grounding**: The connection to the well-established Kuramoto model from statistical physics provides a solid foundation, with Lyapunov stability proofs under symmetry assumptions and clear geometric interpretation of the dynamics.
- **Innovative energy-based reasoning**: The observation that the Kuramoto energy correlates with solution correctness, enabling energy-based voting to improve OOD performance from ~52% to ~89.5% on Sudoku, is a genuinely novel contribution that connects oscillatory dynamics to energy-based reasoning.
- **Comprehensive ablation studies**: The paper systematically ablates key components (projection operator, rotating dimensions N, symmetric constraints, Ω term) with clear performance implications documented across multiple appendices.
- **Well-motivated from neuroscience**: The paper effectively connects to biological phenomena (competitive learning, lateral connections, traveling waves) while maintaining mathematical rigor.

## Weaknesses
- **N parameter usage inconsistency undermines robustness claims**: The paper uses N=2 for robustness experiments (achieving 51-59% adversarial accuracy) but N=4 for all other experiments. Critically, Table 20 explicitly shows that AKOrN_mix with N=4 achieves 0% adversarial accuracy—the same architecture that performs well on object discovery and Sudoku. This dramatic sensitivity to N is neither explained in the main text nor presented as a limitation. The robustness claim appears to depend critically on choosing N=2, yet no guidance is provided on when to use which value.

- **Unfair comparison in robustness benchmarks**: The strongest robustness baselines (Bartoldson et al., Diffenderfer et al.) explicitly use adversarial training. AKOrN claims inherent robustness but achieves substantially lower clean accuracy (88.91-91.23%) compared to Bartoldson et al. (93.68%) and Diffenderfer et al. (96.56%). A direct comparison of AKOrN with and without adversarial training would clarify whether the robustness is truly "by design" or partly a clean accuracy trade-off.

- **Unvalidated energy function for attentive connectivity**: The paper acknowledges "it is unclear whether the energy defined in Eq. 3 is proper" for attentive connectivity, yet the entire energy-based reasoning story (Sudoku, calibration) depends on this energy being meaningful. No empirical validation of energy-solution quality correlation is provided for the attention variant beyond Sudoku.

- **Up-tiling fairness concern**: The up-tiling technique substantially boosts object discovery numbers but is only applied to AKOrN, not to baseline methods. Without ablation showing what up-tiling does for DINO or ItrSA, the claimed AKOrN advantage on object discovery may be partially attributable to the post-processing technique rather than the core oscillatory mechanism.

- **Limited reasoning evaluation**: Beyond Sudoku, no other reasoning or combinatorial optimization tasks are explored. The neuro-symbolic reasoning claims would be strengthened by experiments on additional tasks (graph coloring, scheduling, etc.).

## Nice-to-Haves
- **ImageNet classification accuracy not reported**: The paper trains AKOrN on ImageNet but never reports top-1 accuracy. Reporting linear probe or fine-tuning results would substantiate the claim of learning "better object-bound features" than DINO/MAE.
- **AKOrN + slot attention not explored**: SPOT (35.0/44.7 on COCO) combines DINO features with slot attention. Testing whether AKOrN features + slot attention improve further would clarify whether AKOrN complements or competes with slot-based approaches.
- **Visualization of oscillator dynamics on natural images**: Animated visualizations are promised but not shown in the paper. Direct visualization of how oscillator directions cluster during inference would strengthen the binding mechanism claims.

## Novel Insights
The paper's central insight—replacing static threshold activations with oscillatory dynamics where information is encoded in vector directions on a hypersphere—is genuinely novel and well-executed. The observation that synchronization dynamics naturally implement competitive learning and feature binding is both theoretically grounded and empirically validated. Particularly noteworthy is the demonstration that the Kuramoto energy function, derived from physics principles, correlates with solution correctness in reasoning tasks, enabling energy-based voting to dramatically improve out-of-distribution performance. This suggests an elegant bridge between oscillatory neural dynamics, energy-based models, and neuro-symbolic reasoning that merits further exploration.

## Potentially Missed Related Work
- **Neural Oscillator Networks (Aoyagi, 1995)**: Early work on oscillatory neural networks for phase retrieval that may provide additional theoretical grounding for the N-dimensional generalization.
- **WaveNet-style oscillatory models**: Recent work on oscillatory architectures for temporal processing may share similar motivations but different implementations.
- None identified that substantially change the contribution's novelty or significance.

## Suggestions
1. **Clarify N parameter guidance**: Provide explicit analysis of when to use N=2 vs N=4, and add a section explaining the trade-off between regularity (favored by smaller N) and representational capacity.
2. **Add adversarial training to AKOrN**: Compare AKOrN with and without adversarial training to determine if robustness is additive or just a clean accuracy trade-off.
3. **Ablate up-tiling on baselines**: Show what up-tiling does for DINO/ItrSA to establish a fair comparison, or remove up-tiling from the main comparison.
4. **Validate energy for attention on non-Sudoku tasks**: Demonstrate that energy correlates with correctness on other reasoning/optimization tasks using the attention-based variant.

---

## gc8QAQfXv6

- GT: Accept (Oral) (avg 9.0)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary

This paper investigates catastrophic forgetting in large language models during continual instruction tuning by leveraging function vectors (FV) from mechanistic interpretability as both an analytical tool and intervention mechanism. The authors demonstrate a strong correlation between FV dynamics and forgetting, concluding that forgetting stems from biases in function activation (shifts in P_M(θ|x)) rather than overwriting of task functions. Based on this insight, they propose FV-guided training with two regularization terms that, when combined with existing continual learning methods, substantially reduce forgetting while maintaining model plasticity.

## Strengths

- **Novel mechanistic insight**: The paper makes a valuable contribution by demonstrating that FV similarity is a superior predictor of forgetting compared to traditional metrics (last-layer feature similarity, parameter L2 distance), with strong empirical correlation (R² = 0.873 on Hellaswag).

- **Comprehensive empirical evaluation**: Experiments span 4 LLMs (Llama2-7b/13b-chat, Llama3-8b-chat, Mistral-7B-instruct), multiple task sequences from SuperNI and TRACE benchmarks, and ablation with 4 existing continual learning baselines (IncLora, EWC, O-Lora, InsCL).

- **Compelling intervention evidence**: The causal pathway analysis demonstrates that adding source FV or subtracting biased target FV during inference substantially restores forgotten capabilities, providing direct empirical support for the proposed mechanism.

- **Interesting empirical observations**: The finding that forgetting can naturally mitigate during training (performance drops then rebounds) challenges assumptions in prior work and provides practical insights.

## Weaknesses

- **Missing ablation on loss components**: The proposed method combines FV consistency loss (ℓ_FV) and KL divergence loss (ℓ_KL), but Table 2 provides no ablation separating their individual contributions. It remains unclear whether both components are necessary or if one dominates the improvements.

- **No statistical significance testing**: All reported improvements are point estimates without error bars, confidence intervals, or significance tests. Given the known variance in LLM evaluation, the magnitude of reported gains (e.g., +25.25 on IP metric) should be qualified with uncertainty estimates.

- **Unjustified design choices**: Several key hyperparameters lack justification: the top-10 selection for causal attention heads (why not top-5 or top-20?), layer selection for KL intervention (l=9), and the coarse grid search resulting in α₁=1, α₂=0.08 without sensitivity analysis.

- **Limited task diversity**: All experiments use NLP tasks from SuperNI and TRACE. The approach's effectiveness on reasoning-heavy, multilingual, or multimodal tasks remains unverified, limiting generalizability claims.

## Nice-to-Haves

- Systematic ablation on FV construction (top-k heads, random heads, different aggregation methods) to establish whether the specific FV methodology is necessary for the findings.
- Analysis of failure cases (e.g., the -2.8 FP delta on NI-Seq-M1 with InsCL+FVG) to understand when and why the method degrades training performance.
- Layer-by-layer analysis of FV intervention effects to justify layer selection and understand where in the model the mechanism operates.

## Novel Insights

The paper's most valuable contribution is demonstrating that function vectors provide a principled lens for understanding catastrophic forgetting in LLMs. The finding that forgetting correlates with FV similarity (Cosine(θ_T[0]_e, θ_T[j]_e)) but not with traditional metrics challenges the field's reliance on parameter-distance or hidden-state similarity approaches. The mechanistic insight that forgetting stems from biased activation of the mapping P_M(θ|x) rather than overwriting of P_M(y|x,θ) offers a new conceptual framework—previously learned task functions are not erased but overshadowed by newly introduced biased functions. This reframing has practical implications: rather than protecting parameters, continual learning methods should stabilize how inputs map to task-specific representations.

## Potentially Missed Related Work

- Task arithmetic/merging methods (e.g., TIES-Merging, DARE): These parameter-space approaches to addressing forgetting could provide alternative baselines or be analyzed through the FV lens proposed in this work.
- Extended causal mediation analysis literature: Pearl-style do-calculus analysis could provide stronger causal grounding for the claims in Section 5 beyond the empirical intervention experiments.

## Suggestions

The paper would benefit from adding an ablation study that isolates the contribution of each loss component (ℓ_FV vs ℓ_KL) to determine which mechanism drives the observed improvements, along with statistical significance testing to validate the reported performance gains.

---

## 5kMwiMnUip

- GT: Reject (avg 1.4)
- Predicted: Reject (1.0/10)
- Match: Yes

### Final Review

## Summary
This paper explores five jailbreaking methods (Multishot, Mirror Dimension, Cipher, "Wrong Question," and Textbook) against LLMs, claiming these techniques effectively bypass safety mechanisms and could serve as a benchmark against defenses like LlamaGuard. The paper provides only conceptual descriptions of the methods without reproducible experimental details, specific prompts, or clear evaluation criteria, rendering the quantitative claims unsubstantiated.

## Strengths
- **Timely and important topic**: LLM security and jailbreaking resistance is a critical area of research as these models are deployed in high-stakes applications
- **Attempted taxonomy**: The paper organizes diverse jailbreaking techniques into five conceptual categories, which could provide a useful framework for systematization discussions
- **Creative framing**: The Mirror Dimension (fictional framing) and Textbook (information synthesis) approaches represent novel conceptual framings for understanding jailbreak vulnerabilities, even if the techniques themselves are variations on existing attacks

## Weaknesses
- **No reproducible methodology**: The paper describes each method conceptually but provides no specific prompts used, exact models tested (only vague references to "GPT-3" and "various LLMs"), temperature/sampling parameters, or evaluation protocols. Without this, the work cannot be independently verified
- **Unsupported quantitative claims**: Success rates cited in the conclusion (e.g., "Multishot: 0.20 → 0.80") are not traceable to any legible evidence—the figures appear corrupted or contain unreadable data (Figure 2 shows garbled text like "me}(2) 3fe}"); the central experimental contribution is incomprehensible
- **Title-content mismatch**: The title invokes "Chain of Thoughts Approach" and the abstract mentions "chain-of-thought reasoning" and reinforcement learning enhancement, but neither chain-of-thought methodology nor RL implementation appears anywhere in the paper
- **No benchmarking results**: The abstract claims findings can "serve as a benchmark against LlamaGuard," yet no experiments with LlamaGuard or any defended system are presented anywhere in the paper
- **Unexplained methodology element**: The "Reference Method" appears in figures and results but is never defined or described in the methodology sections
- **Missing ethics discussion**: For a paper explicitly demonstrating how to bypass LLM safety measures, there is no mention of responsible disclosure, ethics review, or potential misuse mitigation—particularly concerning given the "Masterkey" reference (Deng et al., 2024) in related work discusses automated jailbreaking of commercial chatbots
- **No comparison to prior work**: The paper does not quantitatively compare against established baselines (e.g., GCG attacks, MasterKey, GPTFuzzer), making it impossible to assess whether these methods are novel or repackaged versions of existing attacks

## Nice-to-Haves
- Define "jailbreak success" operationally—what threshold distinguishes success from failure, and was it based on automated detection, human evaluation, or content classification?
- Include actual prompt-response examples demonstrating each technique with the model's refusal before jailbreaking and response after
- Provide ablation studies varying components of each attack (e.g., number of shots for Multishot, encoding schemes for Cipher method)
- Fix all figures to present interpretable data; a paper claiming quantitative results cannot have corrupted tables

## Novel Insights
While the paper's experimental contribution is fatally compromised by missing details and corrupted figures, the conceptual framing of some techniques offers modest value. The "Mirror Dimension" approach's insight—that fictional framing can exploit models' propensity for narrative generation—represents a reasonable hypothesis about how creative training data influences safety compliance. Similarly, the "Textbook" method's observation that information synthesis can circumvent safety by aggregating restricted content across safe sources identifies a real vulnerability pattern. However, these observations are presented without empirical validation, making it impossible to assess their validity or practical significance.

## Potentially Missed Related Work
None identified (search was not conducted)

## Suggestions
The paper requires substantial revision before it could meet publication standards. Specifically:
1. **Conduct and document actual experiments**: Provide all prompts used, model versions (GPT-3.5-turbo? GPT-4? LLaMA-2?), API parameters (temperature, top-k, etc.), and exact evaluation criteria for what constitutes a successful jailbreak
2. **Align title and abstract with content**: Remove claims about chain-of-thought reasoning and reinforcement learning unless they are actually implemented
3. **Add ethics statement**: Discuss responsible disclosure, which models were tested (public APIs vs. research models), and justify why the benefit to defenders outweighs risks
4. **Release evaluation code and prompts**: Without this, the claimed benchmarking contribution is impossible to verify or reuse

---

## HD6bWcj87Y

- GT: Accept (Oral) (avg 7.5)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces In-Run Data Shapley, a novel data attribution framework that computes Shapley values for each training iteration and accumulates them throughout a single training run, eliminating the need for retraining-based Data Shapley computation. The key technical contribution is a set of "ghost" techniques that compute gradient dot-products and gradient-Hessian-gradient products in one or two backpropagations respectively, achieving near-zero runtime overhead for first-order approximations. The paper demonstrates applications including data curation for the Pile dataset, analyzing how data contributions evolve across training stages, and providing insights on copyright attribution for generative AI.

## Strengths
- **Elegant theoretical framework**: The paper provides rigorous derivations of closed-form Shapley value formulas for both first-order (Theorem 3) and second-order (Theorem 4) Taylor approximations. The decomposition of cumulative contribution into per-iteration Shapley values via the linearity axiom is mathematically sound and provides a clear interpretation of the method as a "contribution accountant" tracking data value throughout training.

- **Technically novel ghost techniques**: The "ghost dot-product" technique is clever and well-motivated—exploiting the insight that per-sample gradient information is already available during backpropagation, allowing pairwise dot-products without instantiating individual gradient vectors. This directly addresses the computational bottleneck that made prior approaches (e.g., TracIN-Ideal) "computationally infeasible."

- **Significant efficiency improvement with empirical validation**: Table 1 demonstrates substantial speedups (76.2 → 70.5 tokens/sec for first-order, representing only ~8% overhead) and ~20× improvement over direct per-sample gradient computation. Figure 1 validates fidelity against Monte Carlo estimates with RMSE ~ 0.0003 and Spearman correlation > 0.94 for first-order approximation.

- **Valuable empirical insights**: The case studies provide genuinely novel observations, including the finding that ~16% of Pile data receives negative values despite the dataset's curation, the stage-dependent dynamics of domain contributions during training, and the observation that training data contributes significantly to semantically similar (but lexically distinct) outputs—relevant for copyright discussions.

- **Clear connection to prior work**: The paper explicitly relates first-order In-Run Data Shapley to TracIN-Ideal (Pruthi et al., 2020), providing new theoretical interpretation and demonstrating that the prior method's computational burden can be overcome.

## Weaknesses

- **Optimizer mismatch between theory and experiments**: The theoretical derivations are specifically for SGD, yet experiments use AdamW. Remark 7 explicitly states the ghost techniques do not extend to Adam. The paper acknowledges using "SGD as a proxy for Adam," but this means the theoretical claims about exact Shapley computation apply to a different optimizer than what is used for training. Since foundation model pretraining universally uses Adam/AdamW, this creates a gap between the principled framework and practical applicability that the paper does not adequately address.

- **Cumulative approximation error not characterized**: The paper validates Taylor approximation fidelity at a single iteration (iteration 3500) in Figure 1, showing high correlation. However, In-Run Data Shapley accumulates scores across thousands of iterations. The paper does not analyze whether per-iteration approximation errors compound in problematic ways over the full training trajectory, which is crucial for trusting the cumulative values used in data curation decisions.

- **Second-order method lacks clear justification**: Table 1 shows second-order In-Run Data Shapley is approximately 2× slower than first-order (34.4 vs 70.5 tokens/sec), yet Figure 8 shows the second-order approximation provides "no notable improvement" in utility approximation accuracy. The paper claims second-order captures "interaction terms" for duplicate detection, but this benefit is not quantified. Without clearer evidence that second-order provides essential functionality, including it adds complexity and runtime overhead without demonstrated benefit.

- **Foundation model scale claims are aspirational**: The paper claims applicability to "foundation model pretraining" and references models like LLaMA in the introduction, yet experiments use only GPT2-Small (124M) and Pythia-410M. While the smaller-scale experiments are reasonable for methodological papers, the gap between 410M and actual foundation models (7B+) is substantial, and the paper does not provide scaling projections or analysis to support the scalability claims.

## Nice-to-Haves
- Systematic batch size ablation to characterize the efficiency-memory tradeoff of ghost techniques across different hardware constraints
- Validation set robustness analysis showing how results change with different validation corpora choices
- Time-series visualization of cumulative attribution evolution across training iterations
- Theoretical bounds on cumulative approximation error as training progresses
- Additional copyright case studies with varied validation corpora beyond the three examples shown

## Novel Insights
The paper's most valuable conceptual contribution is the shift from algorithm-level to model-specific data attribution. By defining utility per iteration with the specific batch realization embedded, the method produces deterministic scores for a particular training trajectory rather than expected values across random initializations. This addresses a genuine limitation in prior Data Shapley work where "contribution" was averaged over retrainings with different random seeds. The observation that 16% of Pile data receives negative values—despite the dataset's multi-layer curation—suggests current data quality filtering misses systematic issues. The stage-dependent dynamics finding, where general corpora contribute early but domain-specific corpora dominate later, provides a nuanced view of pretraining that simple data filtering methods miss.

## Potentially Missed Related Work
- Choe et al. (2024) - concurrent work on "What is your data worth to GPT? LLM-scale data valuation with influence functions" using similar gradient decomposition techniques; relevant for comparison as it addresses similar scalability challenges
- Schioppa (2024) - gradient sketches for training data attribution studying loss landscape geometry; directly relevant to the gradient-based attribution approach

## Suggestions
1. **Clarify the optimizer usage**: Either (a) implement SGD as the actual training optimizer to match the theory, (b) provide empirical evidence that SGD-based attribution correlates well with AdamW training dynamics, or (c) more prominently acknowledge this as a limitation requiring future work on Adam-compatible ghost techniques.

2. **Add cumulative error analysis**: Track the correlation between cumulative In-Run Data Shapley values and ground-truth cumulative utility at multiple checkpoints (iteration 500, 1000, 5000) to validate that approximation errors do not compound undesirably.

3. **Justify or remove second-order method**: If first-order provides sufficient accuracy for all downstream tasks (data curation, domain detection, copyright analysis), remove second-order to simplify the paper and focus on the efficient variant. If second-order is essential for some applications, provide concrete examples demonstrating this.

---

## kkGIbmpCHU

- GT: Reject (avg 4.8)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
TABDAR proposes a diffusion-nested autoregressive model for heterogeneous tabular data generation that combines masked bidirectional transformers with conditional diffusion models to handle continuous columns while supporting arbitrary column generation orders. The method achieves state-of-the-art performance on 10 datasets across statistical fidelity, downstream utility, and privacy metrics, with reported improvements of 18-45% over prior methods.

## Strengths
- **Comprehensive empirical validation**: The paper evaluates on 10 diverse datasets (continuous-only, discrete-only, and heterogeneous), comparing against 10 baselines including VAEs, GANs, diffusion models, LLMs, and autoregressive methods. This breadth is commendable.

- **Sound methodology for the core insight**: The combination of masked bidirectional attention to simulate arbitrary causal ordering with per-column conditional diffusion for continuous features is technically principled and addresses both stated challenges.

- **Informative ablation studies**: Table 4 demonstrates that both the diffusion loss (most critical) and random ordering contribute to TABDAR's success, providing useful insights into the method's mechanisms.

- **Strong imputation capability**: The method outperforms specialized imputation methods (KNN, GRAPE, MOT, Remasker) on 3 of 4 datasets, demonstrating practical utility beyond unconditional generation.

- **Robustness to hyperparameters**: Tables 5-6 show the model performs well with fixed hyperparameters across diverse datasets, reducing engineering burden.

## Weaknesses
- **Incomplete privacy comparison**: Table 3 and Figure 5 only compare TABDAR with SMOTE, TabDDPM, and TabSyn on privacy metrics. Other baselines (DP-TBART, CTGAN, TVAE) are absent, leaving an incomplete picture of where TABDAR stands relative to the full baseline suite.

- **Missing imputation comparison**: The missing value imputation experiments (Figure 6) do not include other tabular generative models (TabDDPM, TabSyn, CTGAN) that could serve as competitive baselines. Only KNN, GRAPE, MOT, and Remasker are compared.

- **β-Recall underperformance unexplained**: TABDAR achieves 37.2% on β-Recall (coverage), the worst among deep generative methods, compared to 44.1% for TabSyn. The paper dismisses this by noting SMOTE performs best but does not analyze why TABDAR specifically struggles with coverage—a genuine limitation that deserves investigation.

- **Conditional generation not empirically validated**: While the paper claims flexible conditional sampling as a key advantage, only missing value imputation is evaluated. Other conditional generation scenarios (class-conditional, feature-constrained) are not demonstrated.

- **Presentation of improvement claims**: The abstract reports "18-45% improvement" but this aggregates across metrics and baselines. The ablation study shows removing random ordering only degrades performance by ~1.83% on Margin, creating a disconnect between headline claims and isolated contributions.

## Nice-to-Haves
- Per-dataset breakdown of improvements in the main paper to identify where TABDAR excels or struggles
- Ablation against alternative continuous distribution modeling approaches (normalizing flows, Gaussian mixtures)
- Statistical significance tests for the main comparison claims
- Analysis of generation order variance to validate that ordering doesn't meaningfully affect output quality
- 3D or higher-order joint distribution verification beyond the 2D visualizations

## Novel Insights
The paper's primary contribution is adapting and systematically evaluating the combination of masked bidirectional attention with conditional diffusion models for tabular data—a framework previously explored in image generation (Li et al., 2024a). The key insight that masked bi-directional attention can simulate arbitrary causal orderings through appropriate masking is well-demonstrated and enables the model to learn conditional distributions from any column given any subset of observed columns. The empirical validation for tabular data is thorough and demonstrates meaningful improvements in fidelity metrics, though the novelty is incremental rather than breakthrough-level.

## Potentially Missed Related Work
- **Li et al. (2024a)** - The paper does cite this, but direct comparison would strengthen the evaluation
- **Garbin et al.** - Conditional tabular generation methods that use normalizing flows for continuous features
- **Feng et al.** - Adaptive imputation methods that could serve as additional baselines

## Suggestions
1. Include at least one tabular generative model baseline in the missing value imputation comparison to contextualize performance.
2. Add statistical significance tests and confidence intervals to the main fidelity comparisons.
3. Provide per-dataset breakdown in the main paper to identify failure modes and validate broad applicability.
4. Either expand conditional generation experiments or calibrate claims about "flexible conditional sampling" to match demonstrated capabilities.

---

## P49gSPmrvN

- GT: Reject (avg 1.0)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary

This paper presents a method for visualizing the temporal development of scientific discourse by embedding arXiv abstracts (cs.CL, 1995-2024) into high-dimensional vectors using a language model, then projecting to 3D via UMAP with time on the Z-axis. The author claims this enables discovery of research trends (e.g., the emergence of deep learning around 2013-2014) without external ground truth, framing this capability as "inner-referential."

## Strengths

- **Practical exploratory visualization**: The interactive 3D visualization with time on the Z-axis provides an intuitive interface for researchers to explore their field's historical development, which has genuine utility for historiographical research.
- **Adequate dataset scale**: Using 55,667 arXiv cs.CL abstracts spanning 30 years provides sufficient temporal coverage to observe meaningful field evolution patterns.
- **Demonstrates proof of concept**: The paper successfully shows that interpretable spatial features (columns corresponding to explanation-based learning, deep fragment embeddings) can be visually identified and correlated with their publication periods.

## Weaknesses

- **No quantitative validation**: The paper provides only visual evidence and manual interpretation of patterns. No metrics are reported to establish that the observed features are meaningful rather than artifacts—neither the harsh critic nor the balanced reviewer found evidence of clustering quality measures, temporal coherence scores, or comparison to ground-truth topic annotations.

- **Missing baseline comparisons**: The paper does not compare UMAP + GIST embeddings against simpler alternatives (TF-IDF + PCA, LDA topic modeling, doc2vec). This leaves the fundamental question unanswered: does this approach reveal patterns that established methods would miss? The examples shown (explanation-based learning fading, Word2vec spike in 2013) are well-known NLP historiography, raising the question of what *new* insights this method provides.

- **Unjustified methodological choices**: The paper states UMAP uses "default parameters" without specifying values (n_neighbors, min_dist, metric, random seed), and relies on a single embedding model without justification. The spark finder correctly notes that UMAP results vary dramatically with these settings. No ablation studies or sensitivity analyses are provided.

- **Reproducibility concerns**: The paper references a URL for source code, but it is truncated/absent in the submission. Key methodological details (preprocessing steps, pooling strategy for variable-length abstracts, random seeds) are unspecified.

- **Philosophical framing undermines clarity without adding technical value**: The paper dedicates approximately 60% of the introduction to tangentially related philosophical discussions (Drucker's nonrepresentational approach, Bickhard's critique, archaeological "inner-referentiality"). This material does not motivate the technical contribution. Notably, the paper's own Section 4 reasoning undermines the "inner-referential" claim: "word vectors are not inner-referential, because they are defined through comparison with all the text that generated the model." This self-defeating argument suggests the philosophical framing should be removed entirely.

## Nice-to-Haves

- **Formal evaluation protocol**: Define a measurable task (e.g., predict emergence date of research topics, cluster articles by theme) with success metrics to demonstrate scientific utility beyond feasibility.
- **Validation against known history**: Explicitly test whether the method correctly recovers established NLP historiography before claiming it discovers new patterns.
- **Generalizability testing**: Apply the method to other arXiv categories (physics, mathematics) or external corpora to demonstrate robustness beyond cs.CL.

## Novel Insights

The paper demonstrates that a standard pipeline (document embeddings + UMAP dimensionality reduction + 3D visualization with time as an axis) can serve as a practical exploratory tool for bibliometric analysis. However, the examples provided—explanation-based learning fading after the 1990s and the 2013 Word2vec-related publication explosion—are established facts in NLP historiography. The paper does not identify any insights that were previously unknown, raising the question of whether the method adds discovery capability beyond confirming known history. The genuine contribution lies in demonstrating feasibility and providing an interactive visualization tool, but this is better characterized as an application paper requiring stronger evaluation than is currently provided.

## Potentially Missed Related Work

- Existing scientometric visualization tools (VOSviewer, Sci2, Dimension) and their use of embeddings for research landscape mapping—positioning against these would clarify what the paper's approach improves upon.
- Prior work on using word embeddings and dimensionality reduction specifically for tracking temporal evolution of scientific fields.

## Suggestions

Replace the philosophical introduction with a focused discussion that: (1) clearly states the contribution as a visualization tool for temporal discourse analysis, (2) positions the work against existing bibliometric tools, and (3) outlines a concrete evaluation plan (e.g., user study with domain experts, quantitative comparison to baselines). Remove or drastically condense the "inner-referentiality" discussion since it is self-undermining and does not serve the technical contribution.

---

## m29SV0n6DO

- GT: Reject (avg 4.2)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
Toto presents a large-scale empirical study on generative pre-training from videos using autoregressive next-token prediction on discretized visual tokens, pre-training LLaMA-style transformers (up to 1B parameters) on over 1 trillion tokens. The paper evaluates representations across diverse downstream tasks and studies scaling behaviors, finding that visual next-token prediction follows a power law similar to language models.

## Strengths
- **Impressive empirical scale**: Training on 1 trillion visual tokens across 100,000+ hours of video with models up to 1B parameters provides valuable insights into scaling visual autoregressive models.
- **Comprehensive ablation studies**: Systematic investigation of tokenizer choices (dVAE vs VQGAN vs patch), pooling methods (attention vs average pooling showing +7.9% gain), and resolution adaptation via RoPE fine-tuning.
- **Valuable architectural insights**: Attention pooling substantially outperforms average pooling for autoregressive models; middle layers produce best representations; low-resolution pre-training + high-resolution finetuning via RoPE outperforms direct high-resolution training.
- **Diverse downstream evaluation**: Image classification (ImageNet), action recognition (K400), action forecasting (Ego4D), video tracking (DAVIS), object permanence (CATER), and robotics manipulation—providing holistic view of representation quality.
- **Scaling law analysis**: Empirical evidence that visual next-token prediction follows power laws similar to language models, providing useful guidance for future research.

## Weaknesses
- **Limited novelty**: The paper explicitly states "This paper does not describe a novel method"—the approach directly applies GPT-style next-token prediction to visual tokens. Architectural choices (RoPE, SwiGLU, RMSNorm) are borrowed from LLaMA without visual-specific innovations. AIM (El-Nouby et al., 2024) already demonstrated similar autoregressive visual pre-training at scale.
- **Significant performance gap**: Despite 1 trillion tokens and 1B parameters, Toto-1b achieves 75.3% on ImageNet—below MAE (80.9%), AIM-3B (82.2%), and substantially below DINOv2 (86.4%). On K400, Toto-1b (74.4%) trails VideoMAE (79.8%) and MVD (79.4%). The paper's positioning as "competitive" within generative approaches is accurate but masks that masked approaches outperform.
- **No video generation quality evaluation**: As a "generative pre-training" paper, zero generated outputs (images or videos) are shown. No FID, FVD, or qualitative samples are provided, making the "generative" claim difficult to assess.
- **Robot experiments severely underpowered**: Only 16 trials on a single task (cube picking) without confidence intervals or statistical analysis—insufficient for the claim of performing comparably to state-of-the-art robotics vision backbones.
- **No ablation isolating video vs image contribution**: Trains on mixture but never analyzes what videos add beyond images—a fundamental gap for a paper claiming to study "generative pre-training from videos."
- **Mamba architecture comparison unexplained**: Mamba achieves 40.7% vs Toto's 53.2% with similar parameter count, but the paper provides no analysis of why state-space models perform poorly on this task.
- **Incomplete scaling gap analysis**: Paper finds power law exponent of -0.0386 vs GPT-3's -0.048 but offers no explanation—is this due to tokenizer bottleneck, temporal redundancy, or the objective itself?

## Nice-to-Haves
- Video generation samples and metrics (FID, FVD) to demonstrate generative capabilities
- Deeper analysis of attention patterns across frames to validate temporal reasoning claims
- More rigorous robotics evaluation using established benchmarks (LIBERO, MetaWorld) with standardized protocols
- Systematic ablation of dataset contributions (video-only vs image-only vs mixture)
- Analysis of what semantic information the discrete tokenizer vocabulary captures

## Novel Insights
The paper's most valuable contribution is demonstrating that autoregressive visual pre-training scales predictably with compute following a power law, similar to language models, though with higher loss. The finding that attention pooling substantially outperforms average pooling (+7.9%) in autoregressive visual models is important for downstream transfer. The resolution finetuning trick—pre-training at 128×128 and finetuning to higher resolution via RoPE adaptation—offers a practical approach to reduce pre-training compute. However, the core insight that masked autoencoders substantially outperform generative approaches (80.9% vs 75.3% on ImageNet) suggests that for recognition tasks, the "just model the data distribution" objective is less effective than discriminative or reconstruction-based objectives, raising questions about the practical value of generative pre-training for vision unless generation is explicitly needed.

## Potentially Missed Related Work
- **AIM (El-Nouby et al., 2024)**: Most directly comparable prior work on autoregressive visual pretraining. Toto-1b (75.3%) significantly underperforms AIM-3B (82.2%), yet the paper claims "highest accuracy among autoregressive models" without adequately positioning against this concurrent work.
- **Video diffusion models (W.A.L.T, CogVideoX)**: Relevant comparisons for the generative video modeling claim, though these are concurrent works.
- **MagViT / Tokenizer improvements**: Recent work on learning visual tokenizers may offer improvements over the Dall-E dVAE used.

## Suggestions
1. Either strengthen the technical contribution with visual-specific innovations (visual positional encodings, temporal modeling mechanisms) or reframe as a thorough benchmark paper with clearer scope.
2. Add video generation quality evaluation (FID, samples) to justify the "generative" framing.
3. Increase robot experiment rigor with more trials, confidence intervals, and multiple tasks.
4. Systematically ablate video vs image contributions to justify the video pre-training focus.
5. Address the performance gap with masked approaches or provide compelling evidence that generative representations offer unique benefits (e.g., for generation, reasoning) that justify the accuracy trade-off.

---

## EO8xpnW7aX

- GT: Accept (Oral) (avg 8.0)
- Predicted: Accept (8.0/10)
- Match: Yes

### Final Review

## Summary
SymmetricDiffusers introduces a discrete diffusion model for learning probability distributions over finite symmetric groups (permutations), leveraging riffle shuffles as the forward process due to their favorable mixing properties, and proposing a Generalized Plackett-Luce (GPL) distribution for the reverse process that is proven more expressive than standard PL. The method achieves competitive or state-of-the-art results on sorting tasks, jigsaw puzzles, and traveling salesman problems, particularly excelling on longer sequences.

## Strengths
- **Strong theoretical foundation**: The paper effectively bridges classical results from random walks on finite groups (cut-off phenomena, mixing times) with modern diffusion methodology. The riffle shuffle selection is rigorously justified by its O(log n) cut-off time versus O(n² log n) for alternatives (Section 3.1.2).
- **Meaningful theoretical contributions**: Theorem 2 proves the GPL distribution can represent any distribution over S_n (with -∞ scores), while Proposition 1 shows standard PL cannot represent delta distributions. Proposition 3 provides exact TV distance formulas for riffle shuffles.
- **Comprehensive experiments**: The paper tests on diverse tasks (sorting, jigsaw puzzles, TSP) with varying sequence lengths, providing thorough empirical validation and informative ablation studies (Tables 3, 6, 7).
- **Practical denoising schedule**: The TV distance-based heuristic (threshold ~0.005 for mixing, ~0.3 for merging) is well-motivated by theory and validated empirically. Table 7 demonstrates the dramatic impact of schedule choice.
- **Code and reproducibility**: GitHub repository provided with complete training details.

## Weaknesses
- **Unacknowledged theoretical limitation**: Theorem 2's constructive proof requires -∞ scores to represent arbitrary distributions (Lemma 4), which neural networks cannot produce. This is a significant gap between the theoretical claim and practical implementation. The paper does not acknowledge this limitation in Section H or discuss how well GPL approximates distributions with bounded parameters.
- **Missing statistical uncertainty**: Table 2 reports "averaged over 5 runs" without error bars or standard deviations. Given that accuracy varies dramatically across sequence lengths (e.g., 98.0% at n=3 to 0.1% at n=200), error bars are essential for calibrated comparisons.
- **Incomplete ablation of reverse distributions**: For n=200, the paper switches to PL (claiming efficiency) but never compares GPL vs. PL at this scale. This leaves unclear whether GPL's O(n²) parameters cause training instability or memory issues.
- **TSP results presentation**: The 0.18% optimality gap on TSP-20 is genuinely competitive, but the paper could more clearly distinguish when the method matches OR solver quality versus when gaps remain substantial.

## Nice-to-Haves
- Training convergence curves showing loss/ELBO improvement over training steps would validate that the model actually learns target distributions.
- Qualitative analysis of failure modes (e.g., for n=200 sorting) would help diagnose whether the method fails gracefully or produces degenerate outputs.
- Visualization of diffusion trajectories showing how permutation distributions evolve through reverse steps.
- Grid search over TV distance thresholds (e.g., 0.25 vs. 0.35) to validate that the 0.3 threshold is principled rather than arbitrary.
- Inference-time computation comparisons (wall-clock time) to substantiate efficiency claims over alternatives.

## Novel Insights
This paper makes a valuable contribution by connecting two previously separate research areas: classical random walks on finite groups (from probability theory) and modern diffusion models. The key insight—that card shuffling methods provide a natural forward diffusion process for permutation spaces, with riffle shuffles offering the best trade-off between support size and mixing speed—is both theoretically grounded and practically useful. The GPL distribution extends Plackett-Luce with n² parameters, enabling richer expressiveness while maintaining sampling tractability. The denoising schedule based on TV distance provides a principled way to merge reverse steps, reducing computational cost while maintaining performance.

## Potentially Missed Related Work
- **MaskGIT (Chang et al., 2022)** and related parallel decoding methods for autoregressive generation—relevant for the decoding strategies discussed in Section 3.5
- **DIFUSCO (Sun & Yang, 2023)** is actually cited and compared for TSP, but the paper does not discuss how its graph-based discrete diffusion differs from the symmetric group approach
- **Recent continuous-time discrete diffusion models** (Campbell et al., 2022; Sun et al., 2023; Shi et al., 2024) are briefly mentioned in related work but could be discussed more thoroughly regarding adapting them to permutation spaces

## Suggestions
1. **Acknowledge the -∞ score limitation**: Clearly state that Theorem 2 requires -∞ parameters that neural networks cannot produce, and discuss approximation quality with bounded outputs.
2. **Add error bars**: Report standard deviations for key metrics to enable calibrated comparisons.
3. **Ablate GPL vs. PL at scale**: Run GPL on n=200 (or PL on smaller scales) to isolate whether the limitation is expressiveness or training stability.
4. **Clarify SOTA claims**: Frame results as "comparable for n≤32, significantly outperforms for n≥52" to avoid overstating generalization.

---

## P7f55HQtV8

- GT: Accept (Poster) (avg 6.5)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
QuaDiM introduces a non-autoregressive conditional diffusion model for quantum state property estimation (QPE), addressing the limitation that auto-regressive approaches impose unnecessary ordering biases on quantum systems where qubits interact non-sequentially. The method progressively denoises Gaussian noise into measurement distributions conditioned on Hamiltonian parameters. Evaluations on 1D anti-ferromagnetic Heisenberg models up to 100 qubits demonstrate superior performance over auto-regressive baselines in limited-data regimes, particularly for predicting correlation functions and entanglement entropy.

## Strengths
- **Compelling problem motivation**: The paper correctly identifies that auto-regressive models introduce biases by assuming sequential qubit ordering that has no physical basis in quantum entanglement. The experimental evidence (Table 5 in Appendix F.1) confirms that different orderings affect auto-regressive model performance, while QuaDiM remains order-invariant.
- **Technically sound implementation**: The adaptation of Diffusion-LM to discrete quantum measurements via learnable token embeddings and rounding steps is well-executed. The mathematical derivations in Appendices A and B are thorough and correctly apply standard diffusion theory to this domain.
- **Extensive empirical evaluation**: The paper tests across multiple system sizes (L=10, 40, 70, 100), multiple properties (correlation, entanglement entropy), multiple baselines (Classical Shadow, RBFK, NTK, RNN, LLM4QPE), and includes ablation studies on positional embeddings, denoising steps, hidden dimensions, and different POVMs.
- **Demonstrated scalability**: Results on 100-qubit systems show QuaDiM maintains strong performance (RMSE ~0.01 for correlation at Mout=20000) while RNN baselines degrade significantly (RMSE ~0.08), validating the scalability claim.
- **Sample efficiency advantages**: Particularly at low Mout values, QuaDiM shows meaningful improvements—e.g., at L=100, Mout=100: QuaDiM (0.1686) vs LLM4QPE (0.1759), demonstrating practical value for resource-constrained settings.

## Weaknesses
- **Positional embedding claim inconsistency**: The paper states qubits are treated "equally without predefined order," yet the architecture uses trainable positional embeddings, and Section F.5 shows removing them causes catastrophic degradation (RMSE increases from ~0.0125 to ~0.29 at Mout=10000). While the claim may be intended to mean "no sequential dependency during generation," the framing is misleading and should be clarified or revised.
- **Missing computational complexity analysis**: No training time, inference time, or memory usage comparisons are provided. Table 8 shows inference speed varies dramatically with denoising steps (5.7 to 37.4 samples/sec), but baseline inference speeds are not reported, making it difficult to assess practical deployment feasibility.
- **OOD generalization gap is underexplored**: Table 7 shows RMSE approximately quadruples under out-of-distribution conditions (0.0117 → 0.0417 for L=70). The paper claims generalization capability but provides minimal analysis of when and why it breaks down.
- **Limited physical domain**: Experiments focus exclusively on 1D Heisenberg models. The XY model results in Appendix F.2 are only for L=10. Practical quantum computing applications involve 2D lattices and more complex entanglement structures where the non-sequential modeling advantage would be more critical to validate.

## Nice-to-Haves
- Statistical significance testing (paired Wilcoxon tests) to substantiate "outperforms" claims beyond point estimates
- Comparison with other non-autoregressive architectures (e.g., 1D-CNN) to isolate whether diffusion specifically or any non-autoregressive approach provides the benefit
- Visualization of correlation structures at larger system sizes (L=40, 70, 100) to verify spatial pattern capture
- Deeper analysis of why diffusion helps—what aspects of quantum state distributions does it capture that autoregressive models miss?
- Code and model checkpoints for reproducibility

## Novel Insights
The paper makes a conceptually valuable observation that quantum entanglement lacks intrinsic ordering, making the sequential dependency assumption of auto-regressive models physically unmotivated. The non-autoregressive diffusion approach treats the quantum state distribution as a permutation-invariant object to be modeled, which aligns with the symmetric nature of quantum correlations. The empirical finding that QuaDiM scales better than RNN-based approaches to larger systems (L=100) suggests that avoiding sequential dependencies helps the model capture long-range correlations more effectively. However, the critical reliance on positional embeddings indicates that some notion of spatial structure remains important—future work should clarify whether this reflects a fundamental limitation or an implementation choice.

## Potentially Missed Related Work
- Diffusion-based approaches for quantum many-body problems (beyond text generation): Potential connections to quantum Monte Carlo or neural quantum states literature that also explore generative modeling of quantum distributions
- Normalizing flows for quantum state tomography: Could provide a non-autoregressive baseline for direct comparison
- Consistency models or other distillation methods for accelerated diffusion sampling: Relevant to the acknowledged sampling efficiency limitation

## Suggestions
1. **Clarify the positional embedding claim**: Either justify why positional information doesn't contradict "equal treatment" (e.g., positional encoding captures physical locality, not ordering bias) or revise the claim to be more precise about what "equal" means.
2. **Add computational footprint analysis**: Report training time and peak memory usage for QuaDiM vs. baselines at each system size to enable practical deployment assessments.
3. **Extend to 2D systems**: The paper's value proposition is strongest when qubit interactions are genuinely non-sequential, which is more characteristic of 2D lattices than 1D chains where neighboring interactions dominate.

---

## U3PBITXNG6

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
INVERSEBENCH introduces a comprehensive benchmarking framework for evaluating plug-and-play diffusion prior (PnPDP) methods across five scientific inverse problems (optical tomography, MRI, black hole imaging, seismology, fluid dynamics), systematically comparing 14 algorithms against domain-specific baselines. The work addresses a critical gap—existing benchmarks focus exclusively on natural image restoration, while scientific inverse problems involve physics-based forward models with fundamentally different characteristics. The paper provides valuable empirical insights into when PnPDP methods succeed or fail in scientific contexts.

## Strengths
- **Comprehensive scope with clear organization**: 14 algorithms evaluated across 5 diverse scientific domains with distinct forward model properties (linear/nonlinear, with/without closed-form solutions, gradient access), providing unprecedented systematic coverage of PnPDP methods on scientific inverse problems
- **Valuable empirical insights**: Key findings include identifying that methods requiring more gradient queries (DAPS, PnP-DM) are more accurate but computationally expensive and numerically unstable on PDE-based problems; that PnPDP methods struggle with out-of-distribution sources by biasing toward the prior; and that these methods are more robust than end-to-end approaches when forward models change
- **Excellent reproducibility**: Open-source codebase, pre-trained models, and datasets provided; extensive hyperparameter documentation (Table 12) and clear problem formulations enable independent verification
- **Thoughtful problem characterization**: Tables 1-2 systematically characterize algorithmic requirements and problem properties, enabling clear understanding of when different methods apply and their limitations
- **Practical multi-faceted evaluation**: Combined use of accuracy metrics (PSNR, SSIM), data consistency metrics (χ² statistics), and efficiency metrics (forward model/DM queries) with ranking scores

## Weaknesses
- **Small test sets limit statistical confidence**: FWI and Navier-Stokes use only 10 test cases each with 1 validation sample. Table 8 shows EnKG ranging from 0.120 to 2.203 relative L2 error, indicating high variance that makes ranking differences between methods (e.g., 0.201 vs 0.250) unreliable without statistical significance testing
- **Test set contamination for black hole imaging**: The paper generates validation and test images using the same diffusion model later used as the prior. This creates circular evaluation—the prior has implicit knowledge of the test distribution, potentially inflating performance estimates
- **Limited end-to-end learned baselines**: E2E-VarNet and Residual UNet are compared only for MRI. For FWI, Navier-Stokes, and black hole imaging, no learned end-to-end baselines are included. Neural operators or learned baselines trained on the same data as diffusion priors would strengthen comparative conclusions about PnPDP's value
- **Hyperparameter tuning asymmetry**: Fast problems receive 50-100 Bayesian optimization iterations while computationally intensive problems (FWI, Navier-Stokes) receive only 10-30 iterations, likely resulting in under-tuned PnPDP methods for these tasks and confounding cross-problem comparisons

## Nice-to-Haves
- Ablation studies varying diffusion prior quality (training set size: 1K, 10K, 50K) to substantiate claims about prior strength
- Systematic PDE stability experiments quantifying failure rates across step size schedules for FWI and Navier-Stokes
- Statistical significance testing (paired t-tests or confidence intervals) for method rankings
- Per-method failure case studies showing whether failures manifest as mode-collapse, artifacts, or mode-seeking
- Real Event Horizon Telescope observations for black hole imaging to assess real-world applicability beyond synthetic GRMHD data

## Novel Insights
The paper makes its most significant contribution by systematically documenting failure modes of current PnPDP methods that are invisible in natural image benchmarks. Most notably, it identifies that PDE-based forward models (FWI, Navier-Stokes) require stability constraints (e.g., CFL conditions) that many PnPDP algorithms violate, particularly methods incorporating Langevin Monte Carlo which introduces additional noise. This concrete failure mode—where "minor adjustments in step size can lead to either unconditional generation results that ignore measurements or complete failure"—points to a fundamental gap between algorithm design for image restoration versus physics-based scientific inverse problems. Additionally, the finding that PnPDP methods trained on scientific data significantly outperform the same methods using natural image priors (e.g., FFHQ faces on inverse scattering) provides empirical evidence that domain-specific priors are necessary for scientific applications, not merely beneficial.

## Potentially Missed Related Work
- None identified (related work search was skipped)

## Suggestions
- Regenerate black hole imaging test data from a different diffusion model or physical simulation to eliminate circular evaluation, ensuring test distributions are independent of the prior
- Expand test sets for FWI and Navier-Stokes to at least 50-100 samples and increase Bayesian optimization budget for slow problems to match that of fast problems
- Add at least one learned end-to-end baseline (e.g., neural operator) for FWI and Navier-Stokes to enable fair comparison against the full spectrum of learning-based approaches practitioners would consider

---

## DJSZGGZYVi

- GT: Accept (Oral) (avg 9.0)
- Predicted: Accept (8.0/10)
- Match: Yes

### Final Review

## Summary
The paper introduces REPA (REPresentation Alignment), a simple regularization technique that aligns noisy diffusion transformer hidden states with clean image representations from pretrained self-supervised encoders (e.g., DINOv2). The method achieves 17.5× faster convergence for SiT training (matching 7M-iteration FID at 400K iterations) and reaches FID=1.42 on ImageNet 256×256. The core insight is that learning high-quality representations is a key bottleneck in diffusion transformer training, and guiding this process with external representations substantially eases optimization.

## Strengths
- **Compelling empirical results**: The 17.5× training speedup is well-demonstrated across model sizes (B/2, L/2, XL/2), with Table 3 showing SiT-XL/2+REPA at 400K iterations (FID=7.9) matching or exceeding vanilla SiT-XL/2 at 7M iterations (FID=8.3). Final FID of 1.80 (without guidance interval tricks) represents genuine improvement over the 2.06 baseline.

- **Comprehensive ablation studies**: The paper systematically explores target encoder selection (DINOv2, CLIP, SigLIP, MAE, MoCov3, I-JEPA), alignment depth (layers 6-16), and alignment objectives (cosine similarity vs NT-Xent), providing actionable design guidelines. The finding that aligning only early layers (layer 8) is optimal is both practically useful and non-obvious.

- **Clear empirical motivation**: Figure 2 provides strong evidence for the representation gap hypothesis, showing diffusion transformers have significantly weaker representations than DINOv2 despite 7M training iterations, and that alignment remains weak even when semantic quality improves.

- **Reproducibility and extensibility**: Code and checkpoints are released; the method is simple (single regularization term added to existing training objective) and architecture-agnostic enough to be widely applicable.

## Weaknesses
- **Compound improvement in SOTA claim**: The headline FID=1.42 result combines REPA with guidance interval scheduling (Kynkäänniemi et al., 2024). The improvement attributable to REPA alone is the gain from 2.06→1.80 (basic CFG), a 0.26 improvement, not the full 0.64 point gain. The paper acknowledges this with an asterisk but the attribution in the abstract ("state-of-the-art results of FID=1.42") conflates two separate contributions.

- **No wall-clock time analysis**: The 17.5× speedup claim counts training iterations, not actual compute time. REPA requires forward passes through a frozen pretrained encoder for every training sample, adding computational overhead. The paper mentions pre-computation as a speedup but provides no wall-clock comparisons. A method requiring 2× compute per iteration but achieving 17.5× fewer iterations is not clearly a win.

- **Limited architectural diversity**: All experiments are on DiT/SiT architectures (pure diffusion transformers). The paper does not test whether REPA helps U-Net-based diffusion models or hybrid architectures, limiting the generality of the claim that "training diffusion models is easier than you think."

- **Single-run results without variance**: All FID values are reported from single training runs. Given training stochasticity, the reliability of the efficiency claims would be strengthened by reporting mean ± std across multiple seeds.

- **Missing comparison with related distillation methods**: DreamTeacher (Li et al., 2023b) is mentioned in related work as a similar distillation approach but no experimental comparison is provided. This is the most directly related prior work.

## Nice-to-Haves
- **Theoretical analysis**: The paper lacks formal justification for why representation alignment improves generation. A connection to contrastive learning theory or denoising autoencoder analysis (Vincent, 2011) would strengthen the contribution.
- **Time-conditional alignment weight**: The regularization coefficient λ is constant across timesteps. Intuition suggests alignment to clean representations may be more beneficial at certain noise levels; exploring time-varying λ could yield further improvements.
- **Analysis of layer 8 choice**: The paper shows empirically that aligning layer 8 works best but doesn't explain mechanistically why. Understanding this could inform future architectural or training designs.

## Novel Insights
The paper's most valuable contribution is identifying and empirically validating that **representation learning is a key bottleneck in diffusion transformer training**, not just the denoising objective. The insight that aligning early transformer layers to strong self-supervised representations (while leaving later layers free to focus on high-frequency details) provides a simple yet effective regularization is both intellectually interesting and practically impactful. The correlation between encoder quality and generation performance (stronger encoders → better FID) across diverse targets (DINOv2, CLIP, SigLIP, MAE) provides strong evidence that this is a general phenomenon, not an artifact of a particular encoder. This opens a promising research direction for unifying discriminative and generative representation learning.

## Potentially Missed Related Work
- **DreamTeacher (Li et al., 2023b)**: "Pretraining image backbones with deep generative models" — also distills self-supervised representations into diffusion models but uses an auxiliary encoder architecture rather than direct alignment loss; direct comparison would strengthen claims.
- **RCG (Li et al., 2024)**: Mentioned in related work as using pretrained representations for diffusion, but no experimental comparison is provided.

## Suggestions
- Report wall-clock training time including encoder overhead, or provide a theoretical analysis showing pre-computation makes overhead negligible.
- Add error bars or multi-seed experiments to quantify variance in FID improvements.
- Include DreamTeacher as a baseline comparison in future versions.

---

## X8dzvdkQwO

- GT: Accept (Poster) (avg 6.2)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
The paper introduces Fine-tuned Score Deviation (FSD), a method that improves pretraining data detection by fine-tuning LLMs on a small set of non-member examples and measuring the resulting score deviation. The key insight is that fine-tuning preferentially decreases perplexity for non-members compared to members, enlarging the gap between them. Experiments across multiple datasets and models demonstrate substantial AUC improvements (e.g., 0.62→0.91 on WikiMIA for OPT-6.7B).

## Strengths
- **Novel contribution**: First work to leverage unseen (non-member) data for improving pretraining data detection, presenting a fresh perspective on membership inference
- **Strong empirical results**: Demonstrates consistent AUC improvements across diverse benchmarks (WikiMIA, ArXivTection, BookMIA, BookTection, Pile), multiple model families (LLaMA-7B/13B/30B, OPT-6.7B, GPT-J-6B, Pythia-6.9B, NeoX-20B), and multiple scoring functions (Perplexity, Min-k%, Zlib, Lowercase)
- **Practical data efficiency**: Achieves significant gains with as few as 100 non-member examples, making it feasible for real-world deployment
- **Comprehensive ablation studies**: Explores fine-tuning data size, different fine-tuning methods (LoRA, AdaLoRA, IA3), model sizes, and domain generalization
- **Clear methodology**: The FSD score formulation (S(x; f_θ) - S(x; f_θ')) is straightforward and well-explained with visualizations showing score distribution separation

## Weaknesses
- **Temporal shift confound in WikiMIA**: The primary benchmark constructs non-members from post-2023 Wikipedia events while members are pre-2023 events, creating distributional confounds. The paper's own experiments in Table 6 show that baseline performance drops to near-random (AUC ~0.54) when temporal cues are removed via word deletion/replacement. The dramatic improvements on WikiMIA may partly exploit temporal patterns rather than pure memorization signals. While the paper acknowledges this, it is underemphasized given that WikiMIA is the primary benchmark used throughout.
- **Missing statistical uncertainty**: No error bars or variance estimates are reported across runs despite numerous experiments. This is particularly important for the Pile dataset experiments (20 subsets) where some improvements are modest (e.g., CC: 0.541→0.546). ICLR standards typically expect uncertainty quantification.
- **Limited theoretical justification**: The paper lacks analysis of *why* fine-tuning creates differential effects on members vs. non-members. The empirical observation in Figure 2 is well-documented, but no mechanistic explanation (e.g., gradient alignment, Fisher information, loss surface geometry) is provided.
- **Cross-domain limitation**: Table 16 demonstrates that FSD fails when fine-tuning on WikiMIA non-members and evaluating on ArXivTection (Perplexity AUC remains 0.68). The method requires domain-matched unseen data, limiting practical applicability to settings where such data exists.
- **White-box access requirement**: The method requires fine-tuning the target LLM, which is only feasible for open-source models. The paper claims applicability to commercial APIs like GPT-4o but provides no experiments validating this claim.
- **Baseline performance concerns**: Reported baseline AUCs (~0.58-0.68) on WikiMIA are close to random guessing, raising questions about whether the benchmark properly isolates memorization from distributional patterns rather than demonstrating the method's power.

## Nice-to-Haves
- Include confidence intervals for all reported metrics to quantify uncertainty
- Add experiments on closed-source or API-accessible models to support practical applicability claims
- Provide a theoretical analysis of the asymmetric fine-tuning effect (why non-member scores decrease more)
- Show calibration analysis (precision-recall curves) and discuss TPR at very low FPR thresholds (<1%), which matters for privacy auditing
- Include qualitative examples of correctly vs. incorrectly classified samples to illustrate what drives decisions

## Novel Insights
The paper's core contribution is identifying and exploiting an asymmetry in how LLMs respond to fine-tuning: non-member perplexity decreases substantially after exposure to unseen data, while member perplexity remains relatively stable. This observation—visualized in Figure 2—reveals that memorized content resists update more than non-memorized content, which is itself a meaningful insight about LLM learning dynamics. The method elegantly converts this asymmetry into a practical detection signal by measuring score deviation rather than absolute scores, and demonstrates that this signal is compatible with multiple existing scoring functions as a drop-in improvement. The ablation showing that fine-tuning on members also improves detection (though less than non-members) suggests the core mechanism may relate to general domain adaptation rather than specifically "exposing the model to non-members."

## Potentially Missed Related Work
- Mattern et al. (2023): Neighborhood comparison methods for MIA — represents an established baseline that could contextualize FSD's improvement
- Yeom et al. (2018), Salem et al. (2019): Classical loss-based and entropy-based MIA attacks — referenced in background but not included as baselines for comparison
- These are suggestions; the paper's citations cover the primary related work on likelihood-based detection

## Suggestions
- **Strengthen evaluation**: The Pile dataset experiments (Table 15) are the most methodologically sound because they avoid temporal confounds. The paper should prominently feature these as the primary evidence of effectiveness rather than WikiMIA, and conduct more subsets or independent runs to establish statistical significance.

---

## sZQRUrvLn4

- GT: Accept (Spotlight) (avg 6.4)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper addresses the apparent contradiction between known theoretical limitations of message-passing GNNs (which provably cannot count arbitrary subgraphs) and their empirical success in subgraph counting tasks. The authors provide sufficient conditions for when GNNs can count subgraphs on realistic datasets, develop a novel dynamic programming algorithm (TREE-COLSI) for subtree isomorphism that aligns with message passing, prove that GNNs can simulate this algorithm, and validate that the sufficient conditions hold on real-world molecular datasets.

## Strengths
- **Theoretically grounded sufficient conditions**: The (ℓ, k)-identifiability framework (Definition 2, Theorem 2) provides a principled way to understand when GNNs can count subgraphs, moving beyond worst-case impossibility results. The connection to truncated universal covers is elegant and well-motivated.

- **Novel algorithmic contribution**: The TREE-COLSI dynamic program and its "quite-colorful" condition (Definition 3) are original contributions that extend color-coding techniques to settings amenable to GNN simulation. Theorem 4's bidirectional proof is technically sound.

- **Sample complexity analysis**: Theorem 3's pseudo-dimension bound (≤ η_{ℓ,G} + 1) is a significant improvement over the naive |G| bound, directly connecting expressivity to learnability. This addresses a gap where prior positive results required impractically large models.

- **Convincing empirical validation**: Tables 2 and 3 convincingly demonstrate that real-world molecular datasets satisfy the sufficient conditions (>97% of ego-nets are (ℓ, k)-identifiable for ℓ = k + 2), and Figure 4 shows nearly all subgraph isomorphisms are quite-colorful.

- **Technical rigor**: The use of universal covers and connections to the Weisfeiler-Leman framework provides consistent theoretical grounding throughout, and comparisons to Zhang et al. (2024) are appropriately framed.

## Weaknesses
- **Classification framing does not validate counting ability**: The experiments frame subgraph counting as multi-class classification with count values as discrete classes. This does not demonstrate that GNNs can actually *count*—the model predicts which discrete bucket a count falls into, not the count value itself. The paper would benefit from true counting regression experiments (e.g., predicting exact counts or at least reporting nMAE as a continuous metric rather than framing it as classification).

- **Limited domain scope**: All experiments focus on molecular datasets. While motivated by practical relevance, this limits generalizability claims. Other graph domains (social networks, citation graphs, knowledge graphs) may behave differently.

- **No baseline comparisons**: The paper does not compare against prior subgraph counting methods, including Chen et al. (2020), k-GNNs, or subgraph GNNs. This makes it difficult to assess whether the observed GNN performance is particularly good or merely adequate.

- **Incomplete characterization**: The sufficient conditions are not necessary, and the paper does not fully characterize when they fail. The Peptides-func anomaly (one pattern where "none of the subgraph isomorphisms are quite colourful") is mentioned but not analyzed—the model still achieves reasonable performance, suggesting alternative mechanisms exist that are not explained.

- **Practical parameter bounds**: Table 6 shows η_{6,G} grows up to ~7M for some datasets, suggesting the O(η²_{ℓ,G} · ℓ) parameter bounds may be impractical for diverse graph collections. The paper does not discuss how model size scales with realistic graph complexity or whether smaller models achieve similar empirical performance.

## Nice-to-Haves
- Adversarial dataset evaluation (e.g., regular graphs) to explicitly test where sufficient conditions fail and performance degrades.
- Ablation studies on TREE-COLSI components to isolate whether specific algorithmic structure matters versus simpler aggregation strategies.
- Empirical sample complexity validation via learning curves showing test error vs. training set size.
- Quantitative comparison with Zhang et al. (2024) characterizing how many additional patterns become countable under the realistic assumptions versus worst-case bounds.
- Concrete examples explaining why some patterns never become quite-colorful.

## Novel Insights
The paper's core insight is that the gap between GNN theory (which shows they cannot count arbitrary subgraphs) and practice (where they often do count accurately) arises because worst-case impossibility results rely on adversarial graph pairs that rarely appear in real datasets. The (ℓ, k)-identifiability framework formalizes this by showing that if graphs in a dataset can be locally distinguished (via truncated universal covers), GNNs can learn to count subgraphs with bounded sample complexity. The TREE-COLSI algorithm extends this by demonstrating that even when patterns are not inherently "quite-colorful," the target graphs' WL colors can make subgraph isomorphisms satisfy this condition in practice—explaining why standard GNNs succeed on molecular datasets without architectural modifications.

## Potentially Missed Related Work
- Bouritsas et al. (2022) "Improving graph neural network expressivity via subgraph isomorphism counting" — uses subgraph counts as features to improve GNN expressivity; relevant as an alternative approach to the same problem.
- Kanatsoulis & Ribeiro (2024) "Counting graph substructures with graph neural networks" — concurrent work on subgraph counting with GNNs; deserves explicit comparison.

## Suggestions
**Reformulate experiments as true counting tasks**: Change the experimental setup from multi-class classification to regression or structured prediction that outputs exact counts or continuous count predictions. This would more directly validate the paper's theoretical claims about GNN counting ability rather than testing classification with counts as class labels.

---

## 1aF2D2CPHi

- GT: Accept (Oral) (avg 8.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper enables data-free knowledge distillation from CLIP by identifying that existing DFKD methods fail because CLIP's BatchNorm layers encode facial features from web-crawled training data. The authors propose using image-text matching for model inversion, enhanced with style dictionary diversification, class consistency maintaining, and meta knowledge distillation. The method supports customization via text prompts or few example images, achieving 64.81% average accuracy across 12 customized tasks compared to ~7% for existing DFKD methods on CLIP.

## Strengths
- **Significant problem identification**: The observation that CLIP's BatchNorm statistics encode facial features from web data, causing standard DFKD methods to fail, is a novel and important finding that explains a fundamental limitation in applying existing techniques to VLMs.
- **Comprehensive solution design**: The proposed components (style dictionary diversification, class consistency maintaining, meta knowledge distillation) address distinct aspects of the DFKD problem for VLMs, and ablation studies confirm each contributes meaningfully (baseline 60.61% → +SDD+CCM+Meta 64.81%).
- **Extensive experimental validation**: Experiments across 12 customized tasks with detailed ablation studies, comparisons across multiple VLMs (CLIP, BLIP, EVA), and multiple student architectures (ResNet-18, ViT-T) demonstrate broad applicability.
- **Practical flexibility**: Support for both text-based and image-based customization addresses real user scenarios where class names or example images may be available.

## Weaknesses
- **Missing comparison against zero-shot CLIP**: The paper never discusses why a user would choose the student model (~64.81%) over the teacher CLIP directly (~85.65% on Caltech-101). The paper claims benefits for "mobile and IoT edge devices" but doesn't quantify inference speedups or memory savings on real edge hardware or compare against efficient CLIP variants (TinyCLIP, MobileCLIP).
- **Gap between student and teacher performance not adequately justified**: The ~20% accuracy gap between the student model and CLIP zero-shot is substantial. The paper doesn't analyze whether the student offers benefits (faster inference, smaller footprint) that justify this performance loss, nor does it bound the achievable performance with synthetic vs. real data.
- **Style dictionary selection is empirical**: Table 14 shows only marginal differences between 16, 50, and 86 styles, and the final 16-style selection appears hand-tuned without clear guidance for task-specific adaptation. This limits the reproducibility and practical guidance for users.
- **ViT-based CLIP not validated as primary experiment**: While the motivation mentions that "architectures like ViT lack BN layers," the main experiments use ResNet-50 CLIP. Only a single ViT-T→ViT-B result appears in Table 5, not the reverse (ViT teacher). Most users employ ViT-based CLIP models (ViT-B/32, ViT-L/14).

## Nice-to-Haves
- **Direct ablation of image-text matching**: Section C discusses why image-text matching is superior but doesn't directly compare against pixel-space optimization or classification-guided inversion without VQGAN.
- **Per-class synthetic image quality vs. accuracy correlation**: Understanding which classes CLIP can invert well vs. poorly would clarify the method's operating range.
- **Quantitative metrics for synthetic image quality**: FID, LPIPS, or other metrics alongside task accuracy would better reveal the relationship between image quality and distillation effectiveness.

## Novel Insights
The paper's key contribution is the discovery that CLIP's BatchNorm layers, trained on web-crawled image-text pairs containing significant human imagery, encode facial features that corrupt synthetic image generation when used for model inversion. This fundamental insight explains why existing DFKD methods (DeepInversion, CMI, Fast) fail with single-digit accuracy on CLIP while achieving reasonable results (~50%) on ImageNet-trained models. The alternative approach using image-text matching to bypass BN dependencies, combined with style diversification for diversity and meta-learning for generalization across styles, represents a practical solution for customizing compact models from foundation vision-language models without accessing original training data.

## Potentially Missed Related Work
- **Textual inversion / DreamFusion variants**: While related to model personalization, works like Gal et al. (2022) on textual inversion for generative models could provide alternative approaches for handling ambiguous class names.
- **Model inversion attacks literature**: The broader model inversion attack literature (Fredrikson et al., 2015; Zhang et al., 2020) may offer additional techniques for data reconstruction that could complement the proposed approach.

## Suggestions
- Add a direct comparison between the student model and CLIP zero-shot performance in the main paper, with explicit analysis of the accuracy vs. efficiency tradeoff (inference speed, model size, memory footprint) to justify when the student model should be preferred.
- Include experiments with ViT-based CLIP teachers (ViT-B/32, ViT-L/14) as primary results, since these are the dominant CLIP variants users employ, rather than ResNet-50 CLIP.

---

## 6Mdvq0bPyG

- GT: Reject (avg 3.0)
- Predicted: Accept (8.0/10)
- Match: No

### Final Review

## Summary
EfficientQAT introduces a two-phase quantization-aware training framework for LLMs, combining Block-wise Training of All Parameters (Block-AP) followed by End-to-End Training of Quantization Parameters (E2E-QP). The method achieves state-of-the-art results in low-bit quantization (2-4 bits) while dramatically reducing training costs—enabling 70B model quantization on a single A100-80GB GPU in 41 hours with less than 3 points accuracy degradation compared to full precision.

## Strengths
- **Strong empirical performance**: Comprehensive experiments across 7B-70B models demonstrate substantial improvements over existing uniform quantization methods, particularly at 2-bit where EfficientQAT outperforms AutoRound by ~5 points (59.50 vs 54.50 on Llama-2-7B) and matches or exceeds vector quantization methods like AQLM and QuIP#
- **Practical training efficiency**: The two-phase approach reduces memory and time requirements significantly—70B model training completes in 41 hours on a single A100-80GB GPU, compared to 336 hours for AQLM or multi-GPU requirements for other QAT methods
- **Well-designed ablation studies**: Table 6 systematically demonstrates that full parameter training (s, z, W) outperforms partial training variants with clipping, rounding, or step sizes alone, providing clear evidence for the design choices
- **Broad applicability**: Validated across base LLMs, instruction-tuned models (Llama-1/2/3), and multimodal models (LLaVA-1.5), showing consistent improvements
- **Hardware-friendly design**: Uses standard uniform quantization enabling deployment via existing optimized toolboxes (BitBLAS, Marlin, T-MAC)

## Weaknesses
- **No comparison with BitNet b1.58**: The paper claims to "combine advantages of fully trainable parameters and end-to-end training, similar to native QAT (Ma et al., 2024)" but never directly compares against BitNet b1.58 results, despite BitNet being a key motivation for the work
- **Limited architecture diversity**: All experiments focus on Llama-family models (Llama-2, Llama-3, Vicuna); generalization to other architectures (Mistral, Mixtral, Gemma) is not validated, despite claims of broad applicability
- **No downstream task evaluation beyond benchmarks**: Results are limited to zero-shot accuracy on standard benchmarks and perplexity; actual instruction-following quality (MT-Bench, Chatbot Arena) is not evaluated despite being the intended deployment scenario
- **Weight-only quantization scope**: Only addresses weight quantization; weight-activation quantization for inference acceleration is not explored, limiting practical deployment value
- **LLaVA 4-bit degradation**: Table 14 shows EfficientQAT at 4-bit has slight accuracy decreases compared to 16-bit LoRA (58.3 vs 59.6 for 7B), indicating the method may not universally dominate for multimodal tasks

## Nice-to-Haves
- Ablation on iterative Block-AP → E2E-QP refinement to explore whether multiple rounds could close remaining gap to FP16
- Training/validation loss curves to demonstrate convergence stability, especially at 2-bit where STE gradient issues are known concerns
- E2E-QP analysis on larger models to verify whether training both s and z at 70B scale provides benefits justifying additional memory cost
- Per-task breakdown analysis explaining which benchmarks benefit most from EfficientQAT and why

## Novel Insights
EfficientQAT's primary contribution is demonstrating that full-parameter block-wise training within the block-wise reconstruction framework (Block-AP) is both feasible and superior to previous approaches that trained only specific quantization parameters. The insight that complex trainable parameter designs (clipping thresholds, rounding parameters) are unnecessary and that directly training all parameters (W, s, z) achieves better results with acceptable memory overhead is valuable. The two-phase approach also effectively balances the trade-off between block-wise optimization (memory efficiency) and end-to-end optimization (capturing inter-block interactions), with ablation showing Block-AP dominates (8.53 PPL vs E2E-QP's 9.33 PPL alone, but combining yields 7.68 PPL).

## Potentially Missed Related Work
- **BitNet b1.58 (Ma et al., 2024)**: The paper cites BitNet as motivation but does not include direct comparisons, making it difficult to assess the claimed combination of native QAT advantages with PTQ efficiency
- **Q-Distillation methods**: Knowledge distillation techniques for quantization that may offer complementary approaches to the block-wise training strategy
- **Recent weight-activation quantization work**: Methods like SmoothQuant and QServe that address both weight and activation quantization for deployment scenarios

## Suggestions
1. Add direct comparison with BitNet b1.58 on the same model scale to validate the claim of combining native QAT advantages with practical efficiency
2. Test on diverse architectures (Mistral, Mixtral) to validate generalization beyond the Llama family
3. Include instruction-following quality metrics (MT-Bench, Chatbot Arena) alongside benchmark accuracy to demonstrate real-world deployment quality

---

## V1N6MmDY27

- GT: Reject (avg 2.5)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes augmenting deep learning-based autonomous vehicle perception with a commonsense reasoning layer that detects inconsistencies in perception by observing collective vehicle behavior (e.g., all vehicles stopping at an intersection implies a red light or stop sign). The approach demonstrates proof-of-concept improvements for two specific scenarios—traffic light misclassification and obstacle detection—in the CARLA simulator, with an uncertainty-based mechanism to invoke the commonsense layer efficiently. However, the contribution is significantly narrower than the paper's framing suggests: it presents domain-specific rule-based consistency checking, not general commonsense reasoning, and evaluates only two hand-coded scenarios in simulation, far from the SAE Level 5 autonomy claimed.

## Strengths
- **Well-motivated problem**: The paper correctly identifies a real gap—pure deep learning approaches lack the ability to reason about edge cases and inconsistencies in safety-critical autonomous driving scenarios.
- **Modular hybrid architecture**: Decoupling the commonsense reasoning layer from the perception model is a sound design choice that allows the logic layer to be adjusted or extended without modifying the underlying deep learning system.
- **Practical uncertainty integration**: Using evidential deep learning to trigger commonsense reasoning only when needed is practically valuable for real-time deployment, addressing computational efficiency concerns.
- **Meaningful improvements on traffic light detection**: Tables 1 and 3 show consistent, substantial improvements (e.g., Town 1 100: 47.89%→85.46% accuracy) when the commonsense layer corrects perception failures, particularly in adverse weather conditions.
- **Reproducibility commitment**: The paper includes a reproducibility statement and mentions code availability, which is commendable for a simulation-based study.

## Weaknesses
- **Fundamentally overclaimed contribution**: The paper repeatedly asserts this work "paves the way to SAE Level 5 autonomy" and that "commonsense reasoning provides an effective pathway to reach SAE Level 5." This is not supported by evidence. Two hand-coded scenarios in simulation cannot demonstrate a pathway to Level 5, which requires handling arbitrary road layouts, weather, pedestrians, cyclists, adversarial conditions, legal/ethical reasoning, and much more. This overclaiming undermines the paper's credibility and should be removed or drastically qualified.
- **"Commonsense reasoning" mischaracterization**: The paper conflates hand-coded logic programs with commonsense reasoning. The rules (e.g., "if vehicles collectively stop, conclude red light or stop sign") are domain-specific pattern-matching rules, not general commonsense as understood in the field. The paper explicitly states rules are "hand-coded" and cover "only the scenarios within our scope." This should be clearly labeled as domain-specific rule-based consistency checking, not commonsense reasoning.
- **Circular evaluation in obstacle detection**: Table 2 reports 100% accuracy for the combined obstacle model. The paper admits this is because "there was always a collective lane change" by construction—"In these particular recordings, there was always a collective lane change, so the commonsense reasoning easily achieves 100% accuracy." This is not a meaningful evaluation; the logic layer's preconditions are guaranteed to be met. A fair evaluation should include cases where collective behavior is absent or ambiguous.
- **Unacceptably weak baseline**: Traffic light detection baseline accuracy ranges from 24% to 76% (Table 1), which is far below what standard object detectors achieve on this task. The paper does not explain this poor performance or whether the baseline is properly trained. Without a competitive baseline, it's unclear whether commonsense reasoning is adding value or merely fixing an improperly engineered model.
- **No ablation studies**: The paper does not isolate the contribution of individual components (BEV segmentation, behavior detection, DBSCAN clustering, uncertainty estimation). It's impossible to determine whether commonsense reasoning is the key ingredient or whether any post-processing correction would yield similar improvements.
- **No statistical rigor**: Results lack confidence intervals, variance across runs, or statistical significance testing. Given the high variance across towns (Town 4 combined: 40.6% vs Town 1 combined: 85.5%), this is a significant omission.
- **Scalability not addressed**: All commonsense knowledge is hand-coded for two specific scenarios. The paper acknowledges this is future work but does not discuss how to scale beyond manual rule authoring. For safety-critical applications, covering "the entire class of possible misclassifications" (as claimed) is infeasible with hand-coded rules.

## Nice-to-Haves
- Train a competitive baseline traffic light detector (90%+ accuracy) using standard architectures to demonstrate that commonsense reasoning adds value even when the perception model is well-engineered.
- Provide the complete logic rule set in the paper or supplementary materials for full reproducibility.
- Evaluate on real-world datasets (nuScenes, Waymo) to demonstrate sim-to-real transfer.
- Analyze failure modes: explain why Town 4 achieves only 40.6% combined accuracy (worse than Town 1's 85.5%) and when the logic layer itself fails.
- Compare against simpler heuristics (e.g., proximity-based rules) to establish that the complexity of the commonsense reasoning layer is justified.
- Analyze precision/recall trade-offs: for safety-critical applications, high precision (avoiding false positives that cause unnecessary hard braking) may be more important than high recall.

## Novel Insights
The paper's core insight—using collective vehicle behavior as a signal for perception errors—is conceptually sound and addresses a genuine limitation of end-to-end deep learning for autonomous driving. The observation that humans infer traffic signal states by watching other vehicles is valid. The uncertainty-invocation mechanism for efficient commonsense reasoning is practically motivated. However, the paper overclaims both the generality of the approach (calling hand-coded rules "commonsense reasoning") and its significance (claiming a pathway to SAE Level 5). The actual demonstrated contribution is more modest: a proof-of-concept showing that logic-based consistency checking can correct deep learning perception errors in two specific, hand-coded scenarios within a simulation environment.

## Potentially Missed Related Work
- **Neurosymbolic Visual Question Answering for AVs** (Suchan et al., 2021) — The paper discusses this but could more clearly differentiate its incremental contribution beyond the related hybrid AV work it cites.
- **Answer Set Programming for AV decision-making** (Kothawade et al., AUTO-DISCERN) — Discussed briefly but the comparison could be deeper regarding when each approach is appropriate.
- **Bayesian/Robust optimization for AV perception uncertainty** — General uncertainty quantification literature that could provide stronger baselines for the uncertainty-invocation mechanism.

## Suggestions
1. **Remove or drastically qualify the SAE Level 5 claims.** Rephrase the contribution as demonstrating that logic-based consistency checking can improve perception in specific narrow scenarios—a more accurate characterization of what the paper actually shows.
2. **Rename "commonsense reasoning" to "logic-based consistency checking"** or "rule-based verification" to accurately reflect the hand-coded nature of the approach.
3. **Fix the baseline evaluation:** Train a properly competitive traffic light detector or provide clear justification for why the baseline performs so poorly. Without this, the improvements are difficult to interpret.
4. **Improve obstacle detection evaluation:** Include trials without guaranteed collective lane changes to avoid circular evaluation. Show the logic layer's behavior when collective behavior is ambiguous or absent.
5. **Add ablation studies** to demonstrate that the commonsense reasoning component specifically (not just any post-processing) is responsible for the observed improvements.

---

## Pxik3T6Mn9

- GT: Reject (avg 4.5)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces UIFORMER, an uncertainty-aware dual Transformer architecture for human mobility modeling and unsupervised anomaly detection. The model combines feature-level and event-level attention with both aleatoric (data) and epistemic (model) uncertainty estimation to enable robust training and nuanced anomaly scoring that accounts for inherent behavioral stochasticity. Experiments on two synthetic datasets demonstrate meaningful improvements over baseline methods in both masked prediction and anomaly detection tasks, with ablation studies validating the complementary benefits of different uncertainty types.

## Strengths
- **Principled uncertainty decomposition**: The explicit modeling of both aleatoric and epistemic uncertainty, with clear theoretical motivation distinguishing inherent data stochasticity from model uncertainty, is well-executed. The empirical validation showing that aleatoric uncertainty remains stable across training sizes while epistemic uncertainty decreases appropriately strengthens the theoretical claims.

- **Strong empirical validation of uncertainty calibration**: The analysis demonstrating monotonic improvement in prediction performance when excluding high-uncertainty samples (Figure 4) provides compelling evidence that uncertainty estimates are meaningful. The complementary nature of the two uncertainty types—aleatoric helping regression tasks while epistemic benefits categorical predictions—adds theoretical depth.

- **Well-motivated architectural design**: The dual Transformer approach treating mobility data as a sequence-of-sequences (events with nested features) is appropriate for the spatiotemporal event structure, and the within-day positional encoding explicitly captures day boundaries which is important for human routines.

- **Comprehensive ablation studies**: The paper systematically ablates both uncertainty components and the percentile normalization, revealing which aspects contribute to performance gains. The finding that different uncertainty types complement different feature types (AU for regression, EU for classification) provides actionable insight.

## Weaknesses
- **Evaluation limited to synthetic data**: Both SimLA and NUMOSIM are expert-simulated datasets. The paper never explains what constitutes an "anomaly" in these datasets (injected behavioral deviations, specific agent types, etc.), making it difficult to interpret what the model actually detects. Real-world GPS data with documented anomalies would substantially strengthen practical impact claims, particularly for the stated application of "bad-actor or malicious behavior detection."

- **Inconsistency on NUMOSIM (small) agent-level**: UIFORMER achieves AUROC=0.655 compared to SS-MLP's 0.697 on this dataset/metric combination, yet the paper does not discuss this failure case or explain why a simpler baseline outperforms on this configuration.

- **Percentile normalization dominates improvement**: Table 4 shows that pth(Δ_f) alone achieves AUROC=0.789 on SimLA (small), nearly matching UIFORMER's 0.795. The paper frames uncertainty estimation as a central contribution, but the ablation suggests the marginal benefit of uncertainty incorporation is modest. The paper should better articulate why the added complexity of MC Dropout and learned variance is warranted.

- **Unclear dropout specification**: The paper states a "dropout ratio of 0.05" but only shows dropout after embedding layers in Figure 2. Whether dropout is applied uniformly across all layers (standard for MC Dropout) is ambiguous, affecting reproducibility and theoretical validity.

## Nice-to-Haves
- **Statistical significance testing**: Single-run results without confidence intervals or significance tests make it difficult to assess whether improvements are reliable across random initializations.
- **Computation complexity analysis**: No training/inference time or memory footprint reporting, which is important for practical deployment given that MC Dropout requires 50 forward passes.
- **Qualitative anomaly examples**: The case study only shows high/low uncertainty agents on normal behavior. Concrete examples of detected vs. missed anomalies with contextual explanation would improve interpretability.
- **Calibration curves**: Standard uncertainty evaluation (expected calibration error, reliability diagrams) would strengthen the uncertainty quantification claims beyond the monotonic trend analysis.

## Novel Insights
The paper's core insight is that uncertainty-aware anomaly scoring can distinguish between genuine anomalies and naturally variable behavior—for instance, an exploratory individual who frequently deviates from population norms should not be flagged as anomalous simply because their predictions have high error. By incorporating both prediction deviation and uncertainty estimates into the anomaly score, the model can appropriately downweight predictions for inherently stochastic individuals. The empirical finding that aleatoric and epistemic uncertainties complement each other across feature types (numerical vs. categorical) suggests this decomposition has practical value beyond theoretical appeal.

## Potentially Missed Related Work
- None identified. The related work review adequately covers trajectory prediction, uncertainty estimation in deep learning, and anomaly detection baselines.

## Suggestions
1. **Validate on real-world mobility data**: Test on Geolife, Foursquare check-ins, or taxi trajectories with injected or documented anomalies to demonstrate real-world applicability beyond synthetic settings.
2. **Explain the anomaly generation process**: Describe what constitutes anomalies in SimLA and NUMOSIM so readers can interpret detection performance appropriately.
3. **Address the NUMOSIM (small) failure case**: Analyze why SS-MLP outperforms UIFORMER on agent-level detection for this configuration, and discuss whether this reveals a limitation of the uncertainty-based approach.
4. **Clarify the dual-attention contribution**: Add an ablation isolating whether feature-level attention (in addition to event-level) provides value, to justify the dual Transformer complexity over a standard Transformer baseline.

---

## ZSdubdbOoi

- GT: Accept (Poster) (avg 6.0)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes Self-Improving Robust Preference Optimization (SRPO), an offline RLHF framework that casts preference learning as a self-improvement process through a min-max objective jointly optimizing a self-improvement policy and a generative policy. The key theoretical contribution demonstrates that the optimal solution can be expressed independently of the behavior policy µ, and the min-max objective can be reformulated as a practical supervised loss without requiring reward models or online inference.

## Strengths
- **Theoretically principled framework**: The derivation from the KL-regularized RL objective (Eq. 1) to the supervised ℓ₂ loss is mathematically sound. The key result (Eq. 14) extends DPO's insight by expressing preference probabilities in terms of optimal policies without requiring the Bradley-Terry assumption—only requiring that preferences exist.

- **Significant OOD improvement**: The empirical results on XSum (trained on TL;DR) demonstrate meaningful gains: SRPO 5-rev achieves 90% win-rate against gold, representing a 15 percentage point improvement over DPO in the OOD setting where robustness matters most.

- **Novel self-improvement capability**: Unlike DPO/IPO which fail to improve through in-context revision, SRPO consistently improves with each N-revision iteration across both in-distribution and OOD tasks. This demonstrates the practical value of the learned self-improvement policy.

- **Practical contribution**: Converting an adversarial min-max problem into a tractable supervised loss (Algorithm 1) that can be trained with standard techniques is valuable for scalability and aligns with the successful DPO/IPO paradigm.

## Weaknesses
- **Overstated robustness claims**: The paper states SRPO's solution is "completely independent of the behavior policy µ" and "completely robust to its changes." However, Figure 2 shows SRPO's learned policy differs slightly between uniform (µ₀) and skewed (µ₁) distributions. The claim conflates "the optimal solution can be expressed without µ" with "training is robust to µ changes in practice." The qualification "under idealized conditions" or "in the infinite data regime" is missing.

- **Headline result conflates algorithmic and inference-time improvements**: The 15% improvement over DPO comes from 5 iterations of self-revision. The fair comparison should be SRPO 0-rev vs DPO 0-rev, where the gap is substantially smaller (~2-3% on TL;DR). The paper should clearly separate algorithmic improvement from the additional computation of iterative refinement.

- **Unfair baseline comparison for self-improvement**: DPO and IPO were not designed for self-improvement, yet they're evaluated using their own outputs as revision contexts. The paper acknowledges this in passing but doesn't address that this comparison doesn't isolate whether gains come from the training objective or the inference-time revision mechanism itself.

- **Abrupt performance degradation unexplained**: Figure 5 shows significant win-rate drops for all models after revision 1 on Arena-Hard, with SRPO 1-rev still outperforming baselines but substantially worse than SRPO 0-rev. The paper acknowledges this but provides insufficient analysis of why self-improvement fails in this setting or how to determine the optimal revision count.

- **ℓ₂ loss properties undiscussed**: The practical algorithm minimizes an ℓ₂ surrogate rather than directly optimizing the original objective. The paper provides no bounds on approximation error or statistical analysis (consistency, sample complexity) of this estimator. The convex combination claim in Section 4.2.3 also lacks justification.

## Nice-to-Haves
- Compare against additional baselines (KTO, SLiC-HF) in the main experiments to contextualize whether gains are specific to SRPO or general to preference optimization.
- Include ablation of β across methods with equivalent search budgets, as SRPO uses β=0.01 while DPO uses β=0.1 (10× difference).
- Add human evaluation on a subset of samples to complement GPT-4-as-judge.
- Report inference-time computational costs (latency/throughput) to assess practical utility of multiple revision passes.

## Novel Insights
The paper's core insight—that framing preference learning as self-improvement leads to solutions independent of the behavior policy—is genuinely novel and addresses a real limitation in existing offline RLHF methods. The two-policy formulation (generative π and self-improvement π†) is conceptually clean and the mathematical derivation shows how this decoupling removes the µ-dependency that plagues DPO/IPO. The empirical finding that SRPO learns genuine self-improvement capability (improving consistently up to 5 revisions) while DPO/IPO fail at even 1-revision is a compelling demonstration that the training objective shapes what the model can do at inference time, not just what it produces.

## Potentially Missed Related Work
- **KTO (Kahneman-Tversky Optimization)**: Ethayarajh et al. introduces a different framing of preference alignment based on prospect theory that could provide an alternative baseline for comparison.
- **SLiC-HF**: Zhao et al. proposes sequence likelihood calibration with human feedback using supervised learning, which is closely related to the offline preference optimization paradigm explored here.

## Suggestions
The paper should clearly separate the algorithmic contribution (SRPO 0-rev vs DPO 0-rev) from the inference-time refinement capability (N-revision gains) when presenting results, and qualify the robustness claims with "under idealized/exact optimization conditions" to avoid overstating practical robustness.

---

## 5XL8c0Vg9k

- GT: Reject (avg 2.0)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary
The paper proposes an Infinite Parameter Large Language Model (IP-LLM) architecture that decouples model parameter count from computational cost by routing inputs to domain-specific expert parameters at inference time, using a frozen base model (Qwen-7B) for initial processing and routing, with claims of 65% memory reduction during inference compared to dense models of equivalent total parameter count.

## Strengths
- **Addresses real problems**: The paper correctly identifies genuine challenges in LLM deployment—catastrophic forgetting during continual learning, model obsolescence, and memory constraints at inference time. These are active research problems worth investigating.
- **Reasonable architecture concept**: The idea of routing once per inference task (rather than per-token like MoE) to enable selective parameter loading is a reasonable approach for reducing peak memory consumption. The staged pretraining strategy—separating foundational language knowledge from domain-specific knowledge—is a sensible design principle.
- **Catastrophic forgetting mitigation**: Storing knowledge for different categories in separate parameter sets and training new experts without modifying existing ones is a valid approach to enabling incremental updates without catastrophic forgetting. The paper correctly identifies this as an advantage over fine-tuning dense models.

## Weaknesses
- **No empirical results provided**: The paper references Figure 2 showing benchmark comparisons (MMLU, C-Eval, GSM8K, MATH) but provides no extractable numerical values in the text. The central performance claim ("comparable to the dense model") is entirely unsubstantiated. This is a fundamental deficiency for any ICLR submission—readers cannot verify any quantitative claim.
- **Factually incorrect MoE comparison**: The paper states "MoE requires loading all parameters into memory" during inference. This misrepresents how MoE operates—sparse activation means only a subset of experts perform computation during each forward pass. While expert parameters reside in memory (often distributed across devices), this is fundamentally different from dense model inference where all parameters contribute to every forward pass. This technical inaccuracy undermines the paper's comparison claims.
- **Single-expert routing is a critical limitation**: IP-LLM routes to a single expert per entire inference task. For queries spanning multiple domains (e.g., "Explain the mathematical foundations of quantum mechanics"), this design choice would access only one expert's knowledge, potentially degrading output quality. No mechanism for handling multi-domain queries is proposed or evaluated.
- **Unsubstantiated "infinite parameter" framing**: The paper describes a 24.5B parameter model but claims the architecture can "accommodate infinite knowledge." With finite memory for storing expert parameters and finite computational resources, this is misleading. The paper does not define operationally what "infinite" means or bound the actual scalability.
- **Missing baseline comparisons**: No experiments compare against: (1) the Qwen-7B base model used for training, (2) established MoE baselines like Mixtral, or (3) prior work on lifelong learning with MoE. Without these comparisons, the contribution's value cannot be assessed.
- **Incomplete methodology**: Critical training details are absent: data labeling procedure for the 22 categories, training hyperparameters, exact proportion of domain-specific vs. general data used during expert training, and how category boundaries are determined. The paper states experts are trained "sequentially" but Equation 7 suggests independent expert models—a contradiction.

## Nice-to-Haves
- **Latency analysis**: The paper emphasizes memory reduction but ignores latency costs of parameter loading/unloading. In practice, loading 0.75B parameters from storage incurs significant time overhead.
- **Routing accuracy measurement**: The paper claims improved routing accuracy over MoE but provides no quantitative evidence. Measuring and reporting routing accuracy as a function of category count would validate this core claim.
- **Failure mode analysis**: What happens when the router misclassifies an input? How does output quality degrade? This analysis is absent but essential for understanding real-world reliability.
- **Cross-domain generalization**: How well do experts trained on one category generalize to related categories? How does the base model handle unseen domains?

## Novel Insights
The paper's core insight—using a large frozen base model for routing rather than a small router network—may offer improved routing accuracy over MoE, though this is asserted rather than demonstrated. The architecture's approach to lifelong learning (adding new expert parameters without modifying existing ones) provides a clean mechanism for avoiding catastrophic forgetting that differs from prior approaches like regularization-based continual learning. The concept of routing once per inference task rather than per-token could reduce expert-switching overhead, though the single-expert commitment is a significant limitation. Overall, these ideas are directionally interesting but would require substantial empirical validation to constitute genuine contributions.

## Potentially Missed Related Work
- **Lifelong MoE (Chen et al., 2023)**: The paper references this but the comparison in Section 2.4 focuses on differences without acknowledging the similar goal of using MoE for continual learning. A more thorough discussion of how IP-LLM differs in mechanism (staged training, on-demand loading) vs. Lifelong MoE's distribution-specialized experts would clarify the contribution.
- **CoLT5 (Aribandi et al., 2022)**: Selective attention/routing mechanisms for handling long inputs could provide relevant comparison points for routing strategies.
- **Hash layers / adaptive compute**: Work on adaptive computation depth (e.g., using hash functions to route to different parameter sets) provides related context for parameter-efficient routing.

## Suggestions
- **Provide complete numerical results**: Report all benchmark numbers for IP-LLM alongside baselines (Qwen-7B, Mixtral 8x7B, a dense 24B model) on MMLU, C-Eval, GSM8K, and MATH. These numbers are essential for evaluating the paper's claims.
- **Correct the MoE comparison**: Reframe the comparison to acknowledge that MoE achieves sparse activation during computation (not just sparse memory allocation). The actual advantage of IP-LLM is in on-demand parameter loading from storage, not in computational sparsity.
- **Reduce hyperbolic language**: Remove claims about "omniscient and omnipotent artificial general intelligence" and "infinite parameters." Frame the contribution as a memory-efficient domain-routed architecture for continual learning.
- **Specify the expert architecture**: Clearly state how many transformer layers constitute each expert, how the 4 replicated layers per domain work in practice, and how domain labeling was performed on training data.
- **Evaluate routing accuracy**: Report empirical routing accuracy (percentage of queries correctly classified) and analyze failure cases, particularly for multi-domain inputs.

---

## bEgDEyy2Yk

- GT: Reject (avg 1.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper presents a Python implementation of Algorithm 4 (MMJ distance by Calculation and Copy) from Liu (2023) for computing the All Pairs Minimax Path (APPD) distance matrix in undirected dense graphs, claiming O(n²) complexity. Experiments on seven datasets (N=120 to N=10,000) demonstrate significant speedups over O(n³) alternatives, with Algorithm 4 completing N=10,000 in 67 seconds while competitors timeout at N=2,500.

## Strengths
- **Practical value**: This is the first working implementation of an O(n²) APPD algorithm, addressing a recognized gap between theory and practice. The email correspondence with Dr. Murphy (Section 6) corroborates that no practical implementations of SLINK or Cartesian tree approaches were known to the community.
- **Strong empirical results**: The timing comparisons in Table 2 are compelling—orders-of-magnitude improvement (67s vs. >2 hours for N=10,000) is practically significant for machine learning workflows using minimax distances.
- **Reproducibility**: The paper provides specific hardware (3.3 GHz Quad-Core Intel Core i5, 16 GB RAM), explicit timeout conditions (7200s), and references to data sources, enabling independent verification.
- **Clear problem statement**: The minimax path problem, APPD matrix, and equivalence to widest path problem are well-defined with equations (1)-(2).

## Weaknesses
- **No correctness verification**: The paper never validates that Algorithm 4 produces accurate APPD values. For small datasets (N=120, 300), the authors could compare against the verified Floyd-Warshall implementation (Figure 2) to establish ground truth. Without this, speedup claims are hollow—a fast incorrect algorithm is worthless.
- **O(n²) complexity claim lacks rigorous justification**: The paper states nested loops "make it look like O(n³) but it is actually O(n²)" due to varying subtree sizes, but provides no formal amortization argument. The analysis "when i is small, tree1 and tree2 are O(n); when i is large, both are O(1)" is hand-wavy, not a proof that total work is O(n²).
- **Bug in MST_shortest_path implementation**: Figure 3, line 6 uses undefined variable `N` (should be `len(G.nodes())`). This code error undermines confidence in the authors' implementation quality.
- **Unfair baseline comparison**: MST_shortest_path computes bottleneck path distances using MST, but the paper acknowledges MBST is not necessarily MST, meaning this baseline may compute different (incorrect) distances. Comparing Algo 4 against a potentially incorrect baseline is methodologically flawed.
- **Missing comparison to other O(n²) methods**: The paper claims Algorithm 4 is "the first" implementation of O(n²) APPD but never implements or compares against SLINK (which Murphy suggests could compute APPD from dendrogram) or Cartesian trees, despite citing these approaches.

## Nice-to-Haves
- Statistical rigor: Single-run timing measurements lack variance estimates; multiple trials with mean±std would strengthen conclusions.
- Dataset diversity: All seven datasets vary only in size (N), not structure; testing with varied graph properties (edge weight distributions, density variations within dense graphs) would validate robustness.
- C++ implementation: The language comparison (Algo_1, Floyd_Warshall have C++ versions; Algo_4 does not) conflates language effects with algorithm effects.
- Scaling beyond N=10,000: The practical impact of O(n²) vs O(n³) would be clearer with larger-scale experiments (N=50,000 or 100,000).

## Novel Insights
The paper's core insight is pragmatic rather than theoretical: translating a theoretically optimal but unimplemented O(n²) algorithm into working code demonstrates its practical viability. The experimental finding that this implementation achieves 67 seconds for N=10,000 (where verified O(n³) methods timeout) represents a genuine enabling capability for practitioners who need minimax distances for clustering or classification tasks at scale. The email correspondence with Dr. Murphy provides compelling evidence that this implementation gap was indeed unrecognized by the community, making the practical demonstration valuable even without theoretical novelty.

## Potentially Missed Related Work
- **SLINK implementation**: The paper cites Sibson (1973) and suggests (via Murphy's email) that SLINK could compute APPD from its dendrogram output, but this connection is not explored or implemented. This is directly relevant to the paper's "first implementation" claim.
- **Cartesian tree approaches**: Demaine et al. (2009, 2014) are cited as O(n²) theoretical methods but never compared against Algorithm 4.

## Suggestions
1. **Add correctness verification immediately**: For at least N=120 and N=300 datasets, compute the APPD matrix using both Algorithm 4 and the Floyd-Warshall variant (Figure 2), and report maximum absolute difference. This single addition would validate the core claim.
2. **Fix the code bug**: Replace `N` with `len(G.nodes())` in Figure 3, line 6.
3. **Strengthen complexity analysis**: Provide a proper amortized argument for the O(n²) claim, or at minimum a count of total matrix cell accesses across all iterations.
4. **Either implement or clearly dismiss SLINK/Cartesian trees**: If these are truly impractical to implement, explain why Algorithm 4 is implementable when they are not.

---

## u1cQYxRI1H

- GT: Accept (Oral) (avg 10.0)
- Predicted: Reject (5.0/10)
- Match: No

### Final Review

## Summary
This paper proposes IC-Light (Imposing Consistent Light transport), a training methodology for scaling diffusion-based illumination editing models by enforcing the physical principle that linear blending of object appearances under different illuminations equals appearance under mixed illumination. The method enables stable training on over 10 million diverse samples (light stage data, 3D renders, in-the-wild augmented images) across multiple backbones (SD 1.5, SDXL, Flux), while preserving intrinsic properties like albedo.

## Strengths
- **Physically grounded constraint**: The light transport consistency principle (Eq. 3: I(L₁+L₂) = I(L₁) + I(L₂)) provides a theoretically justified regularization that directly addresses the core challenge of ensuring illumination-only modification. This is a principled approach to preventing the model from devolving into a "structure-guided random image generator."
- **Successful scaling demonstration**: The ability to train on 10M+ images across heterogeneous data sources (real light stage captures, rendered Objaverse data, in-the-wild images with synthetic augmentation) with a unified framework is a significant practical contribution. The scheduled probability mixing of data sources (starting 0.5/0.5/0.0, ending 0.35/0.35/0.3) is a thoughtful design.
- **Multi-backbone validation**: Demonstrating the approach works with SD 1.5, SDXL, and Flux shows architectural generality and practical applicability for different deployment scenarios.
- **Clean mathematical formulation**: The extension of the physical linearity constraint to diffusion objectives (Section 3.2) is well-reasoned, including the handling of scaling factors in different prediction objectives (Appendix A).

## Weaknesses
- **Evaluation biased toward training distribution**: The quantitative evaluation uses only a 50k subset of "unseen 3D rendering data" from the training set. This creates circular reasoning—the full method includes 3D data, so it will naturally perform well on 3D evaluation data. The paper's core claim is generalization to "in-the-wild" scenarios, but this is never quantitatively validated on real-world images.
- **Ablation study lacks quantitative support**: Figure 4 shows ablation of in-the-wild data and light transport consistency, but only with qualitative visual examples. Table 1 provides quantitative metrics, but these are for the full test set only, not for the specific ablation variants shown visually. The paper states removing consistency "significantly decreased" albedo preservation but provides no numerical evidence for this claim.
- **No quantitative albedo/intrinsic preservation validation**: The paper's central claim—that the method "preserves intrinsic properties such as albedo" (Abstract, Section 1)—is never measured. No metrics like IIW WHDR scores or albedo consistency measurements are reported. This is a critical gap given that albedo preservation is a primary motivation for the entire approach.
- **MLP adaptation layer unjustified**: The learnable MLP φ(·,·) for bridging HDR/LDR/latent domains is introduced with no ablation or analysis. The 5-layer, hidden=128 architecture appears purely heuristic. The paper does not analyze what this MLP learns or when it might fail.
- **Loss weight λ_consistency=0.1 not ablated**: The consistency loss contributes only ~9% of the total gradient (0.1 vs 1.0), yet this is the paper's core contribution. No sensitivity analysis is provided to justify this weight or show how results vary with it.
- **Missing hyperparameters**: Batch size and total training iterations are not reported. Only training time (hours) is given. This significantly impairs reproducibility.

## Nice-to-Haves
- **User study**: ICLR standards for generative/editing methods typically expect human evaluation of output quality (lighting plausibility, albedo preservation, artifact presence).
- **Standard benchmark evaluation**: Evaluation on established intrinsic image decomposition benchmarks (IIW, Sintel) would directly validate albedo preservation claims.
- **Statistical validation**: No error bars or confidence intervals on quantitative metrics. The LPIPS improvement (0.1025 vs 0.1077 for "w/o light stage") is marginal and may not be statistically significant.
- **Comparison with dedicated intrinsic decomposition methods**: Methods like IntrinsicAnything or IntrinsicDiffusion are mentioned in related work but never compared against, despite overlapping goals.

## Novel Insights
The paper's key insight—applying the physical linearity of light transport as a training constraint in diffusion models—is genuinely novel and well-motivated. The observation that light transport linearity can be transferred to diffusion prediction targets (eps-prediction, v-prediction, flow match) opens a new direction for physics-informed training of generative models. The staged data mixing strategy and the unified handling of heterogeneous data sources (light stage, 3D renders, in-the-wild) also represent practical innovations for the field.

## Potentially Missed Related Work
- **IntrinsicDiffusion (Luo et al., 2024)** and **IntrinsicAnything (Xi et al., 2024)**: These recent works on intrinsic image decomposition using diffusion models overlap significantly with the paper's stated goal of albedo preservation but are only briefly mentioned in related work. A more detailed comparison would contextualize the contribution.

## Suggestions
- **Add quantitative albedo preservation metrics**: Use established benchmarks (IIW WHDR) or compute albedo consistency before/after relighting to substantiate the paper's core claim.
- **Evaluate on held-out real-world data**: Create a test set of real images with ground truth relighting to validate the "in-the-wild" generalization claims beyond 3D rendering data.
- **Ablate the consistency loss weight**: Sweep over λ_consistency values (e.g., 0.01, 0.05, 0.1, 0.5, 1.0) and report quantitative results to justify the chosen weight.
- **Analyze MLP behavior**: Visualize what the adaptation MLP learns and characterize failure cases to build confidence in the approach.

---

## B5Dj4EhZPP

- GT: Reject (avg 4.3)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes "Reciprocal Feedback," a biologically plausible alternative to backpropagation that uses autoencoder-like pseudoinverse feedback connections instead of weight transposes. The key contribution is showing how a modified Recirculation algorithm can learn Moore-Penrose pseudoinverses, which are then used in a Hildebrandt-Graves Newton-like optimization framework to train networks without requiring weight transport. Experiments on MNIST and CIFAR-10 show comparable asymptotic error to backpropagation and faster convergence than Feedback Alignment.

## Strengths
- **Novel theoretical framework**: The paper connects the Hildebrandt-Graves theorem (1927), previously unexplored in neural network training, with locally-learned pseudoinverse feedback connections — a genuinely novel synthesis not found in prior biologically-plausible learning literature.
- **Strong neuroscientific grounding**: The work is well-motivated by recent evidence (Chen et al., 2024; Marino, 2020; Linde-Domingo et al., 2019) suggesting cortical feedback connections function autoencoder-like rather than as pure error feedback, providing a principled bridge between biological and machine learning perspectives.
- **Complete derivation**: The mathematical connection between pseudoinverse learning via Recirculation modifications and Newton-like weight updates is rigorously derived, with complete proofs in the appendix.
- **Reproducibility**: Code is available, hyperparameters are fully specified, and pseudocode for both algorithm variants (two-phase and parallel) is included.

## Weaknesses
- **Critical theory-practice mismatch**: Theorem 1 requires bijective activation functions (Section 4), yet experiments use "noncontinuous, leaky ReLU" (Section 5.1), which is not bijective (not surjective onto ℝⁿ and not continuously differentiable). This discrepancy undermines whether the theoretical guarantees apply to the experiments presented.
- **Missing critical comparison**: The paper extensively discusses the weight-mirror algorithm (Akrout et al., 2019) as a related approach but provides no experimental comparison. Given weight-mirror has been shown to scale well to larger datasets, this comparison is essential for evaluating the paper's practical contribution.
- **Limited experimental significance**: CIFAR-10 results (~40-41% accuracy) are inherently limited by the use of fully-connected architectures, which the paper acknowledges are unsuitable for this dataset. This weakens the significance of the empirical contribution.
- **Insufficient statistical rigor**: Results report only the best of 20 random seed configurations without error bars or confidence intervals, introducing selection bias. Mean ± std across seeds should be reported.

## Nice-to-Haves
- Ablation studies examining: contribution of wake-sleep training, sensitivity to pseudoinverse accuracy, and effect of update frequencies in two-phase version.
- Analysis of how learned pseudoinverses track evolving forward weights during training.
- Convergence rate analysis for Theorem 1 under practical conditions.
- Testing on deeper networks (10+ layers) where biologically-plausible alternatives are most needed.

## Novel Insights
The paper's most significant insight is demonstrating that locally-learned pseudoinverse feedback connections (learned via modified Recirculation with mean-zero noise inputs) can substitute for weight transposes in propagating error, circumventing the weight transport problem. Furthermore, the connection to the Hildebrandt-Graves iteration as a Newton-like optimization method — distinct from Gauss-Newton — provides a fresh theoretical lens for understanding alternative optimization in neural networks. The wake-sleep cycle provides a biologically intuitive training procedure that mirrors both the error minimization (wake) and weight alignment (sleep) phases observed in cortical processing.

## Potentially Missed Related Work
- **Tapson & van Schaik (2013)**: Describes an online, biologically-plausible algorithm for computing pseudoinverses — directly relevant to the paper's pseudoinverse learning mechanism and mentioned briefly but could be discussed more thoroughly as an alternative implementation.
- **Direct Feedback Alignment and Sign Symmetry**: Standard baselines in the biologically-plausible learning literature that could strengthen the experimental comparison beyond just Feedback Alignment.

## Suggestions
1. **Resolve the bijectivity issue**: Either use a bijective activation function (e.g., invertible softplus) in experiments to satisfy Theorem 1's assumptions, or formally prove convergence under non-bijective activations and acknowledge the theoretical gap.
2. **Add weight-mirror experimental comparison**: Given the paper explicitly contrasts its approach with weight-mirror's transpose-based feedback, this comparison is essential for claims of practical contribution.
3. **Report full statistical summary**: Include mean ± std across all 20 seeds rather than only the best result to properly characterize algorithm performance.

---

## IUwqJ8VT4F

- GT: Reject (avg 4.0)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
This paper systematically investigates design choices—architectural variants (Transformer, Convformer, Mamba) and context lengths (K=2, 5, 10, 20)—in max-return sequence modeling for offline RL across 9 diverse D4RL datasets. Using Reinformer (Zhuang et al., 2024) as a base framework, the authors demonstrate that dataset characteristics have greater impact on performance than architectural or context-length choices, that architecture superiority is dataset-dependent (Transformer excels on non-Markovian data; Convformer/Mamba on trajectory stitching), and that surprisingly, training on long sequences while inferring with short contexts (via masking) significantly improves trajectory stitching performance.

## Strengths
- **Comprehensive empirical evaluation**: The 9 datasets × 3 architectures × 4 context lengths grid (108 configurations) provides thorough coverage that systematically addresses prior work's limited evaluation scope. The paper correctly identifies that prior conclusions about architecture superiority (e.g., Mamba/Convformer outperforming Transformer) were biased by evaluation on only Gym datasets.
- **Valuable empirical insights**: The finding that long-trained models (K=20) can dramatically improve trajectory stitching when inferring with short contexts (K=2) via masking (Figure 7 shows Reinformer improvement from ~1.6 to ~36 on Antmaze-medium-play) is a genuinely surprising and potentially impactful discovery.
- **Methodologically sound attention analysis**: The token masking ablation in Section 5.2.1 provides a principled way to examine whether architectures process local vs. global information, and the experiments with swapping attention matrices and value vectors (Table 5) offer insight into why masking helps.
- **Actionable practical guidance**: The concluding recommendations—when to use sequence modeling (high-quality data) vs. classical offline RL (trajectory stitching)—provide practitioners with clear decision criteria.

## Weaknesses
- **Unexplained NaN failure case**: Table 2 shows Reinformer with K=2 on Maze2d-large produces NaN (also noted as "nan" for training steps in Appendix). This critical failure mode is never discussed or analyzed, raising concerns about implementation reliability and whether certain configurations are fundamentally unstable.
- **Weak baseline comparison**: Only comparing against IQL (acknowledged as not current SOTA) limits the paper's claims. The recommendation to "resort to classical Offline RL when stitching is crucial" lacks conviction without comparisons to TD3+BC, CQL, or more recent offline RL methods that practitioners would actually consider.
- **Missing ablation of expectile regression**: The max-return mechanism relies on expectile regression loss (Equation 4), but the paper never ablates whether this component actually helps. A control with α=0.5 (standard MSE) would clarify whether return prediction matters or if architecture changes alone drive the results.
- **Inconsistent training hyperparameters**: Training steps vary from 1w to 20w across configurations (e.g., Reimba K=20 uses 4w for HC-me but 10w for HP-mr), and tau values differ (0.9 vs 0.99 vs 0.999). Without evidence that each configuration is properly converged, comparisons may be confounded by undertrained models.
- **Theoretical claim without formal support**: The paper states that expectile regression "can make the model predict the maximum returns-to-go when α → 1" but provides no proof, citation, or formal analysis of this claim.

## Nice-to-Haves
- **Error bars in main results**: Table 2 reports means only (5 seeds). Standard deviations would enable assessment of whether performance differences are statistically significant—a standard expectation for empirical papers.
- **Comparison to original Reinformer**: The paper builds on Reinformer but doesn't report its numbers on these 9 datasets, making it unclear what the paper's actual contribution beyond the base method is.
- **Formal variance decomposition**: The claim that "dataset characteristics have greater impact" is asserted but not quantified. A simple variance decomposition would make this claim more compelling.
- **Testing on non-D4RL domains**: The paper criticizes prior work for limited evaluation but remedy is adding more D4RL datasets. Testing on other domains (robotics, game-playing) would better support generalizability claims.

## Novel Insights
The paper's most significant contribution is the systematic demonstration that architecture superiority in sequence modeling is fundamentally dataset-dependent rather than universal—prior work claiming Mamba/Convformer superiority over Transformer was observing dataset-specific behavior, not general trends. The attention analysis showing Transformer focuses on global historical information (better for non-Markovian data like Maze2d) while Convformer/Mamba focus on local information (better for trajectory stitching) provides a principled explanation for this dataset-dependency. Most surprisingly, the discovery that masking historical tokens during inference (training on K=20 but inferring on K=2) dramatically improves trajectory stitching—through the mechanism that masked value vectors enable more exploratory behavior (Table 5: from 1.60 to 36.00)—suggests a simple practical technique for combining the benefits of long-sequence training with short-sequence inference flexibility.

## Potentially Missed Related Work
- **Antmaze-umaze datasets**: Table 1 shows these are commonly used for trajectory stitching analysis, yet only medium-play and medium-diverse are evaluated. Antmaze-umaze results would strengthen conclusions about stitching ability across architectures.
- **One-step/decision-aware RL methods** (e.g., Cal-QL, COMBO): These represent current offline RL state-of-the-art and would provide stronger baselines for the "when to use sequence modeling vs. RL" recommendation.

## Suggestions
1. **Investigate and explain the NaN failure** for Reinformer K=2 on Maze2d-large, or remove this configuration from reported results with appropriate caveats.
2. **Add ablations** showing performance without expectile regression (α=0.5) to isolate its contribution.
3. **Standardize training steps** across architectures for fair comparisons, or provide convergence curves demonstrating that each configuration is properly trained.
4. **Include error bars** in Table 2 and main figures to enable statistical comparison of results.
5. **Clarify the contribution statement**: Is this an empirical study validating/contradicting prior architecture claims, an introduction of Reinconver/Reimba variants, or a study of the masking-at-inference discovery?

---

## LJULZNlW5d

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes FGL (Fast Gradient Leakage), a gradient inversion attack for federated learning that combines StyleGAN priors with a joint gradient matching loss (L2 + Cosine + L1) and multi-seed optimization strategies to reconstruct private images from gradients. The method demonstrates high-resolution (224×224) face reconstruction on CelebA with batch sizes up to 60 and claims significant speed improvements over prior work (2.58 minutes vs ~24 minutes for batch=1 attacks).

## Strengths
- **Comprehensive ablation analysis**: The paper systematically evaluates each component (Sinit, Sfinal, Ttrans, Mgrad, Ngrad, Mseed), showing progressive improvement from Top-1=0.0 to Top-1=0.88, providing clear evidence that each proposed technique contributes to attack effectiveness.
- **Genuine time efficiency gains**: The reduction from ~24 minutes (GI, GIAS) to 2.58 minutes for batch=1 attacks is a meaningful practical improvement that could enable more practical attacks in real federated learning scenarios.
- **Technical achievement at scale**: Demonstrating gradient inversion attacks on 224×224 resolution with batch sizes up to 60 extends the capability boundary of GIAs, which prior work struggled to achieve.

## Weaknesses
- **Fundamentally misaligned evaluation metric undermines comparisons**: The paper dismisses standard GIA metrics (SSIM, PSNR, MSE) and instead uses Inception-v3 classification accuracy on reconstructed images. However, this changes the evaluation from "can an attacker recover private data" to "can an attacker generate plausible-looking images." High Top-1 accuracy (0.88) could result from generating recognizable human faces without recovering any actual private individual's image. This makes the comparison against DLG, GI, Fishing, and GIAS (all showing 0.0) questionable—either the baselines were improperly tuned, or the evaluation framework is measuring something fundamentally different.
- **Gradient normalization technique is improperly cited**: The paper cites Xu et al. (2019), Xiong et al. (2020), and Yang et al. (2019) for gradient normalization, but these papers address Layer Normalization in transformer architectures, not gradient-based optimization. This is a significant technical error.
- **Unfair baseline comparison conditions**: All baseline methods are evaluated at batch=1 while FGL's claimed advantage is large-batch performance. Without baselines evaluated at matching batch sizes (e.g., 30 or 60), the "state-of-the-art" claim at larger scales cannot be verified.
- **Batch size 60 performance is weak**: Top-1 accuracy drops from 0.88 (batch=5) to 0.483 (batch=60)—a 45% degradation. The paper presents this as a success without adequately acknowledging or explaining this significant performance cliff.
- **Hyperparameter underspecification**: The joint loss weights α₁, α₂, α₃ are presented as hyperparameters without any guidance on selection, sensitivity analysis, or ablation showing why this specific combination is optimal.

## Nice-to-Haves
- **Pixel-level reconstruction metrics**: Even alongside the proposed feature-based metrics, adding SSIM/PSNR would enable comparison with traditional GIA evaluation protocols and help readers assess actual privacy leakage severity.
- **GIAS comparison clarification**: The paper claims GIAS "did not demonstrate effective attacks" but provides no direct comparison or analysis explaining how FGL differs from or improves upon this closely related generative approach.
- **Distribution shift robustness analysis**: Testing StyleGAN priors from different source distributions (e.g., LSUN, MetFaces) would demonstrate practical attack feasibility when attackers don't have perfectly aligned generative models.

## Novel Insights
The paper's primary contribution is demonstrating that combining multiple gradient matching losses (L1, L2, Cosine) with StyleGAN priors and multi-seed optimization can enable efficient gradient inversion at higher resolutions and batch sizes than prior work. The insight that jointly optimizing with multiple distance metrics helps the optimization escape local optima is reasonable, though the empirical validation relies heavily on the controversial Inception-v3 accuracy metric. The proposed approach represents an incremental but practically useful advance in attack efficiency rather than a fundamentally new attack paradigm.

## Potentially Missed Related Work
- **Li et al. "Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage"**: Directly relevant as another GAN-based gradient inversion attack; direct comparison would strengthen the paper.
- **Zhu & Blaschko "R-GAP: Recursive Gradient Attack on Privacy" (ICLR 2021)**: Provides theoretical analysis of gradient inversion that could contextualize the proposed approach.
- **Hatamizadeh et al. "GradViT: Gradient Inversion of Vision Transformers"**: Demonstrates GIA on ViTs; discussion of architecture generalizability would strengthen the paper.
- **Struppek et al. "Plug & Play Attacks" (PPA)**: The paper cites this for Poincaré distance but could more deeply engage with PPA's approach to model inversion using GAN priors.

## Suggestions
1. **Add SSIM/PSNR comparisons** alongside the proposed metrics to enable fair evaluation against traditional GIA benchmarks. Present both metrics transparently even if performance is lower—this would provide readers with a complete picture of reconstruction quality.
2. **Evaluate baselines at larger batch sizes**: Run GI, Fishing, and GIAS at batch sizes 10, 30, and 60 under identical StyleGAN-pretrained conditions to validate FGL's advantage in scaling.
3. **Fix gradient normalization citations**: Either cite appropriate references for the gradient normalization technique or remove the component and re-run experiments.
4. **Report per-ablation variance**: The jump from Top-1=0.0 to 0.60 with Mgrad is extreme. Run each ablation with 5+ random seeds and report mean ± std to establish statistical confidence.
5. **Clarify CelebA dataset usage**: The paper states "CelebA face dataset (1000 classes, 224 × 224 px)" but CelebA has ~10,177 identities and 178×218 native resolution. Specify the exact experimental setup including which identities/classes were used and preprocessing details.

---

## oYSsbY3G4o

- GT: Accept (Poster) (avg 6.4)
- Predicted: Accept (8.0/10)
- Match: Yes

### Final Review

## Summary
GQT (Graph Quantized Tokenizer) proposes a novel graph tokenization approach that combines multi-task graph self-supervised learning (DGI + GraphMAE2) with Residual Vector Quantization (RVQ) to create discrete hierarchical tokens. These tokens encode local interactions via a GNN encoder, allowing a standard Transformer to focus on long-range dependencies. The method achieves state-of-the-art performance on 20 out of 22 benchmarks across homophilic, heterophilic, and large-scale datasets while delivering substantial memory reduction (up to 30-fold).

## Strengths
- **Strong empirical evaluation**: Comprehensive experiments across 22 diverse benchmarks covering long-range (LRGB), homophilic, heterophilic, and large-scale datasets with proper error bars over 5 runs.
- **Memory efficiency with quantified gains**: The 30-fold memory reduction on ogbn-products is significant for practical deployment, and the paper provides concrete numerical comparisons rather than vague claims.
- **Thoughtful combination of SSL objectives**: Using DGI (contrastive) + GraphMAE2 (generative/distillation) + commitment loss creates a multi-task framework well-suited for learning robust representations without labels.
- **Clean pipeline design**: Decoupling tokenizer training from Transformer training via discrete tokens enables efficient vanilla Transformers to operate on graphs without architectural modifications.
- **Public implementation**: Code availability at the provided GitHub link enhances reproducibility.

## Weaknesses
- **No cross-domain transfer experiments**: The paper claims "robust and generalizable" tokens and positions GQT as a step toward Graph Foundation Models, yet all experiments train the tokenizer separately on each dataset. This fundamental gap undermines the foundational model aspirations—no evidence shows tokens transfer across graphs.
- **Missing codebook utilization analysis**: Standard for vector quantization work, yet the paper never reports codebook usage rates, entropy, or whether codebook entries collapse. This makes it impossible to verify whether RVQ is functioning effectively or producing degenerate representations.
- **Limited ablation scope**: Table 6 shows ablation only on ogbn-arxiv. While Table 8 in the appendix covers Minesweeper, the claim that "every component contributes to downstream performance" lacks broad validation across graph types (homophilic vs heterophilic, small vs large scale).
- **Semantic edge hyperparameter sensitivity**: Semantic edges use KNN with k neighbors (values like 5, 10 in Table 7), but the paper never ablates or discusses sensitivity to this hyperparameter, leaving the design choice unjustified.
- **No attention analysis for long-range claims**: The paper asserts the Transformer can "focus on long-range dependencies" via semantic edges and PPR sequences, but provides no visualization or quantification of attention patterns to substantiate this.

## Nice-to-Haves
- **Cross-dataset transfer learning experiment**: Demonstrate pre-training tokenizer on one graph and fine-tuning on another to validate generalization claims.
- **Codebook analysis**: Report token frequency distributions, code utilization rates, and entropy to verify RVQ effectiveness.
- **Flat VQ vs hierarchical RVQ ablation**: The paper never ablates the number of codebooks (fixed at 3 across all experiments), leaving the benefit of hierarchical quantization unvalidated.
- **Attention visualization**: Heatmaps or statistics showing cross-node attention patterns to verify long-range dependency claims.
- **Comparison with GRAFF/GRIT**: Recent strong baselines for graph representation learning that appear missing from comparisons.

## Novel Insights
GQT introduces the concept of graph tokenization via hierarchical residual vector quantization trained with graph-specific self-supervised objectives. The key insight is that discrete quantization of GNN-learned representations can simultaneously (1) compress graph information into compact tokens enabling efficient Transformers, (2) provide regularization that improves robustness to adversarial attacks compared to reconstruction-based VQ (RQ-VAE), and (3) decouple local structure encoding (via GNN + VQ) from global dependency modeling (via Transformer). The combination of semantic edges (feature-space KNN) with structural serialization (PPR-based sequences) offers a practical mechanism to expose Transformers to long-range interactions that may not exist in the original graph topology, particularly beneficial in heterophilic settings.

## Potentially Missed Related Work
- **GRAFF/GRIT**: Recent strong graph representation learning baselines that should be included for comprehensive "state-of-the-art" claims.
- **VCR-Graphormer on more datasets**: The paper shows "-" for VCR-Graphormer on several datasets (CoraFull, CiteSeer, Squirrel, Chameleon, Questions), yet VCR-Graphormer is a directly comparable tokenizer-based approach and should be evaluated consistently.
- **Flat VQ baselines**: VQ-GNN (Ding et al., 2021) for comparison on the quantization approach.

## Suggestions
1. Add at least one cross-dataset transfer experiment (e.g., pre-train tokenizer on ogbn-arxiv, fine-tune on Cora) to validate generalization claims.
2. Report codebook utilization metrics (usage rates, entropy) in a supplementary table or ablation.
3. Ablate the number of RVQ codebooks (try 1, 2, 3, 6) on one dataset to validate the hierarchical quantization design.
4. Include attention visualization or statistics showing what the Transformer attends to, to substantiate the long-range dependency claims.

---

## n2NidsYDop

- GT: Accept (Oral) (avg 8.7)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper provides the first theoretical analysis of training one-layer transformers with chain-of-thought (CoT) reasoning to solve the k-parity problem. The authors establish three main results: (1) parity cannot be learned efficiently without intermediate supervision in the finite-sample setting, (2) teacher forcing enables learning in a single gradient update from zero initialization, and (3) self-consistency checking with augmented data enables learning in logarithmic steps without teacher forcing. The work extends prior RNN analysis (Wies et al., 2023) to transformer architectures with softmax attention.

## Strengths
- **Technically rigorous**: The paper provides complete proofs for all three theorems in the appendices, with careful gradient analyses and concentration bounds. The extension of parity hardness to finite samples (Theorem 2) meaningfully strengthens prior population-level results.
- **Strong theoretical separation**: The contrast between impossibility without CoT (Theorem 2) and efficiency with CoT (Theorems 5 and 7) establishes a clear and compelling separation that validates why intermediate supervision helps.
- **Novel analysis of end-to-end CoT training**: Theorem 7's treatment of self-consistency checking with augmented data provides theoretical grounding for empirical CoT practices, extending beyond expressivity results to actual training dynamics.
- **Informative experiments**: The numerical experiments clearly demonstrate phased optimization behavior and validate theoretical predictions across k values, with the direct model confirming the impossibility result.

## Weaknesses
- **Theorem 5 overclaims "one gradient update"**: The abstract states the model "can learn parity in one gradient update." The theorem actually shows one gradient step from zero initialization reduces loss to O(d^(-ε/8)), but this is directional improvement rather than convergence to a solved problem. The phrasing suggests stronger convergence guarantees than what is proven. This is not a fatal flaw since the proof sketch shows the gradient analysis, but the framing should be more modest.
- **Weight quantization unexplained in Theorem 7**: The requirement for quantizing weights to integers is essential to the proof but entirely unaddressed in the paper. The experiments don't use quantization, creating a gap between theory and practice. The paper should either justify this assumption theoretically, demonstrate it's not essential, or acknowledge it as a limitation.
- **Missing test-time generalization analysis**: The theory analyzes training loss behavior (gradient direction at initialization for Theorem 5), but no experiment shows the learned weights generalize to unseen test inputs. While Theorem 5 provides a bound on test loss, the experiments only show training loss curves.

## Nice-to-Haves
- **Scaling experiments**: All experiments use d=64 with n=100K. Varying d and n to validate theoretical sample complexity scaling would strengthen empirical validation, though this is not critical given the theoretical nature of the paper.
- **Random initialization discussion**: All proofs assume W^(0)=0, exploiting the uniform attention structure at initialization. The paper should discuss whether results extend to standard random initialization.
- **Clearer self-consistency terminology**: The "self-consistency" mechanism (filtering based on augmented data informativeness) differs from Wang et al.'s majority voting approach. While the paper cites both correctly, more explicit distinction would prevent reader confusion.

## Novel Insights
The paper's core insight is that stepwise reasoning in transformers emerges naturally from optimizing the hierarchical decomposition of complex tasks. The phased optimization phenomenon—where each level of the parity tree activates sequentially during training—provides a compelling mechanistic explanation for how CoT abilities might arise. The analysis of how errors propagate exponentially without teacher forcing but can be controlled through consistency checking offers principled guidance for CoT training. The theoretical characterization of why augmented data enables self-consistency (filtering out uninformative intermediate states) connects abstract mechanism to practical training techniques.

## Potentially Missed Related Work
- **Li et al. (2024a)** is appropriately cited but the paper could benefit from more detailed comparison regarding the distinction between in-context learning and explicit training with CoT chains.
- **Lightman et al. (2024)** ("Let's Verify Step by Step") is cited but the paper could strengthen the discussion of how process supervision in large-scale experiments relates to the theoretical teacher forcing analysis.

## Suggestions
1. **Clarify Theorem 5's claim**: Change "can learn parity in one gradient update" to "one gradient update from zero initialization significantly reduces loss" or similar to avoid overclaiming convergence.
2. **Address weight quantization**: Either justify why integer quantization is essential (and why experiments work without it), or relax this assumption in the analysis.
3. **Add test generalization experiments**: Show accuracy on held-out test samples after training, not just training loss curves.
4. **Discuss random initialization**: Acknowledge that W^(0)=0 is a theoretical convenience and whether results extend to standard initialization schemes.

---


# Summary

Papers: 44 | Accuracy: 84.1%
