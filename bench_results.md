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
## 5IkDAfabuo

- GT: Accept (Oral) (avg 7.5)
- Predicted: Accept (8.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes Prioritized Generative Replay (PGR), a framework that augments online RL replay buffers by conditioning a diffusion model on a "relevance function" F to guide synthetic data generation toward more learning-relevant transitions. The key empirical finding is that curiosity-based relevance functions outperform reward or TD-error-based conditioning, improving sample efficiency across state-based and pixel-based domains. The paper demonstrates that this guidance reduces Q-function overfitting and enables better scaling with synthetic-to-real data ratios.

## Strengths
- **Comprehensive empirical validation**: Experiments span DMC (state and pixel), OpenAI Gym, and DMLab environments with comparisons against model-free (SAC, REDQ, DRQ-V2), model-based (MBPO, DREAMER-V3), and generative baselines (SYNTHER), showing consistent improvements.

- **Instructive mechanism analysis**: The paper effectively diagnoses why curiosity-PGR works through dormant ratio metrics (quantifying overfitting reduction) and curiosity score distribution evolution, showing guidance improves diversity in synthetic data.

- **Informative ablation studies**: The comparison between PGR and PER (Section 5.1, Figure 3a) clearly demonstrates that generating new data (densification) complements prioritization—neither alone achieves PGR's performance.

- **Practical scaling insights**: Analysis showing PGR can leverage larger networks and higher synthetic ratios (up to 0.75) while SYNTHER cannot provides valuable practical guidance for data-efficient training.

- **Clear problem framing**: The "densification + guidance" conceptual framework elegantly unifies uniform replay and prioritized experience replay through the lens of conditional generation.

## Weaknesses
- **Missing guidance scale value**: The CFG guidance scale ω is treated as a hyperparameter but its value is never disclosed, limiting reproducibility. This is particularly critical since CFG directly controls how aggressively generation deviates from unconditional behavior.

- **Incomplete hyperparameter ablations**: Key design choices—guidance scale ω, conditioning dropout probability (0.25), top-k selection ratio, and inner loop frequency—are not systematically ablated. Given these are central to the method, their absence weakens confidence in the claims.

- **Limited failure mode analysis**: The paper shows reward-PGR underperforms unconditional generation but doesn't deeply analyze why. Understanding when guidance hurts (e.g., mode collapse, overly narrow coverage) would strengthen the contribution.

- **Missing direct comparison**: The claim that PGR "goes beyond" PER is not fully validated. A baseline combining curiosity-based prioritization (PER) with unconditional generation would isolate whether benefits come from guidance or from simply generating more diverse data.

- **No theoretical justification**: The paper lacks formal analysis connecting curiosity-based conditioning to overfitting reduction or improved coverage, relying solely on empirical evidence.

## Nice-to-Haves
- Ablation on what aspect of curiosity-based conditioning matters (state novelty vs. action novelty vs. full ICM dynamics error)
- Evaluation on longer-horizon or procedurally generated tasks where unconditional generation more clearly fails
- Concrete examples comparing generated transitions under different relevance functions
- Formal connection to diversity-weighted sampling or PAC-style coverage guarantees

## Novel Insights
The paper's central insight—using curiosity-based relevance functions as conditioning signals for generative replay, rather than reward or TD-error—addresses a key limitation of both uniform and prioritized replay. By demonstrating that the dynamics prediction error from ICM provides a better conditioning signal than value estimates, the authors reveal that exploration-like signals can guide generative models toward more diverse, learning-relevant synthetic data. The dormant ratio analysis provides compelling evidence that this diversity specifically mitigates Q-function overfitting to synthetic transitions, offering a mechanistic explanation for why guidance matters beyond mere data quality improvements.

## Potentially Missed Related Work
- **GRAC** (Shan et al., 2021): Attention-based conditional generation for RL could provide an alternative conditioning mechanism worth comparing
- **Model-based RL with learned constraints** (e.g., CONTRASTS, PILCO variants): These approaches use learned models with uncertainty for data generation but may not be directly applicable to the replay augmentation setting
- None identified that substantially change the paper's positioning

## Suggestions
To strengthen the paper, the authors should report the guidance scale ω value and conduct at least a minimal sweep (e.g., ω ∈ {1.5, 3.0, 5.0}) to demonstrate robustness. Additionally, adding a PER + unconditional generation baseline would better isolate whether the benefit comes from guidance or from the generation component itself.

---

## rPup1cWk4d

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes a novel interpretable data augmentation method that combines energy-based modeling with information geometry, embedding structured data into a statistical manifold via the log-linear model on posets. The core contribution is the "backward projection" algorithm that reverses dimension reduction by projecting onto locally-constructed sub-manifolds using k-nearest neighbors. The method claims competitive performance with autoencoders while offering transparency and interpretability.

## Strengths
- **Novel algorithmic contribution**: The backward projection algorithm (Algorithm 4.1) provides a geometrically intuitive approach to inverting dimension reduction, which could have applications beyond this specific framework.
- **Strong theoretical foundation**: The paper builds on established information geometry concepts (dually-flat manifolds, Bregman divergences, m/e-projections) and connects them to data augmentation in a principled way.
- **Interpretability via explicit construction**: Unlike black-box neural network methods, the many-body approximation sub-manifolds (Equation 1) provide clear semantic meaning (ℓ-body captures ℓ-mode interactions), enabling principled choice of latent dimensionality.
- **Thoughtful ablation studies**: The sensitivity analysis on bandwidth and nearest neighbor count, and the necessity of dimension reduction experiments provide useful insights into method behavior.

## Weaknesses
- **Broken autoencoder baselines severely undermine comparative claims**: Table 3 shows autoencoder achieving 20.80% accuracy on an 11-class problem (Musk) and 21.80% on a 2-class problem (Sonar)—both at or below random chance. These are not valid baselines. The paper's claim of "competitive performance" is meaningless when comparing against non-functional baselines. The autoencoder architecture (2+2 layers) appears underconfigured and/or improperly trained.

- **Experimental scope is insufficient for ICLR standards**: The primary evaluation uses MNIST with only 1,000 training samples (200 per digit). Original accuracy of 68.12% for the autoencoder and 75.37% for the proposed method on 10-class classification is far below standard MNIST benchmarks. More diverse benchmarks (CIFAR-10, larger training sets) are needed to demonstrate real-world utility.

- **Algorithm 4.1 lacks reproducibility**: While Remark 4.1 hints at construction via "fixing some indexes," the Sub-Manifold and Projection operations are described too abstractly for independent replication. Specific details on how D is constructed from nearest neighbors and what projection method is used are missing.

- **Theoretical justification for backward projection is incomplete**: The paper claims divergence-minimizing guarantees for backward projection, but does not rigorously justify why projecting onto a locally-constructed manifold D (from k-nearest neighbors) approximately inverts the forward projection onto B. The mathematical connection between forward and backward projections lacks formal analysis.

- **Normalization embedding loses information for images**: The embedding φ (normalizing pixels to probabilities) destroys absolute intensity information. The paper acknowledges this implicitly by noting "φ⁻¹ empirically reverses the average of original scaling," but this crude approximation is not analyzed for information loss.

## Nice-to-Haves
- Compare against standard data augmentation techniques (CutMix, MixUp, geometric transforms) rather than only autoencoders.
- Add quantitative fidelity metrics (FID, Inception Score, precision-recall curves) beyond visual inspection.
- Evaluate with neural network classifiers (CNNs, transformers) to show augmentation helps modern architectures.
- Systematically ablate the tensor reshaping choice (28×28 → 7×2×2×7×2×2) and justify this design decision.
- Provide convergence/optimality analysis for backward projection.

## Novel Insights
The paper's most valuable insight is treating datasets as probability distributions on a statistical manifold, providing a "meta-learning" perspective on data. The backward projection algorithm offers a novel geometric approach to inverting dimension reduction—finding approximate inverses by exploiting local manifold structure. The connection between many-body approximations and interpretable latent dimensions (where ℓ corresponds to ℓ-mode interactions) is principled and potentially valuable for applications requiring explainability. The observation that "pseudo-non-linear" behavior emerges from linear projections in curved statistical manifolds is interesting but incompletely characterized.

## Potentially Missed Related Work
- **Geometry-aware data augmentation**: Methods like Geometry-Aware Transform (GAT) or manifold-based augmentation techniques that interpolate in learned spaces
- **Diffusion models for augmentation**: Recent work on using score-based models for data augmentation with theoretical guarantees
- **Interpretable generative models**: Methods specifically designed for white-box data augmentation (beyond the autoencoder/GAN focus)

## Suggestions
**Fix the autoencoder baselines first**: Before claiming competitive performance, ensure the autoencoder achieves reasonable accuracy (at least above random chance for all classes). Use standard architectures with proper regularization and training procedures. If the autoencoder truly cannot perform on these datasets, this itself is an interesting finding that should be analyzed rather than hidden.

**Expand empirical evaluation**: Include standard augmentation baselines and modern benchmarks to make the contribution meaningful to the community.

---

## LbgIZpSUCe

- GT: Accept (Spotlight) (avg 7.3)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes MRDS-IR, a probabilistic generative model for multi-region neural dynamics that combines nonlinear within-region dynamics (parameterized by deep neural networks) with linear communication channels parameterized by their impulse response functions. The key methodological contribution is enabling interpretable characterization of inter-area communication through linear systems theory while retaining expressive nonlinear local dynamics, alongside a state-noise inversion-free variational filtering algorithm for efficient inference. The approach is validated through ground truth recovery experiments, reverse engineering of trained RNNs, rhythmic timing tasks, and real V1/V2 recordings.

## Strengths
- **Strong theoretical foundation**: The paper elegantly bridges expressive nonlinear dynamics with interpretable linear communication channels by leveraging established linear systems theory (impulse responses, transfer functions, state-space realizations). The mathematical derivation connecting impulse responses to LDS realizations (Section 2.1) is sound and well-presented.

- **Technically novel inference algorithm**: The state-noise inversion-free variational filtering approach, derived through the Bayesian learning rule, provides a principled and computationally tractable method for hybrid stochastic/deterministic state-space models.

- **Comprehensive experimental validation**: The paper tests the approach across diverse scenarios—matched ground truth recovery (Section 3.1), reverse engineering of trained RNNs for a distributed integration task (Section 3.2), rhythmic timing with output gating (Section 3.3), and real V1/V2 recordings (Section 3.4)—demonstrating consistent performance gains over multiple baselines.

- **Meaningful interpretability**: The rhythmic timing task (Figure 3) demonstrates that learned channel frequency responses reveal passband structure matching the frequency range of the task, providing genuinely interpretable insights about inter-area communication that could inform neuroscientific hypotheses.

- **Competitive real-world performance**: Quantitative comparisons on V1/V2 data (Figure 4E-F) show MRDS-IR consistently achieves lower MSE than MRM-GP, DLAG, and LN/NL baselines across stimuli conditions.

## Weaknesses
- **Incomplete ablation isolating model components**: The paper lacks experiments testing whether the *combination* of nonlinear local dynamics + temporal (non-instantaneous) channels outperforms simpler combinations. The NL baseline (nonlinear dynamics + instantaneous channels) and LN baseline (linear dynamics + temporal channels) are only compared in V1/V2 results (Figure 4E), not in the synthetic experiments where ground truth would enable precise attribution of gains.

- **Missing comparison to MR-SDS**: Table S1 lists MR-SDS (Karniol-Tambour et al., 2022) as existing work but this is never used as a baseline. Given MR-SDS also handles nonlinear dynamics with multi-region communication, this is the most directly relevant comparison missing from the paper.

- **V1/V2 validation lacks neuroscientific depth**: The V1/V2 analysis (Section 3.4) shows interpretable oscillatory structure consistent with drifting gratings and feedforward/feedback asymmetry, but does not connect findings to established neuroscience literature or validate interpretations against known anatomical properties. The supplementary DLAG comparison (Figure S1) is only qualitative. Without stronger validation against domain knowledge, claims about "meaningful features" remain suggestive rather than confirmed.

- **Identifiability not addressed**: The model combines nonlinear dynamics f_k with linear channels H_{k,ℓ}—a highly non-identifiable combination where different parameterizations could theoretically produce identical observations. The paper does not discuss whether recovered parameters are unique or demonstrate that multiple random initializations converge to similar solutions.

- **Hyperparameter selection not justified**: Latent dimensions (L_k) and filter order (M) are chosen without systematic ablation. Section 3.2 uses M=1 while Section 3.3 uses M=2, but sensitivity to these choices is not explored. The Spark Finder correctly notes that different hyperparameter settings could yield substantially different channel estimates.

## Nice-to-Haves
- **Initialization sensitivity analysis**: With deep neural networks for f_k and many channel parameters, solution quality may depend on initialization. Figure S3 shows 8 seeds with varying ELBO, but the paper does not quantify agreement between channel estimates across seeds or identify whether low-ELBO solutions have qualitatively different interpretations.

- **Single-trial predictions for V1/V2**: Figure 4A shows trial-averaged trajectories, which can mask phase variability across trials. Showing single-trial predicted vs. actual neural activity would provide a clearer assessment of model quality.

- **Pole-zero analysis for V1/V2 channels**: Figure 3G shows frequency responses for synthetic rhythmic data, but equivalent analysis for real V1/V2 channels is absent. This would strengthen interpretability claims.

- **Practical guidance on hyperparameters**: The paper would benefit from recommendations on choosing L_k and M based on data characteristics, initialization strategies, and potential failure modes.

## Novel Insights
The paper's core insight—combining nonlinear within-region dynamics parameterized by neural networks with linear communication channels parameterized by their impulse responses—offers a principled middle ground in the expressivity-interpretability tradeoff space. By grounding channel communication in linear systems theory, the approach provides direct access to temporal properties (delays, frequency filtering) through impulse responses and pole-zero analysis, which could enable neuroscientists to formulate and test specific hypotheses about inter-area communication structure. The state-noise inversion-free filtering algorithm further provides a practical tool for tractable inference in these hybrid models.

## Potentially Missed Related Work
- **MR-SDS (Karniol-Tambour et al., 2022)**: Listed in Table S1 but never used as a comparison baseline; this is the most directly comparable existing method with nonlinear dynamics and multi-region communication.
- **Nonlinear LDS variants**: Approaches like Hernandez et al. (2018) on nonlinear evolution via spatially-dependent linear dynamics may be relevant for related methodology.

## Suggestions
1. **Add ablation experiments** in synthetic tasks (integration, rhythmic timing) that isolate: (a) benefit of temporal vs. instantaneous channels, and (b) benefit of nonlinear vs. linear local dynamics, to attribute performance gains precisely.

2. **Include MR-SDS as a baseline** in at least one synthetic experiment where ground truth enables fair comparison of interpretability.

3. **Strengthen V1/V2 validation** by correlating inferred feedforward/feedback signals with known anatomical connectivity or predictions from established neuroscience literature.

4. **Address identifiability** either through empirical validation (multiple random initializations showing convergence) or theoretical discussion of uniqueness conditions.

---

## CTC7CmirNr

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper reveals that masked diffusion models (MDMs) are theoretically equivalent to masked models by proving that the time variable is unnecessary for both training (via ELBO reformulation) and sampling (via the First-Hitting Sampler). The authors identify a critical numerical precision issue in 32-bit Gumbel-based categorical sampling that caused prior work to report inflated performance metrics for MDMs. After fixing this issue, the paper shows MDMs significantly lag behind auto-regressive models (ARMs) in generative perplexity.

## Strengths
- **Rigorous theoretical contribution**: All four propositions (3.1, 3.2, 4.1, 5.2) are mathematically sound, with complete proofs provided in the appendix. The ELBO reformulation by number of masked tokens and the characterization of the optimal time-agnostic model are well-established.
- **Novel algorithmic innovation**: The First-Hitting Sampler (FHS) is a genuine advance that analytically samples transition times, achieving up to 20× wall-clock speedup under specific configurations while maintaining theoretical equivalence to the original reverse process.
- **Important empirical finding**: Identifying the truncated Gumbel numerical precision issue (Section 5) is a significant contribution that exposes a flaw in previous benchmarking methodology. The closed-form analysis explaining how truncation reduces token diversity is theoretically elegant and practically important.
- **Comprehensive experimental validation**: The paper provides extensive ablation studies comparing training variants, sampling strategies, and numerical precision effects. The design of the ablation experiments (including the "truncated 64-bit" control) convincingly isolates the precision issue.
- **Honest limitation discussion**: The authors appropriately acknowledge that text's left-to-right structure favors ARMs, that KV caching incompatibility poses deployment challenges, and that MDMs may retain value for order-agnostic data.

## Weaknesses
- **Limited domain scope**: All experiments focus exclusively on text generation. The paper's negative conclusions about MDMs vs. ARMs, as well as the numerical precision findings, have not been validated for other discrete domains (images, code) where MDMs are also applied. This limits the generalizability of the findings.
- **Configuration-dependent speedup**: The 20× speedup claim holds for the paper's specific setup (vocabulary size 50,526, sequence length 1024, ~600M parameters), but the paper's own analysis in Appendix J.3 shows this can drop to 5× or even 1.07× in other settings (e.g., DiffSound). This nuance deserves more prominent placement.
- **Baseline comparison concerns**: The ARM baseline is trained for half the iterations of MDMs, which may partially explain the perplexity gap. Additionally, comparing encoder-only MDMs against causal ARMs introduces architectural confounding factors that are not fully addressed.
- **Missing masked model comparisons**: The paper theoretically connects MDMs to masked models but does not empirically compare against approaches like MaskGIT or BERT-based generative models to validate the practical sufficiency of simpler masked models.
- **KV caching impact not quantified**: The paper identifies bidirectional attention's incompatibility with KV caching as a fundamental deployment challenge but provides no quantitative analysis of how this affects inference cost at scale compared to ARMs with KV caching enabled.

## Nice-to-Haves
- **Image domain experiments**: Testing whether the numerical precision issue and equivalence claims extend to image generation (where MDMs like MaskGIT are widely used) would substantially strengthen generalizability claims.
- **Scaling experiments**: All conclusions rest on a single 170M model. Evaluating at 1B+ scale would strengthen the claim that "MDMs cannot truly beat ARMs" before the community takes this conclusion seriously.
- **Human evaluation**: The paper relies on Gen PPL and entropy as quality proxies. Human evaluation at matched perplexity levels would clarify whether the measured gaps are perceptible in practice.
- **Retrospective baseline correction**: Applying the 64-bit precision fix to prior published MDM baselines (SEDD, MDLM) and reporting corrected Gen PPL numbers would more comprehensively establish the field's current state of knowledge.

## Novel Insights
This paper provides three genuinely novel observations that advance understanding of discrete generative models. First, the theoretical proof that MDMs reduce to masked models with the time variable replaced by masked ratio provides a unifying framework that clarifies the relationship between these previously distinct approaches—the Beta distribution insight showing time as a continuous relaxation of masked ratio is particularly elegant. Second, the identification that 32-bit Gumbel truncation creates a temperature-lowering effect that was previously misattributed to MDM superiority is a crucial correction to the literature that will impact how future discrete diffusion work is evaluated. Third, reframing the sampling process as a first-hitting time problem enables analytical solutions that avoid expensive categorical sampling across all positions simultaneously—a practical insight that connects theory to efficient implementation.

## Potentially Missed Related Work
- **MaskGIT (Chang et al., 2022)**: As a leading masked model for image generation, direct empirical comparison with MDMs under the corrected evaluation protocol would strengthen the claim that "simpler masked models are sufficient."

## Suggestions
- Move the configuration-dependence analysis of the 20× speedup from Appendix J.3 to the main paper body, as this is crucial context for interpreting the practical significance of the FHS contribution.
- Retrain ARMs with identical training compute (not half iterations) for the comparison in Section 6 to ensure the perplexity gap is not partly attributable to training inequality.
- Include a quantitative analysis of KV caching impact comparing ARMs with KV caching vs. MDMs with FHS on long-context tasks, as this is the paper's strongest argument for why masked models face fundamental deployment challenges.

---

## gInIbukM0R

- GT: Reject (avg 2.5)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary

This paper introduces a quantitative framework for measuring "Emergence" in neural networks by counting paths from inactive to active nodes, and proposes "relative Emergence" (normalized by parameter count) to explain training dynamics and pruning effects. The authors claim that tracking this metric during training can predict when networks are nearing convergence. The work builds on the categorical framework of Li et al. (2023a) and validates it on MLP/CNN experiments with MNIST, Fashion-MNIST, and CIFAR-10 datasets.

## Strengths

- **Novel conceptual framing**: The idea of quantifying emergence through network topology (specifically, path counts between active and inactive nodes) is conceptually interesting and addresses an underexplored area in understanding neural network training dynamics.
- **Practical prediction insight**: The observation that Emergence approaching zero may signal when a network has reached its learning potential has practical value for early stopping decisions, even if preliminary.
- **Theoretical ambition**: The paper attempts to connect categorical/quiver representation theory to practical network analysis, which is an interesting interdisciplinary direction.
- **Multi-architecture validation**: Experiments cover MLPs and CNNs (VGG19) across three datasets, providing breadth in architectural validation.

## Weaknesses

- **Fundamental conceptual ambiguity**: The paper uses "Emergence" to mean at least three distinct things: (1) the mathematical path-counting metric, (2) network complexity/scale, and (3) trainability/efficiency. Section "Emergence Increases with Scale" discusses absolute Emergence (complexity), while "Relative Emergence Correlates with Training Performance" discusses a different concept (efficiency). This confusion undermines the paper's coherence and makes core claims difficult to evaluate.

- **Arbitrary threshold without justification**: The active/inactive node threshold of 0.05 is never justified theoretically or empirically. No sensitivity analysis is provided to show how results change with different thresholds (0.01, 0.1, etc.). This is a critical hyperparameter that fundamentally determines the Emergence value but is treated as an unexamined constant.

- **Disconnect between theory and practice**: The appendix develops a complex categorical framework using quiver representations and derived functors, but the main text applies a simple path-counting formula without explaining the connection. The theory appears to be inspiration rather than a genuine derivation of the proposed metric. No justification is given for why "paths from inactive to active nodes" specifically captures emergent behavior.

- **Insufficient empirical rigor for key claims**: The paper makes strong claims about correlation between Emergence and performance, but:
  - Only 5 epochs are used in the main pruning experiments—far short of convergence
  - No statistical significance tests, confidence intervals, or error bars are reported anywhere
  - Final accuracy differences are minimal (e.g., 95.7% vs 95.1% for MNIST non-pruned vs 70% pruned)
  - No correlation coefficients or quantitative measures of relationship strength are provided

- **Missing test set evaluation**: All reported metrics are training accuracy. Claims about "network performance" and "learning capability" are made without any held-out evaluation, making it impossible to assess generalization.

- **Limited experimental scale**: The paper claims general implications for neural networks, but experiments are limited to simple tasks (MNIST, Fashion-MNIST) and modest architectures. The phenomena of interest in emergence (qualitatively new capabilities) are most relevant in large-scale models, yet no experiments test ResNets, Transformers, or other modern architectures where skip connections or attention would require modifications to the path-counting logic.

- **Relative Emergence lacks theoretical grounding**: Defining relative Emergence as E/#parameters is presented as a key contribution, but this is simply a density normalization without justification for why this particular ratio captures "trainability" or why it should predict learning efficiency.

## Nice-to-Haves

- Ablation study on the active/inactive threshold (0.01, 0.05, 0.1, 0.5) to validate robustness
- Comparison with simpler baselines (e.g., does parameter count alone predict final accuracy?)
- Proper convergence studies extending training to completion rather than arbitrary epoch cutoffs
- Statistical tests with multiple random seeds to validate claimed trends
- Counterexample analysis showing when Emergence incorrectly predicts convergence

## Novel Insights

The paper's most genuinely useful insight is the observation that Emergence decreasing to zero appears to correlate with a network reaching its learning capacity—a potential early stopping signal. The distinction between absolute Emergence (capturing scale) and relative Emergence (capturing efficiency per parameter) also offers a useful lens for understanding the trade-off between network complexity and trainability in pruned models. However, these insights are presented with insufficient rigor to establish their validity.

## Potentially Missed Related Work

- **Neural tangent kernel (NTK) literature**: The connection between network initialization, path-counting (which resembles NTK analysis), and training dynamics could be explored more thoroughly.
- **Lottery Ticket Hypothesis (Frankle & Carbin, 2019)**: The pruning experiments connect to this work but are not explicitly discussed or compared against.
- **Sharpness-aware minimization and loss landscape geometry**: Li et al. (2018) is cited but the specific claims about local minima density and Emergence lack rigorous connection.
- **Network architecture search / efficiency metrics**: The relative Emergence concept could be contextualized against existing efficiency measures like FLOPs or parameter efficiency ratios.

## Suggestions

1. **Conduct sensitivity analysis on the active/inactive threshold** and justify the chosen value theoretically or show robustness across choices.
2. **Add proper test accuracy evaluation** throughout training to validate generalization claims.
3. **Include statistical rigor**: Report means and standard deviations over multiple random seeds, perform significance tests, and include error bars in all figures.
4. **Connect theory to practice**: Clarify how the categorical framework motivates the specific path-counting implementation, or consider presenting the empirical approach as independent inspiration rather than theoretical derivation.
5. **Compare against simple baselines**: Does training loss at epoch 1 predict final accuracy better than Emergence? If so, the complex measure adds no practical value.
6. **Run experiments to completion**: Extend training until clear convergence rather than stopping at 5-30 epochs, to properly validate the correlation claims.

---

## L9j8exYGUJ

- GT: Reject (avg 5.0)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces "distributional reasoning" as a novel framework for understanding multi-hop reasoning in LLMs, proposing that middle network layers activate a distribution of potential intermediate answers, from which a simple linear transformation predicts final answers. The authors demonstrate this phenomenon across four LLMs (Llama-2-7B/13B, Llama-3-8B, Mistral-7B) using 6,547 prompts spanning 14 question types, and introduce a hallucination dataset with fictitious subjects to show the reasoning process generalizes beyond stored knowledge. The work connects neural network reasoning to cognitive psychology's dual-process theories (spread of activation vs. propositional reasoning).

## Strengths
- **Novel conceptual contribution**: The insight that middle layers represent *distributions* over possible intermediate answers rather than single values is genuinely novel and provides a fresh interpretability lens for understanding implicit multi-hop reasoning.
- **Thorough empirical evaluation**: Testing across four models, 14 question types, and 6,547 prompts with consistent results demonstrates robustness of the phenomenon.
- **Innovative hallucination experiments**: Using fictitious subjects/attributes to isolate reasoning processes from memorized knowledge is a clever methodological contribution that provides novel evaluation methodology for internal LLM processes.
- **Interpretability value**: The method enables tracing reasoning failures to specific intermediate hops, offering practical diagnostic utility beyond theoretical insight.
- **Cognitive modeling bridge**: Connecting associative and propositional reasoning paradigms provides meaningful contributions to cognitive science connections with AI.

## Weaknesses
- **No causal evidence established**: The paper relies entirely on correlational analysis (linear regression between A1 and A2 activations). The paper acknowledges this limitation but presents "distributional reasoning" as if describing an actual mechanism rather than an observed pattern. Alternative explanations—where A1 and A2 activations are independently triggered by the subject embedding—are not ruled out.
- **R² > 0.5 threshold is modest**: The paper claims this "strong association" evidence, but R² > 0.5 means less than half the variance is explained. The remaining 50% unexplained variance is substantial and should be discussed more critically.
- **Limited structural diversity in questions**: All experiments use similar question templates ("What is X of the birthplace of Y?"), limiting claims about general reasoning mechanisms. Different prompt structures might elicit different solving strategies as the paper's own Figure 1 acknowledges.
- **Logit Lens dependency**: The entire analysis rests on Logit Lens interpretation, which the paper acknowledges "remains unclear why it works" and may have "undiscovered biases." This is a fragile foundation for strong claims.
- **Layer selection bias**: R² values are cherry-picked at "two-thirds depth" as optimal. The paper does not show variance across layers or whether this specific depth generalizes across question types. A robust phenomenon should not depend this heavily on precise layer selection.

## Nice-to-Haves
- Compare linear regression predictions against trivial baselines (e.g., predicting A2 from subject embeddings, random A1 subsets, or shuffled values) to demonstrate specificity of the transformation.
- Analyze what the learned Q2 matrices actually contain—do they have interpretable structure corresponding to actual transformations (e.g., letter-sound correspondences)?
- Test whether distributional reasoning emerges in tasks requiring logical deduction where there are no natural intermediate answers to distribute over.
- Add comparative case studies showing the same question answered via chain-of-thought vs. direct prediction to demonstrate distributional reasoning differs from explicit step-by-step reasoning.

## Novel Insights
The paper's most valuable contribution is demonstrating that multi-hop reasoning in LLMs operates through a distributional rather than deterministic mechanism in intermediate layers—multiple candidate answers are simultaneously activated in middle layers before a phase transition narrows to the final prediction. This provides a computational model connecting associative (spread of activation) and propositional (structured reasoning) cognitive paradigms within a single neural network. The observation that this process generalizes to fictitious subjects suggests it represents a learned reasoning strategy rather than memorized correlations, offering both theoretical insight into LLM reasoning and practical utility for error tracing.

## Potentially Missed Related Work
- **Elhage et al. (2021)** on transformer circuit analysis — directly relevant to understanding the mechanistic basis for the phase transition, particularly the role of feed-forward blocks
- **Geva et al. (2022)** on transformer feed-forward layers building predictions — relevant to understanding how intermediate concepts emerge in vocabulary space
- **Probe-based studies** examining whether linear representations of concepts exist in middle layers (not just Logit Lens)

## Suggestions
Perform direct causal intervention experiments (e.g., activation patching or ablation of specific A1 activations in middle layers) to establish whether the observed linear transformation from A1 to A2 is actually used by the model, rather than merely correlational. This would significantly strengthen the core claim of "distributional reasoning" as a mechanism.

---

## MFZjrTFE7h

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
D-FINE introduces a real-time object detector that redefines bounding box regression in DETR models through two innovations: Fine-grained Distribution Refinement (FDR), which iteratively refines probability distributions for edges rather than predicting fixed coordinates, and Global Optimal Localization Self-Distillation (GO-LSD), which transfers localization knowledge from deeper to shallower decoder layers. The method achieves 54.0%/55.8% AP on COCO at 124/78 FPS on NVIDIA T4 GPU, and up to 59.3% AP with Objects365 pretraining, while also enhancing various existing DETR models by 2.0-5.3% AP with negligible overhead.

## Strengths
- **Strong empirical results**: State-of-the-art real-time detection performance on COCO, with D-FINE-L achieving 54.0% AP at 124 FPS and D-FINE-X reaching 55.8% AP at 78 FPS. The method also excels on CrowdHuman (55.5% AP for D-FINE-S) and Objects365.
- **Efficient self-distillation**: GO-LSD achieves meaningful gains (+0.5% AP over FDR alone) with only 6% training time increase, avoiding the need to train or maintain a separate teacher model—a practical advantage for deployment.
- **Demonstrated generalizability**: Consistent improvements (+2.0% to +5.3% AP) when applied to Deformable-DETR, DAB-DETR, DN-DETR, and DINO, validating that FDR and GO-LSD generalize across DETR variants.
- **Well-structured ablation studies**: Table 4's stepwise progression from baseline to final model provides clear evidence for each architectural modification's contribution, and Table 5 compares distillation methods comprehensively.
- **Elegant motivation**: The insight that treating bounding box edges as probability distributions rather than fixed values enables better modeling of localization uncertainty is well-grounded in prior work (GFocal/DFL) while addressing their key limitations.

## Weaknesses
- **Incomplete ablation of FDR vs. GO-LSD**: Table 4 shows "+FDR" and "+GO-LSD" as a combined step, not separate rows. This makes it impossible to determine each component's individual contribution and whether they work synergistically or redundantly. Reviewers cannot assess whether the +0.5% AP improvement in Table 5 (from 53.8% FDR alone to 54.5% FDR+GO-LSD) justifies GO-LSD's complexity.
- **Incomplete comparison with RT-DETRv3**: Table 1 includes RT-DETRv3 models but with missing metrics ("-"). Since RT-DETRv3 is a direct architectural predecessor using similar components (same RT-DETR framework, training strategy), this is the most fair comparison. The paper cannot claim to surpass "all existing real-time detectors" without complete RT-DETRv3 numbers.
- **Runtime overhead of FDR not isolated**: FDR predicts distributions over N=32 bins for 4 edges across all decoder layers. The paper shows the net effect of architectural savings + FDR/GO-LSD (+1.0% AP, reduced FLOPs/latency) but does not quantify FDR's computational cost in isolation. This prevents understanding whether FDR adds overhead that is masked by architectural simplifications.
- **Limited scope beyond DETR**: While the paper claims FDR addresses anchor-dependency limitations of GFocal, the method is only demonstrated on DETR-based architectures. CNN-based anchor-free detectors (YOLOX, FCOS) are mentioned as incompatible, but no experiments attempt to adapt FDR for them. This limits the broader applicability claim.
- **Objects365 pretraining comparison unfair**: Table 1 shows YOLOv10 pretrained on Objects365 for 300 epochs while D-FINE uses only 21 epochs. The paper acknowledges this but does not discuss whether D-FINE's superior results (57.1%/59.3% vs 56.1%/58.3% AP) might be partially attributed to more efficient pretraining rather than pure method superiority.

## Nice-to-Haves
- Ablation using earlier decoder layers as teachers for GO-LSD to justify why the final layer is "optimal"
- Analysis of when the method saturates or fails (e.g., correlation between distribution entropy and localization quality)
- Additional dataset evaluations beyond COCO, Objects365, and CrowdHuman (e.g., LVIS, domain-specific datasets) to assess robustness
- Backbone ablation to verify FDR/GO-LSD are not dependent on HGNetv2 specifically

## Novel Insights
D-FINE presents a principled approach to bounding box regression that treats edges as iteratively-refined probability distributions rather than fixed coordinates. The key insight—using the non-uniform weighting function W(n) to enable finer adjustments near accurate predictions while allowing rapid corrections for inaccurate ones—is intuitive and well-motivated. GO-LSD's innovation of distilling from the final decoder layer's refined distributions to shallower layers, using a unified union set of Hungarian matches and decoupled weighting for matched/unmatched predictions, provides a practical self-distillation mechanism that avoids training a separate teacher model. The combination of these components with lightweight architectural optimizations results in meaningful accuracy improvements without sacrificing speed.

## Potentially Missed Related Work
- **Anchor-free distribution-based regression**: While the paper contrasts FDR with GFocal (anchor-based), related anchor-free distribution regression work such as FCOS/ATSS with uncertainty modeling could provide additional context for the localization uncertainty angle.
- **Transformer-specific distillation**: KD-DETR (Wang et al., 2024b) and DetrDistill (Chang et al., 2023) address distillation for DETR families—comparing GO-LSD against these methods would strengthen the self-distillation claims.

## Suggestions
1. Add separate ablation rows for FDR alone and GO-LSD alone in Table 4 to quantify individual contributions.
2. Complete the RT-DETRv3-L/X metrics in Table 1 for a fair architectural comparison.
3. Isolate and report FDR's computational overhead separately from the architectural simplifications to show the net efficiency tradeoff.

---

## OXIIFZqiiN

- GT: Reject (avg 1.5)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces IGCP (Image-Guided Code Patch Framework), a dual-modal approach for patch representation learning combining code patches with synthetic descriptions. The authors provide extensive mathematical scaffolding using measure theory, quantum information theory, and statistical physics, alongside empirical results on patch description generation benchmarks claiming state-of-the-art performance.

## Strengths
- The paper presents a principled multi-objective loss design (PDC, PDM, PDG components) with ablation studies demonstrating each component's contribution to overall performance
- The empirical results show meaningful improvements over baselines (e.g., +4.51% ROUGE-L over CCPGen) on a standard patch description generation benchmark with established evaluation metrics
- The phase transition analysis (Theorem 3.9) connecting learning dynamics to statistical physics offers an interesting theoretical perspective on sample complexity thresholds
- The paper attempts to bridge diverse theoretical frameworks (measure theory, information geometry, stochastic optimization) within a unified treatment for patch representation learning

## Weaknesses
- **Terminology inconsistency undermines clarity**: The framework is called "Image-Guided Code Patch Framework" in the title/abstract but "Integrated Patch-Text Model Framework" in Section 2.3. More critically, despite the "image-guided" framing, the paper processes code patches and text descriptions—never actual images or visual prompts. This is misleading and contradicts the title's promise.
- **Dual-modal claim is untested**: The abstract explicitly states IGCP "achieves state-of-the-art performance in patch description generation" and "exhibits remarkable domain generalization capabilities," yet only generative tasks are evaluated. No predictive task experiments (e.g., security patch classification) are provided, leaving the core dual-modal contribution incompletely validated.
- **Figure 2 discrepancy raises verification concerns**: The text reports IGCP achieves ROUGE-L of 26.92% and BLEU of 22.49%, but the radar chart visualization appears to show substantially lower values for IGCP. This discrepancy undermines result verification.
- **Theoretical framework lacks justification and empirical validation**: The quantum information bottleneck (Section 3.3.1) is presented without explaining why quantum formalism is necessary for classical code patches. The theorems (3.1-3.9) largely repackage standard results (RKHS theory, spectral theorem, representer theorem, SGLD convergence) without novel extensions specific to IGCP. The claimed phase transition (Theorem 3.9) and Łojasiewicz convergence (Theorem 3.7) are asserted but never demonstrated empirically.
- **Missing reproducibility information**: Section 4 omits critical details: model architecture specifications, training hyperparameters (learning rate, batch size, epochs), optimization choices, and computational requirements. The baseline implementations are not described.

## Nice-to-Haves
- Rename the framework to accurately reflect its actual modality (code patches with text descriptions) rather than the misleading "Image-Guided" framing
- Validate theoretical predictions empirically (e.g., measure the predicted phase transition by varying sample size)
- Add cross-domain generalization experiments across programming languages to substantiate the claimed domain generalization capabilities
- Include qualitative examples showing actual generated descriptions alongside baseline comparisons
- Provide comprehensive hyperparameter settings and training curves for reproducibility

## Novel Insights
While the paper's theoretical formalism is extensive, the core contribution appears to be applying contrastive learning (CLIP-style) principles to the code patch domain with a multi-objective loss combining generative, matching, and contrastive objectives. The quantum information bottleneck and free probability analysis, while mathematically sophisticated, are presented as post-hoc formalizations rather than generating novel architectural innovations or predictions. The phase transition insight connecting learning dynamics to statistical physics represents the most genuinely novel theoretical contribution, though it remains unvalidated empirically.

## Potentially Missed Related Work
- CodeT5 (Wang et al., 2021) — recent strong baseline for code understanding and generation tasks that could provide a more competitive comparison
- GraphCodeBERT (Guo et al., 2020) — for graph-based code representation approaches in baseline comparisons

## Suggestions
The authors should focus on validating their claims rather than expanding theoretical machinery: implement predictive task evaluation to support the dual-modal claim, fix the figure-text discrepancy, add statistical significance tests with confidence intervals, and provide complete architectural and training specifications for reproducibility. If the quantum/free probability formalism is retained, explicit connections to practical predictions must be established—otherwise, this content should be moved to supplementary material to maintain focus on verifiable contributions.

---

## y2ch7iQSJu

- GT: Reject (avg 2.0)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces BBsurv, a novel budgeted active learning method for censored survival data where queries can only partially "decensor" data points (reveal information incrementally via an I-oracle). The approach extends BatchBALD's mutual information framework to survival settings and reduces batch selection to a maximum coverage problem, claiming (1-1/e) approximation guarantees for both uniform and non-uniform instance costs. Empirical evaluation on three medical survival datasets (MIMIC-IV, NACD, SUPPORT) demonstrates modest improvements over adapted baselines.

## Strengths
- **Addresses a genuine gap**: The problem of budgeted active learning with incrementally decensorable survival data is novel—no prior work handles this setting where querying reveals partial information rather than complete labels. This has clear practical relevance for clinical trials operating under budget constraints.

- **Sound theoretical framing**: The reduction to maximum coverage and NP-hardness proof are well-motivated. The adaptation to non-uniform costs via budgeted maximum coverage (Section 4.2) extends the framework appropriately.

- **Practical I-oracle formulation**: The mechanism for modeling partial information gain (where queries reveal only I additional years of information) is well-defined and closely mirrors real clinical trial follow-up constraints.

- **Comprehensive experimental conditions**: Testing across three real-world datasets with both uniform and non-uniform cost settings demonstrates breadth of evaluation.

## Weaknesses

- **Theoretical guarantee is undermined**: Algorithm 1 (BBSurv) claims to be a "(1 - 1/e)-approximate algorithm" based on Khuller et al. (1999), but the paper explicitly admits to dropping two key components from Khuller's optimal Algorithm 2—the brute-force H_1 initialization and the O(n³) enumeration loop. The justification that "in deep learning models, often two or three points are not significant enough" and "instances are often very sparse" are intuitive but **unproven heuristics**. The paper cannot simultaneously claim the theoretical guarantee and admit to removing the components that provide it. This is a significant gap requiring either: (a) implementation of the full algorithm, (b) a formal proof that the simplification preserves the bound, or (c) explicit revision to state Algorithm 1 is a heuristic without the guarantee.

- **Weak comparison baselines**: The "sanity check" methods (Closest to Half, Mean Closest to Middle, Clusters for Batch) have no principled connection to information gain or mutual information. BBSurv beating these is expected, not informative. Stronger baselines would include BatchBALD without survival-specific probability adjustments, entropy-based batch selection (which is included but not properly evaluated), or variants isolating each contribution.

- **Marginal and unsubstantiated improvements**: Table 1 shows BBSurv improvements of only 0.01–0.05 MAE-PO over competitors (e.g., MIMIC +5y: 4.23 vs BatchBALD 4.28 = Δ0.05). The paper claims "significantly better" performance without statistical tests. These differences are comparable to reported standard errors (±0.01–0.02), making superiority claims unjustified without hypothesis testing.

- **No ablation study**: It's unclear which component of BBSurv drives improvement—the censoring-aware probability adjustment (Section 5), the increment-aware binning, or the greedy selection. A comparison of (a) BatchBALD without adjustments, (b) with censoring adjustment only, and (c) full BBSurv would clarify contributions.

- **MCtH consistently competitive**: The "Mean Closest to Middle" sanity check method ties or nearly ties BBSurv across nearly all settings in Table 1 and subsequent tables, yet the paper dismisses this finding without explanation. This requires analysis—either statistical evidence that BBSurv is superior, or acknowledgment that MCtH is a strong baseline in this domain.

## Nice-to-Haves
- **Statistical significance testing**: Paired t-tests or Wilcoxon tests across random seeds would properly evaluate whether observed differences are meaningful or within noise.

- **Ablation on time discretization**: The method discretizes continuous time into bins for MTLR. No experiments vary bin count to show robustness—this directly affects mutual information structure.

- **Vary initial training composition**: All experiments use exactly 100 uncensored + 900 censored. Ablation on this ratio would show robustness across different initial knowledge states.

- **Characterize failure regimes**: Analysis of when BBSurv does NOT outperform alternatives (high censorship rates, specific budget-to-pool-size ratios) would guide practical deployment.

## Novel Insights
The I-oracle mechanism provides a clean formalization of partial information gain in survival analysis, modeling the realistic scenario where clinical follow-up reveals only additional years rather than complete event times. This generalizes standard active learning (equivalent to I=∞) and provides a principled framework for reasoning about budget-constrained survival data collection. The extension to non-uniform costs via budgeted maximum coverage is a useful generalization that maintains theoretical tractability, though the practical approximation quality without the full O(n³) enumeration remains empirically validated rather than proven.

## Potentially Missed Related Work
- **Hüttel et al. (2024)** on Bayesian active learning for censored regression is cited but could be more thoroughly compared—while their method uses BALD (individual selection) rather than BatchBALD (batch selection), direct comparisons on identical datasets would strengthen the contribution.

## Suggestions
1. **Clarify the theoretical claim**: Explicitly state whether Algorithm 1 is proposed as a proven (1-1/e)-approximation or as a heuristic approximation, and provide empirical validation across many random instances if claiming the latter.

2. **Strengthen baselines**: Include BatchBALD adapted only for censoring (without increment-aware binning) and proper entropy-based batch selection as meaningful baselines, not just "sanity checks."

3. **Add statistical tests**: Perform paired comparisons with proper significance testing across all random seeds to justify superiority claims.

4. **Analyze MCtH's performance**: Investigate why "Mean Closest to Middle" performs competitively—this may reveal an insight about the problem structure worth discussing.

---

## fBSc0c1IXJ

- GT: Reject (avg 3.0)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces Remote Reinforcement Learning (RRL) with communication constraints, where an actor executes actions without direct reward access—a remote controller observes rewards and must guide the actor over a rate-limited channel. The proposed GRASP method combines channel simulation (importance sampling) to efficiently transmit action samples with behavioral cloning at the actor to establish a shared reference distribution that further reduces communication overhead. The paper demonstrates 4-258× communication reductions across diverse environments while maintaining equivalent learning performance.

## Strengths
- **Well-motivated and novel problem formulation**: RRL with communication constraints models realistic scenarios like remote robot control over wireless channels, edge computing, or human-in-the-loop systems where reward evaluation is costly or centralized.
- **Strong information-theoretic grounding**: Correctly identifies and applies channel simulation literature (Cuff 2008, Li & El Gamal 2018, Theis & Yosri 2022), showing communication cost scales with KL-divergence rather than action entropy.
- **Comprehensive empirical evaluation**: Extensive experiments across 9+ environments with varied action spaces (discrete/continuous), observation types (image/proprioceptive), and agent configurations (single/multi-agent) using multiple RL algorithms (PPO, DQN, SQ, DDPG), with 20 independent runs.
- **Practical communication savings**: The 4-258× communication reductions are substantial and meaningful for real-world deployment scenarios with bandwidth constraints.

## Weaknesses
- **Missing reward-transmission baseline experimental results**: The paper explicitly describes a baseline where "the controller conveys the reward signal to the actor" using 32 bits per timestep and claims "41 times less communication than sending the reward" (Section 8), yet no experimental results for this baseline appear anywhere in the paper. Table 1 and Table 2 only compare ASC (action source coding) against GRASP. This is a significant omission that undermines the central motivational claim—the paper argues GRASP is superior to transmitting rewards but never empirically validates this.
- **Breakout performance gap unexplained**: In Table 1, Breakout shows the largest performance gap between controller and actor (48.8 return gap, 15.2% normalized for GRASP vs 41.4 and 12.3% for ASC). This contradicts the repeated claim that "GRASP does not negatively impact training." No analysis is provided for why image-based observations lead to worse behavioral cloning performance, which is critical for understanding when GRASP succeeds versus fails.
- **No ablation studies**: The paper doesn't isolate the benefit of each component. Key questions remain unanswered: What is the单独 benefit of channel simulation versus source coding? What if the actor used a fixed random policy as Q instead of learning one? How does varying the actor's learning rate or network capacity affect the communication-accuracy trade-off?
- **Actor policy never evaluated in execution mode**: The actor learns a policy via behavioral cloning but never executes it—the paper only evaluates it after training. The text mentions "after training, the controller might transmit its learned policy to the actor, or if the actor's policy is adequate, no further communication is necessary" but provides no empirical validation of this deployment scenario, which is arguably the most practical use case.

## Nice-to-Haves
- **Theoretical convergence analysis**: While information-theoretic bounds are cited, the paper provides no convergence guarantees or sample complexity bounds for GRASP in the RRL setting. Understanding when the controller's learned policy converges and how communication constraints affect this would strengthen the contribution.
- **Noisy channel evaluation**: The method assumes perfect communication; practical wireless scenarios with packet loss or bit errors are not addressed despite the paper's motivation around wireless channels.
- **Channel simulation parameter N analysis**: Algorithm 3 uses a parameter N (number of candidate samples) that critically affects communication cost and action fidelity, yet the paper never reports what N was used or how results vary with different values.
- **KL-divergence growth analysis**: Figure 2 shows KL divergence between controller and actor policies grows during training in some environments, meaning communication cost increases over time. No analysis explains why this occurs or implications for long training runs.

## Novel Insights
The paper's core insight—applying channel simulation from information theory to the remote RL problem—is elegant and well-executed. By recognizing that communicating a *sample* from a distribution (via importance sampling) requires only D_KL[P||Q] bits rather than H(P) + D_KL[P||Q] bits for direct transmission, the authors unlock significant communication savings. The additional insight that the actor can learn a policy through behavioral cloning to serve as Q, thereby reducing communication to just the residual divergence, elegantly combines two established techniques for a novel application. The extension to multi-agent settings demonstrates scalability, though the practical deployment scenario (actor using learned policy independently) remains empirically unvalidated.

## Potentially Missed Related Work
- **Pase et al. (2022)** on rate-constrained remote contextual bandits is appropriately cited but the paper could more prominently contrast its approach (RL with sequential states and transitions) versus this prior work (independent contextual bandits), highlighting the additional challenges introduced by temporal dependencies.
- **Pujol Roig & Gündüz (2020)** on remote reinforcement learning over noisy channels—directly relevant to the wireless communication motivation but only briefly mentioned.
- **Federated RL literature** (Nadiger et al., 2019; Jin et al., 2022)—the paper mentions federated RL as related work but doesn't clearly distinguish RRL's fundamental difference: federated RL typically keeps data localized while learning a shared policy, whereas RRL has a single controller with reward access directing actors with no local RL capability.

## Suggestions
1. **Add the missing reward-transmission baseline results** to Table 1 and 2, or remove the specific quantitative claims (41×, 6.3-343×) if this comparison cannot be included.
2. **Evaluate the actor's learned policy independently** after training in at least one environment to validate the practical deployment scenario.
3. **Conduct ablation studies** varying: (a) fixed versus learned Q distributions, (b) actor network capacity, (c) behavioral cloning update frequency.
4. **Analyze the Breakout gap** by investigating whether image-based observations make behavioral cloning harder, or acknowledge this as a limitation of behavioral cloning for high-dimensional policies.
5. **Report the ordered random coding parameter N** and analyze sensitivity to this choice.

---

## FCBbh0HCrF

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces an event-driven online Vertical Federated Learning framework that addresses the practical misalignment between existing online VFL research (which assumes synchronous data streams) and real-world scenarios where data arrives asynchronously due to events. The authors adapt dynamic local regret (DLR) from Aydore et al. (2019) to VFL with partial client activation, provide theoretical regret bounds under non-convex assumptions, and demonstrate improved stability under non-stationary conditions with reduced communication/computation costs.

## Strengths
- **Compelling problem identification**: The paper makes a valid and important practical observation that existing online VFL research unrealistically assumes synchronous data streams, whereas real-world scenarios (sensor networks responding to events, company transactions triggered by customer actions) generate data asynchronously. This gap is clearly articulated and well-motivated.
- **Rigorous theoretical analysis**: Theorem 1 correctly characterizes how partial client activation affects the dynamic local regret bound, with appropriate modifications from Aydore et al. (2019). The additional term accounting for missing gradient elements from passive clients (2W·G) is correctly derived. Corollary 1 provides O(T^3/4) sublinear regret bound.
- **Comprehensive experiments**: Extensive evaluation across three datasets (i-MNIST, SUSY, HIGGS) with both stationary and non-stationary data streams. The 9-baseline comparison (OGD/SLR/DLR × Full/Random/Event) is thorough and informative.
- **Multi-faceted evaluation**: Beyond accuracy, the paper evaluates computation time and communication costs, providing a holistic view of trade-offs between performance and efficiency.
- **Clear algorithmic presentation**: Algorithm 1 is well-structured with documented steps for server and client procedures.

## Weaknesses
- **Limited methodological novelty**: The core algorithmic contribution is adapting existing DLR techniques (Aydore et al., 2019) to the VFL setting. The main innovation lies in problem identification and formulation rather than methodological innovation. This should be more clearly scoped in the paper.
- **Incomplete view problem remains unresolved**: The paper acknowledges but does not solve the fundamental limitation that passive clients must still send embeddings each round to avoid the "incomplete view" problem. This significantly limits the practical communication-computation savings, since the server must query all clients to obtain complete inputs for each event.
- **Regret bound quality**: The achieved O(T^3/4) bound under non-convex assumptions is sublinear but weaker than O(√T) convex bounds. The additional error term involving p_max/p_min could become large with highly skewed activation probabilities across clients.
- **Event activation mechanism lacks general principled design**: The "Event" activation using threshold on average feature values is designed specifically for experiments. The Random activation is more generally applicable, making the Event activation feel like a dataset-specific demonstration rather than a principled framework component.

## Nice-to-Haves
- Ablation studies on DLR parameters (l, α) were only conducted for Full activation; similar analysis for Random/Event activation would validate robustness of parameter choices under partial client participation.
- Memory/storage overhead analysis for edge devices, given the paper claims suitability for sensor networks with limited resources.
- Systematic study of how different activation patterns (correlated vs. independent across clients) affect performance.
- Per-client embedding trajectories to show how client models evolve under partial vs. full activation.

## Novel Insights
The paper's primary novel insight is identifying that the synchronous data assumption in existing online VFL research is unrealistic for practical deployments where data is generated by events affecting only a subset of clients. This framing opens a new research direction for event-driven online VFL. Additionally, the analysis of how partial activation affects dynamic local regret bounds—specifically how missing gradient elements from passive clients contribute to the regret bound through the 2W·G term—provides a quantitative understanding of the trade-offs involved in event-driven participation.

## Potentially Missed Related Work
- Ganguly & Aggarwal (2023) on non-stationary detection and restart in online federated learning for concept drift handling
- Chen et al. (2020) and Wang et al. (2024) on asynchronous VFL methods that could offer alternative approaches to handling asynchronous data arrival
- Event-triggered control literature (Heemels et al., 2012; Trimpe & D'Andrea, 2014) with more principled theoretical foundations for event-triggering mechanisms

## Suggestions
- The paper should more explicitly scope the contribution as applying DLR to event-driven VFL rather than presenting it as a new framework, to set appropriate expectations about novelty.
- Consider implementing and evaluating one of the suggested approaches (knowledge distillation, self-supervised learning) to address the incomplete view problem, even if partially, to strengthen practical impact.
- Add experiments comparing DLR against adaptive OGD variants (Hazan et al., 2007) to better isolate the contribution of DLR over simpler adaptive methods.

---

## PigfMZMHq1

- GT: Reject (avg 3.7)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces PointNet-KAN, the first integration of Kolmogorov-Arnold Networks (KANs) with Jacobi polynomial basis functions into point-cloud neural networks for 3D classification and segmentation. The authors preserve PointNet's core principles (shared layers, symmetric max-pooling for permutation invariance) while replacing MLPs with KANs. PointNet-KAN achieves competitive accuracy on ModelNet40 (90.5% with normals vs. PointNet's 89.2%) and ShapeNet Part (83.3% vs. 83.7% mean IoU) using a substantially simpler architecture (3 layers vs. 8 layers, ~1M vs. ~3.5M parameters), demonstrating that KANs can serve as viable alternatives to MLPs in point cloud processing.

## Strengths
- **First-of-its-kind exploration**: The paper pioneers the integration of KANs into point-cloud-based neural networks for unordered 3D point sets, opening a new research direction that the authors correctly identify as previously unexplored for this domain.
- **Comprehensive ablation studies**: Tables 4-7 systematically evaluate polynomial degree (2-6) and polynomial type (Legendre, Chebyshev, Gegenbauer) across both classification and segmentation tasks, providing valuable guidance for future work on polynomial-based KANs.
- **Architectural insight preservation**: The authors correctly maintain PointNet's key principles—shared KAN layers and symmetric max-pooling—to ensure permutation invariance, demonstrating principled integration rather than naive replacement.
- **Robustness characterization**: Figures 3-4 provide useful analysis of PointNet-KAN's stability under random point dropout and Gaussian noise perturbation, showing competitive robustness compared to established methods.
- **Reproducibility**: Code is provided; training details (optimizer settings, learning rates, batch sizes, GPU specifications) are clearly documented in Section A.1.

## Weaknesses
- **Confounded experimental comparison**: The primary comparison between PointNet-KAN (87.5% without normals, 90.5% with normals) and PointNet (89.2%) conflates two changes: (1) replacing MLPs with KANs and (2) using a simplified architecture without input/feature transforms. Table 1 shows PointNet without transforms achieves only 77.4%, illustrating how transformative these components are. A controlled experiment comparing PointNet-KAN against a PointNet variant also lacking transforms is needed to isolate the KAN effect.
- **Deeper KAN architectures underperform without explanation**: Section 5.3 reveals that directly replacing all PointNet MLPs with KANs yields 88.9%—worse than both simplified PointNet-KAN (90.5%) and standard PointNet (89.2%). This critical finding is acknowledged but not analyzed. Understanding why deeper KAN networks fail is essential for guiding future architectural choices.
- **No statistical rigor**: All reported metrics are single-run results with no error bars, standard deviations, or confidence intervals. Given that accuracy differences are marginal (≤2%), it is unclear whether observed improvements are meaningful or within noise. ICLR standards typically require ± std from multiple random seeds.
- **Inconsistent hyperparameter justification**: The main experiments use polynomial degree n=4 for classification, yet ablation (Table 4) shows n=2 achieves nearly identical accuracy (89.9% vs. 90.5%) with fewer parameters. Similarly, segmentation uses n=2 with α=β=−0.5 while classification uses n=4 with α=β=1. The paper does not explain why different configurations are optimal for different tasks or justify the additional complexity of higher degrees.
- **Limited real-world evaluation**: On ScanObjectNN (real-world data), PointNet-KAN (69.2%) substantially underperforms PointMLP (85.4%) and even PointNet++ (77.9%), suggesting limited practical utility for noisy real-world scenarios compared to more advanced architectures.

## Nice-to-Haves
- **Visualize learned activation functions**: KANs' core innovation is learning activation functions; showing what the trained Jacobi polynomials look like would validate that the mechanism is actually being leveraged, not just serving as a black-box replacement.
- **Compare B-spline vs. Jacobi KANs**: The original KAN paper uses B-splines; comparing this Jacobi variant against the standard would clarify whether the polynomial basis choice matters for point cloud tasks.
- **Extended PointKAN analysis**: The supplementary PointKAN (KAN-enhanced PointMLP) shows only +0.1% improvement over PointMLP on ModelNet40. More systematic ablations (varying depth, width, polynomial degree within PointKAN) would strengthen the claim that KANs benefit advanced architectures.
- **Training dynamics analysis**: No loss curves or convergence plots are shown. Understanding KAN training stability and optimization behavior would be valuable for practitioners.

## Novel Insights
This work reveals several potentially important observations for the emerging KAN literature: (1) KANs can achieve competitive performance to MLPs on point cloud tasks with substantially fewer parameters and layers, suggesting potential efficiency benefits; (2) polynomial degree and type have minimal impact on performance across a wide range of Jacobi polynomial variants, implying the trainable weighted sum of basis functions may be the key factor rather than the specific polynomial family; (3) deeper KAN architectures (matching PointNet's layer count) actually perform worse than shallower versions, suggesting potential training instability or overfitting issues with complex KAN stacks that warrant further investigation.

## Potentially Missed Related Work
- **KAN-Node (Kashefi, 2024)**: The paper acknowledges this work integrates KANs into PointNet but for 2D CFD applications; clarifying how this differs from the 3D computer vision tasks here would strengthen the contribution claim.
- **Chebyshev-based KANs (SS, 2024)**: This concurrent work also explores alternative polynomial bases for KANs in scientific computing contexts; comparison of findings regarding polynomial type sensitivity would be valuable.

## Suggestions
- **Add controlled baseline comparison**: Create a PointNet variant with the same simplified architecture (no transforms, 3 layers) as PointNet-KAN to directly isolate the effect of KANs vs. MLPs. This single experiment would substantially strengthen the paper's core claim.
- **Report mean ± std across multiple seeds**: Run all experiments with 3-5 different random seeds and report standard deviations to enable meaningful comparison of methods with marginal differences.
- **Investigate why deeper KANs fail**: Conduct gradient flow analysis or ablation on initialization schemes to understand the training dynamics of deeper KAN architectures, as this is critical for future work extending to PointNet++-style hierarchies.

---

## p1HeFnn2AA

- GT: Reject (avg 7.3)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper initiates the study of deep learning for automated design of two-sided matching mechanisms, focusing on the tradeoff between strategy-proofness (SP) and stability. The authors develop novel differentiable surrogates for quantifying ordinal SP and stability violations, use neural networks to represent matching mechanisms, and demonstrate empirically that learned mechanisms achieve better SP-stability tradeoffs than convex combinations of classical mechanisms (DA, TTC, RSD).

## Strengths
- **First deep learning approach to two-sided matching**: The paper opens a genuinely new research direction at the intersection of ML and mechanism design, addressing a fundamentally different problem from prior auction design work due to discrete ordinal preferences.
- **Technically sound core framework**: The equivalence between ordinal SP and first-order stochastic dominance (FOSD) is properly established (Theorem 5, referencing Erdil 2014), and the differentiable surrogates are correctly derived (Theorems 6, 8).
- **Innovative architecture design**: The CNN with 1×1 convolutions using row/column max-pooling preserves permutation equivariance (critical for matching markets) while keeping input channels fixed regardless of market size, enabling scalability.
- **Comprehensive experiments**: Tests across multiple settings (correlated/uncorrelated preferences, markets up to 50×50) with substantial test sets (204,800 profiles).
- **Economically significant finding**: The discovery that learned mechanisms can dominate convex combinations of classical mechanisms is a genuinely interesting result that could motivate new theoretical characterizations.

## Weaknesses
- **Theoretical contribution is thin**: Theorems 6 and 8 are essentially restatements of known equivalences (Erdil 2014). The paper correctly references the prior work but doesn't provide novel theoretical characterizations that would guide understanding of why or when learned mechanisms work.
- **Welfare analysis is inconsistent with non-SP mechanisms**: Appendix F measures welfare assuming truthful reporting, but this assumption is precisely violated for mechanisms with non-zero SP violation. This creates an internally inconsistent evaluation—actual welfare under strategic play could differ substantially.
- **No error bars or confidence intervals**: All figures show point estimates with no measure of variance. Given the stochasticity in training and test sampling, the claim that learned mechanisms are "substantially better" cannot be assessed for statistical significance.
- **Missing comparison to sophisticated hybrids**: The paper compares only against convex combinations of DA/TTC/RSD, but Mennle & Seuken (2017) propose more general hybrid mechanisms. This comparison is necessary to validate the claimed frontier improvement over what theory already achieves.
- **Limited interpretability**: The paper claims mechanisms "raise opportunities for future work in economic theory" but provides no analysis of what structural properties these mechanisms have or why they work better. Without understanding the mechanism, the contribution is incomplete for guiding theoretical work.
- **Distribution-dependent results**: The entire efficient frontier is computed for specific synthetic distributions (uniform random with correlation). The paper doesn't analyze how the frontier changes with different preference environments, making it unclear when learned mechanisms would be practically useful.
- **Scalability with unrestricted preferences**: The adversarial approach to finding defeating misreports requires enumerating all possible misreports, which grows exponentially. The paper handles this only via structured preference domains, limiting applicability to real large markets.

## Nice-to-Haves
- **Structural analysis of learned mechanisms**: Attempt to reverse-engineer or approximate learned mechanisms with known rules (e.g., is there a variant of DA with randomization?). Decision-tree approximations or feature importance analysis would provide interpretable insights.
- **Verification on individual profiles**: Theorems 6 and 8 show surrogates are zero iff properties hold in the limit, but trained networks have non-zero surrogate values. Validation that output distributions satisfy FOSD/ex-ante stability on individual profiles would strengthen claims.
- **Case studies on specific profiles**: Concrete examples showing what the learned mechanism outputs on particular preference profiles would make the contribution tangible.
- **Real preference data**: Experiments on semi-realistic synthetic data mimicking real market structures (e.g., school quality hierarchies) or publicly available preference datasets from school choice applications.

## Novel Insights
The paper's most valuable contribution is demonstrating that deep learning can discover matching mechanisms with SP-stability tradeoffs that dominate simple convex combinations of classical mechanisms. This is a genuine proof-of-concept that ML pipelines can serve as a tool for economic discovery—not just optimization. The finding that near-DAs can be made substantially more strategy-proof (and near-RSDs more stable) opens new questions about the structure of the efficient frontier that pure theory has not characterized. However, this insight remains incomplete without understanding *why* these mechanisms work or *when* the improvements hold, limiting its immediate value for guiding mechanism design practice.

## Potentially Missed Related Work
- Mennle & Seuken (2017) "Hybrid mechanisms: Trading off strategy-proofness and efficiency of random assignment mechanisms" — directly relevant for establishing the comparison baseline of the efficient frontier.

## Suggestions
1. **Add welfare under strategic play**: Measure expected welfare when agents best-respond to learned mechanisms, not just assuming truthfulness. This is essential for evaluating practical usefulness of non-SP mechanisms.
2. **Include statistical uncertainty**: Add error bars or confidence intervals to all figures to assess whether frontier improvements are statistically significant.
3. **Strengthen interpretability analysis**: Provide at least one concrete attempt to characterize or approximate learned mechanisms with interpretable structure.
4. **Compare against Mennle & Seuken's hybrid mechanisms**: This is necessary to establish that the learned frontier represents genuine improvement over existing theoretical characterizations.

---

## bEvI30Hb2W

- GT: Reject (avg 3.0)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
LVM-Net proposes a memory-augmented approach for long-form video reasoning using a fixed-size memory populated by a differentiable neural sampler, enabling single-pass video processing. The method achieves 18x-75x inference speedup on the ReST-ADL dataset compared to Modified TubeDETR, but at the cost of substantially lower prediction accuracy across nearly all query types and durations.

## Strengths
- **Clear problem identification**: The paper correctly identifies key computational bottlenecks in long-form video understanding with transformers (quadratic attention costs, GPU memory limits) and motivates the need for fixed-memory approaches.
- **Sound efficiency mechanism**: The single-pass memory population strategy with query-specific memory reuse directly addresses the stated problem. The architecture correctly avoids redundant video processing across queries.
- **Solid ablation studies**: The paper provides meaningful ablations demonstrating that the neural sampler outperforms uniform random sampling (Table 3) and that continual learning improves over non-continual training (Table 4).
- **Proper evaluation methodology**: Inference comparisons use identical A100 instances with batch sizes maximizing GPU memory utilization, making the speedup claims verifiable.
- **Technical soundness of core components**: The Gumbel-Top-k relaxation with reparameterization trick is correctly applied for differentiable sampling.

## Weaknesses
- **Performance characterization is misleading**: The paper describes LVM-Net as achieving "competitive predictive performance," but Table 2 shows consistent and substantial underperformance compared to Modified TubeDETR (e.g., short activity: 45.3→32.4, a 29% drop; medium object: 25.4→11.9, a 53% drop). A method that is 11x faster but ~30% less accurate represents a significant trade-off that deserves honest acknowledgment, not euphemistic characterization.
- **Single-dataset evaluation**: Only ReST-ADL (4 videos) is used, limiting confidence in generalization. ICLR typically expects evaluation on multiple benchmarks (ActivityNet-QA, Charades, etc.).
- **Training efficiency not discussed**: The batch constraint (no two queries from the same video per batch to avoid race conditions) means maximum batch size becomes n/r. For distributed training, this severely limits training efficiency and is not prominently discussed as a limitation.
- **Missing recent baselines**: MemViT (Wu et al., 2022) is discussed in related work but never compared experimentally. MovieChat, LongVLM, and VideoAgent—mentioned as related approaches for long-video understanding—are not included as baselines.
- **Neural sampler independence claim unjustified**: The paper states "our proposed framework is independent of the choice of neural sampler" but only implements Xie et al., 2019 with no ablation comparing alternative sampling strategies.
- **No memory scaling analysis**: The paper sets m=5880 tokens without ablation showing how performance varies with memory size, limiting practical guidance.

## Nice-to-Haves
- Visualizations showing which patches the neural sampler selects (currently the "discriminative tokens" claim is unsubstantiated)
- Human performance or oracle upper bound on ReST to contextualize whether methods are near ceiling
- FLOP counts and peak GPU memory measurements alongside runtime to make efficiency claims more comprehensive
- Analysis of when the efficiency-accuracy trade-off is acceptable (e.g., minimum queries per video to justify memory population cost)
- Sensitivity analysis on the continual learning hyperparameter p (set to 2 without justification)

## Novel Insights
LVM-Net presents a conceptually interesting approach of using a learned neural sampler to populate a fixed-size memory for long-video reasoning, inspired by human memory mechanisms. The key insight—storing discriminative patches and reusing them across multiple queries rather than reprocessing the video—is architecturally sound and addresses a real practical need. The online continual learning auxiliary loss to address sampler bias toward current query clips is a thoughtful contribution. However, the paper does not establish that this specific combination of components (fixed memory + Gumbel-Top-k sampler + continual learning) represents a clear advance over simpler alternatives (e.g., mean-pooled tokens or random sampling with more memory tokens) or over more recent approaches like MemViT.

## Potentially Missed Related Work
- **MemViT (Wu et al., 2022)**: Memory-augmented multiscale vision transformer; explicitly addresses long-term video recognition with external memory. This is directly relevant and should be compared, not just cited in related work.
- **MovieChat (Song et al., 2024)**: Dense token to sparse memory for long video understanding; addresses the same long-video problem setting.
- **LongVLM (Weng et al., 2024)**: Efficient long video understanding via LLMs; similar efficiency goals for long-form video.

## Suggestions
- **Honest trade-off characterization**: Clearly acknowledge the accuracy- efficiency trade-off and provide guidance on when LVM-Net's approach is appropriate (e.g., applications prioritizing speed over accuracy, or scenarios with many queries per video).
- **Add MemViT as a baseline**: Since it's discussed in related work and addresses the same problem, a direct comparison is essential.
- **Memory size ablation**: Vary m from small (e.g., 1000) to large (e.g., 10000) to understand the performance-memory trade-off curve.

---

## eBS3dQQ8GV

- GT: Accept (Oral) (avg 7.8)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper provides a rigorous mathematical analysis of meta-stable clustering phenomena in a mean-field model of Transformer dynamics, where tokens are modeled as interacting particles on the sphere evolving through layers. The authors prove that for large token counts N, the system exhibits a three-phase evolution (linear, quasi-linear, clustering phases) before converging to a single cluster, with the number of intermediate clusters explicitly determined by the inverse temperature parameter β through a maximizing index of rescaled Gegenbauer polynomials. The meta-stable phase is shown to persist over timescales of O(log N), and explicit quantitative estimates connect hyperparameters to representation "richness."

## Strengths
- **Technically rigorous framework**: The paper establishes a comprehensive mathematical connection between mean-field limits, propagation of chaos, and metastability analysis. Theorem 3.1 (mean-field limit) and Theorem 4.2-4.3 (linear and quasi-linear phases) provide concrete quantitative statements with clear assumptions and complete proofs in the appendices.
- **Novel characterization of metastability**: The three-phase decomposition (linear, quasi-linear, clustering) with explicit O(log N) timescales provides a principled mathematical understanding of how structured representations emerge before complete collapse. The prediction that cluster count k_max depends on β through maximizing γ_k = k(k+d-2)Ŵ_k is both novel and meaningful.
- **Appropriate technical machinery**: The use of Grenier's iterative scheme (Grenier, 2000) to handle nonlinear growth rates and the refined Dobrushin estimates in Lemma C.7 demonstrates sophisticated mathematical technique correctly applied to this problem.
- **Strong numerical support**: Figures 2 and 4 provide compelling visual evidence matching theoretical predictions (β=5→3 clusters, β=7→4 clusters, 3-periodicity emergence from white noise). Figure 3 confirms the O(log N) scaling.

## Weaknesses
- **Assumption 3 is unproven and critical**: Theorem 4.5 (clustering phase) and Theorem C.23 entirely rest on Assumption 3, which states that solutions converge to a k_max-periodic superposition of delta masses. The paper provides only heuristic justification and numerical evidence in Figure 5. Without this assumption, the paper only characterizes the linear and quasi-linear phases—not the actual clustering behavior. This is acknowledged but represents a significant gap in the theoretical contribution's completeness.
- **Condition 3C < Kγ_max in Lemma C.14 lacks explicit guidance**: The continuation argument requires K to be "sufficiently large," but no concrete bounds or constructive guidance on choosing K are provided. This makes the proof difficult to implement in practice.
- **Limited experimental validation of higher-dimensional case**: All numerical experiments are on S^1 (d=2). Section 4.4 claims generalization to d≥3 using spherical harmonics, but no experiments verify this fundamentally different structure. This is a notable gap between theory and validation.
- **Incomplete numerical documentation**: Figure 3 averages over 20 simulations without error bars or confidence intervals. The fixed threshold for simulation termination is not specified, reducing reproducibility.

## Nice-to-Haves
- Systematic ablation over β values testing k_max ≈ √β prediction (Remark 4.1) across β ∈ {1,2,...,15}
- Experiments with d≥3 to validate spherical harmonics generalization
- Explicit finite-N convergence rates in Theorem 3.1, not just asymptotic convergence
- Discussion of finite depth (realistic L) relative to quasi-linear phase duration
- More detail on Q^TK = λId extension mentioned in conclusions

## Novel Insights
This paper provides the first rigorous mathematical characterization of meta-stable phases in mean-field Transformer dynamics, addressing an open problem identified in (Geshkovski et al., 2023). The key insight is that the number of clusters k_max that form in the intermediate regime is determined by the spectral properties of the interaction kernel through maximizing γ_k = k(k+d-2)Ŵ_k—a result that connects the temperature parameter β, embedding dimension d, and representation richness in a mathematically precise way. The O(log N) persistence time of this meta-stable phase explains why finite-depth transformers can exploit rich intermediate representations before total collapse occurs. The use of Grenier's iterative scheme to bypass the limitation that nonlinear growth rates exceed twice the linear growth rate represents a clever application of classical PDE techniques to this modern architecture problem.

## Potentially Missed Related Work
- **Carrillo et al. (2020)**: Already cited but could be discussed more deeply regarding phase transition theory for McKean-Vlasov equations on the torus
- **Han-Kwan & Nguyen (2016)**: Instabilities in mean-field limits—directly relevant to the meta-stability analysis but only briefly mentioned
- **Cohn & Kumar (2007)**: Referenced in Section 4.4 regarding spherical designs, but the connection to the invariant manifold structure could be developed more thoroughly
- **Zhang et al. recent works on neural network dynamics**: May provide alternative perspectives on finite-width effects

## Suggestions
1. **Resolve Assumption 3**: Either prove this assumption under reasonable conditions or provide extensive numerical simulations (larger N, diverse initial conditions, d≥3) to build confidence in its validity before the main clustering claims can be considered established.
2. **Clarify the K condition**: Provide explicit bounds or a constructive method for choosing K sufficiently large in Lemma C.14.
3. **Add error bars and thresholds**: Document Figure 3 with confidence intervals and specify the total variation threshold used for simulation termination.

---

## onvN3zsNMI

- GT: Reject (avg 3.5)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes an "over-memorization" approach that protects training data privacy by redirecting the model's limited memorization capacity toward a constructed "dummy set" of redundant, non-sensitive instances rather than the actual training data. The method involves first optimizing the dummy set to be easily memorable (via a separate coordinate descent procedure), then jointly training the model with both the real and dummy data. Experiments on CIFAR-10/100 image classification and Wikitext-103 language modeling demonstrate reduced membership inference attack success (lower AUROC) with maintained utility, though privacy gains are modest for image classification (AUROC 0.637→0.600) and more substantial for language modeling (0.97→0.80).

## Strengths
- **Novel conceptual contribution**: The idea of embracing memorization rather than preventing it, and strategically redirecting memorization capacity toward innocuous data, is genuinely original and addresses a fundamental tension in ML privacy.
- **Solid theoretical grounding**: The memorization reduction is formally derived using influence functions (Equation 3), providing principled conditions under which the dummy set reduces memorization scores rather than purely empirical observations.
- **Broad empirical evaluation**: Testing on both image classification (CIFAR-10/100 with ResNet-18) and causal language modeling (Wikitext-103 with Pythia-70m) demonstrates task-agnostic applicability of the approach.
- **Systematic ablation studies**: The paper examines dummy set size, sparsity, and partitioning variants, providing useful insights into what properties of the dummy set drive effectiveness (Tables 2, 4).
- **Significant language modeling results**: The AUROC reduction from ~0.97 to ~0.80 on Wikitext-103 represents a substantial privacy improvement, suggesting the method may be particularly effective for LLM fine-tuning scenarios.

## Weaknesses
- **Missing comparison to established privacy defenses**: The paper never benchmarks against differential privacy (DP-SGD), which is the standard baseline for privacy-preserving ML. This omission makes it impossible to assess whether the method is competitive with or complementary to established techniques. For a privacy-focused paper at this venue, this is a significant gap.
- **Limited attack evaluation**: Only the LiRA (reference-based) membership inference attack is evaluated. Stronger attacks such as training data extraction (Carlini et al., 2021), label-only membership inference (Choquette-Choo et al., 2021), or semantic membership inference are not tested. An AUROC of 0.60 still indicates substantial privacy vulnerability.
- **Large statistical uncertainty**: In Table 1, memorization score reductions (0.454→0.441 for CIFAR-10, 1.740→1.684 for CIFAR-100) have very large standard deviations (±1.224, ±2.272), and proper statistical significance tests are not reported. The practical significance of these reductions is unclear.
- **Computational overhead**: Training the dummy set requires a separate model and coordinate descent optimization (Algorithm 2), which is not negligible. The paper does not provide explicit time/memory comparisons or analyze the cost-benefit trade-off relative to simpler privacy methods.
- **Security under adaptive adversaries**: The method assumes adversaries cannot distinguish dummy from real samples. If an attacker can identify dummy data (e.g., via distributional analysis), privacy benefits may vanish entirely. The paper does not address this attack vector.

## Nice-to-Haves
- **Dummy set transferability analysis**: Test whether a dummy set trained for one architecture/dataset transfers to others, which would significantly enhance practical utility.
- **Per-sample memorization distribution**: Show how memorization score distributions change (not just means) to reveal whether the method uniformly reduces memorization or affects only specific samples (e.g., rare vs. common).
- **Training data extraction evaluation**: Evaluate whether the method prevents extraction attacks (Carlini et al., 2021), which directly demonstrate memorization harm rather than just membership inference susceptibility.
- **Privacy-utility Pareto frontier**: Plot AUROC vs. accuracy/perplexity for different dummy set configurations to provide practical guidance on choosing the dummy set size.

## Novel Insights
The paper's core insight—that memorization is a limited capacity that can be strategically occupied rather than suppressed—offers a fresh perspective on the privacy-generalization tradeoff. The formal connection between dummy set effectiveness and influence functions (Equation 3) provides a principled framework: the dummy set must simultaneously improve generalization (when the real sample is absent) and remain uncorrelated with real training data. The empirical finding that trained dummy sets outperform random noise (Table 1) confirms that not all "decoy" data is equally effective—the optimization procedure is essential for making dummy samples genuinely memorable. The language modeling results suggest the approach may be especially valuable for LLM fine-tuning, where membership inference is a known concern.

## Potentially Missed Related Work
- **Canary/guard training methods**: Work on training models to memorize specific "canary" tokens or using decoy data for intellectual property protection may be related, though the focus differs.
- **DP-SGD with multiple rounds**: Methods that iteratively refine privacy guarantees might offer complementary perspectives.
- **Machine unlearning methods**: Approaches to selectively forget memorized information (e.g., gradient-based unlearning) could be relevant comparison points.

## Suggestions
- **Add DP-SGD comparison**: Include at least one experiment comparing privacy-utility trade-offs against differential privacy (e.g., using Opacus or standard DP-SGD with ε=8 or ε=16) to contextualize the method's practical value. This is essential for a privacy paper at this venue.
- **Evaluate robustness to stronger attacks**: Test against label-only membership inference attacks and, if computationally feasible, training data extraction attacks to verify comprehensive privacy improvement beyond LiRA.
- **Report statistical significance**: Conduct proper hypothesis tests (e.g., paired t-tests or bootstrap confidence intervals) for memorization score changes to support claims of improvement.

---

## v4MTnPiYXY

- GT: Accept (Poster) (avg 7.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
Q-SFT proposes a novel offline RL algorithm that reformulates Q-learning as a weighted cross-entropy problem, enabling direct fine-tuning of pretrained language models without architectural modifications. The key insight is that weighting the SFT objective with Bellman backup values causes the resulting probabilities to conservatively estimate Q-values while inheriting pretrained knowledge. The method is evaluated across diverse domains (language games, dialogue, vision-language navigation, robotics) and demonstrates strong empirical performance compared to both supervised and value-based RL baselines.

## Strengths
- **Novel algorithmic insight**: The core idea of casting Q-learning as weighted supervised fine-tuning is genuinely novel and directly addresses a practical barrier that has hindered value-based RL adoption for foundation models—specifically, the need to reinitialize weights or add value heads.
- **Solid theoretical foundations**: Theorem 4.1 provides meaningful guarantees that learned probabilities conservatively bound true Q-values, establishing principled foundations. The proof in the appendix is complete and well-structured.
- **Comprehensive empirical evaluation**: The paper evaluates across diverse domains—language games (Chess, Wordle, 20Q), WebShop, ALFWorld (VLM), and robotic manipulation—comparing against SFT, ILQL, CQL, and Q-Transformer baselines. This breadth strengthens confidence in generalizability.
- **Addresses practical concerns**: The method elegantly solves the architectural mismatch problem, allowing direct use of pretrained logits without modification.
- **Sample efficiency gains**: Figure 3 demonstrates meaningful advantages in low-data regimes compared to Q-transformer, which is practically important.

## Weaknesses
- **Missing critical ablations**: The paper cannot isolate the contribution of key design choices without ablating (1) the policy extraction formula π(a|s) ∝ πϕ(a|s) exp(β pθ(a|s)), (2) the dual-network training procedure (optimizing both πϕ and pθ separately), and (3) the effect of β. An ablation comparing Q-SFT without the behavior cloning pretraining phase would directly test the claim that leveraging pretraining is essential.
- **Limited theoretical scope**: Theorem 4.1 only guarantees bounds for actions where Q*(s,a) ≥ |A|⁻¹. For language models with vocabulary sizes of 30,000+, this condition excludes the vast majority of tokens. The paper dismisses this ("actions with negligible Q-values will not be chosen anyway") but provides no quantification of what fraction of probability mass is covered, leaving uncertainty about practical applicability.
- **No statistical rigor in experiments**: Tables 1-3 report point estimates with no error bars or statistical significance tests. For ALFWorld, where success rates range from 5-45%, this makes it difficult to assess whether differences between methods are meaningful or within noise.
- **Limited scaling validation**: The claim that the method benefits more from pretraining than existing value-based RL is supported by only one experiment (Figure 4) on a single task (20Q) against a single baseline (ILQL). As this is a central empirical claim, testing across multiple tasks and baselines would strengthen confidence.
- **No code release**: For an ICLR paper, open-sourcing the implementation is increasingly expected. The algorithm details in Appendix B are clear, but reproducibility would be greatly enhanced by code availability.

## Nice-to-Haves
- **Decision Transformer comparison**: The paper mentions Decision Transformer in related work but doesn't include it as a baseline in experiments. Given that DT is a strong supervised approach to offline RL for language models, this comparison would help contextualize where Q-SFT provides genuine advantage.
- **Training curves for language tasks**: Only robotics learning curves are shown (Figure 3). Showing learning curves across all tasks would reveal whether Q-SFT consistently learns faster/slower than baselines throughout training.
- **Online RL evaluation**: The paper motivates the method for multi-turn problems that may involve interaction, yet only evaluates in offline settings. Online experiments would broaden the practical impact narrative.
- **Hyperparameter sensitivity analysis**: β varies from 1.0 to 8.0 across tasks (Table 4) with no guidance on selection. A sensitivity analysis would help practitioners apply the method.

## Novel Insights
The paper's core contribution—using weighted cross-entropy where weights are Bellman backup targets to learn Q-values directly from pretrained logits—is a genuinely novel framing that bridges the gap between supervised pretraining and value-based RL. The insight that "pθ(a|s) ≈ w(s,a)·πβ(a|s)" and that choosing w(s,a) = B* pθ(a|s) yields a conservative Q-function estimate is both theoretically sound and practically valuable. The observation that this avoids the unstable regression toward non-stationary Bellman targets (a known failure mode for scaling Q-learning to large models) is well-motivated by prior work. The policy extraction step at inference time (combining the learned behavior policy with the learned Q-likelihood) is elegant and avoids additional training costs.

## Potentially Missed Related Work
- **Decision Transformer (Chen et al., 2021)** — The paper discusses this in related work as similar in using a supervised objective but notes DT has theoretical limitations. However, empirical comparison is missing.
- **IQL (Kostrikov et al., 2021)** — Mentioned as a baseline for robotics but not included in language experiments; could serve as an additional value-based baseline for comparison.

## Suggestions
- Add ablation studies specifically isolating the contribution of (1) the policy extraction formula versus using pθ directly, (2) the dual-network versus single-network training, and (3) the weighted cross-entropy weights versus uniform weights. This would directly validate which components drive the empirical gains.
- Include error bars and statistical significance tests across all experiments to quantify the reliability of reported improvements.
- Extend the scaling experiments to multiple tasks and baselines to substantiate the central claim about leveraging pretraining.
- Consider releasing code alongside the paper to enhance reproducibility and community adoption.

---

## YkMg8sB8AH

- GT: Reject (avg 4.2)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces EquiGX, a novel explainability method that extends Layer-wise Relevance Propagation (LRP) via Deep Taylor decomposition for spherical equivariant graph neural networks operating on 3D geometric graphs. The method derives specific propagation rules for tensor product operations, linear layers, and norm-based non-linearities, enabling backpropagation of relevance scores to input space to explain geometric features. Experiments on synthetic and real-world datasets demonstrate improved explanation quality over six baseline methods across multiple metrics.

## Strengths
- **Addresses a genuine and important gap**: The paper tackles explaining equivariant GNNs for 3D geometric data, where existing XAI methods are largely inapplicable due to their focus on 2D graphs and inability to handle positional/geometric features. This is a valuable contribution for AI-for-science applications where equivariant GNNs are widely deployed but largely opaque.
- **Technically sound derivation**: The LRP extension for tensor product operations leverages the bilinearity property correctly, and the first-order Taylor expansion is exact when zero-order terms vanish (since TP is multilinear in its inputs). The propagation rules for linear layers (Section 3.3) correctly follow standard LRP-ε formulation.
- **Comprehensive layer coverage**: The paper derives propagation rules for all key components of spherical equivariant GNNs—TP operations, linear layers, and norm-based non-linearities—enabling end-to-end relevance propagation from prediction to input space.
- **Richer explanatory power**: Unlike prior methods that only explain node/edge importance, EquiGX separately attributes contributions to node features, edge distances, and edge directions, providing more mechanistic insights into how geometric and positional data influence predictions.
- **Consistent quantitative improvements**: Table 1 demonstrates EquiGX outperforms all baselines across four datasets on both AUROC and AP metrics, with particularly notable gains on Spiral Noise (83.57% vs 79.69% AUROC for best baseline) and ActsTrack (76.96% vs 65.18% AUROC).

## Weaknesses
- **Limited architectural evaluation undermines generalizability claims**: The paper claims contributions for "spherical equivariant GNNs" broadly but only validates on Tensor Field Networks (TFN). Other prominent spherical equivariant architectures like NequIP, Allegro, Equiformer, or MACE are not tested. This significantly limits confidence in the method's broad applicability, particularly since different equivariant architectures use different non-linearities, aggregation schemes, and tensor product formulations.
- **Equal relevance attribution assumption is unjustified**: Equation 7 divides relevance among three components (hidden features, directional part, distance part) equally based on Achtibat et al. (2024), which concerns attention mechanisms in transformers—not equivariant networks. This assumption is particularly problematic given that these components have distinct physical meanings in geometric learning (distances encode bond lengths, directions encode bond angles). The equal-splitting is a strong prior that should either be rigorously justified or explored empirically.
- **No verification that explanations are equivariant**: While the underlying model is rotation-equivariant, the paper never verifies that the computed importance scores transform correctly under input rotations. This is a fundamental property one would expect an explainability method for equivariant models to guarantee or at least test.
- **Missing ablation studies**: The method combines multiple components (TP decomposition, directional attribution, distance attribution, normalization for non-linearities), but no ablation isolates which components drive the improvement. This makes it unclear whether the complex derivation actually matters versus simpler alternatives.
- **Perfect accuracy on synthetic datasets limits discriminative power**: TFN achieves 100% accuracy on both Shapes and Spiral Noise datasets. When models trivially solve tasks, explanations may reveal less about what features truly matter versus what the model happened to learn. This is acknowledged in the literature but worth explicit discussion.

## Nice-to-Haves
- **Statistical significance testing**: While error bars are provided in Table 1, formal significance tests comparing EquiGX to the second-best method (e.g., PG-Explainer on Shapes: 84.31 vs 82.83) would strengthen confidence in the reported improvements.
- **Deeper analysis of BioLip fidelity results**: The paper reports Fidelity ≈ 0.02 when masking binding sites but doesn't discuss what this reveals about model validity. The finding that the model may not use scientifically meaningful binding site information deserves explicit analysis rather than being mentioned almost as an aside.
- **Rotation consistency experiments**: Showing explanations for the same sample under different 3D rotations would validate that structurally equivalent features receive equivalent importance regardless of coordinate orientation.

## Novel Insights
The paper's core insight—that explaining equivariant GNNs requires deriving LRP rules specifically for tensor product operations in spherical harmonic space—is both novel and important. The observation that messages in spherical equivariant GNNs can be decomposed into contributions from (1) hidden features, (2) directional parts (via spherical harmonics), and (3) distance parts (via radial basis functions) provides a principled way to attribute importance to geometric primitives that have physical meaning in scientific applications. This contrasts with prior work that either ignores geometric features entirely or treats them as generic node/edge attributes.

## Potentially Missed Related Work
- **AttnLRP (Achtibat et al., 2024)**: While cited for the equal-splitting assumption, the paper could more thoroughly discuss differences between explaining attention mechanisms versus tensor product operations in equivariant networks.
- **Frame averaging techniques (FAENet, Duval et al., 2023b)**: Discussed briefly in the appendix for categorization but could inform discussion of alternative approaches to ensuring explanation consistency.

## Suggestions
- **Validate on at least one additional spherical equivariant architecture**: Testing on SE(3)-Transformer or NequIP would substantially strengthen the generalizability claims without requiring full re-derivation (the core TP operations are similar across architectures).
- **Add empirical comparison of equal vs. alternative attribution strategies**: An ablation or experiment varying the attribution weights among the three components would validate whether the equal-splitting assumption matters in practice.
- **Include an equivariance verification experiment**: Show that rotating a 3D structure produces correspondingly rotated importance scores, providing empirical evidence that explanations respect model symmetries.

---

## EDJ7cPZk7V

- GT: Reject (avg 6.4)
- Predicted: Accept (7.0/10)
- Match: No

### Final Review

## Summary
This paper uncovers a strong empirical correlation between how quickly neural networks learn individual examples (learning speed) and their likelihood of being forgotten during continual learning—demonstrating a "last-in-first-out" forgetting pattern. Based on this insight, the authors propose Goldilocks, a replay buffer sampling method that filters out examples learned too quickly or too slowly, retaining only those learned at intermediate speeds. Goldilocks is shown to consistently improve multiple continual learning algorithms across various image classification benchmarks while requiring minimal computational overhead.

## Strengths
- **Compelling empirical finding**: The paper demonstrates a remarkably strong correlation (r=0.995, p≤10⁻¹⁰) between learning speed and forgetting susceptibility, visualized clearly through epoch-wise classification matrices. This data-centric perspective on catastrophic forgetting is genuinely insightful and under-explored in prior work.

- **Thorough experimental validation**: The authors test their findings across diverse settings: multiple datasets (CIFAR-10, CIFAR-100, TinyImageNet), architectures (ResNet-18, VGG-16, smaller variants), both task-incremental and class-incremental settings, multiple continual learning backbones (DER, DER++, GEM, AGEM, RPC, X-DER, FDR), and various hyperparameter configurations. This extensive evaluation demonstrates robustness.

- **Practical method with real value**: Goldilocks is computationally trivial (single float per example computed during forward pass), easy to implement, and orthogonal to existing continual learning methods since it only modifies buffer sampling. The consistent improvements (typically 1-3% accuracy gains) across methods represent meaningful practical benefit.

- **Valuable insight for practitioners**: The finding that optimal buffer composition is largely independent of subsequent task identity simplifies hyperparameter selection and provides actionable guidance for deployment.

## Weaknesses
- **Limited theoretical grounding**: The paper provides extensive empirical support and intuitive connections to simplicity bias, but lacks formal analysis explaining *why* learning speed predicts forgettability or *why* mid-learned examples are optimal for rehearsal. This limits the contribution's depth and generalizability beyond empirical observation.

- **Hyperparameter dependency underexplored**: While the paper claims robustness and demonstrates a "broad continuous range" of effective quick/slow values, it lacks systematic sensitivity analysis. The default 20%/20% is heuristic, and the optimal thresholds likely vary with buffer size, dataset, and task count in ways the paper doesn't fully characterize.

- **Limited scope beyond replay-based methods**: The analysis and Goldilocks are designed specifically for replay-based continual learning. The paper does not explore whether the learning-speed-to-forgetting correlation holds or can be exploited for non-replay methods (regularization-based, architecture-based approaches).

## Nice-to-Haves
- **Streaming/single-epoch evaluation**: The paper acknowledges this limitation but provides no experiments. Exploring lightweight approximations (e.g., pre-trained model features as complexity proxies) would broaden applicability.
- **Concrete examples of filtered samples**: Visualizing actual images removed vs. retained at different thresholds would make the "Goldilocks zone" more intuitive and expose edge cases.
- **Statistical significance testing**: While standard errors are reported, formal hypothesis tests comparing Goldilocks to baselines would strengthen the empirical claims.

## Novel Insights
The paper's most valuable contribution is the empirical characterization of catastrophic forgetting from a data-centric perspective: examples learned faster (simpler) are forgotten less readily than those learned more slowly (complex). This "last-in-first-out" pattern—connecting learning speed, simplicity bias, and forgetting—provides both a diagnostic tool (predicting which examples are vulnerable) and a prescriptive one (guiding buffer composition). The finding that optimal buffer composition prioritizes neither the easiest nor hardest examples—those the model already masters or those it struggles to learn—offers a principled middle ground that explains and improves upon uniform sampling. While the connection to simplicity bias is intuitive rather than novel in spirit, the specific instantiation linking learning speed to buffer sampling decisions is a practically useful insight.

## Potentially Missed Related Work
- **Example forgetting metrics**: Toneva et al. (2019) and Maini et al. (2022) are cited, but the paper could more deeply contrast its learning-speed metric with "forgetting events" or "last-right" metrics, clarifying what additional predictive power learning speed provides.
- **Rainbow Memory** (Bang et al., 2021): This concurrent work also focuses on diverse sample selection for rehearsal; a comparison would contextualize Goldilocks' approach.

## Suggestions
- **Systematic hyperparameter analysis**: Conduct a structured sensitivity study showing how optimal quick/slow thresholds vary with buffer size, dataset complexity, and number of tasks. This would provide practitioners clearer guidance than the current "broad range" claim.
- **Theoretical motivation**: Even a sketch of formal analysis—connecting learning speed to PAC-Bayes bounds or optimization dynamics—would strengthen the contribution's theoretical standing.

---

## RdG7LVGnQi

- GT: Reject (avg 4.5)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces LLM-QFA, the first work to apply once-for-all (OFA) training to quantized large language models, enabling a single supernet to generate optimal subnets for diverse deployment scenarios (e.g., edge devices vs. GPU clusters) with different bit-width constraints. The method decouples quantization configurations using separate Low-Rank adapters per (layer, bit-width) pair and introduces a resource-balanced sampling strategy to address training imbalance from uniform sampling. Experiments on LLaMA2-7B, LLaMA2-13B, and Mistral-7B demonstrate comparable or modestly improved performance over QA-LoRA with significant training time reduction (3× fewer GPU hours when deploying to multiple scenarios).

## Strengths
- **First OFA-QAT for LLMs**: This is a genuinely novel contribution that addresses a practical deployment challenge—repeated quantization-aware training per scenario is prohibitively expensive for LLMs. The paper identifies a real gap in the literature.
- **Well-motivated ablation studies**: The Shared LoRA ablation (Figure 7) provides clear empirical evidence that decoupling adapters improves performance across all resource constraints. The uniform sampling ablation similarly validates the resource-balanced sampling contribution.
- **Practical efficiency gains**: The 3× training time reduction (Figure 4) compared to QA-LoRA when deploying to N scenarios is meaningful and directly addresses the paper's stated motivation. This is the primary value proposition.
- **Comprehensive model coverage**: Experiments span three model families (LLaMA2-7B, LLaMA2-13B, Mistral-7B) and two benchmark suites (MMLU, Common Sense QA), with consistent results across settings.
- **Observant problem identification**: The observation that uniform sampling over independent layer-wise bit-widths produces a narrow-variance Gaussian distribution on average bit-width is insightful and non-obvious—it reveals a fundamental challenge in OFA for mixed-precision quantization.

## Weaknesses
- **Train-test leakage in search procedure**: The paper searches for optimal subnets on MMLU (for MMLU results) and ARC-C (for Common Sense QA results), then reports final results on the same benchmarks. This inflates reported numbers. A held-out calibration set should be used for search, with separate test sets for evaluation.
- **Lack of statistical rigor**: Improvements are in the 0.3–3.5% range, but no error bars, confidence intervals, or multi-seed experiments are reported. It's unclear whether these gains are robust or within noise. At minimum, standard deviations across multiple runs should be provided.
- **Vague search procedure**: Section 3.3 lacks sufficient detail for reproducibility. How is "correlation analysis" performed? What metric guides the search space shrinkage? For a search space of 3^32 configurations, the 150-sample budget (100 random + 50 narrowed) seems arbitrary and potentially insufficient.
- **Scalability claims unsupported**: The paper claims the method "can be easily scaled up to even larger models" but provides no evidence beyond 13B. With 3L separate LoRA adapters (96 for 7B, 192 for 13B), scaling to 70B (210 adapters) introduces practical memory management challenges not discussed.
- **Adapter overhead not quantified**: The paper claims decoupled adapters bring "negligible extra costs" but never reports total adapter parameter count or memory overhead. This should be explicitly quantified, especially since 3L adapters are required.
- **Inference-time implications not discussed**: When deploying a specific subnet, how are the correct LoRA adapters selected and merged at runtime? Is there latency overhead compared to single-adapter approaches? The paper focuses on training efficiency but ignores inference considerations.

## Nice-to-Haves
- **Visualization of discovered architectures**: Show heatmaps of per-layer bit-width assignments for different resource constraints to demonstrate that the search finds meaningful patterns rather than random assignments.
- **Comparison with weight-sharing OFA-QAT baseline**: The paper lacks a comparison with traditional OFA (weight-sharing) applied to quantization-aware training. Without this, the claimed advantage of decoupling is only validated through ablation (Shared LoRA), not against an actual competitor.
- **Stronger NAS baselines**: Compare the heuristic search against established methods like NSGA-II or differentiable NAS (e.g., DARTS-style) to contextualize the search quality.
- **Theoretical analysis**: Provide analysis or additional experiments showing *why* independent LoRA updates prevent interference (e.g., gradient similarity metrics, adapter representation analysis).

## Novel Insights
The paper's most valuable insight is identifying and addressing the training imbalance problem in once-for-all quantization-aware training: uniform sampling over independent layer-wise bit-widths naturally concentrates around the mean bit-width due to the Central Limit Theorem, leaving extreme configurations (very low or very high average bit-width) undertrained. The cyclic mixed-Gaussian scheduler provides a simple but effective remedy by sweeping the sampling distribution from high to low bit-widths. This observation applies beyond this specific paper and illuminates a fundamental challenge in OFA for mixed-precision scenarios.

## Potentially Missed Related Work
- **Weight-sharing OFA for quantization** (Tang et al., 2024; 2023): The paper briefly mentions interference in weight-sharing OFA but doesn't empirically compare against such approaches. Including this comparison would strengthen the paper's positioning.

## Suggestions
1. **Use held-out validation sets**: Split MMLU into val/test and ARC-C into calibration/evaluation sets. Search on calibration, report on held-out test to eliminate leakage.
2. **Report statistical significance**: Run experiments with 3+ seeds and report means ± standard deviations. This is essential for the 1–3% improvement range.
3. **Quantify adapter overhead**: Report total LoRA parameter count (should be approximately 3L × 2 × rank × hidden_dim) and memory footprint compared to base model.
4. **Clarify search algorithm**: Specify the correlation metric (e.g., Pearson, Spearman), how sensitivity is computed, and the shrinkage criterion in Section 3.3.

---

## HSi4VetQLj

- GT: Accept (Poster) (avg 6.0)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces Semantically Diverse Language Generation (SDLG), a method that improves uncertainty estimation in LLMs by using importance sampling with an NLI-informed proposal distribution to generate semantically diverse yet likely alternative outputs. SDLG identifies semantically critical tokens through gradient-based attribution and substitutes them with alternatives predicted to alter semantics, while correcting for biased sampling. Experiments on TruthfulQA, CoQA, and TriviaQA demonstrate consistent AUROC improvements over existing methods with claimed computational efficiency gains.

## Strengths
- **Theoretically grounded**: The paper properly derives semantic entropy from first principles and correctly identifies that prior work (Kuhn et al., 2023) used an improper estimator. The importance sampling correction (Eq. 7-8) is theoretically sound and represents a genuine contribution to the semantic entropy framework.
- **Novel sampling strategy**: Using gradient-based attribution via an NLI model to identify semantically critical tokens for targeted substitution is a genuinely new approach. The three-score framework (attribution, substitution, importance) is well-conceived and the ablation (Figure 9) demonstrates each component contributes positively.
- **Comprehensive empirical evaluation**: Experiments span three diverse QA datasets, four model sizes (2.7B–30B), and three correctness metrics, showing consistent improvements. The comparison separating the improper vs. proper semantic entropy estimator effectively isolates the contribution of SDLG's sampling strategy.
- **Practical efficiency gains**: The 15–33% FLOP reduction is significant, and avoiding temperature tuning is a genuine practical advantage.
- **Clear contribution to literature**: The paper addresses a real problem (sampling by chance in semantic uncertainty estimation) with a principled solution.

## Weaknesses
- **Strong theoretical assumption unvalidated**: The paper assumes "all probability mass in p(t|...) is at the actually observed t" (Section B), which is explicitly called a "strong assumption." This is central to the proposal distribution derivation and importance sampling correction, yet the paper provides no empirical validation of this assumption. This weakens the theoretical contribution claim and the justification for the importance sampling correction.
- **Limited model diversity**: All experiments use OPT models only. The paper states "performance trends generalize across transformer-based model families" based on prior work, but this is an assumption not validated for SDLG. Given that SDLG's token scoring depends on specific model-NLI interactions, generalization is not guaranteed.
- **No statistical significance testing**: AUROC differences are reported without confidence intervals. This is particularly concerning for CoQA where improvements are modest (0.4–0.7 percentage points) and may not be statistically significant. ICLR standards require demonstrating that observed differences are not due to random variation.
- **Incomplete ablation of importance sampling correction**: The paper derives importance sampling weights but never validates empirically whether applying these weights improves results. A comparison with SDLG without the correction is essential to justify the added complexity.
- **Token probability threshold and weighting not justified**: A threshold of 0.001 is used for token probabilities but never ablated. The equal weighting of three scores is used without exploration of alternatives.

## Nice-to-Haves
- **Extended model evaluation**: Test on at least one non-OPT architecture (e.g., LLaMA) to support generalization claims.
- **Comparison with recent methods**: MARS (Bakman et al., 2024) and EigenScore (Chen et al., 2024) are mentioned as "complementary" but not empirically compared.
- **Longer-output evaluation**: The paper claims SDLG advantages "emerge in scenarios with longer output sequences" but all datasets produce short answers; experiments on summarization or long-form generation would substantiate this claim.
- **Failure case analysis**: Table 3 shows only success cases. Concrete examples of when SDLG fails (e.g., substituted token leads to incoherent completion, or model "corrects back" to original semantics) would strengthen the paper.

## Novel Insights
The core insight—that semantically diverse outputs should be *deliberately generated* rather than sampled by chance—is valuable and well-executed. The paper's decomposition of token importance into attribution, substitution, and importance scores provides a principled way to "stress-test" the language model's semantic certainty. The observation that correcting the improper semantic entropy estimator alone improves results is an important contribution that others in the field should attend to.

## Potentially Missed Related Work
- **MARS** (Bakman et al., 2024): Meaning-aware response scoring for uncertainty estimation; could serve as a complementary uncertainty measure or comparison baseline.
- **EigenScore** (Chen et al., 2024): Uses internal LLM states for hallucination detection; alternative approach to uncertainty estimation that may benefit from SDLG's diverse samples.
- **Contrastive decoding methods**: While contrastive decoding (Li et al., 2023) is mentioned, its specific application to uncertainty estimation scenarios is not explored.

## Suggestions
1. **Add ablation of the importance sampling correction**: Compare SDLG with and without the weighting term to validate that the correction provides measurable benefit beyond the proposal distribution itself.
2. **Include statistical significance testing**: Report confidence intervals or p-values for AUROC comparisons, especially for CoQA where improvements are smaller.
3. **Validate or relax the proposal distribution assumption**: Either provide empirical evidence that the "all mass at observed t" assumption holds in practice, or explore more robust weighting schemes that don't rely on it.
4. **Test on non-OPT models**: Even a single experiment with LLaMA-2 would substantially strengthen generalization claims.

---

## LqB8cRuBua

- GT: Reject (avg 2.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes Diffusion SigFormer, a two-stage method combining a diffusion-based denoising module (DSDM) with a transformer-based classifier (SigFormer) for electromagnetic signal recognition under interference conditions. The approach adds synthetic noise (Gaussian, Rayleigh, Periodic) to signals via a designed interference mechanism, denoises them using a fixed-timestep diffusion model, and then classifies using a hybrid convolution-Transformer architecture. Experiments on three datasets demonstrate that the diffusion denoising recovers recognition accuracy close to clean signal levels across various interference conditions.

## Strengths
- **Practical problem identification**: The paper correctly identifies a meaningful real-world challenge—electromagnetic signal recognition under interference—and provides a motivated two-stage solution (denoise then classify).
- **Comprehensive experimental coverage**: Testing across three datasets (RML2016.10a, RML2016.10b, BT) with three noise types and ten interference rates provides substantial empirical evidence. The consistent ability to recover near-clean accuracy (e.g., ~85% denoised vs ~88% clean on RML2016.10a) demonstrates the approach's anti-interference capability.
- **Architectural innovation**: Combining convolution with transformer attention in SigFormer Block (adding a 1D conv between Attention and MLP) is a reasonable design choice that addresses the known local feature extraction weakness of pure transformers for 1D signals.

## Weaknesses
- **Non-standard diffusion formulation with insufficient justification**: The paper fixes timestep t such that α_t = 0.5 during both training and inference, meaning the model operates at a single noise level. This fundamentally departs from standard DDPM theory, where models learn across the full noise schedule. The justification ("to satisfy the interference rate relationship") is ad hoc. No comparison with variable-t diffusion models is provided to validate this design choice.
- **Critical missing ablation studies**: The paper claims contributions from both DSDM and SigFormer, but provides no ablation removing DSDM (keeping only SigFormer on noisy signals) or replacing SigFormer with a simpler classifier. This prevents verification of each component's individual contribution.
- **Identical Gaussian and Rayleigh denoising results are unexplained**: Tables 2, 4, and 6 show nearly identical accuracy for Gaussian and Rayleigh noise across all SIR levels (e.g., both ~0.853 on RML2016.10a, both ~0.886 on RML2016.10b). This statistically implausible result suggests the denoising module may not actually be responding to interference level, and the paper offers no explanation.
- **Training-testing mismatch with real-world conditions**: The paper trains at SNR=18dB only, then tests with synthetic localized interference. This does not demonstrate robustness to realistic channel conditions with varying SNR across the full -20dB to 18dB range, which is standard for modulation recognition evaluation.
- **Incomplete SigFormer specification**: Key hyperparameters (patch size, embedding dimension, number of SigFormer Blocks, attention heads) are not reported, hindering reproducibility.

## Nice-to-Haves
- Compare DSDM against conventional denoising methods (wavelet denoising, spectral subtraction, bandpass filtering) to assess whether diffusion models add value over simpler approaches.
- Evaluate across the full SNR range (-20dB to 18dB) using built-in channel noise to enable comparison with existing AMR literature.
- Use per-device accuracy on BT dataset rather than aggregating to brand level, to better leverage the dataset's scale.
- Report computational cost (training time, inference time, model parameters, FLOPs) to assess practical deployability.
- Add confusion matrices under interference conditions to show where denoising provides the most benefit.

## Novel Insights
The paper's primary insight—applying diffusion models to denoise electromagnetic signals before classification—represents a reasonable transfer from the image domain to signal processing. The SigFormer Block design, while incremental, appropriately addresses the known limitation of pure transformers for local feature extraction in 1D signals. The most interesting observation is that the denoising module appears to generalize across noise types (Gaussian, Rayleigh, Periodic), maintaining near-clean accuracy regardless of interference type. However, the identical results for Gaussian and Rayleigh noise warrant investigation—this may indicate the denoising is learning a generic "noise removal" rather than noise-type-specific characteristics.

## Potentially Missed Related Work
- **ResSwinT-SwinT (Ren et al., 2023)**: A two-component radar signal recognition framework using Swin Transformer and denoising networks; directly relevant as both a transformer-based method and a denoising-then-classify approach.
- **ICAN, MT-ResNet, ATR-CNN**: Modern AMR methods with established performance across full SNR ranges on RML datasets; missing from baselines.
- **Signal denoising with diffusion in other domains** (Lan & Huang, 2024; Zhu et al., 2023; Deng et al., 2024): Seismic and DAS signal denoising with diffusion models; provides theoretical context for the diffusion-on-signals approach.

## Suggestions
The paper would benefit significantly from: (1) adding complete ablation experiments isolating each component's contribution, (2) comparing against standard variable-t diffusion to validate the fixed-t design choice, (3) evaluating across full SNR ranges to enable fair comparison with existing literature, and (4) investigating and explaining the suspicious identical results for Gaussian and Rayleigh denoising.

---

## gQlxd3Mtru

- GT: Accept (Oral) (avg 8.7)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper introduces DeepRUOT, a deep learning method for learning Regularized Unbalanced Optimal Transport (RUOT) and inferring continuous unbalanced stochastic dynamics from sparsely time-resolved snapshot data. The key theoretical contribution is a Fisher regularization reformulation (Theorem 4.1) that transforms the SDE-constrained optimization into a more computationally tractable ODE form, avoiding derivative computations in cross terms. The method is validated on synthetic gene regulatory networks, high-dimensional Gaussian mixtures (up to 100D), and real single-cell RNA-seq datasets, demonstrating accurate growth/transition identification and Waddington landscape construction.

## Strengths
- **Theoretically grounded contribution**: Theorem 4.1 provides a novel Fisher regularization reformulation of RUOT that improves computational tractability over prior work (Baradat & Lavenant, 2021) by avoiding explicit differentiation of log p in cross terms. The equivalence proof and connection to the probability flow ODE are solid.
- **Comprehensive empirical validation**: The paper evaluates on diverse datasets including a synthetic gene regulatory network with ground truth, 10D-100D Gaussian mixtures, mouse hematopoiesis scRNA-seq data, and EMT scRNA-seq data. Thorough ablation studies in Table 4 demonstrate the necessity of each component (growth term, mass loss, Fokker-Planck constraint, pre-training).
- **Addresses a genuine gap**: The method handles an important real-world challenge—learning dynamics from sparsely time-resolved snapshots where existing balanced Schrödinger bridge methods fail or require prior knowledge of growth/death processes.
- **Practical utility for single-cell biology**: The ability to infer growth rates and construct Waddington landscapes from snapshot data without lineage information or prior knowledge of proliferation is valuable for trajectory inference applications.

## Weaknesses
- **Unfair experimental comparisons**: For gene regulatory data, DeepRUOT uses σ=0.05 while SF2M uses σ=0.25 due to "numerical instability" in SF2M—this makes the comparison uneven, as DeepRUOT operates in a regime where the baseline cannot stably run. For UDSB, only 3 time points are used vs. 5 for other methods. These methodological differences make it difficult to attribute performance gains solely to the method.
- **Evaluation asymmetry**: DeepRUOT's predictions are evaluated using learned weights while other methods use uniform weights (Appendix C.1), which could systematically favor DeepRUOT in W_1/W_2 comparisons.
- **Heavy hyperparameter burden**: The two-stage training with dataset-specific epoch scheduling (ranging from 20→10 to 35→120) and multiple hyperparameters (λ_m, λ_d, λ_r, λ_f, λ_w) across stages raises reproducibility concerns and limits practical out-of-the-box usability. Table 6 shows sensitivity to epoch scheduling, suggesting careful tuning is required.
- **Missing quantitative false transition analysis**: The paper claims DeepRUOT "eliminates false transitions" but provides only visual evidence (Figure 2). A quantitative analysis comparing transition rates between methods would strengthen this claim.
- **Limited real-data growth validation**: Growth rate accuracy is validated only on synthetic data with ground truth; for real scRNA-seq datasets, there is no ground truth comparison to validate whether inferred growth patterns align with biological priors (e.g., cell cycle markers).

## Nice-to-Haves
- **Explicit σ=0 ablation**: While Table 4 shows "DeepRUOT w/o growth" and mentions that "DeepRUOT without diffusion corresponds to Unbalanced OT," explicitly reporting σ=0 DeepRUOT results would clarify whether improvements come from stochastic modeling or unbalanced handling.
- **Joint embedding learning**: The method requires pre-computed embeddings (PCA) as input, limiting end-to-end applicability. Jointly learning the embedding space with dynamics would increase practical utility.
- **General case experimental validation**: The main paper focuses on a(x,t) = σ²(t)I with experiments; the general formulation in Appendix D.4 with state-dependent diffusion lacks experimental validation.

## Novel Insights
This paper makes a genuine contribution by bridging the gap between theoretically well-studied RUOT formulations and practical computational methods for high-dimensional stochastic dynamics. The Fisher regularization reformulation provides computational benefits by avoiding log-density derivatives in cross terms. More importantly, the work demonstrates that simultaneously accounting for stochasticity and unbalanced dynamics (growth/death) without prior knowledge is both feasible and beneficial—ablation studies confirm that omitting either component degrades performance. The application to Waddington landscape construction offers a principled framework for visualizing cell fate decisions based on inferred dynamics, which could be valuable for the single-cell biology community.

## Potentially Missed Related Work
- **Physics-based trajectory inference methods**: Methods like scVelo/Velocyto (stochastic velocity models) and CellRank (Markov chain-based trajectory inference) are standard in single-cell analysis for inferring differentiation dynamics. A comparison against these established baselines would strengthen the case for OT-based approaches in this domain.
- **UNICORN and related single-cell optimal transport methods**: The paper cites TrajectoryNet and MIOFlow but does not compare against UNICORN or graph-based OT methods designed specifically for single-cell integration.
- **Stochastic interpolation methods**: The conditional flow matching approach used for score learning has connections to recent work on Schrödinger bridge solvers that may provide alternative computational strategies.

## Suggestions
1. **Unify experimental conditions**: Use identical σ values across all methods or include σ sensitivity ablations to enable fair performance attribution.
2. **Standardize evaluation**: Use uniform weights for all methods or report both weighted and unweighted metrics to avoid evaluation asymmetry.
3. **Provide hyperparameter guidance**: Develop simpler tuning strategies or provide default settings that work across datasets, as the current dataset-specific tuning limits reproducibility.
4. **Quantify transition accuracy**: Compute explicit false positive/negative transition rates to support the claim of eliminating false transitions.
5. **Validate growth rates on real data**: Compare inferred growth patterns against biological markers (e.g., cell cycle genes) to demonstrate practical utility beyond synthetic experiments.

---

## jawV7vhGHw

- GT: Reject (avg 4.2)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
PrAViC proposes a unified probabilistic framework for adapting offline 3D CNN video classification models to online settings with early-exit capability. The approach introduces a normalized expected exit time (NE) function incorporated into the training loss, along with architectural modifications to enable recursive frame-by-frame evaluation without recomputation. Experiments across five datasets (UCF101, EgoGesture, Jester, Kinetics-400, and a medical Ultrasound dataset) demonstrate that the framework can reduce decision time while maintaining or improving accuracy.

## Strengths
- **Well-motivated problem**: Real-time video classification with early-exit capability addresses genuine needs in medical diagnostics, autonomous driving, and safety monitoring, with clear practical applications.
- **Theoretical framework for early exit**: The NE function (Eq. 4) provides a principled way to quantify and optimize decision timing, with the asymmetric loss (penalizing late decisions only for positive/class-1 examples) well-justified for emergency detection scenarios.
- **Comprehensive experimental evaluation**: Testing across multiple datasets (UCF101, EgoGesture, Jester, Kinetics-400, Ultrasound) and diverse base architectures (R3D-18, S3D, X3D variants, MobileNetV2, ShuffleNet, SqueezeNet) demonstrates broad applicability.
- **Practical architectural modifications**: The modifications to 3D convolutions (boundary replication), batch normalization (selective temporal statistics), and head design are clearly explained and enable efficient recursive evaluation as shown in Figure 5.
- **Compelling medical application**: Application to Doppler ultrasound video classification for newborn heart defect detection demonstrates real-world relevance and recursive evaluation efficiency with batching (Figure 7).

## Weaknesses
- **Missing computational efficiency analysis**: For a paper claiming "real-time" classification, there are no FLOPs, latency, or throughput comparisons with baseline methods. Figure 7 shows scaling behavior but not absolute inference times needed to assess real-time feasibility. This is a significant omission given the paper's core claims about speed.
- **Incomplete ablation study**: The paper proposes multiple components (modified convolutions, modified batch normalization, mean aggregation, NE loss) but does not isolate their individual contributions. The ablation study only compares PrAViC against CoX3D with frozen pretrained weights, without testing which specific modification drives the accuracy improvement.
- **Notation inconsistency**: The text refers to "α" in one place ("As the value of α approaches 0") but λ in Equation 5, creating confusion about which parameter controls early exit behavior.
- **Limited theoretical depth**: While a probabilistic framework is presented, no convergence guarantees, optimality bounds, or regret analysis for the early exit strategy are provided. The paper would benefit from formal analysis of the accuracy-efficiency tradeoff.

## Nice-to-Haves
- Report inference latency benchmarks (ms/frame) comparing PrAViC against established online methods (TSM, Köpüklü et al., Co3D) to substantiate real-time claims.
- Include multiple random seeds with mean ± std for main results to assess variance.
- Provide a confusion matrix or case study analysis for different λ values to understand when the model makes correct/incorrect early decisions.
- Complete the GitHub repository URL for reproducibility.

## Novel Insights
The paper's key insight is framing early-exit decisions in video classification through a probabilistic lens, where the decision threshold is modeled as a random variable uniformly distributed over [0,1]. This enables deriving a principled expected exit time formula (NE function) that can be incorporated into training. The asymmetric formulation (only penalizing late decisions for positive-class examples) is particularly well-suited for safety-critical applications where early detection of emergencies (heart failure, accidents) is paramount. The architectural insight that existing 3D CNN weights can be preserved with boundary replication and selective batch normalization is practically valuable for leveraging pretrained models.

## Potentially Missed Related Work
- **CoX3D (Hedegaard & Iosifidis, 2022)**: Already cited as most closely related work. The paper could benefit from more detailed comparison, particularly testing whether CoX3D with early-exit thresholds achieves similar NET values.
- **Temporal Shift Module (TSM, Xiao et al., 2023)**: A key online video classification approach mentioned in related work but not included in experimental comparisons. Adding TSM as a baseline would strengthen comparisons against online state-of-the-art.

## Suggestions
- **Add a comprehensive computational efficiency table** comparing FLOPs, latency (ms/frame), and throughput across all methods in Table 1. This is essential for validating real-time claims.
- **Conduct component-wise ablation**: Test separately (a) architectural changes only, (b) with vs. without NE loss, and (c) different aggregation strategies to identify which innovation drives the most improvement.
- **Clarify notation** throughout: use λ consistently and ensure all equations are self-contained with clear variable definitions.

---

## A1HhtITVEi

- GT: Accept (Poster) (avg 6.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
CheapNet proposes a hierarchical interaction-based model for protein-ligand binding affinity prediction that combines atom-level graph encoding with cluster-level representations via differentiable pooling, followed by bidirectional cross-attention between protein and ligand clusters. The model achieves state-of-the-art performance across diverse benchmarks (LBA, LEP, CSAR NRC-HiQ, CASF-2016, DUD-E) with only 1.33M parameters, significantly outperforming pre-trained models like BindNet (47M+ parameters), while demonstrating superior memory efficiency compared to atom-level attention methods.

## Strengths
- **Strong empirical results with efficiency**: CheapNet consistently outperforms all baselines across multiple benchmark categories, including regression (LBA), classification (LEP), and virtual screening (DUD-E), while using dramatically fewer parameters than pre-trained competitors. Ablation studies demonstrate consistent 3-12% improvements when adding the cluster-attention mechanism to different base encoders (GCN, EGNN, GIGN).
- **Modular architecture**: The design is encoder-agnostic, supporting different base GNNs (GCN, EGNN, GIGN), enabling practitioners to trade off between efficiency and performance based on their needs.
- **Memory efficiency with scalability**: The cluster-level attention mechanism substantially reduces memory footprint compared to atom-level attention approaches (GAABind, DEAttentionDTA), enabling larger batch sizes without OOM issues as demonstrated in Figure 3.
- **Comprehensive evaluation**: The paper evaluates across cross-dataset, diverse protein (LBA 30%/60%), LEP, and external benchmarks (CSAR, CASF-2016), with additional extension to protein-protein affinity prediction, demonstrating broad applicability.
- **Interpretability**: Cross-attention visualizations provide biological insights into which protein-ligand cluster interactions drive predictions, useful for drug discovery applications.

## Weaknesses
- **Missing wall-clock time benchmarks**: While "CheapNet" implies computational efficiency and Figure 3 demonstrates memory advantages, no training/inference time or FLOP comparisons are provided to substantiate speed claims. The "cheap" naming is somewhat misleading without this analysis.
- **LEFTNet and GET not evaluated on cross-dataset benchmarks**: Table 1 (cross-dataset evaluation) omits LEFTNet and GET, the most directly comparable cluster-level baselines. These methods are only compared in Table 2 (diverse protein evaluation). The SOTA claim over cluster-level methods on the primary benchmark is therefore incomplete.
- **Permutation invariance claim is overstated**: Section 3.4 claims permutation invariance, but the proof in Appendix A.3 only establishes invariance for the sum pooling operation. The MLP with residual connection applied after attention (Section 3.3.2) operates on the final representation, and while this is unlikely to break practical invariance, the claim should be more carefully qualified.
- **Limited biological validation of clusters**: The paper demonstrates that learned clusters produce good predictions but does not validate that clusters correspond to known pharmacophores, functional groups, or binding motifs. Without this validation, the biological interpretation of cross-attention maps remains suggestive rather than confirmed.
- **Inconsistent cluster size selection**: Different cluster quantiles perform best on different datasets (Q2 for v2013/v2016, Q4 for v2019), suggesting the optimal cluster count may be task-dependent and warrants more principled selection beyond median-based heuristics.

## Nice-to-Haves
- Runtime benchmarks (wall-clock training/inference time) to fully validate the "Cheap" efficiency claims
- Systematic analysis of what chemical/binding properties learned clusters capture, with comparison to domain-motivated clusters (e.g., pharmacophores, structural motifs)
- Failure case analysis alongside success cases to understand limitations
- Formal statistical comparisons using paired tests rather than Z-tests against reported baselines from different sources

## Novel Insights
CheapNet demonstrates that dynamically learned cluster-level representations with cross-attention can effectively bridge the gap between atom-level detail and higher-order molecular interactions, achieving superior performance to both atom-centric models and predefined cluster-based approaches. The key insight is that end-to-end learned soft clustering, rather than geometrically-defined clusters, better captures functionally relevant groupings for binding prediction. The consistent 3-12% performance gains from adding cluster-attention across different base encoders (GCN, EGNN, GIGN) suggests this hierarchical attention mechanism is a broadly applicable contribution. The extension to protein-protein affinity prediction further indicates the framework's generalizability beyond protein-ligand complexes.

## Potentially Missed Related Work
- **OnionNet-2** (Zheng et al., 2021): A multiple-layer convolutional neural network using element-specific and radial cutoff-based intermolecular contact features that achieves strong performance with different feature engineering approach
- **Pafnucy** (Stepniewska-Dziubinska et al., 2018): A CNN-based approach using 3D structure representations that established baselines on PDBbind that are worth comparing against in more detail
- **EquiDock** (Ganea et al., 2022): SE(3)-equivariant graph neural network for protein-protein docking that shares architectural similarities with CheapNet's approach to protein-protein interactions

## Suggestions
- Add wall-clock time or FLOP comparisons to substantiate computational efficiency claims, which would strengthen the "CheapNet" naming and differentiate from memory-only efficiency
- Include LEFTNet and GET results on the cross-dataset evaluation (v2013/v2016/v2019 core sets) using the same protocol to complete the SOTA comparison against cluster-level methods
- Qualify the permutation invariance claim by clarifying it applies to the attention pooling output specifically, not necessarily to the final MLP prediction
- Evaluate the distance-driven cluster interaction weighting proposed in Appendix A.19 as an ablation to determine whether incorporating 3D proximity priors into cross-attention provides measurable benefits

---

## TwJrTz9cRS

- GT: Accept (Oral) (avg 8.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes HiRA (Hadamard High-Rank Adaptation), a novel parameter-efficient fine-tuning method that achieves high-rank updates through Hadamard product with frozen pretrained weights (ΔW = W₀ ⊙ AB where A∈ℝ^{d×r}, B∈ℝ^{r×k}). The key insight is that unlike LoRA's matrix multiplication update (Rank ≤ r), the Hadamard product enables higher rank (upper bounded by Rank(W₀) × r), allowing greater expressiveness while maintaining the same number of trainable parameters and computational complexity. Experiments on commonsense reasoning, dialogue generation, and mathematical reasoning tasks demonstrate consistent improvements over LoRA, DoRA, and MoRA.

## Strengths
- **Novel methodological contribution**: First work to leverage the Hadamard product specifically for high-rank adaptation in PEFT. The approach provides a theoretically grounded solution to LoRA's rank limitation.
- **Strong and consistent empirical results**: HiRA outperforms all baselines across three diverse task categories and two model sizes. For example, HiRA achieves 86.72% average accuracy vs DoRA's 85.20% (+1.52 points) on commonsense reasoning with Llama-3-8B, and 70.81% vs LoRA's 65.89% (+4.92 points) on GSM8K mathematical reasoning.
- **Practical advantages over LoRA**: Same computational complexity but actually lower GPU memory consumption (61.49GB vs 65.48GB) and faster training (14h vs 15h on Llama-3-8B). Like LoRA, HiRA merges seamlessly into pretrained weights, avoiding inference overhead.
- **Comprehensive ablation studies**: Thorough analysis of rank behavior (Figures 1, 7), singular value distributions (Figures 4, 5), impact of rank dimension r (Figure 6), transformer placement (Table 5), and combination with LoRA (Table 6). The HiLoRA variant achieving 87.48% demonstrates complementary benefits.
- **Sound theoretical foundation**: Rank analysis correctly derives the upper bound Rank(W₀ ⊙ AB) ≤ Rank(W₀) × Rank(AB). Theorem 1 provides expressive power analysis showing HiRA's approximation error depends on σ_{r+1}(E ⊘ W₀) rather than σ_{r+1}(E).

## Weaknesses
- **No error bars reported**: The paper states "HiRA is evaluated over 5 runs with different random seeds" but provides no statistical measures (standard deviation, confidence intervals) in any table. The validation set contains only 120 samples, making variance assessment critical for interpreting marginal improvements.
- **Limited model diversity**: All experiments use only Llama-family models (Llama-2-7B, Llama-3-8B). No evaluation on other architectures (Mistral, Qwen, encoder models like BERT) to support generalization claims. ICLR expects broader evaluation.
- **Missing comparison with recent PEFT methods**: LoRA+ (Hayou et al., 2024), ReLoRA (Lialin et al., 2023), and other high-rank approaches are mentioned in related work but not included as baselines, limiting competitive context.
- **No full fine-tuning baselines for all tasks**: While FFT singular values are analyzed in ablation (Figure 4, 5), downstream performance of full fine-tuning is reported only for commonsense reasoning. For CONVAI2 and math tasks, readers cannot assess whether HiRA's gains over LoRA are meaningful relative to full fine-tuning.
- **Theorem 1 assumption underacknowledged**: The analysis assumes W₀ contains no zeros to avoid division by zero in E ⊘ W₀. While pretrained weights typically have near-zero rather than exact-zero values, this limitation is buried in Appendix B and not prominently discussed.

## Nice-to-Haves
- Evaluate on larger models (e.g., Llama-3-70B) to verify scaling behavior.
- Test on additional math benchmarks (GSM8K-Hard, MATH) alongside GSM8K.
- Compare HiRA against LoRA at comparable *effective* rank (e.g., HiRA r=16 vs LoRA r=128+) rather than only trainable parameters, to validate the "high-rank adaptation" claim directly.
- Ablation on nonlinearities: compare W₀ ⊙ AB against element-wise scaling with learned masks to isolate whether the Hadamard product with W₀ specifically matters.
- Analyze what singular directions HiRA captures that LoRA misses through error analysis on tasks where methods disagree.

## Novel Insights
The paper reveals that Hadamard product with frozen pretrained weights provides an elegant solution to LoRA's rank bottleneck. By expressing ΔW = W₀ ⊙ (AB) where W₀ is frozen and AB is low-rank, HiRA achieves effective rank ~2837 vs LoRA's ~32 with identical parameter count (Figure 1). The theoretical insight that Rank(P ⊙ Q) ≤ Rank(P) × Rank(Q) enables high-rank adaptation under PEFT constraints—addressing the fundamental limitation that LoRA's update ΔW = L₁L₂ has Rank ≤ r. The gradient analysis revealing that HiRA's gradients incorporate W₀ information (while LoRA's do not) explains why HiRA can better leverage pretrained knowledge. The HiLoRA combination further demonstrates that HiRA and LoRA capture complementary aspects, with the hybrid achieving the best overall performance.

## Potentially Missed Related Work
- LoRA+ (Hayou et al., 2024): Low-rank adaptation with differential learning rates, mentioned in related work but not compared in experiments.
- ReLoRA (Lialin et al., 2023): High-rank training through low-rank updates, directly relevant to HiRA's goals but excluded from baselines.
- AdaLoRA: Adaptive rank allocation for LoRA, a natural comparison point for parameter-efficient adaptation.

## Suggestions
The authors should add error bars from the 5 runs to main tables and consider evaluating on at least one non-Llama model architecture to strengthen generalization claims. Comparing HiRA against LoRA at matched effective rank (rather than just trainable dimension) would more directly validate the "high-rank adaptation" contribution.

---

## H25xduunIK

- GT: Reject (avg 5.8)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces "Report Cards" — natural language summaries that describe LLM capabilities on specific skills or topics — as a complementary approach to quantitative benchmarks. The authors propose a three-dimensional evaluation framework (specificity via contrastive accuracy, faithfulness via Elo correlation, interpretability via human scoring) and develop PRESS, an iterative algorithm for generating Report Cards. Experiments across MMLU, Adv. AI Risk, and Chinese Grammar datasets demonstrate that Report Cards capture substantive model capabilities and outperform one-pass prompting baselines.

## Strengths
- **Well-motivated problem framing**: The paper clearly articulates limitations of quantitative benchmarks (Goodhart's law, overfitting, poor interpretability) and positions Report Cards as a principled bridge between raw outputs and summary statistics.
- **Comprehensive evaluation framework**: The three-metric approach systematically addresses key desiderata. Contrastive accuracy measures discriminative power, Elo correlation validates faithfulness against ground truth, and human scoring directly assesses interpretability.
- **Strong empirical validation**: Extensive ablation studies cover format choices, iterative refinement, guesser model capabilities, chain-of-thought effects, and de-stylization robustness — providing a thorough understanding of what drives Report Card effectiveness.
- **Compelling de-stylization analysis**: The experiments demonstrating that Report Cards maintain accuracy on de-stylized completions while few-shot baselines degrade significantly provide convincing evidence that Report Cards capture substantive capabilities rather than surface stylistic artifacts.
- **PRESS demonstrates genuine iterative improvement**: The ablation showing consistent gains from iteration 1 to iteration 5 across all topics validates the progressive refinement approach.

## Weaknesses
- **Circularity concern in faithfulness metric for open-ended tasks**: For CN Grammar and Adv. AI Risk, both Card Elo and Oracle Elo rely on the same LLM judge (gpt-4o-mini), meaning R² between them measures judge consistency rather than ground-truth accuracy. While MMLU uses ground truth for Oracle Elo, the open-ended task results should be framed as "judge-consistency" rather than true faithfulness.
- **Limited human evaluation for interpretability**: With only 230 annotations from 18 volunteers, the interpretability claims rest on thin empirical ground. The weak LLM-human alignment (Spearman 0.27-0.40) further limits confidence in automated scoring.
- **PRESS underperforms few-shot on safety/ethics domains**: On Adv. AI Risk (Corrigible-Less-HHH: 90% vs 74%), the paper attributes this to stylistic artifacts but doesn't provide a principled way to determine which method to use in practice. This represents a significant scope limitation.
- **Specificity metric heavily dependent on guesser capability**: Table 7 shows contrastive accuracy varying by ~35 percentage points (e.g., HS Math: 49%→84%) depending on the guesser model. The metric conflates Report Card quality with guesser intelligence, making cross-method comparisons difficult.

## Nice-to-Haves
- **User studies with domain experts**: Testing whether experts (physics teachers, ethicists) make better predictions with Report Cards would directly validate the human utility claim.
- **Failure case analysis**: Examples where the guesser incorrectly matches models despite having Report Cards would clarify whether specificity failures stem from Report Cards or guesser limitations.
- **Cross-domain generalization tests**: Whether Report Cards for MMLU Physics predict behavior on related but unseen tasks (e.g., AP Physics) would demonstrate practical value beyond the training distribution.
- **Report Card content analysis**: A breakdown of how often Report Cards mention actual capabilities vs. stylistic traits vs. generic statements would clarify what the summaries actually contain.

## Novel Insights
The paper's core insight — that iterative summarization by an LLM evaluator can capture nuanced model capabilities in human-readable form — is both intuitive and underexplored. The de-stylization experiments provide particularly valuable evidence: by paraphrasing model completions and showing that Report Cards remain accurate while few-shot baselines collapse, the paper convincingly demonstrates that the summaries capture substantive behaviors rather than surface artifacts. The PRESS algorithm's progressive refinement design, which synthesizes partial summaries through iterative progression and refinement steps, shows consistent improvement across iterations — validating the intuition that summarizing subsets of examples then merging captures more nuanced behavior than one-pass approaches.

## Potentially Missed Related Work
- **QualEval (Murahari et al., 2024)**: A framework improving traditional metrics with qualitative insights for model improvement — related but complementary, as this work focuses on generating reports for human interpretation rather than model improvement.
- **Self-Challenge (Chen et al., 2024)**: A framework where LLMs identify their own limitations by generating challenging test cases — relevant for understanding systematic weaknesses.
- **Model Cards for Language Models (Mitchell et al., 2019; Arnold et al., 2019)**: The paper discusses these but could more clearly differentiate Report Cards (context-specific, skill-based summaries) from Model Cards (intended use, broader documentation).

## Suggestions
- **Redesign faithfulness evaluation for open-ended tasks**: Rather than comparing Card Elo to Oracle Elo (both using LLM judges), directly validate by having humans assess whether Report Card-based model rankings match their own judgments from viewing raw completions. This would provide genuine ground-truth faithfulness validation.
- **Expand human evaluation substantially**: Recruit domain experts (not just crowd workers) for technical topics and collect at least 5-10x more annotations to support interpretability claims robustly.

---

## lHuLMmz3PY

- GT: Reject (avg 3.5)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
LLM-Exp proposes using large language models to dynamically adjust policy exploration in reinforcement learning by periodically analyzing action-reward trajectories and generating environment-specific probability distributions for exploration. The method is designed as a plug-in compatible with DQN and its variants, demonstrated on Atari games with an average 37.27% improvement in human-normalized scores.

## Strengths
- **Novel contribution**: Leveraging LLM reasoning capabilities for adaptive, environment-specific policy exploration is a genuinely new direction in RL research that addresses the rigidity of prefixed stochastic exploration methods.
- **Comprehensive empirical evaluation**: The paper tests 15 Atari environments, compatibility with 6 DQN variants (DQN, PER-DQN, Dueling-DQN, Rainbow, CURL, Double-DQN), and 5 different LLMs, demonstrating broad applicability of the approach.
- **Thoughtful design trade-off analysis**: Ablation studies on the two-stage prompting design, action sampling density (M), and exploration adjusting interval (K) provide valuable insights into performance versus computational cost trade-offs.
- **Practical design considerations**: Text-only inputs reduce computational overhead and enable compatibility with both multimodal and text-only LLMs, making the approach accessible.
- **Reproducibility**: Code is open-sourced with detailed hyperparameters, prompts, and task descriptions provided in appendices.

## Weaknesses
- **Short training horizon**: The paper uses 100k-500k training steps, approximately 100-1000x shorter than standard Atari benchmarks (typically 10-50M frames). This raises questions about whether improvements persist at scale and whether results are robust enough to validate the approach.
- **Limited statistical rigor**: Results are averaged across only 3 random seeds without statistical significance tests or confidence intervals. With the high variance inherent in RL experiments, some reported improvements may not be statistically meaningful.
- **Incomplete mechanism specification**: The paper does not clearly specify how the LLM-generated probability distribution integrates with the existing ε-greedy mechanism. The exact action selection policy π(a|s) is not formally defined, leaving ambiguity about whether the LLM distribution replaces, supplements, or competes with ε-greedy exploration.
- **Unexplained failure cases**: Qbert (-2.80%) and Seaquest (-4.09%) show performance degradation. The paper does not analyze why the LLM-generated exploration strategy hurts in these environments, limiting understanding of the method's reliability and failure modes.
- **Hand-crafted task descriptions**: The detailed prompts (e.g., specific reward values, game mechanics) require substantial manual engineering, which may limit practical deployability and raise questions about whether the LLM is reasoning from the data or simply following instructions.
- **LLM output reliability**: With temperature=1.0, outputs are stochastic. The paper does not verify that outputs are valid probability distributions or discuss handling of malformed outputs, which could impact training stability.

## Nice-to-Haves
- Compare against more sophisticated exploration baselines such as Noisy-DQN, intrinsic curiosity modules, or count-based exploration methods to contextualize the method's contribution.
- Analyze how the LLM-generated exploration distribution evolves over training to validate the core claim of adaptive exploration.
- Test a simple rule-based baseline (e.g., "if reward decreases, increase exploration") to isolate whether the LLM's linguistic reasoning capability provides value over trivial statistical approaches.
- Evaluate with finer granularity on K (update interval) to better understand the performance-cost trade-off curve.
- Provide wall-clock time and monetary cost estimates for the LLM calls to assess practical deployability.

## Novel Insights
The paper presents the compelling idea that LLMs can serve as adaptive exploration strategy generators by analyzing action-reward trajectories and reasoning about the agent's learning status. The observation that purely textual prompts (without visual state information) can still guide effective exploration in pixel-based environments suggests that LLMs can extract meaningful strategic insights from behavioral data alone. The finding that GPT-4o mini outperforms GPT-4o (a larger model) is intriguing and hints that overly sophisticated models may over-fit to specific actions rather than providing appropriately diverse exploration distributions. This work opens a promising research direction at the intersection of LLM reasoning and RL exploration, though the practical viability remains uncertain given the computational overhead of per-episode LLM calls.

## Potentially Missed Related Work
- LLM-based reward shaping methods (Text2Reward, Wayve et al.) — while related work mentions reward function design, direct comparison with these methods could contextualize the exploration contribution.
- Learned exploration strategies with intrinsic motivation (ICM, RND) — stronger baselines for adaptive exploration beyond prefixed stochastic processes.

## Suggestions
- Clarify the exact integration mechanism between the LLM-generated distribution and ε-greedy action selection with pseudocode or precise mathematical formulation.
- Extend training to standard Atari lengths (at least 10M frames) for a subset of environments to validate that improvements are robust at scale.
- Add a simple rule-based baseline to demonstrate that the LLM's reasoning capability is genuinely contributing value beyond simple heuristics.

---

## e1Z4NCQ146

- GT: Reject (avg 5.2)
- Predicted: Accept (7.0/10)
- Match: No

### Final Review

## Summary
ProTrain is a training system that automatically configures memory management policies (CPU offloading, gradient checkpointing, ZeRO techniques) for LLM training on memory-constrained hardware. It introduces a dual-chunk system for model states and an interleaved activation management strategy, combined with memory-aware profiling and cost models to search for optimal configurations. The system demonstrates 1.43× to 2.71× throughput improvements over DeepSpeed, Colossal-AI, and FSDP, enabling training of models up to 75B parameters on a single A100 GPU.

## Strengths
- **Well-motivated problem**: The paper clearly identifies a genuine pain point—manual tuning of memory management settings requiring substantial expertise—and demonstrates concrete impact with the example of GPT-10B on RTX 3090 using only 35.6% GPU memory with default settings.
- **Clean system architecture**: The three-component design (Automatic Memory Management, Memory-Aware Runtime Profiler, High-Performance Training Engine) is well-structured with distinct responsibilities, and the running example in Section 3 effectively illustrates the memory management policy.
- **Comprehensive evaluation**: Experiments span multiple model architectures (GPT-2, OPT, Mistral, LLaMA variants) across three hardware platforms (RTX 3090, A100, AMD MI300X), with thorough throughput, scalability, and ablation studies.
- **Accurate cost models**: The paper provides mathematical formulations for runtime (within 4% error) and peak memory estimation (within 7% error), validated against actual measurements in Figure 5(b) and Appendix D.4.
- **Ablation validation**: Component-wise ablation studies quantify the contribution of each technique (dual-chunk: 1.1× slowdown, interleaved: 1.04× slowdown, overlapped parameter update: 1.22× slowdown when disabled).

## Weaknesses
- **No convergence validation**: The paper claims "ProTrain does not compromise accuracy" but presents no experiments validating model quality—no training loss curves, perplexity comparisons, or downstream task performance. This is a critical omission that undermines the accuracy claim.
- **Baseline comparison fairness**: FSDP's poor single-GPU results (only 1B model) suggest potential misconfiguration. The paper acknowledges DeepSpeed's configurations were "fine-tuned" but doesn't specify exact values, and explicitly notes FSDP uses "default Adam optimizer" which is less efficient than ProTrain's optimized variants. This raises questions about whether performance gains derive from the method or better hyperparameter selection.
- **Limited multi-node evaluation**: ProTrain targets democratizing LLM access, yet evaluations cover only 1-4 GPUs with a single preliminary 2-node experiment. The paper acknowledges this limitation but doesn't quantify performance at larger scales where cross-GPU communication overhead becomes significant.
- **No statistical rigor**: Throughput measurements are single-run values without variance or confidence intervals. The paper should explicitly state that results are stable across runs or include error bars.

## Nice-to-Haves
- **Configuration transferability analysis**: Show whether ProTrain's automatically found configurations for one model transfer to similar architectures, or quantify what profiling overhead is needed for new models.
- **Memory timeline visualization**: Figure 2 shows conceptual layout but not actual measured memory usage over time for ProTrain vs. baselines during a training iteration.
- **Failure case analysis**: Demonstrate robustness by showing scenarios where ProTrain underperforms or encounters OOM errors.

## Novel Insights
ProTrain's key insight is recognizing that the first few transformer layers and subsequent layers have fundamentally different optimal memory management strategies due to execution order: the first chunk executes first in forward pass (so offloading would block computation with no remaining backward pass to hide parameter update latency), while later chunks can overlap data transfers with computation. This dual-chunk intuition is both technically sound and non-obvious, leading to the persistent/non-persistent chunk abstraction. Similarly, the interleaved activation management (alternating swapping blocks with checkpointing blocks) provides a structured approach to balancing communication overlap against memory savings that is more practical than tensor-level optimization while being more efficient than uniform checkpointing.

## Potentially Missed Related Work
- **PackedLLM** (or similar recent works on memory-efficient LLM serving/training): The related work section covers major frameworks (DeepSpeed, Colossal-AI, FSDP, Megatron-LM) but may not cite recent auto-tuning approaches that also aim to automate memory management configuration.

## Suggestions
- Add end-to-end convergence experiments comparing perplexity or downstream task performance between ProTrain and baselines on at least one model (e.g., GPT-2) over a full training run to substantiate the "no accuracy compromise" claim.
- Document exact baseline configurations in sufficient detail for reproducibility, or compare against AutoGPTQ/autotuning systems to demonstrate ProTrain's advantages more rigorously.
- Include error bars or explicitly note single-run measurements to address statistical rigor concerns.

---

## CU8CNDw6Vv

- GT: Reject (avg 5.2)
- Predicted: Accept (7.0/10)
- Match: No

### Final Review

## Summary
This paper addresses a critical meta-scientific problem in ML: spurious correlations benchmarks (e.g., Waterbirds, CelebA, NICO++) disagree on which mitigation methods perform best, sometimes producing negatively correlated rankings. The authors introduce three desiderata (ERM Failure, Discriminative Power, Convergent Validity) to assess benchmark validity, propose a model-dependent statistic K measuring task difficulty due to spurious correlations, evaluate 8 benchmarks and 22 methods, and provide practical guidance for practitioners.

## Strengths
- **Important empirical finding**: The paper convincingly demonstrates that benchmark disagreement is substantial and consequential—Figure 1b shows the best method on Waterbirds is the second-worst on NICO++. This is a genuine practical barrier for researchers and practitioners.
- **Principled framework**: The three desiderata provide a systematic, theoretically-grounded (drawing on Jacobs & Wallach, 2021) approach to evaluating benchmark validity that can guide future work.
- **Methodological contribution**: The K statistic improves on mutual information by accounting for model learnability of both spurious and core features. Validation through synthetic experiments (varying confounder strength, noise levels) is thorough, and robustness to hyperparameters/reference method choice is demonstrated.
- **Comprehensive evaluation**: Analysis spans 8 SubpopBench benchmarks, 22 mitigation methods, and two sanity-check benchmarks (Citybirds, AvP), with extensive appendices supporting the methodology.
- **Actionable practical guidance**: Despite limited validation, the paper provides concrete recommendations (filter to valid benchmarks, use closest benchmark) that address a real practitioner need.

## Weaknesses
- **Limited validation of practical recommendations**: The core practical claims—that averaging over valid benchmarks (6/9 improvements) and using the closest benchmark (5/8 improvements) outperforms standard averaging—are based on small numbers without statistical significance tests. Dollar Street validation yields a marginal 0.6 percentage point improvement (58.65→59.25), which is likely within noise. Claims should be framed more cautiously or validated with larger experiments.
- **Architecture dependence unexamined**: K and benchmark validity assessments are computed only with ResNet-50 (vision) and BERT (language). The paper acknowledges architecture dependence but does not explore whether conclusions transfer across architectures (e.g., ViT, CLIP). Given that benchmark validity is the paper's main contribution, demonstrating robustness across architectures is essential.
- **Circularity in validity assessment**: Evaluating benchmarks against each other (Convergent Validity) creates subtle circularity—if all benchmarks are flawed, there is no ground truth anchor. The Citybirds/AvP sanity checks partially address this but do not fully resolve it.
- **No comparison to baselines for closest-benchmark strategy**: The paper does not report what happens with naive alternatives (always picking a fixed benchmark, random selection). Without these baselines, the practical value of the approach is unclear.
- **Convergent Validity assumption underexplained**: The paper assumes valid benchmarks should agree, but the theoretical justification for why this correlates with "true" construct validity is thin. Additional analysis (e.g., ruling out confounders like similar noise floors or class imbalance) would strengthen this claim.

## Nice-to-Haves
- Ablation analyzing which combinations of desiderata are necessary/sufficient for benchmark filtering
- Formal statistical significance tests for the practical recommendation claims
- Investigation of why text benchmarks (CivilComments, MultiNLI) fail to achieve Convergent Validity
- Analysis of whether robust methods (CRT, ReWeight, ReSample, GroupDRO) share algorithmic properties, to help practitioners generalize to new methods

## Novel Insights
This paper's key insight is that benchmark disagreement in spurious correlations research is not merely a nuisance to be averaged over, but a symptom of invalid measurement tools. The three desiderata framework reframes benchmark evaluation from "which methods perform best" to "which benchmarks actually measure what we care about." The K statistic's ability to capture model-dependent task difficulty (rather than just data-level mutual information) is a useful conceptual contribution, revealing that low-K benchmarks like Dollar Street may have other failure modes beyond spurious correlations. The finding that only a handful of methods (CRT, ReWeight, ReSample, GroupDRO) show both high performance and low variability across benchmarks is practically valuable.

## Potentially Missed Related Work
- None identified—the paper adequately covers SubpopBench (Yang et al., 2023), WILDS (Koh et al., 2021), and related benchmark validity work (Jacobs & Wallach, 2021; Blodgett et al., 2021).

## Suggestions
1. **Strengthen practical recommendation evidence**: Conduct more experiments or reframe claims as exploratory/hypothesis-generating rather than recommendations, given the small margins and lack of statistical significance.
2. **Add architecture ablation**: Test K and benchmark validity with at least one alternative architecture (e.g., ViT or CLIP for vision) to demonstrate transferability of conclusions.
3. **Include baseline comparisons**: Report performance of naive strategies (fixed benchmark, random) for the closest-benchmark selection to contextualize the 5/8 improvement rate.

---

## R1hIXdST22

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (8.0/10)
- Match: Yes

### Final Review

## Summary
MR.Q presents a general-purpose model-free deep RL algorithm that learns state-action embeddings through model-based objectives (predicting rewards, dynamics, and terminations), enabling approximately linear representations of the value function while using non-linear function approximation for value estimation. The method achieves competitive aggregate performance across 118 environments spanning 4 diverse benchmarks (Gym locomotion, DMC proprioceptive, DMC visual, Atari) with a single fixed set of hyperparameters, while being computationally more efficient than model-based alternatives.

## Strengths
- **Comprehensive empirical evaluation**: Testing across 4 benchmarks and 118 environments with rigorous bootstrap confidence intervals is exemplary for RL research, directly addressing the paper's "general-purpose" claim.
- **Thorough design study**: The ablation experiments validate that each design choice contributes meaningfully—particularly the finding that "No MR" (removing model-based representations) causes catastrophic degradation (-0.07 to -0.78) across all benchmarks, confirming the core contribution.
- **Theoretical grounding with practical relevance**: Connecting modern dynamics-based representation learning to classical linear MDP theory (Parr et al., 2008) provides principled justification, while the design study empirically validates the theoretical relaxations.
- **Computational efficiency**: Figure 1 demonstrates MR.Q uses fewer parameters (4.1M vs 187.3M for DreamerV3) and achieves faster training (1.9k vs 236 FPS on DMC), making general-purpose RL more accessible.
- **Valuable community insight**: The finding that "MSE reward loss" and "No unroll" improve Gym but degrade Atari performance highlights benchmark-specific overfitting in RL research, encouraging more rigorous multi-domain evaluation.

## Weaknesses
- **Substantial Atari performance gap**: MR.Q achieves 2.54 mean human-normalized score versus DreamerV3's 3.74 on Atari—a 32% gap. The paper acknowledges this but provides no analysis of which games drive this failure or why model-free approaches struggle relative to model-based ones on this specific benchmark.
- **Theoretical assumptions significantly relaxed**: Theorem 1 shows equivalence under linear assumptions, but the practical algorithm relaxes this substantially (state-dependent dynamics target, target networks, non-linear value function). Theorem 3 requires exact MDP homomorphism (Equation 11), which cannot hold with neural networks. The paper does not quantify how close learned representations come to satisfying this condition, leaving a gap between theory and practice.
- **Incomplete comparison with TD7**: TD7 (Fujimoto et al., 2024), from the same first author, is presented as a state-of-the-art Gym baseline but appears in DMC tables without discussion. Given TD7's strong performance (1.57 mean TD3-normalized vs MR.Q's 1.46), understanding this gap would strengthen claims about MR.Q's contribution.

## Nice-to-Haves
- **Compute-normalized comparison**: The paper claims MR.Q is "simpler and less expensive" but provides only FPS metrics. A wall-clock time comparison with matched compute budgets would strengthen efficiency claims.
- **Scaling study**: The paper targets "general-purpose" learning but never shows how MR.Q scales with embedding dimension or data, leaving unclear whether the single hyperparameters are truly robust across scales.
- **Per-environment analysis in main paper**: Aggregate results dominate the presentation; highlighting specific failure cases (e.g., MontezumaRevenge at ~50, Skiing at -30000) would help readers understand practical limitations.

## Novel Insights
The paper's central insight—that model-based representation learning objectives (predicting rewards, dynamics, and terminations) can enable general-purpose model-free RL with a single hyperparameter set—is both theoretically grounded and empirically validated. The design study reveals a crucial asymmetry: techniques that improve performance on one benchmark often hurt others (e.g., MSE reward loss helps Gym but devastates Atari performance with -0.79 degradation). This suggests that prior RL research may have been inadvertently overfitting hyperparameters to specific benchmarks, and that multi-domain evaluation is essential for understanding what truly generalizes. The finding that non-linear models do not improve over the linear MDP predictor further validates the theoretical claim that maintaining approximately linear representations is beneficial.

## Potentially Missed Related Work
None identified—the paper provides comprehensive coverage of dynamics-based representation learning, general-purpose RL, and state abstraction literature.

## Suggestions
The Atari performance gap warrants deeper investigation. Adding an analysis of which specific Atari games fail and why would either guide improvements to MR.Q or provide principled reasons why model-free methods cannot match model-based approaches on certain discrete-action pixel domains, strengthening the paper's contribution either way.

---

## VB8xHF1Rdl

- GT: Reject (avg 3.5)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper introduces an information-theoretic framework for quantifying representational structure in large language models, proposing three measures—regularity, variation, and disentanglement—derived from linguistic concepts. The core technical contribution is "soft entropy," a novel, differentiable, and memory-efficient method for entropy estimation in high-dimensional vector spaces via projection to the unit sphere with softmax normalization. The authors analyze training dynamics over 2M steps, scaling effects across models from 14M to 12B parameters, and demonstrate correlations between pre-training representational structure and post-fine-tuning GLUE benchmark performance.

## Strengths
- **Novel entropy estimation method**: The soft entropy approach (Section 4) provides a practical alternative to binning or clustering, being fully differentiable, memory-efficient, and parallelizable while performing comparably to existing methods on reference distributions. This enables analysis of large-scale models that would be computationally prohibitive with prior approaches.
- **Comprehensive experimental scope**: The analysis spans training dynamics with 5 initializations over 2M steps, scaling effects across the Pythia suite (14M–12B), controlled comparisons of depth vs. dimensionality in BERT variants, and correlation analyses with GLUE tasks using 25 different initializations. This breadth of experimentation strengthens confidence in the framework's utility.
- **Linguistic grounding and interpretability**: Connecting representational structure to linguistic concepts (regularity/disentanglement/variation) provides intuitive interpretation and situates the work within a broader scientific context, drawing parallels between language as a structured mapping system and learned representations.
- **Subspace entropy for fair model comparison**: The multi-headed subspace entropy approach enables meaningful comparisons between models of different dimensionalities by computing entropy over equally-sized subspaces—a principled solution to a real methodological challenge in comparing models with different hidden sizes.

## Weaknesses
- **Overstated predictive claims**: The paper claims to "predict downstream task performance...before fine-tuning," but demonstrates only Spearman correlations (r ≈ 0.17–0.31 for QNLI). This is correlational analysis, not prediction—nor is it truly out-of-distribution given the in-domain Wikipedia data. The language should be toned down to reflect this, and the practical utility of correlations explaining ~2-10% of variance should be contextualized more clearly.
- **Soft entropy hyperparameters lack principled justification**: The paper uses 50 bins and rescales distances to [-100, 100], acknowledging this is analogous to kernel bandwidth in KDE. However, no guidance is provided for choosing these values, and no robustness analysis demonstrates that conclusions hold across reasonable hyperparameter settings. This leaves uncertainty about reproducibility.
- **No ablation of key design choices**: The paper does not systematically ablate the number of bins, the rescaling factor, or distance metrics (cosine vs. Euclidean), despite acknowledging sensitivity to these choices. This leaves uncertainty about which findings are robust to implementation details.
- **Limited theoretical grounding for soft entropy**: While the paper draws connections to Jaynes' limiting density of discrete points and kernel density estimation, the mathematical relationship between soft entropy and true discrete entropy is not rigorously established. The claim that "cross-dimensional dependencies are relatively small" when using subspace decomposition is asserted rather than demonstrated with supporting analysis.

## Nice-to-Haves
- **Direct comparison with probing baselines**: The paper positions itself as a "non-parametric approach to probing" but provides minimal comparison with established probing methods (e.g., logistic regression probes, MDL probing) on the same downstream tasks. Such comparisons would strengthen the value proposition.
- **Validation of linguistic interpretation**: The paper uses token/bigram/trigram labels as proxies for lexical and contextual information but does not validate that these measures correlate with independently-measured linguistic properties (e.g., syntactic structure, semantic similarity). The linguistic framing would benefit from empirical grounding.
- **Convergence analysis**: Using 256K sentences (or 2.5M) without analyzing whether estimates converge with more data leaves questions about sampling sufficiency unaddressed.
- **Visualizations of representation structure**: Concrete examples showing how representations for different labels are distributed in space (e.g., t-SNE/UMAP) would make the abstract measures tangible.

## Novel Insights
The paper's most compelling insight is the systematic characterization of training dynamics as a shift from token-level alignment to contextual representation—early training rapidly aligns with tokens, followed by contextualization where token disentanglement drops while bigram/trigram disentanglement increases. The finding that larger models proportionally compress representations in subspaces while expanding overall capacity is intriguing and connects meaningfully to Shannon's source coding intuition. The observation that representational structure at pre-training correlates with post-fine-tuning performance across multiple GLUE tasks—even when estimated on non-task data—is potentially valuable for understanding what structural properties support generalization.

## Potentially Missed Related Work
- Information bottleneck theory applications to transformers (Shwartz-Ziv & Tishby, 2017; Saxe et al., 2019) — while cited, a more direct comparison of methodology and findings would strengthen positioning
- Recent work on anisotropy and representation geometry in LLMs that may provide additional validation or alternative interpretations of the entropy measures

## Suggestions
The paper should reframe "predicting downstream performance" as "correlating pre-training structure with downstream performance" and include ablation studies for key hyperparameters (bins, scaling factor) to demonstrate robustness. Adding a dedicated limitations section discussing the assumptions underlying soft entropy and subspace decomposition would strengthen the paper for ICLR review.

---

## EXnDAXyVxw

- GT: Reject (avg 5.0)
- Predicted: Accept (7.0/10)
- Match: No

### Final Review

## Summary
This paper introduces QT-DoG (Quantization-aware Training for Domain Generalization), demonstrating that weight quantization naturally serves as an implicit regularizer that promotes flatter minima in the loss landscape, thereby improving out-of-distribution generalization. The authors further propose EoQ (Ensemble of Quantization), which ensembles multiple quantized models to achieve state-of-the-art performance on the DomainBed benchmark (68.4% average) while being approximately 4× smaller than full-precision ensemble methods. The work represents the first explicit exploration connecting quantization techniques with domain generalization.

## Strengths
- **Novel intersection of fields**: First work to explicitly connect quantization (typically used for model compression) with domain generalization, providing a fresh perspective on both areas. This reframe of quantization as a training regularizer is genuinely insightful.
- **Comprehensive empirical validation**: Extensive experiments across 5 DomainBed datasets (PACS, VLCS, OfficeHome, TerraIncognita, DomainNet), multiple architectures (ResNet-50, ResNeXt-50, DeiT-Small), and ablation studies covering quantization algorithms (LSQ, INQ, OBC), bit precision (5-8 bits), and combinations with other DG methods (CORAL, MixStyle).
- **Resource efficiency with improved accuracy**: EoQ achieves SOTA 68.4% average accuracy while maintaining a memory footprint comparable to single-model methods (1.1×), substantially smaller than ensemble competitors like DiWA (60×) or EoA (6×).
- **Training stability insight**: Figure 3 convincingly demonstrates that quantization stabilizes OOD performance during training—a valuable practical insight where vanilla ERM shows unstable OOD performance while quantized models remain stable.
- **Flatness verification**: Figure 2 empirically validates that QT-DoG finds flatter minima than ERM, SAM, and SWA, achieving comparable flatness to SWAD despite being 75% smaller.

## Weaknesses
- **Non-monotonic bit-precision relationship unexplained**: Table 8 shows 7-bit > 8-bit > 6-bit > 5-bit performance, contradicting a straightforward "more noise = more regularization" narrative. The paper never explains why 7-bit is optimal rather than 8-bit (less noise) or 5-bit (more noise). This undermines the core theoretical story and suggests the mechanism is more complex than presented.
- **Quantization timing (when to quantize) poorly motivated**: The paper quantizes at 2000 steps with Table 13 showing a U-shaped curve (2000 > 1000 > 3000 > 4000) that is never explained. This critical hyperparameter leaves the method partially unreproducible—practitioners have no principled guidance for selecting quantization timing on new datasets.
- **Missing ablation separating quantization from ensemble effects**: EoQ combines both quantization and ensembling, but the paper does not isolate their individual contributions. Comparing EoQ against an ensemble of full-precision models (e.g., ERM Ens. or DiWA-style averaging) would clarify whether quantization specifically enables the improvement or if diversity from multiple initializations is sufficient. This is a significant gap for understanding the mechanism.

## Nice-to-Haves
- **Combine QT-DoG with SWAD**: SWAD is the strongest single-model baseline (66.9%). Combining flat-minima finding (SWAD) with quantization (QT-DoG) is the natural next step to validate mechanism complementarity.
- **Compare against direct weight noise injection**: The paper argues quantization noise improves generalization, but never compares against adding Gaussian noise directly to weights. This comparison would establish what quantization specifically provides beyond generic noise regularization.
- **Quantify GradCAM improvements**: Figure 5 shows cherry-picked examples where QT-DoG focuses on correct image regions, but without percentage statistics across the full dataset, this remains anecdotal.
- **Explore mixed-precision quantization per layer**: The paper uses uniform 7-bit quantization and acknowledges mixed-precision could help but doesn't explore it despite this being an obvious improvement direction.

## Novel Insights
The paper's core insight—that quantization noise during training implicitly regularizes the model toward flatter minima, thereby improving domain generalization—is genuinely novel. This reframes quantization from a deployment-focused compression technique to a training regularizer with generalization benefits. The observation that EoQ (ensemble of quantized models) achieves SOTA with minimal memory overhead compared to full-precision ensembles is practically significant. The training stability plot (Figure 3) demonstrating that quantization makes OOD performance trackable via in-domain validation is a valuable practical insight for model selection in DG.

## Potentially Missed Related Work
- Weight noise injection regularization baselines (An, 1996; Murray & Edwards, 1992) — while cited for the theoretical intuition, direct experimental comparison with Gaussian weight noise is missing, which would help differentiate quantization-specific benefits from generic noise regularization.
- HAWQ (Dong et al., 2019) and mixed-precision quantization work — the paper mentions mixed-precision as future work but doesn't compare against layer-wise or channel-wise mixed-precision approaches during training.

## Suggestions
1. **Add ablation comparing EoQ against full-precision ensemble**: Train 5 full-precision models with ERM and compare against EoQ to isolate quantization's contribution from ensemble diversity.
2. **Explain the 7-bit optimality**: Investigate and report why 7-bit outperforms both 8-bit and 5-bit—perhaps through Hessian eigenvalue analysis or noise magnitude calibration.
3. **Provide guidance for quantization timing**: Either derive a principled method for selecting when to quantize or demonstrate robustness to timing choices across datasets.

---

## XCugWIuHR8

- GT: Reject (avg 3.0)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes "Convex Distillation," a knowledge distillation approach that compresses deep neural networks by replacing non-convex layers with convex-gated architectures in the student model. The method enables label-free compression by matching intermediate activations between teacher and student, eliminating the need for post-compression fine-tuning on labeled data. Experiments on CIFAR-10, SVHN, TinyImageNet, and Visual Wake Words demonstrate competitive performance compared to non-convex distillation, particularly in low-data and high-compression regimes.

## Strengths
- **Novel conceptual contribution**: Combining convex optimization with knowledge distillation addresses a genuine gap—leveraging the expressive power of non-convex teacher models while exploiting the favorable optimization landscape of convex student models is conceptually new.
- **Theoretical grounding**: The paper builds on established convex reformulation results (Pilanci & Ergen, 2020; Sahiner et al., 2021; 2022a) and correctly identifies that convex student modules enable specialized solvers (e.g., R-FISTA) for faster convergence.
- **Label-free capability**: The activation-matching approach (Equation 6) requires no labeled data, providing practical value in annotation-scarce scenarios.
- **Comprehensive empirical evaluation**: Experiments span four datasets and compare against multiple baselines (fine-tuning, non-convex distillation, pruning), with results demonstrating advantages in low-data and high-compression regimes.

## Weaknesses
- **Incomplete ablation isolating convexity benefits**: The paper compares convex distillation against a non-convex version using different optimizers (R-FISTA vs. Adam). It remains unclear whether performance gains derive from convexity itself or superior solver characteristics. A controlled comparison using identical optimizers would strengthen the claims.
- **Polishing step lacks rigorous justification**: The "polishing" technique (Section 4.3) using group elastic net regression on W2 is motivated heuristically. The paper does not clearly explain why information sharing across output dimensions improves results, and the empirical improvement in Figure 7 is marginal.
- **CNN block derivation is loose**: The paper presents the convex CNN block (Equation 8) without rigorous derivation connecting it to the GReLU theorems. While the references (Sahiner et al., 2021; 2022a; Gupta et al., 2021) extend convex reformulations to convolutions, the specific architectural choices lack explicit justification.
- **Conditional performance advantage understated**: The paper claims convex distillation "outperforms" non-convex methods, but Figure 6 shows non-convex methods catching up with relaxed resource constraints. The regime-dependent nature of this advantage is understated in the abstract and conclusions.

## Nice-to-Haves
- Include standard knowledge distillation baselines (Hinton et al. 2015, FitNets, attention transfer) for more comprehensive comparison beyond the bespoke non-convex version.
- Validate on architectures beyond ResNet-18 and MobileNetV3 (e.g., Vision Transformers, EfficientNet variants) to support broader applicability claims.
- Report error bars across figures (beyond Figure 5 which mentions 10-seed repetition) to quantify uncertainty in performance claims.
- Measure actual inference latency, memory footprint, and throughput on real edge hardware to validate deployment claims beyond parameter count metrics.

## Novel Insights
The paper's key insight—bridging non-convex teacher representations with convex student architectures via activation matching—opens a promising research direction at the intersection of convex optimization and deep learning. The observation that convex student models can achieve comparable performance to non-convex counterparts when leveraging rich pre-trained features, particularly in low-data and high-compression regimes, suggests that implicit regularization benefits of convex optimization may be particularly valuable for transfer learning scenarios. This challenges the conventional wisdom that non-convexity is necessary for expressive deep learning.

## Potentially Missed Related Work
- **Contrastive Representation Distillation (CRD)** (Tian et al., 2019) — provides stronger distillation baselines for comparison
- **DeepInversion** (Yin et al., 2020) — alternative data-free distillation method that could serve as comparison baseline
- **WSLD** (Nayak et al., 2019) — addresses similar label-free distillation scenarios

## Suggestions
Conduct an ablation study that isolates the effect of convexity from optimization method by training both convex and non-convex students using the same optimizer (e.g., both with Adam), then separately comparing the effect of using specialized convex solvers vs. general-purpose optimizers. This would cleanly attribute performance differences to the appropriate factor.

---

## KgKN7F0PyQ

- GT: Reject (avg 4.5)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
ReAcTree proposes a hierarchical task planning method that dynamically decomposes complex tasks into manageable subgoals using a tree structure composed of control flow nodes (inspired by behavior trees) and agent nodes (each an LLM-based planner). The method introduces an "expand" action enabling dynamic tree growth, along with episodic memory for retrieving goal-specific in-context examples and working memory for sharing observations across nodes. Experiments on WAH-NL and ALFRED datasets demonstrate consistent improvements over ReAct across multiple LLMs (e.g., 63% vs 24% GSR with Qwen2.5 72B on WAH-NL).

## Strengths
- **Well-motivated approach**: The paper clearly identifies two key limitations of flat LLM planning—single sequential decision-making and context dilution from long trajectories—and addresses both with a principled hierarchical decomposition.
- **Comprehensive experiments**: The method is evaluated across 7+ LLMs (8B-72B parameter ranges) with consistent improvements, demonstrating robustness across model families.
- **Thoughtful memory systems**: The episodic memory with agent-level retrieval and working memory for object location tracking are well-designed components with clear ablation evidence of their contribution.
- **Extensive failure analysis**: Section E provides valuable categorization of failure types (Expand-level, Agent-level, Constraints, Instruction errors) with quantitative breakdowns, which is exemplary for reproducibility and future research.
- **Code availability**: Anonymous code is provided, supporting reproducibility claims.

## Weaknesses
- **Incomplete baseline comparison**: The paper compares only against ReAct, despite discussing multiple directly related methods (DEPS, AdaPlanner, Tree-Planner, LLM-MCTS, ToT) in related work. ICLR expects comparison with the most relevant prior work, not just the oldest baseline. This makes it difficult to assess whether the tree structure itself provides value versus other design choices.

- **Missing component ablation**: A ReAcTree variant without episodic memory (or without any memory system) is not compared against the full system. This ablation is essential to isolate the contribution of the tree decomposition mechanism from the memory systems. The paper shows agent-level vs. task-level retrieval but never quantifies what the memory systems add over the base tree structure.

- **Limited episodic memory construction**: The episodic memory is built from only 5 human-annotated trajectories (one per task category) plus successfully completed training tasks. With such a small memory bank, it's unclear whether episodic retrieval actually occurs during testing or if performance is dominated by the tree structure alone. Analysis of retrieval hit rates would strengthen this claim.

- **Incomplete specification of parallel node aggregation**: The paper states outcomes are "aggregated according to a predefined policy" but never specifies what this policy is. This is a missing detail needed for reproducibility.

- **Partial observability comparison caveat**: The paper evaluates in a partially observable setting adapted from LoTa-Bench, but it's unclear whether ReAct's lower performance (e.g., 24% GSR) reflects algorithmic limitations or the difficulty of ReAct under partial observability. This makes the ReAcTree improvements appear more dramatic than they might be against a ReAct variant specifically optimized for partial observability.

## Nice-to-Haves
- Report confidence intervals or standard deviations across multiple runs to demonstrate statistical significance of improvements.
- Analyze retrieval hit rates to verify episodic memory is actually being utilized.
- Break down performance by task complexity (number of subgoals) to validate that hierarchical decomposition helps more for longer tasks.
- Add visualization of tree depth distributions across successful vs. failed tasks.
- Provide case studies analyzing why ReAcTree sometimes underperforms (e.g., Qwen2 7B on ALFRED).

## Novel Insights
ReAcTree's key insight is that LLM-based task planning can benefit from **dynamic hierarchical decomposition** where agents themselves decide when and how to break down goals, rather than relying on predefined two-level hierarchies. The combination of behavior tree-style control flow (sequence/fallback/parallel) with agent-level memory systems creates a principled framework for managing complexity. The finding that ReAcTree with smaller models (8B) can outperform ReAct with much larger models (70B) is particularly noteworthy, suggesting that decomposing tasks into simpler subproblems may be more effective than simply scaling model size.

## Potentially Missed Related Work
- **Tree-Planner (Hu et al., 2023)**: Constructs action trees and executes actions, similar hierarchical planning approach—direct comparison would contextualize ReAcTree's contribution.
- **LLM-MCTS (Zhao et al., 2024)**: Tree search-based planning with Monte Carlo Tree Search—comparison would highlight differences between learned tree expansion vs. search-based approaches.
- **MOSAIC (Wang et al., 2024a)**: Task planner with tree structure for LLM agents—the paper briefly mentions this but doesn't compare experimentally.

## Suggestions
1. **Add a "ReAcTree-NoMemory" baseline**: Strip both episodic and working memory to isolate the tree decomposition contribution from the memory systems. This minimal ablation is essential to support the paper's core claims.

2. **Implement and compare against at least one tree-based baseline** (e.g., Tree-Planner or ToT) under the same experimental settings to properly contextualize the contribution.

3. **Clarify the parallel node aggregation policy**: Specify the exact success/failure aggregation logic, as this affects how parallel tasks interact.

4. **Expand episodic memory experiments**: Use more training tasks for memory construction and analyze retrieval statistics to verify the memory system is actually being utilized.

---

## 10kBEqYKKN

- GT: Reject (avg 3.0)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper investigates how different prompts influence latent representations in large language models under zero-shot settings, examining whether prompts modify intrinsic dimensionality (via IsoScore and PCA) and whether prompts can be grouped based on their geometric influence. The authors analyze four model families across three binary sentiment classification datasets and find that prompts significantly alter vector distributions, with impact being more model-dependent than prompt-dependent, and that clustering reveals counter-intuitive groupings where semantically similar prompts are not necessarily processed similarly.

## Strengths
- **Broad empirical coverage**: The study examines four major LLM families (Phi, Gemma, Zephyr, Bloomz) across three datasets (Rotten Tomatoes, IMDB, Yelp), providing systematic coverage of model families and scales under consistent experimental conditions.

- **Methodological rigor in geometric analysis**: The use of IsoScore (a rotation-invariant, mean-agnostic metric) combined with PCA variance analysis and K-means clustering provides complementary perspectives on representation geometry that directly address the stated research questions.

- **Valuable empirical finding about model-dependent processing**: The observation that models group prompts by geometric features rather than semantic similarity is an important and non-obvious finding that has implications for understanding how different architectures process inputs.

- **Clear documentation of layer-wise evolution**: Tracking how isotropy evolves across layers reveals that smaller models consistently show higher isotropy, providing concrete evidence about how model scale affects representation utilization.

## Weaknesses

- **Exclusive focus on EOS token representations limits scope**: The analysis extracts only End-of-Sentence token representations, which captures the final output state but not how prompts influence intermediate processing. This severely limits what can be concluded about "how prompts influence representations" during the model's forward pass. The paper acknowledges this limitation but it remains a fundamental constraint on the contribution.

- **Lack of statistical significance testing**: All findings are presented as qualitative observations without statistical tests. The IsoScore variations across prompts (with coefficients of variation ranging from 20% to 131%) are reported without significance tests, making it impossible to determine whether observed differences are reliable or within noise.

- **No quantitative correlation between geometry and performance**: The paper states "bad performance seems to be correlated with extreme isotropy" but provides no quantitative correlation measure. This is a missed opportunity to directly connect geometric measures to task performance, which is central to the paper's motivation.

- **Insufficient explanation of counter-intuitive clustering**: The finding that semantically similar prompts cluster together less often than expected is interesting but under-analyzed. The paper does not characterize what geometric features actually separate the clusters, leaving the finding descriptive rather than explanatory.

## Nice-to-Haves
- Expanding analysis to intermediate token representations (not just EOS) would reveal how prompts are processed throughout the model, not just at the output.
- Adding TF-IDF or n-gram overlap baselines for clustering would establish whether geometric clusters reveal anything beyond lexical similarity.
- Testing on non-sentiment tasks (e.g., question answering, natural language inference) would strengthen claims about generalizability beyond binary classification.
- Releasing code and implementation details would improve reproducibility, which is currently limited by vague descriptions of prompt modifications and clustering methodology.

## Novel Insights
The most valuable contribution is the empirical demonstration that prompts with similar natural language semantics are processed differently by different model families. The clustering analysis reveals that geometric features used by K-means to group prompts do not align with intuitive semantic groupings—for example, "Movie Expressed Sentiment" and "Movie Expressed Sentiment 2" cluster together only 6.71% of the time, while "Movie Expressed Sentiment 2" clusters with "Text Expressed Sentiment" 20.81% of the time. This suggests that the pre-trained knowledge leveraged by models is highly sensitive to input form in ways that are not captured by surface-level semantic similarity, and that architectures, pre-training data, and training paradigms leave detectable traces in how models process their inputs.

## Potentially Missed Related Work
- **Compositionality in contextual embeddings** (Cai et al., 2021): Directly relevant as they study isotropy and cluster structure in contextual embedding spaces.
- **Layer-wise similarity and fine-tuning** (Kornblith et al., 2019): Relevant for understanding how different models produce non-comparable representations.
- **Structural constraints in pretrained representations** (Phang et al., 2021): Related work on clustering in transformer representations that the paper could connect to more explicitly.

## Suggestions
The paper would benefit most from: (1) adding statistical significance tests for isotropy comparisons across prompts and models, (2) computing and reporting quantitative correlations between geometric measures (IsoScore, PCA variance) and task accuracy, and (3) performing probing analysis to identify what linguistic features in the representations drive the observed clustering behavior, thereby explaining why certain prompts cluster together.

---

## gFUomIaycw

- GT: Reject (avg 2.5)
- Predicted: Reject (3.0/10)
- Match: Yes

### Final Review

## Summary
This paper proposes DR-MoE, a dynamic routing mixture-of-experts model for multi-label image classification that replaces HSQ's static gating with input-dependent gating networks. While the approach shows genuine improvements on PASCAL VOC 2007 (94.8% vs 93.7% mAP) and secondary metrics (OF1/CF1), it paradoxically regresses on MS-COCO mAP (85.9% vs 87.1%) compared to HSQ—the most widely-reported benchmark metric.

## Strengths
- Well-motivated problem: The paper correctly identifies static gating's inability to adapt to varying label dependencies as a genuine limitation in HSQ.
- Clear architectural design: The four-component architecture (backbone, transformer query module, shared/task-specific experts, dynamic gating) is logically structured and reproducible.
- Comprehensive ablation studies: Tables 4-5 provide useful analysis of hyperparameters (number of shared experts, gating network complexity), confirming 4 shared experts and 256-dim hidden layers as optimal.
- Effective regularization: Entropy regularization on gating weights (Section 3.9) is a sensible technique to prevent collapse to sparse solutions.

## Weaknesses
- **Misleading performance claims**: The abstract and introduction claim "state-of-the-art" and "outperforming" HSQ, but Table 1 shows DR-MoE (85.9%) actually underperforms HSQ (87.1%) on MS-COCO mAP—the primary metric for MLC. This selective reporting of metrics where the model wins while omitting the primary metric regression is a credibility issue.
- **No explanation for mAP/OF1-CF1 trade-off**: Table 3 shows switching from static to dynamic gating *decreases* mAP (87.1→85.9) while increasing OF1/CF1. The paper never explains why this trade-off occurs or which metric better reflects the actual improvement. Without this analysis, the results are difficult to interpret.
- **Incremental novelty**: The main contribution is replacing HSQ's static gating with per-label MLPs. Dynamic gating in MoE is well-established (GShard, Hash Layer), and the Q2L-style transformer decoder is directly borrowed. This is a reasonable but incremental extension rather than a fundamental innovation.
- **Computational overhead unquantified**: The paper acknowledges "additional computational overhead" but provides zero numbers (FLOPs, parameter counts, latency). With L task-specific gating networks plus shared expert gating, inference cost scales poorly with label count.
- **Gating input limitation**: The gating network Gi takes hi (label-specific representation) as input, meaning the gating decision is based on what the transformer already learned about label i, not on broader image-level context that could inform cross-label dynamics.

## Nice-to-Haves
- Multi-run experiments with statistical significance testing (mean ± std over 3+ seeds) — single-run comparisons are insufficient at ICLR standards.
- Per-class analysis on MS-COCO (only provided for PASCAL VOC) to verify claims about "high heterogeneity" labels benefiting from dynamic routing.
- Aggregate statistics on gating weight distributions across the test set — the paper shows cherry-picked examples but no evidence on how often the model actually uses shared vs. task-specific pathways.
- Exploration of stronger backbones (ConvNeXt, Swin Transformer) mentioned as options but never used.
- Analysis of when and why dynamic routing fails, providing balanced evaluation alongside the claimed successes.

## Novel Insights
The paper's most interesting finding is the asymmetric effect of dynamic gating: it degrades mAP but improves OF1/CF1. This suggests dynamic gating may be better at optimizing for balanced class performance (where each class is weighted equally) rather than average precision (which weights positive examples more heavily). The entropy regularization combined with per-label specialization could be biasing the model toward better handling of rare/difficult labels at the cost of top-prediction accuracy. This trade-off is itself a meaningful contribution but is buried and unexplained in the current presentation.

## Potentially Missed Related Work
- Routing Transformers (Roussel et al., 2021) — dynamic token routing in transformers for efficient attention.
- Hard Mixture of Experts for Large-Scale Multi-label Classification (Dai et al., 2022) — applies MoE specifically to multi-label classification with different routing strategies.
- Sparse MoE foundations (Switch Transformer, ST-MoE) — may offer more efficient routing mechanisms suitable for the MLC setting.

## Suggestions
The paper must honestly acknowledge the MS-COCO mAP regression and provide a thorough analysis of *why* dynamic gating hurts mAP while helping OF1/CF1. Consider adding an auxiliary loss that preserves mAP, adjusting the entropy regularization strength, or investigating whether the label-specific gating input (hi) should be augmented with global image features to capture broader label correlations. Additionally, provide explicit computational comparisons (FLOPs, parameters, latency) to validate the efficiency claims and help readers weigh the trade-offs.

---

## xVw8YNEtH3

- GT: Reject (avg 3.0)
- Predicted: Reject (2.0/10)
- Match: Yes

### Final Review

## Summary
The paper proposes a "Reset method" that combines existing deep learning optimizers (SGD, Adam, AdamW) with manifold optimization theory, applying various Riemannian manifold structures as step-size correction mechanisms to improve convergence and stability in deep learning tasks. However, the core contribution is poorly defined—the exact mathematical form of the "Reset" mechanism is unclear, the theoretical results contain error bounds that grow with iteration count (contradicting convergence claims), and the experiments use non-standard metrics with results that are implausibly uniform across all manifold variants, raising concerns about validity.

## Strengths
- The paper attempts to bridge manifold optimization theory with practical deep learning training, addressing a potentially valuable research direction
- The experimental evaluation covers diverse datasets (CIFAR-10/100, STL-10, SVHN, Market-1501, DukeMTMC-reID) and multiple architectures (DCGAN, StableGAN, Cluster Contrast), providing broad empirical coverage
- The theoretical framework involving Riemannian retraction, vector transport, and Armijo line search on manifolds is technically grounded in established manifold optimization literature (Hu et al., 2020)

## Weaknesses
- **The Reset mechanism is mathematically undefined**: Equation (5) states $x_{i+1} = B_{x_i}(x_i)$ where $B_{x_i}$ is "selected from SGD, Adam, AdamW," but no actual update rule is provided. Without knowing what transformation $B_{x_i}$ applies, the method cannot be reproduced or theoretically analyzed. This is a fundamental flaw.
- **Theorem 4.2 error bounds grow with iterations**: The bounds contain $(i+1)^2$ terms in the numerator, meaning the error $E[\text{grad}f(x_{i+1})] - E[\text{grad}f(x_i)]$ grows with iteration count. This directly contradicts the paper's claim of improved convergence and suggests the theorem does not demonstrate what the authors claim.
- **Non-standard evaluation metric for image generation**: The paper reports "Average Precision (AP)" for GAN tasks, but standard metrics are Inception Score (IS) and Fréchet Inception Distance (FID). The reported values (~98-99%) are implausibly high for these tasks and inconsistent with known benchmarks, raising concerns about what is actually being measured.
- **No comparison to Riemannian optimizers**: The paper claims to improve "manifold optimization" but only compares against Euclidean-space baselines (SGD, Adam, AdamW). No comparison to established Riemannian optimization methods (Riemannian SGD, RAdam on manifolds, methods from Absil et al., Huang et al.) is provided.
- **Implausibly uniform results across manifold variants**: All 10+ manifold combinations yield nearly identical performance (~99.xx%), suggesting the manifold choice is irrelevant or results lack meaningful variation. No variance statistics or significance testing is provided.

## Nice-to-Haves
- Add convergence curves (loss/gradient norm vs. iteration) to empirically verify theoretical convergence claims
- Include ablation studies isolating the Reset mechanism's effect from other factors
- Provide code and full hyperparameters for reproducibility
- Compare against Riemannian Adam or RiemSGD specifically designed for manifold optimization

## Novel Insights
The paper's observation that restart techniques from O'Donoghue & Candes (2015) could be adapted to Riemannian manifold optimization is potentially interesting, and the dynamical systems interpretation of the Reset as inducing "damped harmonic motion" toward saddle points is a reasonable intuition. However, these insights remain underdeveloped because the actual Reset mechanism is undefined and the theoretical results do not convincingly demonstrate improvement over existing methods.

## Potentially Missed Related Work
- Riemannian Adam (RAdam) variants for manifold optimization — directly relevant as a comparison baseline for claimed improvements over manifold optimization
- Zhang et al. (2016) Riemannian SVRG — cited in related work but not compared experimentally despite being a Riemannian optimization method

## Suggestions
The paper should first provide a concrete mathematical definition of the Reset operator $B_{x_i}$ beyond "using SGD/Adam/AdamW." Without this, neither the theory nor the experiments are interpretable. Second, the error bound analysis in Theorem 4.2 must be reconsidered—either the bounds should demonstrate convergence (decreasing with iterations) or the paper should clearly state what they do prove. Third, experiments should use standard evaluation metrics (FID/IS for GANs) and include Riemannian optimizer baselines for meaningful comparison.

---

## BPgK5XW1Nb

- GT: Accept (Oral) (avg 8.7)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
The paper proposes SPA (Spread Preference Annotation), a framework for efficiently aligning LLMs using only a small amount of seed preference data (3.3% of UltraFeedback). SPA iteratively expands preference data using the LLM's own logits for direct preference judgment and applies a noise-aware refinement mechanism with de-coupled noise detection. The method achieves 21.13% win rate on AlpacaEval 2.0 (vs 7.68% for standard DPO with same data), outperforming even Zephyr-7b-β trained on 100% of the data.

## Strengths
- **Compelling empirical results**: SPA achieves substantial improvements using minimal data (7.68% → 21.13% win rate with 3.3% of labels), and even exceeds Zephyr-7b-β trained on 30x more data (10.03% → 21.13%)
- **Strong generalization**: Experiments across multiple model families (Phi-2, Mistral, LLaMA-3, Phi-3) spanning 2.7B to 14B parameters, with consistent improvements
- **Thoughtful ablation**: Table 6 systematically isolates contributions of data expansion, self-refinement, and de-coupled noise detection, demonstrating that the noise-aware components provide meaningful gains
- **Practical methodology**: The approach requires only minor code additions to standard DPO, runs efficiently (5-6 hours per iteration on 4 A6000 GPUs), and is compatible with existing length control techniques
- **Comprehensive evaluation**: Uses both standard and length-controlled win rates on AlpacaEval 2.0, plus MT-Bench, to mitigate evaluation biases

## Weaknesses
- **Unfair comparison with PairRM**: PairRM was trained on ~500K samples while SPA uses 2K samples (Appendix B.2). The main text claims superiority over "external reward models" without adequately clarifying this data scale difference. The comparison demonstrates SPA works with less data, not that direct judgment is superior to a well-trained reward model with equivalent data
- **Missing comparisons with concurrent self-training methods**: Self-Rewarding LM (Yuan et al., 2024), Direct Nash Optimization (Rosset et al., 2024), and SELF-PLAY (Wu et al., 2024) are mentioned in related work but not experimentally compared. These are the most directly relevant baselines for assessing incremental contribution
- **Limited theoretical validation for de-coupled noise detection**: The method approximates a "strongly aligned model" via linear logit extrapolation (Eq. 12) based on Liu et al. (2024), but this geometric mixture assumption is neither validated empirically for DPO-trained models nor is the effectiveness of the approximation tested
- **Marginal self-refinement contribution**: Table 6 shows SR alone provides only +0.29% LC win rate improvement (14.41% → 14.7%), suggesting most gains come from data expansion rather than the noise-aware mechanism. The paper does not adequately explain this limited contribution
- **No sensitivity analysis for hyperparameters**: K=10% noise threshold and λ schedule (1/2, 1/4, 1/8) appear heuristic without ablation, raising questions about robustness
- **Acknowledged length bias**: Response length increases to 2749 chars (Table 10). While length-controlled metrics address this, the tendency to generate verbose responses limits applicability where brevity is preferred

## Nice-to-Haves
- Validate the geometric mixture assumption empirically (e.g., compare preference predictions between approximated model and actual more-aligned model)
- Compare against Self-Rewarding LM and other iterative preference learning methods
- Ablate the K% threshold and λ hyperparameters to justify choices
- Add human evaluation study to validate that improvements reflect genuine human preferences rather than GPT-4 biases
- Test on domain-specific tasks (medical, legal, coding) beyond general instruction following

## Novel Insights
The paper introduces the insight that an LLM's intrinsic reward function (derived from log-ratios with a reference model) can be directly used for self-generated preference annotation, eliminating dependence on external reward models or large well-aligned LLMs for in-context judgment. The key observation is that as the model iteratively improves, external reward models suffer from distribution shift while the self-derived reward remains aligned with the training distribution. Additionally, the de-coupled noise detection via linear logit extrapolation (approximating a strongly aligned model without additional computation) provides a principled approach to filtering noisy self-generated labels, though its empirical contribution is modest compared to the base iterative DPO framework.

## Potentially Missed Related Work
- **Self-Rewarding Language Models (Yuan et al., 2024)**: Directly relevant concurrent work with nearly identical motivation—self-generating preference data for iterative LLM alignment
- **Direct Nash Optimization (Rosset et al., 2024)**: Concurrent iterative alignment approach that could serve as an additional baseline
- **SELF-PLAY Preference Optimization (Wu et al., 2024)**: Another recent self-training approach for preference alignment

## Suggestions
- **Reframe PairRM comparison**: Clearly state that PairRM was trained on 250x more data and add a comparison with PairRM on equivalent data scale to isolate method contributions
- **Add Self-Rewarding LM baseline**: This is the most directly comparable prior work and would clarify the incremental contribution of direct logit-based judgment over LLM-as-judge style self-rewarding

---

## 2iPvFbjVc3

- GT: Reject (avg 3.4)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
VisCE² proposes a reference-free image caption evaluation method that first extracts structured visual context (objects, attributes, relationships) from images using a VLM, then uses this context to evaluate candidate captions. The method achieves state-of-the-art correlation with human judgments among reference-free, tune-free metrics across four datasets, demonstrating that explicit visual context helps VLMs evaluate caption accuracy more effectively than vanilla prompting or embedding-based methods like CLIP-S.

## Strengths
- **Well-motivated and clearly articulated problem**: The paper correctly identifies that modern VLMs generate captions exceeding reference coverage, making reference-based and embedding-based metrics inadequate. The failure case analysis of CLIP-S (overestimating captions with salient objects regardless of accuracy) is compelling and well-supported by examples in Figure 4.
- **Comprehensive experimental design**: Meta-evaluation across four diverse datasets (THumB, Flickr8k-Expert, Composite, Pascal-50S) using multiple correlation metrics (Pearson, Kendall, accuracy) against 10+ baselines provides strong empirical support.
- **Thorough ablation studies**: Systematically validating each component (visual context vs. vanilla prompting, individual context components, backbone VLM selection, extraction vs. evaluation model pairing) demonstrates scientific rigor and provides actionable insights.
- **Reproducibility**: Using LLaVA-v1.5-13B with default hyperparameters and providing full prompts in the appendix enables reproducibility, which is commendable.
- **Informative qualitative analysis**: The heatmaps in Figure 3 revealing score distribution changes and the concrete failure case comparisons in Figure 4 effectively illustrate where VisCE² succeeds and where CLIP-S fails.

## Weaknesses
- **Near-zero recall correlation**: The method shows ρ=0.08 correlation with recall scores on THumB (essentially unchanged from vanilla prompting at 0.08), indicating it only evaluates caption precision, not coverage of salient image content. This fundamental limitation is acknowledged but inadequately addressed—modifications to weight saliency or expand coverage evaluation are absent.
- **Single-run reporting without variance estimates**: Results are reported from "a single run" with no confidence intervals or standard deviations. Given VLM stochasticity, effect sizes like +0.10 on Pearson correlation (0.44→0.54) cannot be properly evaluated for statistical significance.
- **Performance regression on certain categories**: On Pascal-50S, VisCE² shows no improvement over vanilla prompting (80.5→80.8) and underperforms in HM/MM categories compared to other metrics. On Composite, InfoMetIC outperforms VisCE², suggesting fine-tuned methods retain advantages in some scenarios.
- **Compute cost not addressed**: The method requires two VLM inference passes per evaluation (context extraction + evaluation), significantly more expensive than CLIP-S. No cost analysis is provided, limiting practical adoption assessment.

## Nice-to-Haves
- **Statistical significance testing**: Run experiments with multiple seeds/temperature settings to establish variance and confidence intervals.
- **Prompt sensitivity analysis**: The entire method is a prompting strategy—testing alternative prompt formulations for VLMExt and VLMEval would strengthen claims about visual context being essential versus any structured injection helping equally.
- **Calibration analysis**: Are 0-100 scores calibrated to human ratings? Without this, absolute scores have unclear meaning.
- **Fine-tuning comparison**: Testing whether fine-tuning VLMEval on caption evaluation data (as InfoMetIC does) would close the gap with that method would demonstrate true potential.

## Novel Insights
The paper's key insight—that extracting structured visual context (objects, attributes, relationships) as an intermediate representation bridges the image-text gap for caption evaluation—is both intuitive and empirically validated. The ablation showing that the same VLM for extraction and evaluation outperforms mixed-model setups suggests model-specific alignment in how visual contexts are framed. Furthermore, the observation that visual context shifts score distributions from bimodal (good/bad peaks) to trimodal (good/fair/bad) indicates more nuanced evaluation aligned with human judgment. The method's effectiveness with publicly available models (LLaVA) competing with proprietary LLM-based approaches (CLAIRClaude) is noteworthy for practical accessibility.

## Potentially Missed Related Work
- **LLM-as-judge baselines** (e.g., GPT-4V direct evaluation): The paper compares against CLIP-S, PAC-S, and ensemble LLM methods (CLAIR) but does not directly compare against zero-shot GPT-4V or Gemini evaluation using naive prompting. This is the most relevant comparison the community would expect for VLM-based caption evaluation.

## Suggestions
1. Address the recall limitation by adding a secondary pass or modifying prompts to explicitly weight content coverage alongside accuracy.
2. Report variance across multiple runs to enable proper statistical evaluation of improvement significance.
3. Conduct and report a cost-per-evaluation analysis comparing VisCE² against CLIP-S and InfoMetIC to inform practical deployment decisions.

---

## cLws58ZojF

- GT: Reject (avg 3.0)
- Predicted: Accept (6.0/10)
- Match: No

### Final Review

## Summary
This paper systematically explores the design space of Speech-Conditioned Large Language Models (SLMs) through rigorous experiments examining adaptor architectures, trainable modules, masking strategies, LLM choices, and training data. The authors identify a critical gap in existing SLMs—lack of spoken instruction following capability due to text-only instructions in training—and address it by creating a synthetic spoken instruction following dataset (50K samples). Their resulting model SiM significantly outperforms existing SLMs on spoken instruction following tasks while maintaining competitive ASR performance.

## Strengths
- **Systematic empirical methodology**: The paper conducts comprehensive experiments across multiple design dimensions under consistent settings, providing valuable actionable guidance for the community on adaptor choices (finding MLP adaptor outperforms complex architectures), LLM selection (instruction-tuned helps for SLMs contrary to VLM findings), and training data composition.
- **Important discovery with clear evidence**: The finding that existing SLMs lack spoken instruction following capability—even for simple queries like "can I go to the moon?"—is a compelling and well-supported insight. The paper demonstrates this with concrete examples (Figure 4) showing existing models failing to process audio instructions.
- **Comprehensive evaluation suite**: Evaluation across multiple datasets (Alpaca Audio, OpenHermes Audio, LLaMA Questions, LibriSpeech) using both automatic metrics and human preference studies provides strong evidence. SiM achieves 2.71 vs 1.14 for the best existing model on OpenHermes Audio, and 57.7% vs 40.7% accuracy on LLaMA Questions.
- **Practical contribution**: The 50K synthetic spoken instruction dataset and the resulting SiM model provide a directly usable recipe for building SLMs capable of following spoken instructions.

## Weaknesses
- **Missing ablation for spoken instruction data**: The paper claims 50K synthetic spoken instruction samples are "crucial" but provides no ablation isolating their contribution. Performance differences could partially stem from data mixture variations or other training differences, not solely from spoken instruction data.
- **ASR quality gap unexplained**: SiM achieves 3.1% WER while the underlying Whisper encoder achieves 1.3% WER—a 2.4× regression. The paper mentions this but never explains the mechanism or investigates whether this is an inherent cost of the architecture. This undermines the claim of "strong ASR performance."
- **Missing Qwen2 Audio instruction-following scores**: Table 6 omits Qwen2 Audio from instruction-following comparisons despite including it in human preference evaluations (Figure 2) and ASR results (Table 8). Since Qwen2 achieves the best ASR WER (1.8%), its instruction-following capability is essential context for evaluating SiM's claimed superiority.
- **Underspecified evaluation rigor**: Human preference evaluations in Figure 2 show percentages but lack confidence intervals, sample sizes, or statistical significance testing. A margin of 81% vs 84% "wins" could be within noise.
- **Incomplete reproducibility details**: Training hyperparameters (learning rates, batch sizes, warmup schedules, training steps) are entirely absent from the paper. Section references like "Section TBD" in Table 5 indicate incomplete manuscript preparation.

## Nice-to-Haves
- Analyze why instruction-tuned LLMs help SLMs but not VLMs (per Karamcheti et al., 2024)—the discrepancy deserves explanation.
- Study scaling dynamics with varying synthetic dataset sizes (5K, 10K, 25K, 100K) to understand data efficiency.
- Evaluate on noisy speech, accented speech, or diverse recording conditions beyond LibriSpeech test-clean.
- Include attention visualizations to validate that spoken instructions are actually being processed by the LLM.

## Novel Insights
This paper makes a valuable empirical contribution by systematically mapping the SLM design space under consistent settings, a task made difficult by heterogeneous experimental conditions across prior work. The finding that simple 2-layer MLP adaptors outperform complex architectures like Q-Former (which suffers >20% WER and training instability) is counterintuitive and provides practical guidance. The critical insight about the absence of spoken instructions in existing SLM training data explaining their poor instruction-following capability is well-motivated and demonstrated with concrete failures. The discovery that instruction-tuned LLMs help SLMs (contrary to VLM findings) suggests modality-specific dynamics worth investigating. The trade-off analysis between ASR quality (3.1% WER) and instruction-following capability (2.71 score) represents a practical design decision the community must consider.

## Potentially Missed Related Work
- Spectron (Nachmani et al., 2023) — appears in references and Table 7 comparison but could be discussed more prominently given its similar spoken instruction following approach.
- WavLLM — included in some evaluations but its methodology for handling spoken instructions is not compared or contrasted.

## Suggestions
The paper should add a proper ablation study isolating the contribution of spoken instruction data versus other training differences, and provide at minimum learning rate and training step counts for reproducibility. The ASR quality regression versus Whisper deserves either explanation (e.g., through analysis of where errors occur) or explicit acknowledgment as a limitation of the approach.

---

## mb2ryuZ3wz

- GT: Accept (Poster) (avg 5.8)
- Predicted: Accept (6.0/10)
- Match: Yes

### Final Review

## Summary
ALIT (Adaptive Length Image Tokenizer) proposes a recurrent framework that distills 2D image tokens into variable-length 1D latent tokens through iterative rollouts with adaptive memory expansion. Each iteration refines existing tokens while adding new ones (32→256), enabling image-specific representational capacity allocation. The method demonstrates comparable reconstruction metrics to fixed-length baselines (VQGAN, Titok), reveals emergent token specialization for object/part discovery, and shows that required token count aligns with image complexity, OOD familiarity, and downstream task requirements.

## Strengths
- **Strong conceptual motivation**: The idea of variable-length representations tied to image complexity is well-grounded in information theory (Schmidhuber's Low-Complexity Art) and draws compelling analogies to human intelligence and LLM adaptive computation.
- **Extensive and thorough experimentation**: The paper evaluates across multiple datasets (ImageNet-100/1K, COCO, Places, WIT) with diverse baselines (VQGAN, Titok, Matryoshka, FlexViT) across reconstruction FID/L1, linear probing, and task-dependent token analysis.
- **Comprehensive ablation study**: Systematic ablation of model size, training duration, dataset scale, continuous vs discrete tokenization, dynamic halting, and codebook design provides thorough understanding of each component's contribution.
- **Novel emergent property**: The emergent object/part discovery without explicit segmentation supervision (reaching 57.8 mIOU on ImageNet-S attention maps) distinguishes ALIT from patch-bound tokenizers and suggests meaningful semantic specialization.
- **Valuable analytical insights**: The analysis connecting representation capacity to image entropy, OOD familiarity, and downstream task alignment provides practical guidance for adaptive computation in vision.

## Weaknesses
- **Performance gap at maximum token count**: The ~1 FID point gap between ALIT-S (8.02-8.25 at 256 tokens) and VQGAN (7.04) is non-trivial and should be discussed more candidly as a trade-off for flexibility.
- **Not truly end-to-end**: The approach relies on a pre-trained frozen VQGAN for 2D tokenization, making it an additional processing layer rather than a native tokenizer. Joint end-to-end training is left as future work.
- **Dynamic halting utility unclear**: Marked as optional and not used in main Stage 2 experiments, leaving its practical impact during training and potential test-time benefits undemonstrated.
- **Limited downstream application demonstration**: Beyond reconstruction metrics and linear probing, concrete downstream applications (e.g., video tokenization, generative modeling speedup) are not thoroughly explored.

## Nice-to-Haves
- **ElasticTok comparison**: This concurrent work on "adaptive tokenization for images and video" is the most directly related approach and its absence leaves the contribution's distinctness partially unestablished.
- **Single-pass recurrence ablation**: A proper comparison between recurrent processing (N/8 iterations with 32 tokens each) versus a single forward pass with N tokens at equivalent compute budget would isolate whether recurrence provides benefits beyond having more total tokens.
- **Codebook utilization quantification**: While Figure 22 shows codebook sampling distributions, quantifying what fraction of codes are used infrequently (<100 times) would strengthen the OOD detection claims.
- **Attention map comparison with non-recurrent model**: Side-by-side token attention visualization between ALIT and a Matryoshka-style single-pass model would directly test whether recurrence is necessary for the observed object binding.

## Novel Insights
The paper's core insight is that recurrent token refinement with adaptive memory expansion naturally leads to token specialization aligned with semantic content. Unlike Matryoshka-style approaches that learn fixed-length representations in one pass (where the first 32 tokens always remain the same regardless of token budget), ALIT allows existing tokens to progressively sharpen their attention and specialize as new tokens provide additional representational capacity. This is supported by the mIOU improvement across recurrent iterations (Table 6) and the qualitative emergence of object binding (Figures 7, 8, 15, 16). The analysis that ~60% of maximum dataset tokens suffice for near-optimal performance across multiple downstream tasks suggests reconstruction loss may serve as a practical self-supervised proxy for determining per-image token requirements.

## Potentially Missed Related Work
- **ElasticTok (Yan et al., 2024)**: Concurrent work on "Adaptive tokenization for image and video" is the most directly comparable approach—mentioned briefly but not included as a baseline comparison.

## Suggestions
- **Add ElasticTok to experiments**: This is the most directly comparable concurrent work and its absence is notable; even a qualitative comparison would strengthen the paper's distinctness claims.
- **Include a single-pass baseline ablation**: Compare recurrent processing (8 iterations × 32 tokens) vs. a single forward pass with 256 tokens at matched compute to isolate the benefit of recurrence beyond additional capacity.
- **Be more explicit about what "adaptive" means**: Clarify that adaptivity operates at two levels—(1) recurrent refinement allows tokens to specialize progressively, and (2) TSC enables flexible token selection at test time.

---

## koza5fePTs

- GT: Reject (avg 2.0)
- Predicted: Reject (4.0/10)
- Match: Yes

### Final Review

## Summary
This paper presents a comprehensive benchmark suite for evaluating large language models' planning capabilities, encompassing both classical PDDL domains (BlocksWorld, Logistics, Mini-Grid) and natural language tasks (Trip Planning, Calendar Scheduling). The authors systematically investigate three intervention strategies—many-shot in-context learning, supervised fine-tuning on optimal plans, and chain-of-thought reasoning methods (ToT, MCTS, Debate)—across multiple models and generalization scenarios. Key findings include that fine-tuning achieves near-perfect accuracy on in-distribution tasks, many-shot ICL benefits some models but not others, and search-based CoT methods allow smaller models to partially compete with larger ones.

## Strengths
- **Comprehensive benchmark design**: The paper provides a well-structured suite with both formal (PDDL) and natural language representations, including algorithmic generation of instances at configurable difficulty levels. The 28k BlocksWorld samples and scalable generation pipeline are valuable for rigorous evaluation.
- **Systematic methodology**: The authors evaluate three distinct intervention strategies across multiple benchmarks and models, with both in-domain and out-of-domain generalization experiments. The categorization of failure modes (constraint violations, goal failures, out-of-vocabulary actions) provides actionable diagnostic insights.
- **Significant empirical findings**: SFT dramatically outperforms ICL even with smaller models (Gemini 1.0 S achieving 96%+ vs. ~48% with 200-shot ICL). The observation that debate-as-reasoning performs comparably to explicit search methods (MCTS/ToT) is counterintuitive and insightful. The finding that many-shot ICL benefits some models (Gemini 1.5 Pro) but degrades others (GPT-4, Gemini 1.5 Flash) at higher shots is a valuable empirical observation.

## Weaknesses
- **SFT methodology underspecified**: The paper reports learning rate (0.0001), dropout (0.1), and steps (5k) but omits batch size, optimizer choice, weight decay, training batch details, and whether adapters/LoRA are used. This limits reproducibility.
- **OOD generalization gap underexplored**: Table 2 shows training on 3-7 blocks and evaluating on 8-20 blocks yields only 34.2% accuracy—a fundamental limitation that should be more prominent. The paper's "near-perfect accuracy" claim (abstract) is misleading as it only holds for easier instances (67% on 8-20 blocks).
- **CoT methods not evaluated on PDDL benchmarks**: Section 3.1.2 and Figure 4 only show ToT/MCTS/Debate results on Trip Planning and Calendar Scheduling. These methods are never tested on BlocksWorld, Logistics, or Mini-Grid despite PDDL being a core focus. This selective reporting leaves unclear whether search-based methods help for classical planning.
- **Incomparable SFT vs. ICL comparison**: The paper claims SFT "outperforms many-shot ICL" but compares SFT on Gemini 1.0 S against ICL on Gemini 1.5 Pro and GPT-4. The 48-percentage-point gap is confounded by model differences. A fair comparison requires same-model ablation.
- **Regex extraction for NL plan verification unvalidated**: The paper states they use "regular expressions to map the plan in Natural Language back to PDDL" but never reports extraction success rates. Errors in this pipeline directly contaminate accuracy measurements for natural language benchmarks.

## Nice-to-Haves
- Ablation studies for SFT components (optimal plans vs. any plans vs. number of instances) would clarify what drives success
- Statistical significance testing with 95% confidence intervals instead of 70% CI
- Systematic prompt sensitivity analysis beyond the anecdotal comparison in Section 3.4
- Additional open-source models (Llama-3, Mistral) for broader comparison
- Side-by-side concrete examples of successful vs. failed plans for the same problems

## Novel Insights
The paper surfaces several non-obvious empirical observations: the finding that many-shot ICL exhibits model-dependent scaling—with Gemini 1.5 Pro benefiting from additional shots while GPT-4 and Gemini 1.5 Flash degrade—suggests heterogeneous attention patterns or context length limits across model families. The competitive performance of debate-as-reasoning (not explicitly designed for planning) versus structured search methods (MCTS, ToT) indicates that for natural language planning, allowing models to construct and compare multiple solutions may matter more than the specific search strategy employed. The observation that failures concentrate at early plan steps in OOD settings, combined with the discovered correlation between step number and action choice, hints that models may learn positional biases rather than generalizable planning heuristics. The dramatic SFT superiority (96% vs. 48% for ICL) with a smaller model suggests that explicit training on optimal plans instills planning structure that in-context learning cannot replicate, even with extensive examples.

## Potentially Missed Related Work
- **Lehnert et al. (2024) "Beyond A*"** — Uses A* as search mechanism with transformer architecture for planning. The paper mentions this work but positions itself as teaching LLMs themselves rather than hybrid approaches; a brief comparison or ablation against such hybrid methods would strengthen the positioning.
- **LLM-modulo frameworks (Kambhampati et al., 2024)** — The paper briefly mentions hybrid approaches but doesn't compare against them as baselines. A comparison showing whether fine-tuned LLMs can match or exceed LLM-plus-planner systems would contextualize the contribution.

## Suggestions
1. **Validate regex extraction pipeline**: Report extraction success rates for NL benchmarks to ensure accuracy measurements are reliable.
2. **Test CoT methods on PDDL domains**: Evaluate ToT/MCTS/Debate on BlocksWorld, Logistics, and Mini-Grid to establish whether these methods help for classical planning or only for natural language tasks.
3. **Same-model comparison**: Run SFT and ICL with identical models (e.g., Gemini 1.0 S for both) on the same test problems to support the claim that fine-tuning methodology is superior to in-context learning.
4. **Prominently discuss OOD limitations**: The 34.2% accuracy on OOD instances is a fundamental constraint that should be central to the paper's narrative and limitations section, not relegated to a table.

---

## WCRQFlji2q

- GT: Accept (Oral) (avg 9.0)
- Predicted: Accept (7.0/10)
- Match: Yes

### Final Review

## Summary
This paper uses sparse autoencoders (SAEs) to discover linear directions in LLM representations that detect whether the model "knows" facts about specific entities. The authors demonstrate that these entity recognition directions fire selectively on known versus unknown entities across diverse entity types (movies, cities, players, songs), have causal effects on knowledge refusal behavior through steering experiments, and transfer from base models to chat models—suggesting instruction tuning repurposes existing mechanisms. They additionally identify "uncertainty directions" that can distinguish correct from incorrect model answers with moderate accuracy.

## Strengths
- **Strong empirical methodology with converging evidence**: The paper combines SAE-based feature discovery, activation patching, steering interventions, and weight orthogonalization baselines, providing multiple complementary approaches to support claims.
- **Multi-model validation**: Findings are replicated across Gemma 2 2B, Gemma 2 9B, and Llama 3.1 8B with consistent patterns, substantially strengthening confidence in the results.
- **Comprehensive control analyses**: The authors address the token likelihood hypothesis (Appendix S), showing correlations between latent activations and next-token prediction probability are negligible (r < 0.07), and validate on post-cutoff song data (Appendix R).
- **Significant mechanistic insight**: The discovery that base model SAE directions causally affect chat model refusal behavior provides concrete evidence for the "mechanism repurposing" hypothesis during instruction tuning.
- **Reproducibility**: Code is publicly available, entity datasets from Wikidata are described, and methodology is clearly specified.

## Weaknesses
- **Effect robustness varies across models**: The statistical significance tests in Appendix M show mixed results (Gemma 2 2B: 10/10 significant for known direction, 9/10 for unknown; Gemma 2 9B: 10/10 for known, only 1/10 for unknown). The paper acknowledges this but doesn't deeply analyze why the 9B model's unknown direction shows weaker effects, limiting the robustness claims.
- **Large steering coefficients raise off-target effect concerns**: Steering coefficients α ∈ [400, 550] are applied to residual streams with norm ~200-400 (Appendix E), meaning steering adds a vector larger than the original representation. This raises concerns about representational degeneration or unintended behavioral changes beyond the targeted entity recognition effect.
- **Incomplete mechanistic pathway**: While Section 6 shows entity recognition directions affect attention patterns to entity tokens, the paper does not fully trace the circuit from entity recognition to refusal generation. The causal chain from detection → attention modulation → refusal decision remains partially opaque.
- **"Self-knowledge" framing may overclaim**: The paper states models "might have self-knowledge: internal representations about their own capabilities," but findings only demonstrate entity-level detection for factual recall, not broader metacognitive awareness. The terminology in title and abstract suggests broader claims than the data supports.
- **Section 7 (Uncertainty directions) feels preliminary**: The 73.2% AUROC for distinguishing correct from incorrect answers is moderate, and the analysis is relatively brief compared to the main entity recognition findings, creating a disconnected subplot.

## Nice-to-Haves
- **Dose-response curves**: Test how refusal/hallucination rates change as a function of steering magnitude to verify effects are not thresholded or nonlinear.
- **Failure mode analysis**: Show examples where steering does NOT produce expected behavior to clarify the boundaries of generalizability.
- **Ablation study**: Remove the identified directions (not just orthogonalize) to verify the model's known/unknown discrimination ability degrades, strengthening causal claims.
- **Prospective uncertainty prediction**: Test whether Section 7's "unknown" latent can predict hallucinations on held-out questions in new contexts, not just re-predict the test set.

## Novel Insights
The paper's most valuable insight is demonstrating that SAE-discovered directions from the base model causally influence knowledge refusal in the chat model—suggesting that instruction tuning repurposes pre-existing mechanisms rather than creating entirely new ones. This finding has implications for both understanding how RLHF/instruction tuning works mechanistically and for potential intervention strategies. The paper also provides evidence that "knowledge awareness" at the entity level is encoded as interpretable linear directions discoverable through SAEs, with the hierarchical organization showing more generalized features emerging in middle layers. The discovery of uncertainty directions that distinguish correct from incorrect answers, while preliminary, suggests a potential pathway for internal hallucination detection.

## Potentially Missed Related Work
- **Yona et al. (2024)** on LLM uncertainty expression is cited, but the paper could more deeply engage with work on probing for model uncertainty and truthfulness (e.g., "Can LLMs Express Their Uncertainty in Words?" for framing the model's difficulty in expressing when it doesn't know).
- **Marks & Tegmark (2023)** on "summarization motif" is mentioned regarding end-of-instruction tokens, but the paper could more thoroughly connect to this work on how later tokens aggregate information from earlier context.

## Suggestions
- **Strengthen the known/unknown classification validation**: The current approach classifies entities based on whether the model gets attributes correct, which could conflate actual knowledge with lucky guesses. Consider using a second independent attribute set to verify the classification is stable rather than artifact-prone.
- **Analyze why model size affects results**: Investigate whether the weaker effects in 9B models stem from different superposition patterns, more distributed representations, or different knowledge localization—this would deepen mechanistic understanding.
- **Clarify steering coefficient selection**: The choice of α ∈ [400, 550] ("around two times the norm") warrants more justification. Show that smaller coefficients have no effect before claiming the directions are causally relevant at these magnitudes.

---

## e92KW6htFO

- GT: Reject (avg 5.0)
- Predicted: Reject (5.0/10)
- Match: Yes

### Final Review

## Summary

This paper proposes MICE (Memory-driven Intrinsic Cost Estimation) to address constraint violations in Constrained Reinforcement Learning caused by underestimation of cost value functions. The method stores unsafe trajectories in a "flashbulb memory" and generates intrinsic costs based on trajectory similarity to unsafe memories, augmenting the cost value function to correct underestimation bias. The paper provides theoretical bounds on constraint violation and convergence guarantees, with experiments on Safety Gym and Safety MuJoCo demonstrating reduced constraint violations.

## Strengths

- **Problem Identification**: The paper correctly identifies underestimation bias in cost value functions as a distinct issue from the well-known overestimation problem in RL, formalizing why minimizing noisy estimates leads to systematic underestimation. This is a valuable contribution to the CRL literature.

- **Novel Cognitive-Inspired Mechanism**: The flashbulb memory analogy, while loosely connected to actual cognitive science, provides an intuitive framing for a memory-augmented approach. The trajectory similarity-based intrinsic cost offers a principled way to enhance cost estimates for states near known dangers.

- **Theoretical Analysis**: The paper provides meaningful theoretical contributions including convergence guarantees for the extrinsic-intrinsic cost value function (Theorem 3) and a worst-case constraint violation bound (Theorem 2) that is tighter than CPO's bound by an intrinsic term I.

- **Comprehensive Empirical Evaluation**: Extensive experiments across 7 environments with multiple baselines (CPO, PIDLag, Saute, SimmerPID, and additional comparisons in appendix including CUP, IPO, P3O, WCSAC) and thorough ablation studies demonstrating the method's effectiveness.

## Weaknesses

- **Incomplete Convergence Justification**: The convergence proof (Theorem 3) relies on the claim that intrinsic cost c^I "converges to 0 with probability 1" because it decays as γI^k. However, this argument ignores that the memory module is continuously updated with new trajectories as training progresses. The c^I at iteration k depends on memory at iteration k, which itself depends on the evolving policy—creating a moving target. The paper states this without rigorous justification.

- **Theoretical-Practical Gap**: The convergence proof requires conditions (finite MDP, lookup table storage) that fundamentally do not hold in the experimental setup using neural network function approximation. This gap is mentioned but deserves more prominent acknowledgment.

- **Underspecified Generator Architecture**: The intrinsic generator G_φ uses a "random projection layer" but architectural details are sparse. The loss function in Equation 5 regresses the generator toward the same c^I defined in Equation 3, making it unclear what the generator learns differently from directly computing the similarity-based cost. This raises reproducibility concerns.

- **Missing Statistical Significance**: The paper reports mean ± std across 6 seeds but provides no statistical significance tests (confidence intervals, t-tests) to support claims of "substantial reduction" in constraint violations.

## Nice-to-Haves

- **Ablation on memory mechanism itself**: Compare random trajectory storage vs. unsafe-only storage, and memory capacity = 0 (no memory) to isolate contribution of the memory mechanism specifically.
- **Runtime/computational overhead analysis**: The intrinsic generator and memory module add complexity; wall-clock time or FLOP comparison with baselines would help practical deployment assessment.
- **Visualization of intrinsic costs**: Heatmaps of c^I values across state-space regions and how they evolve as memory populates would make the mechanism more tangible.
- **TD3-style baseline for underestimation**: Comparing against using max instead of min for cost value updates would better isolate the contribution of the memory mechanism.

## Novel Insights

The paper offers a genuinely new perspective by identifying that underestimation bias in cost value functions (caused by minimizing noisy estimates) is a key driver of constraint violations in CRL, in direct analogy to how overestimation causes problems in standard RL. The memory-driven approach to enhancing cost estimates for unsafe trajectories, framed through the flashbulb memory metaphor, provides a practical mechanism for corrective signaling. The insight that intrinsic costs can correct underestimation by boosting estimates toward true values—where overestimation is preferable to underestimation in safety-critical applications—is both intuitive and well-motivated.

## Potentially Missed Related Work

- **TD3-based cost estimation**: The paper shows TD3 mitigates but cannot fully eliminate underestimation in Appendix C.2.4, but this work should be discussed in the main text as a natural baseline comparison.
- **Episodic Reinforcement Learning with Associative Memory (Zhu et al., 2020)**: Directly cited but the paper doesn't clearly distinguish MICE's flashbulb memory from this prior episodic memory approach.
- **Intrinsic Fear Model (Lipton et al., 2016)**: Directly cited; the paper claims MICE addresses catastrophic forgetting like this approach but the decay mechanism in MICE (γI^k) seems inconsistent with preventing late-training revisit of dangers.

## Suggestions

- Provide either a rigorous proof or explicit discussion of sufficient conditions for convergence of the joint (Q-value, memory, policy) system, not just assuming c^I → 0.
- Add statistical significance tests to support quantitative claims about constraint violation reduction.
- Include ablation removing the intrinsic generator entirely (computing similarity directly) to validate its necessity beyond computational efficiency.
- More prominently acknowledge the gap between theoretical assumptions (finite MDP, lookup tables) and practical implementation (neural networks).

---

## WRKVA3TgSv

- GT: Reject (avg 3.0)
- Predicted: Accept (7.0/10)
- Match: No

### Final Review

## Summary
This paper introduces GraphModQA, a novel benchmark for evaluating LLMs on dynamic graph modification tasks, and demonstrates that while modern SOTA LLMs have largely solved static graph property tasks, their performance degrades significantly on sequential node/edge modifications—particularly with adjacency matrix encodings. The authors propose Modify-and-Print (MAP) prompting, which instructs models to output intermediate adjacency matrices at each step, yielding substantial performance improvements. The work reveals a critical gap in current LLM capabilities for dynamic graph reasoning.

## Strengths
- **Comprehensive benchmark design**: The 468,750-example GraphModQA dataset systematically varies across five dimensions (3 encodings, 5 modification types, 1-5 modifications, 5 question types, 5 prompting methods), enabling granular analysis of LLM capabilities across conditions.
- **Thorough empirical evaluation**: Four SOTA LLMs (GPT-4o mini, Llama 3.1 405B, Claude 3.5 Sonnet, o1-mini) are evaluated with detailed error analysis categorizing failure modes (e.g., adjacent index confusion, dimension errors), providing actionable diagnostic insights.
- **Introduction of adjacency matrix encoding**: While prior work focused on natural-language-like encodings, this paper demonstrates that the adjacency matrix—a fundamental computational representation—is substantially harder for LLMs, which is novel and practically important for real-world graph algorithm applications.
- **Practical prompting contribution**: MAP prompting is a simple yet effective intervention that consistently improves performance, especially on edge modifications. The observation that MAP outperforms zero-shot even at k=1 (despite there being only one intermediate state) is a valuable empirical finding.
- **Extensive ablation studies**: Analysis of graph type effects (star, path, complete, empty), edge density impacts, and graph size variations provides nuanced understanding of failure modes.

## Weaknesses
- **Limited graph diversity**: All experiments use Erdős–Rényi random graphs. Real-world graphs exhibit structure (power-law distributions, community structure) that could affect LLM performance differently. This limits the generalizability of claims about "critical gaps in LLM capabilities."
- **Missing comparative baselines**: The paper does not compare LLMs against algorithmic baselines (e.g., simple Python code executing modifications) or specialized approaches. Without this, "LLMs struggle" is uninterpretable—are they worse than trivial code?
- **MAP mechanism understanding is limited**: While the paper observes MAP improves performance, it does not deeply investigate *why* it works. The k=1 finding (where MAP should equal zero-shot but outperforms it) suggests an attention-rerouting effect that deserves analysis.
- **Main text focuses almost exclusively on Print Graph**: Other final question types (Node Count, Edge Count, Connected Nodes after modifications) receive minimal analysis, making it unclear whether property-based questions after modifications are equally challenging.

## Nice-to-Haves
- Statistical significance testing for performance differences between prompting methods
- Fine-tuning experiments on GraphModQA to separate capability limitations from prompting limitations
- Extension to directed graphs and weighted graphs, which appear in real-world applications
- Analysis of whether MAP helps on non-matrix representations (incident lists, edge lists)

## Novel Insights
The paper offers several genuinely novel observations beyond the expected contribution of a new benchmark. The discovery that modern SOTA LLMs have largely "solved" static graph property tasks—where PaLM 2 L achieved only 36.5% average accuracy on adjacency matrices while o1-mini achieves 92.6%—is a significant empirical finding that updates community assumptions about LLM graph reasoning. The finding that adjacency matrices are substantially harder than natural-language-like encodings (Incident, Coauthorship) for modification tasks, but comparable for static tasks, suggests a specific failure mode in how LLMs handle dense numerical representations with implicit numbering schemes. The o1-mini structure preservation bias—where the model systematically attempts to preserve graph structure when adding nodes rather than treating them as isolated—is a fascinating emergent behavior that warrants deeper investigation. The MAP prompting insight that explicitly requesting intermediate outputs improves performance even when it shouldn't theoretically matter (k=1 case) suggests that prompting for structural compliance positively influences model attention to the entire task.

## Potentially Missed Related Work
- Perozzi et al. (2024) on soft-prompt tuning for graph encoding (suggests fine-tuning could improve results beyond frozen model performance)
- Zhang et al. (2023) on spatio-temporal graph property tasks with LLMs (related dynamic graph evaluation but different focus)
- None identified that represent major omissions

## Suggestions
The ablation study in Appendix A.7 is limited to Claude 3.5 Sonnet only; extending error analysis to other models would strengthen understanding of model-specific versus task-specific failure modes. Additionally, the paper would benefit from a comparison against an algorithmic baseline (e.g., simple Python script executing modifications) to contextualize whether the observed limitations are specific to LLM architectures or inherent to the task complexity.

---

