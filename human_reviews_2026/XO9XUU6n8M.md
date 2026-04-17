# Skip-It? Theoretical Conditions for Layer Skipping in Vision–Language Models

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Vision–language models (VLMs) achieve incredible performance across a wide range of tasks, but their large size makes inference costly. Recent work shows that selectively skipping VLM layers can improve efficiency with minimal performance loss or even performance improvements. However, this technique remains underused due to the limited understanding of when layer skipping is beneficial. In this paper, we develop a framework that uses information and learning theory to characterize the conditions under which layer skipping enhances efficiency without sacrificing performance. Motivated by these observations, we analyze the evolution of the VLM's hidden representations through the LLM backbone and show that layers with large redundancy as predicted by our framework coincide with those skipped by popular layer-skipping methods in practice, providing a unified theoretical scaffolding for multiple efficient inference techniques. Our experiments demonstrate that skipping such layers yields faster inference that preserves performance, and also show that applying skipping outside these conditions leads to model degradation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper shows that high cosine similarity between adjacent VLM layers indicates where layer skipping can improve inference speed without hurting performance. Experiments confirm early vision layers and late layers have high redundancy and low cross-attention, allowing safe skipping for speedup. However, the theoretical proofs rely on invalid assumptions that don't apply to transformers.

### Strengths
- Establishes formal connections between measurable redundancy metrics (cosine distance) and task-relevant redundancy, providing principled guidance for layer skipping.
- Tests across multiple VLMs and diverse tasks, showing consistent redundancy patterns in early vision and late layers.

### Weaknesses
- The theoretical framework relies on assumptions that don't hold for actual transformer-based VLMs, particularly the Markov chain assumption (X_{ℓ-1} → X_ℓ → Z) which ignores residual connections.
- The identified redundancy patterns (high redundancy in early vision layers and late layers) align with known transformer behavior and are not particularly surprising.
- The claim that "both high redundancy and small cross-attention are necessary conditions" for safe layer skipping is presented without formal proof, relying only on empirical validation.

### Questions
1. Can you provide empirical measurements of the Lipschitz constants (α, β) for the function h(x,y) in actual VLMs? Without these values, how can we interpret the bounds in Theorems 1-2?
2. The Markov assumption X_{ℓ-1} → X_ℓ → Z appears violated in transformers due to residual connections and attention across all layers. Could you either (a) empirically test this assumption, or (b) reformulate your results without it?
3. Your theorems require discrete/finite support for hidden states (Theorems 3-5), but representations in each layer are continuous. What discretization scheme do you propose, and how does discretization error affect your bounds?
4. Have you tested whether your redundancy patterns hold for other tasks (e.g., image captioning) rather than just discriminative VQA tasks? The information flow might differ significantly.
5. Could your redundancy analysis guide other efficiency methods (e.g., which layers to keep during distillation) beyond just layer skipping? This might increase the paper's impact.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a principled framework for determining when layer skipping in vision–language models (VLMs) improves efficiency without degrading accuracy. The central idea is to relate easy-to-measure, adjacent-layer similarity to stronger notions of redundancy that are sufficient for safe skipping. Concretely, the authors define geometric and proximal redundancy based on cosine distance and tail probabilities between consecutive hidden states, and show these imply functional redundancy and are connected to informational redundancy via Fano-type bounds and a lower bound on mutual information upper-bounds performance differences between optimal predictors from layers. The framework is coupled with a cross-modality condition: low cross-attention in a layer band is required so that skipping a modality does not harm the other. Empirically, the paper measures per-layer average cosine distance and proximity probabilities, and visualizes VAR over layers. On LLaVA-1.5 (7B/13B) and LLaVA-NeXT (7B), early and late layers show high redundancy and low VAR, while mid layers concentrate cross-modal fusion. Layer-skipping experiments corroborate the theory: when redundancy and low VAR hold, late entry or early exit  preserves accuracy; violating the conditions degrades performance markedly.

### Strengths
1. Originality. The paper unifies four redundancy notions into a single implication chain and turns them into practical, theory-backed criteria for when late entry and early exit should work in VLMs.  
2. Theory Quality. The analysis is sound and shows that small changes between adjacent layers imply small performance differences, and that low conditional uncertainty connects information- and function-level redundancy; combining this with low cross-modal attention yields clear skip conditions.  
3. Experiments Quality. Measurements of adjacent-layer similarity and cross-attention match the theory: early/late layers are redundant, mid layers do fusion, and skipping only works reliably where the conditions hold while violating them causes clear drops.  
4. Clarity & Significance. The presentation is clean and the framework can standardize safe-skipping decisions, explaining why prior heuristics often work and offering a pathway to reduce inference cost without retraining.

### Weaknesses
1. Metric range consistency. The cosine-distance range is used inconsistently; normalizing the distance or generalizing the bound would resolve this.  
2. Discrete vs. continuous assumptions. Information-theoretic bounds assume finite supports while hidden states are continuous; a simple quantization scheme and error accounting would close the gap.  
3. Markov and Lipschitz assumptions. The required independence and smoothness conditions may be violated in modern architectures; approximate versions and empirical checks would improve realism.  
4. Cross-modal dependence via attention only. Using attention ratios alone doesn’t guarantee causal importance; adding attribution or ablation tests would strengthen the claim.  
5. Compute reporting. Claimed speedups aren’t backed by latency, throughput, memory, or energy measurements; reporting these under fixed settings would quantify real gains.  
6. Comparative baselines. There’s no head-to-head with matched-compute baselines or popular alternatives; adding these would clarify benefits.

### Questions
1. How are continuous hidden states discretized for the information-theoretic bounds, and how sensitive are decisions to the quantization level?  
2. What end-to-end speedups, memory savings, and potential KV-cache overheads do you observe on common GPUs and batch sizes? 
3. Do the skip decisions remain consistent when using alternative similarity measures (e.g., CKA/PWCCA/SVCCA) instead of cosine distance?
4. Please refer to the section on Weaknesses for other questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel visual token pruning method for VLMs, designed to enhance their inference efficiency. The core methodology is operationally simple, involving the skipping of visual tokens at specific layers, such as implementing strategies the authors refer to as 'Late Entry' and 'Early Exit'. The authors place particular emphasis on their analytical framework, which is proposed to determine which layers can be safely skipped. This analysis relies primarily on data derived from cross-attention scores. A key highlight of this work is its demonstration, via the proposed methodology, that visual tokens in the shallow layers of the VLM are similarly dispensable to those in the deep layers. While the latter (the redundancy of deep-layer tokens) is a well-established community consensus, the former (the non-essential nature of shallow-layer tokens) has not been widely discussed. Therefore, the conclusions drawn from this work are of significant inspirational value for the VLM research community.

### Strengths
1.  The paper provides a relatively comprehensive theoretical analysis, establishing a solid justification for determining which layers' tokens can be skipped.
2.  The discovery that visual tokens in most shallow layers can also be skipped is a novel insight that is quite insightful for the VLM research community.

### Weaknesses
1.  **Insufficient Experimental Validation:** For a paper proposing a VLM efficiency optimization method, the experimental validation is rather incomplete. It lacks direct comparisons in terms of both performance and efficiency against other similar methods (e.g., PDrop, TwigVLM et al.).
2.  **Clarity of Results:** The presentation of the final results is unclear. The paper fails to report definitive performance metrics on specific downstream tasks, making it difficult to assess the practical benefits.
3.  **Writing and Terminology:** The paper's clarity needs significant improvement. For instance, the terms "guessing" and "not guessing" in Table 2 are ambiguous, and their intended meanings are not clearly defined. Additionally, the initial mention of "cross-attention" requires a more precise definition in this context. As written, it risks creating confusion with modifications to traditional "cross-attention layers" which hinders reader comprehension.

### Questions
1.  Is the layer-skipping strategy uniform across different tasks, or is it adapted based on the task?
2.  How does the proposed method compare in both performance (accuracy) and efficiency (e.g., latency, throughput) against other SOTA VLM acceleration strategies?
3.  The experiments are conducted on LLaVA-1.5, which may be considered undertrained or outdated. Do the paper's conclusions hold for more recent, powerful models (e.g., Qwen2.5-VL, Gemma3-VL, Qwen3-VL) and across a broader spectrum of tasks?

### Soundness
2

### Presentation
2

### Contribution
2
