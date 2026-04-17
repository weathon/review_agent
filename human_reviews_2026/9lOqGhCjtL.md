# Quantifying Expert Specialization for Effective Pruning in Mixture-of-Experts Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Mixture-of-Experts (MoE) architectures enable efficient scaling of language models through sparse activation. However, their deployment is hindered by a significant memory bottleneck, as all expert parameters must remain resident in memory. Expert pruning is an effective technique to mitigate this issue. Existing methods rely on layer-wise metrics based on either routing behavior or expert outputs. These approaches fail to capture the global influence of an expert on cross-layer information flow. In this paper, we introduce a framework for cross-layer information flow analysis. We propose a novel metric called the Expert Specialization Index (ESI). ESI quantifies the entropy of an expert's influence on downstream routing distributions. This allows it to distinguish between functionally specialized experts and redundant, general-purpose ones. Our analysis on Mixtral-8x7B and Qwen1.5-MoE reveals significant differences in their expert specialization profiles. This leads to a key finding we term architecture-strategy fit. Models with highly specialized experts benefit from preserving the original routing distribution via redirection. In contrast, models with less specialized experts are better served by removing experts and re-normalizing routing probabilities. Supported by experimental results, our ESI analysis allows us to explore how to design compression strategies for different MoE architectures. Our findings provide insights into the relationship between model architecture and effective compression strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the Expert Specialization Index (ESI), which measures the entropy of each expert’s influence on downstream routing to quantify specialization in Mixture-of-Experts (MoE) models. Based on ESI, the authors propose a pruning strategy and the “architecture–strategy fit” principle, showing that models with specialized experts (e.g., Qwen-MoE) benefit from routing preservation, while models with redundant experts (e.g., Mixtral) perform better with deletion.

### Strengths
- Clear motivation and well-written exposition.

- ESI offers an interpretable, global metric for analyzing expert importance.

- Provides architectural insights into differences between MoE models.

### Weaknesses
- While the ESI concept is well-formulated, the underlying idea, i.e., quantifying expert importance via information flow or entropy, resembles existing approaches (e.g., router-logit analysis, mutual-information–based redundancy metrics). The work is more an elegant integration than a breakthrough.

- Experiments are restricted to Mixtral and Qwen1.5-MoE under multi-choice benchmarks. The pruning framework is not evaluated on large-scale generative tasks or long-context scenarios, which are crucial for MoE deployment.

- Computing pairwise cross-layer flow for all expert pairs is potentially expensive (O(L·n²)). The paper does not analyze the scalability or runtime cost of ESI computation.

- Several works have proposed cross-layer or information-based MoE pruning and specialization metrics, but they are not discussed ([1, 2]).

- While ESI-guided pruning improves accuracy modestly over Frequency and HC-SMoE, the gains (1–3%) are relatively small and mostly confined to specific architectures.
---

[1] He et al., Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques  
[2] Xie et al., MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

### Questions
Please refer to the weakness.

### Soundness
2

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
The paper addresses a critical deployment challenge in Mixture-of-Experts (MoE) models. While MoE architectures achieve computational efficiency through sparse activation (only activating *k* experts per token), they face a **memory bottleneck** - all expert parameters must remain in GPU memory to support dynamic routing, severely limiting deployment on commodity hardware.

### Key Contributions

1. **Cross-Layer Expert Flow (CLEF) Framework (§ 3.2)**
    
    The paper introduces a novel cross-layer analysis approach that measures how experts influence downstream routing decisions. (check with equation 2)
    
    it captures meaningful information flow requires all three conditions to be met sequentially - the source must be active, produce significant output, and be received by downstream experts. For the final MoE layer, the framework adapts to compute flow toward the vocabulary space rather than to other experts.
    
2. **Expert Specialization Index (ESI) (§ 3.3)**
    
    Built on CLEF, the ESI quantifies an expert's functional specialization through entropy analysis. (check with equation 3-5) 
    
    interpretation:
    
    - **High ESI** (approaching 1): Low entropy, concentrated influence on few downstream experts → highly specialized function
    - **Low ESI** (approaching 0): High entropy, diffuse influence across many experts → general-purpose, redundant role
3. **Architecture-Strategy Fit Principle**
    
    The paper discovers fundamental architectural differences between MoE models and proposes a principle for matching compression strategies to architecture:
    
    **Architectural Profiles**: Mixtral-8x7B shows continuous ESI distributions indicating functional redundancy among experts, while Qwen1.5-MoE exhibits bimodal patterns with many generalists and few critical specialists (some reaching ESI = 1.0).
    
    **Router Adaptation Strategies**: Router Deletion masks pruned experts to negative infinity forcing probability redistribution, while Router Redirection preserves original routing with pruned experts contributing zero.
    
    **The Key Finding**: High-specialization models (Qwen) benefit from Redirection to preserve specialist routing (+4.48% accuracy), while low-specialization models (Mixtral) benefit from Deletion for adaptive redistribution (+20.28% accuracy).

### Strengths
- The Cross-Layer Expert Flow (CLEF) framework represents a significant conceptual advance. Unlike existing methods that rely on layer-local metrics (activation frequency, output similarity), the authors recognize that an expert's true importance lies in how it shapes information flow through subsequent layers.
- The ESI metric is theoretically sound, using entropy to quantify functional specialization and normalization to [0,1] makes it interpretable across different architectures. The metric captures the intuition that specialized experts (low entropy, concentrated influence) are more critical than generalist experts (high entropy, diffuse influence). The mathematical progression from raw flows to normalized distributions to entropy is clear and justified.
- The principle that compression strategies must align with architectural specialization profiles is a valuable insight. The dramatic performance differences between Delete and Redirect strategies validate this hypothesis.

### Weaknesses
- The paper assumes that "information flow" from expert $E_i^{(l)}$ to $E_j^{(l+1)}$ can be meaningfully captured by multiplying three scalar values. **Information doesn't flow between experts directly** - it flows through the residual stream after layer normalization and attention blocks. Due to this i’m not very sure about this and unable to wrap my head around this.
- The paper treats MoE routing as a fixed, *deterministic* system where information "flows" between experts. But MoE models are designed for:
    - **Load balancing** - routers actively try to distribute tokens evenly
    - **Exploration** - routing has noise/randomness during training
    - **Redundancy** - multiple experts can handle similar inputs

### Questions
- Unable to understand how cross-layer information flow work (figure 1) (mentioned about the same above too.
- MoEs employ load balancing losses and routing noise during training. How does your analysis account for these dynamic aspects? (didnt find anything relevant with repsect to this in the paper)

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
This paper introduces the Expert Specialization Index (ESI), a novel, non-local metric for pruning Mixture-of-Experts (MoE) models. ESI aims to quantify an expert's functional specialization by measuring the entropy of its influence on the routing distributions of subsequent layers. The authors use this metric to find that MoE architectures have distinct specialization profiles: Qwen1.5-MoE is highly specialized, whereas Mixtral-8x7B is functionally redundant. This observation leads to their central claim of "architecture-strategy fit" , which posits that specialized models (Qwen) require "Router Redirection" to preserve routing paths, while redundant models (Mixtral) benefit from "Router Deletion"  and routing probability re-normalization. ESI-guided pruning shows strong results, particularly on the highly specialized Qwen model.

### Strengths
- The paper's core idea—moving from layer-local pruning heuristics to a global, cross-layer analysis of information flow —is well-motivated and represents a more principled approach to quantifying expert importance.
- The "architecture-strategy fit"  is a significant contribution. It provides a compelling, data-driven explanation for why different router adaptation strategies (Deletion vs. Redirection) are required for different MoE architectures.
- The proposed ESI metric is training-free and computationally efficient, requiring only a single forward pass over a small calibration dataset.

### Weaknesses
- The primary claim of "architecture-strategy fit" is supported by experiments on only two models. These models have confounding architectural variables—Mixtral is a standard 8-expert design, while Qwen1.5-MoE is a 64-expert design that also includes shared experts. The conclusions would be far more robust if validated on a wider and more controlled set of MoE architectures.
- The ESI is computed using a remarkably small calibration set of 64 samples. The paper provides no analysis of the metric's stability. If the expert ranks are sensitive to the choice of these 64 samples, the pruning method is not robust.
- The Cross-Layer Expert Flow (CLEF) metric is defined as a product of three terms with little justification. The inclusion of the $||Output(E_i^{(l)})||_2$ term is particularly questionable, as it conflates output magnitude with functional specialization. Furthermore, the softmax temperature $\tau=1$ is presented as a "default" but is an unevaluated hyperparameter that directly controls the metric's sensitivity.
- The ESI-guided method fails to achieve state-of-the-art results on Mixtral-8x7B, performing slightly worse than the HC-SMoE baseline. The authors' explanation—that the metric's choice is less important in redundant models —also implies that their more complex cross-layer analysis provides no benefit over simpler output similarity in this common architecture.
- The appendix results are concerning. On the MathQA task, ESI-pruning underperforms the HC-SMoE baseline significantly. The authors' post-hoc explanation suggests that multi-step reasoning relies on the "collaborative function of multiple generalist experts". This implies a critical flaw: ESI may systematically mistake critical generalists for redundant generalists, pruning experts that are essential for complex reasoning.
- The paper states the CLEF metric is adapted for the final MoE layer to compute flow "towards the vocabulary space" and is normalized by $\log V$. This adaptation is not mathematically defined and is a non-trivial omission, as the vocabulary size $V$ is orders of magnitude larger than the expert count $n_e$, fundamentally changing the ESI calculation for the final layers.

### Questions
- Please provide an analysis of ESI's stability. How much do the expert rankings (and thus the set of pruned experts) fluctuate across different randomly drawn 64-sample calibration sets?
- Can you provide an ablation study on the CLEF formulation? Specifically, what is the performance impact of removing the L2 norm term, which seems poorly justified?
- The failure on MathQA suggests ESI incorrectly penalizes necessary generalists. How does your metric distinguish between a redundant generalist (high entropy, low value) and a critical generalist (high entropy, high value)?
- Please explicitly define the flow calculation for the final MoE layer flowing to the vocabulary space. How does using $\log V$ as the normalizer affect the ESI scores for this layer compared to others normalized by $\log n_e$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
- The paper introduces the Expert Specialization Index (ESI), which quantifies an expert's functional importance by measuring the entropy of its influence on downstream routing distributions across layers, rather than relying on local layer-wise metrics like activation frequency or output similarity.

- Through analysis of Mixtral-8x7B and Qwen1.5-MoE, the authors discover that optimal compression strategies depend on model architecture—models with highly specialized experts (Qwen) benefit from preserving original routing distributions via redirection, while models with redundant experts (Mixtral) perform better with router deletion and re-normalization.

- ESI is derived from a Cross-Layer Expert Flow (CLEF) analysis that models how an expert's output shapes information flow through subsequent layers by combining three factors: gating weight, output magnitude, and downstream reception probability.

### Strengths
- The paper introduces a principled approach to quantifying expert importance through downstream routing influence (Equation 2), moving beyond local layer-wise metrics. The formulation captures three sequential dependencies (activation, magnitude, reception) and provides strong empirical validation through inverse pruning experiments showing 18.77% accuracy drop when removing top-10% ESI experts on ARC-C.

- The discovery that router adaptation strategy must align with model specialization profiles is rigorously demonstrated with performance differences—Redirect outperforms Delete by 4.48% on Qwen1.5-MoE while Delete outperforms Redirect by 20.28% on Mixtral-8x7B at 50% pruning.

- The evaluation compares against state-of-the-art methods (NAEE, HC-SMoE) across 7 benchmarks and two architecturally distinct models (standard MoE vs. shared-expert architecture), with task-specific pruning analysis in Appendix C. The method achieves SOTA results on high-specialization architectures while demonstrating appropriate limitations on low-specialization models.

### Weaknesses
- The ESI formulation (Equations 2-5) contains several arbitrary decisions without proper justification or ablation studies. Why is L2 norm the appropriate output magnitude measure? Why multiply the three factors rather than use alternative aggregations? The temperature parameter τ=1 is set without exploration of its impact. Most critically, the claim that ESI captures "global" influence is overstated—it only measures single-hop dependency (layer l to l+1), not true multi-layer propagation, and no theoretical argument is provided for why entropy of downstream routing distribution fundamentally quantifies functional specialization versus other possible measures.

- The evaluation uses only 64 calibration samples (seemingly arbitrary with no ablation on calibration size) and lacks basic statistical rigor—no error bars, confidence intervals, multiple runs, or significance testing across 7 benchmarks. The architecture-strategy fit principle is derived from only 2 models, which is insufficient to establish a general principle—more diverse MoE architectures are needed for validation.

- The paper acknowledges that "complex, multi-step reasoning may depend more on collaborative function" but doesn't reconcile why ESI, which supposedly captures functional importance, fails to identify important experts for mathematical reasoning.

- No analysis of how the method scales beyond the two tested models (to models with 100+ experts or 100+ layers). The final layer adaptation using vocabulary size V in normalization (Section 3.3) is mentioned but never validated—does this actually work given the massive difference in scale (vocabulary ~32k vs. 8-64 experts)?

### Questions
- Can you provide theoretical justification or empirical ablations for key design choices in Equations 2-5? Specifically: (1) Why is multiplicative aggregation of three factors optimal versus alternatives (e.g., weighted sum, geometric mean)? (2) Why does entropy of downstream routing distribution fundamentally measure "functional specialization" rather than other properties? (3) How sensitive is performance to the temperature parameter τ and calibration dataset size (currently only 64 samples)?

- The principle is derived from only two models with opposite characteristics. Can you test on additional MoE architectures (e.g., DeepSeekMoE, Switch Transformers, models with varying expert counts) to validate this is a general principle? What specific ESI threshold or distributional property determines which strategy to use for new, unseen architectures?

- Can you explain why cross-layer flow analysis fails for mathematical reasoning? Does this suggest fundamental limitations in domains requiring multi-expert collaboration, and if so, how would you detect such scenarios a priori?

### Soundness
3

### Presentation
2

### Contribution
2
