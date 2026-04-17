# Loss-Aligned Structured Pruning for Large Language Models

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Recent advances in large language models (LLMs) have achieved remarkable performance across diverse tasks, yet their increasing size poses significant storage and computational challenges. Model compression, particularly pruning, has emerged as a crucial strategy to reduce memory footprint and computation while preserving predictive performance. In this work, we present LASP, a Loss-Aligned Structured Pruning method that evaluates the contribution of individual model units, such as neurons and attention heads, to the overall performance, subsequently removing those deemed to be of low importance. By combining the activation magnitudes of model units with their gradients with respect to the loss, LASP defines an importance metric that is directly aligned with the model’s objective, thereby ensuring the preservation of performance. To mitigate uncertainty caused by the limited calibration dataset used for importance estimation, LASP incorporates the Upper Confidence Bound (UCB) strategy, refining the selection of low-importance units. In implementation, LASP leverages a moving average to maintain running statistics and reduce storage overhead. Empirical results across diverse LLMs and benchmarks demonstrate that LASP outperforms state-of-the-art baselines, effectively balancing efficiency and performance, thus enabling the practical deployment of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a loss-aligned structured pruning approach for large language models (LLMs). The key idea is to align pruning decisions (i.e., which structures—in this case groups of weights/rows/columns—are removed) with their measured impact on the loss (or proxy thereof) rather than purely on magnitude or heuristic importance.

### Strengths
The paper proposes a loss-aligned pruning metric that directly correlates with the model’s final task loss, which is more task-relevant than using only activations or gradients. The inclusion of uncertainty modeling via an Upper Confidence Bound (UCB) is relatively novel in the pruning literature and provides a principled way to handle estimation uncertainty. The method is empirically validated across Llama and some datasets, demonstrating a good balance between pruning efficiency and performance retention.

### Weaknesses
1. Testing should be conducted on newer models, such as the Qwen3 series. Moreover, whether the method remains effective on reasoning models is also an open question.
2. The evaluation datasets are insufficient. The current datasets are too simple, and the pruned parameters might not cover the activation regions of these models. More diverse and challenging datasets should be included.
3. The proposed method feels too incremental and lacks sufficient novelty or contribution.
4. In addition, the paper is missing important citations that should be discussed, such as SparseGPT [1], LoraPrune [2], Compresso [3], and LLM-Pruner [4].
References:
[1] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot
[2] LoraPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
[3] Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models
[4] LLM-Pruner: On the Structural Pruning of Large Language Models

### Questions
Same as weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents LASP (Loss-Aligned Structured Pruning), a method for efficient compression of Large Language Models (LLMs) through structured pruning. Instead of reconstructing weight matrices or estimating second-order Hessians, LASP introduces a first-order, loss-aligned importance metric that evaluates neuron or head significance using activation–gradient interactions directly tied to model loss. To address data uncertainty from small calibration sets, the method integrates an Upper Confidence Bound (UCB) strategy and employs moving-average running statistics to minimize storage overhead. Experiments on LLaMA, LLaMA-2, and Vicuna models show that LASP achieves superior performance retention—up to 93.5% at 25% pruning—compared with state-of-the-art baselines such as SliceGPT.

### Strengths
1. No fine-tuning required: LASP achieves strong performance preservation without any fine-tuning, making it highly practical for large-scale deployment where retraining costs are prohibitive.
2. More accurate pruning criterion: By aligning pruning decisions with loss gradients rather than proxy metrics like weight magnitude or reconstruction error, LASP offers a theoretically grounded and empirically superior estimation of unit importance.
3. Engineering and algorithmic efficiency: The paper introduces concrete optimizations—layer-wise backward computation, running statistics, and small calibration sets—to make loss-based pruning computationally feasible on single GPUs, effectively addressing the global loss alignment cost.

### Weaknesses
1. Missing comparison with FLAP (AAAI 2024): Since the paper focuses on pruning without fine-tuning (w/o FT), it is crucial to include comparisons with FLAP, which also targets the same setting, to better contextualize LASP’s contributions.
2. Limited model diversity: The evaluation is restricted to the LLaMA and Vicuna series. Experiments on more diverse architectures such as DeepSeek or Qwen would strengthen the generality claims.
3. Lack of pruning-time analysis: Given that the motivation is to avoid fine-tuning costs, it is equally important to report the actual pruning runtime and computational resources, to validate that the method indeed offers an overall efficiency gain.
4. Missing w/ FT evaluation: Although the paper emphasizes the w/o FT setting, fine-tuning typically incurs a moderate cost compared to pretraining. Including w/ FT experiments would reveal the upper bound of model performance and provide a fuller understanding of LASP’s potential.

[1] Fluctuation-based Adaptive Structured Pruning for Large Language Models. AAAI 2024

### Questions
See weaknesses

### Soundness
2

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
3

### Summary
This paper proposes LASP (Loss-Aligned Structured Pruning), a post-training structured pruning framework for large language models (LLMs). LASP evaluates the importance of structural units (neurons or attention heads) by combining their activation magnitudes with gradients of the loss function, thereby aligning pruning decisions with the model’s training objective. To mitigate uncertainty from limited calibration data, LASP introduces an Upper Confidence Bound (UCB)–style regularization that accounts for variance in importance estimates. Empirical results on several open-source models (LLaMA, LLaMA2, Vicuna) show consistent performance improvements over SliceGPT across perplexity and zero-shot reasoning benchmarks. The method is computationally lightweight and hardware-friendly, requiring no retraining.

### Strengths
+ Clear and intuitive formulation. The paper presents a well-motivated first-order approximation of loss change and integrates uncertainty modeling via UCB in a mathematically coherent way. The approach is conceptually simple and easy to implement.

+ Strong experimental coverage. Results are reported for multiple LLM families and scales, with pruning ratios up to 30%. The evaluation includes both perplexity (WikiText-2) and zero-shot reasoning tasks, showing consistent trends.

+ Practicality. LASP is post-training and structured, making it appealing for real-world deployment where retraining is infeasible. Efficiency improvements in memory and latency are quantified.

### Weaknesses
+ Limited novelty relative to prior structured pruning methods. While the “loss-aligned” formulation is conceptually elegant, the underlying idea—using first-order gradient × activation statistics to rank units—is closely related to long-standing importance-based pruning and Taylor expansion–based saliency measures. The use of UCB for uncertainty modeling is a mild novelty but not deeply justified or theoretically analyzed. Overall, LASP reads as an incremental extension of SliceGPT or LLM-Pruner rather than a substantive conceptual advance.

+ Inadequate comparative evaluation. The experimental section primarily contrasts LASP with SliceGPT (with and without LoRA fine-tuning). However, other strong baselines—SparseGPT, LLM-Pruner, or WoodFisher—are discussed but not empirically compared. This omission weakens the claim of “state-of-the-art” performance, especially since SparseGPT has demonstrated superior accuracy-efficiency trade-offs under similar data constraints.

+ Weak theoretical justification for UCB integration. The introduction of the UCB term appears heuristic. There is no formal connection drawn between the multi-armed bandit setting (where UCB is derived) and the pruning uncertainty scenario here. Moreover, the hyperparameter α controlling the uncertainty trade-off is tuned empirically without principled guidance or sensitivity analysis beyond a basic ablation (Figure 3).

+ Lack of interpretability or mechanistic analysis. The paper provides no insight into which neurons or attention heads are pruned or retained, nor any qualitative discussion of redundancy patterns beyond superficial loss plots. This limits understanding of why LASP works and whether its pruning behavior is generalizable across architectures.

+ Presentation and clarity issues. The exposition is generally clear but occasionally repetitive (e.g., re-deriving standard first-order approximations). Figures could better highlight empirical differences between LASP and baselines. Some notation (e.g., the definition of ∆Lu) is inconsistent across sections, and the ablation analysis is relatively shallow.

### Questions
+ Could you formalize the connection between UCB and pruning uncertainty beyond analogy? Is there a principled reason to expect the UCB bonus to approximate the true risk of pruning a unit?

+ Why were LLM-Pruner and SparseGPT omitted from empirical comparison, given their relevance and accessibility?

+ Can LASP be extended to finer-grained structures (e.g., Q/K/V sub-dimensions or per-channel pruning in MLPs), and would the loss-alignment principle still hold?

+ How robust is α across architectures and tasks? Could you propose a heuristic or scaling rule to set α without exhaustive tuning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LASP (Loss-Aligned Structured Pruning), a post-training structured pruning method for compressing large language models efficiently. LASP departs from reconstruction- or Hessian-based approaches by introducing a first-order, loss-aligned importance metric, which measures neuron or attention head importance through activation–gradient correlations directly tied to loss behavior. To mitigate calibration data uncertainty, LASP integrates a UCB-based exploration strategy and moving-average statistics to stabilize importance estimation while keeping storage overhead low. Experiments on LLaMA, LLaMA-2, and Vicuna models demonstrate that LASP maintains up to 93.5% performance at 25% pruning, outperforming SliceGPT and other pruning baselines under the no fine-tuning (w/o FT) setting.

### Strengths
(1) The proposed activation–gradient-based importance metric provides a principled and effective way to measure pruning sensitivity, improving upon traditional weight magnitude or reconstruction heuristics.
(2) The framework introduces practical engineering optimizations—layer-wise backward computation and running statistics—that make loss-based pruning computationally feasible even on limited hardware.
(3) LASP preserves strong performance in the w/o FT setting, significantly reducing deployment overhead for large models.

### Weaknesses
(1) Evaluation focuses on LLaMA and Vicuna only; broader validation on architectures such as DeepSeek or Qwen would better support claims of generality.
(2) Missing comparison with FLAP (AAAI 2024), as FLAP also operates under the w/o FT scenario, omitting this baseline limits contextual understanding of LASP’s contribution.
(3) The paper does not provide detailed runtime or computational cost analysis, leaving unclear whether pruning itself offers net efficiency gains.
(4 )While the paper emphasizes the w/o FT regime, including w/ FT results could highlight the potential performance ceiling and help assess trade-offs more comprehensively.

### Questions
What are the actual pruning-time and resource costs, and how do they compare with existing efficient pruning baselines?
Would including fine-tuned (w/ FT) results clarify LASP’s headroom and performance scalability?

### Soundness
3

### Presentation
3

### Contribution
2
