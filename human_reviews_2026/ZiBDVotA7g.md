# Gated LoRA: Dual-Purpose Projections for Parameter-Efficient Mini-Expert Fine-Tuning

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Low-Rank Adaptation (LoRA) is widely used for parameter-efficient fine-tuning (PEFT) of large language models (LLMs). Yet, its uniform activation of all rank components can lead to task interference and hinder generalization when fine-tuning a model to multiple tasks and datasets. We introduce Gated LoRA, which employs input-dependent gating to selectively activate only the most relevant rank-1 directions. The key design is a dual‑purpose projection: the same matrices that compute LoRA features also drive rank selection, adding no extra trainable parameters. Across nine language understanding benchmarks and diverse LLM backbones, Gated LoRA reduces task interference and achieves up to 2.9-point accuracy gains in multi-task settings and 3.6-point gains in single-task fine-tuning over standard LoRA, while incurring negligible inference latency overhead and no additional training parameters. Our results demonstrate that fine-grained, input-dependent adaptation makes LoRA more robust, adaptive, and interference-resistant, suggesting a way for scalable multi-task PEFT in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the challenges in fine-tuning large language models (LLMs) where full fine-tuning is computationally expensive, leading to the adoption of parameter-efficient fine-tuning (PEFT) methods like Low-Rank Adaptation (LoRA). The proposed solution, Gated LoRA, treats each rank-1 component as a mini-expert and uses dual-purpose projections for both feature computation and rank selection, incorporating top-k or ReLU-based gating with sparsity and orthogonality regularization to enable fine-grained, adaptive updates.

### Strengths
1. The dual-purpose design reuses the down-projection matrix for both adaptation and gating, eliminating the need for separate routing parameters. This approach maintains parameter efficiency equivalent to standard LoRA while enabling input-dependent selectivity. It simplifies integration into existing frameworks without increasing training complexity.

2. Empirical results show consistent accuracy improvements over baselines like LoRA, LoRAMoE, and MoLA across multiple backbones and tasks. Gains reach up to 2.9 percentage points in multi-task settings and 3.6 in single-task fine-tuning. These enhancements demonstrate robustness in handling task diversity.

 
3. Negligible latency overhead during training and inference makes the method practical for real-world deployment. Compared to MoE-based alternatives, it adds minimal computational cost. This efficiency supports scalable application in resource-constrained environments.

### Weaknesses
1. Experiments use only LLaMA family backbones, potentially overlooking generalization to other architectures like GPT or BERT. Differences in model designs might affect gating efficacy. Testing on diverse LLMs would provide stronger evidence.

2. The paper samples equal examples for multi-task training, resulting in a modest dataset size. This may not capture real-world imbalances across tasks. Larger or varied sampling could better simulate practical scenarios.

 
3. Comparisons exclude some recent PEFT variants beyond the listed baselines. Emerging methods might offer competitive alternatives. Including them could highlight relative advantages more comprehensively.

4. The method underperforms on math benchmarks without chain-of-thought prompting. This suggests limitations in compositional reasoning tasks. Adapting gating for advanced prompting techniques might be necessary.

5. There have been some works considering regarding LoRA Adapters as experts to be routed [1,2]. However, these works are not discussed or compared. 


[1] Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging. Arxiv 2025.
[2] SLoRA: Scalable Serving of Thousands of LoRA Adapters. In MLsys 2024.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a PEFT method named Gated LoRA. Its core idea is to reuse the standard LoRA down-projection matrix $A$ as a gating mechanism to dynamically select which rank-1 components (which the authors call "mini-experts") to activate. The selling point of this design is that it achieves input-adaptive sparse activation without introducing any additional trainable parameters. The authors demonstrate through experiments that this method achieves performance improvements over standard LoRA and other MoE-PEFT methods on several single-task and multi-task benchmarks.

### Strengths
1.	Design Simplicity: The "dual-purpose projection" design is very simple, achieving dynamic gating with zero parameter overhead, which is attractive from an engineering perspective.

2.	Parameter Efficiency: In the current research climate focused on PEFT parameter budgets, "not adding any trainable parameters" is a very strong selling point.

3.	Empirical Results: It must be acknowledged that the paper's experimental results are positive. It consistently outperforms standard LoRA across multiple benchmarks (though the improvement margin is around 2-3 points, not disruptive), which proves the method is effective.

### Weaknesses
1.	Incremental Contribution: As mentioned before, the core innovation here is very thin. It is essentially adding a "zero-cost" gate to the existing LoRA framework, making it an incremental combination and optimization of existing technologies (LoRA, MoE).

2.	Metaphor Misuse ("Mini-Expert"): The paper repeatedly calls rank-1 components "mini-experts." This is a significant conceptual overstatement. In a standard MoE, an "expert" is typically a full FN layer with complex non-linear transformation capabilities. A rank-1 matrix, however, merely defines a single update direction in a high-dimensional space. What Gated LoRA does is, more accurately, "dynamic selection of feature directions" within LoRA's low-rank subspace, not routing between multiple complex "reasoning paths." This misuse of metaphor is misleading.

3.	ntroduction of New Complexity (Hyperparameter Complexity): While claiming "no extra parameters," the method introduces new hyperparameters. Gated LoRA's training and tuning complexity is actually increased. The paper does not discuss the sensitivity of these new hyperparameters and only provides fixed values.

4.	Ignoring Training Overhead: As mentioned earlier, this method (especially the ReLU variant) trades significantly increased training time for performance gains. For many resource-constrained scenarios, longer training time can be a more significant issue than a few extra parameters.

### Questions
1.	On the definition of "expert": Can the authors provide a more rigorous argument for why a rank-1 matrix can be called an "expert"? What is its functional comparability (beyond just "being selected") to a full FFN expert in an MoE?

2.	On training overhead: The training time for the ReLU gate (Table 1, 1.26s/batch) is more than double that of standard LoRA (0.61s/batch). Do the authors consider this overhead reasonable? If I were to double the rank r of standard LoRA (which would also increase training time), could it achieve performance comparable to Gated LoRA (r=128, k=32)?

3.	How were these regularization coefficients chosen? Were they fixed at 1.0 for all models and datasets? How would performance change if these values were altered (e.g., 0.1, 0.01)? This is crucial for assessing the method's ease of use.

### Soundness
3

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
This paper introduces Gated LoRA, a new PEFT method to reduce task interference in standard LoRA during multi-task fine-tuning. It treats each rank-1 LoRA as a small expert and uses an input-dependent gate (top-k or ReLU) to activate them sparsely. The method’s core design is a dual-use projection that reuses LoRA’s A matrix to produce both features and gating signals, adding no extra parameters.

### Strengths
- The paper is well written and easy to follow.

- The design that reuses the A matrix for gating is clever and elegant.

### Weaknesses
- The improvement in Table 1 is quite small. It is unclear whether the gain is due to the proposed method or just random variation.

- In some settings, increasing the LoRA parameters even hurts performance. The paper does not show whether the method works reliably across different ranks, especially for low-rank cases.

- There are no examples showing how or why multi-task LoRA fine-tuning causes task interference. Would increasing the rank reduce this interference? And why does multi-task learning lead to interference instead of mutual improvement? This seems a bit counterintuitive.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Gated LoRA, an extension of Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning of LLMs. Instead of uniformly activating all LoRA rank directions, the authors interpret each rank-1 component as a mini-expert and introduce input-dependent gating to select only relevant ranks per token.
A key design is the dual-purpose projection — the same LoRA down-projection matrix A is reused both for feature extraction and gating, avoiding additional routing parameters.

Two selectors are explored: fixed top-k gating and ReLU gating with sparsity regularization.
Experiments on nine NLU benchmarks using LLaMA backbones show moderate improvements (up to +3.6 points) over standard LoRA while keeping training efficiency nearly identical.

### Strengths
The idea of reusing LoRA’s down-projection for gating (dual-purpose projection) is elegant and avoids the parameter overhead common in MoE-style PEFT methods.

The proposed gating mechanism achieves conditional activation without introducing additional routing layers or large routers.

### Weaknesses
[Misleading “MoE-like” terminology]
The paper repeatedly refers to the method as “MoE-style”, yet no actual expert routing or token-level dispatch exists. Gating operates purely within the LoRA rank dimensions (dimension-level gating), not across separate expert modules as in true MoE architectures. The term “mini-expert” is metaphorical rather than architectural.

[No additional memory efficiency beyond LoRA]
The paper’s title and abstract suggest improved “parameter-efficiency” or “memory-efficiency.” In reality, trainable parameters remain identical to standard LoRA; the only efficiency stems from activation sparsity, not parameter reduction. Thus, “memory-efficient” is misleading.

[Limited experimental diversity]
All experiments are confined to LLaMA-family models (LLaMA-2-7B/13B and LLaMA-3.1-8B). There is no evaluation on other architectures (e.g., Mistral, Falcon, OPT, or encoder-only models like RoBERTa), leaving the general applicability unverified.

[Marginal computational benefit]
The reported inference latency gain (≈0.04–0.1s per batch) is minor and may fall within measurement noise.
Claims of efficiency would be stronger with FLOP analysis or hardware-level profiling.

### Questions
Avoid using “MoE-like” unless token-level routing or explicit expert partitioning is introduced. More accurate terms would be “Rank-wise gated LoRA” or “Self-gated LoRA.”

Include experiments on non-LLaMA architectures or smaller encoder-only models to support claims of generality.

### Soundness
2

### Presentation
2

### Contribution
2
