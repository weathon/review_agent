# Layer-wise dynamic rank for compressing large language models

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Large language models (LLMs) have rapidly scaled in size, bringing severe memory and computational challenges that hinder their deployment. Singular Value Decomposition (SVD)-based compression has emerged as an appealing post-training compression technique for LLMs, yet most existing methods apply a uniform compression ratio across all layers, implicitly assuming homogeneous information included in various layers. This overlooks the substantial intra-layer heterogeneity observed in LLMs, where middle layers tend to encode richer information while early and late layers are more redundant. In this work, we revisit the existing SVD-based compression method and propose D-Rank, a framework with layer-wise balanced Dynamic Rank allocation for LLMs compression. We first introduce effective rank as a principled metric to measure the information density of weight matrices, and then allocate ranks via a Lagrange multiplier-based optimization scheme to adaptively assign more capacity to groups with higher information density under a fixed compression ratio. Moreover, we rebalance the allocated ranks across attention layers to account for their varying importance and extend D-Rank to latest LLMs with grouped-query attention. Extensive experiments on various LLMs with different scales across multiple compression ratios demonstrate that D-Rank consistently outperforms SVD-LLM, ASVD, and Basis Sharing, achieving more than 15 lower perplexity with LLaMA-3-8B model on C4 datasets at 20% compression ratio and up to 5% higher zero-shot reasoning accuracy with LLaMA-7B model at 40% compression ratio while achieving even higher throughput.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes D-Rank, a SVD-based compression work for LLM that dynamically allocates compression ranks across layers based on their information density, rather than applying uniform compression ratios.  Given the limited tech contribution, and insufficient experiments, this paper should be improved before it can be published.

### Strengths
well written paper, easy to follow.

### Weaknesses
1. Missing citation and limited novelty. The paper's core contribution—dynamic rank allocation for LLM compression—is not new. Hua et al. (2023) "Dynamic Low-rank Estimation for Transformer-based Language Models" (EMNLP 2023 Findings) directly addresses dynamic rank estimation for transformers. The authors cite the 2022 work from the same group but omit their 2023 follow-up on the exact same problem.  This makes the novelty claims appear overstated.
2. The core technical contribution—using spectral entropy-based effective rank to measure information density—lacks proper justification.  The paper provides no explanation for why spectral entropy is the optimal metric, or does it compare against alternatives like stable rank, nuclear norm, Fisher information (FWSVD), or activation-based importance (ASVD). Table 1 shows effective rank values vary across layers, but doesn't demonstrate these values actually correlate with optimal compression performance. The mathematical formulation (Equations 1-2: normalizing squared singular values into a probability distribution, then exponentiating the Shannon entropy) appears arbitrary without theoretical grounding.
3. Unclear and Potentially Unfair Experimental Settings. The experimental setup is poorly specified, creating uncertainty about fair comparison. For example, Table 3 shows FWSVD with perplexity 18156 at 40% - is this without fine-tuning while D-Rank gets updates? Looking at Table 3 (LLaMA-7B results), the numbers for FWSVD and basic SVD are catastrophically bad (perplexities in the thousands), while ASVD, SVD-LLM, Basis Sharing, and D-Rank are reasonable. This suggests: A) FWSVD and SVD are evaluated zero-shot (no recovery procedure); B) ASVD, SVD-LLM, Basis Sharing, D-Rank get some form of correction/update

### Questions
1. Why is spectral entropy the right metric for information density? 
2. Computing spectral entropy requires all singular values. How numerically stable is this for large matrices? Any regularization?

### Soundness
2

### Presentation
3

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
This paper enhances SVD-based methods by assigning different compression ratios to different layers, which improves model accuracy. Leveraging an "effective rank" metric, the authors assign a higher rank to layers with higher information density. The superiority of this approach is validated on numerous models.

### Strengths
1. The motivation is clear, and the paper is well-written.

2 .The results of hardware performance are provided.

3. A combination with LoRa fine-tuning is provided.

### Weaknesses
1. A comparison with SVD-LLM v2 is missing; that method also assigns different compression ratios to different layers.

2. Experiments on 70B models should be added.

3. Figure 1 should be improved, and a description of the image should be added to the caption.

4. Separate ablation studies for the "dynamic rank" and "balancing" components are lacking.

### Questions
Please see the weaknesses above, and:

1. Please explain the difference between the "effective rank" metric established in this paper and the loss-based metric used in SVD-LLM v2. which is better?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes D-Rank, a layer-wise dynamic rank allocation framework for SVD-based compression of large language models. It introduces an “effective rank” metric based on spectral entropy to measure the information density of each layer’s weight matrices and uses a Lagrange-multiplier optimization to allocate ranks proportionally. The method also redistributes ranks among matrices and adapts to grouped-query attention (GQA) models such as LLaMA-3.

### Strengths
1. The paper identifies layer-wise heterogeneity in information density and attempts to address it with an adaptive rank allocation strategy.
2. The effective rank metric provides a simple quantitative measure for weight information, improving interpretability of rank selection.

### Weaknesses
- The approach mainly extends existing SVD-LLM and Basis-Sharing ideas by replacing uniform rank selection with a simple entropy-based allocation. The Lagrange-multiplier optimization and “effective rank” metric are straightforward and do not represent a substantial algorithmic advance.

- The method depends on activation statistics and per-layer SVD computation, adding complexity without clear end-to-end throughput gains in real inference systems. It lacks compatibility with prefix-cache frameworks (e.g., SGLang) or kernel-fused inference, which severely limits deployment feasibility.

- Although throughput is reported, experiments are small-scale and ignore key system factors such as multi-GPU memory partitioning and cache reuse. The cost of computing effective ranks may offset compression gains.

- The entropy-based rank metric lacks clear connection to model generalization or reconstruction fidelity. No analysis links effective rank distributions to actual representational importance.

### Questions
- How does the additional SVD and activation-statistic computation affect total compression and inference time?

- Does “effective rank” correlate with layer importance beyond empirical intuition? A deeper justification or visualization would help.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets the rank-allocation problem in SVD-based LLM compression and proposes D-Rank. The method first computes an effective rank for each group of layers, then uses a Lagrange-multiplier formulation to assign larger ranks to groups with higher effective rank under a fixed global budget. On top of this, it introduces an attention-specific rebalancing step that shifts rank from query/key to value matrices via a hyperparameter, motivated by the empirical observation that value matrices carry higher effective rank. The method also covers the grouped-query attention case in LLaMA-3, where layer grouping degrades compression, by falling back to group size 1 while still applying dynamic rank allocation. Experiments across multiple LLMs and datasets show consistent improvements over existing SVD-based baselines, and ablations confirm the contribution of each component as well as compatibility with LoRA fine-tuning.

### Strengths
* The rank allocation problem is an important and interesting direction to explore.
* The paper is clearly written, with the methodology and experimental setup explained in a smooth and logical way.
* Using an entropy-based effective-rank metric is a reasonable choice for quantifying information density across layers/groups.
* The experiments are comprehensive, and the ablation studies help isolate the contribution of each component.

### Weaknesses
* The novelty is somewhat limited. Structure-aware and adaptive low-rank allocation compression have already been explored in recent work (for example, see [1]). 
* The attention-specific reallocation is empirically effective, but the large gap between value matrices and query/key matrices suggests that grouping them together may not be ideal. It would be helpful to discuss whether an alternative grouping strategy could address this mismatch and potentially remove the extra hyperparameter $\beta$?

[1] Hua et al; Dynamic Low-rank Estimation for Transformer-based Language Models.

### Questions
* How sensitive is the estimated effective rank to the choice of calibration data? For example, does it change significantly if a different subset or a different source corpus is used?
* How important is the entropy-based effective-rank metric itself? It would be useful to report results with alternative approximation method s (For example, stable rank (Fro. norm over 2-norm) or a simple energy-based threshold such as “95% SV energy”) ?

### Soundness
3

### Presentation
3

### Contribution
2
