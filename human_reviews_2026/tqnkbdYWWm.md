# FlexLoRA: Entropy-Guided Flexible Low-Rank Adaptation

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 2

## Abstract
Large pre-trained models achieve remarkable success across diverse domains, yet fully fine-tuning incurs prohibitive computational and memory costs. 
Parameter-efficient fine-tuning (PEFT) has thus become a mainstream paradigm. 
Among them, Low-Rank Adaptation (LoRA) introduces trainable low-rank matrices and shows strong performance, nevertheless, its fixed-rank design limits flexibility. 
Dynamic rank allocation methods mitigate this issue by pruning redundant directions; however, they often rely on heuristic, element-level metrics that globally sort rank directions without matrix-wise distinction, and they lack mechanisms to expand capacity in layers requiring additional adaptation. 
To overcome these limitations, we propose FlexLoRA, an entropy-guided flexible low-rank adaptation framework that (i) evaluates matrix importance via spectral energy entropy, (ii) supports rank pruning and expansion under a global budget, and (iii) employs zero-impact initialization for newly added singular directions to ensure stability. 
By addressing granularity, flexibility, and stability limitations, FlexLoRA provides a more principled solution for PEFT. 
Extensive experiments show that FlexLoRA consistently outperforms state-of-the-art baselines across benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes FlexLoRA, an entropy-guided dynamic low-rank adaptation method for parameter-efficient fine-tuning of large models. Unlike standard LoRA with fixed rank, FlexLoRA supports both rank pruning and expansion under a global parameter budget. It evaluates matrix importance using spectral entropy, enabling flexible capacity reallocation across layers. Additionally, a zero-impact initialization strategy ensures stable training when adding new rank directions. Experiments show that FlexLoRA consistently outperforms existing dynamic LoRA variants under the same parameter budget.

### Strengths
1. The paper addresses the key limitation of LoRA’s fixed rank by introducing a novel bidirectional rank adjustment mechanism that supports both pruning and expansion under a global budget. It also proposes a new matrix-level entropy-guided importance metric, offering a new perspective on evaluating low-rank matrices importance.
2. The paper presents comprehensive experiments across multiple domains, including Natural Language Understanding, Visual Recognition, and Commonsense Reasoning, along with extensive ablation studies.

### Weaknesses
1. Although the paper introduces a bidirectional rank adjustment mechanism supporting both pruning and expansion, the idea — if we ignore the proposed importance metric — conceptually resembles a combination of AdaLoRA (for adaptive pruning) and IncreLoRA [1] (for rank expansion). Hence, its novelty may be somewhat limited at the framework level.
2. Despite the conceptual soundness, the experimental credibility of the paper is questionable.
   - The tables do not clearly indicate which results are reproduced and which are cited from previous works, making it difficult to assess fairness. Moreover, no detailed experimental settings are provided in either the main text or the appendix, hindering reproducibility and leaving uncertainty about how baseline methods were implemented.
   - The improvements on commonsense reasoning tasks are marginal (e.g., LoRA 85.4% vs. FlexLoRA 85.5%), and given the high variance typically observed in such benchmarks, this raises doubts about the claimed effectiveness.
   - For visual recognition, the reported performance appears lower than that of prior works such as MLAE [2] and GLoRA [3], yet the paper provides no explanation or comparison of configurations, leaving potential inconsistencies unresolved.
   - The ablation study is also incomplete — only a subset of GLUE benchmarks (CoLA, MRPC, RTE, STS-B) is reported instead of the full suite, which may indicate selective reporting of favorable results.

------

[1] IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning. https://arxiv.org/abs/2308.12043

[2] MLAE: Masked LoRA Experts for Visual Parameter-Efficient Fine-Tuning. https://arxiv.org/abs/2405.18897

[3] One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning. https://arxiv.org/abs/2306.07967

### Questions
Additional Questions and Suggestions:

1. Prior study [1] have shown that randomly masking certain parameters within the LoRA matrices can effectively reduce redundancy and improve performance. This raises an important question: how does such a temporary stochastic masking approach compare to the permanent structural adjustments (i.e., pruning or expansion) proposed in this paper? An experimental comparison between random masking and FlexLoRA’s bidirectional rank adjustment would help clarify the relative advantages and trade-offs in terms of performance, stability, and generalization.
2. The paper does not discuss the computational efficiency of FlexLoRA. Since the proposed method introduces additional operations for rank evaluation, pruning, and expansion, it likely incurs extra computational overhead. Given that the performance gains of FlexLoRA appear modest, it is crucial to report and compare the training and inference time costs across different methods to fairly assess their overall efficiency.

------
If the authors can address the above concerns in response, I am willing to raise my score. 

[1] MLAE: Masked LoRA Experts for Visual Parameter-Efficient Fine-Tuning. https://arxiv.org/abs/2405.18897

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
4

### Summary
In this work, the authors propose to evolve the rank of low-rank adapters using entropic regularization on the singular value set. The framework allows both to reduce and to inflate the rank of the adapters, enabling dynamic reallocation across layers.
The proposed method is backed up by a variety of different numerical results.

### Strengths
The work is well written and clear. The proposed method, despite being very simple, seems to be effective against a variety of different baselines.

### Weaknesses
In the presentation of the algorithm, it is not clear which metric is precisely used to rank the importance of the single matrices. I would suggest that the authors include a more precise discussion about this in the revised version.

The proposed method is not able to significantly reduce the number of trainable parameters with respect to other methods like AdaLoRA or GeoLoRA [1]. It is also not clear how the global budget is allocated and kept under the maximal one, as the effect of inflating or not in terms of parameters depends on the layer one is considering. 

[1] S. Schotthoefer et al., GeoLoRA: Geometric integration for parameter-efficient fine-tuning, ICLR 2025.

### Questions
1. From what I understood and from the numerical experiments, the proposed method should not be able to significantly reduce the number of trainable parameters. Is this correct?

For the rest, I would appreciate a comment from the authors about the "weaknesses" section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces FlexLoRA, a method for dynamically adjusting LoRA rank across layers using a matrix-level spectral entropy score. It periodically prunes low-entropy matrices and expands high-entropy ones while keeping a global parameter budget. New directions are added with zero-impact initialization, so the model output is not disturbed at the moment of expansion. Experiments on NLP and vision benchmarks show consistent, if sometimes modest, gains over standard LoRA and prior adaptive methods. Ablations suggest that the entropy score, bidirectional allocation, and the initialization choice each contribute to the improvements.

### Strengths
1. A simple, matrix-level entropy score guides where to prune and where to add rank, avoiding noisy element-wise heuristics.
2. True bidirectional allocation with zero-impact initialization lets the model expand capacity safely while staying within a budget.
3. Strong, cross-domain results (NLP and vision) with clear ablations show each component—entropy, bidirectionality, initialization—adds value.

### Weaknesses
1. Entropy ignores magnitude (energy).
The spectral‑entropy score is scale‑invariant: a matrix with very small Λ but uniform spread can have high entropy and thus be prioritized for expansion, even if its absolute contribution is negligible. The paper briefly compares against Frobenius/nuclear norms (Table 4), but does not explore combined criteria (e.g., entropy × energy) or energy‑gated expansion. This is a conceptual gap given the stated goal of measuring “importance.”
2. Baseline anomalies & reporting details.
In Table 1, LoRA’s MRPC (68.4 Acc) and QQP (63.1 Acc) are unusually low for modern setups, reducing the average (81.7) and potentially inflating relative gains. GLUE typically reports MRPC/QQP F1 scores alongside, or instead of, accuracy. Please clarify the metrics, seeds, and hyperparameters per task, and include mean ± std across multiple seeds.
3. Cost/overhead not quantified.
Maintaining orthogonality (Eq. 2) and dynamically changing parameter shapes may incur compute/memory overhead vs. standard LoRA. No wall‑clock or throughput numbers are provided, especially on large LLM runs.
4. Limited theoretical justification linking entropy to “importance.”
Appendix B argues that the least singular value contributes least to entropy under certain conditions, which supports the pruning rule, but there’s no theoretical link to downstream loss reduction or generalization.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces an entropy-based metric to estimate the importance score of each weight-update matrix and proposes FlexLoRA, a method that dynamically adjusts the rank of LoRA modules by reducing the rank of those with lower importance scores while increasing that of the more important ones.

### Strengths
The framework is clear and intuitive. The authors conduct many experiments on GLUE, commonsense reasoning, and the Visual Task Adaptation Benchmarks, demonstrating the effectiveness of the proposed method.

### Weaknesses
1. The paper omits key system-level metrics such as peak GPU memory usage and FLOPs, which are essential for assessing efficiency.
2. The proposed entropy-guided importance metric is closely related to the effective rank concept from [1], limiting the novelty of the contribution.
3. The reported performance gains are marginal, particularly for large-scale models like LLaMA3-8B. Moreover, in Table 2, the number of trainable parameters in AdaLoRA is only about half that of the proposed model, making the comparison appear unfair.


[1] Olivier Roy and Martin Vetterli, The effective rank: A measure of effective dimensionality.

### Questions
Since the framework adjusts the rank dynamically during training, it would be valuable to report how this process affects the overall training time and computational cost.

### Soundness
2

### Presentation
2

### Contribution
2
