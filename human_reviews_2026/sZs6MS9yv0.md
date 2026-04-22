# HS-SFT: Hybrid Sparse Supervised Fine-tuning for Offline LLM KV Cache Eviction

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Long-context LLMs are constrained by the linear growth of key–value (KV) caches during autoregressive decoding, which incurs pronounced latency and memory overhead. KV eviction mitigates this issue, with existing efforts fall into offline policies with fixed eviction patterns and online policies that adaptively discard cache based on attention scores. While online eviction typically preserves accuracy under standard benchmarks, its performance can collapse in practical multi-turn dialogue scenarios where the query positions vary, and integration with pre-fill acceleration remains challenging. In contrast, offline eviction is infrastructure-friendly and generalizable but commonly sacrifices more accuracy. In this paper, we explore Supervised Fine-Tuning (SFT) for offline KV eviction and demonstrate its efficacy as a simple and powerful alternative to the design of complex online eviction metrics. We further propose Hybrid Sparse Supervised Fine-Tuning (HS-SFT) to explore the optimal offline design of KV eviction within SFT. In particular, HS-SFT employs a straight-through estimator to learn discrete local-window allocations of streaming heads across layers with budget-aware balancing loss, such that under high compression ratios—where dense-head capacity is constrained—the budget can be more effectively skewed to capture critical information.  Across extensive evaluations on a wide array of LLMs and long-context tasks, HS-SFT delivers substantial performance gains over state-of-the-art eviction baselines. For example, with fewer than 4 hours of SFT of LLaMA-3-8B-1048K using a single 8-GPU node, HS-SFT achieves 5.86% and 38.3% higher average accuracy than Duo-attn on Longbench and Ruler-16K at 10% KV budget, respectively. These results position training-aware offline eviction—achieved with simple SFT—as an effective and practical path to scalable long-context inference. Code will be available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose HS-SFT, a method to enhance model quality for KV cache eviction. They employ supervised fine-tuning (SFT) to adapt the model to the StreamingLLM framework and utilize the straight-through estimator (STE) for discrete budget routing, accommodating varying sparsity across attention heads. Experimental results demonstrate that the proposed approach improves generation quality on Llama2 and Llama3 models.

### Strengths
The method is evaluated on LongBench, RULER, and the NAIH benchmark, showing consistent improvements in generation quality for both Llama2 and Llama3 models.

### Weaknesses
- Limited novelty: The idea of leveraging head-wise sparsity and SFT for adaptation is not sufficiently innovative, as similar concepts have been explored in prior work.
- Insufficient technical contribution: Learning-base head-specific sparsity patterns has been widely studied in existing literature (e.g. HeadKV[1]), and the proposed method does not significantly advance this direction.
- Inadequate experimental rigor: The evaluation is limited to Llama2 and a non-official version of Llama3. Broader validation on other model families (e.g. Qwen-32B) and larger-scale models is necessary to strengthen the claims.
- Lacking efficiency analysis: No comparison with online-base methods (Like SnapKV) in terms of inference speed is provided. The potential computational imbalance introduced by dense heads may undermine practical efficiency.

[1] Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning (ICLR2025)

### Questions
See weaknesses, and:

- Why were Llama2 and a non-official version of Llama3 chosen as the primary evaluation models, rather than other widely-used or more powerful models?
- In Table 2, why was Continued Pretraining (CP) conducted on the RedPajama dataset instead of UltraChat? Could there be potential data contamination issues with UltraChat?

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
This paper addresses the performance-deployability trade-off between online and offline KV cache eviction methods. The authors first demonstrate that simple SFT can substantially close the accuracy gap between efficient, infrastructure-friendly offline policies (e.g., StreamingLLM) and heuristic-based online policies (e.g., SnapKV).

Building on this, the paper proposes Hybrid Sparse SFT, a method to learn an optimal offline eviction strategy. HS-SFT divides heads into a fixed dense set and a sparse set. For the sparse heads, it uses a Straight-Through Estimator to learn a discrete, layer-wise local window budget from a set of candidates . This selection is regularized by a budget-aware balancing loss (kv div.) to favor smaller budgets . At inference, this learned policy is a fixed, offline pattern. Experiments show HS-SFT significantly outperforms SOTA eviction baselines, particularly at high sparsity.

### Strengths
- The paper's primary contribution is the highly practical insight that lightweight SFT alone can adapt an LLM to a fixed, offline eviction policy, making it competitive with complex online methods . This is a interesting finding for real-world deployment, as offline policies are compatible with prefill acceleration and are more robust in general-purpose settings (e.g., multi-turn chat) .
- The HS-SFT method itself is a interesting training-aware approach to this problem. Instead of relying on a fixed, hand-crafted offline pattern, it learns a more flexible, layer-specific offline policy. The use of an STE to learn discrete local window budgets combined with a KL-based loss  is a sound and effective mechanism.
- The experimental results are strong, showing clear SOTA performance over other offline and online methods, especially at aggressive, low-budget (e.g., 10%) eviction rates .

### Weaknesses
- HS-SFT initializes its dense head selection using the logic from a prior method, Duo-Attn. The sensitivity to this initialization is not ablated, making it unclear how critical this heuristic is. 
- The learned budget is layer-wise, not head-wise, a trade-off made for inference efficiency. This forces all sparse heads in a layer to share the same local window size, which may be suboptimal. 
- The SFT is performed on a general-purpose dataset. While this shows generalizability, it is not compared against task-specific SFT.

### Questions
- How dependent is HS-SFT on the Duo-Attn initialization for dense heads? What is the performance if the $\alpha$ dense heads are selected randomly before SFT?
- While head-wise selection is noted as future work, an ablation comparing layer-wise vs. head-wise budget learning would be valuable to quantify the performance/efficiency trade-off.
- Does the SFT data domain significantly impact the learned policy? For instance, does fine-tuning on data more representative of the downstream benchmarks (e.g., summarization, retrieval) yield a more effective eviction strategy?

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
3

### Summary
This paper addresses the critical challenge of KV cache efficiency in long-context LLMs by proposing Hybrid Sparse Supervised Fine-Tuning (HS-SFT), an offline eviction strategy that unifies the deployment simplicity of offline methods and the performance retention of online approaches. By leveraging supervised fine-tuning (SFT) and learning layer-wise adaptive local window sizes via a straight-through estimator (STE) with budget-aware balance loss, HS-SFT achieves good performance across multiple benchmarks while maintaining inference efficiency.

### Strengths
1. This paper targets a promising direction, proposes fine-tuning sparse attention with an elaborated algorithm design.
2. Strong evaluation: cover multiple LLMs (Llama-2-7B-32K, Llama-3-8B with 262K/1048K context) and benchmarks (LongBench, Ruler-16K, NIAH), demonstrating cross-model and cross-task generalization.
3. This paper is well-structured.

### Weaknesses
1. Lack of several baselines of online eviction alternatives or query-aware sparse: H2O[1], Quest[2], HShare[3].
2. The paper mentions online evictions’ weakness in multi-turn dialogue, but does not provide explicit evaluation results for this scenario.
3. The paper uses 1B tokens from UltraChat for SFT but does not explore how data size or domain affects performance, simply following another paper. 

[1] Zhang, Zhenyu, et al. "H2o: Heavy-hitter oracle for efficient generative inference of large language models."
[2] Tang, Jiaming, et al. "Quest: Query-aware sparsity for efficient long-context llm inference." 
[3] Wu, Huaijin, et al. "HShare: Fast LLM decoding by hierarchical key-value sharing."

### Questions
1. In Section 2.3, you initialize dense heads using Duo-Attn’s logit map and note that updating dense heads during SFT yields significant gains. However, you do not explore alternative dense head initialization strategies. Could you explain why Duo-Attn’s initialization is optimal, and have you tested whether different initialization methods affect HS-SFT’s convergence speed or final performance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
HS-SFT targets the linear growth of LLM KV caches by improving offline eviction. The paper states that, unlike online attention-score–based eviction, offline schemes remain compatible with prefill acceleration and are more robust to query positions in multi-turn dialogue. The paper first shows that lightweight supervised fine-tuning (SFT) alone largely closes the gap between offline methods (StreamingLLM) and online methods (SnapKV), indicating models can learn to compensate for eviction-induced degradation. 
It then introduces Hybrid Sparse SFT (HS-SFT), which learns discrete, layer-wise local-window budgets for streaming heads via a straight-through estimator and a budget-aware balancing (KL) loss. This allows the cache budget to be skewed toward layers that need more long-range aggregation. At training time a fixed fraction of dense heads is kept while per-layer budget logits are learned from a small candidate se. At inference time, these learned budgets are monotonically recsaled to meet any target sparsity, preserving a offline execution path that remains compatible with prefill acceleration.

### Strengths
The ideas in the paper, in my opinion, is practical and well-designed. It keeps an offline eviction path compatible with prefill acceleration. The paper shows that lightweight SFT can recover much of the offline–online accuracy gap. The paper also adds an STE-based, budget-aware KL mechanism to learn layer-wise local-window budgets.

### Weaknesses
HS-SFT’s routing is layer-level, which the authors note may leave performance on the table. Their comparisons with OSS-style hybrids are limited to the SFT regime and evaluating during pretraining is left as future work. Additionally, the paper misses comparison against other efficient inference time solutions like MorphKV.

### Questions
1. How robust are the learned layer-wise budgets to training-distribution shifts and to the choice/size of the budget set B, since your ablation suggests larger sets didn’t help?

2. Would a head-wise router (instead of layer-level) materially improve your trade-offs, and why is there no comparison to MorphKV? Was it excluded because you restrict to offline, prefill-compatible baselines or do you expect HS-SFT to dominate under those constraints?

### Soundness
3

### Presentation
3

### Contribution
2
