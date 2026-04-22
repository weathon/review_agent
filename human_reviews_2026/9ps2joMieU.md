# Drop or Merge? Hybrid MoE LLMs Compressors via Metric-Driven Adaptive Allocation

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Mixture-of-Experts (MoE) models enhance the scalability of large language models but encounter deployment challenges due to their vast parameter counts. Existing compression methods either drop experts entirely (discarding valuable knowledge) or merge experts (suffering from parameter conflicts), typically employing uniform strategies that ignore the heterogeneous specialization patterns across layers. In this paper, we propose DM-MoE, an adaptive Drop-then-Merge MoE compression framework to address these limitations. Our approach is motivated by two key observations: first, that eliminating a small number of truly redundant experts facilitates more effective subsequent merging, and second, that expert functional redundancy and behavioral similarity serve as reliable indicators for adaptive compression throughout MoE architectures. Building on these insights, we develop a two-stage compression: (1) In the dropping phase, we quantify layer redundancy via mutual information between expert outputs and formulate a constrained optimization problem to derive layer-wise dropping budgets, then select experts based on output impact assessment to retain those with high functional significance. (2) In the merging phase, we adaptively determine the number of expert groups per layer using behavioral diversity metrics, partition experts into functionally similar clusters via graph-based optimization, and merge them using importance-weighted averaging based on activation frequency and output deviation. Comprehensive evaluations on Mixtral, Qwen, DeepSeek and GPT-OSS MoE demonstrate that our  DM-MoE surpasses state-of-the-art methods across models and compression ratios. For Mixtral-8×7B, we retain 96.5\%/89.1\% of original performance at 25\%/50\% expert reduction. Code is available in the Appendix.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents DM-MoE, a two-stage compression framework for Mixture-of-Experts (MoE) language models. The proposed method first drops redundant experts based on information-theoretic redundancy metrics and then merges the remaining experts using a graph-based similarity clustering and importance-weighted parameter averaging. The framework adaptively allocates drop and merge budgets across layers via constrained optimization guided by layerwise mutual information and diversity measures. Experiments on several MoE LLMs (Mixtral, Qwen, DeepSeek, GPT-OSS) demonstrate consistent improvements over prior drop-only or merge-only approaches such as HC-SMoE and Frequency-drop.

### Strengths
- The motivation is reasonable and supported by analysis showing that pre-dropping redundant experts can mitigate parameter conflicts during merging.

- The methodology is clearly formulated, and the optimization-based allocation of drop/merge budgets is well presented.

- Experiments are comprehensive, covering multiple recent MoE models with extensive ablations and theoretical discussions.

- The paper is clearly written, easy to follow, and provides implementation details and pseudocode that enhance reproducibility.

### Weaknesses
- The core ideas of expert dropping and merging are well established in the literature. The main contribution lies in combining these two stages sequentially with layer-adaptive allocation, which is more of an engineering refinement than a fundamentally new concept.

- The reported throughput (Table 8) remains nearly unchanged after compression, indicating that the method primarily reduces memory footprint but not actual inference latency. The title and claims could better reflect this.

- The experiments focus on general reasoning and classification benchmarks. More generation-intensive or reasoning-heavy tasks (e.g., code generation, math reasoning, long-context modeling) are needed to validate general applicability.

- Although claimed efficient, the pairwise metric computation and graph partitioning per layer may become expensive for very large expert counts.

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
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DM-MoE, a two-stage drop-then-merge compression framework for MoE LLMs with layer-adaptive allocation: it (1) computes information-theoretic metrics to drop redundant experts, then (2) clusters and merges the remaining experts using similarity/diversity signals; allocation for both phases is solved via a lightweight constrained optimization, using a small calibration set. The method reports consistent gains across Mixtral, Qwen-MoE, DeepSeek-Lite, and GPT-OSS models.

### Strengths
* Reasonable, modular design. Clear two-phase pipeline with metric-driven, layer-adaptive allocation; algorithms and pseudocode are provided end-to-end. 

* Comprehensive main experiments/appendix. Broad model coverage, statistical significance, and practical notes on optimization cost; calibration/data choices are also specified.

### Weaknesses
* It's claimed that the proposed method "differs fundamentally by introducing a sequential drop-then-merge". This make reader expects the performance gain to be from the combination. However, Table 3 shows that purely uniform dropping is already quite good compared to other baselines. With adaptive allocation, the Drop-only averages 0.596 —i.e., the combined pipeline adds only ~0.5%–1.1% points on average. This supports the idea that drop-then-merge helps, but the incremental gain over a strong drop-only baseline is relatively small. This also somehow contradict with the statement in intro: "Performance collapse from complete dropping". It would be necessary to clearly isolate the gain from each component, as the dropping method is very similar to [1] with a smaller search space.

* It would be better to include more ablation studies on other model to clearly show the effectiveness of all components.

* A related work [2] should be mentioned:.

[1] Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models
[2] SlimMoE: Structured Compression of Large MoE Models via Expert Slimming and Distillation

### Questions
* How is the output-drop baseline implemented? How is it different with the Drop-only(uniform)?

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
The paper introduces DM-MoE, a hybrid, two-phase framework for compressing Mixture-of-Experts (MoE) LLMs without retraining. The authors argue that existing methods—dropping experts (losing knowledge) or merging experts (parameter conflicts)—are suboptimal, especially when applied uniformly. DM-MoE proposes a sequential "drop-then-merge" paradigm. Phase 1 adaptively drops experts, using mutual information to set layer-wise drop budgets via constrained optimization, and removing experts based on low output impact. Phase 2 adaptively merges the remaining experts, using behavioral diversity metrics to set layer-wise merge budgets and graph-based partitioning to cluster functionally similar experts. This approach claims to reduce parameter conflicts and achieve state-of-the-art performance, particularly at high compression ratios.

### Strengths
- The central "drop-then-merge" hypothesis—that strategically dropping redundant experts first reduces parameter conflicts and facilitates a more effective subsequent merge—is intuitive and empirically supported by the ablation study (Table 3).
- The framework's use of metric-driven, layer-wise adaptive allocation is a principled advancement over uniform compression strategies. It correctly identifies that expert redundancy varies by layer (Figure 1, right).
- The entire compression process is training-free and search-free, relying on efficient metric computation and constrained optimization (Table 12), making it a practical, one-shot solution.

### Weaknesses
- The paper claims to address "deployment challenges", yet the method provides zero inference latency reduction. Buried in Appendix C (Table 8), the results explicitly show that runtime (tokens/sec) is unchanged, as inference speed is dictated by the active top-k experts, not the total expert count. The contribution is thus limited only to memory and storage reduction, and framing it as a comprehensive deployment solution is misleading.
- The central "drop-then-merge" argument is weakly supported by its own ablation (Table 3). The full adaptive "Drop Merge" (0.601 avg) is only negligibly better than "Drop Only" (0.596) or "Merge Only" (0.590). The significant performance gain clearly comes from adaptive allocation (e.g., 0.601) versus uniform (0.584), not from the complex two-stage pipeline itself. The paper oversells the sequential combination.
- The perplexity (PPL) comparisons in Tables 9 and 10 are highly suspect. When compressing Qwen3-30B from 128 to 64 experts, the baseline HC-SMOE PPL catastrophically degrades to 72.33 (Wikitext-2) and 148.41 (C4), while DM-MoE achieves 17.79 and 32.28. This massive discrepancy suggests a buggy or misconfigured baseline, as the same method performs reasonably on downstream tasks (Table 1). This invalidates the claim of superior language modeling preservation.
- The choice of metrics (Mutual Information for dropping, "Diversity" for merging) feels arbitrary and justified post-hoc simply because they yielded the best results in the ablation (Table 5). The theoretical analysis (Appendix D) is trivial, merely formalizing the obvious concept that merging dissimilar experts is bad.

### Questions
1) Table 8 in the appendix indicates that the method does not improve inference latency (tokens/sec), as the active top-k experts dictate runtime. Could the authors clarify the precise deployment advantage? Is the contribution strictly limited to reducing memory/storage, and if so, should the claims about addressing "deployment challenges"  be reframed to be more specific?
2) The ablation in Table 3 shows that the full adaptive "Drop Merge" method (0.601 avg) is only negligibly better than adaptive "Drop Only" (0.596) or "Merge Only" (0.590). The primary gain appears to come from adaptive allocation, not the sequential pipeline. Can the authors provide statistical significance for this $\sim$0.5-point gap or further evidence to justify the added complexity of the "drop-then-merge" framework over a simpler, single-phase adaptive method?
3) The baseline (HC-SMOE) perplexity scores in Tables 9 and 10 appear anomalously high (e.g., 72.33 on Wikitext-2 and 148.41 on C4 ). This degradation is far more severe than what is shown in the downstream task results in Table 1. Could the authors please verify and confirm these baseline results? An experimental error here would significantly alter the paper's conclusions.
4) The method uses mutual information ("Inform") for the dropping phase allocation and a "Diversity" metric for the merging phase allocation (Table 5a). This metric-switching seems fine-tuned. Is there a deeper intuition for why the best metric to identify redundancy (for dropping) is different from the best metric to measure diversity (for merging)? Or is this selection purely based on these empirical ablation results?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DM-MoE, a two-stage compression framework for Mixture-of-Experts (MoE) language models. The authors propose to first drop redundant experts based on mutual information between expert outputs, and then merge the remaining ones using a graph-based clustering guided by behavioral similarity. Their experiments on Mixtral, Qwen, DeepSeek, and GPT-OSS models demonstrate improvements over existing MoE compressors such as Frequency-drop, HC-SMoE,  and MC-SMoE.

### Strengths
1. The “drop-then-merge” idea is intuitive and well-motivated. It effectively reduces expert conflict while preserving functional diversity.
The use of information redundancy and behavioral diversity allows for non-uniform compression, which is both data-driven and interpretable.

2. The experimental results are comprehensive. The authors evaluate across several MoE models and a variety of reasoning benchmarks, showing consistent gains under different levels of compression. The ablation studies are detailed. Each component (drop metric, merge metric, allocation strategy) is validated systematically.

3. The method is retraining-free, avoiding expensive fine-tuning or search.

### Weaknesses
1. The calibration data used for the Canonical Correlation Analysis is very limited (16 sequences), and no robustness analysis is provided. See Q1.

2. The approach involves pairwise CCA computations and graph partitioning across experts, which can be expensive for large MoE models. The paper reports roughly a “10-minute” runtime without showing how complexity scales with expert count or model size. This omission makes it hard to judge whether DM-MoE is feasible for massive production-scale MoEs (e.g., 128–256 experts per layer).

### Questions
1. How stable are the mutual information and diversity metrics across different calibration datasets or random seeds? Could domain shifts substantially change the layer allocation?

2. How does DM-MoE compare to standard weight compression methods (e.g., quantization, low-rank approximation, or pruning) in terms of performance–compression tradeoff?

3. Have you explored combining DM-MoE with quantization or other compression techniques? Would the two be additive or interfere?

### Soundness
3

### Presentation
3

### Contribution
3
