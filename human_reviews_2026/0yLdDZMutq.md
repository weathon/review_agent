# GVote: Adaptive Per-request KV-Cache Compression without Manually Setting Budget

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Large language models (LLMs) inference relies heavily on KV-caches to accelerate autoregressive decoding, but the resulting memory footprint grows rapidly with sequence length, posing significant efficiency challenges.
Current KV-cache compression methods suffer from a Procrustes' bed problem: they force diverse workloads into fixed compression ratios, leading to suboptimal resource allocation and inference performance. 
To this end, we present GVote, an adaptive per-request KV-cache compression scheme that eliminates manual budget specification while achieving superior accuracy-efficiency trade-offs. 
GVote operates on the principle that the important keys are the aggregation of keys required by future queries. 
Gvote predicts future query attention demands by Monte-Carlo style sampling potential queries and aggregating selected keys to determine the optimal cache budget without manual specification.
Experimental evaluation demonstrates GVote's effectiveness across multiple benchmarks, including GSM8K, RULER and Longbench. 
Compared to baselines, GVote exhibits 2$\times$ memory reduction while the accuracy maintains higher or comparable.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces GVote, a sampling-based method to determine the KV cache budget and the KV pairs to keep for efficient inference. With a couple minor hyperparameters, the authors show that GVote can automatically reduce memory and speed up inference on several tasks. While the algorithm seems sound, I mainly have some issues with the clarity and motivation.

### Strengths
1) Automatically determining the KV budget is an important problem, and this paper makes progress towards it.
2) Sizable memory reduction and better prefill latency scaling.

### Weaknesses
1) I think there is a lack of background/related work in the paper that makes the field seem too narrow. Given that the field of KV cache compression has matured over the past 2 years, there have been a lot more work than what the authors have included. For instance, many methods that use low-rank compression, sparse+low rank compression, or offloading KV caches to CPU are not touched upon, even though they all fall under the realm of KV caching algorithms. While the authors do not need to explicitly compare against these methods, I suggest a broader coverage of the literature.

2) The claim in Figure 2 that the curves look Gaussian is not convincing visually, since they mostly seem like simple unimodal distributions. A quantitative metric would make this claim seem less suspect.

3) Many (if not all) of the tasks are short generation tasks. To demonstrate the predictive power of GVote, it would better to show results on long generation tasks.

4) Various typos:
    - Line 129: backwards quotations
    - Line 351: KVCache should be two words
    - Line 365: We extensively tests -> We extensively test
    - Equation 1: Should the index for the keys and values go from 1 to t? KV pairs from current and previous tokens are used, not just previous tokens.

### Questions
1) Are models using chain of thought for GSM8K?

2) Is there a way to tune the dial to prioritize accuracy vs. efficiency?

3) Is there a way to constrain the efficiency? For example, if I want at least a 80% reduction in memory, is there a way to enforce this?

4) Do you have throughput metrics? Or does this optimize only for latency?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces an adaptive KV-cache compression scheme that operates per request. By moving beyond fixed-budget compression and choosing the compression ratio dynamically for each input, the approach targets lower memory and more efficient deployment without sacrificing accuracy.

### Strengths
* The motivation is clear and well argued, and the experiments are thorough and well documented.
* The distribution-aware synthetic-query mechanism is sound and convincing.

### Weaknesses
* The reliance on K from the last ground-truth query to stabilize selection is intuitive but under-justified. Please add theory/ablations showing when this helps vs. hurts.
* Figure 6 needs methodological clarity. If results are matched by accuracy, each method’s retained KV size / compression ratio and the corresponding latency/memory should be reported to ensure fair comparability.
* Minor: duplicated sentence at L256–L257.

### Questions
* Do you have empirical or theoretical evidence to justify using the K from the last ground-truth query?
* In Fig. 6, at the matched-accuracy operating point, what are each method’s KV-cache size (or compression ratio) and latency?

### Soundness
3

### Presentation
2

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
This paper addresses the limitations of fixed-budget KV-cache compression, where a single preset budget is applied to all requests regardless of their contextual complexity. This uniform setting can either hurt accuracy on difficult reasoning tasks or waste memory on simpler ones.

To resolve this, the authors propose GVote, an adaptive per-request KV-cache compression scheme that automatically determines the necessary cache budget with low overhead.
GVote is built on the observation that hidden states across the sequence approximately follow a Gaussian distribution. Leveraging this, the method samples synthetic future queries via Gaussian perturbation, aggregates their attention over existing keys through a Monte-Carlo voting procedure, and retains the union of key positions deemed important.

As a result, complex workloads (e.g., mathematical reasoning) naturally receive larger cache budgets, whereas simpler tasks (e.g., QA) are assigned smaller ones, enabling better accuracy–efficiency trade-offs. Experiments on Qwen2.5-7B-Instruct as well as Llama-based models across GSM8K, RULER, and LongBench demonstrate that GVote achieves higher or comparable accuracy at similar or lower memory usage compared to static-budget baselines.

### Strengths
* The paper clearly identifies the inefficiency of fixed-budget KV-cache compression under heterogeneous workloads and proposes a practical per-request budgeting mechanism.
* By combining Gaussian-based synthetic query generation with Monte Carlo voting to approximate future queries, the method offers a technically interesting and novel bottom-up perspective.
* The method demonstrates consistent accuracy–memory improvements across diverse benchmarks (e.g., GSM8K, RULER, LongBench) and model families (e.g., Qwen, Llama), supporting its generality.
* By dynamically estimating per-request budgets, the approach achieves higher accuracy at comparable memory usage, or similar accuracy with lower memory, compared to fixed-budget baselines.
* Sensitivity analyses on hyperparameters (e.g., p_nuc , S) provide useful guidance for applying the method in practice.

### Weaknesses
* The assumption that hidden states follow a Gaussian distribution is only weakly supported; stronger theoretical justification or broader empirical evidence is needed to establish generality.
* The paper provides limited analysis of failure cases in which synthetic queries fail to approximate future queries, leaving the robustness of the proxy insufficiently explored.
* Although the method claims to eliminate manual budget specification, several key hyperparameters still require tuning.
* Comparisons against recent adaptive KV-cache compression approaches are insufficient, making it difficult to assess the precise novelty of the contribution.
* The paper does not adequately discuss how the proposed non-uniform KV layout would integrate with real-world inference systems (e.g., paged attention in vLLM), where memory layout constraints and paging mechanisms may complicate deployment.
* The method focuses on single-request contexts and does not address how KV-cache should be maintained, shared, or incrementally updated across multi-turn dialogue scenarios, which are central in practical LLM applications.

### Questions
* Could you provide broader empirical evidence showing that the Gaussian assumption for hidden states holds consistently across different models, layers, and task types?
* In cases where synthetic queries fail to approximate future attention well, what conditions cause such failures, and how do they impact model performance?
* Could you provide a quantitative evaluation of run-to-run variance to assess the stability of the Monte-Carlo sampling process?
* How would the proposed non-uniform KV layout interact with paged-attention–based systems such as vLLM?
* How should the method maintain and update KV-cache across multi-turn dialogue scenarios, which are common in real LLM applications?
* Since the proposed method leverages hidden-state statistics rather than actual future tokens, would it be feasible to pre-compute a reusable pool of synthetic queries instead of re-sampling for every request?

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
This paper proposes GVote, an adaptive KV Cache compression scheme that avoids manual budget setting.
GVote relies on LLM hidden states’ Gaussian distribution: it uses Monte-Carlo sampling to generate synthetic future queries, aggregates their needed keys to dynamically determine optimal cache budgets, and forms a keep-set via voting.
Experiments  show GVote reduces VRAM while keeping accuracy high or comparable to baselines (StreamLLM, SnapKV, AdaKV). It also has low generation latency and compatibility with modern kernels like FlashAttention.

### Strengths
+ The proposed method maintains its advantage across different architectures (LLaMA, Qwen) and various model sizes (3B-14B), demonstrating that it is not dependent on a specific model structure and has a broad range of applicability.

### Weaknesses
+ In the discussion of "ARE SYNTHESIZED QUERIES GOOD PROXIES?", there is no baseline for comparison. For instance, if tokens from the Prefix and Observation window, as used in SnapKV, are employed as proxies, what will the attention overlap be? This comparison is necessary to evaluate the effectiveness of synthesized queries as proxies.

+ The experimental results in the paper are questionable. According to the reported results, StreamLLM's performance is only slightly worse than SnapKV and AdaKV. However, based on the results in the AdaKV paper, StreamLLM’s performance is significantly worse than the other two methods. This discrepancy is reasonable, as StreamLLM only retains attention sinks and short-term information, which likely limits its overall performance compared to the other approaches.


+ The evaluation datasets used in the paper are not clearly specified. For example, the paper mentions evaluation on Longbench, but Longbench includes 21 different datasets, each with different metrics (such as Acc, F1, Rouge-L, Edit-Sim, etc.). The paper does not explain how the reported "accuracy" was derived.

+ Adaptive adjustment of the budget is not necessarily an advantage, as it also means the budget becomes uncontrollable. In practical scenarios, this could lead to unexpected OOM issues.

+ The paper has not been proofread, and there are numerous issues with the citation formatting.

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
2
