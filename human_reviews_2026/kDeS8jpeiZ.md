# SpareTrain: Fault-Tolerant LLM Training via Low-Cost Dual Modular Redundancy

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Dual Modular Redundancy (DMR) is a highly effective mechanism for detecting silent data corruption (SDC)—a critical reliability concern in large language model (LLM) training—by executing each operation twice. However, its high computation overhead has prevented practical deployment at scale. In this paper, we present SpareTrain, an LLM training system that achieves complete DMR with minimal overhead by repurposing the activation checkpointing mechanism and exploiting idle GPU time. Evaluations on up to 32 H200 GPUs show that SpareTrain improves throughput by 12–35\% over naive DMR, corresponding to only 3–14\% overhead compared to unprotected training, while maintaining full DMR error detection capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses silent data corruption detection by integrating Piggyback-DMR and Deferred-DMR into a novel framework called SpareTrain.

### Strengths
1. The proposed method is applied to high-performance GPUs in a large-scale cluster.
2. The proposed method can be applied to both MoE and non-MoE models.
3. The proposed method introduces only a 3–14% overhead compared to baseline training without SDC protection.

### Weaknesses
1. D-DMR appears to work only with pipeline parallelism, which may limit its applicability.
2. It seems that the proposed method relies on high-performance GPUs, where computation time is significantly faster than communication time. For GPUs like the H20, the effectiveness of the proposed method remains uncertain.
3. We seem to observe a contradiction: we use pipeline parallelism due to memory constraints, yet the proposed method requires additional memory. Does this imply that, with given memory, the baseline could use smaller pipeline parallelism to achieve higher throughput? This seems like a better baseline.

### Questions
In Piggyback-DMR, could some internal SDCs be overlooked, since it only checks the input and output of Activation Gradient Checkpointing?

### Soundness
3

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
4

### Summary
This paper presents SpareTrain, a system for fault-tolerant LLM training that reduces the cost of dual modular redundancy (DMR). It introduces two mechanisms: Piggyback-DMR (P-DMR), which reuses activation recomputation from checkpointing for validation, and Deferred-DMR (D-DMR), which exploits idle GPU time (e.g., pipeline bubbles) to execute redundant checks. A planner coordinates these modes to balance reliability and efficiency. Experiments on up to 32 H200 NPUs show up to 35% speedup over naïve DMR while maintaining full coverage claims. However, the paper provides limited validation of correctness and scalability.

### Strengths
- Addresses an important reliability problem in large-scale LLM training, where silent hardware errors can silently corrupt long-running training.
- The idea of reusing existing computation (recompute or idle slots) for redundancy is intuitive and practically relevant, reducing the prohibitive cost of full DMR.
- The paper demonstrates a non-trivial engineering effort, with experiments on real Ascend NPU clusters up to 8192 devices.
- The proposed planner that automatically assigns P-DMR, D-DMR, or Full-DMR modes based on profiling data is conceptually elegant and system-aware.
- Results show that SpareTrain can significantly lower redundancy overhead while maintaining theoretical detection coverage.

### Weaknesses
Although the paper presents a conceptually reasonable design, the overall analytical depth and experimental coverage are limited, leading to weak interpretability of the proposed DMR planner.

First, the relationship between the DMR planning mechanism and the underlying parallel configuration (such as tensor, pipeline, and expert parallelism) is never formally analyzed.
The effectiveness of both P-DMR and D-DMR clearly depends on these configurations—pipeline depth determines bubble size, tensor-parallel degree affects collective communication cost, and expert parallelism introduces dynamic execution patterns.
However, the paper only illustrates a few heuristic cases and does not provide any mathematical modeling, cost function, or decision boundary describing when a mode switch (e.g., from P-DMR to naïve DMR) should occur.
As a result, the planner behaves like a hand-tuned heuristic rather than a principled optimization framework.

Second, the interaction between checkpointing frequency and P-DMR coverage is completely unexamined.
Since activation checkpointing recomputes every n layers, the interval n directly controls both recomputation cost and P-DMR reuse potential.
Without quantitative analysis of how n influences memory usage, recomputation overlap, or fault detection coverage, it is unclear how the proposed planner would adapt under different checkpoint configurations.

Third, the experimental evaluation is overly high-level.
It reports only end-to-end throughput and speedup ratios, without any breakdown of how many operators are assigned to P-DMR, D-DMR, or naïve DMR modes under different parallel setups.
There is no analysis of idle-window utilization, checkpoint overlap ratio, or the planner’s sensitivity to communication intensity.
Consequently, the claimed throughput improvement lacks interpretability and cannot be correlated with the underlying design decisions.

Finally, the planner’s objective and constraints are only described qualitatively (“maximize reuse given time/memory slack”) without any explicit cost model or optimization formulation.
This makes it difficult to reason about convergence, scalability, or generalization to unseen workloads.

### Questions
- How does the proposed planner adapt to different parallel configurations (e.g., TP, PP, EP) that directly affect communication cost and idle-window size?
- How does the checkpointing frequency (n) influence P-DMR coverage and overall verification cost?
- Is there a quantitative cost model or optimization formulation guiding the planner’s decisions, or are all heuristics empirically tuned?
- Can the authors provide the distribution of P-DMR, D-DMR, and naïve DMR operators under different configurations to explain the observed throughput changes?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SpareTrain, a system designed to provide complete, low-cost fault tolerance for Large Language Model (LLM) training. The primary goal is to detect Silent Data Corruptions (SDCs) using Dual Modular Redundancy (DMR)—which traditionally incurs a prohibitive 100% compute overhead—at a minimal performance cost. The core contribution of SpareTrain is a set of two complementary strategies that exploit existing characteristics of modern LLM training:

1. Piggyback-DMR (P-DMR): This technique cleverly repurposes the inherent computational redundancy from Activation Checkpointing (AC). The forward pass serves as the primary execution, and the re-computation of activations during the backward pass (which AC necessitates for memory savings) is used as the checker execution.

2. Deferred-DMR (D-DMR): This technique leverages the significant GPU idle time present in large-scale distributed training, which arises from communication overhead (e.g., in Pipeline Parallelism (PP) and Tensor Parallelism (TP)) and pipeline bubbles. SpareTrain defers the checker execution of an operation into these idle periods, effectively hiding its computational latency.

These strategies are orchestrated by a three-phase planner that statically (for dense models) or dynamically (for MoE models) assigns operations to P-DMR, coarse-grained D-DMR (for PP bubbles), or fine-grained D-DMR (for TP/EP communication windows).

### Strengths
1. The core strategies are both insightful and effective. The P-DMR concept is particularly creative. Identifying that activation checkpointing already provides a "free" source of redundancy for fault tolerance is a non-trivial and elegant insight. It turns a memory-saving technique into a reliability feature.

2. The paper is exceptionally well-written and easy to follow, especially for a complex systems paper. The concepts of P-DMR and D-DMR are introduced intuitively, and the figures provide clear visual explanations of the baseline, the problem, and the proposed solution's mechanisms.

3. The work is evaluated on relevant, large-scale hardware (H200s) and state-of-the-art models (Llama-3, Mistral-Large, and an MoE model), not just on small-scale toy problems.

### Weaknesses
See Questions

### Questions
1. Long-term Viability of D-DMR: As distributed training frameworks get better at hiding communication latency and reducing pipeline bubbles, the GPU idle time that D-DMR relies on will decrease. How do you see the effectiveness of SpareTrain changing in a future system that has near-perfect overlap and minimal idle time? Does the benefit of SpareTrain then rely almost entirely on P-DMR?

2. In your experiments on Llama-3 and Mistral-Large (Section 5.1, Figure 7), how frequently did the planner's exception case trigger? That is, what percentage of AC segments were assigned to Naïve-DMR instead of P-DMR because the condition ($2 \times \text{Recompute} < \text{Additional Comm.}$) was met? This would help in understanding the practical importance of P-DMR.

3. Does Silent Data Corruption significantly hurt model quality? Please give us some evidence.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Naive DMR doubles computation to detect silent data corruption. This paper SpareTrain overcomes this through two key innovations: Piggyback-DMR using activation checkpointing, and Deferred-DMR Utilizing idle GPU time and perform checker executions asynchronously. This hides redundancy latency behind existing idle periods, minimizing impact on training throughput. SpareTrain includes both offline and online planners that coordinate P-DMR and D-DMR to balance memory, timing, and redundancy coverage.

### Strengths
good originality:
Reusing activation checkpoint recomputation is an creative idea and it works well with existing AC APIs. Deferred-DMR using idle GPU time is also intuitive. The combination of static and dynamic DMR planning introduces a new layer of runtime adaptivity for reliability in distributed training. It's a practical system for achieving full Dual Modular Redundancy (DMR)

good quality:
The paper showcased integration with Llama-3, Mistral-Large, and Llama-4-Scout and up to 32 GPUs. It carefully analyzes overhead sources, memory trade-offs, and timing windows, and conducts ablation studies showing contributions of each design phase. Evaluations demonstrate up to 35% faster than naive DMR with full detection

good Clarity
Checkpointing vs. P-DMR, DMR scheduling diagrams are very helpful to communicate the timing and memory concepts. Despite system-level complexity, the narrative maintains good readability, with consistent terminology and careful explanation of assumptions and constraints.

fair significance
Silent data corruption is becoming a critical reliability issue when scaling to hundreds of thousands of GPUs. Making full DMR feasible at low cost has immediate industrial relevance. SpareTrain’s integration with PyTorch and TorchTitan indicates it could be adopted with minimal workflow disruption

### Weaknesses
While the paper’s DMR planner (static + dynamic) is a well-engineered mechanism for allocating operations across Piggyback-DMR and Deferred-DMR, its soundness depends on the assumption that the profiled execution characteristics remain stable throughout training. This assumption is not always valid. The static planner bases its schedule on a short profiling phase that measures activation checkpointing configuration, operation latencies, and communication windows (GPU idle times). However, these metrics can change substantially across iterations due to variations in 
* pipeline bubble patterns: which depend on load imbalance, microbatch size, and runtime scheduling
* activation checkpointing overhead: which fluctuates with adaptive memory pressure, gradient accumulation steps, or sequence length changes;

It is less clear what's the net win of full DMR over checkpoint rollbacks. From a cost–benefit standpoint, many production training pipelines might prefer to tolerate occasional checkpoint rollbacks over continuously paying 15% runtime overhead for protection against rare SDCs. Unless the paper provides quantitative evidence that SDCs cause non-recoverable or undetectable model degradation despite checkpointing, the necessity and urgency of system-wide DMR protection remain open to question.

### Questions
Would love to see a comparison with checkpoint–resume solution. It remains unclear how frequently such errors materially affect final model quality compared to the dominant sources of variance (e.g., data order, learning rate tuning). When an error or hardware failure occurs, training can resume from the last saved state

### Soundness
2

### Presentation
3

### Contribution
3
