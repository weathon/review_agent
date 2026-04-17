# Aladdin: Joint Placement and Scaling for SLO-Aware LLM Serving

- Decision: Reject
- Scores: 4, 6, 4, 4, 4, 8

## Abstract
The demand for large language model (LLM) inference is gradually dominating artificial intelligence workloads, creating an urgent need for cost-efficient inference serving. While prior work focuses on single-worker optimization, it often overlooks cluster-level coordination across both queries and computing resources. Scheduling requests without considering their uncertainty can lead to SLO violations or overprovisioning, resulting in excessive cost.

In this paper, we present Aladdin, a scheduler that co-adaptively places inference queries and scales computing resources under probabilistic SLO constraints. Aladdin explicitly models request-level uncertainty through stage-wise latency distributions, and places queries based on their statistical profiles to maximize per-worker utilization. To improve robustness and cost-efficiency, we design a flexible constraint interface that supports distribution-aware tail modeling and risk-adjusted capacity allocation. Experiments show that Aladdin reduces serving cost by up to 71\% under the same SLO level compared to standard baselines, which can translate to millions of dollars in annual savings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents ALADDIN, a co-adaptive scheduler for large-scale LLM inference that optimizes query placement under probabilistic SLO constraints. The method uses multi-dimensional bin-packing problem, and introduces a heuristic for online placement with some theoretical guarantees. Experiments on A100/V100 clusters show significant cost reduction (up to 71%) compared to vLLM and some other baselines.

If the questions and weaknesses are properly addressed in the rebuttal, I can increase my score.

### Strengths
- The problem is important and interesting.
- The paper develops a principled formulation with explicit latency modeling, probabilistic SLOs, and analytical tail guarantees.
- The empirical study includes real testbed experiments and large-scale simulations.

### Weaknesses
- Novelty is quite limited as many building blocks (bin-packing, SLO-aware scheduling, KV cache modeling) are adaptations of existing ideas. 
- Experimental baselines can be improved by additional comparisons with concurrent schedulers such as Exegpt or Sarathi-Serve.
- Evaluation focuses on synthetic workloads; no production-scale or multi-tenant trace is shown, which limits real-world validation.
- Some ablation studies on the probabilistic models should be added.

### Questions
- What is the key novelty of the work beyond integrating well-known methods as mentioned above?
- Are there new proof techniques used for the theoretical results or off-the-shelf large deviation bounds?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes ALADDIN, a cluster-level scheduler that jointly (i) chooses worker configurations (GPU count per worker via tensor parallelism) and (ii) places requests online to minimize GPU cost under SLOs. It argues that existing work optimizes single workers or uses SLOs only implicitly, missing cluster-wide coordination and uncertainty in output length. ALADDIN’s contributes stage-wise models.  The stage-wise models are simple, empirically validated, and lead to tractable constraints; the tail-risk formulation connects engineering knobs to SLOs. However, output-length prediction is deliberately naïve (history-based), and true end-to-end SLO guarantees still hinge on θ/γ calibration and the fit quality of per-stage models; some validations rely on simulation rather than only live clusters. The paper is clearly structured with helpful figures/algorithms and situates against vLLM/DistServe/Splitwise lines of work.

### Strengths
1. Joint optimization of placement + scaling with explicit SLO constraints (TTFT/ATGT) at cluster level; complements single-worker optimizations.  Lightweight, validated models (prefill, decode, KV) that are accurate enough for online use.
2. Formal tail-risk view (chance constraints with worker budgets), linking utilization choices to p-tail guarantees.  
3. Higher SLO attainment and p99 ATGT reductions on A100/V100; large simulated GPU savings. Error-aware rebalancing and distributed scheduling extensions for reality.  Clear articulation of why JSQ/Power-of-Two can be suboptimal under KV and decode constraints.

### Weaknesses
1. Uses a naïve, unbiased length predictor; avoids LLM-based predictors due to cost/bias. Accuracy under non-stationary workloads (topic/product shifts) is unclear; guarantees depend on θ/γ tuning and rebalancing efficacy. The live experiments are single-node, small clusters (4×GPUs); the strongest cost savings are in simulation. Cross-region/multi-node contention and network tails aren’t empirically covered.  
2. ATGT is motivated as QoE-aligned, but there’s no user-level study or TTFT/ATGT tradeoff curves; reporting focuses on p99 ATGT and SLO attainment, not TTFT p-tails under load.  Comparisons use JSQ/Power-of-Two; no head-to-head vs. recent SLO-aware/constraint-aware schedulers (e.g., ExeGPT) under identical setups.  
3. For split-phase, decode dominates and results are partly simulated; interaction with paged attention, KV streaming, or chunked prefill in multi-node deployments would strengthen external validity.  
4. Chance-constraint section is principled, but Σ estimation, shrinkage, and βj allocation procedures are not fully specified empirically; how to tune θ/γ to meet a global δ in practice is left abstract.

### Questions
1. How do you estimate Σ online and allocate βj across workers? Can you show a closed-loop controller that tunes θ/γ to hit a target δ (e.g., p99 or p99.9) over time?   
2. When output-length distributions shift (events, product launches), how quickly do models and θ/γ recover?
3. It's reasonable that authors do not include multi-node experiments due to budget issues. But it's interesting to analyze what happens with tens to hundreds of GPUs across nodes with NVLink/NVSwitch/PCIe mixes? 
4. Could the authors report joint TTFT/ATGT tail distributions and DET/ROC-style curves for SLO attainment to justify ATGT vs. TBT and to guide SLO setting?
5. With paged attention / KV streaming and chunked prefill, how do the constraints (Eq. 2–4) shift, and does the heuristic still remain near-optimal? Empirical ablations would help.  
6. What’s the steady-state CPU overhead for scheduling and trace collection at 100–500 req/s? Any tail spikes from scheduling GC or model-update steps?

### Soundness
3

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
The paper introduces ALADDIN, a cluster-level scheduler that jointly optimizes request placement and resource scaling for LLM serving under probabilistic SLOs (covering TTFT and a user-centric decode metric ATGT). It builds simple stage-wise models for prefill/decode latency and KV-cache usage, casts online placement as a multi-dimensional bin packing / MIP with a fast heuristic, adds chance-constrained tail guarantees, and includes error-aware rebalancing plus a distributed variant for high QPS. On A100/V100 with LLaMA-2 7B/13B/70B, ALADDIN reports <10% modeling error and up to 71% GPU cost reduction (and up to 60% on split-phase decode) versus standard baselines.

### Strengths
Originality. Elevates LLM serving from single-node tweaks to a cluster-level joint problem (request placement + resource scaling) with probabilistic SLOs (TTFT/ATGT); a fresh and meaningful problem framing.

Technical quality. Clear stage-wise (prefill/decode) linear models with KV-cache accounting; formulates online placement as multi-dimensional bin packing with a practical heuristic and chance-constrained tail guarantees; includes error-aware rebalancing and a distributed variant for high QPS.

### Weaknesses
1.The system is “designed for single-model serving” and does not support request migration once placed. Production stacks are often multi-model/multi-tenant, where cross-model interference, fair sharing, and migration can materially affect tail SLOs and GPU count.

2.The method uses a “most naive” history-based predictor (low overhead but high error), and relies on rebalancing to compensate.

### Questions
1. Is it possible for author do a experiment with random dataset with uniform sequence length.

2. Is it possible for author do a experiement with multi model serving.

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
4

### Summary
This paper introduces Aladdin, a scheduler for LLM inference serving designed to minimize GPU cost while adhering to probabilistic SLO constraints. The system co-adaptively performs request placement and resource scaling, explicitly modeling request-level uncertainty. It formulates the placement task as an online multi-dimensional bin packing problem solved via a heuristic. Experiments show Aladdin can reduce serving costs by up to 71% compared to baselines.

### Strengths
1. Instead of pursuing a complex and high-overhead predictor, the authors adopt a simple "historical average" estimator, arguing that its "unbiased" nature allows errors to partially cancel out within a batch. This, combined with the proposed re-balancing mechanism, is an interesting and practical design choice.
2. The paper presents an end-to-end system that jointly optimizes worker configuration and request placement. Formulating the placement task as an online multi-dimensional bin packing problem and providing theoretical guarantees under probabilistic SLOs constitutes a contribution.

### Weaknesses
1. The paper's structure could be improved. The first paragraph of the introduction is long; it should be focused on defining the problem. The review of recent methods and research gaps can be the second paragraph. Figure 1 consumes a large amount of vertical space; a horizontal layout would be more space-efficient.

2. Section 2 is overly long. It should be simplified, with non-essential details moved to the appendix. Furthermore, Section 2 and Section 3 are interconnected and could be combined into a single, more cohesive section on problem formulation and modeling.

3. In my mind, the most critical component of this scheduling problem is the output length prediction. If one had access to an "oracle" predictor for the output token number, it would be easy to design an algorithm to optimize SLOs and balance load, likely using standard methods from convex optimization. Therefore, the paper's most interesting claim (at least to me) is its "unbiased" historical predictor, even as the authors argue it is not their main contribution. I believe this point is vital and requires more analysis in the evaluation. What is the system's performance (SLO attainment, cost) if the algorithm is given the true, ground-truth output lengths from the test set? Comparing this oracle-predictor performance to the current results (using the historical average) is a good way to quantify the performance gap and understand how much headroom is left.

4. I think the assumption that input length is a sufficient proxy for output length is not universally true. A query can be semantically profound while being syntactically short. For example, a query like "Prove Fermat's Last Theorem" is only a few words, but its solution (Wiles's proof) spans over 100 pages. The authors need to provide the results from the "oracle" experiment (requested in Weakness #3) to demonstrate to what extent their historical-average assumption  holds and where its limitations lie.

5. The evaluation is limited to the Llama2 model family. The input/output length distributions, as well as performance characteristics, might be different for other modern architectures. To claim generality, It will be great if the author can show results for other models (e.g., Qwen3) to validate that the proposed performance models and scheduling gains are not specific to Llama2.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Aladdin, a scheduler designed to reduce the high cost of LLM inference by co-adaptively placing queries and scaling computing resources under probabilistic SLO constraints. To minimize GPU usage, Aladdin first determines the optimal worker configuration (e.g., tensor parallelism) to maximize per-GPU throughput . It then treats request placement as a multi-dimensional bin-packing problem, using a "best-fit" heuristic to maximize worker utilization while satisfying constraints like KV cache limits and a novel Average Token Generation Time (ATGT) SLO . Finally, it uses an error-aware rebalancing algorithm to adjust new request placements, compensating for inevitable output length prediction errors . Experiments show Aladdin reduces LLM serving costs by up to 71% compared to standard baselines like vLLM while maintaining SLO attainment.

### Strengths
The paper is well structured and clearly presented, with a logical flow that makes the main arguments easy to follow. 

The proposed approach appears to be useful in a Service Level Objective (SLO) context, offering practical insights that could benefit both researchers and practitioners. The authors provide sufficient motivation and background, and the organization of the sections enhances readability.

### Weaknesses
(1) The paper mentions that the testbed is limited to A100 or V100 GPUs. However, in practice, the total number of GPUs in a cluster is large. Therefore, the small scale of the physical testbed seems unconvincing. Moreover, the paper needs to better justify why a simulation can represent a real-world setting.

(2) Using only a few GPUs in the experiments does not make the "cluster-level" claim convincing.

(3) Heterogeneous GPUs: Did the experiments consider heterogeneous GPU environments? It appears that each experiment used only one type of (homogeneous) GPU.

(4) Why is being "non-biased" as important as the paper claims? Even if the method is unbiased under the law of large numbers, it might still introduce large variance at the batch level. I think a biased method could be acceptable. Instead of making the bias as low as possible, we usually want to reduce the MSE, which requires a trade-off between variance and bias (

$$MSE = \text{Variance} + (\text{Bias})^2$$

). Moreover, calibration methods can be used to reduce bias.

(5) Semantic Level: The assumption that prompts of similar length have similar output lengths seems unreasonable. For example, the prompts "What is the answer to 100+100?" and "Give me the longest sci-fi novel possible" have similar input lengths but vastly different expected output lengths. 

(6)Furthermore, according to Figure 2, all max slopes are near 500. This implies that different input length do not represent useful predictive features. I believe even a small BERT-like model could reduce the error significantly. Why is such a model not used?

(7) Based on point (5), I doubt the usefulness of the re-balancing algorithm.

I will consider raise my score if all my concerns are solved.

### Questions
same as the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Aladdin, a probabilistic, SLO-aware scheduler that jointly optimizes query placement and GPU resource scaling for LLM inference. Unlike prior systems that focus on single-worker throughput, Aladdin explicitly models latency distributions under uncertainty and places queries to maximize per-worker utilization. It introduces a flexible constraint interface that enables distribution-aware tail modeling and risk-adjusted capacity allocation with high efficiency.

### Strengths
1. This paper addresses a real problem. The assumptions are realistic. 

2. The method jointly optimize request placement and resource scaling under probabilistic SLO constraints, while existing works treated these problems independently.

3. The method adapts to dynamic workloads with provable efficiency.

4. This work bridges theory and LLM system practice. It has potential for industry deployments after some improvements.

### Weaknesses
I personally like the idea of this paper. The only weakness I see is that the evaluations are limited. It would be better if the authors consider add more evaluations, such as varying different workload, to simulate real-world scenarios better.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
