# Training-Free Self-Scheduling for Efficient LLM Inference Serving

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 4, 2, 4, 6

## Abstract
The ability to deliver fast responses under strict latency requirements is critical for Large Language Model (LLM) inference serving. 
Most existing systems rely on a first-come-first-served (FCFS) scheduling policy, which often suffers from head-of-line blocking. 
While a number of solutions have been proposed, they typically require training additional models or auxiliary predictors, such as BERT, to estimate decoding lengths. 
These approaches limit generalization and necessitate retraining for new domains or distributions.
To address these limitations, we propose self-scheduling with LLM, a novel approach that leverages the reasoning capabilities of the LLM itself without requiring extra training or auxiliary models. 
We systematically investigate a range of feasible strategies and conduct extensive analyses. Experimental results show that our method achieves up to a 5$\times$ improvement in TTFT, a 3$\times$ improvement in TPOT, a 6$\times$ reduction in latency, and a 9$\times$ increase in throughput under both general and domain-specific workloads, with negligible overhead.
This work offers a lightweight yet intelligent scheduling paradigm, demonstrating both practicality and strong potential for LLM inference serving.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes training-free self-scheduling for LLM serving: instead of training a separate length predictor, the deployed LLM itself estimates response length (or ranks requests by expected length) to prioritize short jobs. Reported gains over FCFS include up to 5× lower TTFT and 3× lower TPOT on several workloads.

### Strengths
- The paper is clearly written and well structured; the core idea and its three variants (PrefillOnly, RankOnly, and LengthOnly) are easy to follow.
- The approach is practical and drop-in, avoiding the maintenance of auxiliary predictors while remaining compatible with standard serving stacks.

### Weaknesses
- The work does not include head-to-head comparisons against prior training-based schedulers ([1][2]) or “LLM-tells-its-length” methods ([3]), which makes it difficult to assess competitiveness beyond FCFS.

- The methodology risks inflating TTFT because it decodes tokens to estimate length or ranking before scheduling, so the first token for many requests may be delayed unless all ranking/prefill tokens are rigorously counted and reported. Can you provide more evidence that on why the TTFT is improved? 

[1] Qiu, Haoran, et al. "Efficient interactive llm serving with proxy model-based sequence length prediction." arXiv preprint arXiv:2404.08509 (2024).

[2] Fu, Yichao, et al. "Efficient llm scheduling by learning to rank." Advances in Neural Information Processing Systems 37 (2024): 59006-59029.

[3] Zheng, Zangwei, et al. "Response length perception and sequence scheduling: An llm-empowered llm inference pipeline." Advances in Neural Information Processing Systems 36 (2023): 65517-65530.

### Questions
- Please add controlled baselines that include a trained length regressor/classifier and an LLM-length-prediction method, matched for model, backend, and compute budget.

- Please break out and charge all scheduling overhead (prefill/ranking tokens) to TTFT/TPOT, and report sensitivity under varying load to clarify the true latency impact.

### Soundness
2

### Presentation
3

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
This paper proposes training-free self-scheduling for LLM inference: instead of training an auxiliary length and rank predictor, the server uses the LLM itself to rank incoming requests by expected response length and schedules short ones first to mitigate head-of-line blocking. Three variants are explored.

The paper also introduces a rank-aware anti-starvation mechanism. Experiments across NuminaMath, TACO, ShareGPT report up consistent latency and throughput improvements, with small extra overhead.

### Strengths
1. Cleverly reuses the served LLM for ranking/length estimation, avoiding auxiliary latency/length predictors and reducing system complexity while still improving tail throughput.

2. Empirically shows clear gains over prefill-only (logit probe) baselines, suggesting self-scheduling is a promising direction.

### Weaknesses
1. The paper lacks experiment details and does not specify, up front, which models are used for serving and self-scheduling (this appears only later in Fig. 9). It leads to confusion during reading. 

2. Missing baseline comparisons. There’s no comparisons against other baseline methods for (i) ranking overhead and (ii) rank-match quality.
​
3. Insufficient system analytics across loads: Table 2 analyzes a single operating point (rate = 64, bsz = 1). The paper lacks load sweeps (QPS/RPS); would be good to vary load and report **throughput** as well.

### Questions
1. Can you provide comparisons with baseline schedulers in terms of ranking overhead and system performance (TTFT/TPOT/throughput) under identical settings? Please include rank-match metrics.

2. Insufficient throughput characterization. In addition to Table 1, could you add experiments with varying RPS, showing how goodput/throughput changes with and without self-scheduling? 

3. How does the system behave at low/medium/high RPS? Since re-ranking adds overhead, have you explored adaptive re-ranking frequency (e.g., disable or downsample re-ranking at high load) vs. always re-rank all requests? Quantitative results across load regimes would clarify the trade-offs.

### Soundness
2

### Presentation
2

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
This paper proposes training-free self-scheduling for LLM inference: instead of training an external length predictor, the serving LLM itself briefly “judges” a small pack of pending requests to estimate relative response lengths and decide execution order, aiming to reduce head-of-line blocking under FCFS. The method is positioned as lightweight (no extra models, no retraining) and broadly applicable across domains.
Claimed contributions are: (1) a training-free LLM self-scheduling paradigm; (2) three concrete strategies with tie-aware evaluation; (3) empirical validation of latency/throughput improvements; and (4) a rank-aware anti-starvation mechanism for fairness.

### Strengths
1.Proposes a training-free self-scheduling paradigm that leverages the serving LLM itself—rather than auxiliary predictors—to estimate relative response lengths and order requests, reframing scheduling as an in-model reasoning task. This is a creative repurposing of “LLM-as-a-Judge” ideas to systems scheduling and removes the retraining barrier present in prior work.

2.Executes a multi-dataset study (NuminaMath, TACO, ShareGPT) on an 8×A100 vLLM stack with sensible P95/P99 metrics for TTFT/TPOT/latency; reports consistent advantages of LengthOnly and includes throughput tables under fixed-time and fixed-workload settings.

### Weaknesses
1.Although the proposed system does not introduce an auxiliary predictor, the serving model itself must perform additional reasoning (ranking prompts) before actual decoding. This inevitably increases TTFT and overall latency, since the same LLM is doing both scheduling and generation. Under high-concurrency workloads, it is unlikely that such overhead remains negligible, as the ranking step can stall GPU pipelines and disrupt prefill scheduling.

2.All experiments are conducted at moderate request rates. The authors should explicitly evaluate two contrasting real-world scenarios:
Low-latency regime — set batch_size = 1 to test per-request responsiveness.
High-concurrency regime — stress the scheduler under large-scale Poisson arrivals to test throughput stability.

### Questions
1.Is the ranking step executed synchronously on the same GPU stream as decoding, or asynchronously in parallel?

2.Can the authors provide per-stage timing (prefill, ranking, decoding) to quantify how much TTFT and latency increase per batch?

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
4

### Summary
This paper proposes self-scheduling with LLM, a novel approach that leverages the reasoning capabilities of the LLM itself without requiring extra training or auxiliary models. Given a set of requests, it prompts the LLM to provide a relative ranking of their expected response lengths. The LLM outputs a predicted rank ˆR, where ties are allowed (i.e., multiple requests may be assigned the same rank). Experiments show the performance.

### Strengths
1. Request scheduling for LLM serving system is important.

2. Training-free self-scheduling, especially PrefillOnly, is simple and efficient.

3. Experiments show the performance.

### Weaknesses
1. The main concern is the experiments may not fit practical serving system. In experiments, Req/s seems to be insufficient and does not fully utilize the GPU computation power. Based on Figure 6, TTFT and TPOT decreases with the increase of Req/s. If keep increasing Req/s, TTFT and TPOT should increase.

2. The extra latency can be better evaluated. In section 5.4 EXTRA COST, what is 20 math requests with rate = 64? Could the authors show the result with huge amount of requests?

3. Code is node provided.

### Questions
Separate prefilling and decoding in different servers may improve the performance.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a training-free self-scheduling method  that leverages the reasoning capatilities of LLMs to estimate model output lengths. Instead of relying on auxiliary models or tools to estimate response lengths, the authors propose three strategies to rank latency or predict request lengths for scheduling.

### Strengths
1. The method is simple and straightforward. 
2. The method does not require extra training or auxiliary tools.
3. The method can be adapted widely across different modesl and datasets.
4. The method is efficient and easy to integrate.
5. The authors designed a starvation control method to prevent queries from waiting for long.

### Weaknesses
1. The method requires additional decoding, which slightly introduce additional overhead.

2. I’m concerned about the practical applicability of the proposed method. The starvation control method is somehow naive. In real-world scenarios, some queries with long response sizes may be more important yet still experience long waiting times. Determining scheduling priority based on estimated response length may be weak when handling diverse requirements in real AI services.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
3
