# ATHENA-Serve: An Intelligent Scheduling LLM Serving System via Horizon-Cost Prediction and Hierarchical RL

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Online inference serving for large language models (LLMs) is foundational infrastructure for conversational agents, retrieval-augmented generation, and multi-tenant intelligent applications. Its core objective is to meet strict latency SLOs under heterogeneous and bursty workloads. However, existing systems suffer from bursty arrivals and long-tailed output lengths that drive peak cache pressure and bandwidth contention, as well as the brittleness of FCFS or shortest-job heuristics under noisy length regression and distribution shift—ultimately compounding tail-latency violations and head-of-line (HoL) blocking. We present ATHENA-Serve, a deployable, horizon–cost–aware LLM serving scheduler. ATHENA-Serve converts predicted generation horizons into calibrated memory and compute budgets. Rather than forecasting exact trajectories, it senses each request’s KV-cache usage patterns and peak-footprint signals. Guided by these budgeted signals, ATHENA-Serve proactively constrains batching and concurrency to smooth memory peaks, while conditioning scheduling decisions on global system signals.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes ATHENA-Serve, a scheduling framework that 1) predicts each request's horizon (output length) and maps it to KV-cache and compute budgets via ORACLE, and 2) uses HERA, a hierarchical scheduling policy, to make admission, batching, and decode-concurrency decisions within those budgets. On ShareGPT traces with Llama-7B-Chat, it reports up to 1.64x lower latency than baseline serving systems. The central thesis is that budget semantics plus hierarchical control tame head-of-line effects under bursty, heavy-tailed workloads and stabilize tails.

### Strengths
1. The budgeted formulation is neat: predicted horizons are converted into calibrated KV/compute budgets, which gives the scheduler a clear, interpretable contract for feasibility checks and homogeneous batching. 

2. The hierarchical design decouples global feasibility/admission from local ordering, which reduces search/credit-assignment noise and provides safe guardrails under high load. 

3. The evaluation emphasizes tail behavior across multiple arrival regimes and shows consistent left-shifts in p95 TTFT/E2E relative to FCFS-style baselines.

### Weaknesses
1. The approach hinges on a trained length predictor and a learned RL policy, but there is no sensitivity analysis for predictor accuracy/placement, policy stability, or contention under different loads/models, which makes robustness hard to judge.

2. The empirical scope is narrow (single model, single GPU, one serving stack version, ShareGPT only), so it is unclear how the method behaves with larger models, multi-GPU/cluster settings, or alternative serving backends. 

3. Baselines are limited to FCFS-style systems plus a light "vLLM+Oracle" ablation; stronger SLO-aware or budget-aware schedulers and recent HoL-mitigation systems are not compared, leaving open whether the gains are competitive beyond FCFS.

### Questions
1. Can you provide sensitivity curves for (a) horizon prediction error/bias and (b) RL policy noise, showing p50/p95 TTFT and E2E across load, and clarify where the method starts to degrade?

2. How does ATHENA-Serve perform with larger models and multi-GPU (e.g., TP/prefill/decode disaggregation, KV sharding), and does the budget mapping still prevent head-of-line under cross-device effects?

3. Under matched compute and identical traces, how does ATHENA compare against recent SLO-aware/budgeted schedulers and HoL-mitigation systems beyond FCFS-style baselines, and which parts of HERA (admission vs. ordering vs. concurrency) contribute most of the tail gains?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes ATHENA-Serve, a horizon-aware LLM serving scheduler that converts predicted output-length “horizons” into calibrated compute/VRAM budgets (via ORACLE) and uses a hierarchical controller (HERA) to make admission, batching, and concurrency decisions. The key claim is that budgeted, state-aware scheduling mitigates head-of-line blocking and smooths memory peaks, improving p95 latency on ShareGPT traces with Llama-7B on a single A40 GPU.

### Strengths
- Addresses on an important problem: tail-latency control and HoL mitigation for LLM serving.
- Interpretable decision making: mapping lengths to budgets makes decisions understandable and deployable.
- Clear intuition: translating noisy length predictions into robust budget classes is sensible and may stabilize decisions under shift.
- Hierarchical control framing makes the policy easier to deploy and reason about than an opaque monolith.

### Weaknesses
- All citation formats in the paper are incorrect.
- The motivation is underdeveloped. The paper does not ground the proposed design in concrete SLOs, production traces, or quantified pain points. Without realistic trace analyses (e.g., burst patterns, KV-cache peaks, multi-tenant mix), it’s hard to judge practical necessity over strong heuristics.
- Motivation for hierarchical RL is underdeveloped. The paper does not convincingly show that a learned hierarchical policy is necessary over simpler, robust heuristics (e.g., SJF/SRPT variants with KV-aware caps, max-horizon-per-batch limits, or rule-based admission tuned by load). The current narrative feels like an engineering extension of well-known size-based scheduling with budget guards.
- Scope of evaluation is narrow: single model (Llama-7B-Chat), one GPU class (A40) first-turn ShareGPT prompts only. Lacking multi-turn traces. 
- Overhead and complexity are not quantified: added latency from ORACLE inference, telemetry, feasibility checks, and HERA control is not reported. Gains may diminish when accounting for these costs or under lighter loads.

### Questions
- Can you provide evidences showing how SJF policies fail due to reponsponse prediction error?
- How are budget class boundaries chosen and calibrated across different models and context lengths? Is there a model-agnostic procedure?
- Can a purely rule-based policy (no RL) with horizon caps, KV ceilings, and age-based priority match your results? Please provide a tuned baseline.
- How sensitive is the policy to length-prediction miscalibration or dataset shift?

### Soundness
2

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
The paper proposes ATHENA-Serve, a serving scheduler that couples (1) ORACLE, a lightweight output-length predictor that maps each request to one of 8 “horizon” classes and converts that into KV-cache and compute budget constraints, with (2) HERA, a hierarchical RL controller that uses those budgets plus live system signals (utilization, queue state, burstiness) to make admission, batching, and concurrency decisions. The key idea is to avoid brittle shortest-job heuristics by turning noisy length predictions into calibrated resource envelopes that the scheduler enforces to mitigate head-of-line (HoL) blocking in decode. Across regimes, ATHENA reduces p95 latency more than means, it claims up to 1.64× lower p95 latency vs SOTA on ShareGPT.

### Strengths
The authors provide clear, closed-form KV/compute budgets and tolerance properties that enable stable feasibility checks and near-homogeneous micro-batches. Leveraging Hierarchical RL that separates admission/envelope from ordering, reduces variance, and enforces safety by construction. The results demonstrate consistent p95 TTFT/E2E improvements across multiple bursty regimes, aligning with the max-horizon argument.

### Weaknesses
1. All experiments are single-GPU (A40-48GB), single model (Llama-7B-Chat); no multi-GPU/multi-node results, no interconnect contention, and no MoE or larger models.
2. Comparisons omit several SLO/placement-aware schedulers (e.g., Sarathi-Serve/DistServe split-phase schedulers as configured for SLOs, ExeGPT-style policies, or recent joint placement work). The vLLM+Oracle ablation is helpful but not sufficient. 
3. Results focus on means/p95; there is no formal SLO satisfaction analysis (e.g., p99, violation rates, TTFT vs ATGT trade-offs), nor throughput/tokens-per-GPU or cost metrics. 
4. ORACLE’s calibration is shown only on ShareGPT-like first-turn prompts; robustness under domain drift is not evaluated. 
5. No quantified scheduling overhead at higher request rates; safe-rollback and starvation/fairness properties are argued but not stress-tested at scale.
6. Missing ablations on bin count, tolerance slack, and the contribution of meta-policy vs sub-policy.

### Questions
1. The authors need to report p99 TTFT/E2E, violation rates, and joint TTFT/throughput curves for fixed SLOs across the five regimes. How sensitive are results to stricter tails (p99.9)? 
2.  How does ATHENA perform with multi-GPU (tensor/pipeline parallel) and multi-node clusters where network/KV paging tails appear? Any data with >1 GPU or 70B-class models?
3. Please add SLO-aware schedulers and split-phase systems configured for SLOs (e.g., Sarathi-Serve, DistServe variants, ExeGPT-like controllers), and include an oracle-length upper bound to isolate policy benefits.  
4. How does ORACLE’s calibration hold under topic/domain shifts (e.g., code, long-form writing)? Can you show online recalibration effectiveness and failure modes?
5.  What is the per-tick scheduling latency at 50–200 req/s, and how does tokens/s per GPU change relative to FCFS/SJF?
6. It's interesting to include the study of varying #bins (K), tolerance slack, and confidence-based slack; disable the meta-policy (admission envelope) vs sub-policy (ordering) to quantify each component. 
7. Provide waiting-time distribution/Gini or tail fairness metrics to verify that prioritizing short-budget jobs does not starve long ones, especially under the Stress regime.

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
This paper proposes ATHENA-Serve, a deployable, horizon–cost–aware LLM serving scheduler. ATHENA-Serve converts predicted generation horizons into calibrated memory and compute budgets. Rather than forecasting exact trajectories, it senses each request’s compute demands and memory footprint. Guided by these budgeted signals, ATHENA-Serve proactively constrains batching and concurrency to smooth memory peaks, while conditioning scheduling decisions on global system signals.

### Strengths
1. User requests scheduling in LLM serving is important.

2. The proposed system ATHENA-Serve proactively constrains batching and concurrency to smooth memory peaks.

3. System experiments demonstrate the performance.

### Weaknesses
1. The main concern is the machine learning contribution may not be sufficient. The paper is a system paper on llm serving. Such a system reduces the memory peaks while the model accuracy. The proposed scheduler is like a rule-based policy without learning or reinforcement  learning.

2. The representation learning of such policy is unclear. Ablation study on the hyper parameters can be provided.

3. The convergence or regret analysis on such RL process can be provided.

### Questions
1. Figure 1, batch or battch?
2. Line 116, missing reference

### Soundness
3

### Presentation
2

### Contribution
2
