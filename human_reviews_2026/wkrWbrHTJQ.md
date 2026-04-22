# Cascadia: An Efficient Cascade Serving System for Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Recent advances in large language models (LLMs) have intensified the need to deliver both rapid responses and high-quality outputs. More powerful models yield better results but incur higher inference latency, whereas smaller models are faster yet less capable. Recent work proposes balancing this latency–quality trade-off using model cascades, which route simpler queries to smaller models and more complex ones to larger models. However, enabling efficient cascade serving remains challenging. Current frameworks lack effective mechanisms for handling (i) the huge and varying resource demands of different LLMs, (ii) the inherent heterogeneity of LLM workloads, and (iii) the co-optimization of system deployment and routing strategy.

Motivated by these observations, we introduce Cascadia, a novel cascade serving framework designed explicitly to schedule request routing and deploy model cascades for fast, quality-preserving LLM serving. Cascadia employs a bi-level optimization method: at the deployment level, it uses a mixed-integer linear program to select resource allocations and parallelism strategies based on LLM information and workload characteristics; at the routing level, it applies a Chebyshev-guided method to iteratively co-optimize the routing strategy and the system deployment produced by the deployment level. Our extensive evaluation on diverse workload traces and different model cascades (DeepSeek and the Llama series) demonstrates that Cascadia significantly outperforms both single-model deployments and the state-of-the-art cascade serving baseline, achieving up to 4$\times$ (2.3$\times$ on average) tighter latency SLOs and up to 5$\times$ (2.4$\times$ on average) higher throughput while maintaining target answer quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CASCADIA, a cascade-serving system for LLMs that jointly optimizes (i) GPU/resource deployment of multiple models (including DP/TP/PP choices) and (ii) threshold-based routing across the cascade to meet a target latency–quality trade-off. It uses a bi-level optimization loop: a MILP picks per-model allocations/parallelism given a routing pattern, and a Chebyshev-guided routing solver adjusts thresholds to satisfy a user quality floor. On DeepSeek and Llama cascades, CASCADIA outperforms single-model serving and CascadeServe, reporting up to 4× tighter SLOs and 5× higher throughput, with ablations showing that both resource allocation and parallelism search are necessary.

### Strengths
- Real problem, 2026-relevant: serving fleets of heterogeneous LLMs under SLO and quality constraints is exactly what people are doing.
- The bi-level loop (MILP for deployment + Chebyshev for routing) is a clean way to expose the coupling between “how many requests go to 70B” and “how many GPUs the 70B should get.”
- Strong empirical numbers: up to 4× lower latency SLOs and up to 5× higher throughput than single-model; 1.7–2.5× over CascadeServe; wins also hold for Llama cascades.
- Removing parallelism search or using uniform GPU allocation degrades performance notably, so the system is not just “a nicer scheduler,” it’s actually using its degrees of freedom.
- Online re-scheduling: handling trace1 → trace2 → trace3 shifts and still beating baselines makes the system more believable for production.

### Weaknesses
- Routing quality is based on GPT-4o-as-a-judge. That’s fine for the paper but less fine for on-prem/air-gapped setups; we don’t see how robust the method is to weaker judges.
- Incremental vs existing cascade work: CascadeServe, AutoMix, and even 2025 routing+speculative papers are moving in the same direction.
- No cost/energy accounting, they motivate with resource efficiency and even cite energy papers, but do not report $/req or some energy unit /req, that’s what operators would care about.
- If the quality distributions used in the routing solver are mis-estimated (domain shift, very hard math/coding workloads), the Chebyshev objective could over-route to large models and wipe out the latency gains.

### Questions
1. The routing quality depends on GPT-4o as the judge. If we swap in a weaker/open-source judge (e.g. Llama-3-70B with an arena prompt), does CASCADIA still satisfy the same 𝑞 min, or does it start over-routing to bigger models? A small sensitivity table would help.
2. Rescheduling policy. How exactly do you detect a “significant” workload shift before re-running the scheduler, and do you use any hysteresis to avoid oscillating between plans?
3. Can the deployment MILP be warm-started from the previous solution to keep solve time low when only the routing thresholds change slightly?

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
This paper proposes Cascadia, a novel cascade serving framework for efficient & effective LLM serving. Cascadia employs a bi-level optimization strategy to compute the optimal serving strategy which includes the deployment strategy (i.e., the resource allocation and parallelism strategies) as well as the routing strategy (i.e., which user requests should be consumed by which models). Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed approach.

### Strengths
1. This paper studies efficient and effective LLM serving, which is critical problem for a wide range of real-world applications.
2. This paper introduces Cascadia with rigorous and solid technical developments.
3. Cascadia achieves up to 4x lower latency deadlines and 5x higher system throughputs, which are impressive.

### Weaknesses
1. In Algorithm 1, Cascadia relies on iteratively optimizing both deployment and routing strategies. It remains unclear if the bi-level optimization is theoretically optimal or not. Such guarantees could be critical in practical scenarios.
2. LLM routing is another well-studied technique aiming for efficient & effective LLM serving, which is under-discussed in this paper. Authors may want to discuss and compare to this line of work to better position the contribution of this paper. Several example references are as follows,

[1] Ong, Isaac, et al. "RouteLLM: Learning to Route LLMs from Preference Data." The Thirteenth International Conference on Learning Representations.  
[2] Ding, Dujian, et al. "BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute." Forty-second International Conference on Machine Learning.

### Questions
What is the typical/expected number of iterations required to achieve stable solutions? If the number of iteration tends to be huge, it can lead to non-negligible overheads and compromise the efficiency gains.

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
4

### Summary
This paper introduces CASCADIA, a novel cascade serving framework designed explicitly to schedule request routing and deploy model cascades for fast, quality-preserving LLM serving. CASCADIA employs a bi-level optimization method: at the deployment level, it uses a mixedinteger linear program to select resource allocations and parallelism strategies based on LLM information and workload characteristics; at the routing level, it applies a Chebyshev-guided method to iteratively co-optimize the routing strategy and the system deployment produced by the deployment level. 

It uses mixed-integer linear programming (MILP) to determine the optimal deployment plan given a routing strategy. It balances response latency and output quality. Within each cascade stage, CASCADIA supports various parallelism strategies (e.g., tensor and pipeline parallelism), which allows it to automatically select the optimal strategy based on model size, incoming workload, and routing decisions.

### Strengths
Stengths:

1. Efficient serving multiple LLM to balance accuracy and latency is an important topic.

2. The proposed cascading method intuitively can help the multi-model serving system.

3. Extensive experiments show the performance.

### Weaknesses
1. The main concern is on the real-time efficiency and cost. LLM serving is an online process. If using GPT-4 to judge the small model response, runs GPT-4 takes a few seconds and the cost is expensive. 

2. Time to first token is also very long. For simple prompt, it also needs to wait until GPT-4 finishes the judge.

3. The baselines are insufficient. BERT-based router [1, 2, 3] that directly routes prompt to multiple LLMs can be compared.


[1] https://github.com/vllm-project/semantic-router

[2] Tensoropera router: A multi-model router for efficient llm inference

[3] RouteLLM: Learning to Route LLMs with Preference Data

### Questions
1. Maybe instead of using GPT-4, fine tune a BERT for cascading?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents CASCADIA, an efficient cascade serving system for large language models that jointly optimizes resource allocation and request routing across hierarchies of model sizes, enabling fast, high-quality, and cost-effective LLM inference. By formulating cascade serving as a bi-level optimization problem and dynamically adapting deployment strategies, CASCADIA achieves significantly lower latency and higher throughput compared to single-model and existing cascade baselines, while maintaining answer quality across diverse workloads.

### Strengths
1. The joint formulation of resource allocation and adaptive inference via model cascades is a novel approach to cost-efficient LLM inference and is a very important problem that needs to be solved before cascades can be deployed in real world systems. The paper also considers heterogeneity in model and workload characteristics which are important considerations in real settings.

2. The paper proposes a viable solution to the problem via bi-level optimization that helps to find an appropriate deployment and routing strategy and shows clear improvements over baselines in experiments

### Weaknesses
1. Some of the details of the approach are not clearly explained. I have added several questions below around points that were not clear to me.

2. While the bi-level optimization itself doesn't seem to be taking too long to solve in online settings (Section 4.4), the latency of re-allocating the models/changing the parallelization may be high.

3. The approach does not consider prefix caching even though the traces used in the experiments do contain multi-turn conversations and prefix caching in such settings and may affect the latencies of both the baselines and Cascadia. Even if it is difficult to incorporate prefix caching in the optimization formulation, I believe when running the traces prefix caching should be enabled to see if it causes the results to deviate significantly from what is expected after solving the optimization problem.

### Questions
1. Why do you try to minimize the maximum latency across models, L, in the MILP, when the latency of a query will be the sum of the latencies of the models that it passes through?

2. Why do you consider separate thresholds for each model when determining the routing strategy even though the same input is passed through the models until a satisfactory output is obtained (if the input is the same then the threshold on the output score should also be the same for all models)?

3. $L(\theta)$ in line 263 appears to be non-differentiable. If that is indeed the case, please clarify how $\theta$ is updated to converge to the minima of the optimization problem.

4. Please provide a citation for the baseline CascadeServe in Section 4.

5. How is query complexity measured when looking for a shift in workload characteristics in Section 4.4? 

6. How does CascadeServe handle distribution shifts? Does it not make any changes under distribution shift?

7. Can you quantify the scheduling overhead (line 476) in terms of additional latency (and not just throughput) when re-scheduling under online workloads? For e.g. if there is a spike in the P95 latency during the rescheduling window then that would not be a good thing.

### Soundness
3

### Presentation
3

### Contribution
3
