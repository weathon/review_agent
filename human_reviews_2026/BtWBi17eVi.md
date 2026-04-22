# Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval-induced stalls, which lead to cascading latency—where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4× higher throughput and 5× lower latency, without compromising generation quality. Code is available at https://github.com/tiannuo-yang/SearchAgent-X.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the significant system efficiency bottlenecks in LLM-based search agents, which operate by dynamically interleaving reasoning (generation) and retrieval. The authors first analyze the sources of inefficiency, demonstrating a non-monotonic relationship between retrieval accuracy and end-to-end efficiency. They find that both exact search (high retrieval overhead) and low-recall approximate search (high reasoning overhead) are suboptimal, showing high-recall approximate search is the best compromise.

Second, they diagnose a "cascading latency" problem in existing systems (like vLLM) caused by two primary factors: (1) Improper scheduling, where standard FCFS policies fail to optimize for search-agent workloads, leading to poor KV-cache utilization; and (2) Retrieval-induced stalls, where timing misalignments between asynchronous retrieval and generation steps force requests to wait, resulting in cache evictions and costly recomputation.

To address these challenges, the authors propose SearchAgent-X, a high-efficiency inference framework. SearchAgent-X introduces two main techniques: A priority-aware scheduler that replaces FCFS. It prioritizes requests based on context length, retrieval count, and waiting time to maximize KV-cache reuse. A non-stall retrieval mechanism, an adaptive early-termination strategy for ANN search. This strategy monitors retrieval "maturity" and synchronizes the search termination with the LLM engine's readiness, thus eliminating stalls.

Experiments show that SearchAgent-X significantly outperforms baselines, achieving up to 3.4x higher throughput and 5x lower latency while maintaining the generation quality of exact-retrieval systems.

### Strengths
- The paper tackles a critical and timely problem. As agentic systems that interact with external tools become more common, their inference efficiency is a major bottleneck. This is a strong systems-level contribution.
- The analysis in Section 2.2 is a key contribution. The identification of the non-monotonic efficiency curve (Figure 1) is a clear insight. The decomposition of latency amplification into "improper scheduling" and "retrieval-induced stalls" (Figure 2) is clear, well-illustrated, and clearly motivates the solution.
- The solutions are sound and well-motivated.
The priority-aware scheduler is a smart improvement over FCFS, as it explicitly models the value of a request's KV-cache prefix.
The non-stall retrieval mechanism is the most novel part. It's not just naive early stopping; it's a true co-design where the retrieval component's behavior is dynamically adapted based on the generation engine's state. This tight coupling is a key insight.
- The experiments are comprehensive and convincing.
The system is benchmarked against strong baselines (vLLM with ENN, vLLM with ANN, and vLLM+Cache+ANN).
The evaluation covers both offline and online serving scenarios, showing it's robust.

### Weaknesses
- The priority scheduling mechanism (Section 3.2) feels a bit heuristic. It relies on discretizing three metrics ($R_i, C_i, W_i$) into $G$ levels and assigning priority based on whether any metric exceeds a threshold (Eq. 2). The logic seems sensitive to the thresholds $T_{M,k}$, and the logic for combining these metrics isn't deeply explored. 


- The non-stall retrieval mechanism (Section 3.3) depends on retrieval maturity and "the readiness of the LLM engine." The implementation of this "readiness" signal is critical but vague. Algorithm 1 (line 9-11) abstractly mentions CheckExternalNonStallSignal() when the LLM Engine has waiting requests. How is this practically implemented?  

- The non-stall retrieval mechanism is tested on HNSW and NSW, which are iterative, graph-based ANN methods. The "retrieval maturity" concept is tied to this iterative process. It's unclear how this would apply to other major ANN algorithms (e.g., PQ, IVF) that aren't iterative in the same way. I am curious what are the search solution in real-world industry like perplexity

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents SearchAgent-X, an inference framework for Large Language Model (LLM)-based search agents that incorporate priority-aware scheduling and non-stall retrieval. The design is based on the observation that both excessively high and excessively low retrieval accuracy degrade efficiency. SearchAgent-X improves the throughput by 1.3-3.4x

### Strengths
- The proposed method achieves good latency/throughput improvement while preserving the accuracy of search agents.

### Weaknesses
- No discussion about the previous efficiency optimization method for RAG with multiple rounds of retrievals, including [1-3]
- The non-stall early termination strategy in Section 3.3 seems inconsistent with the retrieval-induced stalls in Section 2.2.2. It is unclear how this optimization resolves the stall caused by being late for the LLM generation steps.
- The paper contains several unclear points. See the questions below.

### References

- [1] https://arxiv.org/abs/2502.20969
- [2] https://arxiv.org/abs/2503.14649
- [3] https://arxiv.org/abs/2505.07833

### Questions
- In Section 2.2, what model and dataset did you use, and what was the configuration of the inference engine (e.g., batch size)?
- In Figure 2a, I cannot see how the latency increases by 83x. I only observe the end-to-end latency rising from around 180s to 240s. Also, why are the data points so coarse-grained (e.g., no points between retrieval latencies of 1s and 4s)?
- In Figure 2b, what is d_i​? How is the scheduling performed here (is it rescheduled after each retrieval step)? Also, how can a non-FCFS policy solve this issue? For example, even if we prioritize request a over request b, wouldn’t that still lead to a cache miss for request b?
- In Figure 2c, what is the latency overhead caused by retrieval-induced stalls? If scheduling is done at the iteration level and chunked prefill is used, I think this overhead should be small.
- Section 4.1 mentions using two models from Search-R1 and ReCall, but the figures do not specify which model was used.
- In Section 4.3, it compares the advantages of prefix caching for top-k=5 and top-k=1, but where is the data for top-k=1?
- Section 4.3 mentions that non-stall retrieval reduced end-to-end latency by 41s, but it is unclear which data this refers to.
- It seems that most of the performance benefit comes from priority scheduling, and I think it is not limited to search-based agents. I wonder if this technique can be generalized to other types of LLM agents that use external tools.

### Soundness
2

### Presentation
2

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
This paper systematically investigates the efficiency bottlenecks of LLM-based search agents and uncovers several insights: (1) retrieval accuracy exhibits a non-monotonic relationship with end-to-end efficiency; (2) search agents are extremely sensitive to retrieval latency, where minor delays can lead to an 83× latency magnification effect; and (3) traditional FCFS scheduling results in low KV-cache utilization. Based on these findings, the paper proposes SearchAgent-X, a system that optimizes search agents through high-recall approximate retrieval, priority-aware scheduling, and a non-stall retrieval mechanism. Experiments demonstrate that the system achieves up to 3.4× higher throughput and 5× lower latency while maintaining generation quality.

### Strengths
1. With the rise of DeepResearch-related work, the paper systematically studies efficiency issues in this emerging scenario, offering significant practical significance and application value.
2. Through empirical studies, the paper discovers many noteworthy phenomena. It provides a deep analysis of the root causes of search agent latency, supported by extensive experiments.
3. Priority scheduling avoids complex weight tuning through discretization, while non-stall retrieval adaptively terminates based on retrieval maturity. Both techniques are easy to implement and model-agnostic.

### Weaknesses
1. The main improvements stem from the high reusability of KV-cache, which works for the Search-R1 pattern. However, some current search agents modify historical context to handle overly long retrieval trajectories (e.g., summarizing previous context or keeping only recent k rounds of retrieval results), which could render the priority-aware scheduling completely ineffective.
2. Priority-aware scheduling cannot be applied to commercial models (e.g., GPT), and the non-stall retrieval mechanism cannot be used to retrieve APIs (e.g., Google). These limitations make the application scenarios quite restricted (local LLM, knowledge base retrieval, KV-cache constrained, Search-R1 pattern).
3. Lack of generalization discussion across different knowledge bases and tasks. For more challenging tasks or models that require more retrieval iterations, many designs in this paper—such as concurrency levels, priority level G, and maturity thresholds—may require substantial adjustments.

### Questions
1. The paper mentions that efSearch=500 may lead to unstable recall. What is the specific performance degradation across different benchmarks?
2. How does SearchAgent-X perform on tasks of varying difficulty? What are the latency and throughput improvements over baselines across different tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates efficiency bottlenecks in LLM-based search agents, which dynamically interleave reasoning and retrieval during generation. The authors identify that both exact and coarse retrieval hurt performance and the framework is highly sensitive to retrieval latency. To address this, the authors propose SearchAgent-X, which features high-recall approximate retrieval, priority-aware scheduling and non-stall retrieval.

### Strengths
- The paper is well-written and easy to follow.
- The paper shows the non-monotonic relationship between retrieval accuracy and efficiency.
- Components (priority scheduling, non-stall retrieval) in SearchAgent-X are well-motivated.
- Comprehensive ablation studies demonstrate robustness of the proposed method.

### Weaknesses
- The baselines do not include recent LLM serving framework like SGLang.
- The priority discretization thresholds and non-stall retrieval maturity criterion involve heuristic settings without extensive sensitivity analysis.

### Questions
- Can the proposed non-stall retrieval mechanism degrade answer quality when retrieval terminates too early?
- Could priority scheduling introduce starvation or fairness issues under high request concurrency?
- How would the framework adapt to multi-agent environments, where multiple reasoning agents interact concurrently?

### Soundness
3

### Presentation
3

### Contribution
2
