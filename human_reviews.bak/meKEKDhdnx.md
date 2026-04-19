# Preble: Efficient Distributed Prompt Scheduling for LLM Serving

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6, 3

## Abstract
Prompts to large language models (LLMs) have evolved beyond simple user questions.
For LLMs to solve complex problems, today’s practices are to include domain-specific
instructions, illustration of tool usages, and/or long context such as textbook chapters in
prompts. As such, many parts of prompts are repetitive across requests. Recent works
propose to cache and reuse KV state of prompts. However, they are all confined to a single-
GPU optimization, while production LLM serving systems are distributed by nature.

This paper proposes Preble, the first distributed LLM serving platform that targets and op-
timizes for prompt sharing. We designed a distributed scheduling system that co-optimizes
KV state reuse and computation load-balancing with a new scheduling algorithm and a
hierarchical scheduling mechanism. Our evaluation of Preble with real workloads and re-
quest arrival patterns on two open-source LLMs shows that Preble outperforms the SOTA
serving systems by 1.5× to 14.5× on average latency and 2× to 10× on p99 latency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In some applications of LLMs, prompts can be much longer than decode length and could be potentially shared across requests. 
However, existing works on KV cache sharing only target a single GPU setting, lacking a cache-aware system for distributed LLM serving.
This paper proposes Preble, a distributed LLM serving platform that utilizes prompt sharing efficiently. 
The core of Preble is a two-level scheduler containing a global request scheduler and a local iteration scheduler.
The global scheduler schedules requests based on exploitation and exploration, trading off between load balancing and prefix sharing.
The local iteration scheduler performs priority-based scheduling to ensure fairness.
Evaluations show that Preble can improve greatly on average and p99 request latency compared to SOTA systems.

### Strengths
1. Good observation that the prefix sharing ratio is high among popular LLM applications.
2. The two-level scheduling algorithms overall make sense.
3. The evaluation results show solid improvement in request latency compared to SOTA systems.

### Weaknesses
1. Lack of justification for some scheduling decisions made in the paper.
2. The workload may not reflect the real distributed LLM serving scenario.

### Questions
Thank you for submitting the work to ICLR 2025! I enjoy reading it. 
I think the problem statement is clear, the design and implementation of the scheduler make sense on a high level, and the evaluation results show clear performance benefits. 
Therefore, I would recommend an accept for this paper. I do have a few questions though and it would be great if the authors could address them.

In terms of the global scheduler for scheduling requests, I am a bit confused about why Preble chooses to favor exploitation over exploration on the point when the amount of recomputation saved (i.e. number of tokens in the matched prefix) is larger than the amount of new computation (i.e. number of tokens in the remaining prompt). 
As far as I am concerned, if a request has a partial overlapping prompt with the prefix cache, it seems to always be beneficial to use the prefix.
This is because, as pointed out in the paper, the prefill latency grows roughly linearly with the number of tokens, so saving the number of tokens needing to compute could reduce prefill latency.
It is true that doing so naively may lead to load imbalance between GPUs, but my feeling is that the load within different GPUs holding the same prefix should play a part in determining whether the scheduler should explore or exploit, instead of merely using the number of tokens as the determining condition.

I am also not sure if the workloads chosen in the paper could fully reflect the real-world distributed LLM serving setting.
How would the sharing ratio be on some chat datasets such as ShareGPT or LMSYS?
In addition, the current evaluation is done on the five datasets respectively. 
However, in a real serving scenario, there may be a set of LLM applications running on top of the same LLM backend at the same time. 
It may be beneficial to show how Preble performs when requests are drawn from different datasets.

Other questions:
1. In Sec 3.2, it says "If an existing tree node only matches partially to the new request, we split the node into the matched part and the remaining part." How often does this happen and would this split incur much overhead?
2. For the case of "exploitation", the algorithm first selects GPUs that have the tree node with the longest tokens in the matched prefix, then selects the GPU with the lightest request load among them. Is this always optimal? What about the case that you have a GPU that has the second longest tokens in the matched prefix but an even lighter load?
3. I find it a bit unintuitive that the value of time window H does not affect the end-to-end results. Have any evaluations been done to evaluate the effect of different time window values?
4. In Sec 4.4, it says "Other workloads and distributions benefit from a different set of techniques." What do you mean by that?
5. In Sec 4.4, it says "One Preble global scheduler can sustain at least 70 to 391 concurrent A100 GPUs", Could you kindly explain how these numbers are calculated and obtained?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a distributed LLM inference framework, Preble, which is specifically designed for long and sharing prompts. Before diving into the solution, the paper first characterizes a wide range of LLM applications and production LLM inference traces that motivate the problems. By designing E2, a distributed LLM request scheduling algorithm, the paper aims to improve LLM usage with long and sharing prompts. Extensive experiments are performed against two popular LLM inference systems: SGLang and vLLM. By running real-world applications, workloads, and GPU testbeds, Preble outperforms existing baselines by significantly improving LLM serving performance.

### Strengths
1 - The paper is well-written and well-organized. I appreciate the intuitive diagrams and figures in the paper.

2 - Motivations and characterizations are clear and inspiring.

3 - The design and analysis are technically clear and sound.

4 - The proposed framework is implemented and evaluated against two popular LLM inference systems, vLLLM and SGLang.

5 - Extensive experiments are conducted, and real-world LLM inference applications are used in the evaluation, which is comprehensive.

### Weaknesses
1 - E2 determines whether it's exploration or exploitation by measuring the computation for matched tokens and computation for unique tokens. This simple design is intuitive enough but remains fixed across different cases. It might be better to make this design dynamic with a knob with load-awareness, where we measure the ratio alpha = (computation for matched tokens / computation for unique tokens) as the knob. For example, if the overall GPU load is quite low (e.g., request rate is low), it would make sense to favor exploitation more by setting this knob alpha lower than 1. This retreats to your fixed design when alpha = 1.

2 - " E2 calculates all three costs as GPU computation time and finds the GPU with the lowest sum." Would it be better if this could be designed as a weighted sum to prevent any dominant components out of the three parts?

3 - The paper mentions that it provides prefill-decoding balancing. How does Preble interact with or integrate with existing continuous batching [1] and chunked prefill [2, 3] techniques?

4 - Minor writing issues: in Appendix C, Figure 11, SGLang should be vLLM in the legends?

[1] Kwon, Woosuk, et al. "Efficient memory management for large language model serving with pagedattention." Proceedings of the 29th Symposium on Operating Systems Principles. 2023.

[2] Zhong, Yinmin, et al. "{DistServe}: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 2024.

[3] Agrawal, Amey, et al. "Taming {Throughput-Latency} Tradeoff in {LLM} Inference with {Sarathi-Serve}." 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 2024.

### Questions
Please see questions in weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a distributed serving system designed to handle large language models (LLMs) with long and shared prompts efficiently. Preble aims to improve the reuse of computed key-value (KV) state across multiple GPUs, overcoming the limitations of current systems that primarily focus on single-GPU optimization. The authors introduce a new distributed scheduling algorithm called E2 (Exploitation + Exploration) to balance the reuse of shared prompt prefixes while managing GPU load effectively. Evaluations using real-world workloads demonstrated that Preble outperforms existing state-of-the-art systems by reducing average latency and p99 latency.

### Strengths
- The authors propsed a first system to address efficient prompt-sharing in distributed LLM environments. 
- The proposed E2 scheduling algorithm seems to effectively balance prefix cache sharing and computation load across GPUs and the authors show significant performance gain compared to SGLang.

### Weaknesses
- Althought the paper claim strong scalability, it is only partially supported by experiments on two four-GPU machines.

### Questions
- What are potential unobserved/unexpected bottlenecks when the systems scales to hundreds or even thousands of machines? Wiil they significantly limit the scalability of the system?

- Given that prompt-sharing can vary beyond exact prefixes, can Preble be extended to support non-prefix caching mechanisms?

- How does Preble’s fairness mechanism prioritize requests if there are multiple users with varying priorities?

- Is there any Trade-off between Cache Utilization and Load-Balancing in E2 Scheduling?

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
The paper presents Preble which optimize for prompt sharing in distributed LLM serving system. More specifically, the paper proposed some load-balancing mechanism that takes considerations into prompt sharing and fairness for large language model in multi-GPU setting, and designed global and local scheduler for coarse assignment and fine-grained scheduling and eviction policy. Experiments show the performance gain comparing to vLLM and SG-lang based baseline.

### Strengths
The paper achieved performance improvements comparing to prompt caching mechanisms (vLLM and SG-lang).

The paper further considered fairness into scheduling process in the optimization in prompt cache.

### Weaknesses
The major concern is that contribution and novelty of this paper is limited. 

Distributed prompt sharing optimization across GPUs: Optimize the inference among multiple GPUs (even in cluster level system) is previously studied by various work. As some of the previous work has been cited by the paper, mem-serve[1], inference without interference [2], mooncake[3] are all large scale systems serves beyond a single GPU. Those systems all considered shared prompts, and can be used with long context in production level. In those papers, global scheduling algorithms are already proposed to route the request with minimal re-computation. Yet, the method is only compared with vLLM and SG-Lang. The paper didn’t discuss the difference between those serving system, as well as considering them as baselines to discuss. 

The localized fine-grained local scheduler: The design of the local scheduler is mainly based on Radix-tree scheduling (SG-Lang), considering fairness added on.

The scenario of the prompt sharing: Previous literature SG-Lang[4], CacheBlend[5] and Prompt Cache[6] have all mentioned the cache reuse opportunity.

[1] MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool  
[2] Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads
[3] Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving  
[4] SGLang: Efficient Execution of Structured Language Model Programs  
[5] CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion  
[6] Prompt Cache: Modular Attention Reuse for Low-Latency Inference

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces Preble, a distributed LLM serving system specifically designed to handle long-context prompt workloads. The authors propose a tree-based prefix-matching algorithm and an online workload-balancing algorithm that combines exploration and exploitation strategies. Preble demonstrates the ability to reduce P99 latency for the Mistral 7B and LLaMA-3-70B models on video QA workloads, using Azure’s LLM request arrival patterns, compared to SGLang and vLLM.

### Strengths
The scheduling algorithm seems interesting. The authors introduce a lot of background of LLM in the appendix.

### Weaknesses
I think the major problem is that the authors mix the concept of high-level cluster scheduling with low-level GPU kernel design, which is very confusing. In Section 3.1, the overall design, the term “data parallel” is used inaccurately. Typically, data parallelism refers to the process of dividing data into chunks and distributing these chunks across blocks and threads in low-level kernel design, instead of splitting data to different servers at a high level.

The statement in the abstract, “production LLM serving systems are distributed by nature,” is misleading. Whether a production LLM requires distributed serving depends on the model size. For instance, in the evaluation section, it’s unclear why the Mistral 7B model is split across multiple servers, as it can easily fit and run inference on a single A100 GPU.

The authors claim to have implemented their scheduling system on top of vLLM, but there is no explanation of this implementation, nor does vLLM appear in any of the evaluation results. To my knowledge, vLLM supports distributed serving and leverages Ray for efficient workload balancing.

### Questions
In Section 3.1, it states, “Our current implementation of Preble scales to at least 70 to 391 GPUs.” Based on my understanding, Preble is designed as a single model serving system. Why, then, does it require such a large number of GPUs to serve a single model? I can serve a LLaMA-3-70B model using just four A100 80GB GPUs. 
How does Preble compare to vLLM?

### Soundness
1

### Presentation
3

### Contribution
2
