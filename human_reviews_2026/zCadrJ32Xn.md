# DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
In large language model (LLM) serving, reusing the key-value (KV) cache of prompts across requests is a key technique for reducing time-to-first-token (TTFT) and lowering serving costs. Cache-affinity scheduling, which co-locates requests with the same prompt prefix to maximize KV cache reuse, often conflicts with load-balancing scheduling, which aims to distribute requests evenly across compute instances. Existing schedulers struggle to reconcile this trade-off, as they operate within a single mapping space, typically applying cache-affinity routing to a subset of requests and load-balanced routing to the rest, without a unified solution to achieve both goals. To overcome this limitation, we propose DualMap, a dual-mapping scheduling strategy for distributed LLM serving that simultaneously enables cache affinity and load balancing. The key idea of DualMap is to map each request to two candidate instances using two independent hash functions based on the request prompt, and then intelligently select the better candidate based on current system states. This design increases the likelihood that requests with shared prefixes are co-located, while evenly dispersing distinct prefixes across the cluster via ``the power of two choices''. To make DualMap robust under dynamic and skewed real-world workloads, we incorporate three techniques: 1) SLO-aware request routing,  which prioritizes cache affinity but switches to load-aware scheduling when TTFT exceeds the SLO, enhancing load balance without sacrificing cache reuse; 2) hotspot-aware rebalancing, which dynamically migrates requests from overloaded to underloaded instances, mitigating hotspots and rebalancing the system; 3) lightweight dual-hash-ring scaling, which leverages a dual-hash-ring mapping to support fast and low-overhead instance scaling without costly global remapping. Experiments on real-world workloads show that DualMap improves effective request capacity by up to 2.25$\times$ under the same TTFT SLO constraints, compared with the state-of-the-art work.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper addresses the challenge of LLM serving systems that utilize KV cache reuse for popular requests. The authors observe that existing systems often face a trade-off, prioritizing either nodes with high cache affinity or nodes that offer better load balancing. The paper argues that these two objectives can be jointly optimized, introducing a system called DualMap. The key insight is the application of the _power of two choices_ technique, where two nodes are probed for each request, and the request is assigned to the more suitable one. The suitability metric is designed to intelligently switch between prioritizing cache affinity or load balancing based on the current system state. Additionally, the authors propose active request rebalancing and the use of consistent hashing to facilitate scaling. The evaluation using two datasets shows that DualMap outperforms existing baselines.

### Strengths
- The paper is well-written, clear, and concise.
- It addresses an important and timely problem in LLM serving: the trade-off between cache affinity and load balancing.
- The proposed solution is systematic and effectively integrates established ideas from the systems literature.

### Weaknesses
- The core problem—hashing, server selection to optimize an objective, and scaling—has been studied extensively in the systems community. The proposed solution bears a strong resemblance to the work in [1]. This raises questions about the work's novelty. While _the application_ (LLM serving) is new, the underlying problem of load-aware request routing seems very similar. The paper would be significantly strengthened if the authors clearly delineated what differentiates DualMap from systems like [1], Chord~[2], or standard consistent hashing. What specific aspects of the LLM serving problem required novel design choices that existing solutions could not address? Explicitly positioning the work, even as a novel application and integration of established techniques to a new domain, would clarify its contribution.

- The assumption that the TTFT of a request $r$ can be accurately estimated a priori (using $TTFT_q$​ and $TTFT_c$​) seems overly simplistic. Request processing time is a complex function, not just of queue length or cache presence. Specifically, the execution time of a batch containing cache misses depends on the mix of prefill and decode requests within it, which is unknown at routing time. Therefore, estimating $T_q​(r, i)$ or $T_c​(r, i)$ just from the queue state seems insufficient. This estimation problem is compounded if the instance-level scheduler does not use a simple FCFS policy. Given these factors, how can the system reliably estimate TTFT and predict potential SLO violations at the moment of the routing decision?

Minor Comments:
- The paper's motivation rests on the importance of KV cache reuse. However, this claim is not substantiated with data. How frequently do large prefix overlaps occur in real-world workloads? Please add a citation or, preferably, a microbenchmark/workload-characterization figure (similar to those in related work) to demonstrate the performance impact and prevalence of this problem.
- Figure 1, which shows the trade-off, would be more effective as a Pareto curve rather than a bar chart. Please also plot DualMap's position on this curve to contextualize its design point.
- Equation 2 uses the term $L(s_i​)$, which is presumably the load of instance $i$. Please define this term explicitly.
- Figure 3 is central to the paper but is difficult to read due to cluttered text. Please consider replotting it, perhaps using a smaller font or simplifying the diagram.
- Figure 7 is key for understanding the SLO-aware strategy but is currently unclear. Please label the strategies directly on the figure to show how they map to the baselines and DualMap. This would make the figure self-contained, removing the need to cross-reference the text (e.g., line 729). Figure 7 in the Sarathi Serve paper is a good example of this.


[1] DeCandia, Giuseppe, Deniz Hastorun, Madan Jampani, Gunavardhan Kakulapati, Avinash Lakshman, Alex Pilchin, Swaminathan Sivasubramanian, Peter Vosshall, and Werner Vogels. "Dynamo: Amazon's highly available key-value store." ACM SIGOPS operating systems review 41, no. 6 (2007): 205-220.

[2] Stoica, Ion, Robert Morris, David Karger, M. Frans Kaashoek, and Hari Balakrishnan. "Chord: A scalable peer-to-peer lookup service for internet applications." ACM SIGCOMM computer communication review 31, no. 4 (2001): 149-160.

### Questions
1. What qualifies as a _sufficiently positive_ benefit $B_r(i \rightarrow j)$​ for triggering an active request rebalancing? Is there a specific threshold, and if so, how is it determined or tuned?
2. In the Appendix (line 665), it is stated that 95% of Conversation dataset requests share a prefix of 2 blocks. What is a block here? What is the size of a block in terms of tokens?

### Soundness
4

### Presentation
4

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
The paper proposes DualMap, a dual-mapping scheduling framework for distributed LLM serving that reconciles the trade-off between cache affinity and load balancing. By mapping each request to two candidate instances via independent hash functions and using SLO-aware routing, hotspot-aware rebalancing, and dual-hash-ring scaling, DualMap achieves both efficient KV cache reuse and balanced workloads. Experiments on real-world traces show up to 2.25× higher request capacity compared to state-of-the-art schedulers.

### Strengths
1. The problem is well defined: the trade-off between cache affinity and load balancing in LLM serving.
2. Extends the “power of two choices” concept to LLM scheduling, offering a novel way to achieve both objectives simultaneously.
3. The evaluation id comprehensive. Benchmarks across models and different baselines clearly show the superior performance.

### Weaknesses
1. The motivation for using two hashes for scheduling is unclear. Is it to save scheduling latency? Or, why not collect global information from all workers and then choose the best one (e.g. based on a weighted sum of prefix-cache and balance benefits)? The paper is very unclear on this point. 
2. While “power of two choices” is cited, formal analysis of DualMap’s convergence or optimality is limited.
3. Lacks of scheduling overhead analysis.

### Questions
1. Is there any challenge integrating Dualmap into disaggregation serving system (e.g. Prefill-Decode disaggregation, Attention-FFN disaggregation)?
2. Could you please further explain the potential drawbacks of using global information for scheduling compared to Dualmap?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents DualMap, a scheduling strategy for distributed LLM serving that considers both the cache affinity and load balancing. DualMap prepares two independent hash functions and selects the better one. It implements SLO-aware request routing, hotspot-aware rebalancing, and lightweight dual-hash-ring scaling. DualMap improves the effective request capacity by up to 2.25x.

### Strengths
- This paper solves the real trade-off in the LLM serving system: load balancing vs. cache affinity.
- Comprehensive evaluation to show the advantage of the proposed method against multiple baselines.

### Weaknesses
- There are a few points that are unclear to me. See questions.

### Questions
- What is the traffic ratio ρ? It seems the paper lacks definition.
- What happens if two hash maps return the same instance?
- How do you choose the instance to migrate ($j$ in Equation 3)? Do you search over all possible $j$?
- In section 3.4, this part was not clear to me: "Because request-to-instance mappings are determined by relative positions on the ring, changes to cluster membership affect only localized regions. This design ensures that most requests retain their original mapping paths during scaling operations"
- I am curious if two is the optimal number for hash maps. Would there be any benefits or disadvantages in having three or more hash maps?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
DualMap build a system to better optimally allocate resources balancing both cache load and cache affinity. They describe ways to determine the appropriate hash length, SLO aware scheduling based on the violation of TTFT, and a rebalancing.

### Strengths
I found this paper to be a good extension on existing work and provide decent/extensive results. 

They contribute and interesting hotspot aware rebalancing and light weight rebalancing.

### Weaknesses
There seems to be a lack of scalability/scheduler overhead analysis in the implementation. It would be interesting to see on more GPUs(even if simulated).

The workload talks about cache migration based on TTFT but it would also be interesting if a direct NVLink transfer/memory cache state awareness was added to this policy.

### Questions
1. Possibly a scalability/scheduler analysis could be good

### Soundness
4

### Presentation
4

### Contribution
4
