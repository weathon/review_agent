# Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
KV caching is a fundamental technique for accelerating Large Language Model (LLM) inference by reusing key-value (KV) pairs from previous queries, but its effectiveness under limited memory is highly sensitive to the eviction policy. 
The default Least Recently Used (LRU) eviction algorithm struggles with dynamic online query arrivals, especially in multi-LLM serving scenarios, where balancing query load across workers and maximizing cache hit rate of each worker are inherently conflicting objectives.
We give the first unified mathematical model that captures the core trade-offs between KV cache eviction and query routing.
Our analysis reveals the theoretical limitations of existing methods and leads to principled algorithms that integrate provably competitive randomized KV cache eviction with learning-based methods to adaptively route queries with evolving patterns, thus balancing query load and cache hit rate. 
Our theoretical results are validated by extensive experiments across 4 benchmarks and 3 prefix-sharing settings, demonstrating improvements of up to **6.92$\times$** in cache hit rate, **11.96$\times$** reduction in latency, **14.06$\times$** reduction in time-to-first-token (TTFT), and **77.4%** increase in throughput over the state-of-the-art methods.
Our code is available at https://github.com/fzwark/KVRouting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the critical trade-off between KV cache eviction and query routing in multi-LLM serving systems. It introduces the first unified mathematical model to formalize this trade-off, identifies theoretical limitations of existing LRU-based eviction policies, and proposes two novel algorithms: RLT eviction and LBGR.

### Strengths
1. The unified mathematical model fills a critical gap by explicitly coupling local cache eviction dynamics with global load balancing, a connection that prior heuristic-based methods failed to formalize.
2. RLT’s randomized eviction mechanism is simple to implement, seems simple but works. 
3. The evaluation covers a broad range of scenarios: 4 benchmarks (synthetic and real-world), 3 prefix-sharing settings, model sizes from 8B to 70B (dense and MoE architectures).

### Weaknesses
1. The theoretical analysis notes that L-LRU’s performance degrades with imbalanced query lengths, but the paper does not explicitly evaluate how RLT/LBGR perform across different query length distributions (e.g., heavy-tailed vs. uniform).
2. The ablation table (Table 1) reports "Average Eviction Time" and "Average Routing Time" but does not compare these to baselines (e.g., L-LRU’s eviction time vs. RLT’s).

### Questions
1. I am wondering how to run the code? seems the authors do not contain a readme.
2. The state-of-the-art baseline is "Cache-Aware+LRU" from SGLang, which switches between hit-rate and load-based routing using a fixed threshold. Have you compared LBGR to dynamic threshold-based routing (where the threshold adapts to query arrival patterns)?

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
2

### Summary
This paper presents the first unified theoretical model that couples KV cache eviction and query load balancing for multi-LLM serving. It identifies the fragility of LRU-based eviction under dynamic query arrivals and proposes two principled algorithms: RLT, a randomized eviction achieving logarithmic competitive ratio, and LBGR, a learning-based greedy routing method predicting end-to-end latency online.

### Strengths
1. This paper is clearly written, and the main problem is well-motivated from a practical LLM-serving perspective.  
2. This paper provides a combination of a theoretical foundation and practical implementation.
3. I appreciate that the authors go beyond heuristic system designs and provide a theoretically grounded formulation together with competitive analysis for the cache eviction process.

### Weaknesses
1. As I understand, RLT may be affected by the random seed. It would be better to include an ablation study evaluating the stability of RLT under different random seed settings.
2. From Figure 6, it appears that the advantage of your method diminishes as the number of workers increases. Could you explain why the proposed approach cannot (or does not need to) scale to a larger number of workers?
3. Writing: It would be better to include a notation table in Section 3 to improve readability and help readers follow the theoretical formulation.

### Questions
Please see weaknesses.

### Soundness
2

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
4

### Summary
The paper proposes a unified model of KV cache–aware load balancing and introduces two algorithms that jointly improve cache hit rate and end-to-end latency. The authors show L-LRU has worst-case O(n) competitiveness while RLT achieves O(logn), and they report large empirical gains across four benchmarks and multiple model sizes.  ￼  ￼  ￼

### Strengths
- Clear problem framing that couples cache eviction with routing
- Strong empirical results across four benchmarks with higher hit rate and throughput￼

### Weaknesses
- Some assumptions are under-discussed (see questions)

### Questions
Thank you for the submission. I like the paper overall, the KVCache scheduling topic for load balancing is timely, the theoretical results are crisp, and the experiments are compelling. I especially appreciated the side-by-side algorithms and the clear latency decomposition in LBGR, which make the motivation and mechanics transparent. That said, a few descriptions felt a bit under-explained to me. I’d appreciate clarifications on the following:
- Theorem 5 assumes βL_{\max}\le B_i and that queries in a batch are distinct. How representative is this in deployed systems? What happens to the bounds or behavior when those assumptions are violated?
- How sensitive are results to the hyper-parameter settings choices across hardware/model sizes? Any guidance for setting them without tuning? 
- Since you normalize many plots, it would help to include at least one table with absolute value so readers can reason about real-world SLOs.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies the concept of KV caching, which is fundamentally important for serving LLMs. The authors focus on providing a mathematical model to understand the interplay between KV cache hits and query load balancing. They provide theoretical results on the poor worst-case performance of algorithms like Leaf-LRU. They then provide two new algorithms: randomized leaf token (RLT) and learning-based greedy routing (LBGR). This new approach outperforms state

### Strengths
1. LLM KV cache managements and query routing is a very critical problem, and the paper does a good job of choosing an important problem to solve
2. The paper does quite a good job at piecing together the theoretical underpinnings of KV cache management, which makes the motivation of RLT and LBGR easy.
3. Section 3.1 does a great job at formalizing the notation and laying the groundwork for further sections. The lemmas are intuitive to understand
4. The experiments are quite extensive, covering large and small, dense and MoE models. The improvements across the board are strong

### Weaknesses
1. The improvements claims made in the intro should be qualified by model type and size, context length, HBM available etc. Otherwise it is hard to trust these numbers. Please take the time to segment the results into small vs large, dense vs MoE, relationship with context length etc.
2. Some recent literature reviews are missing. For example, [1]
3. The figures could be a bit better. For instance, Figure 5 is violating the margin.
4. The idea of the MIP is not used much throughout the paper, so casting the problem as a MIP seems a bit incomplete

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
