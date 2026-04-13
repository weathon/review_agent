## Human Reviewer 1

### Summary
The paper presents DEFT (Decoding with Flash Tree-Attention), a hardware-efficient algorithm that optimizes large language model (LLM) inference for tree-based decoding tasks like few-shot prompting and multi-step reasoning. Current systems struggle with redundant Key-Value (KV) cache loading and poor load balancing, causing inefficient memory use and low GPU utilization. DEFT solves this with KV-Guided Grouping, which reuses shared prefixes to reduce KV cache access, and Flattened Tree KV Splitting, which improves GPU efficiency. Implemented with OpenAI Triton, DEFT achieves significant speedups in attention latency compared to existing methods

### Strengths
Relevance: The paper tackles a timely and significant problem in optimizing LLM inference for tree-based decoding applications, which is highly relevant to current AI research and deployment.
Originality: Introduces a novel attention algorithm, DEFT, leveraging KV-Guided Grouping and Flattened Tree KV Splitting to address memory access inefficiencies and load balancing.
Theoretical Justification: Provides solid theoretical analysis to justify the proposed methods, including IO complexity analysis and discussions on the correctness of the algorithm.
Empirical Validation: Demonstrates significant improvements in end-to-end latency and attention computation across multiple tasks (few-shot prompting, multi-step reasoning, speculative decoding) compared to state-of-the-art baselines. The supplementary material includes extensive experimental results and ablation studies, strengthening the empirical validation.
Comparison with Concurrent Works: The supplementary material provides detailed comparisons with concurrent works, clarifying the advantages of DEFT in handling multi-level tree decoding and addressing unbalanced workloads.
Scalability: The authors provide results demonstrating DEFT's scalability to larger models (up to 34B parameters) and different hardware setups.
Accuracy Preservation: The paper includes analysis showing that DEFT maintains model accuracy, with negligible differences in attention scores and perplexity compared to baseline methods

### Weaknesses
Presentation Clarity: While the supplementary material improves clarity, some sections of the main paper remain dense, and the inclusion of key explanations from the supplementary material into the main text could further enhance understanding. Significant critical information is gained through the supplementary material, specifically regarding reproducibility and algorithm details, which would benefit from inclusion in the main text.
Limited Discussion on Energy Efficiency: The paper still focuses primarily on speedup metrics, and while memory access reduction implies energy efficiency, an explicit discussion or measurement of energy consumption would strengthen the work.
Applicability in Varying Scenarios: Although the authors include experiments with varying tree widths and prompt lengths, further exploration of scenarios with minimal shared prefixes or very small tree widths would provide a more comprehensive understanding of DEFT's applicability.

### Questions
Integration of Supplementary Material: Could the authors consider integrating key explanations and findings from the supplementary material into the main paper to improve readability and clarity for readers who may not delve into the appendix?
Energy Efficiency Metrics: While DEFT reduces IO operations, have the authors considered measuring the impact on energy consumption or providing an analysis of energy efficiency improvements?
Minimal Shared Prefixes Scenarios: How does DEFT perform in scenarios where the shared prefix is minimal or the tree width is very small? Are there any overheads introduced in such cases compared to existing methods?
Realistic Scalability: Do the authors foresee any limitations or challenges in extending DEFT to more common larger model sizes (e.g., 70B parameters, or 400B) or to different model architectures beyond those tested? These larger models generally excel at complex multi-step reasoning tasks compared to the <32B counterparts, which may reveal different patterns in inference and could affect the effectiveness or accuracy retention of your approach.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper presents DEFT (Decoding with Flash Tree-Attention), an optimized algorithm for efficient inference in tree-structured large language model (LLM) tasks, such as few-shot prompting, multi-step reasoning, and speculative decoding. Existing methods face inefficiencies from redundant memory access of shared prefixes and unbalanced GPU workload distribution, which leads to low utilization and slower processing. DEFT addresses these issues with two key techniques: KV-Guided Grouping, which minimizes memory access by loading shared prefix data only once, and Flattened Tree KV Splitting, which enhances GPU utilization by evenly distributing workload across GPU units. Implemented on Triton, DEFT achieves up to 2.5x faster end-to-end speeds by significantly reducing memory operations, making it highly effective for complex, tree-based LLM applications.

### Strengths
1. **Efficient Memory Usage and Balanced Workload Distribution**: DEFT's KV-Guided Grouping minimizes redundant memory access by loading shared prefix data only once, reducing IO costs associated with repeatedly reloading the KV cache. Combined with the Flattened Tree KV Splitting strategy, which evenly distributes data across GPU units, DEFT maximizes GPU utilization by ensuring balanced workload distribution, thus avoiding bottlenecks and maintaining consistent processing speeds.
2. **Enhanced End-to-End Processing Speed**: Compared to state-of-the-art methods, DEFT achieves up to a 2.5x speedup in end-to-end latency, making it highly effective for tasks that require complex, tree-based structures like few-shot prompting and multi-step reasoning.
3. **Scalability Across Tasks**: DEFT demonstrates versatility by performing well across different tree-structured applications, such as speculative decoding, where shared prefix usage and efficient load balancing are particularly challenging.

### Weaknesses
1. **Lack of Comparison with Shared Prefix Infrastructure**: While DEFT introduces novel techniques for memory efficiency and load balancing, it lacks a direct comparison with existing infrastructure solutions like vLLM and DeepSpeed-MII, which already support shared prefix KV cache across different batches. Such a comparison would clarify DEFT’s advantages and limitations relative to widely adopted methods that also aim to reduce redundancy in KV cache management.
2. **Challenges with Distributed Memory and Tensor Parallelism**: DEFT’s current design primarily targets single-device GPU optimization and may not be directly compatible with distributed memory or tensor parallelism setups, which are commonly used to scale large language models across multiple GPUs. Adapting DEFT to work efficiently in distributed environments could require additional modifications to handle inter-device communication and memory sharing effectively, potentially limiting its scalability for very large models.

### Questions
1. Reasoning has become a popular approach to enhance the performance of large language models (LLMs) on complex tasks. Are there any future plans to integrate this method within task pipelines to achieve end-to-end improvements?
2. As noted in the weaknesses, tensor parallelism is widely used to scale large LLMs across multiple GPUs. Will this work be released as an open-source repository to help develop an infrastructure, similar to vLLM or DeepSpeed, that provides a usable framework for the public?
3. The test on speculative decoding sets T from 32 to 256, which is much larger than usual settings (<10), have you test speculative decoding with smaller T value?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces the DEFT (Decoding with Flash Tree-Attention) algorithm, aimed at enhancing efficiency in tree-structured language model (LLM) inference. Traditional approaches often fall short due to redundant memory access, inefficient handling of KV cache for shared prefixes, and poor GPU utilization. DEFT addresses these issues through two primary innovations: KV-Guided Grouping and Flattened Tree KV Splitting. Authors claim that these strategies optimize memory accesses and ensure balanced GPU utilization, leading to significant speed-ups in tree-based tasks like few-shot prompting, multi-step reasoning, and speculative decoding.

### Strengths
1. Authors introduce KV-Guided Grouping, which reuses memory for shared prefixes in the KV cache, minimizing redundant I/O operations.
2. Authors' approach to balanced workload distribution via Flattened Tree KV Splitting leads to better GPU usage.
3. Triton implementation provides strong empirical evidence of the efficacy of the method.

### Weaknesses
1. While single GPU performance is quite good, it is not clear how DeFT can scale to larger models requiring multiple GPUs.
2. Though there is a one-liner on vLLM comparison, there is no numerical comparison with vLLM given that vLLM also implements prefix-based KV-cache sharing.
3. The overhead of QKV PREPARATION PHASE is unclear from the empirical results.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
Tree-structured decoding is gaining more popularity in LLM serving due to the presence of applications such as multi-step reasoning and speculative decoding. Existing inference systems are inefficient due to their failure to be prefix-aware: they either perform redundant recomputation of KV caches for shared prompts, or repeatedly load and store KV caches of shared prompts during attention calculation. This paper presents DeFT, an efficient attention calculation algorithm with prefix-awareness and load-balanced KV cache partitions. DeFT uses KV-guided grouping to group the prefix's KV cache with all shared queries. It then uses flattened tree KV splitting which splits the KV cache into balanced partitions to reduce overhead in computation. Evaluations show that DeFT has better wall-clock time speedup in multiple tree-structured decoding applications compared to state-of-the-art baselines.

### Strengths
1. Tries to solve the important problem that current LLM serving systems are inefficient in computation and IO for tree-based decoding applications.
2. Provides good background on segmented attention and existing attention algorithms.
3. Evaluation results show decent speedup over baselines.

### Weaknesses
1. The paper is hard to follow. The design figures include too many details. Lack of clear explanation of the key techniques including KV-guided grouping and tree KV splitting.
2. Lack of evaluation or discussion on multi-node settings and other baselines.

### Questions
Thank you for submitting the paper to ICLR 2025! I think this paper tries to tackle the important problem of improving GPU utilization for LLM serving under the scenario of tree-structured generation. The paper provides a good background of example tree-structured applications, how existing attention algorithms work and how attention could be calculated in a segmented way. The evaluation of the new proposed algorithm demonstrates solid speedup over existing baselines. I overall feel positive about the paper with a few comments and suggestions for improvements.

The current illustration of the main algorithm in Section 3 is hard to follow.

There are remarks and comparisons here and there.

Figure 3 includes too many notations and details that make the reader hard to recognize which are the baselines and which are the new techniques proposed in the paper. Even after reading all the text, I could not clearly figure out how flattened tree KV splitting works in detail. There are tons of places where the descriptions refer to the Appendix.
However, I think the reader should be able to grasp the intuition and how the algorithm works at a high level by just reading the main text of the paper.

My current understanding is that the core of the DeFT algorithm is to help create balanced and sharable QKV groups during the QKV Preparation Phase. It is probably better to clearly define how KV-guided grouping and flattened tree KV splitting work into two separate subsections, as they are the two main techniques proposed in the paper.

In terms of questions, how do you define the node sizes in the tree KV? 

If the DeFT-Node-Chunk adds additional overhead due to imperfect splits after splitting by the nodes, could we first optimize the tree KV structure to ensure we have nodes of balanced sizes?

In the Attention Calculation Phase, how many techniques introduced in the paper are novel compared to previous works?

In addition, how does the proposed technique compare to [cascade inference algorithm](https://flashinfer.ai/2024/02/02/cascade-inference.html)? The cascade inference algorithm also makes the observation that the KV caches could be shared when there are common prefixes between requests. It first uses a multi-query attention kernel to compute the attention between queries and KV caches of the shared prefix, which goes through L1 cache and registers. Then it uses batch decode attention kernel to calculate for the remaining suffixes, which accesses the global memory and L2 cache.

In terms of experiments, it seems all evaluations are currently completed on a single A100 GPU.
How would the performance be if the algorithm is applied in a multi-node distributed LLM inference setting?
Would any of the parallelization techniques affect the effectiveness of the splitting algorithm?
How would the algorithm perform in a long context LLM serving scenario?

Other questions:

1. For Table 5, why is there an even larger speedup for the case of upper-bound (no attention)?  Isn't the proposed algorithm only optimizing for the attention operation?

2. How would different types of attention operation (e.g. multi-head, multi-query, or group-query attention) affect the performance of DeFT?

3. For Figure 4, what would the latency breakdown be for DeFT-Node-Chunk? Would unpaged versions of DeFT-Node-Chunk and DeFT-Flatten incur similar overhead for KV management?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3