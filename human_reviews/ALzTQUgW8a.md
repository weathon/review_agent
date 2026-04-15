# MagicPIG: LSH Sampling for Efficient LLM Generation

- Decision: Accept (Spotlight)
- Scores: 8, 6, 6, 8, 8

## Abstract
Large language models (LLMs) with long context windows have gained significant attention. However, the KV cache, stored to avoid re-computation, becomes a bottleneck. Various dynamic sparse or TopK-based attention approximation methods have been proposed to leverage the common insight that attention is sparse. In this paper, we first show that TopK attention itself suffers from quality degradation in certain downstream tasks because attention is not always as sparse as expected. Rather than selecting the keys and values with the highest attention scores, sampling with theoretical guarantees can provide a better estimation for attention output. To make the sampling-based approximation practical in LLM generation, we propose MagicPIG, a heterogeneous system based on Locality Sensitive Hashing (LSH). MagicPIG significantly reduces the workload of attention computation while preserving high accuracy for diverse tasks. MagicPIG stores the LSH hash tables and runs the attention computation on the CPU, which allows it to serve longer contexts and larger batch sizes with high approximation accuracy. MagicPIG can improve decoding throughput by up to $5\times$ across various GPU hardware and achieve 54ms decoding latency on a single RTX 4090 for Llama-3.1-8B-Instruct model with a context of 96k tokens.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduced a novel method dubbed "MagicPIG" to reduce the computation cost of self-attention in long context. Specifically, MagicPIG utilizes Locality-sensitive hashing to approximate the attention score distribution and estimate the attention output. While not decreasing the overall cache size required to store Keys and Values, MagicPIG sampled only a fraction of Keys and Values to calculate the attention scores, reducing the overall computation cost.

### Strengths
This paper is exceptionally well-written and clearly presented, making it tremendously helpful for understanding complex topics. Concepts, definitions, and proofs are structured logically, with clear and concise writing. The proposed approach is intuitive and relatively straightforward, with much of the intuition supported by prior explanations. Additionally, the empirical results are strong.

### Weaknesses
I have a few questions:
1. what is the intuition for selecting (K, L) for the hash table size? 
2. For Table 1/2/3, why is latency not included in the comparison? 
3. Does the author believe further improvement can be made by combining this approach with PEFT?

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce MAGICPIG, a heterogeneous system leveraging LSH (Locality-Sensitive Hashing) sampling to estimate a complete attention distribution, overcoming limitations of traditional Top-K sparse attention methods, which can underperform in certain downstream tasks.

### Strengths
1.	MAGICPIG addresses shortcomings of traditional Top-K attention methods in LLMs, which often assume sparsity and suffer in some downstream applications. Using LSH-based sampling, MAGICPIG more accurately estimates the attention distribution, mitigating the bias found in Top-K approximations. The approach is backed by theoretical guarantees and empirical evidence, underscoring its effectiveness in sparse attention acceleration.

2.	MAGICPIG overcomes GPU VRAM constraints by offloading parts of the computation, including hash table operations, to the CPU. This approach is pivotal for scaling LLMs with LSH-based sampling in resource-constrained, practical environments.

### Weaknesses
1.	While the authors discuss CPU-GPU collaboration, they provide limited data on the effects of PCIe bandwidth and CPU-GPU data transfer overhead. This omission may hinder understanding MAGICPIG’s real-world performance across different hardware configurations.

2.	The paper lacks a detailed analysis of the overhead associated with hash tables. As noted by the authors, hash tables could introduce significant memory and computational costs. Therefore, a more thorough evaluation of these overheads would better illustrate the trade-offs of the proposed method.

### Questions
1.	Is the size of the hash table related to model size and sequence length? How does the size of the hash table affect the performance?

2.	What is the time overhead of constructing hash tables, and which factors influence this overhead?

### Soundness
3

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
This paper introduce a novel approach that leverages LSH sampling to approximate the oracle sampling. Empirical evaluation shows improvement over the baseline.

### Strengths
S1. The problem and the solution is well-motivated in the paper.
S2. The CPU-GPU co-design enables the storage of large LSH index table.
S3. The empirical results outperforms baselines.

### Weaknesses
Weak Points
----
W1. In the design of the proposed system, the authors claim that putting the retrieval stage in the CPU side would allow large hash tables. I wonder if moving the full system into GPU would reduce the latency when the GPU memory is sufficiently large to fit the hash table.

W2. The author discussed a few KV Cache reduction methods in Section 2. However, only quest is considered as the baselilne in the experiments. I would suggest the author to add a reasonable justification or add more baselines.

W3. Another direction of accelerating the inference is to quantize the model. How does the proposed method work on quantized LLM is not discussed.

W3. No code is provided. It might be hard for readers to reproduce the results.

Presentation
----
P1. In the abstract, without any notes, the author claims "achieve 110ms decoding latency on a single RTX 4090" while not actually running the code on RTX 4090. I believe this is a false claim without mentioning the simulation.
P2. Although it might be obvious for readers with retrieval and word extraction background, the acronym niah, cwe, and fwe and not explained before usage. 
P3. The numbers in Figure 6 might be a bit outdated. In addition, the connection between CPU and GPU could be faster SXM.

### Questions
See weaknesses.

### Soundness
3

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
This paper proposes MAGICPIG for efficient attention score sampling to resolve the large KV-cache for LLM inference. The observation is that exact top-k attention sampling may not perform well. The proposal is to conduct sampling using locality sensitive hashing (LSH) and use importance sampling to obtain unbiased estimations. Empirical results show that MAGICPIG achieves high accuracy with low computation cost.

### Strengths
1.	The observation that top-k attention does not perform well is interesting.

2.	The idea of using LSH to conduct importance sampling is novel.

3.	MAGICPIG partitions the computation reasonably between GPU and CPU, i.e., the hashing (which involves matrix operations) is conducted on the GPU while attention computation is conducted on CPU.

### Weaknesses
1.	Lacks an intuitive explanation why LSH-based importance sampling works better than exact top-k attention. For the theorical view, I get it that importance sampling provides unbiased estimation while exact top-k attention does not. However, both importance sampling and top-k selects some attention scores to compute. Is it because (i) importance sampling select some scores that top-k will not select or (ii) once sampled, importance sampling assigns higher weights to scores with low sampling probabilities? It will be good if an ablation study can be conducted. For instance, if the case is (i), will it work if combine top-k sampling and sampling some random tokens (or some tokens at regular intervals of the sequence, for a good representation of the sequence)?

2.	The parameter configurations for LSH can be discussed, which involves the number of hash table (H), the number of hash functions for a hash table (L), the number of collisions for a token to be considered as a candidate for attention computation (T). Currently, T is fixed at 2. I understand that to sample a fixed number of attention scores, when H is increased, L should be reduced. We can also increase both H and L, but reduce T. Please provide some insights on how these parameters should be set.     

3.	What are the current execution statistics of the system? When the CPU is computing the sampled attention scores, is the GPU idle? GPU or CPU has a longer running time? If we use a pipeline (e.g., by switching between two mini-batches) to overlap GPU and CPU computation, which one will be the straggler?

### Questions
See the weakness part

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the optimization of long-context LLM inference. Unlike most existing approaches that mainly adopt TopK selection for attention calculation, this paper presents a novel method based on importance sampling, where SimHash is used for estimation. Experiments on a set of benchmarks demonstrate the effectiveness of the proposed method and its superiority over a state-of-the-art TopK selection approach.

### Strengths
S1. This paper studies from a new perspective of estimating attention scores, in contrast to TopK selection that has been widely targeted in existing works. The proposed approach showcases its potential in dealing with the non-sparse case of attention. 

S2. Experiments are highly promising, outperforming Quest in both accuracy and efficiency. 

S3. System design is discussed, with the jobs of CPU and GPU clearly depicted in the figure.

### Weaknesses
W1. The context length seems to be short in the evaluation.

W2. Some parameter evaluations are missing in the experiments. 

W3. Only Llama series models are evaluated.

### Questions
Q1. In Figure 3, did you mean even the exact TopK selection yields a higher relative error than oracle sampling? For oracle sampling, I suppose you estimate the weight of each value vector in the attention. As such, a better result can be obtained, especially for the case when attention is not sparse, where TopK selection treats all non-TopK values as zero-weights. 

Q2. For LSH, why SimHash was chosen? The method proposed by Andoni et al. (Practical and optimal LSH for angular distance, NeurIPS 2015) is a better approach than SimHash and has been used in Reformer. 

Q3. How does the budget B relate to K and L? For each hash probe out of L, there could be multiple k_i's having a hash collision with q. I suppose the number of retrieved k_i's in L hash probes should be reflected to the budget. 

Q4. Following Q3, hash collision could be a problem when the context goes long. In this case, K and L can be adjusted to strike a balance, but it is unclear how they are affected by the context length (the context used in the paper seems to be short, see Q5). 

Q5. What is the maximum context length used in the experiments? It seems to be 96K. It is encouraged to see what if the context goes longer, e.g., 1M, which has been evaluated in some TopK-based approaches such as InfLLM.

Q6. Despite evaluating the importance of centering, Figure 8(a) can be seen also as an evaluation of the impact of L. However, I didn't find the evaluation of K. I wonder how K = 8-10 was determined in the experiment. 

Q7. On LongBench and RULER, the performance is even higher when a smaller set of (K, L) is used, e.g. (8, 75) and (9, 120), in comparison to (10, 150). Why?

### Soundness
3

### Presentation
4

### Contribution
3
