# Expected Attention: KV Cache Compression by Estimating Attention From Future Queries Distribution

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Large language models encounter a significant memory bottleneck during inference due to the Key-Value (KV) cache, which stores past token representations and grows linearly with context length. Although using attention scores to evict KV pairs is promising, it is often impractical in real-world scenarios because the attention scores from future tokens have not yet been computed, and modern implementations like Flash Attention do not materialize the full attention matrix, making past scores inaccessible too. To address these limitations, we introduce $\textit{Expected Attention}$ , a training-free method that estimates a KV pair's importance by approximating how future queries will attend to it. By leveraging the distributional properties of activations in LLMs, we compute the expected attention score in closed form for each KV pair. This score is then used to rank and prune KV pairs with the smallest impact on the residual stream, achieving compression without performance loss. Crucially, our approach works in both prefilling and decoding tasks, consistently outperforming state-of-the-art baselines in both scenarios. We release all our code to enable researchers to implement and build upon our methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Expected Attention, a training-free KV cache compression method that leverages the Gaussian properties of LLM activations to estimate the importance of KV pairs for future tokens. The method is evaluated across multiple benchmarks and model families and shows competitive or superior performance compared to existing baselines. While the idea is novel and well-motivated, the paper has several weaknesses that need to be addressed before acceptance.

### Strengths
1.The idea of using Expected Attention to estimate future KV importance without requiring future queries is innovative and theoretically grounded.
2.Extensive experiments across prefilling and decoding phases, multiple models, and benchmarks (LongBench, Ruler, NIAH, Aime25, MATH-500) demonstrate the method's robustness.
3.The authors commit to releasing code and provide detailed experimental setups.

### Weaknesses
1.The Gaussian assumption of hidden states and queries is central to the method but is not rigorously validated across all layers, models, or tasks. The paper should include more quantitative evidence (e.g., normality tests) beyond visual fits in the appendix.
2.The approximation of the RoPE matrix over a fixed future window T=512 is arbitrary and not well-motivated. How sensitive is the method to the choice of T? An ablation study is necessary.
3.The performance on Gemma3-12B in Table 3 is inconsistent and sometimes worse than TOVA . This should be discussed and analyzed.
4.The computational overhead of computing query statistics and expected attention scores is not quantified. How does this impact latency, especially during decoding?
5.The contribution of each component (e.g., Gaussian assumption, RoPE averaging, adaptive per-head compression) is not isolated. Ablation studies are needed to justify the design choices.

### Questions
See Weakness

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
3

### Summary
The authors propose a form of KV cache compression which operates based on measuring the residual connection contribution of a KV pair instead of the attention score. They predict attention contribution implicitly by modeling the distribution of future attention values.

### Strengths
- The approach is novel and interesting since most prior works focus on attention scores. 

- The approach intuitively makes sense, because the contribution is from both the attention scores and the value vector, while prior work only focuses on the attention scores. If the value vector norm is 0 or close to zero, these high attention scores are null.

### Weaknesses
- Why was RULER only considered up to 16K. Most of the models under consideration in table 1 can go well past 16 natively.  

---

Overall, I would be interested to see how the performance holds up over the baseline models for the longer RULER context lengths and on the full subset of RULER tasks. For instance, the harder retrieval tasks like multi-key 3 seem to be omitted from the current set of experiments. If the authors could show that the method holds up for these harder settings, it would be very compelling.

### Questions
- What is meant by "activation value" in figure 1? Is it the dot product value before the exponential? Or something else

- For clarity, can you add a derivation of how you get the expression on teh RHS of equation 7?

- Why do SnapKV and TOVA show odd patterns of missing the needle at around 100-115K context lengths in Figure 3?

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
The author proposed a KV eviction method based on query vector statistics. It captures outstanding tokens in the KV cache in a query-statistics-aware manner rather than simply using a pooled query or a particular query.

### Strengths
The method implementation is pretty simple.

### Weaknesses
- I am not sure this method is really helpful
- The mathematical analysis is not sufficiently intuitive and expressive to convince the reader to use covariant statistics of query vectors. 
- More importantly, the empirical results are not significantly improved compared to baselines in LongBench. 
- Even in the Llama 3.1, the downstream task performance is drastically dropped compared to baselines (25.5 vs. 66.5), which is a clear signal that something is wrong with the method (or experiment).

### Questions
- What is the latency overhead of computing statistics of queries?

### Soundness
2

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
3

### Summary
This paper presents Expected Attention, a training-free approach for compressing the KV cache during large language model inference. It estimates each key’s expected contribution to future attention by assuming hidden states follow a Gaussian distribution and analytically computing expected attention scores using moment-generating functions. By ranking and pruning low-impact KV pairs based on these scores, the method reduces memory use while maintaining accuracy. Experiments on long-context and reasoning benchmarks (LongBench, Ruler, MATH-500, AIME25) show that Expected Attention matches or surpasses prior training-free baselines such as SnapKV and TOVA, achieving up to 50% compression with minimal performance degradation. Limitations include manual compression ratio tuning and lack of kernel-level optimization.

### Strengths
### Novel yet practical idea: 
The paper introduces a theoretically grounded, training-free method for KV cache compression based on the expected attention formulation, which is both conceptually elegant and computationally feasible.

### Solid theoretical foundation: 
The derivation using Gaussian assumptions and moment-generating functions is mathematically sound and provides clear intuition on how expected attention can approximate future query behavior.

### Extensive empirical validation across model families: 
The method is tested on a wide range of LLMs, including LLaMA-3.1-8B, Qwen3-8B, Gemma-3-12B, and reasoning-focused models such as Qwen-R1-1.5B/3B and Nemotron-14B, demonstrating robustness across architectures and scales.

### Comprehensive benchmark coverage: 
Evaluations span both long-context tasks (LongBench, Ruler, Needle-in-a-Haystack) and reasoning benchmarks (MATH-500, AIME25), showing that Expected Attention maintains accuracy even under substantial compression ratios.

### Weaknesses
### Lack of Efficiency Comparison and Runtime Analysis:
While the paper presents an insightful theoretical formulation and strong accuracy results, its evaluation of practical efficiency is limited. Although Figure 5 reports the peak memory usage of Expected Attention under varying sequence lengths and compression ratios, there is no direct comparison of memory footprint against prior baselines (e.g., SnapKV, TOVA, KeyDiff, KNorm, StreamingLLM). Moreover, inference latency or throughput measurements are not provided, making it difficult to assess the real-world efficiency gains. Given that the primary motivation of KV cache compression is to reduce memory and runtime overhead, a more thorough empirical analysis of these aspects would significantly strengthen the paper.

### Minor point: 
1. Missing caption for Figure 5:
The paper includes two subfigures, (a) and (b), showing peak memory usage under different conditions, but the main caption “Figure 5:” itself is missing. Each figure in the paper should include a clear and complete caption describing its content, in accordance with the conference formatting guidelines.

2. Incorrect caption placement for Figure 4:
The caption for Figure 4 appears above the figure, likely to align with the neighboring Table 2 caption, since table captions are required to appear above tables. However, according to the official ICLR formatting rule (`The figure number and caption always appear after the figure`), figure captions must be placed below the figure. While the intent to maintain visual alignment is understandable, adhering to the guideline would ensure consistency and compliance with the submission standards.

### Questions
### Batch inference comparison: 
All reported experiments appear to use batch size 1, focusing on single-sequence inference. In real-world deployment scenarios, LLM inference often runs with larger batch sizes for better throughput and GPU utilization. How would Expected Attention perform when scaling batch size — both in terms of GPU memory consumption and inference latency — compared to prior training-free baselines? Evaluating this aspect would provide stronger evidence of the method’s practicality for real-world applications.

### Computation efficiency comparison: 
How does Expected Attention compare to other training-free KV cache compression methods in terms of actual GPU memory consumption, inference latency, and throughput? Including such comparisons would clarify the real-world efficiency benefits of the method.

### Soundness
3

### Presentation
3

### Contribution
3
