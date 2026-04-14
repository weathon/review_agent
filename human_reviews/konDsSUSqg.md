# MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression

- Decision: Reject
- Scores: 5, 8, 6, 3

## Abstract
Sparse attention can effectively mitigate the significant memory and throughput demands of Large Language Models (LLMs) in long contexts. 
Existing methods typically employ a uniform sparse attention mask, applying the same sparse pattern across different attention heads and input lengths. 
However, this uniform approach fails to capture the diverse attention patterns inherent in LLMs, ignoring their distinct accuracy-latency trade-offs.
To address this challenge, we propose the Mixture of Attention (MoA), which automatically tailors distinct sparse attention configurations to different heads and layers.
MoA constructs and navigates a search space of various attention patterns and their scaling rules relative to input sequence lengths. It profiles the model, evaluates potential configurations, and pinpoints the optimal sparse attention compression plan.
MoA adapts to varying input sizes, revealing that some attention heads expand their focus to accommodate longer sequences, while other heads consistently concentrate on fixed-length local contexts.
Experiments show that MoA increases the effective context length by $3.9\times$ with the same average attention span, boosting retrieval accuracy by $1.5-7.1\times$ over the uniform-attention baseline across Vicuna-\{7B,13B\}, and Llama3-\{8B,70B\} models. 
Moreover, MoA narrows the capability gaps between sparse and dense models, reducing the maximum relative performance drop from $9\%-36\%$ to within $5\%$ across two long-context understanding benchmarks.
MoA achieves a $1.2-1.4\times$ GPU memory reduction, boosting decode throughput by $6.6-8.2\times$ and $1.7-1.9\times$ over FlashAttention2 and vLLM, with minimal performance impact.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces MoA (Mixture of Attention), a method designed to compress Large Language Models (LLMs) through the application of sparse attention mechanisms. Unlike existing methods that use a uniform sparse attention mask, MoA dynamically tailors distinct sparse attention configurations to different heads and layers, allowing it to adapt to varying input sizes and optimize for both accuracy and latency. MoA leverages long-range dependencies and model-generated summaries to enhance the calibration dataset, improving the alignment between the original dense model and the compressed version. Experiments demonstrate that MoA can significantly boost retrieval accuracy and reduce perplexity while decreasing GPU memory usage and increasing decoding throughput.

### Strengths
1. MoA can adjust to different input lengths, which is important for handling long sequences commonly found in real-world applications.
2.The method shows a marked improvement in retrieval accuracy and a reduction in perplexity compared to uniform attention baselines. By incorporating long-range dependencies, MoA can maintain global attention necessary for tasks requiring long-range retrieval.
3. Using model-generated summaries rather than human-written ones reduces inconsistencies and aligns attention patterns better with the original model's behavior.

### Weaknesses
1. The process of identifying the optimal sparse attention configuration appears to be complex, involving profiling the model, evaluating configurations, and pinpointing the best plan. This complexity may pose a barrier to practical implementation.
2. The effectiveness of MoA seems to depend heavily on the availability of appropriate long-range contextual data, such as the MultiNews dataset. The performance gains might diminish without such data. How robust the method is against different long-range contextual dataset?
3.Overall I believe this paper is good at its idea and implementation. While the topic of efficient decoding of LLMs is very hot, it is expecte to compare with more competitive related methods in the experiments to demonstrate the effectiveness of the proposed method among its competitors.

### Questions
See the weakness for details.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Authors propose a training-free method for determining the optimal sliding window sizes for accelerating the inference speed of frozen Transformer language models (7B-70B parameters) trained using full attention. The authors demonstrate that using uniform window sizes across all model heads under a given sparsity budget leads to performance degradation and propose a method to optimally allocate window sizes to different heads. For each head h, they compute the approximate increase in model loss ΔL_hij resulting from prohibiting a query at position i from attending to a key at position j for all (i,j) pairs. This computation is efficient, depending only on ∂L/∂a_hij and a_hij's, where a_hij is the normalized attention score in the original model. The process requires just a single backward pass per sequence from a small calibration corpus. After determining ΔL_hij values, the authors compute optimal window size allocations per head by formulating a global multi-objective mixed integer programming (MIP) problem that satisfies the global sparsity budget while minimizing the sum of resulting ΔL_hij values. The MIPs can be solved efficiently using available solvers.

The authors' investigation of calibration corpus selection reveals that corpora involving long-range dependencies, such as summaries of long news articles, yield significantly better performance on evaluation tasks and improved retrieval accuracy.

At 50% sparsity, the authors report significant inference speedups with existing LLM frameworks such as FlashAttention2 and vLLM, achieved through various hardware optimizations including static KV-cache, larger batch sizes due to smaller KV cache, reduced attention computations due to sparsity, and specialized CUDA kernels.

The authors demonstrate that these speedups incur minimal performance loss and, compared to existing fast decoding frameworks H2O, infLLM, and StreamingLLM, their proposed method achieves significantly better long-context retrieval accuracy, long-context understanding scores on LV-Eval and LongBench, and perplexity on long-range text corpora.

### Strengths
1. The proposed method is elegant, mathematically principled, and efficient.

2. The method achieves significant decoding speedups with modest performance loss compared to strong baselines. Faster LLM inference is an important and timely topic of research. 

3. The experiments are thorough and comprehensively support the authors' claims.

4. The authors have shared their codebase for reproducibility.

### Weaknesses
1. While the proposed method targets trained full-attention models, the need for heterogeneous window sizes might diminish with the emergence of uniform sliding window methods augmented with Mamba-like state space layers.

2. The methodology description, especially the MIP section, could be clearer.

### Questions
1. During profiling, what is the loss function? Is it the sum of entropy losses at all output positions or just the largest position? You mention that perplexity is evaluated only on the answer part - does this apply during profiling as well? Please clearly define the loss function.

2. Are the choices of (α_r), (β_r) pairs finite? If yes, what are they?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies sparse attention configuration. Particuarlay, the authors identifies two hyper-parameters for sparse attention configuration, and adopts a gradient-based method to configure these two hyper-parameters in an adaptive manner. The author conduct experiments on various long context evaluation benchmarks with a 4k to 16k context length.

### Strengths
The studied problem is important and emerging. The proposed method is evaluated on LLms of various different sizes. Both wallclock time and memory consumption has been reported. The performance gain is consistent and significant.

### Weaknesses
My major concern is on the experiment design. As this study targets general sparse attention, it would be better to also conduct evaluation on popular LLM evaluations (e.g., AlpacaEval). Since the prevailing application for LLM is generation, it is necessary to inspect the potential performance degeneration on those tasks. 

Also, I suggest the author to also evaluate the wallclock time and memory consumption for a longer context (e.g., 128k), since the proposed method should have the largest generation acceleration potential at those cases.

### Questions
Why the proposed method is named "Mixture of Attention"? I feel the name is a bit confusing and would suggest the author to change it to a more straightforward name.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper proposes the Mixture of Attention (MoA), a novel approach for compressing Large Language Models (LLMs) by leveraging sparse attention. Traditional sparse attention methods apply uniform sparse attention masks across all heads and input lengths, limiting the model’s effectiveness due to the lack of diversity in attention spans. MoA addresses this limitation by introducing heterogeneous sparse attention configurations across different attention heads and layers, allowing the model to optimize its focus on both local and global contexts. This approach extends the effective context length, enhances retrieval accuracy, and reduces GPU memory usage. The authors demonstrate that MoA can increase the context length by 3.9 times and improve retrieval accuracy by 1.5-7.1 times, while achieving significant gains in throughput and memory efficiency compared to other sparse attention baselines.

### Strengths
MoA's introduction of heterogeneous elastic rules across different heads and layers allows for more targeted attention configurations, aligning with the diversity of information retrieval needs across various input lengths. This significantly improves retrieval accuracy and extends the effective context length, providing clear benefits over uniform sparse attention models. MoA features an automated pipeline to determine optimal sparse configurations tailored for each model and dataset, based on profiling each attention head's impact on prediction loss. This approach enhances the model's adaptability to specific datasets with long-range dependencies, demonstrating impressive scalability and performance benefits without the need for retraining. The method achieves substantial GPU memory reduction and throughput improvements, showing a notable advantage in environments with limited computational resources. With up to 8.2 times improvement in throughput and minimal retrieval accuracy loss, MoA represents a valuable tool for deploying LLMs on a larger scale with reduced computational burden. The paper presents a comprehensive evaluation across several models and benchmarks, confirming MoA’s efficiency and effectiveness in diverse settings. Its performance on Vicuna and Llama3 models at different input lengths reinforces the robustness and scalability of the proposed method.

### Weaknesses
1. Sparse attention has been extensively explored in prior research, notably in [1] and [2], which diminishes the novelty of this study. Additionally, H2O [3] has already thoroughly examined the feedback from using a sliding window approach.
2. There is a lack of comparative baselines, specifically H2O [3], SnapKV [4], and PyramidKV [5]. Including these would provide a clearer assessment of the proposed method.
3. An evaluation of Needle is necessary to demonstrate the motivation behind preserving long-range dependencies effectively.
4. The proposed method is really similar to a previous method RazorAttention [6]. I recommend adding this to citations.




[1] https://arxiv.org/abs/2402.17762
[2] https://arxiv.org/pdf/2309.17453
[3] https://arxiv.org/abs/2306.14048
[4] https://arxiv.org/abs/2404.14469
[5] https://arxiv.org/abs/2406.02069
[6] https://arxiv.org/abs/2407.15891

### Questions
As weakness

### Soundness
3

### Presentation
3

### Contribution
2
