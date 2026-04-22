# Scaling Overhead Matters: Saliency-Aware Graph-Based Efficient Post-Training Quantization for LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Post-training quantization (PTQ) is essential for efficiently deploying large language
models (LLMs) in resource-constrained environments. Recent PTQ methods
achieved near-binary precision. However, existing binarization methods rely
on position-based heuristics or fixed saliency assumptions, leading to untracked
scaling overhead and limited adaptability across model architectures. We propose
SAGE-PTQ (Saliency-Aware Graph-based Efficient PTQ), a novel approach for
arbitrary-bit quantization of LLMs. Our formulation comprises five key components:
(1) Saliency-Aware Weight Filtering: identifies salient weights based on
weight distribution statistics; (2) Affinity-Based Weight Grouping: Models inlier
weights with a subsampled graph structure to capture attention patterns and
determine the optimal number of weight groups; (3) Dual-Mode Quantizer Optimization:
iteratively optimizes weight matrix quantization, minimizing scaling
overhead by assigning a single per-channel scale to multi-bit salient weights and
a scalar per-group scale to binarized inlier weights; (4) Adaptive Saliency Thresholding:
dynamically adjusts the saliency percentage to optimally minimize quantization
error; (5) Efficient Inference Runtime: implements a layer-wise lookup
to efficiently load binarized weights for accelerated inference. Our approach
achieves an average of 1.03 weight bits and 0.004 scaling bits per matrix, significantly
outperforming SoTA schemes like BiLLM and PB-LLM binarization.
Evaluations on LLaMA-2-7B yield perplexity of 5.87 (vs. 32.48 for BiLLM) on
WikiText2. Compared to BiLLM, our method uses less than 50% of device memory
and only 6.5% of the FP16 model size, enabling 1.5× faster token decoding
on LLaMA-2-70B with a single NVIDIA L40 GPU. demonstrate strong potential
for efficient inference on edge devices. SAGE-PTQ code will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, a new post-training quantization algorithm called SAGE-PTQ has been proposed.

After partitioning weights into salient and non-salient weights, the authors further partitioned non-salient weights into multiple clusters, each of which is assigned a separate scale.

To do so, several modules have been proposed; 1) module to judge saliency, 2) graph-based module used for the partitioning of non-salient weights, 3) dual quantizers, etc.

### Strengths
The method is clearly explained, but there exists some doubt on practicality of the method.

### Weaknesses
1. I have a doubt about the inference speed. After the quantization, each weight matrix of one layer consists of mixed bits where the positions are random. Specifically, in contrast to the standard per-channel or group-wise quantization, weights having the same bit (i.e., salient or non-salient weights) are positioned in random. Also, for non-salient weights where one-bit is assigned, weights sharing the quantization scale are located in random. How to accelerate the inference of such irregularly quantized weights on the real device? Moreover, while the authors compared the inference latency of the original FP16 model and the quantized model obtained via SAGE-PTQ, there is no latency comparison with the standard uniform quantization.

2. An additional concern is that the authors have considered only weights for the partitioning and quantization. However, in almost all PTQ methods, the degradation in the final task loss has been considered much more important than the perturbation in the weight. 

3. Comparison with GPTQ is totally unfair. GPTQ assigns a uniform bit to all weights. Also, it seems that the quantization performed in SAGE-PTQ is non-uniform quantization, not the standard uniform quantization. Why did the authors compare their method with GPTQ?

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SAGE-PTQ, a novel post-training quantization (PTQ) framework designed to compress Large Language Models (LLMs) to near-binary precision while minimizing scaling overhead. The authors identify critical limitations in existing binarization methods, such as reliance on heuristics, rigid weight grouping, and significant untracked memory overhead from scaling factors. The authors conduct extensive experiments on various LLM families (LLaMA, OPT, Vicuna, etc.) and sizes (up to 70B). The results are exceptionally strong, demonstrating significant improvements over state-of-the-art methods like BiLLM and PB-LLM. For instance, on LLaMA-2-7B, SAGE-PTQ achieves a perplexity of 5.87 compared to 32.48 for BiLLM, while reducing GPU memory usage by over 50% and achieving a 1.5x decoding speedup on a 70B model.

### Strengths
1. The core contribution—using graph-based clustering on subsampled weights to adaptively determine the number of inlier groups—is highly novel and technically sound. It provides a principled, data-driven alternative to the rigid, block-based grouping schemes prevalent in prior work. This approach is well-motivated by the paper's own statistical analysis of weight distributions across different models and layers (Appendix B).

2. The paper's central theme, that "scaling overhead matters," is timely and crucial. Many works in low-bit quantization focus solely on weight bit-width, ignoring the substantial memory and computational cost of scaling factors. SAGE-PTQ's dual-mode quantizer design, which assigns only one scalar per inlier group, directly and effectively tackles this problem. The reported scaling overhead of ~0.004 bits per weight is a massive improvement over the ~1.0 bit for BiLLM.

3. The authors demonstrate a clear focus on practical deployment. The inclusion of an efficient inference runtime (Module 5), the analysis of memory footprint (Figure 4), and the end-to-end latency case study on a 70B model (Table 3) are commendable. These results show that the benefits are not just theoretical but translate into tangible reductions in memory and latency in resource-constrained scenarios.

4. The paper is well-written, clearly structured, and easy to follow. The modular design is explained logically, with each component's role justified. The ablation studies (Figure 5) effectively isolate and validate the contributions of the novel components (graph clustering and adaptive saliency). The appendices provide valuable supplementary material, including detailed algorithms and the motivating statistical analysis.

### Weaknesses
1. The graph-based clustering (Module 2) introduces several hyperparameters, notably the subsampling size and the number of neighbors K in the KNN graph. The paper does not provide an analysis of the method's sensitivity to these choices. The robustness of the "optimal" number of clusters found by the Silhouette Score to these settings is an important, unaddressed point.

2. The processed models in this paper only lies on OPT and LLaMA series, the recent new models such as Qwen, Gemma are missed.

3. SAGE-PTQ is a complex, multi-stage pipeline involving statistical analysis, graph construction, spectral clustering, numerical optimization (Brent's method), and an alternating optimization scheme. While the results justify this complexity, it may pose a barrier to adoption and reproduction. The paper claims faster quantization times than BiLLM (Table 4), which is a strong counter-argument, but the implementation complexity itself remains a concern.

### Questions
1. The authors should also discuss the recent SOTA salience-aware papers in related works, such as SliM-LLM[1].

2. I suggest the author refine the design of Figure 1, which is with most blank space, and it is hard to get the details information from this figure.



[1] SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models. ICML 2025

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
This paper proposes an efficient post-training quantization (PTQ) method, SAGE-PTQ, based on saliency awareness and graph clustering to address memory and latency bottlenecks in the deployment of large language models (LLMs). This method solves the problems of untracked scaling overhead, poor model adaptability, and rigid weight grouping in existing binary PTQ methods through five core modules. Experimental verification shows that SAGE-PTQ performs well on multiple model families (1.3B-70B parameters) such as LLaMA, OPT, and Vicuna, with an average weight bit width of only 1.03 bits and a low scaling overhead of 0.004 bits. Compared with existing methods such as BiLLM, SAGE-PTQ reduces confusion by more than 77%, GPU memory usage by 50%, and inference speed by 1.5 times, providing an efficient solution for deploying LLMs on edge devices.

### Strengths
1. The motivation is clear and meaningful, focusing on the core points of LLM binary PTQ.

2. The proposed methods are highly compatible with the problem-solving objectives.

3. The experimental design is comprehensive and shows relatively good performance, and with calibration costs and inference speed reports.

### Weaknesses
1. The saliency measurement based on Hessian has been thoroughly validated by works such as BiLLM and GPTQ, and the conclusion in the paper shows that the magnitude-based method is significantly better. What is the reason of this phenomenon and is it related to binary quantization settings?

2. Why was sparse KNN chosen for clustering instead of other methods, and more quantitative experiments or metrics should be explained.

3. Dual mode quantization introduces complex scaling factors. Will this incur additional overhead for actual inference, or will it adapt to existing inference frameworks or require operator redesign?

4. The paper should provide an individual ablation study for different components.

### Questions
See weaknesses

### Soundness
3

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
5

### Summary
The paper presents SAGE-PTQ, a post-training quantization method for large language models that pushes weight precision down to nearly 1-bit while maintaining strong performance. The method includes several components: saliency-aware outlier detection, graph-based clustering for the remaining weights, a dual-mode quantizer, adaptive outlier thresholding, and a bit-packed inference kernel.

### Strengths
This is a well-engineered solution to a real problem. The design is thoughtful, the evaluation is thorough, and the empirical improvements over prior 1-bit methods like BiLLM are impressive. The paper is particularly strong in showing practical deployment benefits, like reduced memory and faster decoding, and the implementation seems polished.

### Weaknesses
The novelty feels somewhat incremental. The overall idea, ie, handling outliers in higher precision and binarizing the rest, has been done before, and the contributions here read more like refined heuristics than a conceptual breakthrough. The graph-based grouping and adaptive thresholding are reasonable, but not clearly game-changing. The exposition is dense and could be clearer in places, especially around the clustering and threshold modules. It’s also a bit disappointing that edge deployment is heavily emphasized, yet all experiments are on GPUs. Without tests on CPUs or mobile accelerators, the “edge-ready” claim feels premature.

A few other gaps: there’s little discussion of quantization time or graph construction cost, which could be substantial for large models. Some baseline comparisons are missing, especially to popular 3–4 bit methods like GPTQ or AWQ, which would help contextualize how close this 1-bit method gets to higher-bit PTQ in accuracy. Also, the method involves many components and hyperparameters, but the paper says little about sensitivity or robustness.

Overall, this is a strong paper in terms of engineering and results, but the conceptual advances feel modest. I lean weak reject for now, mostly due to the limited novelty and unclear practical trade-offs during quantization. With clearer exposition, a deeper novelty discussion, and broader baselines, I could see this moving toward acceptance.

### Questions
1. How exactly is the saliency threshold determined? Do you optimize per layer, and how expensive is that? 
2. Did you compare graph grouping to simpler clustering like K-means? 
3. How sensitive is performance to the outlier bit budget or saliency percentage?
4. Any insights into how well the bit-packing and lookup work on CPUs or other edge devices? 
5. How does this compare in accuracy to 3–4 bit quantization methods?

### Soundness
3

### Presentation
3

### Contribution
3
