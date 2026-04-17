# Mitigating Error Propagation in Low-Rank Approximation of Large Models via Distribution-Aware Whitening

- Decision: Reject
- Scores: 2, 8, 6, 2

## Abstract
Low-rank approximation has emerged as a cornerstone technique for model compression and parameter-efficient fine-tuning, enabling substantial reductions in computation and memory without altering model architectures. However, existing approaches often overlook the shifts in feature distributions induced by the approximation process, which can lead to error amplification and unstable inference.
We propose a distribution-aware whitening framework that dynamically whitens layer inputs based on the evolving feature distributions, ensuring second-order isotropy of input features. 
This allows that the discarded components in the low-rank approximation are those with minimal impact on model outputs, thereby minimizing cumulative approximation errors across layers.
We theoretically analyze how distribution misalignment leads to error propagation and demonstrate that our approach achieves tighter control over layerwise distortion.
Extensive experiments across various large language models demonstrate the superiority of our method in post-training compression. Moreover, our method also serves as an effective initialization for LoRA-style parameter-efficient fine-tuning.
Our findings highlight the importance of considering feature distributions in low-rank approximations, paving the way for reliable and effective model compression strategies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposed a distribution-aware whitening framework for low-rank approximation of LLMs. The key idea is: when applying low-rank factorization/compression of model weights or fine-tuning via low-rank adapters, the input feature distributions at each layer may shift and become anisotropic, which can amplify approximation error across layers.

### Strengths
The problem addressed is relevant and timely: low-rank approximation and efficient fine‐tuning of large models is important for deployment.

### Weaknesses
1 The core method, i.e., how exactly the whitening is applied, over what time/blocks, how the low‐rank approximation is integrated, how the discarded components are selected, is not clearly or cleanly presented.
2 It is not entirely clear how this method compares to other normalization or whitening‐inspired compression methods in literature; the novelty relative to prior work

### Questions
Have you compared to simpler baselines such as applying a standard normalization before low-rank factorization? 

How much marginal benefit does your whitening bring over such simpler normalization?

### Soundness
2

### Presentation
1

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
This paper tackles a core weakness of SVD-style low-rank compression and LoRA init—ignoring how upstream compression shifts activation distributions—by introducing a distribution-aware whitening framework that re-estimates each layer’s input covariance on the fly, whitens activations, performs SVD in the isotropic space, and maps the result back to the original space. The authors give a clean error-propagation analysis contrasting raw, static-whitening, and dynamic-whitening schemes, showing “whitening drift” can amplify distortion across depth and that distribution awareness tightens layerwise reconstruction bounds. Empirically, across several LLMs and datasets, the method yields lower perplexity than SVD-LLM under fixed and dynamically allocated compression ratios, is more robust when combined with quantization (e.g., GPTQ-4bit), and also serves as a stronger LoRA initialization than PiSSA, improving convergence and downstream scores. Overall, it’s a principled and practically relevant refinement of low-rank approximation that links compression fidelity to evolving feature statistics and demonstrates consistent gains without architectural changes.

### Strengths
### Strong theoretical grounding:
The paper provides a formal analysis of error propagation under different whitening schemes, clearly quantifying how distribution misalignment across layers amplifies reconstruction errors. This theoretical treatment makes the motivation and advantage of the proposed method well-justified rather than heuristic.

### Principled and general framework:
The proposed distribution-aware whitening is simple, modular, and architecture-agnostic. It can be used as a drop-in replacement for conventional SVD in model compression or as a better initialization strategy for LoRA without altering existing architectures or training pipelines.

### Comprehensive empirical validation:
The authors evaluate the method across multiple LLMs (e.g., LLaMA-7B, LLaMA3-8B, Qwen2-7B) and datasets (WikiText-2, PTB, C4) under various compression ratios. Results consistently show lower perplexity and better convergence than SVD-LLM and PiSSA, confirming both compression and fine-tuning benefits.

### Compatibility and robustness:
The approach complements other compression techniques like pruning and quantization rather than competing with them, and it demonstrates stability under lightweight post-compression fine-tuning.

### Weaknesses
While the paper provides strong theoretical and empirical validation in terms of perplexity and convergence, it lacks practical efficiency metrics. Since this is a model compression study, reporting inference throughput, latency, and actual memory usage compared to baselines (e.g., SVD-LLM, pruning-based methods) would provide a more comprehensive understanding of real-world benefits.

### Questions
1. Did the authors perform any retraining or fine-tuning after compression? It would be helpful to clarify whether the reported results are obtained purely from post-compression evaluation or involve additional training steps.

2. Since this work focuses on model compression, it would be informative to report inference efficiency metrics such as throughput  and GPU memory usage compared to baselines.

3. In Table 1, including the baseline performance at a compression ratio of 0% (i.e., the uncompressed model) would help readers better understand the degradation trend and relative impact of compression.

### Suggestions:

1. It would be interesting to compare a compressed large model (e.g., Qwen3-8B compressed to 50%) against a smaller pre-trained model of similar size (e.g., Qwen3-4B). If the compressed Qwen3-8B outperforms Qwen3-4B, it would suggest that training a larger model once and then applying model compression for different deployment scales could be more cost-effective than maintaining multiple model families. Such an analysis could significantly strengthen the paper’s practical value and relevance.


2. Extending the proposed distribution-aware whitening approach to vision models (e.g., Vision Transformer) would also be valuable.
Evaluating its applicability to visual backbones or multimodal architectures—such as MaskAlign [1] or PELA [2]—could demonstrate broader generality beyond language models.

--- 

[1] Xue et al. Stare at What You See: Masked Image Modeling Without Reconstruction. CVPR 2023.  
[2] Guo et al. PELA: Learning Parameter-Efficient Models with Low-Rank Approximation. CVPR 2024.

### Soundness
4

### Presentation
4

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
This paper proposes a distribution-aware whitening framework for low-rank approximation in large language models. The core contribution is addressing the shifts in feature distributions induced by the approximation process, which can lead to error amplification across layers. The method dynamically whitens layer inputs based on evolving feature distributions to ensure second-order isotropy of input features, allowing discarded components in low-rank approximation to have minimal impact on model outputs. The authors provide theoretical analysis on how distribution misalignment leads to error propagation and demonstrate experimental results on various large language models for both post-training compression and as initialization for LoRA-style parameter-efficient fine-tuning.

### Strengths
1. **Important problem**: Addresses a genuine issue in low-rank approximation where distribution shifts can cause cumulative errors across layers. The observation that anisotropic input distributions can cause small singular values to align with high-energy input components (leading to disproportionate output distortion) is well-articulated and provides strong motivation for the proposed approach.
2. **Theoretical motivation**: Provides theoretical analysis linking distribution alignment to error propagation, which adds depth to the empirical work.  The decomposition of whitening drift into initial residual and compression-induced shift (Equation 12) offers valuable insights into the source of performance degradation.
3. **Clear presentation of the core idea**: The motivation and high-level approach are communicated effectively. The progression from problem identification to solution design is logical and well-structured.

### Weaknesses
1. **Some kind of Limited novelty in core technique:** The use of whitening transformations to decorrelate features is a well-established technique in machine learning. While the dynamic, layer-wise application is novel, the core mathematical framework (ZCA whitening via Cholesky decomposition) is standard. The main contribution appears to be the engineering insight of recomputing whitening based on compressed activations rather than a fundamentally new algorithmic approach.
2. **Computational overhead not thoroughly analyzed:** While Appendix A acknowledges additional computational cost from per-layer whitening operations, the paper lacks quantitative analysis of this overhead. Critical missing information includes: (a) wall-clock time comparison during compression, (b) memory overhead during the compression process, (c) impact on inference latency after compression, and (d) scalability analysis showing how overhead grows with model size. For practitioners, these practical considerations are as important as perplexity improvements.

### Questions
1. [**Theoretical guarantees:**]
    - Can you provide tighter bounds that incorporate the regularization parameter ε?
    - How do the Lipschitz constants ρₗ behave in practice for transformer layers?
2. **Scalability:** Your largest model is 13B parameters. Do you expect the method to scale to 70B+ models?
3. **Dynamic compression:** In Section 4.4, you combine your method with dynamic ratio allocation. How does layer-wise variation in compression ratio affect the whitening computation? Should the whitening strategy be different for layers with different compression ratios?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Improving over the earlier methods of low rank decomposition, the authors propose that we should incorporate the compression of earlier layers to recompute the input distributions for current layers since injected error at earlier layers affects the input distributions. They theoretically analyze this error. This idea can be extended to model compression as well as PEFT initialization.

### Strengths
The paper is well written, self-contained (for a person like me who does not keep up with latest in model compression) and easy to read. It identifies and solves an important ignored issue in previous methods.

### Weaknesses
1. I do not buy the idea that low-rank is a promising model compression method (post training compression). For instance, the perplexity values in table 1 essentially say that in most cases model is not useful at all after compression. Is there a reason why authors believe that low-rank is a promising method? Related to this, there are improvements in table 1 for sure with using adjustment proposed. But the absolute values are too large for this table to be meaningful in my opnion.

2. The idea is important, self contained and potentially impactful. However, it is incremental since whitening is a generally established idea. I am not sure if the contribution is justifies an paper. 

3. What happens if you only use SVD-LLM whitening (without adjustment for previous layers) in PEFT ?  I believe the impact of adjusting for previous errors will be minimal when used only for initialization.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
1
