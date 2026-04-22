# Progressive Binarization with Semi-Structured Pruning for LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large language models (LLMs) have achieved remarkable progress in natural language processing, but their high computational and memory costs hinder deployment on resource-constrained devices. Binarization represents the most extreme form of quantization, yet binarized models still contain redundancy that can be further removed. Pruning provides a natural way to eliminate such redundancy, but naïve combination with binarization often results in severe performance degradation. In this paper, we propose Progressive Binarization with Semi-Structured Pruning (PBS$^2$P), a novel post-training framework that seamlessly integrates binarization and semi-structured pruning. We first propose Stepwise semi-structured Pruning with Binarization Optimization (SPBO), which progressively introduces sparsity while optimizing binarization parameters to jointly reduce pruning and quantization error, yielding more stable and accurate compression. Additionally, we propose a Coarse-to-Fine Search (CFS) that first allocates pruning ratios and then refines element selection, further enhancing overall performance. Extensive experiments across multiple LLM families show that PBS$^2$P consistently outperforms state-of-the-art (SOTA) binary post-training quantization methods in both perplexity and downstream accuracy. We will release all the code and models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a pipeline for jointly applying 1-bit quantization and semi-structured pruning to Large Language Models (LLMs). The method is built upon a layer-wise closed-form solution for 1-bit quantization, governed by two control parameters, which minimizes the reconstruction error under the Frobenius norm. For pruning, a global strategy assigns layer-specific sparsity ratios based on inter-layer similarity, and the Optimal Brain Surgeon (OBS) framework is used to select the weights for removal. A key aspect of the pipeline is the iterative update of the quantization parameters after each pruning step to maintain accuracy. The constituent techniques are established, the key contribution lies in the integration of these methods into an end-to-end pipeline.  However, given that SparseGPT also supports joint quantization and semi-structured pruning using the OBS framework, the similarities between the two methods should be explicitly clarified. Experimental results demonstrate that the proposed method outperforms compared methods. However, to strengthen the validation, evaluations on more recent model families like Qwen and comparisons against state-of-the-art baselines such as OmniQuant and  ParetoQ should be included.

### Strengths
The constituent techniques are established, the key contribution lies in the integration of these methods into an end-to-end pipeline. Experimental results demonstrate that the proposed method outperforms compared methods. The presentation is easy to follow.

### Weaknesses
This work presents a pipeline for joint quantization and semi-structure pruning, but it has several weaknesses that limit its current impact. The methodological description lacks clarity in key areas, such as the handling of activation flow during compression (independent vs. sequential) and the specific role of hyperparameters like the block size. The technical foundation is questioned, particularly the choice of the cosine similarity metric without a clear strategy for handling negative values or empirical validation of its distribution. The experimental validation is limited in scope, relying heavily on WikiText2 and older model families like LLaMA, while omitting stronger recent baselines and a thorough comparison to the closely related SparseGPT. Finally, the paper would benefit from a discussion of the method's applicability to broader architectures like MoE and diffusion models.

### Questions
1. Regarding the formulation in Equation 2, it appears the compression objective for a linear layer uses the original, unmodified activations. Could you clarify the following? (a) Is each linear layer's compression treated as an independent objective, using the original network's activations? (b) Or is the compression applied sequentially, where the input to a layer comes from the already compressedprevious layer?

2. You use cosine similarity, which ranges from [-1, 1], to gauge layer importance. Could you please clarify how your algorithm handles cases where the cosine similarity is zero or negative? The subsequent use of a reciprocal (i.e., 1/similarity) in your global pruning ratio assignment would be undefined or invert the intended importance ranking in such scenarios. Was this encountered in practice, and if so, how was it addressed?

3. Could you show some empirical results of the cosine similarity values computed for the layers of a model (e.g., Llama-2 7B)?

4. The results (Tables 1 and 2) indicate the use of a block size of 128. Could you please clarify the role of this parameter in your method? Specifically, is this block size exclusively for the 1-bit quantization process?

5. The experimental evaluation currently reports perplexity results primarily on WikiText2. To more thoroughly and fairly assess the generalizability of the proposed method, it would be beneficial to follow the common practice established by your cited baselines (e.g., GPTQ, BiLLM). Could you please include perplexity results on additional standard datasets, such as PTB and C4?

6. The experimental validation is conducted on established model families like LLaMA (1-3) and OPT. To further demonstrate the relevance and effectiveness of the method, it would be valuable to include results on more recent and widely-used models, such as the Qwen series.

7. The method is presented in the context of standard dense transformer-based LLMs. Could you comment on its potential adaptability to other important model classes (e.g., MoE models, diffusion models)?

8. From a general perspective, the goal of jointly performing quantization and pruning is also a key feature of the SparseGPT [1] framework. Specifically, SparseGPT supports various quantization bit-widths alongside semi-structured pruning, and similarly utilizes the OBS framework for weight selection and error minimization. Given these high-level similarities, could you please provide a more detailed discussion of the fundamental differences between your method and SparseGPT?

9. The experimental comparisons would be strengthened by including recent state-of-the-art methods that support extreme low-bit quantization of LLMs, such as ​​OmniQuant [2]​​ and ​​ParetoQ [3]​​.

10. The paper reports computational savings based on the latency of a single matrix multiplication operation. However, in real-world deployment, end-to-end inference time, which includes I/O overhead, memory access patterns, and other system-level bottlenecks, is a more meaningful metric for evaluating efficiency. Could you please provide measurements of the end-to-end inference latency (e.g., tokens/second) for a complete forward pass on a standard benchmark?

[1] Frantar, Elias, and Dan Alistarh. "Sparsegpt: Massive language models can be accurately pruned in one-shot." International conference on machine learning. PMLR, 2023.

[2] Shao, Wenqi, et al. "Omniquant: Omnidirectionally calibrated quantization for large language models." arXiv preprint arXiv:2308.13137 (2023).

[3] Liu, Zechun, et al. "Paretoq: Scaling laws in extremely low-bit llm quantization." arXiv preprint arXiv:2502.02631 (2025).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces PBS²P (Progressive Binarization with Semi-Structured Pruning), a pure post-training framework that pushes LLMs down to 0.55–0.8 bit weight precision while retaining SOTA perplexity and zero-shot accuracy on LLaMA/OPT families. 
The method alternates two components:
SPBO – step-wise N:M pruning followed by on-the-fly re-optimisation of binarisation scalars α/μ to reduce compound error.
CFS – a coarse-to-fine search that first allocates layer-wise pruning rates via cosine-similarity importance and then picks elements to prune by a Hessian-based second-order criterion.

### Strengths
1. The paper is well-written.
2. The paper introduces PBS2P, a novel post-training framework that seamlessly integrates binarization (1-bit quantization) and semi-structured pruning (N:M sparsity), effectively reduces combined errors from pruning and quantization
3. Ablation tests validate each component (e.g., SPBO, CFS metrics, pruning types), highlighting their necessity and superiority, which strengthens the method's credibility.

### Weaknesses
1. The proposed method involves some predefined constants, such as N_high and N_low in CFS, and hyperparameters like Optimization Steps. It is unclear how to set the values of these predefined constants whether the settings of these constants affect the final compression effectiveness. (I am concerned that there may be difficulties or troubles in setting these constants during practical applications.)
2. The paper only tested zero-shot tasks on relatively old models, such as the Llama1 and Llama2 series. If applied to stronger models (e.g., Llama3 or Qwen3 series) after quantization and pruning, how would it perform on zero-shot tasks?

### Questions
1. How sensitive is the method in the paper to calibration data? Does the distribution of calibration data have an impact? How should calibration data be selected for training?
2. The paper demonstrates efficiency advantages in matrix multiplication. How much efficiency improvement can it bring in normal inference tasks?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PBS2P, a post-training framework that combines binarization with semi-structured (N:M) pruning for LLM compression. The method consists of two key components: (1) Stepwise Pruning with Binarization Optimization (SPBO), which jointly optimizes weight pruning and binarization parameters, and (2) Coarse-to-Fine Search (CFS), a two-stage strategy that first allocates pruning ratios based on layer importance and then selects specific elements using Hessian-based metrics. Experimental results on LLaMA and OPT model families demonstrate improvements over STBLLM and other binary post-training quantization methods.

### Strengths
1.	Well-motivated problem: Combining binarization with pruning to reduce redundancy and overcome performance degradation is a valuable research direction.
2.	Comprehensive experiments: Extensive evaluation across multiple model families (LLaMA-1/2/3, OPT), datasets (perplexity and zero-shot), and model sizes demonstrates broad applicability.
3.	Thorough ablations: Section 4.4 provides a good analysis of design choices (SPBO, search metrics, group size, etc.).

### Weaknesses
1.	Certain techniques are not well explained, which may cause confusion and make reproduction difficult. See specific concerns in the Questions section below.
2.	Computational cost: Inverting block wise covariances even at size 128 is not cheap; the fine stage dominates runtime (109 min on 7B). Complexity and wall-time scaling to 65B/70B should be analyzed more carefully (per-layer cost, number of SPBO alternations τ, M−N steps).

### Questions
1.	The notation in Equation 4 is confusing: what is the shape of 1? If 1 is just a column unit vector, μ should be a scalar. However, the binarization center for each row should be different.
2.	For the Coarse Stage, why not use gradient-based importance (e.g., Fisher information) or loss sensitivity?
3.	For Equation 7, the "+1/2" term for rounding is not explained. Additionally, the concrete choices of N_high and N_low are not presented in the paper.
4.	Theorem 3.1 (Equation 8) is essentially a restatement of classical results from Optimal Brain Surgeon (Hassibi et al., 1993). The "proof in supplementary" claim doesn't add novelty—this is a well-known second-order approximation. What is the difference between Theorem 3.1 and the results from OBS?
5.	Computational cost: 111 minutes for LLaMA-7B is 2.5× slower than ARB-LLM. For larger models (65B), this could be prohibitive. Is there any study on the computation time for large models?
6.	Table 4(c) only analyzes the LLaMA-7B model. While RI causes a degradation, the LI metric seems to provide only a small improvement. There should be more justification on more models for importance selection in the coarse stage and the necessity of adaptive assignment.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Progressive Binarization with Semi-Structured Pruning (PBS²P) for compressing large language models (LLMs). The core component is SPBO (Stepwise Semi-Structured Pruning with Binarization Optimization), which progressively prunes a subset of elements at each step while jointly optimizing the binarized parameters, effectively reducing the combined error from pruning and binarization. In addition, the authors introduce a Coarse-to-Fine Search strategy to improve the accuracy of pruning element selection, further enhancing compression efficiency. Extensive experiments show that PBS²P outperforms existing post-training quantization methods across various LLM families and evaluation metric.

### Strengths
1. Method design shows some innovation: The paper jointly optimizes pruning and binarization, using a stepwise strategy to reduce the error accumulation from single-step pruning.
2. Comprehensive ablation studies: Experiments validate the contributions of the SPBO strategy as well as different metrics and pruning types to performance.
3. Clear presentation: The writing is well-structured, and the workflow and formulas are described in detail, making the approach easy to understand.

### Weaknesses
1. Limited innovation: Although the combination of stepwise pruning and quantization is experimentally validated, it essentially remains a combination of pruning and quantization, resulting in moderate to low novelty.
2. Hardware support limitations: The paper adopts 5:8 and 6:8 N:M sparsity configurations, but public documentation shows that NVIDIA GPUs only natively support 2:4 sparsity. Therefore, higher-ratio sparsity may not achieve hardware acceleration in practice.
3. Unclear hyperparameter selection: The method for setting 𝑁high and 𝑁low is not specified, lacking theoretical justification or search strategy, which reduces reproducibility and interpretability.
4. Optimality of mask decomposition not demonstrated: The stepwise progressive mask decomposition is not proven to be optimal, and there may exist schemes that achieve higher accuracy at the cost of longer runtime. The paper does not explore this trade-off.
5. Method limitations: The SPBO’s stepwise updates rely on calibration data and multiple iterations, increasing computational cost. The impact on efficiency for large-scale model deployment is not thoroughly discussed.

### Questions
1. How are 𝑁high and 𝑁low selected? Is there a transferable principle or tuning strategy?
2. For the 5:8 and 6:8 configurations, is hardware acceleration actually achieved, or are they only used for experimental comparison?
3. Have other mask decomposition schemes been tried? Is there a better accuracy-runtime trade-off?
4. What are the computational overhead and practical deployment costs of SPBO on large models?

### Soundness
2

### Presentation
3

### Contribution
2
