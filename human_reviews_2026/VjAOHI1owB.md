# Yet Another Scaling Axis with Some Free Lunch: Enlarging Token-indexed Parameters

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
The scaling laws of large language models have driven remarkable progress, yet they reveal a fundamental bottleneck: performance gains from adding parameters diminish while computational costs grow near-linearly. To overcome this limitation, we introduce a new scaling axis—token-indexed parameters—that decouples model capacity from computational cost. We present ReToken and Mixture of ReToken (MoRT), which augment transformer layers with token-specific modulation vectors retrieved from learned embedding tables. These vectors are applied through table lookups and element-wise operations, adding negligible FLOPs compared to the backbone's $O(d^2)$ complexity per token. Across dense and MoE backbones (190M–3B parameters), our method consistently reduces training loss and substantially improves downstream task accuracy while maintaining unchanged inference FLOPs and latency. Our scaling law analysis reveals that MoRT introduces a multiplicative improvement factor to the loss function, fundamentally shifting the quality-compute Pareto frontier, achieving equivalent model quality with 20-30\% less compute compared to baseline MoE models. These results establish token-indexed parameters as an effective scaling dimension for continued LLM advancement without proportional compute increases.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces training Dense MLP and MoE transformers with (big) lookup tables. The author provides a lot engineering effects to speed up training and inference, including off-load big lookup tables to CPU, asynchronously prefetch rows of the lookup tables from CPU to GPU, overlapping the communication and computation, distribute lookup tables to different GPUs during training. For a set of baseline transformer models (ranging from 190M to 3B parameters), adding lookup tables boost the performance on many common benchmarks.

### Strengths
- The engineering part of this work is techniquely sound, including off-load big lookup tables to CPU, asynchronously prefetch rows of the lookup tables from CPU to GPU, overlapping the communication and computation, distribute lookup tables to different GPUs during training. It would be a Plus if the author provides source code for this engineering part. 

- This work adapts the embedding lookup table techniques for both dense and MoE architectures. 

- This work also contains a lot of experiments on different size of models (from 190M to 3B parameters). 

- The lookup tables optimization idea is interesting for "low compute" cases.

### Weaknesses
- The comparison between Baselines and the proposed methods is unfair. First, the lookup tables introduces a lot of additional parameters. Taking Table 6 Medium (500M params) for example, size of gpt2 tokenizer is 50k, dimension of embedding is 1024, number of layer is 24. The size of lookup tables is 50k*1024*24 > 1B. The total parameters addup to 1.5B. Considering this number of paramters, the proposed method performs worse than baseline. 

- One may argue that the computation cost (Flops things) of baseline model and the proposed methods are close. However, Flops of different operators are hardly compariable. For example, exp(*) operator costs more time than add(*) operator in modern GPUs. Even though both can be viewed as "one operator". It is clear that lookup tables and GEMM are different operators. Thus Flops comparison doesn't make a lot sense. Comparing methods under the same "pretraining time" is a better option. 


- minor typos: 
line 076: miss reference "(?)"
line 099 - 103: ( 3.1) ( 3.2) ( 3.3) ( 3.4) -> (3.1) (3.2) (3.3) (3.4)

### Questions
- Does every layer contain a lookup table? or the whole network share one lookup table?

- What is the pretraining / inference time cost of the proposed method, compared with baseline?

### Soundness
2

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
3

### Summary
The paper introduces ReToken and Mixture of ReToken (MoRT), new method that scales LLM by adding efficient, token-specific modulation vectors. These vectors augment transformer layers with minimal computational overhead, enhancing model capacity without increasing cost. Experiments show this method consistently improves performance and can reduce computational cost, enabling 25–30% compute savings at comparable performance, establishing a practical new scaling axis for LLMs.

### Strengths
1. Strong Empirical Validation: The paper supports its claims with extensive experiments across both dense and MoE models (190M–3B parameters). Downstream performance tables (Tables 1,2,3) show robust improvements across multiple domains.
2. The authors provide detailed engineering solutions (kernel fusion, embedding parallelism, CPU offloading) that make the approach feasible for deployment.

### Weaknesses
1. While the writing is decent, it could be further improved. A diagram illustrating the model design would make it more intuitive and easier to understand.
2. The description of the baseline in the paper is unclear. It would be better to provide a more detailed definition in the experimental section.
3. The paper does not compare against closely related lightweight scaling or parameter-efficient fine-tuning approaches (e.g., adapters, LoRA, or retrieval-augmented memory layers) under the same compute budgets.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

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
This paper introduces token-indexed parameters as an additional set of sparse parameters, providing a new axis for extending the traditional LLM scaling law with negligible extra computational cost. And the paper conducts a comprehensive experiments and analyses to support this claim.

### Strengths
S1: The paper thoroughly considers the computational overhead introduced by token-indexed parameters and provides experimental results for various settings to ensure the effective scaling of existing Scaling Laws.

S2: The paper additionally explores the incorporation of Embedding Parallel and cross-layer residual connections to maintain model training performance even when the memory access overhead and communication overhead are substantial.

### Weaknesses
W1: Given the vast array of transformer variants currently, the authors should provide a more detailed and visual diagram to clearly explain the method. This would enhance the clarity and accuracy of the presentation.

W2: Since the token-indexed parameters ($N_e \times L \times V \times d$) are often significantly larger than the model backbone parameters (roughly proportional to $L \times d^2$), this can lead to a memory access bottleneck when scaling MoRT to larger model scales, which can greatly impede normal training. The authors should include a dedicated section to discuss the memory access and communication overhead, as these are typically more critical concerns than computational overhead in such research.

W3: As the authors have mentioned in the related work, prior studies have demonstrated that introducing sparse embedding parameters can effectively optimize the scaling law curves and serve as an additional scaling dimension. Therefore, the authors should focus their analysis on the $\eta$ (Parameter Ratio) and $\rho$ (Activation Sparsity) components, plot the loss scaling curves under different $\eta$ and $\rho$ settings, different model scales, and investigate whether the observed performance gains can be transferred across varying model scales, as these are often more important.

W4: The authors have mentioned that cross-layer residual connections can mitigate the issue of computation bubbles caused by excessive memory access and communication overhead, which frequently arises when scaling up to larger model scales. However, the introduction of this mechanism is likely to result in a decrease in performance. The authors should also provide an analysis of the corresponding effects after incorporating this feature.

W5: While in theory, the introduction of RMSNorm-like element-wise operations can help reduce the computational FLOPS, the authors should also provide an end-to-end training efficiency analysis considering the communication/memory access overhead.

### Questions
Q1: Could the authors provide some additional information about the characteristics of the sparse embedding parameters (after training) to offer more insights?

Q2: The author mentioned introducing an extra factor to ensure the variance of the hidden states. Could the authors give some ablation or analysis experiments regarding this modification? This would help the reader better understand the effectiveness and implications of this proposed change.

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
4

### Summary
This paper proposes token-indexed parameters as a new scaling axis for large language models (LLMs), decoupling model capacity from computational cost. The authors introduce two architectures, ReToken and Mixture of ReToken (MoRT), which attach token-specific modulation vectors to Transformer layers via lightweight table lookups and element-wise operations. These additions expand representational capacity without introducing new GEMM operations, keeping FLOPs roughly constant.

### Strengths
1. The idea of “token-indexed parameters” introduces an elegant and under-explored way to increase model expressivity independently of computational cost, complementing existing scaling axes (data, model size, compute, and sparsity).
2. Experiments cover both dense and MoE backbones, a wide range of scales, and diverse downstream benchmarks (knowledge, reasoning, code, and math). The improvements are consistent and significant.

### Weaknesses
1. The assumption that token-indexed parameters multiplicatively modify loss scaling is plausible but lacks deeper theoretical explanation or derivation. A more formal argument (e.g., via bias–variance decomposition or representational capacity theory) would strengthen the claim.
2. While the authors claim negligible FLOPs increase, the memory bandwidth and latency implications, especially for CPU-offloaded MoRT, are only briefly discussed. Quantitative latency comparisons (ms/token) are needed.
3. The paper lacks a structural diagram of ReToken/MoRT placement within the Transformer block or the CPU-offload inference path. This omission makes it difficult to assess reproducibility and whether the described mechanism is realistically implementable at inference time.
4. The abstract and introduction claim 25–30% compute savings for equivalent quality, whereas the scaling-law section and Figure 3 describe only ~20% (1.20×) efficiency gain. Please reconcile these numbers and provide full experimental details, including seeds and variance.

### Questions
1. Could the proposed token-indexed modulation be viewed as a parameterized residual bias similar to adapter fusion or feature modulation (e.g., FiLM)? How do they differ mathematically and empirically?
2. What are the memory and latency implications of CPU offloading for MoRT during inference on real hardware (e.g., A100 vs. H100 vs. TPUv5)? Can the authors provide quantitative profiling?
3. In Figure 1, the ReToken and MoRT models show a training-loss drop of over 0.1 compared to the baseline, an effect usually comparable to doubling model size, so how can the authors justify such a large gain with almost no extra compute?
4. Will the authors release the code and training details to verify these unusually strong results?

### Soundness
2

### Presentation
2

### Contribution
3
