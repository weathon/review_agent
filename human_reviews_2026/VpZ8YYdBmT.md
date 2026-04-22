# Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Large language models (LLMs) demand extensive memory capacity during both fine-tuning and inference. To enable memory-efficient fine-tuning, existing methods apply block-wise quantization techniques, such as NF4 and AF4, to the network weights. We show that these quantization techniques incur suboptimal quantization errors. Therefore, as a first novelty, we propose an optimization approach for block-wise quantization. Using this method, we design a family of quantizers named 4-bit block-wise optimal float (BOF4), which consistently reduces the quantization error compared to both baseline methods. We provide both a theoretical and a data-driven solution for the optimization process and prove their practical equivalence. Secondly, we propose a modification to the employed normalization method based on the signed absolute block maximum (BOF4-S), enabling further reduction of the quantization error and empirically achieving less degradation in language modeling performance. Thirdly, we explore additional variations of block-wise quantization methods applied to LLMs through an experimental study on the importance of accurately representing zero and large-amplitude weights on the one hand, and optimization towards various error metrics on the other hand. Lastly, we introduce a mixed-precision quantization strategy dubbed outlier-preserving quantization (OPQ) to address the distributional mismatch induced by outlier weights in block-wise quantization. By storing outlier weights in 16-bit precision (OPQ) while applying BOF4-S, we achieve top performance among 4-bit block-wise quantization techniques w.r.t. perplexity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new 4-bit data format called FOF4, which leverages the asymmetric property of maximum/minimum values within a block to reduce the waste of degrees of freedom in the quantization codebook. This method can be regarded as a type of non-uniform quantization.

### Strengths
This method leverages the asymmetry of the boundaries in weight quantization to reduce quantization error.

### Weaknesses
1. This method can be regarded as a special type of non-uniform quantization method (1D codebook). Additionally, compared with other KMeans-based methods such as GPTVQ and RPTQ, it may not have advantages in terms of accuracy and speed (without hardware support).
2. This method lacks a comparison with similar basic codebook-based methods like GPTVQ，VPTQ...
3. Current LLMs can achieve W4A4 quantization with almost no loss, which has greater advantages in reducing computational overhead. This method seems unable to achieve this, and it also requires additional unstructured outlier storage and computation.

### Questions
Methods that previously determined normalization constants based on MSE have also achieved good results. Why is the weight based on the maximum absolute value chosen instead? What advantages does this approach offer?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel 4-bit block-wise quantization algorithm called BOF4-S. The proposed algorithm employs an enhanced EM algorithm to obtain the optimal reconstruction levels. Additionally, it normalizes the weights using the signed absolute block maximum, which effectively saves one reconstruction level, thereby adding an extra degree of freedom to the codebook and further reducing quantization error. Moreover, this algorithm excludes outlier weights and store them separately to further enhance the quantization accuracy. Experimental results demonstrate that the proposed algorithm achieves less quantization error and perplexity compared to the conventional methods, while also achieving superior model performance before and after fine-tuning.

### Strengths
1. This paper provides a thorough and clear explanation of the method, making it easy for readers to understand. The experimental results are comprehensive, which strongly supports the effectiveness of the proposed method. 

2. The method addresses several practical issues in current quantization algorithms, such as the wastage of reconstruction levels and the impact of outliers on quantization accuracy. By cleverly saving reconstruction levels, designing an optimized codebook algorithm, and eliminating outliers, it provides an ingenious approach to optimizing quantization algorithms, further enhancing the quantization accuracy.

3. This method achieves improvements in both quantization error and perplexity with minimal time overhead. Furthermore, the LLMs quantized using this algorithm demonstrate better task performance before and after fine-tuning compared to traditional methods, showcasing the method's strong applicability.

### Weaknesses
Although selecting one of the two endpoints as the reconstruction level for the maximum absolute weight provides an additional degree of freedom for the codebooks, it also requires an extra bit to store the sign of this maximum value. Is this overhead justified? Especially since, when using Llama-3.2-3B as the base model, there is no performance improvement on most tasks (Table 2).

### Questions
1. Is it possible for both endpoints of the normalized weights (-1 and 1) to appear at the same time? If so, how is the sign handled?
2. When outliers are removed, their corresponding values in the tensor are replaced with 0. Is this tensor modified before or after normalization? Does directly replacing values with 0 impact the distribution of weights within the current block, potentially interfering with the selection of the optimal codebooks?

### Soundness
3

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
This paper presents a comprehensive study on improving 4-bit block-wise quantization for Large Language Models (LLMs). The authors identify a fundamental issue in existing methods (e.g., NF4, AF4): they optimize the quantization error of the *normalized* weights rather than that of the *original* weights, leading to suboptimal results. To address this, the authors propose an improved Expectation-Maximization (EM) algorithm that directly optimizes the error of the original weights, resulting in a family of quantizers termed BOF4. They also introduce signed absolute maximum normalization, which frees up one codebook entry to enhance representational capacity. Additionally, a mixed-precision scheme is proposed to identify and handle outlier weights. The paper is strongly supported by rigorous mathematical derivations and extensive experiments on multiple LLMs and tasks (both inference and fine-tuning), demonstrating consistent and significant performance improvements over strong baselines.

### Strengths
- **Theoretical Soundness and Novelty:** The paper's core insight—optimizing the end-to-end quantization error of the *original* weights rather than that of the *normalized* weights—is profound and well-articulated. The derivation of new centroid update rules for Lloyd's algorithm (applicable to both MSE and MAE) constitutes a solid theoretical contribution. Clear and comprehensive mathematical proofs further solidify the theoretical foundation.
- **Holistic Methodological Framework:** The paper extends beyond a single idea by introducing a suite of complementary techniques: an optimal codebook based on reconstruction loss (BOF4), a signed normalization scheme (BOF4-S), and a practical outlier-preserving mechanism (OPQ). This integrated approach effectively enhances quantization accuracy from multiple perspectives.
- **Thorough and Convincing Experiments:** The evaluation is extensive, covering multiple model families (Llama, Qwen, Mistral), both inference and fine-tuning (QLoRA) scenarios, and a wide range of benchmarks (perplexity, NLP tasks, code generation). The results consistently show that the proposed methods, especially BOF4-S with OPQ, outperform the baselines.
- **Practical Impact and Reproducibility:** The methods are directly applicable for memory-efficient LLM deployment and fine-tuning. The paper provides optimized codebooks in the appendix and discusses integration with data-aware PTQ methods like GPTQ (Appendix I), enhancing its practical utility and reproducibility.

### Weaknesses
- **Assumption of Gaussian Weight Distribution:** The optimization of the BOF4 codebook relies on the assumption that network weights are Gaussian-distributed. Although Appendix C provides justification that most blocks are indeed Gaussian, especially after OPQ, the performance on models or layers with significantly non-Gaussian weight distributions remains less explored. This could limit the generalizability to certain architectures.

- **Overhead of OPQ:** Although OPQ is shown to have minimal runtime overhead (Appendix G.3), it introduces additional memory overhead for storing the outlier indices and values. A more detailed analysis of this memory-cost/accuracy trade-off, especially for very large models, would be beneficial.

### Questions
- **Q1: Gaussian Distribution Assumption**: The EM algorithm used for BOF4 relies on an assumed Gaussian distribution. How sensitive is the final performance to deviations from this assumption? If a model's weights do not follow a Gaussian distribution, what would be the magnitude of deviation this algorithm might cause?

- **Q2: Selection of Hyperparameter q**: Regarding OPQ, the hyperparameter `q=0.95` was chosen via a limited search. Could you discuss the sensitivity of the results to the choice of `q`? Is this value generally robust across different models and sizes, or does it need tuning?

- **Q3: The Special Sign Bit:** The signed normalization in BOF4-S requires storing the sign of the block maximum if double quantization is applied, as noted in Appendix A. Could you quantify the potential performance degradation if this extra bit is not used and the standard double quantization scheme is applied naively?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a novel approach to improving block-wise training for large language models (LLMs) by introducing refined optimization and scheduling mechanisms that better capture inter-block dependencies during training. The authors demonstrate that their method enhances both convergence stability and downstream task performance, offering a computationally efficient alternative to full end-to-end fine-tuning. The study is well-motivated, clearly written, and supported by extensive experimental validation across multiple benchmarks.

### Strengths
1. The proposed method provides an original perspective on block-wise training, addressing the often-overlooked issue of gradient inconsistency across blocks.

2. The formulation is mathematically rigorous, with theoretical justification for the proposed scheduling strategy.

3. The experiments span several model sizes and datasets, showing consistent improvement over baselines such as layer-wise and progressive tuning.

3. The approach maintains efficiency advantages (reduced memory and training cost) while achieving comparable or superior results to full fine-tuning, highly relevant for real-world large-scale model adaptation.

4. The paper is well-structured, with clear motivation, methodology, and ablation analysis that enhances understanding.

### Weaknesses
1. While the approach performs well on benchmark datasets, it would be useful to see how well it generalizes to non-language tasks (e.g., multimodal or code models).

2.  The method involves scheduling parameters whose influence is only briefly discussed; more detailed robustness analysis would strengthen the contribution.
3. Although the method is efficient, the paper could provide clearer quantification of the additional cost introduced by the new scheduling mechanism relative to vanilla block-wise training.

### Questions
How sensitive is the performance to the choice of block partitioning (e.g., number of layers per block)?

Could the proposed inter-block dependency modeling be integrated with adapter-based fine-tuning approaches?

Have the authors considered evaluating the method’s stability when applied to continual learning or streaming data scenarios?

### Soundness
3

### Presentation
3

### Contribution
4
