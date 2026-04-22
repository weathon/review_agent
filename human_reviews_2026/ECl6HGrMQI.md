# LoRaQ: Optimized Low Rank Approximated Quantization Error for 4-bit Quantization

- Avg Score: 6.50
- Decision: Reject
- Scores: 4, 6, 10, 6

## Abstract
Post-training quantization (PTQ) is essential for deploying large diffusion-based transformers on resource-constrained hardware. However, aggressive 4-bit quantization introduces significant degradation in generative performance. While existing solutions mitigate quantization error through outlier smoothing or rotation techniques, low-rank approximation methods that add auxiliary linear branches to each quantized layer represent a promising new paradigm. Yet, these approaches suffer from computational overhead due to the data movement required by full-precision (W16A16) branches, limiting practical deployment. In addition, data-based calibration contributes to the computational complexity of the quantization process, especially because search policies must evaluate many parameter configurations using a small calibration subset. We propose LoRaQ (low-rank approximated quantization), a data-free calibration approach to optimize quantization error compensation. This method can be used in composition with other PTQ models. LoRaQ further enables mixed-precision configurations by quantizing the low-rank branch itself, overcoming the limitations of prior work. While LoRaQ achieves superior quantization performance than state-of-the-art methods in their native W4A4 setting on PixArt-Sigma and SANA, it also allows for configurations such as W8A8, W6A6 and W4A8 for low-rank branch alongside a W4 main layer. This reduces data movement overhead and enables a fully quantized, hardware-efficient solution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a framework for low-rank quantization of diffusion models. Instead of performing standard weight quantization (W8A8, W4A8, etc.), the authors decompose each weight matrix into a low-rank factorization (LoRA-style), followed by quantization-aware optimization on the factorized matrices.

### Strengths
1. Conceptual Integration of LoRA and Quantization.  The idea of combining LoRA and quantization into a single unified framework is logical and potentially impactful. It effectively exploits the redundancy in diffusion U-Nets and transformer-based blocks.

2. Quantizing diffusion models is increasingly important for deployment on mobile or edge hardware. LoRaQ directly targets this problem, aligning with the trend of compute-efficient generative AI.

### Weaknesses
1. The first picture appears to be a non-vectorial image. It is recommended to convert it to a vectorial image.

2. The method is largely an engineering combination of existing paradigms, LoRA-style factorization and quantization-aware training, with a joint optimization loss. While useful, it lacks a novel theoretical component or formal analysis explaining why low-rank factorization improves quantization robustness.

### Questions
None

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
4

### Summary
This paper proposes LoRAQ, a data-free calibration approach to minimize the weight quantization error. The work extends from existing low-rank approximation method by proposing mixed-precision configuration with quantized low-rank branch. Quantization errors are further minimized by inserting a learned rotation matrix on the low-rank branch. The proposed method is further accelerated by system implementation optimizations.

### Strengths
1. Novelty-wise, this paper moves away from the common data-dependent calibration of block reconstruction and explores a weight-only calibration method. Though the use of low-rank branch and the rotation matrix insertion are well-known ideas, the overall framework remains novel.
2. The exploration on quantizing the low-rank branch opens up a new tradeoff between the rank and the quantization precision of the branch
3. Both quantitative and qualitative results are provided for multiple diffusion models showing the proposed method outperforming SVDquant baseline.
4. The paper has a clear presentation overall, easy to follow.

### Weaknesses
1. For the efficiency evaluation, though the paper claims improved hardware support by removing floating scales and micro-scaling formats, no real runtime measurements are provided in the evaluation to demonstrate the improved efficiency. A latency or throughput comparison needs to be conducted to justify the improvement.
2. The paper proposes multiple techniques, such as the new data-free calibration strategy, adding rotation to the low-rank branch, and performing different format of quantizations on the low-rank branch. However, ablation study is lacking to show the effect of each individual treatment. Ablation is especially needed to show the performance gain brought by the different calibration strategy and the rotation matrix.
3. The paper claims a fiar comparison with SVDquant. However, the proposed method utilizes additional rotation matrix, which may add additional overhead to the inference.

### Questions
Please provide additional results to tackle the three weaknesses mentioned in the previous section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
1

### Summary
I am unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers on 14 Oct 2025.

### Strengths
I am unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers on 14 Oct 2025.

### Weaknesses
I am unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers on 14 Oct 2025.

### Questions
I am unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers on 14 Oct 2025.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes LoRaQ, a data-free calibration approach to optimize quantization error compensation. With W4A4 settings, LoRaQ achieves optimal experimental performance compared to existing methods.

### Strengths
1. The results show that the proposed method significantly improves the metrics compared to existing SVDQuant methods, demonstrating its advantages.
2. The paper claims that they will release the PTQlibrary for transformer blocks, which will contribute to future research.

### Weaknesses
1. The paper mentions that a major advantage of LoRaQ is its model independence, which eliminates the need to calibrate datasets to determine low-rank matrices. This significantly simplifies the quantization process. However, the authors do not explain the performance gains resulting from this simplification. Are there any metrics that can quantify the benefits brought by the model?
2. The paper only conducted experiments on the PixArt-Σ and SANA models. Is it generalizable for other models with different architectures or different numbers of parameters?

### Questions
Please see Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3
