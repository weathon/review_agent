# LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Large multimodal models (LMMs) have achieved impressive performance on various vision-language tasks, but their substantial computational and memory costs hinder their practical deployment. Existing compression methods often decouple low-rank decomposition and quantization, leading to compounded reconstruction errors, especially in multimodal architectures with cross-modal redundancy. To address this issue, we propose LLaVA-FA, a novel efficient LMM that performs joint low-rank plus quantization approximation in the frequency domain. By leveraging the de-correlation and conjugate symmetry properties of Fourier transform, LLaVA-FA achieves more compact and accurate weight representations. Furthermore, we introduce PolarQuant, a polar-coordinate quantization method tailored for complex matrices, and an optional diagonal calibration (ODC) scheme that eliminates the need for large-scale calibration data. Extensive experimental results demonstrate that our proposed LLaVA-FA outperforms existing efficient multimodal models across multiple benchmarks while maintaining minimal activated parameters and low computational costs, validating its effectiveness as a powerful solution for compressing LMMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an efficient LMM that performs joint low-rank plus quantization approximation in the frequency domain to exploit de-correlation and conjugate symmetry, leading to more compact and accurate weight representations. Additionally, authors introduce PolarQuant that separately discretizes amplitude and phase in polar coordinates. Moreover, in order to eliminate the need for large-scale calibration data, authors derive an optional diagonal calibration scheme that approximated the Hessian with row/column means.

### Strengths
- The paper, for the first time, integrates low-rank + quantization optimization in the fourier domain. 

- The proposed method using fourier approximation is elegant and has solid mathematical justification.

- Clear pseudocode for algorithms are provided (line 233 - 252).

- Superior performance achieved across different benchmarks.

### Weaknesses
- The novelty is limited somehow and while Fourier-domain decomposition is interesting, it may be seen as a direct extension of existing LoRA + quantization frameworks.

- Comparison with related baselines (QLoRA[1]) is underdeveloped.

- Some qualitative or visualization results may enhance the presentation.

- The ODC heuristic (row/column averaging) lacks strong theoretical or empirical justification. It’s unclear when this approximation fails or how sensitive the method is to distribution shifts.

[1] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.

### Questions
- Referring to line 066 *"we observe that the weight matrices of LMMs in the frequency domain have a more **compact** spread of singular values as compare to spatial domain."* , what is "compact spread of singular values"

- The proposed method achieve exceptional performance in hallucination benchmark and can you elaborate the reason

- Since important contribution of the paper is reducing the computational and memory costs, it would be better to move analysis to the main paper.

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
This paper targets on efficient Large Multimodal Model (LMM). They bring Fourier approximation to decomposes the weight matrices into a
low-rank plus quantized weight, designing a efficient LMM framework called LLaVA-FA. They also design PolarQuant which is an amplitude-and-phase polar codec to quantize complex matrix, and an optional diagonal calibration (ODC) scheme to approximate Hessian Matrix. Extensive experiment results prove the effectiveness of the proposed method.

### Strengths
1. This paper is written in a very high quality, the figures analyze the problem and illustrate the idea very clearly, especially Figure 1 and 3. 
2. I think this paper targets on an important problem, the efficient LMM. 
3. The idea is interesting and reasonable. I am happy to see Fourier approximation can be applied to LMM since it really has some good characteristics. 
4. The experiments are abundant and clear, proving the effectiveness of the proposed method.

### Weaknesses
1. Just one discussion. This paper choose Fourier approximation, and can we consider other type of approximation? I am happy to see more comparison results. 
2. The authors can have more discussions about the limitations and future work.

### Questions
See weaknesses, especially discussions about the limitation and future work.

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
The paper proposes LLaVA-FA, an integrated compression pipeline that merges low-rank and quantization in Fourier space, accompanied by a concrete algorithmic recipe and a complexity-aware design. More specifically, LLaVA-FA applies a 2D Discrete Fourier Transform (DFT), factorizes the largest spectral components with a low-rank complex SVD, quantizes the residual using a polar-coordinate codec (PolarQuant), and optionally weights the reconstruction objective with a diagonal calibration derived from row and column statistics (ODC). The experiments demonstrate favorable performance at small scales, accompanied by measured efficiency gains in latency, FLOPs, and KV-cache usage.

### Strengths
1. The proposed LLaVA-FA is well-motivated, and supported by a clear theoretical framing.

2. Efficiency evidence with concrete measurements. Specifically, latency, FLOPs, KV-Cache usage, and TTFT are reported, which aligns well with the goals of a model compression study.

### Weaknesses
1. Limited ablation for ODC and calibration choices. The paper mentions that ODC removes the need for large calibration sets, but no direct comparison or ablation is presented to isolate this effect. Adding such results would make the claim more convincing.

2. There are some fairness concerns regarding the baseline comparisons in Table 1. The amount of training data varies across methods, and some baselines use fewer samples than LLaVA-FA while achieving comparable performance.

3. Discussion on a few related works seem to be missing. For instance, in the line of LMM efficiency “CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers, ICML 2024,” and in the line of vision-language weight compression “UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers, ICML 2023.”

### Questions
1. Do the authors plan to release their models?

2. The current results are mainly on small-scale models. Would the authors consider including results on larger models to demonstrate scalability?

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
This paper proposes LLaVA-FA, a compression framework for large multimodal models (LMMs) that addresses the limitations of existing methods, where decoupled low-rank decomposition and quantization often lead to compounded reconstruction errors. LLaVA-FA employs Fourier approximation to integrate low-rank decomposition and quantization within the frequency domain, leveraging two essential properties of the Fourier transform: de-correlation, which reduces spectral redundancy, and conjugate symmetry, which nearly halves parameter storage.To handle complex matrices in the frequency domain, the paper introduces PolarQuant—a polar-coordinate quantization method that discretizes amplitude and phase separately to preserve complex structure. It also proposes an Optional Diagonal Calibration (ODC) scheme, which approximates the full Hessian with row/column means to avoid reliance on large-scale calibration data.

### Strengths
This paper proposes LLaVA-FA, a compression framework for large multimodal models (LMMs) that addresses the limitations of existing methods, where decoupled low-rank decomposition and quantization often lead to compounded reconstruction errors. LLaVA-FA employs Fourier approximation to integrate low-rank decomposition and quantization within the frequency domain, leveraging two essential properties of the Fourier transform: de-correlation, which reduces spectral redundancy, and conjugate symmetry, which nearly halves parameter storage.To handle complex matrices in the frequency domain, the paper introduces PolarQuant—a polar-coordinate quantization method that discretizes amplitude and phase separately to preserve complex structure. It also proposes an Optional Diagonal Calibration (ODC) scheme, which approximates the full Hessian with row/column means to avoid reliance on large-scale calibration data.

### Weaknesses
1. The paper proposes performing low-rank decomposition and quantization in the frequency domain via the Fourier transform, but the experimental section lacks comparisons with existing solutions in the spatial domain, limiting the credibility of its claimed competitiveness.
2. The method shows limited generalization capability, as experiments are only conducted on 3B and 7B-scale LLMs (Qwen-2.5) without evaluation on larger-parameter models.

### Questions
1. How does the proposed method perform when extended to models beyond Qwen-2.5?
2. How is the calibration matrix C constructed in the paper, and has there been any ablation study on the amount of calibration data used?
3. Could you elaborate on the compression time required for models of different sizes?
4. The algorithm flow in the paper provides an option to disable ODC—how does the performance change when ODC is not used?

### Soundness
2

### Presentation
3

### Contribution
2
