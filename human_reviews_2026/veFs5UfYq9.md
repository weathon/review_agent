# KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6, 4

## Abstract
Mixture of Experts (MoE) models have achieved great success by significantly improving performance while maintaining computational efficiency through sparse expert activation. However, their enormous parameter sizes and memory demands pose significant challenges for deployment in resource-constrained environments.
Vector Quantization (VQ) offers a promising approach for ultra-low-bit compression in Large Language Models (LLMs) by constructing and leveraging a codebook—where weight vectors are mapped to the most similar discrete codewords within the codebook. 
However, its direct application to MoEs suffers from significant performance degradation caused by two critical obstacles:  (1) redundant representation among experts leads to VQ repeatedly quantizing similar representations for each expert, resulting in inefficient utilization of the limited codebook capacity; and (2) cumulative outputs bias, amplified by expert aggregation, leads to distributional shifts in the quantized outputs, resulting in degraded model accuracy.
To this end, we propose KBVQ-MoE, a novel VQ framework to enhance extremely low-bit quantization for MoE-based LLMs. 
KBVQ-MoE introduces two lightweight and offline techniques that introduce negligible runtime computational and memory overhead:
(1) Input-driven redundancy elimination, where a Karhunen–Loève Transform (KLT) guided singular value decomposition (SVD) extracts and shares dominant weight components across experts. 
(2) Bias-corrected output stabilization, where vector quantization is applied to expert-specific (i.e., non-redundant) representations and the quantized outputs are corrected with channel-wise affine compensation.
Experiments on various MoE LLMs demonstrate that KBVQ-MoE preserves accuracy substantially better than existing quantization methods. 
For instance, 3-bit quantization of Qwen1.5-MoE-A2.7B achieves an average accuracy of 67.99, nearly identical to the FP16 baseline of 68.07, underscoring the potential of KBVQ-MoE for efficient deployment on edge devices and other resource-constrained platforms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes KBVQ-MoE, a post-training vector quantization framework for MoE LLMs combining a KLT-guided SVD to remove cross-expert redundancy (IDRE) with a lightweight channel-wise affine correction (BCOS) to stabilize outputs after VQ. The key idea is to map expert weights into an input-coherent basis via KLT, extract dominant shared components with SVD and keep them in full precision, then quantize only the expert-specific (i.e., non-redundant) representations and correct distributional shifts by mean–variance matching. The authors claim the affine correction is “not a heuristic adjustment but an unbiased MMSE-optimal estimator.”

### Strengths
* This paper is clearly motivated by two obstacles in MoE VQ, including redundant representation among experts and cumulative outputs bias.

* Empirical result looks promising, particularly at low bits.

* The method acts as an easy-to-use plugin that improves multiple VQ baselines and yields speedups.

### Weaknesses
* Evaluation tasks are limited. The evaluated tasks are normally for pre-trained models, but some of the models are post-trained models (e.g. Qwen3). I would love to see some post-training benchmark (e.g. MMLU, AIME24, HumanEval) results on the Qwen3 model.

* Some typos: (line 756) “A.5 ALL EXPRIMENTS” → “All Experiments.”; (line 81) "uantization" -> "quantization"; (line 105) "non-redundan" -> "non-redundant"

* Some of the related works on MoE quantization can be further discussed or compared with, e.g. [1] [2] [3].

(I'm more than happy to increase my score if my questions are adequately addressed)

[1] Kim, Y.J., Fahim, R. and Awadalla, H.H., 2023. Mixture of quantized experts (moqe): Complementary effect of low-bit quantization and robustness. arXiv preprint arXiv:2310.02410.

[2] Li, P., Jin, X., Tan, Z., Cheng, Y. and Chen, T., 2024. QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts. arXiv preprint arXiv:2406.08155.

[3] Duanmu, H., Li, X., Yuan, Z., Zheng, S., Duan, J., Zhang, X. and Lin, D., 2025. MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design. arXiv preprint arXiv:2505.05799.

### Questions
* How is `Qwen3-30B-A3B ` evaluated? Specifically, did you enable "reasoning" of this model? I would love to see the quantization results on a reasoning LLM.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel quantization method, KBVQ-MoE, designed for MoE-based large language models. The method first applies the KLT and SVD to extract dominant weight structures shared across experts, thereby reducing redundancy and improving codebook utilization. Building on this foundation, it employs vector quantization for expert-specific representations and further proposes a channel-wise affine compensation module to refine the mean and standard deviation of the quantized outputs. Extensive experiments demonstrate superior performance across multiple datasets and LLM architectures.

### Strengths
1. Figures 2 and 3 provide highly illustrative examples that effectively support the paper’s claims on redundancy extraction and channel-wise bias correction.
2. The overall motivation is solid and clearly articulated. The proposed method is both conceptually sound and straightforward, making it easy to understand and reproduce. Moreover, it successfully addresses two critical challenges outlined in the introduction.
3. Table 5 convincingly demonstrates that the proposed IDRE and BCOS modules can be seamlessly integrated into other MoE quantization approaches, highlighting the method’s flexibility and potential for broad applicability.

### Weaknesses
1. Some mathematical symbols should be revised (e.g., matrices should be boldfaced). Additionally, minor grammatical and formatting issues should be corrected, and certain technical details could be described more clearly.
2. The contribution of each component within the IDRE and BCOS modules remains unclear due to the lack of quantitative results and in-depth analysis in the ablation study.
3. The comparison with other MoE-based compression methods, such as EAC-MoE, D2-MoE, and SubMoE, which are illustrated in the related work, should be included for a more comprehensive comparison.

### Questions
1. Is the IDRE technique applied exclusively to router experts or to all experts? The notation in Eqs. (1) and (3) suggests it concerns router experts, while the workflow in Eq. (2) seems to encompass all experts. Clarification on whether the symbol $n$ refers to different concepts across equations would be helpful.
2. What is the rationale for maintaining the shared structure at full precision while quantizing only the expert-specific weights? Why not quantize the shared weights as well?

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
This paper proposes KBVQ-MoE, a vector quantization framework designed specifically for compressing Mixture-of-Experts (MoE) language models. The method addresses two key challenges: (1) redundant representations across experts that waste codebook capacity, and (2) cumulative output bias from quantization errors that get amplified through expert aggregation. The approach combines Input-Driven Redundancy Elimination (IDRE), which uses KLT-guided SVD to extract and preserve shared components at full precision, with Bias-Corrected Output Stabilization (BCOS), which applies vector quantization only to expert-specific weights and corrects distributional shifts via channel-wise affine transformations. Experiments on models like Qwen and Mixtral show strong performance at 2-3 bits, with 3-bit Qwen1.5-MoE-A2.7B achieving near-FP16 accuracy.

### Strengths
1. The paper makes a novel contribution by adapting vector quantization specifically for MoE architectures. The identification of expert redundancy and amplified quantization bias as key bottlenecks is insightful, and the KLT-guided SVD approach creatively aligns weight decomposition with input activation statistics.
2. The technical approach is sound with theoretical justifications provided in the appendices. The experimental evaluation is comprehensive, covering multiple MoE models with thorough ablations. Results show consistent and meaningful improvements at ultra-low bit-widths (2-3 bits), where baseline methods struggle significantly. The modular design demonstrates compatibility with existing VQ methods, increasing practical impact.

### Weaknesses
1. The paper mentions "negligible" computational overhead but provides limited quantitative analysis. How long does the KLT-SVD calibration take compared to standard VQ? What is the actual inference-time cost of the channel-wise bias correction operations? These practical considerations matter for deployment. Could you provide some results on this, like time cost of quantization method.
2. The baseline methods, especially MoEQuant, show surprisingly poor performance at 2-bit in Table 1 (e.g., W2 of 583542 for Qwen1.5). More discussion on this failure would help.
3. The choice of truncated rank k=n/128 appears empirically driven from Table 4 but lacks theoretical justification. While the ablation shows diminishing returns beyond this point, why this specific ratio is optimal across different models and tasks remains unclear. The paper would benefit from analysis connecting rank selection to properties of the expert weight matrices or input distributions.
4. The evaluation relies on simple zero-shot reasoning tasks (ARC, HellaSwag, PIQA, etc.) that may not fully capture model capabilities. Including more challenging benchmarks like MMLU, MATH, and code generation tasks (MBPP, EvalPlus) would better demonstrate the method's effectiveness across diverse domains.
5. Table 1 only shows 2-bit and 3-bit results. Adding 4-bit and 8-bit comparisons would provide a more comprehensive results, especially since 4-bit quantization is common in practice.

### Questions
1. typo: Qwen1.5-Moe-A2.7B->Qwen1.5-MoE-A2.7B

### Soundness
3

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
The paper proposes a vector quantization approach targeted for mixture of expert layer. The idea is to extract redundant representation across experts and keep it in higher precision while expert specific components are vector quantized. Within vector quantization of expert specific components, scaling and bias is applied to improve quantization. The paper is well written and easy to understand. The techniques proposed are intuitive and I appreciate the authors providing actual hardware speedup numbers. The paper is missing some recent non linear quantization baselines and comparison at iso-compression.

### Strengths
1. Paper is easy to understand and well written.
2. Technique proposed is intuitive.

### Weaknesses
1. The paper is missing comparison with recent non linear quantization baselines : VPTQ (https://arxiv.org/abs/2409.17066), AQLM (https://arxiv.org/pdf/2401.06118), QUIP (https://arxiv.org/pdf/2307.13304), QUIP# (https://arxiv.org/pdf/2402.04396), SqueezeLLM (https://arxiv.org/pdf/2306.07629), GPTVQ (https://arxiv.org/pdf/2402.15319),  etc.
2. Among the baselines presented, the compression achieved by various techniques is missing. 
3. Iso-compression results are missing.
4. Evaluation on complex tasks is missing : math understanding, coding, reasoning, long context abilities, etc.

### Questions
1. How does this approach compare with ResQ (https://openreview.net/pdf?id=4qIP1sXcR1)? Although ResQ does activation quantization as well and integer quantization of weights, it also uses eigen value decomposition to isolate high precision components. How does this approach compare with using IDRE proposed in this work?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents KBVQ-MoE, a vector quantization framework tailored for ultra-low-bit compression of Mixture-of-Experts (MoE) large language models (LLMs). The study is motivated by the substantial performance drop observed when conventional quantization methods are directly applied to MoE architectures. In particular, KBVQ-MoE mitigates expert redundancy and output bias through input-driven redundancy elimination and bias-corrected output stabilization mechanisms.

### Strengths
1. The motivation is clear and convincing, highlighting MoE-specific issues of expert redundancy and output bias.

2. The proposed KBVQ-MoE is validated on several representative MoE architectures, demonstrating consistent improvements.

### Weaknesses
1. While IDRE and BCOS are ablated individually, there is no fine-grained study of codebook size sensitivity.

2. Lack more advanced or concurrent MoE-aware compression baselines (e.g., D2-MoE, SubMoE mentioned in related work).

3. The evaluation of computational efficiency is insufficient. The paper only reports a simple “Decoder speed test” in Table 6, without providing detailed analysis of computational or memory overhead.

4. The core motivation of the paper lies in the claim that redundancy elimination and bias correction help stabilize expert output distributions; however, the supporting evidence (e.g., Fig. 2–3) is insufficient. It would be more convincing if the authors compared other methods reported in Table 1 to quantitatively validate the claimed effect.

5. The paper’s presentation lacks rigor in notation and consistency. For instance, the dimension oc in Step 2 is undefined, and the superscript in Equation (3) for the routing expert is unexplained. There are also typos (e.g., uantization → quantization) and inconsistent use of MoE/moe.

6. The paper does not explicitly discuss the limitations of the proposed method.

### Questions
Given that the calibration set contains only 256 samples from RedPajama, could the authors provide ablation results on calibration size to verify its representativeness for computing reliable KLT statistics and bias correction factors? Additionally, how do the authors ensure that no potential data leakage occurs during calibration?

### Soundness
3

### Presentation
2

### Contribution
2
