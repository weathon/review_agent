# SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models

- Avg Score: 5.40
- Decision: Reject
- Scores: 5, 5, 5, 6, 6

## Abstract
Large language models (LLMs) have achieved remarkable progress, but their extensive number of parameters results in high memory usage, significant loading latency, and substantial computational demands. To address these challenges, post-training quantization (PTQ) has emerged as an effective technique for compressing model weights. In the context of PTQ for LLMs, existing uniform quantization methods, though efficient in terms of memory and computational requirements, often struggle to maintain performance. In this paper, we propose SliM-LLM, a Salience-Driven Mixed-Precision Quantization scheme that achieves group-wise bit-width allocation with mixed precisions for efficient LLMs with high accuracy. Building on our observation that salient/important weights often follow a structured distribution, we incorporate two core components to preserve post-quantization performance in LLMs while maintaining efficiency: 1) Salience-Determined Bit Allocation adaptively assigns bit widths to groups within each layer based on their group-level salience, aiming to minimize the reconstruction error of activations; and 2) Salience-Weighted Quantizer Calibration optimizes quantizer parameters by incorporating element-level salience, ensuring that the most critical weights are preserved, further preserving important weights information. With its structured group partitioning, SliM-LLM offers a hardware-friendly quantization approach, maintaining computational and memory efficiency comparable to highly optimized uniform quantization methods. Extensive experiments demonstrate that SliM-LLM significantly improves the accuracy of various LLMs when quantized to ultra-low bit widths. For instance, a 2-bit quantized LLaMA-7B model achieves nearly 6x memory reduction compared to its floating-point counterpart, alongside a 48% reduction in perplexity compared to the leading gradient-free PTQ method, all while maintaining GPU inference speed. Furthermore, SliM-LLM+, which incorporates gradient-based quantizers, reduces perplexity by an additional 35.1%.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a group-wise mixed-precision quantization method, to solve the bottlenecks of mixed-precision LLMs. 
The main contribution covers two novel strategy. One is Salience-Determined Bit Allocation, named SBA. Before this, author implement many observations on Hessian values and weight salience. And determines the optimal bit-width allocation for different groups by minimizing the distance of information entropy with the original weight output. The other is Salience-Weighted Quantizer Calibration, which is named SQC, to enhance the information of significant weights within the group by amplifying the quantizer awareness of salient weight. By calibration parameter τ, optimize the quantizer, "liberating the perception interval during quantization".  Faced with problems of outlier in group, this method find a compromise, and expand the solution space of scale and zero point. The important point is scale sharing in one group. 
The SBA and SQC amentioned is supported in most quantizer, including static-based and gradient-based quantizers. By Experiment result, SliM-LLM can surpass most methods in LLAMA and OPT models.

### Strengths
1. By observation, introduce a method to find salient weight in certain channels. By KL divergence in output activation matrix, effectively minimize the recontruction error in quantization. 
2. Utilize 3 - σ rule to select salience part. By introducing τ, reduce quantization error.
3. The SQC share the parameter in group. without distinguishing weight in one group, it simplifies the memory and infer in hardware

### Weaknesses
1. The SBA Utilize KL divergence of block outputs as a precision allocating metric, depend on a certain calibration dataset. Generalization on datasets is not mentioned, so the method maybe effective in the same dataset. 
2. The SQC introduce a novel quantizer by introducing τ to reduce reconstruction during quantization. The 3 - σ rule and 1% ratio is based on a common observation, not a heuristic method. this theoretical basis is not very strong. Dynamic ratio by layer maybe better. 
3. This paper main solve the bottleneck in mixed-precision, and offers a hardware-friendly quantization. But from method and experiments section, hardware-friendly is not introduced. Including memory saving, time consuming, and superior to other method.
4.  Group-wise mixed-precision focus on different bits during same channel allocation in infer process. It is not mentioned in this paper.

### Questions
1. How to keep generalization on datasets, due to your observation or KL divergence, rely on calibration datasets. If I choose mmlu or humaneval, the observation is same?
2. The SQC introduce a novel method to reconstruct quantization.  what is the difference between salient weight and un-salient weight in your Formula (5), or when optimize τ, the weight or reconstruct error will be different?
3. Hardware-friendly mentioned not introduced in details. How does it save memory and time comsumption in infer process? And its advantages in this aspect, compared with PB-LLM, LLM-MQ?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces SliM-LLM, a framework for low-bit quantization of Large Language Models (LLMs). The key innovation is a mixed-precision approach that uses two main components:
1. Salience-Determined Bit Allocation: Hessian-guided assigns bit widths to groups of weights based on their importance
2. Salience-Weighted Quantizer Calibration: Optimizes quantization while preserving important weights

The method achieves significant compression (2-bit quantization) while maintaining model performance and ensuring hardware-friendly deployment. The framework is validated across multiple LLM families and tasks, showing practical benefits for deploying LLMs in resource-constrained environments.

### Strengths
1. Introduces an innovative two-component framework combining global group-level and local element-level salience
2. The structured approach is hardware-friendly while maintaining good performance. It achieves actual GPU acceleration with reasonable inference speed.
3. Comprehensive experiments across multiple LLM families, showing improvements over existing methods.

### Weaknesses
1. The equations in this paper have significant problems: 
- In Equation 2, the $\alpha$ is not used. The correct equation may be $\hat{w}_b=\alpha \text{sign}(w_f)$.
- In Equation 3, the equation does not hold, please confirm if it is an approximation.
- In Equation 4, using KL divergence for outputs is mathematically incorrect as they aren't probability distributions. Variables g1,...,gn are undefined. The definition of $w_f^{sba}$ needs proper notation (The superscript of $w^{b_0}$ cannot denote for bitwidth selection).
- In Equation 5, the braces are not use appropriately.

﻿2. The main problem of this paper is that the algorithm is poorly explained:
﻿- Critical algorithm details are hidden in appendix. Implementation details are unclear.
﻿- Salience-weighted quantizer calibration is not well-explained, which is hard to understand. In Section 3.2.2, the search space for s and z is undefined. And the role of τ in expanding solution space isn't clearly justified. The authour should clarify how τ expands the solution space and why this is beneficial. 

﻿3. Limited comparison with other extremely low-bit quantization methods. For instance, "Ma S, Wang H, Ma L, et al. The Era of 1-Bit LLMs: All Large Language Models Are in 1.58 Bits [J]. arXiv preprint arXiv:2402.17764, 2024." This method is popular and user-friendly, as it is supported by Hugging Face.

### Questions
1. Is PTQ practical for ultra-low bit quantization in the case of significant dropout?
2. Is there any details about how exactly does τ expand the solution space of s and z?
3. How do you justify using KL divergence as objective function?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes SliM-LLM, a quantization approach for LLMs that employs a salience-driven mixed-precision scheme. The key idea is to allocate bit-widths dynamically, giving more precision on model weights that are "salient". SliM-LLM introduces two main components: Salience-Determined Bit Allocation (SBA), which assigns bit-widths to groups within each layer based on their group-level salience to reduce reconstruction error, and Salience-Weighted Quantizer Calibration (SQC), which fine-tunes the quantization parameters by emphasizing highly influential weights. Experiments demonstrate SliM-LLM's effectiveness in various LLMs, while with noticeable slowdown (Token/s) on GPUs running 7B models.

### Strengths
The experiments are thorough, spanning multiple models and bit configurations.

The paper explains both the SBA and SQC components of the approach well, though some additional clarity in experimental detail would help.

### Weaknesses
Inference slow down: the paper suffers from a significant problem that the inference speed of the Llama2-7B quantized by SLIM-LLM is even slower than the fp16 baseline (Table 5). Users may as well not do any SLIM-LLM quantization: worse PPL and worse speed compared to fp16 baseline.

### Questions
The 2-bit PPL is still much higher compared to fp16, it's not practical for practical use yet. 4-bit weight is still the most widely used deployment case. Can the authors provide 4-bit PPL comparison with RTN, AWQ and GPTQ?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes salience-aware bit allocation and calibration methods for low-bit quantization of large language models (LLMs). Unlike conventional uniform bit allocation, this method allocates bit-width based on the weight salience of each group. Additionally, the quantizer calibration incorporates element-level salience, enhancing overall model quality.

### Strengths
•	The fine-grained bit allocation and calibration method demonstrates effectiveness, achieving improved quality compared to existing post-training quantization baselines.

•	The paper implements inference support for this new quantization algorithm.

### Weaknesses
•	The authors are suggested to evaluate and report the quantization cost, since the proposed approaches introduce searching overhead. 

•	While the theoretical expectation for inference speedup from 16-bit to 2-bit quantization is 8x, the actual inference speed of the proposed approach can even be slower than its FP16 counterpart. This makes the proposed approach less convincing. It may be beneficial to explore state-of-the-art low-bit inference frameworks, such as T-MAC, to see if the issue can be addressed.

•	Figure 4 lacks clarity in its legends and scales, making it difficult to interpret.

### Questions
Since low-bit quantization is popularly used for the LLM deployment on resource-limited user devices, such as laptops and mobile phones. How would you expect the results will be on these devices.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents "SliM-LLM," a quantization method for large language models (LLMs), focusing on salience-driven, mixed-precision quantization to enhance memory efficiency and maintain model performance at low bit-widths. The proposed framework integrates two main steps: Salience-Determined Bit Allocation (SBA), which dynamically allocates bit-widths to weights based on group-wise importance, and Salience-Weighted Quantizer Calibration (SQC), which aims to further improve quantization quality by adjusting quantizer parameters based on local salience. Overall experiments demonstrate that SliM-LLM significantly improves performance metrics (e.g., perplexity) across various model sizes with Llama series, outperforming several state-of-the-art quantization methods in both accuracy and efficiency.

### Strengths
- Clear problem definition and motivation
- Good visualization
- Comprehensive results in various model size

### Weaknesses
- The paper lacks a transparent discussion of potential limitations, such as possible trade-offs between compression rate and inference speed, or cases where the approach may not perform as well, such as for models with different architectural peculiarities. Acknowledging these limitations would strengthen the overall rigor of the paper.
- The experimental evaluation in this paper primarily focuses on the LLaMA series, with additional results on the OPT model series presented in the appendix. However, there is little discussion or empirical validation on architectures beyond these two families, which share many similarities in their transformer structures. This limited scope raises concerns about the generalizability of SliM-LLM across more diverse architectures. The absence of empirical results on these varied architectures leaves open the question of how well the proposed salience-based quantization strategies would perform in models with different salience and outlier channel dynamics. Extending experiments to a broader set of architectures would strengthen the claim of SliM-LLM’s robustness and applicability across LLMs.
- The proposed SliM-LLM approach is designed based on empirical observations of outlier channels and salience distributions in models like LLaMA. However, recent architectures such as Gemma2 incorporate normalization layers following both attention and feed-forward (FFN) modules, which could alter the distribution and behavior of outlier channels. These normalization layers might dampen or redistribute the significance of certain activations, potentially influencing the effectiveness of salience-based quantization strategies. This variation raises questions about how well SliM-LLM’s bit allocation and salience-weighted calibration would generalize to such architectures, where outlier handling may not align with observations made in models without inter-layer normalization. A more in-depth analysis would help determine if additional adaptations are required to support architectures that use different normalization schemes, as seen in models like Gemma2.

### Questions
- The paper briefly mentions spatial clustering of salient weights, but how does the model respond to cases where salience does not cluster as neatly? Are there adaptive mechanisms in SBA and SQC to handle such irregularities?
- Given the additional steps introduced in SBA and SQC, how does the computational cost compare to baseline quantization methods? Specifically, are there latency trade-offs that should be considered, especially for real-time applications?
- The paper focuses on transformers for text, but given the growing interest in multi-modal LLMs, could this quantization approach generalize to models that handle other data types or non-transformer architectures?
- While SliM-LLM claims to reduce overall memory usage, the paper lacks clarity on whether SBA and SQC introduce any hidden memory or storage overheads due to structured bit-width allocation. Further detail here would help assess the method’s efficiency.

### Soundness
2

### Presentation
2

### Contribution
2
