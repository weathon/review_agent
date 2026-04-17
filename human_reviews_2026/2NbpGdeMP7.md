# Boost Post-Training Quantization via Null Space Optimization for Large Language Models

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Existing post-training quantization methods for large language models (LLMs) offer remarkable success. However, the increasingly marginal performance gains suggest that existing quantization strategies are insufficient to support the development of more compressed models. To inspire new directions for future research, this paper introduces the concept of \textbf{null space} into LLMs quantization. We argue that the quantization error can be effectively alleviated by constraining the post-quantization weight perturbation to lie within the null space of input activations. To prove this fresh idea, we propose an intuitive projection method on several PTQ baselines to validate whether the performance will be further improved. Specifically, we devise an efficient and accurate null space projection approximation tailored to the characteristics of LLMs, and then theoretically derive a closed-form solution for an equivalent vector of the obtained projection matrix to satisfy practical inference condition. When validating our method on several milestone PTQ baselines, further performance improvements can be noticed obviously, demonstrating the novel perspective of null space optimization for LLMs quantization is effective. We view this paper the first step to alleviate the quantization error based on the insights of null space, hoping it inspiring future researchers to design more advanced quantization methods. Codes are available at \url{https://anonymous.4open.science/r/q2n-2236}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel post-training quantization (PTQ) optimization approach for large language models (LLMs) by leveraging null space theory, aiming to address the unavoidable quantization error in existing PTQ methods.
The core insight is that constraining the post-quantization weight perturbation $(W-W_q)$  to lie within the null space of input activations can effectively alleviate quantization error, as it minimizes the impact of such perturbation on the final output $(||(W-W_q)x||^2_2)\approx 0$. To validate this insight, the authors propose a framework with 3 key components: eplaces computationally expensive SVD with QR-based eigenvalue decomposition to handle large activation matrices in LLMs; Accurate null space Approximation with a threshold t, addressing the lack of exact zero singular values in LLM activations. Derives a vector $\alpha$ (with regularization coefficient $\lambda=0.2$ to integrate null space optimization.

### Strengths
1. Novel Theoretical Perspective: The paper introduces a fresh and meaningful direction for LLM post-training quantization (PTQ) by leveraging null space theory, addressing a key limitation of existing methods—their inability to mitigate the negative impact of unavoidable numerical errors beyond minimizing quantization error. This perspective, which links weight perturbation constraints to the null space of input activations, fills a gap in current quantization research and provides a new guideline for developing advanced quantization techniques.
2. The framework this paper proposed incorporates three well-designed components to ensure practicality for LLMs. These designs balance performance and efficiency, making the method applicable to real-world LLM inference scenarios. This paper not only introduces the null-space paradigm but also optimizes both speed and accuracy by addressing computational costs and the issue of zero singular values. These designs are orthogonal to existing quantization approaches and represent a plug-and-play new mechanism.
3. The paper conducts thorough experiments across diverse LLMs (e.g., LLaMA3, LLaMA3.1, Qwen2.5), multiple PTQ baselines (e.g., GPTQ, QuIP, LeanQuant), and different quantization scenarios (weight-only, weight-activation). Results consistently show improvements in both language generation (perplexity reduction) and downstream reasoning (zero-shot accuracy increase), demonstrating the method’s generality and effectiveness. Additionally, comparative experiments (e.g., vs. SVD, Adam-NSCL, backpropagation) further validate the advantages of the proposed components.

### Weaknesses
1. The paper acknowledges that the performance improvements brought by the framework are relatively moderate,  especially compared to some state-of-the-art PTQ methods that already achieve near-lossless results at lower bits. Additionally, the test sets mentioned in the paper are overly simplistic and focused exclusively on prefill-type tasks; it remains unclear how the method performs on generation-type tasks, such as GSM8K and HumanEval.

2. The framework relies on manual selection of key hyperparameters (e.g., the ratio threshold t for null space approximation and regularization coefficient $\lambda$ for the closed-form solution) via coordinate descent or grid search. The paper notes this as a limitation, as manual tuning not only increases the complexity of deployment but also may lead to suboptimal performance across different LLMs or quantization baselines, lacking an adaptive hyperparameter optimization mechanism.

3. The paper's approach primarily relies on estimating the null space of input samples, which inevitably makes it highly dependent on the quality of the calibration dataset. While it is undeniable that the method can achieve excellent performance by overfitting to a specific calibration set, its generalization to different scenarios remains questionable—the adaptability of the null space across diverse scenarios appears to be a fundamental challenge that cannot be easily overcome.

I’ll consider raising my score if you can solve my concerns.

### Questions
1. How can the inconsistency of null spaces across different calibration sets be addressed, so that the quantized model performs well not only on one or two specific tasks but generalizes across a broader range of tasks?

2. The notation in the paper is somewhat unclear. For example, in Section 3.2 ("Accurate Null Space Approximation"), both U1 and U2 appear simultaneously—what is the distinction between them, and what role does U2 play, especially since subsequent discussions primarily focus on U1?

3. Are the hyperparameters used in the paper generally applicable? If applied to different large language models, would this approach incur repeated hyperparameter tuning costs?

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
5

### Summary
The paper “Boost Post-Training Quantization via Null Space Optimization for Large Language Models” presents a post-training quantization (PTQ) correction method designed to mitigate quantization errors in large language models. The proposed approach identifies the null space of activations for each layer and formulates an optimization problem that approximates this null-space projection using a vector–matrix Hadamard product.

### Strengths
- The proposed solution has the original element of applying the null-space correction to the quantization error.
- The proposed quantization correction can be seamlessly integrated into existing quantized inference pipelines.
- It introduces no additional latency or memory overhead since it can be merged into the per-channel scaling vector.
- Experimental results demonstrate consistent and significant improvements across various quantization methods, supporting the effectiveness and generality of the approach.

### Weaknesses
- The relationship between eigenvalue decomposition and singular value decomposition of the covariance matrix is well established and widely used. Therefore, the “efficient eigenvalue decomposition” component should not be considered a novelty contribution.
- It would strengthen the paper to include evaluations on quantization methods that explicitly consider activation statistics, such as AWQ. It is not clear, whether such methods can benefit from the proposed approach. 
- The WA quantization results appear unimpressive. Since the perplexity metric is highly sensitive, the results in Table 2 do not convincingly indicate meaningful improvements. The only notable exception is the LLaMA3-70B QuaRot result in Table 7, which shows a substantial difference. This raises the question of whether QuaRot was applied correctly in that case.
- There are a list of inconsistencies or logical errors throughout the paper that hinder the paper clarity: 
- a. Since the covariance matrix is positive semi-definite, its eigenvalues are non-negative. Therefore, taking the absolute value of eigenvalues (lines 177–178) is unnecessary.
- b. The equation in line 668 is invalid when w is not a scalar. The proof should adopt a different formulation.
- c. If a is a vector, the regularization term (a−1) should be expressed as an l2 norm. 
- d. The Hadamard product in Equation (8) should be explicitly denoted to avoid ambiguity.
- e. Figure 3 would be more interpretable as a bar plot or an alternative visualization.


The paper presents strong experimental results and a well-executed quantization correction method; however, the underlying idea offers limited originality, bearing resemblance to existing activation-aware quantization approaches. The use of null-space optimization introduces a degree of novelty, but overall, the contribution feels somewhat incremental for a full-length conference paper. The evaluation could be strengthened by including comparisons with methods such as AWQ to better demonstrate general applicability. Additionally, refining the approach for WA quantization and improving the clarity of exposition, as noted in the comments, would enhance the paper’s quality and impact.

### Questions
- Can you provide the impact of your method with activation-aware quantization approaches like AWQ or smooth quant? 
- When comparing the closed-form and BP solutions, the optimization objectives differ. Specifically, the BP objective (line 458) omits the regularization term present in Equation (8). Please clarify this discrepancy.
- How many samples from the Wiki and C4 calibration datasets were used to compute the null space of X? Table 3 indicates that computing the null space for a single transformer block takes approximately 4 seconds—please specify how many samples were used for that estimation.
- Please, correct the errors pointed out in the "Weaknesses" section.

### Soundness
2

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
This paper proposes a novel perspective on post-training quantization (PTQ) in LLM: constraining the quantization error to the null space of the input activations to mitigate error propagation. The authors present an exemplary implementation—Q2N (Quantize-to-Nullspace)—combined with baselines such as GPTQ, QuIP, LeanQuant, and QuaRot, demonstrating consistent but limited improvements in language generation and downstream tasks.

Overall, this is an interesting mathematical perspective, but the method itself is rather empirical, lacking theoretical depth and experimental persuasiveness, making it a typical "concept-first but engineering-light" paper. It's more of an inspirational exploration than a technology that can truly change the landscape of quantization.

### Strengths
1. Introducing null space theory into quantization error analysis is indeed a first in LLM quantization literature (previously it was mostly used in LoRA-Null, AlphaEdit, etc.), and it is quite inspiring.
2. The core lemma (if the error is in the input null space, the output error approaches zero) is logically rigorous and its derivation is complete.
3. Complete derivation, pseudocode, and open-source links are provided, along with engineering execution specifications.
4. Compatibility has been verified through testing on multiple frameworks, including GPTQ, QuIP, LeanQuant, and QuaRot.

### Weaknesses
### **1. Limited contribution and marginal improvements**  
While the proposed Null Space optimization approach is novel, its methodology is overly simplistic, merely adding an approximate projection operation to existing frameworks such as GPTQ. Experimental results show limited performance improvement—only 1–3 percentage points for most tasks, and no significant improvement for some metrics.

### **2. Over-simplified projection mechanism**  
The authors claim to reduce error propagation by projecting the quantization error (W − Wp)  onto the null space of the input activation. However, in the actual implementation, only a channel-level vector α ∈ ℝ^{C_out} is introduced to replace the entire projection matrix. This substitution is overly simplistic, equivalent to multiplying the output dimension by a scaling factor. It fails to effectively model the high-dimensional structure of the null space and cannot truly guarantee  **(W − Wp)X ≈ 0**.  Therefore, this implementation is more like a "numerical correction" than a true projection optimization.
### **3. Lack of ablation and mechanism analysis**  
The paper fails to analyze which component actually contributes to the reported improvements.  
Multiple elements could influence results—(a) replacing SVD with eigen decomposition, (b) the prefix-suffix ratio for null-space approximation, and (c) the regularization term on **α**—yet none are independently evaluated.  
Both the threshold **t = 0.1** and regularization weight **λ = 0.2** are empirically chosen without justification or sensitivity study, leaving questions about robustness and reproducibility.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel perspective for post-training quantization (PTQ) of LLMs by leveraging null space optimization. The core idea is to constrain the quantization error to the null space of input activations, thereby minimizing the impact on the model's output. The authors propose Q2N, a practical method that includes an efficient approximation of the null space and a closed-form "equivalent projection vector" `α` that can be absorbed into existing scaling factors to avoid any inference overhead. Experiments show that Q2N consistently improves performance when added to state-of-the-art PTQ baselines like GPTQ and QuIP.
The paper is well-written, theoretically sound, and makes an original contribution. The core concept of using the null space is a valuable perspective that opens a new, principled direction for PTQ research. The experimental methodology is robust, providing evidence for the method's effectiveness across a range of models and baselines.

### Strengths
(1) Novel Conceptual Framework: The idea of guiding quantization error into the null space is original and reframes the PTQ problem in a more targeted way than simply minimizing numerical error.
(2) Practical and Efficient Algorithm: The paper translates theory into a practical algorithm, with the memory-free `α` vector being a key element that makes the method viable for deployment without inference overhead.
(3) Comprehensive Validation: The approach is thoroughly evaluated on multiple modern LLMs,  demonstrating its general applicability and robustness.

### Weaknesses
(1) Marginal Gains vs. Complexity: While gains are consistent, they can be modest. A more direct analysis of the trade-off between the one-time quantization complexity and the magnitude of performance improvement would be beneficial.
(2) Hyperparameter Sensitivity: The paper would be strengthened by a more detailed sensitivity analysis for the key hyperparameters `t` and `λ` to better guide practitioners on their selection.
(3) Limited Bit-Width Scope: While the paper focuses on 2-3 bit quantization where the method shows clear benefits, it would be valuable to understand the method's behavior across a wider range of bit-widths (e.g., 4-6 bits) to better characterize when null space optimization provides the most value.
(4) Indirect Evaluation of Null Space Quality: While Table 3 and Table 4 compare different null space estimation methods through downstream performance, the paper would benefit from direct mathematical metrics (e.g., subspace distance, projection error, or spectral analysis) to quantitatively assess how well the Prefix-Suffix Sum Ratio approximation captures the true null space.

### Questions
(1) The paper focuses on low-bit quantization. Have you explored the effectiveness of Q2N at higher bit-widths, such as 4-bit and above? Does the method continue to provide consistent improvements in these less aggressive quantization regimes?
(2) Could you provide a sensitivity analysis plot for hyperparameters `t` and `λ` against a key performance metric (e.g., perplexity)?
(3) The paper uses groupsize=128 for per-group quantization. How does the equivalent projection vector α interact with group-wise scaling factors? Have you explored the sensitivity to different groupsizes, and does the method's effectiveness vary with finer or coarser granularity?

### Soundness
3

### Presentation
3

### Contribution
3
