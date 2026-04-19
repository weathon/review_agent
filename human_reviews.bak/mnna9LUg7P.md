# Quamba: A Post-Training Quantization Recipe for Selective State Space Models

- Decision: Accept (Poster)
- Scores: 6, 6, 5, 8

## Abstract
State Space Models (SSMs) have emerged as an appealing alternative to Transformers for large language models, achieving state-of-the-art accuracy with constant memory complexity which allows for holding longer context lengths than attention-based networks. The superior computational efficiency of SSMs in long sequence modeling positions them favorably over Transformers in many scenarios. However, improving the efficiency of SSMs on request-intensive cloud-serving and resource-limited edge applications is still a formidable task. SSM quantization is a possible solution to this problem, making SSMs more suitable for wide deployment, while still maintaining their accuracy. Quantization is a common technique to reduce the model size and to utilize the low bit-width acceleration features on modern computing units, yet existing quantization techniques are poorly suited for SSMs. Most notably, SSMs have highly sensitive feature maps within the selective scan mechanism (i.e., linear recurrence) and massive outliers in the output activations which are not present in the output of token-mixing in the self-attention modules. To address this issue, we propose a static 8-bit per-tensor SSM quantization method which suppresses the maximum values of the input activations to the selective SSM for finer quantization precision and quantizes the output activations in an outlier-free space with Hadamard transform. Our 8-bit weight-activation quantized Mamba 2.8B SSM benefits from hardware acceleration and achieves a 1.72 $\times$ lower generation latency on an Nvidia Orin Nano 8G, with only a 0.9\% drop in average accuracy on zero-shot tasks. When quantizing Jamba, a 52B parameter SSM-style language model, we observe only a $1\%$  drop in accuracy, demonstrating that our SSM quantization method is both effective and scalable for large language models, which require appropriate compression techniques for deployment. The experiments demonstrate the effectiveness and practical applicability of our approach for deploying SSM-based models of all sizes on both cloud and edge platforms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a post-training quantization method for state space models (SSMs), which enables the utilization of low-bit acceleration in modern GPUs and achieves notably lower generation latency and small accuracy penalty. The method is mainly comprised of two techniques: 1) static 8-bit per-tensor quantization for input activations, and 2) Hadamard transform for outlier-free output activation quantization.

### Strengths
- Overall, the paper is well-structured, well-motivated, and well-written. The right amount of background introduction, motivating examples, and explanation of challenges make this paper easy to understand.
- The paper did thorough experiments to show the generalizability and scalability of the proposed method on various SSM-based implementations. In the results, the proposed Quamba shows a clear latency improvement and makes it possible to run large-scale models on devices with a limited amount of memory.

### Weaknesses
- The novelty of the proposed method is not clear. As mentioned in the text, the paper acknowledges that clipping outliers in input activations with a percentile max is a widely applied technique, and smoothing out the outliers with the Hadamard transform is also not new and has been applied in prior quantization methods for transformer attention computation. The main novelty of this paper therefore may come from being the first to apply these two techniques to SSM.
- There is no significant benefit over QuaRot-SSM, which is a re-implementation of QuaRot for Mamba. The positive side is that this does serve as a strong baseline, however, their performance is very similar in many experiments. Figure 7 in the appendix, to some extent, also shows that the differences in architecture are small.

### Questions
- Could the paper articulate more clearly what aspects of applying the two proposed techniques (clipping and Hadamard transform) to SSMs are novel or challenging compared to their use in other contexts?
- Could the paper provide a better positioning of this paper, in terms of what the main difference is between the proposed method and the prior QuaRot? It seems that a trivial change on the original QuaRot (such as removing the extra Hadamard transform, which is a natural choice) can easily improve the performance.
- In Figure 1 (b) and (c), could the paper provide the data for QuaRot-SSM for a more complete comparison?
- In Figure 2 (b) and (c), it is unclear what y_t means. And why are there multiple (colors of) MSE curves? For example, for the curve of V_bar (red one in (b)), does it still represent the MSE of y_t and how should we understand this curve? Figure 3 (a) is much clearer in this sense.
- Is it possible to support 4-bit quantization with the proposed method?

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
5

### Summary
The paper studies quantization of stat-space models. The propose Quamba which employs 8-bit quantization and Walsh-Hadamard transformations of activations to preserve accuracy.

### Strengths
- Studies quantization of SSMs, which is novel since SMMs themselves are new and quantization studies have not caught up with them.
- The empirical results are not just scoped to accuracy studies using fake quantization, but actual profiling is done using INT8 and cutlass.

### Weaknesses
- Empirically, Quamba does not show advantages compared to Quarot.
- MSE studies are used to compare difficulty of quantizing attention vs SSM (e.g., Fig 2). However, absolute numbers for MSE are not useful when inputs and models are different. A better way to visualize the quantization difficulty would be to employ a normalized MSE. In other words, the L2 norm of the delta divided by the L2 norm of the unquantized tensor, rather than just the L2 norm of the delta.
- Theorem 4.1 is very badly written, I had to refer to the appendix to guess what the authors meant. Note that lowercase b and epsilon were never defined and they are the defining variables of the theorem! Please fix this and introduce b as being an element from the B matrix, and epsilon to be an upper bound of the quantization step. P.S., these are my guesses for what was meant. 
-While I appreciate Theorem 4.1 which states that quantization error compounds over time-steps (correct me if I am wrong). Where was this used? Which aspect of the proposed Quamba leverages Theorem 4.1? It seems like this was just added as decorative math.

### Questions
- Why is TTLT used? It makes more sense to evaluate latency (TTFT) and throughput (token-to-token).
- Since extremely large outliers are identified as problematic for SSMs, have the authors considered using clipping to lower quantization noise. For example, OCTAV may be useful in this setup [1]
- Empirical results with comparisons to transformers are focused on the W8A8 case. However, Transformers have been quantized to 4-bit (e.g., using Quarot).Is it therefore a fair comparison to use 8-bit transformers?

[1] Sakr, Charbel, et al. "Optimal clipping and magnitude-aware differentiation for improved quantization-aware training." International Conference on Machine Learning. PMLR, 2022.

----
Good responses, I am raising to 6.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a post-training quantization method tailored for Selective State Space Models (SSMs) to address challenges in model size reduction and computation efficiency, specifically in hardware-constrained environments. The authors propose an 8-bit static quantization approach that handles weight and activation quantization while mitigating the effects of outliers commonly found in SSM outputs.

Key contributions of the paper include:
1.  A method for input activation quantization that reduces the impact of extreme values by clipping based on the 99.999th percentile, improving quantization precision.
2. Hadamard transforms to map output activations into a space with minimized outliers, allowing for accurate quantization without excessive memory and computational cost.
3. Empirical analysis showing the effectiveness of this quantization approach in achieving a balance between model accuracy and reduced latency, making SSMs more feasible for edge and embedded applications.

The paper provides a comprehensive empirical evaluation, demonstrating how these techniques enable SSMs to perform efficiently on hardware with limited resources.

### Strengths
1. The paper presents a unique approach by applying quantization specifically to Selective SSMs, a topic with limited prior research in this area. The integration of percentile-based clipping and the Hadamard transform for managing outliers in SSMs represents an innovative combination of techniques. These methods are tailored specifically for the challenges of SSM quantization, marking a significant contribution to the field of model compression.
2. The paper demonstrates a thorough experimental setup, with well-defined quantization methodologies for activations. The authors support their approach with empirical evidence, clearly detailing the implementation of 8-bit static quantization and its impact on latency and accuracy. The careful construction of quantization techniques for hardware constraints is grounded in a rigorous understanding of SSM behavior, lending credibility to the study’s findings.
3. The proposed quantization approach has high potential significance, especially for applications involving SSMs in resource-limited environments. By addressing the specific challenges of quantizing SSMs, the paper extends the practical use of SSMs, making them feasible for deployment on low-power hardware without significant accuracy degradation.

### Weaknesses
The paper is innovative in its approach and thorough experimentation. However, there are several critical questions that I raised in the "Question" section, which I believe are essential for the clarity and robustness of the findings. I hope the authors can provide insights on these points, and I look forward to further discussion.

### Questions
1. As a researcher focused on large model quantization on edge devices, I have some concerns about the authors' claim regarding the lack of prominent outliers in transformers. Based on my own experiments and findings [1-2] from recent literature, larger language models (LLMs) actually tend to exhibit more significant outliers, especially in channel-wise activation. Can you clarify your claim about outliers in Transformers and provide evidence or citations supporting your view?

2. Studies [3-4] on transformer quantization have shown that quantizing outliers can substantially impact model accuracy. This observation aligns with the authors' ablation study results. However, I am unclear on why certain techniques in this paper appear to alleviate the impact of outlier quantization. Additionally, shouldn’t each optimization step incrementally reduce accuracy? Surprisingly, the authors' ablation study shows an improvement in accuracy when all three techniques are applied together. Could you please provide a more detailed explanation of why your techniques, when combined, lead to improved accuracy rather than incremental reductions?

3. In the introduction, the authors mention that this quantization method is applicable to both edge devices and cloud computing. However, the experiments only feature the high-compute edge device Orin Nano 8G (and the A5000, which would generally not be considered an edge device). There is no testing on cloud computing hardware, which would have been helpful.

4. For the results in Table 1: Profiling Latency, I find some inconsistencies. As token length increases, both computation and memory should scale linearly, so I would expect a relatively stable speedup ratio. If anything, given the execution overhead of the nonlinear kernel within SSM, there might be a slight increase in latency with token length. However, the authors' data show an unusual spike in performance at $L=512$ and $L=1$. Could you please provide a more detailed explanation of the observed performance spikes, possibly including additional data points or experiments that could help clarify the relationship between token length and speedup ratio?

5. Lastly, could the authors clarify how many hardware measurements were conducted to obtain the reported results? Given the inherent variability in hardware performance, multiple measurements are typically necessary to ensure stability and reliability. It would be helpful to understand the stability of the hardware results presented. Could you please provide details on the number of runs performed for each experiment, any measures taken to account for variability (e.g., averaging results, reporting standard deviations), and how you ensured the stability and reproducibility of your hardware measurements?

[1] Shen, Xuan, et al. "Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 17. 2024.
[2] Xiao, Guangxuan, et al. "Smoothquant: Accurate and efficient post-training quantization for large language models." International Conference on Machine Learning. PMLR, 2023.
[3] Lin, Yang, et al. "Fq-vit: Post-training quantization for fully quantized vision transformer." arXiv preprint arXiv:2111.13824 (2021).
[4] Dong, Peiyan, et al. "Packqvit: Faster sub-8-bit vision transformers via full and packed quantization on the mobile." Advances in Neural Information Processing Systems 36 (2024).

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work propose a PTQ solution for SSM model. Quamba notices the SSMs have highly sensitive feature maps within the selective scan mechanism (i.e., linear recurrence) and massive outliers in the output activations, and utilises two methods of outlier suppression - ①suppresses the maximum values of the input activations ②supply hadamard transform of the output activations, to solve the problem. Experiments show that show that this method can achieve 8-bit PTQ quantisation of SSM almost losslessly.

### Strengths
- The approach is simple and effective, introducing little additional overhead
- The experiment is solid: The author achieved real quantization - by handwriting fused operators and utilizing low-bit computational libraries (e.g. cutlass) to implement Quamba.

### Weaknesses
- The proposed method is effective, however, these methods are also widely used in Transformer, and authors may need to make their connection to SSM more explicit in their writing.

### Questions
- In line 228, you mentioned the **Theoretical error bound for SSM quantization**, and give a proof in appendix. However, this discussion seems to be isolated, so I do not particularly understand the connection between this discussion and the core approach of this paper.
- In Table 1 and Table 2, you compared QuaRot and Quamba in W8A8 format. However, QuaRot mainly focus on **W4A4** solution for quantization. I would like to know if this method can be extended to W4A8 and W4A4 scenarios? If it can, how is his performance compared to QuaRot?

### Soundness
3

### Presentation
3

### Contribution
4
