# MixLLM: Mixed-precision LLM Quantization with Algorithm-system Co-design

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 6, 6

## Abstract
Quantization has become one of the most effective methodologies to compress LLMs into smaller size.
However, the existing quantization solutions still show limitations of either non-negligible accuracy drop or system inefficiency.
In this paper, we make a comprehensive
analysis of the general quantization principles on their effect to the triangle of accuracy, memory consumption and system efficiency.
We propose MixLLM that explores the new optimization space of mixed-precision quantization between output features based on the insight that different output features matter differently in the model.
MixLLM identifies the output features with high salience in the global view rather than within each single layer,
effectively assigning the larger bit-width to output features that need it most to achieve good accuracy with low memory consumption.
We present the sweet spot of quantization configuration of algorithm-system co-design that lead to high accuracy and system efficiency.
To address the system challenge of this sweet spot, we design the two-step dequantization to make use of the int8 Tensor Core easily and fast data type conversion to reduce dequantization overhead significantly.
Extensive experiments show that MixLLM achieves the best accuracy on a variety of tasks for the popular LLMs than a set of state-of-the-art works.
It shows 0.31 lower perplexity and 0.43\% improvement on zero shot tasks for Llama 3 8B than QoQ, with similar memory consumption and system efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a method called MixLLM, aimed at achieving high accuracy and system efficiency through mixed-precision quantization. MixLLM introduces to identify the salience of output features based on loss distance estimates, focusing on global model loss rather than local layer loss. By assigning larger bit-widths to the most critical features, MixLLM achieve superior while maintaining low memory consumption. This paper presents a two-step dequantization process to reduce overhead and efficiently utilize int8 Tensor Core for computation. Experimental results demonstrate the effectiveness of MixLLM across various tasks compared to existing techniques.

### Strengths
1. The introduction of MixLLM explores a novel optimization space in mixed-precision quantization through global salience identification.
2. The paper clearly articulates how the two-step dequantization improves computational efficiency, particularly through leveraging Tensor Core capabilities, demonstrating both practicality and usability.
3. The authors conduct extensive experiments across multiple tasks and models, providing compelling evidence for the advantages of MixLLM in terms of accuracy and efficiency.

### Weaknesses
1. Although the study evaluates several models and datasets, the comparisons with some state-of-the-art methods are not sufficiently comprehensive, potentially limiting the generalizability of the results.
2. The provided Algorithm 1 is not standardized. It should clearly specify the input and output, adhere to a standard for statement format, and include step numbers for each procedure.
3. Provide detailed experimental setups, hyperparameters, and implementation specifics to Enhance Reproducibility.

### Questions
It would be better to expand the discussion on theoretical analysis behind MixLLM and the limitations of MixLLM.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper address the performance degradation that often occurs when LLMs are quantized to low-bit representations. The authors propose a mixed-precision quantization method for model weights, specifically applying both int4 and int8 quantization to the output channels of the weights. The authors introduce an approach based on global loss estimation and second-order Taylor expansion. This method pinpoints the output channels that are most sensitive to quantization errors, allowing the allocation of higher precision to those critical channels. In addition, the authors design a fast i2f conversion technique to enhances the model's inference speed. Extensive experiments, the paper demonstrates that using 20% int8 quantization  can achieve comparable performance  with full-precision models.

### Strengths
1. The paper introduces a novel approach to pinpoint output channels that are difficult to quantize by utilizing global loss estimation and second-order Taylor expansion. This method allows for effective mixed-precision quantization of weights, specifically applying int4 and int8 to the output channels, which significantly reduces quantization loss without adding extra computational overhead.

2.  The design of a fast integer-to-float (i2f) conversion method is a noteworthy innovation. By leveraging GPU int8 computational capabilities and minimizing the latency typically associated with i2f conversions, the authors enhance computational efficiency, which is critical for high-performance applications.

3. The paper is well-written, with clear descriptions and thorough explanations of the methods used. The authors provide comprehensive experiments that demonstrate their approach's effectiveness, showing that using int8 quantization on just 20% of the channels can achieve state-of-the-art results.

### Weaknesses
1. Limited Model Size in Experiments: The models tested in the paper are relatively small. The absence of experiments on larger models, such as those with 30B or 70B parameters, raises questions about the scalability of the proposed method. Evaluating larger models would strengthen the paper by demonstrating the approach's effectiveness across a broader range of model sizes.

2. Potential Efficiency Concerns: The quantization process requires performing global loss calculations and obtaining gradients, which may be computationally intensive. This could pose efficiency challenges, especially when dealing with very large models, and might offset some of the benefits gained from the quantization technique.

3. Hardware Dependency: The implementation relies on specific w8a8 computational kernels to realize the performance gains. As a result, the acceleration may be significant only on certain GPUs that support these optimizations. This hardware dependency could limit the method's general applicability and usefulness across different computing environments that do not have the requisite hardware support.

### Questions
1. Experiments on Larger Models: The models tested in your work are relatively small in scale (up to 8B LLM). Conduct experiments on larger models, such as those with 30B or 70B is expected to demonstrate the scalability and effectiveness.

2. Quantization Time Efficiency: This method involves calculating global loss and obtaining gradients to identify the output channels requiring higher bit-widths. Could you provide more details on the time efficiency of this quantization process? Specifically, how does the computational overhead compare to other quantization techniques, and does it significantly impact the overall efficiency when scaling to larger models?

3. Latency Testing on Different GPUs: Given that your optimizations are designed to leverage GPU capabilities, have you conducted latency tests on different GPU architectures? Providing performance benchmarks across various GPUs would offer valuable support into the general applicability and potential limitations of your method in diverse hardware environments.

4. Release of CUDA Kernel Code: To facilitate community validation and further research, are you planning to release the CUDA kernel code for your custom w8a8 computation? Sharing the code would enable others to replicate your results, integrate your optimizations into their own work, and contribute to advancements in this area.

### Soundness
2

### Presentation
2

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
This paper proposes a new optimization approach for mixed precision quantization of output features, recognizing that different features have varying importance in the model. By allocating larger bit widths to the most critical features, MixLLM enhances accuracy while minimizing memory usage. To improve system efficiency, this paper also implements a two-step dequantization process, allowing the use of int8 Tensor Core computations and fast integer-float conversion to reduce dequantization overhead. The results demonstrate that MixLLM outperforms QoQ while maintaining similar memory consumption and system efficiency.

### Strengths
This paper achieves better results on various LLMs, including Llama 3 8B, Llama 2 7B, Mistral 7B v0.3, Qwen 2 1.5B, and Qwen 2 7B. It also demonstrates improved performance across various benchmarks compared to QoQ, while maintaining similar memory consumption.

### Weaknesses
1. In Table 1, QoQ fails to optimize the Qwen 2 1.5B and Qwen 2 7B models, leading to an unfair comparison.
2. There is a lack of comparison of MixLLM's inference time with other methods.
3. Figure 1 fails to clearly illustrate the complete process of MixLLM, and this paper includes only one figure.

### Questions
1.	Could you show the inference time of MixLLM in comparison to other methods under the same experimental setup?
2.	Could you provide a comparison of the performance and inference time of this method on larger models like LLaMA3 70B and Qwen2 72B against other methods?
3.	Could you provide the ablation experiments on performance and inference time for the 'two-step dequantization' proposed in this paper?

### Soundness
2

### Presentation
2

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
This paper presents (1) a survey of existing post-training-quantization (PTQ) approaches; (2) a new approach for mixed precision based on global salience. 

The proposed method defines a global salience measure of each output channel, similar to that of Kwon et al., 2022 and Kim et al., 2024 but with modifications. Different from previous works that use such salience information to perform sparse-and-dense decomposition, this paper packs output channels into two bit-rate buckets (8-bit symmetric and 4-bit asymmetric). The mixed-precision matmul is then carried out with a fused kernel which scatters output (of matmuls from different bitrate buckets) back to the corresponding indices.

### Strengths
* The paper does a good job summarizing and providing useful critiques to different PTQ approaches in Section 2, including weight-only quantization, weight-activation quantization, outlier separation and mixed-precision approaches. Many of the assessment maybe subjective but accurate according to the reviewer's own experience.

* This proposed method which uses  salience information to perform packing of output channels into different bit-rate bucket is interesting and novel to the knowledge of the reviewer. This process is intuitive, and requires just a simple sorting instead of using iterative procedure, which as the authors pointed out, "saves a lot of computation" (Ln269). The results looks good on 7B-sized models.

### Weaknesses
**The "co-design" analysis**

While the paper claims "algorithm-system co-design", it doesn't contain much principled quantitative analysis. For a good "co-design" system paper, the reviewer would expect the author to use some kind of performance models, such as the roofline model to justify why the chosen design is a "sweet spot". Or at least use performance counters such as cache hit-rate, communication latency, memory bandwidth to substantiate the choice. In the current state, the paper relies too heavily on empirical observations and many claims are not very well explained. For example,
* Ln46: "Besides, the weight-only method can lead to system performance drop for large-batched workloads." \
In what model and what's the batch size when this happens?

* Ln75: "MatMul execution tends to be bound more on the larger weight tensor rather than the smaller activation tensor, which weakens the need to push the activation smaller" \
"Weakens the need" is a quite vague expression and it will be nice to draw your conclusion in a data-driven manner.

* Ln695: "Hard to achieve the peak performance due to the inefficiency of the sparse tensor computation on the GPU"\
What is the utilization of the proposed method and how bad is the utilization on approaches that separate outliers in contrast?

**Reproducibility**

The paper currently does not show throughput / latency numbers (or did I miss it?) nor was the inference kernel implementation provided. Because MixLLM uses different bit-rate (W4.8) and custom inference kernel (packing and scattering), it is important to carefully benchmark the latency and throughput against other methods. Currently the paper only says "this function with the fused epilogue of MatMul to scatter the output to the corresponding indices, which is basically costless" (Ln288) and "MixLLM is on-pair for the system efficiency
when compared to the state-of-the-art weight-activation quantization" (Ln 466)

While Github implementation is not generally required, the reviewer notes that most quantization papers do have OSS implementation (GPTQ, AWQ, LLM.int8, SmoothQuant, QoQ). Since the actual system performance of the scattering kernel is critical to the usefulness of this paper, it would be difficult to verify the results and seriously limit the usefulness of the paper without an implementation of the fused kernel.

### Questions
1. In Eq 4, two design decisions that differentiate the proposed salience measure from previous work is to (1) use non-diagonal Fisher information matrix and (2) not ignoring first-order information. Can you provide ablation results show how important they are to the final results?

2. MixLLM is evaluated with one fixed setting of 20% 8-bit + 80% 4-bit. It will be nice to see how the accuracy / latency / throughput change when this ratio is changed.

3. The author makes the argument that the proposed one-pass method is faster / competitive to iterative methods. It will be nice to show how much speed-up / accuracy difference does this make.

### Soundness
2

### Presentation
4

### Contribution
3
