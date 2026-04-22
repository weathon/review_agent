# AMS-Quant: Adaptive Mantissa Sharing for Floating-Point Quantization

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 4, 0, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities in various kinds of tasks, while the billion or even trillion parameters bring storage and efficiency bottlenecks for inference. Quantization, particularly floating-point quantization, is known to be capable of speeding up LLM inference by reducing memory footprint and data movement during the inference process. For the first time, we advance the floating-point quantization exploration from integer bit-widths to non-integer bit-widths, namely AMS-Quant, to further approach the quantization sweet spot. AMS-Quant incorporates two novel techniques to put it into effect: (1) it proposes Mantissa-bit Sharing, which groups k quantized weights and lets them share the least significant mantissa bit, allowing us to further approach the minimum quantization bit-width without accuracy loss. (2) It introduces Adaptive Searching, which employs an offline optimization strategy to minimize the accuracy degradation introduced by sharing. Moreover, AMS-Quant is also prototyped as efficient CUDA Linear kernels, which translates memory savings into wall-clock latency reduction by reducing memory access. Extensive experiments on large-scale datasets and models show that AMS-Quant can quantize the model to FP-5.33-e2m3 and FP4.25-e2m2, and significantly speed up the LLM decoding over FP16 inference (2.8$\times$ and 3.2$\times$), with negligible accuracy loss.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes AMS-Quant, a novel floating-point quantization method that enables non-integer bit-width representations for LLMs. The method is based on two key components:
- Mantissa-bit Sharing: groups of k quantized weights share their least significant mantissa bit, reducing the effective bit-width.
- Adaptive Searching: an offline optimization that selects the shared bit value minimizing mean squared error across the group.


Authors complement the algorithm with an efficient CUDA kernel implementation that restore quantized weights using bit-level operations.

### Strengths
- Idea is novel and timely. While existing works have explored sharing exponent bits or scaling factors, mantissa sharing is surprisingly unexplored. 
- The paper is aptly written and clearly organized.
- The proposed adaptive searching for mantissa bits provides an effective mechanism to manage accuracy loss.
- Authors provide substantial experimental results to validate their proposed method.

### Weaknesses
- Authors did not report search time for adaptive mantissa sharing algorithm. While it is run offline, it is essential to ascertain the computational cost associated with the search process.
- Non-integer bit-widths based on packing and unpacking bits cannot be integrated with inference frameworks like TensorRT. Likewise, deployment on other hardware devices like TPUs, NPUs etc may need extra effort.

### Questions
- What is the time and compute cost for running adaptive mantissa search, especially for large models?
- Can this method be combined with existing methods like GPTQ or AWQ with ease?
- Can this method be extended to activation quantization?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AMS-Quant, a weight-only floating-point quantization framework for LLMs. It includes two major techniques, i.e., mantissa-bit sharing to boost compression and adaptive searching to reduce quantization error. A custom GPU kernel is developed to realize practical speedups. Experiments show that AMS-Quant achieves these gains with only minimal accuracy degradation.

### Strengths
1. The paper is well-written and the core ideas are clear to understand.
2. The major contributions of this quantization framework are mantissa sharing and adaptive searching, which are both sound for improved floating-point quantization.
3. The experimental results not only show the quantization accuracy but also the actual acceleration enabled by the proposed methods.

### Weaknesses
1. This work is for weight-only quantization, if it can be further extended to an activation and weight quantization, more efficiency gains should be obtained.
2. The mantissa-sharing operation is analogous to block floating-point quantization and micro-scaling formats (e.g., MXINT, MXFP), which share exponents instead. Additional experiments should be conducted to more thoroughly validate the superiority of the proposed methods.
3. Mantissa sharing is applied along the input-channel dimension of the weight tensor. However, this choice is less cache- and memory-friendly than sharing along the output-channel dimension. While the paper justifies the design by noting that activation outliers are typically aligned with input channels, it may be worthwhile to incorporate a small calibration set into the mantissa-sharing and adaptive search procedure to reduce quantization loss, while enabling mantissa sharing along the output-channel dimension for more efficient memory access.

Overall, this is a good-quality paper and I will raise my score if my concerns are well addressed.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper focuses on weight-only quantization by leveraging mantissa-bit sharing, resulting in non-integer bit widths for individual weight elements. It also proposes an adaptive search strategy to determine the optimal shared mantissa value. Based on these techniques and a bit-slicing packing strategy, the authors implement CUDA linear kernels and demonstrate improved latency through more efficient memory access.

### Strengths
They implement CUDA kernels using an efficient restoration algorithm based on the bit-slicing technique.

### Weaknesses
A major limitation of this paper is the lack of comparison with existing methods. Numerous prior works achieve higher (even 2-bit in QuIP or AQLM) compression rates while maintaining comparable accuracy. Without extensive benchmarking against these methods, it is difficult to evaluate the advantages of the proposed strategy. Furthermore, many weight-activation quantization (e.g., Quarot) approaches exist that effectively address KV cache and high-precision computation overhead in attention mechanisms, which makes the proposed method less compelling.

Additionally, the rationale for mantissa-bit sharing is not clearly justified. The method - merely sharing the least significant bits of the mantissa and selecting the optimal value via brute-force search - appears simplistic and does not convincingly constitute a novel contribution. From a practical perspective, applying techniques that reduce additional 1 or 2 bits more substantially might be more impactful than the minor gains achieved through mantissa sharing.

Regarding the packaging and restoration techniques, the approach shows limited novelty compared to the TC-FPx framework and appears largely as an engineering implementation. Finally, the figures do not effectively convey the core concepts and seem to underutilize the available space, reducing their clarity and impact.

### Questions
What are the advantages of using non-integer bit widths and the mantissa-sharing strategy compared to existing quantization methods? Please clarify how this approach provides benefits in terms of accuracy, compression, or computational efficiency relative to prior work.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose mantissa-bit sharing, sharing the least significant mantissa bit across groups of k quantized weights, resulting in formats like FP5.33-e2m3 and FP4.25-e2m2 (1:3 / 1:4 sharing). Per group 1-bit search minimizies the quantization MSE. Further, they implement bit-packing + register level ops for restoring FP16 for CUDA linear kernels. Speedup over FP16 kernels are demonstrated. Finally, their FP5.3(e2m3) hits the sweet-spot across IFEval GSM8k and MMLU on 3 models.

### Strengths
- Simple to implement and leverage
- design is sound, kernel also obtains a speedup in memory-bound settings at the kernel-level.

### Weaknesses
- baselines seem to be limited, AWQ.GPTQ, NF4 should be discussed.
- Are the speedups only kernel-level? it is very important to see full decode latency,  specifically is that a speedup for the model, or for just the kernel in Figure 6? I think its the latter so it might be better to not label them as real model speedups.

### Questions
- Beyond notes on the weaknesses, it would be interesting to see an ablation of increasing k.

### Soundness
3

### Presentation
2

### Contribution
2
