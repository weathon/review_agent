# Post-Training Quantization via Residual Truncation and Zero Suppression for Diffusion Models

- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Diffusion models achieve high-quality image generation but face deployment challenges due to their high computational requirements. 
Although 8-bit outlier-aware Post-Training Quantization (PTQ) matches full-precision performance, extending PTQ to 4 bits remains challenging. 
Larger step sizes in 4-bit quantization amplify rounding errors in dense, low-magnitude activations, leading to the loss of fine-grained textures. 
We hypothesize that not only outliers but also small activations are critical for texture fidelity. To this end, we propose Quantization via Residual Truncation and Zero Suppression (QuaRTZ), a 4-bit PTQ scheme for diffusion models. 
QuaRTZ applies 8-bit min–max quantization for outlier handling and compresses to 4 bits via leading-zero suppression to retain LSBs, thereby preserving texture details. 
Our approach reduces rounding errors and improves quantization efficiency by balancing outlier preservation and LSB precision. 
Both theoretical derivations and empirical evaluations demonstrate the generalizability of QuaRTZ across diverse activation distributions. 
Notably, 4-bit QuaRTZ achieves an FID of 6.98 on FLUX.1-schnell, outperforming SVDQuant that requires auxiliary FP16 branches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Authors aim to improve low-bit quantization (W4A4) of diffusion models. Their method, QuaRTZ, consists of two stages. The first quantizes the model to 8 bits with uniform quantization (minmax range setting). The second stage reduces this further to 4 bits, through a Leading Zero Suppression (LZS) kernel. They state that the latter retains both outliers and least significant bits, the latter of which (they say) is usually ignored in literature—they claim the least significant bits are important for diffusion models that rely on these for creating small details over many steps. They show that the LZS kernel increases the entropy of the 4-bit INT representation, i.e. more information can be conveyed/retained with it.

### Strengths
I think it's a good paper
* Simple but effective method. Seems to outperform more expensive methods (e.g. rotation/transformation literature).
* Good presentation and clear motivation
* Theoretical (lower bound) and practical (hardware/kernel) considerations
* The results are convincing and groupsize ablation valuable

### Weaknesses
**1. Speed w.r.t. baselines'.**

I appreciate the honest comparison to SVDQuant, which in about half of the cases seems better than QuaRTZ. The FP16 stream in SVDQuant is a disadvantage, but as a reader it is hard to understand the speed-accuracy trade-off since there is no SVDQuant vs QuaRTZ benchmarking.

**2. Comparison to block-wise quantization**

It is unclear to me how it compares to existing block-wise quantization methods, e.g. per-block quantization with power of 2 scales or newer NVFP4 format. Although the authors claim that the shifting can happen efficiently (L158), it is unclear what the online overhead is compared to these existing methods. Section 4.4 and Table 1 seem to indicate it's negligible, but I'd like to ask the authors to elaborate further---e.g. what does the PyTorch column mean---a naive PyTorch implementation of QuaRTZ or a PyTorch matmul?

___

I'm happy to raise my score if the above are resolved.

### Questions
* In Eq. 2, is $m$ the maximum integer representation within the block?
* How does QuaRTZ differ from per-block quantization with power of 2 scale?
* How does it compare to MXFP4/NVFP4?
* Please give more details for the LLM results. Static quantization? What sequence length? Other metrics (e.g. CSR or MMLU) would be beneficial.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces QuaRTZ, a 4-bit Post-Training Quantization (PTQ) method for diffusion models. The authors identify that existing methods, while preserving outliers, often fail to maintain texture fidelity at very low bit-widths due to the loss of Least Significant Bits (LSBs). QuaRTZ addresses this by first applying an 8-bit min-max quantization to handle outliers and then compressing to 4 bits using leading-zero suppression, a technique designed to retain LSBs and preserve fine-grained details. The method reports a strong FID of 6.98 on FLUX.1-schnell, outperforming SVDQuant without requiring auxiliary FP16 branches.

### Strengths
1.  The paper identifies an often overlooked source of error in low-bit quantization for generative models—the loss of LSBs—and proposes a principled solution.
2.  The proposed method, QuaRTZ, is training-free, which is highly desirable for practical deployment of large diffusion models.
3.  The approach of balancing outlier preservation with LSB precision through a two-stage process seems effective. The reported result on FLUX.1-schnell is competitive.

### Weaknesses
1.  A weakness is the absence of an operator-level performance analysis. A comprehensive assessment of QuaRTZ's practical benefits requires a detailed breakdown of latency and memory overheads. Including a comparative analysis of speed and memory costs against baselines like SVDQuant would strengthen the paper's claims.

### Questions
1.  Could you provide a detailed breakdown of the runtime overhead for the encoding and decoding stages of QuaRTZ? How does its performance compare to standard 4-bit dequantization on target hardware, such as GPUs, and what is the cost of each stage in your pipeline?
2.  To better understand the practical benefits of QuaRTZ, could you provide a direct comparison of its latency and memory costs against other methods like SVDQuant? A discussion of the relative advantages and disadvantages would also be valuable.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents QuaRTZ (Quantization via Residual Truncation and Zero Suppression), a 4-bit post-training quantization (PTQ) method for diffusion models. While existing 8-bit PTQ methods preserve model quality, extending them to 4-bit quantization introduces significant rounding errors and texture degradation. QuaRTZ addresses this by leveraging a leading-zero suppression mechanism to compress 8-bit activations into 4 bits. The method effectively balances outlier preservation with low-significance-bit precision, supported by both theoretical quantization error analysis and empirical validation across multiple diffusion models. Experimental results demonstrate that QuaRTZ achieves competitive or superior image quality (e.g., FID 6.98 on FLUX.1-schnell) compared to prior state-of-the-art methods.

### Strengths
- The paper is well organized, clearly introducing the proposed method, theoretical analysis, and empirical evaluation.
- The proposed QuaRTZ method is evaluated on multiple diffusion models and diverse generation datasets, demonstrating the generalization of proposed quantization algorithm.
- The paper includes an analysis of quantization error bounds, providing theoretical support for the observed empirical improvements.

### Weaknesses
- The proposed method seems effectively equivalent to adding a power-of-two scaling factor ($2^{0,1,2,3,4}$) per subgroup, which may limit its novelty or distinct advantage compared to existing mirco-exponent-sharing quantization approach [1].

- Section 4.4 discusses GPU kernel implementation, while Appendix D reports area and power metrics—typically associated with hardware accelerators. It is unclear whether the reported latency is measured on GPU or derived from hardware emulation. If latency is GPU-based, the overhead of integer shifts during accumulation in the GEMM loop should be analyzed.

- (Minor) In line 161, the operation on the MMA output appears to require a left-shift since quantized weights are right-shifted by FLAG.

- (Minor) Some citations in the paper appear incomplete or missing (denoted by question marks), which should be corrected for completeness and credibility.

[1] Darvish Rouhani, Bita, et al. "With shared microexponents, a little shifting goes a long way." Proceedings of the 50th Annual International Symposium on Computer Architecture. 2023.

### Questions
1. Could you clarify whether the proposed quantization approach fundamentally differs from power-of-two scaling quantization, or if it can be interpreted as an equivalent formulation?

2. How exactly was latency measured? Was it obtained from actual GPU execution or from a hardware simulation/emulation of a custom accelerator?

3. If the latency is based on GPU measurements, could you provide a breakdown of the computational overhead introduced by the integer shift operations within the main GEMM accumulation loop?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new 4-bit quantization scheme, which first does INT8 quantization and then compresses to 4 bits with subgroup leading-zero suppression. This scheme is found to have higher entropy on layer-wise analysis than naive INT4, indicating better use of all 4 bits. The evaluation results show that it could achieve higher image quality than naive INT4 and prior art SVDQuant on some diffusion models.

### Strengths
An end-to-end story of a new quantization scheme from theoretical analysis, information-theoretic entropy results, hardware-friendliness discussion, and empirical results on real-world diffusion models.

### Weaknesses
- The quality benefit is not universal. The proposed method has lower quality metrics on some diffusion models like PixArt-sigma.
- It is questionable if the proposed method can be efficiently implemented on current GPUs. Also, there are no performance results reported in this paper.
- INT4 quantization has been surpassed by newer quantization schemes (like MXFP4 and NVFP4) on the latest generation GPUs. A comparison against FP4 should be conducted to demonstrate the benefit of the proposed method.
- The quality of the paper is low. It is hard to get the main idea of the proposed method in Figure 1. Also, there are multiple broken references in the paper (for example, in line 297).

### Questions
- Can the proposed method be efficiently implemented on current GPUs? If yes, what is the performance of the new INT4 kernels?
- How does the proposed method compare to FP4?

### Soundness
2

### Presentation
1

### Contribution
2
