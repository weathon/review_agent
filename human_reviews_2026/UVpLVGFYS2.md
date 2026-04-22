# CondiQuant: Condition Number Based Low-Bit Quantization for Image Super-Resolution

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Low-bit model quantization for image super-resolution (SR) is a longstanding task which is renowned for its surprising compression and acceleration ability.
However, accuracy degradation is inevitable when compressing the full-precision (FP) model to ultra-low bit widths ($2\sim4$ bits).
Experimentally, we observed the degradation of quantization is mainly attributed to the quantization of activation instead of model weights.
In numerical analysis, the condition number of weights could measure how much the output value of the function can change for a small change in the input argument, inherently reflecting the quantization error.
Therefore, we propose CondiQuant, a condition number-based low-bit post-training quantization for image super-resolution.
Specifically, we design an efficient proximal gradient descent algorithm to reduce the condition number of weights while keeping the output as unchanged as possible.
With comprehensive experiments, we demonstrate that CondiQuant outperforms existing state-of-the-art PTQ methods in accuracy without computation overhead and gains the theoretically optimal compression ratio in model parameters.
Our code will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper shows a novel quantization method for image super-resolution (SR), named CondiQuant. CondiQuant is based on the observations that the degradation of quantization is mainly attributed to activations, and the feature matrix X across all linear layers is approximately rank-deficient. A basic analysis tells us that the XδW and δXδW are smaller than δXW. Thus, the main trouble comes from how to reduce δXW. The author connects this with the condition number (related to the deviation of singular values) of W, and then proposes a regularization term to minimize the deviation of singular values. A proximal operator for this regularization term, which utilizes SVD, and a circle optimization process is designed. Experimental results prove the effectiveness of CondiQuant.

### Strengths
1. The overall process makes sense.
2. Results are good.
3. The figures are clear.

### Weaknesses
1. As shown in figure2, there are some X that are full-rank. For these layers, does the author still apply the CondiQuant?
2. In table2 (d), what is the configuration of w/o CondiQuant? 
3. Could you provide a discussion about the difference between this condition number-based method and outlier-suppression methods? In my view, both these types stretch the quantization error-significant direction to reduce the error. However,  outlier-suppression methods are explicit stretch direction, whereas this method implicitly stretches the direction.
4. There are many typos in this paper. Please check all paper and fix them.

### Questions
Please see the Weaknesses.

Overall, I consider that this paper is a good paper.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes CondiQuant, a post-training low-bit quantization method for image super-resolution (SR) based on the condition number of the weight matrices. The authors identify that the performance degradation of low-bit quantization is primarily caused by activation quantization rather than weight quantization. By formulating the quantization error as a function of the condition number and using a proximal gradient descent algorithm, CondiQuant efficiently minimizes the condition number while maintaining output fidelity. Extensive experiments demonstrate that CondiQuant achieves state-of-the-art accuracy with minimal computational overhead and theoretically optimal compression and speedup ratios.

### Strengths
Theoretical Motivation: Provides clear numerical analysis linking weight matrix condition number to activation quantization sensitivity.

Efficiency: The proposed method is computationally efficient (less than 19.0 seconds for the full iteration) and introduces no additional modules.

Experimental Validation: Comprehensive experiments and ablation studies demonstrate improved accuracy over existing PTQ methods and robust performance.

Practical Value: Achieves theoretically optimal compression and speedup ratios, making it applicable for ultra-low bit quantization scenarios.

### Weaknesses
Condition number intuition: It remains unclear why reducing the condition number leads to better SR quantization performance, given that larger condition numbers could potentially preserve more activation variance and high-frequency details.

Second-order error δX δW: In extremely low-bit (2-bit) settings, the second-order term δX δW might no longer be negligible. Quantitative analysis of its contribution is missing.

Generality across architectures: The experiments focus mainly on a single SR architecture (SwinIR-light). It is unclear whether CondiQuant generalizes to other SR networks, including high-performance or lightweight variants (e.g., SwinIR-B, MambaIRv2-light).

### Questions
Condition Number and Activation Representation
In the paper, you propose minimizing the condition number of the weight matrices to reduce activation quantization error. Intuitively, a larger condition number indicates higher sensitivity of the output to the input, which may lead to a wider activation distribution and potentially better preservation of texture and fine details. Could you clarify why, in SR networks, a smaller condition number leads to better quantization performance? During the reduction of the condition number, is there a risk of negatively affecting the network’s feature representation or its ability to capture high-frequency textures?

Impact of the Second-Order Term δX δW
In Section 3.2, you mention that the second-order term δXδW can be ignored as it has minimal impact on performance. However, under low-bit quantization (e.g., 2-bit), the magnitudes of δX and δW increase significantly. Does the relative contribution of activation quantization error δX, weight quantization error δW, and the second-order error δX δW change under such extreme low-bit conditions? Is there any quantitative analysis supporting the assumption that δX W remains the dominant term in ultra-low-bit scenarios?

Generalization to Different SR Architectures
CondiQuant demonstrates notable efficiency (execution time < 19.0 seconds) and performance improvement on the SwinIR-light network; however, it remains unclear how well the method generalizes across different super-resolution architectures. Could the authors provide experimental results or analyses to verify CondiQuant’s generalization and applicability to other architectures (e.g., SwinIR-B, MambaIRv2-light) while maintaining its performance gains and computational efficiency?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
1. This paper proposes CondiQuant, a PTQ method for SR that reduces sensitivity to activation quantization by explicitly lowering the condition number of weight matrices.
2. The paper derives a clean link between quantization error and conditioning via Eq. (5), it motivates conditioning-aware reparameterization. Then it uses a simple proximal-gradient scheme: a gradient step to preserve output, plus a closed-form proximal step in the SVD space to compact singular values—no inference-time overhead.

### Strengths
1. The paper is well organized, and I appreciate the strong and intuitive theory: ties activation quantization error to κ(W), giving a clear knob to turn (conditioning) and a reason it should help.
2. The method is practical and lightweight: the proximal step has a closed form in SVD, integrates into standard PTQ flows, and adds zero runtime overhead at inference.
3. This method achieved impressive results on 5 essential benchmarks than baselines, especially on 2-Bit situation.

### Weaknesses
see questions part

### Questions
1. Can this method be applied to arbitray-scale image super-resolution's backbone? Are there any related results? In this scenario the scale factor should not be limited to x2, x3, x4.
2. I think some implementation details could be clearer, exactly which layers get processed/skipped, numerical stability for big SVDs, and integration details with 2DQuant are lightly described.
3. I am curious about the generalization: Even though is method is designed for SR, have you tested on RCAN, EDSR, or larger SwinIR variants? and how about other low-level tasks (denoising, deblurring)? Does the rank-deficiency and the benefit hold up?
4. About compute and memory: For large layers, do repeated SVDs become a bottleneck on consumer GPUs (e.g., RTX 3090)? Are you using truncated/batched SVD? What’s the typical runtime/memory profile?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed Condition number based quantization for super resolution task. Specifically, based on the linear layer output decomposition analysis, the authors (1) found out that the output is mostly affected by the quantization error on activation, (2) to control the impact of activation quantization error, the method chose to control the condition number of the weight matrix. Proximal gradient descent aws used to learn the weight matrix with small condition number while minimizing the output reconstruction error. The authors applied the method to SwinIR light and demonstrated better SR metrics compared to other quantization methods for 2/3/4-bit quantization.

### Strengths
The authors provided detailed error decomposition analysis and proposed the condition number based quantization which is sound and looks effective on the SR task.

### Weaknesses
I have concerns in the following perspectives:
- In equation (3), we know that the activation quantization error is the main factor causing degradation. To control the error, and based on the inequality that $||W \delta x||_2 \leq ||W||_2 ||\delta x||_2$, why cannot we directly penalize the the spectral norm the weight matrix (i.e., the max singular value of the weight matrix W) while minimizing the reconstruction error, i.e., replace the regularization term in equation (7) by spectral norm? In this way, we can also control the sensitivity of output to the activation quantization error. The paper did not provide convincing ablation study.
- Instead of optimizing w.r.t. the condition number, the authors applied a proxy objective. Then the author applied proximal gradient descent. I would like to know given that the proxy objective is differentiable, what would happen if we directly use standard SGD to minimize the objective function. The paper did not provide convincing ablation to show readers why proximal gradient is actually required for the optimization.
- Although the proposed method is not QAT, it actually tuned the model weights during PTQ. (1) how is the weight matrices optimized? I understand the authors used about 100 "calibration images". Are the weight matrices in each linear layer of the SwinIR (like the QKV projection, FFN projection) optimized sequentially? or it is done layer by layer? (2) since the weights of the model are optimized during the quantization process, did the author actually compare with other quantization methods that enabled weight optimization as well?
- I did not see why that condition number quantization is specific to image super resolution task. I believe in most cases, activation quantization error will be the major source of error for other vision tasks as well, possible even in NLP domain as well. Then, only applying this quantization method to image SR is not enough to showcase the generality of the proposed approach. I would recommend authors apply the method to other vision tasks, such as detection and segmentation at least, and even NLP tasks.

### Questions
please refer to weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
