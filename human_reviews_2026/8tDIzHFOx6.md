# SPR$^2$Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Low-bit quantization has achieved significant progress in image super-resolution. However, existing quantization methods show evident limitations in handling the heterogeneity of different components. Particularly under extreme low-bit compression, the issue of information loss becomes especially pronounced. In this work, we present a novel low-bit post-training quantization method, namely static priority-based rectifier routing quantization (SPR$^2$Q). The starting point of this work is to attempt to inject rich and comprehensive compensation information into the model before the quantization , thereby enhancing the model's inference performance after quantization. Firstly, we constructed a low-rank rectifier group and embedded it into the model's fine-tuning process. By integrating weight increments learned from each rectifier, the model enhances the backbone network while minimizing information loss during the lightweighting process. Furthermore, we introduce a static rectifier priority routing mechanism that evaluates the offline capability of each rectifier and generates a fixed routing table. During quantisation, it updates weights based on each rectifier's priority, enhancing the model's capacity and representational power without introducing additional overhead during inference. Extensive experiments demonstrate that the proposed SPR$^2$Q significantly outperforms the state-of-the-arts in five benchmark datasets, achieving PSNR improvements of 0.55 and 1.31 dB on the Set5($\times 2$) dataset under 4-bit and 2-bit settings, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SPR2Q, a static priority-based rectifier routing quantization method. The overall pipeline can be divided into two parts. The first part leverages a low-rank rectifier group to enhance the backbond network. The second part uses offline static routing calibration to obtain the SPR2Q table to assign the optimal increment for each layer. The proposed method achieves SOTA performance on Mamba with five commonly used benchmarks.

### Strengths
- The design is effective and clear, obtaining SOTA performance with both metric and visual comparison on the SR task.
- The writing is clear and easy to follow.

### Weaknesses
- The proposed method is only tested on MambaIRv2-light. However, the proposed method can be safely tested on more models, including MambaIRv2_SR2, SwinIR. Please provide the results on these models to demonstrate the generalization ability.
- Please provide the complexity of the SPR2Q model, including model parameters, storage, inference speedup ratio, and GPU memory usage. These metrics are critical to the model lighting.
- The rank of each recitifier module is important. However, the influence of the rank is not discussed in the ablation study. Please provide the results of various ranks and the results of full-rank.

### Questions
- In Figure 1, it should be "update" instead of "updata". The corresponding 
- What is the limit of the contribution from the Rectifier group size?

### Soundness
3

### Presentation
3

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
This paper introduces Static Priority-based Rectifier Routing Quantization (SPR2Q), a novel post-training quantization method for low-bit image super-resolution. Unlike existing approaches that struggle with severe information loss under extreme low-bit settings, SPR2Q injects rich compensation information into the model before quantization by embedding a low-rank rectifier group during fine-tuning. A static rectifier priority routing mechanism then evaluates each rectifier’s capability offline and updates weights accordingly without adding inference overhead.

### Strengths
The idea of learning compensation information through lightweight rectifiers is very interesting. In addition, the use of dynamic routing during training followed by static rectifier routing is an effective design choice, improving performance without increasing computational cost. Experimental results support the effectiveness of the proposed method.

### Weaknesses
1. The method is tested only on a single SR baseline, MambaIRv2-light. To better demonstrate its applicability and generalization capability, it should also be tested against other baselines, such as MaIR (CVPR 2025), EAMamba (ICCV 2025), and First-Order State Space Model for Lightweight Image Super-Resolution (ICASSP 2025).

2. Since the main contribution lies in the use of multiple rectifiers, a more in-depth analysis of their internal behavior is expected. For example, the authors mention that each rectifier handles different types of information, but visual or statistical evidence is needed to support this claim. In addition, the analysis of optimal gate weights should be expanded to better explain their optimality.

3. The performance of PTQ depends heavily on the training dataset. In particular, the proposed offline routing calibration method appears to target different domains; however, the experiments are conducted using only a single training dataset.

4. The paper would benefit from careful linguistic revision and thorough proofreading.

### Questions
The authors used rectifiers of the same size. Have the authors considered using rectifiers with different capacities, so that multiple experts can more effectively distribute the workload?

### Soundness
3

### Presentation
3

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
This paper proposes SPR2Q, a post-training quantization (PTQ) framework specifically designed for image super-resolution (SR) with an emphasis on Mamba-based architectures. The key insight is that current PTQ techniques do not sufficiently prepare the model for quantization, especially under aggressive bit-width reduction (e.g., 2-bit or 1-bit).

To address this, the authors propose two complementary techniques:
1.	Pre-Quantization Fine-Tuning with Fused Rectifiers (PQFR):
Learnable, low-rank rectifiers are fused into model weights before quantization to reduce error and preserve representational fidelity.
2.	Static Priority-Based Rectifier Routing (SPR2):
A mechanism for statically assigning rectifiers to layers using an offline-calibrated routing table, which introduces diversity without runtime overhead.

Extensive experiments on five standard SR datasets and comparisons against three strong Mamba-specific quantization baselines (PTQ4VM, Quamba, and MambaQuant) show consistent performance improvements, especially in low-bit regimes.

### Strengths
1.	The paper clearly identifies a relevant and under-addressed problem: the difficulty of applying post-training quantization to super-resolution models, especially in architectures like Mamba that are sensitive to numerical errors due to their recurrent components. The need for specialized approaches in SR is well-motivated.
2.	The PQFR module is inspired by LoRA but adapted for PTQ in the SR setting, which is novel. The SPR2 routing table design is simple, efficient, and effective. It offers a practical compromise between dynamic routing (high cost) and naive shared rectifiers (low performance). And the method is well-integrated into existing training pipelines and avoids runtime penalties.
3.	The method requires only modest fine-tuning and has negligible inference overhead. It is applicable in edge scenarios where low-bit inference is essential.

### Weaknesses
1.	The entire evaluation is performed on the MambaIRv2-light model. While the motivation is grounded in the Mamba architecture, the method itself (particularly PQFR) should generalize. Add at least one experiment on a Transformer-based SR model (e.g., SwinIR) or a CNN-based model (e.g., EDSR) to support claims of generality.
2.	The routing table optimization (Equation 12) is essential to SPR2Q, yet the paper omits important implementation details. It is unclear whether the optimization of gating weights is performed via exhaustive search, gradient descent, or heuristics. I would like to see more information about how the routing weights ĝ are optimized, how long this process takes, and whether it scales with model size.
3.	Although the paper claims no additional inference cost, no quantitative measurements are presented. This is important to validate the claim that the fused rectifiers do not affect runtime. I would be great if include a table comparing runtime, model size, and memory usage (before and after SPR2Q), even on a single dataset, to support the efficiency claim.

### Questions
See the weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
his paper proposes a static priority rectifier routing quantization framework (SPR ² Q) for the low bit post training quantization (PTQ) problem of Mamba architecture image super-resolution (SR) model. This framework injects learnable compensation information through a pre quantized fusion rectifier module and combines it with a static rectifier priority routing mechanism to provide diversified compensation strategies. It effectively alleviates quantization information loss under extremely low bit (2-bit, 1-bit) settings and achieves better performance than existing SOTA methods such as PTQ4VM and Quamba on five benchmark datasets including Set5 and Urban100. Especially, it achieves PSNR improvements of 0.55dB and 1.31dB on 4-bit and 2-bit Set5 (× 2) tasks, respectively.

### Strengths
1. The method proposed in the paper appears to be quite general and easy to integrate into existing models.

2. The paper provides experimental results under different model settings and bits, as well as ablation experiments.

3. The paper achieved SOTA performance under different settings.

### Weaknesses
1. The LoRA fine-tuning in the paper requires end-to-end training, which brings additional memory and time overhead compared to other PTQ methods. And the paper uses a lightweight MambaIR model. If other versions such as MambaIR_SR are used, can Lora training be efficiently performed?

2. As the paper requires LoRA fine-tuning and weight updates, I am puzzled whether this is a traditional PTQ method and whether it is reasonable to compare it with other PTQ methods that do not require training?

3. The paper does not report the calibration resource comparison, such as GPU time and GPU memory.

4. The paper did not provide ablation experiments on LoRA's rank and router group size for calibration resources and performance.

5. Does 4bit denotes W4A4 or W4A16? And the quantization memory reduction effect and inference acceleration should be reported.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
