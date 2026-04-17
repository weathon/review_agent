# Zeros can be Informative: Masked Binary U-Net for Image Segmentation on Tensor Cores

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 6

## Abstract
Real-time image segmentation is a key enabler for AR/VR, robotics, drones, and autonomous systems, where tight accuracy, latency, and energy budgets must be met on resource‑constrained edge devices. While U‑Net offers a favorable balance of accuracy and efficiency compared to large transformer‑based models, achieving real‑time performance on high‑resolution input remains challenging due to compute, memory, and power limits. Extreme quantization, particularly binary networks, is appealing for its hardware‑friendly operations. However, two obstacles limit practicality: (1) severe accuracy degradation, and (2) a lack of end‑to‑end implementations that deliver efficiency on general‑purpose GPUs.

We make two empirical observations. (1) An explicit zero state is essential: training with zero masking to binary U‑Net weights yields noticeable sparsity. (2) Quantization sensitivity is relatively uniform across layers. Motivated by these findings, we introduce Masked Binary U‑Net (MBU‑Net), obtained through a cost‑aware masking strategy that prioritizes masking where it yields the highest accuracy‑per‑cost, reconciling accuracy with near‑binary efficiency.

To realize these gains in practice, we develop a GPU execution framework that maps MBU‑Net to Tensor Cores via a subtractive bit‑encoding scheme, efficiently implementing masked binary weights with binary activations. This design leverages native binary Tensor Core BMMA instructions, enabling high throughput and energy savings on widely available GPUs. Across 3 segmentation benchmarks, MBU‑Net attains near full‑precision accuracy (3\% average drop) while delivering 2.04$\times$ speedup and 3.54$\times$ energy reductions over a 16-bit floating point U‑Net. The code is available at https://github.com/ChunshuWu/MBU-Net.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To address the computational and power limitations of U-Net for high-resolution image segmentation on edge devices, the authors propose a Masked Binary U-Net (MBU-Net) method. Through the experimental analysis of binary U-Net, the authors found that introducing zero mask training produces significant sparsity (more than 95% of the layers have zero weights), and the sensitivity of the layers to the mask is roughly uniform. Motivated by this observation, the authors design a cost-aware masking strategy with three-valued weights (-1, 0, +1) and binary activations, and develop an efficient inference framework based on GPU Tensor Core. Experimental results on three medical/image segmentation datasets show that the proposed method achieves about 2x speedup and 3.54x energy reduction compared to FP16 U-Net, but at the cost of 3% accuracy degradation on average. The authors note that the method relies on the native binary matrix multiplication instructions of the GPU, which is limited in performance on the H100 architecture with this support removed, and the accuracy loss caused by binarization may not be acceptable in some application scenarios.

### Strengths
1.In this paper, the zero masking state in the weights of a binary U-Net improves the segmentation quality.

2.The authors report latency evaluations across multiple GPU platforms, demonstrating an average 2.04× speedup and 3.54× power efficiency improvement compared to a 16-bit floating-point U-Net baseline.

### Weaknesses
1.The authors only evaluated their method on the typical U-Net architecture shown in Fig. 2, without providing results on other segmentation models, which undermines the robustness and generalizability of the proposed approach to other network architectures.

2.The BACKGROUND and RELATED WORK sections should be merged into a single section.

### Questions
1.The authors conducted tests on A100, H100, RTX 2080Ti, and Jetson Orin Nano. However, it is unclear how the different hardware architectures of these GPUs affect the proposed method. For instance, A100 uses NVIDIA Ampere architecture, H100 uses Hopper architecture, and RTX 2080Ti uses Turing architecture. What are the implications of these architectural differences on the performance and applicability of the proposed approach?

2.The authors propose a zero masking method in the manuscript, but the logic and parameter settings for applying zero masking to weights are not clearly explained. Is this an adaptive adjustment or a fixed threshold approach?

3.The authors propose a subtractive bit-encoding method tailored for Tensor Cores in commercial GPUs. However, can this method be generalized into a universal and elegant hardware design framework or dataflow scheduling strategy that provides insights for hardware platforms beyond GPUs, rather than serving merely as an experimental optimization technique specific to GPUs?

4.Please see the Weaknesses section for other issues.

### Soundness
2

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
4

### Summary
This paper proposes a masked-binary mechanism with cost-aware layer selection that adds a "zero state" to binary networks, improving expressiveness and stability while preserving bit-operator efficiency. Masking selective low-cost layers achieves near "full-mask" accuracy with predictable hardware performance.

### Strengths
1. The paper addresses real-time, high-resolution segmentation on edge devices, focusing on accuracy, latency, and energy. The empirical insights are straightforward and motivate a practical design, making the contribution both coherent and relevant.
2. The work combines masked binary weights with a cost-aware layer selection and a GPU execution framework using Tensor Cores. The subtractive bit-encoding and native binary operations show strong engineering rigor and enable deployable efficiency gains.
3. Experiments across multiple GPUs and datasets demonstrate near full-precision accuracy with notable speed and energy improvements. The comprehensive metrics and ablations support robustness, portability, and practical relevance.

### Weaknesses
1. The method adds ternary weights to selected layers of a binary U-Net to approach full-precision accuracy with near-binary efficiency. However, the paper does not clearly compare against ternary quantization baselines. Could the authors clarify in which dimensions MBU-Net outperforms classical ternary methods?
2. The paper enhances binary networks by masking some weights to zero. Intuitively, This seems related to sparsity/pruning. Could the authors clarify: can this method be considered as a combination of pruning and binary quantization? If not, what are the key distinctions?
3. While the analysis and experiments are convincing, the authors focus on the classical U-Net. I am concerned about robustness and generalizability: can the proposed method be generalized to similar segmentation architectures (e.g., U-Net variants) for broader applicability?

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The author’s propose a real-time image segmentation approach using UNet which they name Masked Binary UNet. The approach includes an explicit zero-state (which was found to be essential for performance), and layers are quantized uniformly across layers along with an execution framework that maps the model to TensorCores. They show that this approach performs much faster than the full precision variants while maintaining most of its original performance.

### Strengths
S1 - Interesting finding that including zero-mask values seems to preserve the performance of the UNet model. Moreover, an interesting tensor-core deployment scheme was shown. 

S2 - The proposed system performs roughly just as well as the full-precision UNet’s while being much faster.

### Weaknesses
W1 - While INT8 and INT4 models are evaluated for performance (in 4.3), their latency / speed / energy is not shown. Could it be that INT8 and INT4 perform just as well as the proposed method in terms of speed?
 
W2- Authors should’ve compared to another simple baseline, namely using TensorRT for inference optimization / quantization as in https://arxiv.org/pdf/2012.12259. 

W3 - The paper is lacking in experimental results. For example, Section 4.4 summarizes insights already gathered in 4.3 and 4.2.1. On the whole, the table in 4.3 could’ve just had an extra column indicating the FPS and nothing would’ve been lost if 4.2.1 and 4.4 were removed.

### Questions
* Q1 - Why is Section 4.4 considered an ablation study? No parts of the system were ablated? Could’ve tried random costs on the layers as opposed to the weighting scheme as well. 
* My main concern is W1, followed by W2 and W3. Namely, why didn't the authors show INT8 and INT4 in the pareto curve?

### Soundness
2

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
3

### Summary
In this paper, the author argues that traditional binary U-Nets lose too much accuracy and lack efficient GPU implementations. To address this, the paper proposes MBU-Net, which introduces a zero state to make weights ternary and applies it selectively through a cost-aware masking strategy. It also builds a custom GPU framework using subtractive bit encoding so ternary weights can run efficiently on Tensor Cores, achieving near full-precision accuracy with much faster and more energy-efficient inference.

### Strengths
* This paper is well organized and easy to follow, especially for readers who are not familiar with this area.
* The insights that introducing large amount of ‘zero state’ into pure binary U-Nets is extremely helpful on segmentation task is great
* A major strength of this work is its practical GPU execution framework, which provides tangible, measurable speedup on widely available NVIDIA GPUs
* Innovatively unlocks Hardware Potential with “Subtractive Bit-Encoding”, extending the BMMA (Binary matrix multiply) to the ternary computation.

### Weaknesses
* Introducing ‘zero state’ into pure binary U-Net can significantly boost performance on segmentation, the performance of such a method applied on other types of networks and tasks remains unclear. However, the paper’s innovation on the low-level hardware implementation is still very solid.

### Questions
Please refer to the weakness part

### Soundness
3

### Presentation
3

### Contribution
3
