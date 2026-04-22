# LMS: Learnable Maximum Spike with Optimal Spike Representation for High-Performance and Efficient Spiking Neural Networks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Spiking Neural Networks (SNNs) have garnered increasing attention due to their brain-inspired mechanisms. By encoding information with sparse binary spikes, they replace multiplications with additions, substantially reducing energy consumption. However, the binary spike emission inherently leads to significant information loss. In this work, we propose a Learnable Maximum Spike (LMS) neuron, which emits integer values during training and dynamically learns the maximum membrane potential for each layer based on its own membrane potential distribution, thereby determining the maximum integer value the neuron can emit (referred to as spike maximum). Additionally, we introduce a decay balancing coefficient that allows the spike maximum to adapt to the gradient and change in membrane potential distribution between the early and late stages of training, thereby further enhancing the network's performance. Finally, to preserve spike-driven inference, we transform the binary representation problem of emitted values into an integer programming problem, yielding an optimal spike representation of integers that minimizes energy consumption. Extensive experiments have validated the effectiveness of the proposed LMS neuron, which consistently outperforms current state-of-the-art methods on static datasets (CIFAR10, CIFAR100, ImageNet) and a neuromorphic dataset (CIFAR10-DVS). Furthermore, LMS requires less inference memory (**-7.16**\%), shorter inference time (**-12.09**\%), and lower energy consumption (**-61.9**\%).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an improvement over Integer LIF (I-LIF). I-LIF reduces training cost by using integer spikes during training and binary spikes during inference; this work goes one step further and makes the maximum integer value trainable and layer-specific. The maximum is updated online in a sliding-average manner similar to the statistics used in Batch Normalization. After training, each spiking layer has a fixed, layer-wise integer spike maximum. To further cut inference cost, the authors propose an integer-programming-based algorithm that decomposes an integer spike into the shortest possible weighted sequence of binary spikes. Experiments on several datasets show lower energy, lower latency, and higher accuracy than previous methods.

### Strengths
1. The paper is clearly written, the motivation is sound, and the methodological details are well described.  
2. State-of-the-art results are obtained on multiple datasets, confirming the effectiveness of the approach.

### Weaknesses
1. Compared with I-LIF inference, the proposed method appears to require a distinct weight for every time-step during inference. For long sequences, does this incur large memory overhead to store these per-step weights? Moreover, does this prevent the method from generalizing to variable-length sequences? In general, decomposing spikes along the time axis seems questionable—one could directly binarize ANN activations, yielding an “SNN” that is hardly different from the original ANN.  
2. Source code is not provided in the supplementary material, which reduces reproducibility.

### Questions
1. Are the reported inference results obtained with the integer spikes or with the decomposed binary spikes?  
2. The ImageNet accuracy is noticeably lower than that of recent SNN Transformer architectures. Why were those architectures not tested?

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
This paper proposes the Learnable Maximum Spike (LMS) neuron for high-performance, efficient Spiking Neural Networks (SNNs). Addressing binary spike-induced information loss, LMS emits integers during training, dynamically learns layer-specific maximum membrane potential (via membrane potential distribution), and uses a cosine-decayed balancing coefficient for stable training. It also formulates integer-to-binary spike representation as an integer programming problem to minimize inference energy. Experiments on CIFAR10/100, ImageNet, and CIFAR10-DVS show LMS outperforms SOTA methods, with lower inference memory (-7.16%), time (-12.09%), and energy (-61.9%).

### Strengths
First, it dynamically learns layer-specific spike maxima via membrane potential distribution, balancing SNN expressive capacity and computational efficiency.

Second, its cosine-decayed balancing coefficient stabilizes training by adapting to early/late-stage membrane potential changes.

Third, it minimizes inference energy via integer programming-based optimal spike representation, cutting energy by 61.9%.

### Weaknesses
First, it should supplement comparisons with more latest SOTA methods to better highlight LMS’s advantages in diverse scenarios.

Second, the brain diagram shown in Figure 1 is unnecessary as it doesn’t aid in explaining LMS’s core mechanisms.

### Questions
see Weaknesses

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
5

### Summary
This paper introduces a Learnable Maximum Spike (LMS) neuron. Based on each layer’s membrane-potential distribution, the network adaptively learns that layer’s maximum allowable integer spike. To balance rapid distribution tracking early in training with stable convergence later, a cosine-annealed balancing coefficient dynamically adjusts the update rate. At inference, integer outputs are converted into binary spike sequences by formulating an integer programming problem that incorporates firing-rate considerations to minimize energy. Across multiple datasets, LMS achieves higher or comparable accuracy with fewer inference steps, while also reducing memory usage, latency, and energy consumption.

### Strengths
1.The motivation is clear. Given that I-LIF is a strong baseline, revealing its limitations and proposing substantial improvements is important. Replacing I-LIF with the LMS neuron can improve accuracy while reducing energy.

2.The proposed per-layer scheme that learns the maximum integer spike a neuron may emit is conceptually simple and methodologically transparent. Its parsimony, together with the observed gains, makes the approach academically compelling.

3.Casting spike representation as an integer programming problem with an energy-minimization objective fills an efficiency-optimization gap that prior work has not fully addressed.

4.Extensive experiments validate the effectiveness of the method.

### Weaknesses
1.LMS introduces additional MACs, but the paper does not report the proportion of total energy consumption attributable. If that share is large, it would undermine the original motivation for SNNs.

2.The integer-programming solve is performed once after training, yet its solving time as a fraction of the total training time remains unclear.


3.The x-axis labels in Figure 4 are too small to read. Figures 1 and 3 appear partially redundant. Please justify the necessity of both or consider consolidating them.

### Questions
1.As dataset complexity increases, do neurons learn larger maximum integer spike?

2.In I-LIF, is $C$ set equal to $D$? The reported I-LIF inference appears not to use the optimal spike representation. This should be stated explicitly in the caption of Table 4 to avoid confusion.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the Learnable Maximum Spike (LMS) neuron, which is based on integer LIF (ILIF) neurons, for directly trained SNNs. To optimize the existing ILIF, the authors propose a method that optimizes ILIF based on the neuron's expression range. During training, the ILIF expression range is determined by the maximum value at each layer. Furthermore, for further optimization, they perform optimization using IP to minimize spike firing while reducing the inference time step. The proposed method has been validated on several image classification tasks.

### Strengths
1. Optimization for ILIF neurons. The optimization method proposed in this paper will be helpful for efficient application of ILIF.
2. Competitive empirical results at low time steps. The paper reports strong accuracy–efficiency trade-offs: e.g., ImageNet ResNet34 71.36% with T×C = 1×3.33.

### Weaknesses
Insufficient analysis of the optimal spike representation
- What is the overhead of the IP solver?
- As shown in Table 6, there are weights that are non-powers-of-two (3, 6). In this case, a MAC operation is required (Equation 11). Is there a way to eliminate the MAC operation? Is this operation included in energy analysis?
- What is the theoretical analysis for the proposed method?
- What is the decoding method for weighted spikes?

Lack of validation
- Lack of validation for applicability to various model architectures (e.g., Transformer)
- Lack of validation for applicability to tasks other than image classification

Lack of direct discussion of neuromorphic hardware
The paper states that the goal of SNNs is energy efficiency on neuromorphic hardware, but there is insufficient discussion on whether the proposed method is feasible on neuromorphic hardware.

### Questions
Please refer to Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
