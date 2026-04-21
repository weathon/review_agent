# Towards Zero Memory Footprint Spiking Neural Network Training

- Avg Score: 5.75
- Decision: Reject
- Scores: 5, 5, 8, 5

## Abstract
Spiking Neural Networks (SNNs), as representative brain-inspired neural networks, emulate the intrinsic characteristics and functional principles of the biological brain. With their unique structure reflecting biological signal transmission through spikes, they have achieved significant success in processing temporal data. 
However, the training of SNNs demands a substantial memory footprint due to the added storage needs for spikes or events, resulting in intricate architectures and dynamic configurations.
In this paper, to address memory constraints in SNN training, we introduce an innovative framework characterized by a remarkably low memory footprint. We \textbf{(i)} design a reversible SNN node that retains a high level of accuracy. Our design is able to achieve a $\mathbf{58.65\times}$ reduction in memory usage compared to the current SNN node. We \textbf{(ii)} propose a unique algorithm to streamline the backpropagation process of our reversible SNN node. This significantly trims the backward Floating Point Operations Per Second (FLOPs), thereby accelerating the training process in comparison to the current reversible layer backpropagation method. By using our algorithm, the training time is able to be curtailed by $\mathbf{23.8}$ % relative to existing reversible layer architectures.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Training normal spiking neural networks (SNNs) directly incurs a significant memory cost. To tackle this issue, this paper introduces a reversible spiking neural node that eliminates the need to store intermediate states during backpropagation by recalculating the states on-the-fly. The authors claim good performance and memory reduction of the proposed SNN node in several vision classification tasks.

### Strengths
1. The research topic is both intriguing and important. The substantial training cost associated with SNNs, particularly the considerable memory overhead, has troubled the research community, undermining the potential of SNNs. Therefore, it is imperative to propose various methods to curtail the training cost.

2. The experiments demonstrate a substantial reduction in memory usage.

### Weaknesses
1. The description of the neural node in Section 3.2 is convoluted and hard to follow. The underlying logic of the proposed neural node remains unclear. The motivation behind introducing such a neural node is ambiguous. The rationale for segregating the states into two groups at each time step is not motivated. The necessity of introducing $\hat{V^t}$ (Eqs. 3 and 6) is not clearly justified. Moreover, is $[Y_1,Y_2]$ identical to $[X_1,X_2]$ of the next layer? Is $[\hat{V_1^t},\hat{V_2^t}]$ identical to $[V_1^{t+1},V_2^{t+1}]$? To enhance clarity, I recommend that the authors initially expound upon the LIF model, introduce the components of the model that prompt dissatisfaction, and subsequently describe their modifications based on the LIF model.

2. Based on my understanding, $[Y_1, Y_2]$ represents the transmitted signal to other SNN nodes. The issue at hand is that these signals are not binary spikes according to eqs. 2 and 5. However, one of the appealing aspects of SNNs is their utilization of binary events for information processing. In essence, a spiking neuron is active only when it encounters spikes, enabling an event-driven regime and an energy-efficient system. However, the proposed neuron nodes deviate from this advantageous feature by transmitting real-valued signals: neurons are active all the time, and the communication cost is high. 

3. Why the neural node can achieve layer reversibility during training? Considering the feedforward connection from the l-th layer to the (l+1)-th layer, $x^{l+1}[t] = W * y^l[t]$, where W is not invertible. I am curious about the approach taken to recalculate $y^l[t]$ from $x^{l+1}[t]$. After all, VGG is used as the network architecture, where the weight matrice are not invertible.

4. While the issue of high training costs is particularly pronounced for large-scale datasets such as ImageNet, this work exclusively tests the proposed method on small-scale datasets (including Tiny-ImageNet). Consequently, the results may not be as convincing, primarily because the training cost for small-scale tasks is relatively affordable. It would be beneficial to extend the evaluation to more computationally demanding datasets to underscore the true efficacy and efficiency of the proposed method.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a framework for zero memory footprint spiking neural network training, which includes a reversible SNN node design and a streamlined backpropagation algorithm. The contributions of the paper are the reduction of memory usage and training time, making SNNs more feasible for resource-limited environments such as IoT-Edge devices.

### Strengths
1. This study provides vivid illustrations for the computations of the proposed model, which makes it easy and clear for readers to follow.

2. The experimental results look like convincing.

### Weaknesses
Certainly, the authors provide a detailed introduction to spiking computations. However, it is unclear which component or computational step contributes to the memory savings. This lack of clarity might give the impression that the paper reads like a technical report. Providing further elaboration on the modifications and corresponding improvements proposed by this work would be highly beneficial.

In addition, there are a few instances of unclear phrasing:

It is preferable to use "spiking neuron" instead of "spiking neural node" in the machine learning community.

The overall layout of the paper could benefit from improvement, especially on pages 4-6, which can be somewhat arduous to read.

While the derivation of the formula is quite clear, there are still areas that could be refined. For example, using notation like $[V_{11}, V_{21}; V_{12}, V_{22}]$ may be more effective than the current [v_1, v_3; v_2, v_4], which is then divided into V[1] and V[2]. Additionally, the case distinction between V and v may not be of significant importance.

### Questions
See weaknesses.

I will consider raising my score if the authors fixed my doubts.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a new reversible spiking module to circumvent the high memory cost of storing the spatial-temporal computational graph needed for (surrogate) gradient-based training spiking neural networks. Their method trades memory for additional computation during the backward pass, but they introduce a more efficient way to compute the gradient of the reversible module compared to how it is done usually in revertible neural networks. They demonstrate quantitative improvements in terms of memory and compute time of the gradient compared to many previous SNNs approaches, on various vision tasks (CIFAR10/100, DVCGesture, Tiny-ImgaNet...) and network architectures, while maintaining the same level of accuracy.

### Strengths
**Originality:** The paper is original because it combines research from different fields: invertible neural networks and spiking neural networks. To my knowledge it has never been done before.

**Quality:** The quality is good, the paper is very detailed, very quantitative and thorough in its comparison with other work.

**Clarity:** The paper is clearly written, though some sentences could be better formulated.

**Significance:** The approach is significant since compute is generally cheaper than memory access in hardware, so to me it makes sense to trade one for the other in the context of edge AI, which is the main use case of SNNs.

### Weaknesses
The paper is already good, but one weakness I see is that the approach is only tested on static vision benchmarks that arguably do not require a lot of temporal processing. 

The paper would become great if the RevSNN approach was tested on benchmarks that require long sequences, and where the other approaches fail because of out-of-memory. An example of such a task could be image classification from sequences of pixels, as described in [1], but other tasks are possible. The goal is to find a setting where only RevSNN manages to learn the task thanks to its low memory requirement.

[1] Tay, Yi, et al. "Long range arena: A benchmark for efficient transformers." arXiv preprint arXiv:2011.04006 (2020).

### Questions
Questions:

- Does the reversible nature of the RevSNN imposes any constraint on the dimensionality of the layers? I am thinking that the dimensions have to be the same for the module to be invertible. If yes, how is it handled in architectures like VGG?

- I don't understand why Figure 4 mentions GNN (Graph Neural Network?), whereas the architectures in the tables are all ConvNets?

- It would be nice if the plots in Figure 6 shared the same y-axis, so that comparison is easier when varying the timesteps.

The authors might want to consider citing [2] and [3], as I think they are related to their approach.

[2] Bauer, Felix C., et al. "EXODUS: Stable and efficient training of spiking neural networks." Frontiers in Neuroscience 17 (2023): 1110444.
[3] Gomez, Aidan N., et al. "The reversible residual network: Backpropagation without storing activations." Advances in neural information processing systems 30 (2017).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the challenge of memory consumption in the training of Spiking Neural Networks (SNNs). The authors design a special forward and backward computation pattern for SNNs and it does not need to store the intermediate features. Experiments were conducted to show the GPU memory efficiency.

### Strengths
+ The paper studies how to reduce the memory footprint of the SNNs during training, which could be an important issue. 

+ Though the presentation is not clear, the proposed framework seems to be novel.

### Weaknesses
- The methodology part of this work is not well-presented. There are no intuitions or motivations to demonstrate why we have to design this symmetric forward/backward implementation. The notation system is very chaotic, the authors did not mention what is the input/output/membrane potential, and how you denote traditional LIF nodes' implementation. The figures are just copies of the equation. 

- This work did not reach the same level of accuracy as the recent SOTA SNN works. For example: TEBN [1], MPBN [2].

- Compared to the original SNNs, this new framework increases the training latency. It is okay to have a little bit higher latency, since most memory efficiency methods trade space for time. However, in that case, I hope the authors can demonstrate the real challenging case, i.e. heavy GPU load tasks, like ImageNet-1k training with a single GPU. How much batch size can this method and the traditional method reach and what are the accuracies? These results are missing. 


[1] Duan et al., Temporal Effective Batch Normalization in Spiking Neural Networks. 
[2] Guo et al., Membrane Potential Batch Normalization for Spiking Neural Networks.

### Questions
See my weakness above.

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good
