# One-Hot Multi-Level LIF Spiking Neural Networks for Enhanced Accuracy-Latency Tradeoff

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 3, 3

## Abstract
Spiking neural networks (SNNs) hold significant promise as energy-efficient alternatives to conventional artificial neural networks (ANNs). However, SNNs require computations across multiple timesteps, resulting in increased latency, heightened energy consumption, and additional memory access overhead. Techniques to reduce SNN latency down to a unit timestep have emerged to realize true superior energy efficiency over ANNs. Nonetheless, this latency reduction often comes at the expense of noticeable accuracy degradation. Therefore, achieving an optimal balance in the tradeoff between accuracy and energy consumption by adjusting the latency of multiple timesteps remains a significant challenge. In this paper, we introduce a new dimension to the accuracy-energy tradeoff space using a novel one-hot multi-level leaky integrate-and-fire (M-LIF) neuron model. The proposed M-LIF model represents the inputs and outputs of hidden layers as a set of one-hot binary-weighted spike lanes to find better tradeoff points while still being able to model conventional SNNs. For image classification on static datasets, we demonstrate M-LIF SNNs outperform iso-architecture conventional LIF SNNs in terms of accuracy ($2$% higher than VGG16 SNN on ImageNet) while still being energy-efficient ($20\times$ lower energy than VGG16 ANN on ImageNet). For dynamic vision datasets, we demonstrate the ability of M-LIF SNNs to reduce latency by $3\times$ compared to conventional LIF SNNs while limiting accuracy degradation ($<1$%).

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces a one-hot Multi-Level Leaky Integrate-and-Fire (M-LIF) neuron model, which expands neuron outputs beyond binary spikes by using multiple binary-weighted spike lanes. The M-LIF model enhances the accuracy-energy tradeoff by enabling higher accuracy with fewer timesteps compared to conventional SNNs. Experimental results demonstrate that M-LIF SNNs achieve higher accuracy on static datasets like ImageNet and significantly reduce latency on dynamic datasets like DVS-CIFAR10 while maintaining energy efficiency.

### Strengths
1. Significance: The paper tackles a crucial challenge in SNNs by developing a method to reduce the number of timesteps while maintaining high accuracy, addressing the need for more energy-efficient SNNs.

2. Clarity: The paper is exceptionally well-written and organized.

3. Validation: The authors perform comprehensive experiments on both static and dynamic datasets.

### Weaknesses
1. Novelty: The proposed M-LIF neuron model closely resembles existing concepts such as multi-spike, burst spike, and multi-threshold neurons. While the approach of restricting outputs to powers of two introduces some variation, it does not fundamentally differentiate the model from prior research, potentially limiting its originality. And the author did not discuss any of those methods.

2. Performance: Although the paper presents extensive experiments, the results on datasets like CIFAR10 and CIFAR100 under equivalent timesteps (e.g., T=1, S=3 compared to traditional SNNs with T=4) show suboptimal performance. This raises concerns about the effectiveness of the proposed method in improving accuracy.

3. Lack of Hardware Implementation Discussion: The paper does not address how the proposed M-LIF model can be supported by existing or future hardware architectures. It remains unclear how the model can be practically deployed in neuromorphic hardware.

### Questions
1. Could you elaborate on the fundamental differences between your proposed M-LIF neuron model and existing models such as multi-spike, burst spike, and multi-threshold neurons?

2. In your energy calculations, do you factor in the weights associated with different spike lanes? For instance, if your M-LIF model operates with T=1 and S=3, making it equivalent to a conventional SNN with T=4, should the energy consumption not scale proportionally by the number of spikes (i.e., multiply by 4)?

3. How does your proposed M-LIF model align with current or emerging hardware architectures? Could you provide insights into the feasibility and potential challenges of implementing M-LIF neurons in hardware to achieve the claimed energy efficiency improvements?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The manuscript introduces a one-hot multi-level leaky integrate-and-fire (M-LIF) neuron model that represents the output of spiking neural networks (SNNs) as one-hot binary vectors. The focus is on training ultra-low latency spiking models with high accuracy, and the results show that the proposed model can outperform iso-architecture LIF models in terms of accuracy.

### Strengths
The manuscript presents a simple, intuitive approach that is clearly described and achieves high accuracy with just a single timestep on large static image classification datasets, such as ImageNet. This approach offers an interesting solution for developing ultra-low latency SNN models.

### Weaknesses
The manuscript lacks comparisons with other methods that use graded (weighted) spikes. The idea of utilizing multibit outputs (one-hot encoded or otherwise) is not novel, as it has been previously explored, for instance in [1], and is a feature available in neuromorphic hardware like Loihi 2 [2], as well as in commonly used SNN libraries like snnTorch [3]. I recommend that the authors provide a discussion highlighting the unique features of the proposed model in comparison to these existing methods and tools.

Additionally, while the manuscript emphasizes the energy efficiency of the proposed spiking models over other SNN models and artificial neural networks (ANNs), these energy comparisons lack context without specifying the hardware architecture used. Moreover, as noted by the authors, memory access often dominates energy consumption, which is omitted from these comparisons. Given that memory-related energy can be orders of magnitude higher than compute energy, and considering that SNNs may require more memory access than ANNs due to membrane potential updates, the energy comparisons may be unrealistic. I suggest the authors replace these energy comparisons with more objective metrics, such as the number of multiply-accumulate (MAC) operations, memory usage, and average firing rate.

[1] Ponghiran, W., & Roy, K. (2022). Spiking Neural Networks with Improved Inherent Recurrence Dynamics for Sequential Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 36(7), 8001-8008. https://doi.org/10.1609/aaai.v36i7.20771 

[2] M. Davies et al., "Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook," in Proceedings of the IEEE, vol. 109, no. 5, pp. 911-934, May 2021, doi: 10.1109/JPROC.2021.3067593.

[3]  https://snntorch.readthedocs.io/en/latest/index.html-

### Questions
- What unique features distinguish the proposed model from existing models in the literature, as well as from available neuromorphic hardware and open-source SNN projects? How does the proposed model perform in comparison to these alternatives?

- Regarding the energy efficiency claims in the weaknesses section, could the authors elaborate on the assumptions made for the energy comparisons? How might these comparisons change if memory access energy is included?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a novel one-hot multi-level leaky integrate-and-fire (M-LIF) neuron model, which represents activations as a set of one-hot, binary-weighted spike lanes. Experimental results on various datasets demonstrate that the M-LIF model achieves a better trade-off between accuracy and energy efficiency compared to other methods.

### Strengths
1. The paper is well-organized and utilizes some formulas and figures to help understand.
2. The proposed spike lane concept is innovative, as it enhances the information in activation maps while maintaining the energy efficiency characteristic of SNNs. The authors validate the effectiveness of the proposed model through experiments on both static and dynamic datasets. Overall, the M-LIF model achieves the best trade-off between accuracy and energy efficiency.
3. The paper also compares the M-LIF model with LQ-ANN and demonstrates that M-LIF generally performs better with some experiments.

### Weaknesses
1. Given that the proposed spike lanes share similarities with quantized (S-bit activation, S is the number of spike lanes) models, the authors should include additional comparisons, both in terms of accuracy and FLOPs/energy consumption, with state-of-the-art quantization approaches such as [1][2][3]. This would provide a more comprehensive and fair evaluation of the method's performance.

2. Although the authors provide some results for models with varying numbers of spike lanes, an additional ablation study would be beneficial. Specifically, the authors could conduct experiments with a fixed number of time steps, varying the number of spike lanes (S) from 1 to an upper bound (e.g., 8), and plot the accuracy versus S. This would help identify the point of saturation. Additionally, a complementary study could be performed by fixing the number of spike lanes and varying the number of time steps from 1 to an upper bound (e.g., 10). Such an analysis would offer a clearer comparison of the impact of spike lanes versus the time steps on model performance.

[1] Jaehyeon Moon et al, Instance-Aware Group Quantization for Vision Transformers, CVPR 2024

[2] Yefei He et al, BiViT: Extremely Compressed Binary Vision Transformers, ICCV 2023

[3] Yanjing Li et al, Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer, NeurIPS, 2022

### Questions
For the DVS-CIFAR10 dataset, as the authors add an additional multi-level input layer encoding, could the authors provide a latency comparison between the M-LIF model and other existing models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces the one-hot multi-level leaky integrate-and-fire (M-LIF) neuron, aimed at balancing accuracy and energy efficiency in spiking neural networks (SNNs).

### Strengths
S1. It is interesting to explore a new dimension (beyond the time step) for achieving an accuracy-latency trade-off.

S2. The experiments are quite comprehensive, and the source code is valuable.

### Weaknesses
W1. The novelty of employing multi-level thresholds in SNNs appears limited, as similar concepts have been introduced in related works [1,2]. What distinguishes MLIP from these existing approaches?
>[1] Wang et al., "MT-SNN: Enhance Spiking Neural Network with Multiple Thresholds." ArXiv 2023.  
>[2] Feng et al., "Multi-Level Firing with Spiking DS-ResNet: Enabling Better and Deeper Directly-Trained Spiking Neural Networks." IJCAI 2022.

W2. How does MLIF differ from quantization networks? Specifically, does an MLIF with $S=K $ operate equivalently to a K-bit quantized neural network (QNN) with quantized activation values?

W3. Table 1 shows that MLIF significantly increases computational energy consumption, which poses a drawback for MLIP. For instance, on CIFAR10, although MLIP-Transformer with $S=3$ and $T=4$ achieves the highest accuracy (only 0.3% higher than Spike-driven Transformer with $S=1$ and $T=4$), its computational energy usage is considerably higher. This trend is similarly observed on CIFAR100 and ImageNet.

### Questions
The questions I am most concerned about are raised in the Weakness section, particularly W1 and W2.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a one-hot multi-level leaky integrate-and-fire (M-LIF) neuron model to optimize the accuracy-latency trade-off of deep spiking neural networks. The proposed M-LIF model represents the inputs and outputs of hidden layers as a set of one-hot binary-weighted spike lanes. Experimental results demonstrate superior accuracy in image classification results with only one time step.

### Strengths
1. The proposed one times-step SNNs can significantly improve the energy-efficiency by avoiding repeated memory accesses of the membrane potential and reduced number of spikes.
2. The accuracies obtained by the proposed method on ImageNet is good.

### Weaknesses
1. While the paper has moderately strong empirical results, the proposed method is not much different from bit-serial activation quantized neural networks [1], where the spike lanes can be regarded as bits sequentially arriving and accumulating at each neuron. The overhead of the time steps is thus shifted to the spike lanes.

2. I understand there is one-hot encoding in the spike lanes, however, this contribution alone may not be sufficient to warrant acceptance in ICLR. This is also similar to power-of-two quantization [2] that has been extensively explored in several quantization works.

3. I see some empirical comparisons of this method with log-quantized ANNs, and different modes of operations during training. However, during inference, the operations are almost similar. Even during training, I am not sure if the differences significantly affect the accuracies.

4. Several prior SNN works [3-4] have explored the interplay between the ANN activation bit-width and SNN time steps (that bases the foundation of this work), however, they have been ignored in this work.

5. I see only one comparison with one-time-step SNNs in Table 1. It would be better to add more comparisons, including the results from [4].

[1] https://edge.seas.harvard.edu/files/edge/files/precision_batching_bitserial_decomposition_for_efficient_neural_network_inference_on_gpus.pdf

[2] https://openreview.net/pdf?id=BkgXT24tDS

[3] "Are Conventional SNNs Really Efficient? A Perspective from Network Quantization", CVPR 2024

[4] "Can we get the best of both Binary Neural Networks and Spiking Neural Networks for Efficient Computer Vision?", ICLR 2024

### Questions
Please respond to the weaknesses mentioned above.

### Soundness
2

### Presentation
2

### Contribution
1
