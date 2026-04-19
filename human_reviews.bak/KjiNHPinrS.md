# Timesteps meet Bits: Low-Latency, Accurate, & Energy-Efficient Spiking Neural Networks with ANN-to-SNN Conversion

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 6

## Abstract
Spiking Neural Networks (SNN) are now demonstrating comparable accuracy to intricate convolutional neural networks (CNN), all while delivering remarkable energy and latency efficiency when deployed on neuromorphic hardware. In particular, ANN-to-SNN conversion has recently gained significant traction in developing deep SNNs with close to state-of-the-art (SOTA) test accuracy on complex image recognition tasks. However, advanced ANN-to-SNN conversion approaches demonstrate that for lossless conversion, the number of SNN time steps must equal the number of quantization steps in the ANN activation function. Reducing the number of time steps significantly increases the conversion error, incurring a significant drop in test accuracy. Moreover, the spiking activity of the SNN, which dominates the compute energy in neuromorphic chips, does not reduce proportionally with the number of time steps. To mitigate the accuracy concern, we propose a novel ANN-to-SNN conversion framework, that incurs an exponentially lower number of time steps compared to that required in the SOTA conversion approaches. Our framework modifies the SNN integrate-and-fire (IF) neuron model with identical complexity and shifts the bias term of each batch normalization (BN) layer in the trained ANN. To mitigate the spiking activity concern, we propose training the source ANN with a fine-grained $\ell_1$ regularizer with surrogate gradients that encourages high spike sparsity in the converted SNN. Our proposed framework thus yields lossless SNNs with ultra-low latency, ultra-low compute energy, thanks to the ultra-low timesteps and high spike sparsity, and ultra-high test accuracy, for example, $73.30%$ test accuracy with only $4$ time steps on the ImageNet dataset.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a novel ANN-SNN framework including using QCFS to get the pretrained ANN and then apply a three-step conversion. This article changes the encoding method of IF neurons to binary and further enhance the representation ability. The paper also proposes a fine-grained regularizer to reduce the spiking activity of the SNN, which further improves the compute efficiency.

### Strengths
The paper proposes a novel binary encoded IF neuron and modified the conversion process to fit the binary encoding. The idea of the sparsity regularizer is also interesting.

The experimental results are good. Within 4 timestep, the network can get comparable results with ANN even on ImageNet dataset.

### Weaknesses
> “Note that our neuron model simply postpones the firing and reset mechanism until after the input current accumulation over all the T time steps, and does not change the complexity of the traditional.” 

I am not sure what this means. Does it indicate that the neuron here will accumulate the input without firing and reset until the last timestep? Authors should explain why they change the neuron behavior. Also, authors should demonstrate more detailed ablation study on the proposed binary encoding and conversion framework using the normal IF neuron.


The sparsity regularizer is a loss function that constrain the activation values in the ANN. However, as the coefficient parameter lambda increases, the performance of SNN will decrease, so the author should discuss more to prove that the advantage of reducing by  adding constraints is greater than the performance loss. Also, in Figure 4, the spiking activity with lambda=1e-8 is larger than the spiking activity with lambda=0. Does that mean the sparsity regularizer does not work?

### Questions
Please refer to the concerns addressed in weakness.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a Spiking Neural Network (SNN) architecture with low latency and high model accuracy by introducing an approach of shifting input values by 1 bit (or equivalent to multiplying by 2) at each time step. This shifting mechanism takes advantage of the binary nature of spikes in SNNs, resulting in a significant reduction in the spiking length, scaling down the representation to a logarithmic scale.

### Strengths
The paper is written with remarkable clarity, and the experiments are very comprehensive.

### Weaknesses
The novelty of this paper may be questionable, as a previously published paper [1] discusses the use of similar shifting technology for converting Artificial Neural Networks (ANN) to SNN. Their work provides a more general theory, demonstrating that the scale of shifting can vary not only to 2 but also to other fractions such as 3, 4, and beyond. Thus, it can be argued that this paper can be considered a specific case or an application of the more comprehensive theory presented in [1]. Alternatively, another published paper by Kim et al. [2] explores a similar idea but approaches it from the opposite direction in shifting.


[1] Wang, Z., Gu, X., Goh, R. S. M., Zhou, J. T., & Luo, T. (2022). Efficient spiking neural networks with radix encoding. IEEE Transactions on Neural Networks and Learning Systems.

[2] Kim, J., Kim, H., Huh, S., Lee, J., & Choi, K. (2018). Deep neural networks with weighted spikes. Neurocomputing, 311, 373-386.


The primary distinction between this paper and the previous work [1] is the object of shifting, with this paper proposing to shift the input left, while [1] focuses on shifting the output right. However, it can be argued that this contribution may not be substantial enough to warrant a separate paper submission to a prestigious conference like ICLR, as these two shifting policies are mathematically equivalent and yield nearly identical results.

In conclusion, while the paper presents an interesting concept for improving SNN performance, its novelty is questionable, given the existence of related work [1] that provides a more general theory and [2] that explores similar ideas from a different perspective. The contribution of changing the direction of shifting alone may not be sufficient to justify a separate publication in ICLR.

### Questions
It would be beneficial if the authors could provide a clarification regarding the unique contributions of this paper in relation to the previously mentioned work.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents an approach to convert artificial neural networks (ANNs) to spiking neural networks. The method relies on the quantization of activations. It has been shown that the proposed method can improve the accuracy performance of SNNs and reduce their spiking activities.

### Strengths
The method improves the accuracy performance of SNNs on ImageNet dataset.
The spiking activities have been reduced.

### Weaknesses
The proposed method is similar to quantized neural networks. As such, a comparison is required with quantized neural networks in terms of both accuracy and efficiency. 

The type of each vector/matrix (e.g., integer, real, etc) needs to be specified.

### Questions
My main question is the difference between this work and bit-serial quantized networks in terms of computations?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores the source of conversion error in ANN to SNN and proposed an accurate and efficuent conversion method. The error reduction is a step-up over previous methods in identifying some extra mitigation methods that are shown to be effective as experimental results suggests

### Strengths
* This paper is well organized and mostly easy to follow.
* Experiments are conducted extensively in numerous dataset and compared against recent methods.
* Background are discussed throughly.
* Extra error gaps and mitigation methods, are summurized clearly over previous methods.
* Experimental results, expecially at two steps, demonstrates clear improvement in speed and accuracy over previous methods.

### Weaknesses
* My understanding is that this address some of the error gaps uncovered in the QCFS paper. While the paper has demnstrated faster conversion steps and better accuracy performance, would it be possible to compare and discuss energy against at least QCFS method?

### Questions
Please see above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
