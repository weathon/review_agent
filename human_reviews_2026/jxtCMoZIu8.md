# BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Binary Neural Networks (BNNs), which constrain both weights and activations to binary values, offer substantial reductions in computational complexity, memory footprint, and energy consumption. These advantages make them particularly well suited for deployment on resource-constrained devices. However, training BNNs via gradient-based optimization remains challenging due to the discrete nature of their variables. The dominant approach, quantization-aware training, circumvents this issue by employing surrogate gradients. Yet, this method requires maintaining latent full-precision parameters and performing the backward pass with floating-point arithmetic, thereby forfeiting the efficiency of binary operations during training. While alternative approaches based on local learning rules exist, they are unsuitable for global credit assignment and for back-propagating errors in multi-layer architectures. This paper introduces Binary Error Propagation (BEP), the first learning algorithm to establish a principled, discrete analog of the backpropagation chain rule. This mechanism enables error signals, represented as binary vectors, to be propagated backward through multiple layers of a neural network. BEP operates entirely on binary variables, with all forward and backward computations performed using only bitwise operations. Crucially, this makes BEP the first solution to enable end-to-end binary training for recurrent neural network architectures. We validate the effectiveness of BEP on both multi-layer perceptrons and recurrent neural networks, demonstrating gains of up to $+6.89$% and $+10.57$% in test accuracy, respectively. The proposed algorithm is released as an open-source repository.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents Binary Error Propagation (BEP), a new training algorithm for binary neural networks (BNNs) where both weights and activations are binary. The key innovation is a fully binary, bitwise analog of the backpropagation chain rule, allowing end-to-end propagation of binary error signals through both feedforward and recurrent architectures. The method is empirically measured against prior binary optimization methods and QAT on several tasks and model architectures, showing test accuracy improvements.

### Strengths
1. The method is reasonable. And the formalization of a binary version of global credit assignment for BNNs is clear.
2. The empirical results are broad, showing the advantage of BEP.

### Weaknesses
1. The experiments are not sufficient. Although the authors conduct experiments on both MLP and RNN models to demonstrate the effectiveness of their method, the experimental setup appears somewhat toy. Could the authors perform experiments on larger-scale networks (e.g., comparable to ResNet in size) and datasets (e.g., ImageNet-1k)? In addition, since the BNN field has been studied for a long time, could the authors compare BEP with other BNN methods (like [1-3]) beyond QAT to further validate its effectiveness?

[1] Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm

[2] MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?

[3] XNOR-Net++: Improved Binary Neural Networks

2. Have the authors measured the actual training overhead, such as training time and memory usage, to demonstrate that BEP indeed achieves better training efficiency compared to the baseline?

3. The paper format does not appear to follow the ICLR style. The pages seem wider and contain more content, which might be unfair to other submissions.

### Questions
1. Similar to Weakness 1: Can BEP be applied to deeper binary CNNs or transformer-style architectures without encountering collapse or stagnation? Have the authors attempted to apply BEP to modern vision or language models beyond the provided MLP/RNN settings?

2. Batch normalization seems to contribute significantly to performance improvement. Have the authors considered adopting a binary analog or an alternative form of normalization within BEP?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Binary Error Propagation (BEP), a new training algorithm for binary neural networks that avoids surrogate gradients. Instead of real-valued backpropagation, BEP defines a discrete backward rule that propagates binary "desired activations" using only bitwise operations. The method updates integer hidden weights through sparse binary masks that act as an adaptive learning mechanism. BEP is evaluated on MLPs and RNNs across several datasets, showing higher accuracy than quantization-aware training (QAT) and the method proposed in Colombo et al. (2025). The authors claim that this is the first fully binary error Backpropagation (BP) algorithm capable of effectively training BNNs without relying on floating-point gradients.

### Strengths
- The paper is clearly written and well organized, with notation that is consistent and easy to follow.
- The presentation is concise and direct, and I found the mathematical derivations correct and well-grounded.
- The method is novel and well motivated, and the experimental results are consistent with most of the claims.

### Weaknesses
- In Section 3.1, the authors obtain binary input representations using a fixed binarization method (median or thermometer encoding) but do not analyze how different binarization functions affect BEP's performance.

- The margin parameter $r$ in Eq. (1) determines when binary updates are triggered, but its value and sensitivity are not analyzed. Since this parameter effectively affects the learning dynamics, the authors should provide an ablation to show how it influences performance.

- The paper asserts that the sparsity mask $M_l$ fulfills the role of the learning rate in classical BP, yet this is only mentioned in the text. Providing empirical evidence would strengthen this claim.

- Experiments are limited to two- and three-layer MLPs. It would be helpful to discuss whether BEP scales to deeper binary networks, as the discrete backward propagation might face stability issues analogous to vanishing gradients.

- The paper should include a limitations section. While the authors briefly mention possible future directions, they should clearly articulate the current constraints of BEP.

Minor comments:
- The related work section would benefit from including recent works such as "BiPer: Binary Neural Networks using a Periodic Function (CVPR 2024)"
- The small QAT (w/o batchnorm) plots embedded in each subfigure of Figure 2 are difficult to interpret. I recommend improving their presentation or visibility for better clarity.
- In sections 4.3 and 4.4, the term "window length" is not clearly defined.

### Questions
- Have you tested other binarization functions, and can you comment on how sensitive BEP is to this choice?
- The margin parameter $r$ in Eq. (1) controls when updates are triggered. Have you analyzed how different $r$ values affect training stability or accuracy?
- You state that the sparse mask $M_l$ plays a role similar to a learning rate. Could you provide additional analysis to support this interpretation?
- Have you tested BEP on deeper binary MLPs? If not, could you comment on potential challenges in extending BEP to deeper networks?
- Given that most prior work on BNNs targets convolutional architectures, can you elaborate on how BEP might extend to CNNs? 
- Did you test BEP on tasks other than classification? Might it work?

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
The work proposes a novel alternative to the backpropagation algorithm by reformulating it in the binary domain to reduce computational cost. Unlike Quantization-Aware Training, BEP performs both the forward and backward passes entirely in the binary domain, relying solely on bitwise operations. The paper provides theoretical foundations and experimental results that validate this reformulation of backpropagation.

### Strengths
Novelty: the paper introduces an alternative approach to regular backpropagation by reformulating the algorithm to work for binary weights.


Different from other approaches that rely on real value parameters, like a straight-through estimator, this work focuses on the idea of computing forward and backward on the binary domain. 


The work displays consistent improvements over the classification task on two datasets.

### Weaknesses
1. Lack of organization, it is difficult to read smoothly, as figures, such as Figure 2, are on page 7, and mentioned in page 8, same as Table 1, where it is mentioned in page 9, and it is on page 8.


2. The authors claim computational efficiency and memory reduction, but ablation studies over flops, training, and inference time, as well as details over the computational equipment, are not stated.


3. The authors propose several hyperparameters, even though in section 4.4 is an analysis. Further ablation studies should be conducted on the sensitivity of parameters like r and pr.


4. Lack of training details hinders the reproducibility of the work, as it is not stated the epochs or the number of iterations used in the experiments. 


5. The update rule should be further clarified. Even though there are references to previous works, it should be explicitly stated where they come from Eq. (8)


6. The binary mask from Equation 9 is not theoretically explained, and its purpose is not clear.

### Questions
How does BEP scale computationally compared to QAT during training?


1. How sensitive is BEP to initialization and hyperparameter selection across datasets?


2. Does the algorithm have similar results across different tasks to classification, for instance, binary segmentation?


3. How does BEP handle vanishing or exploiting gradients in comparison to the standard backpropagation algorithm?


4. How does the algorithm work for larger spatial dimensions and complex feature datasets, such as the celebA dataset with the multilabel classification task?

### Soundness
3

### Presentation
2

### Contribution
3
