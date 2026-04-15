# OneBNet: Binarized Neural Networks using Decomposed 1-D Binarized Convolutions on Edge Device

- Decision: Reject
- Scores: 3, 5, 3, 3

## Abstract
Nowadays, it is natural to use 2-D convolutions in convolutional neural networks (CNNs) for computer vision, but this paper shows that 1-D binarized convolutions can achieve excellent performance on CPU-based edge devices.  This paper proposes a new structure called OneBNet to maximize the effects of 1-D binarized convolutions.  The proposed 1-D downsampling can perform information compression gradually through two 1-D convolutions,
which can contribute tremendously to the performance improvement in binarized convolutional neural networks (BCNNs).  Compared with 2-D binarized convolutions, a $n \times n$ 2-D binarized convolution is replaced by $n \times 1$ row-wise and $1 \times n$ column-wise 1-D binarized convolutions, thus doubling the effects of adjusting the activation distribution and non-linear activation function.  In the decomposed 1-D binarized convolution, although computational costs are reduced, the number of element-wise activation functions and learnable bias layers can be doubled, which can be a significant burden.  Therefore, we expect that the 1-D binarized convolution is not suitable for all layers, and we present the reason and experimental results proving it.  Based on the above assumption and experimental results, we can provide more optimized structure in terms of performance and costs.  With ResNet as a backbone, we evaluate the proposed model on several conventional image datasets.  In experiments, the proposed model based on ResNet18 achieves 93.4\% and 93.6\% Top-1 accuracy on the FashionMNIST and CIFAR10 datasets.  In the case of training from scratch, the proposed OneBNet based on ResNet18 can produce 63.9\% Top-1 accuracy, showing better performance over the state-of-the-art (SOTA) binarized CNNs based on ResNet18.  When applying the teacher-student training, 68.4\% Top-1 accuracy can be obtained, which overwhelms the existing SOTA BCNNs.  With 5\% additional delay on a single thread of Raspberry Pi, the proposed lightweight model achieves 67.3\% Top-1 accuracy on the ImageNet dataset, outperforming the baseline by 1.8\%.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studied binarized convolutional neural network and proposed to replace nxn 2D BNN by two 1D BNN (nx1 row-wise BNN and 1xn column-wise BNN). Further, the paper designed two basic BNN blocks, 1-D binarized convolutional layer in Figure 1 and downsampling 1-D binarized convolutional layer in Figure 2. Adjustment of activation distribution are analyzed in Section 3.3. By combining above two basic blocks and adjustment of activation distribution, the paper built the architecture based on ResNet18 for image classification tasks.The experimental results validate the effectiveness of the proposed binarized convolutional neural network.

### Strengths
1. The paper provided detailed  analysis of proposed binarized convolutional network network from both theoretical and experimental perspectives.
2. The designed architectures achieved SOTA performances on CIFAR10 and imagenet. 
3. The author's writing is very good, and the entire paper is relatively easy to understand.
4. simple algorithm, easy to follow.

### Weaknesses
1. from engineering viewpoints, the proposed basic blocks seems obviously due to 1）the work of binarizing nx1 and 1xn convolutional neural network existed, and 2) adjustment of activation distribution also existed. The combination of them is not novel enough for top conferences. Thus, the contribution of this method is not very important.
2. The reference format in this paper can be improved, e.g. 
 - "Reactnet: Towards precise binary neural network with generalized activation functions" is an ECCV paper, 
 - "Recu: Reviving the dead weights in binary neural networks" is an ICCV paper.
3. In provided tables, some columns are left aligned, while others are center aligned. It is best to use a consistent format

### Questions
see weaknesses above

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
This paper proposes a new binary neural network called OneBNet, which mainly replaces NxN 2D binarized convolution by Nx1 row-wise and 1xN column-wise 1D binarized convolutions. The proposed model shows strong performance on ImageNet, i.e., ResNet18-based model obtains 63.9% Top-1 accuracy by training from scratch and 68.4% Top-1 accuracy by applying teacher-student training.

### Strengths
+ The proposed method is easy to follow.
+ Latency on Raspberry Pi is reported, which is beneficial for the BNN community.
+ The proposed model shows strong performance on ImageNet, i.e., ResNet18-based model obtains 63.9% Top-1 accuracy by training from scratch and 68.4% Top-1 accuracy by applying teacher-student training.

### Weaknesses
-	It is not new to replaces NxN 2D binarized convolution by Nx1 and 1xN 1D binarized convolutions. For example, SqueezeNext (CVPR’18 workshop) decompose the KxK convolutions into two separable convolutions of size 1xK and Kx1. Although this paper focuses on binary neural network, the novelty of using such strategy for BNNs is still quite unclear.
-	Could the proposed method also suitable for object detection tasks using binary neural network?
-	Several recent methods reported in Table 2 are not included in Table 3.

### Questions
See the weakness part.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to decompose a 2D convolution into two 1D convolutions on a Binarized neural network model to improve inference speed and model accuracy on edge devices. On the basis of the previous Binarized ResNet, the 3x3 convolutional kernel was changed to two sets of 1x3 and 3x1 convolutional kernels, achieving higher accuracy.

### Strengths
This paper is easy to read and understand. The method in this paper is relatively simple, clear, and easy to reproduce. The experimental data in the paper indicates that the improved model outperforms the previous Binarized neural network model in terms of speed and accuracy.

### Weaknesses
1. The contribution and innovation of this paper are insufficient. The decomposition of 2D convolutions into two 1D convolutions used in this article is not a new idea, but a widely studied method. Although its combination with Binarized neural networks may make it more effective, it is easy to consider or attempt.

2. The generalizability of this method has not been verified. The author's experiment only trained and tested the smaller ResNet model, and the dataset only included CIFAR10 and ImageNet.

3. The basic theory of the method in this paper is not sound enough. Binary quantization and 2D convolutional decomposition are methods that sacrifice accuracy for less computational complexity. Why can a combination of the two achieve better accuracy? Substantive improvements may come from element wise calculations and learnable bias in more layers after decomposition, but this is not without cost, as FLOPs cannot accurately reflect the additional hardware overhead this brings. The explanation of Figure 3 also lacks quantitative data support.

4. This method lacks a determination method for parameter selection. The paper mentions that not all convolutional layers of blocks are suitable for such transformations. In Table 1, several selection combinations are attempted, and the best ones are selected for subsequent comparison. This will bring difficulties to practical applications. If the model structure is different and there are more blocks with different channel numbers, it will not be suitable for such selection.

5. The comparison of experiments lacks fairness and universality. Compared to other methods in Table 3, they are all different binary quantization methods and will not significantly change the model structure. This paper essentially changes the structure of the model orthogonal to previous work. Unless compared with other structural optimizations, such changes are unfair.

### Questions
In experimental environments, it said "Like other BCNN models, the first convolutional and last fully-connected layers adopted FP32
weights and activations." Is the latency data measured End2End? If not, are non binarized layers becoming performance bottlenecks?

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The work proposes decomposing 2-D binarized convolution into two 1-D convolutions in different directions to reduce model complexity. Then, the paper investigates the settings where such replacement can be beneficial, followed by experimental verification of the architecture performance.

### Strengths
This work identified the proper settings where replacing 2D convolutions with 1D convolutions can be beneficial. The model performance shown in the paper is promising.

### Weaknesses
Overall the paper should improve on the presentation befoer it is ready for publication. At current state, I am unsure about many technical details. Please also see below.

### Questions
P3: section 3.1 The section is very difficult to follow, please rewrite this.

P4: "However, it shrinks the receptive fields", I am not sure whether this is true, please verify.

Fig. 2. It is not clear how the results from 1D convolution are summed together with ouput of the 1x1 FP32 convolution, given they are of different shape.  

P5: not sure about " It removes the negative part with β" 

P5: Not sure why there should be both  ζ and α

Figure 5: please change Fig. (a) to (a)

Table 1 is very confusing, please clear it up.

Distillation is quite a standard approach for improving model performance. Therefore, I suggest the authors move the results from the distillation experiment to the appendix. 

I suggest adding a model performance comparison with a similar OP level. 

ReActNet presented results of much higher performance (probably higher complexity); I suggest including an experiment that compares those results.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
