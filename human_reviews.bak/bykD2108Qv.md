# Efficient Gradient Clipping Methods in DP-SGD for Convolution Models

- Decision: Reject
- Scores: 5, 3, 3, 5

## Abstract
Differentially private stochastic gradient descent (DP-SGD) is a well-known method for training machine learning models with  a specified level of privacy. 
However, its basic implementation is generally bottlenecked by the computation of the gradient norm (gradient clipping) for each example in an input batch. 
While various techniques have been developed to mitigate this issue, 
there are only a handful of methods pertaining to convolution models, e.g., vision models.
In this work, we present three methods for performing gradient clipping that improve upon previous state-of-art methods. Two of these methods use in-place operations to reduce memory overhead, while the third one leverages a relationship between Fourier transforms and convolution layers. 
To demonstrate the numerical efficiency of our methods, we also present several benchmark experiments that compare against other algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents and studies memory and compute efficient methods for computing per-instance gradient norms in the context of training convolutional layers with DP-SGD. Three different methods are analyzed and studied. The first two methods are fairly simple variations to well-known methods (resp. Im2col followed by GEMM and ghost-clipping) by making them in-place; the third method exploits the connection between convolutions and Fourier transforms to transform the convolution operation to point-wise products in frequency space. The efficiency of the proposed methods are benchmarked against previous state of art methods in some-what artificial settings.

### Strengths
1. Applying FFT to accelerate per-instance gradient computation for convolution layers is new to my best knowledge. 

2. The mathematical formulation in the background section, in particular the use of operator notation to show that the gradient of convolution with respect to the kernel is the transposed convolution between downstream gradient and input, is neat.

### Weaknesses
1. My primary concern is whether the computation of per-instance gradient norms warrants treatment as a research question, as opposed to being a matter of implementation. I admit that it may not be fair to ask this particular paper of this question, as I am aware that there are multiple published works tackling this exact problem. 
These works point to most ML frameworks (e.g. pytorch, tensorflow) returning only the aggregated per-batch parameter gradient as a fundamental roadblock to computing per-instance gradient norms, but this is only true if we restrict ourselves to the python API implemented by these high-level frameworks. 
As such, I have yet to encounter a convincing argument as to why this per-instance gradient norm cannot be trivially obtained by computing it at the kernel level, right after per-instance gradients have been computed, before they are reduced into per-batch gradients. 
I acknowledge that there are nuances as convolutions and their backward computations are often divided and parallelized into per-patch operations, yet the general spirit remains true that the fine-grained (whether per-instance, per input channel, or even per sliding window) gradients are always computed first before they are aggregated via reduce operations, and that careful book-keeping of their norms can provide us the desired per-instance gradient norm with minimal compute and memory overhead. 

2. Measuring memory consumption in a memory-managed language like python is rather unreliable as the memory usage behavior can differ significantly depending on implementation and how memory usage is measured. More details, including source code, is required to make a convincing empirical argument as to the superior memory efficiency of the proposed method.

3. The overall impact of the methods presented in this paper seems limited: method 1 and method 2 are mostly re-hashes of the direct im2col+GEMM method and the ghost-clipping method with minute differences, and the FFT based method, while more novel, is the least performant out of the three in realistic settings where the kernel size is small and the number of channels is high.

### Questions
Please address point 1 in the weaknesses section.

What is $l_x$ in (4)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents three efficient gradient clipping methods for convolution models, which aims to mitigate the computational bottleneck of per-sample gradient norm estimation in differentially private stochastic gradient descent. Two of the proposed methods use in-place operations, while the third uses the connection between Fourier transforms and convolution layers. The experimental results show that the proposed methods outperform other existing methods both in running time and memory consumption.

### Strengths
1. This paper develops three improved efficient methods of the gradient clipping step specifically designed for convolutional layers, addressing the computational bottleneck in differentially private stochastic gradient descent (DP-SGD). 
2. The analysis for the third method is novel, which utilizes properties of circulant matrices to derive an algorithm that scales effectively with both the number of model parameters and batch size, demonstrating superior performance in high-dimensional settings.

### Weaknesses
1. The contributions of the proposed methods are not clear, especially the first two methods which seem only incorporating the in-place operations on the current exsting works (FastGradClip and Ghost Clipping), without significant conceptual advancements.
2. The claimed improvements in computation time are not fully substantiated. In Setting #1 of the section 4 (numerical results), the authors note that “our in-place ghost-norm algorithm was performing significantly worse than the rest of the methods and was increasing the computation time of the experiment significantly, so we have not displayed it”. This raises concerns about whether the proposed methods genuinely enhance computational efficiency.
3. In the Figure 4.1, Figure 4.2, and Figure 4.3 (running time analysis), the y-axis appears negative values, unclear why the running time metric would be negative. Also, the negative values appear in the memory consumption metric in Figure 4.2.
4. The experimental setup lacks clarity, particularly regarding the structure and the number of layers in the CNNmodel used for evaluation. This makes it difficult to assess the generalizability of the results.
5. The paper does not adhere to the ICLR formatting guidelines, which affects readability and may impact the overall presentation quality.

### Questions
1. Could you clarify how the first two methods differ fundamentally from existing techniques like FastGradClip and Ghost Clipping beyond using in-place operations?
2. In the numerical results section, why was the in-place ghost-norm algorithm excluded from some plots due to poor performance? Does this indicate a significant limitation of the method?
3. Could you explain why the y-axis in Figures 4.1, 4.2, and 4.3 shows negative values? Is this an error, or is there a specific reason for representing the data this way?
4. Could you provide more details on the CNN architecture used in your experiments (e.g., number of layers, types of layers)? How might the results change with different CNN architectures?
5. Can you provide more insights on the scalability of Method 3 when applied to real-world large-scale vision datasets, especially regarding runtime and memory overhead?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The paper proposes three new methods for efficiently gradient-clipping in differentially private stochastic gradient descent (DP-SGD) for convolutional neural network models. The key contributions are:

1. They propose two in-place algorithms (Algorithms 3.1 and 3.2) that compute the gradient norm using only O(1) memory storage, in contrast to the linear memory usage depending on model dimensions of prior methods. They also propose a Fourier-based algorithm (Algorithm 3.3) that leverages the structure of convolution layers to compute the gradient norm more efficiently, with a runtime that scales better with the input/output dimensions.
2. They provide theoretical analysis demonstrating the runtime and memory advantages of the proposed methods compared to prior work, depending on the specific model architecture.
3. The experiments shows the advantage of the proposed algorithms across different regimes of model size and channel counts.

### Strengths
1. The proposed algorithms provide significant improvements in memory usage compared to prior art, with theoretical and empirical support.
2. The analysis is rigorous, leveraging properties of circulant matrices and Fourier transforms in a novel way for the gradient clipping problem.
3. The empirical evaluation covers a range of relevant settings and provides clear comparisons to prior work.
4. The methods are general and can be applied to a variety of convolutional neural network architectures.

### Weaknesses
1. The evaluation is limited to just the gradient clipping step, and does not include full DP-SGD training experiments. I understand that the only modified part is gradient clipping, but demonstrating the end-to-end impact would be a more straightforward message. Will the proposed method lead to fast overall convergence, and how is the generalization performance?
2. While the paper provides three algorithms, it does not give any insight on how one should choose between the different proposed methods, given a specific model architecture and resource constraints.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors proposed several algorithms to reduce the space and time complexity of calculating the (squared) norm of gradients particularly for convolutional layers in DP-SGD. They provided theoretical analyses of each algorithm's complexity and verified their effectiveness across various settings.

### Strengths
- Rigorous theoretical analyses on the space and time complexity of each algorithm
- Improved performance across different settings

### Weaknesses
- The authors should clearly specify the problem they are addressing and what lies outside their scope, allowing readers to understand and appreciate the contribution of this work from a broader perspective. Specifically:

    - The authors should outline the general framework for per-sample gradient clipping. For example, in the OPACUS framework, there is a "for" loop that iterates over different layers: (i) the per-sample gradient accumulates across layers; and (ii) different types of layers (e.g., convolutional layers, fully connected layers) use distinct implementations to calculate the gradient norm. In this sense, the authors are tackling a low-level problem and do not address issues such as parallelizing different samples.

    - The authors should clarify the nuanced difference between “time complexity” and “actual runtime”. Practically, the computational complexity of per-sample gradient clipping is only a constant factor less efficient compared to standard training, rather than being linearly dependent on batch size.

- The proposed algorithm is valuable only if it outperforms current best practices. Unfortunately, the authors only use Bu et al. (2022) as the baseline and do not compare against Bu et al. (2023b), which they acknowledge as state-of-the-art, nor do they consider widely used frameworks like OPACUS and JAX_privacy. Notably, OPACUS provides a tailored solution for per-sample gradient clipping in convolutional modules. This comparison seems essential, and I strongly encourage the authors to discuss whether their implementation of the algorithms could be integrated into these existing frameworks.

### Questions
- The term "additional storage" is unclear. My understanding is that storing gradients requires $O(d)$ space, where $d$ represents the number of parameters in the layer. Could you clarify if "additional storage" refers to something beyond this baseline requirement, and that it appears in all prior work?

- I'm unfamiliar with "ghost clipping", and I assume many readers will be as well. It would be beneficial to provide a clear explanation of this concept in the paper.

### Soundness
3

### Presentation
2

### Contribution
2
