# Prospective Learning: Memory-Efficient MLP Training via Brain-Inspired Direct Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
Multi-layer perceptron (MLP) training via backpropagation faces fundamental memory limitations that constrain deployment in resource-constrained environments such as edge devices.
We introduce Prospective Learning, a novel training paradigm inspired by biological prospective configuration mechanisms that replaces gradient-based optimization with direct algebraic weight computation. 
By transforming weight updates into regularized least-squares optimization problems that can be solved analytically layer by layer, it eliminates the need for gradient storage and intermediate activation caching, significantly reducing resource consumption.
Meanwhile, it integrates brain-inspired sparse connectivity initialization and adaptive metaplasticity mechanisms, which support the framework from the aspects of infrastructure initialization and dynamic learning adjustment, respectively.
Experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets show that Prospective Learning achieves competitive accuracy, reduces memory usage by up to 55\% compared with traditional backpropagation, and consistently outperforms existing backpropagation alternatives in memory efficiency. 
This memory-computation trade-off is favorable for edge scenarios where memory constraints dominate.
For example, it achieves 95.44\% accuracy on MNIST using only 38.77MB of memory on edge devices, providing a viable solution for efficient MLP training on memory-constrained edge devices.
Our main code has been anonymously uploaded to \url{https://anonymous.4open.science/r/Prospective-Learning} without any author information.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Prospective Learning (PL), a novel brain-inspired training paradigm for multi-layer perceptrons (MLPs) that replaces gradient-based backpropagation with direct algebraic weight computation. It introduces three key components, prospective configuration, sparse connectivity initialization, and adaptive metaplasticity, to achieve memory-efficient learning without storing gradients or activations. Experiments on MNIST, CIFAR-10, and CIFAR-100 show comparable accuracy to backpropagation while reducing memory consumption by up to 55%, demonstrating strong potential for resource-constrained environments such as edge devices and neuromorphic hardware.

### Strengths
1. The paper introduces a biologically inspired direct optimization algorithm that avoids gradient computation, which is novel and conceptually grounded in neuroscience.
2. It provides solid theoretical analysis of memory and computational complexity, including formal proofs and convergence guarantees.
3. The experiments are comprehensive and reproducible, demonstrating consistent memory efficiency across multiple datasets and hardware conditions.

### Weaknesses
Although I think this paper is solid, there are several concerns:
1. The proposed method is limited to MLP architectures, and its scalability to convolutional-based models is not demonstrated.
2. Although the method reduces memory, the computational cost of algebraic solving (O(d³)) may become a bottleneck for larger models.
3. The biological justification (prospective configuration and metaplasticity) is conceptually appealing but empirically shallow, lacking ablation beyond mathematical analogies. If possible, please add several discussions.

### Questions
1. How does the method perform on more complex networks (e.g., CNNs) where algebraic solving may become infeasible?
2. Could the authors clarify whether regularized least-squares optimization introduces implicit gradient-like dynamics, and how this differs fundamentally from backpropagation in terms of learning signals?
3. What are the energy efficiency and latency trade-offs of Prospective Learning when deployed on neuromorphic or edge hardware compared to other biologically plausible methods, such as STDP or Forward-Forward?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel backpropagation-free training method named Prospective Learning. This method is decoupled into three main stages, a least-squares optimization based phase to compute the optimal weight updates without gradient based computation, a sparse connectivity initialization which reduces the network's parameters and an adaptive metaplasticity phase where the model's sparse parameters are learned over training.

### Strengths
- The paper proposes a novel, backpropagation-free method for training multilayer perceptron (MLP) and convolutional neural network (CNN) models on CIFAR-10 and CIFAR-100.
- Decomposing the learning algorithm into three stages reflects an in-depth analysis of standard learning algorithms for neural networks, such as backpropagation-based methods.
- The proposed learning algorithm can train MLP and CNN models in resource-constrained environments where memory or hardware constraints could make the support for standard backpropagation-based training infeasible.

### Weaknesses
- The experiments are insufficient to demonstrate the advantages of the proposed prospective-learning algorithm over other baselines. For MLPs, the models are trained from scratch using the prospective-learning algorithm. For CNNs, the ResNet-18 model is first pre-trained on ImageNet, and only the linear classifier head is fine-tuned on CIFAR-100.
- It is not clear from the text what advantages the sparse-connectivity initialization offers compared with other baseline approaches.
- The plots could be improved by increasing the font sizes of the axis labels, legends, and tick marks.

### Questions
- Does the prospective learning algorithm support training from scratch for convolutional neural network models?
- What is the advantage of the sparse connectivity initialization over the dense ones?
- Can the prospective learning algorithm be used for different data domains other than vision?
- Can the prospective learning algorithm be used for models that are not convolution based?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors present a framework for learning in neural networks inspired by a recent observation in neuroscience.
They develop three components to reduce the memory footprint by about a half.
They illustrate their method compared to other optimizers on MNIST and CIFAR datasets.

### Strengths
The research area of developing lower profile training algorithms for neural nets is an important one.

The motivation and method are clearly presented.

The numerical results are well presented, including uncertainty.

It's great that the authors provide code; I did not check it carefully.

### Weaknesses
This approach requires solving a linear system for each layer of the network at every training iteration.
The complexity analysis appendix does mention the fact that this is a cubic operation, and so considerably more expensive than what backprop requires.
But this fact goes insufficiently discussed in the body of the article.
This is a serious limitation.
Edge devices are indeed memory limited, but they are also compute-limited, and providing an algo that requires linear system solves *at every iteration* does not seem like it would be of practical impact except in niche scenarios.
Second order methods, such as Newton's method, are broadly deemed beyond the pale in neural network learning essentially for this reason.
All this notwithstanding, there's nothing in principle wrong with developing an algorithm which trades computation for memory.
But this article is not presented in this manner: this should have been thoroughly discussed in the motivation, abstract and conclusion.

The conclusions that this "enables practical MLP deployment on [...] neuromorphic hardware" is not supported by the body of the article.
Neuromorphic platforms are not simply regular computers with a small memory.
Rather, they face entirely different constraints based on the *locality* of information relative to processing units which is not addressed by this article.

The proof-theorem format of Appendix A.4 is not necessary for such straightforward observations.

It would be helpful to better situate the Section 3.4's contributions, the metaplasticity, within the stable of existing adaptive first order methods such as Adam et al. Why is this approach better suited to your framework than what's currently out there?

The datasets are of course somewhat limited in scale, being MNIST, CIFAR-10 and CIFAR-100.
This is critical to the method because any more complex dataset requiring a bigger hidden layer would explode the computational requirements.

### Questions
1) What is fast orthogonalization? It does not appear to be defined or given in a reference.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The study presents the brain-inspired learning mechanism alternative to the joint classsical backpropagation algorithm (BP) + gradient-based learning. The proposed approach consists of three main ingredients: prospective inference, algebraic optimization and metaplasticity modulation. The main purpose of the presented approach is to decrease the memory footprint and make possible to run training of the MLP models on the resource-contrained hardware. The experiments on the MNIST, CIFAR-10 and CIFAR-100 datasets confirm that the proposed Prospective learning framework gives the competitive performance for the less memory than BP and non-BP based algorithms.

### Strengths
The submitted work is clearly written and has an easy-to-follow format. The presented tables and plots provide the necessary data for evaluating the proposed approach. From these experimental results, it follows that the proposed prospective learning framework is memory-efficient compared to BP and non-BP alternatives.

### Weaknesses
Although the presented work has a clear focus and states the target problem, it has many weaknesses and inconsistencies, which I have listed below.
1. The motivation of the proposed approach from the biological mechanisms suffers from the absence of formal convergence proofs or any analytical intuition why the proposed pipeline corresponds to the minimization of classification error on the test set.
2. The main ingredients have clear analogues in the BP-based methods (adaptive learning rates, tricky initializations, and specific update rules); however, no comparison with existing alternatives is presented. I see many "yet another" instances of the classical ingredients for the learning pipeline, without any theoretical proof of why they are better than existing ones. Even the reported results of the ablation study are insufficient since the predefined hyperparameters are used, and no recipe for searching them in other tasks is discussed.    
3. The proposed approach has a lot of hyperparameters, which make it less portable to other datasets and models (MLP with different numbers of layers), e.g. $\varepsilon$ in (2), $\lambda$ in (6), $\tau$ in (10), $K$ in (11), $p$ in (12), $\gamma$ in (17), etc 
4. Although the use of edge devices is the key feature of the presented study, the applications selected for experiments are relatively standard. I am not sure that learning the image classifier on the Raspberry Pi or similar chips is the most popular way to use such devices. Such inconsistency makes the reported results less impressive and raises many questions regarding the target use cases.
5. The proposed approach demonstrates a uniform accuracy drop for the considered datasets; therefore, a comparison with other strategies to reduce memory footprint is necessary. Probably using mixed precision, low-rank approximations, model distillation, or other techniques can provide a similar memory footprint while maintaining or improving test accuracy. More experiments are needed here.    
6. Moreover, I see that prospective learning is a much slower approach than BP-based learning, so I would like to see the discussion on why the memory efficiency is more critical than training duration for the considered setup of exploiting an edge device.

### Questions
See the weaknesses above. 

In addition, please comment on the following questions.
1. Why were the considered datasets selected for benchmarking? I can imagine a promising application of using a prospective learning approach on edge devices is online voice processing or image segmentation in an autonomous vehicle. Both settings also require fine-tuning to the new data.
2. What is "FastOrthogonalize" procedure?
3. What numerical algorithm was used to solve (6)? Using the direct formula (7) can lead to numerical instability due to the properties of the matrix inversion operation.

### Soundness
2

### Presentation
3

### Contribution
1
