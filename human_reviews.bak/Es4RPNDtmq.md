# Robust Weight Initialization for Tanh Neural Networks with Fixed Point Analysis

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 8

## Abstract
As a neural network's depth increases, it can improve generalization performance. However, training deep networks is challenging due to gradient and signal propagation issues. To address these challenges, extensive theoretical research and various methods have been introduced. Despite these advances, effective weight initialization methods for tanh neural networks remain insufficiently investigated. This paper presents a novel weight initialization method for neural networks with tanh activation function. Based on an analysis of the fixed points of the function $\tanh(ax)$, the proposed method aims to determine values of $a$ that mitigate activation saturation. A series of experiments on various classification datasets and physics-informed neural networks demonstrates that the proposed method outperforms Xavier initialization methods (with or without normalization) in terms of robustness across different network sizes, data efficiency, and convergence speed. Code is available at https://github.com/1HyunwooLee/Tanh-Init.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new method to initialize weights for FCNNs and PINNs with tanh activation. The paper claims that the proposed weight initialization method will not lead to diminishing activations for very deep networks unlike Xavier weight initialization. The paper claims that the proposed weight initialization is robust to network depth and number of units in hidden leayers.

### Strengths
Originality
The paper presents a novel weight initialization method specifically designed for tanh-based neural networks, addressing an understudied area in neural network initialization. This approach is distinct in its use of fixed-point analysis to prevent activation saturation and improve training robustness across network sizes, particularly in the context of Physics-Informed Neural Networks (PINNs). By emphasizing robustness and performance consistency in both traditional classification tasks and PINNs, the paper makes a valuable contribution to the field of weight initialization. The originality is strong, given the lack of prior research focusing on tanh-based initialization methods.

Quality
The paper demonstrates high quality in both theoretical and experimental aspects. The method is grounded in rigorous mathematical analysis, leveraging fixed-point theory to derive conditions that ensure stable activation propagation. The provided lemmas, proofs, and propositions add credibility and depth to the approach.
Experiments are well-designed and span various network configurations, datasets (MNIST, Fashion MNIST, CIFAR-10), and applications (PINNs for solving differential equations). The results consistently show that the proposed method outperforms Xavier initialization, particularly in deeper networks and varying network sizes.

Clarity:
Overall the paper is very well written, with some exceptions mentioned in the weakness section. All the sections in the paper are laid out clearly. The notations are consistent across the paper.

### Weaknesses
Significance:
The paper compares their proposed weight initialization with Xavier weight initialization for FCNN with tanh activation. Xavier is known to show diminishing gradients and activations problem for deeper networks, but this is solved by using layer normalization. I am therefore considering this work not significant because the problem that the authors are trying to solve does not exist for Xavier + Layer norm and the authors did not do any comparative analysis with and without layer norm.

Other major issue:
1. In equation 2, the paper claims that a_i^(k+1) follows normal distribution with unit mean. But when number of neurons in layer l-1 (N_(l-1)) is greater than number of neurons in layer l (N_l), then the mean will be greater than 1. When N_(l-1) = 2 * N_l, the mean will be 2. This is not clearly discussed in the paper. If the mean is > 1, then that leads to tanh always saturated.

### Questions
1. Layer norm is added to handle the diminishing activation problem. Any reason why you did not compare the performance of the proposed approach with Xavier weight initialization with Layer norm?

2. In equation 2, the paper claims that a_i^(k+1) follows normal distribution with unit mean. But when number of neurons in layer l-1 (N_(l-1)) is greater than number of neurons in layer l (N_l), then the mean can be greater than 1. When N_(l-1) = 2 * N_l, the mean will be 2. This is not clearly discussed in the paper. If the mean is > 1, then that can lead to tanh saturation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
Weight initialisation is an old topic, and most studies have verified that weight initialization methods can improve the performance of neural networks. However, because that neural network’s depth increasing rapidly,  most neural networks, especially Feedforward Neural Networks (FFNNs) should face the gradient vanishment problem.  In this article, authors proposed one weight initialisation method for FFNNs and Physics-Informed Neural Networks (PINNs). Based on an analysis of the fixed points of the function tanh(ax), this method determines values of $a$ that prevent the saturation of activations during the training progress. In terms of robustness, this method presents a stronger and more efficient performance. In the experiment, verified on MNIST, Fashion MNIST, and CIFAR10 datasets, this method also shows acceptable results compared to the Xavier.

### Strengths
1. this article proposes one novel weight initialization method for the FFNN and PINN.
2. the authors prove that the activation values cannot vanish when increasing the depth of the neural network by using a fixed-point analysis.

### Weaknesses
1. The universality of this method should be improved. For example, more experiments on the feasibility of other neural networks except FNN and PINN.
2. The details of hyperparameters should be mentioned, such as what is the $Threshold$ of FFNN, which training strategy (supervised or unsupervised) of FFNN is used?   
3. FFNN is specifically designed to visualize the trained features, which also should be discussed.
4. The consistency of results should be guaranteed. For example, in Figure 3 (c) and (d), there is a different trend (Xavier tends to equal 
the proposed method), which also should be discussed. In case that after 20 epochs, the performance would be totally different to the presented result.

### Questions
The novelty of this work is strong, and the topic sounds interesting. However, the writing and structure should be revised again. There are some questions that the authors should be concerned about. 

1. In Eq. 2, $\sigma_{z}$ is set to $\alpha/\sqrt{N^{l}-1}$ and $\alpha = 0.085$. From Figure 2, we can find that the optimal value of $\alpha$ is $0.085$. Is there any theoretical reason why $\alpha$ should be $0.085$. Or should we manually try the value accordingly?

2. Additionally, please scribe what is $\alpha$. There is no definition of $\alpha$.

3. In Figure 3 (c) and (d), the proposed method seems to decrease after 6 epochs. Although the accuracy curve can rapidly reach the peak (faster than Xavier), the robustness of this method also should be discussed. For example, the initialization method can first provide prior knowledge to neural networks, but if it can keep the stability of training or not. Or is it the reason for the high $\alpha$?

4. In Appendix A.1, the authors discussed different conditions. When $x = 0$, whether the vanishment problem will abscond. Please highlight the strategy on how this method can process it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work first provides a theoretical analysis of weight initialization when exclusively using tanh
as an activation function. Providing reasons for the clustering behavior when initializing networks.
Based on the developed theory, an initialization scheme is proposed, and the hyperparameter σz
is empirically determined. There are two types of experiments mainly comparing the proposed
initialization with Xavier. The first type of experiment concerns the classification accuracy in the
early training phase. Showing there is an improvement in accuracy across different data sets and
configurations. The second type of experiment concerns solving PDE with PINNS, showing that the
proposed method has a good performance.

### Strengths
Unlike optimization problems with theoretical guarantees on fixed points, weight initialization
is an important task in deep learning. An initialization scheme with theoretical backing can
have a long-lasting impact, even just for a sub-field of deep learning.

Experiments do show significant improvement when using PINNs to solve PDE.

### Weaknesses
The impact depended on exclusively using tanh as an activation function is fundamentally
beneficial in PINNs. As the current state of the paper, there is not enough support for this.

Given that the experiments are not too computationally intensive and the experiment section
only considers a few data sets or PDEs, the demonstrated improvement may not be general.

### Questions
In sections 4.1 and 4.2, the experiment trains for 20 or 40 epochs. Do networks converge to their best accuracy? What is the difference in accuracy when training for more epochs?

The experiments in section 4 are not too computationally intensive, is it possible to include more
data sets or PDE can show the improvements are general?

Can the code used in the experiment can be provided to improve reproducibility?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a novel weight initialization technique specifically designed for neural networks using the tanh activation function. This technique is evaluated against the well-known Xavier initialization method using benchmark datasets. The experimental results demonstrate that the proposed initialization method enhances the convergence speed of Physics-Informed Neural Networks (PINNs) utilizing the tanh function, showing greater robustness to variations in network size. The findings indicate that the new initialization technique outperforms Xavier initialization in solving various problems related to Partial Differential Equations (PDEs).

### Strengths
The strength of this paper lies in the development of a novel weight initialization technique that facilitates faster convergence and enhances performance in physics-informed neural networks (PINNs) utilizing the tanh activation function.

### Weaknesses
1. The comparison of the proposed weight initialization technique solely with Xavier is insufficient; it should also be experimentally evaluated against other state-of-the-art weight initialization methods.
2. Using tanh activation function in entire neural network is not good practice that it has the drawback of the vanishing gradients for the very high and very low values of x.
3. The formulation is more complex than standard methods, which could complicate implementation as shown in Equation (1). 
4. The optimal value of 𝛼 can be highly context-dependent, varying across different architectures, datasets, and tasks, which makes it less universally applicable. Additionally, the choice of 𝛼 can interact with other hyperparameters, such as learning rate and batch size, complicating the overall tuning process during backpropagation, as described in Equation (2).
5. In Section 4.1, the evaluation process utilizes three datasets—MNIST, FMNIST, and CIFAR-10—employing the tanh activation function in every layer. As shown in Table 2, as the number of hidden layers increases, loss gradually increases, which is indicative of overfitting. It would be more effective to use the proposed weight initialization in conjunction with state-of-the-art architectures for training deep neural networks.

### Questions
See above in weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
