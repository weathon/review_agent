## Human Reviewer 1

### Summary
This paper proposed an two-stage sparse structure learning framework for SNNs, which is combined with the PQ index technique and can be implemented through a continuous and iterative learning process.

### Strengths
1. The authors provided a detailed algorithm pseudo-code for their sparse training strategy.
2. It is interesting to combine the sparse structure learning framework of SNNs with PQ index and rewiring techniques.
3. The authors considered various rewiring scopes, including layer-wise and neuron-wise schemes.

### Weaknesses
1. The authors did not comprehensively report the performance of previous works in Table 1. Furthermore, it seems that their performance lags behind previous works. For example, the authors claim that UPR [1] achieves 78.3% (Acc.), 0.77% (Conn.) and 1.81M (Param.) on DVS-CIFAR10, VGGSNN. However, as mentioned in [1], the original text indicates that UPR can also achieve **81.0%** (Acc.), **4.46%** (Conn.) and **2.50M** (Param.) on DVS-CIFAR10, VGGSNN. In addition, STDS [2] can achieve **79.8%** (Acc.), **4.67%** (Conn.) and **0.24M** (Param.). 
In comparison, this work achieves **78.4%** (Acc.), **30%**(Conn.) and **2.76M** (Param.) under the same experimental condition, which is inferior to the previous results.
2. Compared to previous work [1], the authors lack experimental results for SOPs (Synaptic Operations) and power saving ratio.
3. Compared to previous works [1, 2], the authors lack persuasive experimental results on large-scale datasets (e.g. ImageNet-1k).

[1] Shi X, et al. Towards energy efficient spiking neural networks: An unstructured pruning framework. ICLR 2024.

[2] Chen Y, et al. State transition of dendritic spines improves learning of sparse spiking neural networks. ICML 2022.

### Questions
See Weakness Section.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a dynamic pruning framework for spiking neural networks (SNN). It divides each training iteration into two stages to update the weights and structure separately. It utilizes the PQ index to gauge and dynamically adjust structure of sparse subnetworks according to compressibility. Experimental results show that the proposed method achieves competitive performance while reducing redundancy.

### Strengths
1. The proposed method achieves competitive or better performance than the dense model with fewer connections and reduced redundancy.
2. It is a novel idea to dynamically calculate the rewiring ratio.

### Weaknesses
1. The details of the training process is unclear. In Figure 2, the density decreases as the iteration increases. However, as described in Algorithm 1, the remove fraction and the regrow fraction are both $c_i$, therefore the density should keep constant. Please explain why the density decreases and describe the training process in detail.
2. The proposed method requires multiple training iterations. Does it require longer training time as compared to existing methods? Please compare the training time with existing methods.

Minor:

1. Duplicate definition of $c_i$ in Algorithm 1: both labels of input data and rewiring ratio.
2. In Equation 1: undefined symbol $n^k$, you mean $n_k$? Symbol $\epsilon$ is also undefined.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 3

### Summary
This paper addresses over-parameterization in deep spiking neural networks (SNNs) during both training and inference, aiming to achieve an optimal pruning level. The authors propose a novel two-stage dynamic structure learning approach for deep SNNs.

### Strengths
Pruning is essential for making SNNs more practical for neuromorphic hardware, distinguishing it from artificial neural network (ANN) pruning, which is often GPU-oriented.

### Weaknesses
The paper could benefit from consistent citation formatting. For instance, "Hu et al. (2024)" works well at the beginning of a sentence, whereas "(Hu et al., 2024)" fits better at the end. Maintaining good formatting would enhance readability.

The rationale for using a PQ index in the second stage is unclear. Although PQ is a general technique, it would help to see a clearer connection to SNNs specifically and how it improves this work beyond its original intent. Not clear how PQ index is related to SNN.

The paper lacks an analysis of energy consumption, which is essential for evaluating SNNs’ efficiency, particularly in neuromorphic applications.

There’s no substantial improvement in accuracy or network connectivity, which may limit the paper’s impact.

### Questions
What is unique about the proposed work for SNNs specifically? Additionally, given the strong performance of attention-based and Transformer-based SNNs in the field, could this method also be applied effectively to those models?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
5

---

## Human Reviewer 4

### Summary
The authors present a two-stage dynamic pruning method tailored for deep Spiking Neural Networks (SNNs). The first stage evaluates the compressibility of existing sparse subnetworks within SNNs using the PQ index. In the second stage, synaptic connections are dynamically adjusted. The authors assert that this method can significantly reduce the network's sparsity. While the paper's motivation is commendable, there are shortcomings in both the presentation and the novelty of the work.

### Strengths
1. The authors apply adaptive pruning and sparse training methods to deep SNNs, which can further enhance the energy efficiency of these networks. This initiative is worthy of recognition.
2. The two-stage sparse training and learning method proposed by the authors appears to be feasible based on the experimental results presented.

### Weaknesses
1. To my knowledge, adaptive pruning and sparse training are well-established in ANNs. Existing works such as RigL [1]  seem to share similar ideas with the authors' approach. Additionally, DSR [2] also adjusts the rewiring rate. The authors have not provided targeted adaptations of these strategies to accommodate the binary spiking characteristics unique to SNNs.
2. The paper mentions the PQ index but does not provide an explanation, instead merely offering a citation. This could leave readers confused about what the PQ index represents and how it is calculated.
3. The methodology section lacks theoretical discussion. There is no explanation of why these methods are particularly beneficial or suitable for SNNs.
4. The size of Figure 3 is noticeably inconsistent with the other figures, which affects the overall presentation quality.

References:
[1] Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020). Rigging the Lottery: Making All Tickets Winners. International Conference on Machine Learning (ICML).
[2] Mostafa, H., & Wang, X. (2019). Parameter Efficient Training of Deep Convolutional Neural Networks by Dynamic Sparse Reparameterization. International Conference on Machine Learning (ICML)

### Questions
1. Can the authors explain why the proposed method is specifically suited to SNNs rather than Binary Neural Networks (BNNs)? Have special adaptations been made to leverage the unique characteristics of SNNs, rather than applying ANN techniques to SNNs?
2. I recommend that the authors include additional figures to further illustrate their method. This would greatly aid reader comprehension.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4