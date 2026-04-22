# Bridging Between Stable Rank and Data Selection: A Novel Sampling Method for Fast Training of Deep Neural Networks

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Data selection for efficient training aims to reduce the computational cost by selecting a subset of data to approximate the objective function. A number of elegant approaches have been proposed in past years, such as the popular importance sampling and coreset methods. However, their required sample sizes usually have a linear dependence on the dimension of parameter space (or other types of dimensions like pseudo dimension), which could be very large and thus hinder their applications to deep neural networks. In this paper, we aim to provide a deeper understanding of the connection between data selection and the complexity of training space in theory. Inspired by the effectiveness of prevalent low-rank fine-tuning techniques, we propose to study the sample size from the perspective of Gradient Trajectory (GT). Specifically, we measure the dimension of training space by the "stable rank" of gradient trajectory matrix (GT matrix), and propose a novel data selection method called "Stable Rank related Stratified Sampling method (SRS-Sampling)'' to accelerate the training process. Moreover, we establish the theoretical framework between the evolving stable rank of GT matrix and the required sample size. Finally, we conduct a set of experiments across pre-training and fine-tuning to validate the effectiveness of SRS-Sampling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes stable rank related stratified sampling (SRS-Sampling), a loss-stratified data selection method. This approach ensures that the required subset size is proportional to the stability rank of the gradient trajectory (GT) matrix, rather than the parameter dimension. SRS-Sampling was proposed based on a clear theoretical foundation, and methods for efficient computation were also presented.

### Strengths
1. This manuscript is well written with clear theoretical evidences. Also, by clearly outlining the limitations of prior works and the challenges encountered in narrowing the scope to detour these limitations, it is easy to follow the motivation of this research.

2. The proposed idea that utilize the tracking dimensionality of gradient trajectory subspace for data selection, which is unaffected by the number of parameters, is novel and interesting.

### Weaknesses
1. The performance of the SRS-sampling does not differ significantly from the baselines. Most performance gaps are less than 0.5%p, and it is particularly similar to the baseline on the ImageNet-1K and Alpaca datasets. To claim statistical significance for SRS-Sampling, additional p-value-based analysis is required.

2. The explanation of loss-based exponential partitioning is insufficient. It is not readily apparent how this partitioning aids in approximating the objective function of the entire dataset using a weighted subset, leading it to be perceived as a heuristic approach. Furthermore, with large-scale datasets, the number of partitions increases simultaneously, and certain partitions end up with very large intervals. This leads to the presumption that numerous partitions will be nearly empty, resulting in high redundancy.

### Questions
1. I would like to know whether SRS-Sampling remains effective even in situations where the subset ratio is extremely small such as 1% and 5%.

2. I believe a sensitivity analysis is required based on the value of the base used in the exponential partition.

3. To obtain an upper bound for the stable rank, the Jacobian, hidden-layer representations, and gradients are required. Storing and computing these components is expected to necessitate greater memory usage than the baseline. Therefore, a comparative analysis of computational cost is required.

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
The main idea of the paper is to develop a novel data selection method for efficient training of deep neural networks by leveraging the evolving dimensionality of the Gradient Trajectory (GT) subspace. The authors propose measuring the dimension of the training space using the stable rank of the gradient trajectory matrix, which represents a continuous and soft version of rank, reflecting the principal components of the gradient trajectory.

The paper proposes Stable Rank related Stratified Sampling (SRS-Sampling) method that selects a weighted subset of data for training, with the required sample size scaling linearly with this stable rank.

The empirical results on academic benchmarks such as MNIST, CIFAR, etc., show improvements. Experiments on Alpaca dataset with LLama-7B are encouraging.

### Strengths
1. The proposed method uses a smart stratified sampling approach based on how hard the network currently finds each data point (measured via loss), which ensures training speedups without losing accuracy, making it practical for large neural networks.

2. The empirical results show value of the proposed methods.

### Weaknesses
1. The connection between the gradient trajectory (GT) theory and the stratified sampling method is somewhat indirect. The paper leverages the stable rank of the GT matrix to characterize the dimension of the crucial training subspace, which guides the overall sample size needed. However, the stratified sampling based on loss partitions is not directly derived from the GT analysis but rather integrated as a practical sampling strategy inspired by importance sampling. This gap can make it hard to fully grasp how the GT stable rank determines the specific loss-based sampling probabilities and partitioning.

2. While the paper provides theoretical guarantees about sample size scaling with the stable rank and good empirical results, the computational overhead and complexity of estimating stable rank and eigen-pairs for very large models might limit practical real-time usage.

3. The adaptive and exponential partitioning of loss regions is a heuristic design choice without a fully explicit theoretical link to the GT dimension, so the choice of the number/count of regions and the partition base might affect results and requires tuning.

### Questions
1. The evaluation of LLM fine-tuning is conducted on older models and datasets; how well would the method generalize to more recent, larger models and evolving datasets?

2. How sensitive is the method’s efficiency and accuracy to the choice of the partitioning base and the number of loss regions in stratified sampling?

3. What is the computational overhead of estimating the stable rank in very large-scale settings, and how does it affect overall training speedups?

4. Would the approach degrade in scenarios with very noisy or imbalanced datasets where loss values may have less meaningful gradients?

5. Does the adaptive sample size based on stable rank sufficiently capture dynamic changes in training complexity during different training phases, e.g., early vs. late?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel and theoretically-grounded data selection method, named Stable Rank related Stratified Sampling (SRS-Sampling), for accelerating the training of deep neural networks. The key idea is that the effective dimensionality of model training is governed not by the parameter dimension but by the stable rank of the gradient trajectory matrix, which captures the intrinsic subspace of model updates. Building on this insight, this paper proposes a stratified sampling algorithm that allocates sampling ratios across loss strata in proportion to the estimated subspace complexity. Theoretical analysis shows that the required subset size scales linearly with the stable rank rather than with the parameter dimension. Experiments on CIFAR-10/100, ImageNet-1K, and LLaMA-7B fine-tuning demonstrate consistent acceleration and accuracy retention.

### Strengths
- **Strong theoretical insight**: The paper establishes a novel theoretical connection between the stable rank of the gradient trajectory and data sample complexity, extending beyond traditional parameter-dependent coreset theory.

- **Solid empirical validation**: 
(1) Direct Theory Validation: The paper includes crucial experiments that directly support its theoretical claims. Figure 5, which visualizes the stable rank ratio during training, confirms the core premise that training dynamics are low-dimensional.
(2) Rich experiments demonstrated that SRS-Sampling show clear advantages over a wide array of strong baselines across multiple datasets and tasks.

### Weaknesses
(Major)

**Limited ablation on hyper-parameters**. The performance of SRS-Sampling may depend on the number of loss strata and the details of stable-rank estimation. A more extensive ablation would clarify the method’s sensitivity and robustness.

**Computational complexity of stable-rank estimation**.
Although the paper claims negligible runtime overhead (<0.1%), it does not formally analyze the computational complexity of stable rank estimation. A complexity expression or scalability discussion with respect to model dimension and training length would make the analysis more rigorous.

(Minor)

**Typographical and grammatical issues**. Several minor issues should be corrected for clarity and professionalism. For example: "gird" should be "grid" in the Figure 3 caption, and subject-verb agreement should be corrected (e.g., "...gradient structure has..." instead of "have" on Page 3, Line 158).

### Questions
- Could the authors provide an explicit complexity expression for stable rank estimation (e.g., O(k·d) with respect to iteration count k)?

- How sensitive is the method to the number of strata and the sampling ratio allocation strategy? Would adaptive stratum partitioning improve performance?

- Is the stable rank computed once before training, or does it evolve during training? If the latter, how often should it be updated in practice?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The authors propose to study the stable rank of the gradient trajectory matrix, which is a continuous and soft version of rank, newly defined in their paper as the rank of principal components in the gradient trajectory matrix.
- The authors suggest Stable Rank related Stratified Sampling for training data selection, for data-efficient training of deep learning models, where the suggested SRS sampling for training data selection depends on the initial parameter values.
- The authors conducted diverse experiments, including image classification with multiple datasets and LLM fine-tuning.

### Strengths
- The proposed SRS-sampling algorithm is straightforward and easy to understand.
- I haven't read all the proofs, but the provided proof sketches seem reasonable at least to me).

### Weaknesses
- Even the proposed SRS-sampling algorithm (Algorithm 1) is simple, but it is difficult to understand the completeness of the algorithm because it is hard to connect with Algorithm 2, which is linked within Algorithm 1.
- There should be error bars in the experimental results, since some performance gaps are tight.

### Questions
- Please explain Figure 1 in more detail. What are $\theta_T$ and $\theta_{2T}$? What are the green space and the orange space, respectively?
- Line 256: Why is the gradient norm in training a neural network typically bounded? Please specify the assumptions if they are required.
- Line 313: $G = A - BWC$ seems to have some meaning for each $A, B, C$. Could you explain this in more detail? (I know there is an extra explanation regarding this in the appendix as a previous work, but please provide me with an insight about this formulation.)
- In Theorem 2, $\lambda_1$ and $\lambda_2$ are the two smallest eigenvalues of $S$, right? Do they have to be non-negative or something? (And why?)
- typo in Line 352: upper-bound (6)
- Regarding Algorithm 2, so it seems that the upper bound of $sr(D_T)$ can be recursively integrated with each layer's upper bound of $sr(D_T^(l))$, right? I might have lost somewhere between, but why the "additionally" integrating the upper bound of $sr(D_T^(l))$?
- So the selected samples with the SRS perfectly depend on the initial neural network parameter values, where all other settings, including the network structure, loss function, etc., are given, right? Then, what proportion of the training dataset intersects (on average) for the different initialization?
- Could you compare the computational complexity for the baselines as well?

### Soundness
2

### Presentation
2

### Contribution
2
