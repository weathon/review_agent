# Maximum Likelihood Estimation for Flow Matching by Direct Second-order Trace Objective

- Decision: Reject
- Scores: 6, 3, 3

## Abstract
Flow matching, one of the attractive deep generative models, has recently been used in wide modality.
Despite the remarkable success, the flow matching objective of the vector field is insufficient for maximum likelihood estimation. 
Previous works show that adding the vector field's high-order gradient objectives further improves likelihood.
However, their method only minimizes the upper bound of the high-order objectives, hence it is not guaranteed that the objectives themselves are indeed minimized, resulting in likelihood maximization becoming less effective.
In this paper, we propose a method to directly minimize the high-order objective.
Since our method guarantees that the objective is indeed minimized, our method is expected to improve likelihood compared to previous works.
We verify that our proposed method achieves better likelihood in practice through experiments on 2D synthetic datasets and high-dimensional image datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose directly minimizing a high-order objective, specifically the second-order objective, to overcome the limitations of previous methods that focused only on minimizing upper bounds. This new approach aims to optimize likelihood more effectively by reducing the KL divergence between the data distribution and the generated distribution. Experimental results show that the proposed method performs better than existing flow matching techniques on both 2D synthetic datasets and high-dimensional image datasets. So,
to summary, this paper has 
1.Introduced a direct minimization technique for high-order objectives in flow matching, which enhances maximum likelihood estimation.
2.Utilized the gradient of the conditional vector field to calculate the second-order objective without needing simulations, improving computational efficiency.
3.Provided empirical evidence that the proposed method leads to better likelihood compared to previous approaches on several datasets.

### Strengths
1.The proposed method minimizes the KL divergence by directly addressing the second-order objective, offering a more reliable optimization of likelihood compared to earlier methods.
2.The approach demonstrates robustness across various types of datasets, including 2D synthetic and high-dimensional image datasets, indicating its scalability and versatility.

### Weaknesses
1.The method requires explicit computation of the second-order objective, which can be computationally intensive for very high-dimensional datasets, potentially limiting its applicability to extremely large-scale cases.
2.More details about experimental settings, such as learning rate and number of training epochs, need to be provided.
3.The tables in the paper have a lot of empty space below them, which affects the overall formatting. The layout of the paper should be reorganized for better presentation.

### Questions
refer to the question above

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a method for directly minimizing high-order objectives in Maximum Likelihood Estimation (MLE) for flow matching models. The proposed method directly minimizes the higher-order objectives, leading to improved likelihood estimation. The effectiveness of this approach is demonstrated through experiments on 2D synthetic datasets and high-dimensional image datasets, showing superior likelihood and data generation quality compared to previous works.

### Strengths
1.The paper introduces a novel approach that directly minimizes high-order objectives, rather than just their upper bounds, thereby theoretically reducing KL divergence more effectively.

2.The proposed method demonstrates better performance in terms of likelihood estimation (Negative Log-Likelihood) and 2-Wasserstein distance in experiments on both 2D synthetic and high-dimensional image datasets.

3.The use of Hutchinson's trace estimation method reduces the computational cost of calculating high-order objectives, making the approach more efficient.

### Weaknesses
1.While the paper claims that directly minimizing high-order objectives leads to better likelihood maximization, it does not provide rigorous formal guarantees or convergence proofs that this method will always outperform minimizing upper bounds in all scenarios. The theoretical analysis is somewhat limited in terms of offering strong mathematical assurances for the proposed approach's superiority.

2.The theoretical results rely on several assumptions, such as bounding the Fisher divergence by a function of high-order objectives. These assumptions may not always hold in practical applications, especially when dealing with more complex data distributions or models.

3.The experiments are incomplete: the methods compared in the paper have not been evaluated on all datasets. The method proposed in this paper did not achieve state-of-the-art (SOTA) results on CIFAR-10 and ImageNet, and on MNIST, it was only compared with one method.

### Questions
See weekness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper provides a method for directly optimizing the high-order objective for flow matching.

### Strengths
This paper provides a method for directly optimizing the high-order objective, extending the scope of previous work.

### Weaknesses
1. The theoretical results are not novel.
2. The improvement is marginal.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1
