# Deep-ICE: The first globally optimal algorithm for empirical risk minimization of two-layer maxout and ReLU networks

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
This paper introduces the first globally optimal algorithm for the
empirical risk minimization problem of two-layer maxout and ReLU networks,
i.e., minimizing the number of misclassifications. The algorithm has
a worst-case time complexity of $O\left(N^{DK+1}\right)$, where $K$
denotes the number of hidden neurons and $D$ represents the number
of features. It can be can be generalized to accommodate arbitrary
computable loss functions without affecting its computational complexity.
Our experiments demonstrate that the proposed algorithm provides provably
exact solutions for small-scale datasets. To handle larger datasets,
we introduce a heuristic method that reduces the data size to a manageable
scale, making it feasible for our algorithm. This extension enables
efficient processing of large-scale datasets and achieves significantly
improved performance in both training and prediction, compared to state-of-the-art approaches
(neural networks trained using gradient descent and support vector
machines), when applied to the same models (two-layer networks with
fixed hidden nodes and linear models).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a globally optimal algorithm for minimizing 0-1 loss in two-layer ReLU and Maxout networks and shows improved computational complexity. Also, for larger dataset, this paper introduces a heuristic method such that the dataset is feasible for the proposed algorithm.

### Strengths
1. This paper is well structured and clearly written.
2. The problem is well motivated, and the proposed approach is methodologically sound.
3. The discussion of related work is comprehensive and demonstrates a strong understanding of the existing literature.

### Weaknesses
1. The presentation of the Table 1 is not clear enough.
2. The experimental validation is not fully convincing. The experiments show that the algorithm proposed in this paper has better performance compared with baselines, but I am also curious of the running time of different approaches. Also, the proposed algorithm has improved computational complexity compared with literature. I am wondering if the improved computational complexity can be validated experimentally.
3. The evaluation could be strengthened by including additional large datasets with more features (large $N$ and $D$).

### Questions
Please see above.
The meaning of numbers in Table 1 is not clear.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Deep-ICE, a globally optimal algorithm for minimizing 0–1 loss in two-layer ReLU and maxout networks. The method exhaustively and efficiently searches over feature splits via a recursive nested-combination generator with CUDA acceleration, enabling exact training on small datasets and heuristic coreset-based extensions for larger problems. The authors provide formal correctness claims, complexity analysis, and empirical comparisons to SVMs and MLPs.

### Strengths
- The algorithmic description and theoretical details are well-structured and readable.
- Constructing an efficient search over all feature splits is nontrivial and technically interesting.
- CUDA implementation and memory optimization increase the practical relevance.
- Addresses an important research direction: global optimization of neural networks under 0–1 loss.

### Weaknesses
- The claim that two-layer networks are interpretable (unlike linear models) needs stronger justification, especially given nonlinear thresholds.
- Several relevant exact [1] or gradient-based [2, 3] 0–1 loss optimization methods are not cited nor discussed.
- For example, on the dataset from Figure 1, EXACT (with Tanh activation) achieves 18 errors with 2 hiddens and 16 errors with 5 hiddens, substantially outperforming MLP’s 25 errors.
- A Python interface to the CUDA implementation would be beneficial.

### Questions
- What exactly makes a 2-layer ReLU/maxout model “interpretable”? Can the authors provide interpretability examples or a measure?
- How does Deep-ICE compare to global optimization methods like [1] and EXACT [3]? Can runtime and performance comparisons be added?
- Do the authors have plans to release a Python library interface for CUDA to improve usability?

The score can be adjusted based on the responses (especially related work).

[1] Efficient global optimization of two-layer relu networks: Quadratic-time algorithms and adversarial training (2022)

[2] Algorithms for direct 0–1 loss optimization in binary classification (2013)

[3] EXACT: How to train your accuracy (2024)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents the first algorithm for finding the globally minimal empirical risk of two layer neural networks under 0–1 loss. The algorithm achieves polynomial complexity for fixed input feature size D and hidden feature size K, i.e. $O(2^{K-1}N^{DK+1})$ compared with previous $2^KC_1N^{DK+C_2}$. When combined with heuristics for large-scale problems, such as coreset selection, the proposed algorithm demonstrates strong out-of-sample performance.

### Strengths
1.	This paper introduces the first globally optimal algorithm for the empirical risk minimization problem of two-layer maxout and ReLU networks, i.e., minimizing the 0-1 loss.

2.	Experiments demonstrate better performance than those of SVMs and the same maxout network trained with gradient descent.

3.	The paper develops an efficient recursive nested combination generator for GPU execution.

### Weaknesses
1.	There are confusing statements in the paper. In line 105, the paper says ”our algorithm demonstrates strong out-of-sample performance, even when **training accuracy is lower than** that of SVMs or DNNs trained with gradient descent” and the in line 483 the paper claims that the proposed method “achieves significantly **higher training accuracy** than SVMs or two-layer neural networks, still perform well on unseen data when model complexity is properly controlled”. The two claims appear to be in conflict. Besides, there is no clear evidence or discussion in the paper to support either of them.

2.	Another concern is about computation efficiency as the method needs to enumerate data points. Is it possible to have computation time comparison?

### Questions
1.	In table 1, there are two numbers delimited by ‘/’. What do the two numbers denote? What is the difference?

2.	The paper argues that study of two layer of neural network will benefit model interpretability. However, the model output is a linear combination of hidden units, which is hard to interpret. If possible, please explain more about why two layer of neural network benefit interpretability.

3.	Typo in 69,70, $C_1, C_2$ should be switched. What is $K_-$ in line 383?

### Soundness
2

### Presentation
3

### Contribution
3
