# Local Stepsizes Accelerate Distributed Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Distributed optimization is a core enabling technique for large-scale machine learning, multi-agent systems, and decentralized control, allowing both data and computation to be distributed across multiple agents. A key challenge in the design of distributed optimization algorithms lies in selecting appropriate step sizes. Most existing distributed algorithms rely on a coordinated global step size across the agents, which may be challenging to implement in a fully decentralized setting with many agents. Although some efforts have emerged trying to develop adaptive or uncoordinated step size strategies for distributed optimization, these approaches generally yield inferior performance compared with their coordinated counterparts, where a global universal step size is designed under the same step size strategy. In this work, we present a somewhat surprising finding that local step sizes for distributed optimization (with no coordination) can outperform their global step size counterparts. The results are obtained using a rigorous computer-assisted performance-characterizing technique (semidefinite programming) for optimization algorithms and are applicable to all convex and smooth objective functions. To the best of our knowledge, this is the first time that such results have been established for general objective functions in a rigorous and systematic manner. Experimental results using benchmark datasets confirm the theoretical discoveries.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed a novel algorithm, which combines EXTRA and NIDS. Then, This paper analyzed the performance of the proposed method with PEP and showed that the proposed method can enjoy the advantages of both EXTRA and NIDS, converging faster than EXTRA and NIDS.

### Strengths
* This paper proposed a novel algorithm, which combines NIDS and EXTRA and uses a different stepsize among nodes.
* The performance of the proposed method was analyzed using PEP, and this paper demonstrated that the proposed method can converge faster than NIDS and EXTRA.

### Weaknesses
* Using the different stepsizes among the nodes is interesting, but how can we tune the stepsize in practice, e.g., for deep learning tasks? This paper evaluated the proposed method with simple convex functions, in which the smoothness can be computed easily. However, when we use the proposed methods for more complicated tasks, it is very expensive to tune the stepsize so that each node can use a different stepsize.
*  The convergence rate of the proposed method is not analyzed in this paper.
* The constraint shown in Eq. (5), $\sum_{i} \nabla f_i (x^\star) = 0$, is too strong. The quantity of $\frac{1}{N} \sum_i \nabla f_i (x^\star)$ is often used to measure the data heterogeneity for the convex optimization, e.g., [1], but this PEP analysis simplifies the problem and assumes that $\frac{1}{N} \sum_i \nabla f_i (x^\star)=0$.
* This paper only compares the proposed methods with NIDS and EXTRA. Comparing them with other optimization methods, such as decentralized gradient descent and SONATA [2], would strengthen this paper.


## Reference
[1] Koloskova et. al.,  A unified theory of decentralized SGD with changing topology and local updates. In ICML 2020

[2]  Sun et. al., Distributed Optimization Based on Gradient-tracking Revisited: Enhancing Convergence Rate via Surrogation, SIAM 2020

### Questions
See the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a rigorous and systematic theoretical discovery that uncoordinated stepsize in distributed optimization could lead to faster convergence speed. This work builds on the powerful performance Estimation Problem framework and converts convergence analysis into an SDP formulation. The authors then show for EXTRA and NIDS type of algorithm, both worst case and average case performance can improve under the right stepsize choices.

### Strengths
I really enjoyed the theoretical framework to analyze performance of distributed optimization methods. I believe the research community's view is not that we prefer coordinated stepsize, we would rather have uncoodinated stepsize for implementation purposes, but it was very hard to guarantee convergence with those. This paper precisely handles that scenario and gives answers for when we could get convergence guarantees. The use of SDP and interpolation enables stability/convergence analysis of various methods in a systematic way. Numerical results compared both average case and worst case performances.

### Weaknesses
Previously stepsize $1/L_i$ was analyzed and failed to converge for some problems. I believe the fix here is actually Eq (7). The authors assumed that individual optimal solutions are close to the global solution and therefore even if the methods pull towards the local minima, the final solution won't be too far away from the true solution. This is a somewhat cheating assumption. Sure for any problem realization, such R exists, but the reason for divergence is precisely the fact that R is not equal to 0. So rather than saying the different stepsizes are helpful, it was really a small R that enabled good performance. When R is small, almost all distributed optimization method will work beautifully, as this separation of local min was the reason behind zig-zag in many previous papers.

### Questions
Can the authors propose the "best" stepsize rule of thumb? Beyond having a structured analysis of performance, what does this PEP framework give us? Can this be done in a distributed way?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper explores the impact of local step sizes on the convergence of decentralized optimization. The paper states that local step sizes can lead to faster convergence in distributed optimization compared to a global step size, although this paper does not provide any theoretical guarantees for this claim. Additionally, the paper introduces a new algorithm also without any theoretical guarantees, which may be faster in some settings compared to other algorithms.

This paper should be rejected for the following reasons: (1) the results are not well justified by either theory or practice, (2) the experiments are conducted with insufficient set parameters, and (3) The new algorithm (Algorithm 1), provided without theoretical guarantees of convergence.

### Strengths
This paper introduces the interesting research question: “Can local step sizes respecting local geometries of individual objective functions outperform their counterpart with a global universal step size in distributed optimization under general strongly convex objective functions?”

### Weaknesses
**Main argument**

To prove the rejection, I provide the following contradictions for each of the contributions:

The first stated contribution of this paper is "rigorously proving that local step sizes can indeed provide faster convergence in distributed optimization than their global step sizes". However, this paper does not include any rigorous theoretical guarantees for this claim. Can you provide proof for this statement?

 The second contribution claimed by the authors is "we have also developed a new distributed optimization algorithm that can exploit local step sizes to accelerate distributed optimization", but the theoretical guarantees for the algorithm are not provided in the paper. Can you provide proof or any kind of evidence that the proposed approach is faster than others?

 The third contribution stated in this paper is “ To rigorously compare the performance of distributed optimization using local step sizes versus global coordinated step sizes, we generalized the existing distributed PEP framework in two key aspects: firstly, we modified it to accommodate the boundedness restriction of optimal solutions in practical applications of distributed optimization; secondly, we revised it to reduce computational complexity, which is crucial given that the existing PEP approach is computationally intensive.”.  However, using only the PEP framework is not sufficient for a rigorous comparison between the use of local and global step sizes in distributed optimization. Usually, authors are inspired by the use of PEP frameworks to formulate rigorous theoretical results, for example, arXiv:1803.06600, arXiv:2101.09741.

The fourth contribution stated as “we conduct a comprehensive set of experiments on both synthetic and real-world datasets to validate the correctness of our theoretical findings and demonstrate the effectiveness of the proposed algorithm.”.  But these experiments are not sufficient or complete due to the fact that they only use two functions or groups of functions. Also, the smoothness constants L_i differ by no more than a factor of 10. The experiments do not provide convincing evidence of the correctness of the key statement.

### Questions
**Things to improve the paper that did not impact the score:**

1. Provide theoretical guarantees for each statements in this work 

2. Provide experiments for different settings of parameters

### Soundness
1

### Presentation
3

### Contribution
1
