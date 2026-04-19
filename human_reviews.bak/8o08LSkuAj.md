# Learning with Exact Invariances in Polynomial Time

- Decision: Reject
- Scores: 6, 6, 8, 6

## Abstract
We study the statistical-computational trade-offs for learning with exact invariances (or symmetries) using kernel regression over manifold input spaces. Traditional methods, such as data augmentation, group averaging, canonicalization, and frame-averaging, either fail to provide a polynomial-time solution or are not applicable in the kernel setting. However, with oracle access to the geometric properties of the input space, we propose a polynomial-time algorithm that learns a classifier with \emph{exact} invariances. Moreover, our approach achieves the same excess population risk (or generalization error) as the original kernel regression problem. To the best of our knowledge, this is the first polynomial-time algorithm to achieve exact (not approximate) invariances in this context. Our proof leverages tools from differential geometry, spectral theory, and optimization. A key result in our development is a new reformulation of the problem of learning under invariances, as optimizing an infinite number of linearly constrained convex quadratic programs, which may be of independent interest.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studied the statistical-computational trade-offs for learning with exact invariances using kernel regression over manifolds,
proposed a polynomial-time algorithtm for the first time and proved the generalization error bound of the algorithm
by utilizing some tools from differential geometry, spectral theory, and optimization.

### Strengths
Learning with invariance (or other properties of learning problem) is important for both theoretical studies and algorithm design in machine learning. This paper proposes the first computationally efficient algorothm for kernel leanring over manifolds,
and introduces some new theoretical tools from differential geometry and spectral theory,
which I think might be of independent interest.

### Weaknesses
The main weakness of this paper is the generalization bound which is same with learning without invariance,
failing to demonstrate the benefit introduced by invariance.
Although it has been mentioned as a future work, it would be more interesting if the current algorithm already satisfies this property.
In contrast to the bound in [1], 
the upper bound in this paper not only maintains a larger problem-dependent constant but also a larger dependence on the norm of $f^\ast$.

References

[1] Behrooz Tahmasebi and Stefanie Jegelka. The Exact Sample Complexity Gain from Invariances for Kernel Regression. NeurIPS, 2023.

### Questions
(1) In line 150, whether the expectation operator $\mathbb{E}_{S}$ should be corrected as $\mathbb{E}$. 
I think the expectation is w.r.t. the distribution over manifolds not the trainning data.
	
(2) The current analysis of the generalization error just used standard techniques, but did not make full use of the invariance.
	Is the reason that the current analysis is not tight for the proposed algorithm, or that the proposed algorithm did not fully leverage the invariance?

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
The authors consider the setting of invariant regression when the regression function is smooth $s$-Sobolev space. It is well known that for $s>d/2$, the $s$-Sobolev space is an RKHS and for regression without invariance constraint, the ridge estimator is minimax optimal.  As with any kernel method, a forward pass through such an estimator requires at worst $n^3$ time. However, the ridge estimator may not be invariant under the group of interest. In this paper, the authors provide a new estimator that achieves the same error rate as the usual ridge estimator, is invariant under the group of interest, and can be implemented in polynomial time. Note that this estimator may not be minimax optimal as the true regression function with the required constraint lies in a smaller class.

### Strengths
1. The paper is well-written and is easy to follow. 
2. The estimator is pretty natural. While the proof of Theorem 1 is relatively straightforward, I find the idea of re-representing the invariance constraints in the coefficients along basis expansion clever.

### Weaknesses
1. I did not find any apparent weakness. The paper delivers on its promise of developing an invariant estimator for Sobolev regression. 

2. One might argue that the lack of experiments takes away from the paper, but I am fine with a theory paper with sound and rigorous proof not including experiments. The paper’s conceptual contribution is, in my view, sufficient to address this limitation.

### Questions
In Section 3, the authors mention that group averaging can be too costly for large groups. But what about group averaging only over the generator? Does the kernel ridge regression with the group averaging over the generator set $S$ yield an invariant estimator? If so, what do we know about its generalization error?

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
4

### Summary
The authors show that it is possible to learn with invariance on compact smooth manifolds with a finite group action in a manner that simultaneously enjoys polynomial sample complexity and computational complexity in the problem parameters.

### Strengths
The paper is well-written and flows well. The exposition is very clear. The authors also highlighted the novelty and the gap that is filled. The proof sketch is nice and I find that conveys the essential technical aspects.

### Weaknesses
I think the first bullet point on Line 085 could be made more clear: the main results of this paper shows one point on the statistical-computational trade-off curve, as I understand. I don't see how to interpret the results as showing how to trade-off statistical efficiency to gain computational-efficiency. I might be misunderstanding this bullet point however.

### Questions
Is the case of (positive dimensional) lie group actions addressable using this approach?

Theorem 1 should make it clear that $\hat{f}$ is $G$-invariant.

Line 234: how about the second oracle (the one about computing inner product between shifted eigenfunctions). Is that also efficiently computable?

Are there other examples besides the sphere for which the oracle calls can be computed efficiently? E.g., if the authors can talk about an example with Stiefel manifolds, that would be nice to know about.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a computationally efficient algorithm for  learning with exact invariances over RKHS. Specifically, on one hand, the authors demonstrated that the  proposed algorithm only has polynomial-time complexity, in sharp contrast to the previous  algorithm with invariances. On the other hand, an advantage compared to KRR is that the proposed algorithm can enforce group invariances without loss in statistical convergence rate.

### Strengths
1. The paper is well-written and easy to follow.

2. A novel algorithm is proposed with two main advantages: computational efficiency and the enforcement of group invariances.

3. The theoretical analysis in this paper is comprehensive and rigorous. The statistical rate of the proposed algorithm  is provided, along with detailed comparisons to the related algorithm.

### Weaknesses
**Writing:**

The dimension $d$ first appears before it is defined. I suggest the authors introduce the definition of $d$ earlier for clarity.

In Line 144,  the labeled samples are drawn from the product manifold $\mathcal{M}\times R$ rather than from $\mathcal{M}$. 


**Numerical experiments:**

While I find the theoretical aspect interesting and well-founded, it may be beneficial to conduct simple numerical experiments to support the improvement of the proposed algorithm in computational efficiency and enforcing group invariances,  while preserving the same learning rate as the original KRR.

### Questions
One concern I have is  about the computational complexity comparison between the proposed algorithm and the original. You claim that your algorithm reduces the time cost from the super-exponential in $d$ to the polynomial time. However, to my knowledge, the standard KRR method can only solve the low-dimensional task, where the dimension $d$ is typically small and can be neglected compared to the number of sample points.

I also note that  the previous algorithm has computational complexity of $n^2$ (if I can ignore $d$), while your algorithm lower the complexity to $n^{(2+\alpha)/(1+\alpha)}$. I suggest that the authors carefully discuss this issue.

### Soundness
3

### Presentation
3

### Contribution
3
