# On Rademacher Complexity-based Generalization Bounds for Deep Learning

- Decision: Reject
- Scores: 6, 3, 3

## Abstract
We show that the Rademacher complexity-based approach can generate non-vacuous generalisation bounds on Convolutional Neural Networks (CNNs) for classifying a small number of classes of images. The development of new contraction lemmas for high-dimensional mappings between vector spaces for general Lipschitz activation functions is a key technical contribution.  These lemmas extend and improve the Talagrand contraction lemma in a variety of cases. Our generalisation bounds are based on the infinity norm of the weight matrices, distinguishing them from previous works that relied on different norms. Furthermore, while prior works that use the Rademacher complexity-based approach primarily focus on ReLU DNNs, our results extend to a broader class of activation functions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper first generalizes the Talagrand contraction lemma to high-dimensional mapping case, then uses these new results to derive non-vacuous generalization bounds for CNNs. Their results show that the Rademacher complexity does not explicitly depend on the depth of the network for CNNs with some specific types of activation functions. some empirical results on the MNIST image classifications are also given.

### Strengths
1. This paper provides a novel and useful contraction lemma, which extends the previous Talagrand contraction lemma to high-dimensional mapping case.
2. Based on the new contraction lemma, the authors derive the bounds of the Rademacher complexity for CNNs, which does not explicitly depend on the depth of CNNs. This improves the results in previous papers.
3. The authors derive some non-vacuous generalization bounds for CNNs.

### Weaknesses
1. This paper only considers some specific types of activation functions.
2. The conclusion that the obtained Rademacher complexity does not explicitly depend on the depth of CNNs is not very convincing, since by Equations (30), (31), (34), the Rademacher complexity may exponentially depend on the depth of CNNs if the right-hand-side of (30), and (31) are large than 1. More discussions on this claim should be added.

### Questions
See the Weaknesses part. In line 251, "Let $\psi(x)$ is" should be "Let $\psi(x)$ be" .

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work studies the Rademacher complexity for the class of CNNs using any of ReLU family activation functions. Based on the authors' new version of Talagrand's contraction lemma, they provide the upper bound on the Rademacher complexity. Also, they conducted numerical experiments verifying the gap between the upper bound and error.

### Strengths
They try to identify activation functions and consider wider hypothesis space (e.g. $\mathcal{H}_+$) to provide new contraction coefficients.

### Weaknesses
**Questionable significance**

Though this work provides the new contraction lemma, in my opinion, the result is not so significant.
Compared to the known vector-valued Talagrands' contraction lemma ([1, 2]), theorem 2 seems to be able to have poor upper bounds since there is an additional term $\frac{1}{\sqrt{n}}| \psi (0) |$, which can even worsen the upper bound since the denominator is sublinear.
To clarify, please add more details and rigorously compare the lemmas in [1] and [2].
(e.g. why considering $\mathcal{H}_+$ lead better upper bound, ...)


**Non-vacuous bound**

The authors claim that the result provides a non-vacuous bound for CNNs with a small number of classes.
To verify this, they conducted several experiments under various setups.
However, as illustrated in Nagarajan & Kolter (2019), the bound may be vacuous depending on the norms of matrices (e.g. kernel) even if we carefully choose a set of parameters.
Thus, to prove the tightness, please provide a theoretical analysis of how their bounds behave under different matrix norm conditions in light of the results from Nagarajan & Kolter (2019).

**Comparison with other CNN bound**

Though there are lines of research studying generalization bound for CNNs, the authors do not provide any comparison with other known bounds. To highlight the novelty of the work, kindly suggest including a specific comparison section in their paper, highlighting key differences between their approach and existing CNN generalization bounds.

**Clarity Concerns**

There is no proof sketch or idea provided within the main text for any of the theorems; full proof in the main text or deferred to the appendix. For better readability, kindly suggest that the authors include brief proof sketches or key insights for the main theorems in the main text while keeping full proofs in the appendix.



**Reference**

[1] Maurer (2016), A vector-contraction inequality for Rademacher complexities

[2] Foster and Rakhlin (2019), $\ell_\infty$ Vector Contraction for Rademacher Complexity

### Questions
**1.About the remark 3**

The second bullet in remark 3 states that theorem 2 improves the bound for the case $m=1$. However, in Lemma 1 (Talagrand's lemma for $m=1$), the contraction rate is $\mu$, not $2\mu$.


**2. depth-dependent**

In theorem 8, the upper bound $F_L$ of $\mathcal{R}_{n}( \mathcal{F} )$ is defined using the recursive relation (34). 
Because of $\gamma_i$, $F_L$ seems depending on the depth, unlike the explanation in the remark 9. Could you provide a bit more details?

---

I am hoping that the authors will provide the clarifications stated therein in the rebuttal phase

### Soundness
1

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
4

### Summary
This paper explores the theoretical underpinnings of generalization in deep learning, specifically focusing on Convolutional Neural Networks (CNNs). The authors propose a new approach using Rademacher complexity to derive non-vacuous generalisation bounds for CNNs with certain activation functions like ReLU, Leaky ReLU, and Sigmoid. These bounds, unlike prior results, do not explicitly depend on the network depth. The authors validate their findings through experiments on MNIST image classification.

### Strengths
The authors employ the Talagrand's contraction lemma to develop novel   Rademacher-complexity-based generalization bound for CNNs and specialze the bounds to some common types of activation functions.

### Weaknesses
1. The paper's maths and presentation need to be significantly improved.
2. The theoretical contributions seem not sound enough.

### Questions
1. Lemma 1: $\epsilon$ is not defined.
2. The definition of network length should be provided at the beginning of the paper. It actually refers to the number of layers in Section 6. It would be better to adopt the conventional name "network depth".
3. What is the benefit of the bound that does not depends on the network length? It could not reflect the effects of depth in deep learning.
4. In Theorem 8, $L$ is not defined. Futhremore, $F_L$ depends on $L$. Does it contradict with the authors' claim?
5. What do the numerical results in Section 5 imply? There is no elucidation. Futhremore, the authors only implement one experiment which is far from enough for verification.
6. The title is not accurate. This paper only studies CNNs.

### Soundness
2

### Presentation
1

### Contribution
2
