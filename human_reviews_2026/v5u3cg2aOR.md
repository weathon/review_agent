# Why and When Deep is Better than Shallow: An Implementation-Agnostic State-Transition View of Depth Supremacy

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Why and when is deep better than shallow? We answer this question in a framework that is agnostic to network implementation. We formulate a deep model as an abstract state-transition semigroup acting on a general metric space, and separate the implementation (e.g., ReLU nets, transformers, and chain-of-thought) from the abstract state transition. We prove a bias-variance decomposition in which the variance depends only on the abstract depth-$k$ network and not on the implementation (Theorem 1). We further split the bounds into output and hidden parts to tie the depth dependence of the variance to the metric entropy of the state-transition semigroup (Theorem 2). We then investigate implementation-free conditions under which the variance grow polynomially or logarithmically with depth (Section 4). Combining these with exponential or polynomial bias decay identifies four canonical bias-variance trade-off regimes (EL/EP/PL/PP) and produces explicit optimal depths $k^\ast$. Across regimes, $k^\ast>1$ typically holds, giving a rigorous form of depth supremacy. The lowest generalization error bound is achieved under the EL regime (exp-decay bias + log-growth variance), explaining why and when deep is better, especially for iterative or hierarchical concept classes such as neural ODEs, diffusion/score-matching models, and chain-of-thought reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors study the power of depth for neural networks by viewing deep networks as abstract 
state-transition semigroups and proving architecture-agonistic Rademacher complexity bounds for these abstract models.
This leads to a bias-variance decomposition, where the variance is the Rademacher complexity and the bias is determined 
by how well actual networks can approximate those idealized abstract models. 

Then, they provide several sufficient conditions under which the variance grows polynomially or logarithmically. 
Combing these two cases with the exponential or polynomial decay of the bias leads to four regimes, and in each 
of these regimes, the authors identify the optimal depth by equating the bias and variance terms. Finally, several
examples are given for these regimes.

### Strengths
* Overall, this is a clean and well-written paper. 
* The idea of decoupling the implementation details and idealized networks through abstract state transitions is 
  interesting and executed well.

### Weaknesses
The main issue of this paper is that it is more of a paper about abstract Rademacher complexity instead of deep learning 
theory. 

Architecture-independent Rademacher complexity bounds have gone out of fashion since around 2020, and I believe 
this is for good reason. It was empirically shown in the seminal paper [1] that modern networks can fit random noises
but still generalize well if trained properly, indicating that vanilla Rademacher complexity may not be a good way 
to measure generalization in deep learning. This is why most of the papers after that consider specific models 
and/or algorithmic regularization, hoping to show that certain biases induced by the architecture and/or training 
algorithms make the network generalizable even when the network has the capacity to fit random noises. 
I do not think this paper manages to resolve/go around the above issue of implementation-agnostic generalization 
bounds, as it is still based on bounding the covering number of some abstract generic deep networks.

To be fair, the authors do provide several sufficient conditions under which the covering number grows polynomially 
with the depth (Sec. 4.1). However, most of them are either hard to check or impose strong constraints, and the 
examples provided are also either abstract or about a very specific non-conventional problem (neural operators).

A small suggestion: 
It might be better to rewrite Table 1 based on the target error. For example, suppose that the target error is $\epsilon$.
Then, we can derive the needed depth for the bias to be small and then the number of samples required to make the 
variance small. The optimal generalization error is usually not the quantity of interest once it falls bellow a certain 
threshold. Trying to achieve the optimal generalization error makes the depth grows as the number of samples grows, 
which makes the bound look strange, since intuitively having more data should always help the model. 

[1] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals. Understanding deep learning requires rethinking generalization. 2017

### Questions
I would be willing to raise my score if the authors can find a concrete example demonstrating the applicability of their results.  
More specifically, could you find a concrete target function class (composition of quadratic networks with certain properties, 
functions with a tree-like hierarchical structure, or something at a similar level of concreteness), and a class of 
abstract models and their implementations, for which the results of this paper can be used to derive non-vacuous 
bounds?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper develops a theoretical framework for understanding when and why deeper neural networks outperform shallow ones. The key innovation is formulating deep networks as abstract state-transition systems on metric spaces, separating the mathematical analysis from specific implementations. The authors prove a bias-variance decomposition where variance depends only on the abstract depth-k network structure, then analyze how covering numbers of "word balls" (compositions of k functions) grow with depth. They identify four regimes (EL/EP/PL/PP) based on whether bias decays exponentially or polynomially and whether variance grows logarithmically or polynomially with depth. The framework provides explicit optimal depths and explains why depth helps most for hierarchical/iterative tasks like neural ODEs, diffusion models, and chain-of-thought reasoning.

### Strengths
- The approach taken in the paper appears to be genuinely novel and of interest to mathematicians.
- The mathematics introduced in the paper is backed up by extensive proofs and discussion.
- The task of identifying optimal depths of neural networks is of great importance to the field.

### Weaknesses
- The relationship between theoretical concepts to real neural networks is weak
- There are no genuine examples provided to assist readers in understanding the practical benefits of the work
- The paper is very mathematically dense, reducing its overall impact on a broad community. This work feels more at home in a mathematics journal than a broader conference. It gives the impression that little to no effort was made to make the work accessible to the majority of machine learning practitioners.

### Questions
- In the paper, the authors write, "In practice, trained networks obtained by optimization are not
identical to exact empirical risk minimizers, but we assume they are identical for simplicity". This appears to be quite a substantial approximation. Can the authors shed light on how much of a difference one can expect in a real setting where gradient descent is used to train models?
- Were any practical experiments performed, perhaps on small networks, to validate the theory shown here in practice?
- Given that the authors claim to have found a means of selecting optimal depths of networks, how does one practically apply this to their models today?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides an implementation-agnostic explanation of why and when deep networks outperform shallow ones through a novel bias–variance decomposition. The authors show that while the variance term depends only on the depth-k state-transition structure (not on the network type), the bias decays with depth—often exponentially—leading to optimal finite depths $
k* >1$. They identify four canonical bias–variance regimes and show that exponential bias decay with logarithmic variance growth (the EL regime) explains depth supremacy in hierarchical or iterative models such as neural ODEs, diffusion models, and chain-of-thought reasoning.

### Strengths
The paper is well written and easy to follow. Analyzing the generalization error of neural networks through a bias–variance decomposition is a compelling and insightful approach, offering a clear framework to understand the benefits of depth.

### Weaknesses
Although the contribution is interesting, several aspects remain unclear to me. Please see my questions below.

### Questions
-**Section 2.3**: I am confused about the role of the embedding operator — does it address optimization difficulties or something else? The authors state ``In practice, trained networks obtained by optimization are not  identical to exact empirical risk minimizers, but we assume they are identical for simplicity.'' It is not clear to me why this section is titled “Regularized Empirical Risk Minimization”; the connection to regularization is not well explained, and the section feels conceptually confusing.
I would suggest that authors explain this section and the different sources of errors clearly. 

-**Section 3.3**: I am confused about the novelty of Theorem 1. How does this result differ from or extend the existing results presented in Mohri’s book? Foundations of Machine Learning, Chapter 3. 

-**Theorem 1**:  I think the error term  $ε_{imp}$ should also depend on the structure of the network space. Please clarify. 

-While the authors analyze the variance across different structural settings, it is not clear how the approximation (bias) error behaves exactly. Or did I perhaps overlook that part?

### Soundness
3

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
3

### Summary
This paper provides a framework to prove bounds on the generalization performance of deep neural networks.

The setting is abstracted as learning a function in a function class which consists of a composition of blocks lying in some $F \subset C(X,X)$, composed with a read-out head in $H \subset C(X)$. 

A novel bound combining Rademacher complexity of H, plus a Dudley-style integral involving the covering numbers of $B_k := \underbrace{F \circ F \circ \dots F}_{k \mbox{ times }}$, is proved.

Subsequently, the paper presents several sufficient conditions for the covering numbers of $B_k$ to be either $O(1)$ or $\poly(k)$ or $\exp(O(k))$ as $k$ grows.

Finally, the paper considers the tradeoff between the bias term (deeper architectures yield lower bias because they are more expressive), and the variance term (deeper architectures yield higher variance because they are more expressive), and derives an optimal depth based on this tradeoff.

### Strengths
The theoretical results are precisely presented and appear to be sound to the best of my knowledge.

Theorem 2 provides a nice way to conveniently bound the generalization error.

In Section 4, there is a plethora of sufficient conditions under which the covering number of $B_k$ remains bounded. These involve nice connections with various other branches of mathematics.

### Weaknesses
* Overall, I do not see what is fundamentally different between this work and prior works, beyond proposing a more general framework. The introduction's motivation of the paper is that classical measures of generalization suggest that generalization should worsen with depth, and this paper provides an alternative perspective. However, I am not convinced that this paper provides a truly different view.

    - The bounds on covering number of the class in this paper increase with depth (similarly to generalization error bounds growing with depth in the prior work that this paper cites).
    - This paper gets around this by positing that the bias decreases with depth, and computing the depth for an optimal a bias-variance tradeoff. But the idea that bias might decrease with depth is the motivation of past works that prove expressivity separations between neural networks of different depths.

* I am also not convinced that providing the generalization bounds in this work (based on covering number) are the most promising path to understand why networks generalize. Many networks in practice, such as image networks are overparametrized, and can memorize random labels when trained. See, e.g. "Understanding deep learning requires rethinking generalization" by Zhang et al., 2016. Therefore, the optimization must be providing some regularization that allows the network to learn a simple solution. Getting a concrete understanding of what that regularization is seems to be a more promising path.

### Questions
See weaknesses:
* How is the approach fundamentally different from prior works providing generalization bounds?

### Soundness
4

### Presentation
3

### Contribution
2
