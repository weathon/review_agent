# Global Convergence Rate of Deep Equilibrium Models with General Activations

- Decision: Reject
- Scores: 3, 6, 1, 5

## Abstract
In a recent paper, Ling et al. investigated the over-parametrized Deep Equilibrium Model (DEQ) with ReLU activation. They proved that the gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function. This paper shows that this fact still holds for DEQs with any general activation that has bounded first and second derivatives. Since the new activation function is generally non-linear, bounding the least eigenvalue of the Gram matrix of the equilibrium point is particularly challenging. To accomplish this task, we need to create a novel population Gram matrix and develop a new form of dual activation with Hermite polynomial expansion.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The training dynamics of over-parameterized DEQs are revisited in this study. The authors extend prior studies on ReLU DEQs and establish the linear convergence of training DEQs with general activations using a unique population Gramme matrix and a new kind of dual activation with Hermite polynomical expansion.

### Strengths
1. This paper provides a fine-grained analysis of the gradient dynamics of DEQs. It extends the results of the ReLU case in [1] to more general cases.

2. This paper proposes a novel population Gram matrix and develops a new form of dual activation with Hermite polynomial expansion. It appears that the proposed technical contributions can also be applied to the analysis of explicit neural networks.

[1] Ling Z, Xie X, Wang Q, et al. Global convergence of over-parameterized deep equilibrium models. International Conference on Artificial Intelligence and Statistics. PMLR, 2023: 767-787.

### Weaknesses
1 About the weight assumption. The authors assume that $W_{ij}\sim N(0,2\sigma_w^2/m)$ and $U\sim N(0,2/d)$. I do not understand the reason of using the scaling parameter "2". In ReLU case the scaling parameter "2" is commonly used for simplicity, but this paper investigates general activations.

2 About the existence and the uniqueness of $K$ (Proposition 12). In order to make sure Eq. (112) holds, one needs to make sure $2q^2\tilde{L}_q\sigma_w^2<1$ where 
$\tilde{L}_q=\frac{16L^2}{q^2} (\frac{\sigma_w^2}{m} \mathbb{E}G+\frac{3}{2})$
(as implied by Eq.(111)). However, Proposition 12 only requires that Assumptions 1and 2 hold, i.e. $\sigma_w^2<\frac{1}{8L^2}$. I do not think this condition is sufficient to guarantee $2q^2\tilde{L}_q\sigma_w^2<1$.

 Moreover,  the properties of $\mathbb{E}G_{11}$ are unclear. This makes the proof less rigorous.

3 Lemma 10 (Proof of Lemma 4 in [1]) plays a key role in the proof. However, [1]' proof works for ReLU function. The authors should explain the applicability of the proof to general activation functions.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors extend the framework by Ling et al. who showed linear convergence rate for the gradient descent applied to the quadratic loss function for over-parametrized Deep Equilibrium Model (DEQ) with ReLU activation. The same rate is obtained when ReLU is replaced by an activation function with bounded first and second derivatives. To obtain this rate, the authors bound the least eigenvalue of the Gram matrix of the equilibrium point by means of Hermite polynomial expansions.

### Strengths
The assumptions required in the main theorem 8 are fulfilled for commonly used activation functions such as sine and tanh.

The claimed theoretical statements evidence are supported by numerical experiments on MNIST and CFAR-10 datasets.  

The analysis techniques based on dual activation with Hermite polynomial expansion seem somehow original and elegant.

### Weaknesses
The authors state several auxiliary results in Section 5 that are essential for their main theorem but are not proved in the main the appendix. 
Not being an expert in this field and with a very limited amount of time (too short to read in detail the 21 pages of supplementary material), it is quite hard to judge whether the framework is correct or not. I believe that this work, possibly sound and surely interesting, would be worth publishing but the current conference format (coming together with a short allocated review time) might not be the best fit.

### Questions
Could you give examples of activation functions that would not fulfill the conditions of Theorem 8?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper extends a previous result from Ling et al. on the global convergence of DEQs proved for ReLU activations to a more general class of activations.

### Strengths
- the problem of understanding the theory of DEQs is interesting

### Weaknesses
- **contribution**: it is not clear what is the exact contribution of this work. To me it seems that it's merely an extension of the work of Ling et al. but it doesn't bring new theoretical ideas, proof techniques or closes a gap between theory and practice. This is largely reflected by the fact that a large portion of the text is extremely similar to the paper of Ling et al. It can also be seen in the fact that entire parts of the paper are dedicated to things that would usually be in the appendix like extended proofs, extended historical perspectives on generalization or examples. 
Moreover, while there is a claim that "a novel population Gram matrix" or "new form of dual activation with Hermite polynomial expansion" are introduced in this work, it is clear from reading them that they are direct extensions of Ling et al. or Daniely et al.
- **clarity**: so many notations are introduced (some even very unusual like $T$ for the equilibrium point of DEQs) which makes the paper difficult to follow.

### Questions
- what are the contributions of this work on top of extending the proof of Ling et al. to other activation functions?
- why is it important to extend the proof of Ling et al. to other activation functions?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper extends the NTK-like analysis of wide Deep Equilibrium Models converging linearly with gradient descent, which was previously proven only for the ReLU activation ([Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf)), to encompass general activation functions through advanced analysis of the Gram matrix.

Specifically, Deep Equilibrium Model ([Bai et al. (2019)](https://arxiv.org/abs/1909.01377)) defines the model output as the equilibrium value of a recursive equation, which corresponds to the output of infinitely-deep network with the same weight across all layers. The linear convergence of the overparameterized (wide) Deep Equilibrium Model using gradient descent is proven in ([Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf)) for the ReLU activation, following the NTK analysis. The proof of linear convergence requires lower bounding a certain gram matrix. To extend the result for ReLU to general activation functions, the lower bound argument should be more abstract, which is the main contribution of this paper.

### Strengths
### Convergence of Deep Equilibrium Model has not been proven for general activation functions

The previous work for the ReLU activation ([Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf)) provides a detailed literature review on the NTK-like analysis of Deep Equilibrium Models. As far as I understand, [Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf) is the first non-asymptotic analysis for the ReLU and there are no subsequent work for general activation functions. Thus I think the problem this paper addresses itself is new to a certain extent.

### All proofs seem to be correct at high level.

While there are some rough edges (e.g., in Lemma 2, a constant $C$ is introduced but not used anywhere in the statement), the overall argument leading to the theorem appears to be sound.

### Weaknesses
### The proof follows [Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf), and the modification required for dealing with general activation functions looks very basic.

In reviewing this paper, I have thoroughly examined the proof method of [Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf), which is a prior study. Through this examination, I have identified that the fundamental flow of the proof in this paper is essentially the same. For example, the following correspondence exists:

Theorem 2 in [Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf) - Theorem 7

Theorem 3 in [Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf) - Theorem 8

While the introduction of Hermite decomposition distinguishes it from the case limited to ReLU, it is worth noting that a significant portion of the proof remains identical to the original one. In evaluating this paper, it seems that the crucial point to consider is the novelty and significance of using Hermite decomposition to establish a lower bound on the kernel's eigenvalues, especially in comparison to the prior work that did not incorporate this technique. However, I am aware that such an idea is widely used in other relevant literature (e.g., [Misiakiewicz (2022)](https://arxiv.org/abs/2204.10425)). Due to these reasons, the technical contribution of this paper appears somewhat incremental, which is why I have reservations about recommending its acceptance.

### The literature review appears to be lacking in depth.

This paper seems to have overlooked several works on the application of NTK to DEQ, which are thoroughly explained in [Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf), and only mentioning [Ling et al. (2023)](https://arxiv.org/pdf/2205.13814.pdf). The Introduction chapter appears to begin with a very general discussion (introduction to deep learning) while omitting a review of literature directly relevant to this paper. As my suggestion, it might be beneficial to introduce DEQ first, followed by a presentation of existing analyses and challenges associated with it. This approach could provide a more comprehensive overview within a similar page length.

### The proof sketch is a mere list of claims.

The paper seems to just list theorems without providing sufficient explanations about what is fundamentally novel. Many of these theorems can be linked to references in prior literature. In my understanding, the novelty lies in the proof methods of Theorem 7 and 8, so please provide detailed explanations about them.

Furthermore, Section 6 appears to be overly verbose. In my opinion, it could be expected to hold, and even if it doesn't, the focus should be on proving it for just one activation function.

### (Minor) ``any general activation that has bounded first and second derivatives.'' (abstract) requires modification

In Theorem 8, the authors assume non-vanishing Taylor coefficients on the dual activation function. I do not think think bounded first and second derivatives suffice to satisfy this assumption.

### Questions
- It might be helpful if the paper could illustrate the challenges and difficulties of dealing with general activation functions by contrasting the proof methods with those from previous literature, which could highlight aspects that I might have overlooked.

- Could you provide more detailed information about the relevant literature?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
