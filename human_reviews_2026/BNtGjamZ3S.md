# A Bootstrap Perspective on Stochastic Gradient Descent

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Machine learning models trained with stochastic gradient descent (SGD) can generalize better than those trained with deterministic gradient descent (GD). In this work, we study SGD's impact on generalization through the lens of the statistical bootstrap: SGD uses gradient variability under batch sampling as a proxy for solution variability under the randomness of the data collection process. We use empirical results and theoretical analysis to substantiate this claim. In idealized experiments on empirical risk minimization, we show that SGD is drawn to parameter choices that are robust under resampling and thus avoids spurious solutions even if they lie in wider and deeper minima of the training loss. We prove rigorously that by implicitly regularizing the trace of the gradient covariance matrix, SGD controls the algorithmic variability. This regularization leads to solutions that are less sensitive to sampling noise, thereby improving generalization. Numerical experiments on neural network training show that explicitly incorporating the estimate of the algorithmic variability as a regularizer improves test performance. This fact supports our claim that bootstrap estimation underpins SGD's generalization advantages.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a theoretical framework for understanding the generalization properties of Stochastic Gradient Descent (SGD). The authors decompose the generalization gap and introduce the concept of "algorithmic variability", which they analyze through the lens of statistical bootstrapping. Based on this decomposition, the authors construct two novel regularizers and empirically validate that their inclusion can lead to improved generalization performance on tasks including sparse regression and neural network training.
However, there are still some concerns to me. Therefore I lean to a rejection at the time being. 
Specifically, I am not sure whether the idea in this paper has significant differences to algorithm stability, and whether the derivation of this paper is meaningful. 
See below for more details.

### Strengths
1. The paper posits that SGD uses the gradient variability (caused by mini-batch sampling) as a "bootstrap estimate.
2. This paper proves that the expected generalization gap is determined by the trace of the product of the solution's Hessian matrix and the "algorithmic variability" matrix.
3. This paper designs a new regularizer based on the theoretical findings. 
4. The authors further provide empirical evidence on this regularizer.

### Weaknesses
1. [Major Concern] It seems that Assumption 2 directly leads to a small Variability (Eqn 3). However, the authors did not discuss it much. If so, I cannot be convinced that Eqn (3) is the dominate term compared to Eqn (4), where Eqn (4) also contains the epsilon[2, T] term. 
2. [Major Concern] I am not convinced that this paper has significant differences with the line of algorithmic stability. The authors claim in Line 466 that "this paper considers "Hessian-weighted and evaluated at the solutions"". It seems that algorithmic stability can include this case with pretty minor changes. Due to the simplicity, algorithm stability just bound the Hession with smoothness, and use iteration to reach the solution. But starting from the definition of algorithm stability, these are not necessary. The authors shold provide more evidence on how this paper performs differently with algorithm stability. 

[Minor]
1. The authors claim that "we prove rigorously that by implicitly regularizing the trace of the gradient covariance matrix, SGD controls the algorithmic variability." According to the paper's derivation, the algorithmic variability is bounded by two components (corresponding to the latter term in Eq. 6 and Eq. 7). While the authors convincingly connect the implicit regularization of SGD, as identified by Smith et al. (2021), to the first component (Eq. 6), they do not provide evidence or argumentation that SGD also implicitly regularizes the second component (Eq. 7). Consequently, the claim that SGD "controls the algorithmic variability" in its entirety appears to be an overstatement. 
This significantly limits their theoretical contribution, as the work seems to demonstrate that vanilla SGD only addresses a part of the problem identified by the authors. 
2. The paper's analysis of the proposed regularizers, Reg1 and Reg2, lacks sufficient depth regarding their interplay and individual utility. For instance, given that the authors identify Reg1 as an existing *implicit* regularizer of SGD, a crucial discussion is missing on the utility of its *explicit* inclusion. What is the tangible difference between applying Reg1 explicitly versus relying on its implicit effect? Would applying only Reg2, which is the component not addressed by vanilla SGD, be a more practical and principled approach? The paper would be substantially strengthened by ablation studies that dissect the individual contributions of Reg1 and Reg2 and clarify their roles in guiding SGD towards better-generalizing solutions.
3. The practical significance of this work is severely hampered by the unaddressed computational overhead of the proposed regularizers. Both Reg1 and Reg2, as defined, require the computation of the full-batch gradient at each training step. This is a prohibitive cost for large-scale datasets and fundamentally contradicts the core philosophy of SGD, which is designed precisely to avoid such computations. The absence of any discussion on this issue, or on potential efficient approximations, makes it difficult to assess the empirical value of the proposed method. As it stands, the practical guidance offered by the paper appears limited.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies SGD's impact on generalization for machine learning models. Based on the provided  analyses, it proposes two regularization schemes, which are shown to benefit generalization for a few toy datasets.

### Strengths
The question raised in the paper is important and the paper tsts a new regularization method based on the analyses and shows that it might benefit generalization

### Weaknesses
The theoretical contribution appears to be incremental, as, to my understanding, the main insights came from Smith et al. (2021).  The empirical evaluation is very limited, as the results are tested only on a very specific synthetic dataset with a sparse prior and FashionMNIST.

### Questions
1) I did not understand how the analyses are specific to the SGD as opposed to the non-stochastic GD. As the opening sentence of the abstract mentions the difference between generalization of GD and SGD as a motivation, I would like to ask the authors to elaborate more on this. How can we see from the bounds derived in the paper that SGD might outperform GD? 

2) As for the regularizers part, what are the novel insights made in the paper compared to Smith et al. (2021)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper tries to understand SGD from the view of bootstrapping: SGD favors minima with smaller variance of stochastic gradient.

### Strengths
1. The top example in Section 2 is attractive and illustrative.

### Weaknesses
1. The presentation of the theoretical part is a bit confusing.
- The theoretical results are listed as Lemmas 1 and 2 as well as Proposition 1, without a theorem that usually serves as the center of discussions. This makes me confused about what is the main theoretical contribution of the paper.
- The discussions after Lemmas 1 and 2 mainly discuss why the lemmas hold, and do not actually help with the understanding of the theoretical results (especially for Lemma 2, whose righthand side has a lot of terms).
2. My understanding is that the core of the theoretical analysis is the correspondence of Equations (6) and (7) with Equations (10) and (11), which provides a viewpoint from the implicit regularization of SGD by "bootstrapping" the gradients. However, this part lacks a comparison against GD or noisy GD.
3. According to my understanding, the technical contribution is minor. Lemmas 1 and 2 are basically Taylor expansion, and Proposition 1 is basically the strong law of large numbers.

I would honestly confess that I do not understand all the details of the paper, and would be happy to discuss with the authors, other reviewers and the AC. My score of 2 currently represents my unconfident understanding. i think the intuition of the paper is good, but the theoretical part may need improvements.

### Questions
1. Can the authors show more details of the algorithm SGDwReg2, especially how to estimate the term Reg2?
- If Reg2 is estimated in an exact way, then SGDwReg2 requires knowledge of the entire dataset at each minibatch update. In this case, is it possible to design an adaptation of SGD that incorporates the idea of SGDwReg2 but without the requirement of the entire dataset?
- If Reg2 is approximated, can the authors show the details of approximation?
2. How does the bootstrapping view compare with the idea of variance-reduction techniques like SVRG?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to provide a novel eccplanation for the superior genetalozation property of SGD compared wirh GD, from a boostrap perspctive. Specifically, under certein assumptions, the authors show that the generalization error can be decomposed into a dominant Hessian=-preconditioned algorithmic variability term and several small terms. They further argue that the algorithmic variavbilit is stronhly correlated to the accumulated empirical covriance of gradients. As a consequence, they empirically estalish that SGD regularizes algorithmic variability as a bootstrap estimate, and hence improving the generalization error through this correlation.

### Strengths
This paper is clearly written and has a nice structure.

### Weaknesses
Although the authors provide an upper bound on the generalization error via algorithmic stability, the paper does not explicitly establish how SGD regularizes this term theoretically. Moreover, there is no theoretical characterization of the generalization gap between SGD and GD. Another concern arises from the assumptions: while Assumption 1 appears standard, Assumption 2 is rather demanding and may not hold in many scenarios: existing theoretical results generally suggest that the upper bound on uniform algorithmic stability grows with the number of iterations. This implies that the bias term, rather than variance, often dominates the generalization error. From this perspective, the argument that “SGD generalizes better because it regularizes the gradient variance” may not be entirely convincing.

### Questions
No further questions.

### Soundness
2

### Presentation
3

### Contribution
1
