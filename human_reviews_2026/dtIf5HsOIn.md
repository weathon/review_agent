# A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Understanding the dynamics of feature learning in neural networks (NNs) remains a significant challenge.
  The work of (Mousavi-Hosseini et al., 2023) analyzes a multiple index teacher-student setting and shows that a two-layer student attains a low-rank structure in its first-layer weights when trained with stochastic gradient descent (SGD) and a strong regularizer.
  This structural property is known to reduce sample complexity of generalization.
  Indeed, in a second step, the same authors establish algorithm-specific learning guarantees under additional assumptions.
  In this paper, we focus exclusively on the structure discovery aspect and study it under weaker assumptions, more specifically: we allow (a) NNs of arbitrary size and depth, (b) with all parameters trainable, (c) under any smooth loss function, (d) tiny regularization, and (e) trained by any method that attains a second-order stationary point (SOSP), e.g. perturbed gradient descent (PGD). At the core of our approach is a key $\textit{derandomization}$ lemma, which states that optimizing the function $E_{x} \left[g_{\theta}(Wx + b)\right]$ converges to a point where $W = 0$, under mild conditions. The fundamental nature of this lemma directly explains structure discovery and has immediate applications in other domains including an end-to-end approximation for MAXCUT, and computing Johnson-Lindenstrauss embeddings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors prove a de-randomization lemma which reveals information about the loss landscape of problems that fit into a certain structural framework. The lemma essentially says that when trying to optimize a smooth function over a matrix W, such that W is multiplied by a standard normal vector and passed as an argument to the smooth function, than any candidate solution W* which is approximately second order stationary (which includes all local minima and potentially some saddle points) is close to 0, with the proximity depending on how close the point is to being second order stationary. 

This lemma is then demonstrated on 3 diverse problems of interest including:
1. proving a statement about training dynamics for neural networks discovering low dimensional structure.
2. a method for derandomizing the rounding procedure which is applied to the Semi definite program solution of a continuous version of the max cut problem
3. a method of finding deterministic JL embedding matrices.

The lemma provides an interesting insight into the loss landscape for a certain class of functions. The examples demonstrate an interesting and diverse range of applications which have a common structure.

While interesting, I think demonstrating the importance of the main lemma hinges on the examples. The first example which received the majority of the attention in the paper is solid and interesting but feels somewhat marginal in terms of extending  the result of Mousavi-Hosseini et al., 2023. I do think the other two examples are very interesting but I do not totally understand the impact of the lemma in regards to the problems it is applied to. If this was highlighted more clearly and I can be convinced of the significance this lemma brings to the MaxCut and JL applications, I would think the paper is much stronger.

### Strengths
The various applications are interesting as they consider very different but high profile problems which are of interest to many different communities. 

The general insight into the loss landsape is also interesting and I agree with the authors that others may find new applications of this lemma to different problems.

### Weaknesses
It seems as though for the L-smooth function who's expectation we want to optimize, the sample complexity for finding a solution with perturbed gradient descent is growing in L. When applying the lemma to optimize probabilities (written as expected indicators) the smooth approximation to the indicators requires larger L for better approximation. In these cases there is an obvious tradeoff between sample complexity and accuracy. I think this is clear for the max-cut theorem but is sort of hidden in the JL example and perhaps this should be made more clear. 

To follow up on the point above, given the application of the lemma to these two problems, it would be interesting to have some insight into how quality of approximation of the indicator affects quality of the final solution.

It would be nice to have a bit more context into the final two applications in terms how effective the proposed de-randomization problems are. You suspect that you have shared the first optimization-based approach for
derandomizing the Goemans-Williamson algorithm, but have not commented on how this approach compares to the other mentioned methods of  of conditional expectations, small-bias spaces, or explicit pseudorandom constructions. For JL you mentioned that you match the SOTA results but also later that actually recover another method and there is no real comparison between your or other methods (See the first question in the 'Questions' section).

I think the connections of the structure used in your lemma and these different applications are interesting. It is not totally clear to me however what exactly applying your lemma to these problems is contributing to the study of these problems. Is it new methods for solving these problems? Or contributing to the theoretical understandings of these problems or something else? If it is new methods I think a better comparison is warranted.

### Questions
For JL how exactly does your method differ from what others have already done? Or is the contribution of this example to demonstrate that your lemma can be applied and hence provide insight into the loss landscape for this problem?

A small note: I believe in appendix C (line 903) the definition of $\ell'$ should include the $\| W_\|\|^2_F$ not $\| W_\perp\|^2_F$ right?

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
2

### Summary
This paper develops a general derandomization framework for structure discovery in neural networks and other domains. Building on the prior work of (Mousavi-Hosseini et al., 2023), which showed that SGD can lead to low-rank structure in the first layer of a two-layer neural network under strong regularization, this paper extend the result to a broader setting with trainable bias parameters, general smooth loss functions and optimizers. The core contribution is a key derandomization lemma stating that optimizing a general regularized loss in expectation form converges to the origin point, under mild conditions. The authors also apply the same lemma to derive derandomized algorithms for the MAXCUT problem and Johnson–Lindenstrauss embeddings.

### Strengths
1. The authors propose a new derandomization lemma for analyzing feature learning behaviour of neural networks. The obtained results extend and generalize the analysis in (Mousavi-Hosseini et al., 2023).  
2. The authors applied the result for derandomization in other domains including MAXCUT and JL embeddings.

### Weaknesses
1. The assumption of first and second order smoothness is restrictive and does apply to many practical scenarios, such as ReLU networks, forcing the authors to adopt the approximation in Section 4.2. Can this result be extended to non-smooth settings such as the original ReLU activation?  
2. The applications to MAXCUT and JL embedding are actually not new since, as acknowledged by the authors, there are already known derandomized algorithms for both problems. Can this result be applied to new problems where derandomized algorithms are unknown?

Typos and other comments: 

1. in line 903, the definition of $l’_{\\theta’}$ below equation (18), $\\lambda\\|W{\\bot}\\|\_{F}^2$ should be $\\lambda\\|W\_{\\|}\\|\_{F}^2$.  
2. The “Summary of our contributions.” section seems to be missing.

### Questions
See strengths and weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript studies representation learning in neural networks in a multi-index setting. In particular, the authors show that, using the definition of approximate second-order stationarity, the result in (Mousavi-Hosseini et al., 2023) can be established under more general conditions. They provide algorithms that find approximately second-order stationary points under smoothness assumptions and also suggest alternative uses of these algorithms for MAX-cut and embedding finding.

### Strengths
- The paper extends earlier result (Mousavi-Hosseini et al., 2023) to a broader setting.

### Weaknesses
- The paper does not guarantee learning the teacher directions; rather, it shows that the component of the student weights in the subspace orthogonal to the teacher directions vanishes. However, this does not guarantee recovery of the teacher directions. For example, consider the setting where the teacher is $y = \mathrm{He}_4(\langle \theta, x\rangle) + \epsilon$ for some unit vector $\theta$, and the student is $\hat y = \mathrm{He}_2(\langle w, x\rangle)$, $w \in \mathbb{R}^d$, with $x \sim N(0, I_d)$ and square loss. Then $0$ will be an approximate stationary point for any $\rho > 0$, even though it is a saddle point. In this sense, the result does not guarantee learning.

- The example above relies on the orthogonality of the Hermite polynomials $\mathrm{He}_4$ and $\mathrm{He}_2$. Hermite decompositions in multi-index settings have been studied extensively since (Mousavi-Hosseini et al., 2023), and there is now a rich literature that discusses the sample complexity of recovering teacher model with first-order algorithms based on its Hermite decomposition in Gaussian space [1,2,3]. The manuscript does not compare its results to this literature. What improvement does this paper provide? The manuscript would benefit from this discussion.


[1] Gérard Ben Arous, Reza Gheissari and Aukosh Jagannath. “Online stochastic gradient descent on non-convex losses from high-dimensional inference.” J. Mach. Learn. Res. 22 (2020): 106:1-106:51.

[2] Alex Damian, Loucas Pillaud-Vivien, Jason D. Lee and Joan Bruna. “Computational-Statistical Gaps in Gaussian Single-Index Models.” ArXiv abs/2403.05529 (2024): n. pag.

[3] Jason D. Lee, Kazusato Oko, Taiji Suzuki and Denny Wu. “Neural network learns low-dimensional polynomials with SGD near the information-theoretic limit.” ArXiv abs/2406.01581 (2024): n. pag.

### Questions
- In Eq. (3), $U$ is not bold; I believe this is a typo.

- Following my points above, in line 260 the authors claim:  
  “Our goal is to show that the perpendicular component $W_{\perp}$ converges to zero, implying that the first-layer weight matrix $W$ converges to a rank-$k$ matrix,”  
  which is incorrect. This statement should be revised.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Consider the problem of optimizing any function of the form $\mathbb{E}[g(Wx + b)] + \lambda \\|W\\|^2_F$, where $x$ is drawn from a standard Gaussian, with respect to parameters $W, b$. The main result of this paper states that any second-order stationary point (SOSP) of this objective must have $W = 0$ (and an approximate version holds for approximate SOSPs). This has several applications:
1. Neural network optimization: when we try to optimize the squared loss between a "teacher" network of the form $h(Ux)$ and a "student" neural network with initial layer weights $W$, then at any SOSP $W$ will have its component orthogonal to $U$ approximately vanish. This could be seen as optimization "discovering" the (typically low-rank) subspace of $U$.
2. Derandomization: this lemma can be cleverly applied to certain randomized algorithms (such as MAX-CUT or Johnson-Lindenstrauss) in obtain a deterministic solution that (nearly) matches the performance of the randomized one.
The proof of the main lemma is simple and relies crucially on Stein's lemma for Gaussian random variables.

### Strengths
The main result of the paper is appealingly clean, general, and powerful. The applications are creative and interesting. In the case of neural network optimization, the paper recovers a "structure discovery" result which had been proved much more painstakingly (and under more restrictive assumptions) in prior work. The derandomization applications are conceptually thought-provoking and potentially of independent interest in complexity theory and randomized algorithms. The paper is mostly written quite clearly. I did not verify all the proofs in detail but the main ideas seem technically sound (and in particular the main lemma proof looks correct). Overall, this looks like a good paper.

### Weaknesses
A weakness of the paper is that it is stylized in certain important ways. Probably the biggest concern is that all the results rely on perfect Gaussianity of the random variables, because of the crucial use of Stein's lemma in the main result. Another concern is that the functions involved be smooth, which necessitates the use of smooth activation functions etc. Finally, all the claims require regularization and only hold for $\rho$-SOSPs with $\rho$ very small compared to the regularization strength $\lambda$. This limits the interpretation of some of the more practical applications, esp concerning neural network optimization. Indeed, in that application it is not clear that this meaningfully explains "structure discovery" in any way that illuminates what might be happening in a real world setting; rather this seems like more of a curiosity about the Gaussian. The algorithmic applications only really seem suitable for cases where the algorithm designer synthetically injects Gaussian randomness into an algorithm (as with randomized rounding and random projections), not cases with "real world" randomness. This is probably fine, but it bears noting.

### Questions
1. I am curious if the results in this paper extend even in an approximate way to distributions that are close to but not perfectly Gaussian (e.g. sub-Gaussian, log-concave, etc). Would be curious to hear the authors' thoughts on this and more generally the restrictiveness of the Gaussianity assumption.
2. In general throughout the paper, the authors need to be clearer about what the variables being optimized are in any given objective function. Even the term SOSP is always really "SOSP of certain optimization variables". This is sometimes but not always clear from context. In particular, in section 4, I was quite confused by the step in going from Eq 5 to Eq 6, because $W_{\parallel}$ suddenly disappears from the objective altogether. When we actually optimize the risk, we optimize all of $W$ --- i.e., really we are considering an SOSP of both $W_{\bot}$ and $W_{\parallel}$. It is also extra confusing because $\ell'$ implicitly depends on $W_{\parallel}$. I had to spend some time convincing myself that we can effectively work as if $W_{\parallel}$ were a constant equal to its SOSP value. Perhaps such manipulations are obvious to an optimization audience but I think they are somewhat subtle and deserve very clear, explicit exposition (perhaps in the appendix, if space does not permit).
3. The interpretation of the main lemma as a "derandomization" lemma deserves clearer explanation. It only really comes up in Section 5 and even there it is a bit conceptually subtle. In fact this is a good opportunity in the paper to state purely in words what the main lemma is really saying.

### Soundness
4

### Presentation
3

### Contribution
3
