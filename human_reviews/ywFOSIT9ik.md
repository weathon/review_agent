# Revisiting Zeroth-Order Optimization:  Minimum-Variance Two-Point Estimators and  Directionally Aligned Perturbations

- Decision: Accept (Spotlight)
- Scores: 8, 6, 6, 6, 8

## Abstract
In this paper, we explore the two-point zeroth-order gradient estimator and identify the distribution of random perturbations that minimizes the estimator's asymptotic variance as the perturbation stepsize tends to zero. We formulate it as a constrained functional optimization problem over the space of perturbation distributions. Our findings reveal that such desired perturbations can align directionally with the true gradient, instead of maintaining a fixed length. While existing research has largely focused on fixed-length perturbations, the potential advantages of directional alignment have been overlooked. To address this gap, we delve into the theoretical and empirical properties of the directionally aligned perturbation (DAP) scheme, which adaptively offers higher accuracy along critical directions. Additionally, we provide a convergence analysis for stochastic gradient descent using $\delta$-unbiased random perturbations, extending existing complexity bounds to a wider range of perturbations. Through empirical evaluations on both synthetic problems and practical tasks, we demonstrate that DAPs outperform traditional methods under specific conditions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the author(s) propose a new perspective on zeroth-order methods, focusing on an optimization problem that minimizes the variance. After carefully crafting the problem instance, based on constraints that are standard in literature, some comments on classic choices of fixed-length perturbations are made. This is the right motivation to proceed and advocate for  "directionally aligned perturbations", the other "optimal choice". In particular, the theorems derived in this branch for convergence of SGD under standard assumptions make explicit the role of higher moments of the distribution, while still recovering nice to parse upper bounds. To conclude, the author(s) come back to the directionally aligned perturbations and show some experiments for synthetic datasets and a Language task. The practical feasibility of their optimization problem is also discussed.

### Strengths
The paper presents a nice idea to describe a framework. The optimization problem set up is easily solvable, making the narrative flow. The result is approachable, clear, allowing to derive convergence rates that are expressive of all the dependencies on the various parameters. The plot converges towards "defending" these directionally aligned perturbations, which are of interest even simply because the standard choice is the other optimal one. By passing through simple examples, they are also able to experimentally verify their formulation. Overall a very standard but structured paper in optimization.

### Weaknesses
See also the questions below. 

The main weakness is about wording: you claim you identify the optimal distribution in the abstract, while indeed you do not. I feel like the sentence should be adjusted. You find a sufficient condition for optimality subject to a construction of isotropic perturbations (you $\delta$-unbiasedness) and a Taylor approximation. This to me is not identifying an optimal distribution, nor identifying anything optimal at all, if not only wrt your specific criterion, which then you would have to specify anyway. 

I feel like the other main weakness is experimental validation. I believe you have brought the best results you could find, and still, the improvement is marginal. For example, figure $3$ right is really an improvement in machine precision. Figure $4$ is more promising. I also acknowledge that we should not care about SOTA but about understanding, so this is a weakness that is not suggesting any further comment. 

The other point is that you do not discuss the accumulation of errors when you (i)
 perform the Taylor approximation and (ii) perform the gradient estimation. I believe the two should be theoretically explored further to understand in restricted settings how much is lost wrt the convergence theorems. 

Lastly, no limitations are discussed. 


###### Typos
Please do not count these as weaknesses. 
- You never define $\nabla f(x; \xi, v)$, (e.g. line 085), since the notation for derivatives variable, I would define it. 
- "In the mean while" (line 166), meanwhile; 
- The numbering of lists as $(1), (2)$ etc resembles a lot equations, not a typo but a potential source of anti-dynamic reading; 
- line 215, you say $< 0$, probably meant to be finite. 
- line 264 "to a specific types of..."
- line 269 "achiving"
- "classitcal" (line 472)
- "it solves the projection is..." (line 864)
- corollary 3.2 (b) the sentence is not correct logically. If we add the assumption by choosing further specific scalings we get the result, right?

### Questions
1. Have you analyzed the impact of the Taylor approximation in your analysis? 
2. Have you analyzed the impact of the difference between theoretical analysis and estimating gradients in the wild? I acknowledge the experiment for the easy functions, where it is exact. 
3. Why would the assumption of isotropic noise $\mathbb{E}[vv^\top] = \delta I_d$ be valuable in terms of perturbations? Why not something else?



I am very open to changing my score once the issues I have raised are addressed!

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the zeroth-order gradient estimator and identifies the optimal distribution of random perturbations that minimize the gradient estimator's variance.
The problem is formulated as a constrained optimization problem. And it is shown that the optimal perturbations maintain a fixed length or align directionally with true gradient.
These gives two classes of random perturbations that achieve the minimum variance : Constant magnitude perturbations and Directionally aligned perturbations.
Convergence of SGD with both these classes of perturbations are proved. And some experimental results are shown.

### Strengths
The problem studied is of significant interest to the optimization community. And it shows two classes of random perturbations that give minimum variance.

### Weaknesses
In the main theorem (Theorem 2.2), what about only if part ? Does it happen that equality holds in theorem only if the given conditions (a) or (b) is satisfied ?
A discussion of this would be interesting.
The experimental results are weak. Only one practical application of language model optimization is given.
No comparisons with other constant magnitude perturbations: random coordinate/direction sampling and Rademacher distribution are shown.
Why DAP perturbations give better performance than uniform perturbation in experiments is not clear. As the theorem says that theoretically both give minimum variance.

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the two-point zeroth-order gradient estimator and identify the optimal distribution of random perturbations that minimizes the estimator's variance. This paper formulates it as a constrained functional optimization problem over the space of perturbation distributions. This paper reveals that optimal perturbations either maintain a fixed length or align directionally with the true gradient. While existing research has largely focused on fixed-length perturbations, the potential advantages of directional alignment have been overlooked. To address this gap, this paper delves into the theoretical and empirical properties of the directionally aligned perturbation (DAP) scheme, which adaptively offers higher accuracy along critical directions. Additionally, this paper provides a convergence analysis for stochastic gradient descent using $\delta$-unbiased random perturbations, extending optimal complexity bounds to a wider range of perturbations. Through empirical evaluations on both synthetic problems and practical tasks, we demonstrate that DAPs outperform traditional methods under specific conditions.

### Strengths
This paper explores the two-point zeroth-order gradient estimator and identify the optimal distribution of random perturbations that minimizes the estimator's variance. This paper formulates it as a constrained functional optimization problem over the space of perturbation distributions. This paper reveals that optimal perturbations either maintain a fixed length or align directionally with the true gradient. While existing research has largely focused on fixed-length perturbations, the potential advantages of directional alignment have been overlooked. To address this gap, this paper delves into the theoretical and empirical properties of the directionally aligned perturbation (DAP) scheme, which adaptively offers higher accuracy along critical directions. Additionally, this paper provides a convergence analysis for stochastic gradient descent using $\delta$-unbiased random perturbations, extending optimal complexity bounds to a wider range of perturbations. Through empirical evaluations on both synthetic problems and practical tasks, we demonstrate that DAPs outperform traditional methods under specific conditions.

### Weaknesses
I don't think the study over problem Eq. (3) is meaningful. 
Under the theory of this paper, the random coordinate is better than Gaussian random vector.
However, just as pointed out in Theorem 1 of  "Fine-tuning language models with just forward passes'', the Gaussian random vector can provide a "Dimension-Free Rate''.
Unfortunately, the random coordinate can not  guarantee this ``Dimension-Free Rate'' even it is good under the thooery of this paper.

The experiments in this paper do not show significant advantages of DAP over other estimations.

### Questions
No.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the two-point zeroth-order gradient estimator, specifically focusing on the problem of identifying the optimal distribution of random perturbations that minimizes the estimator's variance. In Section 1, they briefly introduce the preliminary concepts and raise the motivating questions of the work. They first question whether it would be possible to determine the class of optimal distributions of random perturbations in a zeroth-order estimator to minimize its variance, and provide Theorem 2.2 as the answer. In Theorem 2.2, they introduce two sufficient conditions for the question, which are constant magnitude perturbations and a novel condition called directionally aligned perturbations (DAPs). In Section 4, they take a closer look at DAPs and provide a sampling strategy for practical implementation. Finally, in Section 5, they demonstrate the practical effectiveness of DAPs through two experimental setups with a synthetic example and language model optimization.

### Strengths
To the best of the reviewer's understanding, the core contributions of this paper are two parts:
- This paper formalized the problem of characterizing the class of optimal distributions of random perturbations in a zeroth-order estimator to minimize its variance and provided sufficient conditions.
- Based on the first contribution, they conceptualize the novel condition which they name DAPs, provide a way to use it practically, and demonstrate the effectiveness by experiment.

The reviewer thinks these are meaningful contributions. The paper also shows that the complexity of SGD with two-point gradient estimation achieves the best-known sample complexity when the perturbation distribution $V$ is chosen to achieve the minimum variance.

Also, the reviewer thinks the writing of the paper is overall nice.

### Weaknesses
This can be closer to a question than a weakness, however, the reviewer is confused about the underlying logic and contribution of Section 3. The reviewer may be missing some elementary points, but they are still confused about the sufficient and necessary condition for minimum variance.

- The most confusing point was the relation between the fourth-order moment. In Theorem 2.2, as addressed in the Remark, it seems (2) has minimum variance when equality holds. But is the converse also true? It seems the terms related to the fourth-order moment only appear in the upper bound. If the converse doesn't hold, isn't the finiteness of the fourth-order moment neither a sufficient nor a necessary condition for achieving minimum variance? In this context, is the finiteness of the fourth-order moment an additional assumption (other than minimum variance) imposed to obtain the results in Section 3?
  
- The reviewer thinks the observation about the influence of the fourth-order moment addressed in the Remark of Theorem 3.1 can be meaningful by itself. However, the reviewer thought the main focus of the paper was the conditions for minimum variance and specifically DAPs. Yet, the first half of Section 3 seems to be just a convergence analysis of SGD, with the assumption of the finiteness of the fourth-order moment. According to the authors' explanation, the proof heavily relies on arguments considered in prior works.
  
- In short, what is the role of Theorem 3.1 in the overall context of the paper? It seems Theorem 2.2 is used in the proof; is it crucial? The reviewer thinks it would be better to address a quick overview of Section 3 in the overall context of the paper at the beginning of the section. The reviewer felt lost when first reading Section 3.

### Questions
- Could you provide the explanation related to the reviewer's questions in the weakness?

- Is (a) and (b) in Theorem 2.2 sufficient conditions for achieving the minimum variance, or are they also necessary conditions?

- It seems the authors claim in the remark about (a) of Theorem 3.1 that a small $\delta$ leads to more gradient updates. However, it appears that Theorem 3.1 provides an upper bound result, so it may not serve as logical evidence for your discussion. Or do you have a lower bound result as well?


**Minor questions:**
- Is the definition of parameters in (a) and (b) of Theorem 3.1 the same? Is there a reason you are repeating them?
- Did you try to write $< \infty$ in line 215?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
***Summary:***
In the paper the authors study (sufficient?) conditions on the  distribution of the sampling directions in order to build a two-point finite difference estimator of the gradient that is at the same time unbiased and with minimal variance. Then they state convergence results for SGD using this kind of estimators (in the non-convex and stronly-convex case), showing that they achieve the optimal complexity in terms of dimension. Finally they focus on DAP (directionally aligned perturbation), a new estimator which satisfies unbiasedness and minimal variance. They design an algorithm to implement it and show promising numerical experiments.

### Strengths
***Main comments:***
The review of the literature is complete, the problem is meaningful and relevant, the theoretical results are significant, the proofs are correct. The paper is interesting and well-written, the presentation is both concise and comprehensible.

### Weaknesses
The paper is well written but there are some points with imprecise statements.  See the comments in the Questions box. 

1) The stochastic optimization setting (in $\xi$) is not needed in the first part of the paper but only for SGD (Section 3), and it creates confusion.

2) P3, Theorem 2.2: the inequalities are clear, but it is not clear to me what you can deduce from them, as they give only a lower- and upper-bound on the quantity you want to minimize.

### Questions
1) P3, Theorem 2.2: the inequalities are clear, but it is not clear to me what you can deduce from them, as they give only a lower- and upper-bound on the quantity you want to minimize. Is it true that the variance is minimal if and only if $\rho_V=0$? Or $\rho_V=0$ is only a sufficient condition? Are conditions (a) and (b) sufficient conditions to get $\rho_V=0$? Apparently no, since from (b) you can not get $\rho_V=0$. To me, it is not completely clear the logic of the reasoning neither the statement. This is true especially in connection with the comment on Gaussian Smoothing at P4: does the fact that $\rho_V>0$ imply that Gaussian Smoothing does not achieve minimal variance? From the inequalities of the Theorem you just know that the variance is lower-bounded and upper-bounded by two different quantities...

2) P6, DAP: for the unknown gradient $\nabla f(x)$, you can apply a small batch of perturbations to obtain an estimated gradient. Second level question: with which distribution do you sample $v$ for the estimator of the gradient used in DAP?


More bibliography:

- Cai, Mckenzie, Yin, Zhang: Zeroth-Order Regularized Optimization (ZORO): Approximately Sparse Gradients and Adaptive Sampling; SIAOPT 2022

- Cai, Mckenzie, Yin, Zhang: A One-bit, Comparison-Based Gradient Estimator; ACHA 2022

- Rando, Molinari, Villa, Rosasco: Stochastic Zeroth order Descent with Structured Directions; COAP 2024

- The paper [Rando, Molinari, Villa, Rosasco: An Optimal Structured Zeroth-order Algorithm for Non-smooth Optimization] has been published in NeurIPS 2023

- Akhavan, Chzhen, Pontil, B. Tsybakov: A gradient estimator via L1-randomization for online zero-order optimization with two point feedback; NeurIPS 2022\\

***Minor comments:***

P2: the formula in Contribution 1 is not correct, should be $\nabla f(x;\xi)$ without the $v$

P2: explain why the constraint $\mathbb{E} vv^T = \delta I$ gives the unbiasedness of the gradient approximation (this is true only for $\mu \to 0$); why do you say it is a linear constraint?

P3, L124: first line of the equation is wrong; in the second line, where is the second order term with $M_c(v)$? Explain better the approximation you make...

P4, DPA: highlight that, in the practice of zeroth order optimization, this condition can not be imposed like it is, since $\nabla f(x)$ is not available

P4, L187: $a^Tv=\pm \sqrt{\delta}\|a\|$

P4, L206: $\hat{\nabla}f(x;\xi)$ has not been defined, but only $\hat{\nabla}f(x;\xi, v)$

P4, L210: comment that the quantity $\min_t \|\nabla f(x_t)\|$ is not something you can check in the practice of zeroth order optimization; in particular, you don't know which one is the best iterate accordingly to the criterion $\|\nabla f(x_t)\|$

P4, L210: $f^*$ not defined

P5, L222: $\mathbb{E}_{\xi} f^*_{\xi}$ not defined

P5, L226: say that $c$ is the strong-convexity constant (it appears only in the definition in the appendix)

P5, L238: comment (a) is not clear to me

P5, Corollary 3.2: "If choosing" is not correct (used twice); the bound $\leq \varepsilon$ has constants involved that are omitted

P6, Fig. 1: the caption is not clear to me

### Soundness
3

### Presentation
3

### Contribution
3
