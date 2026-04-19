# Gradient correlation is a key ingredient to accelerate SGD with momentum

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Empirically, it has been observed that adding momentum to Stochastic Gradient Descent (SGD) accelerates the convergence of the algorithm.
However, the literature has been rather pessimistic, even in the case of convex functions, about the possibility of theoretically proving this observation.
We investigate the possibility of obtaining accelerated convergence of the Stochastic Nesterov Accelerated Gradient (SNAG), a momentum-based version of SGD, when minimizing a sum of functions in a convex setting. 
We demonstrate that the average correlation between gradients allows to verify the strong growth condition, which is the key ingredient to obtain acceleration with SNAG.
Numerical experiments, both in linear regression and deep neural network optimization, confirm in practice our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the acceleration of Stochastic Nesterov's Accelerated Gradient (SNAG) method over the SGD algorithm. In particular, the study is based on Vaswani et al, 2019, in which the acceleration is first proved based on the Strong Growth Condition (SGC) of the stocastic gradient. This paper extends the previous paper by showing an accelerated almost sure convergence result for SNAG, and develops condition that lead to a better SGC coefficient. Based on this condition, they show how the SGC coefficient changes as the batch size increases. The paper also verifies the condition using experiments.

### Strengths
1. The paper provides an extension of the original convergence theorem in Vaswani et al, 2019 in the almost sure convergence form, leading to a more comprehensive understanding of the SNAG algorithm.

2. The paper's result covers both convex and strongly convex case. In particular, both the PosCorr and the RACOGA condition can lead to the SGC without assuming the strong convexity.

3. Centered around the SGC, the paper develops conditions that implies the SGC, which allows the paper to investigate the relationship between the batch size and the SGC coefficient.

### Weaknesses
1. Although the almost sure convergence result provided in Theorem 4 deepens our understanding of the SNAG method, I believe that the major focus of this paper is still on how the gradient correlation can lead to a better SGC coefficient, which gives acceleration for SNAG. From this perspective, the result in Theorem 4 seems a bit disjoint from the other sections of the paper.

2. Although the RACOGA condition holds in general with a coefficient of $c \geq -\frac{1}{2}$, it does not seem to be easy to find a tight $c$ for the objectives, as evaluating this lower bound involves analyzing the pairwise inner product between gradients for all choices of the parameters in the parameter space. Furthermore, when $c$ approaches to $-\frac{1}{2}$, the SGC coefficient $\rho = \frac{N}{1 + 2c}$ approaches infinity, leading to a trivial condition.

3. The experimental verification of the paper seems quite weird. It is noticed that, in the linear regression case, the gradient correlation involves both the inner product term and the $\mathbf{a}_i^\top \mathbf{a}_j$, sign of the residual terms $\mathbf{x} ^\top\mathbf{a}_i - b_i$'s. In particular, different signs of the residual terms could lead to completely different lower bound on the gradient correlation. However, it seems that in the experimental design the paper considered only the correlation between the data. Moreover, it may contradict the claim of the paper that RACOGA helps acceleration since in Figure 1.(a) the green curve, with a smaller RACOGA coefficient, led to a faster convergence than the blue curve.

### Questions
1. How is Theorem 2 different from the results in Vaswani et al, 2019? It would be nice if the paper could include a detailed comparison of the two results.

2. In Example 3, the paper demonstrates the benefits of PosCorr condition over the traditional way of verifying the SGC condition by showing that $\rho_K\leq \frac{N}{K}$. Why is $\frac{N}{K} \leq \frac{L_{(K)}}{\mu}$, so that it could be considered an improvement?

3. What is $\lambda$ in Figure 1?

### Soundness
3

### Presentation
2

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
The authors study the convergence of SGD with momentum--Stochastic Nesterov Accelerated Gradient (SNAG) and obtained an improved rate compared to vanilla SGD. More precisely, they consider the strong growth condition that, intuitively, quantifies the amount of noise. Using this definition, they are able to achieve e.g. $o(\frac{1}{n^2})$ convergence rate (SGD has $o(\frac{1}{n})$) for convex functions. For certain objective functions, the authors propose a way to compute the strong growth condition and a new condition RACOGA. In addition, the authors have numerical experiments to verify their results.

### Strengths
1. The paper is well-organized and easy to read. The material in the supplement serves as a good complement to the main paper. Experimental results are presented in a clear way with nice plots and great details.

2. The main result is indeed very interesting to the community and gives some insight into a long-standing question. The theoretical contribution mainly comes from Theorem 4 which provides an almost surely convergence for SNAG showing a speed-up compared to SGD. 

3. By proposing a new characterization of SGC, the authors improved the assumption so that it only depends on the size of the dataset and batch size. Using this, the authors proposed a new condition--RACOGA. 

4. The authors also discuss the relation between batch size and gradient correlations which brings interesting insights into when to use stochastic and non-stochastic versions of these algorithms.

### Weaknesses
1. Proof for theorem 4 heavily relies on an existing result (Sebbouh et al. 2021, theorem 9), which one could argue it weakens the theoretical contributions of this work.

2. I appreciate that the authors made an effort to compare RACOGA with gradient diversity and gradient confusion and agree with the authors that they are not identical, but they do look quite similar.

### Questions
1. It would be nice to have a table that summarizes results from theorem 1-4 (perhaps including results from the literature) so that readers don't have to go back and forth to compare them. 

2. Authors have remark 5 to explain the results from theorem 4 which does help me to understand it. I wonder if there is any intuition about why $\rho_k$ plays a different role in convex vs strongly convex cases. Also, for the strongly convex case, it seems we need less noisy data for SNAG to beat SGD, because we want $\rho_k$ to be small. Am I understanding this correctly? For continuous strongly convex function, there is a unique minimizer, meaning it won't stuck in some local minimizers. How does this fit into this theory?

### Soundness
3

### Presentation
4

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
The paper studies stochastic versions of Nesterov's accelerated gradient descent (NAG). These algorithms have been previously shown to converge at the same accelerated rates as NAG when the stochastic gradient estimates satisfy the so-called strong growth condition (SGC). Specifically for functions satisfying a finite sum structure, this paper finds a sufficient condition (RACOGA) in terms of gradient correlation that implies the strong growth condition, consequently implying that SNAG converges at an accelerated rate in those settings. Numerical experiments are provided to verify the implications of the RACOGA condition on accelerated convergence of stochastic algorithms.

### Strengths
Previous works have shown that stochastic versions of NAG converge at the same accelerated rates when the gradient estimates satisfy the strong growth condition (SGC). While they provide heuristics that suggest that SGC is a reasonable assumption in the context of overparametrized deep learning, it is not always clear when the condition is actually satisfied. This work addresses that gap in the literature. The authors show that for functions of the form $f=\sum_{i=1}^N f_i$, positive gradient correlation (i.e. $\langle f_i, f_j\rangle \geq 0$ for all $i,j$) is sufficient to guarantee the strong growth condition for the gradients. This result also gives a bound for the strong growth parameter ($\rho$) in terms of the batch size, which is important for choosing optimal parameters for the SNAG algorithms. The main contribution of the paper is a gradient correlation condition (RACOGA) which implies SGC for functions with a finite sum structure. This further implies that SNAG converges at an accelerated rate in those settings. I think this is a useful contribution and a step in the direction of better understanding why momentum-based stochastic gradient algorithms perform well in practice. The authors provide numerical experiments to back their claims, which I found interesting and insightful as well.

### Weaknesses
1. Line 70: "However, even the question of the possibility to accelerate with SNAG in the convex setting is not solved yet."
This is either unclear or inaccurate or both. There are several works which address the convergence of accelerated methods in the stochastic setting, both under SGC and with classical Robbins-Monro bounds, at least for smooth objectives. For a rigorous statement, the authors should specify the geometric assumptions, smoothness assumptions, and assumptions on the gradient oracle. Since the authors are aware of previous works on acceleration in convex optimization under SGC noise, it is unclear what meaning is intended.

2. More concerningly, the next sentence says "Finally, note that our core results (Propositions 1-2) do not assume convexity, and thus they could be used in nonconvex settings." Juxtaposed with the previous sentence, it gives the reader the impression that the authors have addressed the question of acceleration for non-convex functions, which is not true. It is true that the conditions PosCorr and RACOGA studied in Propositions 1 and 2 imply SGC even for non-convex functions. But SGC/PosCorr/RACOGA alone is not sufficient for any of the accelerated convergence results provided here or in previous works, some form of convexity is still required. The current phrasing is misleading since, again, it conflates conditions on the noise in the gradient estimates and on the geometry of the objective function, which are in general independent. If there is a relation in the setting the authors consider, they need to explain and emphasize this. I do not see the implication.

3. The authors claim one of their main contributions is "new almost sure convergence results (Theorem 4)". However, almost sure convergence is already covered by corollary 5 in Gupta et al. "Achieving acceleration despite very noisy gradients" arXiv:2302.05515. That paper studies a stochastic version of NAG under a condition similar to SGC. The authors should highlight the differences in their results.

4. The theorem 4 statement suggests that the authors recover a rate almost surely, but in the current presentation, it is unclear what precisely is meant. Even for $O(n^{-2})$: Is there a random variable C such that $f(x_n) - f(x^*) \leq C/n^2$ simultaneously for all $n$ (and almost surely in probability), or does the random constant $C$ depend on $n$? And, what is meant by $o(n^{-2})$? For a machine learning venue, they should state a non-asymptotic quantitative bound. Almost sure convergence is a notion of convergence which is *not* induced by a metric on a space of random variables. As such, there is no immediate way of making sense of the notion that $f(x_n)$ and $f(x^*)$ are $o(n^{-2})$-close in a specific sense. More explanation is needed. The same concern applies to Theorem 2.

5. The title of the paper is "Gradient correlation is **needed** to accelerate SGD with momentum", which makes it sound like gradient correlation is a necessary condition (i.e. if it is not satisfied then SGD with momentum does not converge at an accelerated rate). But I did not see a result proving that in the paper. The results actually claim that it is a sufficient condition. The title does not accurately reflect the main results.

6. $L_{(K)}$ acts as an "effective" Lipschitz-continuity parameter of the gradient, depending on the batch size $K$. The results for SGD (Theorems 1 and 2) are provided in terms of $L_{(K)}$ without assuming SGC but the results for SNAG (Theorems 3 and 4) are provided in terms of the SGC parameter $\rho_K$. Then these two results, derived under different conditions, are compared to conclude that SNAG does not accelerate over SGD unless $\rho_k<\frac{L_{(K)}}{\sqrt{L}}\cdot C$ (where $C$ is a constant that differs in the convex and strongly convex cases). This seems like an unfair and misleading comparison to me. Both, $L_{(K)}$ and $\rho_K$, measure the stochasticity of the gradient estimates but in different ways. The authors demonstrate in Appendix E.2 an example where $L_{(K)}$ is a tighter estimate of the effective Lipschitz constant than $L\rho_k$. That does show that if the smoothness parameter $L_i$ of each summand $f_i$ is known, then using $L_{(K)}$ would allow us to choose a larger step-size for SGD than the one provided by $1/L\rho_k$. However, a fair comparison between SGD and SNAG can only be made if the same assumptions and information are used to calculate the step size, but there are no convergence results available for SNAG that directly make use of $L_{(K)}$. This feels like comparing apples and oranges. If the authors want to argue that you can use a larger step size for SGD than for SNAG, they should justify why Nesterov would blow up with that step size.

### Questions
1. Was the Algorithm 2, in its given form, first introduced by Nesterov (2012) "Efficiency of coordinate descent methods on huge-scale optimization problems."? If yes, the authors should cite that paper. I appreciate Proposition 4 in the appendix showing that the more common two parameter NAG algorithm (Algorithm 8, with $\tau=0$) can be obtained as a special case of this algorithm with a reparametrization.

2. Proposition 2 suggests that RACOGA holding with $c>-0.5$ is sufficient to verify the SGC. But in Figure 1(a), SNAG does not accelerate over SGD despite the RACOGA values being greater than -0.12. Is there an explanation for this apparent discrepancy?

3. Just to confirm, in the experiments, were GD and NAG used with the full batch gradient at each step (e.g. were all of 50k images used for the CIFAR-10 experiment at each training step)? If yes, this might be worth specifying explicitly since most of the times in machine learning experiments, NAG refers to Algorithm 8, even when it is used with mini-batch gradients.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the possibility of obtaining accelerated convergence of the Stochastic Nesterov Accelerated Gradient (SNAG) method. The authors provide a clear proof that the average correlation between gradients allows to verify the strong growth condition, which is essential for achieving accelerated convergence in convex optimization settings. Furthermore, the paper includes comprehensive numerical experiments in both linear regression and deep neural network optimization, empirically validating the theoretical findings. The experimental results are clear and concise. These contributions advance the understanding of momentum-based stochastic optimization techniques and demonstrate the practical effectiveness of SNAG in enhancing convergence rates.

### Strengths
1. Originality:
- Proposes the hypothesis that Stochastic Nesterov Accelerated Gradient (SNAG) can accelerate over Stochastic Gradient Descent (SGD) and proves that this hypothesis is valid when SNAG is under a Strong Growth Condition. 
- Provides new asymptotic almost sure convergence results for SNAG. 
- Gives the new characterization of the SGC constant by using the correlation between gradients.
- Introduces a new condition named Relaxed Averaged COrrelated Gradient Assumption (RACOGA).
2. Quality and clarity:
- Clearly shows when $f$ is convex and $\mu$-strongly convex, it shows the possibility of acceleration of SNAG over SGD is highly dependent on the SGC constant $\rho_K$, where $\rho_K < \sqrt{\frac{L^2_{(K)}}{\mu L}}$.
- Provides clear and explicit steps for proofs.
- The numerical results are readable and show a clear difference in convergence speed among different algorithms.
3. Significance:
- People can get faster and better results by applying the condition proposed in this paper.

### Weaknesses
* The text and formulas are a bit dense; the author can add a table to compare the convergence speed of SGD and SNAG under different conditions.
* The graphs look good. However, that would be better if the author gave more detail about the explanation for the graph, for example, what the "small values" of RACOGA mean on the graph.
* The colors in the right graph for Figure 1(a) are similar, author can use more contrasting colors.

### Questions
* While RACOGA has been demonstrated to facilitate the acceleration of SNAG over SGD in convex and strongly convex functions, how does RACOGA perform in non-convex optimization scenarios, such as those commonly found in deep neural network training? Can RACOGA be effectively applied to these more complex models, or are there additional considerations needed to achieve similar acceleration benefits?
* The paper highlights that large RACOGA values enable the acceleration of SGD with momentum. However, what practical methods or criteria can be used to identify or achieve large RACOGA values in real-world applications?
* How robust is SNAG's performance to variations in RACOGA across different types of datasets and optimization problems?

### Soundness
3

### Presentation
3

### Contribution
3
