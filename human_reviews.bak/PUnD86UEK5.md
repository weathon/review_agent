# Adam Exploits $\ell_\infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6

## Abstract
Adam outperforms SGD when training language models. Yet this advantage is not well-understood theoretically --  previous convergence analysis for Adam and SGD mainly focuses on the number of steps $T$ and is already minimax-optimal in non-convex cases, which are both $\widetilde{O}(T^{-1/4})$. In this work, we argue that the exploitation of nice $\ell_\infty$-geometry is the key advantage of Adam over SGD. More specifically, we give a new convergence analysis for Adam under novel assumptions that loss is smooth under $\ell_\infty$-geometry rather than the more common $\ell_2$-geometry, which yields a much better empirical smoothness constant for GPT-2 and ResNet models. Our experiments confirm that Adam performs much worse when the favorable $\ell_\infty$-geometry is changed while SGD provably remains unaffected. We also extend the convergence analysis to blockwise Adam under novel blockwise smoothness assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors give a unified proof for AdaSGD, Adam, and blockwise Adam under a new Lipschitz and noise assumption. The new assumption does not only consider the gradient Lipschitz of the full gradient but also the Lipschitz property for each block (for SGD, the full parameter, for Adam, each coordinate).  From the theoretical results, the authors find that (1,1)-norm is the critical value for convergence. Experimental results validate the conclusion for Adam-type algorithms.

### Strengths
1.  The authors propose a unified algorithm that contains Adam, AdaSGD, and blockwise Adam.

2. The authors propose a more detailed assumption on gradient Lipschitz to characterize the underlying function carefully. Thus, they can give a tighter bound than giving the overall Lipschitz constant.

3. The authors find that (1,1)-norm of hessian is positively related to the performance of Adam in both theoretical analysis and experimental validation.

### Weaknesses
1.  From my point of view the theorem in section 3.3 has already covered the results in section 3.2, making section 3.2 meaningless.

2. The authors claim that in their proof, we can see the reason that Adam can be better than SGD, while the explanation of the results is only given by $\sup_x ||\nabla^2 L(x)||_{1,1} \leq \sup_x ||\nabla^2 L(x)||_2$. It should have some reasonable examples.

3. In Table 1,  since the convergence of AdaSGD is related to 2-norm instead of (1,1)-norm, why do the authors not report the 2-norm instead of reporting (1,1)-norm twice?

4. There are some typos in the paper: 

e.g., The first line of notation $\sum_{i=1}^d$ instead of $\sum_{i=1^d}$, $\infty$ instead of $infty$. The name of the algorithm is "Adam-mini" instead of $Adamini$. I do not carefully check every detail of the writing, but the authors should go through and correct the typos.

### Questions
1.  Do the results in Section 3.3 cover results in Section 3.2? If so, why should we introduce section 3.2? If not, can you specify the major difference?

2. A simple example: when we optimize a quadratic function with a positive diagonal matrix. In this case, Adam converges much faster than SGD while the (1,1)-norm of the matrix is always larger than the 2-norm of the matrix, because the (1,1)-norm is the sum of diagonal value while the 2-norm is the maximum value of the diagonal entries.  It seems that the result in the simple case contradicts the results provided in section 3.3. 

Can the authors give some concrete examples showing the correctness of the claim?

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
This paper presents a new theoretical analysis of the Adam optimizer, highlighting its advantages over vanilla SGD in training deep neural networks like GPT-2 and ResNets. By relying on loss smoothness under $l\infty$-norm geometry instead of the more common $l2$-norm smoothness assumption, the authors argue that Adam's success over SGD is due to its coordinate-wise adaptivity which allows to exploit some properties of the presented models.The analysis is also extended to blockwise Adam which is a generalized form of Adam.

The proposed theory is empirically verified Adam on rotated and unrotated loss landscapes, verifying the claim that Adam uses non-rotation-invariant features.

### Strengths
1. Even if the paper could be even more polished (see Question 1 for a comprehensive list of needed corrections), the paper is overall very-well written and interesting. I carefully read the main text and the appendices and found the proofs very clear and did not find mathematical errors.

2. The authors introduce what seems to me is a novel framework to better capture Adam's coordinate-wise adaptivity, namely the $l\infty$ geometry that allows them to get tighter convergence bounds than previous work (Défossez et al., 2022) that used $l2$ smoothness of the objective function.

3. The authors demonstrate Adam's sensitivity to rotation. This is used to show that Adam overperforms SGD thanks to non-rotation-invariant properties.

4. Blockwise-Adam is proposed allowing adaptive updates across parameter groups, which could be used for large models.

5. An empirical validation on models like GPT-2 and ResNet-18 is provided, which supports the theoretical findings. The results on rotated Adam are in my opinion particularly interesting and the Appendix D.1 answers a natural question that arises when reading the paper as to how to apply an orthogonal rotation on the parameters on large models.

6. The provided insights could be used to further improve adaptive methods in deep learning and maybe even design adaptive methods "adapted" to specific models as  the discussed coordinate-wise adaptivity is probably heavily linked to specific properties of the studied models.

### Weaknesses
1. The convergence rate improvements seem a bit incremental when compared to (Défossez et al., 2022) and might not translate into practical gains.

2. The use of non-standard $l\infty$ smoothness assumption may limit the generalizability of the proposed results.

3. The empirical analysis of rotation sensitivity is very interesting but a bit limited in scope. The effort of including a ResNet-18 to explore a different kind of architecture is commendable but a more diverse set of architectures would be very interesting to study. It might provide deeper insight into why Adam is impacted by certain rotations and also in which cases do the non-rotation-invariant properties of Adam easily emerge.

4. The blockwise Adam variant is interesting but under-exploited as there is no practical guidance on choosing parameter blocks, even though the related works of Adamini (Zhang et al., 2024b) and Adalayer (Zhao et al., 2024) are mentioned.

5. The high memory consumption of Adam is only mentioned in the introduction. In practice, Adam's higher memory usage when compared to SGD (three times as much) is a real drawback, particularly for large models relying on the Transformer architecture which is studied in this work (GPT-2). One could ask if training a three times bigger model with SGD before reducing its size with common techniques (distillation, pruning) would lead to better results.

6. The focus on the case $\beta_1=0$ seems a bit far-fetched when looking at the results of Table 1 as the final losses obtained are far better with ($\beta_1 = 0.9, \beta_2 = 0.99$)

### Questions
1. A thorough proofread is needed as some typos/redundancies exist even though the paper is quite polished (eg. l.49 "If Adam optimizes much slower more slowly after" -> "If Adam optimizes much slower" or "If Adam optimizes more slowly"; l.102: "for $p \in [1, infty]$ " -> "for $p \in [1, \infty]$"; l. 459: "OpenWebText corups"->"OpenWebText corpus".

2. A naive question one could ask is what would the results of rotated Adam be when compared to Adam on standard CNNs/MLPs and recurrent neural networks? What could be the intuition behind that?

3. Are there any specific patterns in the rotations that worsen performance, or does any random rotation lead to degradation? Could the pattern be extracted from the data/model? A more detailed work on the rotation sensitivity would be very interesting.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies the superior performance of Adam compared to SGD from a theoretical perspective. 

Specifically, existing analysis on Adam is commonly performed under the $\ell_2$ norm. This paper argues that Adam's advantages arise from its sensitivity to the $\ell_\infty$ geometry of the loss landscape, rather than the $\ell_2$ geometry. By introducing $\ell_\infty$ smoothness measures, the authors develop a new theoretical framework to explain Adam's faster convergence rates. The authors further extend this framework to cover blockwise Adam variants such as Adalayer and Adam-mini.

To support the theoretical findings, the paper presents experimental evidence that:

1. Adam performs poorly on rotated loss landscapes, demonstrating its dependence on non-rotation-invariant properties. 

2. Adam outperforms SGD in cases where the $\ell_\infty$ smoothness measure is significantly smaller than the $\ell_2$ smoothness measure, demonstrating that $\ell_\infty$ measures are more adequate to characterize Adam's performance.

### Strengths
1. The paper draws an insight that Adam is permutation-invariant, but not rotation-invariant is crucial, while SGD is rotation-invariant. I believe this property is highly related to the performance difference between Adam and SGD. 

2. Based on $\ell_\infty$ smoothness measures, the paper provides a general framework to analyze Adam and its blockwise variants such as Adam-mini and Adalayer. This contribution is timely, given the increasing interest in blockwise optimization approaches aimed at reducing memory overhead in training large language models. The framework of this paper can potentially guide the design of new Adam variants.

3. Experiments show that different $\ell_\infty$ smoothness measures indeed lead to different performance of Adam. This provide a strong evidence of the theoretical findings.

### Weaknesses
While the paper provides good insights into Adam's sensitivity to $\ell_\infty$ geometry, the proposed theorems using the $ ||\cdot ||_{1,1}$ norm may not fully capture this sensitivity, particularly in explaining the performance gap between SGD and Adam. Two specific concerns are as follows:

- For convex problems with a positive semi-definite Hessian $B$, it holds that:

$$   ||B||_{1,1} \geq \mathrm{trace}(B) \geq || B ||_2 $$

Thus, for a wide class of problems, we have $ \sup_x ||B||2 $ smaller than $\sup_x ||B||_{1,1}$. However, the authors claim ``the latter is typically much smaller when Adam optimizes faster.''  This assertion seems counterintuitive given the inequality. It requires further justification in practical neural network training scenarios.


- In the quadratic loss experiments (Section 4.1), Table 1 shows that Adam still outperforms both SGD and AdaSGD, even for quadratic cases $\sup_x ||\nabla^2 L(x)||_{1,1}$ is larger than $\sup_x ||\nabla^2 L(x)||_2$. This result appears inconsistent with the authors' theoretical argument, as pointed out in the last bullet point.

### Questions
1. Zhang et al. (2024a) also characterize the advantage of Adam to SGD. They have a theory on quadratic problems. Can the authors compare the results to that of Zhang et al. (2024a)?

2. The authors point out that rotating the loss can easily hamper the performance of Adam. However, in practice, Adam performs on par or better than SGD for most neural networks. This means that deep neural networks are not ``randomly rotated''. Why? Can the authors discuss this?

### Soundness
3

### Presentation
3

### Contribution
3
