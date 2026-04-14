# Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 6

## Abstract
We study the implicit bias of the family of steepest descent algorithms with infinitesimal learning rate, including gradient descent, sign gradient descent and coordinate descent, in deep homogeneous neural networks. We prove that an algorithm-dependent geometric margin increases during training and characterize the late-stage bias of the algorithms. In particular, we define a generalized notion of stationarity for optimization problems and show that the algorithms progressively reduce a (generalized) Bregman divergence, which quantifies proximity to such stationary points of a margin-maximization problem. We then experimentally zoom into the trajectories of neural networks optimized with various steepest descent algorithms, highlighting connections to the implicit bias of Adam.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the implicit bias of the steepest flow with respect to a general norm in deep homogeneous neural networks. Theoretically, they present two major results. 1) The soft margin will increase 2) The limit points of the parameter directions are general KKT points of the margin maximization optimization problem under some assumption of the norm. They also show some experimental results concerning the trajectories of different steepest descent algorithms and the connection to the implicit bias of Adam.

### Strengths
This paper extends the result in [Lyu & Li 2020] to steepest flow for general norms with the lightest extra assumption on the norm. While previous papers focused on gradient descent and other variants of gradient descent, this paper gives a general result for any steepest flow.
The claimed results seem sound and the paper is nicely written.

### Weaknesses
The paper asserts that it presents results for the steepest descent; however, the theoretical results in Section 3 are limited to steepest flow. According to Lyu & Li (2020), additional work is required to extend the findings from flow to descent.

Regarding the general loss function, the paper demonstrates that the soft margin is monotonically increasing but fails to address the convergence of KKT points.

Moreover, the paper overlooks several relevant references, including:

1. Chatterji, N.S., Long, P.M., & Bartlett, P. (2021). "When does gradient descent with logistic loss interpolate using deep networks with smoothed ReLU activations?"

2. Cai, Y., Wu, J., Mei, S., Lindsey, M., & Bartlett, P.L. (2024). "Large Stepsize Gradient Descent for Non-Homogeneous Two-Layer Networks: Margin Improvement and Fast Optimization."

3. Nacson, M.S., Srebro, N., & Soudry, D. (2019, April). "Stochastic gradient descent on separable data: Exact convergence with a fixed learning rate."

4. Ji, Z., Srebro, N., & Telgarsky, M. (2021, July). "Fast margin maximization via dual acceleration."

### Questions
- Can the authors extend the existing results on steepest flow to steepest descent? If not, it would be more accurate to clarify that the current theoretical results apply solely to steepest flow.

- Is it feasible to demonstrate KKT point convergence for the general loss function?

- In Lyu & Li (2020), tight convergence rates for both the loss function and parameter norms were established. Are there any mathematical challenges in achieving similar results here? Additionally, they showed that the soft margin approximates the normalized margin. Could you provide a similar analysis in this context?

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
2

### Summary
This work studies how the prediction margin changes as the training proceeds in a homogeneous network

### Strengths
The problem of prediction margin is important and of fundamental importance

### Weaknesses
I am not sure if I understand the meaning of the result. Also, I find the setting of the theory too restrictive

1. I do not understand Figure 1-right. What does it imply? How does this relate to the theory?
2. The theory only applies to the "exponential loss," which does not make much sense to me.  

Lastly, I feel I should comment that I do not have the related background knowledge to assess the technical advancement made by this work

### Questions
How close is the steepest descent to actual GD? Does the theory hold for GD?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the implicit bias of  steepest flow with general norm in homogeneous neural networks. It shows that the soft-margin is monotonically increasing after the loss is below a threshold. Furthermore, it shows that the steepest flow converges to the generalized KKT point. Finally, experiments demonstrate the similarity of Adam algorithm and Signed-steepest descent algorithm, in terms of implicit bias.

### Strengths
1. The convergence of the general steepest descent to the general maximum margin solution is novel. 
2. The connection between the implicit bias of Signed steepest descent with Adam is interesting.

### Weaknesses
The set of results and proofs seem to be a straightforward generalization of [Lyu and Li, 2019]. The technical contribution is not strong.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper studies the implicit bias of steepest descent methods with infinitesimal step size (i.e., steepest flow) and presents an interesting generalization of Lyu & Li (2020)'s result on gradient flow (steepest flow with L2 norm). Similar to gradient flow, it is shown that steepest flow has a bias towards KKT solutions to a margin maximization problem, but the margin being maximized here is normalized properly according to the norm used in steepest flow, which may not be the L2 norm in general.

### Strengths
1. Understanding the implicit bias of optimization methods is an important problem in deep learning theory, and the paper takes a margin-based view of implicit bias and sheds light on how different methods may optimize different notions of margin for deep models.
2. It is a bit surprising that such an implicit bias of steepest descent can be rigorously proved in the general case. Previously, it is known that this can be proved for linear models, and for deep homogeneous networks, Lyu & Li (2020) generalized the result for GD. It is reasonable to imagine that one could somehow generalize Lyu & Li (2020)'s result to steepest gradient, but their proof apparently has a lot of dependence on the L2 geometry induced by GD. After having a careful read of the proof (though I skipped a few details), I believe the authors found the right way to do this generalization and worked out all the technical issues, including those related to non-smoothness.
3. The paper is well-written and easy to follow.
4. Experiments in simple settings validated the theoretical results.

### Weaknesses
1. While I really appreciate that the authors worked out every technical detail to generalize the proof of Lyu & Li (2020), I also noted that the overall proof outline has not changed much from Lyu & Li (2020), which means this paper does not actually bring a brand new high-level proof idea. This is reasonable since any implicit bias analysis of steepest flow automatically implies an analysis of gradient flow, but the fact that the paper fails to prove the real KKT conditions and only manage to prove a weaker notion (generalized KKT) suggests that a perfect generalization from gradient flow to steepest flow may indeed need a completely new proof strategy.
2. Same as all previous works studying the max-margin bias, this paper only analyzes the late phase, where the training loss is already very small. While the asymptotic analysis of implicit bias is interesting, this also makes the phenomenon less relevant to the practice.

Minor issue: there are large blanks on some pages in the appendix.

### Questions
This paper only proves the convergence to generalized KKT solutions for sign gradient flow, which is an important case of steepest descent. I wonder if the authors have any thoughts on whether failing to prove KKT in this case is an artifact due to the proof techniques or if some tricky hard cases actually exist.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates the implicit bias of the family of steepest descent algorithms, including gradient descent, sign gradient descent, and coordinate descent, for deep homogeneous neural networks. The authors aim to understand how these optimization algorithms implicitly select solutions from the perspective of margin maximization. The main contributions of the paper are:

1. A rigorous analysis of the implicit bias of steepest descent algorithms in non-linear, homogeneous neural networks, showing that an algorithm-dependent geometric margin increases during training, under the same set of assumptions made in Lyu&Li.
2. The definition and characterization of a generalized notion of stationarity for optimization problems, demonstrating that these algorithms reduce a generalized Bregman divergence, which quantifies proximity to stationary points of a margin-maximization problem.
3. Experimental validation of the theoretical findings by training neural networks with various steepest descent algorithms, highlighting connections to the implicit bias of Adam.


I think the class of the optimization problems studied in this paper is general and the theoretical results of the paper are quite clean. Although the main proofs follow the framework developed by Lyu and Li, the proofs are nontrivial (I read most of the proofs in the main text and a few in the appendix). Overall, I am leaning towards accepting the paper. With additional clarifications of the relation with prior work, this paper would be a valuable addition to the literature on deep learning theory.

### Strengths
**Strong Points:**

1. The paper provides rigorous theoretical analysis of the implicit bias of steepest descent algorithms. The results and proofs extend the l2 norm case (gradient descent) by Lyu&Li to general norms. The theoretical results are clean and some of the extension are nontrivial. 

2. The class of optimization algorithms studied in this paper is very general and include several important optimization algorithms as special cases.

3. The paper connects theoretical insights to practical optimization methods like Adam, which is widely used in the deep learning community. This connection may help to understand the benefit of Adam.

### Weaknesses
**Weak Points:**
The authors briefly discuss some prior work on implicit bias of optimization algorithms other than GD in the related work section. But I found the inclusion of related work and the discussion here is somewhat insufficient. For example, Gunasckar et al. 18 already studied the implicit bias of several class of optimization algorithms. However, the citation here is very superficial and it is unclear to me how the results in this paper supercede (or compared with) their results. It is also unclear to me how the results by Wang et al. on Adam is related to the results of this paper. If we simplify Adam to signGD, it seems that the results in Wang et al. is inconsistent with this paper? I think more detailed discussion is necessary.

Moreover, the following papers also obtain implicit bias results towards L_p margin with p different from 2 (for special models). Is there any concrete relation between these results and the results in this paper.  
Kernel and Rich Regimes in Overparametrized Models (already cited but not discussed in details)
Implicit Bias in Deep Linear Classification: Initialization Scale vs Training Accuracy

### Questions
minor comment: 
1. The sentence before Theorem 3.1. The proof is quite similar to that in Lyu&Li. It seems to me that this paper used Cauchy-shwartz but Lyu&Li explicitly used the Pythagorean theorem (which only holds in inner prod space but not general normed space). If I am correct, then there is not much difference.

2. I think it would be better to discuss briefly the relation between D-generalized KKT and ordinary KKT between Theorem 3.8 and Cor 3.8.1. and why not to prove something like Cor 3.8.1. directly.

### Soundness
4

### Presentation
3

### Contribution
3
