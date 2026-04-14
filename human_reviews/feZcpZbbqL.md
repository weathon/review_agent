# Finding Second-order Stationary Points for Generalized-Smooth Nonconvex Minimax Optimization via Gradient-based Algorithm

- Decision: Reject
- Scores: 6, 6, 3

## Abstract
Nonconvex minimax problems have received intense interest in many machine learning applications such as generative adversarial network, robust optimization and adversarial 
Recently, a variety of minimax optimization algorithms based on Lipschitz smoothness 
for finding first-order or second-order stationary points have been proposed.
However, the standard Lipschitz continuous gradient or Hessian assumption could fail to hold even in some classic minimax problems,
rendering conventional minimax optimization algorithms fail to converge in practice.
To address this challenge, we demonstrate a new gradient-based method for nonconvex-strongly-concave minimax optimization 
under a generalized smoothness assumption.
Motivated by the important application of escaping saddle points, we propose a generalized Hessian smoothness condition, 
under which our gradient-based method can achieve the complexity of $\mathcal{O}(\epsilon^{-1.75}\log n)$ 
to find a second-order stationary point with only gradient calls involved, 
which improves the state-of-the-art complexity results for the nonconvex minimax optimization 
even under standard Lipschitz smoothness condition.
To the best of our knowledge, this is the first work to show convergence 
for finding second-order stationary points on nonconvex minimax optimization with generalized smoothness.
The experimental results on the application of domain adaptation confirm the superiority of our algorithm compared with existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors consider the nonconvex minimax problems. Specifically, the paper proposes a gradient-based accelerated method for finding a second-order stationary point  for the nonconvex-strongly-concave minimax optimization under the so-called generalized smoothness condition. This can be regarded as extension of the existing work Luo et al. (2022), Huang et al. (2022) and Yang et al. (2023) with the Lipschitz Hessian condition.  The theoretical comparison is listed clearly in Table 1.

### Strengths
The paper addresses an important theoretical questions on designing efficient optimization algorithms for finding second-order stationary points of nonconvex minimax problems under general smooth condition.  It extended the existing work to more general cases which can arise in various machine learning problem.

### Weaknesses
1. While the extension to the general smooth case is interesting and important, the design of optimization algorithms relies on  the combination of the Nesterov’s classical Accelerated Gradient Descent  and tricks from Li and Lin (2022). Could the authors comment what are the main novelty on the algorithmic design compared with the existing literature?

2. The proofs seem to heavily depend on the previous work Luo et al. (2022), Huang et al. (2022) and Yang et al. (2023) which assume Lipschitz Hessian condition.  The proof novelty can be incremental (see my questions below). 

3.  The computational gain from the experiment compared with RAHGD and Clip RAHGD seems limited.

### Questions
1. The authors summarised the main challenges in Section 4.1. It would be helpful if  the authors describe in the proof sketch which lemmas are main novel ones. 

2. The formulation of Domain-Adversarial Neural Network given by (9) seems not right.  There are no label for target domain. In the loss function $L_2$ contain $y$ for target domain. Do I miss something here?

### Soundness
3

### Presentation
4

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
This paper provides the first first-order algorithm for non-stochastic generalized-smooth nonconvex-strongly-concave minimax optimization that achieves iteration complexity even better than SOTA for Lipschitz-smooth setting.

### Strengths
As shown in my summary, the claim of both weaker smoothness assumption and stronger result (lower iteration complexity) is significant.

### Weaknesses
Some points need to be clarified as shown below, especially the question 1 below about intuition of algorithm design. A possible proof error may invalidate the main convergence result as shown in question 14 below.

### Questions
(1) In line 5 of Algorithm 1, why can $\zeta=\nabla_x f(\hat{z},\hat{y})$ help get rid of saddle point? At the end of Algorithm 1, why is $\hat{e}$ the NC direction? In the revised paper, could you provide intuitive explanations of these questions or cite the works that contains such explanations? 

(2) In line 88, by "Speically", do you mean "Specially" or "Specifically"? 

(3) In line 103, do you mean to change $\mathcal{O}(\sqrt{\kappa}\epsilon^2)$ to $\mathcal{O}(\sqrt{\kappa}\epsilon^{-2})$?

(4) In line 216, "with Eq. (2)".

(5) In Eq. (5) of Lemma 4.2, can $\max\{u\ge 0|u^2\le 2\mathcal{L}[\Phi(x_0)-\Phi^*]\}$ be simplified to $\sqrt{2\mathcal{L}[\Phi(x_0)-\Phi^*]}$? 

(6) In line 320, "reset to be an epoch" might be read together. Do you mean "we define an epoch to be a round from $k=0$ to the iteration that triggers one of these conditions and resets k to 0"? 

(7) In line 325, does "hypergradient estimation of $\hat{z}$" mean "estimation of the hypergradient $\nabla\Phi(\hat{z})$"? 

(8) In line 331: "an second-order" to "a second-order". 

(9) You could define $\Phi^*$ around its first appearance. 

(10) In Lemma 4.2, $G$ depends on $x_0$ but the conclusions do not depend on $x_0$, why? Can $x_0$ be any arbitrary point? Should we add $\|x'-x\|$ to the end of the equation of item 4, based on Eq. (36)? Adding this also will make the conclusion consistent with the inequality $\|\nabla^2 f(\mathbf{u})-\nabla^2 f(\mathbf{u}^{\prime})\| \leq(\rho_0+\rho_1\|\nabla f(x)\|)\|\mathbf{u}-\mathbf{u}^{\prime}\|$ in Definition 3.2 which contains $\|u'-u\|$. 

(11) $t_0$ and $t_K$ are defined but not used in Lemmas 4.6 and 4.7. 

(12) Right under Eq. (8), what are the roles of $\Phi$ and $\ell$ (for examples, are they predictor and loss function respectively)? In Eq. (9), what are $D_{\mathcal{S}}$, $D_{\mathcal{T}}$ and $h$? 

(13) Some other hyperparameter choices like $B,r,K,\mathcal{J}$ could be revealed in your experiment for reproducibility. 

(14) Possible error in Proof of Lemma 4.6: Should the second $\tilde{x} _ {t_0}$ be $\tilde{x} _ {t_0-1}$? Since their distance is not surely to be smaller than $\mathcal{O}(\epsilon)$, I am not sure if Lemma 4.6 holds.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper studied the nonconvex-strongly-convex minimax under the generalized-smooth setting, and proposed a ANCGDA method to find a second-order stationary point in nonconvex-strongly-concave minimax optimization with generalized smoothness. It provided the convergence analysis of the proposed method under some conditions. It also provided some numerical experimental results on the proposed method.

### Strengths
This paper proposed a ANCGDA method to find a second-order stationary point in nonconvex-strongly-concave minimax optimization with generalized smoothness. It provided the convergence analysis of the proposed method under some conditions. It also provided some numerical experimental results on the proposed method.

### Weaknesses
From algorithm 1 of this paper, the proposed algorithm basically extends the existing methods in [1] to minimax optimization. The novelty of the proposed algorithm is limited. Meanwhile, the convergence analysis mainly follows the exiting results.


[1] Huan Li and Zhouchen Lin. Restarted nonconvex accelerated gradient descent: No more polylogarithmic factor in the complexity. In International Conference on Machine Learning, pp. 12901–12916. PMLR, 2022.

The numerical experiments in the paper are limited. The authors should add some numerical experiments such as GAN and multi-agent reinforcement learning.

### Questions
From Figure 1, what does the horizontal axis represent?

Under the Definition 3.4, the generalized Hessian continuous condition may be generate the smoothness condition ? If not, please give some negative examples.

What is the full name of ANCGDA  in the paper?

### Soundness
2

### Presentation
2

### Contribution
2
