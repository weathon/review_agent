# $\alpha$-Divergence Loss Function for Neural Density Ratio Estimation

- Decision: Reject
- Scores: 3, 3, 3

## Abstract
Density ratio estimation (DRE) is a fundamental machine learning technique for capturing relationships between two probability distributions. State-of-the-art DRE methods estimate the density ratio using neural networks trained with loss functions derived from variational representations of $f$-divergence.
   However, existing methods face optimization challenges, such as overfitting due to lower-unbounded loss functions, biased mini-batch gradients, vanishing training loss gradients, and high sample requirements for Kullback-Leibler (KL) divergence loss functions.
   To address these issues, we focus on $\alpha$-divergence, which provides a suitable variational representation of $f$-divergence.
   Subsequently, a novel loss function for DRE, the $\alpha$-divergence loss function ($\alpha$-Div), is derived.
      $\alpha$-Div is concise but offers stable and effective optimization for DRE.
   The boundedness of $\alpha$-divergence provides the potential for successful DRE with data exhibiting high KL-divergence.
      Our numerical experiments demonstrate the effectiveness in optimization using $\alpha$-Div.
   However, the experiments also show that the proposed loss function offers no significant advantage over the KL-divergence loss function in terms of RMSE for DRE. This indicates that the accuracy of DRE is
 primarily determined by the amount of KL-divergence in the data and is less dependent on $\alpha$-divergence.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents a new $\alpha$-divergence-based method for density ratio estimation (DRE). Compared to existing DRE methods, the presented method addresses optimization challenges such as overfitting due to lower-unbounded loss functions, vanishing training loss gradients, etc. Numerical simulations demonstrate the effectiveness of the presented method.

### Strengths
The paper is easy to follow.

Stable and accurate DRE is essential for machine learning, especially for GANs. The authors have provided some valuable and original research results.

Conventional DRE methods, as well as the presented $\alpha$-Div, are compared through the lens of lower boundedness and the vanishing gradient problem.

### Weaknesses
The paper should be revised carefully. There are many typing errors. For example, see $\hat E P[R]$ and $\hat E P[\phi]$ in Line 107 and the word "simbole" in the title of Table 2. Also, either $E_{Q}[\phi^{\alpha}]$ in Eq. (6) or $E_{Q}[e^{\alpha T}]$ in Eq. (8) is incorrect.  

Some important statements are questionable. See Questions below.

Experiments are only done on simulations. It would be great if the presented method could be varied in practical scenarios associated with GANs.

### Questions
Line 161 says that $\nabla_{\theta} E_{Q}[log \phi_{\theta}] = E_{Q}[\nabla_{\theta} log \phi_{\theta}]$ does not hold because $E_{Q}$ is an integral. Why? Note that the distribution $Q$ is independent of $log \phi_{\theta}$. Similarly, other statements about the biased gradient problem are questionable.

Line 193 states that "These results demonstrate that major f-divergence loss functions satisfy the conditions for Equation (3)", which is clearly not true. See the KL and Peason $\chi^2$ cases.

Where is the proof of Eq. (4)?

### Soundness
2

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
The authors identify four major issues with existing f-divergence loss functions for DRE. They discuss train-loss hacking, biased gradients, vanishing gradients and sample rates. All of these issues are theoretically analyzed. The authors introduce an $\alpha$-divergence loss function to tackle these issues and show theoretical results for it. Experiments were conducted to investigate the effectiveness of new loss function.

### Strengths
- The authors address important issues in DRE.
- Improving DRE is an important problem that has significant impact on other ML disciplines, e.g., OOD detection and two-sample testing.
- Interesting ideas of how to investigate possible improvements of f-divergences for DRE.

### Weaknesses
+ Related work for DRE should in general be discussed more elaborate and in more detail. Novelty in comparison to these works is unclear.
+ Insufficient comparison to other neural net or kernel-based DRE methods such as [1,2,3], insufficient comparison with other f-divergences mentioned in the submission.
+ Claimed issues of f-divergences loss functions do not seem to be a problem; as performance of $\alpha$-Divergence is worse than KL-divergence for DRE in the third experiment.

[1] Kato, Masahiro, and Takeshi Teshima. "Non-negative bregman divergence minimization for deep direct density ratio estimation." International Conference on Machine Learning. PMLR, 2021.
[2] Rhodes, Benjamin, Kai Xu, and Michael U. Gutmann. "Telescoping density-ratio estimation." Advances in neural information processing systems 33 (2020): 4905-4916.
[3] Menon, Aditya, and Cheng Soon Ong. "Linking losses for density ratio and class-probability estimation." International Conference on Machine Learning. PMLR, 2016.
[4] Gruber, Lukas, Holzleitner, Markus, Lehner, Johannes, Hochreiter, Sepp, and Zellinger, Werner (2024). Overcoming Saturation in Density Ratio Estimation by Iterated Regularization. arXiv preprint arXiv:2402.13891.

### Questions
Experiments:
  + Why are there different datasets for each experiment, how were the parameters of the distributions chosen?
  + How was $\alpha$ chosen in your experiments and how is this hyperparameter selected in general?
  + Did KL-divergence face any of the claimed issues in the third experiment?
  + Did nnBD-LSIF face any of the claimed issues in the second experiment?

Theory:
+ Biased gradients
   * I did not immediately see why the gradient for KL is biased, can you show this formally? Biased gradients are only claimed for KL, what about other f-divergences?
   * Correct me if I'm wrong, but most kernel-based methods for DRE don't suffer from biased gradients.
+ Vanishing gradients
   * I do not really understand (3): Isn't (i) already implying vanishing gradients?
   * For me it is not clear how the limits in Table 2 are computed, a more formal derivation is needed.
   * Why should (3) happen during training? Why do very small/large density ratios imply (3), in particular how does this determine the gradient in parameter space?
   * Please explain the correctness of the two equivalences in line 347 in more detail.
   * Please show the computation of the limits in Table 4 in more detail.
+ Sample rate
   * A bad sample rate is only shown for KL-divergence, what about the other f-divergences? How does this relate to fast sample rates for DRE as discussed in [4]?
   * Describe the implications of Theorem 6.2 for the sample rate in more detail.

### Soundness
2

### Presentation
2

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
This work studies the problem of density ratio estimation (DRE), i.e., estimating the ratio $p(x)/q(x)$ using neural networks, where both $p$ and $q$ are probability densities. The authors argue that the common approach of optimizing a variational formulation of the KL-divergence (whose optimum is $p/q$) has a number of difficulties, namely: overfitting, biased and vanishing gradients, and high effective sample size. The authors then consider an alternative formulation based on Amari's $\alpha$-divergences instead, and develop a parametrization which does not suffer from the same problem as the original one. The author demonstrate these properties empirically, yet they also show that in terms of regression error, there is no significant advantage of using their method instead of the KL-divergence loss.

### Strengths
- **S1.** The problem addressed by the authors is of great interest to a part of the ML community.
- **S2.** The authors gather a good summary of the problems that existing approaches have.
- **S3.** The authors make a great effort into explaining the details of their proposed loss, which looks like a sensible approach.
- **S4.** While I have only skimmed through them, the results in the appendix on real-world data looks good, and I am not sure why it is not in the main manuscript.

### Weaknesses
- **W1.** The presentation could be improved, e.g., there are some clear typos that should be corrected ("hucking" instead of "hacking" in section 6.1, "Peason" in Table 1, "simbole" in line 231).
- **W2.** Some of the "intuitive explanations" of the paper are wrong or are questionable at best. For example:
  1. The problem of train-loss hacking is attributed here to the loss not being lower-boundered, but I'd say that the problem is explained through the fact that there is a finite number of training samples. You could still lower-bound the loss, if the lower-bound is only tight at the limit, use the same argument as in section 4.2.
  2. Line 161 is wrong. Under mild conditions, the linearity of the integral allows you to swap the integral and gradient, as long as $Q$ does not depend on $\theta$, which I understand is the case. 
  3. The arguments in section 4.4 sound cyclic, i.e., arguments (i) and (ii) of lines 181-184 are equivalent to the gradients vanishing.
  4. (more nitpicky, but:) The assumption in the appendix for the theorem 6.1 is local L-continuity, which is worth mentioning in the main paper.
- **W3.** I have found several errors and questionable explanations in the manuscript, which make me worry about whether the results presented in the manuscript hold after fixing them. For example:
  1. Line 269 is false under fairly general assumptions. You can swap gradients and integral as long as Q does not depend on $\theta$ (or if Q is reparametrizable).
  2. The last two columns of Table 2 are all wrong as far as I can see ($E$ should be $\hat E$ and the $Q$ in the last column should be $P$.) 
  3. While not "wrong" per se, as it does not change the arguments of the paper, writing a difference between vector infinities in Table 4 is rather concerning.
  4. I quickly skimmed through the proofs, and I do not understand Eq. 57 at all. Are the authors writing a gradient as the limit of a function as $\psi$ approaches $\theta$? Where is this definition coming from?
- **W4.** The baseline methods to compare their approach with change across experiments, which I cannot fully undestand. Why not using nnBD-LSIF for the experimerints in section 7.3?
- **W5.** This is more of a personal taste, but I really dislike some of the wording chosen in this manuscript, e.g., using "naive" and "merely" as freely as in lines 48-49. It downplays the contribution of previous works in an unfair manner.

### Questions
See above.

### Soundness
1

### Presentation
2

### Contribution
2
