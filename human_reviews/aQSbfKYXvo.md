# The Effect of Personalization in FedProx: A Fine-grained Analysis on Statistical Accuracy and Communication Efficiency

- Avg Score: 5.20
- Decision: Reject
- Scores: 6, 8, 6, 5, 1

## Abstract
FedProx is a simple yet effective federated learning method that enables model personalization via regularization. Despite remarkable success in practice, a rigorous analysis of how such a regularization provably improves the statistical accuracy of each client's local model hasn't been fully established. Setting the regularization strength heuristically presents a risk, as an inappropriate choice may even degrade accuracy. This work fills in the gap by analyzing the effect of regularization on statistical accuracy, thereby providing a theoretical guideline for setting the regularization strength for achieving personalization. We prove that by adaptively choosing the regularization strength under different statistical heterogeneity, FedProx can consistently outperform pure local training and achieve a minimax-optimal statistical rate. In addition, to shed light on resource allocation, we design an algorithm, provably showing that stronger personalization reduces communication complexity without increasing the computation cost overhead. Finally, our theory is validated on both synthetic and real-world datasets and its generalizability is verified in a non-convex setting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper has studied how personalization in FedProx impacts the statistical error of FedProx and  the trade-off between local computation cost and communication cost. First,  this paper provided sharper statistical error bounds by adaptively tuning the regularization parameter $\lambda$ in FedProx based on the degree of statistical heterogenelity parameter $R$. Then this paper provided a new algorithm to solve FedProx and analyzed the rate of convergence on both local models and global model, showing personalization can increase the computational cost per iteration but also reduce the number of communications. Finally, numerical experiments have been conducted to validate the theoretical findings.

### Strengths
This paper studied a fundamental problem that how statistical heterogenelity impacts the process of personalized federated optimization,
gave new statistical error bounds under different levels of statistical heterogenelity,
and established a new algorithm that shows an interesting trade-off between local computation cost and communication cost.

### Weaknesses
(1) The authors claimed that their statistical error bounds improve previous results which I do not fully agree with. The main weaknesses of this paper include the unfair comparisons over previous results and the limited improvement in statistical errors over previous results. There are three reasons.

(i) The statistical error bounds in this paper omitted the optimization error of algorithm solving FedProx.
In other words, this paper just analyzed the statistical error of ERM estimators.
However, previous results [1] incorporate the optimization error of algorithm solving FedProx,
that is, the statistical error of approximate ERM estimators.

(ii) The statistical error defined in this paper is slightly different from previous work [1].
While the two definitions are closely related, there exists a gap represented by a problem-dependent constant.

(iii) Except for problem-dpenednt constants,  this paper only improves previous results [1] by a factor of $O(1/\sqrt{m})$ 
in the case of $R\leq \frac{1}{m\sqrt{n}}$. 

(2) This paper aims to provide contributions in theory. In addition to the theoretical results, 
it is necessary to explain the technical challenges on the theoretical analyses and the corresponding technical controbutions, both of which are currently unclear to me.
Besides, there are many typos in the current version.

(i) In Eq. (29), (30), (32) and so on, the expectation operator $\mathbb{E}[\cdot]$ are omitted.

(ii) In Eq. (33), (35), (36), I think $\tilde{w}^{(i)}$ should be corrected as $w^{(i)}_*$.

References

[1] Chen et al. Minimax estimation for personalized federated learning: An alternative between fedavg and local training? JMLR, 2023.

### Questions
Could the authors elucidate the technical challenges and contributions regarding the theoretical analyses?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper investigates the effect of regularization in FedProx, providing a theoretical guideline for setting the regularization parameter for achieving personalization. Experiments conducted on synthetic and real datasets validate the theoretical analyses.

### Strengths
1. It is intuitive that GlobalTrain is more suitable for i.i.d. data, while LocalTrain (and personalized FL) is more appropriate for non-i.i.d. data, and FedProx represents a hybrid approach. This paper substantiates this intuition and formally provides theoretical guidelines to demonstrate how the regularization term in FedProx affects model convergence.
2. In comparison to similar works, the theoretical results in this paper are more precise with fewer assumptions. Furthermore, the experimental results perfectly validate the theoretical analyses.

### Weaknesses
1. Figures in the paper could be larger for better readability.

### Questions
N/A

### Soundness
4

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
3

### Summary
This paper proposes a theoretical guideline for adaptively setting the strength of the regularization under different levels of statistical heterogeneity to achieve personalization in FL. The authors demonstrate that by adaptive tuning the regularization to strengthen the personalization, communication complexity can be reduced without increasing the computation cost overhead.

### Strengths
1. This paper addresses a practical issue that commonly exists in the FL framework the statistical heterogeneity is prevalent. The motivation for optimizing regularization strength in Fed Prox is compelling. 
2. The paper is well-written and has a logical structure. The mathematical derivations are rigorous and precise. The authors also provide code to facilitate reproducibility.

### Weaknesses
1. The experiments in this paper use only one real dataset, MNIST, which may not comprehensively illustrate the effect of personalization on statistical accuracy under various levels of heterogeneity. In addition, the model used in the non-convex case is relatively simple. Including more datasets and exploring more complex models could better demonstrate the main contribution of this paper.
2. The experimental results primarily focus on the model's error. However, it would also be beneficial to include accuracy to show the model performance with different lambdas. 
3. The descipriotns and analyses of the experiments lack clarity and it would be better to improve the content. For example, L. 492-493 references Fig.4 in the appendix as displaying error-related results, but Fig. 4 (a) actually shows accuracy variation. Improving these experimental explanations would make the results more interpretable for readers. 
4. I am curious whether this adaptive regularization approach can be extended to scenarios where clients are sampled in each communication round.

### Questions
Refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The FedProx algorithm enables personalization through penalization on the distances between local models and the global model. 
This paper theoretically analyzes the regularization strength influence the statistical rate and show that FedProx can consistently outperform pure local training and achieve a nearly minimax-optimal rate by properly choosing the penalization quantity based on the relative size of heterogeneity $R$ and local sample size $n$.  Experiments on synthetic and MNIST datasets are given.

### Strengths
- The rates are reasonable and established under common assumptions in federated learning.
- The paper is written in a way that is easy to follow.

### Weaknesses
- It is claimed that FedProx achieves a nearly minimax-optimal statistical accuracy but it is not compared to other works like Scaffold where heterogeneity $R$ does not appear in the upper bound.

- Empirical results are only given for synthetic datasets and MNIST on logistic regression. Other personalization methods are not compared. 

- The adaptive strategy of choosing of $\lambda$ relies on unknown quantity and therefore limits its contribution.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The paper studies the FedProx learning rule for multi-distribution learning. It establishes statistical rates as a function of first-order heterogeneity, showing how the regularization (to form consensus) should be set in a heterogeneity-aware manner. The authors also provide an algorithm to solve the empirical FedProx objective, providing its convergence analysis in the noise-less setting. Some experiments are also provided to verify the theory's predictions.

### Strengths
The paper identifies an interesting problem in federated learning: the issue of personalizing more v/s less as a function of data heterogeneity.

### Weaknesses
**Incorrect units**: Theorem 1 seems wrong, as stated. In particular, the units of the convergence rate are incorrect, i.e., the r.h.s. does not have the units of the squared parameters. For instance, if we blow up both the parameter space and function space by the same factor but very large factor, then r.h.s will grow much more quickly because of the $C_2$ and $C_4$ terms that will blow up due to the additional $\sqrt{1+2\mu}$ factor as opposed to say $\rho/\mu$ which will grow with the same rate as the l.h.s. To put it succinctly, the result is not scale invariant, and the authors are likely either hiding some boundedness assumptions or omitting important problem parameters in the Theorem statement. Even the choice of $\lambda$ is incorrect. $\lambda$ should have the same units as strong convexity, but it does not. It is unclear what the indicator function does in the choice of $\lambda$ because comparing $R$ with a unitless/scaleless quantity such as $1/\sqrt{n}$ makes no sense. For another sanity check, note that when each client's objective is the mean estimation objective (i.e., mean squared error of the estimator), $\mu=1$ and only dependence on the right in (4) should be $\rho^2$ and not $\rho^4$. I request the authors to correct this. 

In light of the above issue, it is hard for me to interpret the convergence result, which is the main result of this paper, but I will try to disambiguate what the correct convergence rate should have been. 

**Vacuous guarantees**: Let's consider (5) for the particular case of mean estimation so that $\mu=1$. I claim that the upper bound in (5) is worse than the best of two trivial algorithms: (i) estimate the mean using only the samples on machine $i$ (i.e., pure local training), (ii) estimate the mean using all the samples across all the machines (i.e., consensus optimization). To see this, consider the first and third terms of (5), which, when combined, simplify to:$$\min(\frac{\rho^2}{mn} + R^2, \frac{\rho^2}{n}),$$
which is precisely the best of simple algorithms (i) and (ii). Since (5) has another term in the upper bound, the guarantee of Theorem 1 can not beat this trivial baseline and is thus vacuous. Furthermore, the best of (i). (ii) is extremely simple to implement: do the usual federated learning and pure local training side by side by splitting the samples in half. Then, compare these two solutions for each machine and pick the best one. 

**Noiseless setting**: It is tough to justify the noiseless setting of the optimization result because, in the absence of noise, there is no benefit of collaboration whatsoever, and even the Theorem statement in the paper suggests setting $\lambda=0$ to do purely local training. Thus, studying optimization with a complex procedure in such a setting is a moot point. The authors brush this away, speaking as if dealing with noise were trivial and techniques like variance reduction can be added without much effort to their algorithm. That does not seem to be the case.  

In light of the above issues, I strongly recommend rejecting the work, as it does not meet the rigor requirements for publication at ICLR.

### Questions
1. Why is $w_\star^{(g)}$ the appropriate true global solution in the heterogeneous setting? It seems the authors are conflating between the average of optimas and the optima of the average function. In federated learning, the aim is usually to recover the latter, and both these solutions can be further apart. To see the difference between these two solutions, see the section on fixed point discrepancy in [Patel et al.](https://arxiv.org/abs/2405.11667). 

2. It is unclear what is $\tilde w^{(i)}$; what do the authors mean by local models? Do they use this to refer to each machine's ERM solutions, which are unique given the strong convexity?

### Soundness
1

### Presentation
1

### Contribution
1
