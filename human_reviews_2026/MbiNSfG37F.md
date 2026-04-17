# Fast Rerandomization for Balancing Covariate in Randomized Experiments: A Metropolis–Hastings Framework

- Decision: Reject
- Scores: 6, 4, 6, 8, 2

## Abstract
Balancing covariates is critical for credible and efficient randomized experiments. Rerandomization addresses this by repeatedly generating treatment assignments until covariate balance meets a prespecified threshold. By shrinking this threshold, it can achieve arbitrarily strong balance, with established results guaranteeing optimal estimation and valid inference in both finite-sample and asymptotic settings across diverse complex experimental settings. Despite its rigorous theoretical foundations, practical use is limited by the extreme inefficiency of rejection sampling, which becomes prohibitively slow under small thresholds and often forces practitioners to adopt suboptimal settings, leading to degraded performance. Existing work focusing on acceleration typically fail to maintain the uniformity over the acceptable assignment space, thus losing the theoretical grounds of classical rerandomization. Building upon a Metropolis-Hastings framework, we address this challenge by introducing an additional sampling-importance resampling step, which restores uniformity and preserves statistical guarantees. Our proposed algorithm, PSRSRR, achieves speedups ranging from 10 to 10,000 times while maintaining exact and asymptotic validity, as demonstrated by simulations and two real-data applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses a well-known limitation of rerandomization in experimental design: the computational inefficiency of classical rejection sampling under small balance thresholds. The authors propose a Metropolis–Hastings–based framework to accelerate rerandomization while retaining theoretical guarantees of uniformity over acceptable treatment assignments.

### Strengths
1. Clear problem motivation: The paper identifies the genuine computational bottleneck of rerandomization.
2. Strong theoretical grounding: The paper rigorously proves stationarity and uniformity.
3. The method proposed looks very clear and practical for me.
4. The paper is well writtern and easy to follow.

### Weaknesses
To be clear, I am not an expert in this field. I generally enjoy reading the paper and am not fully confident in providing any weakness. Below are just what in my mind, which can be wrong.
 
1. Technical contribution: Although there are some interesting ideas in designing the algorithm, the analysis and some ideas themselves (including the rejection sampling) seem to be standard. The main contribution seems to be applying them to the rerandomization context.
2. More general discussion on how to set the temperature can be helpful, together with some sensitivity analysis of the parameter $T$.
3. The paper is quite statistical-methodological with limited ML relevance. I think it would be beneficial to frame it for a more general ICLR audience.

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PSRSRR, a Metropolis–Hastings–based framework for ensuring uniformity when accelerating rerandomization in randomized experiments. Classical rerandomization acceleration method can ensure covariate balance but does not ensure uniformity over the feasible region, making classical theoretical inference results for rerandomization inapplicable. The authors model pair-switching of treatment assignments as a Markov chain, prove its stationary distribution, and then apply a sampling-importance resampling step to restore uniformity. A practical “early-stopping” version is introduced to trade exactness for speed. Simulation and real-data experiments suggest PSRSRR achieves similar statistical validity to classical rerandomization while being substantially faster.

The paper is methodologically careful and builds on solid theoretical ground. The derivations of the stationary distribution and uniformity correction are mathematically correct and appropriately referenced. The experimental setup is standard and sufficient to demonstrate feasibility.

However, the claims of theoretical novelty are somewhat overstated—the work mainly adapts standard MCMC and rejection-sampling concepts to rerandomization. The early-stopping heuristic, which drives most of the practical efficiency, lacks a rigorous analysis of its bias relative to the true uniform distribution. 

The paper is clearly written, with a logical flow and helpful diagrams. The algorithms are well-specified. References are thorough and well-chosen.

### Strengths
-	Provides a clean Markov-chain interpretation of pair-switching rerandomization as well as additional rejection sampling with clear theoretical justification.
-	Demonstrates solid empirical performance: substantial computational gains with minimal loss of balance quality.
-	Algorithm is clear and easy to implement.

### Weaknesses
This paper proposes an algorithm that ensures uniformity when conducting pair-switching rerandomization, which is Algorithm 2 in the paper. However, Algorithm 2 introduces additional rejection sampling step, making the procedure even slower. Thus, the author proposes a heuristic approach that does rejection at every step, which is the PSRSRR in the paper. The main problem is that the heuristic PSRSRR lacks theoretical guarantee on uniformity with only empirical evidence shows improvement on uniformity over the PSRR algorithm.

For this reason, I am wondering if there’s a theoretically rigorous way for conducting inference given the PSRSRR sampling scheme. For example, whether the inference procedure in (Zhu and Liu, 2022) is applicable here. On the other side, the author could also give some theoretical guarantee on the deviation from uniformity from the proposed procedure and explain how that deviation will influence the inference procedure.

To summarize:
-	Limited novelty: core ideas (pair-switching + MCMC + resampling) are straightforward extensions of prior PSRR and general MCMC principles. 
-	Heuristic implementation lacks theoretical error bounds or diagnostic tools to quantify deviation from uniformity.  Only numerical evidence of improvement is shown.

### Questions
See above.

### Soundness
3

### Presentation
4

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
Rerandomization aims at producing a random treatment assignment that yields treatment and control groups with similar covariate means, measured by the Mahalanobis distance. For some theoretical results and methods to apply, it is useful to sample a treatment assignment uniformly from the set of all assignments whose Mahalanobis distance between groups does not exceed a small threshold $a > 0$. However, existing rerandomization methods either fail to guarantee uniformity or lack computational efficiency.

First, this paper introduces a rerandomization algorithm called Truncated Pair-Switching (Algorithm 1). The authors derive the explicit form of the limiting and stationary distribution (Theorem 1) of the Markov chain generated by this algorithm.

Then, they propose Algorithm 2, which samples exactly from the desired uniform distribution (Theorem 2) by applying rejection sampling to Algorithm 1, where the acceptance probability is inversely proportional to the limiting probability obtained in Theorem 1, truncated to the target set of assignments.

As a computationally cheaper heuristic, the authors integrate Algorithms 1 and 2 into a single loop rather than nesting them, forming Algorithm 3 (Pair-Switching Rejection Sampling Rerandomization; PSRSRR).

Experiments on both synthetic and real datasets compare the proposed method (PSRSRR) with existing ones in terms of mean squared error (MSE), confidence interval (CI) length, coverage rate, statistical power, type I error, and computational time. The proposed method consistently performs among the best in both statistical accuracy and computational efficiency, with especially strong advantages in higher-dimensional settings.
The paper also confirm that the PSRSRR produces uniform samples despite being a heuristic.

### Strengths
- The paper provides useful theoretical guarantees (Theorem 1 and Theorem 2), which seems to be a novel and good contribution to the literature.

- The proofs of Theorems 1 and 2 seem correct to me (except the part using Lemma 4, for which the notation is not clear).

- The heuristic version of the algorithm (Algorithm 3) is computationally efficient (Figure 2), while being superior or comparable to previous methods in terms of statistical performance (Figure 1).

- The proofs are also written in an accessible way.

### Weaknesses
- Algorithms 1 and 2 are similar to PSRR (Zhu and Liu, 2022) especially when we set $N = 1$, but the paper does not clearly mention this fact.

- I appreciate that the authors' effort to make the proof self-contained by restating the theorem from Harchol-Balter (2024) as Lemma 5. However, there are undefined symbols and I do not understand what it states.

- There are a few unclear points in the proof of Theorem 1, which is making it difficult to understand. (See the Questions.)

- PSRR is slightly faster for larger $p$ in Figure 2.

- There seems to be no numerical results for Algorithm 2.

- It is difficult to see if the proposed method is better than GSW. GSW seems advantageous in the coverage rate while PSRSRR is better in CI length, but these criteria might be in a trade-off relationship.

### Questions
### Major comments
- Please clearly explain the contribution relative to Zhu and Liu (2021).

- Please rephrase Lemma 5 using symbols defined in this paper.

- In Theorem 2 and the text below it, shouldn't the part of PSRSRR be rather Algorithm 2? I thought PSRSRR refers to Algorithm 3, which has no theoretical guarantee.

-----
### Minor comments
- If we can interpret Algorithm 1 as a Metropolis-Hastings method, I thought there is a proposal distribution. What is the proposal distribution?

- l.64: "robust alternative"---alternative to which method?

- l.122: what is $Cov(...)$ exactly?

- In Corollary 3, what is "the projection of potential outcomes on covariates" exactly?

- In Corollary 3, $S_X^2$ --> $S_{XX}$?

- In A.2 PROOF OF THEOREM 2, $g$ might not be defined.

- Are there any numerical results for Algorithm 2? In particular, it would be interesting to compare its running times with those of Algorithm 3.

- ll.321-323: "We provide strong indirect evidence for the near-uniformity"---Where can we find this result?

- Around l.698 in the proof of Theorem 1, we might need $n_c + n_t \ge 3$ to be able to find such three indices $p, s, q$.

### Soundness
3

### Presentation
2

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
The paper proposed a new sampling method to generate a uniform distribution on all acceptable assignments in the context of randomized experiments. It generalizes existing methods based on pair-switching to a two-step reject sampling algorithm with uniformity guarantee and further heuristically combines the two steps into a single-chain continuously reject sampling algorithm. The proposed method is proven to be efficient and accurate with high sampling quality through simulation studies.

### Strengths
- **originality**: the proposed method is novel and effective. It adds to the existing toolbox of randomization algorithms. 
- **quality**: the paper includes theoretical results along with extensive solid simulations. 
- **clarity**: the paper is clearly drafted. 
- **significance**: The efficient algorithm to generate high-quality random assignments to experimental units is an essential component in experimental design.

### Weaknesses
Although the paper provides a theoretical guarantee for the two-stage reject sampling algorithm, the practical algorithm is a heuristic revision of the theoretically justified one. The author empirically investigated the impact of the ad-hoc adaption, but there is still a theoretical gap. It would be perfect if any theoretical guarantee for the single-chain algorithm could be developed.

### Questions
- The proof of Theorem 1 could be significantly simplified by (i) verify the proposed distribution is stationary (ii) irreducibility + aperiodicity of a finite MC implies uniqueness, existence and convergence of the stationary distribution. 
- In figure 2, it looks like PSRSRR performs bad on small n large p cases. What could be the reasons?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers a method for randomized experiments under a Metropolis-Hastings framework. The proposed algorithm efficiently balances covariates under experiments. The algorithm's performance is examined by simulations and two real-data applications.

### Strengths
This paper uses a well-known algorithm for covariate-balanced experiments and the method is immediately applicable.

### Weaknesses
- While the pair-switching strategy considered in the paper is noted as a proposal to reduce the computational cost, its primary limitation is that it functions merely as a algorithmic fix. It only addresses the speed of the re-randomization, not the fundamental criteria being optimized. Therefore, it it remains a "sub-optimal" design. It does not solve the main problem, which is the failure to prioritize covariates that are more strongly associated with potential outcomes. While the authors prioritize the uniform assignment over covariates, that should not be the objective.  Thus, the strategy in that paper is ultimately an "incomplete solution." By focusing narrowly on reducing the heavy computational burden, it fails to address the more significant, underlying theoretical deficiency. It offers no progress toward the more precise quantification of covariate heterogeneity that is needed. It simply provides a faster way to arrive at a sub-optimal result.

- The theoretical results presented in the paper appear to be only minor modifications of those in the existing literature. To strengthen the paper's contribution, the authors are encouraged to consult the paper cited below and other recent theoretical articles.

- Liu, Z., Han, T., Rubin, D. B., & Deng, K. "Bayesian Criterion for Re-randomization." arXiv preprint arXiv:2303.07904 (2023) and accepted at *Journal of the American Statistical Association*.

### Questions
Please refer to the limitations listed in *Weaknesses*.

### Soundness
1

### Presentation
2

### Contribution
1
