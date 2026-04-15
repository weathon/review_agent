# Distributional off-policy evaluation with Bellman residual minimization

- Decision: Reject
- Scores: 8, 3, 6, 6

## Abstract
We consider the problem of distributional off-policy evaluation which serves as the foundation of many distributional reinforcement learning (DRL) algorithms. In contrast to most existing works (that rely on  supremum-extended statistical distances), we study the expectation-extended statistical distance for quantifying the Bellman residuals and provide the corresponding theoretical supports. Extending the framework of Bellman residual minimization to DRL, we propose a method called Energy Bellman Residual Minimization (EBRM) to estimate the return distribution. We establish a finite-sample error bound for the EBRM estimator under a realizability assumption. Additionally, we introduce a variant of our method based on a multi-step bootstrapping procedure to enable multi-step extension. By selecting an appropriate step level, we obtain a better error bound for this variant of EBRM compared to a single-step EBRM, under non-realizability settings. Finally, we demonstrate the superior performance of our method through simulation studies, comparing it to other existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces novel target objectives for distributional RL along with a comprehensive error bound analysis.

### Strengths
* The underlying theory seems solid and well-founded, say the derived convergence rates from Theorems 2 and 3 appear reasonable. 

* Theorem 3 also handles non-realizable settings which are not seen from previous works.

### Weaknesses
* Theorem 3 dosen't degenerate to Theorem 2 under similar settings, which opens the possibiliy of whether a better proof/result exists.

### Questions
None

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a new method of distributional OPE, EBRM, which is derived from a discrepancy metric of distribution-valued functions based on the energy distance.

The discrepancy metric has two nice properties. First, it allows us to bound the distributional OPE error in terms of the distributional Bellman residual. Second, it is straightforwardly estimated from the offline sample when the state-action space is finite (i.e., in the tabular setting).

In the tabular setting, the authors also constructed algorithms to estimate the minimizer of the proposed discrepancy metric for realizable and unrealizable cases, respectively.

Finally, some experimental results are provided.

### Strengths
- Addressing interesting problem of OPE (not just in the distributional one): Discrepancy between the estimation error and the estimation objective, where the latter is typically defined with the Bellman residual.

### Weaknesses
-  The key inequality (7) involves a large constant $C_{sup}=p_{max}/p_{min}$, which makes it uninteresting. If we are allowed to use a constant of $O(1/p_{min})$, then the max-extended discrepancy $\eta_\infty$ is trivially bounded with the expectation-extended discrepancy $\bar{\eta}$.
- Limited experiments. Specifically, the environment is somewhat small and artificial.
- Some of the notation are broken: For example,
  - What does $b_μ (s, a) ≥ p_{min}$ mean in continuous state space? If it is the density of $b_μ$ with respect to some base measure, what is the base measure?
  - The definition of the kernel (33) does not seem to give $k(x + c, y + c) = k(x, y)$.

### Questions
- What is the limitation of the proposed method?
- In the experiment, why don't you thoroughly feature and discuss the results of more standard and fair metrics (like the Wasserstein-based one in the appendix)?
- Are the weakness points I raised correct?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considered a fundamental problem of the distributional reinforcement learning, termed as the distributional off-policy evaluation. The authors proposed a new algorithm based on the Bellman residual minimization, and characterized the finite-sample statistical error bound under realizable and unrealizable settings. Empirical results on synthetic environments also demonstrate the effectiveness of the proposed setting.

### Strengths
* A comprehensive analysis of distributional off-policy evaluation.
* Technically sound in general.

### Weaknesses
* I'm not fully convinced that the proposed distance measure has some meaning in practice. For me it is much more like a distance measure induced by the theoretical need, instead of motivated from some practical consideration. I can provide an example on this unnatural feeling: for OPE on expected return, we care more about $\|E_{(s, a) \sim \rho} Q(s, a) - E_{(s, a) \sim \rho} \hat{Q}(s, a) \|$ instead of $E_{(s, a) \sim \rho} \| Q(s, a) -  \hat{Q}(s, a) \|$, and the latter one also suffers from a slow convergence rate.
* For the unrealizable setting part I feel there are also many designs that are purely for the proof without so many insights, e.g. $G$ function in (18).
* I believe that there will not be significance rate difference for the realizable and unrealizable settings, and the difference may also come from the algorithm aside from the reason mentioned by the author (e.g. no need for additional operations in the realizable setting). Such reason should be make more clear in the paper.

### Questions
* Can the authors revise the paper to highlight some intuitions behind such algorithm design?
* Can the authors dive into the rate mismatch between the realizable setting and the unrealizable setting?

I haven't had time to go through the proof line by line (as there are about 50 pages to read) but most of proof steps align with the standard techniques and I believe there will not be severe issues. I will raise some questions if some steps of the proof is unclear or potentially wrong.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses a critical aspect of RL by delving into the realm of distributional off-policy evaluation. Specifically, authors introduce a method called Energy Bellman Residual Minimizer (EBRM) to estimate the return distribution, which extends the framework of Bellman residual minimization to distributional RL. This work also establishes finite-sample error bounds for the proposed framework and introduces a multi-step extension for non-realizable settings. Simulations are performed for performance validation and comparison to existing methods.

### Strengths
This paper consider the problem of off-policy evaluation in Distributional RL, which concerns estimating the (conditional) return distribution of a target policy based on offline data. Existing reinforcement learning methods predominantly focus on the expectation of return distribution, while distributional RL methods, which extend this focus to the entire return distribution, have shown promising results. However, in off-policy evaluation within DRL, these methods lack theoretical development, error bounds, and convergence rate analysis. The paper addresses this gap by proposing the EBRM method, accompanied by its finite-sample error bounds, providing a theoretical basis for using expectation-extended distance in measuring distributional Bellman residuals. In particular, authors establish the application of expectation-extended distance for Bellman residual minimization in DRLs. A multi-step extension of the proposed method in non-realizable settings is also discussed, with corresponding finite-sample error bounds. The paper significantly contributes to the theoretical understanding of distributional off-policy evaluation in DRL.

### Weaknesses
The theoretical guarantees of the proposed methods rely on a set of assumptions, some of which appear to be strong, especially for the proof of Theorem 2 and Theorem 3. Authors should comment on how such assumptions can be satisfied when being applied and provide examples for further illustration. In particular, it is of interest to highlight the essential assumptions that are central to the statistical error bounds, and comment on the performance when relaxing some of the assumptions. Due to such assumptions, it is questionable whether the proposed method (EBRM) has wider applicability compared to the existing methods.

### Questions
1. Compared to FLE (Wu et al., 2023), how the proposed method excels in policy learning? In particular, as demonstrated in the empirical evaluations, EBRM does outperforms FLE in terms of statistical accuracy. Does the main reason lie in the fact that FLE focuses on learning the marginal distribution while the proposed method estimates the conditional return distribution?

2. This paper focuses on discounted infinite-horizon tabular RL. Is the current result in Bellman residual minimization generalizable to average-reward MDPs or linear MDPs, and how does it compare to "A Maximum-Entropy Approach to Off-Policy Evaluation in Average-Reward MDPs"?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
