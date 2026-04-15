# FedDRO: Federated Compositional Optimization for Distributionally Robust Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3, 5

## Abstract
Recently, compositional optimization (CO) has gained popularity because of its applications in distributionally robust optimization (DRO) and many other machine learning problems. Large-scale and distributed availability of data demands the development of efficient federated learning (FL) algorithms for solving CO problems. Developing FL algorithms for CO is particularly challenging because of the compositional nature of the objective. Moreover, current state-of-the-art methods to solve such problems rely on large batch gradients (depending on the solution accuracy) not feasible for most practical settings. To address these challenges, in this work, we propose efficient FedAvg-type algorithms for solving non-convex CO in the FL setting. We first establish that vanilla FedAvg is not suitable to solve distributed CO problems because of the data heterogeneity in the compositional objective at each client which leads to the amplification of bias in the local compositional gradient estimates. To this end, we propose a novel Distributed-DRO (D-DRO)~framework that utilizes the DRO problem structure to design a communication strategy that allows FedAvg to control the bias in the estimation of the compositional gradient. A key novelty of our work is to develop solution accuracy-independent algorithms that do not require large batch gradients (and function evaluations) for solving federated CO problems. We establish $\mathcal{O}(\epsilon^{-2})$ and 
 sample and $\mathcal{O}(\epsilon^{-3/2})$ communication complexity in the FL setting while achieving linear speedup with the number of clients. We corroborate our theoretical findings with empirical studies on large-scale DRO problems with multiple real datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigate an interesting problem of federated compositional optimization and propose a novel CO setting, which is more general compared to previous studies. Furthermore, the authors also present the FedDRO algorithm with provable theoretical guarantees. Empirical studies are also conducted.

### Strengths
1. This paper is well-originazed and clearly written.

2. The paper investigates an important problem and introduce a general federated compositional optimization setting.

3. Experimental results are provided to verify the performance of the proposed algorithms

### Weaknesses
1. The authors have introduced a general setting for DRO problems, including DRO with KL-Divergence and $\chi^2$-Divergence. Are there other DRO optimization problems that fit this setting?

2. The exact form of the optimization problem in two experiments should be specified.

3. How is the dataset distributed across each client? Does it reflect the diversity in data distribution?

4. Utilizing the small batch size is one strength of FedDRO. However, this isn't evident in the experiments. Specifically, |b|=16 in FedDRO, while |b|=4 (or 8) in GCIVR (with the client count m=8). Therefore, experiments should be conducted with small batch size or more clients (ensuring large batch size in GCIVR).

5. The experiments are repeated only five times, and it would further strengthen the reliability if reruning the experiments at least ten times. Furthermore, presenting the loss curve would also add to this reliability.

### Questions
See questions in the 'Weakness'.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new compositional optimization algorithm called FedDRO for distributionally robust learning. The proposed algorithm integrates novel communication and optimization techniques to tackle biased gradient estimates, client drift, etc. in compositional optimization. Convergence analysis and numerical experiments are conducted to validate the proposed method.

### Strengths
The proposed approach improves upon the previous compositional optimization methods in terms of convergence rate and the need for large batch gradients. It's also technically novel to utilize the low-dimensional communication of the compositional functions $g(\cdot)$ to trade for better convergence.

### Weaknesses
The low-dimensional $g(\cdot)$ still causes additional $O(\epsilon^{-2})$ communication, whose impact may vary based on specific forms of $g(\cdot)$ and the synchronization settings.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper developed a federated learning method for compositional optimization problem, which can be applied to DRO problems. Both theoretical and empirical results are provided to show its performance. However, this paper overclaims its contributions. It solves a simple compositional problem so it is not surprising to see a better bound. But the authors hide this point. This could mislead the community.

### Strengths
This paper provides a good literature review. 

Theoretical analysis is provided.

### Weaknesses
1. This paper overclaims its contributions.

2. The writing is poor. It could mislead the community.

3. Some operations are not practical.

4. Some claims are NOT correct.

### Questions
1. Different from existing federated compositional optimization, this paper investigated a simple case, i.e., the outer-level function is deterministic, not a stochastic function as the first three baselines in table 1. However, the authors never mentioned this critical difference. For this simple objective function, the authors claim they can achieve better convergence rates. This is not the case because the problem settings are different. This paper overclaims its contribution. The authors should clearly state the difference in the problem setting. Otherwise, it will mislead the community. 


2. The novelty is incremental. For this simple compositional optimization problem, it is trivial to extend existing theoretical analysis to this kind of problem. I didn't see any challenges here. In particular, the outer-level function is deterministic, and $y$ is synchronized in each iteration. Then, compared with the non-compositional optimization problem, there are no additional challenges in convergence analysis. 

3. For Eq.(8), what is the reason for using the storm estimator? The standard moving average estimator should also work. Could you please provide more discussions?

4. This paper claims it can achieve better communication complexity than existing works. It is NOT true. Specifically,  the proposed algorithm communicates $y$ in every iteration. Then, the communication complexity is the same as the number of iterations. It is much worse than existing approaches.

5. No experiments to compare the proposed algorithm with the federated baselines.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper develops a momentum-type federated learning algorithm for Compositional Optimization (CO) problems. The authors provide an $O(\epsilon^{-2})$ sample complexity and $O(\epsilon^{-3/2})$ communication complexity for their approach. Numerical experiments on large-scale Distributionally Robust Optimization problems demonstrate the effectiveness of their method.

### Strengths
1. The paper addresses an important problem in the field.

2. The paper is well-written, with clear mathematical notations. The table comparing related works provides insightful information on their contributions. The discussions around each assumption are useful. 

3. The discussion surrounding the theorems is useful for understanding their results.

### Weaknesses
Please see below.

### Questions
**Abstract:**

- The following expression has already been discussed in existing literature. See Lemma 2.1 of [R1] and Section 1.2 of [R2]. What are your findings in contrast to these?

    “We first establish that vanilla FedAvg is not suitable for solving distributed CO problems due to data heterogeneity in the compositional objectives at each client. This leads to the amplification of bias in the local compositional gradient estimates.”

- The following sample and communication complexity for CO problems have already been established. See [R3] and [R4] for details on sample and communication complexity, and [R4] for linear speedup under the heterogeneity assumption (Assumption 3.4 in your paper). Please clarify the differences.

    “We establish an $O(\epsilon^{-2})$ sample and $O(\epsilon^{-3/2})$ communication complexity in the FL setting while achieving linear speedup with the number of clients.”

**Section 4 (Algorithm Design):**

Algorithm 1 is similar to the method proposed in [R3] and [R4]. Specifically, the improved communication complexity and sample complexity are obtained from Eq (7) and the momentum update in Eq (8), which are already studied in [R3] and [R4].

**Experiment:**

- The main focus of the paper is federated learning under the heterogeneity assumption. However, this setting is not apparent in the experiment evaluation.


- It would be useful if the authors provided a comparison to variance reduction and momentum-based methods designed for heterogeneous federated composition problems [R1-R4].

- Why is there a jump in test accuracy in Figure 1 after 100 communications?



Please let me know in your response if I misunderstood your contribution, and I will be happy to update my score.




[R1] Tarzanagh, D.A., Li, M., Thrampoulidis, C. and Oymak, S., 2022, June. Fednest: Federated bilevel, minimax, and compositional optimization. In International Conference on Machine Learning (pp. 21146-21179). PMLR.

[R2] Yang, S., Zhang, X. and Wang, M., 2022. Decentralized gossip-based stochastic bilevel optimization over communication networks. Advances in Neural Information Processing Systems, 35, pp.238-252.


[R3] Feihu Huang. Faster adaptive momentum-based federated methods for distributed composition optimization. arXiv preprint arXiv:2211.01883, 2022

[R4] Tarzanagh, D.A., Li, M., Sharma, P. and Oymak, S., 2023. Federated Multi-Sequence Stochastic Approximation with Local Hypergradient Estimation. arXiv preprint arXiv:2306.01648.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
