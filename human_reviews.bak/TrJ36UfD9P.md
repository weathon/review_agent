# Methods with Local Steps and Random Reshuffling for Generally Smooth Non-Convex Federated Optimization

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Non-convex Machine Learning problems typically do not adhere to the standard smoothness assumption. Based on empirical findings, Zhang et al. (2020b) proposed a more realistic generalized $(L_0,L_1)$-smoothness assumption, though it remains largely unexplored. Many existing algorithms designed for standard smooth problems need to be revised. However, in the context of Federated Learning, only a few works address this problem but rely on additional limiting assumptions. In this paper, we address this gap in the literature: we propose and analyze new methods with local steps, partial participation of clients, and Random Reshuffling without extra restrictive assumptions beyond generalized smoothness. The proposed methods are based on the proper interplay between clients' and server's stepsizes and gradient clipping. Furthermore, we perform the first analysis of these methods under the Polyak-Łojasiewicz condition. Our theory is consistent with the known results for standard smooth problems, and our experimental results support the theoretical insights.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces three novel federated learning (FL) algorithms—Clip-LocalGDJ, CLERR, and Clipped RR-CLI. The first method adds gradient clipping to localGD on the server side. Building on the first method, the second method adapts reshuffling to local updates. The third method extends the second method to address the client drift problem by shuffling the clients. The convergence of all methods is analyzed under $(L_0, L_1)$ generalized smoothness in non-convex settings without assuming data heterogeneity. Additionally, this paper analyzes the proposed methods under the Polyak-Łojasiewicz condition within the generalized smoothness setting.

### Strengths
1. This work adapts gradient clipping to local GD with convergence guarantees.

2. This work fills in several theoretical gaps:

2.1. Convergence analysis for an FL method with partial participation of clients under the $(L_0, L_1)$-smoothness assumption.

2.2. Convergence analysis for FL methods under both $(L_0, L_1)$-smoothness and Polyak-Łojasiewicz conditions.

### Weaknesses
1.This work claims that its analysis uses weaker assumptions by not assuming data heterogeneity. However, in the convergence upper bounds presented in Theorems 1, 3, and 4, the right-hand side contains terms such as $\Delta^$ or $\bar{\Delta}^$, which are closely related to data heterogeneity. Specifically, $\Delta^*$ is an example of data heterogeneity as defined in earlier work [1]. This undermines the claim of weaker assumptions since the convergence analysis still implicitly depends on data heterogeneity through these terms. It would be better to make it clear that what is the cost of removing the heterogeneity assumptions.

[1] Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, Zhihua Zhang:
On the Convergence of FedAvg on Non-IID Data. ICLR 2020

2.  There are many works [2, 3, 4] that provide convergence guarantees without heterogeneity bounds in their complexity. It would be more complete to add these related work.

 [2] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh:
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. ICML 2020: 5132-5143.

 [3] Quoc Tran-Dinh, Nhan H. Pham, Dzung T. Phan, Lam M. Nguyen:
FedDR - Randomized Douglas-Rachford Splitting Algorithms for Nonconvex Federated Composite Optimization. NeurIPS 2021: 30326-30338.

[4] Chandra Thapa, Mahawaga Arachchige Pathum Chamikara, Seyit Camtepe, Lichao Sun:
SplitFed: When Federated Learning Meets Split Learning. AAAI 2022: 8485-8493 2020.

3. After Corollary 1 and Corollary 2, the authors claim that when $L_1 = 0$, their iteration complexity matches that of local GD and GD. However, from my understanding, the motivation for analyzing generalized smoothness is not just to recover known results in the smooth case. In Zhang et al. (2020b), it was empirically shown that gradient clipping can achieve faster convergence. Therefore, to strengthen the theoretical contributions of this work, the authors could add more theoretical discussions on how the iteration complexity of the proposed methods accelerates beyond local GD, rather than simply stating that their complexity matches the previous results in the smooth case. A discussion similar to Remark 5 in Zhang et al. (2020b) would be helpful.

### Questions
please refer to the weakness.

### Soundness
2

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
4

### Summary
The paper proposes three federated learning algorithms for solving minimization problems under a generalized smoothness assumption. More specifically, the authors propose a new method, called Clip-LocalGDJ, that uses local steps, a second one that combines local steps with Random Reshuffling, while the third method allows also for partial participation of the clients in the federated learning regime. For all the above methods, convergence guarantees are established under a generalized smoothness assumption, matching the existing results under the classical smoothness assumption. In addition, the authors provide theoretical guarantees for the convergence of each method in the  Polyak-Łojasiewicz (PL) setting, proving a linear rate of convergence. Extensive experimental results are provided, validating the theory and testing the effectiveness of each method in practice.

### Strengths
- The analyses of all algorithms are made under the relaxed smoothness assumption and the results are tight in the sense that they coincide with the already established ones in the case when the classical smoothness assumption is true.
- The Clipped RR-CLI algorithm combines local steps, random reshuffling and partial client participation and is analyzed for distributed non-convex and ($L_0, L_1$) smooth problems. This is the first algorithm that combines all the above three properties and is being analyzed under such relaxed conditions.
- The paper is well-written and the presentation is easy to follow for the reader.

### Weaknesses
- Since the analysis is provided under the most relaxed assumptions that exist in the literature, do you believe that an analogous proof could be extended in cases where the underlying problem is a minimax optimization problem or in scenarios where extrapolation is also used in the process? 
- How does the stepsizes used in Theorems 1, 2 compare with the ones in previous works? Do the best stepsizes found in the Experimental Section coincide with the ones predicted in the theory? 
- How might the large-scale nature of data in practical federated learning settings pose challenges for implementing the proposed algorithms? What potential obstacles do you see in adopting these methods widely in practice?

Minor Typos:
- In Section 1.3 (line 186): " clipped IGWang et al. (2024)" the reference should be added correctly.
- In Appendix E: it would be better to incorporate beyond the plots some further details on the additional experiments.

### Questions
- Which algorithm of the three should be preferred in practice? In particular, given that Algorithm 3 allows also the partial participation of the clients, it seems to be the best choice in practical scenarios where all clients are not available during the time that the algorithm is run. However, in cases where all clients are available, are there any potential disadvantages of using Algorithm 3 in comparison to Algorithm 2? Additionally, in full participation cases should one prefer in practice Algorithm 2 that utilizes also Random Reshuffling over Algorithm 1? 
- Algorithm 3 allows for partial participation by performing a reshuffling of the clients that participate to solve the underlying problem. Is there any practical application, where performing random reshuffling in the clients turns out to be more beneficial than the classical sampling with-replacement technique for partial participation? Are there any theoretical insights or empirical evidence on the trade-off between these two approaches?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies and analyzes the federated learning algorithms under the nonconvex setting with the generalized smoothness condition. In the deterministic setting where each client can compute its full gradient, the authors propose a new method with local steps with jumping. In the stochastic (finite-sum) setting, the authors propose two algorithms with partial participation and shuffling on both clients and data sides. All the algorithms proposed are new, and rigorous convergence analysis under generalized smoothness is provided in this paper.

### Strengths
- The authors provide a comprehensive analysis for federated learning under generalized smoothness conditions, which I believe is novel and new.
- It is interesting to see random reshuffling is analyzed under generalized smoothness for federated learning problems.
- The spectrum of the federated learning algorithms with their convergence results in this paper is quite substantial.

### Weaknesses
- In the section of related work, the authors missed recent advances since 2020 in random reshuffling (e.g., [1-3]), which I believe should be included for a complete context.
- The global step size $\theta_t$ requires evaluating the full gradient, i.e., $\|\nabla f(x_t)\|$, which requires further computation and communication costs.
- Algorithm 2 returns $x_T$ in the statement, but the convergence is only analyzed for the best iterate. The statement of Algorithm 2 possibly needs rephrased.
- The experiments are mostly conducted on a synthetic function (4). 

[1] Cha, Jaeyoung, Jaewook Lee, and Chulhee Yun. "Tighter lower bounds for shuffling SGD: Random permutations and beyond." International Conference on Machine Learning. PMLR, 2023.
[2] Cai, Xufeng, Cheuk Yin Lin, and Jelena Diakonikolas. "Empirical risk minimization with shuffled SGD: a primal-dual perspective and improved bounds." arXiv preprint arXiv:2306.12498 (2023).
[3] Koloskova, Anastasia, et al. "On Convergence of Incremental Gradient for Non-Convex Smooth Functions." arXiv preprint arXiv:2305.19259 (2023).

### Questions
Besides the weaknesses mentioned, I have the following questions.
- Is it possible to analyze the local steps for Algorithms 2 and 3? Would the local steps bring benefits with random reshuffling, or are there any technical difficulties that the authors foresee?
- Is it possible to analyze the case of different inner stepsizes for different clients?
- In general how does one choose the constants $c_0, c_1$ in the step sizes?
- Is Appendix E on additional experiment details unfinished? There are only a few figures with captions that extend beyond the box.

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
3

### Summary
This paper proposed and analyzed new methods with local steps, partial participation of clients, and Random Reshuffling for Federated Learning with generalized $(L_0, L_1)$ smoothness assumption. Moreover, they also analyzed these methods with the additional Polyak-Łojasiewicz condition. Their results are consistent with the known results for standard smoothness, and they also conducted experiments to support their theoretical claims.

### Strengths
The authors proposed new methods with local steps, Random Reshuffling, and partial participation for Federated Learning and proved the convergence of their algorithms under $(L0, L1)$-smoothness assumption and under additional PL assumption.

### Weaknesses
(1): It would be better if the authors can add a conclusion/discussion section at the end of the main text.

(2): The section E Additional experimental details are missing.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
