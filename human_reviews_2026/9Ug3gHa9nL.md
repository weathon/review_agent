# Offline Policy Learning for Nonparametric Contextual Bandits under Relaxed Coverage

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
This paper is concerned with learning an optimal policy in a nonparametric
contextual bandit from offline, and possibly adaptively collected
data. Existing methods and analyses typically rely on i.i.d. offline
data, and a uniform coverage condition on the behavior policy. In
this work, similar to the single-policy concentrability coefficient,
we propose a relaxed notion of coverage that measures how well the
optimal action is covered by the behavior policy for the nonparametric
bandits. Under this new notion, we develop a novel policy learning
algorithm by combining the $k$-nearest neighbor method with the pessimism
principle. The new algorithm has three notable properties. First and
foremost, it achieves the minimax optimal suboptimality gap for any
fixed coverage level (up to log factors). Second, this optimality is attained adaptively, without requiring prior knowledge of the coverage level of the offline data. Last but not least, it maintains these guarantees even with adaptively collected offline data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies offline policy learning in nonparametric contextual bandits when the offline dataset may not satisfy the traditional uniform coverage condition. Specifically, it introduces a relaxed coverage assumption that focuses on how well the optimal action is covered by the behavior policy, extending the idea of the single-policy concentrability coefficient from offline RL. The work bridges the gap between nonparametric statistics and offline reinforcement learning, demonstrating that near-optimal policies can be learned from limited or adaptively collected data without strong coverage assumptions.

### Strengths
The introduction of the relaxed coverage notion is a key novelty. It generalizes prior assumptions and reflects more realistic conditions found in practice, e.g., healthcare or recommender systems where only near-optimal actions may have sufficient representation. The combination of nonparametric estimation with the pessimism principle is novel in this setting.

Theoretical contributions are rigorous. The authors establish tight minimax upper and lower bounds, which is significant because such tightness under partial coverage is nontrivial. The algorithms are derived systematically from the theoretical framework and are accompanied by detailed analyses. The proofs demonstrate a careful decomposition of the excess risk using margin and smoothness conditions, following classical nonparametric learning methods.

The paper is well-structured, starting with motivation and related work, followed by formal setup, algorithm descriptions, theoretical results, and detailed proofs.

The results make a meaningful theoretical advancement. They extend the landscape of offline learning from parametric to nonparametric models, achieving near-minimax rates without stringent assumptions.

### Weaknesses
The paper is entirely theoretical. While this is acceptable for a theory-focused venue, it would be valuable to include synthetic experiments demonstrating how the proposed algorithms perform with varying coverage levels. This would help illustrate the adaptivity of KNN-LCB in practice and bridge theory to application.

While the authors position the paper within offline policy learning, it could benefit from a stronger conceptual bridge to recent OPE results (e.g., minimax rates in OPE under limited coverage). This would help position the contribution more clearly relative to established theory in both RL and statistics.

The analysis assumes known smoothness and margin. The authors mention that adaptation to unknown smoothness is impossible without extra assumptions, but it would be helpful to provide more context, e.g., under what mild regularity assumptions adaptivity could be feasible.

### Questions
How can $C^\star$ be estimated or approximated from data in practice? Even if not required for the algorithm, understanding it empirically might provide diagnostic value.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the important problem of offline policy learning for nonparametric contextual bandits under a relaxed coverage condition. The relaxed coverage notion that only requires the optimal action to be covered, in contrast to the more stringent uniform coverage assumptions common in prior work. They introduce two algorithms: a binning-based method (BIN-LCB) and a more sophisticated k-nearest neighbor method (KNN-LCB) that incorporates pessimism. Theoretically, they establish a minimax lower bound and prove that both algorithms achieve a suboptimality gap that matches this lower bound up to logarithmic factors. A key strength of the KNN-LCB algorithm is its adaptivity—it achieves this optimal rate without prior knowledge of the coverage coefficient and its guarantees hold even for adaptively collected data.

The paper tackles a non-trivial and relevant problem, and the technical results appear sound and significant. The adaptive algorithm (KNN-LCB) is a particularly elegant solution to the practical challenge of unknown data quality.

### Strengths
- Minimax Optimality: The paper provides a complete minimax characterization of the problem. Theorem 3.2 establishes a fundamental lower bound, and Theorems 3.1 and 4.1 show that both proposed algorithms are minimax optimal up to log factors. This provides a clear picture of the problem's statistical limits.
- Adaptive Algorithm (KNN-LCB): The design of Algorithm 2 (KNN-LCB) is a major contribution. Its ability to adapt to the unknown coverage level in a data-driven manner is highly desirable for practical deployment. The mechanism of choosing 
k(a) separately for each arm and test point to balance bias and variance is interesting
- Handling Adaptive Data: The theoretical guarantees are robust to adaptively collected offline data, which is a common scenario when the batch data comes from a previously deployed online learning algorithm. This significantly broadens the applicability of the results beyond the simpler i.i.d. setting.

### Weaknesses
- Lack of Empirical Validation: The most significant weakness of the paper is the complete absence of empirical results. Given that the KNN-LCB algorithm is a primary contribution, it is crucial to demonstrate its performance on synthetic or real-world datasets. How does it compare empirically to natural baselines (e.g., direct importance weighting, vanilla k-NN, or the non-adaptive BIN-LCB)? Does it indeed adapt to varying levels of coverage as theory predicts? Without empirical validation, it is difficult to assess the algorithm's practical utility and performance in finite-sample regimes.
- Assumption of Known Smoothness and Margin Parameters: The algorithms and their theoretical guarantees rely on prior knowledge of the smoothness parameter β and the margin parameter α. As the authors note in the discussion (Section 6), these are typically unknown in practice. While they cite literature suggesting that adaptation to unknown β is generally impossible, the paper would be significantly strengthened by a discussion or even a heuristic approach for setting these parameters, or an investigation of the consequences of misspecification.
- A naive implementation of Algorithm 2 would have a high computational cost. For a test point x, for each arm a, it must compute $U_k^a(x)$ for all k from 1 to N to find the minimizer k(a). This leads to a per-test-point complexity of $O(K * N)$, which is prohibitive for large datasets N. The paper should discuss computational considerations and potential approximations (e.g., using efficient nearest neighbor search, or searching over a subset of k values).

### Questions
- What happens if we do not assume the margin parameter? What is the role of margin parameter in this result and guarantees?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem of offline policy learning for nonparametric contextual bandits. The authors consider a setting where the offline dataset may be collected adaptively and, crucially, relax the standard uniform coverage assumption to a much weaker single-policy concentrability condition that only requires the optimal action to be covered. Under this relaxed coverage and standard smoothness/margin conditions, the paper makes four key contributions: (1) It establishes the minimax optimal rate for the suboptimality gap, which is of order $$ \widetilde{O}\left(\left(N / C^{\star}\right)^{-\frac{\beta(1+\alpha)}{2 \beta+d}}\right) $$ ; (2) It proposes a binning-based algorithm (BIN-LCB) that achieves this rate but requires knowledge of the coverage coefficient $$C^{\star}$$ ; (3) Its main algorithmic contribution is a novel k-nearest neighbor-based algorithm (KNN-LCB) that adaptively achieves the minimax optimal rate without prior knowledge of $$C^{\star}$$ ; (4) All theoretical guarantees hold under adaptively collected data, significantly broadening their applicability.

### Strengths
The primary strength lies in formulating and solving the problem under a relaxed, single-policy concentrability condition, a major advancement over restrictive uniform coverage assumptions. The most notable innovation is the KNN-LCB algorithm, which is proven to be minimax-optimal without prior knowledge of the coverage coefficient $$C^{\star}$$ . Furthermore, the work establishes a complete minimax framework, providing matching upper and lower bounds, and all guarantees hold under adaptively collected data, enhancing the theory's rigor and practical relevance.

### Weaknesses
KNN-LCB requires an expensive procedure for each test point, involving searching over all data points for each action to determine the optimal number of neighbors, leading to a theoretical per-decision complexity of $$O(K \cdot N \cdot d)$$ . This is prohibitively expensive for large datasets.

### Questions
Given the computationally expensive nature of the KNN-LCB algorithm, which requires $O(K \cdot N \cdot d)$ operations per test point, do you see a pathway to significantly reduce this complexity to logarithmic or near-logarithmic scales, such as $O(K \cdot \log N \cdot d)$ ? For instance, could efficient data structures like KD-trees or approximate nearest neighbor techniques be integrated into your theoretical framework while preserving the minimax optimality guarantees?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops a nonparametric theoretical framework for offline policy learning in contextual bandits under relaxed coverage.
The setting considers a finite action set and continuous contexts, under smoothness and margin conditions on each action's reward function $f_a(x)$.
The authors introduce a coverage coefficient $1/C^\star$, quantifying the minimum probability that the optimal arm is taken over the covariate space, and derive the minimax lower bound on the suboptimality of policy learning:
\$$ \tilde O\left[\left(\tfrac{N}{C^\star}\right)^{-\frac{\beta(1+\alpha)}{2\beta+d}}\right],
\$$
They propose two nonparametric (pessimistic) algorithms: BIN-LCB, using binning-based local averages, and KNN-LCB, using adaptive knn regression, both combined with lower confidence bounds. Both (nearly) achieve the above rate. The work is entirely theoretical, extending pessimistic offline learning analyses to the nonparametric regime.

### Strengths
**S1.** Clean minimax characterization of policy learning performance under relaxed coverage and smooth reward assumptions.

**S2.** Rarely explored nonparametric perspective of offline policy learning in contextual bandits.

**S3.** Conceptually easy-to-understand algorithms (BIN-LCB and KNN-LCB) that enjoy good theoretical guarantees.

### Weaknesses
**W1. No empirical validation** is provided, even though offline contextual bandits are typically evaluated on practical benchmarks. Almost all papers in offline policy learning (even highly theoretical ones) include empirical results to illustrate performance.

**W2. Limited practicality of the proposed algorithms.** BIN-LCB has very high space complexity when $M$, $d$, or $K$ are large, and KNN-LCB incurs high inference time that scales with the dataset size $N$. As a result, the methods are practical only in low-dimensional, low-data regimes. Connecting to the point above, the absence of experiments prevents any demonstration of their strengths.

**W3. Insufficient discussion of the coverage condition.** The paper provides no clear interpretation of what happens when coverage is already sufficient (large $1/C^\star$). The lower-bound theorem assumes $C^\star$ is large enough: i.e., that the problem is sufficiently hard. This is understandable, so that we can derive the lower bound, but it should be discussed, especially whether the proposed algorithms become overly conservative in that case.

**W4. Unclear dependence on the number of actions.** The theoretical bounds do not make the scaling in $K$ explicit, even though this dependence is important in offline contextual bandits.

**W5. Missing references.** The paper omits directly relevant work on pessimistic offline contextual bandits, including [1, 2, 3] and many more. Authors should include these papers and the references therein. To find such papers, authors can read [3], which provides a unified framework citing many pessimistic approaches. While these works consider policy classes and are thus parametric with respect to the policy (though nonparametric with respect to rewards, except [2]), they remain relevant.

[1] https://arxiv.org/pdf/2006.10460 (AISTATS)

[2] https://www.researchgate.net/profile/Olivier-Jeunen/publication/353481957_Pessimistic_Reward_Models_for_Off-Policy_Learning_in_Recommendation/links/60ffe2ea1e95fe241a8ee976/Pessimistic-Reward-Models-for-Off-Policy-Learning-in-Recommendation.pdf (RecSys)


[3] https://arxiv.org/pdf/2406.03434 (UAI)

**W5. Missing discussion.** Numerous recent papers (e.g., [4, 5] and references therein) derive bounds that depend on coverage of $\pi^*$ rather than coverage of the whole action space. They are still different (e.g., parametric either on the reward or policy), and often assume the behavior policy is static (not adaptive). Discussing those would strengthen the position of the paper.

[4] https://arxiv.org/pdf/2402.14664 (AISTATS)

[5] https://arxiv.org/pdf/2309.15771 (ALT)

### Questions
**Q1.** Could the authors clarify the precise dependence of the theoretical bounds on the number of actions $K$?

**Q2.** What happens when the dataset already provides good coverage (large \$1/C^\star\$): will the algorithms become overly conservative in this regime?

**Q3.** Could the authors provide explicit analyses of the space complexity, computational cost, and inference time for both BIN-LCB and KNN-LCB?

**Q4.** Could the authors include experiments demonstrating cases where BIN-LCB or KNN-LCB outperform existing pessimistic offline learning methods?

**Q5.** A broader question: Is pessimism fundamentally necessary here? For example, some recent works [1, 2] suggest that greedy approaches outperform pessimistic ones in terms of average suboptimality (not worst-case suboptimality). Thus, using pessimism or not would depend on the setting (e.g., frequentist or Bayesian, etc.) and metric used (e.g., suboptimality or Bayesian suboptimality, etc.). Given that your algorithms incorporate pessimism, could you share your perspective on this?

[1] https://arxiv.org/pdf/2306.01237 (ICML)

[2] https://arxiv.org/pdf/2402.14664 (AISTATS)

### Soundness
3

### Presentation
2

### Contribution
3
