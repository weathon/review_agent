# Multi-Objective Bandits with Hierarchical Preferences: A Thompson Sampling Approach

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 2, 4

## Abstract
This paper studies multi-objective bandits with hierarchical preferences, a class of bandit problems where arms are evaluated according to multiple objectives, each with a distinct priority level. The agent aims to maximize the most critical objective first, followed by the second most important, and so on for subsequent objectives.We address this problem using Bayesian decision-making strategies. Although Bayesian methods have been extensively studied in single-objective bandit settings, its effectiveness in lexicographic bandits remains an open question. To fill this gap, we propose two TS-based algorithms for lexicographic bandits: **(i)** For Gaussian reward distributions, we introduce an multi-armed bandit algorithm that achieves a *problem-dependent regret bound* of $O(\sum_{\Delta^i(a)>0}\frac{\log(mKT)}{\Delta^i(a)})$, where $\Delta^i(a)$ denotes the suboptimality gap for the objective $i\in[m]$ and arm $a\in[K]$, and $m$ is the number of objectives. **(ii)** For unknown reward distributions, we design a stochastic linear bandit algorithm with a *minmax regret bound* of $\widetilde{O}(d^{3/2}\sqrt{T})$, where $d$ is the dimension of the contextual vectors. These results highlight the adaptability of TS strategy to the lexicographic bandit problem, offering efficient solutions under varying degrees of knowledge about the rewards. Empirical experiments support our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Thompson Sampling–based algorithms for multi-objective bandits with hierarchical (lexicographic) preferences. It proposes two algorithms. When the distribution type is known and is Gaussian, DK-TSLB algorithm achieves logarithmic gap-dependent regret. The other algorithm (DF-TSLB) works under the setting of stochastic linear bandits and achieves $\sqrt{T}$ gap-independent regret. The theoretical analysis proves that these bounds are comparable to single-objective TS algorithms, except for an additional multiplicative factor that quantifies hierarchical trade-offs. Simulations show the advantage of the proposed algorithms with respect to baseline methods.

### Strengths
1. To the best of the reviewer’s knowledge, this is the first TS-based solution for lexicographic multi-objective bandits. The results extend the reach of TS into multi-objective decision-making with lexicographic preferences, which is becoming highly relevant for real-world applications (recommender systems, clinical trials, etc).

2. The paper provides a detailed theoretical analysis with complete proofs. Both problem-dependent and minimax regret bounds are established.

3. The paper is easy to follow. Assumptions and algorithms are clearly described. The results make sense.

### Weaknesses
1. The hierarchical tradeoff parameterized by $\lambda$ restricts the set of multi-objective learning problems to which the proposed methods are applicable. The hierarchical tradeoff may not hold in practice. This requires empirical justification (with some well-known datasets used in bandit algorithm evaluation) to improve the credibility of the current work. 

2. Experiments are limited to toy settings (10 arms, 3 objectives). No evaluation on real-world datasets. 

3. Dependence of the regret on the additional multiplicative factor $\Lambda_i(\lambda)$ is an unresolved question. A lower bound analysis is not provided in the paper. 

4. The paper does not compare against more recent lexicographic multi-objective methods beyond PF-LEX [A, B]. These works also involve the hierarchical tradeoff parameter $\lambda$. A detailed comparison in terms of algorithm implementation, regret bounds, and proof techniques should be given. Specifically, the filtering approach in these works should be compared with the filtering approach of the TS-based algorithms (conceptually, what is new in the paper?). The novelty and challenges of extending the results to the TS-based algorithms should be articulated clearly to evaluate the originality and significance of the proposed approach.

[A] Xue, Bo, et al. "Multiobjective Lipschitz bandits under lexicographic ordering." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.

[B]  Xue, Bo, et al. "Multiple trade-offs: An improved approach for lexicographic linear bandits." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 20. 2025.

### Questions
1. Indicate what the summation is over in the regret bound in line 20 of the abstract. 

2. Algorithms require the knowledge of the tradeoff parameter $\lambda$. What happens if $\lambda$ is misspecified? An experimental evaluation of the proposed algorithms under misspecification will yield an understanding of the robustness of the proposed algorithms. 

3. Can $\lambda$ be estimated online? An experimental evaluation of adaptations of algorithms that can estimate $\lambda$ online will strengthen the contribution.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes and analyzes Thompson sampling-like algorithms for multi-objective bandits with hierarchical preferences (lexicographic ordering). Two distinct settings are investigated: (1) MOMAB: Finite, independent arms with Gaussian rewards and Gaussian priors, where the distribution parameters are known up to the prior. (2) MOSLB: Stochastic linear bandits with shared parameters across arms and general bounded reward distributions, where distributions are unknown. The authors present Algorithm 1 (DK-TSLB) for the MOMAB setting and Algorithm 2 (DF-TSLB) for the MOSLB setting. Both algorithms employ a priority-based arm elimination procedure. Algorithm 1 builds upon Bayesian inference, while Algorithm 2 utilizes TS-like posterior sampling. A key assumption is that suboptimal expected rewards in higher-priority objectives ($j<i$) can lead to at most $\lambda$-times the improvement in lower-priority objectives ($i$). Under this structure, the authors derive an instance-dependent regret bound for Algorithm 1 and an instance-independent regret bound for Algorithm 2. Both algorithms achieve sublinear regret for each objective $i \in [m]$. Notably, the regret guarantee for the primary objective ($i=1$) is shown to be no worse than standard single-objective bandit algorithms.

### Strengths
The concept of using the parameter $\lambda$ to model the trade-off structure between hierarchically ordered objectives is clever, simplifying both the problem setup and the subsequent analysis. It is an interesting theoretical finding that, under the proposed framework, optimizing for lower-priority objectives ($i > 1$) does not degrade the regret performance for the primary objective ($i=1$) compared to single-objective optimal algorithms.

### Weaknesses
1. **Role and Definition of Thompson Sampling**: The paper frames its contribution around Thompson Sampling, but TS doesn't appear to be the core mechanism driving the results. The arm elimination procedure, which manages exploration across objectives based on the $\lambda$-structure, seems far more critical.

- Algorithm 1 (DK-TSLB) does not implement the characteristic posterior sampling step of TS; it uses Bayesian inference (calculating posterior means and variances) combined with confidence bounds for elimination. It would be more accurately described as a Bayesian UCB or Active Arm Elimination algorithm.

- While Algorithm 2 (DF-TSLB) does incorporate posterior sampling ($\tilde{\theta}$), it's unclear from the presentation and analysis whether this sampling step is truly essential for achieving the regret bound, or if using the posterior mean ($\hat{\theta}$) would suffice (related to Question 1).

2. **Novelty Relative to Prior Work**: Given that TS itself might not be the key innovation, the main contribution appears to be the application of the $\lambda$-based hierarchical elimination procedure to MOMAB and MOSLB settings. However, a very similar structure (Assumption 5) and elimination logic (Algorithms 2 & 4 involving $\lambda_i = 1+\lambda+...+\lambda^{i-1}$) was previously introduced in Xue et al. (2024), "Multiobjective Lipschitz Bandits under Lexicographic Ordering". While cited, the current paper does not clearly state the extent to which it adopts this existing mechanism, making the novelty seem incremental rather than substantial.

3. **Disparate Settings and Analyses**: The two settings investigated (MOMAB with known distributions vs. MOSLB with unknown distributions) feel somewhat disconnected.
- The models differ significantly (independent arms vs. shared linear parameters).
- The algorithms employ different core mechanics (Bayesian UCB vs. TS with sampling).
- The resulting regret bounds are of different types (instance-dependent vs. instance-independent).
- This lack of alignment makes the paper feel like a combination of two separate results rather than a unified contribution built around a single core idea (beyond the shared $\lambda$-elimination concept).

4. **Exponential Dependence on Objective Priority**: Although acknowledged briefly by the authors as an open problem 2, the exponential dependence of the regret bounds on the objective index $i$ (via the $\Lambda^i(\lambda)$ term) is a significant limitation. This could render the theoretical guarantees impractical for problems with even a moderate number of objectives ($m$) or a large trade-off factor ($\lambda$). A more thorough discussion and analysis of this limitation is warranted.

### Questions
1. **Necessity of Sampling in Algorithm 2**: Could the authors clarify why posterior sampling ($\tilde{\theta}$) is necessary in Algorithm 2? Specifically, what would happen if $\hat{\theta}_t^i$ (the posterior mean) were used instead of $\tilde{\theta}_t^i$ for arm elimination (in Algorithm 3)? Would the algorithm fail, or would the proof break down? If the proof fails, which specific step or lemma becomes invalid?

2. **Choice of Regret Analysis**: Why was an instance-dependent regret analysis performed for MOMAB (Thm 1) but an instance-independent one for MOSLB (Thm 2)? Could a problem-independent bound be derived for Algorithm 1, or a problem-dependent bound for Algorithm 2, to allow for a more consistent comparison or demonstrate robustness across analysis types?

3. **Relation to Xue et al. (2024) and Justification of $\lambda$**: Beyond the difference in bandit models (MAB/SLB vs. Lipschitz), what is the key conceptual difference in the $\lambda$-based assumption and the arm elimination scheme compared to Xue et al. (2024)? Could the authors provide a stronger justification for the $\lambda$ assumption itself? Is it purely an analytical tool, or does it reflect structures found in real-world lexicographic problems? How easily can $\lambda$ be estimated or bounded in practice? 

4. **Empirical Verification of Scaling**: Does the empirical performance shown in the experiments (Figure 1) reflect the exponential scaling with the objective index $i$ suggested by the $\Lambda^i(\lambda)$ term in the theoretical regret bounds, particularly for the second ($i=2$) and third ($i=3$) objectives?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper considers a multi-objective version of the multiarmed bandit problem, where each arm is associated with a vector of reward distributions. In this setting, there are various possible notions of regret and the authors chose to define it with respect to a lexicographicall optimal arm. The authors give two algorithms, one for the case of standard  finite armed bandit setting with Gaussian rewards and the other for linear bandits with unknown reward distributions. There is some limited experimental evaluation.

### Strengths
- The paper is relatively well written
- It does a decent job of literature review.

### Weaknesses
Unfortunately i have several criticisms of the paper. 

- The paper takes the notion of lexicographically optimal arms from [Huyuk and Tekin 2021] but their notion of regret is not the same.In particular, the present paper measures the regret for task $i$ of playing an arm with respect to the lexicographically optimal arm, which does not seems to make sense: more natural would be to compare  it against the reward of the optimal arm for that task. This is reflected also in the need to introduce the condition in (2) and (3), which seems contrived.
- The authors make no attempt to motivate the lexicographically optimal arm, nor their notion of regret. The original paper  [Huyuk and Tekin 2021] do give some motivating examples, but correspond to their notion of regret, not to this paper's.
- Algorithm 1 is not Thompson sampling at all!  It is closer to the Bayesian UCB in reference [Kaufmann et al 2021a]. The technical arguments seem to me quite similar, so there is limited novelty.
- The authors claim that methods based on Pareto dominance "assume equal importance across all objectives, ...” (lines 86–87). I don't think Pareto optimality makes any such assumption, but just recognize objectives are incomparable hence there is a tradeoff on the Pareto frontier.

### Questions
- Could you please clarify that your definition of regret is not he same as in [Huyuk and Tekin 2021] and if so, then justify why it is reasonable?
- Why do you call Algorith 1 Thompson sampling? isn't t closer to the Bayesian UCB from [Kaufmann et al 2021a]?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The submission investigates multi-objective bandit problems (MOMABs and MOSLBs) and introduces two Thompson Sampling (TS)-based algorithms: Distribution-Known Thompson Sampling for Lexicographic Bandits (DK-TSLB) and Distribution-Free Thompson Sampling for Lexicographic Bandits (DF-TSLB), tailored for MOMABs and MOSLBs, respectively. Theoretical analyses provide rigorous regret bounds, presented in Theorems 1 and 2, to substantiate the proposed methods. Furthermore, empirical simulations demonstrate results that are largely consistent with the theoretical findings.

### Strengths
- The submission explores the potential of TS based methods for multi-objective bandits.
- Concrete methods considering both distribution-known and distribution-unknown scenarios are proposed and analyzed rigorously.

### Weaknesses
- Two strict assumptions, the existence of a lexicographic optimal arm (LOA) and the validity of inequalities (2) and (3), limit the applicability of the proposed methods.
- Based on the intuition in (line 207), there may exist weaker and more realistic formulations to model the value improvement during solution transition.
- The key steps in the proofs (lines 760 and 911) directly invoke assumptions (2) and (3), making the analyses less technical. 
- The results cannot be applied to the single-objective MAB case, as the regret bound becomes suboptimal; thus, one must rely on existing methods instead.
- The experiments do not include Gaussian reward functions to validate DK-TSLB. 
- In the MOSLB experiments, finite arms are used instead of the theoretically assumed infinite arm setting, creating a discrepancy between theory and experiments.

### Questions
- It seems the word “trade-offs” in “capture the trade-offs between conflicting objectives, …” (line 200) implies the user has a preference among the objectives. Can this paper address user preference?
- Are assumptions (2) and the existence of LOA independent assumptions, or does one imply the other?
- In the proofs (Appendix B and C) of Theorems 1 and 2, at which specific steps is the LOA assumption utilized?

### Soundness
3

### Presentation
3

### Contribution
2
