# Order-Optimal Instance-Dependent Bounds for  Offline Reinforcement Learning with Preference Feedback

- Decision: Reject
- Scores: 6, 8, 3, 5, 5

## Abstract
We consider offline reinforcement learning (RL) with preference feedback in which the implicit reward is a linear function of an unknown parameter. Given an offline dataset, our objective consists in ascertaining the optimal action for each state, with the ultimate goal of minimizing the {\em simple regret}. We propose an algorithm, \underline{RL} with \underline{L}ocally \underline{O}ptimal \underline{W}eights or {\sc RL-LOW}, which yields a simple regret of $\exp ( - \Omega(n/H) )$ where $n$ is the number of data samples and $H$ denotes an instance-dependent hardness quantity that depends explicitly on the suboptimality gap  of each action.  Furthermore, we derive a first-of-its-kind instance-dependent lower bound in offline RL with preference feedback. Interestingly, we observe that the lower and upper bounds on the simple regret match order-wise in the exponent, demonstrating order-wise optimality of {\sc RL-LOW}. In view of privacy considerations in practical applications, we also extend {\sc RL-LOW} to the setting of $(\varepsilon,\delta)$-differential privacy and show, somewhat surprisingly, that the hardness parameter $H$ is unchanged in the asymptotic regime as $n$ tends to infinity; this underscores the inherent efficiency of {\sc RL-LOW} in terms of preserving the privacy of the observed rewards. Given our focus on establishing instance-dependent bounds, our work stands in stark contrast to previous works that focus on establishing worst-case regrets for offline RL with preference feedback.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors consider offline reinforcement learning (RL) with preference feedback in which the implicit reward is a linear function of an unknown parameter. They propose an algorithm, RL with Locally Optimal Weights (RL-LOW), which yields a simple regret which decays exponentially as n goes large. Furthermore, they derive a matching instance-dependent lower bound in offline RL with preference feedback. Moreover, they also extend RL-LOW to the setting of differentially private RL and show that the hardness parameter 
 is unchanged in the asymptotic regime as n tends to infinity.

### Strengths
1. The problem of deriving an instance dependent sub-optimality bound for offline RLHF is well-motivated and interesting.

2. The paper presents the first instance dependent bounds for the setting, while the upper bound matches the lower bound in the order. The paper is solid and the proof looks correct to me.

3. The authors also extend the result to DP-RL setting and the MDP setting, which makes this paper a comprehensive work.

4. The simulation supporting the theoretical findings is convincing.

5. The paper is well-written in general.

### Weaknesses
1. The setting is kind of simple. For the bandit setting, although the reward is linear, the setting is actually tabular (finite states & actions). For the MDP setting, the transition kernel is known. While I understand to derive a gap-dependent analysis as in the paper, the setting is required to be tabular, it seems that the locally optimal weights heavily depend on the tabular nature of the problem.

2. The improvement on the worst-case bound is limited. Meanwhile, the instance-dependent upper bound seems to be well-expected, since given the minimum gap, the failure probability of identifying the optimal arm will decay exponentially as n converges to $\infty$.

3. Previous work [1] derived an instance-dependent sub-optimality upper bound $O(\frac{1}{\sqrt{n}}+\frac{1}{n\epsilon})$ for standard offline RL (explicit reward) with DP. The two terms in [1] seem to match the order of the two terms in (22) of Theorem 5.3. It will be interesting if the author could provide a worst-case version of Theorem 5.3 and compare with the result in [1] to further showcase the cost of privacy.

[1] Offline Reinforcement Learning with Differential Privacy, Dan Qiao and Yu-Xiang Wang.

Overall, I think the instance-dependent sub-optimality bound for offline RLHF is meaningful and the strengths outperform the weaknesses. Therefore, I lean towards acceptance. I will be happy to raise my score if my comments are solved.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The work studies simple regret bounds of offline reinforcement learning with preference feedback. Formally, we are given a dataset generated as follows:
1. A state $k$ is drawn from some fixed distribution $\rho$.
2. Two actions $a_0, a_1$ are picked. 
3. A Bernoulli variable $\sigma$ is generated based on the soft-max of the rewards associated with $(k, a_0), (k, a_1)$.

We want to assign each state $k$ a candidate action $a(k)$ so as to minimize the simple regret defined as $  \mathbf E_{ k \sim \rho } [  \text{reward}(k, a^*(k)) - \text{reward} (k, a(k))  ) ] $, where $a^*(k)$ is the action that maximizes the reward of state $k$.

The main contribution is an algorithm that achieves a simple regret bound of $\exp(- O(n/H))$, where $H$ is some complexity parameter that depends on the instance, and a lower bound construction showing that $\exp(- \Omega(n/H) )$ is necessary.

### Strengths
The main novelty of the algorithm is the design of Locally Optimal Weights. At a high level, the weights are used to weight to what extend one should take into account the information about a datapoint about state $k’$, action $i’$, and action $j’$ while comparing the rewards of action $i$ and action $j$ on state $k$. Moreover, the weights are constructed to minimize a variance proxy for the comparison result so as to enhance the estimation precision. The technique is sophisticated, and shown in experiment to have non-trivially improved the average simple regret bound.

### Weaknesses
The extension to the MDP setting, where one tries to find optimal policies, rather than just actions that maximize per-state rewards, is rather superficial. First, the authors assume that the transition probabilities are known a priori (though this is the case in prior work as well), which significantly weakens the algorithm’s applicability. Besides, there is likely no mention of the computation complexity of their algorithm in this setting. It seems like the algorithm is just doing brute-force to find a policy $\pi$ that maximizes $\mathbf E_{k \sim d^{\pi}}[ \hat r_{k, \pi(k),1} ]$, which is prohibitively expensive.

### Questions
1. How is $d^{\pi}$ formally defined? Is it the stationary distribution under the policy $\pi$?
2. It seems like the consistency requirement is mainly to ensure that the dataset at least contains enough information to compare each pair of actions $i,j$ on a state $k$. Is there a setting where such assumption is relaxed and one instead assumes that there is a joint distribution over state and action tuples so that one can ignore the highly unlikely state-action tuples in the final regret bound?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies an offline preference-based policy learning problem. This work utilizes the locally optimal weights to construct an instance-dependent algorithm. In theory, the instance-dependent upper bound and lower bound are derived. Further, an extension to the data-privacy setting is studied.

### Strengths
1.  The preference-based problem is important in the offline RL community. 
2.  Some theoretical guarantees are established for the proposed algorithm.

### Weaknesses
1. This work considers a linear reward setting, which is more stringent compared to the general reward function setting in the existing literature: a. Chen, Xiaoyu, et al. "Human-in-the-loop: Provably efficient preference-based reinforcement learning with general function approximation." ICML b. Zhan, Wenhao, et al. "Provable offline preference-based reinforcement learning." ICLR. On the other hand, the reward functions are complex and usually require non-linear function approximation. 

2. The reward function is not trajectory-based defined but is just defined on the state-action pairs. The state-action pair setting is inconsistent with the practical applications, particularly natural language processing. See Ramamurthy, Rajkumar, et al. "Is reinforcement learning (not) for natural language processing: Benchmarks, baselines, and building blocks for natural language policy optimization." ICLR. 

3. This paper mentions the underlying problem is a PbRL. However, following careful reading of the setting in Section 2, the problem considered in the paper is essentially a contextual bandit problem. This greatly limits the contribution of the work.

3. The instance-dependent setup is not well explained in the preference-based settings. Particularly, the coverage of the offline data is not discussed in the work. 

4. Definition 2.2: The condition on the consistent instance is strong, which is usually not true in general offline settings. 

5. Equation (6): The clip on the implicit reward is difficult. That is, the bound in the implicit reward is not easy to measure in practice. 

6. The estimation of the locally optimal weight seems very sensitive over the low-coverage samples. Also, the computational complexity is high for estimating the local optimal weight. 

7. Equation (7): the condition of the consistent instance essentially implies good coverage following the definition of the locally optimal weight. 

8. The instance-dependent upper bound seems superficial as compared to the setting in the general offline RL. For example, the dimension of the feature mapping is not in the bound. See Nguyen-Tang, Thanh, et al. "On instance-dependent bounds for offline reinforcement learning with linear function approximation." AAAI.  Some further explanation and discussion on this point are required. 

9. The extension to a privacy-preserve setting is trivial, but the motivation for ensuring privacy in the PbRL problem is not well discussed. 

10. There is a lack of comprehensive empirical studies on the algorithm. The evaluation of the offline PbRL benchmark is necessary. This can better position the paper in the existing literature.  

11. The ablation studies on the instance-dependent upper bound and lower bound in terms of the key parameters are needed. 

12. The presentation of the paper is not clear. There are many notations and some of them are not necessary.

### Questions
See weakness.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces RL-LOW, an algorithm aimed at minimizing simple regret in offline reinforcement learning (RL) with preference feedback. 

The paper claims that RL-LOW achieves tight instance-dependent bounds, offering theoretical guarantees beyond existing minimax approaches. 

Additionally, the algorithm is extended to a differential privacy setting, maintaining its instance-dependent guarantees.

### Strengths
1. **Instance-Dependent Bounds**: 

The algorithm’s ability to achieve instance-dependent error bounds is a major strength. This focus provides finer theoretical guarantees compared to minimax bounds.

2. **Theoretical Contributions**: 

The paper does a thorough job of proving both upper and lower bounds for the regret, providing order-optimality guarantees in terms of the hardness parameter $H$.

### Weaknesses
1. **Limited Practical Relevance of Instance-Dependent Bounds**: 

While the instance-dependent bounds are novel, I feel they are disconnected from practical applications. In many real-world scenarios, problem instances aren’t clearly characterized in terms of hardness, which limits the utility of these bounds. It is also unclear how people could use these refined bounds to benefit practice.

2. **Some parts Hard to Understand**: 

The paper suffers from clarity issues throughout. Key concepts, such as the distinction between the instance-dependent and worst-case bounds, are not well-explained. For instance, the introduction of the hardness parameter $H$ is not intuitive, and sections like Definition 2.2 on consistent instances are particularly difficult to digest. More explanation or discussions would benefit understanding.

3. **Experiments Lack Validity**: 

The experimental validation is weak. The simulations are based on a synthetic dataset, and while this is standard for theoretical work, the lack of real-world testing makes the results less convincing. Furthermore, the comparisons with PESSIMISTIC MLE are only on narrow setups, leaving questions about the general applicability of RL-LOW.

4. **Unclear Motivation for Differential Privacy**: 

The introduction of differential privacy seems tacked on and not well-motivated. There’s little discussion on why privacy is critical for this particular problem setting, and the paper introduces the concept late in the related work section rather than making it part of the core problem setup. This reduces the relevance of the extension. Furthermore, it is also unclear whether the bound under different privacy (in Theorem 5.3) is tight or not in terms of $\epsilon$ and $\delta$.

### Questions
1. **BTL Model Assumptions**: 

The Bradley–Terry–Luce (BTL) model used for preference feedback is quite limited in terms of modeling real human preferences. To what extent could the assumptions of this model be relaxed, and what impact would this have on the results?

2. **Algorithmic Novelty vs. Analytical Refinement**: 

Is the prior theoretical guarantee of RL-LOW due to its algorithmic design or a more refined analysis? A comparison of RL-LOW’s performance with that of PESSIMISTIC MLE under the same conditions would help clarify this point.

3. **Clarification on Assumptions**: 

Definition 2.2 needs further explanation, which appears somewhat confusing. This definition relies on the collected data $N_{k,i,j}$, which is random, while the concept of a consistent instance seems to be a property of the feature $\phi(k, i)$. This suggests that for a given feature $\phi(k, i)$, it may sometimes be consistent and other times not, depending on the data. Could the authors clarify why this assumption is necessary and provide more insight into its implications? Additionally, it would be helpful to understand how consistency is determined in practice.

4. **Clarification on "Instance-Dependent" Bound**: 

The so-called instance-dependent bound appears more like a "gap-dependent" bound, as it heavily relies on the suboptimality gaps between actions. This raises the question of how the results would change if these gaps did not exist or were negligible. In such cases, would the proposed algorithm still achieve meaningful bounds, or would its performance degrade? Could the authors explain how the proposed method would handle instances where the suboptimality gaps are small or non-existent?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper studies offline reinforcement learning with preference feedback under the Bradley-Terry-Luce Model. There is a known feature function \phi that maps state-action pairs to a vector embedding such that the preference of two actions is determined by a linear function of their feature through a sigmoid transform.

The offline dataset have fixed s,a1, a2, but the label is random (from the BTL model). The target state distribution is \rho.  The offline dataset appears to be fixed for s, a1,a2 but have random Boolean label (preference).

The main claims are: (1) An algorithm that enjoys a gap-dependent bound that decays exponentially in the number of samples n. (2)  A lower bound showing that the rate of decay (on the exponent) is optimal up to a constant. (3) A label-differentially private algorithm with appropriate privacy and utility bound.

### Strengths
Strengths

-	The results appear to be new. 
-	The presentation of the technical results is clear.
-	The logarithm of the upper and lower bounds of the suboptimality match up to a universal constant. The weak notion of optimality (on the exponent) may appear to be esoteric, but it is appropriate in cases when the error bound is exponentially small.
-	In a specialized setting, the result knocks out a log n factor in the worst-case expected “simple regret” of Zhu et al. (2023).

### Weaknesses
Weaknesses:

-	The model of offline RL being considered in the paper is crudely simplified. The target state distribution \rho is fixed rather than controlled by the chosen policy.  The policy only controls the actions.  This setting should be called “Offline Contextual Bandits under distribution-shift” rather than offline RL.  A generalization is given in Section 6 but still requires the transition dynamics to be known (still quite far from a typical offline RL setting).

-	The whole point of assuming a linear feature model in BTL model is that the learning bound does not depend explicitly on the cardinality of the state space and that the model can learn without seeing all states, i.e., the learner can generalize to unseen states.  The setting in this paper assumes linear features, but also requires state/action spaces to be finite and that the distribution samples each state with non-zero probability.  For example,  Zhu et al's bound depends polynomially in d.  When S = {0,1}^d,  this is log S. I personally think that it is more natural if the authors just focus on the tabular case (with \phi being the trivial one-hot feature).

-	Related to the above, the claim on computational efficiency is only valid when S and A are finite and small.  Typical settings in RL with function approximation require dependence on log S to be considered efficient. 

-	The fourth strength above (Proposition 3.4 compared to Theorem 1.1 of Zhu et al 2023) is not a fair comparison.  In the setting of Zhu et al, there is no dependence on the cardinality of the state space.  The “coverage” is described by the properties of the feature covariance matrix.  The results of this paper require all states (in the support of target distribution \rho) to have many observations to be non-vacuous.

### Questions
*These are not quite questions, but comments that I think may help the authors.*

-	I would suggest that the authors not just focus on the dependence on n, but more thoroughly inspect the “coverage”, i.e., how the discrepancy between N and \rho affects the learning bounds.  This is well-studied in offline RL (non-preference feedback), e.g., https://proceedings.mlr.press/v97/chen19e.html  https://arxiv.org/abs/2106.04895  https://arxiv.org/abs/2203.05804  


-	Since the setting is label-DP, e.g., in the “Our Contributions” setting, it should be appropriately referred to “label-DP”.  This is in contrast to existing studies on differentially private reinforcement learning that considered individual trajectories to be the contribution of a user (to be protected).  I agree that label-DP seems natural in the RLHF setting. 

-	More discussion on the following body of work will help the paper to situation its novel contribution in a broader context:

1.	Gap-dependent analysis for Dueling Bandits, e.g.,  https://proceedings.mlr.press/v130/yang21a/yang21a.pdf 

2.	Contextual Dueling Bandits (and many follow-up work, e.g., https://proceedings.neurips.cc/paper/2021/file/fc3cf452d3da8402bebb765225ce8c0e-Paper.pdf who talked about large-gap as an open problem) https://proceedings.mlr.press/v40/Dudik15.html 

3.	Gap-dependent bound for offline RL is well-studied but not cited (a).	Tabular setting:  https://arxiv.org/abs/2206.00177  (b).	Linear function approximation https://arxiv.org/abs/2211.13208 

Note that in those cases even if the rewards are known, a 1/n rate is required in the lower bound. The main difference is that the current paper assumes away any planning and does not measure the regret in terms of the value function.

### Soundness
4

### Presentation
3

### Contribution
2
