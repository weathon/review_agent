# Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning

- Decision: Accept (Poster)
- Scores: 2, 8, 4, 8

## Abstract
We study online adversarial imitation learning (AIL), where an agent learns from offline expert demonstrations and interacts with the environment online without access to rewards. Despite strong empirical results, the benefits of online interaction and the impact of stochasticity remain poorly understood. We address these gaps by introducing a model-based AIL algorithm (MB-AIL) and establish its horizon-free, second-order sample-complexity guarantees under general function approximations for both expert data and reward-free interactions. These second-order bounds provide an instance-dependent result that can scale with the variance of returns under the relevant policies and therefore tighten as the system approaches determinism. Together with second-order, information-theoretic lower bounds on a newly constructed hard-instance family, we show that MB-AIL attains minimax-optimal sample complexity for online interaction (up to logarithmic factors) with limited expert demonstrations and matches the lower bound for expert demonstrations in terms of the dependence on horizon $H$, precision $\epsilon$ and the policy variance $\sigma^2$. Experiments further validate our theoretical findings and demonstrate that a practical implementation of MB-AIL matches or surpasses the sample efficiency of existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors derive second order sample complexity bound in the context of imitation learning where the learner can collect reward free trajectories.
As a result the expert samples and MDP interaction complexity only scales with the maximum reward variance where the maximum is taken over both the reward and policy class and not on the horizon. These are very interesting results.

My low score at the moment is because there seems to me that there is a contradiction between the lower bound in theorem 5.9 and the BC upper bound, which scales only with the expert variance and not with the maximum variance over the policy and reward class.

### Strengths
The analysis relies on elegant techniques and it looks correct to me.

The results are important for the imitation learning literature.

The experiments are convincing.

### Weaknesses
Here are my main suggestions on how to improve the presentation of your work. I think that the most important one is weakness 5 since it identifies a potential technical flow. I am willing to raise my score significantly once I am sure that there is no technical issue there.

1) In Remark 5.6, the authors discuss that in environments where the expert is complex ( ie Pi contains many functions) than the BC bound in Foster et al 2024 becomes suboptimal compared to MB-AIL, claiming in this way that online interactions are beneficial. However, even in the offline case, it is possible to obtain bounds that are independent of $\Pi$ but that depend on the covering number of the $Q$ value class (see [1]). I suggest that the authors add a comparison with this result, and they suggest that it is reasonable to expect that $\abs{\mathcal{R}} \leq \abs{\mathcal{Q}}$ because for a fixed reward function, I can obtain many different $Q$ functions changing policy and dynamics.

2) In the context of Linear MDP, algorithms such as BRIG or FRA-IL[2] can work when observing only the reward features visited by the expert and not the chosen actions. This setting captures as an important example of state-only imitation learning. Similarly, in the context of general function approximation considered by your work, I think that it's enough to observe all the reward functions in the class $\mathcal{R}$ evaluated at the expert state action pairs. Observing the actions seems unnecessary. Pointing this out makes the work even stronger because it would make your algorithm applicable when BC is not (because BC needs to observe the actions).

3) I found Remark 5.5 a bit odd because if expert and dynamics are deterministic, then trivially only one trajectory is needed. Therefore, $1/\epsilon$ bound in this setting is disappointing. Noticed that Foster et al. 2024 attained $\mathcal{O}(1/\epsilon)$ assuming only a deterministic expert but not transitions. Also in Table 2, for the case deterministic expert and deterministic transitions (right bottom cell ), I think that one trajectory is enough for BC because simply copying the actions can guarantee remaining with probability one on the support of states visited by the expert. Can the same result be proven for MB-AIL ?

4) The expert sample complexity of MB-TAIL scales with the maximum variance over the policy class while the BC bound scales with the maximum variance in class. Can you improve the bound to match the variance dependent bound in BC ? Please fix this in Table 1 where this difference is not visible. 

5) This also seems at odd with the lower bound in Theorem 5.9. The authors there state that any online or offline algorithm needs to pay the maximum variance in class. How is this possible considering that the bound for BC scales only with the expert variance. Moreover, how is it possible to prove a lower bound on $N$ for any offline algorithm since by definition an offline imitation learning algorithm uses $N=0$?

[1] Moulin et al. 2025, Inverse Q Learning Done Right: Offline Imitation Learning in $Q^\pi$ realizable MDPs
[2] Moulin et al. 2025, Optimistically Optimistic Exploration for Provably Efficient Infinite-Horizon Reinforcement and Imitation Learning

Minor

At line 358, I think it should $\Pi$ and not $\mathcal{R}$.

### Questions
The lower bound seems related to the construction used in [2] Section 5.3. Could you elaborate on the differences?

Related to weakness 2. Does MB-AIL need to observe the actions or only the values of the reward functions in $\mathcal{R}$ evaluated at the state actions in the expert dataset?

 In the algorithm, how to solve efficiently the problem at Step 7 where the maximization is over both policies and transition dynamics in the confidence set. Are you planning to use extended value iteration? Can this be done in a more computationally efficient way using corresponding exploration bonuses?

Please answer also to my questions raised in the Weaknesses Section.

### Soundness
1

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
3

### Summary
This work takes a theoretical perspective on adversarial imitation learning (AIL), providing results that clarify the benefits of leveraging online interactions and the role of stochasticity in both the expert policy and the environment’s transition dynamics.

### Strengths
This is a well-presented and solid theoretical contribution to AIL. The paper formulates an interesting in-depth analysis, leading to theoretical results that, to my knowledge, were not previously established in the literature.

### Weaknesses
1. It is not fully clear what new practical insights the theoretical analysis provides for the design of MB-AIL. While the bounds are technically interesting, the paper does not convincingly articulate how these results translate into concrete algorithmic guidance or practical improvements.

2. The presented algorithm MB-AIL is derived by decomposing the regret into the error of reward and error of the model. Other studies, not mentioned in the Related Work, have previously leveraged this division in their analysis. Refer to:

[1] Chen Y, Giammarino V, Queeney J, Paschalidis IC. Provably efficient off-policy adversarial imitation learning with convergence guarantees. arXiv preprint arXiv:2405.16668. 2024 May 26.

[2] Shani L, Zahavy T, Mannor S. Online apprenticeship learning. InProceedings of the AAAI conference on artificial intelligence 2022 Jun 28 (Vol. 36, No. 8, pp. 8240-8248).	

Minor: 

A few typos and grammar errors: “stragety” in Line 71, “as an practical example” Line 277, “Despite of this difference” in Line 282.

### Questions
1. What new practical insights do the theoretical results provide for the design of MB-AIL? Are there concrete guidelines that practitioners can draw from the analysis?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how an agent can learn to imitate expert behavior using both expert demonstrations and online interactions, without access to reward signals. The authors propose a new algorithm called Model-Based Adversarial Imitation Learning (MB-AIL). It combines model-based reinforcement learning with adversarial imitation learning by separately learning a reward function from expert data and a world model from online interactions.

The paper provides strong theoretical results. It proves that MB-AIL achieves horizon-free, second-order sample complexity bounds, meaning its efficiency depends on the variance of returns and improves in more deterministic settings. It also establishes information-theoretic lower bounds showing that MB-AIL is nearly minimax-optimal—it uses about as few samples as theoretically possible. Experiments on MuJoCo and GridWorld tasks confirm the theory, showing that MB-AIL matches or surpasses some imitation learning methods in sample efficiency.

### Strengths
1. The paper derives horizon-free, second-order upper bounds for adversarial imitation learning under general function approximation. This is technically meaningful and advances the theoretical understanding of how stochasticity affects AIL.
2. The idea of learning transition models in AIL is natural and well-motivated. Splitting the learning into adversarial reward estimation + model estimation + optimistic planning is elegant and leverages the strengths of model-based RL in a principled way.

### Weaknesses
1. As stated in the introduction, the paper's main goal is to provide a tight characterization of the benefits of online interaction in imitation learning. However, the result is unsatisfying and does not clarify when adversarial imitation learning outperforms BC. The paper claims that online interactions are helpful only when $\log (N_{R}) < \log (|\Pi|)$. In practice, however, it is difficult to compare the complexity of the policy class and the reward class, which are typically neural networks.
2. The lower bound result (Theorem 5.9) appears to be incorrect. If I understand correctly, Theorem 5.9 claims that achieving an approximately optimal policy requires both $\Omega (\sigma^2 \epsilon^{-2})$ expert demonstrations and $\Omega\left(\sigma^2 \cdot \log ^2|\mathcal{P}| \exp (-N) \epsilon^{-2}\right)$ online interactions. However, as shown in Table 1, [Foster et al., 2024] proved that BC requires $\Omega (\sigma^2 \epsilon^{-2})$ expert demonstrations but zero online interactions, which conflicts with Theorem 5.9.
3. The paper provides a practical version of MB-AIL, but this version differs significantly from the original algorithm, creating a notable gap between theory and practice. The original algorithm constructs a version space over transition models and solves a joint optimization problem over both the policy and transition model. The practical version ignores this design entirely—it simply learns transition models using maximum likelihood estimation and then runs RL on the learned model. From this view, such a model-based adversarial imitation learning has been proposed in [1]. 
4. The experimental validation is insufficient to substantiate the paper’s claims that MB-AIL matches or surpasses the sample efficiency of existing methods. The empirical study is limited to a single toy GridWorld environment and three MuJoCo tasks, which do not provide enough diversity or complexity to convincingly demonstrate the generality of the proposed approach. Moreover, the comparisons are restricted to only two adversarial imitation learning baselines, omitting several important and state-of-the-art methods (e.g., [1, 2]). Including these stronger baselines is essential for a fair and comprehensive evaluation. Without broader empirical coverage and more competitive baselines, the experimental evidence remains too weak to support the strong performance claims made in the paper.

References:

[1] Juntao Ren, Gokul Swamy, Zhiwei Steven Wu, J Andrew Bagnell, and Sanjiban Choudhury. Hybrid inverse reinforcement learning.

[2] Gokul Swamy, David Wu, Sanjiban Choudhury, Drew Bagnell, and Steven Wu. Inverse reinforcement learning without reinforcement learning.

### Questions
1. The analysis assumes that the true reward function, transition dynamics, and optimal policy are contained within the function classes (Assumption 3.5). How sensitive is MB-AIL to mild violations of this assumption?
2. Remark 5.6 states that AIL is better than BC when $\log (N_{R}) < \log (|\Pi|)$. Could the authors empirically validate this claim by controlling the complexity of the policy networks and reward networks? 
3. The established theory shows that the variance term plays a key role in AIL. However, the experiments do not validate the role of variance in theory. Could the authors show experiments where the environment or policy stochasticity is systematically varied?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper has proposed MB-AIL, a model-based adversarial imitation learning algorithm (Algorithm 1). Second-order regret/sample complexity upper bounds are provided in Section 5.1 (see Theorem 5.1 and Corollary 5.2), and minmax lower bounds are provided in Section 5.2 (Theorem 5.9). Preliminary experimental results are provided in the appendices.

### Strengths
- Overall, this is a very strong theoretical machine learning paper. To the best of my knowledge, the proposed MB-AIL algorithm is novel. This paper has established both upper regret/sample complexity bounds for MB-AIL and minmax lower bounds. They together justify that MB-AIL is near-optimal in both its use of online interaction and its dependence on expert demonstrations.

- This paper has done a very good job of literature review, as summarized in Table 1.

- Extensive and rigorous discussions on the theoretical results are provided in Section 5. Many of such discussions are thought-provoking.

Overall, I recommend accepting this paper.

### Weaknesses
- It is not clear to me why the authors do not include **any** experimental results in the main body of the paper. Of course, there is a page limit, but I think the authors can rewrite the paper to include at least one experimental result. I think this will make the paper more "balanced" between the theoretical results and the experimental results.

- My understanding is that the realizability assumption (Assumption 3.5) is a major weakness of this paper. In particular, with function approximation, this assumption rarely holds in practical problems. I fully understand that this is a standard assumption made in theoretical papers to ensure the analyses are tractable, and analyzing the proposed algorithm without this assumption can be very difficult. However, I recommend that the authors add some experimental results when Assumption 3.5 is (mildly) violated to justify that the proposed algorithm is robust to (mild) model misspecification. (It is not clear to me if such experimental results already exist in the appendices; if so, please explain.)

- In Theorem 5.1, the regret bound depends on three notions of complexities: the eluder dimension $d_E$, the bracketing number for the model class $\mathcal{N}_{\mathcal{P}}$, and the covering number for the reward class $\mathcal{N}_{\mathcal{R}}$. Please better explain why we need three different notions of complexities to establish this regret bound. The readers should be able to understand the intuitions without reading the proof.

- I recommend accepting this paper, but I do feel that this paper is too "dense" for a conference paper. I recommend that the authors further polish the writing to make it more readable.

- There are still minor typos in the paper. Please make another pass and fix them. Examples:
  - In the first equation on Page 4, $a_{h'}$ should be $a_t$
  - In equations 3.1 and 3.2, the expectation over the initial state is missing.

### Questions
Please try to address the weaknesses listed above.

### Soundness
3

### Presentation
3

### Contribution
4
