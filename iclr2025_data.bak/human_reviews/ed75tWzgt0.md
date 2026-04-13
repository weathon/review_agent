## Human Reviewer 1

### Summary
This work introduces a theoretical two-player zero-sum game framework for reinforcement learning from human feedback (RLHF), where the max-player approximates the Nash equilibrium and the min-player approximates the best response by each choosing their own optimistic rewards. The theoretical algorithm achieves $\tilde{O}(\sqrt{T})$ regret in linear environments. It also proposes two practical algorithms: a two-agent algorithm TANPO and a single-agent algorithm SADPO to approximate TANPO. These algorithms outperform several baseline methods in experiments.

### Strengths
1. This paper gives a theoretical sound result for self-play style RLHF algorithm, achieving a sub-linear regret in linear two-player zero-sum games.
2. The algorithms have better performance than baselines as shown in experiments.

### Weaknesses
1. The discussion on data diversity is not persuasive enough.
2. The motivation of using a single-agent approximation is not discussed.
3. As usually Nash learning results do not assume BT model or other transitive preferences, this work only focuses on BT model, lacking discussions on non-transitive preferences.
4. It would be better if the authors can plot a curve of regret to show that it is $\tilde{O} (\sqrt{T})$ (even if in toy environments).

### Questions
1. On top of page 4, why is $\mathcal{R}$ defined on two actions instead of one action as in Equation (3)?
2. Is the final step of Equation (7) $V_r (\pi^t, \mu)$ instead of $V_r (\pi^t, \dagger)$?
3. What is the sampling strategy of online DPO?
4. Regarding data diversity: Why does larger $|\log \pi_\text{ref} (a^1) - \log \pi_\text{ref} (a^2)|$ reflect higher data diversity? In my opinion, diversity is about the **overall distribution** of the sampled responses (more like $D_{KL} (\pi_\text{ref} (\cdot | \text{TANPO}) || \pi_\text{ref} (\cdot | \text{online DPO}))$), instead of the difference in a **local pair**.
5. Regarding single-agent approximation: What is the motivation of using a single-agent approximation? Is it more time and space efficient than two-player version?
6. Previous Nash learning results do not assume transitive preferences. Can your results applied to non-transitive preferences?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper studies provably efficient self-play training algorithms of RLHF, which aims to 1) derive theoretical guarantees for the self-play framework and 2) improve with active exploration. The authors propose both an easy-to-implement two-agent algorithm and its single-agent version. Numerical results are provided to demonstrate the effectiveness of the proposed algorithms.

### Strengths
The proposed algorithms show improvements to the baselines in the numerical experiments.

### Weaknesses
1. The self-play framework is motivated by general preference models, while this paper limits to the standard BT reward model;
2. Confusion and (elementary) mathematical typos appear in critical definitions and analysis, indicating the paper's current version may not be ready for publication (see Questions below).

### Questions
1. Policies $\pi$ and $\mu$ are fully decoupled in the value function (3), and the minimax value is always zero, which makes the use of Nash equilibrium as a solution concept quite confusing;
2. There are typos in deriving the second equations in both (5) and (7). For instance, $\arg\max_\pi\min_\mu V_{\hat{r}_t^1}(\pi,\mu)$ in (5) is always zero while the RHS is non-zero. Additionally, there is a typo in Eq.(12).
3. A minor suggestion on writing: In Section 2, I think it would be better to introduce "Two-Agent Zero-Sum Games" in the context of the self-play training framework.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes a new self-play RLHF algorithm. It effectively balances the trade-off between exploration and exploitation.  In TANPO, two players are trained using different loss functions to ensure more diverse and informative data collection. And, in SADPO,  a single-agent approximation of  TANPO, which is supported by both theoretical analysis and empirical evidence. Experiments are conducted on multiple evaluation benchmarks, including AlpacaEval 2.0, MT Bench, and PairRM. It is a good paper.

### Strengths
This paper proposes a new self-play RLHF algorithm. It effectively balances the trade-off between exploration and exploitation.  In TANPO, two players are trained using different loss functions to ensure more diverse and informative data collection. And, in SADPO,  a single-agent approximation of  TANPO, which is supported by both theoretical analysis and empirical evidence. Experiments are conducted on multiple evaluation benchmarks, including AlpacaEval 2.0, MT Bench, and PairRM.

### Weaknesses
No

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The submission investigates a game theoretic approach to RLHF to increase data diversity.

### Strengths
RLHF is an important problem.

---

Review summary: I'm not an expert in RLHF, so I'll defer to other reviewers for the strength of the submission along that axis. But as far as its game-theoretic approach, the submission seems a bit confused, as articulated in the weaknesses section. Still, RLHF is an empirical field, so the its empirical results (about which I am not qualified to speak) shouldn't be underweighted.

### Weaknesses
> the max-player aims to maximize the summation of (i) the expected Nash equilibrium value function and (ii) the negative estimation loss of that reward function. Similarly, the min-player seeks to maximize the summation of (i) the expected best response value function based on the max-player’s strategy and (ii) the negative estimation loss of that reward function.

These sentences doesn't make sense:
1. Value functions are already expected values.
2. Both the Nash equilibrium value function and the best response value function are a unique functions mapping from decision points to values. They are not an objective that a player can maximize.
3. The submission's use the language "that reward function", indicating a reference back to a previously mentioned reward function. But there is no such previously mentioned reward function.

---

As a aesthetic matter, "Two-Agent Nash Policy Optimization" is a pretty tasteless name. There is a large literature of policy-based algorithms for computing Nash equilibria in zero-sum games.

---

> In the Nash equilibrium, the max-player’s strategy and the min-player’s strategy are mutual best responses, meaning each is optimal given the strategy of the other (Nash et al., 1950).

-> given the other

---

> where V (π, µ) is a general function that captures the payoffs based on the strategies π and µ.

"general" is superfluous here.

---

The submission's usage of regret, while not wrong, is a bit atypical for games. Rather than quantifying a player's external regret over the iterations actually played, it quantifies that player's exploitability. Under this usage of regret, the relationship between sublinearity and convergence to Nash is less fundamental than it is games literature. Specifically, while in games literature, sublinearity is necessary for average iterate convergence, here it is just quantifying convergence speed.

---

I don't understand the motivation of setting up the equation (3) as a zero-sum game. The objective the submission describes is additively factorable across players, so you can just drop all the min terms and maximize to find the Nash equilibrium.

---

> To simplify this setup, we propose the Single-Agent Diversity driven Policy Optimization (SADPO) algorithm as a single-agent approximation of TANPO. The SADPO optimization objective is similar to min-player objective (13).

If we just care about data diversity, why bother with a game theoretic setup in the first place?

---

I don't understand Table 3. Why are there min players and max players on both axes?

### Questions
I included some in the weaknesses section above.

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
3

### Confidence
3