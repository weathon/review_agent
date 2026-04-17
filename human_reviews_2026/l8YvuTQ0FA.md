# Is Pure Exploitation Sufficient in Exogenous MDPs with Linear Function Approximation?

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Exogenous MDPs (Exo-MDPs) capture sequential decision-making where uncertainty comes solely from exogenous inputs that evolve independently of the learner’s actions. This structure is especially common in operations research applications such as inventory control, energy storage, and resource allocation, where exogenous randomness (e.g., demand, arrivals, or prices) drives system behavior. Despite decades of empirical evidence that greedy, exploitation-only methods work remarkably well in these settings, theory has lagged behind: all existing regret guarantees for Exo-MDPs rely on explicit exploration or tabular assumptions.
We show that exploration is unnecessary.
We propose Pure Exploitation Learning ($\texttt{PEL}$) and prove the first general finite-sample regret bounds for exploitation-only algorithms in Exo-MDPs. In the tabular case, PEL achieves $\widetilde{O}(H^2|\Xi|\sqrt{K})$. For large, continuous endogenous state spaces, we introduce $\texttt{LSVI-PE}$, a simple linear-approximation method whose regret is polynomial in the feature dimension, exogenous state space, and horizon, independent of the endogenous state and action spaces.
Our analysis introduces two new tools: counterfactual trajectories and Bellman-closed feature transport, which together allow greedy policies to have accurate value estimates without optimism.
Experiments on synthetic and resource-management tasks show $\texttt{PEL}$ consistently outperforming  baselines. Overall, our results overturn the conventional wisdom that exploration is required, demonstrating that in Exo-MDPs, pure exploitation is enough.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper shows that pure exploration—model estimation with greedy action execution—is sufficient for rate-optimal regret in exogenous MDPs when the exogenous process is the only unknown. The paper covers both the tabular MDP case and the case of MDPs with linear function approximation. In both cases, they show a regret of order $O(\sqrt{K})$.

### Strengths
- The paper is well written and easy to follow. Overall, the authors do a good job of providing intuition about the results and explaining why the algorithm works.
- I am not up to date with the literature on exogenous MDPs, but from what I can gather, the results seem novel and interesting (I did not have time to check all the proofs).

### Weaknesses
- The main weakness of the paper is that the setting is quite specific. In particular, the assumption that the only unknown component is the exogenous process seems quite strong to me. The authors justify this choice as being common in the literature, but I do not have enough knowledge of the literature to judge this claim. However, I can understand that in certain operational research applications, this assumption might be reasonable.

### Questions
- Could you provide some intuition as to why pure exploration is sufficient to achieve rate-optimal regret in this setting? Is it because the endogenous process is fully known, and the exogenous process is independent of the actions, so that it can be learned at a "fast" rate (approximately $1/\sqrt{K}$) without impacting the regret too much?
- In the linear function approximation setting, could you comment on how the regret depends on the problem parameters (such as d, anchor points, etc.)? Is this dependence optimal, or could it be improved?
- The assumption that anchor points are known seems strong. Could you comment on this assumption? Is it possible to relax it?

Overall, I think the paper provides interesting and novel results on exogenous MDPs, even though the setting is quite specific.

### Soundness
3

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
3

### Summary
This paper studies exogenous MDPs where the noise in the transitions is independent on the action taken by the learner.
Under this setting the authors show that no exploration mechanism is needed in Bandits Problem and Tabular MDPs where the transition dynamic is known.
Moreover, the result is extended to the linear case.

### Strengths
I think that the question tackled here is interesting, as several problems of practical interest are usually solved by algorithms that do not use any exploration mechanism.

### Weaknesses
I think that the assumption about the existence of the anchor set is not strong; however, I do think that it is a strong assumption that the learning algorithm knows this set.
Why is it not possible to guarantee the invertibility of $\Sigma$ by defining $\Sigma = \sum^N_{n=1} \phi_n \phi_n^T + \beta I$ where $\beta$ is a small scalar and $I$ is the identity matrix ? In this way, switching from Least Square to Ridge regression it should be possible to avoid the assumption.

Why do you use the anchor points to define the matrix $\Sigma$ instead of using the data collected during the policy rollout phase? It seems weird to me that the rollout phase only uses the encountered exogenous states to estimate the exogenous to exogenous transition matrix. In contrast, the encountered states and actions are discarded.

### Questions
1) How is the initial exogenous state chosen? Is it sampled from a fixed distribution, or can it be chosen adversarially?

2) If the transition dynamics (the $f$) in tabular MDPs were not known, would pure exploitation still suffice?

3) In the tabular guarantees, there is no dependence in the number of endogenous states and actions ($S$ and $A$). However, the feature dimension $d$ shows up in Theorem 2? If I represent a tabular MDP by choosing the features that are one one-hot encoded vector of dimension $d = SA$, I would get a dependence on $\sqrt{SA}$  if I apply your Theorem 2 in this setting, while Theorem 1 avoids this dependence.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper makes a contribution to the reinforcement learning literature attempting to characterize when greedy policies are sufficient for provable guarantees, or when exploration is not necessary. The authors show that this is indeed the case for exogenous MDPs, where there is an exogenous Markovian component that is unaffected by the learner's actions. Their variant of LSVI achieving sublinear regret crucially hinges on an assumption that the endogenous transitions are deterministic and known, as well as three technical assumptions on an anchor set that is closed under the Bellman operator with non-negative residuals. A tabular method only requires the former.

### Strengths
- The paper makes a solid contribution to the RL literature. Understanding exactly when greedy exploitation is sufficient for provable guarantees is a topic that has gotten more attention lately. 
- This tends to occur when there is sufficient environment noise, and Exo-MDPs appear to be one such case. To my knowledge, this is novel.
- The authors tackle bandits and both tabular and linear MDPs, showing that this holds somewhat more broadly than just in an isolated case. 
- The paper is largely well written, save for a few issues on clarity.

### Weaknesses
1. Not much intuition is provided on exactly why such a positive result is possible. At a very very high level that probably amounts to skimming over a lot of subtleties, it seems to be because the learner can decouple the exogenous transitions, of which no exploration is necessary to learn them, from the endogenous transitions, whom are deterministic and known. Assuming that one can do so, the whole problem then reduces to learning the exogenous transitions for input to learning a Q-function via the standard LSVI procedure.
2. The assumptions are quire strong. Needing the existence of an anchor set that is known to the learner, plus additional assumptions on said set, is quite strong in the RL literature (I am reading this as its alternate name, a coreset). It is understandable that strong assumptions are necessary, but one could also (for instance) assume that the minimum eigenvalue of the design matrix is bounded under any policy. Characterizing exactly why this is needed, how it can be obtained (it is folklore that Frank-Wolfe-like procedures get you one, but it should be stated), and the relevant intuition would go a long way towards helping the unfamiliar reader and justifying these assumptions.
3. The proofs in the appendix, especially for the linear MDP section, are quite poorly written. The authors state that they make a counterfactual analysis. It is unclear to me where this is done, or what exactly this means. This is striking -- I've seen the linear MDP proofs many, many times, but it's not clear to me where it happens.

### Questions
1. How much of this relies on the assumption of known deterministic endogenous dynamics?
2. Can the anchor set assumption be weakened? What could replace it?
3. Where is the counterfactual analysis?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a pure exploration learning framework in Exo-MDPs, where the state decomposes into exogenous and endogenous components. Starting from a simple Exo-Bandit warm-up, they show that in tabular Exo-MDPs, one can estimate the exogenous transition kernel and computed the policy via dynamic programming, deriving a theoretical regret bound. For linear function approximation, they introduce an algorithm called LSVI-PE, which combines post-decision state and counterfactual trajectory analysis, and provide a corresponding theoretical analysis.

### Strengths
They present the first near-optimal regret bound for Exo-MDPs under linear function approximation and theoretically establish that it is independent of the endogenous state and action cardinalities. Furthermore, they rigorously prove that the exogenous process in EXO-MDPs evolves independently of the policy, thereby removing the need for explicit exploration. This is supported by $\tilde{\mathcal{O}} (\sqrt{K})$ regret guarantees in both tabular and linear function approximation settings.

### Weaknesses
1. The modelling assumptions are somewhat restrictive, as the theoretical results rely on the exogenous state space being discrete and on the endogenous transition and reward functions being known.
2. The regret bound scales linearly with $|\Xi|$ in both the tabular and linear function approximation settings, which may limit scalability. In particular, LSVI-PE can exhibit degraded performance when the anchor placement is suboptimal and $\lambda_0$ becomes small.

### Questions
Given the relatively small experimental scale—in terms of state/action set sizes, horizon length, and the number of episodes—and that the baseline is limited to an optimism-augmented variant, would it be feasible to scale up the experiments and include additional baselines, particularly other pure-exploitation methods?

### Soundness
3

### Presentation
2

### Contribution
2
