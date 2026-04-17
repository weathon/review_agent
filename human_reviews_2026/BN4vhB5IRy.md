# Exploration for Deployment-Efficient Reinforcement Learning Agents

- Decision: Reject
- Scores: 2, 4, 4, 4, 2

## Abstract
Reinforcement learning (RL) provides a rich toolbox with which to learn sequential decision making policies. Notably, the ability to learn solely from offline interaction data has been a highly successful modality for training real-world policies. However, a gap exists in this paradigm when the offline dataset does not cover all the behaviors necessary to extract optimal policies. Naively, one can pre-train a policy using offline RL and fine-tune it using online RL; this can lead to catastrophe in settings like healthcare and autonomous driving, where deploying an unverified policy is irresponsible. Deployment efficient learning is a potential solution, where the number of distinct data collection policies is relatively low compared to the number of updates to the policy. We argue that safely improving a dataset requires a deployment efficient algorithm with a carefully constructed data collection policy. We introduce a framework with a stationary exploration policy that aims to reduce out-of-distribution uncertainty while maintaining strong returns. We establish theoretical guarantees of this exploration framework without finetuning and demonstrate our method on a large-scale supply chain environment with real-world data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on offline-to-online RL by designing a stationary policy to improve exploration. This paper proposes an exploration policy that aims to reduce the uncertainty in the offline data while staying within the known safe regions. Empirical results on the gridworld and supply chain environment are reported.

### Strengths
(+) The paper is written clearly.

### Weaknesses
(-) The analysis in this paper is not well justified, e.g., Assumption 5.3.

(-) The experiment section lacks standard baseline methods. Only naive baselines are compared.

### Questions
Q1. How is the uncertainty $u$ computed in Def 5.1?

Q2. In Line 208, what are the references for the following statement: "This assumption is also made by most offline RL algorithms using Equation (1) as their objective"?

Q3. Can you provide a comparison with standard offline RL algorithms?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies deployment efficient reinforcement learning, where the number of distinct data collection policies is relatively low compared to the number of updates to the policy. The authors argue that safely improving a dataset requires a deployment efficient algorithm with a carefully constructed data collection policy. They introduce a framework with a stationary exploration policy that aims to reduce out-of-distribution uncertainty while maintaining strong returns. They establish theoretical guarantees of this exploration framework without finetuning and demonstrate their method on a large-scale supply chain environment with real-world data.

### Strengths
1. The setting of deployment efficient RL is important.
2. The paper is technically solid, the proof looks correct.
3. There are experimental results supporting theoretical results.

### Weaknesses
1. The setting is not clear and seems restrictive. It is not clear whether the setting is tabular or continuous. According to the definition of $u(s,a)$, it seems that the state-action space must be discrete, which is very restrictive in real-world applications. Is it possible to extend the method to more general function approximation? How is $u(s,a)$ rigorously defined there?

2. Given Assumption 5.3, Theorem 5.4 seems to be a straightforward result. Bounding the difference between the real MDP and a pessimistic absorbing MDP is a standard approach in various previous works.

3. The selection of the policy $\pi_{exp}^\star$ is according to the standard approach of maximizing the reward (here is the uncertainty measure $u$) with a KL constraint. Could you please explain the novelty here?

### Questions
Please see the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies deployment-efficient exploration for offline-to-real RL: when continual online updates are infeasible, deploy a small number of stationary exploration policies to enrich data, then retrain offline. It introduces an uncertainty-weighted, KL-regularized explorer and an MPC variant that plans with a shaped reward  A theoretical bound links performance gaps to optimal-policy visitation of dataset “unknowns,” motivating targeted exploration. Experiments on a toy gridworld and a supply-chain simulator suggest better data collection (and comparable evaluation returns) than simple ϵ-greedy/UCB baselines under limited deployments.

### Strengths
The formalization is reasonable, but the main bound (Theorem 5.4) hinges on strong assumptions (offline optimizer finds the optimal policy for the pessimistic MDP; the KL on visitation replaced by a KL on actions), and the experiments do not fully validate the conditions under which the framework is guaranteed to help. The supply-chain evaluation uses an indirect, multi-simulator protocol with heuristic uncertainty and limited ablations.

### Weaknesses
1. Assumptions behind the bound are strong and under-tested.
Theorem 5.4 assumes the offline learner finds the optimal policy for the pessimistic MDP and then replaces a visitation-level divergence with a policy-level KL. The paper does not empirically probe when these surrogates are tight (e.g., via measuring actual state-visitation drift vs. action KL), nor does it test sensitivity of performance to that approximation.
2. Baselines are weak for the stated setting.
In the deployment-constrained regime, there exist stronger comparators than ϵ-greedy/naive UCB: deployment-efficient MBO (e.g., model-based data collection à la prior work cited), conservative online fine-tuning with verification/shielding, uncertainty-aware behavior cloning with data-selection, or “collect-once” behavior-regularized explorers. The paper cites several lines but does not instantiate competitive versions under the same deployment budget.

### Questions
Metricization of deployment efficiency.
Can you report: (a) number of distinct deployed explorers, (b) trajectories per deployment, (c) per-deployment collect-return and regret, and (d) any safety/constraint violation proxies during exploration?

### Soundness
2

### Presentation
2

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
The paper proposes a deployment-efficient exploration framework for offline-to-online reinforcement learning in safety- or cost-sensitive domains. It defines dataset suboptimality via visits to uncertain state-action pairs, derives a bound linking missing coverage to return loss, and constructs stationary single- and multi-step exploratory policies that stay near the dataset policy while targeting uncertain regions, with theory and experiments to validate effectiveness.

### Strengths
1. The paper gives a clear suboptimality characterization: $J\left(\pi^{\star}, M\right)-J\left(\pi_D^{\star}, M\right) \leq \frac{2 R_{\max }}{(1-\gamma)^2} d_{\pi^*}\left(U_D\right)$, which cleanly isolates "missing states of the optimal policy" as the only thing exploration needs to fix. This is a useful, actionable target for data-collection policies.

2. The exploratory policy is derived from a principled divergence-regularized objective so exploration is explicitly balanced against staying close to a verified baseline policy - exactly what deployment-efficient RL needs.

3. The paper provides two constructive approximations (single-step exponential reweighting and multistep MPC-style planning with an uncertainty bonus) that show the abstract objective can be instantiated in both discrete and continuous/large action spaces without online policy finetuning.

### Weaknesses
1. The key quantity $u(s, a)$ ("likelihood of being outside the dataset") is only heuristically estimated from counts/GMMs; the guarantees rely on it tracking the true unknown set $U_D$, but the paper does not give an error-to-performance translation for misspecified $u(s, a)$.

2. The bound and the construction assume the offline policy $\pi_D^{\star}$ is already optimal on the pessimistic MDP $M_D$; this is a strong assumption that pushes difficulty into the offline learner and is not relaxed in the main theorem.

3. The multi-step exploratory policy requires a learned dynamics model $\hat{T}$ and reward $\hat{r}$; the method does not analyze model-bias accumulation in the planning rollout, so it is unclear how far from the dataset support the MPC variant can safely explore.

### Questions
1. The objective $\max_\pi \mathbb{E}_{d^*}[u(s, a)]-\beta D_{\mathrm{KL}}(\pi \| \pi_D^*})$ is motivated via a visitation-level KL; can the authors formalize when the policy-level KL is no longer a good surrogate (e.g., when dataset-induced state marginals differ a lot)?

2. The suboptimality bound depends on $d_{\pi^*}\left(U_D\right)$, which is not observable. Is there a practical upper bound in terms of the learned $u(s, a)$ that could be monitored to decide when to stop deploying the exploratory policy?

3. For the multi-step version (Eq. (5)), how sensitive is the exploration target to model errors in $\hat{T}$ near the boundary of the dataset support, and can a conservative backup (e.g., HALT transition) be added without breaking improvement?

4. The framework fixes $\pi_{\exp }$ during a deployment window to satisfy verification constraints. Could the analysis be extended to a piecewise-stationary schedule and still retain the same form of the suboptimality reduction bound?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework for deployment-efficient exploration in reinforcement learning—motivated by real-world constraints where policy updates require verification and redeployment is costly. The setting lies between offline and online RL: the agent can collect additional data through limited, pre-approved deployments but cannot update its policy mid-deployment.

The authors suggest learning an offline conservative policy, then constructing an exploration policy that tilts the behavior distribution toward uncertain regions, $\pi_{\text{exp}}(a|s) \propto \pi_D(a|s)\exp(u(s,a)/\beta)$, where u(s,a) is an uncertainty function. The idea is to collect “useful but safe” new data for later offline retraining. They present several theoretical results linking coverage to sub-optimality and conduct experiments in a Gridworld domain and a supply-chain simulator.

### Strengths
The problem setting—safe data collection under limited deployment budgets—is realistic and practically relevant, especially for industrial RL applications. The high-level motivation of optimizing exploration to improve future offline training is a good idea, though this is well studied in the area of reward-free RL (e.g. Jin et al. 2020, Wang et al. 2020, Chen et al. 2022, Wagenmaker et al. 2022, Amortila et al. 2024). The inclusion of "real-world" experiments in a supply-chain simulator demonstrate some potential, though the method used in Section 6.2 does not seem related to the methods proposed in the rest of the paper. 

Citations: 
Jin et al. 2020: https://arxiv.org/pdf/2002.02794
Wang et al. 2020: https://arxiv.org/pdf/2006.11274
Chen et al. 2022: https://arxiv.org/pdf/2206.10770
Wagenmaker et al. 2022: https://arxiv.org/pdf/2201.11206
Amortila et al. 2024: https://arxiv.org/pdf/2403.06571

### Weaknesses
**Theoretical novelty and correctness**

The theoretical developments seem to mostly build on Kidambi et al. 2021 (including the main theoretical bound), but somehow without its formality or correctness. The new developments are sloppy and in parts incorrect. Many equalities are handwaved completely and/or incorrect as stated (e.g. Equation 1: \hat{p} is undefined, not specified which (s,a) is being considered), many approximate equalities $\approx$ that are informal and unclear). The main theoretical result (Theorem 5.4) is almost verbatim a restatement of known results from Kidambi et al., 2021 on coverage-based sub-optimality bounds in for pessimistic offline RL. Fundamental quantities for the method to work, such as a definition for the uncertainty function u(s,a), are left under specified. Since u(s,a) drives the entire exploration method, leaving it abstract makes the method non-operational. In fact, obtaining a proper notion of uncertainty is well-studied and one of the defining challenges in reward-free RL (e.g. Amortila et al., 2024) and bonus-based exploration (Bellemare et al. 2016, Pathak et al. 2017, Pathak et al. 2019, Ash et al. 2022). And as mentioned before, though finding exploratory policies that cover the distribution of $\pi^\star$ is a good idea (since this would allow for subsequent offline RL), this is not a novel idea.

**Experimental limitations**

The experiments are limited to a toy gridworld and a supply-chain simulator, with very weak baselines (greedy and ε-greedy policies). No comparisons are provided to standard offline-to-online RL, reward-free methods, or bonus-based exploration as mentioned above. Without these, it is impossible to judge whether the proposed method adds practical value. Furthermore, the experiments do not seem to actually measure the deployment efficiency of their method. The results themselves seem modest over epsilon-greedy and not accompanied by ablations on the different choices of the uncertainty function. 

Overall, the paper’s motivation is good, but the theoretical development is largely a repackaging of prior results, the new contributions are handwavy and left undefined, and the experiments lack strong baselines, ablations, and a quantitative measure of the claimed deployment efficiency. 

Citations: 
Bellemare et al. 2016: https://arxiv.org/pdf/1606.01868
Pathak et al. 2017: https://arxiv.org/pdf/1705.05363
Pathak et al. 2019: https://arxiv.org/pdf/1906.04161
Ash et al. 2022: https://arxiv.org/pdf/2110.11202
Jin et al. 2020: https://arxiv.org/pdf/2002.02794
Wang et al. 2020: https://arxiv.org/pdf/2006.11274
Chen et al. 2022: https://arxiv.org/pdf/2206.10770
Wagenmaker et al. 2022: https://arxiv.org/pdf/2201.11206
Amortila et al. 2024: https://arxiv.org/pdf/2403.06571

### Questions
- How should one compute the uncertainty function u(s,a) in practice outside of tabular domains?
- What differentiates Theorem 5.4 from prior results such as those in Kidambi et al. (2021)?
- How does your exploration method compare to prior reward-free methods, ensemble disagreement methods, curiosity-driven methods, or count-based exploration on the same tasks?

### Soundness
1

### Presentation
2

### Contribution
2
