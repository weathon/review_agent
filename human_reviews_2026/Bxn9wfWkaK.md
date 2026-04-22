# TRAM: Test-time Risk Adaptation with Mixture of Agents

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Deployed reinforcement learning agents must satisfy safety requirements that emerge only at test time—evolving regulations, unexpected hazards, or shifted operational priorities. Current risk-aware methods embed fixed risk models (typically return variance) during training, but this approach suffers from two fundamental limitations: it restricts risk expressiveness to trajectory-level statistics, and it induces uniform conservatism that reduces behavioral coverage needed for effective deployment adaptation. We propose **TRAM** (Test-time Risk Adaptation via Mixture of Agents), a deployment-time framework that composes risk-neutral source policies to satisfy arbitrary risk specifications without retraining. TRAM represents risk through occupancy-based functionals that capture spatial constraints, behavioral drift, and local volatility—risk types that trajectory variance cannot encode. Our theoretical analysis provides localized performance bounds that cleanly separate reward transfer quality from risk alignment costs, and proves that risk-neutral source training is minimax optimal for deployment risk adaptation. Empirically, TRAM delivers superior safety-performance trade-offs across gridworld, continuous control, and large language model domains while maintaining computational efficiency through successor feature implementation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes TRAM, a test-time risk adaptation method. TRAM first trains a set of risk-neutral policies on source tasks. Then, in the target task, it evaluates these policies based on their value and occupancy-measure-based risk, and finally selects the action from the optimal policy for execution.

### Strengths
- The test-time risk adaptation problem studied in this paper is relatively novel.
- The method is well-motivated and supported by solid theoretical foundations.

### Weaknesses
- Motivation
    - The motivation for studying the problem is somewhat vague. The authors should provide more concrete real-world examples to better justify the importance of studying test-time risk adaptation.
    - In my view, especially for risks associated with safety constraints, test-time risk adaptation may not be the most reasonable approach. The proposed method effectively assumes that the risk for every state-action pair in the target task is known, and then evaluates the risk of policies (with known occupancy measures) accordingly. However, if the risk function of the target task is already known, it would be more straightforward to relabel the data collected from source tasks (used to compute occupancy measures) using this risk function, and then apply offline reinforcement learning directly on the relabeled dataset. Such an approach would likely yield better policy performance.
- Methodology
    - The method description lacks clarity. The paper does not provide sufficient implementation details, either in the main text or the appendix — for example:
        - How is the target-task vector w in the successor feature framework computed?
        - How are successor features implemented within the LLM?
    - Transfer via successor features typically relies heavily on the similarity between the source and target tasks. The authors should describe in detail how the source tasks used to train source policies are designed.
    - The method also depends on the choice of the risk function. It is unclear how one should select appropriate risk functions for different settings — are they provided directly by each target task, or does the framework assume fixed risk functions per task type?
    - The occupancy measures of the source policies are not clearly defined. How are they computed, and how is their accuracy ensured? If the occupancy is learned, could this lead to out-of-distribution (OOD) issues? For instance, if a policy exhibits OOD behavior in the target task, the subsequent occupancy estimation might become inaccurate, leading to compounding errors.
    - The paper appears to model risk as a one-step quantity, i.e., only the immediate risk from taking an action a in state s, while ignoring potential future risks caused by the same action. This could be problematic under safety constraints, where certain actions can lead to infeasible or unsafe future states. It is unclear whether the proposed method can handle such scenarios.
- Experiments
    - The GridWorld environment used in the experiments is too simple, and the Reacher environment is discretized, which weakens the empirical validation. The authors should evaluate the method on more complex and realistic continuous-control environments.
    - The LLM-based experiment is highly under-specified. Key details are missing, such as how occupancy is defined and computed in the LLM setting, how successor features are implemented, how value estimation is performed, and how the comparison with GPT-4 is conducted. The current level of description is insufficient to be convincing.
    - The paper lacks an analysis of factors affecting policy performance, such as how the risk penalization coefficient c influences results, or how the scale and diversity of source policies affect performance.
    - The experimental results do not include information on random seeds, standard deviations, or statistical significance, which makes it difficult to assess the robustness and reliability of the findings.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the problem of deploying reinforcement learning (RL) agents in environments where risk or safety constraints emerge only at test time potentially differing from training conditions. The authors propose TRAM , a framework in which multiple source policies are trained risk-neutrally and then combined at deployment using an additional risk functional to guide action selection.

Overall, this is a well-motivated and theoretically grounded paper with a clear contribution. However, the empirical evaluation is relatively limited. With broader experiments (across diverse environments, datasets, and baselines) and a clearer framework this work could become a significant contribution to safe RL and safety alignment research.

### Strengths
1. Deferring risk modeling to deployment is a compelling and practical idea for domains where safety constraints are uncertain or variable. It reflects real-world conditions where not all safety-critical scenarios can be foreseen during training.
2. By defining risk as functionals over discounted occupancy measures, the approach captures hazards beyond return variance or short-term constraint satisfaction.
3. The provided performance bounds are well-motivated and clarify when and why mixtures of risk-neutral policies can achieve near-optimal risk-sensitive performance.

### Weaknesses
1. Constructing a meaningful risk function over discounted occupancy measures is nontrivial. In safe RL, risk is typically represented as a per-step cost for constraint violation signal, not a function of the occupancy distribution. It remains unclear how feasible or interpretable this is in practice. Is it correct that the occupancy measure is estimated in this framework through neural features of the Q-function approximator?
2. The test-time objective $\tilde{Q}=Q−c \rho$ diverges from the original constrained formulation $max Q s.t. \rho<\delta$. The hyperparameter c (risk weight) critically affects results but is insufficiently analyzed. Sensitivity or tuning strategies are not discussed.
3. It is unclear how the arg max in Eq. 4.2 is computed, whether by sampling or continuous optimization. Since Q may be non-convex over the action space, the maximization could be computationally challenging during deployment.
4. Algorithm 1 selects an action deterministically based on the maximal adjusted Q value. In stochastic or continuous domains, this deterministic selection may be brittle and could benefit from probabilistic smoothing or ensemble weighting.
5. Theorem 4.1: The definition of $pi^*$ is unclear whether it refers to (i) the optimal policy of the constrained problem (Sec. 3.1), (ii) the surrogate penalized form (Sec. 3.2), or (iii) the deterministic maximizer in Eq. 4.2 and algorithm 1. Clarifying which optimal policy the bound compares against is essential for interpreting the guarantees.
6. Section 5.2 (“Continuous Control: Scalability”) is underdeveloped. The description in the main text is too brief to convey insights about efficiency or scalability; most details are relegated to the appendix.
7. For empirical validation, the paper would benefit from evaluation on more complex or high-dimensional envs from safe RL benchmarks, including safe offline RL or policy customization tasks.
8. The paper does not explicitly address potential failure modes or robustness issues, an important omission for safety-oriented work. When might TRAM fail, and under what conditions would baseline methods outperform it?
9. Extension suggestion: The LLM experiment is a strong proof of concept. Extending to VLA (Vision-Language-Action) models for safety alignment would further demonstrate practical relevance, showcasing how TRAM could adapt generalist policies to context-specific safety constraints.

### Questions
(Refer also to the corresponding points raised in the Weaknesses section.)
1. How sensitive is TRAM’s performance to the risk weight c? Was this hyperparameter tuned per environment, and how robust are results across different values?
2. The claim that risk-neutral training is minimax-optimal needs clearer assumptions. How many source policies are required? How realistic is the assumption of no retraining at deployment in safety-critical domains? Any intuitive explanations on this?
3. The LLM alignment experiment lacks sufficient detail on the target task, evaluation protocol, and definition of the risk tolerance. Moreover, the “Risk-free transfer” baseline achieves strong target rewards. What is the risk tolerance setting that makes it inferior to TRAM?
4. For Eq. 4.2, how is the argmax over actions implemented? Is it solved analytically, sampled, or approximated via policy gradient during deployment?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method for risk-aware reinforcement learning to adapt at test time to new rewards without retraining. First, multiple policies are trained on problems with different rewards. Then, at test time, the method selects the action from the mixture of policies with the best reward on the test task. Sub-optimality bounds for the proposed method are derived. The proposed method is evaluated on a grid-world planning, continuous control, and large language model alignment tasks, and results show effective adaptation.

### Strengths
- Guarantees: The paper derives suboptimality bounds for the proposed algorithm. 
- The proposed algorithm is straight-forward to implement: The method consists of selecting the best action from a mixture of pre-trained agents on different training problems, without additional re-training on the target problem.
- Evaluation on diverse tasks: The method is evaluated on different tasks such as grid-world planning, continuous control, and large language model alignment.
- The application to LLMs is interesting: By only involving two source policies, reward hacking is mitigated with reasonable computational overhead.

### Weaknesses
- Mismatch between motivation and considered setting: The introduction discusses challenges like changing dynamics and distribution shifts. However, the method assumes a fixed MDP with only changes in the reward or additional penalized constraints, which contradicts some of the arguments made in the introduction.

- The proposed algorithm is computationally expensive: Multiple MDPs need to be solved at training time, and their solutions stored. At test time, the different pre-trained policies need to be evaluated on the test task, before being ranked to select the best action. The linear setting studied in section 4.2 alleviates some of these limitations, but assuming linear rewards can be restricting.

- Potentially better method in the linear setting: In the linear setting, one could potentially directly train a policy on the test problem, which could result in a better policy at potentially lower computational costs. Comparisons with this approach are missing. 

- Theoretical results are limited: The bounds in the Theorem 4.1 and Corollary 4.2 are functions of the localized reward discrepancy measure. This measure is difficult to evaluate or bound at training time, as it depends on the target optimal policy and reward that are unknown at training time. As a result, the usefulness of these results is limited. The bound in (4.9) assuming a linear reward structure seems more useful, and could potentially be related to the covering number of the weights space to obtain stronger uniform error bounds for the algorithm. Also, it is not too surprising that the error bounds depend on this metric (i.e., that the algorithm will give a good policy if one of the training time rewards $r_j$ is close enough to the target reward $r_T$). Finally, the analysis would also apply to standard RL problems without penalized risk constraints. The analysis does not seem to use particular features of the risk-aware problem, so the insights regarding this specific setting are limited.

### Questions
- What prevents the method and theory to extend to test problems that have different stochastic transition dynamics $p$ and discount factors $\gamma$?
- In the linear setting, is the proposed method better than directly training a new policy on the test task?
- In Table 2, what do values in the column "GPT-4 Win vs TRAM" represent?

### Soundness
2

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
4

### Summary
This paper proposes TRAM, a method for test-time risk adaptation. The approach trains a mixture-of-experts policy under risk-neutral (risk-free) settings and, at test time, selects actions based on adjusted Q-values using occupancy-based risk functionals. Experiments are conducted on three domains, including MiniGrid, Reacher, and a language-model-based decision-making task, demonstrating that TRAM outperforms baseline methods.

### Strengths
- The paper is well motivated and clearly written.
 - Test-time risk adaptation is an important and practically relevant problem.
 - To the best of my knowledge, the proposed formulation of combining mixture-of-experts with risk-adjusted test-time action selection is novel.

### Weaknesses
- While the method is demonstrated on a continuous control task (Reacher), the environment is still low-dimensional (4D state space). It is unclear whether the method can scale to higher-dimensional, more complex domains where computing and evaluating Q(s, a) is substantially more challenging. The computational cost of Q evaluation appears to be a key bottleneck that may limit applicability.
 - The current experimental settings are relatively simple. Demonstrations on more complex environments would be needed to convincingly establish the method’s effectiveness and generality.
 - Additional ablations would strengthen the paper. For instance, evaluating the effect of the number of experts in the mixture, as well as analyzing how expert diversity influences performance, would help provide insight into how and why the method works.

### Questions
- Could the method be evaluated on more complex safety-oriented or high-dimensional environments (e.g., Safety-Gym) to better assess scalability and generality of the method?
 - How does performance vary with the number of experts in the mixture-of-experts model?
 - What criteria should be considered when selecting or training experts to ensure policy diversity and effective risk adaptation at test time?

### Soundness
2

### Presentation
3

### Contribution
3
