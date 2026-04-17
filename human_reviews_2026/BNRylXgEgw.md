# Provable Policy Optimization for Reinforcement Learning from Trajectory Preferences with an Unknown Link Function

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
The link function, which characterizes the relationship between the preference for two trajectories and their cumulative rewards, is a crucial component in designing RL algorithms that learn from preference feedback. Most existing methods, both theoretical and empirical, assume that the link function is known (often a logistic function based on the Bradley-Terry model), which is arguably restrictive given the complex nature of preferences, especially those of humans. To avoid mis-specification, this paper studies preference-based RL with an unknown link function and proposes a novel zeroth-order policy optimization algorithm called ZSPO. Unlike typical zeroth-order methods, which rely on the known link function to estimate the value function differences and form an accurate gradient estimator, ZSPO only estimates the sign of the value function difference. It then constructs a parameter update direction that is positively correlated with the true policy gradient, eliminating the need to know the link function exactly. Under mild conditions, ZSPO provably converges to a stationary policy with a polynomial rate in the number of policy iterations and trajectories per iteration. Empirical evaluations further demonstrate the robustness of ZSPO under link function mis-specifications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a zeroth-order optimization algorithm for preference-based RL with an unknown link function. The algorithm updates the policy parameters toward a direction that increases the value function by using random perturbations and majority votes. The authors provide convergence guarantees to a stationary point and experiments demonstrating that the algorithm outperforms baselines when the link function is misspecified.

### Strengths
- The proposed algorithm is simple to implement and directly optimizes the policy without relying on a reward model, similar to DPO and NLHF.

- The theoretical analysis is thorough and mathematically sound.	

- The empirical results are also promising: the algorithm performs better than baselines that assume a BT model under misspecified settings, over all three environments.

### Weaknesses
- The motivation could be stronger. The authors could further discuss why it is important to consider broader classes of link functions beyond the BT model, given that the BT model performs well empirically and has theoretical justification through the Borda rule in social choice theory.
- Moreover, while the linear link function is used to demonstrate misspecification in experiments, it is unclear how practical this model is. It is also unclear how much performance degradation occurs in the proposed algorithm (compared to others) if the true link function were sigmoid.
- Scalability to high-dimensional settings such as LLMs is unclear. It seems unlikely to me that the proposed approach would scale efficiently.
- The bound converges to 0 as $D \rightarrow \infty$, but practically it is difficult to use large $D$ in online settings.

### Questions
- In line 161, the global optimizer $\theta^*$ is defined but never used. It seems that no theoretical results establish convergence to a global optimizer.
- In experiments, what value of $D$ was used? I only found $D=1$ in the stochastic gridworld example.

### Soundness
3

### Presentation
4

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
This work extends the standard Bradley–Terry model for preference-based RL by proposing a method that does not assume knowledge of the link function.
The core algorithm, Zeroth-Order Sign Policy Optimization (ZSPO), achieves independence from the explicit link function by only estimating the sign of the value difference. The authors theoretically establish ZSPO's convergence to a stationary policy under smoothness and distinguishability assumptions. Empirical evaluation on classic control benchmarks shows the method's strong robustness when the link function is misspecified.

### Strengths
•  Addresses a critical, well-motivated problem by successfully generalizing preference-based Reinforcement Learning to remove the dependence on a known link function. 

•  The paper features a clear and insightful articulation of the deep connection between dueling bandit theory and preference-based RL, which helps situate the proposed method within the broader theoretical landscape.

### Weaknesses
• The core theoretical framework and results appear to be a minor adaptation of the Zhang & Ying (2024)'s ZPG framework, rather than offering significant conceptual originality.
-	Zhang, Qining, and Lei Ying. "Zeroth-order policy gradient for reinforcement learning from human feedback without reward inference." arXiv preprint arXiv:2409.17401 (2024).
• The reliance on a majority-vote mechanism requiring an extremely large number of rollouts (e.g., $N=1000$) raises serious concerns about practical inefficiency and high computational cost, making it potentially infeasible for realistic RLHF applications.
• The comparison across algorithms is questionable by the use of different preference oracles (e.g., BT versus linear) for each method. This weakens the direct comparability of the empirical results.
• Although the experiments demonstrate empirical performance improvement and strong robustness to link-function misspecification, the paper provides no corresponding theoretical analysis to explain the underlying mechanism. Specifically, it is unclear how the proposed method mitigates optimization bias or guarantees convergence stability under such misspecification.

### Questions
•  Can the authors clarify what theoretical elements are genuinely novel beyond those inherited from ZPG?
•  How does the majority-vote approximation error behave in smaller N regimes (e.g., N=32,64) relevant to real RLHF scenarios?
•  Why are different preference oracles used for different algorithms? Wouldn’t this lead to unfair comparisons?

### Soundness
3

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
5

### Summary
This paper addresses the problem of **preference-based reinforcement learning (PbRL)** under the assumption that the **link function (σ)** — which defines the relationship between trajectory reward differences and preference probabilities — is **unknown or potentially misspecified**.

To tackle this, the authors propose **ZSPO (Zeroth-Order Sign Policy Optimization)**, an algorithm that enables provable policy improvement without explicit knowledge of σ.

ZSPO estimates only the **sign of the value difference** between two policies and determines an ascent direction based on this binary (±1) signal.

The method thus relies solely on **directional information** instead of full reward magnitude or an assumed link model, effectively leveraging **1-bit preference feedback** for policy updates.

Notably, while ZSPO is described as a **zeroth-order, gradient-free** algorithm, it is not entirely gradient-free in implementation.

The update rule still follows a **gradient-ascent-like structure**, where the estimated sign signal acts as a coarse directional proxy for ∇θV(πθ).

In this sense, ZSPO avoids explicit gradient computation or backpropagation but retains the overall framework of gradient-based policy optimization.

Theoretically, the paper provides convergence guarantees for ZSPO under general MDP settings, showing that the expected policy gradient norm decreases polynomially with respect to the number of iterations and trajectory comparisons.

Empirically, across CartPole-v1, HalfCheetah-v5, and Hopper-v5, the method demonstrates strong robustness under **link function misspecification** (e.g., true link linear vs. assumed logistic), outperforming baselines including RM+PPO, Online DPO, ZPG, and Evolution Strategies in both stability and final performance.

### Strengths
1. **Novel Theoretical Formulation** — Unlike most prior works in preference-based RL that assume a known link function (typically the logistic Bradley–Terry model), this paper explicitly formulates the problem under *unknown link functions* and provides the first formal convergence guarantee in this setting.
2. **Simple but Powerful Algorithmic Insight** — The paper shows that effective policy improvement is achievable using only the **sign** of value differences, discarding magnitude information. This leads to a minimal yet expressive framework for **1-bit feedback–based policy optimization**.
3. **Mathematical Rigor** — The analysis is well-structured, offering explicit convergence rates (Theorem 1 and Corollary 1), a clear definition of the distinguishability constant ε\*_D, and an interpretable sample-complexity bound. The proof framework extends Lyapunov-drift analysis into a zeroth-order optimization context, which is an elegant theoretical contribution.
4. **Empirical Persuasiveness** — The experiments systematically simulate link-function misspecification across three Gymnasium control environments, demonstrating that ZSPO remains consistently robust compared to all baselines.
5. **Clear Relation to Prior Work** — The paper effectively situates ZSPO within the landscape of existing preference-based RL methods such as DPO, ZPG, ES, and RM+PPO, providing a fair and transparent comparative evaluation.

### Weaknesses
1. **Complexity of Online Preference Collection** — The algorithm requires multiple batch-level trajectory comparisons and majority-vote queries per iteration, which may be impractical or expensive in real human-in-the-loop systems.
2. **Discontinuity in Policy Updates** — Since the update direction is derived from 1-bit sign feedback, it can be highly sensitive to noise. Although convergence is theoretically ensured, the variance of updates may be substantial in practice.
3. **Limited Experimental Scope** — All experiments rely on synthetic preference oracles; no evaluation with real human or LLM feedback is included, which limits claims of real-world applicability.
4. **Baseline Comparability Issues** — Some baselines differ in setup: RM+PPO uses semi-offline reward modeling, while DPO includes KL regularization. Hence, the comparisons are not entirely controlled.
5. **Lack of Qualitative Analysis** — The paper could benefit from a visualization or sensitivity study showing how different link-function shapes (e.g., steep, flat, linear) influence learning dynamics.

### Questions
1. **Practical Scope of Link-Function Uncertainty** — In what real scenarios (e.g., human crowd evaluations, noisy LLM feedback) does the “unknown link function” assumption most accurately apply?
2. **Stability of Policy Updates** — Since 1-bit sign gradients might cause oscillation or overshooting in policy space, have you explored adaptive step-size or momentum strategies to mitigate instability?
3. **Human-in-the-Loop Applicability** — When extending to actual human feedback, how could the number of comparisons per iteration (N, D) be reduced? Would active preference sampling or uncertainty-driven querying help?
4. **Partial-Information Extensions** — If partial knowledge of the link function is available (e.g., monotonicity or σ′(0)), can the method be modified to accelerate convergence beyond the purely sign-based approach?
5. **Policy-Label Feedback (Relation to PPL Work)** — How might ZSPO adapt if behavior-policy labels are provided, as in online PPL settings? Could incorporating such distributional information improve gradient-direction accuracy?
6. **Comparison with Pebble’s Online PPO** — The Pebble framework (Wirth et al., 2021; Wang et al., 2023) also conducts online preference-based learning using PPO within a fully interactive loop. Given that RM+PPO in this paper follows a semi-offline setup, a comparison with Pebble’s *online* PPO (which shares the same data collection and update loop structure) could provide a more direct empirical baseline.
    
    Could the authors clarify why this comparison was not included, and whether differences in data collection protocol or link-function assumptions made it infeasible?

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
4

### Summary
The paper presents a zero order preference based algorithm that does not require the specification of the link function, i.e., the function that connects the preference with the underlying rewards. Common algorithms use the Terry-Bradly model (a sigmoid) which can limit the performance of these algorithms in the case of preference misspecification as argued by the authors. The algorithm is a zero-order algorithm that directly searches in the parameter space of the policy. It chooses a random update direction and then computes via the preference feedback whether this direction improved performance or not. The main contribution of the paper is the theoertical analyis showing convergence bounds in terms of the gradient norm of the algorithm. Moreover, the algorithm is compared against baselines on 3 continuous control tasks.

### Strengths
- the theoretical analysis seems sound (but I could not check every detail)
- The new algorithm has proofable convergence
- Using preference-based RL without specification of the link-function has not been studied so far in the literature

### Weaknesses
- The assumptions made by the algorithm are very far from beeing practical. The algorithm requires that we generate D trajectories for both policies N times in order to compute a single gradient update. In practice, it will be very hard for humans to compare D trajectories. Most practical RLHF algorithms compare trajectories instead of policies, i.e., they can learn from the comparison of single trajectories instead of a batch of trajectories.
- The black-box manner of the algorithm also brings severe limitations (which is however also comparable to the recent ZPG algorithm). As the preference compares policy parameters and not trajectories, it is for example very hard for the algorithm to take random initial states into account. This will only work if we massively increase the number of trajectories D in the comparison. 
- The experimental evaluation is not convincing. It looks very noisy. Its unclear how many seeds have been used, but from the plot it seems way too small to make a proper statistics. Authors should use at least 10 (better 20) seeds to get better statistics
- The experimental setting is also not fully clear to me. Are the baselines evaluated in a similar manner then the algorithm (use a batch of trajectories for the preference comparison) or are they applied to single trajectory pairs? Algorithms such as DPO are not black box, so they can leverage single trajectory comparisons in a much more straightforward way. It is an unfair comparison if these algorithms are evaluated only on preferences on trajectory batches as they do not have the same limitations as the presented algorithm. 
- More ablations should be performed. For example, it would be insightful if the derived bounds hold at least approximately (i.e. by showing how the performance changes with number of comparisons N and batch size D). It would also be good to show experiments with different ground truth link functions, in particular, if the ground truth link function is indeed the terry-bradley model, how would the algorithm perform against the baselines that use the terry-bradley assumptions.

### Questions
- Please specifiy what exactly a "linear link function" is (used for the experiments)
- The link function is not formally defined in the beginning. That would help the understanding
- It would be helpful to see the number of trajectories or number of samples on the x axis in the results instead of the number of iterations
-

### Soundness
3

### Presentation
3

### Contribution
2
