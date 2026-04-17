# Taming OOD Actions for Offline Reinforcement Learning: An Advantage-Based Approach

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 4, 2, 4

## Abstract
Offline reinforcement learning (RL) learns policies from fixed datasets without online interactions, but suffers from distribution shift, causing inaccurate evaluation and overestimation of out-of-distribution (OOD) actions. Existing methods counter this by conservatively discouraging all OOD actions, which limits generalization. We propose Advantage-based Diffusion Actor-Critic (ADAC), which evaluates OOD actions via an advantage-like function and uses it to modulate the Q-function update discriminatively. Our key insight is that the (state) value function is generally learned more reliably than the action-value function; we thus use the next-state value to indirectly assess each action. We develop a PointMaze environment to clearly visualize that advantage modulation effectively selects superior OOD actions while discouraging inferior ones. Moreover, extensive experiments on the D4RL benchmark show that ADAC achieves state-of-the-art performance, with especially strong gains on challenging tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The proposed ADAC improves on existing approaches by using an advantage-like function to evaluate OOD actions and modulate Q-function updates.

### Strengths
1. The paper is well-structured, making it easy to follow..
2. It provides a comprehensive evaluation of the proposed method on the D4RL benchmark.

### Weaknesses
1. My primary concern is that the author claims the advantage function as an auxiliary correction for OOD evaluation is the main contribution of the method, essentially a value regularization technique. However, I noticed that the authors also use a BC term as a policy constraint, which seems only loosely related to the claimed approach. Could the authors provide performance results without the BC regularization term?

2. I remain concerned about the validity of $A(a|s)$. Theoretically, the implicit value is trained only within the sample distribution. However, when OOD actions are used, the next state may shift to an out-of-distribution regime, making the use of the value from these OOD states as a correction term questionable. How reliable is this approach?

3. The comparison is made with methods from 2022, but we are now in 2025. I would expect the authors to compare their method against more recent, state-of-the-art methods.

4. There should be an analysis of the sensitivity with respect to $\alpha$ and $N$.

5. To ensure a fair comparison, most researchers use MLP as the network backbone. The authors use a ResNet architecture instead. Could they provide an ablation study to justify this choice?

6. The theoretical error bound is precisely $\| A(a|s) \|_{\infty}$, so if the advantage function is not used, a tighter error bound would result. This could either indicate that the proposed method is ineffective or suggest potential issues with the theoretical analysis.

### Questions
Please see Weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This method proposes a novel advantage-based method for offline RL, which evaluates OOD actions via a novel advantage-like function and uses it to modulate the Q-function update discriminatively.

### Strengths
* This paper finds that the (state) value function is generally learned more reliably than the Q-value function. It uses the next-state value to assess each action indirectly.
* The experiments show that ADAC achieves SOTA performance.

### Weaknesses
* ADAC needs to sample multiple actions from the behavior policy, which may bring more computational burden.
* Why is the V-function generally learnt more reliably than the action-value function?

### Questions
(1) The motivation of this paper appears similar to that of A2PR: “Existing methods counter this by conservatively discouraging all OOD actions, which limits generalization.” Since A2PR is also an advantage-based approach for offline RL, it would be helpful to include a comparison between ADAC and A2PR in terms of performance, as well as a discussion highlighting their conceptual differences.

(2) In Section 5.2, could you also plot the value function or Q-values along the trajectories? This would help better illustrate how the learned value estimates evolve during the rollouts.

(3) The transition and behavior cloning models are implemented as 4-layer and 5-layer MLPs, respectively, which differ from the architectures used in other baselines. Could you provide an additional experiment where all methods use the same model architecture for a fair comparison?


Reference:

Liu, T., Li, Y., Lan, Y., Gao, H., Pan, W., & Xu, X.. Adaptive Advantage-Guided Policy Regularization for Offline Reinforcement Learning. In International Conference on Machine Learning (pp. 31406-31424). PMLR.

### Soundness
3

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
This paper proposes ADAC, an advantage-based method designed to selectively handle out-of-distribution actions in offline reinforcement learning. By introducing an advantage-guided target for critic learning, the approach aims to distinguish beneficial OOD actions from harmful ones, mitigating the over-conservatism of prior offline RL methods. Experiments are conducted on multiple D4RL benchmarks and include visualization studies.

### Strengths
> Comprehensive experimental tasks.

The authors evaluate the proposed method on a wide range of benchmark tasks, and the empirical section includes substantial experimental data.

>Visualization

The PointMaze visualization clearly illustrates how ADAC distinguishes between good and bad OOD actions, providing intuitive insight into the method’s behavior.

>Clear motivation.

The paper provides a reasonable motivation for selectively addressing OOD actions rather than uniformly penalizing them.

### Weaknesses
> Incomplete related work analysis.

The discussion of recent related work is not sufficiently comprehensive. In particular, the paper overlooks recent studies on OOD detection, OOD state/action correction in offline RL. The relationship between ADAC and those works should be analyzed in more depth to clarify novelty.

> Unsubstantiated claim: “state value functions are easier to learn.”

The statement that state-value functions are easier to learn than action-value functions is asserted without theoretical or empirical justification. A more rigorous analysis or empirical evidence is needed to support this claim.

> Outdated baselines and questionable SOTA claim.

The experimental comparison relies primarily on older offline RL baselines (e.g., CQL, IQL, TD3+BC). Recent methods (2023–2025) are missing. Without comparing against these newer baselines, the claim of achieving “state-of-the-art” performance is not convincing.

> Missing Reproducibility Statement.

The paper does not include the required Reproducibility Statement section mandated by ICLR submission guidelines.

> Unexplained hyperparameter choices.

The reason for choosing the number of action samples N=25 during action selection is not explained. The sensitivity to this hyperparameter should be analyzed or justified.

>Insufficient explanation of key formulas.

Several important equations are not clearly explained:

Equation (9): The role of the “Quantile” or expectile component and how it influences the advantage-guided target should be clearly elaborated.

Equation (14): The rationale for introducing the proposed advantage term into the critic update requires a detailed explanation or theoretical motivation.

### Questions
The idea of distinguishing between beneficial and harmful OOD actions through an advantage-based target is interesting and intuitively appealing. However, the paper lacks sufficient theoretical justification, omits key related works, and compares mainly against outdated baselines while claiming SOTA performance. Moreover, some implementation choices and formulas are underexplained.

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
This paper addresses the issues of OOD action evaluation errors and value overestimation caused by distribution shift in Offline RL, and proposes an Advantage-based Diffusion Actor-Critic algorithm (ADAC). Its core idea leverages the characteristic that the state value function is more amenable to reliable learning than the Q-function. Specifically, it approximates the optimal value function from the dataset and introduces a novel advantage function to conduct differentiated evaluation of OOD actions—encouraging beneficial ones while suppressing harmful ones. Additionally, the algorithm incorporates a diffusion strategy to model the action distribution.

### Strengths
The paper is written in a clear and fluent manner, making it highly accessible. Its logic is solid, with rigorous and straightforward theoretical analyses. The experimental results are relatively comprehensive and demonstrate significant effectiveness.

### Weaknesses
1. The baselines provided in this paper are relatively outdated, as they all focus on works published before 2023.
2. One of the paper’s innovations lies in introducing a new calculation method for the reward function. However, the experiments **lack a comparison** between this new reward function and classical ones, making it unclear which part contributes to the improved algorithm performance.
3. The algorithm learns the value function, Q-function, dynamics, advantage function, and policy network simultaneously. Overall, this design is relatively redundant, leading to excessively high algorithm complexity.

### Questions
1. In the calculation of the reward function, 25 actions are selected, but the underlying logic and further analysis for this choice are lacking.
2. If the diffusion model is replaced with an MLP or other network structures, how much performance loss would occur?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Advantage-based Diffusion Actor-Critic, a novel offline RL algorithm aimed at addressing the challenge of OOD actions in offline RL. The key idea is to assess the advantage of actions using a pretrained value function that compares the estimated value of the next state with a quantile-based threshold derived from the behavior policy. The method integrates this advantage into a modified Bellman backup to guide Q-learning and employs a diffusion model as the policy class. ADAC yields strong empirical results across D4RL benchmarks.

### Strengths
- Good writing makes the paper easy to follow
- Simple and effective idea
- Various experiments to support the effectiveness of ADAC.

### Weaknesses
- The advantage-computing method in Eq. 9 seems ambiguous. The original advantage definition is $A(s,a) = Q(s,a)-V(s)$, which evaluates the advantage of action $a$  among other actions in state $s$. However, the advantage in ADAC is calculated based on the next state's value $V(s')$, which is quite different. Meanwhile, if the author needs to show the effectiveness of such an advantage-computing method, an ablation study on this should be conducted.
- More recent offline RL frameworks should be compared, such as [1,2].
- The D4RL benchmark is quite outdated. Could you please provide more experiments on other robotic simulation benchmarks like: LIBERO [3] and ManiSkill [4]?

I would like to raise my score if my concerns are all addressed. 

[1] A2PO: Towards Effective Offline Reinforcement Learning from an Advantage-aware Perspective. NeurIPS 2024

[2] Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization. ICLR 2025

[3] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. NeurIPS 2023

[4] Demonstrating GPU Parallelized Robot Simulation and Rendering for Generalizable Embodied AI with ManiSkill3. arXiv 2024

### Questions
See weaknesses above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ADAC (Advantage-based Diffusion Actor-Critic) for offline RL. The main idea is to evaluate potentially OOD actions by how much next-state value they can reach, using a value function learned by expectile regression as a proxy for the dataset-optimal value. Based on this, the authors define an advantage that compares E_{s’}[V(s’)] under candidate action a to a κ-quantile of values from actions sampled from the behavior policy, and then modulate the critic target with A(s,a) in the Bellman backup. Experiments on D4RL show strong averages across Gym, AntMaze, Adroit, and Kitchen, plus a PointMaze study visualizing that ADAC recovers near-optimal, straight-line paths absent from the dataset. Ablations and a brief efficiency comparison are also provided.

### Strengths
- Paper is well-written and easy to read.
- A novel formulation of an OOD filter. Defining advantage via next-state value relative to a κ-quantile of behavior actions is simple, tunable, and aligns with the claim that V is often more reliable than Q in offline data. The analysis that expectile regression moves V toward a dataset-optimal value is helpful context. 
- Compelling qualitative evidence. The PointMaze visualization clearly shows ADAC stitching suboptimal trajectories and discovering straight-line solutions missing from the dataset.
- Good empirical results. ADAC is competitive or better on most D4RL suites, with large gains on AntMaze, which is where OOD action selection matters most.
- Useful ablations/diagnostics. κ-sensitivity, advantage statistics across tasks, and practical soft-clip for stabilizing extreme A add transparency.

### Weaknesses
- Missing prior work, prior work already links OOD action selection to “optimal next-state value.” - POR (Policy-Guided Offline RL) [1], which trains a guide policy toward optimal next states and uses that signal to permit OOD generalization. Please cite and compare against POR.
- Accuracy of A(s,a) hinges on policy and model constraints. Although A(s,a) is intended to promote good OOD actions, its reliability depends on (i) the transition model producing realistic s’ for actions sampled from \pi since extreme OOD actions can yield poorly extrapolated s’; and (ii) the value function’s generalization on OOD states. Errors in either component can make A(s,a) over-optimistic or over-conservative, weakening the claimed advantage-selection benefit.
- System complexity and tuning burden. The method adds several moving parts: an expectile-trained V, a learned dynamics model, and extra hyperparameters (quantile threshold \kappa, expectile \eta, penalty/weight \alpha, sampling counts). This increases implementation complexity and tuning overhead, which may limit practicality and comparability across domains.
- Reporting/variance. Several results show large standard deviations (e.g., AntMaze large variants in Table 1; Adroit pen tasks), and all scores use only 4 seeds. Strong claims (e.g., “SOTA across almost all tasks”) should be tempered or backed with more seeds. 

[1] Xu et al, A Policy-Guided Imitation Approach for Offline Reinforcement Learning, NeurIPS 2022.

### Questions
1. Why is the A(s,a)-augmented critic provably better than standard TD? Can you provide a formal error decomposition (e.g., showing bias/variance reduction, a tighter fixed-point error bound) that demonstrates Q learned with A(s,a) surpasses vanilla TD under reasonable assumptions? 
2. DQL baseline. If my understanding is correct, removing A(s,a) reduces the method to Diffusion Q-learning with your current infrastructure. Please report the results of DQL baseline under the same codebase, architectures, sampling counts, and training/evaluation settings. This would isolate the key benefit of the proposed advantage term.

### Soundness
3

### Presentation
3

### Contribution
2
