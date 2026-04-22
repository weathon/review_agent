# Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Single-domain offline reinforcement learning (RL) often suffers from limited data coverage, while cross-domain offline RL handles this issue by leveraging additional data from other domains with dynamics shifts. However, existing studies primarily focus on train-time robustness (handling dynamics shifts from training data), neglecting the test-time robustness against dynamics perturbations when deployed in practical scenarios. In this paper, we investigate dual (both train-time and test-time) robustness against dynamics shifts in cross-domain offline RL. We first empirically show that the policy trained with cross-domain offline RL exhibits fragility under dynamics perturbations during evaluation, particularly when target domain data is limited. To address this, we introduce a novel robust cross-domain Bellman (RCB) operator, which enhances test-time robustness against dynamics perturbations while staying conservative to the out-of-distribution dynamics transitions, thus guaranteeing the train-time robustness. To further counteract potential value overestimation or underestimation caused by the RCB operator, we introduce two techniques, the dynamic value penalty and the Huber loss, into our framework, resulting in the practical Dual-RObust Cross-domain Offline RL (DROCO) algorithm. Extensive empirical results across various dynamics shift scenarios show that DROCO outperforms strong baselines and exhibits enhanced robustness to dynamics perturbations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates dual robustness in cross-domain offline RL, addressing both train-time robustness against source-target dynamics mismatch and test-time robustness against deployment-time dynamics perturbations. The authors introduce a novel Robust Cross-domain Bellman (RCB) operator with theoretical guarantees, and develop the practical DROCO algorithm using ensemble dynamics models, dynamic value penalty, and Huber loss. The experimental results across 32 scenarios demonstrate that DROCO achieves good performance.

The authors employ dynamics perturbations of large magnitude, which may diverge from realistic scenarios. More critically, to put the theory into practice, the algorithm adopts a practical scheme that deviates from its core theoretical assumptions: using an ensemble of dynamics models trained on limited target data to approximate uncertainty. To compensate for the potential value estimation errors caused by this deviation, DROCO further introduces a dynamic value penalty and the Huber loss.
 
Although the method demonstrates performance surpassing baselines and enhanced robustness on specific Mujoco tasks, its effectiveness is highly dependent on a set of sensitive hyperparameters that are difficult to tune effectively in a purely offline setting. Furthermore, the experimental scope is confined to locomotion tasks and has not been validated on more challenging benchmarks, which leaves the generalizability and practical value of its claimed "dual robustness" open to question.

### Strengths
The problem formulation is both novel and practically important. While existing cross-domain offline RL methods focus exclusively on train-time robustness, this work is the first to systematically study both train-time and test-time robustness together. The motivation is compelling, with Figure 1 clearly demonstrating that policies trained with limited target domain data are highly vulnerable to test-time dynamics perturbations. This observation reveals a critical gap in current approaches that assume deployment environments will match the target domain exactly.

The theoretical contributions are well-developed and rigorous. The RCB operator elegantly handles dual robustness by applying robust Bellman updates only to source domain data while using standard updates for target data.

### Weaknesses
My primary concern is the insufficient analysis of generalization. Moreover, the experiments are confined entirely to MuJoCo tasks. Maybe authors can consider more experiment for validation. 

The paper's own sensitivity analysis (Section 5.3) showed that the optimal values for β and δ vary significantly across different tasks and datasets. In a real-world offline scenario, it is nearly impossible to tune these parameters to their optimal values for a new task due to the inability to validate against the target environment. The method's claim to practicality is diminished if its strong performance relies on meticulous, per-task tuning that cannot be replicated in practice.

### Questions
See Weaknesses

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
3

### Summary
This paper enhances offline RL by introducing two types of robustness: train-time robustness and test-time robustness. Train-time robustness addresses the dynamic shift between the source-domain dataset and the target-domain dataset, while test-time robustness focuses on the shift from the target dynamics to the deployment dynamics. Considering that (1) deployment dynamics may change in real-world settings and (2) existing methods often fail to generalize well under such changes, I believe this paper makes a valuable contribution.

The proposed method centers around a novel Robust Cross-domain Bellman (RCB) operator, which integrates two types of Bellman operations: (1) the standard Bellman operator for training on the target-domain dataset and (2) a “robust” Bellman operator for policy evaluation on the source-domain data. The theoretical analysis, in my view, can be derived from classic results in robust reinforcement learning.

The theoretical justification mainly concerns the convergence (contraction) property of the proposed RCB operator. I believe that the conclusions for both the idealized and practical cases can be obtained with relatively minor modifications to existing robust RL proofs.

The empirical results (Table 1, on MuJoCo tasks) adequately demonstrate the advantages of the proposed algorithm.

### Strengths
- The theoretical justification is solid, covering both the idealized case (Proposition 4.1) and the practical case (Proposition 4.3).

- The motivation and analysis for both train-time and test-time robustness  (Proposition 4.4 and 4.5) are meaningful and potentially impactful, although their direct relevance to practitioners might be limited.

- The empirical evaluation is convincing. The chosen baselines are sota methods for offline RL and cross-domain offline RL, yet the proposed algorithm (DROCO) achieves significant improvements over them.

### Weaknesses
I do not see any major weaknesses worth highlighting.

### Questions
Q1. Lipschitz Q-function assumption:
Why do you cite recent studies to justify the Lipschitz continuity assumption? If I am not mistaken, this assumption is a standard one in Q-learning and can be found in many textbooks.

Q2. Test-time robustness:
It is somewhat difficult for me to see the benefits of DROCO regarding test-time robustness. Could you please elaborate on this aspect explicitly in the Experiments section?

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
The paper studies dual robustness in cross-domain offline RL. It considers train time robustness to distribution shift between source and target, and test-time robustness to perturbations around the target dynamics. It proposes a Robust Cross-domain Bellman (RCB) operator that performs robust backups (min over uncertainty set) on source data and standard backups on target data. Experiments are performed on D4RL MuJoCo tasks to show performance gains and test time stability.

### Strengths
+ The goal of achieving both train time and test time robustness in offline RL is well motivated.
+ The RCB operator that separates robust and standard updates is simple especially with the duality result which simplifies the uncertainty set of distributions to one over states.
+ The paper gives good empirical results with RL benchmarks showing that the approach outperforms baselines under moderate dynamics shifts.

### Weaknesses
- The robust Bellman backups and the derived contraction properties are standard. As far as I understand, the main idea is to split robust and standard updates, which is conceptually incremental. The paper seems conceptually incremental for ICLR.
- The setup is restrictive as only dynamics shift is modeled. Typically, there is shift in reward, observation, state/action spaces, etc. 
- The theoretical results largely follow directly from known robust RL results. For example Prop 4.1 showing the contraction is immediate for discounted robust Bellman operators. The train time conservatism and lower bound properties (e.g., Prop 4.4) are classic robust RL analyses. Even the test time guarantee of Prop 4.5 arguing that the performance is better than the worst case value when the true perturbation lies inside the set is by construction and standard in DRO. 
- The framework is restricted to Wasserstein distance without extensions to TV/MMD or other divergences.
- It is unclear how to choose or tune the uncertainty radius. Also if the domain shift is large, the approach is likely going to be over conservative given the requirement that the target lies in the uncertainty set. 
- I did not quite understand the value penalty and the Huber loss parts as they are treated superficially without sufficient depth.

### Questions
What is the precise theoretical and/or algorithmic novelty beyond the split backup? Any coverage aware or calibration guarantees that are new? The paper needs better positioning in relation to vast literature on robust offline RL/DRO. 

Can your approach extend to TV/MMD balls (even approximately)? What breaks or becomes intractable?

Any results for reward or state shift to demonstrate generality?

How does performance scale with the amount of target data and epsilon? Where does RCB become too conservative?

The source and target behavior policies are often different. Can you analyze this setting?

### Soundness
3

### Presentation
3

### Contribution
2
