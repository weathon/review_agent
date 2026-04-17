# Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
We study offline reinforcement learning of style-conditioned policies using explicit style supervision via subtrajectory labeling functions. In this setting, aligning style with high task performance is particularly challenging due to distribution shift and inherent conflicts between style and reward. Existing methods, despite introducing numerous definitions of style, often fail to reconcile these objectives effectively. To address these challenges, we propose a unified definition of behavior style and instantiate it into a practical framework. Building on this, we introduce Style-Conditioned Implicit Q-Learning (SCIQL), which leverages offline goal-conditioned reinforcement learning techniques, such as hindsight relabeling and value learning, and combine it with a new Gated Advantage Weighted Regression mechanism to efficiently optimize task performance while preserving style alignment. Experiments demonstrate that SCIQL achieves superior performance on both objectives compared to prior offline methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a method for style alignment in the offline RL setting using implicit Q learning and advantage weighted regression. Styles are defined using hard coded functions which is then used as a reward to learn a style value function. This value function is combined with a task value function (independent of style) to train a style aligned policy. Experiments are conducted on the circle and halfcheetah environments, showing significant performance advantage over baselines such as SORL and SCBC. Ablation experiments demonstrate how different temperature parameters prioritize task performance and style alignment.

### Strengths
* The proposed solution is quite simple and sound.
* The effectiveness of the proposed method is clear.

### Weaknesses
* I think the presentation can be improved if the authors moved some of the plots in the appendix to the main paper.
* Some details in the method can be better explained.
* I find the need to tune the temperature parameter and its sensitivity a downside of the proposed method.

### Questions
* In (12), can you add a text explanation of the equation? Is the gating saying that if the style advantage is high enough such that the sigmoid output is 1 then you can incorporate task advantage? In theory the advantage function at optimality is zero $\max_{a}Q(s, a) = V(s)$, the sigmoid output is 0.5, so you are still using a small weight on the task reward advantage.
* In the results, you did not include an in-depth explanation of the different datasets. Can you explain how you expect the method to behave differently for different datasets? From Table 1, it looks like halfcheetah-vary performs worse on the baseline methods than the other halfcheetah datasets. Why?
* Can you comment on the sensitivity of the temperature parameters?
* I would suggest moving some of the plots in the appendix to the main paper so that people understand what style means.
* (Minor) there are a lot of typos in the paper. Please fix.

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
This paper proposes SCIQL, an offline reinforcement learning algorithm designed to learn policies that optimize task reward while exhibit specific behavioral styles. Building upon IQL, SCIQL extends it to the style conditioned setting and introduces GAWR mechanism to balance the two advantage terms.

### Strengths
The proposed GAWR mechanism and sub-trajectory labeling provide a simple yet effective way to integrate style supervision into offline RL. Empirical results on Circle2D and HalfCheetah environments show that SCIQL consistently achieves higher style alignment scores compared to the baselines.

### Weaknesses
1. The problem formulation is conceptually unclear. If style alignment and task reward are inherently conflicting, the object should be to balance the trade-off between the two. However, the current formulation seems to sacrifice task reward to increase style conformity, which raises the question of whether this trad-off is explicitly modeled.
	
2. Given that style alignment and task reward clearly conflict as shown in Section 5.3, the evaluation might be better framed in a Pareto optimality context rather than using single averaged metrics. Without such discussion, it is difficult to interpret whether improving style alignment at the cost of lowered reward constitute genuine progress.

3. The paper defines style labels as discrete categories obtained via predefined labeling functions. Could the authors clarify why a discrete formulation was chosen over a continuous style ? Using continuous representations might allow smoother interpolation between styles and potentially improve generalization to unseen or mixed style combinations.

4. The evaluation is restricted to toy circle 2d and halfcheetah environments., which are relatively simple and low-dimensional. It would strengthen the work to include results on more diverse environments, such as other MuJoCo or Atari tasks or humanoids-tyle control demands where stylistic variations are more naturally expressed.

5. It would be valuable to assess whether the proposed method can extrapolate (or interpolate) to unseen style labels or novel combinations of style labels that were not encountered during training.

### Questions
1. Is z a multi-dimensional vector aggregating multiple criterion-specific labels, or a single discrete label ? If it is the former, the description around lines 180-190 should be revised to clarify how multiple criterion labels are annotated and used in z.

2. Minor typos. Line 453, Twhile -> while. Line 169, " is reversed.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper propose a new view of the stylized policy learning problem as a generalization of the goalconditioned RL and introduce SCIQL algorithm which uses hindsight relabeling and Gated Advantage Weighted Regression mechanism to optimize task performance.

### Strengths
This paper provides a unified formulation of behavioral style learning via programmatic sub-trajectory labeling, and introduces the SCIQL+GAWR framework that effectively balances style alignment and task performance in the offline RL setting.

### Weaknesses
The reliance on hand-crafted style labeling functions constrains scalability to more abstract or subtle styles, and may require domain expertise when applied to complex environments. The algorithmic pipeline is relatively intricate, increasing implementation burden, and evidence on large-scale real-world or high-dimensional robotic systems remains limited

### Questions
The proposed approach relies on hand-crafted sub-trajectory labeling functions; how scalable and generalizable is this design to tasks where styles are abstract, high-level, or difficult to encode programmatically?

While the method demonstrates strong performance in simulated benchmarks, there is no evaluation on real-world systems or higher-dimensional robot control tasks. Can the authors comment on the expected practicality and robustness of SCIQL in real settings?

The overall pipeline introduces multiple components and optimization stages; how sensitive is the method to hyperparameters, and can the authors provide an ablation isolating the contributions of each module to ensure that improvements are not due to increased model complexity?

The approach assumes accurate style labels from the labeling functions. How does performance degrade under noisy or imperfect style annotations, and can the method handle ambiguous or overlapping style categories?

The paper positions programmatic style labeling as scalable, but could the authors discuss potential avenues for extending the framework to automatically learn style representations, or integrate human feedback when labeling heuristics are insufficient?

### Soundness
3

### Presentation
2

### Contribution
3
