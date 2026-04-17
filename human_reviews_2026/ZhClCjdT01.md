# On-policy Reinforcement Fine-tuning with Offline reward for Multi-step Embodied Planning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2, 2

## Abstract
Embodied planning requires agents to make coherent multi-step decisions based on dynamic visual observations and natural language goals. 
  While recent vision-language models (VLMs) excel at static perception tasks, they struggle in interactive environments.
  In this work, we introduce an on-policy reinforcement fine-tuning framework with offline rewards, that preserves the generalization benefits of RFT while addressing the challenges of sparse rewards and costly interaction, supported by solid theoretical guarantees.
  Our approach is evaluated on Embench, a recent benchmark for interactive embodied tasks, covering both in-domain and out-of-domain scenarios. 
  Experimental results show that our method significantly outperforms models of similar or larger scale, including GPT-4o-mini and 70B+ open-source baselines, and exhibits strong generalization to unseen environments. 
  This work highlights the potential of reinforcement-driven reasoning to advance multi-step planning in embodied AI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an on-policy Reinforcement Fine-Tuning (RFT) framework for multi-step embodied planning. To address the challenges of costly online interaction and sparse rewards, the method utilizes an offline reward signal computed by comparing model-generated action sequences to expert demonstrations, primarily based on the length of their correct action prefix. The approach involves an initial SFT stage, followed by policy optimization using GRPO. Experimental evaluation on the Embench benchmark is conducted to demonstrate the effectiveness of the proposed pipeline.

### Strengths
The paper presents a valuable application of Reinforcement Fine-Tuning to the embodied planning domain. The designed reward function, which combines a prefix-based accuracy reward with a structured format reward, successfully adapts the RFT paradigm to this task and shows a clear performance improvement over SFT alone, as evidenced in the ablation studies.

### Weaknesses
- The proposed method's core reward mechanism is fundamentally dependent on pre-defined expert trajectories. This strong reliance on supervised demonstration data limits the agent's capacity for strategic exploration and discovery of novel solutions beyond the provided examples, constraining its generalization to new scenarios.

- Insufficient Baseline Comparisons. The experimental section lacks comparisons with previous baselines specifically designed for embodied task planning. For instance, the method is not compared against DPO-based methods like EPO, D2PO, SFT-based planners like WAP (62.7%), or other RL/RFT-based VLM methods such as RL4VLM (51.2%), VAGEN (52.8%).

- There is a critical conceptual flaw in the training setup. The paper studies a multi-turn, interactive problem but explicitly avoids "online execution" and "collecting interactive feedback" (Line 142). Consequently, during training, the model's rollouts are generated from the sequence of states in the expert trajectory, not from the states resulting from the model's own previous actions. This severs the feedback loop that is central to sequential decision-making, as the state at step_{i+1} is independent of the action taken at step_i. This reduces the problem from interactive Sequential Decision-Making to a static Sequence Generation task, failing to address the core challenge of adapting to dynamic state transitions.

- The entire study is conducted using a single backbone model, it is important to demonstrate the generality of the proposed RFT framework across different model architectures.

### Questions
- The EB-ALFRED benchmark and the proposed method's training data are both derived from the ALFRED dataset. Is there any overlap between the tasks or trajectories used for RFT training and those in the Embench evaluation sets?

- The main results table omits the "Long Horizon" subset of Embench. Could you provide the results on this subset? Long Horizon capability is also important in embodied planning.

-  What's the backbone model used for SFT and RFT? The main text does not explicitly state this.

### Soundness
2

### Presentation
2

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
This paper tackles training VLMs in embodied environments, where sparse rewards and costly simulator rollouts make RL difficult. The authors propose an *on-policy* RFT method that uses *offline rewards* computed by comparing model-generated trajectories to expert demonstrations, avoiding expensive environment interaction. The approach combines supervised imitation learning with offline RFT and provides theoretical guarantees that the learned policy approximates the true online optimum. Experiments on embodied reasoning benchmarks show substantial gains over strong open- and closed-source baselines, indicating improved planning and generalization.

### Strengths
- The theoretical analysis is clear and shows that the offline objective is directional aligned with the true reward. This mathematical justification is appreciated rather than solely relying on empirical intuition. 
- The idea is intuitively motivated and well executed. Avoiding the cost of simulator rollouts while retaining the performance is impressive. 
- By eliminating the need for rollouts, the approach makes the path towards scalable training more practical and accessible.

### Weaknesses
- The method's reliance on offline expert trajectories means it cannot adapt when the environment, task structure, or available actions differ significantly from those seen in training. All experiments are conducted within variations of similar embodied reasoning environments (ALFRED and Habitat).
- On the same note, the learned policy assumes a shared task schema and action grammar; if either changes, the offline reward function provides no meaningful signal.
- The paper repeatedly emphasizes computational and practical efficiency as a key motivation, but never provides concrete evidence (e.g., training time, resource usage, or sample complexity) to support this claim.
- While the paper demonstrates improvements from the combined SFT→RFT pipeline and compares several internal reward variants, it never evaluates a control setup that performs *on-policy fine-tuning without the proposed offline reward*. A fairer test would include something like "SFT → on-policy fine-tuning without offline reward" versus "SFT → on-policy fine-tuning with offline reward." Comparing the offline reward signal against the true environment reward (when available) would reveal how closely it approximates real task success and how much performance is lost or retained relative to genuine RL feedback.

### Questions
- Could the authors clarify whether their approach can be extended to environments with continuous actions or richer affordances, where the notion of a discrete prefix match no longer applies?
- How sensitive is the method to suboptimal or noisy expert trajectories?
- Given the offline reward’s limitations on unseen tasks, could the authors comment on the feasibility of combining a small number of online rollouts with their offline reward for fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a method for fine-tuning Vision-Language Models (VLMs) for embodied agent planning using reinforcement learning. The primary contribution is an offline reward mechanism designed to overcome the challenges of sparse rewards and costly simulation. This reward is calculated by comparing model-generated rollouts to expert trajectories. Experiments conducted on the Embench benchmark demonstrate that this approach achieves better performance compared to the original base models, Supervised Fine-Tuning (SFT) models, and several open-source LLMs.

### Strengths
1. The paper is well-motivated. It aims to address an important problem: the sparse rewards and costly simulation for online RL embodied AI training. 

2. The experimental results show it outperforms the base model and the SFT model by a big margin on both EB-ALFRED and EB-Habitat.

### Weaknesses
1. The reviewer found many technique details are not presented clearly and needs further clarification. A critical point requiring clarification is the precise mechanism for generating the "model rollouts." The authors state that the method aims to address costly simulation, yet it is unclear whether these rollouts require interaction with a simulator. This ambiguity makes it difficult for the reader to fully understand the process.

- If no simulator is used: The manuscript should provide a detailed explanation of how these rollouts are performed (e.g., auto-regressively by the model). Furthermore, it is unclear how the method maintains or approximates an "on-policy" data distribution in the absence of an interactive environment.

- If a simulator is used: This appears to contradict the paper's core motivation of avoiding high simulation costs. The authors should explicitly address this potential contradiction and clarify the computational costs associated with this step, explaining how it is less "costly" than traditional online methods.

A more explicit description of this rollout process is essential for understanding the method's novelty and practical advantages.

2. In Line 82, the authors claim their work is the first to apply RL to optimize VLM for embodied tasks. This claim appears to be an overstatement, as several existing works utilize RL for similar tasks. For example, [a] employs both Supervised Fine-Tuning (SFT) and online RL to fine-tune embodied agents, demonstrating strong generalization performance across diverse benchmarks. A comparison with [a] in the experimental section would significantly enhance the informativeness of the paper.

3. The description of the data filtering process in Lines 190-193 lacks the necessary precision. The manuscript uses ambiguous terms such as 'too few' and 'too many' without providing formal, quantitative definitions. To ensure the method is reproducible, the authors must specify the exact numerical thresholds or statistical criteria used to implement this filtering step.


[a] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons, Szot et al., CVPR’25

### Questions
1. The proposed approach relies on expert trajectories for reward computation. How robust is this method if, as is often the case in real-world scenarios, expert data is expensive or unavailable, and only sub-optimal trajectories can be obtained?

2. Could you clarify the meaning of the terms k+1 and n+1 as they appear in equation (3)?

3. How are out-of-distribution issues addressed in the reward computation? Specifically, when the policy diverges from the expert policy, do you assume  it always leads to suboptimal actions?

4. The results in Table 1 are presented without confidence intervals (CIs). Given that reinforcement learning methods are known to have high variance, CIs are essential for a statistically robust evaluation and for making reliable comparisons between approaches. Please add confidence intervals (or standard deviations) for all results and specify the number of runs (e.g., random seeds) used to compute these statistics.

5. Please compare with more RL baselines such as [a]

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an on-policy reinforcement fine-tuning (RFT) framework for multi-step embodied task planning, aiming to close the gap between VLMs that perform well on static perception tasks and those required to act coherently in dynamic environments.

The method introduces an offline reward mechanism that compares model-generated trajectories with expert demonstrations, thereby avoiding costly simulator interaction and sparse environmental feedback. Rewards are computed via a quadratic prefix-based function measuring how long the predicted action sequence matches the expert trajectory (Eq. 3), combined with a small format-regularization term (Eq. 4). Optimization is performed through GRPO.

The authors claim theoretical guarantees showing that policies optimized under offline rewards remain bounded with respect to true online-reward policies. Experiments on Embench demonstrate improvements over both open-source and closed-source baselines. On EB-ALFRED (in-domain), their model reaches 49.2% success rate, outperforming Qwen2.5-VL-72B (40.8%) and LLaMA-3.2-90B (35.2 %); on EB-Habitat (out-of-domain) it achieves 22.4%, outperforming similarly-sized models.

Ablation studies test reward allocation and computation strategies, showing the prefix-based reward performs better than embedding-similarity metrics and that combining accuracy (1.0) + format (0.5) yields the best results. Overall, the work claims to enable scalable, simulation-free RFT for embodied reasoning.

### Strengths
- Clear motivation and strong empirical performance: The problem of costly online reward collection in embodied environments is well-motivated (in the real world at least), and experiments on Embench show tangible gains over large-scale baselines.
- The reward function is sound and simple. The optimal policy under this reward also proves close to the optimal policy under a sparse environment reward. Empirical results also show the effectiveness of this design.
- Comprehensive benchmarking and ablations: a good collection of open source and closed baselines models of difference sizes and reward variants are compared, offering a broad empirical picture.

### Weaknesses
### Unclear definition of the underlying MDP / “on-policy” claim
Section 2.1 formulates a POMDP with observations o_t and actions a_t, but Section 2.2 states that training avoids simulation. If the agent never steps an environment, there is no transition function P(s’|s,a) and thus no genuine new observation after an action. The process effectively samples full trajectories from the model and scores them offline against stored expert data, an imitation-style static optimization, not an interactive MDP. The paper should clarify what “on-policy” means here and explicitly state that the “environment” is frozen in the dataset. As written, the theoretical framing in Sec. 2.3.2 over-extends beyond what is actually optimized.

---

### Reward definition lacks justification and rigor.
In Eq. 3 the reward assumes the predicted and expert action sequences have equal length $k$, yet this is never guaranteed: model outputs can terminate early or differ in length. It is unclear how mismatched sequences are handled. The quadratic term $n(n+1)/k(k+1)$ appears ad hoc: the "$+1$" factor and quadratic scaling are not theoretically motivated beyond "smoothness".

---

### Limited discussion of computational efficiency.
The paper repeatedly claims that online reinforcement fine-tuning in embodied simulators is "prohibitively expensive" and motivates the offline-reward design on this basis. However, no quantitative evidence is provided: there are no comparisons of training time, GPU hours, or memory/FLOP usage between the proposed offline RFT and a standard online-interaction setup.

Moreover, modern embodied simulators are parallelizable: dozens or hundreds of episodes can run concurrently on a single node using GPU-based rendering. For these environments, the cost of stepping simulations is often negligible compared with large-model forward/backward passes. Thus, the argument that simulator interaction is the main computational bottleneck is not well-substantiated.

### Questions
- The authors should clarify the exact decision process underlying their “on-policy” RFT framework: if no simulation is performed, what defines the MDP’s transition dynamics and how are new observations obtained after each predicted action?
- Relatedly, the term “on-policy” needs to be precisely defined: does it merely refer to sampling from the current model, or does the data distribution evolve in any way to reflect policy behavior?
- The reward formulation also requires further justification: it assumes equal-length action sequences between model and expert trajectories, yet the paper does not explain how mismatched lengths are handled, nor why the quadratic prefix reward was chosen over simpler alternatives.
- Finally, the claimed computational advantage of avoiding simulator interaction should be supported with quantitative evidence. It would be helpful to specify whether it is possible to use parallelized simulation for training within EmbodiedBench, and to report actual training or rollout costs. Clarifying these points would make the technical contributions and efficiency claims much more convincing.

### Soundness
2

### Presentation
3

### Contribution
2
