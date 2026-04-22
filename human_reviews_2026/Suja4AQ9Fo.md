# Both Local Validity and Global Effectiveness Matter:  Decoupled Credit Assignment for Long‑Horizon Agentic Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
The natural-language action space of Large Language Model (LLM) agents creates a real risk of invalid outputs (e.g., API rejections, parsing errors). Consequently, in Reinforcement Learning (RL) for long-horizon LLM agents, learning to generate a locally valid action in each turn is as crucial as selecting globally effective one. However, this requirement was overlooked by the prevailing additive paradigm for credit assignment in agentic RL. Specifically, it computes an action's credit by summing an estimated local score with the trajectory-level score. This paradigm assigns a ``contribution" score to all actions regardless of their validity, allowing invalid actions to be assigned positive credit, especially in positive trajectories. To address this, we propose Multiplicative Gated Rewards (MGR), which decouples local action-level validity from global effectiveness. MGR uses a fact-based validity signal, derived from direct environment feedback and syntactic validity, to determine the action-level score (e.g., $\pm$1). This score is then multiplied by the magnitude of the trajectory-level score. This ensures the action's validity strictly governs the reward's polarity, preventing credit misassignment. Experiments demonstrate that our method improves training stability and achieves SOTA performance on long-horizon LLM agent benchmarks. Code of MGR has been uploaded in the Supplementary Material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Multiplicative Gated Rewards (MGR), a novel reinforcement learning framework for LLM agents that decouples local action validity from global trajectory effectiveness in long-horizon tasks. The key idea is that every action produced by an LLM agent must be both locally valid (syntactically and executably correct) and globally effective (helpful toward task success). Existing additive credit-assignment methods conflate these dimensions, rewarding invalid actions that appear in successful trajectories and penalizing valid actions in failed ones. MGR addresses this by introducing a multiplicative gating mechanism that strictly enforces validity at the action level before scaling by global success magnitude.

### Strengths
1. The paper tackles an important problem in LLM-agent RL, credit assignment, which directly affects training stability and generalization.
2. The proposed decoupling between sign (validity) and magnitude (strategy) is both intuitive and empirically effective, leading to clearer reward signals and better long-term reasoning behavior.
3. Across ALFWorld and AppWorld, MGR outperforms SFT, GRPO, GiGPO, and Loop, especially on harder tasks requiring longer reasoning chains.

### Weaknesses
1. While MGR’s “factual validity” is well-defined, its design relies on hard-coded heuristics, error detection, format checks, repetition counts, rather than learned notions of action value. This may incentivize the agent to game surface-level rules rather than develop deeper understanding of action utility, leading to potential reward hacking or brittleness across unseen domains.
2. Since MGR’s local reward is purely syntactic or execution-based, it lacks any semantic notion of whether the action actually helps the task. The trajectory-level term compensates only indirectly. The agent could thus learn to maximize validity without improving task-level reasoning if the environment reward is sparse or delayed.
3. The method introduces several coefficients ($\alpha$, $\beta$, $\gamma$, $q$, $p_min$, schedule thresholds) whose interdependence could make it sensitive to domain or dataset scale. The paper provides defaults but lacks a robustness or sensitivity analysis beyond $\beta$.

### Questions
See weaknesses

### Soundness
2

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
The paper proposes a novel technique for credit assignments in long-horizon agent tasks for LLMs. Specifically, they propose two orthogonal components that make up their reward through a multiplicative gate, local action validity reward (+1 or -1 sign based on the action validity) and a trajectory level advantage score. The method outperforms other techniques like GRPO and Loop on ALFworld and AppWorld tasks.

### Strengths
1. The credit assignment problem being tackled is important
2. The paper's idea is interesting and well motivated
3. The ablations study and analysis are insightful

### Weaknesses
1. The method mostly relies on heuristics for the local-level action feedback, which seems like added bias and transfer on other tasks is uncertain
2. It's likely quite tricky to use this method on a variety of problems where it's not always clear what constitute as an invalid action, or where there is not an automated feedback for invalid actions
3. There are no uncertainty/error bounds in the experiments.
4. Introduction of various new hyperparameters makes me wonder how sensitive the method is to them, and how difficult they would be to tune on new tasks.
4. The method is tested on only two tasks, it's possible that the method and hyperparameters have been overfitted on just these two tasks.

### Questions
1. How would the method be applied to tasks where action-level validity signals are not available? 
2. How sensitive is the method to the various hyperparameters introduced, and how well does it transfer to other tasks?
3. Is it possible to add more seeds to get error bounds?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Multiplicative Gated Rewards (MGR), a reinforcement learning framework for long-horizon LLM agents that separates local action validity from global effectiveness. Instead of adding rewards, MGR multiplies a validity signal with a trajectory-level score, ensuring invalid actions are penalized even in successful trajectories. A dynamic gating mechanism creates an implicit curriculum, first learning valid actions, then strategic sequencing. Experiments on ALFWorld and AppWorld show MGR achieves state-of-the-art performance over more challenging tasks compared with baselines.

### Strengths
1. The paper is easy to follow.
2. The paper systematically analyzes the key issue of credit assignment in long-trajectory learning for LLM agents and proposes a practical solution to address it.
3. The experiments are convincing, as MGR demonstrates a remarkable advantage on more challenging L2 tasks.

### Weaknesses
1. The method introduces several hyperparameters for control, such as α and β, whose sensitivity remains unclear, which is critical for assessing the robustness and practical applicability of this approach.
2. The method relies on explicit action validity signals from the environment, limiting its applicability to tasks with well-defined executable actions and making it unsuitable for open-ended or semantic language tasks without clear feedback.

### Questions
1. Can the method be applied to more complex environments, such as embodied settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a reward function, Multiplicative Gated Rewards (MGR), for long-horizon LLM agents, which decouples "local action validity" (step is syntactically/executably valid) from "global effectiveness" (trajectory quality), then fuse them multiplicatively so invalid steps contribute zero (or negative) credit even in successful rollouts. A stochastic gate balances valid steps in failed trajectories to avoid overwhelming positives. Experiments on ALFWorld and AppWorld with Qwen2.5-7B/Qwen3-8B show higher success rates than baselines; ablations indicate the gate, critic, and repetition penalty matter materially.

### Strengths
- Addresses credit assignment problem, i.e. additive credit lets invalid steps get positive credit in successful trajectories.
- Method is simple, orthogonal to PPO/GRPO training, and given in clear pseudocode; implementation details for validity checks are explicit for both environments.
- Consistent gains on ALFWorld/AppWorld; learning curves show improved stability and faster rise in both action-validity and task success.
- Some ablations provided: removing gating/critic/repetition penalty degrades sharply, supporting design choices.
- Method naturally induces a curriculum

### Weaknesses
- Novelty is incremental. Multiplicative gating is a principled reweighting/masking; not a new credit-assignment theory.
- Local validity is heuristically defined and environment-specific; portability beyond text/code interfaces is unclear. Provide a (ideally formal) definition up front
- Gating schedule is heuristic. The batch-stat–driven sign flip lacks theory; sensitivity and failure modes aren’t fully characterized.
- Reporting focuses on success rates; analysis on sample efficiency and exploration would be appreciated.

### Questions
- Does the stochastic sign gate ever destabilize learning or introduce bias, especially early in training when the proportion of valid actions is low?
- Can MGR be applied on top of learned or preference-based reward models (e.g., PRM, CAPO, RLAIF)? If so, is the gating still beneficial, or redundant?
- Is there any quantitative evidence that the multiplicative formulation improves gradient signal quality?

### Soundness
3

### Presentation
3

### Contribution
2
