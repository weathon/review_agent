# Learning from Imperfection: Mistake-Aware LLM Finetuning for Robust Planning

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
While finetuned Large Language Models (LLMs) for embodied planning excel at producing reliable plans for the given environment, they do so in a very narrow area of operation. Usually one wrong step in such a plan is sufficient to get the agent into an unseen scenario. Current training paradigms focus on preventing mistakes by learning from optimal demonstrations, but neglect the crucial skill of recovering when deviating from the correct plan. To address this gap, we propose Mistake-Aware Finetuning (MAF), a novel training methodology that explicitly teaches agents to recover from planning errors. In the MAF paradigm, the model is exposed to plans containing intentional mistakes, but a targeted loss mask ensures it only learns from the subsequent, correct recovery actions. This allows the model to learn the association between a failure state and its resolution without being negatively influenced by the erroneous action itself. We demonstrate the effectiveness of MAF by finetuning a Llama-3B model on two distinct environments. Our approach substantially improves the task success rate from 21% to 72% on the Tower of Hanoi puzzle and from 67% to 96% on the complex MiniGrid MiniBossLevel. Furthermore, to probe the generalisation capabilities of our method, we introduce Unlock-To-Unlock-N, a novel and challenging benchmark designed to test long-horizon planning. We demonstrate that our MAF-trained agent not only performs robustly in this new benchmark but also exhibits generalisation capabilities that extend beyond the training dataset. It highlights the MAF’s immense potential for developing lightweight, robust embodied agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on learning from imperfect demonstrations where some mistakes is made at certain step of the trajectory. The article proposed a data generation method to construct a data combines with optimal and imperfect trajectories. Then a targeted loss masking is used to train the LLM to recover from errors. To validate their method, the author proposes a new benchmark: Unlock-to-Unlock-N and does some experiments.

### Strengths
1. The idea of learning from imperfect demonstrations is novel.
2. The author proposes a new benchmark: Unlock-to-Unlock-N.

### Weaknesses
1. The setting is impractical. The method requires to know at which step the mistake is made in advance, while in many practical settings it is impossible. 
2. The method is too simple. Only ignore the mistaken step to train is just a simple weighted behavior cloning with weights only being 0 and 1. We desire a more innovative and effective approach. For example, some offline RL methods can weight these behaviors with a learnt Q. 
3. The baseline is too simple and not enough. For example, you can include an offline RL baseline with the right step rewarded by 0, wrong step rewarded by -0.1 and final success rewarded by 1, where the policy is learnt with a Q as weighting function on the behaviors.

### Questions
See weaknesses.

### Soundness
1

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
5

### Summary
This paper presents a fine tuning technique for agentic planning tasks which makes the model more robust after a mistake.

### Strengths
The novelty in the paper is that the mistakes are introduced artificially during training and since they know which step is noisy, they do not consider that for model training (like masking). This way model learns how to robustly recover from past mistakes. The results are verifying the uselfulness of this approach on multiple benchmarks.

### Weaknesses
The title is misleading. It reads like this is an approach which is learning from its mistakes (like RL/DPO does), hence model makes less mistakes. The goal is not making less mistakes but being more robust after making a mistake. In agentic world, this is usually called as error recovery. Otherwise learning from mistakes is a heavily studied area with multiple papers (see below)
One big problem in the setup is that, the introduced mistakes do not effect the remaining actions. For example picking up an apple is introduced as a mistake but this can be ignored during the execution of the rest. In many cases the mistakes result in scenarios which are very hard to recover from, such as boiling an egg while making an omelette in home robot scenarios. Also for conversational cases, usually a mistake is harder to recover from, such as "set a timer" "for how long?" "15 minutes" "ok setting the timer for 50 minutes" "no i said 15 15"
In that respect I find the evaluation setup limited to certain tasks.

### Questions
Despite allocating a section on RL, I'd prefer to see quantifiable comparison with RL based approaches.
Explain the difference of your approach from the following papers:
https://arxiv.org/pdf/2403.02502
https://arxiv.org/abs/2406.11176
https://arxiv.org/pdf/2501.01702

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper tackles the brittleness of LLM-based planning agents, where a single error can derail the entire plan, by introducing Mistake-Aware Finetuning (MAF). Unlike conventional fine-tuning approaches that rely solely on perfect demonstrations, MAF intentionally injects errors into plans and masks the corresponding loss for those erroneous steps. The model is then trained only on the subsequent corrective actions, enabling it to associate failure states with appropriate recovery behaviors. Experiments are conducted on two domains: i) Tower of Hanoi, representing symbolic, logic-based planning, and ii) MiniGrid MiniBossLevel, representing embodied, partially observable environments. MAF significantly improves task success rates: from 21% to 72% on Tower of Hanoi and 65% to 94% (up to 96%) on MiniGrid. Additionally, the paper introduces a new benchmark, Unlock-To-Unlock-N, to evaluate long-horizon generalization, demonstrating that MAF-trained agents can generalize to unseen planning depths.

### Strengths
- Clarity and Motivation: The paper clearly defines the brittleness problem in LLM-based planners and convincingly argues for shifting the learning objective from error prevention to error recovery. This conceptual pivot is both intuitive and novel.
- Methodological Simplicity: MAF requires only minimal modification (loss masking) to existing fine-tuning pipelines. Its lightweight design supports scalability and ease of integration into existing LLM-planning frameworks.
- Empirical Rigor: Consistent performance gains are demonstrated across symbolic (Tower of Hanoi) and embodied (MiniGrid) tasks. The inclusion of the Unlock-To-Unlock-N benchmark provides evidence for the model’s generalization to long-horizon planning scenarios.
- Analytical Depth: The paper thoughtfully interprets MAF not merely as “error correction” but as learning a mapping between failure contexts and corrective actions. The synergy between MAF and step-by-step decoding is particularly well-analyzed, reinforcing the method’s robustness.

### Weaknesses
- Limited Experimental Settings: All experiments are conducted in symbolic or textual simulation environments (Tower of Hanoi and MiniGrid), which lack sensor noise, actuation errors, or physical constraints. As the authors themselves acknowledge in Section 5, MAF’s effectiveness in embodied or multimodal agents remains untested. The current validation is confined to text-level proxies of planning, rather than full embodied intelligence.

- Synthetic Mistakes: The mistakes used for training are artificially injected (e.g., random subgoal removal or insertion in MiniGrid, suboptimal moves in Tower of Hanoi). Such synthetic perturbations do not fully capture real-world error modalities (e.g., perception noise, actuator drift, memory errors, or environmental dynamics). Consequently, it is unclear whether MAF’s recovery capability would transfer to more realistic, noisy embodied scenarios.

- Lack of Comparison with Related Baselines: Although the paper conceptually relates MAF to lightweight alternatives to reinforcement learning (RL) approaches such as [1,2,3,4,5,6,7], it does not include quantitative comparisons or efficiency analyses against these baselines. Without such evaluations, it remains uncertain whether MAF’s masking-based fine-tuning provides a substantial benefit beyond existing (offline) RL or trajectory relabeling methods.

- Underdefined “Mistake-Aware” Taxonomy: The paper promotes “mistake awareness” as a central contribution but does not systematically define or categorize different error types. The injected mistakes are limited to planning-level symbolic errors, while real embodied agents may face more diverse and interacting failure sources (e.g., perceptual, causal, or motor-level mistakes). The lack of error taxonomy and per-type recovery analysis makes the notion of “mistake-awareness” somewhat shallow and limited to simple plan deviations.

[1] Off-policy deep reinforcement learning without exploration. Fujimoto et al., 2019.

[2] Conservative Q-Learning for Offline Reinforcement Learning. Kumar et al., 2020.

[3] AWAC: Accelerating Online Reinforcement Learning with Offline Datasets. Nair et al., 2020.

[4] Offline Reinforcement Learning with Implicit Q-Learning. Kostrikov et al., 2021.

[5] Decision Transformer: Reinforcement Learning via Sequence Modeling. Chen et al., 2021.

[6] Offline Reinforcement Learning as One Big Sequence Modeling Problem. Janner et al., 2021

[7] Hindsight Experience Replay. Andrychowicz., 2017.

### Questions
- How would MAF perform in multimodal or embodied settings involving perception and motor control noise? Do the authors consider sensor-based observations or continuous control domains in the current submission?

- Can the authors provide a more systematic definition or taxonomy of “mistakes” beyond subgoal insertion/removal? How might MAF handle errors such as perceptual, causal, or memory origin?

- Have the authors considered benchmarking MAF against offline RL methods, which similarly learn from suboptimal trajectories? How does MAF compare in terms of sample efficiency or robustness?

- Does this submission consider integrating MAF into larger embodied systems (e.g., robot task planners using visual-linguistic inputs)? If so, how are non-symbolic (or textual) state representations addressed?

### Soundness
2

### Presentation
2

### Contribution
2
