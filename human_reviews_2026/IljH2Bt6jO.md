# Training a Vision-Language Model for Diverse Exploration in Open GUI World

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Vision-language models have emerged as capable computer-use agents, showing increasing potential to automate a wide range of computer tasks through graphical user interfaces. However, their effectiveness remains bounded by a fundamental limitation: current LLM- or VLM-based agents struggle to generalize to unfamiliar applications and remain heavily dependent on large-scale, human-curated datasets. To address this, we introduce ScreenExplorer, a novel VLM-based agent designed for autonomous exploration in real, dynamic, open-ended GUI environments. Through end-to-end training with an exploration-driven objective, our approach enables sustained interaction and diverse discovery without relying on predefined task structures. Specifically, we introduce a world model-inspired curiosity reward that helps the agent to overcome the cold-start phase of exploration, coupled with state-change-based exploration rewards to encourage agent's intrinsic motivation for venturing into novel states. Additionally, an experience stream distillation mechanism is designed to systematically accumulate and refine exploratory policies, enabling continual learning from gathered experiences. Extensive evaluations demonstrate that ScreenExplorer achieves remarkable generalization and diverse exploration capabilities in unseen applications, significantly outperforming static deployment baselines. This work establishes a new paradigm for GUI agents to progressively learn through autonomous exploration, moving beyond static dataset dependency toward adaptive, lifelong learning in complex digital worlds.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study proposes a reinforcement learning agent that autonomously explores GUI environments using a vision-language model. The proposed agent employs a reward function composed of nine components, including immediate, subsequent, and alignment rewards, together with a World Model–based curiosity reward to effectively explore unseen interfaces. To achieve more stable learning under the sparse and high-variance nature of GUI environments, the authors adopt Group Relative Policy Optimization (GRPO), demonstrating that it enables increasingly diverse exploration as training progresses.

### Strengths
- The study appropriately defines the observation and action spaces, which are essential aspects when applying reinforcement learning to web environments, and provides detailed explanations along with a plausible experimental setup.
- It conducts extensive experiments, including ablations, to analyze how visual reward signals and the world model influence exploration.
Defining a reward function in GUI environments is inherently challenging, but this work provides more combinations of the rewards than just simple single-step return rewards by introducing additional components such as the subsequent change reward, and empirically examines their effects on exploration through ablation studies.
- Given the sparse and highly fluctuating rewards characteristic of GUI environments, the choice to use GRPO instead of PPO to reduce variance is reasonable.

### Weaknesses
- This work uses visual/text diversity at the trajectory and group levels as its primary metrics, but it remains unclear whether these metrics truly correspond to meaningful web exploration rather than simply capturing random or superficial behavioral differences.
For instance, in Figure 5, the agent consistently selects the web browser across all episodes, and this could be due to the browser’s inherently higher visual diversity rather than intentional exploration, suggesting that the agent may have converged to a local optimum without exploring other diverse applications.

- The authors does not provide quantitative evaluations of the World Model’s prediction accuracy or qualitative visualizations comparing predicted versus actual screens.
Given that fine-grained predictive understanding is critical in such settings, presenting only the finding that the World Model helps during the cold-start phase, without evidence of its predictive fidelity, feels somewhat incomplete.

- Since several proposed reward components are directly tied to the evaluation metrics themselves (e.g., World Model–based reward), there is a risk that the agent’s exploration behavior becomes overly dependent on these self-referential metrics, potentially leading to biased or misleading exploration if the World Model is inaccurate.

### Questions
- Is there any way to verify the quantitative results of the World Model?
- In the GUI environment, does a sufficiently converged agent always choose the web browser?
If so, would this behavior correspond to a local optimum?

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
4

### Summary
This paper presents ScreenExplorer, a vision-language agent designed for exploration and interaction within real, dynamic, and open-ended GUI environments. The framework integrates a curiosity-driven reward mechanism that leverages a learned world model to enhance exploratory behavior; a reinforcement learning pipeline based on RL; and an experience stream distillation procedure that improves adaptation and reduces reliance on manually curated datasets.

### Strengths
1.	The combination of RL, a learned world model, and experience stream distillation forms a coherent, reproducible pipeline for self-supervised GUI exploration.
2.	The world model, trained on paired image-text state transitions, introduces an intrinsic curiosity reward that improves cold-start exploration and advantage variance.
3.	The paper analyzes reward components, showing that removing the world model or alignment rewards degrades performance.

### Weaknesses
1.	Although the paper claims that the agent is “rewarded for both successful interaction and exploration novelty,” the implementation shows that “successful interaction” merely refers to producing syntactically valid JSON actions that alter the GUI state. There is no external or task-based success criterion (e.g., reaching a goal, executing a correct function, or completing a workflow). As a result, the learned policy optimizes purely intrinsic objectives without assurance that these behaviors translate into useful or goal-directed interactions. This limits the practical significance of the reported improvements in exploration diversity and weakens the claim that ScreenExplorer enhances “interaction capability.”

2.	The system is designed almost entirely around exploration incentives. Curiosity-based rewards and diversity metrics are maximized without complementary exploitation or goal conditioning. Consequently, the policy may overfit to visually novel yet semantically meaningless actions (a form of the “noisy-TV” problem acknowledged by the authors). This imbalance raises doubts about whether the learned behaviors can generalize to structured GUI tasks requiring planning or consistency. The paper would benefit from experiments showing how ScreenExplorer behaves when explicit goals or extrinsic feedback are introduced.

3.	Despite the “open-world” framing, all reported experiments appear confined to a limited set of GUI applications with similar layouts and interaction patterns. There are no results demonstrating generalization to unseen interfaces, visual themes, or operating-system contexts. Without evaluation on held-out GUI types, the claim of “generalizable exploration” remains speculative. Additionally, the paper does not discuss potential domain adaptation techniques or cross-environment fine-tuning strategies.

4.	The paper defines nine reward components spanning format validity, visual and textual novelty, intent alignment, and world-model curiosity. However, their relative weighting, normalization, and mutual interference are underexplored. The only detailed ablation concerns the world-model term; other components (e.g., intent alignment or diversity scores) lack sensitivity analyses. Without systematic tuning or normalization (beyond GRPO’s implicit standardization), there is a risk of reward hacking, where the model exploits specific reward structures instead of learning genuinely diverse behaviors.

### Questions
1.	Could the authors clarify whether any extrinsic (task-success) reward or evaluator signal exists, and how the agent’s utility is measured beyond diversity?
2.	How sensitive is performance to the weighting of the nine reward components? Have normalization or coefficient ablations been tested?
3.	What prevents the curiosity reward from degenerating into the “noisy-TV” effect (repeating visually novel but meaningless actions)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
ScreenExplorer introduces a VLM-based agent trained via reinforcement learning in real GUI environments to enable autonomous exploration without relying on predefined task structures. The key innovation is a hierarchical reward system combining state-change rewards, world model-based curiosity signals, and intent-state alignment rewards for GRPO. Experiments demonstrate improved exploration diversity compared to static baselines, with the model improving from worst to best performer through RL training.

### Strengths
- Addresses the important but under-explored challenge of autonomous GUI exploration in open-ended environments, moving beyond task-specific training.
- The multi-faceted reward function elegantly combines immediate feedback (format, instant change), long-term diversity (subsequent change), curiosity (world model predictions), and grounding (intent-state alignment).
- Clear improvements in exploration diversity metrics, with ScreenExplorer-7B achieving 0.55 average diversity compared to 0.25 for GPT-4o and 0.43 for Qwen2.5-VL-72B.
- The world model curiosity reward successfully addresses the exploration cold-start problem by increasing advantage variance (Figure 4), enabling the 3B model to overcome initial learning barriers.

### Weaknesses
- The paper critically lacks evaluation on actual GUI tasks. While exploration diversity is measured extensively, there's no evidence that this exploration improves performance on established benchmarks like WebArena, VisualWebArena, or Mind2Web. This is a fundamental gap. Exploration is only valuable if it improves task performance.
- The diversity metrics (visual/textual sequence diversity) don't clearly correlate with useful exploration. The agent might be exploring irrelevant states (e.g., clicking random news articles) without learning transferable skills.
- Evaluation is restricted to a single Linux desktop environment. No evidence of generalization to other platforms (Windows, macOS, mobile) or complex web applications.
- While ablations show which rewards contribute to diversity, they don't demonstrate which exploration behaviors actually help downstream task learning. The "noisy TV problem" is mentioned but not thoroughly addressed.
- The filtering process for experience streams relies on GPT-4o-mini or manual curation, but there's no analysis of what makes exploration trajectories valuable for learning.

### Questions
- What is the performance on established benchmarks? How does ScreenExplorer perform on WebArena, VisualWebArena, or Mind2Web after exploration pre-training? Does exploration diversity correlate with task performance? Can you show that models with higher exploration diversity actually perform better on downstream GUI tasks?
- How does the approach handle task-specific fine-tuning? After exploration pre-training, how should the model be adapted for specific tasks?
- What prevents meaningless exploration? How do you ensure the agent explores task-relevant states rather than just clicking randomly to maximize state changes?
- How does exploration transfer across environments? Does exploration in Linux desktop environments transfer to web or mobile applications?
- What is the quality of distilled behaviors? Do distilled models learn meaningful exploration strategies or just memorize trajectories?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ScreenExplorer, a vision-language model trained via reinforcement learning for open-world GUI exploration. The agent learns to interact with real desktop interfaces without predefined goals, driven by multi-term rewards that capture both state changes and semantic alignment. A world-model-based curiosity signal promotes novelty, and experience stream distillation helps consolidate diverse experiences for continual improvement. Experiments in a Linux GUI environment show consistent gains in exploration diversity and novelty compared to strong VLM baselines.

### Strengths
- The paper explores a fresh and timely problem (open-world GUI exploration), moving beyond fixed task datasets toward agents that can learn to explore software interfaces on their own through curiosity-driven interaction.
- It proposes a well-designed framework that combines a world-model-based curiosity signal, multi-part state-change rewards, and GRPO optimization, with experience stream distillation helping the agent gradually improve through self-collected experience, this is intuitive and effective.
- Experiments in a realistic Linux desktop environment show noticeable gains in exploration diversity and novelty compared to strong VLM baselines, and the analyses clearly demonstrate how the curiosity signal and alignment rewards contribute to the results.
- The paper is clearly written and well-organized.

### Weaknesses
- The evaluation is done only in a custom Linux GUI environment with a small set of apps and layouts. While this makes the study controlled and clean, it doesn’t reflect the variety and complexity of real-world interfaces like web or multi-window systems. It’s therefore unclear if the exploration policy and curiosity module would still work well in broader or more realistic GUI settings.

- The ablation studies focus on removing reward terms or curiosity signals but don’t test the effect of experience stream distillation, reward weights, or world-model design. As a result, it’s hard to tell which parts of the system actually drive the performance gains, rather than the whole pipeline working together.

-  While the framework is well-designed, many parts (like the curiosity module, GRPO training, and distillation) are adapted from existing methods. The paper mainly integrates these components rather than introducing new techniques, so the overall novelty feels more like a solid system combination than a new algorithmic idea.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
