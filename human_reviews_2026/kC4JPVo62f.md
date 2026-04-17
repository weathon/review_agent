# DriveRX: A Vision-Language Reasoning Model for Cross-Task Autonomous Driving

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Effective autonomous driving hinges on robust reasoning across perception, prediction, planning, and behavior. However, conventional end-to-end models fail to generalize in complex scenarios due to the lack of structured reasoning. While recent vision-language models (VLMs) have been applied to driving tasks, they typically rely on isolated modules and static supervision, limiting their ability to support multi-stage decision-making. We present AutoDriveRL, a unified training framework that formulates autonomous driving as a structured reasoning process over four core tasks. Each task is independently modeled as a vision-language QA problem and optimized using task-specific reward models, enabling fine-grained reinforcement signals at different reasoning stages. Within this framework, we train DriveRX, a cross-task reasoning VLM designed for multi-stage decision-making. DriveRX achieves strong performance on the public benchmark, outperforming GPT-4o in behavior reasoning and demonstrating robustness under complex or corrupted driving conditions. DriveRX serves as a high-level semantic reasoning backbone, producing structured stage-wise reasoning chains that enhance decision consistency. These outputs also provide high-quality supervisory signals for annotation and downstream planning/control models.
We will release the AutoDriveRL framework and DriveRX to support future research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents DriveRX, a vision-language-action model designed to enhance reasoning and decision-making for autonomous driving. The authors aim to leverage language-based reasoning to handle complex or ambiguous driving cases and show some performance improvements over generalist vision-language models. The paper also includes an evaluation on a trajectory prediction benchmark to demonstrate the model’s potential for better planning.

### Strengths
The paper shows a clear pipeline for data construction and training framework of VLM training, and it shows some improvement over general-purpose models like Qwen and other large VLMs, which suggests that domain-specific fine-tuning helps the model better understand driving scenes.

### Weaknesses
I have some concerns about the motivation and experimental setup. Most recent works that use VLMs for driving focus on long-tail reasoning or trajectory prediction, and they evaluate on clear benchmarks with established baselines. This paper shows results in Table 3, but the setup isn’t well explained, so it’s hard to understand what the numbers actually mean or how fair the comparison is.

The model is not compared against any strong end-to-end driving systems, such as UniAD or VAD, which are the standard baselines for trajectory prediction.

It’s also missing comparisons with other VLM-based methods, such as DriveVLM, OmniDrive, or ORION. In fact, ORION is a very close baseline, and based on the results here, this proposed method doesn’t outperform it.

Overall, the improvement seems marginal, and the paper doesn’t convincingly show that this method leads to any meaningful gains in driving performance. The motivation would be stronger if the authors could clearly demonstrate where this approach provides a unique advantage over existing systems.

### Questions
Can the authors clarify the evaluation setup for Table 3? How are the metrics calculated? 

Why are the results not compared with other domain-specific models (e.g., UniAD, VAD, DriveVLM, OmniDrive, ORION)?

What's the main usecase for this VLM? Is it aiming for improving existing self driving system?

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
5

### Summary
This paper presents DriveRX, a vision-language reasoning model for autonomous driving trained via the AutoDriveRL reinforcement learning framework.
AutoDriveRL decomposes driving into perception, prediction, planning, and behavior tasks, each optimized with task-specific reward models.
Using structured reasoning and joint RL training, DriveRX achieves strong generalization and robustness on DriveBench and DriveLM-Hard benchmarks, outperforming GPT-4o in behavior reasoning.
The model also supports downstream trajectory prediction and action generation through reasoning-based distillation.

### Strengths
+ Proposes a unified RL framework (AutoDriveRL) enabling interpretable, multi-stage reasoning across driving subtasks.

+ Demonstrates state-of-the-art performance and robustness, surpassing larger models under both clean and corrupted conditions.

+ Extends practical impact by showing reasoning-enhanced transferability to trajectory and control tasks.

### Weaknesses
**For the image input**: In the structured reasoning process you designed, including prediction and planning, you only feed the model multi-view images, which makes it very hard to ensure the model can capture temporal information like speed — this doesn’t really make sense. Although some other VLAs also only use images, they at least provide historical ego‐states to ensure the ego car’s temporal information.

**Application**: The authors design a very complex reasoning process; I am quite puzzled about what scenario the authors expect their work to be applied in. If it’s just high-level prediction, then they have not opened up a large gap compared with current open-sourced VLMs such as InternVL3-8B, Qwen3-VL-8B, etc. (And I must emphasize: the authors test on DriveBench but that is not truly OOD because both DriveRX and DriveBench data are constructed based on DriveLM.) I am not sure whether DriveRX has better performance than the open-sourced VLMs in other domains of driving scenarios (e.g., DriveAction).

**Latency**: Because I am confused about the application of DriveRX. If it is used for low-level planning, then latency is a big challenge; and based on Table 3 results, its performance gap with existing VLAs model is significant (e.g., Auto-VLA).

### Questions
+ Could the authors clarify the time latency when using an LLM for scoring during training?

+ For the prediction and planning tasks, did the authors explore using a rule-based reward model? If so, how does it compare (advantages/disadvantages) to the LLM-based reward?

+ Could the authors provide results on the AlphaDrive benchmark for planning, so we can assess true out-of-distribution (OOD) performance?

+ Based on Table 8, I don’t understand why the gap between SFT (supervised fine-tuning) and RL in the behavior task is so large—this phenomenon isn’t observed in other related work. Could the authors explain?

+ **Most importantly, I’d like the authors to clearly articulate the intended application scenarios for DriveRX:**

(i) As a strong backbone for low-level planning tasks?

(ii) As a data annotator for long-tail driving scenarios?

(iii) As a reward model to improve other VLA/VLMs’ low-level planning performance?

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
4

### Summary
This paper proposes AutoDriveRL, a reinforcement learning (RL) framework for vision-language models (VLMs) applied to autonomous driving, decomposing driving into four interpretable sub-tasks: Perception, Prediction, Planning, and Behavior. Each sub-task is framed as a VQA problem with a task-specific reward model. Based on this framework, the authors train DriveRX, a cross-task reasoning VLM that aims to unify multi-stage decision-making. Experiments on DriveBench and DriveLM-Hard show that DriveRX outperforms GPT-4o and other baselines in the Behavior task and maintains robustness under visual corruptions. They further extend DriveRX to (1) DriveRX-Agent for trajectory generation and (2) DriveRX-VLA for closed-loop control via reasoning-based distillation.

### Strengths
This study introduces a reinforcement-learning–driven, multi-task vision-language reasoning framework designed to enhance the coordination between perception, reasoning, and action in autonomous driving. By attaching task-specific reward models to each stage of the reasoning process, the approach provides finer-grained supervision, improving interpretability and stability.

It organizes the problem into four subtasks that reflect the cognitive pipeline of autonomous driving—from perception to behavior. A Visual Question Answering (VQA)–style data formulation is used to unify the data structure, supporting cross-task consistency and a cleaner, more integrated learning signal.

Empirical results show that DriveRX achieves 62.02 on the Behavior task, outpacing GPT-4o (55.04) and the strongest open-source vision-language models. The method also demonstrates robust performance under visual corruption, underscoring its resilience. Baseline comparisons cover a broad spectrum, including commercial models (e.g., GPT-4o), generalist models (LLaVA, Qwen2.5-VL), reasoning-oriented models (MM-Eureka, R1-OneVision), and domain-specific systems (DriveLM, Dolphins), all evaluated under consistent settings.

Beyond core reasoning, the approach extends to trajectory prediction and control through DriveRX-Agent and DriveRX-VLA. These extensions illustrate how high-level reasoning can be bridged with low-level action generation, pointing toward a unified driving agent that blends perception, reasoning, and control in a single framework.

### Weaknesses
1. Ambiguous Reinforcement Learning Details
Equation (2) defines GRPO, but the intra-group advantage term (Eq. 3) is shared across tokens, not time-dependent, this simplifies training but may degrade credit assignment.
No mention of reward normalization schedule, policy rollout horizon, or sampling temperature, which are essential for RL reproducibility.
The rule-based and LLM-based reward models are described qualitatively but lack explicit scoring functions, thresholds, or examples (Appendix M.1 is referenced but not shown in main text).

2. Missing Quantitative Ablations
The framework introduces multiple key design choices (reward composition, GRPO vs PPO, per-task weighting, data filtering), yet no ablation results are provided in the main paper.
For example, how much does AutoDriveRL contribute compared to supervised fine-tuning on the same data? The gain from GRPO remains unquantified.

3. Unclear Task Coupling
The paper claims cross-task reasoning, but tasks are trained independently with shared parameters, without conditioning on previous task outputs (explicitly stated in §3.1).
This design limits true reasoning chaining; the reasoning chain is only semantic, not computationally causal. Thus, DriveRX may not genuinely achieve process-level reasoning across tasks.

4. Evaluation Ambiguities
The GPT score metric is used exclusively, judged by GPT-3.5-Turbo. This raises reliability and circularity issues — the model is trained with LLM-based rewards and then evaluated by another LLM judge.
There is no mention of inter-rater agreement, or calibration with human labels (Appendix I claims alignment but gives no correlation coefficients in the main text).
Metrics like ADE/Col. in Table 3 and Driving Score in Table 4 are reported but lack variance or standard deviation, leaving statistical significance unclear.

5. Distillation Pipeline Unclear
In §6, reasoning data from DriveRX is distilled into DriveRX-VLA, but the exact mapping from language to action tokens and supervision loss are unspecified.In §6, reasoning data from DriveRX is distilled into DriveRX-VLA, but the exact mapping from language to action tokens and supervision loss are unspecified.
It’s not clear whether the VLA model receives image input, text prompts, or both during inference — this affects the claimed “unified backbone” property.

6. Interpretability and Example Limitation
Only one qualitative case (Figure 3) is provided. No multi-turn reasoning trace or counterfactual analysis is shown, weakening claims of interpretability.

### Questions
1. Ambiguous Reinforcement Learning Details
Equation (2) defines GRPO, but the intra-group advantage term (Eq. 3) is shared across tokens, not time-dependent, this simplifies training but may degrade credit assignment.
No mention of reward normalization schedule, policy rollout horizon, or sampling temperature, which are essential for RL reproducibility.
The rule-based and LLM-based reward models are described qualitatively but lack explicit scoring functions, thresholds, or examples (Appendix M.1 is referenced but not shown in main text).

2. Missing Quantitative Ablations
The framework introduces multiple key design choices (reward composition, GRPO vs PPO, per-task weighting, data filtering), yet no ablation results are provided in the main paper.
For example, how much does AutoDriveRL contribute compared to supervised fine-tuning on the same data? The gain from GRPO remains unquantified.

3. Unclear Task Coupling
The paper claims cross-task reasoning, but tasks are trained independently with shared parameters, without conditioning on previous task outputs (explicitly stated in §3.1).
This design limits true reasoning chaining; the reasoning chain is only semantic, not computationally causal. Thus, DriveRX may not genuinely achieve process-level reasoning across tasks.

4. Evaluation Ambiguities
The GPT score metric is used exclusively, judged by GPT-3.5-Turbo. This raises reliability and circularity issues — the model is trained with LLM-based rewards and then evaluated by another LLM judge.
There is no mention of inter-rater agreement, or calibration with human labels (Appendix I claims alignment but gives no correlation coefficients in the main text).
Metrics like ADE/Col. in Table 3 and Driving Score in Table 4 are reported but lack variance or standard deviation, leaving statistical significance unclear.

5. Distillation Pipeline Unclear
In §6, reasoning data from DriveRX is distilled into DriveRX-VLA, but the exact mapping from language to action tokens and supervision loss are unspecified.In §6, reasoning data from DriveRX is distilled into DriveRX-VLA, but the exact mapping from language to action tokens and supervision loss are unspecified.
It’s not clear whether the VLA model receives image input, text prompts, or both during inference — this affects the claimed “unified backbone” property.

6. Interpretability and Example Limitation
Only one qualitative case (Figure 3) is provided. No multi-turn reasoning trace or counterfactual analysis is shown, weakening claims of interpretability.

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
This paper introduces AutoDriveRL, a unified reinforcement learning framework that models autonomous driving as a structured reasoning process across perception, prediction, planning, and behavior. It trains a cross-task reasoning model, DriveRX, which outperforms GPT-4o in behavior reasoning and shows strong robustness in complex driving scenarios.

### Strengths
1. The formulation of each sub-task as a vision-language QA problem is interesting.

2. The experimental results are promising.

### Weaknesses
1. The introduction section feels somewhat wordy and difficult to follow. It’s not clearly structured around three key points: what common issues exist in current works, what direction and method are proposed, and how the proposed method is designed. These aspects should be laid out more clearly.

2. Additionally, the paper categorizes the autonomous driving pipeline into perception, prediction, planning, and behavior. Typically, autonomous driving is divided into three main tasks, perception, prediction, and planning, while behavior is often treated as part of prediction. What is the rationale for introducing behavior as a separate stage?

3. The description of the task-specific reward models lacks clarity. How the reinforcement signals are computed and whether they are consistent across tasks could be better elaborated.

4. It’s unclear how much each reasoning stage (perception, prediction, etc.) contributes to overall performance,  finer-grained ablation or visualization would improve interpretability.

### Questions
1. How robust is DriveRX to distribution shifts, e.g., unseen weather or sensor noise, beyond the “corrupted” conditions mentioned?

2. Are there any failure cases observed during deployment or simulation that reveal limits in the cross-task reasoning design?

### Soundness
3

### Presentation
3

### Contribution
2
