# Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Vision-and-Language Navigation (VLN) poses significant challenges for agents to interpret natural language instructions and navigate complex 3D environments. While recent progress has been driven by large-scale pre-training and data augmentation, current methods still struggle to generalize to unseen scenarios, particularly when complex spatial and temporal reasoning is required. In this work, we propose SkillNav, a modular framework that introduces structured, skill-based reasoning into Transformer-based VLN agents. Our method decomposes navigation into a set of interpretable atomic skills (e.g., Vertical Movement, Area and Region Identification, Stop and Pause), each handled by a specialized agent. To support targeted skill training without manual data annotation, we construct a synthetic dataset pipeline that generates diverse, linguistically natural, skill-specific instruction-trajectory pairs. We then introduce a novel training-free Vision-Language Model (VLM)-based router, which dynamically selects the most suitable agent at each time step by aligning sub-goals with visual observations and historical actions. SkillNav obtains competitive results on commonly-used benchmarks, and establishes state-of-the-art generalization to the GSA-R2R, a benchmark with novel instruction styles and unseen environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SkillNav, a mixture-of-experts framework designed to decompose vision-and-language navigation (VLN) into atomic skills and route-level reasoning. The approach aims to improve compositional generalization by assigning navigation instructions to specialized experts and reordering sub-instructions using LLMs. The authors also introduce several skill-based datasets to support the design and use of VLMs as the router for different experts. Experiments on R2R and GSA-R2R demonstrate the effectiveness of the proposed SkillNav especially in GSA-R2R with styled instructions.

### Strengths
1. Interesting motivation and idea. The paper presents a clear and appealing idea of decomposing navigation into modular “skills” and routing instructions through specialized experts. This formulation aligns well with the broader goal of enhancing compositional reasoning in VLN. The authors also provide distinct datasets for training each expert, which adds practical value and can facilitate future research in this direction.
2. The proposed SkillNav framework effectively combines the generalization ability of LLM-based methods with the strong task-specific performance of supervised VLN models through a hierarchical structure. This hybrid design is promising and represents a good balance between accuracy and efficiency.
3. Clear and well-organized presentation. The paper is well-written and easy to follow. The motivation, methodology, and experiments are presented in a coherent and logical flow, making the main contributions easy to understand and evaluate.

### Weaknesses
1. Misalignment between motivation and method. The motivation and the proposed method don’t quite line up. While the paper claims to tackle compositional reasoning through an MoE setup, the approach feels more like a form of data augmentation. The results in Table 4 also suggest that the MoE component doesn’t really make a difference — the no-router variant performs almost the same. The largest improvement happens in the Test-N-Scene split of GSA-R2R, but the method doesn’t include any mechanism specifically designed for handling scene-style instructions. This improvement might actually come from the reordering module, where the LLM helps better interpret the instructions, rather than from the MoE design itself.
2. Modest experimental gains and potential data leakage. The experimental results are not very convincing. On the standard R2R benchmark, the method falls short compared to SRDF, which also augments instructions in a similar way. The claimed improvement mainly comes from GSA-R2R, but that dataset includes scenes from HM3D, which are also used in training by ScaleVLN and the proposed SkillNav. This overlap raises a concern about possible data leakage, which could partly explain the performance gains.
3. Missing ablations and unclear MoE behavior. The paper would benefit from more ablation studies to verify whether the MoE setup actually works as intended. Right now, the model uses VLMs to directly predict the skill without any fine-tuning, which is odd given that the paper also introduces a skill-labeled dataset. It’s unclear why that dataset wasn’t used to train or adapt the experts. In addition, there’s no analysis showing how the routing mechanism selects the right expert for a given instruction. Without this, it’s hard to tell whether the model is really leveraging multiple experts or just behaving like a black box.

### Questions
1. In Lines 48–52, the authors state that existing methods tend to memorize examples, limiting their effectiveness in unseen environments. Could the authors provide evidence for this claim? Some recent works, such as ScaleVLN and SRDF, have already narrowed the performance gap between val-seen and val-unseen, which seems to contradict that statement.
2. More explanation is needed regarding the “atomic skills” derived from NavNuances. Since these form the foundation of the proposed method, readers would benefit from a clearer understanding of what each skill represents and how they are defined or annotated.
3. In Table 3, the landmark detection performance is noticeably lower than the other subtasks. Could the authors elaborate on why this happens? Does it relate to data imbalance, annotation difficulty, or inherent limitations of the current detection pipeline?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SkillNav is a modular VLN framework that decomposes navigation into atomic skills, uses an LLM to reorder instructions into subgoals, and employs a VLM-based router to pick the right skill at each step. Each skill has a synthetic, skill-focused dataset and a specialized agent fine-tuned on a DUET backbone, then all agents are integrated for execution. The method attains strong R2R results and state-of-the-art generalization on GSA-R2R, and shows skill-wise gains on NavNuances.

### Strengths
1. Clear modular design with interpretable skills, temporal reordering, and a VLM router that localizes subgoals and selects a single best skill. 
2. Practical synthetic data pipeline that enables skill-specific supervision without human annotation.
3. Reasonable empirical results on GSA-R2R with competitive R2R performance and skill-level improvements on NavNuances.

### Weaknesses
1. It would be great to see if the method can generalize to a broader setting, such as real-world robotic settings or computer-use agent settings. The current evaluations on VLN tasks are somewhat limited and artificial.
2. Compared with other baselines such as SRDF, the model still falls behind and there is a significant performance gap between their model and other baselines on benchmarks.
3. Router effectiveness depends on the chosen VLM, and ablations show nontrivial variance across routers.

### Questions
1. Can the model fit into the current MLLM-based paradigm?

### Soundness
3

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
4

### Summary
The paper proposes SkillNav, a mixture of skill-based framework for Vision-and-Language Navigation (VLN). It decomposes instructions with an LLM-based Temporal Reordering module into ordered sub-goals, then uses a VLM-based Action Router to select among specialized skill agents at each step. The approach aims to improve compositionality, interpretability, and OOD generalization on R2R and GSA-R2R.

### Strengths
1. Clear, modular design with interpretability: Temporal reordering → sub-goal localization → skill routing → action, making intermediate reasoning explicit and auditable.
2. Thoughtful skill taxonomy and expansion: Builds on NavNuances (Direction Adjustment, Vertical Movement, Landmark Detection, Area/Region ID) and adds Stop & Pause and Temporal Order Planning to address frequent failure modes.
3. Strong empirical results under distribution shift: On R2R (Val-Unseen/Test-Unseen) and GSA-R2R (R/N; Basic/Scene), SkillNav achieves competitive to SOTA performance; notably, prior SRDF is strong on R2R but generalizes poorly to GSA-R2R, whereas SkillNav holds up better.
4. Two-stage training that encourages reusable skills: Agents share a DUET-based, skill-agnostic backbone (trained on R2R + ScaleVLN + Temporal synthetic data) before skill-specific fine-tuning—clean separation of training for reuse and specialization.
5. Ablations that isolate key components: Experiments vary reordering on/off and router choices (Random, Qwen2.5-VL-7B-Instruct, GLM-4.1V-9B), showing consistent gains from both Temporal Reordering and VLM routing

### Weaknesses
1. Router dependence on external VLMs: The action router relies on large VLMs in a zero-shot fashion. This raises cost, stability, and reproducibility concerns (model drift, API dependence). Can the authors quantify runtime/cost trade-offs and variance across VLM choices?
2. Synthetic, skill-directed instructions may bias learning: Since skills are trained on tightly targeted synthetic data, do agents overfit to linguistic “triggers”? Consider small-scale human validation or cross-style tests beyond the current splits. (Design suggests targeted datasets per skill.)
3. Coverage and interaction of skills: The taxonomy is compelling, but there is limited quantitative analysis of coverage (how often each skill is needed) and inter-skill interference. A distribution/co-occurrence and error-attribution study would clarify completeness.
4. Reproducibility details: Training/fine-tuning hyperparameters, routing prompts, and fallback policies are not fully specified; reproducing the full stack (especially router behavior) may be challenging.
5. Reproducibility details: Training/fine-tuning hyperparameters, routing prompts, and fallback policies are not fully specified; reproducing the full stack (especially router behavior) may be challenging.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
