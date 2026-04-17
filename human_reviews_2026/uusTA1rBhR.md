# Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4

## Abstract
Safe and feasible trajectory planning is critical for real-world autonomous driving systems.
However, existing learning-based planners rely heavily on expert demonstrations, which not only lack explicit safety awareness but also risk inheriting undesirable behaviors such as speeding from suboptimal human driving data.
Inspired by the success of large language models, we propose Plan-R1, a two-stage trajectory planning framework that decouples principle alignment from behavior learning.
In the first stage, a general trajectory predictor is pre-trained on expert data to capture diverse, human-like driving behaviors.
In the second stage, the model is fine-tuned with rule-based rewards using Group Relative Policy Optimization (GRPO), explicitly aligning ego planning with principles such as safety, comfort, and traffic rule compliance.
This two-stage paradigm retains human-like behaviors while enhancing safety awareness and discarding undesirable patterns from demonstrations.
Furthermore, we identify a key limitation of directly applying GRPO to planning: group-wise normalization erases cross-group scale differences, causing rare, high-variance safety-violation groups to have similar advantages as abundant low-variance safe groups, thereby  suppressing optimization for safety-critical objectives.
To address this, we propose Variance-Decoupled GRPO (VD-GRPO), which replaces normalization with centering and fixed scaling to preserve absolute reward magnitudes, ensuring that safety-critical objectives remain dominant throughout training.
Experiments on the nuPlan benchmark demonstrate that Plan-R1 significantly improves planning safety and feasibility, achieving state-of-the-art performance, particularly in realistic reactive settings.
Our code is available at https://github.com/XiaolongTang23/Plan-R1.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to use RL with rule-based rewards like GRPO to align ego planning with principles for trajectory planning models. Overall, the authors utilize a two-stage paradigm: pretraining and post-training to align with rules. They also modify GRPO into VD-GRPO, so as to remain safety-critical objectives remain dominant. Experiments are conducted on nuPlan dataset, and advantage against other previous methods are shown.

### Strengths
1. The authors study the interesting and important problem on whether recent advances in post-training language model, or basically GRPO, can help improve trajectory planning models. Their observations and modifications on GRPO provide clear and inspiring answer to this question. Especially, they conduct ablation study, analyzing the dominating term of advantages, and show that during GRPO training the unsafe samples are not assigned enough importance.
2. Experiments on nuPlan prove the validity of their proposed method.
3. The dual-model rollout, which utilizes frozen pre-trained model as 'world model' for other driving agents, enables the trained model to interact with other driving agents. This is also an interesting design choice.
Overall, the discovery about GRPO is interesting, and related improvement method, i.e. VD-GRPO, is reasonable and shown to be effective.

### Weaknesses
1. Experiments are mainly conducted on nuPlan, not including other datasets. Given that said, experiments are thorough on nuPlan.

### Questions
Have you tried to use your fine-tuned model as 'world model' for other driving agents? Though I indeed recognize that this might not give too much performance uplift.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Plan-R1, a two-stage trajectory planning framework for autonomous driving. A motion predictor is first pre-trained to capture diverse human-like driving behaviors, and then fine-tuned with rule-based reinforcement learning to align with safety and planning principles. The authors further identify a critical limitation in GRPO's normalization for safety-critical domains and introduce Variance-Decoupled GRPO (VD-GRPO) to preserve absolute reward magnitudes. Combined with a dual-model interactive rollout design, Plan-R1 achieves state-of-the-art performance on the nuPlan benchmark, especially under reactive closed-loop evaluation.

### Strengths
1. The authors clearly diagnose how per-group variance normalization in GRPO suppresses rare but safety-critical violations, and propose a principled solution that consistently enhances safety optimization without sacrificing secondary objectives.

2. Using a frozen world model for surrounding-agent responses ensures interaction-aware rollouts while preventing instability in non-ego behaviors. This design proves crucial for strong performance in reactive scenarios.

3. Plan-R1 achieves new state-of-the-art performance in both non-reactive and reactive nuPlan evaluations, with significant gains in safety metrics such as collisions and drivable area compliance. The ablations convincingly support the contributions of both VD-GRPO and the dual-model rollout.

### Weaknesses
1. Limited analysis of world model reliability when ego deviates from expert behavior. The frozen world model is assumed to remain accurate when the ego policy explores beyond regions well-covered by expert data. However, there is no case study or quantitative analysis showing how the world model behaves under large ego deviations or unusual interaction patterns, which could introduce compounding errors during RL fine-tuning.

2. Tokenization design lacks ablation on discretization choices. The trajectory discretization process (e.g.spatial quantization granularity, or temporal segmentation interval) may significantly influence expressiveness and planner performance. The paper does not provide ablations or analysis on these design factors.

### Questions
Same as weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Plan-R1, a two-stage framework that formulates trajectory planning as language modeling. The idea of decoupling behavior learning from principle alignment, inspired by LLM training paradigms, is both novel and elegant. The proposed Variance-Decoupled GRPO (VD-GRPO) effectively addresses the limitation of standard GRPO by preserving safety-critical gradients, leading to substantial performance gains on the nuPlan benchmark. The experimental section is comprehensive, including detailed ablations and clear visualizations, which convincingly demonstrate the method’s safety and feasibility improvements.

However, the theoretical justification of VD-GRPO (especially the fixed scaling constant) could be further strengthened, and an additional evaluation on another dataset (e.g., CARLA) would help validate generalization. Overall, this is a well-written and technically sound paper with strong empirical results and a creative conceptual contribution.

### Strengths
The paper introduces a novel and well-motivated idea of framing trajectory planning as language modeling through the two-stage Plan-R1 framework. The decoupling of behavior learning and principle alignment is conceptually elegant and practically effective. The proposed VD-GRPO clearly addresses a key limitation in standard GRPO, preserving safety-critical gradients and improving rare-event optimization. Experiments on the nuPlan benchmark are extensive and convincing, with strong gains in both non-reactive and reactive settings. The writing is clear, the figures are informative, and the work demonstrates a high level of technical maturity.

### Weaknesses
While the empirical results are strong, the theoretical justification of VD-GRPO remains limited. The fixed scaling constant is treated as a hyperparameter without principled analysis of its effect on convergence or stability. The dual-model setting, with a frozen world model, may introduce distribution drift in long-term interactions. Moreover, evaluation is restricted to nuPlan; results on other benchmarks such as CARLA or Waymo would strengthen claims of generalization.

### Questions
- Have the authors analyzed the learned motion tokens to see if they correspond to interpretable motion primitives or semantic driving actions?

- How would Plan-R1 perform under partial observability or sensor noise conditions compared to diffusion-based planners?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a two-stage framework called Plan-R1 for safe, feasible trajectory planning in autonomous driving. In the first stage, the model learns diverse, human-like driving behaviors via pretraining on expert data. In the second stage, the model is fine-tuned with rule-based reinforcement learning (RL) to align trajectory planning with explicit principles such as safety, comfort, and traffic rules. To address the limitation of standard GRPO (Group Relative Policy Optimization)—which can dilute safety-critical signals in multi-objective planning—Plan-R1 proposes Variance-Decoupled GRPO (VD-GRPO), which preserves absolute reward magnitudes to ensure the dominance of safety objectives. Experiments show that Plan-R1 significantly improves planning safety and feasibility on the nuPlan benchmark, especially in challenging reactive settings.

### Strengths
• Clear two-stage framework: Plan-R1 decouples behavior learning from principle alignment, retaining human-like behavior while enhancing safety awareness and removing undesirable patterns present in expert data (p.1, lines 054–058).  
• Novel VD-GRPO: In response to standard GRPO’s limitations, VD-GRPO replaces in-group normalization with centering and fixed scaling, effectively preventing rare but critical safety-violation signals from being washed out and ensuring safety-critical objectives dominate training (p.2, lines 075–083; p.6, lines 295–311).  
• Rule-based rewards: Instead of relying on human preference data, Plan-R1 uses rule rewards that offer consistent, unbiased supervision, avoiding bias and improving scalability and reliability (p.3, lines 122–127; p.5, lines 266–269).  
• Dual-model design: During RL fine-tuning, a trainable ego planner explores alternative decisions, while a frozen copy of the pretrained model acts as a reactive world model to predict other agents’ responses, enabling stable, interaction-aware joint prediction (p.2, lines 066–069; p.5, lines 256–265).  
• Significant performance gains: On the nuPlan benchmark, Plan-R1 achieves state-of-the-art performance in both non-reactive and reactive settings, notably surpassing Diffusion Planner by +4.89, +7.98, and +7.11 points in the reactive setting, substantially improving safety and feasibility (p.7, Table 1, lines 370–377).  
• Clear qualitative results: Figures 2 and 3 clearly show how Plan-R1 avoids undesirable behaviors common in pretrained or expert-data-only models, such as speeding, off-road driving, and collisions.

### Weaknesses
• Definition of pivots: The paper states that trajectories are discretized into motion tokens but does not delve into how these “pivot” points are chosen or defined, nor whether such discretization might miss key kinematic or geometric features (p.5, lines 217–223).  
• Detailed analysis of VD-GRPO: Although VD-GRPO is proposed, there is limited theoretical analysis of how its parameters (e.g., the fixed scaling constant c) affect training dynamics; the discussion is mostly empirical (p.9, lines 453–460; p.15, lines 799–807).  
• Missing user-uploaded images: The text mentions “Image generation: enabled,” but no actual images are provided, making it impossible to assess the quality or completeness of the figures referenced in the paper.

### Questions
1) In VD-GRPO, the choice of the fixed scaling constant (c) is empirical. Is there a more theoretical way to determine an optimal c, or is it highly task- and reward-design-dependent? (See p.15, lines 799–807.)  
2) The paper notes that VD-GRPO “replaces in-group normalization with centering and fixed scaling.” Could you provide more concrete implementation details of this “fixed scaling,” and how it differs mathematically from traditional normalization methods? (See p.6, lines 302–303.)  
3) In the ablation study of §4.3, Table 2 shows that VD-GRPO substantially improves the collision metric (+3.45) but has no effect on the comfort metric (+0.00). Could you explain in detail why VD-GRPO impacts some metrics more than others, and whether this relates to the high priority assigned to safety objectives in the reward function? (See p.8, Table 2; p.9, lines 453–458.)

### Soundness
2

### Presentation
2

### Contribution
2
