# Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 6

## Abstract
Reward models (RMs) play a critical role in aligning AI behaviors with human preferences, yet they face two fundamental challenges: (1) Modality Imbalance, where most RMs are mainly focused on text and image modalities, offering limited support for video, audio, and other modalities; and (2) Preference Rigidity, where training on fixed binary preference pairs fails to capture the complexity and diversity of personalized preferences. To address the above challenges, we propose Omni-Reward, a step toward generalist omni-modal reward modeling with support for free-form preferences, consisting of: (1) Evaluation: We introduce Omni-RewardBench, the first omni-modal RM benchmark with free-form preferences, covering nine tasks across five modalities including text, image, video, audio, and 3D; (2) Data: We construct Omni-RewardData, a multimodal preference dataset comprising 248K general preference pairs and 69K instruction-tuning pairs for training generalist omni-modal RMs; (3) Model: We propose Omni-RewardModel, which includes both discriminative and generative RMs, and achieves strong performance on Omni-RewardBench as well as other widely used reward modeling benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses two challenges: (1) modality imbalance and (2) preference rigidity by introducing `Omni-RewardData` and `Omni-RewardBench`. To tackle modality imbalance, the authors collect a large-scale multimodal preference dataset and benchmark across 9 tasks. To address preference rigidity, the authors collect human preference pairs with free-form preference descriptions to enable more fine-grained analysis of human preferences.

The authors systematically evaluate the performance limitations of current reward models (RMs) on multi-modal scenarios and find that they perform poorly on Omni-RewardBench. More importantly, performance varies significantly across different modalities—a phenomenon they conceptualize as "modality imbalance".

### Strengths
1. Introduces free-form preference descriptions to resolve the preference rigidity issue and enable RMs to learn more fine-grained human preferences.
---
2. Collects a human preference dataset from diverse tasks (including T2T, TI2T, T2I, T2V), allowing the model to generalize to novel multi-modal scenarios and adapt to human preferences across different tasks.
---
3. Introduces synthetic instruction-tuning data to allow RMs to adapt to customized evaluation and fine-grained user preferences.
---
4. Provides clear ablation analysis on incorporating both mixed multimodal data and synthetic instruction-tuning user specifications.

### Weaknesses
1. The 3.7K samples in `Omni-RewardBench` were collected from only three PhD students in computer science, which may not reflect general human preferences. The diversity of the benchmark is questionable.
---
2. The mixture of the multimodal preference dataset is lacking an explanation. What would be a good multimodal data mixture ratio?
---
3. Table 3 provides concise evidence for the authors' claim that "using mixed multimodal data leads to significantly better generalization across tasks." However, more rigorous analysis is needed to determine how each modality's task data contributes to preference prediction in other modalities.

### Questions
See above.

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
This paper aims to solve two core challenges in aligning Artificial Intelligence (AI) behaviors with Reward Models (RMs): (1) Modality Imbalance , where existing RMs focus primarily on text and image, with limited support for modalities like video and audio; and (2) Preference Rigidity, where models trained on fixed binary preference pairs fail to capture complex and personalized preferences.

### Strengths
1. One of the paper's most valuable contributions is its clear definition of "Preference Rigidity". Existing RMs typically learn a static reward function $r(x, y)$. This paper proposes transforming it into a dynamic, controllable function $r(x, y | c)$, where $c$ is a free-form text criterion. 

2.Omni-RewardBench is a direct and successful implementation of the problem defined above. It is the first benchmark to systematically evaluate RMs

3. The 69K instruction-tuning dataset built to address "preference rigidity" is a clever engineering contribution

### Weaknesses
1.The model achieves SOTA on VL-RewardBench. Given that the training data (Table 7) includes large datasets like RLAIF-V, was any data contamination check performed to ensure that (near) duplicates from VL-RewardBench or Multimodal RewardBench were not present in the training set?

2.The authors' own Omni-RewardData (Table 7) appears to only contain data for T2T, TI2T, T2I, and T2V tasks. The key modalities used to demonstrate this gap in the benchmark (T2A, T23D, TI2I) are absent from the training data.

### Questions
Please see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work proposes Omni-Reward, an omni-modal reward model benchmark that extends beyond image and text modalities and supports free-form preferences instead of status quo binary preferences as well as a new dataset Omni-RewardData containing preference and instruction tuning multimodal data pairs. The authors use this data to train Omni-RewardModel, which has both discriminative and generative reward model variants, that shows strong empirical performance on their own proposed benchmark, as well as public reward model benchmarks.

### Strengths
* This work proposes an omni-reward model that covers 5 modalities. While multi-modal RMs are not new [1-2], no prior work to my knowledge also covers audio and 3D modality tasks, which is novel and a valuable contribution. The authors also contextualize their work with respect to these prior works (L309-312)
* The paper is well written and easy to follow
* The dataset contribution is valuable to train multi-modal reward models, which the authors demonstrate via the Omni-RewardModel. 
* The trained Omni-RewardModel shows strong performance on the proposed Omni-RewardBench (Tab 1), often beating open source models while lagging behind proprietary models on a few tasks, as well as VL-RewardBench (Tab 2) where it achieves state-of-the-art performance

### References
[1] Zang et al. "InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model", *arXiv 2025*

[2] Wang et al. "Unified reward model for multimodal understanding and generation.", *arXiv 2025*

### Weaknesses
This work has no significant weaknesses in my view. I believe the benchmark, dataset, and trained reward models will be valuable for the preference learning and alignment community. I believe this paper is nearly of spotlight / oral quality, and is only lacking in contextualization to an important area for reward modeling: heterogeneous preference modeling.

There are several recent works that move beyond binary preference pairs and homogenous modeling, though they do not use fully free-form preferences. How does Omni Reward modeling fit in this landscape, as pluralistic alignment to diverse human values is becoming more and more relevant? I suggest adding a comment in the conclusion or brief discussion in the revision about this.

* Knox et al. "Models of human preference for learning reward functions", *TMLR 2024* :  discussing the limitations of learning rewards from binary preference pairs
* Sorensen et al. "Position: a roadmap to pluralistic alignment." *ICML 2024* : a discussion of the importance of pluralistic alignment
* Chen et al. "PAL: Pluralistic Alignment Framework for Learning from Heterogeneous Preferences", *ICLR 2025* : moving beyond preference pair modeling towards heterogeneous reward models
* Liang et al. "Generative Reward Modeling via Synthetic Criteria Preference Learning", *ACL 2025* : moving beyond preference pairs to a preference criteria tree

### Questions
* Is there a specific reason you did not use Pick-a-Pic (v1 or v2) for the text-to-image preference data or ImageReward as a reward model baseline (Tab 1)
* I am curious how does Qwen 3 Omni perform in the T2A and TA2T tasks? I do not expect this experiment in the rebuttal, but it would be nice to include (at least for the audio-specific tasks)

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper reframes reward modeling as criterion-conditioned, omni-modal preference learning to overcome modality imbalance and rigid, binary notions of “helpfulness.” It contributes three components. (1) Omni-RewardBench: a five-modality, nine-task benchmark in which each pairwise comparison is explicitly conditioned on a free-form textual criterion. (2) Omni-RewardData: 317K preference pairs combining general supervision with instruction-tuning data that teaches models to read and follow criteria. (3) Omni-RewardModel: a discriminative Bradley-Terry scorer and a generative, critic-style RM trained with RL that verbalizes brief reasoning before deciding.

### Strengths
The benchmark is notably broad (five modalities and nine tasks) with human-labeled pairs and explicit free-form criteria, evaluated under both w/ ties and w/o ties settings, which is a clear advance over prior suites. The dataset combines large general preference pools with an instruction-tuning subset whose criteria/labels are multi-model verified, improving criterion-following. Empirically, the study is broad, and the BT variant achieves strong accuracy on the new bench and competitive results on public suites.

### Weaknesses
Human annotations come from a small PhD group; after quality control, disagreements are removed, yielding consensus-only labels that may under-sample ambiguous cases. Some modality imbalance remains in distribution. The instruction-tuning split relies on GPT-4o-generated criteria/labels which could encode stylistic biases (though mitigated via verification). Finally, beyond the CoT ablation, the paper would benefit from richer qualitative error analyses across modalities to illuminate failure modes.

### Questions
1. Omni-RewardBench covers more modalities/tasks than prior human-labeled suites but with a comparable or even smaller total size. Are per-task and per-modality sample sizes sufficient?

2. What were the exact quality-control rules for filtering criteria and preference labels? And how does that affect the benchmark data quality?

3. What is the proportion of ties within the preference labels per task/modality?

### Soundness
3

### Presentation
3

### Contribution
3
