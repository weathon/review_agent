# KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Augmentations and Constraints

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Large Multimodal Models encode extensive factual knowledge in their pre-trained weights. However, its knowledge remains static and limited, unable to keep pace with real-world developments, which hinders continuous knowledge acquisition. Effective knowledge injection thus becomes critical, involving two goals: knowledge adaptation (injecting new knowledge) and knowledge retention (preserving old knowledge). Existing methods often struggle to learn new knowledge and suffer from catastrophic forgetting. To address this, we propose KORE, a synergistic method of KnOwledge-oRientEd augmentations and constraints for injecting new knowledge into large multimodal models while preserving old knowledge. Unlike general text or image data augmentation, KORE automatically converts individual knowledge items into structured and comprehensive knowledge to ensure that the model accurately learns new knowledge, enabling accurate  adaptation. Meanwhile, KORE stores previous knowledge in the covariance matrix of LMM's linear layer activations and initializes the adapter by projecting the original weights into the matrix's null space, defining a fine-tuning direction that minimizes interference with previous knowledge, enabling powerful retention. Extensive experiments on various LMMs, including LLaVA-v1.5-7B, LLaVA-v1.5-13B, and Qwen2.5-VL-7B, show that KORE achieves superior new knowledge injection performance and effectively mitigates catastrophic forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a multimodal knowledge injection framework that integrates structured data augmentation with a null-space constraint. It aims to balance new knowledge learning and old knowledge preservation through automatic multimodal task generation and covariance-based regularization.

### Strengths
The paper is visually well-presented with clear figures and tables, and the writing is coherent and easy to follow.

### Weaknesses
1.	The method uses GPT-4o to generate augmented data, which introduces the risk of incorporating external knowledge. This may lead to unfair comparisons with baselines that do not rely on external models. In other words, the performance gains could partly result from distillation from GPT-4o rather than from the proposed method itself.
2.	The “Knowledge-Oriented Constraint” closely resembles AlphaEdit’s [1] null-space approach. The paper should clarify its conceptual or technical novelty beyond AlphaEdit.
3.	KORE-Augmentation and KORE-Constraint operate independently without clear interaction, making the framework appear as two parallel components rather than an integrated system.
4.	Several conceptually or methodologically related works were not discussed. The paper should include an analysis and discussion with [1][2][3].
5.	The presentation of the theorems in the paper is not standardized and is difficult to follow. For example, in Theorem 1, the symbols are not defined within the statement itself but are introduced later in the proof. The theorem statements should be formalized and rewritten in a clearer, more rigorous manner.

[1] AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models, ICLR

[2] LoRASculpt: Sculpting LoRA for Harmonizing General and Specialized Knowledge in Multimodal Large Language Models, CVPR

[3] LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation, COLM

### Questions
1. The roles of the two theorems in the paper are unclear; their purpose and contribution to the overall method are not explicitly explained.
2. The LoRA rank used in this paper appears relatively high compared to prior works. How does the proposed method perform under lower-rank settings?
3. In Table 5, the W/o Constraint variant performs about 3% better than KORE on the EVOKE dataset. Does this indicate that the proposed constraint may negatively affect performance on EVOKE?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the KORE method, which achieves a balanced integration of new and existing knowledge in large models through the collaboration of KORE-AUGMENTATION and KORE-CONSTRAINT. The former automatically generates multi-turn dialogues and multimodal task data to promote knowledge internalization, while the latter uses null-space constraints to prevent forgetting. Experiments demonstrate that KORE significantly enhances the knowledge learning and retention capabilities of multimodal models.

### Strengths
- KORE-AUGMENTATION is highly innovative and serves as a reasonable and effective data augmentation approach.
- The method achieves strong empirical results, reaching state-of-the-art performance.

### Weaknesses
- The core idea of KORE-CONSTRAINT is quite similar to AlphaEdit [1] (both employ projection onto the null space to mitigate interference with prior knowledge), which weakens the originality of this work.
- Experiments are conducted only on the EVOKE benchmark, while another equally important benchmark in this field, CoIN [2], is neglected.

[1] [2025-ICLR] Alphaedit: Null-space constrained knowledge editing for language models  
[2] [2024-NeurIPS] CoIN: A benchmark of continual instruction tuning for multimodel large language models

### Questions
- AlphaEdit [1] also employs projection onto the null space to mitigate interference with prior knowledge. Could you clarify how your approach differs from theirs?
- SEFE [2] consists of two components, ASD and RegLoRA. Since the authors of SEFE did not apply ASD to the EVOKE dataset, did you fully reproduce ASD on the EVOKE benchmark for comparison, or are your SEFE results based solely on its RegLoRA component?

[1] [2025-ICLR] Alphaedit: Null-space constrained knowledge editing for language models   
[2] [2025-ICML] SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning

### Soundness
3

### Presentation
4

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
The paper investigates the continual learning in Large Multi-Modal Models. To solve the balance of learning new knowledge and knowledge retention, they propose a KOPE. Extensive experiments demonstrate that KOPE achieves new knowledge injection performance and mitigates catastrophic forgetting.

### Strengths
The proposed method sounds reasonable, which makes new knowledge structured, and the covariance matrix keeps previous knowledge.

The experiment in the paper is sufficient, including multiple MLLMs and multiple downstream tasks.

### Weaknesses
The survey is insufficient, e.g., related works and baselines have few papers in 2025.

For Figure 5, why is Full-FT lower than KOPE? Theoretically, full fine-tuning is an upper bound for any method.

During the training, how to split the dataset into new knowledge datasets and old knowledge datasets?

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
