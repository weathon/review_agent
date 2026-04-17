# Reframing Dense Action Detection (RefDense): A New Perspective on Problem Solving and a Novel Optimization Strategy

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
In dense action detection, we aim to detect multiple co-occurring actions. However, action classes are often ambiguous, as they share overlapping sub-components. We argue that the dual challenges of temporal and class overlaps are too complex to be effectively addressed as a single problem by a unified network. To overcome this, we propose decomposing the task into detecting temporally dense but unambiguous sub-components underlying the action classes, and assigning these sub-problems to distinct sub-networks. By isolating unambiguous concepts, each sub-network focuses solely on resolving dense temporal overlaps, thereby simplifying the overall problem. Furthermore, co-occurring actions in a video often exhibit interrelationships, and exploiting these relationships can improve the method performance. However, current dense action detection networks fail to effectively learn these relationships due to their reliance on binary cross-entropy optimization, which treats each class independently. To address this limitation, we propose providing explicit supervision on co-occurring concepts during network optimization through a novel language-guided contrastive learning loss. Our extensive experiments demonstrate the superiority of our approach over state-of-the-art methods, achieving substantial improvements across different metrics on three challenging benchmark datasets, TSU, Charades, and MultiTHUMOS.  Our code will be released upon paper publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the task of temporal dense action detection, a fundamental and challenging problem in video understanding. The authors propose to decompose dense action labels into two components: action-entity and action-motion, to alleviate the difficulty of modeling overlapping and co-occurring actions. In addition, they employ a noisy contrastive learning objective to provide explicit supervision for co-occurring concepts. Experiments on three benchmark datasets show moderate performance gains compared to prior methods. While the paper is clear and well organized, the conceptual novelty is somewhat limited, and some claims are overstated. The decomposition into entity and motion branches aligns closely with well-established two-stream and relational modeling paradigms in video understanding. Furthermore, the method introduces additional network capacity, making it difficult to disentangle gains due to the proposed design from those due to the larger network.

### Strengths
-	This paper is well organized and easy to follow. 
-	This paper targets a general and important task for video understanding, temporal dense action detection.
-	The decomposition of actions into entity and motion components is conceptually intuitive and may help address overlapping action scenarios.

### Weaknesses
-	The paper claims novelty in addressing simultaneous temporal and class overlaps, but this challenge has been widely recognized in earlier dense detection and multi-label video models. 

-	The statement (L156–L157) suggesting that prior two-stream networks focus only on low-level spatiotemporal features because they are trained end-to-end lacks conceptual clarity and justification. 

-	The two-stream design introduces increased model capacity, making performance comparisons with single-stream baselines potentially unfair. The ablation studies do not convincingly separate the effects of decomposition from additional parameters.

### Questions
The main technical concerns are outlined in the weaknesses section.

A minor question relates to the results. I noticed this manuscript was made public earlier this year, but the results in that version differ from those in the current one. Given that the overall methodology remains largely unchanged, what are the key differences?

### Soundness
3

### Presentation
3

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
This paper introduces RefDense, a framework designed for dense action detection that addresses the challenges of temporal and class overlaps through problem decomposition. The approach consists of two key components: first, actions are decomposed into entity and motion components, with dedicated sub-networks tasked to detect each, thereby simplifying individual learning objectives. Second, a contrastive co-occurrence loss leveraging language guidance is proposed to explicitly capture relationships among frequently co-occurring actions, overcoming the limitation of standard binary cross-entropy loss that treats classes independently. The method is evaluated on the TSU, Charades, and MultiTHUMOS datasets.

### Strengths
1. The paper is clearly written and easy to follow. 
2. The proposed method demonstrates performance gains ranging from 0.4% to 2.1% across the benchmark datasets.

### Weaknesses
1. The idea of decomposing actions into entity and motion components bears resemblance to established paradigms like two-stream networks and several recent works.  [1] Dual detrs for multi-label temporal action detection, CVPR 2024.
[2] Decomposed cross-modal distillation for rgb-based temporal action detection, CVPR 2023.

2. The construction of labels for the sub-networks, specifically the "action-entity" labels, may be problematic. In untrimmed videos, certain entities (e.g., "hammer" in the provided example) might be present throughout the entire video duration, even when the corresponding action is not being performed. This could lead to ambiguous and noisy supervision for the Action-Entity sub-network. The authors should address this potential issue and justify the robustness of their labeling process.

3. The experimental comparisons appear to be limited to other dense action detection methods. To better position the work, it would be valuable to include comparisons with recent state-of-the-art methods in temporal action localization on the same datasets, which would provide a broader perspective on its performance.

4. The ablation studies could be more comprehensive. Key questions remain unanswered: What is the performance of each sub-network (Action-Entity and Action-Motion) when trained and evaluated independently? Is the observed performance gain primarily due to the increased network capacity (using two sub-networks) or the core idea of decomposition? A controlled experiment, for instance, comparing against a single network of comparable parameters, would help isolate the true source of improvement.

5. The figures could be improved for clarity: Figure 1 would benefit from concrete examples of actions (e.g., "pour water") to more directly illustrate the concepts of entity and motion decomposition. There is a typo in Figure 2; the second sub-network is currently labeled "Action-Entity" but should presumably be "Action-Motion."

6. There is a confusing use of the symbol tau in the manuscript. It is used to represent the temperature coefficient in Equation 8 but denotes a window size in Table 2. To avoid confusion for the reader, it is strongly recommended to use distinct symbols for these different parameters.

### Questions
1. What is the fundamental conceptual or technical advancement of your decomposition framework compared to these existing approaches? 

2. How does the Action-Entity sub-network distinguish between an entity being merely present versus being actively involved in an action? Could you provide an analysis or examples from the validation set showing that the entity labels are temporally precise and not overly noisy?

3. How would your method, RefDense, perform against these recent temporal localization models in terms of average precision?

4. What is the standalone performance (e.g., on the decomposed task) of the Action-Entity and Action-Motion sub-networks?

5. Is the performance improvement primarily due to the increased model capacity from having two sub-networks? Have you conducted a controlled experiment comparing RefDense against a single, larger network with a comparable number of parameters?

### Soundness
2

### Presentation
2

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
The manuscript suffers from critical flaws in methodological transparency, experimental rigor, and practical relevance that cannot be addressed through minor revisions. The lack of reproducible details for label decomposition and L_CoLV, confounded generalization experiments, and incomplete engagement with related work undermine the validity of the claimed contributions. To be reconsidered, the authors would need to: (1) fully specify all methodological details (e.g., L_CoLV formulation, GPT-4 prompts), (2) conduct ablation studies to isolate the impact of key components (e.g., parameter count vs. decomposition), (3) validate performance across more diverse qualitative examples, and (4) address practical constraints like computational efficiency and LLM accessibility.

### Strengths
This paper proposed a strategy of decomposing the task into detecting temporally dense but unambiguous components underlying the action classes, and assigning these sub-problems to distinct sub-networks

### Weaknesses
1. The core contributions of RefDense—action label decomposition via GPT-4 and Contrastive Co-occurrence Language-Video Loss (L_CoLV)—lack sufficient detail to support reproducibility and validity
2. The experimental evaluations, while extensive, suffer from biases, unaddressed confounders, and incomplete reporting that undermine the credibility of the claimed performance gains. For example, Confounded Generalization Experiments: When embedding PAT and MS-TCT into the RefDense framework (Table 5), the paper reduces the parameter count of each sub-network (e.g., PAT from 270M to 144M) while claiming "total embedding dimensionality is the same."
3. Inconsistent SOTA Benchmarking: Many SOTA comparisons rely on re-run results (marked †) using the authors’ own code, but fail to validate that experimental conditions (e.g., optimizer hyperparameters, training epochs, data augmentation) match the original papers. 
4. Insufficient Discussion of Limitations and Practicality. The paper does not evaluate decomposition performance with open-source LLMs (e.g., LLaMA-3, Mistral) to assess accessibility. Besides, this paper does not discuss the scalability of LLM-based label generation for larger datasets (e.g., beyond 10k videos in Charades).
5. The related work section fails to engage with recent or relevant literature, leading to an inaccurate positioning of RefDense’s novelty. For example, it oversimplifies Two-Stream Networks and neglects Vision-Language action detection precedents:

### Questions
1. The core contributions of RefDense—action label decomposition via GPT-4 and Contrastive Co-occurrence Language-Video Loss (L_CoLV)—lack sufficient detail to support reproducibility and validity
2. The experimental evaluations, while extensive, suffer from biases, unaddressed confounders, and incomplete reporting that undermine the credibility of the claimed performance gains. For example, Confounded Generalization Experiments: When embedding PAT and MS-TCT into the RefDense framework (Table 5), the paper reduces the parameter count of each sub-network (e.g., PAT from 270M to 144M) while claiming "total embedding dimensionality is the same."
3. Inconsistent SOTA Benchmarking: Many SOTA comparisons rely on re-run results (marked †) using the authors’ own code, but fail to validate that experimental conditions (e.g., optimizer hyperparameters, training epochs, data augmentation) match the original papers. 
4. Insufficient Discussion of Limitations and Practicality. The paper does not evaluate decomposition performance with open-source LLMs (e.g., LLaMA-3, Mistral) to assess accessibility. Besides, this paper does not discuss the scalability of LLM-based label generation for larger datasets (e.g., beyond 10k videos in Charades).
5. The related work section fails to engage with recent or relevant literature, leading to an inaccurate positioning of RefDense’s novelty. For example, it oversimplifies Two-Stream Networks and neglects Vision-Language action detection precedents:

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the challenges of dense action detection, specifically temporal and class ambiguity, by proposing a decomposed approach. The method breaks down ambiguous actions into unambiguous temporal components, assigning them to specialized sub-networks to simplify temporal overlap resolution. Furthermore, it introduces a language-guided contrastive loss to explicitly model the relationships between co-occurring actions, overcoming the limitations of independent class treatment in standard binary cross-entropy. The approach demonstrates superior performance, achieving substantial gains on TSU, Charades, and MultiTHUMOS benchmarks.

### Strengths
+ This paper decomposes the complex problem of dense action detection into simpler sub-tasks of detecting unambiguous temporal components, allowing specialized sub-networks to handle temporal overlaps more effectively.
+ The method demonstrates superior and substantial performance improvements over state-of-the-art methods across multiple challenging benchmark datasets.

### Weaknesses
- The performance gain might be better explained by the sub-networks specializing in foreground entities and actions. This specialization reduces the impact of the background after feature concatenation, which is a perspective that diverges from the authors' stated motivation.
- Missing visualization and quantitative results of two sub-network. The qualitative result comparison among the predicted action-entity, action-motion and the final detection result can help readers understand the reasons for the effectiveness.
- The performance improvements shown in Table 3 and Table 5 are incorrect. Please recheck these tables.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2
