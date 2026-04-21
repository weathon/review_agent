# Multi-Task Learning with Hypernetworks and Task Metadata

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
Multi-task learning architectures aim to model a set of related tasks simultaneously by sharing parameters across networks to exploit shared knowledge and improve performance. Designing multi-task architectures is challenging due to the trade-off between parameter efficiency and the ability to flexibly model task differences at all network layers. We propose a novel multi-task learning architecture called Multi-Task Hypernetworks, which circumvents this trade-off, generating flexible task networks with a minimal number of parameters per task. Our approach uses a hypernetwork to generate different network weights for each task from task-specific embeddings and enable abstract knowledge transfer between tasks. Our approach stands out from existing multi-task learning architectures by providing the added capability to effectively leverage task-level metadata to explicitly learn task relationships and task functions. We show empirically that Multi-Task Hypernetworks outperform many state-of-the-art multi-task learning architectures on small tabular data problems, and leverage metadata more effectively than existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a multi-task architecture for tabular data. The proposed architecture includes a hypernetwork which is used to generate model weights for different tasks from task-specific embedding and metadata. The resulting multi-task model is a soft-parameter sharing network that can get good accuracy on datasets with metadata.

### Strengths
- The paper is easy to follow and the proposed method is easy to understand.
- The experiments show the effectiveness of the proposed method especially the usage of meta in tabular data.

### Weaknesses
- Motivation and Practical Usage:
  - Limited Applicability: One major issue pertains to the paper's motivation and practical usage. The proposed method is primarily suited for tabular data, severely restricting its versatility due to its reliance on metadata.
  - Limited Advantages: Furthermore, the paper falls short in demonstrating notable advantages. It does not address key considerations, such as storage or computational efficiency. The motivation behind the trade-off between parameter efficiency and task accuracy, as mentioned in the abstract, is insufficiently explored. In many Multi-Task Learning (MTL) scenarios, like those involving multiple vision tasks in autonomous driving, one expects to reduce storage costs through parameter sharing and even lower computational costs through computation sharing. Regrettably, the proposed method offers no benefits in these two aspects, as it relies on soft-parameter sharing.

- Novelty:
  - Lack of Novelty: Another significant concern pertains to the novelty of the proposed approach. The paper introduces hypernetworks as a central component, which is not a novel concept in MTL. Hypernetworks have already been extensively explored in both Natural Language Processing (NLP) [1] and Vision MTL scenarios [2], diminishing the originality of the paper's contribution.
  - Common Practice: Moreover, employing task embedding as input is a conventional practice in the field of MTL. This conventional approach further diminishes the uniqueness of the proposed method.
  - Limited Generalization: While the paper introduces an insightful use of metadata, its applicability is severely constrained. This novel aspect can only be employed effectively in the context of tabular data, limiting its broader relevance and potentially affecting its acceptance at top-tier conferences.

[1] Mahabadi, Rabeeh Karimi, et al. "Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks." arXiv preprint arXiv:2106.04489 (2021).    
[2] Liu, Yen-Cheng, et al. "Polyhistor: Parameter-efficient multi-task adaptation for dense vision tasks." Advances in Neural Information Processing Systems 35 (2022): 36889-36901.

### Questions
- What do you think makes your method different from the existing methods? In other words, what novelty would you like to emphasize in your method?
- Do you think it is possible to use metadata in general multi-task problems like vision tasks and NLP tasks?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This article presents a method for multi-task learning using hypernetworks, specifically designed for small tabular datasets by leveraging their metadata. A common hypernetwork produces network weights derived from both learned embeddings and the task's metadata. These weights are tailored for each task and are utilized to produce the final predictions relevant to that task.

### Strengths
- The article is well-composed and systematically arranged.

- The concept of employing hypernetworks to produce weights specific to each task is intriguing, particularly with the incorporation of supplementary data (referred to as metadata in the paper). This approach holds potential for crafting robust multi-task networks.

- The assessments are thorough, encompassing many statistical nuances.

### Weaknesses
- The evaluation size for the multi-task learning approach seems limited, with fewer than 1,000 training samples. Determining the efficacy of the suggested deep learning technique from such a compact dataset can be challenging. Other multi-task learning studies have set more expansive benchmarks such as PASCAL-Context, NYUD, and Cityscapes [1], 

- While the paper touches upon some older deep learning techniques (like Cross-stitch and Sluice), it misses out on discussing or comparing several recent deep learning-based multi-task learning methodologies. Notable omissions include InvPT [1], TaskPrompter [2], TaskExpert [3], MQTransformer [4], and MTFormer [5]. Notably, TaskExpert [3] adopts a strategy of learning task-specific gating networks for generating task-dependent weights during task-specific decoding, which bears a resemblance to the hypernetwork learning process to some extent.

- The parameter size and computational efficiency of the methods listed in Table 2 are not shown. I understand that Fig. 6 provides some information about efficiency, but we would like to see the effectiveness vs. efficiency.

References:

[1] Inverted Pyramid Multi-task Transformer for Dense Scene Understanding. ECCV 2022

[2] TaskPrompter: Spatial-Channel Multi-Task Prompting for Dense Scene Understanding. ICLR 2023

[3] TaskExpert: Dynamically Assembling Multi-Task Representations with Memorial Mixture-of-Experts. ICCV 2023

[4] Multi-Task Learning with Multi-Query Transformer for Dense Prediction. arXiv 2022

[5] MTFormer: Multi-Task Learning via Transformer and Cross-Task Reasoning. ECCV 2022

### Questions
- Can you provide a rationale for evaluating multi-task models using just a single tiny-scale multi-task benchmark? 

- How does the performance stack up against one or two of the aforementioned cutting-edge multi-task architectures?

- How does the model perform on datasets such as PASCAL-Context?

I would be open to revising my final score after reviewing the authors' response. Thank you.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a hypernetwork to generate flexible task networks for each task from different task-specific embeddings and metadata. They show empirically that the proposed method outperforms many MTL architectures on small tabular data problems, and leverage metadata more effectively than existing methods.

### Strengths
1. The motivation of this paper is clear.
2. The proposed method can generate flexible task networks.

### Weaknesses
1. The novelty of this paper is limited. The ideas of utilizing hypernetworks for MTL and learning task-specific embeddings have been extensively employed in existing MTL research. The proposed method of this paper is relatively straightforward and does not provide significant novel insights, just an application work not a research paper.
2. The authors claim that "We additionally show experimentally that the task embeddings learn “meaningful” task representations, in that they are predictive of task-level knowledge." However, Section 3.3 only shows the task embeddings can be used to reconstruct metadata and is based on a  model that is needed to be learned. How can this demonstrates the meaningfulness of learned task embeddings? Furthermore, the reconstructed performance is poor.
3. The authors only conducts experiments on small-scale datasets, with the maximum dataset size being 796. It seems challenging to generalize their method to address large-scale MTL tasks. This is because tackling large-scale MTL problems typically requires a target network model with a significantly larger parameter set, which becomes difficult to predict using the hypernetwork. Although the authors propose a method to reduce the dimension of parameters in the hypernetwork generator using an existing technique, it remains unclear whether this compression technique would successfully scale up the hypernetwork to effectively handle large-scale real-life MTL problems.
4. The authors claim on the superior performance of their method paper, by saying the proposed method outperforms SOTA methods. However, the compared methods in this paper are not SOTA methods.  More recent SOTA methods are needed to discuss and compare.

### Questions
see Weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents Multi-Task Hypernetworks that aim to leverage the information from different tasks to improve the performance on each task. In particular, hypernetworks with weight compression are employed in this work to allow knowledge transfer across tasks while introducing a minimal number of trainable parameters. Furthermore, the authors propose to utilize the metadata, which could help during training. Experimental results on different tabular datasets validate the effectiveness of the proposed method.

### Strengths
- The paper is well-written and easy to follow. The authors present their method in detail.
- The proposed method outperforms other baselines on different experimental setups. Ablation studies are conducted thoroughly to understand the effectiveness of each component.

### Weaknesses
- The proposed method is a simple combination of employing hypernetwork and more informative inputs (using handcrafted metadata). Moreover, the idea of utilizing HyperNetwork for Multi-task learning is not novel. It has been studied in prior work (e.g. [1,2]).
- The construction of metadata requires domain knowledge from ML practitioners and potentially includes target leakage features.
- The number of trainable parameters in the proposed method is larger than the target network, which is inefficient for training. This is addressed by using chunk embeddings for different small parts of the main network.

[1] Navon, Aviv, et al. "Learning the Pareto Front with Hypernetworks." International Conference on Learning Representations. 2020.

[2] Lin, Xi, et al. "Controllable pareto multi-task learning." arXiv preprint arXiv:2010.06313 (2020).

[3] von Oswald, Johannes, et al. "Continual learning with hypernetworks." International Conference on Learning Representations. 2019.

### Questions
- Please add the number of trainable parameters/training budgets used for comparative methods in the main tables.
- Can you compare your proposed hypernetwork with weight compression against the chunking technique?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
