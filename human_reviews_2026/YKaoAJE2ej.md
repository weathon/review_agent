# Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
A generalist agent must continuously learn and adapt throughout its lifetime, achieving efficient forward transfer while minimizing catastrophic forgetting. Previous work within the dominant pretrain-then-finetune paradigm has explored parameter-efficient fine-tuning for single-task adaptation, effectively steering a frozen pretrained model with a small number of parameters. However, in the context of lifelong learning, these methods rely on the impractical assumption of a test-time task identifier and restrict knowledge sharing among isolated adapters. To address these limitations, we propose Dynamic Mixture of Progressive Parameter-Efficient Expert Library (DMPEL) for lifelong robot learning. DMPEL progressively builds a low-rank expert library and employs a lightweight router to dynamically combine experts into an end-to-end policy, enabling flexible and efficient lifelong forward transfer. Furthermore, by leveraging the modular structure of the fine-tuned parameters, we introduce expert coefficient replay, which guides the router to accurately retrieve frozen experts for previously encountered tasks. This technique mitigates forgetting while being significantly more storage- and computation-efficient than experience replay over the entire policy. Extensive experiments on the lifelong robot learning benchmark LIBERO demonstrate that our framework outperforms state-of-the-art lifelong learning methods in success rates during continual adaptation, while utilizing minimal trainable parameters and storage.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
DMPEL builds a progressive task-agnostic LoRA expert library and addresses the stability and plasticity trade-off that previous adapter-based continual imitation learning approaches have overlooked in the context of lifelong robot learning. It employs a lightweight routing module with expert coefficient replay, which stores only routing scores and corresponding observation features, to dynamically compose previously learned experts through top-k selection. On the LIBERO benchmark, it improves forward transfer, achieves near-zero forgetting, and is compared against recent adapter-based continual imitation learning methods.

### Strengths
- Addresses an important problem in current lifelong robot learning algorithms, particularly under settings with adapter-based model expansion.
- Shows an efficient design choice by integrating the replay mechanism directly into the routing module, enabling stable and memory-efficient lifelong robot learning.
- The idea of expert merging for lifelong robot learning aligns with adapter-based continual learning in vision and language domains, demonstrating that expert composition consistently improves performance. This is a valuable property because even minor merging errors can lead to significant performance drops.
- Provides strong baselines, including multiple adapter-based continual imitation learning approaches, ensuring a comprehensive comparison, and reports higher performance than training adapters independently with task IDs.
- Offers analysis with clear ablations that substantiate the paper’s main claims, and emphasizes reproducibility through a well-documented report that includes a complete appendix with methodological details, hyperparameters, and supplementary material.

### Weaknesses
- Many building blocks are combinations of ideas from PEFT-based continual learning, so the method is not fundamentally new. Nevertheless, demonstrating that these ideas work effectively in robotic settings is a meaningful contribution. The analysis supports the claims, yet it remains unclear why DMPEL is particularly effective for robots. As with most problems in this area, the underlying insights and their connection to robot learning are under-explained.
- There is a potential fairness concern in comparison: under Top-3 expert routing, DMPEL effectively operates with roughly three times more expert parameters than Seq-FT, TAIL, or IsCiL within the same layer family. While LoRA experts account for only a minor portion of the full model and baselines lack expert-merging capability, evaluating DMPEL (Top-3) against TAIL under matched expert-parameter budgets (or using a Top-1 variant) would make the empirical evidence more convincing.

### Questions
- How can it be applied in terms of granularity? For example, at the level of an action chunk, skill, option, or something else?
- In Figure 3, is the reason the model does not reach multi-task performance due to the low-rank expert’s rank size being insufficient? If the number of trainable parameters per continual learning phase were increased, it seems the performance could improve. Was there another limiting factor?
- For DMPEL trained with Top-3 routing, must the routing during training and evaluation always be the same? For example, can you train with Top-N and evaluate with Top-3, or train with Top-3 and evaluate with Top-1, allowing constraint adaptive scaling?

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
5

### Summary
This paper proposes DMPEL for efficient lifelong learning with reduced catastrophic forgetting. The method establishes a LoRA library for each task, then learns a lightweight router to merge the LoRA experts for new tasks. To avoid forgetting, this paper proposes an expert coefficient replay method, which requires low storage space. Extensive experimental results show that DMPEL achieves a better success rate in forward transfer with almost zero forgetting.

### Strengths
1. Extensive experiments show that DMPEL is a good lifelong learning framework compared to multiple baselines. 
2. Extensive ablation studies demonstrate the functionality of each component.

### Weaknesses
1. Establishing a LoRA library for each task is quite normal, and the expert coefficient replay is also quite standard. 
2. Figure 2 shows that the model inference time is over 50 ms, while the baseline is around 30 ms. What is the latency caused by router computation? What is the latency for averaging the weights? Is there any approach to improve efficiency? If the expert synthesis interval is greater than 1, the model latency will increase rapidly when expert synthesis occurs. 
3. Figure 7 indicates the embedding domain shift problem of SeqFT. SeqFT sequentially fine-tunes the model on new tasks, so it is normal to observe embedding shifts when fine-tuning on a new domain. DMPEL should be compared with other lifelong learning methods.

### Questions
1. How is the expert trained? Is the expert contrained by the router?
2. Figure 3 reports the lifelong learning performance comparison with multiple baselines. Why do some baselines, such as MT, not appear in the FWT and NBT figures? Does PT mean pretraining?
3. Figure 3 introduces MT (multitask learning), but MT does not appear in the baselines listed in Section 5.1. Why is that? Is MT considered an upper-bound performance?
4. DMPEL only introduces 1.2M trainable parameters. Why is the latency still so large?
5. Figure 6a shows that the action head is reused. In the manipulation tasks, the action output mainly consists of end-effector translation, rotation, and gripper movement. Therefore, it is reasonable that one task could cover the entire action space. Could you visualize the action space of each task to verify whether the first task covers the action space of the following tasks? If absolute position is used as the action, the situation may differ. 
6. Figures 11 and 12 show the expert selection during task execution. Is there a more detailed analysis of whether previous knowledge is being reused? For example, in vision, if an object appears in previous tasks (not in the pretraining tasks), the vision experts learned from the first task containing that object should be selected.
7. Appendix B.1 mentions that the router is padded with zeros when transferring from task k to task k+1. What is the network structure of the router? Is the router retrained for each new task? If it is retrained, why is padding needed?
8. When learning a new task, does the expert library help faster training? If library is big enough, is it possible to only finetune the router?

### Soundness
3

### Presentation
2

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
This paper presents a novel framework that integrates modular programmatic policies to enhance the generalization and adaptability of robotic manipulation tasks. The authors propose a dynamic mixture-of-programs architecture, where each subprogram is responsible for a specific manipulation primitive, and a controller dynamically selects and combines them according to task context and sensory inputs.

### Strengths
The paper is written in a clear and coherent manner, with well-organized sections.
The technical flow is easy to follow, and figures are appropriately used to illustrate the overall architecture and key mechanisms.
The manuscript exhibits a complete and logical structure, from problem definition to experimental validation.

### Weaknesses
The paper uses a large number of mathematical symbols and subscripts, some of which appear without a clear prior definition or consistent formatting. This can make certain derivations harder to read, especially for readers not deeply familiar with the notation conventions used.
The current version provides only a brief acknowledgment of potential limitations. I suggest that the authors add a more thorough discussion to strengthen the paper’s self-critique and transparency.

### Questions
1. At line 197, $s$ has already been used to represent state, so at line 210, it would be better to use a different symbol for success rate.
2. The FWT calculation method introduced in Equation 2 differs from the FWT calculation methods presented in [1] and [2]. Can the authors explain why?
3. How are the expert models obtained? Do the expert models increase as tasks increase? If the expert models increase with the number of tasks, then the router's output dimension is constantly increasing—how do the authors address this issue? If the expert models do not increase with the number of tasks, can this method adapt to unexpected tasks? For example, if the predefined number of tasks is 10, but due to changing requirements, the tasks now increase to 20, can this method still adapt? It would be even better if a corresponding experiment could be provided.
4. How do the authors think about the approach of training a separate model for each individual task in CRL—that is, training a dedicated small model for each task? Is this approach better? What is the authors' perspective on this issue? PEFT methods still require more parameters and greater computational resources as tasks increase. From the perspective of parameters and computation, PEFT and separate model training for each task are essentially the same. How do the authors view this issue?
5. In line 376, ER does indeed cause storage and replay computational overhead, but the expert coefficient replay mechanism proposed in this paper also correspondingly causes storage and replay computational overhead.
6. For Figure 5(a), can the authors provide experimental results for all experts + module-wise coeff?
7. What do the labels vision, text, state, ... in Figure 6(a) mean? I suggest providing explanations in the caption as well.
8. The current experiments focus on tasks with consistent state and action spaces. How does [3] address this problem when facing tasks with inconsistent state and action spaces?

references

[1] Continual World: A Robotic Benchmark For Continual Reinforcement Learning

[2] t-DGR: A Trajectory-Based Deep Generative Replay Method for Continual Learning in Decision Making

[3] Solving Continual Offline RL through Selective Weights Activation on Aligned Spaces

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes DMPEL for lifelong robot learning. Unlike prior approaches that rely on oracle task IDs or isolated adapters, DMPEL introduces: A progressively built low-rank expert library using LoRA modules, A lightweight router to select and combine experts for current tasks dynamically, and an expert coefficient replay mechanism to reduce catastrophic forgetting by replaying low-dimensional router coefficients instead of full trajectories.

### Strengths
(1) Dynamic expert composition allows fine-grained adaptation to new tasks without oracle task identifiers.

(2) Full task modularity via LoRA enables parameter efficiency and knowledge sharing across tasks.

### Weaknesses
(1) Quadratic complexity from dynamic expert mixing and routing could scale poorly for larger networks or very long task sequences.

(2) Limited task diversity in evaluation – experiments are restricted to LIBERO simulated environments, not real-world or cross-domain tasks.

(3) No comparisons with learning-based retrieval baselines like contrastive skill indexing or diffusion-based task retrievers.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2
