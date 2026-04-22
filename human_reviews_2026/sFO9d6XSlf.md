# Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Fine-tuning vision-language models (VLMs) on robot teleoperation data to create vision-language-action (VLA) models is a promising paradigm for training generalist policies, but it suffers from a fundamental tradeoff: learning to produce actions often diminishes the VLM's foundational reasoning and multimodal understanding, hindering generalization to novel scenarios, instruction following, and semantic understanding. We argue that this catastrophic forgetting is due to a distribution mismatch between the VLM's internet-scale pretraining corpus and the robotics fine-tuning data. Inspired by this observation, we introduce VLM2VLA: a VLA training paradigm that first resolves this mismatch at the data level by representing low-level actions with natural language. This alignment makes it possible to train VLAs solely with Low-Rank Adaptation (LoRA), thereby minimally modifying the VLM backbone and averting catastrophic forgetting. As a result, the VLM can be fine-tuned on robot teleoperation data without fundamentally altering the underlying architecture and without expensive co-training on internet-scale VLM datasets. Through extensive Visual Question Answering (VQA) studies and over 800 real-world robotics experiments, we demonstrate that VLM2VLA preserves the VLM's core capabilities, enabling zero-shot generalization to novel tasks that require open-world semantic reasoning and multilingual instruction following.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces *VLM2VLA* to overcome the trade-off where training a powerful VLM to control a robot leads to a loss of its general understanding. By first translating robot actions into natural language, VLM2VLA bridges the gap between the VLM's pretraining and the robot data. This technique allows the model to be efficiently fine-tuned with LoRA, keeping its original knowledge intact. This successful preservation is experimentally validated by the robot's ability to demonstrate advanced semantic reasoning and zero-shot generalization on novel, complex tasks.

### Strengths
- The paper is well-written and structured.
- The problem is clearly stated and is highly interesting and relevant to the robot learning community.
- The solution is quite novel. The idea of using natural language to describe actions is interesting and exceptionally well-presented and motivated in the paper, particularly in Figure 3.
- The experimental setup is strong; however, the results are most fascinating in the Out-of-Distribution (OOD) tasks, especially those where instructions were given in different languages. This particular study provides strong evidence that the *VLM2VLA* method effectively preserves the VLM's capabilities. 
- The limitations and future work are clearly stated.

### Weaknesses
While this work presents many positive aspects, I offer a few suggestions that could further strengthen its quality and experimental validation.
- The baselines $\pi_{0.5}$ [1] and MolmoAct  [2] should also be evaluated on the real-world tasks for completeness, consistent with the other baselines in Section 4.2. Including these results would significantly strengthen the experimental validation and make the overall conclusions more robust.
- I believe the performance on the VQA tasks for the corresponding reference VLM models for $\pi_{0.5}$ [1] and MolmoAct [2] should be added as done with OpenVLA by including Prismatic VLM and VLM2VLA by including Gemma-3.
- While the chosen tasks are compelling and relevant, the overall number of experimental tasks discussed in the paper remains limited.

[1] Intelligence, Physical, et al. "$\pi_ {0.5} $: a Vision-Language-Action Model with Open-World Generalization." arXiv preprint arXiv:2504.16054 (2025).

[2] Lee, Jason, et al. "Molmoact: Action reasoning models that can reason in space." arXiv preprint arXiv:2508.07917 (2025).

### Questions
I have a couple of questions regarding the experimental methodology:
- Have you ablated different VLM models for the purpose of task curation?
- Was the robot learning fine-tuning based exclusively on the Bridgev2 dataset, or did it also include real demonstrations collected from the exact kitchen setup depicted in Figure 5? 
- I am also interested in whether the In-Distribution (ID) tasks are considered ID solely based on the instruction and objects, or if they also rely on the identical physical kitchen setup and object locations.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the VLM2VLA pipeline, which turns VLMs into VLAs while maintaining their generalization performance for OOD settings. It addresses the critical issue of VLA pretraining degrading VLM features over time, due to the large domain shift between VLM training and action generation. The concept is based on addressing the distribution mismatch between actions and text, which typically undermines the knowledge gained from VLM pretraining.

The method involves modeling actions using words and then fine-tuning the VLM on that action word data. Since the VLM has encountered many instances of numbers in the context of science texts, etc., it already has a good understanding of them and can easily grasp the concept of action logic. Crucially, it's a VLA training pipeline that uses LoRA to maintain the VLM features. It also uses a hierarchical reasoning setup in combination with the text actions: high-level subtask -> mid-level motion planning -> low-level action. Results on VLM benchmarks and various real-world manipulation setups demonstrate that the proposed method is able to preserve VLM knowledge while serving as a good VLA with improved OOD performance.

### Strengths
- This is an overall very insightful paper with good experiments and results, which is helpful for the VLA community.
- It also tackles a critical issue in VLA pretraining, where the VLA training process destroys the VLM's foundational features
- The Action as Language Representation is a novel with good results but limited ablations
- Demonstrating that this method can work with VLA training solely using LoRA is a significant advantage for achieving efficient VLA adoption from VLMs.
- The OOD experiments and results are promising and demonstrate that the method effectively maintains VLM generalization.

### Weaknesses
- The current main weakness of the paper is the weak ablations of their key contributions: no ablation of how the 3-stage word actions impact the training, with baselines that only predict low-level actions, mid-level ones, etc
- Low-level action has translational DoF only, without any more complicated movements
- Very low inference speed, with several seconds for a single action prediction
- No VLM-centric ablations. It would have been interesting to study how different VLM backbones affect the performance of this pipeline

### Questions
- Can you add ablations for the hierarchical word action setup to study the impact of each hierarchy level on the performance? Especially since all are generated by Gemini and not humans. How important are good plans? Would be adoption work also with false traces? More insights here would strengthen the paper
- Additional tests with other VLMs would also be great to see how the pipeline performs across this axis
- Is LORA for all training stages a consequence of limited compute? How would it behave with smaller VLMs and full finetuning? Any insights on LORA rank and downstream performance?
- Does it scale to SIMPLER benchmark evaluation with 7 DoF? If so, what are the results?

### Soundness
3

### Presentation
4

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
This paper proposes VLM2VLA, a data-centric method for fine-tuning Vision-Language Models (VLMs) into Vision-Language-Action (VLA) models while minimizing catastrophic forgetting. The key idea is to represent robot actions as natural language phrases rather than discrete tokens or continuous vectors, allowing fine-tuning via LoRA without modifying the base VLM architecture.

### Strengths
1. Casting low-level actions as text tokens is conceptually elegant and well aligned with the VLM’s pretraining distribution, avoiding token-level mismatches common in prior VLAs.
2. Experiments show that multimodal reasoning and multilingual understanding are preserved, as in across diverse VQA benchmarks and real-world robotic tasks.
3. The proposed pipeline of using LoRA fine-tuning only avoids costly co-training or architecture modifications.

### Weaknesses
1. The comparison with baselines on robotics tasks are limited. MolmoAct and π0.5 are mentioned in Table 1 but not reported in robot experiments. Other VLA works that address the catastrophic forgetting problems[2] or unifying the architectures[1] are not compared.
2. Robot experiments are relatively simple and there are no simulation experiments, which could have provided scalable quantitative metrics.
3. While compared with an action expert ablation with discrete tokens, there is no comparison with flow-matching or diffusion-based action experts such as π0.5 and MolmoAct, which provides state-of-the-art continuous control.
4. The proposed three-stage hierarchy (subtask → motion plan → action chunk) is promising but conceptually overlaps with system-1/system-2 reasoning paradigms (planning vs. acting). A comparison with a 2 system VLA is needed to validate the proposed method, where a system2 is responsible for planning and translating and a system1 is responsible for action generation.

[1] Lin, Fanqi, et al. "OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning." arXiv preprint arXiv:2505.11917 (2025).

[2] Huang, Huang, et al. "Otter: A vision-language-action model with text-aware visual feature extraction." arXiv preprint arXiv:2503.03734 (2025).

### Questions
1. What’s the X axis of figure 6 and does the images of fish and country flag correspond to the plot?
2. Why only are translational dofs considered?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focus on a reasonable issue, finetuning VLM to VLA will cause the VLM lost its memory for QA but overfit to robot control. The paper proposes to solve this issue by convert the low level action to natural language format, then use lora finetuning.

### Strengths
1. The scope of this paper makes sense, which i agree that fine-tuning the VLM to VLA wil let the VLA lost its QA reasoning ability and overfit to specific control actions prediction.

2. The paper is well organized and easy to follow.

### Weaknesses
1. Even the paper throws a meaningful problem, which is how to fine-tune a vlm to vla without degrade its ablity for reasoning, but the paper's solution is not cool enough, since just format low level actions as language then train it using formal vlm training, this is not novel and used by many papers like [1][2]

2. Not sure what is the paper's main contribution, seems Table 1 want to claim VLM2VLA performs better in series VQA task, but, do the VQA person will use VLA for VQA task, or Robotics ppl care these VQA task? So not sure if evaluate on these VQA task really make sense if this work focus on VLA.


[1] Niu, D., Sharma, Y., Biamby, G., Quenum, J., Bai, Y., Shi, B., ... & Herzig, R. (2024). Llarva: Vision-action instruction tuning enhances robot learning.
[2] Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., ... & Han, K. (2023, December). Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (pp. 2165-2183). PMLR.

### Questions
See Weakness for big concern. Besides those:
1. Seems in section 4.2, it shows some results on robotic control task, but there is some conerns:
a. it deos not compare with a strong baseline, like pi0
b. the task is too simple, mainly pick and place, which i think should focus more on long horizon task, which might better show the method can keep more reasoning or planning ability of the pretrianed VLM.

### Soundness
3

### Presentation
3

### Contribution
2
