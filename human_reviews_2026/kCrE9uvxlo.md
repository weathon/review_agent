# DeepThinkVLA: Enhancing Reasoning Capability of Vision-Language-Action Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 8, 4

## Abstract
Enabling Vision-Language-Action (VLA) models to "think before acting" via Chain-of-Thought (CoT) is a promising path to overcoming the data-hungry nature of end-to-end robot policies. However, progress is stalled by a fundamental conflict: existing models use a single autoregressive decoder for both sequential CoT reasoning and high-dimensional, parallelizable robot actions. This architectural mismatch degrades motor control and fails to forge a strong causal link between thought and action. We introduce DeepThinkVLA, which resolves this conflict through a tightly integrated architecture and training strategy. Architecturally, our hybrid-attention decoder generates sequential CoT with causal attention and then switches to bidirectional attention for fast, parallel decoding of action vectors. This design is complemented by a two-stage training pipeline: we first use Supervised Fine-Tuning (SFT) to teach the model foundational reasoning, then apply Reinforcement Learning (RL) with task-success rewards to causally align the full reasoning-action sequence with desired outcomes. This synergy leads to state-of-the-art performance, achieving a 97.0\% success rate on the LIBERO benchmark \textcolor{red}{and demonstrating robust generalization on the high-fidelity RoboTwin 2.0 benchmark, outperforming the strongest baseline by 21.4\%.}. Our ablations confirm the design's effectiveness: the hybrid architecture alone outperforms standard decoders by 15.5\%, and the final RL stage provides a crucial 2\% boost to secure top performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DeepThinkVLA, a vision-language-action (VLA) model that enables robots to "think before acting" through explicit Chain-of-Thought (CoT) reasoning. The authors point out that existing VLAs face a fundamental architectural conflict, as a single autoregressive decoder must handle both sequential reasoning and parallel, high-dimensional actions. DeepThinkVLA addresses this by employing a hybrid-attention decoder, using causal attention for CoT generation and bidirectional attention for fast, parallel action decoding. An outcome-based reinforcement learning (RL) stage further aligns reasoning with task success, improving performance beyond simple imitation learning. The model achieves a 97.0% success rate on the LIBERO benchmark.

### Strengths
- The paper points out a fundamental yet previously underexplored architectural conflict between sequential reasoning and parallel action prediction in VLAs.

- It proposes a novel hybrid-attention decoder, combining causal attention for reasoning and bidirectional attention for action generation, effectively bridging this gap.

- The authors recognize the potential mismatch between CoT reasoning and task success, introducing an outcome-based RL stage that directly aligns the reasoning-action sequence with task performance.

- The figures are clear and informative, effectively illustrating the proposed model architecture and training/annotating pipeline.

### Weaknesses
- Annotating CoT is crucial for the proposed approach, yet the paper lacks sufficient evidence showing that the pipeline for constructing an embodied CoT dataset performs well in more diverse or realistic settings.

- Although the outcome-based RL stage improves task success by 2% on the LIBERO-Long suite, this gain is relatively modest, and the paper does not explore why the improvement is limited or how the method might scale with larger datasets or more complex tasks.

- The description of the CoT dataset construction pipeline is somewhat confusing. Section 3.3 only mentions detecting gripper state changes to identify keyframes, but elsewhere a human reviewer is referenced (Figure 2). The role and extent of human involvement are unclear.

- If human reviewers are indeed required for CoT annotation, this could introduce a bottleneck in scalability and limit the practicality of the proposed approach for large-scale or real-world robotic datasets.

### Questions
- What is the main reason behind the architectural conflict between sequential reasoning and parallel action prediction? Could the authors elaborate on why a single autoregressive decoder cannot effectively handle both modalities?

- The paper combines SFT pretraining with RL fine-tuning, a paradigm often used in large language models. What are the key differences when applying this paradigm to VLAs?

- How reliable is the fine-tuned local VLM used for CoT annotation, and how might its quality or bias affect the overall performance of DeepThinkVLA?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper is about enabling chain-of-thought in vision-language-action models for applications in robotics. The claimed contributions are by introducing (a) a hybrid-attention decoder that switched between causal attention and bidirectional attention and (b) a training strategy consisting of supervised fine-tuning of the model followed by reinforcement learning with a success-based reward function. According to the presented results, this approach reached high success rated on a common benchmark. Investigation the individual merits of the components of the proposed approach, the authors find that the architecture alone contribute 15.5% improvement over SOTA and the RL component another 2%.

### Strengths
- The paper identifies and acts upon an important and relevant problem in the robotics and embodied AI community.

- The paper is written clearly and has a good structure. Particularly, the description in form of the "probabilistic decomposition" is appreciated relative to the common "diagram with arrows approach" used in many other submissions. 

- The approach seems innovative and original in the way CoT and action generation are combined, but given the large body of work on VLA models and CoT and the fast growth of this research area, this is hard to judge at this point. 

- The paper provides the pipeline with which the data augmentation pipeline for generating the dataset that is necessary to train their model. As this type of data is hard to obtain, this is an important part of the overall approach. 

- The reinforcement learning component makes a few interesting contributions. Particularly the way the reward and the advantage are computed and how the objective is regularized. 

- The experiments use a current and relevant "base model" (Pertsch 2025) and therefore provide a good basis for evaluating the proposed approach against SOTA.

### Weaknesses
- The benchmark / dataset used in experiments only focusses on robotic manipulation tasks. It is clear that the authors have to use what is available, but from a robotics perspective, these are still fairly limited tasks. 

- One of the main issues of using VLAs in this way is the question of what happens when the robot or its abilities are different from the training data. The paper does not test the case there the robot has to learn to solve the tasks in a different way from what the VLA thinks is right. Because there exists a reward signal, this should be an interesting evaluation regarding to generalization ability of the approach. An example would be to limit the joints of the downstream agent so that it has to diverge from the VLA.

### Questions
- The ablation experiment on page 7 is bit confusing. Could the authors clarify why this experiment evaluates the role of CoT. Why are the results of masking vs. random so different? Does this man that training with CoT is merely an auxiliary task and that it is not really needed in the final inference?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DeepThink VLA, which make improvements from base model pi zero -FAST by doing the following 

1) On architecture, uses a hybrid-attention architecture to enable both autoregressive CoT reasoning and efficient parallel action decoding. 
2) On training, uses a two stage pipeline. First do SFT on reasoning data curated by querying key frames from a large model, then do RL with task success as reward to fully ground reasoning in robot task.

### Strengths
This is a strong paper with good results. It is well-written and easy to read. The core idea of using CoT to enrich the model's internal representations is novel. This approach appears to be very effective, as demonstrated by the strong experimental results, particularly on long-horizon tasks. The ablation study on the role of CoT is convincing, showing it strengthens representations during training.

### Weaknesses
1) The benefits of the RL stage is not obvious to me, especially given the strong results of SFT. The paper would be strengthened if it could further demonstrate the unique benefits of RL. For example, are there OOD or more complex generalization experiments where the RL-trained model significantly outperforms the SFT-only model? Such experiments would provide stronger evidence that RL is crucial for aligning reasoning for novel scenarios, rather than just providing a minor boost on in-distribution tasks.

2) The paper presents the hybrid-attention architecture as a main contribution, but it is not a particularly novel idea.

### Questions
Do authors consider other thinking modalities to improve model representations? e.g. bounding boxes, end effector trajectories, etc.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors study VLAs, and want to update VLAs from models that directly predict action chunks to ones that first generate CoT tokens in language space before generating action chunks.

To do so, the authors construct a CoT dataset out of robot episodes. Assuming that changes in the gripper state correspond to semantically important intermediate steps of the episode, they pull out image frames with gripper change and ask for language descriptions by a more powerful VLM. Then, the remaining intermediate frames are labeled by a weaker VLM that can be run locally.

This is done to create an SFT dataset, and the model is further trained using GRPO. The reward for GRPO is sparse, covering both task success and whether the CoT is well formatted or not. The authors run their experiments on the LIBERO benchmark.

The authors additionally argue that for robot control in particular, action chunks need not be generated autoregressively, since they are a single coherent piece of information, so it is okay to decode all the chunked tokens in parallel to speed up inference.

### Strengths
The ways to construct a good embodied dataset for robot control is still pretty open and further work there is interesting. RL is also an interesting area of study, given that many VLA approaches have focused on just doing imitation learning and have not done as much exploration into further RL work on top.

### Weaknesses
In terms of the attention decoding, I don't think the proposed Hybrid-Attention is meaningfully different from the CoT-VLA-7B Zhao et al 2025b cited work, which generates subgoals as a CoT using causal autorgressive attention, then a full parallel attention for actions like this work. The only difference seems to be using text based subgoals compared to image based subgoals.

The paper spends little time acknowledging that the results for DeepThinkVLA use an additional wrist-mounted camera, which is very important to results (as noted in Figure 4 this is the difference between an 86% policy and a 94.2% policy, a pretty significant boost. This weakens the comparisons in Table 1, since almost all numbers do not use a wrist camera (with the exception of the 96.8% spatial number for pi_0)

Experiments are primarily done in simulation, compared to prior work which included some real robot results as well. Similarly the authors assume a perfect task success detector, due to using sim, which somewhat works against the argument that RL would be helpful.

The authors say that when the CoT is ablated to a blank <think></think> section, success rates are not that different, and argue that this shows the CoT benefit is primarily on learning better representations. But to me this seems like it's showing that CoT is not that necessary for learning action behavior. That's not to say it is useless in general, to me it suggests the LIBERO benchmark is just not strong enough to evaluate the usefulness of embodied CoT, and by extension this paper (which only uses LIBERO) is not making a strong enough case for the CoT.

This, combined with other works using similar bidirectional mechanisms + added real robot results, makes me unsure this is a large enough contribution for ICLR.

### Questions
When there is a reward of 1 for "if CoT format correct"< what does that actually mean? I did not find an example explaining what incorrect CoT format looked like.

### Soundness
3

### Presentation
3

### Contribution
2
