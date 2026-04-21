# Subtask-Aware Visual Reward Learning from Segmented Demonstrations

- Avg Score: 6.20
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 5, 6

## Abstract
Reinforcement Learning (RL) agents have demonstrated their potential across various robotic tasks. However, they still heavily rely on human-engineered reward functions, requiring extensive trial-and-error and access to target behavior information, often unavailable in real-world settings. This paper introduces REDS: REward learning from Demonstration with Segmentations, a novel reward learning framework that leverages action-free videos with minimal supervision. Specifically, REDS employs video demonstrations segmented into subtasks from diverse sources and treats these segments as ground-truth rewards. We train a dense reward function conditioned on video segments and their corresponding subtasks to ensure alignment with ground-truth reward signals by minimizing the Equivalent-Policy Invariant Comparison distance. Additionally, we employ contrastive learning objectives to align video representations with subtasks, ensuring precise subtask inference during online interactions. Our experiments show that REDS significantly outperforms baseline methods on complex robotic manipulation tasks in Meta-World and more challenging real-world tasks, such as furniture assembly in FurnitureBench, with minimal human intervention. Moreover, REDS facilitates generalization to unseen tasks and robot embodiments, highlighting its potential for scalable deployment in diverse environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces REDS, a novel framework aimed at improving reward learning for reinforcement learning (RL) in complex robotic tasks. The framework tackles the problem of designing reward functions, which typically require extensive human effort and domain-specific knowledge. Instead of relying on predefined reward functions, REDS uses segmented video demonstrations where subtasks are labeled, allowing the model to learn from these segmentations with minimal supervision.

The key innovation lies in aligning video segments with subtasks using contrastive learning and employing the EPIC (Equivalent-Policy Invariant Comparison) distance to ensure that the learned reward function is consistent with ground-truth subtasks. The model is tested on both simulated tasks (Meta-World) and real-world tasks (FurnitureBench) and demonstrates superior performance in long-horizon tasks with multiple subtasks.

### Strengths
1. A novel framework for reward learning.
2. Experiments are very extensive, including both simulated and real-world tasks (FurnitureBench).

### Weaknesses
1. The method relies on pre-trained visual and text encoders (e.g., CLIP), which may not be optimal for robotic tasks. The paper acknowledges that subtle visual changes are not well-handled, and reward quality could be further improved with better pretraining.
2. While REDS reduces the need for hand-crafted reward functions, it still requires segmented demonstrations. The quality and quantity of these demonstrations could impact the performance.

### Questions
See weakness.

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
5

### Summary
This paper presents a novel visual reward learning framework that aligns video demonstration segments with learned rewards, effectively indicating progress in subtask completion. The performance improvements across eight simulation tasks and a real-world assembly task demonstrate the method's efficacy. The visualizations of the rewards, the framework's generalization capabilities, and the impacts of various design choices significantly enhance the comprehension of the learned reward.

### Strengths
- The paper is well-organized, providing clear descriptions of the proposed approach, including effective reward visualizations and experimental setups.
- Experiments are thorough and convincing. Results are favorable compared with SOTA approaches like VIPER and DrS. The comprehensive ablation study clearly delineates the impact of each component.
- The experiments on generalization ability and real robots are impressive. It is good to see these properties are held in the proposed method.

### Weaknesses
While I think this paper deserves acceptance based on its strong results and presentation, I have several concerns that should be addressed in the next revision:

- **Context-Aware Reward Signals**: While intuitively beneficial, learning context-aware reward signals may require extensive annotation, potentially weakening the motivation for this work.
    - Existing studies, such as VIP and XIRL, have successfully learned context-aware reward signals without context labels but are not compared in this paper.
    - The method bears resemblance to that presented in [1], which employs purely unsupervised learning without requiring subtasks; this similarity should be discussed and compared.
    - Although the authors suggest that large models could be used to alleviate annotation burdens, this claim is not substantiated in the current version. I recommend either removing this mention or including an experiment that compares large-model-based annotations with human-annotated ones.
- **Clarification of Motivation**: The motivation behind the proposed method needs more clarity. In Line 049, the authors reference prior approaches in the "One Leg Task," but this is not empirically supported in Figure 2 or subsequent sections.
- **Extension to Other Domains:** The applicability of the proposed method to other domains is unclear. Specific rules for subtask segmentation and the threshold $T_U$ should be explicitly described. The authors are encouraged to discuss how to adapt existing methods to other domains, what components need more attention, and potential failure scenarios.
- **More Experiments**: The experiments could be more convincing. While the authors claim that the proposed reward can better handle long-horizon tasks, the selected tasks are not too long-horizon, in my perspective. It is suggested to extend One Leg task to Two/Three Leg Tasks, and see the changes in learned reward and the corresponding performance.
- **Generalization Failures**: It is impressive to see experiments on generalization ability. However, it would be beneficial to present cases where the proposed method struggles. For instance, the authors could include discussions on scenarios such as changes in camera view or background.
- I also include several smaller questions below that should be addressed in the next revision as well.

### Questions
- What rules are used for the division of each task? Could different rules yield varying downstream performance?
- In Table 1, why do DrS, VIPER, and REDS perform worse than Sparse Reward after the offline phase (approximately 1.1 vs. 1.8)?
- What is the effect of $T_U$? Could the authors conduct experiments to explore this?
- In Table 2, while REDS achieves the highest EPIC score, what happens if hand-engineered rewards are replaced with sparse ones (e.g., step-style rewards like in the One Leg Task)? Would the conclusion remain valid?
- In Figure 4, REDS appears continuous in (a) but step-like in (b), despite both rewards being learned to minimize the distance between $\phi$. What accounts for this difference in appearance of the learned rewards?
- Why is Diffusion Reward not included in the experiments? It may outperform VIPER.
- Does the format of the language instruction influence the results?
- What explains the sudden drops for VIPER and DrS and flat curve of ORIL in Figure 10?

[1] Sermanet, P., Xu, K., & Levine, S. (2016). Unsupervised perceptual rewards for imitation learning. RSS, 2017.

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
This paper introduces an inverse reinforcement learning approach that enables the extraction of a dense reward signal from video demonstrations with minimal human intervention. The reward model is designed to consider both the observation and subtask information. This connection between the reward and subtask information intuitively breaks down the intricate long-term task into manageable subtasks. Empirical findings further validate the benefits of integrating subtask information into the reward learning process.

### Strengths
i) In this paper, subtask information is integrated into the reward learning process. During training, the subtask is included in the reward function input as a text embedding that provides instructions on completing a specific subtask. In the inference phase, the text embedding is substituted with a video embedding as an additional input.

ii) The approach employs the EPIC loss function to reduce the disparity between the predicted reward sequence and the ground truth reward. Experimental results demonstrate superiority of this approach.

### Weaknesses
i) In the training phase, this paper decomposes the overall task into multiple subtasks based on domain knowledge. However, the reliance on predefined instructions from the environment for task decomposition raises concerns about practical applicability. Some environments may lack such predefined knowledge, necessitating human annotations or the need for a learned task decomposition model when extending this approach to new environments. This raises doubts about the novelty and scalability of this methodology. 

ii) This paper introduces the function $\psi$, which maps observations to subtasks. However, providing more details about this function is essential. Specifically, elucidating the form of the subtask output by $\psi$ is crucial since it serves as the ground truth for reward learning, a key component for the success of this method. Therefore, the authors should elaborate on the detail of the subtasks produced by $\psi$.

iii) During the reward learning, this method utilizes the EPIC loss function to quantify the disparity between the estimated reward and the proposed ground truth. While EPIC was initially introduced in a prior work, the rationale behind its selection and comparisons with previous loss functions for reward learning should be further discussed by the authors. This explanation should include both intuitive reasoning and empirical evidence.

iv) During the inference phase, the substitution of text embedding with video embedding for subtask information is addressed. Although an alignment mechanism is employed to bring video and text embeddings closer in the latent space, the loss function appears to overlook the maximization of distances between irrelevant embeddings. This oversight could be crucial when dealing with subtasks that share similar text descriptions or video content. It is advisable for the authors to investigate the effectiveness of learned embeddings within tasks involving similar subtasks, although this aspect is not a primary concern.

### Questions
What if we utilize the text embedding directly as input during the inference phase? Does this approach exhibit a substantial performance advantage over the method that employs video embedding?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses the challenge in reinforcement learning (RL) of relying heavily on human-designed reward functions, especially for long-horizon tasks. It introduces REDS (Reward Learning from Demonstration with Segmentations), which infers subtask information from video segments and generates corresponding reward signals for each subtask, utilizing minimal supervision. The framework employs contrastive learning objectives to align video representations with subtasks and uses the EquivalentPolicy Invariant Comparison (EPIC) distance to minimize the difference between the learned reward function and ground-truth rewards.

### Strengths
1. This paper is written clearly and highlights an important challenge for long-horizon reinforcement learning.
2. This paper provides a reward model training method that utilizes both expert demonstration videos and suboptimal videos.

### Weaknesses
- The method seems to rely heavily on carefully predefined subtasks or key completion points (like in table 6). This may limit the generalizability of the method.

### Questions
1. Note that in the experimental setting of this article, expert demonstration videos of the current task are already available. Should imitation-learning-from-observation-based methods also be used as a baseline, for example, to directly derive rewards based on the similarity between the agent trajectory and the expert demonstration video?
2. When learning a new task, how can we determine the appropriate way to divide it into subtasks? For instance, should it be broken down into 2, 3, or 4 subtasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Demonstration with Segmentations (REDS), for long-horizon robotic tasks that are usually composed of numerous subtasks. The key intuition is to use subtask annotations or descriptions for expert demonstrations to aid in learning a dense reward function. The reasoning behind this approach is that the subtasks occur in a particular order in expert demonstrations, and this order can be used as ground truth to learn the reward function. 

The reward model or the reward predictor is defined as a function of the sequence of prior observations and the current subtask. The model is trained using the Equivalent-Policy Invariant Comparison (EPIC) metric. The goal is to match the learned reward function to the ground-truth reward function obtained from the ordering of subtasks in expert demonstrations. To identify the current subtask, the paper uses the cosine similarity between the CLIP image embeddings of the observations and the CLIP text embeddings of the textual descriptions of the subtasks required to achieve the goal. Therefore, before the execution of an episode, the subtasks have to be defined in natural language.

Additionally, the authors propose to continually learn the reward function from suboptimal demonstrations obtained from an RL agent. Here, the RL agent is trained using the reward function obtained from expert demonstrations. This process is repeated multiple times.

 The proposed approach outperforms other reward learning baselines for manipulation tasks from the Meta-world and the FurnitureBench environments.

### Strengths
Overall, the paper is well written, and the problem is very well explained. The authors have clearly delineated their contributions from previous works. The experiments and ablations are detailed and informative. With respect to the proposed approach, the strengths are:
The approach requires minimal human supervision in terms of defining the subtasks accurately.
The proposed approach generalizes well for manipulation tasks with unseen objects. Additionally, since the reward model does not depend on actions, the authors show generalization capabilities to different robots. For example, the reward model trained only with the Panda arm can be used to train RL agents for the Sawyer arm.
Dependence on foundation models like CLIP for subtask identification reduces the adversarial effects of visual artifacts like varying lighting conditions, locations of objects, etc.

### Weaknesses
The expert demonstrations would always contain the subtasks in a particular order. This might lead to poor reward signals when the subtask estimation turns out to be incorrect. Such instances could occur when the RL agent is exploring. 
The effect of the hyperparameter epsilon, which is used to enforce progressive reward signals within each subtask, is not clearly explained. The authors show an ablation for the cases with and without regularization loss. But, the effect of epsilon is not clearly described. 
From Fig. 3, it appears that the baselines have not converged for the Lever Pull, Sweep Into, and Push cases. 
The proposed approach is not compared against CLIP-based zero-shot reward models like [1].
LIV [2] also learns dense rewards from just observations and text descriptions. The proposed approach has to be either differentiated from or compared against this paper.

[1] Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning. 
[2] LIV: Language-Image Representations and Rewards for Robotic Control.

### Questions
Questions:
Is there a way to quantify the subtask identification performance? This could be useful as the identified subtask affects the reward prediction. 
An ablation study depicting the effect of epsilon, which is used to enforce progressive reward signals, will be useful. CLIP-based subtask identification and ground-truth rewards from subtask ordering are not sufficient to enforce the required expert behavior. From my understanding, this is implicitly enforced using progressive reward signals. Therefore, an ablation study for epsilon will be crucial.
The authors need to explain how their approach is different from [1] and [2].

[1] Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning. 
[2] LIV: Language-Image Representations and Rewards for Robotic Control.

### Soundness
3

### Presentation
3

### Contribution
3
