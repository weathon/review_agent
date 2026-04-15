# Rapid Learning without Catastrophic Forgetting in the Morris Water Maze

- Decision: Reject
- Scores: 3, 5, 6, 6

## Abstract
Machine learning models typically struggle to swiftly adapt to novel tasks while maintaining proficiency on previously trained tasks. This contrasts starkly with animals, which demonstrate these capabilities easily. The differences between ML models and animals must stem from particular neural architectures and representations for memory and memory-policy interactions. We propose a new task that requires rapid and continual learning, the sequential Morris Water Maze (sWM). Drawing inspiration from biology, we show that 1) a content-addressable heteroassociative memory based on the entorhinal-hippocampal circuit with grid cells that retain knowledge across diverse environments, and 2) a spatially invariant convolutional network architecture for rapid adaptation across unfamiliar environments together perform rapid learning, good generalization, and continual learning without forgetting. Our model simultaneously outperforms ANN baselines from both the continual and few-shot learning contexts. It retains knowledge of past environments while rapidly acquiring the skills to navigate new ones, thereby addressing the seemingly opposing challenges of quick knowledge transfer and sustaining proficiency in previously learned tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the challenge of catastrophic forgetting in the context of continual learning. The proposed study focuses on the sequential Morris Water Maze (sWM) task which is inspired by several mechanisms used by biological systems. It combines a content-addressable memory system and a convolutional network architecture to implement these mechanisms in the context of ANNs. This model excels at fast learning, generalization, and continuous learning, outperforming baselines in both continuous and few-shot learning settings.

### Strengths
The paper reads well and can be followed straightforwardly.

### Weaknesses
1. It is not clear that the proposed task is of practical importance.


2. Comparison is extremely limited and considers old methods such as EWC.

3. The code is not provided which makes judgment about reproducibility challenging.

### Questions
The major novelty that I see in this paper is building connections between biological systems and ML. However, it is not clear how relevant the proposed task is and how effective it will be in the context of CL. The question is then why the paper has not done an evaluation according to the precedent given the fact that CL is an extremely well-established field by including SOTA methods.


============Post Rebuttal=============
Thanks for the rebuttal. I changed my rating accordingly. I don't find this work compelling and mostly find a proof-of-concept level work which is not clear whether it will be of practical relevance. The task is a limited synthetic task and I cannot think of a major benefit for future research.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on catastrophic forgetting in lifelong learning scenarios. Inspired by inspiration from the spatial learning mechanisms observed in biological neurons,  this paper introduces a new task, sequential Morris Water Maze (sWM), for rapid adaptation and continual learning.  Furthermore, the paper presents a lifelong learning approach built upon the Memory Scaffold with Heteroassociation (MESH) architecture, designed to promote generalization within the Water Maze environments.  However, the description of the proposed method requires further clarity.

### Strengths
The paper studies a practical and important problem: lifelong learning without catastrophic forgetting, focusing on the performance in sequential Morris Water Maze tasks.  

The proposed method, a bio-inspired lifelong learning framework based on MESH, provides a reasonable method to mitigate catastrophic forgetting in changing environments.

Furthermore, in the experimental evaluation conducted on sequential Morris Water Maze tasks, the proposed method demonstrates superior performance compared to previous approaches, as reported in the paper.

### Weaknesses
1) Enhancing the paper's readability, especially for readers less familiar with neuroscience, would improve its significance. This could involve providing clearer (maybe more intuitive) explanations of certain concepts, such as the entorhinal cortex, the neocortical-entorhinal-hippocampal circuit, the memory scaffold, Hebbian learning, and grid cell patterns.

2) The description of the Morris Water Maze environment lacks clarity. What are the observations and sensory-cells in Figure 2. Is the observation meant to represent the agent's view as high dimensional vector? What are place and grid cells, and what are their dimensionalities? Does the memory store grid cells?  What are the differences between various environments, apart from variations in the goal positions?

3) The description of the proposed method requires further clarity. An explanation of the design architecture for the policy is needed. Additionally, specify which parts of the MESH incorporate attention mechanisms. How is the entire network trained? Is it trained using Reinforcement Learning? Define the objective function for training. Does the statement "The policy requires no further training" imply that the displacement network also doesn't require training?

4) The experimental evaluation lacks comprehensiveness. It would be valuable to compare the proposed method with more state-of-the-art continual learning approaches, such as ER [1] and A-GEN [2]. Moreover, in the paper [3], a strategy involving the replay of similar experiences is used for continual learning. It would be insightful to discuss the relationship between the proposed method and the paper [3] on memory replay.

[1] A. Chaudhry, et al “On tiny episodic memories in continual learning”, arxiv 2019.  
[2] A. Chaudhry, et al “Efficient lifelong learning with a-gem”, ICLR 2018.  
[3] A. Abulikemu, et al “Online Model Adaptation with Feedforward Compensation”, CoRL 2023.

### Questions
1) It is beneficial to provide clearer (maybe more intuitive) explanations of some concepts, such as neocortical-entorhinal-hippocampal circuit, memory scaffold, and grid cell patterns.

2) What are the observations and sensory cells in Figure 2? Does the memory store grid cells?  What are the differences between various environments, apart from variations in the goal positions?

3) How is the entire network trained? Is it trained through Reinforcement Learning?  What is the training objective?  Additionally, specify which part of the network utilizes attention mechanisms.  

4) Consider highlighting the unique aspects of the proposed method compared to replay-based approaches, such as ER [1], A-GEM [2], and Feedforward [3].

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a novel continual learning benchmark based on the Morris Water Maze test of spatial learning in animals, as well as a dedicated neuroscience-inspired continual learning method combining Memory Scaffold with Heteroassociation framework, a randomly-initialised CNN, and an attention module. Through experiments on the sequential Morris Water Maze benchmark, the authors show their method outperforms standard continual learning baselines by effectively retaining past knowledge and quickly adapting to new environments.

### Strengths
The new benchmark is a valuable contribution to the continual learning community. The method has a strong neuroscientific grounding and it brings together existing components in an original way. The paper is well presented and nicely structured. The writing is clear and the figures are very helpful in conveying the main points of the argument. The ablation study provides sufficient justification for the individual design choices.

### Weaknesses
The empirical evaluation is the main weakness of the paper. The authors compare their method to only two continual learning baselines, both of which are quite old. In addition, the replay buffer sizes that are used in the experiments are rather small (200-1800) The proposed method seems to be custom-designed for the navigation task, so while it can serve as a model of how rodents learn to navigate in new environments, it is not a practical continual learning method that could be applied to an arbitrary task. To give existing baselines a fighting chance, I would recommend having two separate networks: one mapping observations and goal position into some latent representation and another mapping these to actions. In the first environment, train both networks. For each new environment, re-train only the first network.

### Questions
Why does replay buffer exhibit such poor performance on the last task?

Will the benchmark be made available? Is it a framework to produce random environments or just a fixed dataset?

Are the associations between displacement representations and actions simply memorised by the attention module?

Is the grid code of the goal location available to your method straight away?

How is the goal location provided to the network for replay and naive methods?

What exactly is stored in the rehearsal buffer? Observation-action pairs?

Have you tried increasing the grid size of the environment?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a task-specific maze path-finding network that is suitable for continual learning. The key contribution is to decouple the policy, and the localization module, and the memorized goal location. The experiments show that the proposed network can learn 5 environments with no forgetting, whereas the baseline policy network completely forgets the previous task.

### Strengths
- The idea of an end-to-end network that is capable of continual learning is interesting, even if it can only handle path finding tasks.
- The results suggest that the proposed network clearly solves continual learning.

### Weaknesses
- It makes sense that the policy network is invariant across tasks, since given true localization and goal location it only needs to learn a good search algorithm. But the place cell and grid cell may still suffer from catastrophic forgetting. Does the model get another set of newly initialized place cells and grid cells when switching to a different environment? Otherwise, how does it prevent forgetting? I would appreciate further clarification on this part.
- I understand that the proposed method is tailored to a path finding task, however, to make it more generalizable, it would be better to test on other types of maze tasks (perhaps with more complex visual features and map topology). Moreover, I don’t see why the task needs to be a water maze (with no walls) vs. a real maze.
- I would appreciate more clarity on model training and loss functions. An algorithm block can also strengthen the presentation clarity of the paper.

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
