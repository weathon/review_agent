# Instant Policy: In-Context Imitation Learning via Graph Diffusion

- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
Following the impressive capabilities of in-context learning with large transformers, In-Context Imitation Learning (ICIL) is a promising opportunity for robotics. We introduce Instant Policy, which learns new tasks instantly from just one or two demonstrations, achieving ICIL through two key components. First, we introduce inductive biases through a graph representation and model ICIL as a graph generation problem using a learned diffusion process, enabling structured reasoning over demonstrations, observations, and actions. Second, we show that such a model can be trained using pseudo-demonstrations – arbitrary trajectories generated in simulation – as a virtually infinite pool of training data. Our experiments, in both simulation and reality, show that Instant Policy enables rapid learning of various everyday robot tasks. We also show how it can serve as a foundation for cross-embodiment and zero-shot transfer to language-defined tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a novel method for rapid robot policy learning. The proposed "Instant Policy" framework allows robots to instantly learn tasks from just one or two demonstrations by modeling in-context imitation learning (ICIL) as a graph-based diffusion process. The approach uses graph representations for demonstrations, observations, and actions, combined with a learned diffusion process, enabling efficient and scalable task learning. It can be trained using pseudo-demonstrations, offering a virtually infinite source of training data, and achieves high success rates on various tasks in both simulated and real-world environments.

### Strengths
1. The paper presents a substantial amount of experiments in both simulator and real-world, showcasing the effectiveness of the proposed method.
2. The proposed method demonstrates good performance in adapting to unseen tasks.
3. The idea of using pseudo-demonstrations to ensure that the learned network can be used for a wide range of real-world tasks is interesting.

### Weaknesses
1. I think there is a notation mistake in Line 163 $\{\mathcal{F}_g^i,p_g^i\}^6_{i=1}$.
2. Since the method consists of three kinds of graphs, the ablation study is missing.
3. The method section is hard to follow, it will be better with a clearer illustration figure.

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
4

### Summary
In the presented work, the authors introduce Instant Policy, a novel approach for in-context imitation learning (ICIL). The key contribution of this work is that demonstrations are represented as a graph, including the observations and actions trained from pseudo-demonstrations generated procedurally in simulation. Furthermore, given a single demonstration of the target task, the proposed method is able to generate a suitable control policy by re-framing the ICIL problem as a diffusion-based graph generation problem. The presented work is interesting for the perspective of not requiring data for the target manipulation tasks but is instead able to use an approach similar to learning from play.

### Strengths
- The graph-based diffusion process is interesting, particularly how the robot's motion is generated through the iterative diffusion process.
- The graph diffusion process in section 3.3 is well described, and generally, the paper is relatively easy to read.
- The results of the presented method are convincing, consistently outperforming the chosen baselines.
- The modality change in section 4.3.1 is very interesting and should be expanded on (the cross-embodiment transfer could be placed in the appendix instead)

### Weaknesses
- In section 3.2, it is unclear how actions are encoded. The graph seems to be encoding relative spatial relationships between key-points on task-relevant objects and the gripper, however, the "Action Representation" section is intuitively unclear. How are the position-based embeddings learned with respect to the actions? What exactly are the actions?
- Given that demonstrations are encoded as a sequence of relative spatial relationships, why are action nodes needed? Wouldn't it be possible to just "condition" the motion on the current observation with respect to the demonstrated graphs and optimize some form of minimal deformation of the graph? (similar to Vogt, David, et al. "A system for learning continuous human-robot interactions from human-human demonstrations." _2017 IEEE International Conference on Robotics and Automation (ICRA)_. IEEE, 2017.) (this might be a candidate for some discussion in the related work)
- The authors claim that, unlike BC, their method does not require much data; however, generating pseudo-demonstrations in simulation is also an expensive process. How does generating pseudo-demonstrations compare to alternative approaches, e.g., this one (Lynch, Corey, et al. "Learning latent plans from play." _Conference on robot learning_. PMLR, 2020.)? (this might be a candidate for some discussion in the related work)
- The generalization to novel objects within the same semantic group seems to stem from the model's ability to generalize due to the selection of only a few key points in the point cloud, making the "novelty" of the semantically similar objects questionable. How does the model handle novel objects from semantic groups that were never part of the training data, which would be a common occurrence in household tasks?

### Questions
- How are sub-goals formulated in the diffusion process? Is it just some horizon the diffusion model is able to generate "in one go"?
- How many pseudo-demonstrations are used to train the initial model across how many objects?
- Transferring the policy from simulation to the real robot seemed to have required only 25 examples (5 rollouts across 5 tasks). How was performance assessed, and would more samples result in significantly better results?

### Soundness
2

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
3

### Summary
The paper presents Instant Policy, a method aimed at improving In-Context Imitation Learning (ICIL) by enabling task learning from just one or two demonstrations without additional training. It introduces a graph-based representation to effectively combine demonstrations and observations, framing ICIL as a diffusion-based graph generation problem. The authors also explore training with procedurally generated pseudo-demonstrations, which allows for scalable data generation. The results suggest that this approach can perform well across various tasks in both simulation and real-world contexts, showing potential for scalability and generalization.

### Strengths
1. The main contribution of this work is the Instant Policy framework for In-Context Imitation Learning, which enables immediate robotic skill acquisition from just one or two demonstrations using a novel graph diffusion process. 
2. The motivation is well-founded. Large Language Models (LLMs) demonstrate impressive capabilities in in-context learning. The paper analyzes the underlying reasons behind the emergence of these capabilities. It also addresses the challenges of acquiring such abilities in the field of robotics and proposes solutions for each challenge identified.
3. The experimental results and demonstrations indicate that the model possesses certain in-context learning capabilities. It also showcases interesting abilities in downstream applications, including the capacity to learn from demonstrations even after cross-embodiment transfer and modality change during the reasoning phase.

### Weaknesses
1. The writing in the paper could be improved, particularly in the model training section, which is somewhat unclear. Figure 3's left image does not effectively illustrate this process.
2. The collection and use of pseudo-demonstrations are not well explained, especially regarding how the model utilizes them and the differences in their usage compared to the RLBench dataset or real-world datasets. For details, see the question section.

### Questions
1. About baselines: Do the baseline models also learn from pseudo-demonstrations during the training phase? In the inference phase, is the accessed context modality video, images, or a data structure similar to INSTANT POLICY (e.g., point clouds, points)?
2. About training dataset: What types of data were used to train the models tested in simulation versus those tested in the real world? Did both use pseudo-demonstrations generated from ShapeNet?
3. About related works: What is the main difference between your work and One-Shot / Few-Shot Imitation Learning approaches, such as [1]?
4. About inference: I noticed that the model predicts T=8 future actions, and I'm curious whether the model completes the task based on the provided demonstration by making just one prediction.

[1] Zhang, Xinyu, and Abdeslam Boularias. "One-Shot Imitation Learning with Invariance Matching for Robotic Manipulation." arXiv preprint arXiv:2405.13178 (2024).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Instant Policy method for in-context imitation learning. Instant Policy represents the state as a graph computed from a point cloud and object segmentation. Edges in this graph connect gripper nodes from the current observation to gripper nodes in the in-context demonstrations. Instant Policy generates actions through a graph diffusion process that produces a gripper state in the graph. The model is trained with "pseudo-task" data of a gripper interacting with simulated objects in arbitrary ways. Experiments show that Instant Policy can perform tabletop manipulation tasks from two demonstrations better than ICIL baselines. Instant Policy can also work with cross-embodiment demonstrations and language instructions.

### Strengths
1. Instant Policy empirically outperforms ICIL baselines in 12 RLBench tasks by a large margin. This indicates that the graph representation and graph diffusion process from Instant Policy provides meaningful improvements over prior approaches. 

1. The pseudo-demonstrations are an effective way to scale data collection for robot manipulation tasks in ICIL. This data source is scalable because it simulates interactions with arbitrary objects without the requirement that they are dynamically or kinematically feasible. The value of this data is confirmed by all methods achieving some non-zero ICIL performance when trained with this pseudo-demonstration source.

1. The architecture of Instant Policy is ablated in Table 2, confirming that the action and diffusion mode are important for performance. 

1. The paper shows scaling results for Instant Policy in Figure 6. 

1. Instant Policy outperforms baselines on real-world robot experiments. These results show that the Instant Policy assumptions of point cloud and object segmentation hold for real-world scenarios where these sensors will be noisy. 

1. The paper shows several interesting extensions of Instant Policy. Table 4 shows Instant Policy generalizes to new objects in the real-world. The graph state representation of Instant Policy allows it to work with human demonstrations for the context as well. Finally, language instructions can also replace the provided demonstration context to work with a language-specified tasks.

1. Supplementary G provides useful transparency about architecture and training decisions that did not work, which will be useful for researchers in the field.

1. Qualitative results are provided in the supplementary section's analysis of the failure modes, the supplementary website, and in Figure 5. 

1. The paper and code provide sufficient details for reproducing the results.

### Weaknesses
1. It is unclear how to extend the Instant Policy to partially observable settings. Instant Policy relies on all the objects in the scene being represented via nodes in the local graph. Even with addressing partial observability by appending previous local graphs, Instant Policy still would struggle with object clutter or occluded objects in containers. The graph state representation that enables Instant Policy to perform well in ICIL in simple tabletop tasks could be a hindrance when scaling to more complex task settings.

1. The pseudo-demonstration generation lacks collision avoidance behaviors. As the trajectories are not necessarily feasible and are just with a single object, they demonstrate a simple interaction motion. Such data does not show how to avoid other obstacles, such as reaching around another object or into a tight space. This limitation is also confirmed in the RLBench failure cases in L891, which mention that the robot gripper typically moves in a straight line toward objects. Based on this and the previously mentioned weakness, the Instant Policy method seems limited to tabletop manipulation where the scene is fully observable and the gripper only needs to move in a straight line to interact with objects. 

1. Instant Policy has two major components: the graph structure and the graph diffusion process. The baselines do not use this same diffusion process or graph structure. The paper does not investigate which of these components of graph diffusion or local graph structure is more important. A simple next-action prediction architecture operating over the graph structure input, like the GPT2 baseline, may be sufficient for good performance like Instant Policy. 

1. The method assumes object segmentation to form the local graph. However, this is not a major weakness since the paper demonstrates this works in the real world with noisy observations, so this assumption is likely possible in many realistic settings.

1. The paper uses a fixed number of demonstration steps for all demonstrations being trained on (L819). While Table 2 shows the method is robust to the demonstration length, to scale to more diverse demonstration scenarios, the method must also handle demonstrations of varying lengths where demonstrations with more complex behavior require more steps. 

Minor:
1. L88 mentions that Instant Policy achieves "true generalization to unseen tasks". However, the paper never defines what is meant by true generalization and why prior approaches lack this capability.

1. The definition of the term "pseudo-demonstrations" was delayed until late in the paper in Section 3.4. I suggest adding some high-level explanation of what's meant by "pseudo-demonstrations" earlier in the paper since this term is often referenced earlier in the paper.

### Questions
1. Perhaps I missed it, but what does the "*" in Table 1 next to the baseline numbers indicate?

1. Why are orientations of objects in the environment fixed to be within some range (L872)? Why not increase the range of rotations in the pseudo-demonstrations or have the policy generalize to these new rotations?

1. Do the objects in RLBench tasks overlap with the objects used in the pseudo-demonstrations? This is important for assessing the generalization abilities of Instant Policy trained only on the pseudo-demonstrations.

1. What exactly does "plateaus at varying levels" on L430 mean? Does this mean the performance does not exceed the numbers in Table 1, even for larger policy sizes?

### Soundness
4

### Presentation
4

### Contribution
3
