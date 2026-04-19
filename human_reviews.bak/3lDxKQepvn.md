# Latent Task-Specific Graph Network Simulators

- Decision: Reject
- Scores: 8, 6, 3, 6

## Abstract
Simulating object deformations is a critical challenge in many scientific domains, with applications ranging from robotics to materials science. 
Learned Graph Network Simulators (GNSs) are an efficient alternative to traditional mesh-based physics simulators. Their speed and inherent differentiability make them particularly well-suited for inverse design problems such as process optimization.
However, these applications typically offer limited available data, making GNSs difficult to use in real-world scenarios. We frame mesh-based simulation as a meta-learning problem and apply conditional Neural Processes to adapt to new simulation scenarios with little data. In addition, we address the problem of error accumulation common in previous step-based methods by combining this approach with movement primitives, allowing efficient predictions of full trajectories. We validate the effectiveness of our approach, called Movement-primitive Meta-MeshGraphNet (M3GN), through a variety of experiments, outperforming state-of-the-art step-based baseline GNSs and step-based meta-learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper first propose a meta-learning framework to efficiently learn generalizable mesh-based dynamic prediction tasks. Different from previous graph neural simulators which predict the state updates in a step-by-step manner, the proposed M3GN targets to predict the whole trajectories by a conditional neural process to effectively diminish the error accumulation issue.

### Strengths
Strength:
1. Adopting meta-learning to deal with dynamic prediction tasks is novel, especially the concept of regarding each tajectory as a new task is interesting. 

2. The authors consider past information and the eventual state of the collider as the condition to predict the subsequent movement trajectory, which make the network infer the future from the past rather than remember the dynamic behaviour of a certain material. In addition, predicting the whole rest path by a single forward pass could significantly improve the efficiency, compared with previous Graph-based single timestep prediction.

### Weaknesses
Weakness:
1. This paper is highly related to the Graph-based Neural Simulators. However, in the related work section, the latest advancements in this field are not included, and most of the work discussed is from 2023 or earlier. This could make the paper appear somewhat outdated. I believe this section could benefit from a more comprehensive overview of the field, especially more works from 2024. Below are two of the latest advancements about Graph Network Simulators that I recommend the authors to discuss them in Section 2.1 ,or better, use them as baselines for comparison. However, given the tight rebuttal timeline, it is also tolerated that concurrent works were not included for comparison.

    (1) "DEL: Discrete Element Learner for Learning 3D Particle Dynamics with Neural Rendering" 2024   ..  This work integrate traditional Newton mechanics into the graph network design to benefit from mechanics priors for longer term prediction.

    (2) "Equivariant graph neural operator for modeling 3d dynamics" 2024  ..  This paper deal with dynamic prediction tasks as trajectory-level rather than next-step level by operator learning, which is somewhat relavent with this reviewing work. Also, it handle the equivariant issues. 

2. For Equation 3, does it use past trajectory collider states when encoding z because I saw that you seem to only use the latest state, or does it rely solely on the historical information of the deformed object? I believe it would be more reasonable to use all the historical information of the collider here as well, since the deformation of the mesh is passive. 

3. If this method is trained on an elastic dataset, can it generalize directly to elastoplastic materials? I believe it would be worthwhile to discuss the generalization across different materials in the experiments, rather than limiting it to variations in mechanical parameters within the same material. 

4. Line 276 mentions that the context information z is concatenated with the node features. Is the same z concatenated to each node?

5. Finally, the neural network predicts a set of weights, and the shape of the weight matrix is 𝑇, 𝐷,3. Which basis functions are these weights applied to in order to obtain the predicted trajectory? Are they precomputed from the historical trajectory? If yes, how?

In appendix A.2 "Initially, we integrate a relative goal position as part of the node weights w" What's the exact mean of the relative goal position? 

I will raise the score if most of concerns are well addressed by the authors.

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose a graph network simulator that combines movement primitives and trajectory-level meta-learning. The network uses the simulation history as the context information to predict the deformation for objects with unknown properties. They also use probabilistic dynamic movement primitives to represent the future trajecteries and directly predicts the full simulation trajectories instead of iteratively predicting the next-step. Experiments show that it outperforms STOA in different simulation tasks. Abalation studies validate the effectivenss of the design choice.

### Strengths
This work aims to address two important problems in learning-based simulation:

1. It treats the simulation as a trajectory-level meta-learing problem and use trajectory history as the context to predict future trajectories.

2. It mitigates the problem of error accumulation by using ProDMP to directly predict the full simulation trajectories.

The paper is well structured and written.

### Weaknesses
1. Some descriptions are unclear and some important details are missing.
(1) in line 242, "graph edges between the deformable object and collider are added based on physical proximity to model interactions
 between objects." what is the physical proximity exactly? Since the deformation mesh node position for the end timestep is unknown, I suppose we cannot use that to compute the distance. Whether this edge creation is done only for known timesteps or if it's updated during prediction?

(2) in line 231, why is the term c_1y_1(t) + c_2y_2(t) only depending on the inital conditions? What is the representation of the pre-computed basis fuction \phi?

2. More detailed description of the training/val/test split should be added. Specify how trajectories are divided between training, validation, and test sets. What are different between training and test? Clarify if test trajectories involve different objects, material properties, or initial conditions than training trajectories. In the limitation part, it is claimed ''We currently consider each trajectory as a task, and require initial states of this trajectory as a context set during inference."

3. Since the method needs a trajectory with simulated states as context, the author better include a runtime comparison between your method (including context computation) and traditional simulators for predicting the same number of future timesteps and discuss the trade-offs between computation time and accuracy compared to traditional simulators.

### Questions
1. What is the timestep for simulation?

2. A figure illustraing all the relation and symbols of input, output can be added. Fig.3 Right is not information for undertanding the task setting.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces Movement-primitive Meta-MeshGraphNet (M3GN), a model for simulating object deformations in data-limited scenarios. M3GN combines meta-learning and movement primitives to improve the adaptability and accuracy of Graph Network Simulators (GNSs) by framing mesh-based simulation as a meta-learning task.

### Strengths
The paper takes a novel approach to enhancing rollout stability by predicting entire future mesh states, and it incorporates a meta-learning scheme to improve adaptability within the simulation framework.

### Weaknesses
While the approach appears novel, the rationale behind certain modules in the model is unclear, and the results do not provide sufficient evidence to justify their inclusion. Also, the paper is not clearly written and sometimes hard to follow. The detailed comments and suggestions are listed below.

### Questions
1. The model's architecture is not clearly explained, and it is unclear why certain modules are necessary. For example, from the results, it seems that MGN, even without history information, can surpass M3GN in performance. This raises questions about the value of incorporating historical information in M3GN. Moreover, the experimental results do not clearly demonstrate the necessity or advantages of using a meta-learning scheme. A thorough analysis on how meta-learning benefits model performance would be valuable, including ablation studies comparing model performance with and without meta-learning.
2. The authors claim that the baseline MGN does not incorporate historical information, which appears inaccurate. In certain datasets, MGN does include history. For a fair comparison, the MGN baseline should also be evaluated with historical data to assess its impact on performance.
3. The results section only reports the average MSE across all time steps. It would be helpful to provide a comparison of MSE over the number of prediction steps, as this would give insight into the model's performance stability over time as claimed in the paper.
4. Based on Figure 3, the proposed M3GN method does not appear to use ground truth collider information. If this is the case, does the collider state being predicted by the mode? How accurate is the collider state prediction, especially when history steps are limited? Additionally, including collider ground truth (as in MGN) is actually intuitive and makes sense, as the primary goal of developing a simulation model is to understand how a solid deforms under varying contact forces and obstacle displacements. Predicting these external forces may not be necessary for achieving this objective.
5. It would be informative to visualize the node-level latent task descriptions learned by the model. Such visualizations could help in understanding how task-specific information is represented.
6. The datasets used in this paper have relatively small node counts compared to those in previous MGN studies or those used in other related papers. When the number of nodes increases significantly, it is concerned that M3GN may struggle due to the large number of historical steps required. Comparing M3GN’s memory usage with MGN’s would provide a more comprehensive evaluation. 
7. The authors consider each trajectory as a separate task with varying context sizes. However, this approach may not align with the broader goals of meta-learning, as tasks are typically defined by consistent properties such as the same material setting. Currently, the meta-learning setup seems more focused on adapting to different context sizes rather than generalizing across diverse tasks.
8. As the input context size changes, will the number of predicted steps vary as well? If so, the model’s ability to generalize to different context sizes is unclear, and it may not be as flexible as MGN in this respect. Any experiments or evaluation on this aspect? Additionally, splitting single data points into multiple input-output sets seem to increase the effective amount of training data for M3GN, potentially creating an unfair comparison with MGN which use less training data.
9. The authors do not specify how material properties are incorporated. Also, it is unclear whether the test data involve material properties that are in-distribution or out-of-distribution relative to the training data. Providing this information is crucial for evaluating the model's generalization capabilities.
10. The authors mention that material node features are not added to M3GN. Given that these features enhance MGN's performance, it would be useful to understand the rationale for this exclusion and perform related ablation study.
11. Although the authors mention other methods in related work besides MGN, these methods are not included in the baselines. Some of these methods have better accuracy and efficiency. Including these additional baselines would provide a clearer view of M3GN’s comparative performance.
12. Will the data used in this study be publicly available? Making the dataset accessible would facilitate further research and replication studies.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a graph network simulator for mesh-based simulation on material study. The framework is constructed on a meta-learning problem and applies conditional Neural Processes to address data limitations. This paper shows both qualitative and quantitative experiments.

### Strengths
1. This paper shows a clear motivation for initial state uncertainty and data limitation, which are all critical problems in related research fields.

2. Consider the "node-level latent features," which is, to the best of my knowledge, a novel method for solving such a problem.

3. The results of the new simulation task in the paper are convincing for the proposed method.

### Weaknesses
1. Some methodology details are unclear, especially in the "Probabilistic Dynamic Movement Primitives" section and "Meta-Learning and Graph Network Simulators."

### Questions
1. How does ProDMP generate smooth trajectories based on the predefined conditions of the initial state? Please give detailed justification and explanation.

2. Could the author provide a detailed explanation of how a meta-learning problem can contribute to simulating new scenarios?

### Soundness
3

### Presentation
3

### Contribution
3
