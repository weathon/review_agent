# Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination

- Decision: Accept (Poster)
- Scores: 8, 8, 5

## Abstract
A world model provides an agent with a representation of its environment, enabling it to predict the causal consequences of its actions. Current world models typically cannot directly and explicitly imitate the actual environment in front of a robot, often resulting in unrealistic behaviors and hallucinations that make them unsuitable for real-world robotics applications. 
To overcome those challenges, we propose to rethink robot world models as learnable digital twins. We introduce DreMa, a new approach for constructing digital twins automatically using learned explicit representations of the real world and its dynamics, bridging the gap between traditional digital twins and world models.
DreMa replicates the observed world and its structure by integrating Gaussian Splatting and physics simulators, allowing robots to imagine novel configurations of objects and to predict the future consequences of robot actions thanks to its compositionality.
We leverage this capability to generate new data for imitation learning by applying equivariant transformations to a small set of demonstrations. Our evaluations across various settings demonstrate significant improvements in accuracy and robustness by incrementing actions and object distributions, reducing the data needed to learn a policy and improving the generalization of the agents. 
As a highlight, we show that a real Franka Emika Panda robot, powered by DreMa’s imagination, can successfully learn novel physical tasks from just a single example per task variation (one-shot policy learning).
Our project page can be found in: https://dreamtomanipulate.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a novel paradigm for constructing world models that serve as explicit representations of real-world environments and their dynamics. By integrating advances in real-time photorealism, such as Gaussian Splatting, with physics simulators, the authors propose a system capable of generating new data for imitation learning. Additionally, the paper demonstrates the application of this model in real-world scenarios, showing how data collected from the world model can be used to train robots via imitation learning, with promising results when transferring learned behaviors to real-world tasks.

**Strengths:**

1. The paper introduces an innovative approach by leveraging world models to generate robotic data for imitation learning, which is a contribution to the field.
2. The experiments are detailed, covering both simulation environments and real-world robot demonstrations, providing a robust evaluation of the approach.
3. A creative method for augmenting data used in imitation learning is introduced, which could lead to improved learning efficiency.

**Weaknesses:**

1. The absence of publicly available source code limits the reproducibility of the results. It is suggested to release the code during the rebuttal stage.
2. Some figures in the paper need improvement, as the text in several instances is too small to read clearly.
3. The predictions demonstrated in the paper are limited to simple tasks and physics environments, and future work should focus on extending these predictions to more challenging tasks and complex physical simulations.

In conclusion, the paper presents a compelling framework that blends world modeling with imitation learning, but there are areas for improvement, particularly in terms of figure clarity, task complexity, and providing source code for reproducibility.

### Strengths
1. The paper introduces an innovative approach by leveraging world models to generate robotic data for imitation learning, which is a contribution to the field.
2. The experiments are detailed, covering both simulation environments and real-world robot demonstrations, providing a robust evaluation of the approach.
3. A creative method for augmenting data used in imitation learning is introduced, which could lead to improved learning efficiency.

### Weaknesses
1. The absence of publicly available source code limits the reproducibility of the results. It is suggested to release the code during the rebuttal stage.
2. Some figures in the paper need improvement, as the text in several instances is too small to read clearly.
3. The predictions demonstrated in the paper are limited to simple tasks and physics environments, and future work should focus on extending these predictions to more challenging tasks and complex physical simulations.

### Questions
Could you please show some performance results in more complex physical environments and challenging tasks? Even if they were unsuccessful, it would be helpful to see such results, even though they are not included in the paper.

### Soundness
3

### Presentation
2

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
The paper augments a small set of demonstrations with imagination to improve few-shot imitation learning in both RL bench and real-world robotic tasks. The imagination comes from learning compositional objects models through Gaussian Spatting and replaying demonstrations with varied objects poses in a simulator.

### Strengths
The paper presents a novel strategy of data augmentation by first acquiring object models and then leveraging simulations to ensure correct dynamics of imagined demonstrations instead of learning worlds models of both objects and dynamics simultaneously. The strategy is shown to have meaningful improvement on few-shot imitation performance in sufficient sim and real tasks. The approach is also thoroughly invested with ablations showing the significance of the imaging with roto-translation of original demos. The paper is well written with clear motivations and goals and sufficient results to support the claim.

### Weaknesses
1. The paper should compare to other data augmentation approaches such as MimicGen or Digital Twin, which is significant extensive line of work worthy of more elaboration and discussion. Similar approach such as [1] should be discussed or cited. 
2. The method relies on open-vocabulary tracking models to segment objects, which limit the approach to non-articulated objects. It is unclear how such segmentation models can capture individual robot link or object parts connected with articulated joints accurately. Also, it is unclear how to extend the simulator to incorporate articulated objects interaction after the parts have been learned. 
3. Existing approach of verify imagined demonstration is rudimentary. 

[1] MIRA: Mental Imagery for Robotic Affordances

### Questions
1. What’s the relationship with digital twin line of work?
2. How to learn 3d models for parts of articulated objects and how to imagine demonstrations that manipulate articulated objects?
3. Can you replay imagined trajectories in simulator to verify correctness? Will that help improve imitation success?
4. How does error build up in the pipeline of segmenting object masks -> learning objects models through Gaussian Spatting -> imagine demonstrations -> learning policy? Perhaps some ablations or quantitative metrics to measure error will be useful! 
5. Can you add one more baseline method of data augmentation to compare to?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes DreMa, which integrates object-centric Gaussian splatting with a rigid-body physics simulator, to “imagine” new demonstrations to train imitation learning models. These imagined demonstrations are obtained in simulation by applying robot transformations and rotations to objects extracted from Gaussian splatting.

### Strengths
- The paper tackles the data efficient regime, and their proposed approach uses a single demonstration

- The paper validates the proposed approach with real-world experiments, and demonstrates a system that can perform manipulation tasks

### Weaknesses
My main issue with the paper is that the work is positioned within the “world model” literature, while the proposed method seems to fall under real2sim and data augmentation strategies. The “world model” is used to generate an augmented set of demonstrations in simulation in order to train an imitation learning model offline, and it is not used during online control. The set of equivariant transformations used to generate the augmented demo set is hand-designed, and likely task-specific. 

The simulated results would be more convincing if they were expanded to include more than 3 RLBench tasks, but the method is limited to non-articulated objects. Additionally, the real-world tasks seem to only include blocks or boxes. How does DreMa perform with more complex shapes? How does DreMa perform when the scene includes a mix of objects with different shapes? 

- It is unclear what base imitation learning model is used by the proposed DreMa method. Is this just PerAct? Are observations to the policy directly captured from cameras, or by rendering Gaussian splats?

- There are no metrics on runtime performance of the proposed method, while the introduction mentions the “real-time” performance of Gaussian splatting. 

- Table 1: Comparisons to PerAct (Shridhar et al., 2023) are done using only 5 episodes per task, while PerAct uses 25 episodes per task. 

- The related work section would benefit from a broader discussion of particle-based simulation approaches, such as: https://arxiv.org/abs/1810.01566, https://arxiv.org/abs/2312.05359, https://arxiv.org/abs/2405.01044, https://arxiv.org/abs/2303.05512. A comparison to ManiGaussian (https://arxiv.org/abs/2403.08321) would also be helpful.

### Questions
Figure 2: It would be helpful to include how “open-vocabulary tracking models” fit into the pipeline. 

L107: “Two recent works demonstrated predicting future states can be applied to robotics” - This statement can be made more precise, since predicting future states / modeling forward dynamics for control is not a new idea in robotics.

L128: Why is (Cheng et al., 2023) used as a reference for the zero-shot capabilities of foundation models?

L157: There is some overloading of the term “manipulation”, which in the context of the paper seems to refer to a “manipulable” or controllable world model, rather than a world model explicitly designed for robot manipulation tasks. 

L174: Set $\mathcal{X}$ notation is used for a sequence.

L267-268: Other objects in the world that the robot arm may interact with could also be articulated? ie. cabinets 

L409: Is the PerAct baseline in the multi-task setting only trained on the subset of RLBench tasks you’re evaluating on? How many demonstrations were used? Was PerAct trained with data augmentation?

L412-413: Was this model selection using a validation set done over the entire course of training? How does this compare to just using the final model after training for 600k steps?

L418: Why “112, 61, and 81” demonstrations for the three tasks? How were these number of “imagined” demonstrations chosen? 

L485: How were OOD test conditions chosen? Would they be within the set of equivariant transformations evaluated in Table 2? 

[minor editing]

Figure 2: part 3, typo in Gaussian

L105: “gaussian” -> “Gaussian”

L233: $x_{n,k}$ should be $y$?

### Soundness
2

### Presentation
2

### Contribution
2
