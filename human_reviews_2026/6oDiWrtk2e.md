# Generative Simulation for Dexterous Hands

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Data scarcity remains a fundamental bottleneck for embodied intelligence. 
Existing approaches use large language models (LLMs) to automate gripper‑based simulation generation, but they transfer poorly to dexterous manipulation, which demands more specialized environment design. Meanwhile, dexterous manipulation tasks are inherently more difficult due to their higher degrees of freedom. Massively generating feasible and trainable dexterous hand tasks remains an open challenge. To this end, we present **GenDexHand**, a *generative simulation pipeline* that autonomously produces diverse robotic tasks and environments for dexterous manipulation. **GenDexHand** introduces a closed‑loop refinement process that adjusts object placements and scales based on vision‑language model (VLM) feedback, substantially improving the average quality of generated environments. Each task is further decomposed into sub‑tasks to enable sequential reinforcement learning, reducing training time and increasing success rates.
Our work provides a viable path toward scalable training of diverse dexterous hand behaviors in embodied intelligence by offering a simulation-based solution to synthetic data generation. Our anonymous website: https://sites.google.com/view/gendexhand.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a generative simulation pipeline to produce robotic tasks and environments for dexterous manipulation. It consists of three stages: (i) task proposal and environment generation (using Claude Sonnet 4.0 and assets sampled from DexYCB, RoboTwin/Robotwin, PartNet-Mobility, (ii) multimodal large language model refinement (using Gemini Pro 2.5 to explicitly adjust object size, placement, andorientation), and (iii) policy generation (using motion planning + RL).

### Strengths
- The idea is interesting and new.
- The generative pipeline design seems solid.
- paper is well written and well structured. the examples given in the paper are easy to follow and the supplementary materials provide quite a lot details.

### Weaknesses
- My biggest concern is the efficiency and effectiveness of the generative method in terms of sim2real. The authors did not provide any sim2real experiments to evaluate the quality of their generated data.
- The experiments only use a small set of tasks. 
- Figure 2 caption description is not consistent with the previous statement. It says the process consists of four stages, in which it counts environment proposal and generation as two separate stages, whereas in the introduction section, it claims the pipeline consists of three stages and counts environment proposal and generation as one stage.

### Questions
- Does the RL policy need to be retrained for every task generated? or the RL policy can be trained on one/several subtasks and generalize across tasks?
- How many hours does it take to generate X samples on Y device?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a generative simulation pipeline for dexterous hand manipulation. It has several stages—task proposal and environment generation, MLLM refinement, and policy generation, with designs including closed-loop MLLM-driven scene adjustment, subtask decomposition, and hybrid motion planning/reinforcement learning (RL) for policy training. Experiments show it can generate physically plausible tasks and achieves a 53.4% average improvement in task success rate compared to baselines.

### Strengths
1. The entire pipeline appears to be feasible.
2. The paper is well-structured and clearly written.

### Weaknesses
1. The work represents only an incremental improvement over existing gripper-based data generation approaches [1,2] and lacks novelty.

2. Most experiments in the paper do not actually require a dexterous hand — the tasks can largely be accomplished with a simple gripper, revealing a lack of truly dexterous manipulation tasks. Only a in-hand manipulation task truly needs dexterous hand, however this task don't need this pipeline actually, because a lot of previous works have done it well. 

3. The provided video demonstrations are short and feature relatively simple tasks, lacking examples of complex or long-horizon manipulation.

4. The work lacks the results of imitation learning experiments trained on its data. The collected data needs to be validated for its effectiveness in training autonomous policies — otherwise, the dataset itself holds limited practical significance.

5. Due to the absence of real-world robot experiments, the authors fail to demonstrate the practical usefulness of the proposed pipeline for real dexterous robotic manipulation.

[1] GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs

[2] RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation

### Questions
See Weakness.

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
The authors present a GenDexHand, a simulation framework to generate tasks and data, tailored specifically for dextrous manipulation tasks. The use LLMs to generate tasks, MLLMs to refine them, and they benchmark various methods to generate data for these tasks. While the ideas are sound and experiments are promising, there are several missing details in the paper about the scope of the benchmark, and experiments are sparse. Please see comments below for additional details.

### Strengths
- the prompts to generate tasks and data are elaborate and well-thought out. they are clearly laid out in the appendix, making it transparent how the system works.
- using MMLMs to refine tasks is a practical idea, and the authors show specific examples how this is applied in practice to obtain more realistic tasks.

### Weaknesses
- The experiment to quantify task diversity via cosine similarities of the text embeddings is just one specific metric, but is not a holistic way to measure task diversity. For example, how many assets are incorporated? How many skill families are present, compared to other works? What is the distribution of the number of stages per task? How many environments are present? This information is missing in the current manuscript.
- The experiments only feature three tasks (figure 4), and the tasks are not very diverse (two are basic pick-place tasks). I presume there are more tasks that this framework can generate, but the main text does not mention the full scope of tasks.
- The paper presents a simulation framework with the goal of generating diverse data, presumably to build real-world robot agents, but there are no experiments or discussion about how to use the generated data for transfer to real world environments and tasks.

### Questions
- How many tasks in total are generated by this simulation framework? What are the skill families present?
- It's unclear why without subtask decomposition, the episode length is 400 steps, but with subtask decomposition the episode length is 200 steps. Is this a fair comparison?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents GenDexHand, which uses VLMs to automate environment and data generation in simulation for dexterous hand manipulation tasks. The main contributions are:  
1. It focuses on dexterous hand manipulation.  
2. It uses VLMs not only to create tasks but also to check and refine them.  
3. It studies policy learning for the proposed tasks, including task decomposition and motion planning integration.
Experiments are done on several simulation tasks, and task diversity is reported.

### Strengths
The writing is clear and easy to follow.  
It is good to explore using VLMs for dexterous task design and simulation setup.  
It is also good to study reinforcement learning, frozen joints, and motion planning.

### Weaknesses
- The main weakness is that there are no real-world experiments. This means the paper cannot show if the simulation is actually useful in real applications, which makes it less convincing for manipulation research.  
- Another weakness is that the idea of using VLMs for simulation setup is no longer new (unlike GenSim or RoboGen). Many researchers now question the value of VLM-generated simulations. Reviewers may expect a solid and practical simulation benchmark such as THOR or RoboCasa. The authors are encouraged to build a far more diverse and convincing VLM-based simulation benchmark that includes extensive real-world validation. For the tasks in this paper, setting them up manually would be easier and more controllable than using prompts.  
- From the policy learning point of view, compared to recent studies, it is hard to say that subtask decomposition, motion planning, or freezing degrees of freedom are contributions. If the authors want to highlight policy learning, they should provide more insights, compare with state-of-the-art methods and include real-world demonstrations. 
- The results in Table 1 are also not convincing, and more experiments are recommended.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
