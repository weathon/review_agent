# RoboMoRe: LLM-based Robot Co-design via Joint Optimization of Morphology and Reward

- Decision: Reject
- Scores: 8, 6, 8, 4, 4

## Abstract
Robot co-design, the joint optimization of morphology and control policy, remains a longstanding challenge in the robotics community. Existing approaches often converge to suboptimal designs because they rely on fixed reward functions, which fail to capture the diverse motion modes suited to different morphologies. We propose RoboMoRe, a large language model (LLM)-driven framework that integrates morphology and reward shaping for co-optimization within the robot design loop. RoboMoRe adopts a dual-stage strategy: in the coarse stage, an LLM-based Diversity Reflection mechanism is proposed to generate diverse and high-quality morphology–reward pairs and  Morphology Screening is performed to reduce unpotential candidates and efficiently explore the design space; in the fine stage, top candidates are iteratively refined through alternating LLM-guided updates to both reward and morphology. This process enables RoboMoRe to discover efficient morphologies and their corresponding motion behaviors through joint optimization.  The result across eight representative tasks demonstrate that without any task-specific prompting or predefined reward and morphology templates, RoboMoRe significantly outperform human-engineered design results and competing methods. Additional experiments demonstrate robustness of RoboMoRe on manipulation and free-form design tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses a fundamental challenge in evolutionary Robotics, i.e., robot design automation, and proposes a novel perspective that evolves reward functions simultaneously in order to account for distinct motion modalities across morphologies. Specifically, the authors introduce a dual-stage strategy: a coarse stage where an LLM is prompted to propose diverse and promising morphology-reward pairs, and a fine stage where the solutions are further optimized. Extensive experiments demonstrate the effectiveness of the proposed approach, RoboMoRe, compared with several baselines.

### Strengths
1.The paper is well-written and easy to follow. The abundant illustrations and qualitative results greatly help with reader’s understanding. 

2.The paper is well motivated. The design of reward function is left simple and static in previous robot co-design studies. The introduction of reward shaping facilitates more customized and precise fitness evaluation across different morphologies, leading to improved optimization efficiency and overall performances. 

3.I greatly appreciate the authors’ endeavour to examine their method in various settings, such as template-based and free-form robot design, as well as different environmental perturbations. The results validate their adaptability and generality across diverse scenarios. 

4.The authors provide a comprehensive discussion of limitations and future directions, which would greatly inspire further studies.

### Weaknesses
1.I believe the motivating example given in Appendix D is worth incorporating into the main text, so that the logic flows more naturally. 

2.The authors could also include a brief outline at the beginning of the appendices so that the contents are more organized.

### Questions
1.Since the reward functions are evolved by an LLM, I wonder whether this would implicitly introduce a bias towards robot morphologies with more interpretable motion patterns instead of those with high potential but less intuitive reward functions to evolve? 

2.Could the authors provide a couple of example reward functions for soft voxel robots? Since the dynamics of SVRs are far less intuitive than articulated rigid robots, I wonder whether LLMs can still evolve interpretable reward functions?

### Soundness
4

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
The paper presents RoboMoRe, a framework that integrates large language models (LLMs) into the robot co-design process, enabling the joint optimization of morphology and control. By leveraging natural language descriptions and structured reasoning, the system aims to generate robot designs that are both functional and specialized for given tasks. The authors highlight how the LLM facilitates creative design exploration while remaining grounded through simulation-based evaluation. The approach is validated through multiple case studies showing task-specific robot designs and control policies optimized jointly via reinforcement learning and design iteration.

### Strengths
1. Insightful finding on co-design: The study convincingly shows that reward shaping must be robot-dependent, illustrating that task-based optimization alone fails to generalize across morphologies. This is an important and underexplored insight in robot design problem.

2. Timely and important topic: The intersection of LLMs and robotic design is a rapidly growing area with significant potential impact. The authors target an important problem - how generative models can extend the space of feasible robot designs.

3. Sound evaluation protocol: The experiments are thoughtfully designed, combining simulated environments with objective metrics that quantify both control performance and morphological diversity.

### Weaknesses
1. While the integration of LLMs is appealing, much of the technical pipeline (e.g., reinforcement learning for control, parametric morphology search) builds directly on established frameworks, so the question is whether the contribution is appealing enough to a general machine learning audience.

### Questions
1. How sensitive is the system to prompt phrasing? Did the authors test multiple prompt templates for reward generation, and if so, how consistent were the resulting performance outcomes?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces the idea of utilising LLMs to judge and augment the fitness value with a generated reward function for the joint co-optimisation of agent policies and embodiment variables.
Overall, the paper presents an interesting approach to reward generation for the problem of co-design. I aprpeciate that it is not quite straightforward to compare against other approaches and find suitable metrics to compare approaches (see comments below). However, I judge this paper as an interesting research into this direction with proividing sufficient details and insight.

### Strengths
- Care was taken to evaluate the use of LLMs fairly, eg by masking elements of the XML files to prevent possible training data contamination.
- Using an LLM-generated reward is a relatively novel idea for co-design, albeit learned rewards have been used before in co-design (see below) and Eureka has been used in a behaviour-only RL setting.
- The paper is good to follow, well written and visualisations are used nicely to support the reader's understanding.

### Weaknesses
- The use of efficiency as fitness/volume is not quite clear to me. Why do you not use torque or fitness/energy instead? It seems to me that the algorithms can easily game this metric by producing as thin geom-elements as possible without increasing actual real world efficiency (asi n, energy spent per forward unit of movement).
- The literature review/discussion of related works is a bit incomplete, as the idea of using learned reward functions has already been explored in previous work (see eg [R1]), albeit to the best of my knowledge not with LLMs.
- In tables 2 and 3, it is not clear to me why the best result is not presented in bold, but only the proposed method. ROboMoRe does not always seem to be the best performing method in the respective metrics.

[R1] Rajani, Chang, et al. "Co-imitation: learning design and behaviour by imitation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 5. 2023.

### Questions
see above

### Soundness
3

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
The paper presents a 2 stage approach for robot codesign – the first is a coarse stage driven by LLMs to build diversity of designs, and the second is a fine stage where top designs are iteratively refined through alternating LLM guided updates to both reward and morphology. The method is shown to significantly outperform competing methods of agent design.

### Strengths
The method seems novel and exposes a way to utilise LLMs for morphology design. The results seem promising as well.

### Weaknesses
Some of the parts of this work need to discussed more clearly. For example, it is not clear what efficiency of a design means formally. In addition, there seems to be a lack of comparisons with existing non-LLM methods such as transform2act [1] etc., Even if this is not an equivalent method, it would still be good to include comparisons for better reference. In addition, the idea of refining the rewards is not very clear to me because it inherently changes what a "performing" agent is.

[1] Yuan, Ye, et al. "Transform2Act: Learning a Transform-and-Control Policy for Efficient Agent Design." International Conference on Learning Representations.

### Questions
1.	With regards to lines 67-68, while different rewards may be more relevant to different designs, modifying the rewards for each design may also inadvertently promote unwanted behaviours for each design (reward hacking). For example, if the ultimate goal is to move forward as fast as possible, while encouraging jumping behaviours could help this objective, it could also lead to unwanted behaviours such as jumping in place if there is a reward introduced for jumping.

2.	Also, if the reward functions of two designs are dependent on the designs themselves, how does one compare the performance of two designs? The performance can no longer be measured in terms of rewards as the reward structure would be different for each agent.  Is there instead some universal measure of performance, independent of the reward function which is considered?

3.	Below eq (1), I am not sure if it is correct to say that it would be a local optimum – because as long as R_0 is the correct reward function for the agent (that is, it accurately captures the quality of behaviours from the agent), $\theta^*$ would indeed be the global optimum. My contention, consistent with 2. above is that as soon as one varies R as in Eq (2), there would be no meaning to ‘high performing’ agents because the performance of two agents would no longer be comparable

4.	If the purpose of stage 1 is to promote diversity, could approaches like diversity if all you need [2] (applied to the design space) be applied?

5.	How is “efficiency” as discussed in the results defined formally?

6.	Why are the resulting designs seemingly all symmetrical? Is symmetry imposed on the designs?

7.	Do the frequency of the stages matter? For example, instead of alternating cycles of course and fine stages, would it possibly be more effective to perform multiple cycles of each stage, before moving to the other stage?

[2] Eysenbach, Benjamin, et al. "Diversity is All You Need: Learning Skills without a Reward Function." International Conference on Learning Representations.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RoboMoRe, an LLM-driven framework for robot co-design that jointly optimizes morphology and control policy. Unlike prior methods that rely on fixed reward functions, RoboMoRe uses an LLM to generate and refine morphology-reward pairs through a two-stage process: coarse exploration and fine iterative refinement. Experiments on eight tasks show that the method discovers more efficient designs.

### Strengths
+ This work applies emerging LLM capabilities to the classic robot co-design problem.
+  The paper identifies a limitation of LLM-based design -- tendency toward repetitive morphology outputs, and introduces the concept of diversity reflection to address it.
+ The method is evaluated on multiple tasks, though the experiments are performed in simple simulation environments.

### Weaknesses
-  The novelty of the work appears limited. The key contributions seem to center on prompt engineering, e.g., prompting the LLM to “reflect” on prior results to increase diversity. The reward shaping and morphology filtering mechanisms also appear straightforward (e.g., discarding repeated designs), and the coarse-to-fine pipeline resembles a standard iterative refinement process, albeit executed via an LLM.

-  The paper does not provide sufficient detail regarding the diversity reflection mechanism. For example, the authors state that “the LLM reflects on previously generated samples and deliberately produces new candidates,” but it is unclear how this reflection is implemented to encourage diversity.

-  Similarly, the description of reward shaping lacks clarity. For example, the paper notes “It is therefore used to evaluate all morphology candidates.” How exactly is this evaluation performed?    

- Line 223 refers to an “LLM-generated reward function,” yet the reward functions for tasks appear manually defined as shown in the paper’s Appendix. This creates confusion. My interpretation is that the manually defined functions compute rewards based on LLM-generated morphologies, rather than the LLM generating the reward function itself. The authors should clarify this distinction.

-  More broadly, the use of the term co-design feels unconvincing. The process appears sequential: morphologies are generated first, and then controllers are optimized. Effective co-design in robotics should explicitly consider the interplay among morphology, motion behavior, and physical constraints during the design process. Robot morphology should be designed with careful consideration of the robot’s motion control and physical constraint requirements. The current framework does not clearly demonstrate such coupling.

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2
