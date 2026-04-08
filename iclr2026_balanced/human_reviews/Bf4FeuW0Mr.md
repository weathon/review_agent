## Human Reviewer 1

### Summary
This paper presents DemoGrasp, a framework that first augments demonstrations with pose and object randomization and a reinforcement learning (RL) residual policy, and then distills the results into an imitation learning policy. Extensive experiments are conducted in both simulation and the real world. The ablation studies on RL-sampling comparison, camera configuration, and RL action spaces are well designed.

### Strengths
The experiments are extensive, and the real-world demonstrations looks robust.  
The analysis of RL-sampling comparison, camera configuration, and action spaces is insightful.  
RL residual learning and RGB sim-to-real transfer are challenging, and the paper addresses them successfully.  
The presentation of the paper is clear and well structured.

### Weaknesses
- One weakness is that more details about the experimental settings should be provided, like RGB randomization details in the paper.
- It is suggested to replace some edited videos on the website with uncut, continuous grasping videos.

### Questions
- It would be helpful to discuss more recent related works in the paper.  
- The quality of figures and videos can be improved, preferably with higher resolution.
- It would be good to include an open-source commitment in the paper to increase its reproducibility.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper shows framework for dexterous grasping policies using RL with single demonstration augmented. The policy could grasp a huge amount of objects and its appreciate the author conduct steps for sim-to-real, instead of simply reply the trajectory in simulation. The trajectory dataset and the sim-to-real for these trajectories could be a good contribution.

### Strengths
Appreciate the sparse reward and simple collision penalty. Easy to read and follow the idea. The presentation is good. The experiments are comprehensive.

### Weaknesses
Minor weakness: 

1. the abstract contain many contents but not organize well. It get reader confuse that the author using RL in simulation but directly switch to imitation learning in the real-world without any context.

2. line 090, please specify what is the "prior methods" mentioned, eg cite the corresponding paper.

3. line 105-106, please specify what is "previous state-of-the-art-methods" here.

4. Front size in figure 1 is too small to recognize.


Major weakness:

1. when perform sim-to-real transfer, the author select the successfully RL rollout, then apply an additional training stage and apply domain randomization in imitation stage. why not directly train a vision based policy and directly transfer to the real. 

2. For two stage training, why select stage based RL and do a selection and then do BC? Have you ever research train a state based RL first and then perform a teacher-student pipeline to distill a vision policy, which can also be able to transfer to real.

3. For the sim to real, the author mention the domain randomization for the RGB and depth. Have you apply any method on robot control side. eg. simulation hand is usually more compliant than real hand, especially for the inspire hand, which may like to broken if the collision is too large. Also for the arm, have you notice any control gap between sim and real.

4. Would be more interested the pipeline and application to extend to beyond grasping task. 

5. How much randomization for the object initial pose? For a simple demo augumentation, does the larger randomization area decrease the max success rate of RL? 

6. While appreciate conduct the BC for sim-to-real, more experiments to justify the BC training would make experiments more comprehensive. Eg. better generalization, or better design choice, or help on implementation.

### Questions
See weakness above

### Soundness
4

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents a method to learn dextrous grasping from just a single demonstration. The key differentiator with existing approaches is to formulate the problem as a 1 step markov decision process where the policy predicts a residual trajectory acting as a demonstration editor. The entire trajectory is treated as a single action executed using a lower level motion planner. This makes exploration and reward design much more simpler than imitating the entire trajectory. Once a state-based policy is trained in simulation, a vision-action dataset is collected to then learn a vision-based policy using behavior cloning.
The results show generalization to novel objects as well as zero shot transfer to real world robots.

### Strengths
The paper is well structured and easy to follow. It provides sufficient background, comparison to existing literature and clearly establishes the motivation for the paper. 

The paper provides sufficient details about training, provides comprehensive quantitative comparison against relevant baselines across different object datasets and robot embodiments. The results and the supplementary videos show effective grasp success rates. 
The paper also presents in depth ablation studies that sheds light on the necessity for RL instead of sampling based methods, the action space used in RL, camera configurations as well as demonstration quality. The proposed method performs better than baselines in all scenarios. 

An additional strength of the paper is real-world results which is the ultimate test for any robot learning method.

### Weaknesses
While the results are promising, I think the biggest weakness is the lack of generalizability of the proposed method to more dynamic environments where a reactive policy might be required to avoid collisions or adjust a grasp mid motion.

Adding details about how the language conditioned policy is learned can be extremely beneficial for a reader. Similarly, more details on the lower level motion planner that executes the trajectory would be really helpful.

### Questions
1) The problem formulation is not too clear - In section 3.1, the authors mention that the goal of the policy is to maximize the discounted sum of returns (like standard RL), and the equation suggests that the horizon “T” is the length of the demonstration trajectory. But this is at odds with the single step MDP that the authors claim in the beginning. Even in table 12, the episode length is set to 1. How exactly is the problem formulated and what is the discount factor?

2) What is the motion planner used? How robust is the policy to the motion planner?

3) The vision based policy is closed loop per grasp attempt, correct? What is the frequency of that loop? 

4) It would also be great to know how tactile feedback can be incorporated into this framework in future work section.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper presents a method for training dexterous grasping policies from a single demonstration in simulation. The key idea is to frame the problem as demonstration editing, where a reinforcement learning (RL) policy learns to generate residual corrections that enable grasping of novel objects. This formulation helps address the exploration challenge and leads to more efficient learning. Extensive real world evaluations are carried out to validate the method.

### Strengths
The paper is well-written and easy to follow.

The proposed method is conceptually sound and addresses an important problem in dexterous manipulation.

### Weaknesses
1. It is unclear what range of novel objects this approach can effectively handle given only a single demonstration on one object. For example, if the demonstration involves grasping a bottle, it may be difficult to transfer that experience to grasping a flat plate. It would be valuable to include experiments illustrating which types of object transfers are successful, and to discuss how such transferability might be predicted a priori.

2. The computational efficiency of the proposed method appears to remain high compared to sampling-based or template-based approaches. Including quantitative comparisons of computational cost and a discussion of when this approach would be preferable to these alternatives would strengthen the paper.

3. The high-level idea is not that surprising, given recent papers like DexMachina.

### Questions
See Weakness 1. I would like to see more discussion on this.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
5