# Curriculum Reinforcement Learning via Morphology-Environment Co-Evolution

- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Throughout long history, natural species have learned to survive by evolving their physical structures adaptive to the changes of environments. In contrast, current reinforcement learning (RL) mainly focuses on training an agent with a fixed morphology (e.g., skeletal structure and joint attributes) in a fixed environment, which can hardly be generalized to changing environments or new tasks. 
In this paper, we optimize an RL agent and its morphology through ``morphology-environment co-evolution (MECE)'', in which the morphology keeps being updated to adapt to the changing environment, while the environment is modified progressively to bring new challenges and stimulate the improvement of the morphology. This leads to a curriculum to train generalizable RL, whose morphology and policy are optimized for different environments. Instead of hand-crafting the curriculum, we train two policies to automatically change the morphology and the environment. To this end, (1) we develop two novel and effective rewards for the two policies, which are solely based on the learning dynamics of the RL agent; (2) we design a scheduler to automatically determine when to change the environment and the morphology. In experiments on two classes of tasks, the morphology and RL policies trained via MECE exhibit significantly better generalization performance in unseen test environments than SOTA morphology optimization methods. Our ablation studies on the two MECE policies further show that the co-evolution between the morphology and environment is the key to success.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a curriculum reinforcement learning approach, MECE, which optimizes an RL agent's morphology and environment through co-evolution. The authors train two policies to automatically modify the morphology and change the environment, creating a curriculum for training the control policy. Experimental results demonstrate that MECE significantly improves generalization capability compared to existing methods and achieves faster learning. The authors emphasize the importance of the interplay between morphology and environment in brain-body co-optimization.

### Strengths
1. The paper is well-structured, with a relatively clear introduction.

2. The paper includes comprehensive experiments on rigid robot co-design tasks, demonstrating the superiority of the proposed algorithm. The ablation studies effectively isolate each component's contribution and provide valuable insights into the algorithm's effectiveness.

### Weaknesses
The significance of the paper's contributions is a bit unclear. It is not the first to propose using co-evolution method to co-design brain, body and environment. The proposed methods should be compared with more strong baselines. Curiously, can and how this system extend to the real world?

### Questions
1. How general is the proposed approach, beyond the tasks and environments considered in the experiments?

2. Is the proposed MECE method computationally efficient?

3. Have you encountered any scalability issues when applying MECE to more complex tasks or environments?

4. It is not clear to me how environments are produced and how the agents perform in your environment (Figure 4), do you have a video?

5. It seems that MECE's performance is not much better than Transform2Act, can you provide more results on different tasks?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the problem of joint optimization of the policy and the morphology of a learning agent. The authors’ motivation is described in the claim written in the introduction: “a good morphology should improve the agents’ adaptiveness and versatility, i.e., learning faster and making more progress in different environments.” To realize it, the authors propose the novel framework where the morphology and the training environment are jointly evolved. In the proposed MECE scheme, three policies are introduced: one for the control of an agent’s action, one for the evolution of the morphology, one for the evolution of the training environment. Inside this scheme, the authors define reward functions for the training of the morphology policy and for the training of the environment policy. The authors have performed comparison with several baseline approaches on three control tasks and ablation studies have been conducted to confirm the effectiveness of each algorithmic component.

### Strengths
1. A novel framework for morphology optimization aiming at obtaining a morphology under which a control policy  can be quickly adapted to an unseen environment.

2. Promising empirical results compared to baseline approaches, but the experimental procedure is questionable, see below.

### Weaknesses
1. As far as I understand from what is written in the introduction, the motivation of the morphology optimization in this paper is to obtain a morphology with which the agent can adapt its policy quickly to unseen tasks. It is also written in the second question of the experiments. However, it seems that the reported results in figures are average performances of the agent obtained at each training time step on randomly-selected environment. Therefore, the performance evaluated in this paper is the one for domain randomization. It is different from the motivation. The efficiency of the adaptation of the policy under the obtained morphology is not evaluated. My understanding might be wrong as the evaluation procedure was not clearly stated. Please clarify this point. 

2. It could be better if the design choices of the proposed approach is more elaborated. In particular, it is not clear how the reward functions (1) and (2) reflect the author’s hypotheses “a good morphology should improve the agent’s adaptiveness and versatility, i.e., learning faster and making more progress in different environments” and “a good environment should accelerate the evolution of the agent and help it find better morphology sooner”. It is also not clear why the authors want to train policies for morphology evolution and environment evolution instead of just optimizing the probability distributions over these spaces, despite the fact that these policies are not used afterwards and only the obtained morphology is used in the test phase. 

3. The clarity of the explanations could be improved. First, the notation inconsistencies makes it confusing. For example, r^m vs r_m, r^E vs r_e, and E and Env. If they are the same, please use the same notation. Algorithm 2 was also not very clear. How could pi_m be updated by using D where transition history doesn’t necessarily have a reward information r_m? The same applies for pi_e.

### Questions
Please clarify the points given in the weakness section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents an approach to co-optimize both the morphology and environments of robots. The morphology and controller of the robot is updated, while the environment is progressively changed. The result of the employed co-evolutionary approach are environments that progressively get more complex, providing a good learning signal for the agent. The approach is compared to ablated versions, which demonstrate that the co-evolution of morphology and environment is beneficial, in addition to comparisons with modifications of methods such as POET, which typically only optimize the robot’s controller but not its morphology.

### Strengths
- Interesting approach that could make robots more robust to varying environments
- Good ablation baseline comparisons

### Weaknesses
- Environment modifications seem limited (e.g. only environment roughness in the case of the 2D environment)
- Comparisons to other methods are a bit ad hoc, e.g. as the authors note, POET was not developed to deal with changing morphologies. In addition to randomly sampling environments here, I would suggest a slightly more advanced baselines that samples environments of increasing complexity
 
Minor comment:

"CMA-ES (Luck et al., 2019) optimizes robot design via a learned value function.” -> their method is not called CMA-ES. CMA-ES is used an evolution strategy for  optimisation

### Questions
- "When the control complexity is low, evolutionary strategies have been successfully applied to find diverse morphologies in expressive soft robot design space” -> how does the control complexity in this paper compare to the one by Cheney et al.? One could say the soft robots in Cheney et al. (2013) are more complex than the robots co-evolved in this paper.
- How expensive is the approach of co-evolving the three different policies? And how does the computational complexity compare to the other baseline approaches?
- It would be good to see some pictures of the evolved environments
- What would happen if you start 3d-locomotion and gap-crossover with the same initial robot as in 2d-locomotion? There already seems to be a lot of bias given with the initial design.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
