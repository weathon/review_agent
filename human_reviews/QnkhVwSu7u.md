# ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Interpretable Reward Design in Robotics

- Decision: Reject
- Scores: 3, 8, 5, 5

## Abstract
Reinforcement learning (RL) has demonstrated compelling performance in robotic tasks, but its success often hinges on the design of complex, ad hoc reward functions. Researchers have explored how Large Language Models (LLMs) could enable non-expert users to specify reward functions more easily. However, LLMs struggle to balance the importance of different features, generalize poorly to out-of-distribution robotic tasks, and cannot represent the problem properly with only text-based descriptions. To address these challenges, we propose ELEMENTAL (intEractive LEarning froM dEmoNstraTion And Language), a novel framework that combines natural language guidance with visual user demonstrations to align robot behavior with user intentions better. By incorporating visual inputs, ELEMENTAL overcomes the limitations of text-only task specifications, while leveraging inverse reinforcement learning (IRL) to balance feature weights and match the demonstrated behaviors optimally. ELEMENTAL also introduces an iterative feedback-loop through self-reflection to improve feature, reward, and policy learning. Further, ELEMENTAL reward functions are interpretable. Our experiment results demonstrate that ELEMENTAL outperforms prior work by 24.4\% on task success, and achieves 41.3\% better generalization in out-of-distribution tasks, highlighting its robustness in LfD.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper framework that can utilize a combination of fast simulation, demonstration, and VLMs. The main idea is to (i) use VLMs to useful state features from environment code and image from demonstration, (ii) run IRL algorithm to learn reward and the policy from state features , and (iii) use the state feature matching counts as a reflection metric to compare how the learned policy is close to demonstration. The main contribution of this paper lies in investigating a way to utilize demonstrations and VLMs for improving IRL, which seems a promising direction to pursue.

### Strengths
- Idea is intuitive and direction seems promising
- Improved performance from incorporating VLMs into IRL pipeline

### Weaknesses
Overall, I like the idea of this paper but it's missing too many results/analysis/discussion to support why & how the method works and what happens during the training.

The main weakness of this paper is that it's only reporting the numbers in main table and does not provide results that can help readers understand how the proposed idea works and that can support the claims made in the paper. For instance,
- Despite the claim, the paper is missing any result or discussion/analysis on the interpretability of rewards, and how it is helpful
- Despite the proposed framework has multiple iterations of training, it's not clear how the performance changes across the iteration, how the *interpretability* of reward improves.
- Is VLM really understanding what demonstration is? What would happen if the VLMs receive sub-optimal data from random exploration in the same tasks? What would happen if you give demonstrations in a different way?
- How crucial it is to give demonstrations in visual observation? what would happen if you give demonstrations as a sequence of states to LLMs instead of using VLMs?
- It's missing analysis/experiments that investigate the effect of VLM choices on the effectiveness of the framework. It could be also nice to include results that show how the method is sensitive to the choice of VLMs. For instance, how good the VLMs should be good to enable this framework to work? Would open-source models be okay? 

Also, experiments are missing some details and baselines:
- Details on experimental setup is not clear. Are all the methods using the same resources for training? It's not clear if all the models are trained until convergence.
- It seems to me that BC performance is a bit weak but it's difficult to understand why as there are not that many experimental details. What would be the performance if we use a powerful BC algorithm such as DiffusionPolicy? 

Chi, Cheng, et al. "Diffusion policy: Visuomotor policy learning via action diffusion." The International Journal of Robotics Research (2023): 02783649241273668.

### Questions
See Weaknesses

### Soundness
2

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
3

### Summary
Using large language models (LLMs) for specifying reward functions has shown success in reinforcement learning. However, existing methods like EUREKA describe tasks and rewards only through language. To remedy this, ELEMENTAL combines learning from demonstration (LfD) and LLMs for reward engineering to extract features, learn a policy and reward, and iteratively refine features. ELEMENTAL leads to better generalization and performance than existing methods.

### Strengths
This paper makes a clear extension from EUREKA, by incorporating visual inputs. In addition, it proposes a novel framework of feature extraction, learning, and reflecting. In particular, I find the self-reflection loop to be compelling

The diagram and writing are clear.

### Weaknesses
The authors mention that ELEMENTAL helps “align robot behavior with user intentions better” and that EUREKA allows humans to “interpret and interactively refine the robot’s behavior” and is more “user-aligned” (line 144). There are no experiments or further discussion of this, and it is not explored in this work.

Further discussion on the effect of self-reflection, such as the types of features that are discovered through self-reflection, would be interesting.

### Questions
1. What is the training time needed for these tasks?
2. Why is Peng et al. (2024b) not included as a baseline?
3. How does this work compare with reinforcement learning from VLM rewards?
4. A more thorough appendix would be useful. For instance, what dimension are the features? What are example features?
5. What is the environment state space for the tasks (line 248)?
6. In Table 2, what are the differences in the EUREKA and ELEMENTAL implementations? For example, are the VLMs / LLMs different, and might that account for the reduced generalization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a novel approach to reward tuning that combines language goals and visual user demonstration inputs with Vision-Language Models (VLMs) to address ambiguities inherent in language-only reward tuning. The method consists of three stages. In the first stage, simulator code, language-based goal descriptions, and images of user demonstrations are provided to a VLM, which then generates code to calculate features relevant to the task. In the second stage, a reward function and policy are learned online using maximum-entropy Inverse Reinforcement Learning (IRL). Finally, in the third stage, rollouts from the policy are used to compute the discrepancy between the feature counts in the user demonstrations and those in the actual rollouts. This discrepancy is fed back into the VLM for iterative refinement, alternating between the second and third stages to optimize performance. Experiments conducted on Isaac Gym tasks demonstrate that the proposed method achieves superior performance compared to state-of-the-art (SOTA) language-only reward tuning and IRL methods that do not leverage VLMs.

### Strengths
- The proposed method leverages VLMs to propose features relevant to the task rather than generating the entire reward function. This approach minimizes the risk of overfitting to environments encountered during VLM/LLM training and reduces issues with low code execution rates, as confirmed by the paper’s experimental results.
- The proposed framework takes in both a language goal and visual demonstrations, effectively addressing the ambiguities associated with using only one type of input.

### Weaknesses
- The requirement for MDP environment code as an input limits this method to simulated environments. In real-world applications, this would require explicitly specifying all relevant objects and dynamics, which could be impractical or infeasible.
- To what extent does the assumption that the reward is a weighted sum of the feature vectors limit the expressiveness of the reward function? This limitation excludes more complex functional forms, such as exponentials, logarithmic functions, or features in the denominator, potentially limiting the method's ability to capture nuanced task-specific details.
- When using a superimposed image as a visual demonstration in tasks like navigation, there is an inherent ambiguity in capturing the temporal direction of actions. For tasks where superimposed images are unsuitable, the method selects approximately four keyframes from the demonstration, introducing an additional need for keyframe identification.
- The method may struggle with highly complex simulation environments, as it requires the entire MDP environment code as input.

### Questions
- How does the allowance of up to 3 attempts impact the method's effectiveness? For tasks where the method fails after 3 attempts, would increasing this threshold lead to successful code executions, or would these tasks likely fail at this stage regardless of additional attempts?
- How does the proposed reward in the experiments compare with the ground truth reward? Would be good to see comparisons and analysis on this
- In Table 1, over how many iterations is the proposed method trained? Does each iteration consistently lead to performance improvements? It would be helpful to see reward curves plotted against iteration count to better understand the effectiveness of the reflection stage.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes "ELEMENTAL", an approach for integrating user demonstrations in language-model-based reward specifications such as in EUREKA. The paper argues that such demonstrations could alleviate the ambiguity of language task specifications. The approach has 3 phases. The first one involves prompting the VLM with text and demonstrations to obtain an executable feature function. In this stage, the demonstration is represented as either a superimposed image or four keyframes. In the second stage, the aim is to learn a reward function that is linear in the features and tries to match the demonstrations, and in an inner loop updates the policy via PPO and using the obtained reward. In the last stage, the agent reflects on its feature function using the discrepancy of the feature counts in the dataset and policy-generated trajectories. The paper includes several simulation-based evaluations of the method and mainly compares it to EUREKA as a baseline. The results show a consistently significant improvement in performance in comparison to EUREKA.

### Strengths
- the paper is very well written and self-contained. It's also quite a smooth read.
- the proposed approach is quite interesting and using demonstrations to alleviate language ambiguity is a needed step for automatic reward design.
- the experiments in simulation are quite extensive and the results support the main claims of the paper.
- the experiments include important ablations of some aspects of the method.

### Weaknesses
- the paper lacks motivation for the choice of an IRL-based approach (with reward linear in the features) to include visual inputs as opposed to following the EUREKA-style approach.
- the paper is only validated in simulation. It would be interesting to see whether these approaches could alleviate the reward engineering usually required to handle real-world problems such as jerky motions and unsafe behaviors. Just one real-world experiment with a robot would suffice. For instance, looking at your simulation environments, an experiment with either the Franka, ANYmal, or ShawdowHand would be a great addition to the evaluation. This could also be a sim-to-real transfer experiment.
- the paper lacks ablations of the various normalization steps (equations 6 and 7).
- the paper does not include any information on the amount of time needed for a full run of the algorithm on the various tasks. I think this aspect is very important for readers to decide whether the approach is feasible for their applications and for future methods to improve upon this. I would suggest comparing the wallclock time of running your method and EUREKA.

I am willing to raise my score if these points are properly addressed.

### Questions
- how is keyframe selection performed for the manipulation tasks? This aspect is important to understand the assumptions you make (do you assume to have access to such keyframes from an oracle/user for each demonstration?) or do you have some method to get them (very important for reproducibility)?
- In Table 2, is this the reward or success rate or what? why is it missing a standard deviation? why is there such a difference between ant original and ant with reversed observation? this difference might indicate that the statistical significance of the claims based on this table is questionable. Please update the paper (text and table caption) to make the metric clearer.
- how important/necessary are the normalization steps?
- ELEMENTAL without visual inputs is consistently worse than EUREKA, which brings the question of whether a different approach to reward design (not IRL-based) is more suitable for this kind of problem. Can you please discuss this?
- intuitively, what would be the advantage of this method versus using a VLM itself as a success detector? success detectors like the ones presented in [1] could in theory also receive textual instructions and demonstrations. I agree that they would give out a sparse reward signal, or if they are modified to produce denser rewards this value would be quite uncalibrated, but I am interested in the authors' opinion on this matter.

[1] Du, Yuqing, et al. "Vision-language models as success detectors." arXiv preprint arXiv:2303.07280 (2023).

### Soundness
3

### Presentation
4

### Contribution
2
