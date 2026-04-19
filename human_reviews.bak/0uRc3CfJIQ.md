# ORSO: Accelerating Reward Design via Online Reward Selection and Policy Optimization

- Decision: Accept (Poster)
- Scores: 3, 5, 8, 8, 6, 5

## Abstract
Reward shaping is critical in reinforcement learning (RL), particularly for complex tasks where sparse rewards can hinder learning. However, choosing effective shaping rewards from a set of reward functions in a computationally efficient manner remains an open challenge. We propose Online Reward Selection and Policy Optimization (ORSO), a novel approach that frames the selection of shaping reward function as an online model selection problem. ORSO automatically identifies performant shaping reward functions without human intervention with provable regret guarantees. We demonstrate ORSO's effectiveness across various continuous control tasks. Compared to prior approaches, ORSO significantly reduces the amount of data required to evaluate a shaping reward function, resulting in superior data efficiency and a significant reduction in computational time (up to 8×). ORSO consistently identifies high-quality reward functions outperforming prior methods by more than 50% and on average identifies policies as performant as the ones learned using manually engineered reward functions by domain experts.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This manuscript proposes the Online Reward Selection and Policy Optimization (ORSO) to frame shaping reward selection as an online model selection problem. It automatically identifies promising shaping reward functions, balancing exploration and exploitation with provable regret guarantees. The ORSO method significantly improves sample efficiency and reduces computational time compared to traditional methods that fully evaluate each shaping reward function.

### Strengths
The idea is a little simple but effective.

### Weaknesses
Some experiments and theories can be added.

### Questions
1. The proposed method generates a set of candidate reward functions for the online selection phase. Does having K reward function candidates mean that the number of candidates is fixed? If all these candidates are not suitable or not the best, what is the solution?

2. The experiments compared the performance of policies trained using No Design, Human, Naive Selection, and ORSO to show the superiority of ORSO. However, the impact of selecting different reward functions for ORSO algorithm on experimental results has not been analyzed. If possible, please provide relevant experiments to demonstrate the experimental differences caused by selecting different reward functions.

3. Figure 4 shows the normalized cumulative regret for different selection algorithms. The manuscript mentioned that the ORSO’s regret can become negative, indicating that it finds reward functions that outperform the human baseline. The minimum value is zero in Figure 4, I didn’t observe the negative values.

4. There is a similar paper ORSO: Accelerating Reward Design via Online Reward Selection and Policy Optimization published in ARLET 2024. What is the difference between these two works? Has the ARLET paper been cited?

5. The number of references is small, and more recent articles on reward shaping can be added.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposed an Online Reward Selection and Policy Optimization (ORSO) algorithm for reinforcement learning. ORSO pre-generates some candidate reward functions by linearly combining some reward components or by LLM, while learning, ORSO dynamically evaluates which candidate reward function can lead to better policy optimization, then selects the optimal candidate to guide the learning process.

### Strengths
The approach is easy to implement and effective at selecting reward functions, it also shows fast convergence in terms of both computational time and sample efficiency.

### Weaknesses
1. As this method follows a "pre-define and select one" paradigm, the final optimal performance that ORSO can achieve heavily depends on how good are the pre-generated candidate reward functions.
2. The author states that this is a reward shaping approach, but the paper doesn't compare it with any reward shaping or reward selection baselines. If the authors could compare ORSO with some representative reward shaping algorithms (such as [1][2][3][4]), it would better showcase its advantages.

[1] Devidze, Rati, Parameswaran Kamalaruban, and Adish Singla. "Exploration-guided reward shaping for reinforcement learning under sparse rewards." Advances in Neural Information Processing Systems 35 (2022): 5829-5842.

[2] Zheng, Zeyu, Junhyuk Oh, and Satinder Singh. "On learning intrinsic rewards for policy gradient methods." Advances in Neural Information Processing Systems 31 (2018).

[3] Memarian, Farzan, et al. "Self-supervised online reward shaping in sparse-reward environments." 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021.

[4] Burda, Y., Edwards, H., Storkey, A., and Klimov, O. (2018). Exploration by random network distillation. In International Conference on Learning Representations.

### Questions
1. Regarding Weakness 1, about candidate reward function generation, As described in  section 5.1.2, the candidate reward functions are directly generated through LLM, Could you clarify:

(a) What are the specific forms of these reward functions? Are they related to the observations, features, and/or pre-defined reward components? Can these generated rewards capture all the necessary aspects to define effective rewards?

(b) If the optimal reward function is not included in the generated candidates, how does ORSO ensure that the final optimized policy is indeed optimal?

2. the policy optimizes based on a given candidate reward function, would this make it difficult to ensure that the policy is optimizing the task's original objective (the environmental reward function defined by the MDP)?

3. Frequent switching of reward functions may lead to significant shifts in the policy's learning objectives. For instance, in a maze task, if reward function #1 focuses on avoiding obstacles, while reward function #2 focuses on resource collection, switching between these two may lead to inconsistent learning targets. Would this cause instability in the learning process?

4. I'm unclear about the evaluation metric in the experiments, specifically, in Figure 2 (left), it shows performance as a percentage of the human-designed reward function. In Section 5.1.1, the paper states, "No design is with the task reward function r for each MDP". I assume this refers to the original environmental reward function, which should be the primary objective the agent aims to optimize. However, in Figure 2 (left), the "No design" baseline is around half of the human-designed reward (I assume this figure reports cumulative rewards under each baseline's own reward function). This seems unfair and could introduce bias for deviating from the MDP’s original task objective. 

For example, suppose the MDP provides rewards of $0, 0, 1$ for states $s_1, s_2, s_3$ (only 3 states). A human-designed reward function might assign $0, 1, 1$ for $s_1, s_2, s_3$. Consequently, the cumulative reward under the human-designed reward function would be higher, and it also proposes new targets (both $s_2$ and $s_3$ are equally important). From my understanding, the performance should be evaluated consistently on the original MDP reward (the true objective), meaning that the "No design" case should actually serve as an upper bound.

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
4

### Summary
This paper introduces ORSO, a method that aims to increase the efficiency of the reward shaping selection process by casting it as an automated online model selection problem.
ORSO is a general algorithmic framework that implements the following steps: (i) a reward function generation method is used to provide a set of candidate shaping reward functions, (ii) a selection algorithm is used to select a shaping reward function, (iii) an RL algorithm is used to train the policy associated with the selected shaping reward function for a set amount of iterations, (iv) the trained policy is then evaluated against the task reward function and the utility is used to update the parameters of the selection algorithm. This process is repeated until a predefined computational budget is exhausted.
While the components within the ORSO framework are modular and exchangeable, this work uses (i) an LLM based generator as the reward function generation method, (ii) PPO as the RL algorithm, (iii) D3RB, as the reward function selection algorithm (ablations are additionally conducted with Exp3, D3RB,UCB, ETC, and EG).
Experiments are conducted across tasks of varying difficulty, 3 budgets, and 3 reward functions sets.
Results indicate that the ORSO performance in terms of task return scales with budget, and is comparable - and can surpass - that of human defined shaping reward functions; additionally ORSO is twice as fast as a prior shaping reward function selection method (EUREKA).

### Strengths
Claims: (1)  ORSO reduces both time and computational costs by more than half compared to earlier methods, making reward design accessible to a wider range of researchers. (2) By formalizing the reward design problem and providing a theoretical analysis of ORSO’s regret when using the D3RB algorithm, we also contribute to the theoretical understanding of reward design in RL.

- The work proposes an original formulation of the shaping reward function selection process, viewing it as an online model selection problem.
- The regret based analysis provides a clear and intuitive way to monitor ORSO's performance and efficiency gains.
- Thanks to the problem formulation, the method is kept elegant in its simplicity and largely amounts to the application of existing approaches into a unified framework.
- Results on a variety of tasks and domains against available baselines support the efficiency and performance claims.

Clarity:
- The writing is clear, the paper is well structured, and appropriate context is provided to the reader.

Significance:
- This work tackles the issue of reward design for RL. This has been and continues to be one of the most significant challenges keeping RL from widespread successful deployment in the real world.

### Weaknesses
Contribution:
- Ultimately, ORSO searches over a set of shaping reward functions. While the framework is simple and elegant, to my understanding, it ultimately relies on and is limited by the performance of the "shaping reward function generator".
- ORSO is only benchmarked against methods for which a performant human-engineered reward function can be defined. Impact would be higher if method could generalize beyond these settings.
- ORSO in its current form does not seem to offer the flexibility to deal with changing / annealing of shaping reward functions throughout training, a common technique in reward shaping literature.

### Questions
- Unique contributions could be made clearer and explicitly called out in the introduction.

- Lines 54-59: I understand you are highlighting unique challenges compared to standard multi-arm bandit settings, yet ORSO uses the same selection algorithms typically used to solve such multi-arm bandit problems. The paper could be clearer in defining exactly which components of ORSO are key to address the unique challenges presented.

- I recommend expanding on the various resampling strategies, if more than one was tried out, and their impact on performance as this seems to be a key ingredient to the method's success.

- I would recommend adding the synthetically generated best-performing shaping reward functions for each task to appendix E. Are the reward functions sensible to the human reader? This has implications on how well these shaping reward functions could be further refined by human experimenters, and possibly give insights on their logical soundness.

- Also, was any constraint, structure, human knowledge beyond the imposed when prompting the generation of such rewards or could the prompt be arguably generated programmatically (if so, I recommend just stating it - the code base is not available during the review process to verify)?  While not directly related to the ORSO contribution, this is arguably important to showcase as ORSO heavily relies on the existence of an automated way of generating reward functions without human priors.

- Please look at the weaknesses section and help clarify if any are can be addressed.

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
This paper studies automated reward shaping by posing it as an online reward selection problem. Instead of multiple training runs to observe the impact of different shaping functions, this work aims to identify the best one within a fixed time budget. To this aim, the authors develop ORSO, a regret-minimizing approach that utilizes multi-armed bandits where a candidate shaping function is an arm. More specifically, ORSO uses the D3RB algorithm to select an arm. Upon selection of an arm, ORSO trains a policy corresponding to the said arm for a fixed number of iterations and then evaluates the policy with respect to the task rewards. The paper provides regret guarantees, assuming that a learner monotonically dominates all learners and its average performance increases monotonically. 

The paper evaluates a practical implementation of ORSO in continuous control tasks with varying complexity whose rewards are either sparse or unshaped. The experimental results show that ORSO is faster than an LLM-based reward design method, can surpass human-designed rewards, and performs better as the budget increases. The authors also provide an ablation study for different arm selection strategies and different numbers of candidate shaping functions.

### Strengths
- The paper is very well-written. The motivation is clearly explained, and the problem and assumptions are well described. Moreover, given the assumptions, the proposed approach and its theoretical guarantees are clear.

- The experimental set-up makes sense, and the evaluated baselines allow us to see how reward design is critical, humans can be sub-optimal at it, and naive attempts are prone to fail.

- The experimental results clearly showcase its advantages in comparison to the baselines. In addition, the ablation study evaluates the impact of different components of ORSO and provides detailed insights.

### Weaknesses
- I urge the authors to move the related work, at least the most relevant parts, to the main document.

- Assumption 4.2 seems limiting. A discussion of why the assumptions are viable or how they are needed would strengthen the paper's arguments. It would be even better to explain their role in causing the contrast with the regret guarantees in Pacchiano et al. (2023).

- As the quality of candidate shaping functions plays an important role, an ablation study to understand the impact of wrong/redundant candidates would help the reader understand the limitations of ORSO.

### Questions
- In what cases would the monotonicity assumption be violated? Do the environments in the experimental set-up violate or obey the assumption? How would ORSO handle such violations?

- Future work mentions exciting directions. Since the naive approach is failing, how likely is a VLM-based reward design method to fail?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Online Reward Selection and Policy Optimization (ORSO), an approach that defines reward selection as an online model selection problem. The approach uses exploration strategies to identify shaping reward functions.

### Strengths
Soundness
======
The approach is generally sound. 
	
Significance & Related work
=========
The paper presents an in-depth related work section, and well defined preliminaries (note redundancy of section 2)  that lead to demonstrations of several results. 

Experimentation
=========
The paper presents an in-depth ablation analysis of the performance of various selection algorithms.

Presentation
=========
The paper is well written.

### Weaknesses
Soundness
======
It is unclear in the experiments in Fig 2 what ‘human-level performance’ or ‘human-designed reward function’ is and how it is defined/computed. Note that the proof for D1 needs to be rewritten for clarity to show base case and inductive hypothesis, should proof by induction still be the chosen approach.
	
Experimentation
=========
The paper presents an in-depth ablation analysis of the performance of various selection algorithms, however, the impact of poorly chosen task rewards needs to be analysed. 

Presentation
=========
Presentation is good, as above, Section 2 is too short and redundant.

### Questions
* What is human designed reward and how it is computed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper employs the data-driven online model selection algorithm D^3RB (Pacchiano et al., 2023) to choose between candidate reward functions for reinforcement learning, where these candidates are generated through an LLM. It replaces the naive reward selection from Ma et al. (2023) with D^3RB, enabling more efficient online selection among candidate rewards. A simple example shows how D^3RB helps prevent budget exhaustion by avoiding over-allocation to a single option. The paper further evaluates the algorithm’s effectiveness on Isaac Gym (Makoviychuk et al., 2021), comparing baseline and human-designed rewards. Ablations also explore various bandit exploration strategies, including UCB, EXP3, ETC, EG, and Naive.

**References**

- Pacchiano, A., Dann, C., & Gentile, C. (2023). *Data-driven regret balancing for online model selection in bandits.* arXiv preprint arXiv:2306.02869.
    
- Ma, Y. J., Liang, W., Wang, G., Huang, D.A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., & Anandkumar, A. (2023). *Eureka: Human-level reward design via coding large language models.* arXiv preprint arXiv:2310.12931.
    
- Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Macklin, M., Hoeller, D., Rudin, N., Allshire, A., & Handa, A., et al. (2021). *Isaac Gym: High performance GPU-based physics simulation for robot learning.* arXiv preprint arXiv:2108.10470.

### Strengths
The paper addresses a significant issue in reinforcement learning, as sparse rewards indeed pose challenges, and reward shaping can provide dense feedback, potentially accelerating learning. The writing is clear, with well-presented graphs and experiments, and a well-defined motivation. The paper also successfully applies a recent online model selection algorithm, demonstrating in the proposed benchmark how the algorithm can aid in selecting the appropriate reward functions with limited budget and can scale effectively with increased budget and candidate reward functions.

### Weaknesses
**Writing**: In Section 3.1, ORSO is presented as an effective and efficient method for reward function design. However, my understanding is that ORSO simply selects a reward function from a set of candidates, with the reward generation handled separately. If this is correct, it would be helpful if the paper clearly defined ORSO’s components, specifying whether reward generation is part of ORSO or assumed to be user-provided. Clarification here would improve understanding.

**Complexity**: The proposed method requires maintaining $K$ separate policies for each reward, which could create substantial memory overhead. Standard approaches in reward shaping often aim to achieve similar benefits while maintaining only a single policy, highlighting a potential drawback in scalability.

**Performance Guarantees**: Traditional reward shaping studies typically ensure that the shaped reward aligns with a base reward function. Here, there is no guarantee that the proposed objective aligns with the base reward, leaving the outcome quality solely dependent on the reward generation process. This could raise concerns about consistency in performance.

**Experiments**:

1. The paper does not compare with established reward shaping methods, such as those by Ng et al. (1999), Zou et al. (2019), Zheng et al. (2018), Sorg et al. (2010), and Gupta et al. (2023). Including these comparisons would strengthen the experimental evaluation.
    
2. Iterative resampling appears to be essential for obtaining high-quality reward functions, as it involves reinitializing policies and refining rewards over iterations. However, the paper lacks discussion on the resampling process's challenges, frequency, and impact on performance. Additionally, it is unclear if resampling is an integral part of ORSO or a separate process.
    
3. In the experiments, performance is evaluated against human-designed rewards, but the actual evaluation should ideally be based on the baseline reward. This raises questions about the metrics used for evaluation. In Figure 2, the upper bounds should correspond to **No Design**, as the objective should be to assess performance against the MDP’s base reward.
    
4. The reward generation process seems to require access to code-level details of the MDP, which may not be feasible in cases where the environment is not code-based. Discussion of this limitation would improve transparency regarding the method’s applicability.
    

**References**

- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* In ICML.
    
- Zou, H., Ren, T., Yan, D., Su, H., & Zhu, J. (2019). *Reward shaping via meta-learning.* arXiv preprint arXiv:1901.09330.
    
- Zheng, Z., Oh, J., & Singh, S. (2018). *On learning intrinsic rewards for policy gradient methods.* Advances in Neural Information Processing Systems.
    
- Sorg, J., Lewis, R. L., & Singh, S. (2010). *Reward design via online gradient ascent.* Advances in Neural Information Processing Systems.
    
- Gupta, D., Chandak, Y., Jordan, S. M., Thomas, P. S., & da Silva, B. C. (2023). *Behavior Alignment via Reward Function Optimization.* arXiv preprint arXiv:2310.19007.

### Questions
The weaknesses section raised some key questions, which I’ll summarize here for clarity:

1. ORSO functions only as a selection algorithm, correct? Reward generation isn’t part of the algorithm itself?
    
2. Does the paper simply apply the D^3RB algorithm, or does it introduce theoretical improvements? This aspect is somewhat unclear.
    
3. As I understand it, there’s no guarantee that the optimal policy obtained with the selected reward aligns with the optimal policies for the base reward (i.e., **No Design**), correct?
    
4. Could you clarify the evaluation metrics? The ideal benchmark should be based on the base reward, as that’s ultimately the reward we aim to optimize.
    
5. The terminology around “effective budget” seems confusing. Based on the proposed algorithms, the effective budget for environment interactions should be $TN$ rather than $T$, since each iteration assumes running the algorithm for at least $N$ steps to yield a final policy. Could you clarify this? This also seems to affect Figure 1—does preferred reward selection occur after $N$ iterations or a single one? As I understand it, each reward would at-least have to be evaluated $N$ times, hence making a minimum budget of $KN$, right?

### Soundness
2

### Presentation
3

### Contribution
2
