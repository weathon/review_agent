# GUARD: A Safe Reinforcement Learning Benchmark

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3, 5

## Abstract
Due to the trial-and-error nature, it is typically challenging to apply RL algorithms to safety-critical real-world applications, such as autonomous driving, human-robot interaction, robot manipulation, etc, where such errors are not tolerable.
Recently, safe RL (i.e., constrained RL) has emerged rapidly in the literature, in which the agents explore the environment while satisfying constraints. Due to the diversity of algorithms and tasks, it remains difficult to compare existing safe RL algorithms.
To fill that gap, we introduce GUARD, a Generalized Unified SAfe Reinforcement Learning Development Benchmark.
GUARD has several advantages compared to existing benchmarks. First, GUARD is a generalized benchmark with a wide variety of RL agents, tasks, and safety constraint specifications. Second, GUARD comprehensively covers state-of-the-art safe RL algorithms with self-contained implementations. Third, GUARD is highly customizable in tasks and algorithms. We present a comparison of state-of-the-art safe RL algorithms in various task settings using GUARD and establish baselines that future work can build on.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors address the challenge of applying reinforcement learning (RL) algorithms to safety-critical real-world applications, such as autonomous driving and human-robot interaction, where errors are intolerable. They introduce GUARD, the Generalized Unified SAfe Reinforcement Learning Development Benchmark, which offers several key advantages over existing benchmarks. GUARD provides a versatile framework with a wide variety of RL agents, tasks, and constraint specifications, encompassing a comprehensive range of state-of-the-art safe RL algorithms with self-contained implementations. Moreover, GUARD is highly customizable, allowing researchers to tailor tasks and algorithms to specific needs. Through GUARD, the paper conducts a comparative analysis of state-of-the-art safe RL algorithms across various task settings, establishing essential baselines for future research. The paper also acknowledges future work to address limitations and broaden the spectrum of available task types within GUARD.

### Strengths
The paper's strength lies in the creation of GUARD, a pioneering benchmark in the field of safe reinforcement learning, which significantly surpasses existing benchmarks. GUARD's unique contributions include an extensive and generalized framework encompassing 11 different types of agents, 7 distinct robot locomotion tasks, and 8 safety constraint specifications. Furthermore, it offers a unified platform with comprehensive coverage of 8 state-of-the-art safe RL algorithms, all implemented with a consistent and independent structure. GUARD's high customizability allows researchers to effortlessly tailor robot locomotion testing suites to their specific needs, with self-customizable agents, tasks, and constraints. This approach fosters clean code organization and facilitates the integration of new algorithms, making GUARD a versatile and powerful tool for advancing research in the domain of safe reinforcement learning.

### Weaknesses
- Although I agree with the authors that there are two groups of safe RL methods, i.e., Hierarchical and end-to-end safe RL, I personally think separating the methods by theoretical guarantee should be a better choice. Also, on-policy or off-policy should also be a choice.
- It would be great if the author could conclude a bit why the methods with theoretical guarantee for constraint satisfaction cannot satisfy the constraint at the early phase of training.  
- The paper covers 8 state-of-the-art safe RL algorithms, but an important off-policy safe RL method WCSAC (published in AAAI-21 and MLJ-23) is missed.

### Questions
- The safe RL algorithms have their own advantages. How can the author compare these algorithms in a more fair manner？
- How can the author ensure these baselines have their best performance in the environments?
- Could the author also address the weaknesses listed above?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces GUARD, a Generalized Unified Safe Reinforcement Learning Development Benchmark, designed to address the challenges of applying RL in safety-critical real-world applications. GUARD offers a comprehensive and customizable environment for evaluating state-of-the-art safe RL algorithms across diverse agents, tasks, and safety constraints. It extends the benchmark beyond existing limits by accommodating various agent types, locomotion tasks, and safety requirements. GUARD promotes consistent and reliable performance comparisons, serving as a valuable tool for advancing safe RL research and bridging the gap between theory and practice in real-world applications.

### Strengths
- Generalization: GUARD provides a wide variety of agents, tasks, and safety constraint specifications, making it a versatile benchmark for testing safe RL algorithms. It accommodates diverse real-world scenarios, ensuring that research is not limited to specific domains.
- Unification: GUARD promotes a unified platform for evaluating safe RL algorithms. By maintaining consistency in experiment setups, it facilitates reliable performance comparisons across different algorithms and controlled environments.
- Self-Contained Algorithms: The benchmark's self-contained structure and independent implementations of algorithms ensure clean code organization and eliminate dependencies between different algorithms. This design facilitates the seamless integration of new algorithms for further extensions, making GUARD a user-friendly and developer-friendly tool.

### Weaknesses
- In Section "4.2 UNCONSTRAINED RL", it is mentioned that TRPO is state-of-the-art, which is clearly not correct given recent improvements to RL policies like agent57 [1] or muzero [2].
- Closely related to the above point, except for USL, all other considered algorithms are for at least 2 years ago, which is in contradiction to what is mentioned in the abstract: "GUARD comprehensively covers state-of-the-art safe RL algorithms". I would strongly suggest including more recent algorithms.
- A few typos: e.g. in Section "5.1 ROBOT OPTIONS", "whosn" -> "shown"
- Figure 4 labels are overlapping making it hard to read.


[1] Badia, Adrià Puigdomènech, et al. "Agent57: Outperforming the atari human benchmark." International conference on machine learning. PMLR, 2020.
[2] Schrittwieser, Julian, et al. "Mastering atari, go, chess and shogi by planning with a learned model." Nature 588.7839 (2020): 604

### Questions
- Any future plans to support the maintenance of the framework? This is very important since there have been a lot of similar works with limited support/maintenance that first, soon fall behind the literature, and second, make researchers and developers reluctant to use them. 
- Any plans to include more algorithms, other than TRPO-based ones?
- Will the authors share the source code of GUARD for free usage?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a novel benchmark for safe RL. Since the popular libraries `safety gym` and `safety starter agents` are not being maintained, there is a need for a maintained library. The authors put a significant effort into the benchmark tasks and the algorithms. They implemented several tasks as well as several benchmark algorithms. They also present the numerical experiments for the tasks and the algorithms.

### Strengths
1. A unified benchmark is really needed for safe RL, especially, when `safety gym` is not maintained anymore
1. The authors presented several different tasks, which appear to be very customizable
1. The benchmark comes with implemented baseline algorithms

### Weaknesses
1. The tasks do not seem to be well-tuned or well-designed depending on the viewpoint. The target cost (which is equal to zero) is not achieved in many experiments, which suggests that the safe algorithms did not learn to be “safe”. I think designing the tasks and tuning the algorithms would be one of the main contributions of this work, but it seems to be lacking.

1. It would be great to provide some details on implementation (apologies if I missed them). For example, it would be good to know if other libraries are being used as the base, e.g., RLLib or torchRL. Also, how to customizable are the algorithms? Can we use Lagrangian TRPO with a safety layer (provided that there are two different types of constraints)?  

1. The choice of TRPO as the base on-policy algorithm may not be the optimal one. It is quite heavyweight and PPO would be preferable at least as an option. Note that it is possible to implement PPO and TRPO in one class as done in the Open AI safety agents. I also recommend implementing Stooke et al 2020 as another baseline, which in my experience works really well.

I appreciate that these weaknesses may seem like minor ones, but I feel it is very important to get these things right the first time. If the authors have better results for the algorithms and the tasks I will consider raising the score.

### Questions
Page 1 “....human-competitive performance in a wide variety of tasks, such as games s (Mnih et al., 2013; Silver et al., 2018; OpenAI et al., 2019; Vinyals et al., 2019; ?),”” broken reference 

Missing important reference: 
Stooke, Adam, Joshua Achiam, and Pieter Abbeel. "Responsive safety in reinforcement learning by pid lagrangian methods." International Conference on Machine Learning. PMLR, 2020.

Very recent references that the authors may consider for future works:

Zuxin Liu, Zhepeng Cen, Vladislav Isenbaev, Wei Liu, Steven Wu, Bo Li, and Ding Zhao. Constrained variational policy optimization for safe reinforcement learning. In International Conference on Machine Learning, pages 13644–13668. PMLR, 2022. 

Sootla, A., Cowen-Rivers, A. I., Jafferjee, T., Wang, Z., Mguni, D. H., Wang, J., & Ammar, H.  Saute rl: Almost surely safe reinforcement learning using state augmentation. In International Conference on Machine Learning (pp. 20423-20443). PMLR, 2022.

Yu, Haonan, Wei Xu, and Haichao Zhang. "Towards safe reinforcement learning with a safety editor policy." Advances in Neural Information Processing Systems 35 (2022): 2608-2621.

Wachi, Akifumi, et al. "Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms." arXiv preprint arXiv:2310.03225 (2023).

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors propose GUARD: a safe reinforcement learning benchmark that comes with several interesting safe RL environments and several implementations of model-free constrained RL algorithms, both old and new. Specifically the authors consider the Constrained Markov Decision Process (CMDP) with cumulative constraints. They implement several model-free algorithms CPO (Achiam et al. 2017), TRPO-lagrangian (Bohez et al. 2019), TRPO-FAC (Ma et al. 2021), TRPO-IPO (Lie et al. 2020), PCPO (Yang et al. 2020), including two hierarchical algorithms TRPO-SL (Dalal et al. 2018) and TRPO-USL (Zhang et al. 2022). In addition they provide an unconstrained baseline TRPO (Schulman et al. 2015). 

By equipping each algorithm with the same MLP architecture the authors obtain a fair comparison between all these methods across a number of environments. The environments can be configured with 8 different control robots, over 4 possible task and 3 possible constraint configurations. The authors provide some discussion on the relative complexity of these possible configurations, accompanied by a discussion regarding the performance profile of the different algorithms.

### Strengths
(1) The paper is very clear and the benchmark is described incredibly thoroughly, I only had some minor confusion which can be adressed in the questions.

(2) This benchmark is clearly a direct competitor with the likes of Safety Gym and newer benchmarks like Safety-Gymnasium (Ji et al. 2023) and Omnisafe (Ji et al. 2023). However, it distinguishes itself by having 8 completely different robot configurations and some new task and constraint configurations.

(3) The problem is well motivated. And GUARD provides a new set of interesting environments that I am sure researchers will be looking forward to tackling.

### Weaknesses
(1) Only model-free algorithms are implemented in this benchmark. Model-based approaches to safe RL have grown in interest recently, since they exhibit better sample complexity, it would be nice to see maybe two popular model-based from the literature available here.

(2) Little backwards compatibility with Safety Gym. Clearly Safety Gym (Ray et al. 2019) has inspired this work, it would be nice if the 3 robots and 3 environment configurations from Safety Gym were captured by GUARD.

(3) The number of algorithms implemented is far from exhaustive. Of course I don't expect every safe RL algorithm to be implemented, but unfortunately other benchmarks like Safety-Gymnasium (Ji et al. 2023) and Omnisafe (Ji et al. 2023) do have a more comprehensive suite of algorithms available.

(4) Lack of a comprehensive discussion of safe RL and the limitations of considering CMDPs. In the original Safety Gym paper, Ray et al. discuss more broadly safe RL and safe exploration, and where the problem they are trying to adress sits in the literature. They also adress the critiques of CMDP and considering simple cumulative constraints.  It would be nice if a similar discussion of the broader context of this research was present in the paper and how you might adress other problems in the future.

For these reasons I am not recommending acceptance of the paper. For me the benchmark does seem useful and the novelty of the environments makes the benchmark distinct from similar "competitors", for lack of a better word. Issues (1)-(3) are not big issues for me. However, I am reviewing a paper not a benchmark and issue (4) does poses a problem in this instance. Without providing a thoughtful discussion of the broader context of safe RL the paper seems more like a user manual than an interesting and thought provoking read. If the authors adress this weakness I would consider recommending acceptance, since I do see merit in this work.

### Questions
What is the reward peformance measure, discounted accumulated reward, or total reward accumulated throughout the entire episode?

Simialry is the cost performance measure, the discounted accumulated cost, or total cost accumulated throughout the entire episode?

How is the cost rate performance calculated?

Is the target cost set to 0.0, if so did this pose any instability issues?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
