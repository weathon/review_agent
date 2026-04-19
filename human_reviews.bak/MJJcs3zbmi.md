# Discovering Temporally-Aware Reinforcement Learning Algorithms

- Decision: Accept (poster)
- Scores: 8, 8, 5

## Abstract
Recent advancements in meta-learning have enabled the automatic discovery of novel reinforcement learning algorithms parameterized by surrogate objective functions. To improve upon manually designed algorithms, the parameterization of this learned objective function must be expressive enough to represent novel principles of learning (instead of merely recovering already established ones) while still generalizing to a wide range of settings outside of its meta-training distribution. However, existing methods focus on discovering objective functions that, like many widely used objective functions in reinforcement learning, do not take into account the total number of steps allowed for training, or “training horizon”. In contrast, humans use a plethora of different learning objectives across the course of acquiring a new ability. For instance, students may alter their studying techniques based on the proximity to exam deadlines and their self-assessed capabilities. This paper contends that ignoring the optimization time horizon significantly restricts the expressive potential of discovered learning algorithms. We propose a simple augmentation to two existing objective discovery approaches that allows the discovered algorithm to dynamically update its objective function throughout the agent’s training procedure, resulting in expressive schedules and increased generalization across different training horizons. In the process, we find that commonly used meta-gradient approaches fail to discover such adaptive objective functions while evolution strategies discover highly dynamic learning rules. We demonstrate the effectiveness of our approach on a wide range of tasks and analyze the resulting learned algorithms, which we find effectively balance exploration and exploitation by modifying the structure of their learning rules throughout the agent’s lifetime.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigates the problem of discovering reinforcement learning algorithms via meta-learning. The paper reveals that incorporating temporal data regarding the agent's lifetime empowers the newly discovered algorithm to dynamically refine its objectives, such as the exploration-exploitation tradeoff, thereby fostering the development of a higher-performing agent. The proposed method can be readily combined with two existing algorithm, LPG and LPO, and demonstrates its effectiveness on a wide variety of benchmarks, including MiniGrid and MinAtar.

### Strengths
- Novelty: Although the proposed method is a simple modification of previous algorithms, the idea of utilizing temporal information is well-motivated and novel.
- Presentation: The authors meticulously analyze the evolution of the discovered objectives throughout the agent's lifetime, detailed in Section 5.3, with Figures 3 and 4 providing compelling visual representations of these dynamic shifts.
- Experiment: The authors meticulously design experiments to rigorously assess the generalization ability of the proposed method to different environments or training hyperparameters (e.g. the number of environment interactions).

### Weaknesses
- Novelty: The application of antithetic sampling for gradient estimation, while prevalent in black-box optimization, appears to lack originality.
- Ablation: The authors do not conduct a comprehensive ablation study comparing the proposed TA-LPG with the standard LPG. Specifically, they introduce two significant modifications, incorporating temporal information and employing antithetic sampling, without isolating the effects of each change.

### Questions
- Ablation: Could you please provide me a clear understanding of the individual contributions of the two proposed component to the performance improvements?
- Hyperparameter: Setting the entropy coefficient of PPO to 0.0 in the Brax environment appears to significantly limit exploration in my view and poses questions about the experimental design. To address this and strengthen the validity of your results, I recommend conducting a systematic hyperparameter search on the entropy coefficient of PPO within the Brax environment.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes to meta-learn a better objective function for reinforcement learning (RL) tasks by taking into account the information of the task horizon.
Specifically, with the help of the time-step information, the proposed algorithms could find a better balance between exploration and exploitation.
Furthermore, this work shows that meta-gradient methods fail to adapt to different horizons, while evolution strategies can do better in this case.

### Strengths
The idea behind this work is simple and effective, supported by solid experiments and detailed analysis.

### Weaknesses
- The idea of incorporating time-step information is not novel in RL, such as this [work](https://proceedings.mlr.press/v80/pardo18a.html), which should be included in the related work.

- Lack of ablation study: In Section 4.1 Equation 9, $n/N$ and $\log(N)$ are included as part of the input to the agent. It is not clear or argued how good this option is compared with other options, such as $n$ and $N$, $N-n$, or $n/N$.

### Questions
- Section 4.1, after Equation 10, "Since TA-LPO is only meta-trained on a single environment and horizon, we do not include the $\log(N)$ term since it does not vary across updates." Would including the $\log(N)$ term decrease the performance?
- Section 5.2, Paragraph "Lifetime conditioning improves performance on out-of-distribution environments." Figure Figure 2: typo.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a training horizon to meta-reinforcement learning algorithms to discover objective functions that depend on the learner's lifetime. The basic idea is to add the information about the lifetime to the input vector of the meta-learning algorithm. The authors propose two algorithms, Temporally-Adaptive Learned Policy Gradient (TA-LPG) and TA-Learned Policy Optimization (TA-LPO) by extending LPG and LPO, respectively. Then, the authors found that the evolutionary strategy is more appropriate than meta-gradient approaches to optimize the upper-level objective function.

### Strengths
Based on the proposed idea, the authors implement two meta-reinforcement learning algorithms: TA-LPG and TA-LPO. It implies that the idea can be applied to various algorithms, although input augmentation depends on the algorithms. Systematic experiments show that TA-LPG and TA-LPO outperform LPG and LPO, respectively.

### Weaknesses
Although the experimental results support the authors' hypothesis that conditioning on the lifetime is helpful in meta-learning, there are no theoretical justifications. In addition, the way to incorporate the lifetime depends on the algorithm, and the experimenter carefully has to design augmentation.

### Questions
1. The function $U_\phi (x_t \mid x_{t+1}, \ldots, x_T)$ is defined in Section 3.2, but it is unclear because $T$ is not explained. Is it the total number of interactions, denoted by $N$ later?  
2. Two variables $n/N$ and $\log N$ are augmented in TA-LPG. Is $n/N = 0$ when $N$ is unbounded? In my understanding, $N$ is determined and fixed before training. Would you explain $N$ and provide an example task? 
3. In TA-LPO, $ \frac{n}{N} x_{r, A}$ is augmented. It is not straightforward to me because it is proportional to $x_{r, A}$, which means linearly-dependent. Would you explain the problem if $n/N$ and $\log N$ are added as the authors did in TA-LPG? 
4. In the final paragraph of Section 4.2, the authors mentioned, "We found this stabilized training and led to higher performance..." Would you explain "this stabilized training" in detail? If it means the rank transformation, it is used in Salimans et al. (2017). Discussing the relation between two equations would be better if the stabilized training means Eq. (12) rather than (11). 
5. I do not fully understand the final paragraph of Section 3.1. In the manuscript, LPG and LPO are selected as the base algorithms. However, the authors mentioned as follows: In our work, we focus on instances of meta-RL that parameterize surrogate loss functions with $\phi$ and apply gradient-based updates to $\pi_\theta$ (Houthooft et al., 2018; Kirsch et al., 2019; Bechtle et al., 2021). Does it mean that three algorithms are implemented somewhere?
6. Equation (10): Is $x_{r, A}$ a typo of $x_{p, A}$? 
Please update the reference Kirsch et al. (2019) to Kirsch et al. (2020). Please see https://openreview.net/forum?id=S1evHerYPr

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
