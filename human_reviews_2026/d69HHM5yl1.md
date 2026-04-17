# NBSP: A Neuron-Level Framework for Balancing Stability and Plasticity in Deep Reinforcement Learning

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
In contrast to the human ability to continuously acquire knowledge, agents struggle with the stability-plasticity dilemma in deep reinforcement learning (DRL), which refers to the trade-off between retaining existing skills (stability) and learning new knowledge (plasticity). Current methods focus on balancing these two aspects at the network level, lacking sufficient differentiation and fine-grained control of individual neurons. To overcome this limitation, we propose Neuron-level Balance between Stability and Plasticity (NBSP) method, by taking inspiration from the observation that specific neurons are strongly relevant to task-relevant skills. Specifically, NBSP first (1) defines and identifies RL skill neurons that are crucial for knowledge retention through a goal-oriented method, and then (2) introduces a framework by employing adaptive gradient masking and experience replay techniques targeting these neurons to preserve the encoded existing skills while enabling adaptation to new tasks. Numerous experimental results on the Meta-World, Atari, and DMC benchmarks demonstrate that NBSP significantly outperforms existing approaches in balancing stability and plasticity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for helping with the stability-plasticity tradeoff on a neuron-level basis. It proposes a metric to identify the neurons that are crucial for performance, and uses gradient masking and replay on top to preserve the already learned patterns. 
The paper performs experiments on Metaworld, Atari, and DMC benchmarks, and compares their method with multiple methods proposed in continual learning, across multiple metrics, and shows that their approach (NBSP) can outperform without the addition of parameters or using complex networks.

### Strengths
1. The paper is well-written, and there is a natural flow for related work and methodology.

2. The paper is well-motivated, and the proposed methodology does in fact avoid complex NN designs and the use of large networks, which are crucial benefits in deep RL.

3. The paper leverages the use of multiple evaluation metrics for plasticity, stability, and overall performance for assessing their method and comparing it to others. Ablation studies explore the components of the proposed method thoroughly, and also the use of the algorithm in other agents than SAC is included.

### Weaknesses
1. The score does not capture what the paper intends to.  $1 - R_{\text{over}}(\mathcal{N})$ does not only include neurons whose activity hold a negative correlation with performance, as stated and intended in line 233,  but also other cases: 
$$
1 - R_{\text{over}}(N) 
= 1 - \frac{\sum_{t=1}^{T} 1\Big[ 1[a(N, t) > \bar{a}(N)] = 1[q(t) > \bar{q}] \Big]}{T}
$$

$$
= \frac{T - \sum_{t=1}^{T} 1\Big[ 1[a(N, t) > \bar{a}(N)] = 1[q(t) > \bar{q}] \Big]}{T}
$$

$$
= \frac{\sum_{t=1}^{T} \Big( 1 - 1\big[ 1[a(N, t) > \bar{a}(N)] = 1[q(t) > \bar{q}] \big] \Big)}{T}
$$

where the numerator is:
$$
\begin{cases}
0, & \text{if } a(N,t) > \bar{a}(N) \text{ and } q(t) > \bar{q},\\
1, & \text{otherwise.}
\end{cases}
$$
and "otherwise" includes all the other cases, i.e., (a < ā, q > q̄), (a > ā, q < q̄), (a < ā, q < q̄)
whereas, as stated in the paper, it was intended to account for only  (a < ā, q > q̄), meaning " when the activation of a neuron falls below its average activation, the agent performs well."


2. Three seeds are not enough to make strong claims about the performance of the method compared to the others. Can you take a few of the methods in one of the settings and run more seeds for them to strengthen the claims?

3. Although the paper does a great job of exploring the effect of the NSBP hyperparameters on the model performance, other methods’ hyperparameters are underexplored. How are the other baselines' hyperparameters tuned? Are they using their default hyperparameters? If so, it might not be fair to use those hypers as they were found for the settings they were experimented on. In particular, is COTASP tuned, since it consistently underperforms in all the metaworld benchmarks? What is the reason for its performance collapse?

4. The paper emphasizes the distinction between CRL and DRL. However, the experiments are mostly done in settings designed for CRL ( non-stationary or task change in the environment). Can you elaborate on the reason for this emphasis and its relation to your method?

### Questions
1. The average goal metric works for settings where there is a specific goal or an episode. How does your method and this metric extend to continuing settings where there are no episodes, or a clear/binary notion of success?

2. How do the experiment results change after fixing the score function?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a neural-level method to overcome catastrophic forgetting issues in deep reinforcement learning (DRL). They propose to both mask the gradient of task-specific neurons and train the neural network with multi-task experience replay. Experiments show improved results compared to baselines on a DRL benchmark.

The paper is well-written, the method is novel and the preliminary results are promising. My main concern is about the experimental protocol. First, the authors compare their approach to baselines of the field only in the Meta-World benchmarks. Second, there is no clear reasons for not testing the method on supervised benchmarks that are more widely used in continual learning, such as adapted versions of cifar100 and imagenet. I suspect that the "Experience replay only" version will achieve much better performance in supervised learning. In addition, the authors do not provide a clear analysis of why Experience replay and neuron masking are complementary; the bad results of "Experience replay only" require explanations. Third, it is unclear how the method scales when considering more than four tasks. 

The authors should address these issues before resubmission.

Minor comments:
- "reasonably broad range of values (from 0.15 to 0.3)", it is unclear why the authors mention this specific range, given the shape of the curve in Figure 4.
- It is unclear why increasing the number of masked neurons "compromises their learning capacity and causes the true RL skill neurons to adjust their activations to accommodate new tasks, ultimately reducing stability".

### Strengths
The paper is well-written, the approach is novel and relevant

### Weaknesses
Experiments are insufficient

### Questions
Please, see above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces NBSP (Neuron-level Balance between Stability and Plasticity), a framework that tackles the stability–plasticity dilemma in deep reinforcement learning (DRL). Unlike prior methods that regulate stability and plasticity at the network or parameter level, NBSP operates at the neuron level. It identifies RL skill neurons, neurons whose activations correlate strongly with task goals (e.g., success rate or return), and protects them during new task learning. NBSP combines Goal-oriented neuron identification to detect neurons critical for knowledge retention; and Adaptive gradient masking and experience replay, selectively applied to those neurons.

Experiments on Meta-World, Atari, and DMC benchmarks show that NBSP improves the trade-off between stability and plasticity, achieving higher Average Success Rate (ASR), lower Forgetting Measure (FM), and higher Forward Transfer (FWT) than nine competitive baselines (for example EWC, NPC, CoTASP, NE, UPGD). Ablation studies confirm the complementary roles of gradient masking and experience replay, and demonstrate that the goal-oriented neuron identification strategy is crucial to performance.

### Strengths
- The neuron-level approach to balancing stability and plasticity is well-motivated. Defining “skill neurons” through goal-oriented correlation is creative, interpretable, and biologically inspired.
- The method is conceptually simple yet effective, adding minimal complexity to standard SAC/PPO agents.
- Strong empirical evidence supports the claims. Meta-World, Atari, and DMC results consistently outperform baselines.
- Ablations and hyperparameter analyses (e.g., actor vs. critic masking) are thoughtfully designed.
- The paper is well-structured. Motivation, methodology, and evaluation flow logically, with clear notation and metric definitions (ASR, FM, FWT).
- Addresses a fundamental and persistent challenge in continual deep RL.
- NBSP’s interpretability and light implementation could make it broadly useful for future continual-learning research, including extensions to supervised or self-supervised setups.

### Weaknesses
- Experiments use only 3 seeds; more repetitions or confidence intervals would increase result reliability.
- The method’s dependence on the top-m% neuron threshold (m = 0.2), mask coefficient (α = 0.2), and averaging window size is not analyzed. A sensitivity study would strengthen robustness claims.
- The paper lacks variance or gradient-norm analyses explaining why masking high-score neurons stabilizes training.
- Atari and DMC results are summarized in tables but lack detailed learning curves or per-task analysis.
- The algorithmic description omits specifics such as how masks interact with target networks and the replay schedule.

### Questions
- How sensitive is NBSP to the hyperparameters?
- Have you measured gradient-norm variance or activation statistics to confirm reduced interference?
- Are both actor and critic masked at every step, or alternately? How does masking interact with target networks?
- What is the computational and memory overhead (FLOPs, wall-time, replay buffer size) compared to vanilla SAC?
- Have you examined the behavior of anti-correlated (“negative-score”) neurons?
- Would NBSP generalize to supervised continual learning or offline RL where reward signals differ?
- The paper excludes the final layer from neuron scoring. Why? Have you tested whether masking output neurons (for example, in the critic’s value head) hurts performance or stability?
-What failure modes did you observe, for example, situations where neuron correlation misidentifies unimportant units, or where masking accumulates and stalls learning?

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
3

### Summary
The authors propose the NBSP framework to solve the stability-plasticity dilemma in DRL (Deep Reinforcement Learning). The method uses a goal-oriented strategy to identify "RL skill neurons" and combines adaptive gradient masking with experience replay to balance learning new skills while retaining old ones. Experiments on multiple benchmarks demonstrate that NBSP outperforms existing methods in multiple metrics such as ASR.

### Strengths
1. This paper introduces a neuron-level approach to address the stability-plasticity dilemma in DRL, proposing the concept of "RL skill neurons." While similar concepts have been discussed in other fields, their introduction to DRL is (to my knowledge) a novel contribution.
2. The experimental design is sound and thorough, featuring multiple benchmarks, diverse metrics, and a wide variety of baseline comparisons. Furthermore, the inclusion of extensive ablation studies to validate component effectiveness and explore the mechanisms of RL skill neurons significantly strengthens the paper's conclusions.
3. The paper is well-structured, clearly written, and utilizes well-designed figures and tables to effectively communicate the research outcomes.

### Weaknesses
1. The experimental setup is relatively simple:
    - Regarding the length of continual learning sequences, the paper tests on short task chains, focusing on lengths of 2 or 4 (at most), and lacks experiments on longer chains (e.g., over 10 Atari games).
    - Regarding task relatedness, the tasks tested are highly correlated, such as window-close/window-open in Meta-World and Cartpole-swingup/Cartpole-balance in DMC.
    - Finally, the tasks are relatively simple, raising questions about the method's generalizability to more complex environments.
2. The paper introduces a considerable number of hyperparameters:
    - Several hyperparameters are introduced for the RL skill neurons, and the two core techniques (gradient masking and experience replay) also bring their own.
    - **It is worth noting that the authors conducted sensitivity analyses, and the current results suggest the method is not overly hyperparameter-sensitive.** 
    - However, it currently lacks a systematic selection criterion and relies on manual tuning, which raises doubts about its broader applicability to other tasks and algorithms.

### Questions
1. Could the authors provide more detailed implementation details for the baselines? For example, the original NE paper used the TD3 algorithm, but in this paper, it was implemented with SAC. Could the authors clarify how this algorithmic transition was handled to ensure a fair comparison?
2. While gradient masking is shown to significantly improve success rates, it resembles reducing the learning rate on certain neurons. Could the authors include experiments that reduce the learning rate across the entire network to demonstrate the necessity of selectively lowering it only on RL skill neurons?
3. The authors emphasize balancing stability and plasticity in DRL. However, both gradient masking and experience replay tend to enhance stability. Does this suggest that, for vanilla DRL algorithms, improving stability is more crucial for metrics like ASR? Alternatively, could NBSP be combined with other techniques that enhance plasticity, such as plasticity injection?

### Soundness
4

### Presentation
3

### Contribution
4
