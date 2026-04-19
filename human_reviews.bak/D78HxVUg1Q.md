# Robustness to Multi-Modal Environment Uncertainty in MARL using Curriculum Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 1

## Abstract
Multi-agent reinforcement learning (MARL) plays a pivotal role in tackling real-world challenges. However, the seamless transition of trained policies from simulations to real-world requires it to be robust to various environmental uncertainties. Existing works focus on finding Nash Equilibrium or the optimal policy under uncertainty in one environment variable (i.e. action, state or reward). This is because a multi-agent system itself is highly complex and unstationary. However, in real-world situation uncertainty can occur in multiple environment variables simultaneously. This work is the first to formulate the generalised problem of robustness to multi-modal environment uncertainty in MARL. To this end, we propose a general robust training approach for multi-modal uncertainty based on curriculum learning techniques. We handle two distinct environmental uncertainty simultaneously and present extensive results across both cooperative and competitive MARL environments, demonstrating that our approach achieves state-of-the-art levels of robustness.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies multi-modal environment uncertainty in Multi-agent Reinforcement Learning (MARL). The authors introduce a formulation for addressing multi-modal uncertainty and design a robust training method based on curriculum learning to manage two distinct environmental uncertainties. Experimental results are provided across various MARL settings. The paper's contributions include a theoretical formalization of the optimal policy problem under multi-modal uncertainty, the design of an effective curriculum learning strategy for this problem, and experimental validation for its method.

### Strengths
1. The paper proposed a method to apply curriculum learning (CL) to address multi-modal environment uncertainty in multi-agent reinforcement learning (MARL), that is an area not explored extensively in prior research.
2. The paper is written with explicit attention to detail. Each section, from experiments to conclusion, clearly elaborates on its respective topic.

### Weaknesses
1. While the methodology and results are well-detailed, it would be better to include a brief introduction to curriculum learning and its traditional applications for readers unfamiliar with the concept. For example, in 5.1 EFFICIENT LOOKAHEAD CL, it’s hard to understand what algorithm 1 is doing, especially without adequate context of explanations. The same problem in 5.2 EFFICIENT LOOKAHEAD CL FOR MULTIPLE UNCERTAINTIES, algorithms 1a and 1b. 

2. The problem formulation is too straightforward that is a simple combination of the robust MARL problem formulation from the existing literature, thus it lacks a unique perspective or innovative twist that could differentiate this approach from previous works.

3. When careful review of the "related work" section in the paper, I noticed that the author's coverage of previous research is not comprehensive. More critically, there seems to be a misreference. For instance, in the proof of existence of NE policy the author refers to Han et al. (2022), but from the context, it appears that He et al. (2023) should have been cited. Such oversight can not only confuse readers but also potentially mislead other researchers. 

4. After a thorough review of the paper, I observed that the author's reliance on the theories from He et al. (2023) and Kardes ̧ et al. (2011b) to demonstrate the existence of Nash equilibrium within their problem formulation appears to be flawed. The central issue is that both of these references do not account for multi-modal uncertainty. To be more precise, neither of them consider the three distinct types of uncertainties. Therefore, using these theories as foundational proof in the current context may lead to incorrect or incomplete conclusions. 

5. In reviewing the experimental section, I found several aspects unclear, which I believe need further elaboration for the reader's comprehensive understanding. Firstly, in section 6.1 titled "ROBUSTNESS UNDER UNCERTAINTY IN A SINGLE PARAMETER," the exact baseline algorithms the author is comparing with their own are not explicitly mentioned. It raises the question, are different baseline algorithms used under different uncertainties? Even if a particular baseline algorithm only considers a single type of uncertainty, isn't it worthwhile to compare this baseline with the proposed method under other uncertainties? Furthermore, there are several RL algorithms that consider action uncertainty. Shouldn't the author consider the multi-agent versions of these algorithms as potential baselines? Regarding state uncertainty, while Han et al. (2022) might not offer a comparison for various uncertainty values, there are other MARL algorithms that do. Examples include "A Robust Mean-Field Actor-Critic Reinforcement Learning Against Adversarial Perturbations on Agent States" and "Robust multi-agent reinforcement learning with state uncertainty." It would be beneficial for the paper's completeness and comparative analysis if these methods were included in the evaluations.

### Questions
Please refer to Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper addresses multi-modal robustness in multi-agent reinforcement learning (MARL). While most prior work on robustness only focused on a single aspect, e.g., adversarial actions, observations, rewards, or transitions, the paper attempts to connect these aspects under a unified perspective. Therefore, the paper first proposes a Markov game that explicitly considers uncertainty sets w.r.t. actions, observations, rewards, or transitions and formulates a maximin value function and policy to characterize a robust Nash equilibrium w.r.t. to these uncertainty sets by assuming that they are adversarial. As a solution, the paper proposes a simple curriculum learning scheme that gradually increases the variance in two uncertainty sets w.r.t. to some convergence parameter. The approach is tested on three multi-particle environments.

### Strengths
The paper addresses a problem that is relevant to the MARL community.

### Weaknesses
**Novelty**

The paper assumes that all agent observations are Markovian, i.e. fully observable, since the policies do not require any history. Thus, the scope of the theoretical concepts is limited. As far as I understood, the problem formulation is an instantiation of the zero-sum Markov game framework of Littman 1994, where the agents represent player 1 and the uncertainty sets represent player 2. The value function and policy definition in Section 4.2 are therefore not novel (this is not a major problem of the paper - but also no major contribution).

Furthermore, I do not understand the exact motivation for highlighting these uncertainties in a separate setting as more general settings like Dec-POMDPs also define uncertainties w.r.t. observations, actions, and other agents [1] and are already known to be difficult to solve, i.e., NEXP-complete. Thus, it does not surprise me that coping with several uncertainty factors simultaneously is a hard problem as the branching factor simply increases in that case. On the other hand, there exists a lot of state-of-the-art work in MARL that addresses Dec-POMDPs like QMIX [2], QPLEX [3], MAPPO [4], etc. However, there is neither a discussion nor an evaluation that includes these works.

The curriculum learning scheme is simply borrowed from prior work and, therefore, not particularly novel either.

**Soundness**

Since the paper builds upon existing frameworks, e.g., the Markov game of [1], the assumptions seem to be sound.

However, I am confused by some contradicting statements regarding the assumptions:
- In Section 4.2: *“Ideally, we would like to make our model robust to all four model uncertainties”* - In Section 3.2, epistemic uncertainty is defined as “model uncertainty”. However, according to the paper, aleatoric uncertainty is being focused on.
- According to Section 4, it is stated that the problem consists of 4 uncertainty aspects. However, in Section 5.1, it is stated that *“We do not have transition dynamics uncertainty”*. What am I missing here? Is the focus on **2 out of 3** uncertainty aspects or on **2 out of 4**?

The curriculum learning approach just increases uncertainty parameters depending on their convergence. However, I do not know, how convergence is actually measured. To me, it seems like a multi-objective optimization problem, where the uncertainty parameters need to be balanced, which would confirm, why simultaneous optimization of all uncertainty aspects is difficult (please, correct me if I am wrong). However, to maintain an adequate balance, the curriculum may need a mechanism to “step back” if one of the aspects dominates and training gets biased towards a single aspect (which the paper actually claims to avoid).

**Clarity**

Despite having knowledge in the field, I had difficulties in fully understanding the paper. There are many grammar mistakes (where subjects or articles are missing) and unclear or ambiguous expressions that could be interpreted in different ways. For example:
- *“Though there has been some work … but they have been studied individually”* - makes only little sense to me
- *“uncertain parameter”* - this is ambiguous. Are the parameters (of the model or environment) uncertain or are these parameters specifying some uncertainty aspect?
- *“It has surpassed baseline”* - Which baseline (algorithm)?
- *“complexity of finding Nash equilibrium and optimal Bellman equation”* - The optimal Bellman equation is given and does not need to be found. However, optimal solutions, i.e., policies that satisfy the Bellman equation, need to be found.
- *“thus requiring the need”* - either *“requiring <something>”* or *“there is a need of <something>”*

**Significance**

Conceptual significance is limited due to the restriction to just two uncertainty aspects out of the four (or three?) mentioned above.

In the experiment section, essential details are missing, like the algorithm for the base method, i.e., is it a centralized learning algorithm like MAPPO or just independent learning, the architecture, and common hyperparameters. After reading the paper and appendix, I do not feel sufficiently confident to understand the setup enough in order to reproduce the results. From an external perspective, the results could just mean anything due to the lack of information.

Furthermore, there is no comparison with other robust MARL approaches. There is only a “Baseline”, which I do not know further (no algorithm, no hyperparameters, etc.). Thus, I cannot confirm the significance of the results. As the paper claims to address a weakness of prior robust MARL, there should be at least a direct comparison with the respective approaches to confirm the improvement over the single-modal state-of-the-art.

**Minor Comments**
- Sometimes sets are written with \mathcal{} and sometimes they are not, for example $S$ and $\mathcal{S}$ in Section 3.1
- In Equation 2, the reward is assigned to a normal distribution. However, if the reward is stochastic, it should be *sampled* from the Normal distribution instead.
- Equation 4 only applies to domains with continuous actions. However, continuous action spaces are not explicitly assumed in the text before.
- *“​​multi-agent static games”* - *“multi-agent **stochastic** games”?*
- There is a misuse in notation, e.g., in the value function the policy of agent $i$ conditions on the joint state $s_t$ (it should condition on its individual state $s^i_t$), while the joint policies of all other agents $-i$ also condition on the joint state $s_t$.
- Last sentence on page 6: *“We show”* is something missing here?
- The plots in Figure 2 are very cluttered due to many lines. I suggest just showing the respective best and worst settings as well as one setting in between to improve readability without hurting the actual takeaway message.


**References**

[1] F. Oliehoek et al., "A Concise Introduction to Decentralized POMDPs", 2016

[2] T. Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", ICML 2018

[3] J. Wang et al., "Qplex: Duplex Dueling Multi-Agent Q-Learning", ICLR 2021

[4] C. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022 (Benchmark Track)

### Questions
1. I do not understand the motivation behind Figure 1. It merely shows the consequences of different uncertainty aspects, which are not surprising when regarding them in isolation. How does it motivate multi-modal solutions, e.g., how does it approach the problem in contrast to single-modal solutions?

2. *“Note that the range of value of observations and actions is quite small as compared to that of reward. Thus, the magnitude of robustness is different for different uncertainty parameters.”* - How can I confirm this? There are no ranges provided, and equations 2, 3, and 4 restrict values in the same manner, i.e., through the truncated Normal distribution.

3. How is convergence measured in Algorithm 1?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Recent works in multi-agent learning focus on finding the Nash Equilibrium or optimal policy with the objective of achieving robustness towards a single environment variable. The paper aims to tackle uncertainty in multiple environment variables. Authors present a curriculum learning approach utilizing efficient lookahead for multiple variable uncertainty in states, actions, rewards and transition dynamics. The paper characterizes a robust Nash Equilibrium as part of a robust markov game. Curriculum learning is then employed using efficient lookahead with multiple parameters. The model is trained to handle two uncertainty variables concurrently. Experiments are carried out in cooperative and mixed cooperative-competitive tasks.

### Strengths
* The paper is well positioned within the multi-agent learning community.
* Authors address an important and challenging problem.

### Weaknesses
* **Statement of Claims:** My main concern is the statement of claims. The paper significantly overstates some of its claims by stating that it is "the first to formulate the generalised problem of robustness to multi-modal environment uncertainty in MARL", "first work to handle multi-modal uncertainty in MARL" and "first work to handle action uncertainty in MARL". Note that there is a breadth of literature which addresses environment uncertainty in multiple parameters using surprise minimization [1], variational exploration [2], intrinsic motivation [3], adversarial learning [4] and generative models [5]. Additionally, a wide variety of established MARL algorithms build on uncertainty estimation from single-agent RL methods.
* **Writing and Presentation:** In general, the paper is not well-written. Statements and explantions provided by authors are vague and not well structured. For instance, Section 1 does not motivate the problem of uncertainty estimation or multi-agent learning. Sections 3 and 4 overly explain the problem formulation with inconsistent notation and complicated vocabulary. Theorem 1 does not have a formal statement. Finally, Section 5 does not provide any technical detail or explanation of the proposed method. The paper requires significant attention from a presentation standpoint.
* **Approach Description:** Authors formulate the multi-modal learning problem in a curriculum learning setting with efficient lookahead. However, the paper does not provide any technical explanation or intuition of the method. Sections 5.1 and 5.2 provide the general idea of learning two parameters using a single model but do not describe the method. For instance, authors could elaborate on efficient lookahead, SkipAhead and TrainTillSuccess. Authors could also provide details on training procedure and reasoning behind their design choices. In its current form, the paper does not add technical contribution to the problem setting.
* **Experiment Setup:** While the experiment section presents the high-level protocols, it does not explain the curriculum learning task descriptions or training and evaluation setup. The section does not highlight the complexity of tasks and what authors wish to observe from their empirical evaluation. Furthermore, results presented by authors are lacking intuition and reasoning behind their explanations.
* **Results and Baselines:** Authors claim state of the art performance but the work itself does not consider any baselines from existing literature. Results only demonstrate the performance of agents across different hyperparameter configurations and variation in rewards. This does not account for assessing uncertainty from environment variables or improved robustness. The paper could consider baselines which evaluate robustness using exploration[6], intrinsic motivation[1,3,5] and offline datasets[7,8]. Furthermore, the paper could evaluate the importance and complexity of a curriculum using ablation studies.


### Minors
* unstationary $\rightarrow$ non-stationary
* fining $\rightarrow$ finding
* situation $\rightarrow$ situations
* $\pi^* = {\pi_1^*, \pi_2^*, ..., \pi_N^*} $. Do you mean $\pi_*$? 
* Can you please formally state Theorem 1.


[1]. Berseth et. al., "SMiRL: Surprise Minimizing Reinforcement Learning in Unstable Environments", ICLR 2021.  
[2]. Mahajan et. al., "MAVEN: Multi-Agent Variational Exploration", NeurIPS 2019.  
[3]. Jaques et. al., "Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning", ICML 2018.  
[4]. Fickinger et. al., "Explore and Control with Adversarial Surprise", arxiv 2021.  
[5]. Suri et. al., "Surprise Minimizing Multi-Agent Learning with Energy-based Models", NeurIPS 2022.  
[6]. Mahajan et. al., "Generalization in Cooperative Multi-Agent Systems", arxiv 2022.  
[7]. Pan et. al., "Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification", ICML 2022.  
[8]. Wang et. al., "Offline Multi-Agent Reinforcement Learning with Coupled Value Factorization", AAMAS 2023.

### Questions
* Can you please explain your reasoning behind the claims? How is the proposed approach the first to handle uncertainty in MARL?
* What is the reason behind using a curriculum? What is efficient lookahead? How do SkipAhead and TrainTillSuccess work? Can you please technically explain the proposed approach and its intuition?
* How many tasks are a part of the currciculum? What is the complexity of these tasks? What is the training and evaluation setup used during experiments?
* Can you please consider baselines from existing literature? How does the proposed method achieve state of the art performance? How was robustness and uncertainty measured using rewards/success rates? What is the importance of a curriculum in the learning process?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a general robust training method for multi-modal uncertainty based on curriculum learning techniques. They handle two distinct environmental uncertainty simultaneously and present extensive results across both cooperative and competitive MARL environments.

### Strengths
They focus on developing robustness to two uncertain parameter at a time and introduce the curriculum learning technique to improve sample efficiency.

### Weaknesses
1.The article is poorly written and many variables are not defined in details. For example, what is the meaning of TrainTillSuccess in algorithm 1? And what is TrainToSucc and SkipAhead？

2.The core contribution of this paper is just a simple application of curriculum learning in robust RL with little innovation. And there is no detailed analysis of the sensitive parameters of curriculum learning in this paper, such as the effect of different settings of \delta(\labmda) ?

3.The experimental results are not convincing. Why the results are not represented using a curve of mean-standard deviation commonly used in reinforcement learning, such as fig2?

4.I didn't understand what the essential difference is between the solutions for single uncertainty and double unceritainties. What is the essential difficulty caused by the two unceritainties?

### Questions
I have already stated the problem in weakness

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
