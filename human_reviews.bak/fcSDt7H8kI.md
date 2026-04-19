# Boosting Reinforcement Learning with Extremum Experiences

- Decision: Reject
- Scores: 5, 5, 3, 3

## Abstract
Reinforcement learning research has achieved high acceleration in its progress starting from the initial installation of deep neural networks as function approximators to learn policies that make sequential decisions in high-dimensional state representation MDPs. While several consecutive barriers have been broken in deep reinforcement learning research (i.e. learning from high-dimensional states, learning purely via self-play), several others still stand. On this line, in our paper we focus on experience collection in high-dimensional complex MDPs and we propose a unique technique based on experiences obtained through extremum actions. Our method provides theoretical basis for efficient experience collection, and further comes with zero additional computational cost while leading to significant sample efficiency gains in deep reinforcement learning training. We conduct extensive experiments in the Arcade Learning Environment with high-dimensional state representation MDPs. We demonstrate that our technique improves the human normalized median scores of Arcade Learning Environment by 248% in the low-data regime.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a new algorithm, MaxMin TD Learning, by modifying $\epsilon$-greedy exploration in DQN. Specifically, with probability $\epsilon$, the argmin action is selected given the state-action values. Theoretically, this leads to higher temporal difference error under certain assumptions. In practice, the proposed algorithm is shown to achieve higher sample efficiency than DQN with $\epsilon$-greedy exploration.

### Strengths
As far as I know, the presented idea is novel and easy to implement. Generally, the paper is easy to follow. The advantage of the proposed algorithm is supported by both theories and experiments. All algorithms are tested in 100K Atari games.

### Weaknesses
The major weaknesses are insufficient experiments, a gap between theory and experiments, and a lack of explanation.

- How large is $\mathcal{D}(s)$, $\delta$, and $\eta$ in practice? Is $\mathcal{D}(s) − 2\delta − \eta$ positive or negative in practice?
- In Section 4, a fixed step size is used. How does the performance of the algorithms vary with different step sizes?
- In Section 4, $\epsilon$ is chosen from $[0.15, 0.25]$. In practice, a smaller $\epsilon$ is usually used. How is the performance of the algorithms with smaller $\epsilon$, such as $\epsilon \in [0.01, 0.05]$? How sensitive is MaxMin TD learning to $\epsilon$ compared to DQN?
- In Figure 4, not all tasks (e.g. Amidar, Bowling, BankHeist, and StarGunner) are trained with 200M frames although it is claimed so.
- In Section 3, it is claimed that, in the early phase of the training, in expectation over the random initialization $\theta \sim \Theta$, the TD error is higher when taking the minimum value action than that of a random action. However, this contradicts the experimental results shown in Figure 3, especially Figure 3(a).
- Lack of explanation: Why would a higher TD error help exploration and speed up training in general? I don't see a clear connection between them. I believe that it is very important to explain the logic behind.

### Questions
- In Definition 3.2 & 3.5: What is $\hat{a}(s,\theta)$?
- In Proposition 3.4, $a_t \sim \mathcal{U},(\mathcal{A})$: typo.
- In Section 4, it is mentioned that the maximum achievable reward in 100 steps is 10. However, the learning curve in Figure 1 (b) is above 10 in the end. How do you get the data used in Figure 1?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The work looks at improving sample complexity of deep reinforcement learning (RL) algorithms from the lens of experience collection. A new method based on minimizing state-action value function to increase information gain is proposed. Modifying episilon-greedy, the algorithm leads to more novel experiences by taking actions with the smallest Q-value. Experimentally, the proposed method demonstrates significant improvement in sample complexity in the Arcade Learning Environment, without additional learning parameters.

### Strengths
1. The proposed method is well motivated and empirically shows significant improvements in sample efficiency.
2. The paper, in general, is well-structured.

### Weaknesses
1. The first few definitions is unclear and unintuitive.
2. The definition of $\hat{a}$ is confusing in Definition 3.2
3. There needs to be a related work section. It is unclear how this approach position among existing works.
4. Figure 1 is too small
5. Missing standard deviation in Figure 3
6. [Minor] the repetition of the questions in conclusion seems like a waste of space to me.

### Questions
1. Based on Figure 4, Max-Min TD seems to have higher variance, why is that?
2. Would the method be as effective in sparse-reward setting given that it ties directly to the size of TD?

### Soundness
2 fair

### Presentation
2 fair

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
The paper proposes an exploration method based on minimizing the state-action value function. The method is incorporated into temporal difference based on Q-learning with function approximation. Experiments are conducted using a toy chain MDP and several Arcade Learning Environments. The results are compared to the $\epsilon$-greedy baseline.

### Strengths
- The paper addresses the important problem of exploration in reinforcement learning.
- It attempts to provide theoretical justification and analyzes empirical results using toy examples and standard benchmark tasks.

### Weaknesses
- Several claims in the paper require further evidence.
- The empirical evaluation lacks detail.
- Details are lacking in addressing the research questions and contributions proposed in the introduction.

### Questions
An assumption of the proposed method is that the Q-function, in the initial phase of training, would assign similar values to similar states.
“...early in training the Q-function, on average, will assign approximately similar values to states that are similar…”, “....when the Q-function on average assigns similar maximum values to consecutive states”. 
It is unclear how this assumption holds. If a random Q-function processes different consecutive states, then the output value might be arbitrary and not necessarily dependent on the input, even for slightly varied states. Thus, the output could be any random number, not necessarily a similar value.

The text in the plots of Figure 1 is too small and difficult to read. What do each of the plots represent? Are they for different $\epsilon$ values? Which plot corresponds to which value? Also, how does a change in ε affect the results of the proposed MaxMin TD learning?

It is mentioned that "All of the results in the paper are reported with the standard error of the mean”. However, Figure 2 shows the results for the median on the y-axis. Could you clarify what this means?

The claim “.....thus creates novel transitions in exploration with more unique experience collection.” is made, but no evidence is presented in the paper. The results are only compared based on reward performance. How can we be certain that the change in results is due to this particular claim?

In Table 1, the Human Normalized Median is 0.0927 for MaxMin TD and 0.0377 for $\epsilon$-greedy. If 1 is the highest achievable score, then these numbers appear quite low. Do both algorithms fail to learn anything useful? In that case, stating a 248% improvement seems misleading.

What is the QRDQN algorithm baseline in Figure 5? It is not discussed in the paper. What is the difference between $\epsilon$-greedy in Figure 1 and Figure 5? While it is briefly mentioned in the footnote of the supplementary materials, detailed references are not presented.

It is mentioned in the introduction as a contribution that the proposed method "...reaches approximately the same performance level as model-based deep reinforcement learning algorithms," suggesting that the proposed method performs better than model-based. However, no model-based baseline is presented in the experiments, nor is it explained in the text.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new exploration strategy in reinforcement learning which focuses on taking extremum actions with minimum Q-values. Theoretically, the authors attempt to prove that the TD computed by taking the action with minimum Q-value (denoted as $a_{min}$) is above average (i.e., expected Q-value for a uniform policy) by an amount approximately equal to the disadvantage gap, which is referred to the expected Q-value for a uniform policy minus the Q-value for $a_{min}$. The proposed MaxMin TD Learning policy follows the $\epsilon$-greedy style, where the proposed algorithm takes $a_{min}$ instead of uniform random action for exploration.

### Strengths
- This paper proposes an interesting idea of improving the exploration efficiency by taking extremum action, which refers to the action with minimum Q-value. 

- The method comes with a nice theoretical motivation, where the authors show the proof of the relationship between TD error inferred by taking $a_{min}$ compared to that for a uniform policy, showing that taking $a_{min}$ as the extremum action more frequently could lead to novel transitions that accelerate learning.  

- The proposed method is very general and simple to apply, leading to no additional computational overhead compared to vanilla $\epsilon$-greedy.

- The authors show comparison results with UCB and $\epsilon$-greedy on a toy chain MDP domain and large-scale experimental results by comparing with NoisyNets and $\epsilon$-greedy on Atari 100K.

### Weaknesses
- The theoretical contribution of this paper relies on several strong assumptions: (1) expected rewards for a uniform random policy and the $a_{min}$ is $\eta$-uniformed; (2) the Q-value for consequent states $s$ and $s'$ has little difference ($\delta$-smooth); (3) the initialized Q-function results in a policy that is close to uniform random. The main theoretical conclusion that the TD achieved by $a_{min}$ is above-average by an amount approximately equal to the disadvantage gap ($D(s)$) would be wrong if $\delta$ and $\eta$ are not close to 0, because the gap actually equals to $D(s) - 2\delta - \eta$. Also, based on my experience, for value functions parameterized by deep neural nets, the initial policy distribution characterized by the initialized Q-functions is often fairly biased from a uniform distribution. In practice, the value $\epsilon$ would need to gradually decay during the initial phase of training, which means that in practice the theoretically derived conclusion will quickly be invalid in a real training regime. 

- In the proposed Algorithm 1, the RL agent always takes $a_{min}$ for exploration action, and no action with intermediate Q-values could be taken for exploration. Unless $a_{min}$ would keep changing among the action set throughout the training, I think the proposed method would easily result in sub-optimal policy compared to $\epsilon$-greedy due to the limited exploration strategy.  For exploration, I'm not convinced it would be generally beneficial to always take a_{min}, and in practice, a_{min} is not guaranteed to always lead to the largest TD error. Though the authors attempt to claim their proposed method is better than UCB through the simple motivating task on chain MDP, I'm still not convinced that the MaxMin TD Learning could generally beat the strong UCB policy variants when tackling challenging RL domains like Atari 2600. 

- I think the empirical results on the motivating example are flawed. The authors show the learning curves of MaxMin TD, $\epsilon$-greedy, and UCB, but it is unclear how the exploration policies for the two baselines are specified. For $\epsilon$-greedy, it seems that the authors fix the $\epsilon$ value, otherwise, I expect a well-tuned $\epsilon$-greedy with $\epsilon$ decay properly defined will succeed in the simple chain MDP. It is suspicious why $\epsilon$-greedy converges to a sub-optimal average return. I also wonder if MaxMin TD learning can learn properly without $\epsilon$ decay. It is unfair if the authors allow MaxMin to employ a decayed $\epsilon$, while keeping that for $\epsilon$ or UCB fixed. Please specify the details of each policy.

- For the large-scale Atari 100K evaluation, the baselines are insufficient. As the algorithm focuses on exploration policy, at least it should compare with the UCB-variant of baselines. Also, neither noisy networks nor $\epsilon$-greedy is the SOTA method on Atari 100K. The authors should employ stronger baselines. 

- The learning curves for the noisy net are missing in the Atari 100K figures (e.g., Fig 2 and Fig 4). They should be added at least to each game's learning curve.  

- It would be more convincing if the authors could evaluate MaxMin TD Learning on a more inclusive range of tasks, e.g., Atari 2600 and mujoco, where the method could work on top of both value-based and policy-based algorithms to verify its generality.

### Questions
Please refer to the WEAKNESSES section.

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair
