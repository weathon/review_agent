# Annealed Implicit Q-learning in Online Reinforcement Learning

- Decision: Reject
- Scores: 6, 6, 3, 3

## Abstract
In continuous action online reinforcement learning, actor-critic methods are predominantly used.
However, compared to Q-learning-based discrete action algorithms that model the optimal Q-value, continuous action algorithms that model the Q-value for the current policy and perform policy improvement solely through policy updates suffer from low sample efficiency. 
This study investigates whether an algorithm that implicitly estimates the optimal Q-value, typically used in offline RL, is also effective in online RL. It is demonstrated that a loss function aimed at achieving optimality distorts the distribution of Q-values, leading to overestimation bias, and that this distortion and bias increase as learning progresses. To address this issue, we propose a simple algorithm that anneals optimality. Our method significantly outperforms widely used methods such as SAC and TD3 in online DM Control tasks. Additionally, we demonstrate that annealing improves performance and enhances robustness to the hyperparameter related to the optimality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a sample-efficient actor-critic method by modifying the policy evaluation loss in online scenarios. They propose to do so by leveraging the expectile loss shown effective in offline scenarios. Additionally, the authors propose to decrease the learned expectile during training to perform implicit Q-learning at the beginning of the training and transition to a policy evaluation regime at the end of the training.

### Strengths
1. The paper investigates an important topic in RL. 

2. The proposed method is clearly explained and well-motivated.

3. The presented algorithm outperforms relevant baselines in the considered experiments.

### Weaknesses
1. The novelty of the proposed approach is limited as the paper's main contribution lies in applying Implicite Q-Learning to an online setting with a linear schedule for the expectile. Nevertheless, the findings are interesting. They would be worth sharing with the community if a more detailed study would be done. Here are some ideas to increase the depth of the analysis:

     ~a.  The authors should comment on the choice of learning the Q-function directly instead of using a separate value function as done in IQL. This choice should normally harm the performances as the learned expectile is now influenced by the stochasticity of the environment. Evaluating the proposed algorithm in a highly stochastic MDP would help justify this choice.~ 

     ~b. AIQL is only applied to SAC. Applying it to TD3 and CrossQ, for example, would strengthen the authors' claim.~

     c. Investigating the importance of the expectile regression loss compared to the quantile or Huber quantile regression loss would be interesting.

     ~d. Analyzing the influence of the hyperparameter $T$ would be beneficial for understanding its influence on AIQL.~ 

2. Changing the target expectile value during the training makes the loss non-stationary, potentially harming performances. The authors do not comment on this point.

~3. The code is not provided, which limits the reliability of the experiments. Additionally, the baselines’ hyperparameters are not reported.~

### Questions
~1. Line 155, the authors claim that the scenario where all action values are equal has the largest overestimation bias. Can this be justified?~

2. To mitigate the issue presented in Weakness 2, the authors could consider having a Q-network with several heads, where each head is responsible for learning one expectile, similar to Quantile-Regression Deep Q-Network [1]. The head used for training the policy would evolve during training, starting from a low target expectile value to finish with a high target expectile value as proposed by the authors. I would be interested in the authors' comments about this suggestion.

[1] Dabney, Will, et al. "Distributional reinforcement learning with quantile regression." AAAI. 2018.

– Remarks – 

~A. Line 35, a stop between “efficiency” and “This” is missing.~

~B. Line 42, for clarity, I would replace “optimal value” by “maximal value”.~

~C. Line 90, the word “initial” is missing before “probability distribution”.~

~D. Line 120, the value function should not take the action $a$ as input.~

~E. Line 196, some parentheses are close, but they have never been opened. I suggest removing them.~

~F. Line 215, replacing $t$ by $\min(t, T)$ in the right side of the equation would be more accurate.~

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new approach to improve sample efficiency in continuous action online RL by gradually reducing optimality bias, which helps control overestimation in Q-values. The method outperforms SAC and TD3 on DM Control tasks, offering better performance and robustness.

### Strengths
- Tackles a key challenge in online RL: boosting sample efficiency through better value function estimation.
- Presents a clear argument that overestimation and Q-value skewness are major issues, and introduces expectile loss in the critic architecture to help address them in continuous action tasks.
- Shows impressive performance gains compared to popular methods like SAC and TD3.

### Weaknesses
- Section 2.2 (Line 115): Why is it stated that "In IQL, near-optimal values are estimated by using τ close to 1"? Could you clarify this choice?
~~- Sections 4.1 & 4.6: Could you expand your analysis to discuss Exp1 in Section 4.6? Based on the explanation in Section 4.1, it seemed the natural annealing strategy might align with Exp1 in Section 4.6, yet it performed relatively worse than the linear and sigmoid strategies.~~
~~- Section 4.2: In the XQL (Garg et al., 2023), the extension of SAC (X-SAC) showed comparable performance to SAC, even outperforming it in Hopper-Hop. However, the results in Figure 4 don’t seem to align with this finding. Could you provide any reasoning or justification for this discrepancy?~~

### Questions
none

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a new method in online RL, focusing on the setting with continuous action space. The work starts from the advantage of expectile and the observation of overestimation bias in function approximation. Then a new algorithm named Annealed Implicit Q-learning (AIQL) is proposed, which is based on Soft Actor-Critic (SAC) and the tool of expectile inspired by Implicit Q-learning (IQL). Experiments are then to show the comparisons in performances.

### Strengths
> **Clarity**
- Despite some lack of notation explanation, the paper gradually leads readers to the proposed method through observation of gaps and theoretical discussion, making the main method easy to understand.

> **Significance**
- The proposed method is tested in various benchmark environments.

### Weaknesses
> **Originality**
- The proposed method mainly follows the ideas of both IQL and SAC.

> **Quality**
- The claims regarding the overestimation bias are not sufficiently convincing. Equation (1) is derived following the strong assumption that Q-functions are independent from actions, which tends to be an over-simplification of the scenario. In addition, the experiment shown in Figure 2 mainly concerns a single training step, but does not explain why such bias can still accumulate after online exploration (apart from claiming such result by 'if the bias becomes too large').
- In Section 3.3, the advantage of tuning $\tau$ is described, after which a linearly decreasing $\tau(t)$ is proposed. However, there is neither theoretical/empirical observation showcasing how different $\tau$ affects the training in different stages (as a support to the advantage part), nor justification of the choice of linear functions (as a support to the proposal part).

> **Clarity**
- In Section 2.1, $d\_0$ is not clearly defined by just claiming it is 'the probability distribution', which actually seems to be the initial state distribution. And the horizon $T$ is not defined.
- In the review of IQL, the loss function for the parameters of Q-functions misses the dependence on $V\_\psi$.

> **Significance**
- In Section 3.1 the paper emphasizes its advantage by estimating directly the optimal value compared to methods like SAC, while in actual implementation according to Section 4.1, the proposed method is just an expectile-loss version of SAC. More clarifications on distinctions between methods could be helpful.
- In some experimental results in Figure 4, SAC reaches higher average returns earlier than AIQL.

### Questions
- Why is there no comparison with some online version of IQL, if any?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper uses an annealed implicit Q-learning loss in online RL with SAC, to replace the standard MSE loss. The loss is annealed in a way that increases bias in the early stage of training and reduce bias toward the later stage.

### Strengths
- The paper proposes a very simple trick that can potentially improve the sample efficiency of training the Q-function in SAC.
- The experiments show a major improvement in hopper-hop and humanoid-run environments.
- The paper provides a large set of results on various $\tau$ values being annealed or kept fixed.

### Weaknesses
## Lacks justification of expectile loss

While the paper spends a significant portion trying to motivate the $\tau$ schedule, it is still unclear why introduce the implicit Q-learning loss to begin with? What benefit does it offer over the thoeretically guaranteed objective of SAC in online RL?

In other words, it is unclear why AIQL outperforms SAC in the environments considered. Figure 7 shows that the skewness and bias of SAC are the minimum, and lower than AIQL. Then what is the root cause of AIQL outperforming SAC?


## The annealing schedule of optimality is not sufficiently justified
I am still unable to understand why the bias should be kept high in the beginning (and lower in the end)?
Section 3.2 discusses overestimation bias and that too much overestimation bias would lead to failure in learning, which makes sense.
Section 3.3 says "$\tau$ control the trade-off between optimality and bias", but this relationship was never justified. Therefore, the connection between skewness/bias and $\tau$ value is not well established.


## The scheduling of tau depends heavily on the environment
Essentially this paper requires two kinds of hyperparameters:
- the start and end values of $\tau$.
- the total duration of learning (set to 3M steps in this work) to set up the linear schedule.

But how does one select the decay schedule for an environment where the schedule is not known a priori? Even for the humanoid tasks, the convergence is not fully achieved, e.g., humanoid-run should reach max return of ~450 if trained for longer. So, how does one define the annealing schedule in such cases?

I would encourage the authors to demonstrate their idea on more diverse environments apart from DM Control, to fully understand when the expectile loss and the annealing schedule can help or hurt. And also how to set the hyperparameters in general.

## IQL loss on algorithms other than SAC
The paper claims "tested the modeling of optimal Q-value using implicit Q-learning loss in continuous-action online RL", but only shows results on SAC. Would similar improvements be expected in TD3 as well? The authors already have the TD3 baseline, so I expect this experiment to not be so hard as the incorporation of expectile loss should be pretty simple?

## Interpretation of the values of $\tau$ annealing
In Section 4.3, Annealed (0.7) and Fixed (0.7) perform quite well, on par with Annealed (0.9). I am not sure whether there is a strong justification for the paper's insight that one needs to have a high bias in the beginning. In other words, I am still not sure where the learning gains come from.

Similarly, Fixed (0.6) does really well (713, 822) but Fixed (0.5), i.e. SAC, underperforms quite a bit (657, 765). What happens between 0.5 and 0.6 that causes the improvement? I believe investigating this could give a clear insight about why expectile loss is helping, which this paper is currently missing.

### Questions
1. The paper mentions:
> As learning progresses, the policy approaches the optimal policy, reducing the necessity for high optimality in value function learning.

Why is high optimality not necessary towards the end? If the value function is suboptimal, then the policy would also be suboptimal.

2. The paper categorizes hopper-hop and humanoid-run as "more challenging tasks". What makes hopper-hop more challenging than quadruped-run?

3. Replacing SAC's value function update with expectile loss does mean that the entropy regularization is removed, correct?

4. What is the reason for AIQL to perform so well in hopper-hop? It seems this is the main reason for the cumulative results to be so good.

### Soundness
3

### Presentation
2

### Contribution
2
