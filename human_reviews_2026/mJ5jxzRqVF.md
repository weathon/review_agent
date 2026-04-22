# Reward Inflation Paradigm Through the Lens of Monetary Economics

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 4, 0

## Abstract
Reward is fundamental to reinforcement learning (RL), where the agent treats it as an incentive to maximize.
This appearance is akin to a rational human who maximizes their income.
However, in the real economy, money expands to stimulate economic growth.
Inspired by this principle of monetary economics, we introduce a novel RL paradigm, reward inflation, which gradually increases the reward scale during training.
Analogous to inflationary policies used by central banks to stimulate economic growth, reward inflation acts as an incentive stimulus for agents to accelerate policy learning.
Reward inflation can be applied in two ways: fixed or adaptive.
Motivated by the Fed's monetary policy, we propose FedeRL, a dynamic controller for adaptive inflation.
Theoretical analysis suggests that the effect of reward inflation is threefold: (1) induces recency bias in temporal-difference learning, (2) amplifies policy gradients, and (3) enhances neural activation.
Empirical results corroborate these insights, showing that moderate inflation improves performance on continuous control tasks.
Moreover, FedeRL performed even better than fixed inflation and outperformed comparable baselines.
By translating economic growth principles into RL, our approach offers a novel perspective that strengthens policy optimization and addresses fundamental RL objectives.
The implementation code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FedeRL, a method inspired by monetary economics that dynamically scales reward functions.
Specifically, the authors first motivate why adjusting the scales of the rewards can be beneficial during training, using concepts from macroeconomics.
This is followed by the FedeRL algorithm, which, inspired by how central banks work, adjusts the inflation rates based on gradient norms of the function approximator.
Finally, the authors show empirically that FedeRL can reduce dead neurons and improve performance in MuJoCo when used with the soft actor-critic algorithm.
Overall, the paper proposes an RL algorithm that achieves improved empirical results, but its theoretical framing is underdeveloped and, in its current form, does not add meaningful insight.

### Strengths
The proposed method is simple to implement and appears to perform well on MuJoCo tasks.
The idea of modifying the rewards to stabilize training is an interesting one, which may provide new insights into algorithm designs.

### Weaknesses
The connection to economic theory appears largely superficial.
While the paper draws some analogies to economic concepts, these do not seem to inform the design of the proposed algorithm (FedeRL) in a meaningful way.
The “economic inspiration” is mentioned primarily at a narrative level rather than as a source of substantive methodological insight.
As a result, the framing may give the impression of novelty without contributing actual conceptual or theoretical value to the RL problem.

The proposed method, while effective, also introduces eight new hyperparameters ($Y$, $\beta_s$, $\beta_l$, $K$, $\rho_Y^{*}$, $\lambda_1$, $\lambda_2$, $\tau_\rho$), and a study of these parameters is lacking.

### Questions
1. Can the authors clarify precisely how the economic analogy contributes to the algorithm’s design?
2. How sensitive is the algorithm regarding the hyperparameters?
3. In the implementation of SAC, are the rewards scaled first before being stored into the replay buffer? Or is it scaled on-the-fly during training depending on the global step? If I understand correctly, Assumption 1 is based on the setting where all rewards are scaled by the same factor. However, if the replay buffer contains rewards that are scaled differently, it is not clear to me what Q-learning would converge to, if it converges at all.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduce a reward inflation method that increase the reward scale during training.  Theoretical analysis suggests three main effects: (1) recency bias, (2) gradient norm amplification, and (3) enhanced neural activation.

### Strengths
1. This idea is interesting and novel
2. the result is promising

### Weaknesses
1. The proposed approach assumes a noise-free environment when adjusting the reward scale. However, if consider stochasticity—such as observation noise, reward corruption, or environmental uncertainty, the change of reward scale also affect these noise and may also have negative affect on the learning.

2. The hyperparameter study on soft update coefficent will strength the paper.

### Questions
1. The reward inflation process is happened before saved into replay buffer or after sample from replay buffer? if happened before replay buffer, then this will cause the issue that reward sclae uneven in the replay buffer which may limit the performance.

2.Is there a upper bound for the inflation which cause learning collapse (idea also from economics)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces "reward inflation," a novel setup for reward rescaling during RL inspired by monetary economics, where a 'nominal' reward scale is gradually incremented by some percent during training (inflation if a positive percent, deflation if a negative percent). The authors propose that when acting as an inflation, this reward modulation acts as an "incentive stimulus" that amplifies policy gradients, induces a recency bias in learning, and increases network activation (i.e. reducing the percent of dead neurons). They also propose FedeRL, an adaptive mechanism for setting the reward inflation percent with an analogy to a central bank which attempts to dynamically adjust inflation based upon an economic environment's speed of change. In this RL application, FedeRL dynamically adjusts the inflation rate based on two exponential moving averages of the policy gradient norms to stabilize training. Empirically, both fixed inflation and the FedeRL adaptive inflation method are shown to outperform the standard SAC baseline and other related methods on continuous control tasks.

### Strengths
- This work presents a highly original and creative paradigm by drawing a compelling analogy from monetary economics to RL. I genuinely found this interesting to read and consider.
- The adaptive FedeRL inflation rate determination is quite practical and shows clear, intuitive, and interpretable behavior.
- Consistent empirical performance gains are shown over the standard (0%) baseline and other comparable methods (like LR scheduling and PER).

### Weaknesses
- The primary theoretical effect (gradient norm amplification) is described as functionally very similar to learning rate scheduling, and yet the paper doesn't fully disentangle why "reward inflation" is a superior mechanism. When comparing against learning rate scheduling, this is only done to a limited degree (+ vs - 2%) when a more complete range of searching is completed for FedeRL and static inflation rates.
- The new FedeRL controller introduces its own set of hyperparameters (natural rate, short and long range moving norm averages, etc.), which may be difficult to tune and it is clear from Table 1 that these values might require tuning per task. These limitations are not explored.
- The inflationary rate is motivated heavily from a monetary perspective, though it is in the end a rather simple change to reward rescaling and any negative side effects (or limitations) of such are not much explored. E.g. catastrophic forgetting could be made much worse by any positive inflationary value. Furthermore, it is unclear whether the training time for models with inflationary rates which are non-zero has to be limited to avoid the eventual exponential rise in training impact. How the model training times must be limited is not discussed herein.
- This method is only applied with the SAC RL algorithm. Ensuring that these benefits transfer to alternative RL algorithms would be ideal and important for demonstrating generality.

### Questions
A response to the above weaknesses would be much appreciated.

Though the inflationary process is described as similar to learning rate scheduling, it might be more accurate to say that it could be similar to (perhaps even equivalent to) a particular scheduler for the learning rate combined with a change to the discounting term (gamma). Is this correct?

Ideally to address the above, the nominal return (sum of discounted future rewards) could be unpacked and given a form with respect to the regular reward. This should allow an identification of the degree to which these inflationary measures are simply equivalent to a modification of the discounting factor along with a learning rate change. Doing so could significantly demystify this method and also ensure that claims are not overblown. Note that this would also indicate that the baselines for comparison should also be extended with a sweep over learning rate scheduler rates and discounting factors.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes a "reward inflation" paradigm inspired by monetary economics where the "nominal reward" scale is gradually increased during training, and introduces FedeRL, an adaptive controller mimicking the Federal Reserve's monetary policy. The authors analyze three effects of this approach: inducing recency bias, amplifying gradient norms, and enhancing neural activation. Empirical results on MuJoCo tasks are shown.

### Strengths
No

### Weaknesses
The paper's entire premise hinges on an analogy between RL rewards and money. This analogy is incorrect and unreasonable. It mistakes a simple heuristic (non-stationary reward scaling) for a economic principle.

### Questions
No

### Soundness
1

### Presentation
2

### Contribution
1
