# Distributional Sobolev reinforcement learning

- Decision: Reject
- Scores: 5, 1, 3, 3

## Abstract
Distributional reinforcement learning (DRL) is a framework for learning a complete distribution over returns, rather than merely estimating expectations. In this paper, we extend DRL on continuous state-action spaces by modeling not only the distribution over the scalar state-action value function but also its gradient. We refer to this method as Distributional Sobolev training. Inspired by Stochastic Value Gradients (SVG), we achieve this by leveraging a one-step world model of the reward and transition distributions implemented using a conditional Variational Autoencoder (cVAE). Our approach is sample-based and relies on Maximum Mean Discrepancy (MMD) to instantiate the distributional Bellman operator. We first showcase the method on a toy supervised learning problem. We then validate our algorithm in several Mujoco/Brax environments.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel approach to distributional reinforcement learning, Distributional Sobolev RL, which models both the distribution of random returns and the action-gradients. The paper also empirically demonstrate the effectiveness of proposed algorithm. While incorporating Sobolev training is an interesting and promising idea, I have some concerns regarding the current manuscript.

### Strengths
The motivation is clear, and the idea of learning distribution over action gradients is good.

### Weaknesses
- **Experiments**: There are several issues that need to be clarified in the toy experiments. For example, the red and blue dashed lines in Figures 3 and 4 are confusing. In Figure 1, it is difficult to distinguish the blue and green lines. For MuJoCo experiments, I have big concerns about the **fairness comparison**. The authors only present the performance results after $0.25\times10^6$ training steps, while the default setting is  $1\times10^6$. The potential cause I guess is that DSDPG is computationally cost, but this adjustment seems to benefit DSDPG since it is a model-based algorithm utilizing pre-trained transition information, which helps faster convergence. In contrast, DDPG is a model-free algorithm that is not sample-efficient, particularly in the early stages.  Furthermore, using actor-critic-based methods to demonstrate the advantage of incorporating gradient information into distributional modeling may not be ideal since the decision-making is based on the actor, while the critic (return distribution) is primarily used to train the actor network.

- **Random state-action Sobolev return**: First, It is unclear whether Sobolev returns are actually used for decision-making. Second, Equation (15) is confusing, and it appears to be missing a term involving $\frac{\partial s'}{\partial s}\frac{\partial s}{\partial a} x$.  Third, as the Sobolev return is defined as the derivative of the return with respect to the state, I have a concern regarding the handling of complex state spaces, such as pixel-based states in Atari games. The computational burden will significantly increase as the dimension of the state grows. Since the current experiments are implemented on Mujoco, where the state and action are vectors, I am curious about how well DSRL will perform in more complex environments.

  

- **Discussion with other methods:**  The discussion of related methods is limited. It would be beneficial to expand on this, especially regarding the connections to existing distributional RL approaches (QR-DQN, C51). The use of MMD minimization for learning the distributional Bellman operator is inspired by MMD-DQN [1], but this connection should be discussed in more detail. Additionally, tuning the bandwidth parameter $h$ in the multiquadratic kernel is crucial since kernel-based methods are sensitive to bandwidth selection.

[1] Distributional Reinforcement Learning via Moment Matching. AAAI 2021.

### Questions
Q1: I have concerns regarding sample-based distributional RL. While some works utilize samples to represent return distributions, these methods are not mainstream. One significant drawback is the computational inefficiency associated with using generative models to represent return distributions. Furthermore, sample-based methods involving maximum likelihood estimation and specific distributional parameterization may suffer from model misspecification, making it difficult to capture the distributional Bellman equation.

Q2: Have the authors considered using diffusion models to model transition probabilities, given their impressive empirical performance?

Q3: Have the authors considered implementing Distributional Sobolev RL on top of value-based RL?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
This paper proposed a so-called Distributional Sobelev Deterministic Policy gradient, where the proposed method uses a distributional critic loss while incorporating the gradient of a random variable. Particularly, in RL scenarios, the authors also leverage a differential world model of the environment in order to infer gradients from observations. They also conducted some experiments on the toy supervised learning tasks and several Mujoco envs.

### Strengths
The proposed algorithm seems to incorporate different kinds of methods, such as Sobolev learning and conditional VAE.

### Weaknesses
**A poor motivation**. In the introduction part, it seems that the authors try to motivate the paper to improve the (stability of) policy gradient algorithm by incorporating the gradient information in the policy optimization. This is a very straightforward and even trivial idea. Also, the authors tried to develop a distributional version of algorithms that is orthogonal to the existing algorithm and has already been studied before, such as [1]. It was very confusing to understand the motivation behind it, but I ended up being frustrated. 

**No methodological contribution**. Distributional RL and Sobolev are already known methods, and I do not think it is meaningful to do such kind of combination research. Also, the proposed method is constrained. For example, the paper directly limits the scope to a deterministic policy gradient method starting from Eq.(2). The gradient to be incorporated into the training requires the differentiability of the environment. To this end, the authors proposed a world model by CVAE, but it is not practical for real and potentially complicated environments.

**Poor Writing**. It is very hard to understand what the authors want to express. Many expressions are very inaccurate, and notation usage is sloppy. For example, what is the likelihood estimation issue in Line 215? How to define the derivative of a random variable over a in Eq.8? Note that the return Z is a random variable conditionally over the policy on s and a. I am confused to understand this point. The reference of Czarnecki et al., 2017 is also not complete, making me feel the writing of this paper is very non-professional. Z^S in Eq. 11 is not consistent with Z_S, which occurs many times. 

**The experiments are trivial and limited**. It is not clear if it is necessary to have experiments on supervised learning tasks since the focus of this paper is RL. The empirical improvement in Mujoco is insignificant, making me feel that the proposed method is even not sound.


[1] Distributed Distributional Deterministic Policy Gradients (ICLR 2018)

### Questions
Please see Weakness.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper proposes “Distributional Sobolev RL” to extend distributional RL by modeling both the return and gradient distributions of the state-action value function. Using a one-step world model with a conditional Variational Autoencoder (cVAE) and Maximum Mean Discrepancy (MMD) for the Bellman operator, the approach is validated on toy tasks and Mujoco/Brax environments.

### Strengths
Learning the gradient in addition to the distribution is an interesting idea.

### Weaknesses
I failed to understand this paper and am quite confused about the proposed approach. We should check if it is a personal difficulty of me or if other reviewers also feel confused. If the latter, this indicates room for significant improvements in the paper.

I have listed my questions in the Questions section below.  These questions prevent me from understanding the proposed algorithm. In addition, Section 3.1 has few intuitive explanation. for example, why is eq (16) proposed? The paper states, "Depending on the applications Eq. 11 or 16 may make more sense ". This may need further clarification (when should we prefer 11 or 16?). Since I couldn’t follow Section 3.1, I was unable to understand Section 3.2 as well.

Overall, I find this paper lacks rigor and has limited (theoretical) justification. The definitions in section 3.1 are not rigorous enough. 

The related work section is not organized well. The authors should put most related works here instead of in the introduction. Also, many prior works appear to be missed.

Minor issue: Figure 1 appear somewhat sloppy: the right side is cut off, and so $Z^{next}$ is incomplete.

### Questions
- The $f$ in eq (12) is not bold, which took me a while.
- In Line 178, how do you assume the action-gradient $x^\text{action}$ exists? First, I believe you need some differentiable conditions; second, if actions are discrete, how is the gradient defined in this case?
- Similarly, what if transition is discrete? Then is $\frac{\partial }{\partial s} x^{\text{return}}$ well defined?
- Similarly, is $\frac{\partial }{\partial a} R(s,a)$ well defined given that $a$ may be discrete?
- In Line 191, what does "gradient computations of the bootstrapped target" mean? Why do you assume it is differentiable?
- In eq (16), the notation $\nabla_s$ is unclear since there is no $s$ on the right. This is confusing.
- SImilarly, $\nabla_a$ is also unclear in eq (16). It is also in eq (11).
- Estimating the return distribution and the gradient seem orthogonal. Could you explain why you want to merge them in this paper?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper meticulously formulates a new model-based actor-critic method for distributional RL. The proposed method utilizes a conditional VAE as their world model, that learns to predict both the next state and reward, conditioned on the current state and action. This model, trained along with the policy and critic networks, allows computing the MMD loss for the critic, and introduces uncertainty in the action gradients (of the returns) along with the returns themselves. The authors elaborate the required mathematical background to explain their method and design choices, and show the advantage of the distributional approach and the usage of gradient information in a toy problem. Combined with RL, the proposed method is tested in five mujoco environments, showing a certain superiority over other approaches.

### Strengths
1. Mathematical background and notations: Comprehensive and helps to explain the proposed method. I am not sure which ones are novel though, if some of them, it is worth noting.

2. The extension to distributional RL is done elegantly, using a rather simple world model, which allows using other types of models.

3. Experiments seem extensive in terms of benchmarks, although see my notes in the weaknesses section.

### Weaknesses
### Experiments:
1.  My major concern with this work is the empirical performance of the proposed method; It performs very similar to DDPG with MMD (only minor improvements) and in most tested environments, vanilla DDPG performs the best. Considering that the toy problem does not involve RL, I think that the method is not justified enough. I suggest looking for other environments that show the value of such method, maybe ones that are more stochastic?

2. Lack of baselines: although the method is meant for the distributional setting, mean-based methods could be applied to the same environments, and there are many that should be compared, if not for direct comparison it helps for understanding the scale of performance improvement.

### Related Work:
This section seems very limited. Since the introduction describes some, I would move this section to the beginning of the paper, and maybe merge with the introduction section.

### Clarity:
For me, it is not clear enough what is novel in this method; The usage of the gradient info has been done, hence the method extends this approach to distributional RL?
The same applies for the mathematical analysis. I think it should be clearer -- either differ between existing and new/ add a statement of the paper's contribution.

### Minor Comments:
1. line 449: Deterministic Deep Policy Gradient -> Deep Deterministic Policy Gradient.

2. line 456: both variants of
DSDPG, using action gradient and DSDPG using state-action" -- rephrase

### Questions
1. What are the paper's contributions?

2. Is it possible to extend your method to support other actor-critic methods? (e.g., PPO, A2C, etc.)

3. In your experiments, it seems that you evaluate the learning curve, while eventually most methods perform in par after enough interaction. Have you tried limiting the data (to show better sample complexity)?

4. One of the advantages of using a model-based method, given an accurate enough model, is the ability to train without direct interaction (only from the world model) have you tried it?

### Soundness
3

### Presentation
2

### Contribution
2
