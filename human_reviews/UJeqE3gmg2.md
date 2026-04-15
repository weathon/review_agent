# Robust Policy Optimization with Evolutionary Techniques

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3

## Abstract
Learning-based techniques to train control policies of autonomous agents often assume that the agent experiences 
are sampled according to a specific dynamical model for the environment. However, 
environmental dynamics can change, due to intentional or unintended environmental changes. While domain randomization and robust
learning can handle some distribution shifts, significant environmental shifts may necessitate re-training to learn policies optimal in the 
changed environment.
We present an algorithm called `Evolutionary Robust Policy Optimization' (ERPO)
inspired by evolutionary game theory (EGT) to address
the problem of incrementally and efficiently adapting policies to an altered 
environment. We give theoretical guarantees on the convergence of our 
algorithm to the optimal policy under the assumption of sparse rewards.
We empirically demonstrate that our algorithm outperforms several state-of-the-art 
deep RL algorithms in many gym environments. Specifically, we are 
able to adapt policies using fewer training steps while getting 
higher rewards and requiring lower overall computation time.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed "Evolutionary Robust Policy Optimization (ERPO)", which aims to adapt policies to an altered environment using fewer training steps while getting higher rewards and requiring lower overall computation time.

### Strengths
see above

### Weaknesses
This paper seems to be an incomplete work, fails to illustrate its motivation and main technical contribution, and misses a lot of important baselines (e.g., from meta-learning literature) in its experiments. Hence, I think it is difficult to evaluate the novelty, effectiveness, and importance based on its current version that needs to be improved significantly.

some comments:

1. I wonder why evolutionary game theory is a proper solution to robust policy optimization.

2. The title is too broad to reflect the main contribution of this work. 

3. Please add more baselines and compare ERPO with them in extensive environments in order to evaluate ERPO's soundness.

4. incorrect citation format.

### Questions
see above

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The manuscript presents a novel method for transfer learning that combines ideas from evolutionary game theory and reinforcement learning. The method operates only with discrete state-action spaces. Overall, the results showcase that the proposed method, ERPO, consistently outperforms several baselines.

### Strengths
- The paper is generally well-written, easy to follow and the ideas are conveyed effectively.
- Strong empirical results on several tasks
- Theoretical convergence guarantees

### Weaknesses
- No limitations or weaknesses are described in the text
- No hints on how the method can be extended to continuous state-action spaces are given
- The authors mention *"Simulation models are generally simplistic and fail to consider environmental variables ... so they cannot be directly deployed in such applications"*. I have the following issues with the statement:
    - Simulation models are not generally simplistic. Even the less realistic simulators contain quite sophisticated procedures and models behind. Changing "simplistic" to "non-realistic" or just referring to the well-known Sim2Real gap is enough.
    - Most real-world applications are continuous in nature and are quite difficult to discretize or solve with discrete state-action spaces. The manuscript proposes a method that can only operate with state-action spaces. Thus, this sentence is not the best "motivation" for the proposed method.
- The above comment leads to my main "complaint" from the presented work: since most realistic robotic/autonomous systems applications are continuous in nature, how is the proposed method solving the issue it claims to solve? All the experiments as well have nothing to do with robotic applications.

Typos/Minor comments
===================

- Page 2, first paragraph: *"approaches Rajeswaran et al. (2016)train"* -> there is a space missing
- Page 3, Section 3, first paragraph: *"theory (EGT) Smith (1982),Sandholm (2009)"* -> there is a space missing
- Page 3, last sentence: *"As we make state-wise updates, we modify the replicator equation to be We modify this replicator equation as follows:"*

### Questions
- What are the main limitations of ERPO?
- How can we extend ERPO to the continuous case?
- How can ERPO be useful in realistic applications of autonomous systems/robotics?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the problem of adapting reinforcement learning (RL) policies to significant changes in the environment dynamics in robust RL. Many existing methods, like domain randomization and robust policy optimization, fail when test environments differ substantially from training. The authors propose an evolutionary robust policy optimization (ERPO) approach to adapt policies without full retraining. Assuming access to the optimal blackbox policy on the original environment, ERPO explores the new environment using an $\epsilon$-soft version of this policy. It incrementally improves the policy by weighting state-action pairs from fitter trajectories more highly, inspired by evolutionary game theory. Experiments show superior results against methods only trained on the old environment (referred to as base models) or only trained on the new environment. They also compare their algorithm to domain randomization methods such as PPO-DR.

### Strengths
The paper's setup tackles an important real-world challenge - adapting black-box RL policies without full retraining. The simplicity and intuitiveness of ERPO are also strengths. Updating actions based on relative expected rewards is an interesting idea. This evolutionary approach avoids needing gradients for the new environment. However, there are some major concerns about the author's implementation of this idea, the theory, and the writing quality and experiments. Specific issues are detailed in the next section recommending rejection.

### Weaknesses
The main concern is about the soundness of the algorithm and theoretical claims. Even if we accept the sparse reward assumption, the conclusion of the authors that "*the value of a state can be approximated by the average return across all trajectories containing the state*" is an intuitive statement and needs concrete evidence. Even if we accept this argument, the proof of Theorem 1 is still incorrect. In particular, the proof mixes up the behavior policy $\pi^i\_{train}$ and the learned policy $\pi^i_{new}$ and assumes they have the same sampling distribution, which is clearly not the case. Moreover, I think the current proof, even by fixing all the previous issues, would not work unless we define 
$$\pi^{i+1}(a|s) = \pi^i(a|s) \times \frac{\mathbb{E}[f(\tau\_{(s,a)})]}{\mathbb{E}[f(\tau\_{s})]}$$
where, in the denominator, we have $\tau\_{s}$ instead of $\tau\_{s'}$. 

Besides the previous concerns, I think the empirical comparison to other methods is unfair. In particular, the proposed method essentially uses the information from the old environment (through the optimal policy) and the data from the new environment. A more fair comparison would initialize the policy of any method that trains on the new environment as the previous policy (e.g., using a cross-entropy loss). The fact that PPO eventually gets to the optimal solution (even without the correct initialization) suggests that initializing its policy with $\pi\_{old}$ will result in comparable or even better results than ERPO. 

In summary, concerns include the following:
1. Unjustified approximations in analysis
2. Logical gaps in the convergence proof
3. Unfair comparative evaluations against methods not exploiting old policy information

### Questions
Please refer to the previous section.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair
