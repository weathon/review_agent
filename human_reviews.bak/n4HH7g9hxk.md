# Deep Exploration with PAC-Bayes

- Decision: Reject
- Scores: 5, 5, 6, 3

## Abstract
Reinforcement learning for continuous control under sparse rewards is an under-explored problem despite its significance in real life. Many complex skills build on intermediate ones as prerequisites. For instance, a humanoid locomotor has to learn how to stand before it can learn to walk. To cope with reward sparsity, a reinforcement learning agent has to perform deep exploration. However, existing deep exploration methods are designed for small discrete action spaces, and their successful generalization to state-of-the-art continuous control remains unproven.  We address the deep exploration problem for the first time from a PAC-Bayesian perspective in the context of actor-critic learning.  To do this, we quantify the error of the Bellman operator through a PAC-Bayes bound, where a bootstrapped ensemble of critic networks represents the posterior distribution, and their targets serve as a data-informed function-space prior. 
We derive an objective function from this bound and use it to train the critic ensemble. Each critic trains an individual actor network, implemented as a shared trunk and critic-specific heads. The agent performs deep exploration by acting deterministically on a randomly chosen actor head. Our proposed algorithm, named PAC-Bayesian Actor-Critic (PBAC), is the only algorithm to successfully discover sparse rewards on a diverse set of continuous control tasks with varying difficulty.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper presents a novel method for exploration in deep reinforcement learning. The method uses a PAC-Bayesian perspective to derive a novel critic update rule, which it then uses as part of an actor-critic setup. They actualize this using a bootstrapped ensemble of critics

### Strengths
Originality:
 * The method is the first to formulate the deep exploration problem from a PAC-Bayesian perspective and overall their method seems original.

Quality:
 * Overall the paper is well written. They achieve decent results, especially in their custom environments.

Significance:
 * Exploration continues to be a difficult problem in a variety of settings. I believe that the problem setting is well motivated and the paper has the potential to be significant from this perspective.
* The paper has some decent results

### Weaknesses
Some key issues:
 * The paper lacks any real analysis of the results. The main takeaway seems to be "our method is better" and that's it. Why does PBAC outperform baselines in the environments where it does? Why doesn't it see a similar improvement for the other benchmarks? Are there any other insights you can share
 * I find it concerning that the main place you see a win is in your own custom environments, because it's difficult to know if this win is because of your algorithm truly doing better or if the baselines could be tuned to achieve similar performance. I think it would be good to see some other common exploration benchmarks.

### Questions
Experiments questions:
 * Can you give more intuition for how to interpret the positional delay parameter $c$? You say in the paper that it is 2 and 1 for ant and humanoid respectively but what do these values mean? How many timesteps does it take for the agent to reach these distances? Can they be reached occasionally through random actions or do they absolutely require an exploration bonus?
 * Why no very sparse version for the humanoid environment?
 * How are hyperparameters tuned for comparison algorithms? Is there any chance that if you tuned their hyperparameters to the extent that you tuned yours then you would see similar performance? I see you tuned BootDQN-P but how about important exploration benchmarks like DRND?
 * Why does your method perform better on the sparse version of humanoid than the original?

Tiny notes / questions:
 * Figure 3 needs axis labels

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes an approach for deep exploration in sparse reward environments by applying a PAC-Bayesian framework in RL. Empirical results are presented in sparse reward environments however this method does not work in dense reward environments and lacks theoretical convergence guarantees.


Modeling the Posterior Distribution consists of the following pieces: The posterior distribution is represented as an ensemble of $K$ critic networks, each weighted equally. Distributions $\rho$ and $\rho_0$ are modeled directly in function space rather than weight space.
A data-informed prior $\rho_0$ is constructed based on the critic targets, providing the most recent information about action values. This prior is modeled as a normal distribution estimated from the critic targets. The KL divergence between $\rho$ and $\rho_0$ 
is then calculated using probability density functions evaluated at the outputs of the critic ensemble.

### Strengths
- The PAC-Bayesian framework provides a flexible way to assess generalization, allowing any posterior distribution to be optimized for data, making it highly adaptable. It's a nice area of research.
- The paper covers aspects of PAC-Bayesian RL, ensuring the bound’s validity and robustness in training.
- Applying the PAC-Bayesian framework for exploration in RL is novel and challenging. So I appreciate the effort.
- Empirical across various continuous control tasks with different reward structures.

### Weaknesses
- PBAC’s structured exploration approach is less effective in dense reward environments, particularly when high rewards are frequent. In these settings, random exploration often performs as well or better.
- The method lacks convergence guarantees,  limiting confidence in its stability and long-term performance across diverse tasks. 
- Looking at the results, particularly the reward plots and Figure 4, it seems that the complexity of this method outweighs its advantages. The performance improvements shown aren’t particularly impressive, and given the considerable complexity involved.

### Questions
- There are several established sparse reward testbeds in robotics. I’m curious why the authors opted to modify Gym environments for sparse rewards. Sparse reward testbeds, such as Meta-World and D4RL, naturally simulate environments with sparse feedback. These frameworks offer realistic benchmarks that directly address sparse rewards, enabling comparison with other established state-of-the-art methods on these platforms without the need for manual modifications.

- Choice of IQM over Reward Plots: Could you elaborate on the decision to present performance results in Table 1 using IQM scores rather than traditional reward plots? I’m curious if this choice was intended to mitigate variability across seeds or to provide a more concise comparison across tasks. Would reward plots offer additional insights, or do IQM tables capture the main performance differences effectively? I understand that reward plots can sometimes be misleading due to high variability across seeds, especially in sparse reward environments where performance can vary significantly. 


**Miscellaneous Comments**
- Table 1: In the table description, "IQM of the area under learning curve" should be corrected to "area under the learning curve."

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the challenge of efficient exploration in reinforcement learning with sparse reward tasks. This is a challenging problem that has wide reaching applications in continuous control domains. They review the literature on (Bayesian inspired) deep exploration algorithms that are designed to find solutions in this setting. They then propose a novel algorithm called PAC-Bayesian Actor critic that leverages a PAC bound to achieve efficient exploration. In practice their method involves fitting a bootstrapped ensemble of function approximators (neural network) regressed to represent the error of the one-step TD Q-learning objective. This is the first principled PAC Bayesian treatment of an Actor-Critic algorithm and it seems to perform well (albeit the evaluations are on relatively standard 'simple' control domains).

### Strengths
- This is a relevant problem in the field of reinforcement learning and from the reviewers limited exposure to PAC-Bayesian theory the derivation seems sound and is intuitive. 
- The empirical results showcase the algorithms effectiveness particularly in tasks where existing methods for deep exploration struggle (i.e. very sparse 'ant' target finding domain, humanoid etc.) which gives a convincing picture.
- I have limited understanding of PAC-Bayesian methods but am an expert on off-policy RL. From what I can tell the derivations do indeed seem novel and the idea of modelling the error of the TD-operator with a PAC-Bayesian approach seems highly promising and worthwhile for the community.

### Weaknesses
- The practical implementation of the well motivated general algorithm requires a series of approximations (e.g. ensemble of networks, Gaussian assumption on the adaptive prior etc.) which are well described in the paper but strike me as somewhat complicated and ad-hoc. It would have been great to see some ablations of different choices here and provide the reader with understanding of why these choices are reasonable (I understand at least the ensemble of networks follows prior art. but the rest seems not already well established).
- The core derivation is relatively simple and clear, but the resulting method (due to the points above) becomes quite complicated. Making me wonder how easy it would be to reproduce and built on top of it.
- The empirical results still only consider fairly low-dimensional domains and no vision based policies/critics (perhaps with the exception of the humanoid domain which is fairly high-dimensional). And the domains are also mostly not naturally sparse reward problems but have been 'sparsified' from their dense reward versions. I have some concerns whether the presented method would also work in more relevant modern settings such as learning vision based policies in e.g. robotics and or more expressive function approximator classes such as using transformers or other large models. Do the authors have any intuitions on these and/or could one more relevant complex domain be considered? E.g. as in recent papers on RND etc.

### Questions
A discussion of the weaknesses above would be highly appreciated for me to raise my score. Additionally I have a problem in understanding one crucial part of the paper:
- The authors mention that due to the long episodes in RL a PAC-Bayesian treatment has to be done based on TD errors (and these only consider consecutive state-action-state tuples and not trajectories). This makes intuitive sense to me, however I could not fully understand how the treatment still captures the uncertainty over the full trajectory space when the estimator used for bootstrapping (i.e. of the Bellman error) conflates everything to the expectation over future trajectories. Could the authors help me understand this more clearly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper studies RL in continuous spaces, and is specifically motivated by sparse-reward problems that require exploration. They take a PAC-Bayesian perspective, and derive an algorithmic approach motivated by a PAC-Bayes analysis to address such hard exploration problems. Experimental results are given illustrating its performance on several Mujoco tasks, and that it is able to solve problems with sparse rewards.

### Strengths
The PAC-Bayes perspective has not been given significant attention in the RL community, and this work takes a step towards addressing that. Furthermore, the proposed algorithm does indeed appear to solve sparse-reward settings effectively.

### Weaknesses
1. The main theoretical result (Theorem 2) is vacuous, as is noted by the authors. It is not clear to me what purpose this result serves given this—it is trivially true that $\| Q_\pi - X \|_{P_\pi}^2 \le \frac{R^2}{(1-\gamma)^2}$, which is all this bound claims to show (unless the negative variance term can be shown to cancel out this term, but this avenue is not pursued here). Furthermore, this bound is used to motivate the algorithm, yet given that it is vacuous, one cannot draw any real conclusions from it, and thus the motivation for the algorithm becomes unclear. I would suggest removing this result and cleaning up the algorithmic motivation.
2. The experimental evaluation is limited, only considering DMC environments, and several other Mujoco tasks. Furthermore, the gains over existing methods are relatively marginal—in the majority of examples, it performs comparably to existing methods (or the stochasticity is high enough that it is not possible to give a clear ranking). 
3. The writing of this paper could be improved and the overall story made more explicit. In particular, as I understand it, the main idea is to apply a PAC-Bayes analysis to get a measure of uncertainty, then use this measure of uncertainty to induce exploration in a UCB-like manner. This was not clear, however, from reading the paper. For example, there are statements like “implement the bound” (line 306) which seem to imply this but are ambiguous (what does it mean to implement a bound?). I would encourage the authors to tighten the story and make connections between the analysis and resulting algorithm more explicit. There were various other unclear statements or issues with the exposition, for example:
	* Line 266: It was not clear to me why a function is replaced by a distribution here. This may be standard in PAC-Bayes analysis (I am not too familiar with PAC-Bayes), but is not standard in the RL literature, and I would suggest further exposition here to make clear why the loss is now over a distribution.
	* Line 302-303: In the sentence starting with “However…”, what is an “existing bound rigorously developed for a specific purpose”? Some explanation of this statement (or relevant citations of such a bound) would be helpful.
	* It was difficult to parse what the final algorithm actually is. Many of the details given in Section 3.2 are not necessary for the main body, and make it difficult to determine which points are the most salient. It would be very helpful to give an algorithm box putting all the pieces together.
4. Another paper that would be good to compare against is [1].

[1] Lee, Kimin, et al. "Sunrise: A simple unified framework for ensemble learning in deep reinforcement learning." International Conference on Machine Learning. PMLR, 2021.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
