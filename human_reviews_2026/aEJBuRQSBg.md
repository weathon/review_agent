# Softmax for Continuous Actions: Optimality, MCMC Sampling, and Actor-Free Control

- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
As a mathematical solution to entropy-regularized reinforcement learning, softmax policies play important roles in facilitating exploration and policy multi-modality. However, the use of softmax has mainly been restricted to discrete action spaces, and significant challenges exist, both theoretically and empirically, in extending its use to continuous actions: Theoretically, it remains unclear how continuous softmax approximates hard max as temperature decreases, which existing discrete analyses cannot handle. Empirically, using a stand actor architecture (e.g., with Gaussian noise) to approximate softmax is subject to the limited expressivity, while leveraging complex generative models can involve more sophisticated loss design. Our work address these challenges with a simple Deep Decoupled Softmax Q-Learning (DDSQ) algorithm and associated analyses, where we directly implement a continuous softmax of the critic without using a separate actor, eliminating the bias due to actor's expressivity constraint. Theoretically, we provide theoretical guarantees on the suboptimality of continuous softmax based on a novel volume-growth characterization of the level sets in action spaces. Algorithmically, we establish a critic-only training framework that samples from softmax via state-dependent Langevin dynamics. Experiments on MuJoCo benchmarks demonstrate strong performance with balanced training cost.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Deep Decoupled Softmax Q-Learning (DDSQ), an actor-free Maximum Entropy Reinforcement Learning algorithm that attempts to address limitations observed in existing actor-critic methods.

Specifically, the authors highlight that conventional actor architectures often induce overly simplistic action distributions, leading to a mismatch between the learned policy and the true softmax distribution. Furthermore, while more expressive generative architectures can capture complex policies more accurately, they tend to introduce training instabilities and significant computational overhead, especially in the context of maintaining and tuning two complex neural networks (actor and critic), as they introduce extra handcrafted loss functions.

To overcome these challenges, instead of training an actor that approximates the optimal softmax policy, DDSQ leverages Langevin Dynamics to directly sample from it, which it augments with self-normalized importance sampling for particle initialization, specular reflection to prevent boundary stagnation, and a grid-search over step-size schedules to obtain near-optimal action samples. Moreover, the authors provide convergence results for continuous softmax value iteration.

### Strengths
- Establishes polylogarithmic suboptimality bound between softmax and hardmax value iteration for continuous actions.

- Though it does not always outperform the showcased baselines, DDSQ balances is significantly faster than the usual best performing one (DACER).

### Weaknesses
1. L15: At this level, it is unclear what is meant by continuous softmax approximating hardmax.

2. L46-47 seem to imply that the authors are the first to use a softmax policy over continuous actions, which is inaccurate.

3. The "Premilinaries" section does not contain a paragraph on the actor-critic framework.

4. The softmax distribution is referred to as energy-based (L159) without prior definition.

5. L220 does not define $\tau^k$.

6. The motivation behind establishing the bound to the hardmax policy is not clear. This contribution needs to be better stated.

7. Specular reflection is not defined.

8. The authors state that SNIS expedites convergence, but they do not explain how it does so. Differently, they acknowledge that it is subject to high variance and that it is only used as a "coarse initialization scheme". These two statements seem contradictory.

9. Unclear Langevin-incurred backpropagation computational cost during the critic learning phase.

10. Missing references to similar actor-free work: Messaoud et al. introduced S2AC[1], an SVGD-based actor with a tractable density that is used to estimate the entropy of the SVGD-induced policy. Similarly to LD, the induced distribution is a softmax over Q-values. Differently, the entropy can be approximated in closed form. Also, the authors propose a particle truncation heuristic for divergence control is very similar to SNIS initialization in the case of DDSQ, insofar as it eliminates particles that are not close to the target. S2AC also learns the initial distribution which significantly improves convergence to the target.

[1] Messaoud, S., Mokeddem, B., Xue, Z., Pang, L., An, B., Chen, H., & Chawla, S. S2AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic. ICLR 2024.

### Questions
1. What is $\mu(x)$ in L155.

2. L61: How are these candidate step schedules defined in DDSQ? They are not properly introduced in section 4.

3. Why is MCMC not applicable in constrained action spaces in L146?

4. What exactly is boundary stagnation?

5. What is the formula for specular reflection?

6. When learning the critic, are you backproping through the Langevin Dynamics steps? If yes, how expensive is this?

7. How many LD steps are used in the experiments?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the softmax (Boltzmann or energy-based) policies with respect to a Q function in reinforcement learning. Firstly, the paper provides a suboptimality gap of the stationary point of the softmax operator. Then the paper proposes DDSQ, a new value-based algorithm with MCMC sampling for continuous control. DDSQ uses self-normalized importance sampling (SNIS) to sample the initialization points of Langevin dynamics and specular reflection to adjust the sampling step due to the bounded action space. The paper gives some theoretical analysis of both techniques. Empirically, it shows the proposed algorithm achieves competitive performance in Gym MuJoCo and demonstrates multimodal action distributions in specific cases.

### Strengths
The paper has the following strengths:
1. The problem of sampling from the Boltzmann policies of the Q function has been discussed and of interest to the community (e.g., Haarnoja et al., 2017; Messaoud et al., 2024).
2. It provides some useful theoretical analysis. Section 3 discusses a suboptimality gap of the softmax operator, which characterizes how the optimality gap grows with respect to the temperature and the introduced volume function. To my knowledge, such a relationship has not been studied for RL with continuous actions.
3. Overall, it is well written with good figures to explain concepts and phenomena. Figures 1 and 2 illustrate the volume function and stagnation issue of the Langevin dynamics with bounded space very clearly.The mathematical notation is also mostly clear.

Haarnoja, T., Tang, H., Abbeel, P., & Levine, S. (2017). Reinforcement learning with deep energy-based policies. ICML.

Messaoud, S., Mokeddem, B., Xue, Z., Pang, L., An, B., Chen, H., & Chawla, S. (2024). S^2AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic. ICLR.

### Weaknesses
Meanwhile, the paper also has some weaknesses that need to be addressed:
1. Limited empirical investigation. 
  - While strong performance has been achieved in MuJoCo environments, it is unclear how the proposed methods work in other continuous control environments (e.g., DeepMind Control Suite, MetaWorld etc.). Considering there are many algorithmic components and hyperparameters, it’s unclear whether the choice of these components and hyperparameters are robust and still work in other domains. 
  - In addition, it is unclear if and how much each component impacts learning. A proper ablation study is needed to understand them better.
2. Limited discussion of and comparison to closely relevant works. While the paper compares the DDSQ with a few relevant works, it misses a very closely related work (Messaoud et al., 2024). Messaoud et al. also investigate different sampling methods for Boltzmann policies. The paper should provide a comparison to their method and include their algorithms as baselines.

While theoretical statements in the paper are nice, they are not technically strong and are not the core contributions of the paper. On the other hand, the current empirical investigation is limited, so I tend towards a negative assessment of the paper at this stage.

### Questions
Could the authors clarify the below questions?
1. I couldn’t find a clear definition of specular reflection. What does it do specifically?
2. What’s the definition of $W$ in Theorem 2? It’s mentioned to “weigh the importance ratio”, but what is it exactly?
3. Theorem 2 includes the cardinality of the action space in the bound. Does that imply that this result is mostly useful in the discrete case but not the continuous setting?
4. It appears that Theorem 3 doesn’t necessarily imply exponential rate decay. Could the authors provide a discussion on the conditions under which it is such a decay?
5. What’s the impact of mixing the stochastic and deterministic generators? How important and sensitive is this design choice?
6. Similar to 5, what’s the impact of SNIS initialization, specular reflection, and the grid search for the sampling step size schedule?
7. What’s the inference speed of the proposed sampling strategy, compared to something like the Gaussian policy in SAC?

Other minor questions and suggestions with little impact on the rating:
1. Line 200: It might be a good idea to give some intuition about assumption, describing how it restricts the shape of the Q function. For example, what kind of Q functions would be more likely to satisfy the assumption, and when would the assumption be violated.
2. Lines 263 - 269: the discussion of jitted implementation should be put in the experiment section, which reduces interruption to the flow of introducing the core algorithms.
3. Some references (like (4) (7) (10)) are not referencing equations or anything that is properly labeled. It’s better to avoid them to reduce confusion.
4. Footnote 2 and Appendix A.3.4 together are a bit confusing – they contradict with the state-dependent temperature (Line 344). I can only infer how the temperature is set after reading the hyperparameter table. Perhaps add more explanation so that it’s clear what is used.
5. The hyperparameter table in Appendix A.2 does not include $p_e$. Maintaining a complete list of hyperparameters helps improve reproducibility.

### Soundness
2

### Presentation
2

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
Sampling in proportion to $\exp Q(s,\cdot)$ is one of the greatest challenges of SAC as we are seemingly unable to parametrize the inverse CDF as an MLP. This limits the representation capabilities of the policy. The author propose an MCMC approach for sampling directly from $\exp Q(s, \cdot)$ with a warm starting approach. This approach differs from existing literature (except the one case mentioned in the questions) in that they do not learn a forward diffusion policy. This simplifies and accelerates training greatly. Experimental results show that the method is competitive in MuJoCO domains while also being able to sample from multi-modal distributions.

### Strengths
The paper solves a real problem. It does so elegantly while not introducing yet another thing to learn. This to me is the best part of this work and I think the community will find it interesting. 

This work is interesting on the theoretical side as well. This work extends Song et al.'s result on the suboptimality of softmax policies to (bounded) continuous actions. The results are clear and to the best of my knowledge correct. Similarly, the MCMC (with initialization) algorithm is similarly analyzed.

### Weaknesses
The writing is confusing at the beginning, I was expected entropy regularized RL (a la SAC) not the softmax Bellman operator.

The experiments do not include deepmind control nor do they try to ablate the effect of different components like SNIS initialization, reflection correction, automatic temperature tuning, double Q learning, and the added compute requirement. In particular, a comparison with TD3PG and explicitly gradient would have been appreciated.

A non toy example where multi modality is useful would have greatly elevated the paper, an example of this would be ant maze where one of the path is randomly blocked at test time.

### Questions
Do you have an intuition on the tradeoffs of using forward vs backward mode differentiation for calculating the gradient of the energy function? Which one do you use.

What happens if Assumption 1 does not hold? I understand that the second integral in theorem 1 becomes unboundable, but do you think that similar results could hold or do you have a counter example?

Is Assumption 2 really reasonable? Is this equivalent to bounded gradients for Q?

In theorem two, can W be defined explicitly? Similarly can R (reflection) be also defined in the main text? 

Do you think that it would be relevant to cite and compare with [1] ?


# nits
105 -> what property 
can the bar in Figure 3 be made thicker?
866 what does $\mu(\cdot,\cdot)\in\Pi$ mean?


[1] Vineet Jain, Tara Akhound-Sadegh, and Siamak Ravanbakhsh. "Sampling from Energy-based Policies using Diffusion." Reinforcement Learning Journal, vol. 6, 2025, pp. 2291–2307.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper studies entropy‑regularized optimization and its use in softmax Q‑iteration. It claims a polylogarithmic sub‑optimality gap between hard and soft Bellman operators, analyzes a reflective Langevin sampler on bounded domains, proposes a state‑dependent temperature schedule, and gives a differential‑entropy estimator for continuous actions along with a brief SNIS analysis.

### Strengths
The paper tackles a problem that if solved could allow for KL regularization in continuous action MDPs.

### Weaknesses
It seems this papers SNIS setup can require exponentially many samples in the effective horizon $H=1/(1-\gamma)$ which for the standard discount factor of $\gamma = 0.99$ would require $m = \exp(100)$. This seems like an argument against such a method since it suffers from the so-called curse of the horizon [1].  

In general the paper is hard to follow and would benefit greatly from more care and polish in its presentation. 


[1] Liu, Qiang, et al. "Breaking the curse of horizon: Infinite-horizon off-policy estimation." Advances in neural information processing systems 31 (2018).

### Questions
1. Is it really reasonable to assume to minimum volume is lower bounded? 
2. Can you somehow tune $\lambda$ to avoid the exponential blow up in theorem 2 or is this unavoidable? 


some comments

1. Definition 3 seems like an assumption ("we assume" is stated in the definition). 
2. Definition 2 please separate the lower bound on the volume as an assumption. Since this is essentially definition 3 (which should be an assumption), you should introduce the notation here. 
2. line 35, are -> is
3. line 081, include $\gamma \in [0,1)$.
4. line 143, what is paragraph 10, is it the 10th paragraph in your manuscript (somewhere on page 3 or 4?) or the tenth \paragraph{} wrapper? Maybe call it a remark ? 
5. line 131: do you mean policy gradient (not policy optimization) since policy optimization is the process of learning a good policy (fitted $Q$ iteration (DQN) is a policy optimization method)?

### Soundness
2

### Presentation
1

### Contribution
2
