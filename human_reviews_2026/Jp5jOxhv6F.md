# Unsupervised Mode Discovery for Fine-tuning Multimodal Action Distributions

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
We address the problem of fine-tuning pre-trained generative policies with reinforcement learning while preserving the multimodality of the action distributions of such policies. Current methods for fine-tuning generative policies (e.g. diffusion policies) with reinforcement learning improve task performance but tend to collapse diverse behaviors into a single reward-maximizing mode. To overcome this, we propose MD-MAD, an unsupervised mode discovery framework that uncovers latent behaviors in generative policies, together with a mutual information metric to quantify multimodality. The discovered modes allow mutual information to be used as an intrinsic reward, regularizing reinforcement learning fine-tuning to improve success rates while maintaining diverse strategies. Experiments on robotic manipulation tasks demonstrate that our method consistently outperforms conventional fine-tuning, achieving high task success while preserving richer multimodal action distributions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method to fine-tune pre-trained generative policies with reinforcement learning while preserving their multimodal behaviours. MD-MAD introduces an unsupervised mode discovery process that identifies latent behavioural modes within pre-trained policies and quantifies multimodality via conditional mutual information. This metric serves as an intrinsic reward during RL. Experiments on robotic manipulation and 2D navigation show that MD-MAD maintains multimodal action distributions and high task performance.

### Strengths
1. The paper identifies a clear and underexplored gap, which is a meaningful contribution to the RL + generative modelling community.

2. The figures in this paper is beautiful and informative.

### Weaknesses
The main reasons for my hesitation to recommend acceptance are twofold:

The presentation of the method does not clearly convey its motivation or the underlying logic. Even though I am familiar with this research area, I found it difficult to fully grasp the proposed framework. In particular, some passages, for example, lines 194–197 and the use of “therefore” in line 234, lack sufficient explanation to justify the transitions between ideas. I would encourage the authors to improve the exposition of Section 4 by walking the reader step by step through the theoretical reasoning and explicitly clarifying the assumptions and derivations.

I am also concerned about the theoretical formulation of the "multimodal policy". The definition in Question 2 does not appear to align well with standard understandings of multimodality in probability distributions. The authors may want to revisit the conventional definition of a multimodal distribution (e.g., as summarised on Wikipedia, or more rigorously in recent ICLR work [1]) and better explain how their definition connects to or generalises these prior formulations. Establishing this link would make the theoretical foundation of the paper more convincing.

### Questions
1. Around line 52, the authors mention that RL fine-tuning often biases the policy towards reward-maximising behaviour at the expense of diversity. I think they are correct and RL fine-tuning *should* bias the policy towards reward-maximising behaviours and ignore sub-optimal modes. Although the authors refer to the issue of reward misalignment, they should further clarify their motivation and explicitly explain why this bias is considered problematic in their setting.

2. The multimodal policy $\pi(a \mid s, w)$ suggests that the action distribution can have multiple peaks. However, the definition in line 182 does not seem to reflect this property. To my understanding, the definition implies that a policy is multimodal if some different input noises induce different action distributions — even if each distribution itself is unimodal. Accordingly, a policy would be unimodal if the action distribution is independent of the input noise. Could the authors confirm whether their notion of multimodality is consistent with previous work, or if it represents a distinct concept? 

3. There is a typo in line 207.

4. It is unclear why line 234 states that “distinct values of $z$ can therefore select different behaviours.” After reading Algorithm 1 in the Appendix, my interpretation is that the authors cluster state–action pairs and represent each cluster using the latent code $z$. When PPO is trained, the aim is to control the actor’s behaviour by conditioning on the cluster (source) of the actions. Could the authors confirm whether this interpretation is correct?

**References**

[1] Wang, M., Jin, Y., \& Montana, G. (2025). Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning. ICLR.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors use mutual information maximization to quantify and promote multimodality of a behavioral policy that is fine-tuned to maximize reward. The framework discovers latent behavior in generative policies that are multimodal. The method is shown to work in some basic environments, and against some baselines.

### Strengths
The problem of how to generate multimodal behavior is important. The paper is well written in easy to follow.

### Weaknesses
The objective of mutual information maximization does not necessarily promote multimodality. I think there is a general confusion in the paper, as it assumes that higher mutual information between a latent variable and an action is equivalent to multimodality. This is not the case. Therefore, maximizing mutual information does not produce multimodality unless the environment has already several reward modes – as it is the case in the basic example studied. 

Treating skills as modes in a latent space of a pre-trained policy has been addressed before in 

https://openreview.net/forum?id=HhbHw2yInZ

in a slightly different setting using mixture policies, truly leading with multimodal behaviors. Further discussion and comparison between scopes and methods should be given. 

The derivation of the lower bound is not novel, and resembles, and seems equivalent to the once provided in 
https://arxiv.org/abs/1907.01657

Therefore, the mathematical derivation is not novel, and no references are provided. 

The environments considered are two simple and not high dimensional. For instance, one could use the Ant-v4 environment in a locomotion problem with multiple rewards to test whether the methods produce multimodal behavior in a setting like that of Fig. 4.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes MD–MAD, a framework for fine‑tuning pre‑trained diffusion (or flow) policies with RL while explicitly preserving multimodality. Core ideas: (i) formalize multimodality via conditional mutual information; (ii) discover latent behavioral modes by reparameterizing a steering policy with a latent variable z and training an inference model $q_\phi(z\mid s, a)$; (iii) use the resulting MI lower bound as an intrinsic reward to regularize RL fine‑tuning. 

Experiments on a 2D Gaussian‑mixture landscape and robotic manipulation (Reach, Lift, Avoid) show improved task success and mode retention versus DPPO, residual policies, and steering baselines.

### Strengths
1. On methodology: First, multi-modality MI is practical via a variational lower bound and a latent-conditioned steering policy. Second, such intrinsic MI rewards integrate cleanly with PPO and are agnostic to the RL algorithm. Third, presentations on formulation and motivation are clear. 

2. On Experiments: MD–MAD variants recover full mode coverage under balanced and unbalanced rewards (Table 2, p. 8). On ManiSkill/D3IL tasks, it retains diversity (SR_M, mc@0.8, entropy) with minimal loss in SR, often improving it (Tables 3–4, p. 9).

Also, the appendix clearly situates itself among direct fine‑tuning / residual / steering strategies and shows MD–MAD is orthogonal and can augment them.

### Weaknesses
1. Method:
(1) The theoretical rationale is mostly heuristic; no formal guarantees are provided for convergence or z identifiability. 
(2) Introducing z as the latent factor governing w is risky; without additional structure or constraints, z could become arbitrary or trivial, making the reason it works unclear.
(3) Training and maintaining both the steering and inference networks add complexity and hyperparameter sensitivity.

2. Experiments:
(1) Stress tests mainly vary reward landscapes (rotations/imbalance) and task types; there is little evaluation under dynamics shifts, sensor noise, or partial observability, limiting robustness claims.
(2) The paper uses MI estimates and entropy-based indicators but lacks more direct trajectory diversity metrics.
(3) I think the [mode discovery + MI] backbone should include the unsupervised RL baselines,  VALOR [1], DIAYN [2], and Controllability-Aware Skill Discovery [3].

I did not check the details in the appendix due to limited review time. 

[1] Achiam, Joshua, et al. "Variational option discovery algorithms." arXiv preprint arXiv:1807.10299 (2018).

[2] Eysenbach, Benjamin, et al. "Diversity is all you need: Learning skills without a reward function." arXiv preprint arXiv:1802.06070 (2018).

[3] Park, Seohong, et al. "Controllability-aware unsupervised skill discovery." arXiv preprint arXiv:2302.05103 (2023).

### Questions
1. Is the latent stable across seeds/perturbations (e.g., confusion matrices over modes, NMI/ARI, per-mode SR)?

2. How does MD–MAD behave under dynamic changes or observation noise? Any evidence that the MI estimator remains stable when the state distribution drifts during fine‑tuning? 

3. Could you add an independence/disentanglement constraint to improve the controllability of z (e.g., KL regularization, TC-style regularization)? (Re. Weak 1 (2))

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the authors present MD-MAD, an unsupervised discovery algorithm for finding modes of multimodal action policies. This algorithm tries to solve the problem of loss of multimodality when fine-tuning a multimodal behavior policy. The way the authors address this is by trying to associate a latent-conditioning behavior eliciting model, which given the conditioning would generate noise that when passed into a policy will create the actions from the right modes. The authors train this through a mutual information maximization lens, by optimizing for the mutual information between the latent conditioning and the generated actions.

The authors show experiments in two domains: one in a simple gaussian mixture environment, and another set on simulated ManiSkill environments. In the gaussian setup, the authors show that they are able to extract the modes. In the simulation setup, the experiments show that adding MD-MAD as a regularization maintains higher multimodality.

### Strengths
1. This work finds a shared middle ground between unsupervised skill discovery work and multimodal imitation learning. The connection is quite interesting, since the skill extraction behaves similarly in both cases, and the similarity of algorithms is quite insightful.
2. MD-MAD is compatible with diverse fine-tuning paradigms, such as policy decorator, residual learning, or DPPO. This property is quite helpful in terms of being useful to practitioners.

### Weaknesses
1. The simulated environment benchmarking are only on three tasks, and the extent to which multimodality exists in those tasks is not clear in the first place. A better way to evaluate the multimodality of trained policies would be to use something like the Franka Kitchen environment where there is clear distributional multimodality in behavior and the results are much more easily quantifiable.
2. Without any real robot experiments, it is hard to tell whether this would scale in real, and what kind of bottlenecks could be there in the process.
3. Similar to unsupervised skill discovery, finding the right parametrization for the latent space Z seems challenging for large real world datasets, and merits more discussion.
4. The jump from single-action multimodality to trajectory level multimodality is not clear – given that policies can vary or multiplex similar Zs for different actions at different subsets of the state space.

### Questions
1. The algorithm currently uses a two stage method, but with some human labels or demonstrations, would it be possible to do this in a single stage?
2. What are the challenges towards stability that the authors see in this work? Unsupervised skill discovery never took off due to stability problems in real application, so what are the primary stabilization approaches that can benefit MD-MAD?

### Soundness
2

### Presentation
2

### Contribution
2
