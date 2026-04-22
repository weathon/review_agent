# Robust Multi-Agent Reinforcement Learning with Diverse Adversarial Agent Generation and Contrastive Policy Encoding

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Multi-agent reinforcement learning (MARL) has emerged as a promising approach for learning coordination policies in multi-agent systems (MAS).  However, policies trained by conventional MARL algorithms often overfit to specific team behaviors, limiting their ability to generalize and remain robust when faced with teammate failures or adversarial interventions. Such limitations pose significant challenges to the deployment of MARL in real-world applications. To address these issues, we propose a novel co-evolutionary robust MARL framework that enhances the robustness and generalization of MARL algorithms under policy disturbances and adversarial agents within MAS. Our framework comprises two key components: (1) DAAG: a Diverse Adversarial Agent Generator optimized via an information theoretic objective to produce behaviorally diverse and challenging adversarial agents, and (2) CAPE: a Contrastive learning-based Agent Policy Encoder that continuously learns informative representations of adversarial agents’ policies encountered during training, which are integrated into the MARL agents’ policy learning process to enable dynamical adaptation to diverse and evolving adversarial policies. These two components are optimized in a co-evolutionary training paradigm, enabling cooperative agents to robustly co-adapt alongside increasingly diverse adversaries. Comprehensive experiments conducted on the Predator-Prey and SMAC benchmarks demonstrate that our framework significantly outperforms baseline methods in both robustness and generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an adversarial multi-agent reinforcement learning (MARL) method to improve robustness in cooperative systems, where agents can change their behavior adversarially to compromise the originally learned coordination.

The methods consist of three components:
1. DAAG (Diverse Adversarial Agent Generator), which produces diverse adversarial behavior.
2. CAPE (Contrastive learning-based Agent Policy Encoder), which learns an embedding of adversarial agent policies
3. A co-evolutionary training scheme, which alternately trains functional agents and adversaries.

The approach is evaluated in a simple predator-prey task and some SMAC tasks, showing better performance than prior MARL methods.

### Strengths
The paper addresses an important (yet well-studied) problem.

It is mostly well-written and easy to understand. I view the work as sound because it builds on well-established methods and definitions (see Novelty below).

### Weaknesses
**Novelty**

Robustness via adversarial reinforcement learning is a well-established field in MARL with several previous works. The paper focuses on adversarial agents, where some of the original functional agents can be randomly replaced by adversarial counterparts. This has already been introduced in [1,2]. The problem formulation, where some agents act adversarially, is known as a *mixed cooperative-competitive game* [1,3].

The paper promotes diversity for improved robustness in MARL, a topic that has been studied in various prior works [4,5]. Policy-based diversity through latent variables has been studied in [6].

Learning policy embeddings for conditioning the behavior of the functional agents is a well-known practice known as *opponent modeling* [7,8], and the aggregation of policy embeddings via attention has also been proposed in the past [9,10,11].

The co-evolutionary training, i.e., alternating updates to the functional and adversarial parts, is a common practice in adversarial reinforcement learning [1,11].

Given the publication years of the listed prior works, I regard the proposed method as incremental and recommend a more thorough literature review and discussion.

**Quality**

While the text is mainly well-written, all figures have diminishingly small fonts, which makes them unreadable when printed. There is a possibility that I may have missed crucial details due to this.

**Significance**

I do not view the proposed method as particularly ground-breaking, since it is essentially based on adversarial training with diverse opponents, which is well-known in the literature [1-11].

The experiments only present the learning progress by testing each (baseline) approach periodically against a pre-defined set of adversary agents. However, I could not find any information about how these adversary agents were created, i.e., how biased the test actually is.

To evaluate the worst-case robustness of each approach, I recommend running an adversarial MARL algorithm on the fully trained functional agents, similar to the M3DDPG paper (the proposed method might work on average on a small test set, but potentially vulnerable to a true exploiter, according to [19]).

I also recommend comparing with other agent-replacing adversarial approaches, such as [1,2] and alternative policy embedding approaches [6,8], to put the work more into the context of the existing body of literature.

**Minor**

According to the problem definition, observations are considered to be Markovian in this paper, as the policies condition only on observations. To fix this, the definition should be rewritten such that the policies condition on the action-observation histories [12].

**Literature**

[1] Phan et al., "Learning and Testing Resilience in Cooperative Multi-Agent Systems", AAMAS-20

[2] Li et al., "Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game", ICLR-24

[3] Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NeurIPS-17

[4] Jaderberg et al., "Human-Level Performance in 3D Multiplayer Games with Population-Based Deep Reinforcement Learning", Science-19

[5] Vinyals et al., "Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning", Nature-19

[6] Mahajan et al., "MAVEN: Multi-Agent Variational Exploration", NeurIPS-19

[7] Nashed et al., "A Survey on Opponent Modeling in Adversarial Domains", JAIR-22

[8] He et al., "Opponent Modeling in Deep Reinforcement Learning", ICML-16

[9] Iqbal et al., "Actor-Attention-Critic for Multi-Agent Reinforcement Learning", ICML-19

[10] Phan et al., "Attention-Based Recurrence for Multi-Agent Reinforcement Learning under Stochastic Partial Observability", ICML-23

[11] Pinto et al., "Robust Adversarial Reinforcement Learning", ICML-17

[12] Oliehoek et al., "A Concise Introduction to Decentralized POMDPs", 2015

### Questions
1. Why does the approach focus/limit itself to discrete styles? With a continuous latent vector, sampled from a multivariate distribution, more variability would be possible.
2. There is an apparent information asymmetry between functional agents and adversarial agents (which condition on the global state, according to Line 193). What is the motivation behind this asymmetry? In Appendix D.2, why does the performance decrease when K is chosen to be too large?
3. Experiments: How were the “unseen adversarial agent policies” created? Since training is co-evolutionary, are the adversaries from the same learned distribution/generator?
4. Experiments: How robust are the policies when facing an adversary directly trained against the fixed functional policy (like in the M3DDPG paper)?
5. Figure 3a: Without the colors, I only see 3-4 clusters out of the 10 generated policies. Is this supposed to be sufficiently diverse? Is there any intuition why the dots corresponding to policy 9 are spread into 3 separate clusters?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Conventional MARL tends to overfit to specific team behaviors and becomes brittle when teammates fail or adversarial interventions occur. Prior robustness methods lack control over behavioral diversity, and evolutionary approaches are computationally heavy and require separate optimization. This paper addresses these limitations by proposing DAAG, which generates behaviorally diverse adversarial agents, and CAPE, which continuously learns informative representations of adversarial agents’ policies. This paper integrates these two components into an end-to-end pipeline and trains them in a co-evolutionary paradigm. Experiments show that the framework achieves improvements in robustness and generalization compared to baselines.

### Strengths
1. Adversarial attacks have been used to improve robustness in MARL, but a major limitation is overfitting to specific attacks. To address this, learning adversarial policies with behavioral diversity is crucial. Prior work has tried to increase diversity using evolutionary methods, but these approaches are computationally expensive and require separate (non–end-to-end) optimization. This paper proposes Diverse Adversarial Agent Generator (DAAG), an unsupervised mutual information-based method for diverse policy generation, which enables style code conditioned adversarial policies. 


2. Owing to partial observability in multi-agent systems, observations are noisy and information-limited, making it hard to discriminate diverse adversarial policies from observations alone. Different types of adversarial attacks demand attack-dependent adaptations of the cooperative agents’ policies, which prior methods struggle to learn. This paper proposes CAPE, which introduces a contrastive learning based discriminative representation that is fed into both the policy and the critic, enabling clear separation among diverse adversarial policies. In addition, this representation supports continual learning via an Instance-wise Relation Distillation (IRD) loss.


3. Several prior approaches are not end-to-end, requiring separate optimization for adversarial and cooperative agents, which limits their ability to co-adapt. This paper integrates the proposed components into an end-to-end pipeline, enabling co-evolutionary training in which cooperative agents adapt as adversarial policies evolve. Experiments show improvements in robustness over baselines, demonstrating the effectiveness of the proposed components.

### Weaknesses
1. This paper addresses the learning of diverse adversarial policies and the training of agent policies that adapt to varying attack types. Increasing the methodological novelty and reflecting the multi-agent characteristics more explicitly would strengthen the contribution. DAAG, conditioned on style codes, can learn a variety of adversarial policies; the approach is largely aligned with skill-based RL such as DIAYN [1]. CAPE enables agent policy learning that accounts for diverse adversarial policies via a contrastive-learning–based representation; the design is closely related to contrastive learning RL such as CURL [2]. Incorporating elements that are more specific to multi-agent systems, and the adversarial attack setting would further enhance the originality.
2. This paper leverages DAAG to enhance the diversity of adversarial policies, which broadens the variety of attack actions. That said, restricting each episode to a single, fixed adversarial agent may actually constrain diversity in multi-agent settings it fixes the attack target identity, limits the interaction patterns the team experiences, reduces coverage of agent-subset combinations. In line with recent work (e.g., ROMANCE; WALL [3]), allowing dynamic multi-agent attacks where the adversary can select one or more agents at each timestep can expand the attack space combinatorially, introduce temporal variation. Considering these factors, it should be possible to implement more diverse adversarial policies.
3. Many studies on robust RL and adversarial attacks evaluate not only robustness to their own proposed attack but also to other attack methods and broader distribution shifts. Incorporating such evaluations here would better substantiate the paper’s robustness claims. In addition, the field already offers many robust MARL and adversarial RL baselines; expanding the comparison set would make the results easier to trust and compare. The current experiments focus on Predator–Prey and SMAC (2s3z, 3m, MMM); demonstrating robustness and generalization across a broader range of settings such as additional SMAC scenarios and other benchmark suites would further strengthen the empirical evidence.
4. The paper notes that prior methods suffer from high computational cost. Given that the proposed method includes multiple components and trains models such as transformers, non-trivial computational overhead is also likely. It would therefore be helpful to provide theoretical comparisons or experimental evidence (e.g., training time) demonstrating improved efficiency. The method introduces multiple components and many hyperparameters, which may increase training sensitivity; additional ablations and sensitivity analyses would be valuable. Since the main MARL backbone is MAPPO, confirming effectiveness on a value-based method such as QMIX would strengthen claims of generality. 

[Reference]

[1] Eysenbach, Benjamin, et al. "Diversity is all you need: Learning skills without a reward function." arXiv preprint arXiv:1802.06070(2018).

[2] Laskin, Michael, Aravind Srinivas, and Pieter Abbeel. "Curl: Contrastive unsupervised representations for reinforcement learning." International conference on machine learning. PMLR, 2020.

[3] Lee, Sunwoo, et al. "Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning." International conference on machine learning. PMLR, 2025.

### Questions
1. I’m curious whether the proposed method given its multiple components and the need to train models such as transformers reduces computational cost (training time) compared to prior approaches like ROMANCE.

[Minor typos]

1. The math fonts are inconsistent across equations—for example, ($adv$, $\mathrm{adv}$) and ($style$, $\mathrm{style}$). It would be helpful to standardize the notation throughout.

2. In Equation (12), it is written as $f_{\psi}^{\mathrm{global}}$, but it appears this should be $f_{\gamma}^{\mathrm{global}}$.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles robustness in cooperative multi-agent reinforcement learning when a subset of teammates can be adversarially perturbed a limited number of times. The authors propose a co-evolutionary framework combining (1) DAAG: a style-conditioned adversary population generator that uses a mutual-information based objective to encourage behavioral diversity and adversarial quality; and (2) CAPE: a contrastive policy encoder that learns discriminative embeddings of encountered adversarial policies (local + global encoders, contrastive loss, and instance-relation distillation as continual learning regularization). Training alternates between evolving adversaries and updating the cooperative policy (MAPPO backbone) with the learned embeddings as additional context. Experiments (SMAC maps, Predator-Prey) show improved robustness to unseen/stronger attackers and no degradation in clean (no-attack) settings.

### Strengths
1. Addresses a practically important and timely problem in cooperative MARL (robustness to compromised teammates). 

2. Improves on adversary-cooperative alternating style robust MARL work and proposes a more fine-grained, end-to-end framework. Method combines complementary mechanisms (adversary diversity + policy encoding) in a principled way and integrates nicely with standard RL backbones (e.g., MAPPO). 

3. Empirical evaluation includes unseen adversary tests, ablations, and embedding analyses that support the claims.

### Weaknesses
The main weakness lies in the unclear novelty and limited empirical analysis. 

DAAG and CAPE combine familiar ideas of adversary generative co-evolution and contrastive representation, which are also seen in prior works like ROMANCE, but the paper does not highlight what is fundamentally new beyond integrating them (e.g. the mechanism that contributes to more efficient training and versatile anti-perturbation performance compared to prior works). The mutual-information "style" objective and IRD regularizer lack quantitative evidence of their claimed effects, such as improved diversity or embedding stability. The training alternation between adversaries and the cooperative policy is also insufficiently explained; how CAPE updates interact with adversary evolution, and whether this alleviates forgetting, remains unclear. 

Experiments are restricted to small SMAC and Predator–Prey tasks, with no sufficient details on how the unseen test adversaries differ from training ones or on computational cost comparisons. Ablations are too coarse to isolate key components (e.g., IRD, local vs. global encoders), leaving it unclear which mechanisms are most significant to robustness. Besides, more experiment on generalized perturbation setting (e.g. >=2 adversaries being simultaneously compromised) and computation cost comparisons against prior works will help better validate the effectiveness of your work.

### Questions
1. How does the test adversary pool differ from the training adversary population (e.g. pool scale, explicit illustration of the statistical difference) ?

2. How does your method behave when multiple teammates are simultaneously replaced by adversaries during episodes?

3. What is the computational cost of the proposed method compared to the baselines (e.g. MAPPO and ROMANCE) ?

4. Request for more fine-grained ablations. E.g. remove IRD while keeping contrastive loss; study individual contributions of the local and global encoders.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel framework for robust MARL that jointly trains cooperative agents and adversarial counterparts through a co-evolutionary paradigm. The proposed methods are the DAAG and CAPE. DAAG produces behaviorally distinct adversarial agents via an information-theoretic objective maximizing mutual information between latent style codes and generated trajectories, and the CAPE employs contrastive and continual learning to encode adversarial behaviors into discriminative policy representations that guide adaptive coordination. Experiments on Predator–Prey and SMAC benchmarks demonstrate gains in robustness and generalization compared to existing MARL baselines.

### Strengths
- It seems to be an interesting method to leverage mutual information to create diverse adversarial agents for improving the robustness of the MARL.
- The proposed methods seem to achieve better performance compared to the correctly selected baselines.
- The paper is relatively well written and easy to follow.

### Weaknesses
- Even though there is an analysis in Figure 3 regarding the difference between different styles, it is unclear how it translates qualitatively to agent behavior. It would be beneficial to have some qualitative analysis of the behavior of the agents (e.g., in SMAC).

- More ablation would be beneficial. It is not clear how the IRD loss affects the overall training. How does it improve the performance of the method? Each of the DAAG and CAPE methods seems to have many parts; it is unclear how each part of those methods affects the overall performance.

- It is also unclear how the proposed method behaves under the traditional adversarial agents that are optimized for the worst rewards

- In Figure 1 right side, the contrastive loss box is unclear, as there is no input to the loss box.

### Questions
- How does the proposed method perform when encountering the adversarial agents that are used in the baselines where adversarial agents are optimized for minimizing the reward only?

- In equation 14, for "a not equal to i", do you mean "a not equal to k"? Or does it mean the negative sample should not come from the same agent? It is unclear here; please clarify.

### Soundness
4

### Presentation
3

### Contribution
4
