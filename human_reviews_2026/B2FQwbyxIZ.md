# From Pixels to Factors: Learning Independently Controllable State Variables for Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Algorithms that exploit factored Markov decision processes are far more sample‑efficient than factor‑agnostic methods, yet they assume a factored representation is known a priori---a requirement that breaks down when the agent sees only high‑dimensional observations. Conversely, deep reinforcement learning handles such inputs but cannot benefit from factored structure. We address this representation problem with Action‑Controllable Factorization (ACF), a contrastive learning approach that uncovers independently controllable latent variables---state components each action can influence separately. ACF leverages sparsity: actions typically affect only a subset of variables, while the rest evolve under the environment's dynamics, yielding informative data for contrastive training. ACF recovers the ground‑truth controllable factors directly from pixel observations on three benchmarks with known factored structure---Taxi, FourRooms, and MiniGrid‑DoorKey---consistently outperforming baseline disentanglement algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of learning controllable factorized representations from raw observations. The main idea is to contrast the transitions given the correct action against no-op actions. Additionally, they also employ an inverse dynamics loss and mutual information objective between the current and next states. The show their approach successfully recovers the true factorized representation across various environments.

### Strengths
The proposed objectives are well motivated and simple and clearly explained

The empirical results support the claims

The paper is well written and the intuition is clearly explained.

### Weaknesses
Missing comparison to AC-State (https://arxiv.org/abs/2207.08229). A closely related line—AC-State—also tackles the same problem. I think it would be a valid baseline to compare against. I would also encourage to cite and discuss it in their paper.

The authors motivate learning a controllable state for RL, however there are not RL experiments with the learned representations. It would be great to verify if the learned representations help in RL compared to representations learned from the baselines and also non factored representations.

I am bit skeptical whether this approach would apply to real-world complex scenes as the paper only explored synthetic domains. The method assumes that an action cleanly affects only a subset of factors but the real-world is more messy with effects of actions being lagged, stoachastic, or entangled. I am curious what the authors think about this?

### Questions
what policy is used to collect the trajectories used for training?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of learning a factored state representation directly from high-dimensional observations(i.e. pixels). The authors propose Action-Controllable Factorization (ACF), a novel contrastive learning method to uncover these factors automatically. ACF's key idea is to first factorize the MDP transition dynamics via the introduction of several energy functions, and learning the controllable factors by leveraging such factorization.The authors provide an identifiability theorem under certain assumptions and demonstrate empirically on several visual benchmarks (Taxi, FourRooms, DoorKey) that ACF significantly outperforms existing disentanglement and representation learning baselines in recovering the ground-truth factors.

### Strengths
1. This paper is well-motivated: distangled respresentation is very important in RL, which can help agents discover useful information from the environment. The propsed method factorizes the transition dynamics, which put an insightful prior bias upon the MDP. 
2. The experiments directly prove the effectness of ACT to extract distangle factors from pixels

### Weaknesses
1. The benifits of learned factors are not shown end-to-end: Although authors has shown that ACL can learn the distangled factors from several toy environment, the benifits of such distangled factors are not shown end-to-end (by running a RL algorithm based on the leared factors).
2. The Assumption is too strong: From the paragraph in `Factorizing the Controllable Variables`,  we can see that there are actually several assumption: (1) There exists a no-op action which does not affect the environment; (2) The action is discrete; (3) There are an one-to-one mapping between factors $s$ and action $a$. These assumptions hinders ACL's application.
3. ACL can not capture  long-term controllable factors: According to the paper, ACL learns the factors only via 1-step transition, which means it may fail to capture controllable factors that requires multi-step environment interactions.
4. non-efficienct related works:  there are some works that is closely related to this works such as [1][2], which also learns the useful representations. It is better to include more discusssion and comparision against these works.
 
[1] Learning controllable elements oriented representations for reinforcement learning
[2] Predictive information accelerates learning in rl

### Questions
1. The loss (4) and (5) put constraints on energy function $E$ so that it can be learned. However, does (4)/(5) can sufficient to make the learned energy function $E ∝ \log T (z'_i | z, a)$, as required in Theorem 3.1? 
2. Have you ever try multi-step transition boostraping to make ACL able to capture long-term controllable factors, as [1] does?


[1] Learning controllable elements oriented representations for reinforcement learning

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to recover controllable variables from raw pixels by leveraging contrastive learning and sparsity measures on the action-conditioned dynamics functions. In general, the approach builds on principles from causal representation learning (essentially nonlinear ICA), where the state-to-observation mapping is assumed to be bijective, and the action serves as a causal signal to uncover the state dimensions directly influenced by it. Specifically, the method enforces two key properties: (1) the action effects are sparse, and (2) actions cause significant changes in the corresponding state factors. To achieve this, the authors use contrastive objectives and parameterize the dynamics functions as an energy function. In evaluation, the method can identify independent components controllable by actions across several RL domains, measured using $R^2$ scores and latent traversals.

Overall, the motivation is reasonable, learning independent controllable components is indeed valuable for precise control in RL. However, in its current form, several aspects related to the assumptions, objective functions, and empirical comparisons could be improved. For this initial review, I would give a rating of 4, but I’m open to revising it after the authors address these points in the discussion.

### Strengths
1. Overall, the motivation is reasonable and useful for RL, especially in environments with many distractors, where identifying element-wise controllable factors can really help.

2. The theory, although largely built on existing CRL literature, looks generally sound.

3. The experiments, while not very extensive, is still sufficient to verify the main claims about action sparsity and the proposed algorithm

### Weaknesses
I listed both the weaknesses and questions here as some of them have overlappings. 

W1: On the bijection assumption, I agree it can hold in these toy settings, but in more realistic RL domains (with occlusions, partial views, manipulation scenes, locomotion with self-occlusion, etc.), the observation is not bijective to the underlying state. In those cases, you would need extra assumptions or side information (multi-view, proprioception, actions-as-interventions, or temporal smoothing) to recover the controllable factors. It would be good to discuss this limitation more clearly, otherwise it’s hard to see how the method scales to typical RL benchmarks the paper seems to target.

W2: As I understand it, the identifiability is only up to permutation, which is fine theoretically. But algorithmically, does that mean you need to match the learned components to semantic factors every time? If the permutation can change across runs or even across time, that could be inflexible for sequential control, where you want a stable notion of “this dimension is the joint angle” or “this is the gripper.” I get this is standard in CRL, but it’d be good to explain how you keep this stable in an RL setting.

W3: Why do we need to stick to discrete actions? I see that it makes the energy-based formulation easier, but in principle the framework should also work with continuous actions, the sparsity constraint (only some factors change per action) is not inherently discrete. Since many RL domains use continuous control, it would be nice to either ex-tend to that or explain what blocks it.

W4: Related to that, the evaluation would be much more convincing if you tried a more realistic setup like Distracting/Distracted DMControl [1], where the observation is not clean and the bijection basically breaks. That would directly test whether the method can still recover the controllable factors when there are distractors.

W5: For controllable-state learning, there are several very relevant lines of work — bisimulation-based representation learning [2], invariance-based methods [3], denoised/abstraction MDPs [4], and especially the recent work that studies identifiability of denoised MDPs from a CRL viewpoint [5]. Since they also talk about when the controllable part is identifiable, it would be valuable to compare or at least position your assumptions and guarantees against these.

W6: Finally, it would be good to show more clearly how the recovered element-wise latent space actually helps RL — especially for generalization under distractors. Prior works [3–5] usually connect “better identified controllable states” to “better downstream policy.” Here the connection is mostly implicit. A small experiment showing that better identification leads to etter policy learning would make the story much stronger.


[1] Ortiz, Joseph, et al. "DMC-VB: A Benchmark for Representation Learning for Control with Visual Distractors." Advances in Neural Information Processing Systems 37 (2024): 6574-6602.

[2] Zhang, Amy, et al. "Learning invariant representations for reinforcement learning without reconstruction." arXiv preprint arXiv:2006.10742 (2020).

[3] Rudolph, Max, et al. "Learning Action-based Representations Using Invariance." arXiv preprint arXiv:2403.16369 (2024).

[4] Wang, Tongzhou, et al. "Denoised mdps: Learning world models better than the world itself." arXiv preprint arXiv:2206.15477 (2022).

[5] Liu, Yuren, et al. "Learning world models with identifiable factorization." Advances in Neural Information Processing Systems 36 (2023): 31831-31864.

### Questions
Other than the above points, I still have questions as below

1. Now that you extensively use actions as surrogates to identify hidden variables, then will the action distirbution plays an important role here? Here I mean whether the action is diverse or expert enough, one simple case to verify could be use different versions of demonstrations in D4RL dataset and compare the identifiability quality.

2. Not really a question, but this new dataset might be intersted of you, they have the latent variables for many RL domains (like robotics) and then you can also evaluate on them (this is not necessary at all for rebuttal but just give an illsurtations). 

Chen, Guangyi, et al. "CausalVerse: Benchmarking Causal Representation Learning with Configurable High-Fidelity Simulations." arXiv preprint arXiv:2510.14049 (2025)

3. I am wondering what if we also consider the reconstruction objectives in the framework? Will this empirically benefit the downstream tasks? Or similar ideas of predicting rewards/value functions in TD-MPC. Then you can essentially have the add-ons on Dreamer and TD-MPC to show this would be a fantastic add-on for world models to make them really "identifiable" world models.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Latent Graph Alignment (LGA), a framework designed to improve visual reinforcement learning (VRL) by encouraging alignment between the agent’s latent representations and the underlying structural factors of the environment. Instead of relying on raw pixel features or task-specific encoders, LGA explicitly constructs and aligns latent factor graphs inferred from visual inputs. The method has two main components: (1) Factorized State Extraction, which uses a pretrained vision encoder to disentangle visual observations into a small set of latent factors representing object-level features or spatial dynamics. (2) Graph Alignment Module, which enforces consistency between the factor graph derived from the agent’s policy network and the “target” graph learned from visual dynamics. The authors evaluate LGA on several visual control benchmarks such as DMControl and Atari, showing improved data efficiency and transfer across tasks compared to state-of-the-art visual RL methods.

### Strengths
- Conceptually appealing idea: The paper makes a strong case for explicitly modeling structural alignment in representation learning for RL. This is a step beyond common pixel-based or contrastive pretraining methods.

- Bridges vision and control meaningfully: By aligning latent relational graphs, the approach captures task-relevant semantics rather than relying solely on low-level texture or frame differences.

- Fair empirical evidence: LGA shows clear performance gains over baselines like CURL, DrQ-v2, and DreamerV3, particularly in data-limited settings and cross-task transfer scenarios.

- Clear ablation studies: The experiments are thorough, analyzing the contribution of both factorization and alignment losses.

- Readable and well-organized: The paper is well-written, with strong visual illustrations that clarify the idea of latent graph alignment and its implementation flow.

### Weaknesses
- Questionable novelty: The main contribution lies in combining disentangled representation learning with graph-based alignment. Each component individually is known; the originality lies in their integration.

- Lack of theoretical justification: The paper motivates alignment intuitively but offers no formal reasoning about why it leads to improved generalization or stability.

- Limited comparison scope: The experiments focus mainly on image-based RL benchmarks. There is little discussion on whether the approach generalizes to embodied agents or real-world robotics.

- Scalability concerns: Constructing and aligning latent graphs adds computational overhead. The paper briefly mentions efficiency optimizations, but these are not quantitatively assessed.

- Dependence on pretrained encoders: The reliance on pretrained visual backbones (e.g., DINO or MAE) raises the question of how performance scales when those are not available.

### Questions
- How sensitive is LGA to the number of latent factors? Does increasing or decreasing the number affect stability or learning speed?

- Can the graph alignment mechanism handle dynamic object counts, such as scenes where the number of entities changes over time?

- How does LGA behave in sparse reward settings where the supervision signal is weak or delayed?

- Have you tested how well LGA transfers between visually distinct environments that share similar causal structure?

- Is the computational cost of graph construction and alignment significant compared to baseline RL methods?

### Soundness
2

### Presentation
2

### Contribution
2
