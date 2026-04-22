# AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Current deep reinforcement learning (DRL) approaches achieve state-of-the-art performance in various domains, but struggle with data efficiency compared to human learning, which leverages core priors about objects and their interactions. Active inference offers a principled framework for integrating sensory information with prior knowledge to learn a world model and quantify the uncertainty of its own beliefs and predictions. However, active inference models are usually crafted for a single task with bespoke knowledge, so they lack the domain flexibility typical of DRL approaches. To bridge this gap, we propose a novel architecture that integrates a minimal yet expressive set of core priors about object-centric dynamics and interactions to accelerate learning in low-data regimes. The resulting approach, which we call AXIOM, combines the usual data efficiency and interpretability of Bayesian approaches with the across-task generalization usually associated with DRL. AXIOM represents scenes as compositions of objects, whose dynamics are modeled as piecewise linear trajectories that capture sparse object-object interactions. The structure of the generative model is expanded online by growing and learning mixture models from single events and periodically refined through Bayesian model reduction to induce generalization. AXIOM learns to play various games within only 10,000 interaction steps, with both a small number of parameters compared to DRL, and without the computational expense of gradient-based optimization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces AXIOM, a fully Bayesian, gradient-free reinforcement learning agent that learns from pixels within 10,000 interaction steps. AXIOM models environments as interacting objects through expanding mixture modules for perception, identity, dynamics, and interactions, updating online via variational inference and pruning with Bayesian Model Reduction. Using active inference for action selection, it is evaluated on the proposed Gameworld 10k benchmark of ten object-centric games, where it achieves faster and more data-efficient learning than DreamerV3 and BBF while maintaining interpretable object-level representations.

### Strengths
1. Semantically meaningful representations: AXIOM’s vector tokens correspond to interpretable physical quantities such as position, color, velocity, and size, providing a level of semantic transparency rarely achieved in prior slot-based object-centric models.

2. Direct comparison with strong baselines: The paper includes quantitative evaluations against state-of-the-art agents such as BBF and DreamerV3.

3. Introduction of a new benchmark: The authors propose Gameworld 10k, a suite of ten minimalist, object-centric games designed to systematically study sample efficiency, generalization, and object-centric reasoning in low-data regimes.

4. Comprehensive methodological exposition: The paper offers a detailed and well-structured description of AXIOM’s components—including perception, identity, dynamics, and interaction models—supported by ablations and qualitative analyses that clarify each module’s function and contribution.

### Weaknesses
1. Limited evaluation scope: The experiments are restricted to the newly introduced Gameworld 10k benchmark, leaving the method’s scalability and effectiveness on more complex or standard environments (e.g., Atari, DMControl, MuJoCo) untested.

2. Potential baseline comparability concerns: Since DreamerV3 and BBF were trained for the new environment, performance differences may partially stem from adaptation or implementation details rather than fundamental algorithmic superiority.

3. Presentation and qualitative analysis limitations: The visualization in Figure 1 could be clearer in illustrating the model’s structure and information flow. And given AXIOM’s strong interpretability claims, the paper would benefit from more extensive qualitative examples demonstrating object tracking, dynamics understanding, and policy reasoning.

### Questions
1. Generality of empirical validation: Could the authors extend evaluation to standard benchmarks such as Atari/Procgen/etc to demonstrate AXIOM’s effectiveness and scalability in more complex, visually diverse, and long-horizon environments? I understand the main contribution, but if the method cannot be further scaled, then the value of the claims would be diminished.

2. Necessity of the object-centric assumption: To what extent is the object-centric prior essential for AXIOM’s functioning? Could the same expanding Bayesian framework and active inference mechanism be adapted to environments without explicit object structure or physical dynamics, and if so, what modifications would be required?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents AXIOM, an object-centric, Bayesian world model that learns directly from pixels without gradient updates. It decomposes perception and dynamics into modular mixtures for object discovery, type identification, motion prediction, and interaction modeling, with adaptive expansion and pruning through Bayesian model reduction. Using expected free-energy planning, AXIOM achieves fast, sample-efficient learning on the custom Gameworld-10k benchmark, outperforming baselines like DreamerV3 and BBF.

### Strengths
**Strengths**

**1. Good Writing:** The paper presents a clear, fully probabilistic framework that decomposes perception, dynamics, and interaction into modular mixture components. AXIOM’s architecture is transparent, where each latent variable has a defined physical or semantic meaning (slot, type, mode, interaction).

**2. High sample-efficiency:** Within only 10k interaction steps, AXIOM achieves competent performance across multiple tasks, often surpassing baselines such as DreamerV3 and BBF. This demonstrates the practical advantage of its object-centric, non-gradient Bayesian updates, which yield faster credit assignment and stable learning in low-data regimes.

### Weaknesses
**Weaknesses:**

**1. Generalization to complex tasks:** The Gameworld-10k suite is tailored to object-centric, sparse-interaction dynamics with low visual complexity. While useful for probing the proposed priors, it risks design–method coupling and may inflate relative gains versus deep baselines optimized for high-dimensional, long-horizon settings. Claims of generality are not justified without external baselines. The paper acknowledges not scaling to “complicated control tasks typical of the RL literature,” which limits generalization claims. On the top of that, methods like Dreamer/BBF are known to benefit from larger encoders and replay; compressing inputs (resizing, pooling), altering frame-skip, and a small data budget can handicap them.

**2. Priors encode environment knowledge:** The observation pipeline (pixel tokens with coordinates, nearest-neighbor interaction features, fixed radii) injects strong inductive bias that suits the Gameworld physics.The model’s success appears heavily contingent on its encoded inductive biases like explicit object decomposition, locally linear dynamics, and spatial interaction priors. While these assumptions fit the Gameworld environments almost perfectly, they effectively hard-code much of the task’s physical structure and visual simplicity. Consequently, AXIOM’s reported sample efficiency may reflect prior alignment rather than genuinely better inference or planning. Although the authors briefly acknowledge that AXIOM’s object-centric and locally linear priors restrict its applicability to simple domains, this is treated more as future work than as a central limitation. The paper lacks quantitative evidence on how sensitive performance is to these priors or how gracefully the model degrades when they are violated.

### Questions
1. If you remove pixel coordinates and nearest-neighbor features, can a learned CNN and attention recover performance?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes to use mixture models (MM) over object-centric representations to learn a dynamics model. The MMs provide a flexible way to expand to additional objects (during testing/training). The authors also propose a benchmark -- Gameworld 10K which has a 10000 steps budget for the agent to achieve the maximum score. Because they use MMs, the number of learnable parameters are significantly smaller when compared to the current state-of-the-art model based RL approaches (like Dreamer). They show fast adaptation to newer object shapes and colors in their experiments.

### Strengths
1. A novel way to employ model-based planning (without any neural networks or gradient optimization) that can potentially, in the future be an avenue for fast adaptation.

2. I appreciate that the authors provided anonymized code -- I had a brief look at it.

### Weaknesses
1. The core claim of "robustness to environmental perturbations" is not necessarily applicable to AXIOM in particular. As the authors point out, Dreamer and AXIOM are both similarly robust to such perturbations, and BBF instead outperforms both when it comes to robustness. So, I'm not fully convinced of this claim of robustness.

2. There are too many components in the model -- which isn't inherently a bad thing -- however, I wonder if this will scale up to more realistic observations. For instance, can this model generalize and outperform Dreamer on robotic suites like Robosuite / ManiSkill for object manipulation?

3. For a benchmark, to compare different algorithms, I'd like to see IQM [1] metrics since, in certain environments (Figure 3), it is hard to gauge if AXIOM is significantly better than Dreamer/BBF.

----

**References:**

[1] Deep Reinforcement Learning at the Edge of the Statistical Precipice, Rishabh Agarwal et al., NeurIPS 2021

### Questions
4. Why is there a need for this benchmark? HackAtari (https://github.com/k4ntz/HackAtari), for instance, enables environment modifications, and since ATARI 100k is an existing and well-known benchmark, creating it for the purpose of showcasing AXIOM work is not a compelling reason.

5. Why is the reward not a function of the next state $\mathcal{O}_{t+1}^k$ and instead predicted from rMM? I'd like the authors to explain their rationale behind this.


6. Which model is being used for DreamerV3? Is it the base model?

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
4

### Summary
The paper develops a model consisting of multiple mixture components for doing reinforcement learning. It develops a formalism for training that large mixture model.

### Strengths
The paper defines a large model with many different mixture model components for learning a variational posterior on observing trajectories within an RL problems. It applies variational inference and a mixture component split merge structure to develop an inferential procedure that can be used in planning. Instead of learning to optimize reward from the outset, it uses an approximate Bayesian approach for concurrent world modelling and refinement of parameter distribution training. Arguably this concurrency provides a big win in terms of data-efficiency.

### Weaknesses
(Please sort out the references - there is a significant lack of care in the references, capitalisation is all over the places - Gauss is a proper noun etc. This does not reflect well on the work). The paper has an overabundance of gratuitous references to the work of Karl Friston. Bayesian agent architectures have been around for decades prior to Parr et Al. Beliefs are always updated incrementally as new
evidence emerges, it doesn't need another Friston reference to establish that. Nor is mixture adding and merging a Friston idea (SMEM did this in 1998, followed up by many others). And again throughout the paper. One might be forgiven for believing this paper was an exercise in H-index hacking. 

Altogether, I do not see any indication as to how the authors relate this to other Bayesian reinforcement learning implementations, or to general model-based reinforcement learning, and world modelling. Despite the copious references to Friston, there seems to be incredibly little reference to most other Bayesian RL work at all? see e.g. the review of Kang, Tobbler and Dayan, and many more. There is decades of work in this area, and this work should really be set in the context of the existing RL literature. The model itself is multiple factorised mixture models, with a heuristic updating scheme for the structure. The exact relationship to Bayesian agents is a little lost in the actual formalism. There is some confusion as to be Bayesian (over model parameters) and having probabilistic state updates (which this shares with most POMDP related formalisms). The planning aspect is a little cryptic, and lacks specifics. 

The paper would be improved if it explicitly and precisely defined its problem setting, assumptions, and way the results are assessed. There is no problem statement at all. 

I don't believe the priors over parameters has been defined here.

Does this relate strongly to the "RL as inference" angle of Mark Toussaint and later, Surgey Levine?

The recurrent mixture model captures most of the structure of the model. It deals with the reward structure. i.e. the strong dependence of the instantaneous reward on the state, which has to be learnt. Yet it is quite worrying as that seems to be handled in 7 as an _independent_ categorical distribution between state and reward. This can't make sense, yet that is what it says. Reward modelling is critical but is not dealt with explicitly and is hidden away here. That seems quite worrying, and I wonder here is there is actually any real reward modelling happening.

The posterior factorisation assumptions seem to be worrying, in that it decouples things that are strongly coupled. The internal posteriors over the mixture parameters could be discussed further.

However the main weakness of the paper is that it dismisses evaluating the approach on any of the standard settings used in this context, instead defining their own unestablished setting just for them, so we can't tell how good it is as a test. This leads to irrelevant benchmarking, as it is close to impossible to know the performance of the proposed method. At the very least the authors should give results on existing benchmarks, critique those and propose further benchmarks.

### Questions
Please could you explain your reward modelling. This seems really unclear at the moment.

### Soundness
2

### Presentation
2

### Contribution
2
