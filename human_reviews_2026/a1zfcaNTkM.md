# ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning

- Decision: Accept (Poster)
- Scores: 8, 8, 2, 8

## Abstract
Long-horizon embodied planning is challenging because the world does not only change through an agent's actions: exogenous processes (e.g., water heating, dominoes cascading) unfold concurrently with the agent's actions. We propose a framework for abstract world models that jointly learns (i) symbolic state representations and (ii) causal processes for both endogenous actions and exogenous mechanisms. Each causal process models the time course of a stochastic cause-effect relation. We learn these world models from limited data via variational Bayesian inference combined with LLM proposals. Across five simulated tabletop robotics environments, the learned models enable fast planning that generalizes to held-out tasks with more objects and more complex goals, outperforming a range of baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces ExoPredicator, a framework for learning abstract world models that capture both endogenous (agent-controlled) and exogenous (environment-driven) causal processes. Unlike standard world models or symbolic planners that assume all changes stem from the agent’s actions, ExoPredicator explicitly models external processes that unfold concurrently over time (e.g., water boiling after switching on a kettle).

The framework represents states as collections of predicates grounded by vision-language models (VLMs) and describes temporal dynamics via causal processes—structured tuples encoding conditions, effects, and probabilistic delay distributions. The system jointly learns (i) state abstractions (via LLM-guided predicate invention) and (ii) causal process models (via Bayesian structure and parameter learning with variational inference).

Experiments across five simulated tabletop robotics environments (Coffee, Grow, Boil, Domino, and Fan) show that ExoPredicator learns accurate symbolic models from limited data, enabling efficient A* planning over “big-step” transitions and outperforming hierarchical RL, VLM-planning, and STRIPS-style baselines. Ablations indicate that both Bayesian model selection and LLM guidance are crucial for performance.

### Strengths
1. Strong motivation and relevance. The paper tackles the important problem of integrating state and temporal abstraction in world modeling, particularly addressing exogenous processes that evolve independently of agent actions.
2. Elegant conceptual framing. The formulation of causal processes as structured tuples with probabilistic delays provides a clean and general formalism for modeling both endogenous and exogenous dynamics.
3. Expressive yet practical abstraction. Defining abstract states through predicates queried via VLMs strikes a balance between symbolic expressivity and perceptual grounding.
4. Clear mechanistic interpretation. The notion of a big-step transition function is intuitive and well-motivated, allowing the planner to reason at a temporally coarse level.
5. The use of a variational lower bound for parameter estimation and Bayesian model selection for structure learning supports data-efficient model discovery.
6. Innovative use of LLMs. The integration of LLMs for proposing new symbolic forms of causal processes (Sec. 5.2) and predicates (Sec. 5.3) is a creative and well-executed design choice.
7. Well-designed experiments. The set of five environments captures a diverse range of exogenous phenomena, and the ablation studies provide useful insight into which components matter most.
8. Empirical results are convincing. ExoPredicator achieves strong generalization to unseen tasks, often outperforming baselines despite limited training data.

### Weaknesses
1. Questionable novelty claim. The paper argues that prior models (e.g., Dreamer, world models) “do not consider exogenous processes.” This is somewhat overstated. Model-based RL frameworks routinely model all sources of state transition—including environment-driven stochasticity. 
2.  Overstatement regarding options/skills. The claim that options or skills cannot capture the temporal evolution of exogenous dynamics may be too strong. In principle, semi-MDP formulations with temporally extended actions can incorporate control-exogenous effects into the transition model, although this is not their main focus.
3. Assumption of known skill set. The framework assumes the availability of predefined high-level skills (Pick, Place, etc.), which limits autonomy and may constrain generalization to new embodiments.
4. Limited empirical diversity. While the environments are well-constructed, they are all tabletop simulation tasks with relatively low-dimensional perceptual spaces. It remains unclear how well the method scales to real-world or visually richer domains.

### Questions
1. What is the practical motivation of the frame axiom potential for enforcing state persistence over time? 
2. Variational Representation: How are the variational distributions in sec. 5.1 represented and parameterized in practice?
3. When prompting the LLM to propose symbolic forms for exogenous processes, what kinds of templates or grammatical constraints are imposed?
4. Minimum Description Length: How is the MDL prior or description length of a causal process computed? Is it based on symbolic length, number of predicates, or something else?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces ExoPredicator, a framework for learning abstract world models that capture both endogenous skills and exogenous causal processes with stochastic and possibly delayed effects. The model jointly learns a symbolic state abstraction via predicate invention and a library of causal processes whose conditions, effects, and delay distributions are inferred from limited trajectory data using variational Bayesian inference guided by language model proposals. Planning operates in big steps with an A star search over high level actions, including a NoOp action that allows the agent to wait for exogenous dynamics to unfold. Across five PyBullet tabletop domains Coffee, Grow, Boil, Domino, and Fan the method achieves high solve rates and generalizes to harder held out tasks with more objects and more complex goals. It outperforms vision language planning, hierarchical RL, and STRIPS style operator learning baselines and includes ablations that highlight the roles of Bayesian model selection and LLM guidance.

### Strengths
- Novel problem framing that explicitly models exogenous processes with stochastic and delayed effects within an abstract world model. This goes beyond standard STRIPS or option based abstractions that assume instantaneous effects and agent centric dynamics.
- Integrated learning and planning system. The method couples predicate invention, Bayesian structure and parameter learning for causal processes, and a big step planner that reasons over concurrent exogenous dynamics via a NoOp wait action.
- Principled inference. The paper derives a variational objective using arrival time latent variables to avoid fine grained simulation and learns both process weights and delay distributions. The planning semantics are precise and the big step transition is specified with care.
- Empirical breadth and generalization. Strong results across five simulated domains with generalization to harder held out tasks. Convergence in at most three online iterations with only one or two demonstration seeded training tasks per seed shows good sample efficiency.
- Comparisons and ablations. Consistent improvements over ViLA, MAPLE, and VisPred. Ablations show that both LLM guidance and Bayesian model selection matter. Manual delay parameters underperform learned ones which supports the value of the inference component.
- Clarity of problem setup. The paper carefully distinguishes endogenous skills from exogenous processes and motivates why temporal abstraction for world dynamics is needed for long horizon planning.

### Weaknesses
- Evaluation is limited to simulated tabletop worlds with built in closed loop skills. There is no validation on real robots or with noisy perception which leaves open questions about robustness and deployment.
- Only three random seeds and a modest number of training tasks are used. While this highlights sample efficiency it makes statistical confidence harder to assess. Confidence intervals and significance tests are not reported.
- The approach relies on a fixed library of low level skills and on a proprietary LLM for proposals. This raises reproducibility concerns and limits analysis of end to end learning without pre specified skills.
- The failure case in Boil indicates that learned conditions can miss important disjunctive structure. More stress tests on overlapping or interacting exogenous processes would strengthen the case.

### Questions
- Clarify the assumed family for delay distributions. Are they discrete on timesteps or continuous then discretized. What priors are used for these distributions and for process weights.
- Provide more details on the segmentation and clustering used before process discovery. How sensitive is learning to the clip length and the clustering hyperparameters.
- Report wall clock times per online iteration and per plan for each environment. Include counts of LLM calls and average cost to understand practicality.
- How does the planner choose the duration to wait with NoOp. Is the wait until first abstract change always optimal or could waiting through an intermediate change be beneficial.
- How does the framework prevent spurious processes from being added when predicates are noisy. Is there any calibration of VLM predicate accuracy or robustness to false positives and false negatives.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper tackles long-horizon planning in dynamic environments where changes arise both from an agent’s actions and from independent external processes. It proposes a framework for learning abstract world models that capture symbolic state representations and causal processes with delayed effects.

### Strengths
The paper addresses a timely and important problem in robotics: learning abstract world models that can reason over both agent-driven and environment-driven dynamics. The motivation is clear and well-grounded, and the overall direction aligns with the community’s growing interest in long-horizon planning.  Experiments on simulated tabletop tasks show improved generalization compared to several baselines. However, the framework’s current complexity and heavy mathematical formulation limit its accessibility. A major revision to simplify the presentation or improve clarity would make the contribution more broadly useful.

### Weaknesses
- Overly complex framework and poor readability. The methodology is overloaded with notation and nested definitions that make it extremely difficult to follow. Many variables and terms are introduced without clear motivation, explanation, or citation. It is unclear which components are novel and which are borrowed or adapted from prior works. While the motivation in the abstract and introduction is sound, the formalism feels unnecessarily complicated for the problem being addressed.

- Insufficient experimental validation. The experiments are minimal and do not justify the complexity of the proposed framework. The evaluation focuses mainly on overall success rate, without ablations or detailed analyses that could demonstrate the contribution of each component. More fine-grained experiments are needed to show why the proposed system must be this complex and how each module contributes to its performance.

- Conceptual confusion around endogenous vs. exogenous processes. The paper defines exogenous processes as those that occur independently of the agent’s actions, yet uses “filling water” as an example, which is directly caused by the agent turning on the faucet. This blurs the distinction between endogenous and exogenous mechanisms. In practice, the framework seems to capture delayed consequences of the agent’s own actions rather than truly external processes. The paper would benefit from a clearer conceptual definition and concrete examples that distinguish these cases.

### Questions
See Weaknesses

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes to explicitly model state and temporal abstractions of *exogenous* natural processes, those not controlled by the agent directly, separately from the *endogenous* processes that are result of the agent actions. They propose an approach that leverages a VLM to process the perceptual input  and LLM to propose and filter potential predicates. They evaluate their method in simulated robotic manipulation experiments where this exogenous dynamics are part of the task.

### Strengths
- The paper is generally well-written and motivated
- The proposes a full approach that leverages LLMs and VLMs to improve sample efficiency in robotics planning.
- The evaluation is done in varied robotic manipulation environment and includes baselines that cover diverse approaches to solve these problems such as VLM planning, HRL and ablations to their own method. 
- Their method effectively outperforms the baselines in both sample efficiency and performance
Overall this paper introduces an effective way to use modern LLMs and VLMs to create abstractions for planning that handles the processes that are not fully controlled, include natural dynamics and that have delayed effects.

### Weaknesses
- It is clear that searching in the space of predicates is extremely costly and intractable via brute-force. It seems that LLMs provide a way to amortize the cost of it. However, it still seems a very large search problem. How computationally expensive is the method in comparison to some of the baselines? 
- The notation is quite cluttered and it's a bit hard to parse, though it can be understood. Perhaps some rework on simplifying notation could help the reader and I'd encourage the authors to do it.

### Questions
See above

### Soundness
4

### Presentation
4

### Contribution
3
