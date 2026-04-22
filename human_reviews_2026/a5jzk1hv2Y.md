# Automating the Refinement of Reinforcement Learning Specifications

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Logical specifications have been shown to help reinforcement learning algorithms in achieving complex tasks. However, when a task is under-specified, agents might fail to learn useful policies. In this work, we explore the possibility of improving coarse-grained logical specifications via an exploration-guided strategy. We propose **AutoSpec**, a framework that searches for a logical specification refinement whose satisfaction implies satisfaction of the original specification, but which provides additional guidance therefore making it easier for reinforcement learning algorithms to learn useful policies. **AutoSpec** is applicable to reinforcement learning tasks specified via the SpectRL specification logic. We exploit the compositional nature of specifications written in SpectRL, and design four refinement procedures that modify the abstract graph of the specification by either refining its existing edge specifications or by introducing new edge specifications. We prove that all four procedures maintain specification soundness, i.e. any trajectory satisfying the refined specification also satisfies the original. We then show how **AutoSpec** can be integrated with existing reinforcement learning algorithms for learning policies from logical specifications. Our experiments demonstrate that **AutoSpec** yields promising improvements in terms of the complexity of control tasks that can be solved, when refined logical specifications produced by **AutoSpec** are utilized.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces AutoSpec, a framework designed to automatically refine logical specifications for reinforcement learning (RL) tasks. While logical specifications can guide RL agents towards complex goals, a common issue is that coarse-grained or under-specified definitions may prevent agents from learning useful policies.

The core idea of AutoSpec is to use an "exploration-guided strategy" to automatically search for a more detailed specification. This "refined" specification is stricter than the original but provides additional guidance to the RL algorithm, making the learning process easier. Crucially, the framework guarantees "soundness": any trajectory satisfying the new specification must also satisfy the original coarse one. Theoretical justifications are provided. Experiments demonstrate that agents using specifications refined by AutoSpec can solve more complex control tasks than before.

### Strengths
1. The issue of "specification too coarse to learn from" is a significant practical hurdle in specification-based RL. This paper tackles this problem directly, which is highly valuable.
2. The primary contribution of AutoSpec is its ability to refine specifications without human intervention. This greatly lowers the barrier to using logical specifications, which might otherwise require extensive manual tuning by domain experts. Also, theoretical justifications are provided to show the framework doesn't just modify specifications arbitrarily; it guarantees that the refined specification is a valid "subset" of the original. This is crucial.
3. The refinement processes to abstract graphs are clear and intuitive, the order of refinement are also reasonable. Overall, the algorithm is easy to follow.

### Weaknesses
1. I am not fully convinced by the name *SpectRL* specification logic. To me, they are just standard subset of Linear Temporal Logic (LTL), clarifying this in the paper clearly is sufficient. We do not really need a separate notation.
2. The paper mentions that AutoSpec searches for a refinement in order. How large is this search space for each environment, and what is the overhead of this additional procedure? These are not mentioned in the paper.
3. As discussed in the paper, the reliability of the AutoSpec are heavily dependent on its base algorithm. This limitation is understandable, yet it would be better if additional discussion and proposals can be provided.
4. As I mentioned in weaknesses 1, this work only discusses a subset of LTL, while other works have already proposed some insights and solutions to the problem [1, 2]. Please consider add discussion of these works and give your own insights.
5. This is minor, I noticed that two important concepts mentioned in the abstract never mentioned again in the main text. Please explain  what are "under-specified" and "exploration-guided strategy" in the main text. I know these are the summarizations of later component, but it is better to make them clear.

[1] Qiu et al. Instructing goal-conditioned reinforcement learning agents with temporal logic objectives. NeurIPS 2023.

[2] Jackermeier and Abate, DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL. ICLR 2025.

### Questions
1. See weaknesses 2, please explain the search space and overhead of AutoSpec.
2. See weaknesses 3, please explain the difficulty of refinement in AutoSpec.
3. See weaknesses 4, can you discuss [1] and [2] in the paper, and provide some insights on whether AutoSpec can be applied to $ \omega $-regular LTL specifications?
4. It seems that AutoSpec detect the failures of a policy and perform refinement, this is good. However, it is possible to perform "active" refinement rather than "passive"? This could be very interesting.

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
4

### Summary
The paper considers the problem of automatically refining logical specifications in order to help specification guided reinforcement learning algorithms. The main intuition is that when the specification is very coarse, these algorithms find it hard to learn effective policies. So they propose identifying problematic specifications and refining them to help the algorithms converge faster and also help with guided exploration.

This work uses the SpectRL specification logic which can be represented as a graph that captures different ways to satisfy the specification. They present several types of refinement procedures that modify the graph and these procedures consist of refining goal/target regions, or adding additional intermediate target regions.

In their experiments they show that their method greatly helps specification guided RL algorithms to learn effective policies in large gridworld environments as well as robotic manipulation task with obstacles.

### Strengths
This is one of the first few works to consider the problem of automatic refinement of RL specifications based on collected feedback from the training of policies. This is a fundamental issue because if the specification is too coarse grained then algorithms would find it hard to effectively explore the state space and learn good policies.

The problem they consider is studied in depth and many refinement techniques are proposed that are sound. The benchmarks they consider are also interesting. This work also opens up many interesting related directions that can be explored.

### Weaknesses
While the contributions of the paper are substantive, they can be presented better. The introduction can be expanded to give further intuition with respect to the problem being solved. Specifically, the notion of abstract graph is never introduced informally even though it is central to the paper. Perhaps it would be helpful to take the example in figure 1, and present in some detail about what the refinement procedures would produce and how it would make the learning task easier. Similarly, logical specifications for RL are also never introduced.

The related work section can also be organized better into paragraphs.

### Questions
1. Are there possible failure modes where the refinement procedure would follow a wrong chain of refinements that make the learning task much harder? Perhaps this deserves a short discussion?

2. In the current algorithms, the different refinements are applied in a specific order. Do you imagine situations where this order can be detrimental? 

3. Why or why not dynamically choose which refinement to apply at each step?

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
5

### Summary
Specification guided reinforcement learning often fails when initial logical task descriptions and their labeling functions are too coarse. The paper proposes $\mathrm{AutoSpec}$, a framework that refines SpectRL specifications through an exploration driven loop and four refinement procedures, while guaranteeing that any trajectory that satisfies the refined specification also satisfies the original specification. The method integrates with existing algorithms and is demonstrated with $\mathrm{DIRL}$ and $\mathrm{LSTS}$, where refinements help recover learnability on tasks that were previously hard to solve.

### Strengths
- The paper addresses an important challenge in formulating specifications for RL. Automatically refining predicates and specifications provided to the agent is a promising direction, particularly because crafting appropriate predicates is difficult and loosely defined specifications can be hard to satisfy.
- The framework is integrated with established specification guided algorithms and the experiments illustrate how the refinements interact with the different exploration strategies of $\mathrm{DIRL}$ and $\mathrm{LSTS}$.

### Weaknesses
- The empirical scope is narrow. Only two domains are considered, n Rooms and PandaGym. This limits the evidence for scalability and diversity of specifications. A broader study that samples many predicate regions or includes a less contrived multi room world would strengthen the case.
    - The specifications tested upon are quite limited (only 1 or 2 refinements per specification needed). A sample driven testing approach (say randomly chosen predicate regions) or a less contrived 100 room example would be more convincing of the scalability of the approach.
    - I appreciate the carefully chosen experiments for an intuition of what is happening, but some further generalization studies would help (e.g. more than 2 refinements needed and whether $\mathrm{AutoSpec}$ covers the search space appropriately).
- When there are no successful samples, certain refinements cannot be computed, as observed for $\mathrm{LSTS}$ on the complex specification. The approach is sound but not complete and it may fail to find a refinement even if one exists.
- `AvoidRefine` only enlarges the avoid set or equivalently reduces the safe set, without permitting relaxations when the avoid region is overly conservative. Algorithm 3 defines the refined safe region by removing the convex hull of recent failure states, which can bias the learner away from potentially optimal paths if the initial avoid labeling is narrow or misaligned.  This is acceptable in most situations,  but the onus is on the user specifying the initial predicate regions to start with conservative definitions. A discussion of when to relax an avoid constraint would be valuable.

### Questions
1. What is the computational overhead of $\mathrm{AutoSpec}$ in the reported settings, relative to running $\mathrm{DIRL}$  or $\mathrm{LSTS}$  alone? A wall clock comparison and a complexity view in terms of the number of edges and sampled trajectories per refinement would help readers assess practical costs.
2. How do the procedures behave when a specification needs several consecutive refinements? Is there an observed depth beyond which refinements fail to improve satisfaction probability or become unstable?

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
4

### Summary
The authors propose a method for automatically refining specifications defined using the SpectRL framework and that are used for specification-guided reinforcement learning. Their technique produces a provable refinement of the original specification (a trace satisfying the refined specification implies that the trace also satisfies the original specification). By using the refined specifications to retrain the RL policies, the authors observe that the newly trained policies have higher specification satisfaction rates. In other words, better specifications result in better policies.

### Strengths
- Problem statement is relevant, interesting, and well defined.
- The authors do a good job of explaining the preliminary material needed to understand their work.
- The results show that their method significantly improves the performance of the re-trained policies to meet the original specification after refinement.

### Weaknesses
- Some of the presentation of the AutoSpec framework is lacking... specifically Figure 2 really doesn't clarify what the PastRefine refinement procedure does. What's the relationship between the two parts of the figure? It also leaves a ton of open whitespace, which looks sloppy.
- It would be nice to have visuals on how each of the refinement procedures is working, not just PastRefine.
- The experimentation and presentation of it is lacking significantly
	- Only use 2 experimental setups (n-Rooms and PandaGym)
	- There are 2 tunable hyperparameters (the probability threshold and the number of traces to sample) the details of which are never mentioned for their experiments.
	- They never discuss the cost of doing the specification refinement (how long does it take? does retraining the policy take as long?) This would mean training 2x since we have to train all over again to integrate specification refinement.
	- They never describe how many times the experiment was attempted. Did they train a bunch of different policies and try it multiple times? Or are the results they show just from training one policy for each of the specification-guided RL algorithms mentioned and trying their framework on it? I am assuming the former, which is limited experimentation in my opinion.
	- They don't compare to any other specification refinement or generation techniques for specification-guided RL.
	- Figure 4 is a plot of "Best Path Cost" on the y-axis vs. "Number of Timesteps" on the x-axis. They never introduce what path cost is or what it means to have best path cost. How should these plots be interpreted?
	- Figure 5 doesn't have any labels on the x and y axes, so it is unclear what results are being demonstrated.

Minor comments / typos:
- Missing space between guarantees and citation (Lechner et al.) on pg. 2, line 104
- I believe the "AddRefine: Introducing Waypoints." part should be given a new line in section 3.1. All of the other specification refinement procedures are given their own new lines when introduced.

The paper introduces a compelling strategy for improving specification-guided RL by refining the specifications, but it lacks strong experimentation to be convincing. While apparently theoretically sound, much more experimentation would be necessary (more experimental setups, try on more specification-guided RL algorithms and for multiple different trained policies, a stronger ablation study than shown in section 4.3 by again running more experiments, also experiments controlling the tunable hyperparameters, experiments comparing to other related methods, and better presentation of the results).

### Questions
Please address and discuss weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3
