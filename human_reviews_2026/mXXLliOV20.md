# Unveiling Complex Collective Behaviors from Simple Rewards

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Recent studies have shown that multiple agents trained through reinforcement
learning can surprisingly exhibit swarm behaviors from simple rewards, without any
rewards specifically encouraging aggregation. Explaining how complex collective
behavior emerges from these simple rewards is an intriguing research problem,
but the underlying process remains a black box up to now. This paper aims to
reveal the hidden rules in this process. Specifically, we discovered that the reason
agents are able to develop complex behaviors from simple rewards is that they
implicitly learn the geometric fields of the environment and utilize these structures
as desired targets for coordinated movement. This finding is supported by two
distinct tasks: a competitive predator-prey pursuit and a cooperative multi-robot
shape assembly. 1) In the competitive environment, prey agents surprisingly converge
toward the boundary of the predators’ Voronoi diagram, demonstrating that they
are able to spontaneously learn Voronoi diagrams without any guided rewards. To
gain the above insights, we propose a two-stage EEC (Ego-observation → Egobehavior
→ Collective-behavior) explanatory framework. This includes a novel
analytical tool called the Agent Response Map (ARM), which reveals agents’
decision-making patterns across space and identifies regions of aggregation and
avoidance. 2) The proposed method is extended to a more realistic
and challenging cooperative robot-swarm task: Shape assembly, to validate its
generality and practical utility. The insights and tools presented in this paper
may provide a new perspective on the connection between AI-driven multi-agent
systems and real-world biological systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce approaches to analyze multi-agent reinforcement learning experiments, referred to as EEC. These include Ego-observation -> Ego-behavior, followed by Ego-behavior -> Collective behavior. The first stage of this analysis is the application of SHAP to trained actor and critic networks. In the second stage, a so-called Agent Response Map (ARM) is constructed, showing how agents respond to different scenarios within a simulation in order to shed light on their collective behavior. 

In the paper, the authors focus on a predator-prey model with the prey punished for being captured and the predators being rewarded for capturing. As input, the agents receive information about their position along with the relative position of the other agents in the simulation. During ablation studies, this information is limited. The authors find via SHAP analysis that relative position is the essential ingredient in producing collective behavior in the prey. Further, they identify that the prey appear to move towards the edges of the Voronoi diagram formed using the predator agents as nodes. An additional study is mentioned relating to a robotics task, but the results are not presented in the main paper.

The approach to analyzing the learned policy can be seen as an interesting result and contribution of the paper. The learned policy of the prey, essentially to maximize their distance from predators, is nice to see, but not necessarily of great surprise.

### Strengths
The paper tackles an important problem in reinforcement learning, namely, better understanding the emergent policy of agents. Further, the visual representation of the policy analysis makes the outcomes clear to readers. Further, by explaining the emergent policy of the agents, the authors shed light on a common RL problem.

### Weaknesses
There were several concerns raised while reading the paper. First, the concerns regarding the study will be addressed:

- A predator is mentioned in the abstract before any mention of the kinds of problems that were studied
- At times, it felt as though the prior work was not adequately addressed. Ignoring the fact that the section does not appear in the main paper, even in the appendix, and with the language used in the main text, there are works approaching this problem on very similar systems. E.g., https://arxiv.org/pdf/2110.01307, 10.1088/2632-2153/ad5f73 as two notable examples.
- In some cases, the analysis is convoluted. Discussing Voronoi diagrams is reasonable, but is the central point that the agents maintain a maximum distance from the predators? This is a far simpler explanation for what appears to be the same result.
- Due to the problem addressed, while interesting, the results are not particularly novel.
- Throughout the paper, as will be addressed below, critical details such as the RL algorithm, information on PBC, and other factors are simply not discussed. This makes understanding the results difficult and reproducibility all but impossible.

Outside of the study, it is also important to mention the following concerns:

- Moving related work to the appendix is not an appropriate way to reduce the size of a paper. Those rules are put in place for a reason. This also goes for removing the RL theory from the paper. It is never clearly stated what kind of RL is being studied. The authors mention actor and critic, thereby insinuating it, but no details are provided. This also goes for the lack of detail regarding the simulations and reproducibility. This is also done in the case of the final study in a robotic setting. Overall, the paper cannot be considered self-contained.

### Questions
- Is the “sight” 360 degrees?
- Is the position given to the agents global? This would contradict the statement “the agents must coordinate their motion using only local information.”
- Is the ARM computed for fixed predator positions (This is explained later to be true, but needs to be clarified before Figure 2). In Figure 2, are the authors using PBC or not? Judging by the trajectories, it seems like they are. This changes the expected outcome and is thus critical information. Again, I believe this is mentioned later in the paper, but if the authors wish to include and mention these results in the figure, this information should be introduced at that point.
- In the absence of relative position information, is it surprising that the agents cannot form groups?
- When saying, “equidistant from the predator along two opposing directions,” is it meant equidistant between two predators? Hence, along the Voronoi diagram formed using those predators as nodes? This would come down to the agents maximizing their distance from both predators, not exactly a surprising result, although admittedly interesting that the agents learned this form of geometry given the reward used.
- Can the authors explain why the angular velocity is so high along the Voronoi lines? I would have expected that the agents would stop turning in these regions and move relatively straight

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the question of how complex collective behaviors, such as herding and flocking, can emerge in multi-agent reinforcement learning (MARL) systems trained with only simple survival-based rewards. The authors propose a two-stage explanatory framework, EEC (Ego-observation → Ego-behavior → Collective behavior), to dissect this process. The core contributions are: 1) a novel visualization tool called the Agent Response Map (ARM), which probes and maps the learned policy across the environment space, and 2) the surprising discovery that prey agents learn to converge towards the boundaries of the predators' Voronoi diagram, which acts as a "line attractor" or a dynamic safety zone. The paper uses SHAP and ablation studies to identify critical observational features and validates its findings in a predator-prey environment, with a brief extension to a robot shape-assembly task.

### Strengths
1.  The paper tackles a fundamental question at the intersection of AI, biology, and complex systems: the emergence of collective intelligence from simple, local rules. Providing a mechanistic explanation for this phenomenon in a modern MARL context could be a useful enabler for analyzing future algorithms.

2.  The proposed Agent Response Map (ARM) is a highly intuitive tool for interpreting MARL policies. Visualizing the policy's output as a field over the state space provides a global, geometric understanding of agent decision-making that is hard to obtain from trajectory analysis alone. This is a methodological contribution to MARL interpretability.

3.  The discovery that agents implicitly learn and exploit the Voronoi diagram of predators is interesting. The framing of this boundary as a "line attractor" is a powerful and clear explanatory concept.

### Weaknesses
1.  The work is dependent on the periodic boundary conditions and simple dynamics, which create a situation where a predator can approach from two opposing directions. Further, predator-prey problems in 2D are quite the special case and may not be of huge generality. The line attractor property is not theoretically backed.

2.  The extension to the multi-robot shape assembly task in Section 5 feels disconnected from the main analysis. The section is very brief, with most details relegated to the appendix. It does not clarify whether an analogous geometric structure (like the Voronoi diagram) emerges or how the EEC framework provides unique insights in that context.

3.  The ARM is constructed using a "virtual probe" agent with zero velocity. While this is a clever way to isolate the policy's spatial response, a trained policy is conditioned on the full state, including velocity.

### Questions
1.  Could you elaborate on the role of the periodic boundary conditions in the emergence of the Voronoi diagram as an attractor? Have you run experiments in a large, un-walled environment or on other topologies? If so, does a similar geometric structure emerge, or do the agents adopt a different strategy?

2.  Did you experiment with using a probe agent with non-zero (e.g., average) velocity? How sensitive is the structure of the ARM to the probe agent's own state, and would this change the interpretation of the results?

3.  The paper shows agents converge to this solution, but could you speculate more on why the MARL algorithm finds this specific Voronoi-based strategy? Is it demonstrably the most optimal path, or is it simply always the case for the simple problem dynamics with continuous policies?

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
3

### Summary
This paper investigates why complex swarm-like behaviors emerge in multi-agent reinforcement learning (MARL) systems trained only with simple survival-pressure rewards, without any explicit incentives for grouping. The authors propose a two-stage explanatory framework, EEC (Ego-observation → Ego-behavior → Collective behavior), and introduce a visualization tool called the Agent Response Map (ARM) to reveal how individual policies respond spatially across the environment. Through SHAP-based feature attribution and ablation analysis, they show that prey agents’ actions are dominated by the relative position of the nearest predator rather than by neighbors’ orientations. ARM further demonstrates that agents implicitly learn to align along the Voronoi boundaries of predators, forming a line attractor that explains aggregation and coordination. The framework is also applied to a robotic shape-assembly task to illustrate generality.

### Strengths
1. Clear and timely problem framing. The paper targets an interesting and underexplored question — explaining emergent collective behavior in MARL.

2. Interpretability framework. The combination of SHAP analysis and the proposed ARM visualization provides new insight into how local observations translate into swarm-level structure.

3. Mechanistic insight. Discovering that agents implicitly learn the predators’ Voronoi boundaries as safety zones is a compelling and intuitive explanation for emergent coordination.


4. Readable and well-structured. The paper is easy to follow and the figures effectively communicate the key findings.

### Weaknesses
1. Limited generality. The entire analysis is based on a predator–prey pursuit–evasion setting with homogeneous agents. It remains unclear whether the same mechanism explains other emergent behaviors in heterogeneous or task-driven MARL.

2. Primarily interpretive, not algorithmic. The work does not propose new learning methods or measurable performance improvements; its contribution is mainly analytical.

3. Dependence on specific geometric assumptions. The observed Voronoi-boundary alignment may arise from the planar and isotropic design of the environment, and might not hold in 3-D or irregular terrains.

4. Qualitative validation of ARM. Although ARM is visually compelling, its evaluation is descriptive rather than quantitative; a more formal measure of explanatory power would strengthen the claim.

5. Computational scalability. The framework requires repeated policy queries across the full spatial grid, which may become infeasible for high-dimensional environments.

### Questions
1. How sensitive are the ARM findings to the number of predators or to sensor noise in observation features?

2. Could the same EEC + ARM framework explain collective behaviors in cooperative tasks without adversaries (e.g., flock navigation or foraging)?

3. Is the “Voronoi line attractor” still observable when agents have non-Euclidean sensing or act in 3-D space?

4. Can ARM be extended to quantify causal influence rather than correlation？

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
3

### Summary
This paper investigates why complex collective behaviors can spontaneously emerge in multi-agent reinforcement learning (MARL) environments even when agents are only trained with simple survival/catching rewards. The key contributions are:
1. The paper identifies an intuitive yet unexpected mechanism: agents implicitly learn the geometric risk field of the environment, especially the Voronoi boundaries induced by predators, and use these boundaries as coordination targets. Consequently, agents spontaneously gather along these boundaries, which act as linear attractors. This finding is supported through visualization and empirical analysis.
2. The authors propose a two-stage explanatory pipeline EEC. Stage 1 identifies key sensory inputs via SHAP-based attribution and controllability tests. Stage 2 introduces the Agent Response Map (ARM), a new tool using a virtual invisible probe to scan space, visualize policy responses, and reveal spatial discontinuities and agents’attractor lines.
3. The method is further demonstrated in a shape-assembly task with robotic agents, showing that the framework generalizes beyond the predator–prey domain and has potential for practical multi-robot coordination.

### Strengths
Originality is strong. The idea that agents implicitly learn and exploit geometric invariants (Voronoi boundaries) is novel and not widely explored in MARL interpretability. The ARM tool is a new methodological contribution that bridges local policy interpretation with global spatial structure.

Quality is good, but could be stronger. The authors combine attribution analysis (SHAP), ablation tests, and spatial visualization (ARM) effectively. The experiments cover multiple scenarios—bounded vs. periodic maps, predator vs. no-predator, and transfer to robot shape assembly. However, the main claim (“Voronoi boundaries act as attractors”) is still mostly supported by visual trends, not rigorous quantitative evidence. The results are suggestive but need more formal validation (e.g., dynamical analysis, statistical robustness).

Clarity is above average. The main storyline is clear and supported by good visuals (ARM heatmaps, SHAP plots, time–distance graphs). The EEC pipeline is conceptually clean. That said, several implementation details (e.g., network structure, hyperparameters, training seeds, ARM sampling strategy) are deferred to the appendix or missing altogether, which makes reproducibility difficult. Moreover, some claims (like “implicit Voronoi learning”) would benefit from clearer quantitative definitions.

Significance is high. Understanding why emergent collective behavior arises under simple rewards is a long-standing question in MARL, robotics, and swarm intelligence. If validated, the proposed mechanism could reshape how we interpret learned coordination and provide a useful diagnostic tool (ARM) for multi-agent systems.

### Weaknesses
Lack of quantitative proof for the “line attractor” / Voronoi learning claim

Suggetions:

1. Attractor Quantification: Introduce perturbation experiments around Voronoi boundaries. Measure return-to-boundary time and convergence speed compared with baselines (random policies, Vicsek-style local rules). If Voronoi boundaries are true attractors, trajectories should converge faster and more consistently.

2. Local Linearization: Compute local Jacobians of the policy-induced velocity field along the boundary (e.g., via finite differences) to show contraction in the perpendicular direction.

3. Alternative Boundary Comparison: Compare Voronoi boundaries with alternative geometric features (e.g., iso-value contours of the critic or potential field) to demonstrate that Voronoi is indeed the most predictive structure. 

Insufficient robustness and quantitative metrics for ARM

Suggestions:

1. Conduct parameter sensitivity analysis for ARM: vary the probe’s initial orientation, speed, neighbor count, and input ordering to verify that results are consistent.

2. Define a quantitative metric for ARM discontinuity (e.g., spatial gradient variance) and report it across seeds to establish reproducibility.

3. Perform predictive validation: use ARM maps to predict where agents will aggregate in actual rollouts and measure the prediction accuracy.

Missing Baseline Comparisons

Suggestions:

1. Compare against classical local-interaction models (Vicsek, Couzin, Reynolds) trained or tuned under the same reward conditions to test whether similar Voronoi clustering emerges.

2. Extend the explanation method comparison beyond SHAP: e.g., causal feature ablation (random permutation or noise injection) to confirm that input importance reflects causality, not just correlation.

Reproducibility Gaps

Suggestions:

1. Provide full training details (architecture, optimizer, learning rate, batch size, seeds, hardware, total steps).

2. Release code or demo scripts to ensure transparency and facilitate community adoption.

Limited discussion on generalization and real-world constraints

Suggestions:
1. Expand shape-assembly experiments to non-convex or multi-component target shapes and report whether ARM still identifies meaningful geometric attractors.

2. Discuss (or test) how occlusion, noise, or communication delay affects the learned geometry—important for practical multi-robot coordination.

### Questions
1. Can you provide statistics on how agents return to Voronoi boundaries after perturbation, compared with random baselines?

2. How sensitive are ARM maps to probe orientation, initial position, or sampling resolution?

3. How are Voronoi boundaries computed under periodic boundary conditions or multiple predators? Are they always continuous?

4. How many random seeds were used? What is the variance in emergent behavior across runs?

5. How do results change with different prey/predator ratios or varying perceptual neighbor counts?

6. How does your learned strategy differ quantitatively from rule-based models (in clustering degree, survival rate, or energy efficiency)?

### Soundness
3

### Presentation
3

### Contribution
3
