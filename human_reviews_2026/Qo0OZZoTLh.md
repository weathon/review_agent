# Virtual Community: An Open World for Humans, Robots, and Society

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 8, 6, 8

## Abstract
The rapid progress of AI and robotics may profoundly transform society, as humans and robots begin to coexist in shared communities, bringing both opportunities and challenges. To explore this future, we present Virtual Community—an open-world platform for humans, robots, and society—built on a universal physics engine and grounded in real-world 3D scenes. With Virtual Community, we aim to enable the study of embodied social intelligence at scale. To support these, Virtual Community features: 1) An open-source multi-agent physics simulator that supports robot, human, and their interactions within a society; 2) A large‑scale, real‑world aligned environment generation pipeline, including vast outdoor space, diverse indoor scenes, and a community of grounded agents with rich characters and appearances. Leveraging Virtual Community, we propose two novel challenges. The Community Planning Challenge evaluates multi‑agent reasoning and planning in open‑world settings, such as cooperating to help agents with daily activities and efficiently connecting other agents. The Community Robot Challenge requires multiple heterogeneous robots to collaborate in solving complex open‑world tasks. We evaluate various baselines and demonstrate the challenges in both high‑level open‑world task planning and low‑level cooperation controls. We have open-sourced our project and hope that Virtual Community will unlock further study of human-robot coexistence in open worlds.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Virtual Community, an open-world simulation platform for
studying interactions between multiple embodied agents. These embodied agents
can be either robots or humans. Virtual Community includes 3D open worlds where
multiple agents can interact with each other. Creating these worlds requires a
few steps. First is generating the world itself; this step starts with
geospatial data and then refines the data to create meshes that are easy to
simulate. The meshes are also photorealistic due to careful curation of
textures. The second step is to create the agents in the world; this is done by
prompting LLMs. Each agent has certain personalities and other attributes and
also a daily schedule of activities. Finally, all these agents and several
robots are simulated in the Genesis physics engine. To motivate studying
multiple embodied agents, this paper further introduces two challenges; the
Community Planning challenge studies social interactions among humans, while the
Community Robot Challenge studies how robots can interact with humans.

### Strengths
1. It is clear to me that the authors took a lot of time to polish this paper,
   as well as their accompanying website and codebase. In particular, I am
   impressed with the level of care taken to document the code and make it easy
   for folks to get started with Virtual Community.
1. The generation pipeline makes a lot of sense. In particular, I appreciate the
   use of LLMs to generate the agents automatically. It is nice that the
   resulting environment only requires a single GPU to run episodes.
1. I overall find the simulator quite novel and believe it provides a useful
   platform for studying embodied intelligence. I think the challenges are also a good
   starting point for the community; I am curious to see what methods can be created
   to solve the current challenges, and what new challenges can be created in the future.

### Weaknesses
I honestly could not find any major weaknesses with the paper. I have left
various minor comments and questions below.

### Questions
1. Line 241: "define their embodiments" -> "given their embodiments"?
1. I think it would be helpful to provide a line or two describing the Genesis
   engine, since Genesis is mentioned several times in the paper.
1. Section 4.1 mentions that the agents use a hierarchical planner. I am clear
   that the low-level actions are for navigating between locations. What are the
   high-level actions?
1. For the community assistant tasks, are the assistant agents robots or humans?
   Overall, it might be helpful to clarify in Section 4 whether "agent" refers
   to human or robot.
1. Do the authors anticipate creating more challenges within Virtual Community
   in the future?
1. I am curious whether the authors see any relation between this work and
   recent world modeling works like Genie 3?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The goal of this paper is to introduce a simulation platform, built on top of the Genesis physics platform, that supports the simulation of urban spaces and populations of heterogeneous agents.


The contributions of this work are as follows:  
1) Urban 3D scene generation pipeline.
2) Indoor 3D scene generation pipeline.
3) Multi-agent population generation pipeline.
4) Extension of Genesis to support humanoid agents.
5) Support for a large heterogeneous population of agents.
6) Multi-agent interaction tasks: Community Planning Challenge and The Community Robot Challenge.

### Strengths
1) Significance (human-robot interaction and embodied social intelligence are crucial research topics).
2) The work advances the capabilities for embodied social intelligence simulation and research.

### Weaknesses
1) The main weakness is work presentation quality (sentence formulation and writing style). While the paper's contributions are valuable, the clarity and style of the writing could be improved to reflect the quality of the underlying work better. A couple of comments:

- L33-36: Not clear how performed experiments (evaluation of baselines on introduced tasks) help to answer questions raised in the abstract: how robots cooperate or compete, how humans form social relations and communities and how humans and robots coexist. In other words, it feels like the these questions are not aligned with the rest of the paper. A more appropriate formulation would be "With Virtual Community, we aim to enable the study of embodied social intelligence at scale".


- L37-40: "A large-scale, real-world aligned community generation pipeline, including vast outdoor space, diverse indoor scenes, and a community of grounded agents with rich characters and appearances.". Community generation pipeline includes outdoor spaces, indoor spaces and community of agents? A better formulation might be "environment generation pipeline".

- Not consistent usage of terms in describing agents: L37: Virtual Community ... "supports robots, humans" agents. Then in L91-92 ...in Community Planning challenge ... "agents interact with humans and other agents". Then L300-301: community assistant tasks "in which agents cooperatively plan to assist multiple humans". And L301 community influence task "in which agents competitively plan to efficiently connect and interact with other agents". This makes it confusing to understand what types of agents are supposed to be involved in a particular task. 

- Another semantically overloaded phrase is open world. "Open-world" - used 30 times, "open world" - used 19 times. However, it is not explicitly defined what it means in the context of the work.

2) It is hard to draw insights about embodied social intelligence from selected baselines and performed experiments. Tables demonstrate which baselines perform better compared to other baselines.

### Questions
1) Grounding Validator performs scene grounding validation (checks whether generated agent profile matches scene). What about validation of generated agent profiles and community relations, to make sure they reflect real worls scenarios?

2) What is the maximum size of the population that Virtual Community supports?

3) L83: Remove dot.

4) L1069-1073 repeats L1098-1103.

5) L1208-1209: "We utilize the according URDF file" -> "We utilize the corresponding URDF files"?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents Virtual Community, a city-scale simulation platform that unifies human avatars, multiple robot types, and social structure (profiles, schedules, relationships) within a single physics-enabled engine. It proposes two families of benchmarks: Community Planning (carry/delivery/search and a social-influence setting) and Community Robot (heterogeneous multi-robot mobile manipulation). The system couples real-world geographic data (OSM/Google) with a generative asset pipeline to produce large outdoor areas, indoor scenes, and populated communities. Baselines include heuristic/MCTS/LLM planners and IK+RRT vs. PPO for manipulation.

The work is ambitious and likely to be useful to the community, but several claims and evaluations need tightening (especially around “physically real” and manipulation without oracle priors).

### Strengths
1. Scope & Unification – A rare integration of city-scale environments, human–robot co-presence, and social graphs with time-aligned schedules, enabling questions that neither indoor-only simulators nor purely social agents can address.
2. Real-world Grounding at Scale – Uses real geographic data and a generative pipeline to create semantically rich, traversable outdoor/indoor spaces (transport, buildings, POIs). This improves task relevance for long-horizon planning.
3. Well-defined Benchmarks – Clear task families (carry/delivery/search/influence) with consistent observations/actions and success/time/follow metrics; results reveal non-trivial failure modes for LLM planners and for manipulation.
4. Transparent Baselines – A modular navigation stack, multiple planners (Random/Heuristic/MCTS/LLM), and classical IK+RRT vs. RL for manipulation provide a solid starting point and set realistic expectations.
5. Community Value – the platform can serve as a bridge between embodied AI and social simulation, attracting researchers from planning, VLA, multi-agent RL, and HRI.

### Weaknesses
1. Over-claim on “physical realism.”
The implementation appears primarily rigid-body + contact with kinematic attach/detach for human–object/vehicle interactions. There is limited validation of compliant contact, actuator dynamics, deformables/fluids, or grasp stability. The current evidence supports physics-enabled rather than physically real.

2. Manipulation relies on oracle priors; RL underperforms.
Success rates drop notably without oracle grasps; end-to-end RL struggles in sparse, long-horizon settings. This weakens the claim that the platform currently supports robust mobile manipulation in open worlds.

3. State/Progress tracking in LLM planning.
LLM planners excel at search but degrade on multi-step, progress-dependent tasks (carry/delivery). Lack of explicit memory/plan monitoring leads to mis-ordering and cost underestimation.

4. Limited quantitative validation of physics and assets.
The paper acknowledges outdoor detail gaps but does not provide measurable physical consistency tests (e.g., friction/solver sensitivity) or cross-sim comparisons to indoor high-fidelity platforms.

5. Navigation & scheduling cost modeling.
While transit/indoor-outdoor transitions are supported, there is no ablation on commute cost modeling (wait times, congestion, transfers) or task ordering policies, which likely drive failures in open-world itineraries.

### Questions
1. What exact physics features are enabled at run-time (solver, time step, iterations, contact model, friction model), and how sensitive are benchmark outcomes to them?

2. How are kinematic attachments triggered/terminated, and do they bias success metrics versus true closed-loop grasp stability?

3. Can the authors provide statistics on map connectivity and accessibility (e.g., average indoor-outdoor traversal times, transit wait distributions) and correlate them with failures?

4。 For the influence task, how are dialogue vs. target-selection effects disentangled? Any evaluation with stronger memory or planning scaffolds?

### Soundness
3

### Presentation
3

### Contribution
4
