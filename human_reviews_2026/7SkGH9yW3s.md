# Hierarchical Object-Oriented POMDP Planning for Object Rearrangement

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
We present an online planning approach and a new benchmark dataset for solving multi-object rearrangement problems in partially observable, multi-room environments. Current object rearrangement solutions, primarily based on Reinforcement Learning or hand-coded planning methods, often lack adaptability to diverse challenges. To address this limitation, we propose a Hierarchical Object-Oriented Partially Observed Markov Decision Process (HOO-POMDP) planner that leverages object-factored belief representations for efficient multi-object rearrangement. This approach comprises of (a) an object-oriented POMDP planner generating sub-goals, (b) a set of low-level policies for sub-goal achievement, and (c) an abstraction system converting the continuous low-level world into a representation suitable for abstract planning. To enable rigorous evaluation of rearrangement challenges, we introduce MultiRoomR, a comprehensive benchmark featuring diverse multi-room environments with varying degrees of partial observability (10-30\% initial visibility), blocked paths, obstructed goals, and multiple objects (10-20) distributed across 2-4 rooms. Experiments demonstrate that our system effectively handles these complex scenarios while maintaining robust performance even with imperfect perception, achieving promising results across both existing benchmarks and our new MultiRoomR dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a novel method for multi-room, multi-object rearrangement in a partially observable setting. The rearrangement process operates in two phases. In the walkthrough phase, the agent explores the environment to build a map, record receptacle locations, and gather scene information; objects are then placed randomly in the environment. In the rearrangement phase, the agent uses the constructed map, the known set of object classes, and the specified goal locations to perform the rearrangement. The approach employs a hierarchical planning framework, with a high-level planner based on PO-UCT to generate abstract actions such as move, pick, and place, and a low-level controller that uses reinforcement learning for pick-and-place behaviors and A* for navigation. The method explicitly addresses challenges posed by blocked paths and blocked goal locations, which earlier approaches do not handle effectively. Experimental results demonstrate that the system successfully operates under partial observability and scales to scenarios with up to 20 objects. Additionally, the paper introduces MultiRoomR, a benchmark featuring more rooms, more objects, and a greater variety of rearrangement configurations.

### Strengths
1. The paper is well-written and clearly structured, making the motivation and methodology easy to follow.
2. The proposed method is novel, particularly in how it discovers partially observable objects. It also effectively handles blocked-path scenarios, which prior work does not address.
3. The approach benefits from using PO-UCT as an iterative planner, which reduces dependence on learned policies, thereby lowering the risk of catastrophic failures often seen in purely neural methods.
4. The experimental evaluation is extensive and thorough, including strong comparisons to prior approaches and informative ablation studies that highlight the importance of each system component.
5. The newly introduced MultiRoomR benchmark is a valuable contribution that provides a more challenging and realistic testbed for future research on multi-room, multi-object rearrangement.
6. The authors have released the code and provided detailed algorithmic descriptions to support reproducibility, and they include the MultiRoomR dataset along with the supplementary material to facilitate future research.

### Weaknesses
1. Limited novelty in methodology: While the system is thoughtfully designed, the novelty is incremental. The method primarily combines existing components (PO-UCT, PPO-based low-level skills, and A* navigation) into a pipeline, and the core contribution lies in their integration. The main new challenge addressed is the blocked-path scenarios, which, although important, limits the conceptual innovation relative to prior hierarchical rearrangement frameworks.
2. Scalability concerns due to PO-UCT: The reliance on PO-UCT for high-level planning means computation scales with the number of simulations and search depth, which is strongly influenced by the number of objects and environment complexity. As a result, planning time may grow significantly as task complexity increases, limiting scalability to larger or more cluttered scenes.
3. Limited generalizability to real-world deployment: Because the high-level decisions are generated through Monte-Carlo tree search rather than a learned policy, the system must perform online planning at each step. This limits generalization and makes real-world application challenging, as the approach does not amortize planning into a neural policy that could execute quickly without repeated simulation.
4. Problem formulation relies on extra prior knowledge: The problem setup assumes access to the set of object classes that need to be rearranged, which is not typically available in prior works and may not be realistic in practical settings. Furthermore, the method assumes that the user provides precise 3D goal locations for objects. In real-world applications, such explicit spatial goal specifications are uncommon; instead, agents are generally expected to use commonsense semantic priors to infer plausible goal locations (e.g., as in TIDEE (Sarch et al.) or Housekeep (Kant et al.)). Integrating semantic goal inference would make the approach more practical and aligned with real-world deployment expectations.
5. Simplifying independence assumption in belief modeling: The belief update assumes objects are independent, which does not hold in real household environments where objects commonly co-occur, occlude each other, or exhibit relational structure (e.g., items inside containers or grouped functionally). This assumption restricts the model’s ability to reason about relational or compositional uncertainty and may limit performance in more complex, cluttered scenes.

### Questions
1. To better understand the scalability of the proposed approach, could you clarify the minimum number of simulations and the required search depth needed to achieve the best performance for different numbers of rooms and objects? Additionally, how does compute time grow as these parameters increase? A quantitative analysis of simulation budget vs. performance vs. time would help assess how the method scales to larger and more complex environments.
2. In the appendix Table 3, the “Time (m)” column is not clearly defined. Does this value represent the average total clock time to complete an entire rearrangement episode (planning + execution), or only the planning time? Additionally, is this reported per episode or normalized per object?
3. Have you considered incorporating semantic goal inference or preference-based goal discovery, similar to TIDEE or Housekeep?
4. Can the framework be extended to reason about correlations (e.g., object occlusions, functional groupings)? Do you observe failure cases where this assumption becomes limiting?
5. Can you provide insights into the most common failure modes? For example, is performance more sensitive to detection errors, navigation failure, or belief update ambiguity?
6. Could you comment on the feasibility of deploying this framework on a real robot?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a hierarchical planning algorithm for the object-rearrangement problem, in the presence of partial observability. The three primary components of the algorithm are: 1) high-level POMDP planning in an abstract state space, 2) macro-actions which are implemented using Markov methods, and 3) an object-factored belief state. The result is a practical and high-performing algorithm for this important robot subtask. The algorithm is thoroughly evaluated on a novel benchmark task, where its performance exceeds previous methods'.

### Strengths
- This is an important problem that displays many of the characteristics of real-robot tasks.
- The approach is practical and simple, and builds on a lot of recent work focused on exploiting structure in POMDPs to generate efficient solutions.
- The method is sufficiently structured that I expect it would work on a real robot, with some engineering effort required obviously.
 - Using Markov low-level controllers to support a more abstract level that can then plan taking partial-observability into account has appeared in a few places, and has worked very well. This is a highly appropriate use of that strategy.
- The evaluation is thorough and the new benchmark proposed is valuable.
 - The paper is well written (though not well-structured, see below) and easy to read.

### Weaknesses
- The paper is oddly structured. There's a Related Work section that occurs before the Background section, which makes no sense. How am I to understand the RW section when I haven't been precisely told what problem you are solving yet? Then the Background section is called Problem Formulation, though instead of a formulation (which is a precise mathematical description of the task), we are given some implementation details about AI2Thor. The paper and ideas have to stand on their own outside of AI2Thor. There's some problem formulation going on in section 4 (!!), where the authors define an OO-POMDP. Anyway I had to re-read the paper several times, out of order, to actually follow, which is bad.
- The authors talk about embodied AI, but there are no bodies in this paper. Also, the first sentence is a little melodramatic. I'm not sure that multi-object rearrangement in a simulator is such a fundamental challenge. It's a good problem to work on though. Anyway I would tone some of that down.
 - It's not clear what the authors mean by "hand-coded" planning approach. Do they mean someone hand-codes a plan? Or a planner? Hand-coding a planner is what the planning research community does all day! Or do they mean, hand-design for a specific case? In that case, that's what this paper does! But they seem to be criticizing prior work here, so I think some clarify about what exactly they are criticizing would be good.
 - The paper is related to structured models of POMDPs. It cites an extends OO-POMDPs, which is appropriate, but I think MOMDPs are Merlin's work on local observability is probably also relevant. 
 - Occasionally the paper uses a parenthetical citation as a noun, which should be fixed.

### Questions
In the table, why does PK not have a 100% success rate? Or maybe that's not a percentage?

Can you please confirm my understanding that the low-level policies for the macro-actions are essentially Markov? Are there any implications for that>

### Soundness
3

### Presentation
4

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
The authors propose a hierarchical solution for an Object-Oriented Partially Observable Markov Decision Process (OOPOMDP). The system decomposes the problem by using a high-level abstract POMDP planner (based on PO-UCT) to reason over object-factored belief states and generate high-level actions (e.g., MoveToLocation, PickPlaceObject). A low-level policy executor, composed of classical planners and trained RL policies, is then responsible for executing these sub-goals. They also built a multi-room evaluation on top of ProcThor, creating a new multi-room dataset for evaluation.

### Strengths
The authors demonstrate an improvement over their selected baselines of FHC, VRR, and MSS. I found the inclusion of different ablations, such as having an oracle belief or a perfect object detector, helps validate their design choices. Their multi-room benchmark also highlights the limitations of the methods they chose to compare against.

### Weaknesses
My main concerns are as follows:
* The object independence assumption seems to directly conflict with trying to improve performance on scenarios involving blocked goals or blocked paths. 
* Requiring a pre-task walkthrough to build a map of the static environment is a significant practical limitation. This makes the solution inapplicable to new environments and the methodologies' robustness to any change in the map, or if there were blocking objects during this phase of planning. 
* The error analysis clearly identifies that the low-level pick/place policies are a major bottleneck, which makes it hard to isolate the planning performance.
* The manual initialization of possible abstract actions (described in Abstract OOPOMDP Planner) limits the system's generality. The planner does not discover actions but instead searches over a pre-defined set based on object locations. This would limit the tasks to which this can be applied.
* The paper's presentation is often unclear, making it difficult to fully grasp the methodology and its positioning within related works. Key details of the planning and abstraction systems are distributed between the main text and the appendix, hindering readability.
* The anonymous code link provided in the paper is expired.

### Questions
* How does FHC perform with an Oracle object detector?
* Do other methods require an initial walk-through of the environment?
* How sensitive is the planner to an imperfect static map from the "walkthrough phase"? What happens if a static object, like a chair, is moved between the walkthrough and the rearrangement phase?
* Given that low-level policy failures are a key bottleneck, do you have any data on the planner's success rate? For example, during failed episodes, does the planner still produce a semantically correct sequence of abstract sub-goals?
* Could you clarify the algorithmic novelty of your algorithm's planner? The paper positions it as an extension of OO-POMDPs from search to rearrangement. Is the primary difference simply the inclusion of PickPlace actions and their corresponding belief updates?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a Hierarchical Object-Oriented Partially Observable Markov Decision Process (HOO-POMDP) planner for multi-object (scene) rearrangement problems. This includes a Object-Oriented POMDP planner for generating sub-goals, with low-level policies for achieving these sub-goals and a method for converting the low-level world into a representation for abstract planning. In addition, the paper presents a benchmark simulation environment (called MultiRoomR) of multi-room environments with different levels of partial observability, obstructions, and multiple objects. The authors present results of experimentally evaluating the HOO-POMDP against baselines in the context of various benchmarks.

### Strengths
(i) The paper focuses on the problem of multi-object rearrangement, and formulates it as a probabilistic sequential decision-making problem under partial observability.

(ii) The paper provides a new benchmark for multi-object rearrangement in the form of multi-room environments.

### Weaknesses
(i) One key problem with the paper is that it makes claims that are not fully substantiated. For example, the authors mention that object rearrangement solutions are based on RL and hand-coded planning methods; it is not clear what the authors mean by "hand-coded planning methods", but there are many other ways of solving this problem. Also, existing probabilistic planners can handle large domains and uncertainty resulting from partial observability. 

(ii) The discussion of related work unfortunately seems to be limited to the more recent RL-based or deep networks-based methods; there is no acknowledgement or discussion of the rich literature in hierarchical planning, including those based on RL and POMDPs, e.g., [1-3]. The existing papers have already explored  factorization of state space (and belief updates) in complex domains. The authors need to clarify how their proposed approach is different and makes a new contribution.

(iii) Following up on previous points: it is unclear if/how the proposed approach is an hierarchical planner. The proposed pipeline processes sensory inputs to obtain concepts that are then used to compute a plan for the rearrangement task. This is the standard pipeline in many AI systems that process sensor inputs and plan actions. The "abstraction system" is just a manually encoded method that is executed to map the sensor inputs to the representation for POMDP planning. This can be contrasted with prior work on hierarchical planning with POMDPs [1, 2] where there is an actual hierarchy of POMDPs and tasks to be performed. In addition, these systems support complex state spaces, uncertainty in perception and actuation, different objective functions (e.g., optimize for travel distance and time), and have been used for planning on physical robot platforms.

(iv) From the description of problem formulation and the algorithms, it is not clear if there is any actual uncertainty in actuation; even the uncertainty in perception is captured by a very simple model, and the "partial observability" seems to be a reference to the fact that the agent's view of the domain is limited at any point in time. It is also not clear why RL policies are needed for the low-level execution of simple movements in a simulation environment.

(v) In the experimental evaluation, the baselines seem to be systems that pose the rearrangement task as a learning problem (e.g., with deep networks), or seek to complete the task after removing the limited uncertainty introduced in perception. It is also not clear what it means to "remove the hierarchical planning" in the HOOP-HP baseline, and how it impacts the corresponding state (also action, observation) space. Given the limited noise in the system, it is unclear why there is no comparison with a state of the art classical planning or probabilistic planning system. Finally, the statistical significance of the results shown in Table 1 is unclear.

[1] Joelle Pineau and Sebastian Thrun. High-level Robot Behavior Control using POMDPs. National Conference on Artificial Intelligence (AAAI), 2002.

[2] Mohan Sridharan, Jeremy Wyatt and Richard Dearden. Planning to See: A Hierarchical Approach to Planning Visual Actions on a Robot using POMDPs. Artificial Intelligence, 174 (11):704-725, 2010.

[3] Harsha Kokel, Sriraam Natarajan, Balaraman Ravindran, and Prasad Tadepalli. RePReL: A Unified Framework for Integrating Relational Planning and Reinforcement Learning for Effective Abstraction in Discrete and Continuous Domains. Neural Computing and Applications, 35: 16877-16892, 2023.

### Questions
Please address comments in the "weaknesses" section above.

### Soundness
2

### Presentation
3

### Contribution
2
