# MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 8, 6, 6, 6

## Abstract
Mobile manipulators in households must both navigate and manipulate. This requires a compact, semantically rich scene representation that captures where objects are, how they function, and which parts are actionable. Scene graphs are a natural choice, yet prior work often separates spatial and functional relations, treats scenes as static snapshots without object states or temporal updates, and overlooks information most relevant for accomplishing the current task. To overcome these shortcomings, we introduce MomaGraph, a unified scene representation for embodied agents that integrates spatial-functional relationships and part-level interactive elements. However, advancing such a representation requires both suitable data and rigorous evaluation, which have been largely missing. To address this, we construct MomaGraph-Scenes, the first large-scale dataset of richly annotated, task-driven scene graphs in household environments, and design MomaGraph-Bench, a systematic evaluation suite spanning six reasoning capabilities from high-level planning to fine-grained scene understanding. Built upon this foundation, we further develop MomaGraph-R1, a 7B vision–language model trained with reinforcement learning on MomaGraph-Scenes. MomaGraph-R1 predicts task-oriented scene graphs and serves as a zero-shot task planner under a Graph-then-Plan framework. Extensive experiments show that our model achieves state-of-the-art results among open-source models, reaching 71.6% accuracy on the benchmark (+11.4% over the best baseline), while generalizing across public benchmarks and transferring effectively to real-robot experiments. More visualizations and robot demonstrations are available at https://hybridrobotics.github.io/MomaGraph/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes 1) MomaGraph, a unified scene representation that integrates both spatial and functional relationships, 2) MomaGraph-Scenes, a dataset of annotated, task driven scene graphs in household environments, 3) MomaGraph-Bench, a VQA benchmark, and 4) MomaGraph-R1, a 7B vision-language model RL fine-tuned on MomaGraph-Scenes. The paper shows MomaGraph-R1 is the best performing open-source model on MomaGraph-Bench. The paper validates the system in real-world on a humanoid in unseen environments.

### Strengths
- The paper is clearly written, motivates the contents progressively, and thus easy-to-read. 
- The paper introduces a lot of content, from unified scene graph representation, to dataset, to benchmark, to model.

### Weaknesses
- Aside from its own proposed task, the paper only evaluated on one external benchmark, BLINK. It would be more compelling to see the scene graph and model being compared on more commonly used benchmark such as Habitat and Matterport 3D. 
- While the paper introduced the mechanism of dynamic scene graph update (Section 4.3), the defined task is static visual question answer, instead of interactive embodied question answer.

### Questions
- From the definition of MomaGraph (Section 4.1), the method is constrained to a single indoor room. Is there any particular for the constraint?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes MomaGraph, a task-conditioned scene-graph representation that unifies spatial and functional relations and includes part-level interactive nodes. Authors introduce (i) MomaGraph-Scenes, a new dataset of richly annotated, task-driven scene subgraphs with multi-view images; (ii) MomaGraph-Bench, a multi-choice VQA benchmark covering six reasoning capabilities from action sequencing to visual correspondence; and (iii) MomaGraph-R1, a 7B VLM trained with RL (DAPO) using custom graph-alignment reward to predict task-oriented scene graphs and then plan under a Graph-then-Plan paradigm. The method achieves 71.6% acc on MomaGraph-Bench and narrows gap to closed-source models, with +11.4% over Qwen2.5-VL-7B base. Also show it transfers to real-robot tasks (cabinet, microwave, TV, light).

### Strengths
- well-motivated problem as there are clear limitations in existing scene graphs (single relationship types, static scenes, etc) with convincing preliminary experiments
- I like how the work provides a complete comprehensive pipeline from representation design to dataset collection, to model training with RL, to benchmark construction, to real-world deployment
- I think the joint modeling of spatial + functional relationships with part-level nodes is intuitive and well-executed
- Strong empirical results (+11.4% over base model) and I like the comprehensive evaluation including in-person

### Weaknesses
- state-aware dynamic updates are presented as a major contribution but have no experimental evaluation. How often are these updates needed? How accurate are they? I would like to see ablation comparing task success with dynamic updates vs without. I also would like to see examples of success vs. failed disambiguation and in what number % of cases, this disambiguation was needed.
- the dataset size of 1050 graphs seems modest. This may limit the generalization to diverse households or household tasks.
- missing comparison with OpenFunGraph, a directly comparable method for functional scene graph generation with part-level interactive elements.

### Questions
- How sensitive is performance to the reward function weights (wa, wf, wl in Eq. 2)?
- What is distribution of failure modes? Would like to see some type of fail case analysis of this system
- What is the computational cot of graph generation at inference time?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper targets the pain point that (i) representations that are “spatial-only” or “function-only” cannot robustly support long-horizon, executable tasks, and (ii) VLMs tend to hallucinate when asked to directly plan. In response, it proposes **MomaGraph**, a **part-level, state-aware unified task scene graph** \(G_T=(N_T, E_T^{s}, E_T^{f})\). After each interaction step, it **dynamically updates** the functional/spatial relations via  
\(G_T^{(t+1)} = U(G_T^{(t)}, a_t, s_{t+1})\),  
so that an initial “one-to-many” hypothesis (one goal, many possible objects/affordances) progressively converges toward “one-to-one”.  

The reasoning paradigm is **Graph-then-Plan**: multi-view observations + instruction → generate the task graph → plan an action sequence from the graph → execute and write back updates. This realizes a closed loop of “perception → representation → planning → execution → updated representation.”

The authors also release **MomaGraph-Scenes** (~1,050 subgraphs, 6,278 images, 350+ scenes, 93 instructions) and a hierarchical benchmark **MomaGraph-Bench** (six capability types × four difficulty levels). They train on Qwen2.5-VL-7B using DAPO reinforcement learning with graph-alignment rewards.

Experiments show that **a unified (spatial + functional) representation** and **Graph-then-Plan** both outperform “single relation only” or “direct planning,” across both open- and closed-source baselines. Their 7B variant reaches **71.6%** on the benchmark, a **+11.4%** absolute gain over the base model, degrades less on high difficulty and under cross-view correspondence (including BLINK subsets), and is demonstrated on real robots.  

Overall, MomaGraph offers an integrated “representation + algorithm + data + benchmark” framework that systematically reduces hallucination while improving executability and consistency.

### Strengths
Originality: The paper unifies spatial and functional relations in a single task scene graph, refined down to part-level nodes (e.g., handles, buttons). It then applies state-aware dynamic updates \(G_T^{(t+1)}=U(\cdot)\) so that one-to-many hypotheses collapse toward one-to-one matches as the agent interacts. By pairing this with Graph-then-Plan, it decouples representation from planning and offers a holistic “representation + algorithm + data + benchmark” solution.

Quality: The training data covers ~1,050 subgraphs / 6,278 multiview images / >350 scenes / 93 instructions, annotated at part level and serialized in JSON. The benchmark spans 294 scenes / 1,446 images / 352 graphs / 1,756 instances, and evaluates six capability types across four difficulty tiers. The DAPO RL setup with graph-alignment rewards (actions, spatial/functional edges, node completeness, plus format/length constraints) is clearly specified, and the ablations/error analyses convincingly explain *why* the method works.

Clarity: The paper formalizes \(G_T=(N_T,E_T^s,E_T^f)\), illustrates dynamic graph updates (e.g., knob-to-stovetop examples), and shows multiview alignment and question types (action sequencing, spatial reasoning, affordance, precondition-effect, goal decomposition, cross-view correspondence). Figures are illustrative and make reimplementation feel achievable.

Significance: On MomaGraph-Bench, Graph-then-Plan consistently outperforms direct planning across open- and closed-source models. MomaGraph-R1 (7B) reaches 71.6%, +11.4% over the base model, with significantly smaller performance drop at higher difficulty tiers. On BLINK / cross-view correspondence subsets, it achieves new SOTA-like performance (+3.8% / +4.8%), demonstrating that structured intermediate representations reduce hallucination and boost robustness in realistic household long-horizon tasks.

### Weaknesses
The paper does not yet evaluate the robustness of the dynamic update module \(U(\cdot)\) under perception noise (false positives, false negatives, latency, occlusion). The core claim is that interaction plus write-back makes one-to-many → one-to-one, thereby reducing hallucination. This only generalizes if the update remains stable under imperfect observations.

Two actionable asks:
1. Add robustness stress tests: during evaluation, inject controlled noise (missed detections / spurious detections / delayed observations / occlusions) into the perceived scene before calling \(U(\cdot)\). Keep the rest of the pipeline fixed. Compare “no-graph / direct planning” vs. “Graph-then-Plan” on overall and per-capability metrics.
2. Report simple stabilization baselines, e.g. (a) hysteresis / confirmation delay (require repeated evidence before pruning edges or committing affordances), and (b) temporal smoothing such as EMA over edge/part confidence. This would clarify whether the proposed dynamic convergence is robust in non-ideal conditions and help attribute gains specifically to the structured Graph-then-Plan constraint.

### Questions
On the dynamic update \(U(\cdot)\) and convergence:

Please provide an explicit, executable update rule including:
(a) How uncertainty is represented (per-node / per-edge confidence? prior vs. posterior maintenance?);
(b) The conditions for pruning and confirmation (thresholds, evidence accumulation across multiple consistent observations, conflict detection and rollback policy);
(c) The convergence criterion (when do you “lock in” a one-to-one mapping from an initially one-to-many affordance hypothesis? how do you prevent oscillation or premature commitment?);
(d) Where this update module sits in the full inference loop, ideally with pseudocode or a flow diagram.

Concretizing these details would move “interaction → write-back → convergence” from a conceptual claim to something reproducible, and would strengthen the argument that performance gains are truly due to Graph-then-Plan’s structured constraint.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
At its core, this paper argues that embodied agents should reason with scene graphs that encode both spatial and functional (i.e. task-specific) relations. The authors (i) curate a dataset (MomaGraph-Scenes), (ii) introduce a multi-choice evaluation suite (MomaGraph-Bench), and (iii) train an RL-tuned 7B VLM (MomaGraph-R1) with a graph-alignment reward. On the benchmark, the model reaches 71.6% accuracy (+11.4% over the best open-source baseline), with transfer to public benchmarks and qualitative real-robot demos.

### Strengths
1. Transfer to established benchmarks such as BLINK is solid. MomaGraph-R1 leads strong open-source baselines on BLINK, consistent with the claim that graphing first reduces VLM hallucinations. Table 2 describes the most interesting result, since SOTA VLMs achieve atmost 60-65% accuracy on BLINK. 

1. The benchmark has breadth across reasoning skills. The benchmark spans tiered tasks (T1–T4). Table 1 shows that unified spatial+functional graphs beat spatial-only / functional-only graphs across tiers.

1. Interesting downstream real-robot evaluation. The paper evaluates four household tasks on a mobile manipulator illustrate the full Graph-then-Plan pipeline (multi-view perception -> graph -> plan -> skills).

### Weaknesses
1. Ablations are thin on training choices. The paper motivates RL but doesn’t include SFT/ICL baselines trained to produce the same JSON graphs, nor a sensitivity study on reward weights (only a config table/curves). 

1. Unclear novelty vs. prior “Graph-then-Plan" work. Prior work already builds or consults scene/task graphs before planning, e.g., GRID [1]  and VeriGraph [2]. Please spell out what’s new here: unified spatial+functional + part nodes, the graph-alignment RL objective, and state-aware updates, and which components drive the gains. 

1. Robot results are qualitative. The real-robot section describes four tasks but omits success rates, number of trials, failure modes, and timing—making it hard to assess robustness. Per-task SR/NSR and common failure modes would strengthen the evidence. 

References

[1] Dai et al. Optimal Scene Graph Planning with Large Language Model Guidance. https://arxiv.org/abs/2309.09182

[2] Ekpo et al. VeriGraph: Scene Graphs for Execution Verifiable Robot Planning. https://arxiv.org/abs/2411.10446

### Questions
1. Data diversity claim vs. skew. The paper claims that MomaGraph-Bench is diverse. However, Figure 8 shows that a majority of scene graphs come from two action-function pairs (pull-open_close_control, and press-device_control), which, to me, seems fairly narrow. Could you elaborate on your claim and situate your benchmark in the context of similar benchmark suites (such as BLINK)?

1. Currently, the only transferable task measured is BLINK. I would like to see evaluation of atleast one other task, such as EmbodiedBench, which aligns with your ultimate goal of embodied agent manipulation. 

[1] https://embodiedbench.github.io/

### Soundness
3

### Presentation
3

### Contribution
2
