# Dyana: Benchmarking Dynamic Hand Intelligence

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Most existing hand grasping benchmarks focus on static objects, which fails to capture the challenges of dynamic, real-world scenarios where targets move and precise timing becomes critical. We first propose the Dynamic Grasp Suite (DGS), a unified platform for dynamic grasp evaluation, and Dyana-12M, a large-scale benchmark with 12M frames of human-hand dynamic grasp trajectories. Dyana-12M represents target motion with three interpretable trajectories: straight-line, circular-arc, and simple-harmonic, which compose into arbitrarily complex trajectories. DGS standardizes interfaces and protocols, supporting the evaluation of three major model zoo: vision–language–action (VLA) agents, diffusion policies, and vision–language models (VLMs). 
Together, DGS and Dyana-12M establish a new paradigm for dynamic grasping, shifting evaluation from static scenes to motion-aware, temporally aligned assessment at scale.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an evaluation framework for dynamic hand–object interaction and a large-scale dataset DYANA of human-hand grasp trajectories with targets following three interpretable motion primitives (line, arc, simple harmonic). The suite compares VLAs, diffusion policies, and VLMs under unified I/O and metrics.

### Strengths
- The paper is well organized and easy to understand.
- The motivation is good. The community lacks a unified benchmark for dynamic grasping tasks.
- The authors develop a large scale human-hand dynamic grasping dataset.
- The metrics are defined well

### Weaknesses
- I’m not fully convinced about benchmarking VLAs/diffusion policies together with VLMs that aren’t designed for robotics control. It risks mixing capability gaps with interface/latency quirks.
- The dataset is generated in Unity; visual robustness and sim2real transfer are critical issues.
- The diversity of the dataset is narrow. Motion comes from a small set of primitives and compositions; object/material variety is limited. It’s not obvious this covers the range of real-world grasp dynamics (curvature, accelerations, occlusions, camera motion, lighting, etc.).
- No sim2real experiments. There’s no real-robot study to show that rankings or trends in Dyana carry over. Without at least a correlation plot (sim metric vs. real success), it’s hard to trust transferability.
- Many VLAs/diffusion policies were trained on robot hands/grippers. Evaluating them on a human-like hand in Unity may measure adaptation pain more than “dynamic hand intelligence.”

### Questions
- Could you fix the caption issue in figure 1? There are two Grasp Deviation. I believe one of them should be Location Deviation.
- Do you have any real-robot results (even small-scale) showing that model rankings or metric deltas in Dyana predict real performance?
- Is DYANA an acronym? If so, please expand it on first use. If not, a one-line note on why you chose the name (and why the dataset is Dyana-12M) would help.

### Soundness
2

### Presentation
3

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
This paper addresses the problem of grasping moving objects, in particular the pre-grasp stage. 
The main focus of this work is to studies how well current models can reach and grasp the moving objects. 
To assess the grasping quality, this work proposes a few metrics, including completion time, pre-grasp trajectory smoothness and final grasp success rate.
The proposed benchmark generate simulated object motions, using a number of pre-defined basic motions.
To evaluate a model, e.g. VLA or Diffusion Model, the benchmark provides RGB observation to the model, and expect the model to output accurate global hand wrist pose and finger articulation.

### Strengths
This paper proposes to benchmark the grasping performance of moving objects. This problem setting is interesting and novel. 

This paper studies a number of basic motions, these motions form a good basis for more complex motions.

This paper evaluates several state-of-the-art models on their benchmark, showcasing the limitation of current models.

### Weaknesses
# Reality Application 
One issue of static object benchmark is whether such setting aligns well with real-life scenarios. For example, typically the objects are static w.r.t the environment before grasping. For this dynamic grasping benchmark, it is unclear how the benchmark performance will correlate to the real-life performance. The setting is novel but It is unclear in what specific robotics scenario we will need to grasp moving objects.

# Ambiguous Definition of Valid Grasps

The evaluation is centring around the successful  grasp, described in Line 348. However, the definition of “a valid grasp” is unclear. 
Does a valid grasp account for objects sliding along fingers? Does it account for objects with non-uniform mass distribution? How are the forces evaluated? 
Since the objects are in motion, what happens when a moving object hits the hand? Does the object bounce on the hand? 

In real life, human will assess above questions before performing a successful grasp of a moving object. However, these factors are not considered in the proposed benchmark. 

# Pre-grasp vs Grasping

The proposed evaluation metrics in Sec. 4.2 are primarily focusing on the pre-grasp stage: how fast the hand reaches the object, how smooth are the trajectory during reaching, whether the hand can locate above the objects. The only metric for grasping is the "Grasping success rate". It seems that this paper assume that the grasping is a solved problem once the hand is above the object, which I disagree. 

In addition, "Grasping error" measures the distance between fingertips and the object surfaces at a successful grasp; this suggest that the validness of a grasp might merely be distance-based. However, a valid grasp is more than distance-based contact: "force closure" is another important factor affecting a grasp, and how forces interact with the object moving speed is not considered in the evaluation.

### Questions
I find this paper not easy to follow. Below are a few examples: 

Line 49-53  the current evaluations are limited to static objects, but then the authors say “the blind spot … whether models can anticipate motion”. Does the Dyna benchmark evaluate methods on object motion forecasting? 

Line 056, what does it mean to “expose” parameterized target-motion generators? 

Line 056, what is observation-action API?

Line 066, what does rollout mean here?

Line 105: “CHOMP, ITOMP, TrajOpt” references needed.

Line 146: “Clothoids provide G2 connectors when needed”. What does this sentence mean? 


In addition, I aslo found that this paper focuses on evaluating zero-shot methods, that is why VLA, DIffusion and VLMs are adopted for testing. 
If the evaluation methods are not limited to zero-shot methods, we would expect a more traditional multi-stage method to work. By "traditional multi-stage" I mean training a depth estimator, a motion predictor and an explicit hand action model that output the global wrist and local finger poses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to address the limitations of existing benchmarks that focus on static objects and introduce a new framework for benchmarking dynamic hand intelligence. The authors propose DGS, a simulation platform built in Unity, and Dyana-12M, a large-scale dataset of 12 million frames featuring human-hand trajectories for grasping objects moving along predefined atomic paths. The framework includes standardized evaluation protocols and hierarchical metrics to assess performance. The authors evaluate several state-of-the-art policy models and VLMs to show that these models struggle with dynamic grasping tasks.

### Strengths
- The proposed Dyana-12M dataset is extensive and incorporates features tailored specifically for dynamic scenarios. The "Observe-before-act" mechanism is a novel and critical feature for evaluating a model's ability to infer motion patterns before execution.
- The hierarchical evaluation metrics, which cover success rates, trajectory quality, and completion speed, are comprehensive.
- The paper provides a valuable empirical study by benchmarking a wide range of recent and relevant models.

### Weaknesses
- The manuscript suffers from numerous contradictions, typos, and undefined terms that severely impede readability and trust in the results. The camera setup is described as providing "egocentric RGB images... from a fixed camera" (line 158). These terms are mutually exclusive. An egocentric camera moves with the agent, while a fixed camera does not. Figure 1 contains two separate axes with different values labeled "Grasp Deviation," making the chart ambiguous and difficult to interpret. Key acronyms are used without definition. For instance, "HOI" is used multiple times before it is ever defined, forcing the reader to guess its meaning (Hand-Object Interaction).
- The paper fails to provide a formal definition of the task and does not adequately justify key modeling assumptions. The task of "Dynamic Localization & Grasping" is never formally defined. The paper lacks a clear mathematical formulation (e.g., as a POMDP) specifying the state, action, observation spaces, and the objective function. This makes it difficult to understand the precise problem the models are supposed to solve. Moreover, the benchmark assumes a "free-floating (hand-centric) embodiment". This abstracts away the robot arm, which is a major source of kinematic and dynamic constraints in any real-world application. This assumption is not discussed or justified, and it raises concerns about the practical relevance of the benchmark, as some evaluated trajectories (e.g., Fig. 2) might be physically unrealizable for a robot.
- Key design choices for the simulation environment and trajectory representation are not well-defended. The paper states that DGS is implemented in the Unity engine, but provides no justification for this choice over other simulators like Isaac Sim or MuJoCo, which are more common in robotics research and often offer better physics fidelity and sim-to-real support. The motivation for using linear, arc, and harmonic motion primitives is brief. The claim that these "compose into arbitrarily complex trajectories" is very strong and not sufficiently supported. Details about "For models from different ... through lightweight adaptation" (lines 201-204) are also not discussed.
- The paper relies exclusively on quantitative tables and charts, providing no visual examples of the generated trajectories. For a paper introducing a new benchmark and simulation environment, showing visual examples of success and failure cases is critical to demonstrate the nuances and challenges of the task.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes a benchmark for hand grasping for dynamic objects, which can capture dynamic, real-world scenarios where the grasp targets are dynamic -- in contrast to prior hand grasping benchmarks focused on static objects. To this end, it introduces the Dynamic Grasp suite (DGS), which is a unified, online evaluation platform for dynamic grasp with parameterized motion generations. It also presents Dyna-12M benchmark containing 12M frames across 180K dynamic hand grasp trajectories, constructed on top of DGS. The paper also provide standardized evaluation for diverse model faimilies including VLA agents, diffusion policies and VLMs.

### Strengths
**(1) Contributing a novel and useful benchmark**

This paper introduces a large-scale benchmark and evaluation platform for hand grasping of dynamic objects — an underexplored area in the field. I believe this can be a valuable contribution to the research community. I particularly appreciate the authors’ effort to release an online evaluation platform and provide standardized evaluations for existing model suites, which will further enhance the accessibility and usability of this benchmark.

**(2) Reproducibility**

The authors provide an implementation of the proposed benchmark, which significantly improves reproducibility.

**(3) Good presentation quality**

Overall, the presentation quality of this paper is solid. The figures are well-designed, and the text is clearly organized and easy to follow.

### Weaknesses
**(1) Real–synthetic domain gap of motions**

The proposed framework employs parameterized target-motion generation, enabling automated data collection and annotation, as demonstrated in the dataset comparisons in Table 2. However, this motion parameterization inherently limits the expressiveness of the generated motions, which may limit its ability to model complex real-world dynamics. Additional explanations or discussions on this aspect would be helpful.

**(2) Lack of discussion on limitations**

Related to the above point, the paper lacks sufficient discussion of the current limitations of the proposed benchmark. Including such a discussion would provide more informative insights to the research community and better guide future research directions.

### Questions
Please see the weaknesses section. I especially wonder how significant the real–synthetic domain gap is in the generated motions.

### Soundness
2

### Presentation
2

### Contribution
3
