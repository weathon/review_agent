# Creative Robot Tool Use by Counterfactual Reasoning

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
We propose a causal reasoning framework for creative robot tool use in which a novel object is correctly identified to functionally substitute for a tool that is no longer available to the robot. During training, our framework discovers properties of the source tool that are causally relevant to the task by conducting counterfactual experiments in a physics-based dynamics model. We reconstruct the geometry of the tool from vision and systematically intervene on its properties using 3D shape editing to generate counterfactual tool variants. During deployment, candidate tools are classified based on their similarity to the source tool with respect to the discovered causal features. By reconstructing the task in a dynamics model, our approach grounds tool use in the physics of the problem.
    We illustrate our approach in reaching a distant object with different sticks, in scooping candies from a bowl with different kitchen items, and in using different boxes as platforms to step up to retrieve an object from a high shelf. We surpass state-of-the-art methods in repurposing everyday objects and show that the discovered features align with human judgment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposed a causal reasoning framework that helps robots to find appropriate tools from unseen candidates for a specific task. During training, the authors first asks a VLM to propose features of an object that are relevant to completing the task, then new objects are synthesized by altering these features. These new objects are then used in simulation to figure out the actual importance of these features, which are used by classifiers to determine the right tool during inference.

### Strengths
1. The authors leverage VLM's commonsense knowledge to help robots complete tasks, which is a promising direction.
2. Real-world test setups were established.

### Weaknesses
1. The proposed approach did not appear sufficiently generalizable to more complicate tasks and objects. The experiments are conducted on only three tasks.
2. The illustrations are difficult to understand. Particularly on the right side of Fig.2, the title suggests "classification" but it is unclear where the classification step is at. And why is the same image of the novel object is considered "approximately equal to" to different shapes? Are the two shapes "approximately equal" too?
3. Typos. In Fig.2-left, the feature "suggestor" (which should be suggester?) is labelled with (e), whereas in the texts, it is labelled with $\phi$. In Line 222, I did not find grasp points in Fig.3, so I assume it should be Fig.2?

### Questions
1. The authors use simulations with modified 3D objects to determine whether the suggested features are relevant to the task. Is it possible to replace this procedure with VLMs, which would be less time-consuming.
2. In Line 194, what are the 12 features in "top 6 most voted features out of 12"? Are they fixed for all tasks?
3. With the relevant features, how does the classification step work? The texts in the paper are not very clearly written.
4. Sec 4.2 is titled "Task Performance and Inference Time Analysis", where is the inference time analysis? I only find one sentence with time-related information ("it requires minutes to scan a single object"), and it is not about inference.

### Soundness
2

### Presentation
2

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
This paper is concerned with robots that can identify appropriate tools for their task at runtime, even if the available tools have never been seen before.  To address this challenge, the paper presents a computational pipeline that combines foundation models, procedurally generated object models, and physics simulators.  Given an existing tool known to be relevant to a task, new tools available in the environment are compared to the existing tool through several computational steps.  The steps include identification of relevant features on the existing tool, systematically warping each feature of the existing tool in a procedural object generator, simulating task execution with the warped versions to determine which work best, and then matching the best-performing warps to the novel tools in a new scene.  Foundation models and other deep models are used for several stages of this pipeline, including identification of relevant features, segmenting objects into parts, and reconstructing the current scene in a simulator.  Experimental results suggest that the proposed system has a higher success rate than several ablations and baselines.

### Strengths
- This paper works on an interesting problem and the high-level conceptual approach is appealing.  Simulation and procedural object warping are good ways to ground the error-prone recommendations of foundation models.
- The comparison with human-annotated object features is a nice touch.
- The work appears to be a substantial undertaking, given the combination of several large machine learning models and several robotic hardware platforms.

### Weaknesses
- One significant weakness concerns the technical novelty of the approach.  While some of the conceptual ideas may be novel, at the technical level, it appears that the computational pipeline does not involve any new method or model.  Rather, it combines several existing off-the-shelf foundation and other models in sequence.

- There are also deficiencies in the empirical analysis.  Table 1 presents overall success rate, but I did not see mention of the number of trials used to calculate success rate.  It is also not clear how much variation was present in each experimental condition (object poses, starting positions of the robots, etc.).  Several other "results" are only qualitative. In particular, lines 370-391 describe several trends anecdotally but do not provide any numerical data as quantitative empirical evidence for these claims.  And section 4.4 ("classification metrics") does not provide any metrics, it only references three images showing pictures of different objects.  Overall, there could have been a more comprehensive and intentional variation in experimental conditions, showing how performance varies in relation to those experimental conditions.  Details on computational expense would also be welcome, considering the large foundation models involved.

- Since the method uses foundation models trained on internet-scale data, the claim that "novel" tools have "never been seen before" is questionable.

- I saw several typos and grammatical errors; the presentation could be improved with more proof-reading.  Here are some:
    - section 3 line 127: missing period
    - page 4 line 189: "details if the controller" -> "details of the controller"?
    - page 4 line 203: "editor that gets a feature" > "editor gets a feature"?
    - page 5 line 222: "figure 3" should be "figure 2", right?
    - page 5 line 246-7: "that the all", "feautures"

### Questions
- It was unclear where the full feature set $F$ comes from.  If it is pre-defined per task, then can this method really be described as "open world"?

- If SAMPART3D requires a 3d object model as input, where does the 3d object model come from?  Is it the scene reconstruction?  But Fig 2 does not have a corresponding arrow from the scene reconstruction.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of creative robot tool use, enabling robots to repurpose or substitute everyday objects when the intended tool is unavailable. Existing approaches, such as affordance prediction or vision language reasoning, rely on correlation and semantic similarity rather than physical or causal understanding, which limits generalization to unseen tools. The authors propose a causal reasoning framework that discovers why a tool works by identifying its causally relevant physical features. A vision language model first proposes candidate features such as length, curvature, or mass. A shape editing module then generates counterfactual variants of the original tool by modifying these features, and a physics simulator evaluates each variant to determine which changes affect task success. The robot thereby learns a set of causal functional features that explain task performance. When encountering novel objects, the robot selects substitutes that share those causal properties. Experiments in simulation and real world manipulation tasks show that the system finds physically plausible substitutes, outperforms affordance and VLM based baselines, and produces interpretable causal explanations.

### Strengths
- Combines language model based hypothesis generation with physical counterfactual simulation.

- Produces interpretable explanations of which features make a tool functional.

- Enables successful substitution with previously unseen objects.

- Includes both simulation and real world experiments with clear gains over strong baselines.

### Weaknesses
- The causal analysis assumes high fidelity physics and accurate object reconstruction, which may reduce reliability in real world scenes.

- The evaluated tasks are limited in diversity and focus mainly on object centric scenarios; extending to deformable tools would strengthen the claim.

- Because candidate features originate from a vision language model, causally important but linguistically obscure attributes such as stiffness or friction may be missed.


- The related work section would benefit from citing closely related lines of research:

H. Chen, C. Zhu, S. Liu, Y. Li, and K. Driggs Campbell, “Tool as Interface: Learning Robot Policies from Observing Human Tool Use,” CoRL 2025.

M. Xu, P. Huang, W. Yu, S. Liu, X. Zhang, Y. Niu, T. Zhang, F. Xia, J. Tan, and D. Zhao, “Creative Robot Tool Use with Large Language Models,” arXiv:2310.13065, 2023.

### Questions
How robust is the causal feature discovery under perception or simulation noise

Can the method identify non linguistic causal attributes such as material compliance or friction

How transferable are discovered features across tasks, for example from scooping to pushing

How might the system generalize to multi step tool compositions or sequential creativity

### Soundness
3

### Presentation
4

### Contribution
3
