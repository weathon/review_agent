# Learning Hierarchical and Geometry-Aware Graph Representations for Text-to-CAD

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Text-to-CAD code generation is a long-horizon task, requiring the translation of instructions into a long sequence of interdependent operations. This process is exceptionally fragile, as minor early errors can propagate through the sequence and ultimately invalidate an entire complex assembly. Existing methods typically decode instructions directly into executable code (e.g., bpy) without an explicit representation of assembly hierarchy or geometric constraints. This flat decoding strategy vastly expands the search space, amplifying local errors and leading to cascading failures in contextual operations. We address this gap by learning an intermediate representation: a hierarchical and geometry-aware graph. The graph represents an assembly-based decomposition, with multi-level nodes modeling the product's parts and components, and edges defining the explicit geometric constraints between them. Rather than mapping text directly to code, our graph paradigm first predicts high-level structure and constraints, then conditions the sequencing of operations and program generation, thereby narrowing the search space and improving both geometric fidelity and constraint satisfaction. Furthermore, we introduce a structure-aware progressive curriculum learning mechanism to enhance the model's ability to generate sophisticated decomposition graphs, allowing it to handle more complex assemblies. The mechanism constructs graded tasks via controlled edits to object structure, probes the model’s capability boundary, and synthesizes boundary examples for subsequent training rounds. We also introduce a 12K-instruction dataset annotated with instructions, geometric decomposition graphs, action sequences, and bpy code, together with metrics for node- and hierarchy-level graph accuracy and a measure of constraint satisfaction. Extensive experiments show that our approach outperforms existing methods in terms of both geometric fidelity and accurate fulfillment of geometric constraints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces GraphCAD, a three-stage framework for the text-to-CAD generation task, that includes a hierarchical, geometry-aware graph as an intermediate representation between text and executable CAD code. By explicitly modeling assembly hierarchy and geometric constraints, and by employing a structure-aware progressive curriculum learning scheme, the method achieves higher geometric fidelity and better constraint satisfaction than previous end-to-end baselines.

### Strengths
- This work proposes a novel and well-motivated formulation of the Text-to-CAD generation task by treating it as a structured reasoning problem rather than a direct text-to-code mapping. This reformulation is conceptually clear and empirically validated, leading to improvements in CAD generation performance.

- A new dataset (BlendGeo) is curated that pairs textual instructions with hierarchical geometric decomposition graphs, action sequences, and executable CAD code. This dataset is a valuable contribution that could facilitate future research in structured 3D reasoning and program synthesis.

- The manuscript is clearly written and well-organized, with intuitive figures (e.g., Figures 2 and 7) that effectively illustrate the hierarchical decomposition process and the overall data annotation pipeline.

### Weaknesses
1. The ablation studies confirm that each stage (graph, action planning) helps, but they do not fully explore why or where the improvements arise. For instance, showing qualitative failure cases when geometric constraints are omitted or visualizing how curriculum iterations expand the model’s capability boundary could make the training dynamics more interpretable.

2. The three-stage pipeline introduces latency (≈1.7 min per sample) and multi-model coordination complexity. Although the authors argue that this cost is acceptable, a more quantitative trade-off analysis between inference time and geometric fidelity would strengthen the claim.

### Questions
1. Will the proposed dataset (BlendGeo) be released publicly? Since the dataset is one of the key contributions of this paper, it would be important to clarify the release plan and accessibility for future research.

2. Why are three separate models fine-tuned for the three stages of the pipeline? This design significantly increases the overall complexity, computational cost, and memory footprint. Could the authors explain why a single unified model (e.g., fine-tuned jointly or multitask) was not adopted, and whether they explored this alternative?

3. What is the accuracy of the fully automated data generation process before human filtering? Specifically, what proportion of the automatically generated data was filtered or corrected by human annotators? It would be informative to show examples of (a) automatically generated data and (b) samples that were rejected or corrected during manual validation. I am particularly curious about the typical types of errors made by the LLM/VLM in the automated annotation pipeline.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a method for text-to-CAD generation. It decomposes the problem hierarchically, addressing different levels of abstraction in the design process. CAD code is generated progressively in three stages: an abstract graph, sequences of operations, and detailed CAD code. This generation process is achieved by three dedicated language models, finetuned by a proposed training strategy. A new dataset is curated for this task.

### Strengths
1. Proposes a novel hierarchical and geometry-aware graph as an intermediate representation for Text-to-CAD, clearly improving structure and constraint handling.
2. The experiments are comprehensive.
3. The proposed dataset is a valuable contribution to the field.

### Weaknesses
1. Lack of visualizations of the generated CAD models to better assess quality.
2. Evaluation heavily depends on LLM/VLM judgments, which may introduce bias.
3. The generation process heavily relies on large language models. Since generation of CAD code requires precise generation of numerical values, which may be difficult for LLMs, it is unclear how well the method generalizes to more complex designs. The authors should provide more analysis on this aspect, justifying the robustness of their approach. 
4. The proposed method can be viewed as a chain-of-thought strategy. It's unclear how much the hierarchical decomposition contributes to the performance compared to a simpler CoT approach or a zero-shot approach with a strong reasoning LLM.

### Questions
Please refer to the weaknesses section. I'm willing to increase my score if the authors can adequately address those concerns.

### Soundness
2

### Presentation
3

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
This paper tackles the challenge of text-to-CAD code generation, a long-horizon reasoning task prone to cascading errors. The authors propose a hierarchical, geometry-aware graph as an intermediate representation. A structure-aware progressive curriculum learning mechanism further enhances graph generation by gradually increasing structural complexity. The authors also introduce a new dataset. Experiments demonstrate significant improvements in geometric fidelity and constraint satisfaction over existing baselines.

### Strengths
1. The paper is overall well-written and easy to follow.
2. The proposed method is conceptually simple and clearly presented.
3. The motivation for capturing geometric constraints in CAD generation is well justified.
4. The hierarchical graph decomposition appears practical and effective.

### Weaknesses
1. The captioning cost using closed-source LLMs should be reported. Moreover, it would be valuable to evaluate the performance of free, open-source LLMs such as the Qwen-VL series for captioning.
2. Table 3 shows that the three-stage pipeline outperforms the end-to-end baseline; however, the time cost should also be compared to provide a clear trade-off analysis.
3. The curriculum learning algorithm is rather common, and this paper appears to present only an application of it to CAD generation, which limits its novelty.

### Questions
Please address the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose utilizing a hierarchical and geometry-aware graph that decomposes an assembly into its constituent parts (nodes) and explicit geometric relationships between them (edges). Furthermore, a curriculum learning approach is adopted for training a decomposition model. In addition to this, the authors also introduce BlendGeo dataset consisting of 12K quadruplets consisting of user instructions, geometric decomposition graphs, action sequences, and blender python code.

### Strengths
- the problem of lack of geometric reasoning and explicit structure is well motivated for Text-to-CAD application
- Building the BlendGeo dataset for the research community

### Weaknesses
- the representation novelty is overstated to some extent. The hierarchical geometry-aware graph closely mirrors the assembly graph representation established in [1], [2]
- the gains from SAPCL appear modest and it is unclear to me how much the curriculum learning contributes
- key metrics (Attr, Spat, Inst) rely on GPT-5 for evaluation and I’m concerned about objectivity and reproducibility.

[1] Hierarchical Graph Learning for Material Prediction and Recommendation in Computer-Aided Design
[2] Material Prediction for Design Automation Using Graph Representation Learning

### Questions
1. The improvement from SFT to SAPCL in Table 2 for Avg is modest. Can you provide some reasoning for this?
2. At what complexity or maybe part count does the graph representation become essential?

### Soundness
3

### Presentation
3

### Contribution
3
