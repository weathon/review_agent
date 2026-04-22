# SG-Tailor: Inter-Object Commonsense Relationship Reasoning for Scene Graph Manipulation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Scene graphs capture complex relationships among objects and serve as powerful priors for 3D scene understanding tasks, yet their manipulation, such as adding nodes or modifying edges, remains underexplored and highly challenging. Even a single edge change can propagate conflicts across the graph due to intricate interdependencies, making the task computationally difficult. We propose $\textbf{SG-Tailor}$, an autoregressive model for structure-aware scene graph editing that generates commonsense edges for newly added nodes and resolves conflicts arising from edge modifications to ensure globally coherent graphs. For node addition, SG-Tailor queries the target node, forms candidate pairs with existing nodes, and predicts the appropriate relationships, while for edge modification it introduces a $\textbf{Cut-and-Stitch}$ strategy that repairs conflicts and adjusts the graph holistically. Extensive experiments demonstrate that SG-Tailor substantially outperforms prior approaches and can be seamlessly integrated as a plug-and-play module for downstream tasks such as scene generation and robotic manipulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work addresses the conflict issues that arise during the graph manipulation steps of generated scene graphs for downstream tasks.

### Strengths
Comprehensive description of methods and results.

Demonstrates effectiveness in practical settings and for downstream tasks.

### Weaknesses
### 1. Unelaborated Problem Definition
The main problem addressed in the paper is information loss during the manipulation steps. This is acceptable if no method has previously attempted to solve it. However, the third contribution claims to outperform other competitors in these steps, implying solved or solvable issues by competitors. Compared to the competitors, what problems are tackled should be more elaborated. 

### 2. Unclear Contribution on Problem Proposal
It seems that the literature may already cover this work, making it important to clarify its novelty. However, Section 3 is vague about whether the formulation itself constitutes a contribution. If it is novel, this should be explicitly stated; if not, the differences from prior work and relevant references should be provided.

### 3. Weak Novelty of Proposed Methods
The method is a straightforward transformer with masked training and does not introduce a particularly novel idea. In particular, the graph neural network community already employs a variety of triplet variants with advanced context-based input variations.

### 3. Issues in Empirical Validation
1) MPNN (2019) is selected as the baseline for comparison without justification. It is one of the earliest works in graph neural networks, and many GNNs have since been developed. Furthermore, it does not support the main argument on the impact of manipulation method proposal, but its generality use in graph building mechanisms, even though the transformer architecture design is not the main contribution of the paper.

2) Downstream tasks for robotics applications present only qualitative results, even though the dataset allows quantitative evaluation in the SG-bot work. Previous results are primarily analytical, and quantitative downstream results are essential to justify the importance of graph manipulation.

### Questions
Minor Point 

The term “Reasonable Scene Graph” can be confused with “reasoning-enabled scene graph,” and “reasonable” itself is somewhat vague. In my understanding, it simply refers to a scene graph that does not conflict with human knowledge.

### Soundness
2

### Presentation
2

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
The authors propose the SG-Tailor framework, which autoregressively reasons inter-object relationships to resolve semantic conflicts during scene graph manipulation. SG-Tailor utilizes a "Cut-And-Stitch" strategy, redefining the graph-level operations into cut and stitch steps and providing a novel perspective. The effectiveness of SG-Tailor is demonstrated through extensive experiments on 3D scene datasets, where it significantly outperforms both MPNN and state-of-the-art LLM baselines in generating coherent graphs.

### Strengths
- Clear problem definition and motivation. Well-structured paper which was very easy to follow.
- Comprehensive experiments on various datasets and effective baseline selection on multiple aspects.
- Interesting cut-and-stitch strategy, suggesting a novel viewpoint to graph operations and offering a solid framework.

### Weaknesses
- Downstream bottlenecks: As the authors acknowledge, the model's practical utility is constrained by the limitations of the downstream modules (like Graph-to-3D) and the fixed predicate vocabulary of the datasets.
- Novelty seems minor: The paper's contribution seems to lie in its formulation of the scene graph manipulation task, its practical application, and its strong empirical results, rather than on a fundamental architectural or theoretical advance.

### Questions
- Some parts of the methodology seem computationally heavy, e.g., brute-force way of editing. Can you provide a comparison on computational complexity with baseline methods? Also, I would like to know the scalability of this method. How does the performance (both speed and accuracy) degrade as the number of objects in a scene graph grows?
- Cycle rates don’t reflect the degree of incoherence (e.g., two or more conflicts) or other forms of spatial contradictions (other than cycles). Can you provide other metrics that can also reflect such aspects?

### Soundness
3

### Presentation
4

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
This paper presents SG-Tailor, an autoregressive model designed to address conflicts in scene graph manipulation tasks. Scene graphs capture complex relationships among objects and serve as a key component in generating and manipulating 3D scenes. However, existing methods for manipulating scene graphs struggle to handle semantic conflicts that arise when nodes or edges are modified. SG-Tailor introduces the "Cut-and-Stitch" strategy, which allows the model to infer reasonable relationships for newly added nodes and resolve conflicts caused by edge modifications, producing coherent scene graphs. Extensive experiments demonstrate that SG-Tailor outperforms existing methods on multiple benchmarks and can be seamlessly integrated into downstream tasks, such as scene generation and robotic manipulation.

### Strengths
1. Importance of the problem selection: Scene graph manipulation is a key challenge in computer vision and robotics, and the paper focuses on solving the issue of relationship conflicts, which has practical application value.

2. Method innovation: The Cut-And-Stitch strategy is a novel approach that decomposes scene graph manipulation into cutting and stitching steps, offering both intuitiveness and effectiveness.

3. Experimental comprehensiveness: The paper evaluates the method across multiple datasets, demonstrating its generalization ability.

4. Practicality and scalability: The method can be integrated as a plug-in module for downstream tasks, such as robotic manipulation.

### Weaknesses
1. The core method of the paper is based on autoregressive models, which is a common technique in natural language processing (NLP) and some graph learning tasks. Although applying this to scene graph manipulation is somewhat novel, the paper fails to sufficiently demonstrate the essential differences from existing works, such as SGNet, MPNN, or LLM-based methods.

2. The Cut-And-Stitch strategy lacks theoretical support: The paper claims that the Cut-And-Stitch strategy can effectively resolve conflicts, but it does not provide theoretical analysis or mathematical proof to demonstrate its optimality. For example, how can we ensure that the stitching step always produces a conflict-free graph after the cutting step?

3. Generalization ability is questionable: The paper claims that SG-Tailor can be applied to multi-task learning and downstream applications (e.g., robotic manipulation), but the experimental section only briefly mentions this, lacking detailed results or user study data. For instance, the robotic manipulation experiment (Appendix A) only provides qualitative examples without offering quantitative metrics or comparisons with professional methods.

### Questions
1. How does the specific implementation of the Cut-And-Stitch strategy ensure global consistency?
2. How does SG-Tailor handle large-scale scene graphs? When the number of nodes increases, does the sequence length exceed the model's context window?
3.  What is SG-Tailor's performance in real-time systems (such as robotic interactions)? How robust is it to input noise? The paper mentions downstream applications (such as robotic manipulation), but the real-time performance has not been evaluated. If the input scene graph contains annotation errors or noise (such as incorrect relationships), can the model correct them? Figure 8 demonstrates robotic manipulation results, but there is no quantitative analysis of the success rate. Please add task-level metrics (such as planning accuracy, execution efficiency).

### Soundness
3

### Presentation
2

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
This paper focuses on how to maintain the consistency of the scene graph after inserting or manipulating the node and proposes the SG-Tailor for robust scene graph manipulation.

### Strengths
1. The problem statement of scene graph manipulation is valid and convincing.
2. The proposed structure shows significant improvements in the experiment results.

### Weaknesses
1. The paper lacks theoretical analysis on the proposed method and the motivation of the SG-Tailor structure is unclear. Why is it a good idea to formulate inter-object relationship reasoning into the autoregressive sequence generation task? It will be good if authors can provide any bounds based on equation 7.
2. The result on SceneVerse37K is only compared with MPNN. A single baseline is not sufficient to claim the effectiveness of the proposed structure. Can authors give the results from other models?

### Questions
1. In table 4, why there are two MPNN appears in the 3D-FRONT dataset section. Are they different?
2. For the cycle rates, table 2 shows great improvement to avoid generating unnecessary parts of the graph. What is the cost for that? Will SG-Tailor take longer to generate the graph parts? In the use case mentioned in appendix A, what is the overall execution time for the baseline and the SG-Tailor?

### Soundness
4

### Presentation
4

### Contribution
3
