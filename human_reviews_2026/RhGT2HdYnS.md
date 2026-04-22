# Transformers tend to memorize geometrically; it is unclear why.

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
In sequence modeling, the parametric memory of atomic facts has been predominantly abstracted as a brute-force lookup of co-occurrences between entities. We contrast this associative view against a geometric view of how memory is stored.
We begin by isolating a clean and analyzable instance of Transformer reasoning that is incompatible with memory as strictly a storage of the local co-occurrences
specified during training. Instead, the model must have somehow synthesized its
own geometry of atomic facts, encoding global relationships between all entities,
including non-co-occurring ones. This in turn has simplified a hard reasoning
task involving an ℓ-fold composition into an easy-to-learn 1-step geometric task.

From this phenomenon, we extract fundamental aspects of neural embedding
geometries that are hard to explain. We argue that the rise of such a geometry,
despite optimizing over mere local associations, cannot be straightforwardly
attributed to typical architectural or optimizational pressures. Counterintuitively,
an elegant geometry is learned even when it is not more succinct than a brute-force
lookup of associations. Then, by analyzing a connection to Node2Vec, we
demonstrate how the geometry stems from a spectral bias that—in contrast to
prevailing theories—indeed arises naturally despite the lack of various pressures.
This analysis also points to practitioners a visible headroom to make Transformer
memory more strongly geometric. 

We hope the geometric view of parametric
memory encourages revisiting the default intuitions that guide researchers in areas
like knowledge acquisition, capacity, discovery and unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper implies a geometric form of memory in Transformer to encode global structure information rather than only local associations. Experiment results demonstrate promising results

### Strengths
1. Paper is well written and easy to follow
2. Experiment is good and result is encouraging

### Weaknesses
I have my concern in the IN-WEIGHTS PATH-STAR TASK experiment. In this experiment, there is some information leakage, since training and testing is done in the same graph. In order to claim "IN-WEIGHTS REASONING IS LEARNED", testing should be done in new graph (with new graph topology), or with permutated labels for the node.

### Questions
1. see above weakness
2. In section 4, spectral analysis is mainly done in Node2vec, any connection with Graph Laplacian for the Transformer side?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates how Transformers memorize and reason over graph-structured data stored in their weights. The authors present a "path-star" graph task where, contrary to expectations, the model succeeds at finding "in-weights" paths. They posit this success is due to the emergence of "geometric memory" (where embeddings encode global graph structure) rather than "associative memory," or a simple lookup of local facts. The authors argue that the emergence of this geometric structure is not explained by standard pressures (like simplicity bias) and hypothesize that it stems from a "spectral bias," which they explore in the context of a simpler Node2Vec model.

### Strengths
1. The paper presents an interesting phenomenon which explores in-context learning and an alternative form of memorization via their experimentation of the path-star task.


2. The paper articulates the difference between "associative memory" (local, pairwise) and "geometric memory" (global, structural), offering a helpful lens for analyzing how models store knowledge.

### Weaknesses
1. The main explanation for the phenomenon (spectral bias) is explored only in a simpler Node2Vec model. The paper fails to provide a convincing experimental or theoretical link to show that this same mechanism is responsible for the results observed in the Transformer.

2. The visual evidence presented for the geometric memory in the Transformer is unconvincing. The UMAP plots (e.g., Fig 3b, Fig 10) are messy and do not show the clear, well-separated clustering that the text claims, forcing the argument to rely on heatmap interpretations.

3.  The paper sets up a compelling puzzle (success in Transformers) but then pivots entirely to a different model (Node2Vec) for a solution. This jump makes the overall argument feel disconnected and leaves their central question ultimately unanswered and confusing.

### Questions
1. Why should we assume that the spectral bias observed in a 1-layer Node2Vec model is the primary mechanism at play in a complex, multi-layer Transformer? Could the observed geometric memory in Transformers be an emergent property of the attention mechanism?

2. Could the usage of standard sequential positional encoding undermine the claim of geometric memory? The model is learning a node's position in a sequence, not its inherent position in a graph, which seems to reinforce the very "associative" view the paper claims to refute.

Side note: 
1. Your Fig. (1) has a weird error. The caption "Node2Vec Associative Memory..." on the left orange figure has overlapping texts.

### Soundness
3

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
3

### Summary
The paper proposes an in-weights variant of the path star task, where the goal is to memorize a star graph topology (in existing approaches the graph varies and is not memorized). Similarly to existing approaches, the model is trained on paths  from leafs to the center node and is evaluated on paths from the center to the leaves. Notably, in this setting the model generalizes to the test case (this is not the case in the  original path-star task). The authors investigate reasons for why this happens and argue that transformer memory works geometric.

### Strengths
-  **(S1) Novelty,  Contribution:**  The in-memory star-path task is novel and the observation that the model generalizes well is interesting.
- **(S2) Significance:** Understanding generalization capability of transformers is an important problem.
- **(S3) Experiments:** Empirically testing the different hypothesis on transformer memory / generalization is interesting. This paper contains several good observations. In particular: the model implicitly learns distances between nodes without being explicitly trained  for it.

### Weaknesses
While I think that the empirical work in this paper is solid, there are significant problems in the writing and presentations that hold this paper back:

- **(W1) Problem Presentation:** The problem central to this paper (Path-Star task) is not defined properly in the paper. While a good explanation exists, it is buried on page 25 in the appendix. This makes it difficult to understand the paper. In contrast, the paper devotes almost half a page to Figure 1 which is difficult to understand as it requires 10 lines of caption. As a side-note: What does the color in  Figure 1  encode?

- **(W2) Writing:** The writing often makes it difficult to understand the paper. Consider this:
>Early on during training, the model fits the later tokens in the target —
concretely nodes $v_2, \ldots , v_{\text{goal}}$ — as the unique child of the previous token provided in the input
during next-token training. This is known as a Clever Hans cheat: the model uses a local rule that
relies on witnessing part of the ground truth prefix $(p, r<i)$ to predict the next token $r_i$ — as against
only using the prefix p to predict the answer tokens

From this explanation, it is not clear what the "Clever Hans" cheat actually is. In contrast, the paper that introduces this problem [The Pitfalls of Next-Token Prediction](https://arxiv.org/pdf/2403.06963) makes the cheat quite clear. Furthermore, sentences are often convoluted making it  difficult to follow:
> Next, we cast doubt on statistical pressures by scrutinizing the applicability of simplicity bias


- **(W3) Significance:** The paper is not clear on what its contributions are. It is mainly a flood of hypotheses, observations and related works. From the abstract: 
>  Our insight is that global geometry arises from a spectral bias that—in contrast to prevailing intuition—does not require low dimensionality of the embeddings. Our study raises open questions concerning
implicit reasoning and the bias of gradient-based memorization, while offering a simple example for analysis. Our findings also call for revisiting theoretical abstractions of parametric memory in Transformers.

What does this mean? While the structure "Hypothesis" -> "Observation" is interesting it makes it very difficult to follow. For example, Observations often weaken Hypothesis requiring extra explanation after the observation. 

-**(Minor Weakness) Node2Vec:** The section about Node2Vec seems a bit orthogonal to the rest of the paper.

**To sum up,** the combination of a weak presentation, unclear writing and the paper not being clear about its own contributions makes it difficult for me to determine what the paper really contributes. I recommend that the authors significantly rewrite their text with a focus on clarity. As I do not believe that this can be done during rebuttal, I vote for reject. Furthermore, it seems that related work and connections to literature are present in many positions in the paper, making it difficult to  judge what is novel and what is not.

### Questions
- What sets your contributions apart from [The Pitfalls of Next-Token Prediction](https://arxiv.org/pdf/2403.06963)?
- Can you give me a concrete list of contributions?

## Miscellaneous

- The entire first paragraph is oddly phrased.
>Neat representations materialize when a model compresses redundancies in the data. On the
other hand, when faced with incompressible atomic facts (like the birth date of a celebrity), a
model would memorize these associations like a lookup table.

 What are ''neat representations''? The "however" also does not quite feel right here.

> We argue that this phenomenon implies a form of geometric memory in Transformers that must be contrasted with the associative memory view posited by Bietti et al. (2023). 

What are geometric and associative memories?


- Figure 6 in the appendix does an excellent job at explaining the setting of this paper. It would  be great if parts of it were in the main paper.
- Figure 2: The logarithmic (?) y-axis is slightly misleading and makes 48% look like 90%.
- "However, perhaps when the target we train on **is** an in-weights path, the cheat is not easy to learn—say, due to the nature of parametric memory."
- In the list of contributions:"We argue that the emergence of the geometric memory over the associative memory does not follow directly from existing intuitions."  Arguing is  not a contribution. Do you demonstrate or prove this? 
- Inconsistent subsubsections: some sections have a single subsubsection. Either use multiple ones (consistently) or do not use subsubsections.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
--------------An empirical study with insufficient analytical support-----------------------------

The paper studies how Transformers, when trained to memorize graph edges (in-weights), often learn a “geometric” parametric memory—node embeddings that reflect global relationships—rather than a purely local associative lookup structure. The authors construct an in-weights path-star task (first train on edge bigrams to put a fixed graph into the weights, then finetune on path prediction) and find that Transformers can perform nontrivial multi-hop path-finding for large graphs where in-context next-token training fails. The success contradicts standard associative-memory abstractions. To explain how local pairwise supervision can yield global geometry, the paper studies simpler 1-layer Node2Vec models and presents empirical evidence of a spectral bias: embeddings align with Fiedler-like eigenvectors of the graph Laplacian, while a time-varying coefficient matrix converges to having those eigenvectors in its null space. The authors argue a similar spectral mechanism may be responsible in Transformers, though Transformer embeddings are less “clean” than Node2Vec’s, indicating room for improvement.

Overall, the paper presents a clear phenomenon, thorough experiments, and a promising connection to spectral dynamics in simpler models.

### Strengths
•	Clear, analyzable phenomenon: Turning the known in-context failure for path-star graphs into an in-weights setting produces a striking contrast and isolates implicit in-weights reasoning.
•	Extensive experiments: Results span multiple graph types and sizes and include meaningful ablations (edge directions, pause tokens, first-token-only loss, etc.). Visualizations and heatmaps strengthen the empirical case.
•	Theoretical connection: Mapping the emergence of geometry to spectral bias in Node2Vec-style learning is a useful and plausible explanatory route.
•	Useful framing: Presenting associative vs. geometric memory as competing data structures is intuitive and helpful for framing future research.
•	Reproducibility effort: The paper includes many implementation details, hyperparameters, and procedural choices; the appendices are substantial.

### Weaknesses
1.	Theoretical rigor is limited.
o	The claim that Node2Vec dynamics converge to embeddings aligned with Fiedler vectors and that the coefficient matrix has those vectors in its null space is primarily empirical. The paper acknowledges the lack of a formal proof; still, the scope and assumptions of the empirical claim should be clarified (e.g., dependence on initialization, optimizer settings, sampling, etc.).
2.	Generality and applicability to natural language are unclear.
o	The experiments are on symbolic graph tasks and specific topologies. It is not yet clear whether the geometric memory phenomenon plays a comparable role in real-world language-model memorization or knowledge integration.
3.	Explanation for why Transformers avoid associative solutions is incomplete.
o	The paper effectively shows simple statistical/architectural/supervisory explanations are insufficient, but lacks a direct diagnostic showing why gradient descent prefers geometric solutions in practice. More direct analysis of optimization trajectories or mechanisms would strengthen this point.
4.	Some training choices require deeper analysis.
o	Pause tokens, mixed forward/backward edge supervision, and the particular interleaving strategy are important in practice; their exact roles (and whether they generalize) need more principled investigation.
5.	The link between Node2Vec and Transformer remains somewhat speculative.
o	Transformers include attention, deep layers, residual connections and layer-norm; how these aspects alter or obstruct the spectral dynamics observed in Node2Vec isn’t fully explored. Additional ablations to bridge the gap would be helpful

### Questions
Experiments & results

•	Stability: Provide multi-seed runs and report variance (e.g., success rate, epochs-to-convergence). Many plots show single runs.

•	Learning dynamics: The result that token accuracies rise “in tandem” is informative. Please show whether this pattern is robust to changes in learning rate, batch size, optimizer, or initialization.

•	Pause tokens: Quantify the relationship between number of pause tokens and required reasoning depth. Does increasing model depth (more layers) substitute for pause tokens?

•	Mixed edge supervision: Present a fine-grained ablation where the fraction of backward edges is varied continuously (e.g., 0%→100%) to show how performance scales.

•	Quantitative geometry metrics: In addition to heatmaps and UMAPs, report objective metrics (e.g., intra-path vs. inter-path cosine separation, Silhouette score, alignment with true Fiedler vectors) with error bars so comparisons (Transformer vs Node2Vec vs associative) are objective.

Theory & analysis

•	Node2Vec dynamics: Move the full derivation of Lemma 2 into the appendix (if not already) and explicitly list all assumptions (batching, softmax normalizations, self-probabilities, sampling, optimizer choices). Clarify when the empirical “self-stabilizing” dynamics are expected to hold and when they may fail.

•	Proposition 1: The bits and l2 norm arguments are useful. Flesh out boundary conditions—e.g., effect of weight tying, multiple outputs per input, or non-uniform degree distributions.

•	Transformer mechanism: Consider intermediate ablations that simplify the Transformer toward Node2Vec—e.g., single-layer Transformer with only embedding+unembedding, linear attention, or frozen attention weights—to trace whether spectral alignment persists and how components contribute.

### Soundness
3

### Presentation
3

### Contribution
3
