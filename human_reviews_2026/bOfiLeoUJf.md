# Query-Specific Causal Graph Pruning Under Tiered Knowledge

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
We present a systematic method for pruning edges from causal graphs by leveraging tiered knowledge. We characterize conditions under which edges can be removed from a causal graph while preserving the identifiability of (conditional) causal effects. This result enables causal identification on simplified graphs that are substantially smaller than the original graphs. The approach is particularly valuable when researchers are interested in causal relationships within specific tiers while accounting for broader influences from other tiers without fully specifying them. Building on this, we introduce a query-specific causal discovery algorithm that takes a causal query and observational data as input and returns a graph tailored specifically to that query. Through both theoretical analysis and empirical studies, we demonstrate that our discovery algorithm can achieve exponential speedups compared to the existing method when tiered knowledge is available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper focuses on the challenge of answering causal queries in complex systems without having to fully specify or learn an entire causal graph. This comes from the motivation that in some settings the causal model can involve many variables, but a researcher may only be interested in the effect of certain treatments on certain outcomes. The paper focuses on how incorporating background knowledge (tiered knowledge) about the system can improve both identifiability of causal effects and computational efficiency.   They introduce a graph pruning approach to remove certain edges from the causal graph without affecting the identifiability (causal identification on simplified graphs). Based on this, the paper introduces a query-specific causal discovery algorithm that takes as input not just observational data but also a specific causal query of interest to only learn the portion of the causal graph relevant to that query. The paper shows theoretically and empirically that their method can achieve exponential speedups

### Strengths
- I think the paper is clear and well written. Sometimes a bit too heavy in notation, but this is perhaps necessary for introducing the theoretical results. I would encourage having more intuitive introductions to some of the notations (for instance c-components/tiered knowledge). I enjoyed the graph examples in the paper, they made the understanding easier. 
- the research question is relevant in the field, further improvements in this direction can potentially help causal effect estimation be more applicable in practical scenarios. It is also important to start from this kind of theoretically-grounded contribution to understand to what extent identifiability can hold with different kinds of domain knowledge.
- the theoretical results show that the proposed algorithm is sound and complete
- the experimental result show promising results in terms of efficiency without sacrificing correctness of the learned structures

### Weaknesses
- I think the paper could be strengthened by adding some examples of real world applications in introduction where tier knowledge is available to practitioners
- The authors should explicitly mention as well that this paper relies on acyclicity of the causal structure
- in Line 078, the notation $Pr_{B_1}(B_3,C_3)$ is introduced, however the subscript notation had not previously been introduced. The authors should fix this. 
- related to the previous sentence, also when mentioning 'they are identifiable in the pruned graph G' I would add a reference to Fig 1b, to make it easier for the reader to understand the example.
- Minor: format of the title seems different from the ICLR layout

### Questions
- can you provide some examples of practical applications where we actually have tiered knowledge? 
- It is hard to imagine a practical setting in which we can rule out bidirected edges across tiers. I know that the authors mention that this extension is left for future work, but what kind of steps would be necessary for this extension?
- it seems that the advantages of the algorithm decrease be increasing the size of the conditioning set $Z$. What happens when this is substantially larger? It would also be nice to see the improvement in terms of num tiers as percentage of the entire set of nodes, and then analyse how this changes with different sizes of underlying causal graph

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
2

### Summary
This paper studies the identification and computation formula for causal (treament) effect and conditional causal effect under tiered knowledge. It shows the identification and computation can be infered from a simplified (pruned) graph based on the original graph using the tiered knowledge. The pruning procedure is given and a query-specific causal discovery method is proposed based on the pruning procedure.

### Strengths
- The paper well-motivated. It introduces the concept of query-specific learning, which reframes causal discovery from global reconstruction to task-focused learning, suitable when full-graph recovery is unnecessary or infeasible.
- The exposition is generally clear and the examples (especially motivating ones in the introduction) make the idea intuitive.

### Weaknesses
See questions.

### Questions
- One of the main message of this work is: to check the idenfication of a given (conditional) causal effect, it suffices to look at a prune graph, then use the existing identification formula. However, it there truely benefit of doing so in terms of computation? Would calling identification formula on the whole graph reduce to calling identification formula on the pruned graph? What is the gain of using the pruned graph when there is an extra step to prune the graph?
- The experiment only shows the speedup against existing methods on causal discovery. With the main motivation to be causal effect identification check, would there be experiment showing the speedup in this check?
- Why does the experiment in Figure 5 seem not depend on $|Z|$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
A method is presented that prunes edges from causal graphs under tiered knowledge. Conditions for the removal of edges are specified that don't affect the identifiability of (conditional) causal effects. These are subsequently used to develop a query-specific causal discovery procedure.

### Strengths
The presented method is potentially computationally more efficient than the FCITiers algorithm it is based on. The presentation is clear.

### Weaknesses
- Bidirected edges across tiers are not allowed, which is a strong limitation. In fact, in the medical context example (lines 121-123) used to justify the assumption of tiers, I don't believe it can be assumed that there are no latent confounders between, say, treatment and recovery. This restriction makes me doubt whether this paper is appropriate for a top conference like ICLR.
- The paper conflates ADMGs and MAGs. Initially, ADMGs are used for causal graphs, and their intuitive semantics are explained. But the theoretical developments later in the paper use MAGs and PAGs, which have different semantics. I am not sure the presented results are correct if the ground-truth graph is an ADMG that is not a MAG.
- The pruning method seems to be incremental compared to existing pruning techniques.
- Proposition 4.2 proves the possibility of exponential speedup, but this only applies to "a class of distributions". This makes the statement rather weak. (And indeed, the proof constructs a family of graphs that contains one graph for each $n$.)

### Questions
- How is the data generated in the experiments?
- How are propositions 3.5 and 3.6 related? Would it be possible to just use Proposition 3.6?
- How does this work compare to existing work using pruning, such as [1]?

[1] Tikka, Santtu, and Juha Karvanen. "Enhancing identification of causal effects by pruning." Journal of Machine Learning Research 18.194 (2018): 1-23.

### Other comments
- "casual" -> "causal" in line 97

### Soundness
1

### Presentation
2

### Contribution
1
