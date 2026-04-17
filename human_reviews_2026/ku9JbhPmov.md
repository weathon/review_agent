# Caterpillar GNN: Replacing Message Passing with Efficient Aggregation

- Decision: Reject
- Scores: 4, 8, 6, 2, 2

## Abstract
Message-passing graph neural networks (MPGNNs) dominate modern graph learning.
Typical efforts enhance MPGNN's expressive power by enriching the adjacency-based aggregation.
In contrast, we introduce an efficient aggregation over walk incidence-based matrices that are constructed to deliberately trade off some expressivity for stronger and more structured inductive bias.
Our approach allows for seamless scaling between classical message-passing and simpler methods based on walks. 
We rigorously characterize the expressive power at each intermediate step using homomorphism counts over a hierarchy of generalized caterpillar graphs. Based on this foundation, we propose Caterpillar GNNs, whose robust graph-level aggregation successfully tackles a benchmark specifically designed to challenge MPGNNs.
Moreover, we demonstrate that, on real-world datasets, Caterpillar GNNs achieve comparable predictive performance while significantly reducing the number of nodes in the hidden layers of the computational graph.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Caterpillar GNNs as a novel aggregation scheme over walk–incidence matrices. The design intentionally trades some expressivity for a stronger inductive bias that can be tuned via a height parameter derived from color refinement.
Theoretically, the authors prove that homomorphism counts from generalized caterpillar graphs are exactly as expressive as counting colored walks on height-h colorings, and that their efficient aggregation recovers this power while remaining comparatively efficient.
Empirically, Caterpillar GNNs solve a topology-driven benchmark that challenges MPNNs and, on common graph-classification datasets, achieve comparable accuracy with fewer computational nodes per graph.

### Strengths
The work introduces original ideas on designing GNNs. The presented model processes the occurrences of colored walks in a manner that de-duplicates repeated structural patterns in the input graph, effectively reducing the size of the computational graph. This class of GNN is very distinct from prior approaches. The fine-grained characterization of expressivity below 1-WL is also an intriguing new direction for the theory of GNNs and their inductive biases. The presentation is also clear to follow.

### Weaknesses
I think its claimed efficiency of Caterpillar GNNs is not convincingly demonstrated, despite being presented as a major feature of the approach. The preprocessing runtime is shown to be cubic in the number of vertices, which is limiting for large graphs and far above the trivial preprocessing needed for standard message-passing GNNs.

While the number of nodes in the computational graph is one factor that determines the efficiency, the sparsity of the underlying aggregation matrices is another critical factor that is not discussed in sufficient detail. As real-world graphs often have a high degree of sparsity, message-passing GNNs can scale to large graphs with >>10K vertices on single GPUs by using optimized sparse matrix operations. The paper does not analyze the sparsity of the $C_t^A$ matrices and the compute needed when using them in the suggested aggregation scheme.

The experiments also do not provide any runtime measurements for preprocessing and model training, which would be key to establish the scaling of Caterpillar GNNs in practice. The experiments also focus entirely on small-scale graphs with at most a few hundred vertices.

I do think the presented class of GNNs and the expressivity results are novel and interesting, but the unconvincing efficiency claims are a significant weakness as the paper makes efficiency a key part of the motivation.

### Questions
1. For the experiments on real-world datasets, what is the end-2-end runtime of training on each dataset when including the pre-processing time for obtaining the canonical walk subsets?
2. How sparse are the matrices $C_t^A$ compared to the adjacency matrix $A$?
3. Can the suggested walk scheme handle edge-features or continuous node features, which are both common on real-world graph learning tasks?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Caterpillar graphs are a subclass of tree graphs and thus computing homomorphisms from caterpillar graphs is strictly less powerful than using all trees.
Nevertheless, the paper shows that this reduced expressivity is often enough and introduces an architecture that works by counting the occurences of colored walks in the input graph which in practice is as powerful as using standard message-passing. 
The paper focuses on a clean and thorough introduction of the method, its modifications to make it efficient (including proofs) ,and its connection to existing work in expressiveness research for graphs.

### Strengths
A fundamentally new architecture, an indicator that more expressiveness is not everything (at least for many situations), rigorous theoretic derivation of expressiveness properties and connections to homomorphism counting.

### Weaknesses
The experiments could have been more extensive. This includes both runtime (completely missing) as well as additional datasets. E.g. Table 3 in the appendix makes it look like the caterpillar-GNN achieves SOTA results which is clearly not the case. 

Details:
- Figure 2: Just looking at the figure makes it hard to guess what is happening here, even including the caption did not help too much. It would be nice to state that on top of the table its "all possible walk patterns" and possibly to visualize the example given in the caption about two walks "gb" ending in vertex 3 by showing those two walks and connecting it to the corresponding position
- 191: its sounds odd to talk about walks of color $\textbf a$, maybe "color pattern" or just "pattern"?
- 206: it would be nice to state that the overall matrix $W$ will then be reduced to $|V|\cdot t$ (from $|V|^t$)
- 231: what does this paragraph aim for? Or rather, what part about the MPGNN aggregation are you unhappy about and thus suggest to change it?
- 239: I find this whole paragraph on EA pretty hard to parse and would have liked a bit more high-level information on what the paragraph is about. What is "t-th efficient"? Is it possible to add "(e.g. A and I from MP above)" to the first sentence introducing M?
- 358: would it be possible to clarify which aspect of fig 1 is important in this context? I think this could be more clear.
- Fig 7: the datasets feel a little outdated and the MPNN performance tends to be rather low (even though the description makes clear that some effort was put into it).

### Questions
1) how much capacity for tuning the practical part of the method do you see?
2) What are the main obstacles against extending the method to directed graphs?
3) Given the low performance on ZINC (which is about counting cycles), do you expect it to perform well in real-world tasks in general? Especially as many of those rely on small cycles. Would it be possible to also run experiments on e.g. MolPCBA and OGBN-Arxiv? (as examples of somewhat more recent datasets)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose EA, a mechanism to replace Message-Passing in GNNs, which uses an aggregation over walk incidence-based matrices and trades some expressive power for a stronger, more structured inductive bias. They provide proof of its expressiveness using a hierarchy of generalized caterpillar graphs and their graph homomorphism counts.

### Strengths
- The mechanism presented is novel and well-founded, connecting random walk-based models with message-passing GNNs.
- There is a rigorous theoretical analysis of the expressive power of the method using homomorphism counts.
- The paper contains multiple interesting theoretical insights. I also find the use of automata theory to prove results from Sec. 3 and Sec. 4 interesting.
- Different from some other random-walk-based methods, Caterpillar GNNs can preserve (permutation) equivariance and determinism, which can be desirable properties.
- The paper introduces the new nStepAddition benchmark, showcasing scenarios where Caterpillar GNNs outperform MPNNs.
- The paper is very pleasing from an aesthetic perspective; the figures and formatting are beautiful.

### Weaknesses
The main weaknesses of this paper are scoping and empirics:
- **Scoping:** The paper is primarily theoretical, proposing a novel aggregation mechanism along with the new Caterpillar GNNs and providing an extensive theoretical analysis. However, its presentation within the constraints of a short conference paper feels limited. While the main text introduces some of the key ideas and results, much of the theoretical novelty and discussion are deferred to the Supplementary Material. I recognize the challenge of fitting a substantial body of work into a 9-page format, but this raises the question of whether the contribution would be better suited as (1) a longer journal submission or (2) multiple, more focused conference papers with narrower scope. As it stands, the overall narrative feels somewhat diffuse, and the contributions appear scattered rather than cohesive.
* **Empirics**: The paper is notably weak in terms of empirical evaluation. The stated goal from an empirical perspective is to study the scaling properties of EA and compare the Caterpillar GNNs with MPNNs. While some results are presented in this direction (e.g., Figures 4 and 7, Appendix Tables 2–4), they remain relatively superficial and unconvincing from a practitioner’s perspective. Both MoleculeNet and TUDataset provide limited grounds for quantitatively assessing modern architectures. Moreover, the paper lacks runtime comparisons, parameter counts, or other metrics that would allow a fair assessment of computational efficiency, reporting only the number of nodes in the computation graphs. In addition, there are no comparisons with more expressive baselines such as _k_-WL-based models (e.g., GNN-SSWL [1], IDMPNN [2]) or subgraph-aware GNNs (e.g., ESAN [3]). Comparisons with Random-Walk-based methods like CRaWL [6] or NeuralWalker [7] are also missing. Finally, the evaluation omits standard expressivity benchmarks, including EXP and CSL [4], which are widely used to assess a model’s ability to capture higher-order structural patterns.

[1]: https://proceedings.mlr.press/v202/zhang23k/zhang23k.pdf
[2]: https://proceedings.mlr.press/v202/zhou23n/zhou23n.pdf  
[3]: https://arxiv.org/pdf/2110.02910  
[4]: https://arxiv.org/pdf/2010.01179  
[6]: https://arxiv.org/abs/2102.08786
[7]: https://arxiv.org/pdf/2406.03386

### Questions
Please see weaknesses above.

I find the core idea of the paper appealing, and I believe the theoretical contributions are both interesting and valuable for certain subcommunities. However, the presentation of the theoretical ideas could be more focused, or alternatively, the paper could benefit from a more comprehensive empirical analysis. I am somewhat ambivalent overall but lean toward accepting the paper primarily for its theoretical merits. That said, I believe there is substantial room for improvement, and I would also find a rejection reasonable.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents Caterpillar GNN, which augments the message passing process to incorporate node-walk incidence matrices derived from random walks. Because the number of random walks increases exponentially in the length of the walks, the authors propose a deterministic walk selection approach, which limits the number of random walks of length 1 to T to n, where n is the number of graph nodes and T is the maximum length of the random walks. Using a least-squares based approximation, the selected walks B_t are mapped into new matrices C_t that are used instead of the adjacency matrices during the message passing operations. The authors choose the random walks in a deterministic and permutation equivariant manner.

### Strengths
- The approach appears to be novel and incorporating important insights from graph theory and automata theory fields.

- Deterministic and permutation-equivariant walk selection adds a principled structural constraint.

### Weaknesses
1) Poor organisation and clarity

- The exposition is dense and introduces key definitions late, which makes it difficult to follow the main ideas. 

- The first three pages (i.e., from the beginning until Section 3) introduce the terminology and concepts used in the paper.  However, a formal definition of “caterpillar”, a fundamental concept of this submission, does not appear until page 6 in Section 4. On page 1, the authors refer to caterpillars simply as a subclass of trees, without providing any further details. 

- In Section 1, an informal theorem (Theorem 4.1) is illustrated by a figure (Figure 1). However, neither a proof is provided nor there is an explanation for the figure. As a result, the relationship between colored walks and caterpillars remains unclear. 

- In Section 2, an unnecessarily complex and unconventional notation is used to describe the way MPGNNs operate. 

- The proof of the arguably the most important theorem of the paper (Theorem 4.2) is given in the appendix with minimal explanation and by citing some other Lemmas given in the appendix.

2) Efficiency and scalability issues

- The method presented is expensive despite the complexity reduction referred to as efficient aggregation. With sufficiently discriminative node or edge features, the number of colors will be in the order of the graph nodes. The complexity of walk selection will then be O(n^4) according to Theorem 3.1, where n is the number of graph nodes.

3) Limited experimental validation

- Experiments are sparse and inconclusive. It would help if the authors provided a comparison or ablation showing why deterministic selection outperforms stochastic alternatives.

### Questions
- Why is the deterministic, permutation-equivariant walk selection introduced in Section 3.1 considered the best possible choice? 

- How does Caterpillar GNN compare, in tractability and expressiveness, to simpler alternatives such as using powers of the adjacency matrix?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes sacrificing some expressivity to achieve more efficient computation. I believe it makes substantive advances in both ideas and theory; however, objectively, it needs better organization and polishing, and it should provide more convincing experiments.

### Strengths
Relaxing strict graph isomorphism tests to obtain more efficient computation is an interesting idea that could help the graph learning community strike a better balance between theory and practice.

### Weaknesses
* To be fair, the paper’s clarity could be much improved; I spent considerable time ensuring I understood the main ideas. I recognize that a theory-heavy paper cannot be fully popularized, but there are concrete steps that would markedly raise readability. For example:

  * Provide a glossary at the start of the appendix. It may be long in this case, but it is necessary.
  * “Caterpillar graphs” are not formally defined until the end of page 6, yet the term appears frequently before that. The same issue applies to Theorem 4.1. I suggest reorganizing the paper so that Section 4 appears earlier in the main text.
  * Section 3 walks readers through the derivation of EFFICIENT AGGREGATION (EA). I am not convinced these details are needed in the main text for all readers; they may fit better in the appendix.
  * Add a table or relationship diagram clarifying the contributions of all theorems/lemmas/corollaries in the main text and appendix, and how they relate to each other.
  * What kind of benchmark is “NSTEPADDITION”? Is it introduced by this paper, or reported previously? Please cite prior work or provide a more detailed description in the appendix.

* The trade-off between expressivity and computational efficiency is a long-standing challenge in graph learning, so the idea of “EFFICIENT AGGREGATION” is appealing. Unfortunately, I do not see the paper emphasizing this advantage with sufficient evidence. Why is EA “efficient”? Readers will want to know how much computational improvement Caterpillar GNNs achieve over prior GNNs known for high expressivity. Expanding Table 1 to compare both expressivity and complexity would help.

* The experiments are few and confusing.

  * In Section 5.1, what exactly does “bottleneck” mean? The NSTEPADDITION benchmark is also described vaguely. What are the precise MP settings used for comparison? Is the conclusion “more expressivity hurts” restricted to NSTEPADDITION and Caterpillar GNNs, or is it general?
  * In Section 5.2, I do not see the rationale for using the average number of nodes in the computation graph on the x-axis. Computational cost (or efficiency) is not linearly tied to the number of nodes. I suggest replacing it with the height of the Caterpillar GNN. Also, §5.2 lacks comparisons against other highly expressive GNNs (and even MP), which makes it hard to substantiate the advantages of Caterpillar GNNs.
  * Experiments are conducted only on the (almost) smallest graph-classification datasets, with no larger datasets or richer task settings.
  * In the appendix, Table 4 shows MP almost always outperforming Caterpillar GNNs across settings. This raises doubts about the practical utility of Caterpillar GNNs in real-world scenarios.
  * Regarding the authors’ claim of efficient aggregation, the paper should also report the comparison of time and memory consumption with other GNNs.

### Questions
See weakness

### Soundness
2

### Presentation
1

### Contribution
2
