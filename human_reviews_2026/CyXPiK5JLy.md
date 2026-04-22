# Get the GIST of Graphs with Intersection Signature

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
Graph Transformers have emerged as a promising alternative to Graph Neural Networks (GNNs), offering global attention that mitigates oversmoothing and oversquashing issues. However, their success critically depends on how structural information is encoded, especially for graph-level tasks such as molecular property prediction. Existing positional and structural encodings capture some aspects of topology, yet overlook the diverse and interacting substructures that shape graph behavior. In this work, we introduce Gisty Intersection Signature Trait (GIST), a structural encoding based on the intersection cardinalities of k-hop neighborhoods between node pairs. GIST provides a permutation-invariant representation that is theoretically expressive, while remaining scalable through efficient randomized estimation. Incorporated as an attention feature, GIST enables Graph Transformers to capture fine-grained substructures together with node-pairwise relationships that underlie long-range interactions. Across diverse and comprehensive benchmarks, GIST maintains a uniformly strong performance profile: head-to-head evaluations consistently favor GIST, underscoring its role as a simple and expressive structural feature for Graph Transformers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GIST, a graph transformer architecture that encodes structure through pairwise intersections of k-hop neighborhoods. The motivation is to provide an alternative to distance- or Laplacian-based positional encodings by measuring structural overlap directly. Theoretical claims include improved expressiveness over SPD-WL, RD-WL, and RRWP, and invariance under graph isomorphism. Experiments show consistent improvements on standard molecular and vision graph benchmarks, using sketch-based approximations (MinHash and HyperLogLog) for efficiency.

The main problem is that the contributions are overstated. The neighborhood intersection idea is already established in the literature, particularly in Chamberlain et al., where the same approximation techniques are used. The paper’s formulation adds little beyond integrating that encoding as a transformer bias. The technical development is inflated and overpormises.

### Strengths
Using k-hop commong neighborhoods as a proxy for structural information is an established technique in the literature. This paper proposes its application also as a structural encoding for graph transformer architectures, which is a natural and promising next step.

The proposed encoding has a simplicity to it that parallels prominent encodings such as RWSE or LapPE which can make it easily applicable for a wide variety of tasks.

The empirical results are very strong. The reported results are pushing the boundary on a variety of standard benchmarks, commonly surpassing elaborate state of the art architectures.

### Weaknesses
## Novelty / Attribution

The attribution to Chamberlain et al is insufficient. The idea of using pairwise encodings of k-hop neighborhoods as an efficient proxy for substructure information is the core of Chamberlain et al., which is already referenced but only in a vague sense regarding the methods to approimate these encodings. There, the formulation of “Following […], we propose […]” is not appropriate in my opinion. The approximation approach for pairwise k-hop common neighborhood is precisely present in Chamberlain et al., seemingly using the exact same techniques. In a way this work could even be summarized as using the techniques of Chamberlain et al. as a structural encoding for graph transformers. The one vague reference to their work is therefore  insufficient for proper attribution and recognition of novelty.

Given this large overlap with prior work, it remains unclear to me where the major novelty lies in the submission.

## Technical Development

The technical discussion in this paper is unfocused and unnecessarily opaque. Some concrete points towards this are listed below:

Line 191: why not just have the sum run for $x < k_u$ and $y < k_v$? Also it might be even simpler to define it via the neighborhoods, i.e., use that
$(N_{k_u}(u) \setminus N_{k_u-1}(u))$ are the edges exactly k_u steps from u and then take the intersection of this set with the corresponding set for v.

Line 198: similarly, just use $V(G) \setminus N_k(v)$ for the nodes that are further away than k from v and intersect with the precise k_u neighborhood as in the previous comment.

Line 205: given the previous comments, I find don’t understand the point in emphasizing the definition of $C_{k_u,k_v}$ since it seems to just complicate and obfuscate Definition 3.1 for no benefit.


Line 220: what is the point of the hash function on a single multiset? Theoretically you gain nothing since it’s just a bijection and in practice I think you use the actual multiset instead of the hash.


Theorem 3.3: the statements should be made more precise to avoid misleading the reader. Looking at the details in the appendix reveals that the results are much more narrow in scope than the theorem’s phrasing implies:
- It should be made clear that n here is the number of vertices because that does relativize points 1 and 2 quite a lot.
- In point 2 & 3,  “no less expressive” is very misleading. What is shown is that there exists one pair of graphs that can be distinguished by GIST(6) but not by RRWP(3) or RD-WL, nothing more.  
- In point 3 of the theorem the some kS,kR is needlessly opaque. The proof is just for 3 and 6 specifically.
- In the proof you begin item 2 with a conjecture that seems unrelated to what is proven or what is stated in the theorem? 

On top of this, SPD-WL and RD-WL are not defined and seemingly not even cited(??) in this submission. 
I would have expected a statement on the relation to 2-FWL since that would be the most obvious point of comparison for representation on all node-pairs. It seems intuitive to me that 2-FWL actually is strictly more expressive than GIST and it would be helpful (and I believe of more interest than the RD-WL or RRWP results) to have a clear picture where to place GRIT in terms of expressivity.

Definition 4.1 is established standard and should be referenced. At the very least it should be made clear that this is not a contribution of the submission.

### Questions
Is there any reason that Theorem 4.3 is not trivial? Does this not follow immediately from the k-hop substructure encoding being invariant under isomorphism? If so clarify that this is the case when stating the theorem or drop it entirely. (Or maybe make it a proposition about the encoding?). If not, please make it clear in the paper why this theorem is needed.

Can you please clearly describe what the difference is between what you propose and what is already done in Chamberlain et al.? Given that can you please make the novelty of this submission clearer.

### Soundness
3

### Presentation
1

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
This paper introduces GIST (Gisty Intersection Signature Trait), a novel structural encoding method for Graph Transformers. GIST encodes the intersection cardinalities of k-hop neighborhoods between node pairs, producing a permutation-invariant and expressive representation of graph substructures. The authors propose an efficient randomized estimation algorithm using MinHash and HyperLogLog, reducing computational complexity. GIST is integrated into the self-attention mechanism as an additional structural bias, enhancing the model’s capacity to capture fine-grained substructures and long-range dependencies.

Empirical evaluations across ZINC, Peptides, MoleculeNet, and LRGB benchmarks demonstrate consistent improvements over strong baselines such as GraphGPS, GRIT, Subgraphormer, and SPSE. Theoretical analysis shows that GIST is at least as expressive as several existing distance-based or random-walk-based positional encodings, and in some cases strictly more expressive.

### Strengths
1. The idea of using intersection cardinalities of k-hop neighborhoods is both intuitive and mathematically grounded. It provides a meaningful bridge between substructure-level and global structural information.
2. The paper offers a formal expressiveness comparison to other positional encodings (e.g., SPD-WL, RD-WL, RRWP). The proof sketch and theoretical guarantees add credibility and clarity to the method’s contribution.
3. The use of randomized hashing (MinHash + HyperLogLog) to approximate neighborhood intersections is elegant and practical. The computational reduction is well justified, and implementation details are transparent.

### Weaknesses
1. While GIST’s theoretical expressiveness is well discussed, the paper does not empirically isolate how much of the gain comes from the theoretical novelty versus implementation-level improvements (e.g., intersection vs. distance-based encoding).
2. Although the authors claim robustness, the main text lacks quantitative sensitivity analysis (delegated to the appendix). A concise presentation of trade-offs between accuracy and computational cost would strengthen the practical message.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
The paper “Get the GIST of Graphs with Intersection Signature (GIST)” proposes a novel structural encoding method for Graph Transformers, addressing their well-known difficulty in effectively capturing graph topology.Traditional Graph Neural Networks (GNNs) suffer from oversmoothing and oversquashing due to local message passing, while Graph Transformers, though capable of modeling long-range dependencies, rely heavily on how structural information is encoded. Existing encodings (e.g., shortest path, Laplacian, or random walk features) only partially represent graph structure and often fail to model interactions among diverse substructures.

To overcome this, the authors introduce GIST (Gisty Intersection Signature Trait) : a structural encoding based on the intersection cardinalities of k-hop neighborhoods between node pairs. GIST serves as a permutation-invariant and expressive representation that captures fine-grained substructure interactions across the entire graph.

The main contributions are:

Novel Encoding Framework: GIST encodes graph structure using pairwise intersection features of k-hop neighborhoods, effectively capturing both local and global substructure relationships.

Efficient and Scalable Computation: The authors design a randomized estimation method (using MinHash and HyperLogLog) to compute GIST efficiently, reducing complexity from O(k2n4)

Integration with Graph Transformers: GIST is incorporated as a learnable attention bias (“GIST attention”), allowing Transformers to use these structural cues for improved node pair interactions. The method is proven to be isomorphism-invariant and theoretically more expressive than several existing structural encodings.

Comprehensive Evaluation: Extensive experiments on 14 benchmark datasets (including ZINC, MoleculeNet, Peptides, MNIST, and CIFAR10) show that GIST consistently achieves state-of-the-art or near state-of-the-art performance. It also demonstrates strong scalability, robustness to hyperparameters, and effective generalization to node- and graph-level tasks.

Overall, the paper contributes a simple, theoretically sound, and empirically effective structural encoding that strengthens Graph Transformers’ ability to model complex substructures and long-range dependencies in graphs.

### Strengths
Originality:
The paper introduces a highly original structural encoding method, GIST (Gisty Intersection Signature Trait), that reframes how Graph Transformers capture graph topology. Unlike prior positional or distance based encodings, GIST leverages the intersection cardinalities of k hop neighborhoods to characterize higher order substructure interactions. This formulation is both conceptually novel and technically elegant, offering a new way to integrate relational information without relying on handcrafted motifs or message passing. The randomized estimation approach using MinHash and HyperLogLog further demonstrates creative thinking in scaling theoretical insights to practical use.

Quality:
The technical quality of the paper is strong. The authors provide solid theoretical grounding, showing that GIST is isomorphism invariant and strictly more expressive than several established encodings such as SPD WL and RRWP. The experimental design is thorough, covering 14 benchmark datasets across molecular property prediction, long range dependency modeling, and graph classification. Comparisons with a comprehensive set of strong baselines including GraphGPS, GRIT, and Subgraphormer reinforce the robustness of results. Implementation details, ablation studies, and scalability analyses are clearly reported, adding confidence in reproducibility and validity.

Clarity:
The paper is well organized and generally easy to follow despite the technical depth. The motivation is clearly articulated, with intuitive figures such as substructure intersection visualizations that help convey the core idea. Definitions are rigorous yet readable, and the progression from theoretical formulation to algorithmic implementation is smooth. The writing effectively balances formalism and intuition, which makes the contribution accessible to both theoretical and applied audiences in graph learning.

Significance:
The work makes a substantive contribution to the design of structure aware Graph Transformers, a rapidly growing research area. By providing a scalable, expressive, and permutation invariant encoding, GIST has clear potential to become a standard component in future graph Transformer architectures. Its ability to model complex substructure interactions extends the reach of Transformers to tasks in chemistry, biology, and materials science, where structural detail is crucial. The simplicity and generality of GIST also make it a promising foundation for future work in efficient graph representation learning and model interpretability.

### Weaknesses
Limited conceptual comparison to closely related structural encodings:
While GIST presents a novel formulation based on intersection cardinalities, the paper could strengthen its conceptual positioning relative to recent graph Transformer encodings that also aim to capture higher order or substructure information. For example, works such as SPSE (Airale et al., 2025) and FragNet (Wollschlager et al., 2024) introduce path and fragment based encodings that similarly emphasize richer structural dependencies. The paper briefly cites these methods but does not provide a deeper analytical or empirical comparison to clarify what unique structural biases GIST captures beyond them. A more systematic ablation comparing GIST against these structurally similar encodings under matched model capacity would make the originality claim more robust.

Theoretical exposition could be expanded for clarity and intuition:
Although the authors provide a theoretical proof of expressiveness, the connection between the formal results (e.g., comparisons with SPD WL and RRWP) and the empirical improvements remains somewhat abstract. The theory section would benefit from a more intuitive explanation or illustrative examples that show how intersection cardinalities encode structural distinctions that distance based or random walk encodings miss. This would help bridge the gap between theoretical guarantees and the observed empirical performance.

Experimental scope could better test generalization and scalability claims:
The experimental results are comprehensive, but most benchmarks are medium scale molecular or synthetic datasets. To convincingly support the claim of scalability to large graphs, additional experiments on large scale graph datasets such as OGBN Papers100M or OGBG Code2 would be valuable. These would more concretely demonstrate GIST’s computational advantages from the MinHash and HyperLogLog estimations and clarify the tradeoff between approximation accuracy and model performance.

Limited ablation on key design components:
While the paper includes some ablations, they focus primarily on hyperparameters (e.g., hop count or hash functions). It would strengthen the work to include controlled analyses that isolate the contribution of the intersection based features themselves. For instance, comparing GIST attention against versions using only shortest path distance or neighborhood overlap without full k hop intersection vectors would more precisely quantify the gain from the proposed encoding. Similarly, evaluating the impact of randomized estimation error on downstream accuracy would support the efficiency claims more convincingly.

Presentation and reproducibility aspects:
Although the paper is generally well written, some definitions (e.g., the construction of k hop substructure vectors) are dense and may hinder reproducibility for readers less familiar with the notation. Providing pseudocode or a high level algorithm box summarizing the computation of GIST and its integration into attention would improve clarity. In addition, since the code release is deferred, the paper would benefit from including at least partial implementation details (e.g., hash function parameters, runtime complexity benchmarks) to ensure results can be independently verified.

### Questions
Clarification of theoretical expressiveness:
The paper claims that GIST is strictly more expressive than SPD WL and comparable to or stronger than RRWP and RD WL encodings. Could the authors provide an intuitive explanation or illustrative example showing how intersection cardinalities capture structural information that distance or random walk based encodings cannot? This would help readers connect the theory with the observed empirical gains.

Quantifying approximation error and scalability tradeoffs:
The proposed efficiency improvement relies on MinHash and HyperLogLog estimations. How much variance or bias do these approximations introduce in practice, and how does this affect model accuracy? Reporting runtime and accuracy tradeoffs, or error bounds on representative datasets, would make the scalability claims more convincing.

Comparison to recent structural encodings:
Recent approaches such as SPSE (Airale et al., 2025) and FragNet (Wollschlager et al., 2024) also capture substructure relationships. Could the authors clarify how GIST’s intersection based features differ conceptually or empirically from these methods? A targeted comparison under similar model settings would help isolate GIST’s unique contribution.

Generality and applicability beyond molecular datasets:
Most experiments focus on molecular graphs, which have regular and chemically meaningful structures. How would GIST perform on graphs with very different properties, such as citation or social networks? Discussing whether intersection based encoding remains effective in sparse, irregular, or high degree graphs would strengthen the generality claims.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduces GIST (Gisty Intersection Signature Trait), which is a structural encoding for graph transformers that is based on the intersection cardinalities of k-hop neighborhoods between node pairs. The paper provides a set of theoretical expressivity claims for GIST and also conducts experiments to support the GIST’s practical utility.

### Strengths
- The motivation of incorporating neighbourhood intersection information as a structural encoding is sound and well laid out.
- Empirical results are quite strong, and the authors show consistent performance gains across several different datasets and domains.

### Weaknesses
- My main concern is with the novelty of this work. It seems like the GIST encoding/features (computed with MinHash and HyperLogLog) are exactly the same as those presented in Chamberlain et al.
    - How does GIST meaningfully build on the work presented in Chamberlain et al.? It seems like the authors simply took the features presented in that work and applied them to graph transformers here, which is not enough of a contribution to the machine learning field.
- The stated theoretical contributions don’t seem very well defined or supported. In particular:
    - RD-WL (as mentioned in statement 2 of Theorem 3.3) is not defined or cited anywhere. What exactly is this? 
        - Furthermore, the authors only provide a single example for this, which does not constitute a rigorous proof. Can this statement be strengthened or formalised?
    - The proof for statement 3 of Theorem 3.3 also only provides a single example, and “some values of $k_R$ < $k_g$” is extremely vague. Do the authors have any more specific results for conditions under which this holds (ie for what $k_R$ and/or $k_g$)?
    * Theorem 4.3 seems quite trivial, and I’m not sure it’s notable enough to be considered a significant theoretical contribution.

Overall, this paper doesn’t seem fully fleshed out yet, and might be more suitable for a workshop submission in its current state (owing to the strong empirical results). Therefore, I recommend it for rejection.

References:

Chamberlain et al. Graph Neural Networks for Link Prediction with Subgraph Sketching. In ICLR 2023.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
