# Triangle Multiplication is All You Need for Biomolecular Structure Representations

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
AlphaFold has transformed protein structure prediction, but emerging applications such as virtual ligand screening, proteome-wide folding, and de novo binder design demand predictions at a massive scale, where runtime and memory costs become prohibitive.
A major bottleneck lies in the Pairformer backbone of AlphaFold3-style models, which relies on computationally expensive triangular primitives—especially triangle attention—for pairwise reasoning.
We introduce Pairmixer, a streamlined alternative that eliminates triangle attention while preserving higher-order geometric reasoning capabilities that are critical for structure prediction.
Pairmixer substantially improves computational efficiency, matching state-of-the-art structure predictors across folding and docking benchmarks, delivering up to 4x faster inference on long sequences while reducing training cost by 34%.
Its efficiency alleviates the computational burden of downstream applications such as modeling large protein complexes, high-throughput ligand and binder screening, and hallucination-based design.
Within BoltzDesign, for example, Pairmixer delivers over 2x faster sampling and scales to sequences 30% longer than the memory limits of Pairformer.
Code is available at https://github.com/genesistherapeutics/pairmixer.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work applies triangle multiplication in the large scale of current biomolecular structure prediction. Such application scenario has not been investigated with models using triangle multiplication. The authors show that, for the Pairformer backbone, when maintaining triangle multiplication, and omitting triangle attention, the training and inference efficiency can be improved.

### Strengths
1. The paper is well-written, clearly depicting the methodology;
2. The training cost and inference speed are both improved;
3. The large-scale biomolecular structure prediction scenario is of great practical importance;
4. The experiments are sufficient.

### Weaknesses
1. This work lacks of methodological contribution. The authors omit other modules and maintaining only the triangle multiplication. This modification is considered very trivial and not that novel. The importance of triangle multiplication has already been investigated by several proceeders, as the authors themselves claimed in the paper. The author is the first one deleting triangle attention for PairFormer.  The main contribution is testing if we can achieve better performance-efficiency trade-off using only triangle multiplication. 
2. The efficiency improvement is not that satisfying. To be fair, it is adequate when the methodological contribution is enough. But since the contribution is minor, I would expect giant efficiency leap to complement the limited contribution.

### Questions
Whether there is any non-trivial part for deleting triangle attention and other modules? This could be a potential methodological contribution if there is any.

Suggestion: If my evaluation remains consistent after I read the authors' response, I would suggest that the author considers submitting this work to an application-oriented venue, e.g., Nat. Comm., Nat. Comp. Sci., Sci. Adv. The value of this work is mostly about the application results but without meaningful methodological insights. After giving stronger application evaluations, packing as a tool box or executable empirical platform, this work is more suitable to those top application-oriented venues. The methodological novelty, in my opinion, is enough for them. 

I am also open to change my mind if the author can prove the methodological value that I failed to see.

### Soundness
3

### Presentation
4

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
This paper addresses the computational efficiency issue in AlphaFold and argues that the triangle attention operation contributes little to the final predictive performance while introducing significant computational overhead. Based on this observation, the authors propose the PairMixer block, a simplified variant of the Pairformer block used in AlphaFold3, in which operations with minimal performance benefits are removed. Experimental results on the RCSB test set demonstrate that PairMixer achieves comparable performance to the original Pairformer while offering up to a 4x improvement in inference speed.

### Strengths
1. This paper focuses on an important research problem, i.e., to accelerate AlphaFold and make it more lightweight, which is critital for down-stream applications like virtual screening;
2. This paper is well-written and clearly-structured. The figures effectively support the understanding of the proposed method, and the authors provide sufficient background and preliminaries to contextualize their work.

### Weaknesses
1. The contribution of this paper appears limited. The proposed method can be viewed primarily as an engineering optimization of the original Pairformer, without introducing substantial new insights. Without deeper analysis or justification of the design choices, the current contribution may not meet the novelty threshold typically expected for a venue such as ICLR. Furthermore, the finding that the triangle attention module contributes minimally to performance is not particularly surprising; this has been informally noted by several researchers through ablation studies, even though such observations have not been formally published.

2. The experimental evaluation of PairMixer is insufficient. The results are reported only on the RCSB dataset (533 structures), which limits the generalizability of the conclusions. It is recommended that the authors adopt a broader evaluation protocol, such as the one used in Boltz-2, to strengthen the empirical validation of their method.

3. The use of the phrase “all you need” in the title, while common in machine learning literature, is not appropriate in this context. The PairMixer/Pairformer serves only as the model trunk within a structure prediction framework, whereas the diffusion module plays an equally important role. Therefore, the current title may overstate the scope and completeness of the proposed contribution.

### Questions
No further questions. Please see the “Weaknesses” section for details.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Pairmixer, a modified AlphaFold3-style backbone that (a) deletes triangle attention and sequence updates from the Pairformer backbone, (b) keeps only triangle multiplication + pairwise FFNs to update the pair representation, and (c) leaves the single-sequence representation unchanged and feeds it directly to the downstream diffusion module. The experiments are shown with maintaining folding / docking accuracy while significantly improving speed (up to 4× faster inference on 2048-token sequences and 34% less training compute), and unlock downstream design workflows that previously ran out of memory.

### Strengths
Reviewer appreciates the following contributions:

- **impactful and practical**: the paper addresses a real bottleneck in AlphaFold-style models by removing triangle attention for faster, more scalable inference and training. Furthermore, it enables long-sequence and large-complex modeling that was previously infeasible due to memory or computational limits.

- **Empirical validation**: Demonstrates near-identical accuracy to AlphaFold3-class baselines across folding, docking, and binder design tasks, with up to 4× speedup and lower memory use. Experiments include comparisons on Boltz-1, Transformer, and ablations, FLOPs analysis, and realistic large-scale design benchmarks.

- **Simple but be efficient**: the paper shows that triangle multiplication alone suffices for capturing higher-order geometric consistency, providing a clearer understanding of what inductive biases matter. In particular, the final architecture will be the form of *Pairmixer = triangle multiplication + FFN over z, recycle it N times*. 

- **presentation**: the paper is well-written and easy to follow.

### Weaknesses
- **Method Novelty**: Novelty somewhat incremental: Prior works (e.g., Genie2, MiniFold) already suggested triangle multiplication is key; this paper mainly extends the idea to AF3 scale rather than introducing it conceptually. This requires further analysis to highlight key differences between Pairmixer versus prior works, for e.g., what should we do to adapt for the protein structure design task?

- **Limited generalization tests:**
While the paper benchmarks extensively on protein–protein and protein–ligand systems, all evaluations remain within domains similar to the training distribution of Boltz-1 (PDB-scale protein complexes). The work does not assess generalization to more diverse biomolecular systems, such as RNA–protein assemblies, RNA-only structures, metalloproteins, or highly flexible/transient complexes. These categories often require different geometric reasoning and long-range constraints, where triangle attention might still provide advantages. Without such tests, it’s quite unclear whether the proposed architecture truly generalizes beyond well-structured protein complexes, or if its performance degrades on systems with **unconventional topologies or more dynamic conformational behavior**.

### Questions
**Missing sequence update ablation:**
The paper removes sequence updates but doesn’t isolate their effect. It’s unclear how much of the performance change comes from dropping triangle attention versus removing sequence updates.

**Lack of discussion on limitations vs AlphaFold3:**
Can authors discuss where the simplified model may fail compared to full AlphaFold3 — for example, in modeling highly flexible regions, RNA–protein complexes, or subtle side-chain rearrangements requiring long-range attention?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a new operator, Triangle Multiplication, as a core mechanism for relational reasoning and geometric representation learning. The method is presented as a lightweight yet expressive alternative to traditional self-attention, aiming to capture higher-order interactions among triplets of entities efficiently. The authors apply this operation to tasks, showing that it can achieve competitive or improved performance compared to transformer-style baselines.

The conceptual idea is creative and well-motivated; however, the presentation lacks sufficient technical clarity, the experimental scope is somewhat limited, and the empirical analysis does not fully demonstrate the operator’s claimed generality.

### Strengths
1.	Interesting conceptual direction: The idea of moving from pairwise attention to triangle-based relational modeling is novel and aligns with emerging research on geometric and higher-order attention.
2.	Simplicity of the operator:  The formulation is elegant and could potentially be a computationally efficient substitute for attention in specific contexts.
3.	Potential for extension: The proposed mechanism could inspire further work in 3D molecular or graph-structured domains, where triplet relations are natural.
4.	Readable overall motivation:  The high-level rationale and related work are generally well-written.

### Weaknesses
- Limited Comparative Breadth

The experiments benchmark against a few baselines but omit several directly relevant contemporary models, including:
	1. Higher-order attention variants (e.g., Tensor Attention, Relational Transformer)
	2. Geometric and 3D reasoning frameworks (e.g., SE(3)-Transformer, EGNN)
	3. Diffusion-based relational models and equivariant graph networks.

Without these comparisons, it is difficult to judge whether Triangle Multiplication provides a fundamentally better abstraction or merely a reparameterization of higher-order attention.

- Lack of Theoretical Clarity

The mathematical definition of “Triangle Multiplication” is presented at a high level but lacks rigorous derivation or clear connection to known tensor operations:
	1. The operator’s expressive power (what functions it can approximate) is not discussed.
	2. No complexity analysis is provided—readers cannot tell if it scales better than self-attention for large N.

- Weak Empirical Validation

While the experiments show some improvement, they remain qualitative and dataset-limited:
	1. The selected tasks are small-scale and do not reflect real-world complexity.
	2. No ablation studies are provided to isolate the contribution of the triangle operator vs. other components.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
2
