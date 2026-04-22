# Wavelet-Induced Rotary Encodings: RoPE Meets Graphs

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
We introduce WIRE: Wavelet-Induced Rotary Encodings. WIRE extends Rotary Position Encodings (RoPE), a popular algorithm in LLMs and ViTs, to graph-structured data. We demonstrate that WIRE is more general than RoPE, recovering the latter in the special case of grid graphs. WIRE also enjoys a host of desirable theoretical properties, including equivariance under node ordering permutation, compatibility with linear attention, and (under select assumptions) asymptotic dependence on graph resistive distance. We test WIRE on a range of synthetic and real-world tasks, including identifying monochromatic subgraphs, semantic segmentation of point clouds, and more standard graph benchmarks. We find it to be effective in settings where the underlying graph structure is important.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces WAVELET-INDUCED ROTARY ENCODINGS (WIRE), a novel method that adapts the successful Rotary Position Encodings (RoPE) from sequence models (LLMs/ViTs) to arbitrary graph-structured data.

### Strengths
1. Generalization of RoPE: WIRE demonstrates that RoPE is a special case of their method, recovering the original encoding when applied to $N$-dimensional grid graphs (like 1D sequences or 2D images).
2. The method leverages the graph spectrum (eigenvectors of the graph Laplacian) as generalized coordinates for rotation. 
3. Like RoPE, WIRE directly modifies the query ($\mathbf{q}$) and key ($\mathbf{k}$) vectors. This design maintains compatibility with linear attention and KV-caching, allowing for $\mathcal{O}(N)$ scaling in the number of tokens/nodes $N$, which is highly advantageous for large graphs compared to $\mathcal{O}(N^2)$

### Weaknesses
1. The largest weakness is that the core mechanism of using rotation matrices to encode relative position is directly derived from the highly successful Rotary Position Encoding (RoPE), the fundamental rotational approach used in this paper is borrowed, diminishing the conceptual novelty of the encoding function itself. 

**The paper's empirical results consistently show only marginal improvements. This fact, combined with the lack of novelty in the underlying Rotary Encoding mechanism, challenges the work's overall contribution.**

2. Marginal Empirical Efficacy vs. High Precomputation Cost (Primary Concern): While the method is efficient during inference 
($\mathcal{O}(N)$ scaling with linear attention), the necessary one-time precomputation of the graph Laplacian eigenvectors requires $\mathcal{O}(N^3)$ complexity for exact calculation. 
3. Given that the empirical performance gains over competitive graph position encoding baselines (like Random Walk Encodings or standard Laplacian PE) are often marginal (e.g., less than 1\% increase in accuracy), the practical cost-benefit trade-off is difficult to justify for large-scale, static graphs. This suggests the regularization benefit may not warrant the severe precomputation bottleneck.
4. The paper theoretically links WIRE's structural bias to the graph resistive distance (effective resistance). However, the experimental results fail to provide a clear, predictive understanding of when this specific distance metric is superior to other widely-used metrics (e.g., shortest path distance, diffusion distance). The authors should include an ablation study on diverse graph topologies (e.g., clustered, scale-free, dense vs. sparse) to demonstrate the specific structural properties that make WIRE's resistive distance bias definitively advantageous.

### Questions
q1. The theoretical connection to graph resistive distance is a key selling point. However, the paper does not show when this specific bias is superior. Please design and execute a targeted ablation study comparing WIRE to other distance-based Position Encodings.

q2.  The number of spectral dimensions ($m$) is a hyperparameter. Please elaborate on the relationship between $m$ and the graph topology.

q3. What is the maximum graph size $N$ for which the precomputation time of the Laplacian is still practical compared to the total training/inference time?

q4. For sparse graphs, please provide detailed benchmarks comparing the total training+precomputation time of WIRE against baselines

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
3

### Summary
The paper proposes Wavelet-Induced Rotary Encodings (WIRE), which extends Rotary Position Encodings (RoPE) to graphs by replacing Euclidean coordinates with spectral coordinates derived from the graph Laplacian. The authors claim that WIRE generalizes RoPE, preserves equivariance under node permutations, exhibits dependence on resistive distance, and remains compatible with linear attention. Experiments span synthetic graph tasks, point-cloud classification, and standard graph benchmarks.

### Strengths
This work attempts a theoretically grounded fusion of spectral graph theory and positional encoding. The asymptotic relationship identified between resistive distance and expected attention scaling presents an intriguing theoretical perspective. Furthermore, its compatibility with linear attention mechanisms suggests promising computational practicality. Finally, the implementation itself appears concise, modular into existing Transformer-based pipelines.

### Weaknesses
The proposed method demonstrates several critical shortcomings from both graph-theoretic and methodological perspectives. The central construction, which leverages Laplacian eigenvectors as positional encodings, is fundamentally well-established in prior literature, undermining claims of novelty. Furthermore, the paper’s reference to the framework as "wavelet-induced" lacks conceptual rigor, as the implementation relies exclusively on global eigenfunctions of the graph Laplacian, without genuine multiresolution analysis, scale-adaptivity, or localized wavelet bases. Consequently, the chosen terminology misrepresents the spectral construction and obfuscates important distinctions between localized and global harmonic analyses.

The manuscript additionally overstretches its claim regarding permutation equivariance (Remark 1, p. 4). Specifically, eigenvectors of the graph Laplacian inherently lack canonical orientation due to arbitrary sign flips and orthogonal transformations within eigenspaces corresponding to repeated eigenvalues. Thus, without explicit strategies for eigenbasis alignment, the approach inherently fails to maintain invariance under graph isomorphisms. This issue severely limits the practical permutation equivariance claimed by the authors, thereby undermining their theoretical positioning.

Assertions regarding the model’s expressivity, particularly that it surpasses standard graph neural networks (GNNs) by distinguishing graphs indistinguishable under the Weisfeiler–Lehman (1-WL) test, are mathematically unfounded. The existence of isospectral but non-isomorphic graphs, a well-documented phenomenon, indicates that spectral encodings alone cannot reliably enhance graph discriminability beyond classical GNN benchmarks. Consequently, the claim of superior expressive power, as formulated in the manuscript, is both misleading and lacks rigorous theoretical justification.

Moreover, the provided theoretical results, notably Theorem 2, exhibit limited practical relevance and rigor. The resistive-distance connection presented (Equation 7) arises under restrictive assumptions, including random frequency sampling and infinitesimal perturbations (small-ω Taylor expansions). Such assumptions are invalidated during practical model training, wherein frequencies are learned parameters rather than random Gaussian samples. Hence, the deterministic interpretation of resistive-distance dependence is compromised, relegating this insight to an approximate, rather than fundamental, relationship.

The scalability claims related to graph-level computations also appear problematic. Although linear complexity $O(N)$ is emphasized, even approximate computations of leading Laplacian eigenvectors are practically known to exceed linear complexity in realistic scenarios. The so-called "efficient diagonalization" strategy (§A.2) merely reiterates known random-feature kernel approximations, conflating approximate low-rank feature-space expansions with accurate spectral decomposition. This misrepresentation overlooks critical aspects such as spectral accuracy, eigenvector orthogonality, and associated numerical stability.

Conceptually, the authors conflate the notions of coordinate systems and feature encodings. By simultaneously using Laplacian eigenvectors as direct inputs and as positional coordinates for rotary encodings, the method inadvertently duplicates identical spectral information. Observed empirical gains (Tables 1–4) may thus reflect implicit data augmentation rather than the introduction of novel geometric inductive biases, undermining the method’s purported theoretical foundation.

### Questions
1. How do you resolve the non-uniqueness of eigenvectors and sign ambiguity under node permutation, especially for graphs with repeated eigenvalues?

2. Can you provide a concrete example of two isospectral graphs and demonstrate whether WIRE distinguishes them?

3. Why call the method “wavelet-induced” when no localized or scale-separable transform appears?

4. In Theorem 2, what happens when the frequencies are learned rather than sampled from $\mathcal{N}(0,  \omega I)$? Does the resistive-distance term persist?

5. Have you compared against diffusion-based or random-walk positional encodings that already capture resistive-distance–like behavior?

6. How sensitive are your reported results to small perturbations of the eigenbasis (e.g., random rotations within degenerate eigenspaces)?

7. Does WIRE preserve equivariance on disconnected or weighted graphs, or does the Laplacian normalization alter the claimed invariance?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes WIRE (Wavelet-Induced Rotary Encodings), a RoPE-style positional encoding for graphs. Nodes receive spectral coordinates from low-frequency Laplacian eigenvectors; these act as rotation angles for queries/keys, yielding a RoPE-like rotation that is permutation-equivariant and compatible with linear attention. Theory shows (i) RoPE is a special case on grid graphs and (ii) in expectation over random frequencies and for small $\omega$, attention logits are down-weighted proportional to effective resistance. Experiments on synthetic tasks, point-cloud segmentation, and graph benchmarks show modest gains, especially with linear-attention Performers.

### Strengths
- The paper presents a simple mechanism that preserves RoPE’s efficient 2×2 block structure; clearly compatible with linear attention.

- WIRE is equivariance with respect to node permutations and it has clean connection to graph spectra; nice unification as RoPE is a special case of WIRE (grid graphs).

- The effective-resistance perspective is appealing and could inspire principled topological masking.

### Weaknesses
- First concern. The theoretical analysis feels shallow and fragile. Theorems 1 and 2 are interesting but not sufficiently strong. The claim of being “more expressive than standard GNNs” is asserted but not formally quantified (e.g., beyond 1-WL on explicit classes). For a more complete complexity analysis, the paper should provide end-to-end cost bounds for computing mmm eigenvectors versus the gains over RoPE, including scaling with graph size and batch construction.

- Second concern. WIRE loses translation invariance. RoPE’s translational invariance is a key ingredient for length generalization. In WIRE, on the one hand, “translations” in spectral space lack a consistent geometric meaning across graphs; on the other hand, translation invariance is explicitly lost, as acknowledged by the authors. I view this as a weakness of WIRE.

- Third concern (empirics). The empirical gains are marginal and do not close the gap to full Transformers. On point clouds, improvements over baselines are small and may fall within run-to-run variance. On benchmark graphs, WIRE-Performer typically offers only small, incremental gains over Performer, while still underperforming the quadratic Transformer on most datasets.

### Questions
- How sensitive is performance to $m$? Please provide curves of accuracy vs. $m$ and runtime/memory vs. $m$, and discuss truncation error.

- How do results change under different kNN/radius graphs or learned graphs? Does WIRE still help when the topology is noisy or misspecified?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes WIRE (Wavelet-Induced Rotary Encodings), extending Rotary Position Encoding (RoPE) from sequences and grids to graph-structured data. WIRE constructs spectral coordinates for each node from the Laplacian eigenvectors and then applies a RoPE-style rotation to queries and keys in the attention mechanism. It claims the following: (i) WIRE generalizes RoPE and recovers it on grid graphs. (ii) It is permutation-equivariant, compatible with linear attention, and (under assumptions) relates to graph effective resistance. (iii) Experiments on synthetic and benchmark graph datasets show performance competitive with prior Laplacian or distance-based encodings.

### Strengths
- Conceptual clarity: Presents a clean and mathematically consistent generalization of RoPE to arbitrary graphs.

- Theoretical analysis: Proves properties such as permutation-equivariance and RoPE-equivalence on grids.

- Computational efficiency: Keeps O(N) complexity per node and avoids storing full attention matrices that might be beneficial for large graphs.

- Simplicity and compatibility: Can be easily plugged into any transformer or hybrid GNN-Transformer architecture.

- Broad scope: Demonstrates that spectral node embeddings can serve as rotational coordinates, offering a bridge between spectral graph theory and token rotations.

### Weaknesses
- Lack of novelty vs. existing wavelet/spectral encodings: The idea of using spectral or wavelet-based multiscale coordinates is already established in Multiresolution Graph Transformers and Wavelet Positional Encoding (WavePE) published with Journal of Chemical Physics [1] and Range-aware Graph Positional Encoding (HOPE-WavePE) published with Machine Learning: Science and Technology [2], both of which exploit multi-resolution, high-order, permutation-equivariant and wavelet-based positional features. WIRE largely reuses the same spectral foundations, adding only a RoPE-style rotation.

- Limited multiscale analysis: Despite its title, WIRE does not employ true multiresolution wavelet transforms or scale-adaptive kernels as in prior works. The "wavelet" aspect is mostly nominal.

- Insufficient empirical depth: Experiments appear small-scale and lack direct comparisons with WavePE, HOPE-WavePE, or other wavelet/graph transformer baselines.

- Missing connection to prior literature: No citation or discussion of earlier wavelet-based graph positional encodings, making the contribution seem incremental.

- Unclear practical benefit: Improvements over LapPE or RWPE are modest; ablation on frequency selection or spectral truncation is missing.

*** References:

[1] Nhat Khang Ngo, Truong-Son Hy, and Risi Kondor, Multiresolution Graph Transformers and Wavelet Positional Encoding for Learning Long-Range and Hierarchical Structures, Journal of Chemical Physics, Volume 159, Issue 3, DOI 10.1063/5.0152833.
URL: https://pubs.aip.org/aip/jcp/article-abstract/159/3/034109/2903066/Multiresolution-graph-transformers-and-wavelet?redirectedFrom=fulltext

[2] Viet Anh Nguyen, Nhat Khang Ngo, and Truong-Son Hy, Range-aware Positional Encoding via High-order Pretraining: Theory and Practice, accepted at Machine Learning: Science and Technology, DOI 10.1088/2632-2153/ae1acd.
URL: https://iopscience.iop.org/article/10.1088/2632-2153/ae1acd

### Questions
How does WIRE fundamentally differ from prior Wavelet Positional Encoding (WavePE) and HOPE-WavePE methods that already model multiresolution graph structure through spectral or wavelet bases? Could you quantify whether the rotary transformation itself, not just spectral features, provides measurable advantages in accuracy or robustness?

### Soundness
3

### Presentation
3

### Contribution
3
