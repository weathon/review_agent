# Platonic Transformers: A Solid Choice for Equivariance

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
While widespread, Transformers lack inductive biases for geometric symmetries common in science and computer vision. Existing equivariant methods often sacrifice the efficiency and flexibility that make Transformers so effective through complex, computationally intensive designs. We introduce the Platonic Transformer to resolve this trade-off. By defining attention relative to reference frames from the Platonic solid symmetry groups, our method induces a principled weight-sharing scheme. This enables combined equivariance to continuous translations and Platonic symmetries, while preserving the exact architecture and computational cost of a standard Transformer. Furthermore, we show that this attention is formally equivalent to a dynamic group convolution, which reveals that the model learns adaptive geometric filters and enables a highly scalable, linear-time convolutional variant. Across diverse benchmarks in computer vision (CIFAR-10), 3D point clouds (ScanObjectNN), and molecular property prediction (QM9, OMol25), the Platonic Transformer achieves competitive performance by leveraging these geometric constraints at no additional cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces considers the problem of designing transformer architectures for points sets, which are equivariant to rotations and translations. Previous work has accomplished this by using novel equivariant transformers which are more computationally intensive than standard (non-equivariant) transformers. The method suggested in this paper relaxes the problem to equivariant to a discrete subgroup of the rotation group,  and using this relaxation shows how standard transformers can be modified to accomplish equivariance without compromising efficiency.

### Strengths
Reading this paper was a pleasant experience and the mathematics seems correct. The basic idea of the paper makes sense to me: building an efficient equivariant transformer built on the basis of existing transformers. The empirical results on QM9 especially give some basic corroboration of this idea: comparable accuracy with far superior inference time.

### Weaknesses
1. The extent of the empirical results is not very convincing: 
* In QM9, only two of the targets are reported. A skeptic could suspect that these are the only targets on which the method performs well. Results on all targets should be reported.
* In OMol25, I understand the point you make in the first table. Still, it would be more convincing if you could also compete with the reported results from the Esen  paper. Take all the training time you like. 
2. I have some point on the writing I will discuss in the questions section. These can be addressed in a small revision for the camera ready version.

I will increase my score if these issues are resolved.

### Questions
Regarding your discussion of frame averaging in line 424: is there any essential reason why one would require a separate forward pass for each frame element? Couldn't you, as a preprocessing step, turn the input from an n by 3 input to an n by 3 by 4 input (where 4 is the number of SO(3) invariant frames).


**Please do not related to what is below in the rebuttal** these are just suggestions for improvements of writing  for the next version of the paper:
* Line 47: "thereby expanding..." I feel like the logic of this sentence could be improved
* Line 84: the second $k_j$ should be $v_j$
* The explanation in Section 2 is pretty good, but it would be very helpful if you defined $p$ in the beginning, explained what its dimensions are and what translation and rotation invariance would mean for $p$. 
* The footnote in page 4 continues to page 5 which is a bit strange.
* While reading the paper I was wondering whether SO(3) equivariance was being relaxed to equivariance over the chosen subgroup. The answer only became apparent to me when this information was disclosed in subsection 5.1. I would appreciate being more forthcoming about this information earlier and more frequently.
* Line 356: I didn't follows why ScanObjectNN is considered a non-equivariant task. Do objects appear their in a predefined orientation?
* Line 402: impact is misspelled
* The grammar in the last sentence in the ethics statement is off.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents an efficient method to build equivariance in the standard Transformer architecture with respect to discrete symmetry group $G \subset SO(3)$. 
The method is specifically based on rotary positional embedding (RoPE), where the main idea is to run RoPE-based attention in parallel over a fixed set of group elements (reference frames) with shared weights across the selected frames (frames are sampled from the finite subgroup of $SO(3)$). 
The authors evaluate their methods on different input domains including: 2D images (CIFAR-10 dataset), 3D point clouds (ScanObjectNN dataset), and 3D molecules (QM9 and OMol25 datasets).

### Strengths
- I think the proposed idea is simple and interesting. With limited modifications to the standard Transformer, the authors can achieve equivariance w.r.t. discrete subgroups from SO(3) in an efficient and effective way.

- The authors also show good evaluations for their proposed method, and that it can be applied to different input domains (images, point clouds, and molecules).

### Weaknesses
* The baselines are limited to convolution in the case of vision domains. I think the paper should include stronger models in the literature and works that consider similar symmetry subgroups.
* The proposed method only achieves approximate equivariance w.r.t. $SE(3)$ or equivariance w.r.t. finite subgroups of $SO(3)$, but the paper claims multiple times using the phrase “full equivariance to Euclidean transformations”. I think some rewrites might be required, and the authors should state clearly that the goal of the idea is to build approximate equivariance/equivariance w.r.t. discrete subgroups. 
* The propositions need some more illustration, and it would be benefical to mention which symmetry group is considered. For example, Prop. 1 states  “A global roto-reflection $R \in G$ applied to the input point cloud.." Do you mean the discrete subgroup in this case?

### Questions
As 3D point clouds and molecules require continuous equivariance to the SO(3) symmetry group, how do the authors deal with this case? Did you apply rotation augmentations during training?

### Soundness
2

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
The paper presents the Platonic Transformer, which incorporates geometric equivariance into standard Transformers by defining attention relative to Platonic solid symmetry groups. Using a group-theoretic reinterpretation of Rotary Position Embeddings, it achieves equivariance to translations and discrete rotations without changing the Transformer architecture or cost. Experiments across vision, 3D, and molecular tasks show that it matches or exceeds state-of-the-art models while preserving efficiency.

### Strengths
By defining attention relative to Platonic solid symmetry groups and reinterpreting Rotary Position Embeddings as dynamic group convolutions, it unites group theory with Transformer attention in a principled way. The work is technically strong, clearly presented, and demonstrates broad empirical effectiveness across image, 3D, and molecular domains.

### Weaknesses
1. While the Platonic Transformer effectively maintains architectural flexibility, its design introduces a nontrivial trade-off between group size and model expressivity. To control computational overhead, the authors fix the total feature dimension and proportionally reduce the channel size per group element as $|G|$ increases. Although this strategy preserves overall parameter count and computational cost, it implicitly limits the representational capacity of each frame, potentially weakening the model’s ability to capture rich geometric variations when larger symmetry groups are used. Consequently, there exists a tension between achieving stronger equivariance (by enlarging $|G|$) and maintaining sufficient feature expressivity within each frame. A more thorough analysis or ablation of this trade-off, quantifying how equivariance benefits degrade or saturate as channel capacity per group element decreases, would strengthen the paper’s claims on scalability and generalization.

2. Comparison with stronger baselines: While results are competitive, the paper omits direct comparisons with recent high-performing equivariant Transformers such as SE(3)-Transformer (Fuchs et al., 2020), Euclidean Fast Attention (Frank et al., 2024), and Geometric Algebra Transformers (Brehmer et al., 2023). Including these would contextualize the claimed efficiency-equivariance trade-off more rigorously.

### Questions
1. Appendix L suggests potential computational gains from Fourier-domain implementations. Could the authors provide more concrete benchmarks or scaling results demonstrating the actual speedup at different model sizes? How does this compare to standard Transformer throughput?

2. Equivariant attention interpretability: Can the authors provide qualitative analyses or visualizations of the learned attention patterns across Platonic frames? Such results could clarify whether the model learns distinct geometric filters or redundant orientations.

### Soundness
3

### Presentation
3

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
The paper asks how to endow Transformers with strong geometric inductive bias—translation and (discrete) rotation/reflection equivariance—without the heavy machinery and overhead of conventional equivariant networks. The authors propose the Platonic Transformer, which “lifts” features to a group axis indexed by elements of a discrete subgroup $G \in O(3)$, then shares weights across these frames while keeping the rest of the Transformer (including RoPE) unchanged. Attention is computed per-frame and then aggregated, giving an equivariant architecture with the same computation graph as a standard Transformer.

### Strengths
Originality: 
1.  “lift-and-share” design: introducing a group axis and letting ordinary RoPE-attention operate unchanged per reference frame, with equivariant weight sharing across frames.
2. Insightful dynamic-convolution perspective that clarifies RoPE’s inductive bias; pragmatic use of finite Platonic groups offers a tractable middle ground between full continuous equivariance and invariance.
Quality:
1.  Readable narrative from lifting to equivariant linears to attention; figures help (e.g., pipeline diagram and scaling plots).
2. Multi-domain evaluation (images, point clouds, molecules) supports general usefulness of the bias; explicit invariant vs. equivariant attention comparison.
Significance:
1. This method serves as a near drop-in replacement, preserving the original computation graph and standard modules. Furthermore, its linear-time complexity, achieved via a Fourier-domain implementation, ensures excellent scalability.

### Weaknesses
Limited Fundamental Novelty: While the combination of ideas is novel, the fundamental building blocks are well-known. The use of group convolutions to ensure equivariance in linear layers is a cornerstone of G-CNNs (Cohen & Welling, 2016). The concept of lifting features to a group and operating on them is also standard in this field. The attention mechanism itself is the standard RoPE-attention, but applied in parallel. The method can be viewed as an instance of a Message Passing Neural Network (MPNN) where messages are computed via attention over multiple "views" (the reference frames), and updates are performed by group-convolved MLPs. The contribution is thus more of a sophisticated and effective architectural design rather than the introduction of a fundamentally new principle of equivariance. This is not a fatal flaw, as the design is very clever, but it positions the work as an incremental (though important) step forward.

Analysis of Design Choices: The paper introduces several crucial design choices but could benefit from a more in-depth analysis of their impact.
1. The choice to fix key vectors ($k_j$ =1) in the linear attention variant is justified by observed training instability on molecular datasets. The hypothesis that this disentangles geometry and signal is interesting but remains a hypothesis. A more rigorous investigation into this instability and potential alternatives (e.g., regularization, different initializations for the key network) would strengthen this design claim.
The comparison between equivariant attention (Eq. 11) and invariant attention (Eq. 12) is only done theoretically. An empirical ablation study comparing these two would provide concrete evidence for the benefit of the more expressive, orientation-dependent attention patterns.

Missing related work on frame methods：
1. A new perspective on building efficient and expressive 3D equivariant graph neural networks，  Neurips 2024
2. AlphaNet: Scaling Up Local Frame-based Atomistic Foundation Model，  Npj CM, 2025

### Questions
\textbf{Heads vs. Group Size:} The framework maps group elements to attention heads. What is the interplay between the size of the group $|G|$ and the number of heads per group element `nhead`? For a fixed total number of heads ($|G| \times \text{nhead}$), have you explored the trade-off between using a larger, more expressive group (e.g., Octahedral, $|G|=24$) with `nhead=1` versus a smaller group (e.g., Tetrahedral, $|G|=12$) with `nhead=2`? This could shed light on whether it's more beneficial to have more geometric frames or more feature diversity per frame.

 \textbf{Stability of Learned Keys:} Could you please elaborate on the instability observed when using learned keys in the linear convolutional variant on QM9/OMol25? Did the training loss diverge, or did it just converge to a poor result? Is it possible that this instability is an optimization artifact that could be mitigated with, for example, a different learning rate, initialization, or a regularizer on the key-producing network, rather than a fundamental issue requiring the removal of learned keys?

 \textbf{Practical Computational Cost:} Table 4 shows impressive inference times compared to other geometric networks. However, the claim is that the cost is identical to a standard Transformer. Group convolutions (Eq. 8), when implemented in the spatial domain, may have different memory access patterns than a dense matrix multiplication. Could you provide a more direct wall-clock time comparison between a single layer of your spatial implementation and a standard \texttt{TransformerEncoderLayer} with an equivalent total feature dimension (i.e., \texttt{d\_model = |G| * d\_hidden}) and number of points, to precisely quantify any practical overhead?

\textbf{Invariant Attention Ablation:} The discussion in Section 4.2 contrasting equivariant and invariant attention scores is very clear. Given that implementing the invariant score via sum-pooling seems straightforward, was this variant tested? If so, how did it perform? If not, what is your intuition on how much performance would be lost by sacrificing orientation-dependent attention patterns?

### Soundness
3

### Presentation
3

### Contribution
2
