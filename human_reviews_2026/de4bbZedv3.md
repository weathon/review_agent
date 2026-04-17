# InertialAR: Autoregressive 3D Molecule Generation with Inertial Frames

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Transformer-based autoregressive models have emerged as a unifying paradigm across modalities such as text and images, but their extension to 3D molecule generation remains underexplored. The gap stems from two fundamental challenges: (1) tokenizing molecules into a canonical 1D sequence of tokens that is invariant to both SE(3) transformations and atom index permutations, and (2) designing an architecture capable of modeling hybrid atom-based tokens that couple discrete atom types with continuous 3D coordinates. To address these challenges, we introduce InertialAR. InertialAR devises a canonical tokenization that aligns molecules to their inertial frames and reorders atoms to ensure SE(3) and permutation invariance. Moreover, InertialAR equips the attention mechanism with geometric awareness via geometric rotary positional encoding (GeoRoPE). In addition, it utilizes a hierarchical autoregressive paradigm to predict the next atom-based token, predicting the atom type first and then its 3D coordinates via Diffusion loss. Experimentally, InertialAR achieves state-of-the-art performance on 7 of the 10 evaluation metrics for unconditional molecule generation across QM9, Geom-Drug, and B3LYP. Moreover, it significantly outperforms strong baselines in controllable generation for targeted chemical functionality, attaining state-of- the-art results across all 5 metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes InertialAR, a transformer-based autoregressive model for 3D molecule generation. The key innovations are a canonicalization of the atom positions (via inertial frames) and atom indices (via RDKit), in order to generate a 1D sequence of tokens, using a novel 3D geometry-aware positional encoding. Experiments show that InertialAR performs well on QM9, GEOM-DRUGS and the large B3LYP dataset for unconditional generation, and can further be conditioned on molecule class reasonably well.

### Strengths
The appeal of canonicalization of 3D geometry in order to apply well-known techniques from 1D sequence modelling makes sense to me. The new GeoRoPE mechanism builds upon RoPE-3D by adding the pairwise distance using the Nyström low-rank approximation method, and should be useful in other 3D modelling contexts.

### Weaknesses
The canonicalization of the atom indices is unclear and needs to be explained better. How are the identifiers computed? To the best of my knowledge, there is no ‘smooth’ way to canonicalize atom indices. Some additional experiments with the canonicalization would be helpful to understand when it breaks; for example, do you find that similar molecules have very different canonicalization orders?

### Questions
* Is your denoising network also a transformer?
* How does your model decide when to stop generation?
* Does your model support beginning from a random molecule fragment, since the canonicalization of atom indices in the fragment might be very different from the full molecule?
* Do you have any experiments where you prompt the model to generate longer molecules than seen during training?
* Are the training splits the same for all methods on QM9?
* How important is the GeoRoPE mechanism? Do you have experiments with simple RoPE-3D for comparison?
* Can you explain the canonicalization procedures (both positions and indices) for a symmetric planar molecule such as benzene?

### Soundness
3

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
4

### Summary
InertialAR is a transformer-based, autoregressive framework for 3D molecule generation by (i) canonical tokenization: aligning molecules to an inertial frame and applying a deterministic atom reordering for SE(3) and permutation invariance; (ii) GeoRoPE: a geometry-aware attention that blends rotary embeddings of relative orientation with pairwise distance features; it achieves strong or SOTA stability/validity on QM9 and on the large B3LYP-1M set.

### Strengths
1. The two-step canonicalization: aligning each molecule to an inertial frame with a deterministic sign convention, then applying a deterministic atom reordering removes SE(3) and permutation ambiguities without specialized equivariant networks.
2. It achieves SOTA or near-SOTA validity/stability on QM9 and GEOM-DRUG and shows big gains on the large B3LYP benchmark

### Weaknesses
1. To pick axis signs, the authors choose a “fourth node” (the atom farthest from the origin) and require it to lie in the first quadrant of the xy plane; this rule unambiguously fixes signs but could flip when the farthest atom changes under small perturbations, i.e., the frame is not continuous [2]. Same situations will happen when principal moments tie, small geometric changes can swap frames and thus token order
2. The authors does not situate this design within prior work on PCA-based pose selection, graph canonical labeling or alternative symmetry-handling strategies [1][2][3][4].
3. **Appendix D–F are incomplete**. Appendix E still contains a placeholder like “Will extend this to a more formal way.”, and Appendix F includes unresolved “??” figure references; Appendix D also contains informal notes in the text. Please finalize these sections by replacing placeholders with complete derivations, figures, and cross-references, or move unfinished material to a clearly labeled supplemental.

[1] Frame Averaging for Invariant and Equivariant Network Design. Omri Puny, Matan Atzmon, Heli Ben-Hamu, Ishan Misra, Aditya Grover, Edward J. Smith, Yaron Lipman.

[2] Equivariant Frames and the Impossibility of Continuous Canonicalization. Nadav Dym, Hannah Lawrence, Jonathan W. Siegel.

[3] Equivariance via Minimal Frame Averaging for More Symmetries and Efficiency. Yuchao Lin, Jacob Helwig, Shurui Gui, Shuiwang Ji.

[4] A Canonicalization Perspective on Invariant and Equivariant Learning. George Ma, Yifei Wang, Derek Lim, Stefanie Jegelka, Yisen Wang.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

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
The paper introduces InertialAR, an autoregressive (AR) model for 3D molecule generation. It addresses two key challenges:
1. Tokenization: It creates a canonical 1D sequence of atoms by aligning the molecule to its inertial frame (for $SE(3)$ invariance) and then applying a canonical reordering (for permutation invariance).
2. Hybrid Prediction: It models the hybrid (discrete type, continuous 3D coordinate) token using a hierarchical AR paradigm.vThis involves a new Geometric Rotary Positional Encoding (GeoROPE) to make the attention mechanism geometry-aware and predicts coordinates using a Diffusion Loss, which was found to be superior to direct regression.

The model achieves state-of-the-art performance on unconditional generation (QM9, Geom-Drug, B3LYP) and significantly outperforms baselines in controllable generation of specific functional groups.

### Strengths
**Technical Quality**: The GeoROPE architecture is a creative and effective way to inject geometric information into the attention mechanism, combining relative positions (RoPE-3D) and pairwise distances (Nyström) into a single attention score.

**Significance & Performance**: The model demonstrates exceptional performance, not just on standard benchmarks but also on a large-scale dataset (B3LYP) and a highly practical controllable generation task, showing SOTA results across all metrics for the latter.

### Weaknesses
**Originality**: The use of a molecule's inertial frame as a canonical reference is a common solution to the $SE(3)$ invariance problem.

**Robustness Not Addressed**: The paper does not discuss the stability of the inertial frame canonicalization. For symmetric molecules (degenerate eigenvalues) or flexible molecules (where small conformational changes could flip the axes), the token sequence could become unstable, which is a significant problem for an AR model.

**Missing Ablation Studies**: The paper proposes several new components (Inertial Frame, RoPE-3D, Nyström, Diffusion Loss) but lacks ablations to test their individual contributions. 

**Reproducibility**: Key implementation details are missing, most notably how the Nyström approximation anchor points ($m$ points) are selected, which is critical for implementing GeoROPE.

### Questions
1. How do you ensure the inertial frame tokenization is robust? What happens if a small conformational change or molecular symmetry causes the principal axes to flip, resulting in a different canonical sequence?
2. Could you please quantify the "poor performance" of using a simple L2 loss for coordinates? What were the key metrics when the Diffusion Loss was replaced with direct regression? 
3. What are the implementation details for the Nyström anchor points? How many are used ($m$), and how are they selected (e.g., fixed, per-molecule, or sampled)?

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
4

### Summary
This paper proposes a novel autoregressive generation framework InertinalAR for 3D molecules. Multiple contributions are presented including canonical atom ordering, coordinate mapping with inertial frames, geometric rotary positional encoding in conditional feature extration network. Experiments on unconditional and class-conditional 3D molecule generation show that the performance of InertinalAR is promising.

### Strengths
- This paper proposes a novel 3D molecule generation framework. Some contributions like atom canonical ordering and keep SE(3) invariance by inertial frame based coordinate projection is very useful.
- Generally the experimental results are good and promising.
- The writing of this paper is good and clear.

### Weaknesses
- Some details need clarification. What is the ordering of eigenvalues in line 186? How ordering by the refined identifiers is done in line 213?
- A major novelty contribution of this paper is the use of geometric rotary positional encoding (GeoRoPE) together with a transformer architecture as the backbone network. However, no ablation study of this architecture is conducted so it is unclear what is the impact of this architecture on performance. Could we just use a 3D graph neural network or graph transformer model and keep everything else the same? More ablation studies are needed to make this work more solid.

### Questions
No additional questions.

### Soundness
2

### Presentation
3

### Contribution
3
