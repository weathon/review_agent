# Unified Cross-Scale 3D Generation and Understanding via Autoregressive Modeling

- Avg Score: 7.00
- Decision: Reject
- Scores: 10, 6, 8, 4

## Abstract
3D structure modeling is essential across scales, enabling applications from fluid simulation and 3D reconstruction to protein folding and molecular docking. Yet, despite shared 3D spatial patterns, current approaches remain fragmented, with models narrowly specialized for specific domains and unable to generalize across tasks or scales.
We propose Uni-3DAR, a unified autoregressive framework for cross-scale 3D generation and understanding. At its core is a coarse-to-fine tokenizer based on octree data structures, which compresses diverse 3D structures into compact 1D token sequences. We further propose a two-level subtree compression strategy, which reduces the octree token sequence by up to 8x.
To address the challenge of dynamically varying token positions introduced by compression, we introduce a masked next-token prediction strategy that ensures accurate positional modeling, significantly boosting model performance.
Extensive experiments across multiple 3D generation and understanding tasks, including  small molecules, proteins, polymers, crystals, and macroscopic 3D objects, validate its effectiveness and versatility. Notably, Uni-3DAR surpasses previous state-of-the-art diffusion models by a substantial margin, achieving up to 256\% relative improvement while delivering inference speeds up to 21.8x faster.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper seeks to tackle cross-scale generation and understanding. They propose an octree-based coarse-to-fine tokenizer with subtree compression. They also propose Masked Next-Token Prediction (MNTP), a simple autoregressive method for these tokens whereby they duplicate each token and have the first token contain positional information while masking out the content and the second token have the content. They evaluate their method on a wide suite of tasks such as single-frame generation, multi-frame generation, token-level understanding, and structure-level understnanding, and across a wide range of domains, achieving impressive results.

### Strengths
1. The authors bring together a wide range of structure modeling tasks under a unifying framework
2. They achieve substantial improvements over baselines, especially in crystals & docking, which are challenging tasks.
3. They achieve a 21.8x speedup over diffusion models, which have been the dominant architecture for 3D structure generation tasks. 
4. The framework is versatile as it supports both generation and understanding (token‐level and global).

### Weaknesses
- The approach leads to pretty long sequences, though the compression scheme reduces it by up to 8x.

### Questions
- Any comment on the ability to scale to very large structures/long sequences?
- Are there ways to incorporate SE(3) invariance/equivariance into the tokenization?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a single framework with an autoregressive Transformer that can be applied to different 3D structures at different scales (e.g., molecules, crystals, proteins, and macroscopic shapes). The main idea is to build an octree data structure by dividing the space of the input data into different sub-structures, then apply a tokenizer that encodes each sub-structure and concatenate the outputs to build 1D token sequence. The 1D tokens are then modeled by an autoregressive transformer trained with masked next token prediction.  The authors show results on different tasks including: small molecular generation, crystal generation and structure predictions, and macroscopic 3D object generation.

### Strengths
- The main strength of this paper is that it shows a single model and approach can be applied across different 3D inputs and tasks. 

- The paper shows strong evaluations and performance on different 3D tasks: 3D molecular generation (on QM9 and Geom-drug datasets), crystal generation and structure prediction (on Carbon-24, MP-20, and MPTS-52 datasets), and macroscopic 3D object generation (on samples from ShapeNet dataset),  protein pocket predictions and molecular docking.

- The authors also evaluate the inference speed of their model against the Geometric Latent Diffusion model (GEOLDM) on QM9 dataset and show that it can be up to 21% faster.

### Weaknesses
- As mentioned by the authors, it will be interesting to see how their proposed approach can applied to do a single foundation model and train jointly on these domains.  

- Another point is that the proposed method uses a general Transformer model (lack of SE(3) equivariance by design). So it will be beneficial to see some discussion on this and how for e.g., rotating input structures might affect the generalization of the model.

### Questions
Is there any ablation or intuition on how the performance and efficiency change as you vary the octree depth and finest-cell size?

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
Uni-3DAR is an autoregressive framework for cross-scale 3D generation and understanding. The authors present a sequential representation of 3D structures based on octree compression of 3D grids and autoregressive learning techniques. The novel tokenization scheme converts 3D grid structures to 1D tokenization with a hierarchical octree-based 3D decomposition. The encoding scheme is further compressed by a 2-level subtree compression to reduce sequence length.  A fine-grained structural tokenization scheme is added to fine-grained "patches" to encode atomic types and precise coordinates. The authors also develop masked next-token prediction, a serialization scheme to enable dynamic and sparse generation and inference of these "3D patches".  The authors validate this unified framework across a wide array of benchmarks, including small molecule generation, crystal structure prediction, macroscopic 3D object generation, protein pocket prediction, and molecular docking.

### Strengths
1. The authors propose an effective method for 1-dimensional representation of 3D data to potentially represent systems at many different scales and perform generation and prediction tasks 
2. The sparse generation techniques provide a very good solution to the sparsity in the 3D spatial representation problem
3. The evaluation suite is impressive and varied. Uni-3DAR is used for multiple tasks, with strong results in nearly all categories, ranging from small molecule generation, crystal structure prediction, protein pocket prediction, and molecular docking. Such strong results signal the proposed representation can be a strong candidate for unified modeling and valuable for the community as a whole.

### Weaknesses
1. At the core, the serialization scheme converts 3D information into a sequence. This begs whether a simpler canonicalized point-cloud serialization method could effectively produce similar results to the octree-based representation. Ordering ambiguities are mentioned as a negative for point-based sequences (line 1045). However, that is easily solved by selecting a canonical ordering (like sorting by each dimension). 
2. The serialization method results in a lack of inductive bias, such as rotation and translation symmetries, which can be crucial in many scenarios, especially in multi-frame generation scenarios where errors may propagate over time. 
3. While the octree-based compression is lossless, the 3D voxel grid is not, and is only accurate up to the discretization resolution (line 1080). Furthermore, it is not clear how the decoding of continuous positions is performed during inference in the main paper. Are additional prediction heads used for the discretized coordinates?

### Questions
1. Does a new model need to be trained when the depth is changed to accommodate differently scaled systems? In other words, when the L changes between small molecules and large proteins, can the same model be used?
2. Not entirely sure what the inset in Fig. 2, " In this example, in-cell positions are discretized into $[0,50)$, with a default value of 25." 
3. Is the VQ-VAE-based fine-grained tokenization needed for macroscopic objects? Why not have a single fine-grained tokenization strategy?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Uni-3DAR, an autoregressive framework for 3D generation and understanding that claims to unify cross-scale modeling (from molecules to macroscopic objects). Its core is an octree-based tokenizer that converts sparse 3D structures into compressed token sequences. The method reports state-of-the-art results across numerous benchmarks and a significant inference speed-up (up to 21.8x) over diffusion models.

### Strengths
1. Good Empirical Results: The framework achieves SOTA or highly competitive performance across a very broad and diverse set of tasks, including molecular generation, crystal structure prediction, 3D object generation, and molecular docking.

2. High Stated Efficiency: The paper reports a massive inference speed-up (7.5x-21.8x) compared to diffusion-based models on the QM9 generation task, which is a significant practical advantage.

3. Versatile Architecture: The framework demonstrates impressive architectural versatility, showing how a single autoregressive paradigm can be adapted to handle both generation and understanding tasks at both microscopic and macroscopic scales.

### Weaknesses
1. "Unified Cross-Scale" Claim is Misleading: The central claim of unified cross-scale modeling is questionable. Models were trained independently for each task and dataset, meaning no cross-scale information was actually shared. The "unification" is purely architectural, not a single trained model, which conflicts with the impression given by the title.

2. Requires Domain-Specific Tuning: The framework is not "plug-and-play." It requires significant domain-specific adaptations, such as different "3D patch" definitions (single atom vs. VQ-VAE patch) and octree depths (L) for different data types, undermining the "unified" concept.

3. Limited Algorithmic Novelty: The paper appears to be more of a strong systems-level integration of existing components (octrees, AR models) than a fundamental algorithmic innovation. The theoretical advantage of the octree tokenization over using direct point cloud representations is not clearly established, beyond adding hierarchical information.

### Questions
1. Reproducibility: The lack of source code makes it impossible to verify the model's behavior and the paper's claims.

2. Constraint Enforcement: How does the autoregressive model enforce hard conditional constraints during generation? For example, in the CSP task, how is the model guaranteed to produce a crystal with the exact atom types and counts specified in the condition?

3. Unfair Efficiency Comparison: Was any post-processing or rejection sampling used to filter the generated outputs to meet conditional constraints (e.g., in CSP or molecule generation)? If so, the reported inference times are misleading and not a fair comparison to one-shot diffusion models, as the true cost (generation + filtering) could be substantially higher. I strongly encourage the authors to make their source code available.

4. Generality of Speed-up: The reported speed-up is for a single task (unconditional QM9 generation). What is the inference speed comparison for more complex conditional generation tasks like CSP, where filtering might be necessary?

### Soundness
2

### Presentation
4

### Contribution
2
