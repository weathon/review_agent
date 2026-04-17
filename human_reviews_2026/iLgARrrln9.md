# Implicit Reconstruct Spatiotemporal Super-Resolution Microscopy in Arbitrary Dimension

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
High-resolution 4D fluorescence microscopy imaging, essential for deciphering dynamic biological processes, is typically challenged by insufficient spatiotemporal resolutions, including restricted t-axis sampling density to prevent photobleaching and issues with anisotropic resolution.  To address these challenges, we propose an implicit neural representation-based arbitrary scale super-resolution framework, termed SpatimeINR, which leverages spatiotemporal latent representation in conjunction with a multilayer perceptron for 4D rendering, while incorporating cycle-consistency loss to ensure fidelity with the original data. Extensive experiments on lung cancer cell and C.elegans cell membrane fluorescence datasets demonstrate that our approach can accurately reconstruct the nonlinear dynamic motion of biological samples along the time axis and significantly outperforms state-of-the-art methods in both temporal and spatial (4D) super-resolution tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SpatimeINR, a novel framework designed to address the challenges of low spatiotemporal resolution, phototoxicity, and anisotropy in 4D fluorescence microscopy. The method uses an implicit neural representation (INR) to achieve arbitrary-scale super-resolution for 4D data (3D space + time). The motivation is clear, and the experimental results are solid. However, the method has not introduced domain-specific priors from biological processing.

### Strengths
(1) The motivation is clear and makes sense.

(2). The paper is well-written and easy to follow.

(3) The experimental results effectively validate the effectiveness of proposed method.

### Weaknesses
(1) The paper focuses on "biological processing." However, the 4D INR modeling of cellular behavior does not appear to incorporate specific domain knowledge from biology. in fact, the proposed method is more like a general-purpose 4D SR approach.

(2) Similar to LIIF, the trained network is capable of outputting SR results directly without fitting each sample individually. If the extracted input features from the encoder can be reused when querying different upsampling scales for the same input? Authors should provide more detailed analysis regarding inference time and the reusability of features for arbitrary-scale super-resolution.

(3) Missing discussion with recent 2D ASR methods using explicit Gaussians.

[1] Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution

[2] Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling

### Questions
See weakness.

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
4

### Summary
This paper introduces SpatimeINR, a model designed for 4D super-resolution of microscopy images. Unlike traditional methods that upscale images by fixed factors, SpatimeINR constructs an implicit neural representation (INR) over the spatiotemporal domain 
(x,y,z,t). This enables continuous sampling at arbitrary spatial and temporal resolutions. The model consists of two core components: a spatiotemporal feature extractor and an INR renderer. It is trained using a reconstruction loss, with an additional cycle-consistency loss to enhance perceptual quality. Experimental results show that SpatimeINR achieves state-of-the-art (SOTA) performance on two microscopy benchmarks.

### Strengths
- The paper is clearly written and easy to follow.

- The proposed method enables arbitrary-scale 4D super-resolution.

- The model demonstrates SOTA performance on evaluated benchmarks.

### Weaknesses
- The method’s novelty is somewhat limited. The key components—INR-based super-resolution and cycle-consistency loss—are already well-studied, and the contribution mainly lies in combining established techniques and applying them to the microscopy domain.

- The baseline comparison appears outdated; for example, SVIN may not be the strongest reference point. It would be more convincing to compare against recent natural image super-resolution methods retrained on microscopy data (e.g., [1]).

- The experiments are performed on benchmarks different from those used by prior works, making it difficult to fairly assess whether SpatimeINR truly outperforms existing methods.

[1] Ekanayake, Mevan, et al. "SeCo-INR: Semantically Conditioned Implicit  Neural Representations for Improved Medical Image Super-Resolution." *2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*. IEEE, 2025.

### Questions
- In ablation, instead of testing a combination of three different components, can you decouple it a bit to show how each component contributes to performance individually ( like INR+some components).

- Can you also your model’s results on datasets such as Cardiac, which are used by baselines like UVI-Net, to better demonstrate generalization and fairness in comparison.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes SpatimeINR, an implicit neural representation framework for arbitrary-scale spatiotemporal super-resolution of 4D fluorescence microscopy data. The method integrates spatiotemporal latent representations, a conditional MLP, and a 4D volumetric rendering module with cycle-consistency loss. Extensive experiments on lung cancer cell (A549) and C. elegans cell membrane fluorescence datasets demonstrate that SpatimeINR outperforms existing traditional and deep learning methods in reconstructing complex 4D dynamics, both quantitatively and qualitatively.

### Strengths
1. The work successfully extends implicit neural representations (INR) to full 4D (3D space + time) biological imaging, enabling continuous and arbitrary-scale reconstruction. This addresses a significant gap in prior work, which often focused on 2D/3D or fixed-scale super-resolution.
2. The method introduces a principled approach to encode per-voxel temporal trajectories as latent codes, effectively capturing dynamic information across both spatial and temporal dimensions (as illustrated in Figure 3).
3. The authors conduct thorough experiments on two challenging real-world datasets, covering both spatial and temporal super-resolution tasks. Ablation studies (Section 5.3) convincingly demonstrate the importance of each component: spatiotemporal encoding, 4D rendering, and cycle-consistency loss.

### Weaknesses
1. The use of variable x in Equation (1) is ambiguous—it is used to denote both spatial coordinates and full 4D spatiotemporal coordinates. This should be clarified to avoid confusion.
2. The latent code extraction process (Section 3.2) lacks a clear description of how queries between low-resolution grid points are handled (e.g., nearest-neighbor vs. interpolation). The downsampling operator D(⋅) used in the cycle-consistency loss (Equation 11) is not defined.
3. Several recent and highly relevant INR-based methods for 4D medical imaging are not discussed or compared, including:
Saitta et al. (2024): INR for 4D flow MRI super-resolution.
Xu et al. (2024): Self-supervised INR for 3D fluorescence microscopy.
The omission of these works weakens the positioning of SpatimeINR within the current research landscape.
4. The description of the proposed framework is not comprehensive enough. Critical implementation details for key components—such as the encoder architecture, the latent code sampling strategy, and the feature fusion mechanism in the MLP—are omitted, hindering a complete understanding and reproducibility.
5. Lack inference speed and memory usage.

### Questions
The cycle-consistency loss formulation references a downsampling operator $\mathcal{D}(\cdot)$, but its definition is not made explicit. Is this operator simply a uniform average, or does it model specific physical/optical effects? Please clarify its construction and impact on reconstruction fidelity.
Can the authors provide timing benchmarks and memory use for both training and inference across their datasets, and comment on applicability to even larger 4D datasets (for example, developmental time-lapse volumes or multi-field MRI)?

### Soundness
2

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
2

### Summary
The paper introduces an implicit neural representation framework for arbitrary-scale 4D spatiotemporal super-resolution in fluorescence microscopy. By modeling volumetric time-varying data as a continuous implicit function and combining local sampling, a conditional MLP, and cycle-consistency constraints, the method reconstructs high-resolution 4D sequences from low-resolution inputs without relying solely on boundary frames. Experiments on lung cancer cell and *C. elegans* microscopy datasets show that SpatimeINR outperforms both traditional interpolation and deep learning baselines, preserving fine structures and improving downstream segmentation performance, demonstrating strong capability in handling complex, non-periodic cellular dynamics.

### Strengths
- The paper makes a meaningful contribution by extending arbitrary-scale implicit neural representations to continuous 4D spatiotemporal microscopy, a setting rarely addressed in prior SR research. The combination of spatiotemporal latent codes, local INR rendering, and cycle-consistent degradation represents a thoughtful hybrid of existing ideas adapted to biological imaging constraints.
- The technical components are well-motivated and carefully implemented: latent trajectory encoding per voxel, mixed uniform and importance sampling for 4D coordinates, and conditional MLP rendering. The experiments are rigorous, covering multiple scaling factors, two biological datasets, and downstream segmentation with MedSAM. Ablations clearly verify the necessity of each module.
- The paper is clearly written with intuitive figures and consistent mathematical notation. The model architecture, training losses, and data preprocessing steps are explained in detail, enabling reproducibility.

### Weaknesses
Need more ablation studies to quantify the sensitivity of the model to architectural and data-related design choices, including:

- the effect of latent dimensionality on spatiotemporal fidelity,

- the role of temporal context length during encoding,

- robustness to varying noise levels or photobleaching patterns.

### Questions
- How does SpatimeINR fundamentally differ from dynamic NeRF / time-conditioned neural fields (e.g., D-NeRF, Nerfies, HyperNeRF) or grid-based spatiotemporal implicit models? Clarifying this distinction and ideally including at least one representative comparison (or explaining why not) would help position the contribution

- Why adopt a voxel-wise latent trajectory encoding instead of alternatives like continuous temporal embeddings or learned spatiotemporal hash grids? Empirical or conceptual justification would strengthen the design motivation.

- Could the authors provide qualitative examples where SpatimeINR fails or struggles (e.g., extremely rapid motion, highly anisotropic data, low SNR situations)?

### Soundness
3

### Presentation
3

### Contribution
3
