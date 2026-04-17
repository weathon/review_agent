# VoMP: Predicting Volumetric Mechanical Property Fields

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Physical simulation relies on spatially-varying mechanical properties, typically laboriously hand-crafted. We present the first feed-forward model to predict fine-grained mechanical properties, Young’s modulus ($E$), Poisson’s ratio ($\nu$), and density ($\rho$), throughout *the volume* of 3D objects. Our model supports any 3D representation that can be rendered and voxelized, including Signed Distance Fields (SDFs), Gaussian Splats and Neural Radiance Fields (NeRFs). To achieve this, we aggregate per-voxel multi-view features for any input, which are passed to our trained Geometry Transformer to predict per-voxel material latent codes. These latents reside on the trained manifold of physically plausible materials, which we train on a real-world dataset, guaranteeing the validity of decoded per-voxel materials. To obtain object-level training data, we propose an annotation pipeline combining knowledge from segmented 3D datasets, material databases, and a vision-language model. Experiments show that VoMP estimates accurate volumetric properties and can convert 3D objects into simulation-ready assets, resulting in realistic deformable simulations and far outperforming prior art.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes VoMP, an end-to-end framework that predicts voxel-wise physical property fields from a voxelized 3D representation aggregated from unprojected 2D features. The method employs a MatVAE trained on real material databases to ensure physically plausible outputs, a Geometry Transformer that transforms the input features to predict per-voxel latent codes in the material space, and uses a decoder to reconstruct continuous volumetric material fields from these latents. To train this model, the authors construct a large-scale GVM dataset with voxel-level annotations generated via VLM guidance. Experiments across various 3D inputs demonstrate generalization and physically consistent simulation results.

### Strengths
- Proposes a framework for predicting voxel-level material properties from arbitrary 3D representations, defining a novel and important task in 3D physical modeling.
- Introduces a MatVAE that maps material parameters into a physically valid latent space, effectively smoothing predictions and constraining outputs to realistic ranges.
- Employs a VLM to automatically construct large-scale annotated datasets, with the resulting GVM dataset being valuable for future research and other methods.
- Demonstrates higher overall quality than prior methods, with notable improvements in physical consistency and simulation fidelity.

### Weaknesses
- Although interior structure plays a crucial role in physical simulation, the current voxelization approach (projecting 2D multi-view features) is not well-suited for capturing internal geometry variations, leading to limited understanding of interior geometry.
- The voxelization process may cause significant information loss in high-frequency or fine geometric regions, reducing the spatial precision of the predicted material fields.
- The transformer design uses only Swin-Attention without the hierarchical downsampling (and upsampling) scheme of a full Swin-Transformer, which restricts the receptive field and may limit the model’s ability to capture long-range structural dependencies. The VAE of TRELLIS is only for local geometry encoding and compression, and using that structure for global material property prediction is not suitable.
- The task and dataset inherently probably contain inconsistencies due to the semantic labeling process based on VLM and the task’s inherent ambiguity. A pure regression formulation may not be ideal for modeling such uncertainty, even with the help of a latent MatVAE. Would a generative or probabilistic approach better capture these ambiguities?

### Questions
A bit of concern around L244: Since transformers inherently support variable-length sequences, why is cropping necessary in your implementation? What specific limitation or efficiency concern motivates this design choice? The explanation that “dynamic resampling ensures the model is exposed to different parts of the asset over epochs and has a larger number of effective max voxels” seems counterintuitive. Random subsampling may lead to distribution shift during full-sequence inference, effectively making inference out-of-distribution compared to training. Moreover, if subsampling changes which spatial regions of the object are visible or centered, the geometric semantics of each cropped region could shift, potentially confusing the model. How do you ensure consistency of geometry–material correspondence under such random cropping or resampling?

### Soundness
3

### Presentation
3

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
This paper presents a novel feed-forward model (VoMP) to automatically predict the physical properties for 3D objects, which is an interesting and core problem for setting up the physical simulations requiring these properties. It proposes two important modules, MatVAE and Geometry Transformer, to map the interior voxel to mechanical values in a physically valid manner. The experiments demonstrate the effectiveness and speed of the proposed model.

### Strengths
1. The core design is novel and effective. The problem is decoupled into two parts, which guarantee valid physical latent space learning and fast plausible prediction. The ablation study confirms the superiority of the proposed architecture
2. The feedforward nature achieves inference in seconds compared to the previous optimization approach, making it a useful tool for scalable simulation pipeline setup
3. The paper includes a comprehensive set of experiments, including strong quantitative comparisons, ablation studies that justify the design choices.

### Weaknesses
1. One remaining concern is the reliance on VLM-generated ground-truth. An expert-annotated dataset is hard to get; hence, the usage of VLMs and external knowledge is an interesting point for data construction. But how to guarantee the correctness of the VLM predicted value is also crucial, especially for real future downstream task usage. 
2. The Geometry Transformer is trained using a fixed $64^3$ voxel grid, which is rather low resolution for complex 3D assets. This seems like a major bottleneck that prevents the model from capturing fine-grained material details.

### Questions
1. The Geometry Transformer's architecture uses $64^3$ voxel grid, which seems a limitation. What are the bottlenecks preventing scaling this transformer up to $128^3$ or even $256^3$? Would this influence fine-grained detail prediction? It's interesting to analyze whether the performance could also scale up with respect to the resolution.
2. A remaining concern is the material annotation accuracy. Although there is external knowledge to help with the annotation, how can to guarantee the correctness for training and evaluation? Is there any small expert-annotated dataset including these three values that can be used to serve as an independent test of physical accuracy? This problem seems implied in Figure 6, where in (b), VoMP outperforms all previous approaches significantly, while in (c) ABO-500 benchmark, the improvement is not obvious. How to demonstrate that the model is not mimicking the VLM-predicted behavior but truly learning from physics.
3. The model allows internal structure prediction. How to deal with the situations of occlusion?

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
4

### Summary
VoMP predicts spatially varying volumetric mechanical properties—Young’s modulus, Poisson’s ratio, and density—from 3D assets, taking any renderable and voxelizable representation as input and outputting per-voxel fields ready for accurate simulation. It aggregates multi-view DINOv2 features into a voxel grid and uses a TRELLIS Geometry Transformer to produce per-voxel material latents. These latents are decoded by a 2D MatVAE trained on real-world triplets, which enforces physically valid material values. A data pipeline combines part segmentations, material databases, textures, and a vision–language model to create large-scale volumetric supervision and a new benchmark. Experiments show lower errors than NeRF2Physics, PUGS, Phys4DGen, and Pixie, while running in a few seconds per object and enabling realistic deformable simulations. Overall, the work delivers a fast, representation-agnostic, feed-forward approach to simulation-ready material fields via multi-view voxel features and a physically grounded latent material prior.

### Strengths
- Representation-agnostic, feed-forward pipeline that predicts per-voxel mechanical properties in seconds, with outputs directly usable in simulators.
- Data contribution: a VLM-based pipeline that annotates ~1.6K part-segmented 3D shapes with **volumetric** materials; unlike PIXIE’s Pixelverse (surface-biased), this provides volumetric supervision.
- Extensive quantitative and qualitative results with detailed ablations, plus realistic FEM simulations.
- Clear writing and thorough explanations.

### Weaknesses
- Scalability is limited by the size and diversity of the training set (≈1.3K shapes) and the need for part-segmented annotations, making it harder to scale than methods that avoid volumetric labeling.
- Most experiments use author-curated data; because the test set follows a similar distribution, generalization to independent datasets (e.g., Objaverse) is uncertain.
- Resolution is bounded by fixed-grid voxelization, which restricts fine-detail fidelity.

### Questions
To what extent does the method capture discontinuities at part boundaries without “bleeding” material across seams, and is there any explicit penalty encouraging jumps where appropriate?

Does the model predict uncertainties per voxel (e.g., variance over E, ν, ρ)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes VoMP, a feed-forward system that predicts volumetric mechanical property fields including Young's modulus, Poisson's ratio and density- for 3D objects across multiple 3D representations (meshes, Gaussian splats, SDFs, NeRFs). The pipeline first voxelizes the object, lifts multi-view DINO image features into the volume, and feeds per-voxel features to a transformer trained to output material latent codes. MatVAE is a VAE trained on a database of real-world materials so that the decoding process guarantees materials lie on a learned manifold of physically plausible parameters. The authors construct a Geometry with Volumetric Materials (GVM) dataset via a VLM-assisted annotation pipeline that leverages part segmentations, PBR textures, and real-material ranges. VoMP generalizes across 3D formats and produces simulation-ready fields that drive high-fidelity FEM / simplicits simulations without tuning, outperforming recent baselines both in accuracy and speed.

### Strengths
1.  New Dataset: The GVM pipeline combines part segmentation + PBR textures + VLM prompts constrained by a curated materials range table to annotate 37M voxels, a scale jump over earlier works that only annotated sparse points. This will be useful to the community and encourage future research in this under-explored area.
2. Volumetric Representation: Unlike previous models that focus heavily on surface properties or struggle with interior prediction (e.g., NeRF2Physics and PUGS due to feature field limitations), VoMP is designed to predict materials throughout the object volume. The system uses a unified multi-view rendering and voxelization pipeline, successfully handling meshes, SDFs, NeRFs, and 3D Gaussian Splats.
3. MatVAE: The introduction of MatVAE- a VAE used to learn a low-dimensional latent space of valid real-world material triplets- is a significant contribution. This component ensures that the decoded per-voxel materials are physically plausible, even during interpolation or sampling, a guarantee largely absent in prior methods that may output simulator-specific or non-physical parameters. The paper explicitly shows that interpolation in the latent space yields valid materials, unlike naive linear interpolation in the material space.

### Weaknesses
I do not have a major concern, several minor points:
1. Reliance on Approximate Input and Voxelization Resolution: The final output resolution is limited by the fixed-grid voxelization. This can lead to oversmoothing in highly heterogeneous regions and approximation errors when mapping results back to highly detailed input geometry, especially for thin structures or internal complexities. Also the authors do not provide a comprehensive study on the impact of input & output resolution.
2. Assumption of Isotropic Materials: The annotation pipeline and the resulting model fundamentally assume that materials at the part level are isotropic (properties are uniform in all directions). This is a critical limitation for common composite or natural materials, such as wood or carbon fiber, whose physical behavior depends strongly on direction (anisotropy). The model cannot currently capture this crucial complexity.
3. While the VLM is carefully guided by MTD and visual cues, the core physical property assignment for 37M voxels relies on the VLM's inference capabilities. Failure cases in similar VLM-based approaches (like Phys4DGen's dependency on MLLM consensus) highlight the brittleness of this approach. The use of a VLM introduces a reliance on an external, possibly inaccurate, source of physical knowledge, rather than deriving parameters purely from visual features and learned physics principles. I would like to see more discussions with regard to this.

### Questions
1. Generalizability of Voxelization: The feature aggregation step requires solid voxelization, especially challenging for sparse representations like NeRFs and GSs which are fundamentally surface or density fields. Given the extensive training data is derived from high-quality segmented meshes (GVM), how reliable is the voxelization scheme for novel, noisy, or incomplete real-world captured NeRFs/GSs? Does the current method of voxelizing 3D Gaussian splats via rendering depth maps and carving space introduce artifacts or inconsistencies that affect the predicted internal properties in practice?

### Soundness
3

### Presentation
4

### Contribution
3
