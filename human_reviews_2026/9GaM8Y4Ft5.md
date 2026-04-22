# SeMa3D: Lifting Vision-Language Models for Unsupervised 3D Semantic Correspondence

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
We tackle unsupervised dense semantic correspondence for 3D shapes, focusing on severe \textbf{non-isometric} deformations and \textbf{inter-class} matching--a regime where conventional functional map pipelines fail due to ambiguous geometric cues.  We propose \textbf{\emph{SeMa3D}}, a framework that integrates semantic knowledge from vision-language foundation models to build robust vertex-level descriptors. Specifically, SeMa3D aggregates multi-view features from visual foundation models, with a novel \emph{colorization} strategy that mitigates semantic inconsistencies across renderings, and further enriches them with \emph{text embedding fields} to capture higher-level information. These descriptors are fused with geometric priors and aligned through a functional map formulation to ensure smooth, globally consistent correspondences. To achieve semantic matching, we introduce a \emph{region-aware contrastive loss} that leverages geodesic distances and zero-shot semantic part proposals (\eg, head, leg), injecting structural intent (\eg, ``head$\rightarrow$head'') into the mapping. Extensive experiments on challenging benchmarks show that \ourmethod outperforms existing methods in both extreme non-isometric and inter-class scenarios, achieving strong accuracy and generalization without relying on 3D labels or category-specific training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles dense 3D semantic correspondence under severe non-isometry and inter-class matching scenarios. The authors propose SeMa3D with the following components: 
1. synthesizes multi-view consistent and natural texture from generative models
2. lifts multi-view visual features from SD-DINO and fuses them with language embeddings from SigLIP to compose vertex features 
3. optimizes a functional-map correspondence, with a region-aware contrastive loss guided by part proposals from pre-trained model. 

On a wide range of inter-class, intra-class and non-isometric benchmarks, SeMa3D produces superior performance comparing to various baseline methods.

### Strengths
1. The overall paper is with clear problem focus and motivation: to addresses the 3D correspondence in inter-class and non-isometric cases, simple geometric cues alone will usually fail, thus integrating more semantic feature from a wide range of pre-trained models explicitly should always help.

2. The combination of view-consistent colorization, and the fusion of SD-DINO visual features with SigLIP text embeddings seems straightforward and works well.

3. The authors have conducted extensive experiments to compare the proposed method with a wide range of baselines, also did extensive ablation experiments to validate each proposed components.

### Weaknesses
The work relies heavily on directly using a combination of V(L)FMs and texturing model. While the performance gain from these components is naturally expected, the authors didn't fully analyze the potential failure modes from these components, e.g. bad textures, occlusion bias, prompt sensitivity. At least some qualitative results and analysis on inconsistent texture or noisy prompts are helpful.

Similarly, part segmentation outputs can be very noisy, while the loss uses margins to tolerate it, the paper lacks some qualitative robustness analysis vs. segmentation quality.

The adopted multi-stage pipeline may be heavy, current paper didn't give a thorough runtime and resource details, also it'd be good to compare this metric to other baselines.

### Questions
Mostly see the weaknesses above, some other questions:

1. What are the end-to-end runtimes per pair, and memory requirements? Any caching or down-stream speedups once features are precomputed? What' the cost for train and test, respectively.

2. How sensitive is the overall performance to the specific part list or to SigLIP’s language prompts across datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes SeMa3D for unsupervised dense semantic correspondence between 3D shapes, targeting non-isometric deformations and inter-class matching where functional-map pipelines struggle. The method lifts multi-view visual features from VFMs, enforces view-consistent colorization, augments descriptors with language embeddings for part names, and optimizes a functional-map objective coupled with a region-aware contrastive loss that encodes part-to-part relations. Experiments show lower geodesic error than prior art on inter-class SNIS and on non-isometric SMAL and TOPKIDS, while matching state of the art on near-isometric FAUST, SCAPE, and SHREC19.

### Strengths
- Originality: The paper lifts both visual and linguistic cues from VLMs to 3D surfaces and injects semantic priors through a region-aware contrastive loss, which is a clear step beyond purely geometric or visual-only descriptors.

- Quality: The method is carefully constructed with view-consistent colorization, multi-view back-projection, semantic region proposals, a functional-map objective, and an explicit semantic contrastive loss. Ablations isolate contributions of colorization, SigLip language embeddings, and the contrastive loss.

- Significance: SeMa3D substantially improves inter-class matching on SNIS and offers consistent gains on non-isometric datasets while retaining top performance on classical near-isometric benchmarks, suggesting practical benefits for cross-category and deformable scenarios.

### Weaknesses
- Performance relies on high-quality texture synthesis and zero-shot part segmentation. Failure modes from colorization artifacts or segmentation errors are only partially explored.

- The approach assumes a fixed part proposal set. It is unclear how sensitive results are to vocabulary choices, cross-category ambiguities, or missing parts.

### Questions
- How robust is SeMa3D to segmentation noise and to different part vocabularies. Please quantify sensitivity to incorrect or missing regions and report failure cases.

- Do results hold on real scanned meshes with holes and partiality, or in downstream tasks like cross-category part transfer or manipulation. Any small scale study would be helpful.

### Soundness
4

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
The paper proposes SeMa3D, an unsupervised framework for dense 3D semantic correspondence that tackles non-isometric and inter-class matching settings where classic functional-map pipelines struggle. The method synthesizes view-consistent textures for untextured meshes,  extracts multi-view semantic features with SD-DINO, and augments them with SigLIP text embeddings keyed to zero-shot part proposals, and couples these descriptors with a functional-map objective plus a region-aware contrastive loss to inject part-level priors. Across SNIS (inter-class), SMAL/TOPKIDS (non-isometric), and FAUST/SCAPE/SHREC19 (near-isometric), SeMa3D matches or outperforms recent baselines, with especially notable gains on inter-class SNIS. Ablations indicate that view-consistent colorization, adding language features, and the region-aware contrastive term each contribute measurable improvements.

### Strengths
1. Concatenating SD-DINO image features with SigLIP text embeddings tied to zero-shot part labels yields descriptors that disambiguate semantically similar but geometrically dissimilar regions—key for inter-class cases.
2. The dynamic-margin formulation encodes distances over a semantic-region graph and complements the functional-map objective, improving robustness to segmentation noise.

### Weaknesses
1. Success depends on a chain of external components—view-consistent texturing, multi-view rendering, zero-shot region proposals, SD-DINO, and SigLIP—so errors can cascade; real-world scans with incomplete geometry/texture or unusual categories might degrade performance. The paper could include an analysis of the error accumulations.
2. While SeMa3D shows improvements, part of the edge over DenseMatcher could stem from using SeMa3D’s own zero-shot region proposals to run that baseline.

### Questions
DenseMatcher projects multi-view 2D foundation features to meshes, refines with a 3D network, and solves correspondences via functional maps, evaluated on a colored-mesh dataset (DenseCorr3D) for robotic manipulation. What are the biggest differences between SeMa3D and DenseMatcher? What is the performance of SeMa3D on the DenseCorr3D benchmark?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes SeMa3D, a new framework for unsupervised dense semantic correspondence between 3D shapes, especially targeting non-isometric deformations and inter-class matching (e.g., human-to-horse). The core idea is to leverage vision-language models (VLMs) to extract semantic features from both visual and linguistic domains, and integrate them into a functional map framework to establish robust and semantically meaningful 3D correspondences without any manual annotations.

### Strengths
The method introduces Vision-Language Models for 3D Correspondence, where they use SD-DINO (visual) and SigLip (linguistic) models to extract semantic features from multi-view renderings. It combines visual and text embeddings to form rich, semantic-aware vertex descriptors.

The method applies SyncMVD for high-quality, cross-view consistent texture synthesis on raw 3D shapes, which ensures stable and consistent multi-view feature lifting.

The method achieves Zero-Shot Semantic Region Proposal by using SATR for zero-shot 3D part segmentation (e.g., head, leg, torso) without manual annotations, enabling semantic region-aware matching.

### Weaknesses
The method uses several pretrained models, which may have a strong dependence on their performance. Are there any correction schemes if the pretrained models are inaccurate? 

The method is designed for 3D meshes with consistent topology. It is limited to Mesh-Based Shapes and cannot directly apply to point clouds or unstructured 3D data.

The pipeline involves multiple stages: texture synthesis, multi-view rendering, VLM feature extraction, and functional map optimization. It may be slower than end-to-end or purely geometric methods.

There is no explicit handling of severe topological changes. While robust to topological noise (e.g., TOPKIDS), it may struggle with extreme topological variations not seen during training.

### Questions
Please answer and discuss the problems in ``Weakness''.

### Soundness
3

### Presentation
3

### Contribution
3
