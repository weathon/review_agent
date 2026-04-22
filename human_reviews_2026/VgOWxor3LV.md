# Stroke3D: Lifting 2D strokes into rigged 3D model via latent diffusion models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Rigged 3D assets are fundamental to 3D deformation and animation. However, existing 3D generation methods face challenges in generating animatable geometry, while rigging techniques lack fine-grained structural control over skeleton creation. To address these limitations, we introduce Stroke3D, a novel framework that directly generates rigged meshes from user inputs: 2D drawn strokes and a descriptive text prompt. Our approach pioneers a two-stage pipeline that separates the generation into: 1) Controllable Skeleton Generation, we employ the Skeletal Graph VAE Sk-VAE to encode the skeleton's graph structure into a latent space, where the Skeletal Graph DiT Sk-DiT generates a skeletal embedding. The generation process is conditioned on both the text for semantics and the 2D strokes for explicit structural control, with the VAE's decoder reconstructing the final high-quality 3D skeleton; and 2) Enhanced Mesh Synthesis via TextuRig and SKA-DPO, where we then synthesize a textured mesh conditioned on the generated skeleton. For this stage, we first enhance an existing skeleton-to-mesh model by augmenting its training data with TextuRig—a dataset of textured and rigged meshes with captions, curated from Objaverse-XL. Additionally, we employ a preference optimization strategy, SKA-DPO, guided by a skeleton-mesh alignment score, to further improve geometric fidelity. Together, our framework enables a more intuitive workflow for creating ready-to-animate 3D content. To the best of our knowledge, our work is the first to generate rigged 3D meshes conditioned on user-drawn 2D strokes. Extensive experiments demonstrate that Stroke3D produces plausible skeletons and high-quality meshes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors propose Stroke3D, a novel framework that generates mesh-skeleton pairs from 2D strokes and text prompts. The pipeline first uses an Sk-VAE/DiT combination to achieve controllable skeleton generation, conditioned explicitly on the 2D strokes for structure. It then synthesizes the mesh by enhancing a model with the TextuRig dataset and applying the SKA-DPO preference optimization for geometric fidelity.

### Strengths
1. The paper tackles an interesting and novel task: generating skeleton-mesh pairs from 2D strokes and text prompts, which has not been explored in prior work.

2. The proposed method demonstrates effective results for skeleton-mesh generation. The two-stage generation pipeline effectively decouples skeleton and mesh generation, allowing each component to be optimized independently while maintaining geometric consistency through the skeleton-conditioned mesh generation stage.

### Weaknesses
1. The paper misuses the term "rigged model." In standard 3D graphics terminology, rigging includes both skeleton and skinning weights. Since this work only predicts the skeleton without skinning weights, the authors should use precise terminology such as "skeleton prediction" to avoid misleading readers about the actual technical scope.

2. I do not agree that the dataset is one of the main contributions. (1) It is curated from UniRig rather than being newly collected. (2) Since both SkDream and UniRig source from Objaverse-XL, potential duplicates should be identified and reported. (3) The authors should justify why they did not use MagicArticulate's Articulation-XL, which contains more rigged models. While Articulation-XL lacks textures, these could be obtained from the original Objaverse-XL sources.

3. The method predicts only connected joint positions without their hierarchical tree structure. Prior works (RigNet, UniRig, MagicArticulate) all predict parent-child relationships between joints, which are essential for animation workflows. Without this connectivity information, the output cannot be directly used in standard 3D animation pipelines.

4. The category selection from Articulation-XL (human, animal, plant) lacks justification. Plants are neither a major category in the source dataset nor commonly associated with skeletal rigging needs, unlike toys or articulated objects. The authors should provide selection statistics and explain why underrepresented categories like plants are included while more relevant categories are excluded.

5. The text prompt is only used for skeleton generation, while mesh generation proceeds without textual guidance. This is counterintuitive, as mesh geometry and appearance are more naturally conditioned on text descriptions than skeletal structure.

### Questions
1. Missing citation: 
Auto-Connect: Connectivity-Preserving RigFormer with Direct Preference Optimization, which also uses DPO for skeleton generation.

2. The current placement of figures, particularly Figure 4 and Figure 5 (Experimental Results), is suboptimal as they appear within the Method section. One suggestion is to rearrange the figures to fit the paper structure.

3. Verify and correct the mark used for the B2B-CD (All) metric in Table 1, as it appears to be incorrect.

4. Include a brief, clear statement in the main paper that you select part of the MagicArticulate Dataset for both training and testing, rather than only mentioning this information in the Appendix.

### Soundness
2

### Presentation
2

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
The paper introduces Stroke3D, a novel framework that generates rigged 3D meshes from user-drawn 2D strokes and text prompts. Unlike prior methods that generate a 3D mesh first and then rig it, Stroke3D adopts a skeleton-first generation pipeline. Experiments on MagicArticulate and SKDream benchmarks show that Stroke3D outperforms baselines such as RigNet, UniRig, and SKDream, achieving better Chamfer Distance and SKA scores. The system demonstrates both structural control and high visual quality, enabling intuitive rigged 3D asset generation.

### Strengths
1. First framework enabling rigged 3D generation directly from 2D strokes and text.
2. Outperforms baselines such as RigNet, UniRig, and SKDream, achieving better Chamfer Distance and SKA scores.
3. Introduce TextuRig, a dataset of textured and rigged 3D meshes with captions

### Weaknesses
1. Other pipelines generate skeletons from 3D meshes, while this paper generates skeletons directly from 2D strokes. Therefore, comparing skeleton quality metrics between these methods is not entirely fair, since the input modalities differ significantly in both information richness and structural constraints.

2. The contributions of this paper are relatively limited. It mainly proposes a model that generates 3D skeletons from 2D strokes and introduces a modest extension of the SKDream dataset, rather than delivering fundamentally new insights or large-scale advancements to the field.

### Questions
1.How is the camera viewpoint determined when projecting the 3D skeleton into 2D strokes?

2.What are the specific design details of the Skeletal Graph Variational Autoencoder (Sk-VAE)? According to the paper, the VAE decoder requires the set of edges to reconstruct the skeleton. So how are these edges obtained during the generation process?

### Soundness
3

### Presentation
3

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
This paper proposes a novel framework for synthesizing rigged 3D skeletons from 2D drawn strokes and text prompts. The authors first curate a dataset from existing open-source 3D rigging collections, then train skeleton VAE and diffusion models to enable generation conditioned on 2D strokes. To further enhance performance, the paper introduces a skeleton-mesh alignment process guided by an optimization step, improving the model's fidelity. Experiments demonstrate that the proposed method outperforms existing approaches.

### Strengths
1. The paper addresses an interesting and novel task: generating 3D articulated objects from a combination of 2D strokes and text descriptions. The problem itself is well-motivated and interesting.
2. The work improves upon existing baselines in several areas, including dataset curation, VAE design, and a post-hoc optimization stage. This results in a stronger baseline for future research in rigged 3D generation.
3. The paper is well-written, clearly structured, and easy to follow.

### Weaknesses
1. My main concern revolves around the paper's technical novelty. The proposed method appears to be a successful combination of existing technologies: 
* A data curation step that refines datasets from prior work (Unrig, MagicArticulate).
* Slight modifications to the existing Sk-dream model architecture.
*  The application of a post-training optimization phase, which is a common technique for refinement.
Consequently, the contribution seems to lie more in effective engineering and integration than in providing new fundamental insights.

2. The experimental evaluation is not sufficient. A key claim is the ability to generate rigged objects from 2D sketches, yet there is no quantitative metric to evaluate the consistency between the input 2D sketch and the generated 3D object. I suggest that the authors include such a metric (e.g., projection-based distance) to substantiate their claims.

3. The process for creating the training data is not sufficiently detailed. Specifically, the paper should clarify how the paired 2D drafts and their corresponding 3D ground-truth (GT) models were generated. This information is crucial for reproducibility.

### Questions
A potential limitation of the method appears to be its robustness to imperfect user inputs. The paper primarily showcases results with clean, well-defined 2D drafts. However, in real-world scenarios, user-drawn strokes are often noisy, shaky, or not perfectly straight. How would the model perform under such conditions? I would encourage the authors to provide an analysis or at least a qualitative study on the model's sensitivity to variations in input stroke quality.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents Stroke3D, a skeleton-first pipeline that produces rigged 3D meshes from user-drawn 2D strokes plus a text prompt. The method learns a latent space for skeletal graphs with a Skeletal Graph VAE (Sk-VAE) and performs denoising in that space using a Skeletal Graph Diffusion Transformer (Sk-DiT) conditioned on text and stroke features; the decoded skeleton then conditions a mesh generator. A second stage improves skeleton-to-mesh synthesis by curating TextuRig (a textured, rigged subset with captions) and applying skeleton–mesh-alignment-guided Direct Preference Optimization (SKA-DPO). On MagicArticulate (skeletons) and SKDream (meshes), the approach lowers Chamfer distance and raises SKA scores, with qualitative results indicating better adherence to stroke structure and semantics. The work is promising but constrained by data coverage, evaluation breadth, and reliance on VLM-assisted curation.

### Strengths
1. Quality. The technical description is detailed and concrete, including the conditioning mechanism, classifier-free guidance, and the SKA-DPO objective. Training protocols and ablations (e.g., stroke guidance aiding convergence; the DPO margin study) are appropriate. Quantitatively, Stroke3D improves CD metrics and SKA scores over strong baselines.
2. Clarity. The end-to-end pipeline is clearly presented with informative figures (data preparation and overall architecture), and the contributions are explicitly itemized. The appendices document alignment and data-processing choices relevant for reproducibility.
3. Significance. A stroke-and-text interface can lower the barrier to authoring rigged, animation-ready assets. The TextuRig curation and alignment-guided preference optimization are broadly useful ideas for skeleton-to-mesh generation.

### Weaknesses
1 Data dependence and coverage. Performance and generalization rely on curated sets (MagicArticulate, SKDream, and TextuRig). The paper acknowledges dataset limitations and sensitivity to rare concepts (e.g., plants before DPO). Scaling and coverage remain open issues.
2 Stroke simulation vs. real inputs. Structural conditioning is trained on perturbed 2D projections of 3D skeletons rather than large-scale human sketches. This domain gap may reduce robustness to messy real drawings; analysis of viewpoint or stroke ambiguity is limited.
3 Evaluation breadth. Skeleton evaluation is primarily CD-based and mesh alignment uses SKA on a 108-sample set following SKDream. There are no user studies or downstream animation stress tests (e.g., auto-skinning stability under motion), and comparisons to recent autoregressive skeleton generators under stroke constraints are sparse.
4 Ablations and controls. While DPO-margin and stroke-conditioning ablations are provided, the incremental effects of TextuRig versus SKA-DPO are not fully disentangled across categories, and variance over random seeds is not reported.

### Questions
1. Robustness to real sketches. How does performance change with real, noisy human strokes (varying density, missing joints, occlusions)? Please provide sensitivity curves versus stroke sparsity/noise.
2. Generalization to rare/unseen concepts. Can you quantify success rates for long-tail categories (e.g., “samurai,” “turtle”) and the effect of prompt phrasing beyond qualitative examples?
3. Downstream animation quality. Beyond SKA/CD, do the produced meshes remain stable under standard auto-skinning and motion-retargeting tests (e.g., penetration, joint collapse) across sequences?

### Soundness
3

### Presentation
3

### Contribution
3
