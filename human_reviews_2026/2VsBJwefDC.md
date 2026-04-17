# HoloPart: Generative 3D Part Amodal Segmentation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
3D part amodal segmentation--decomposing a 3D shape into complete, semantically meaningful parts, even when occluded--is a challenging but crucial task for 3D content creation and understanding. Existing 3D part segmentation methods only identify visible surface patches, limiting their utility. Inspired by 2D amodal segmentation, we introduce this novel task to the 3D domain and propose a practical, two-stage approach, addressing the key challenges of inferring occluded 3D geometry, maintaining global shape consistency, and handling diverse shapes with limited training data. First, we leverage existing 3D part segmentation to obtain initial, incomplete part segments. Second, we introduce HoloPart, a novel diffusion-based model, to complete these segments into full 3D parts. HoloPart utilizes a specialized architecture with local attention to capture fine-grained part geometry and global shape context attention to ensure overall shape consistency. We introduce new benchmarks based on the ABO and PartObjaverse-Tiny datasets and demonstrate that HoloPart significantly outperforms state-of-the-art shape completion methods. By incorporating HoloPart with existing segmentation techniques, we achieve promising results on 3D part amodal segmentation, opening new avenues for applications in geometry editing, animation, and material assignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
HoloPart introduces a novel diffusion-based model for 3D part shape completion and formally introduces the 3D part amodal segmentation task with two benchmarks (ABO and PartObjaverse-Tiny). It outperforms SOTA shape completion methods on these two benchmarks.

### Strengths
1. The 3D part amodal segmentation task and the ability to infer occluded 3D geometry of less complex geometric structures (i.e., Parts) are important for 3D understanding and can be beneficial for broad applications.
2. HoloPart outperforms competitors on the benchmarks on ABO and PartObjaverse-Tiny datasets.

### Weaknesses
## Major Weaknesses
1. In the two-stage pipeline, the 3D part segmentation method is crucial for final performance. SAMPart3D is not a very robust method. The segmented part definitions are not in control and often require additional merging for evaluation. Missing parts or incompatible segmented part definitions will significantly impact performance. So I don't think it is a very good initialization for the benchmark evaluation.

## Minor Weaknesses
1. Leveraging 3D generative priors indeed overcomes the limitations of scarce training data. However, it will also introduce their failure modes and may still fail on challenging cases for current 3D shape diffusion models.

### Questions
1. Based on weakness 1, given a rendered image of a mesh having 3D part annotations, could we use the 2.5D geometry of this view as the initialization for part completion (only consider visible parts in this view)?  It is a more challenging part-completion scenario. Could you demonstrate HoloPart's performance on this 2.5D part geometry on a small scale (e.g., 20-50 examples)?

2. It would also be valuable to provide additional qualitative and quantitative results of HoloPart on car and airplane categories (68 meshes in total) in the 3DCompat++ [1] dataset to show HoloPart's generalization ability in outdoor objects and with more semantically meaningful part definitions. 3DCompat++ has rendered images, and it would be easy to extract the 2.5D part geometry.

[1] 3DCOMPAT++: A comprehensive dataset for 3D object understanding with fine-grained part annotations

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes HoloPart, a framework that decomposes a complete 3D object into multiple complete and coherent 3D parts. The pipeline first employs SAMPart3D to obtain semantic mesh segmentation masks, and then feeds each incomplete segmented mesh into the proposed model, which reconstructs the complete 3D part meshes.

The contributions lie primarily in the architectural design of the proposed model and the construction of a new dataset for training and evaluation. The method is compared against state-of-the-art shape completion approaches — PatchComplete, DiffComplete, and SDFusion — and achieves superior performance across Chamfer Distance, IoU, F1-Score, and Success Rate metrics. Ablation studies further validate the effectiveness of the Context-Aware Attention, Local Attention, and the influence of the Guidance Scale.

### Strengths
1. The paper tackles a valuable and underexplored problem, focusing on the generation of complete 3D parts rather than full-object completion.
2. The construction of a dedicated dataset for 3D part completion is a useful contribution that can facilitate future research in this direction.

### Weaknesses
1. The approach relies heavily on existing 3D part segmentation techniques, which could limit its robustness when segmentation quality is poor.
2. The task formulation assumes the input is a complete object mesh, yet the pipeline includes VAE compression and flow matching in latent space. It is unclear whether the reconstructed meshes remain geometrically consistent with the original object after decoding.

### Questions
1. Since the pipeline encodes the 3D mesh via a VAE, performs flow matching in latent space, and then decodes it back, does the reconstructed mesh deviate geometrically from the original input mesh? If so, how do the authors mitigate or correct such deviations?

Things to improve the paper that did not impact the score:
- Table 2: The Success Rate metric is mentioned in the text but not shown in the table — please clarify where it is reported.

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
3

### Summary
The paper proposes a 2 step procedure to do 3D amodal segementation. Stage 1:  it uses an off-the-shelf approach to segment out the semantically consistent but visible portions of the occluded object.  Stage 2:  uses a local and global context aware diffusion method to  complete the occluded 3d subobject. Additionally they also propose a data pipeline strategy to curate  3D shapes with labeled semantic sub parts and their full (amodal) geometry for training and evaluation

### Strengths
1) The central idea of completing the occluded 3d sub part with a local and globally conditioned diffusion model is quite appealing

2) The datapipeline strategy to curate paired data is simple and easy to engineer and would be quite useful for future models.

3) Evaluate their model for various settings and compare to various baselines.

### Weaknesses
1) Reliance on segmentation quality of the off-the-shelf model in stage-1. Difficulties would arise due to some type of domain shift, due thin structures or heavy occlusions that prevent the stage 1 model from being able to segment the sub part well.

2) In general two stage training pipelines are a bit clunky as it requires two models to be trained unlike end-to-end trained models.

### Questions
I am curious to know about the compute comparison between the proposed model and baselines, in terms of flops/training time/ number of model parameters.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces 3D part amodal segmentation, a new task aimed at decomposing a 3D shape into its complete, semantically meaningful parts, including portions that are occluded. The authors propose HoloPart, a novel diffusion-based generative model that takes incomplete part segments and completes them by leveraging both local part geometry and global shape context.

### Strengths
1. The task presented in the paper is well-motivated and practical.
2. The dual local and context-aware attention mechanisms is a good design for balancing fine-grained part detail with overall shape consistency.
3. The presented results look good.

### Weaknesses
1. It is unclear how well the model generalizes to novel part compositions not well-represented in the finetuning data, as the completion may be heavily reliant on the part-whole priors learned from the ABO and Objaverse datasets.
2. The method's handling of semantic ambiguity is not discussed. For instance, if a mask incorrectly bridges two distinct semantic parts (like a chair leg and the seat), the model's behavior is unpredictable.
3. The generative completion process can introduce geometric hallucinations that deviate from the original shape's implicit structure. For instance, in Figure 13 (the turkey), the completed parts appear to add new geometry not implied by the original surface, and the final merged object shows significant inter-part overlaps.
4. There is a disconnect between the paper's "amodal segmentation" framing and its core "part completion" contribution. The method is entirely dependent on the quality of the initial segmentation and lacks any mechanism to refine, correct, or handle the noisy and often semantically incorrect outputs of the first stage, thus not fully addressing the end-to-end segmentation problem. Changing the narrative of the paper would make the contribution more aligned.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
