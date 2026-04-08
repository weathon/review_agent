## Human Reviewer 1

### Summary
This paper introduces a new 3D texture generation framework that isolates the color palettes, strokes, and textures from the style image and maps them to the surface of a given 3D mesh/asset. This framework consists of a trained diffusion model for multi-view stylized image generation and a 3D texture baking process to project the stylized views into UV space.

The major contribution of this paper is the introduction of the "jigsaw" operation, which involves random shuffling and masking of image patches to destroy the semantic structure while preserving the style statistics, allowing supervised training. Experiments show better style fidelity and cross-view consistency compared with StyleTex, MV-Adapter, and 3D-Style-LRM, with competitive runtime.

### Strengths
1. The design of patch-shuffle + mask achieves style-content disentanglement theoretically and empirically. Quantitative analysis and mathematical proof are provided to demonstrate the effectiveness and functionality of this design in terms of style-semantic decoupling and mean/variance preservation. Additionally, this method enables supervised learning in a clean and scalable manner.

2. Even though the multi-branch attention and reference-attention formulation are not very novel, they have been demonstrated to be simple and effective for this task, which is valuable for future study.  

3. The writing of this paper is clear and logically structured, with professional schematic diagrams and comparison figures.

### Weaknesses
1. The scope of this topic is relatively weak, focusing on the texture stylization of 3D assets. Recent 3D style transfer works can be applied at both the object and scene levels. Can this method be transferred to general 3D style transfer? Can the authors compare this method with recent 3DGS-based style transfer models (by giving six or more views to these models)?
2. Patch shuffling has already been studied in **StyleAdapter** to reduce the semantic information. What is the difference between your observation and theirs.
2. No perceptual or human evaluation of realism and consistency is provided.
3. Only 20 test meshes are used for evaluation, which may be too small to demonstrate the generalization ability of the proposed methods.

### Questions
1. Based on the content of this paper, I suggest that the authors change the title to "3D Texture Generation" instead of "3D Style Transfer," as the latter is too general and not suitable for this paper.
2. If more comparisons with recent 3DGS-based methods can be provided, I would like to increase the score accordingly.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This work proposes Jigsaw3D, a method for generating style-image conditioned textures for meshes. A key observation of this work is that existing datasets conflate semantics and style within the textures of their meshes. To address this, Jigsaw3D proposes a texture processing step that spatially shuffles and randomly masks patches of the reference textured renders. This allows the resulting patches to be independent of the object semantics and isolates the style component of the texture. Using these "jigsawed" patches, this method is able to train a multi-view diffusion model that is conditioned on cross-attention with these patches, enabling style-only conditioned multi-view diffusion. The method is evaluated both qualitatively and quantitatively, reporting improved results over baseline methods.

### Strengths
- This work identifies an issue with current image guided texturing approaches (style and semantics are very interconnected) and proposes a method that solves this problem, providing an original contribution to the community.
- The proposed “jigsaw” pre-processing step enables the creation of a dataset of style jigsaw  images and textured mesh pairs from a dataset of just textured meshes. This is a creative solution to the problem of disentangling style from semantics and can be used by followup works aiming to separate style and semantics.
- Jigsaw3D’s training-based approach leads to faster run times than per-shape optimizations using score distillation and is thus more practical than distillation methods.

### Weaknesses
- The desired degree to which semantics of the input mesh should be preserved seems very subjective. In image guided texturing there is a tradeoff between preserving the semantics on the input mesh and adhering to the style of the reference image. Jigsaw3D seems to preserve more semantics in some examples and less in others. For example, in Fig. 4, the lamp in StyleTex colors the shade with a more traditional paper texture while the base is colored bronze and the Jigsaw3D result colors the entire thing bronze. Some might argue that the full bronze texture has more fidelity to the reference image while the bronze base and paper shade has better semantic preservation of the input shape. However in the same figure, for the example with the pink floral dress on the car, the Jigsaw3D has more semantic preservation and less fidelity to the reference image. From these examples it appears that StyleTex and Jigsaw3D are comparable qualitatively, and it’s not clear what the desired behavior in these cases should be.
- In some cases (bronze lamp, cartoonish houses on bus), the results from MVAdapter appear comparable to Jigsaw3D and MVAdapter also has a similar fast runtime. Thus from the qualitative results, it is not clear that Jigsaw3D performs better than baseline methods.
- Given that a main claimed contribution of this work is the disentanglement of style and semantics I would expect more experiments explicitly showing this. Tab. 1 reports Gram and AdaIN metrics, but why these capture style / semantics disentanglement is not explained.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces JIGSAW3D, a novel framework for disentangled 3D style transfer that transfers artistic styles from 2D reference images to 3D meshes while preserving object identity and multi-view consistency. The core innovation is a Jigsaw transform that disentangles style from content by randomly shuffling and masking non-overlapping image patches, destroying global semantics while preserving local style statistics (e.g., mean, variance). This enables scalable synthesis of pseudo-paired style-texture training data from existing textured 3D assets without curated style datasets. The authors propose a multi-view diffusion model conditioned on disentangled style features and geometric cues (normals/positions). A key component is the Reference Attention mechanism, which dynamically injects style features into the denoising U-Net alongside self-attention and multi-view attention blocks. Stylized multi-view outputs are baked into UV textures via visibility-aware reprojection and seam-aware blending.

### Strengths
1. Originality. This paper presents a genuinely creative formulation of the 3D style disentanglement problem. The proposed jigsaw transform adapts the idea of patch shuffling—originally from 2D stylization—to the 3D domain in a surprisingly effective way. By disrupting spatial coherence while preserving local statistics, the authors successfully synthesize pseudo-paired style–texture data from unpaired assets (e.g., Objaverse), providing an elegant workaround for the lack of curated 3D style datasets. The reference attention mechanism—which combines self-, multi-view, and style-guided cross-attention—is both principled and practical. While its components are not entirely new, their integration for disentangled 3D style transfer feels original and well-motivated.

2. Quality. The method is technically solid and well-executed. The jigsaw operation (Eq. 1–2) convincingly preserves style statistics while removing semantics, and the inclusion of geometric cues through normal and position maps strengthens structural fidelity. The multi-view diffusion pipeline is coherent and thoughtfully designed, and the final texture baking—featuring visibility-aware reprojection and UV inpainting—demonstrates strong attention to practical details. The experiments are thorough, both visually compelling and quantitatively convincing. Overall, the technical quality is high and the implementation appears carefully considered.

3. Clarity. The paper is very clearly written. The motivation is intuitive and well-argued, and the methodology section flows logically from data preparation to generation and baking. Figures and equations are well integrated into the narrative, enabling readers to grasp the key ideas efficiently. The ablation and limitation sections are appropriately placed and easy to follow. It is one of the clearest papers in this area.

4. Significance. The potential impact of this work is notable. The ability to perform scalable, multi-object style transfer without per-asset optimization directly addresses a major bottleneck in 3D content creation, with immediate applications in gaming, VR, and digital art pipelines. The jigsaw-based disentanglement strategy may also inspire future research beyond stylization, such as in texture synthesis or domain adaptation. The authors’ plan to release code further enhances the potential influence of this work.

### Weaknesses
1. Novelty Overstated; 2D-to-3D Adaptation Not Fully Justified. The proposed jigsaw transform—combining patch shuffling and masking—closely resembles ideas from the 2D style transfer literature, where positional embeddings are shuffled for disentanglement. While the adaptation to 3D is interesting, the paper does not clearly explain why spatial patch shuffling provides a unique advantage for preserving geometry compared to feature-space methods. Suggested Improvement: Include experiments that compare the jigsaw-based disentanglement with feature-space alternatives such as Gram or AdaIN-based approaches in a 3D setting. It would also help to quantify geometric distortion (for example, using Chamfer distance) to justify the necessity of the jigsaw operation.

2. Inadequate Evaluation of Geometric Fidelity. The paper claims to maintain high geometric fidelity, but no quantitative evaluation is provided. The visual examples are persuasive, yet they are insufficient for complex geometries or subtle distortions. Suggested Improvement: Add geometric metrics—such as normal consistency or Chamfer distance—to objectively assess the preservation of shape. Testing on geometrically challenging assets (e.g., fractal or thin structures) would also strengthen the claims.

3. Insufficient Ablation of Core Components. Several critical design choices are not thoroughly validated. In particular, the necessity of the jigsaw transform is not tested against an unshuffled baseline, and the contribution of the three attention branches (self, multi-view, and reference) is not individually analyzed. Suggested Improvement: Provide ablations that remove key components—such as using the raw reference image without shuffling, or disabling the reference attention—to show their quantitative effect on style disentanglement and consistency.

4. Limited Validation of Multi-Object Stylization. The multi-object stylization results look promising, but the evaluation remains qualitative. It is unclear how consistent the style remains across objects within the same scene, or how geometry coherence is maintained when multiple shapes are involved. Suggested Improvement: Introduce scene-level measurements such as inter-object style similarity or surface normal alignment. Testing in more complex, cluttered environments would better demonstrate scalability and robustness.

5. Baking Process Robustness Not Fully Explored. The texture baking process, including seam-aware blending and UV inpainting, is only briefly discussed. Potential failure cases such as heavy occlusion or non-manifold geometry are not examined, and no quantitative UV-space evaluation is given. Suggested Improvement: Show examples of baking failures and describe how the system handles them. Reporting UV-space metrics like seam error or texture continuity would provide stronger support for the robustness of the method.

6. Efficiency Claims Unsubstantiated. The paper emphasizes scalability without per-asset optimization but does not include timing benchmarks. Without concrete measurements, it is difficult to assess whether the method is truly more efficient than prior optimization-based approaches. Suggested Improvement: Report training and inference time per object, and include comparisons with standard diffusion or optimization-based baselines to demonstrate actual runtime improvements.

### Questions
1. Necessity of Spatial Patch Shuffling. Could the authors clarify why spatial patch shuffling is essential for 3D style disentanglement? Similar effects have been achieved in 2D through feature-space shuffling. A comparison with a latent-space shuffling baseline would help justify this design choice.

2. Quantitative Evaluation of Geometric Fidelity. The paper emphasizes preserving geometry but relies mainly on visual comparisons. Could the author provide quantitative metrics such as Chamfer distance or normal-based errors between the original and stylized meshes?

3. Ablation on Jigsaw and Reference Attention. Can the authors include ablations to show (a) the effect of removing the jigsaw operation and (b) the contribution of each attention branch (self, multi-view, reference)? Quantitative results would make these components’ roles clearer.

4. Generalization to Complex Geometries. How does the method perform on objects with fine details, thin structures, or non-manifold geometry? Some examples or metrics on such challenging cases would strengthen the generalization claim.

5. Efficiency and Scalability. Since the paper highlights efficiency compared to per-asset optimization, could you report inference time, training cost, and memory usage?

6. Interpretation of Masking in the Jigsaw Operation. What role does stochastic masking play in style transfer? Does it mainly promote robustness or feature diversity? Results with different mask ratios could clarify its effect.

7. Failure Cases in UV Baking. Could the paper discuss typical failure cases in the baking process and how they are mitigated? A few visual examples would improve transparency.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper presents JIGSAW3D, a novel 3D style transfer method to restyle the texture of a 3D assets with a provided reference image. 
Specifically, JIGSAW3D operation contains spatial shuffling and random masking of the reference patches. It can suppress object semantics and isolate stylistic statistics. 

The main contributions are two folds: (1) Jigsaw-based reference to construct a pseudo-paris dataset for supervised training; similar operation can be applied to the real reference image at inference stage. (2) Training reference-attention module incorporating self, multi-view, and reference attentions to inject disentangled styles to the diffusion model. 

Experiments are conducted on Objaverse. Reference images are from both WikiArt dataset and collected 40 extra images. Baselines include three 2025 method for stylisation. Both quantitative and qualitative results are shown. Ablation studies w.r.t. Jigsaw are provided.

### Strengths
## Strengths 

- 1. The idea of Jigsaw has been explored in feature learning, e.g., Unsupervised Visual Representation Learning [R1], which shows effectiveness in context prediction. This paper leverages this simple but powerful operation, and combines it with masking operation, which is also a helpful operation in self-supervised learning such as MAE [R2]. The Jigsaw3D serves as a simple but effective operation to destroy semantics while preserving style information contained in image patches. Overall, it is a smart and novel approach to disentangle semantic from style. 
- 2. A pseudo pair dataset is constructed from Objaverse by generating 6 orthogonal views as ground-truth and 4 random views as reference images. This method enables supervised training but is free from heavy pair annotation. 
- 3. Reference attention consists of three attention modules: self-attention, multi-view attention, and reference attention. It leverages the cross-view interaction and inserts reference image styles into the 3D views. 
- 4. Experimental results are diverse and evident. 
> Quantitative comparisons are conducted with three recent (2025) methods, incicating the better performance and lower cost time.   
> Qualitative results are shown on both WikiArt and self-collected reference images.   
> Diverse results w.r.t. multi-object, partial, tieable scenes are shown. 


[R1] Doersch, C., Gupta, A., & Efros, A. A. (2015). Unsupervised visual representation learning by context prediction. In Proceedings of the IEEE international conference on computer vision (pp. 1422-1430). 
[R2] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16000-16009).

### Weaknesses
## Weaknesses
- 1. Paper oranisation and description of the pseudo pair dataset construction part can be improved. In current version of writing, the training/inference stages are kind of mixed together. 
> E.g., in my understanding, during training in Figure 2, I should stand for a random view refence image from the Objaverse. During inference, the reference image is user provided.   
> In Sec 3.1 and Sec 3.2, it is also confusing to distinguish the constructed dataset/pseudo reference image and the real user-provided reference image at inference time.   
> Moreover, from Fig 2, it seems like during inference of each reference image stylisation, the model is fine-tuned. Is this the case? Or only during training on the pseudo paired dataset, the model is finetuned and keep frozen at inference time? 
- 2. In Fig. 3, as Divisions increases, Gram Matrix increases significantly. While in Tab. 1, Gram is a metric expected to be the smaller the better. How to understand this inconsistency? 
- 3. Why the jigsaw patch sizes are different during training and inference? 
- 4. Will the self-collected 40 reference images be released? This will facilitate future comparison. 
- 5. The proof in A.2 is trivial.

### Questions
The questions are provided in the above Weaknesses points 1. ~ 4.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3