# 3D PixBrush: Image-Guided Local Texture Synthesis

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
We present 3D PixBrush, a method for performing image-driven edits of local regions on 3D meshes. 3D PixBrush predicts a localization mask and a synthesized texture that are guided the object in the reference image. Our predicted localizations are both globally coherent and locally precise. Globally - our method contextualizes the object in the reference image and automatically positions it onto
the input mesh. Locally - our method produces masks that conform to the geometry of the reference image. Notably, our method does not require any user input (in the form of scribbles or bounding boxes) to achieve accurate localizations. Instead, our method predicts a localization mask on the 3D mesh from scratch. To achieve this, we propose a modification to the score distillation sampling technique
which incorporates both the predicted localization and the reference image, referred to as localization-modulated image guidance. We demonstrate the effectiveness of our proposed technique on a wide variety of meshes and images.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces 3D PixBrush, a method for locally editing 3D meshes by synthesizing textures guided by a reference image. It leverages a learned localization mask to accurately position the texture on the mesh and then synthesizes the texture to adhere to the style of the reference image. The method uses a modified score distillation sampling technique with both predicted localization and the reference image to provide guidance. It requires no manual user input (scribbles or bounding boxes) to achieve accurate localizations. The paper shows qualitative and quantitative results on a variety of meshes.

### Strengths
The strengths include:
-The approach of using a reference image to both localise and texture a specific region of a 3D mesh without manual user input is a new and interesting idea in texture editing. 

-The paper generally explains the methodology well, particularly the Localization Modulated Image Guidance (LMIG) component.

-The figures demonstrate the method's ability to transfer textures from reference images to 3D meshes in a semantically meaningful way. Some of the examples (e.g., the diverse textures on the same human head) are visually pleasing

-The paper includes quantitative evaluations using CLIP R-precision, showing improvements over the CVPR 2024 baseline method.

### Weaknesses
-The description of some components lacks sufficient technical detail. For instance, it would be good to specify the architecture of the Neural texture network in section 3.1.

-ablation studies are present, they do not analyze all of the hyperparameters that exist.

-The evaluation metric CLIP R-Precision may not fully capture the perceptual quality of the results.

### Questions
-Please provide more technical details about the Neural Texture network and the structure for the text guided 3D Generation component.

-The method predicts localization masks - how are these handled to avoid sharp transitions and artifacts when applying the synthesized textures?

-Please clarify what specific novel contributions does the paper make on top of other global style transfer methods?

### Soundness
3

### Presentation
3

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
The paper proposes a novel framework for image-guided local texture editing on 3D meshes. The method leverages Score Distillation Sampling (SDS) loss to jointly optimize two neural field representations: a localization map and a texture field.
Unlike prior works that either require users to manually define the localization map or rely solely on text prompts for guidance, this approach integrates the IP-Adapter to enable reference-image-based texture generation. Moreover, to ensure that the generated textures remain consistent with the global localization structure while preserving fine-grained local details, the authors introduce a Localization Map–Integrated Guidance (LMIG) mechanism, which masks the attention operation in the IP-Adapter using the predicted localization map.

### Strengths
1.It further introduces a Localization Map–Integrated Guidance (LMIG) mechanism, which integrates a mask into the cross-attention module of the IP-Adapter to ensure that the generated texture aligns with the predicted localization map.

2.Extensive experiments demonstrate the effectiveness and superior performance of the proposed approach.

### Weaknesses
1.The novelty of this work is limited, as the IP-Adapter used for image conditioning is already widely adopted in the community, and the proposed LMIG merely incorporates a predicted mask into the attention map.

2.The paper employs the IP-Adapter as the image guidance mechanism, extending 3D PaintBrush from text-guided editing to image-guided texture generation. However, since 3D PaintBrush already predicts both the localization map and the local texture, the novelty of this extension is rather limited and not clearly distinguished.

3.In the proposed LMIG mechanism, there is no analysis or justification provided regarding why the masked attention map can improve the consistency between the localization map and the texture. The initial localization map is obtained through a text-guided loss, which may not be fully aligned with the reference image content. Despite this potential mismatch, the localization map is directly used to modulate the attention map, while the final output simultaneously influences the localization map itself. This design introduces a potential circular dependency and raises questions about the stability and interpretability of the proposed mechanism.

### Questions
1.The CMD[1] framework performs multiview 3D editing, jointly modifying both the mesh geometry and texture, where local texture editing is treated as a sub-process. Could the authors elaborate on the advantages of their method compared with CMD? It would also be informative to evaluate the proposed approach on some of the cases used in the CMD paper.

2.I am curious about how the LMIG mechanism behaves when the text prompt does not correspond well to the reference image, and how the initial localization mask influences the final results under such conditions.

3.Since the image-guided loss also updates the localization map, how would an inaccurate mask applied in the attention operator affect the optimization of the localization map?

\[1\] [https://dl.acm.org/doi/10.1145/3721238.3730722](https://dl.acm.org/doi/10.1145/3721238.3730722)

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces 3D PixBrush, a novel method for image-guided local texture synthesis without user-specified mask, specifically on 3D meshes. The approach leverages a reference image and a text prompt to predict both a localization mask and a corresponding texture, enabling precise, automated edits without user-provided inputs like scribbles or bounding boxes. Key innovations include Localization Modulated Image Guidance (LMIG), which integrates text-driven coarse localization with image-driven refinement by modulating cross-attention features in an IP-Adapter model. It provides extensive qualitatively and quantitatively evaluation results.

### Strengths
1. **Good design of editing pipeline**: I like the whole design of editing pipeline, where the localization and texture prediction evolves through two branches, and localization is then used in LMIG. Although it still needs some warmup steps in localization, it’s already very different from prior methods which usually adopt two or three separate editing stages.
2. **Novel design of the masked cross-attention**: Although the method is well-grounded on previous techniques, it still introduces smart modifications which needs careful observations. The use of masked cross-attention convincingly demonstrate the necessity of itself, and it’s natural to be included in this framework.
3. **Extensive convincing results**: I think the provided qualitatively results are compelling, with diverse examples across different meshes and references. Also the quantitative metrics and perceptual studies provide strong evidence of superiority over prior methods. And I really appreciate the analysis of robustness to initialization and compositionality.

### Weaknesses
1. Although the design of the localization prediction is appreciated, it is limited to text prompt. Do the authors think this framework also supports user-specified mask or localization?
2. **Computational Efficiency**: Optimization requires 4 hours per edit on an A40 GPU, with reasonable results after 1 hour. It lacks comparisons to faster alternatives or discussions on acceleration.
3. **Limited Editing Operations Scale**:  it seems like it only supports texture synthesis on the medium scale part of the object, which can be well described by the text prompt SDS loss. But how about the texture synthesis on parts of smaller scale, like nails on the fingers of a hand and so on? I believe that requires new design of the localization prediction.

### Questions
- How sensitive is LMIG to the choice of cross-attention weighting? Are there guidelines for tuning these across different mesh complexities or reference image styles?
- The authors mention about the extensions to NeRFs or videos—could you elaborate on potential challenges, such as view consistency or temporal coherence, in adapting LMIG to these domains?

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
5

### Summary
The paper introduces 3D PixBrush, a method for image-driven local editing of 3D meshes. The method allows users to modify specific regions of a 3D object based on a reference image, automatically determining where and how to apply the edits—without requiring any manual input such as scribbles or bounding boxes. The authors demonstrate the effectiveness of 3D PixBrush across a wide range of meshes and reference images.

### Strengths
S1. The storyline of this paper is clear and easy to follow.

S2. Editing the local texture of a mesh driven by a reference image is an interesting topic.

S3. The proposed framework is reasonable.

### Weaknesses
W1. In section 3.4, the authors claim that the key to their method is the ability to supervise the local texture edits with image guidance. However, the use of SDS with image guidance has already been explored in the literature [1, 2]. Novelty of the method and contributions are limited.

W2. The paper only reports the CLIP score as the quantitative evaluation results, which is not comprehensive for quantitative evaluation. All the ablation studies are demonstrated by qualitative examples. I have no idea whether these results are cherry-picked or not. Moreover, it’s a 3D editing task. The paper only showcases single-view results. There are no multi-view renders or videos of the generated 3D mesh. 

W3. In line 305, the authors said, “see supplementary material.” However, I didn’t find the supplementary material. 

[1] IPDreamer: Appearance-Controllable 3D Object Generation with Image Prompts.
[2] VP3D: Unleashing 2D Visual Prompt for Text-to-3D Generation.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
