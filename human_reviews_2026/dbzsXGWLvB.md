# TRELLISWorld: Training-Free World Generation from Object Generators

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Text-driven 3D scene generation holds promise for a wide range of applications, from virtual prototyping to AR/VR and simulation. However, existing methods are often constrained to single-object generation, require domain-specific training, or lack support for full 360-degree viewability. In this work, we present a training-free approach to 3D scene synthesis by repurposing general-purpose text-to-3D object diffusion models as modular tile generators. We reformulate scene generation as a multi-tile denoising problem, where overlapping 3D regions are independently generated and seamlessly blended via weighted averaging. This enables scalable synthesis of large, coherent scenes while preserving local semantic control. Our method eliminates the need for scene-level datasets or retraining, relies on minimal heuristics, and inherits the generalization capabilities of object-level priors. We demonstrate that our approach supports diverse scene layouts, efficient generation, and flexible editing, establishing a simple yet powerful foundation for general-purpose, language-driven 3D scene construction. We will release the full implementation upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TRELLISWorld, a training-free framework for large-scale 3D scene generation. The main contribution is that it enables scene-level generation without requiring scene-level training data, leveraging object-level pretrained 3D generative models (Trellis) instead.
The method divides a 3D scene into multiple overlapping patches. For patches with shared regions, a weighted averaging over the overlapping areas is applied during generation to ensure local consistency. Ablation studies show that using a 3D cosine mask for blending produces smoother transitions than direct averaging.
Compared with SynCity, TRELLISWorld achieves a slight improvement on CLIP Mean, but at the cost of significantly higher generation time and computational resources.

### Strengths
1. Training-free and data-efficient: The method does not rely on scene-level 3D data; it extends an object-level pretrained model to generate full 3D scenes.
2. Effective patch blending: The use of a 3D cosine mask during blending helps preserve the central region of each generated patch while enabling smooth transitions at the edges.

### Weaknesses
1. Evaluation inconsistency: For perceptual alignment comparison, the paper states that 18 close-distance views were uniformly sampled, yet the presented examples are mostly long-distance views. Showing close-up comparisons would provide a more convincing evaluation.
2. Limited controllability: The method only supports text-based control, which limits fine-grained scene manipulation. Designing prompts for each patch can be cumbersome, even though the authors mention using LLMs to generate multiple prompts. However, how these prompts are assigned to specific patches to ensure coherent spatial layout is unclear.
3. Low-resolution figures: The figures are generally low in resolution, making it difficult to assess the fine-grained 3D details of generated objects.

### Questions
1. The overall quality of the results is hard to assess since most examples only show low-resolution overviews of entire scenes. It is strongly recommended that the authors include close-up visualizations and higher-resolution renderings to better demonstrate scene quality.
2. As the method relies on Trellis, a text-to-3D model, each patch requires a corresponding text prompt. However, the paper lacks a detailed explanation of how prompts are managed or distributed across patches (e.g., after generating multiple city-related prompts via LLMs, how are they spatially allocated to form a coherent scene?).

Things to improve the paper that did not impact the score:
- Figure 2 should be moved to page 4, closer to where it is referenced.
- In Formula (2), please clarify the meaning of the $\mathcal{O}$ notation and $[\cdot]_f$.
- For a tile size of 4×3×1, what is S? In Figure 3(a), it seems the layout is divided into 8×6 tiles. So is S = 0.5?
- In Figure 4, “without blending” is misleading since simple averaging aggregation is also a form of blending. Consider using “average” and “weighted average” or others instead.
- The tiled decoder mentioned in the ablation study is difficult to locate in the Method section; please reorganize the text to highlight this component more clearly.

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
The paper aims to lift the ability of 3D object generation models to 3D world space. It introduces a training-free approach, trellisworld, to achieve 3D world generation. Based on a text-to-3D model, trellisworld denoises multiple 3D tiles in parallel and use an average weighting mechanism to aggregate the results at each timestep, thereby generating coherent 3D world.

### Strengths
Overall, the paper introduces a training-free method to generate 3d world using a 3D object generator. It provides a simple and effective method to achieve meaningfull applications.

### Weaknesses
The originality of the paper is somehow limited. The article claims the difference between them and syncity is that syncity depends on image inpainting, but this is merely a difference in the conditional mechanism. Aside from this conditional mechanism, the overall pipeline, which involves generating tiles and then blending, is very similar. Furthermore, the mechanisms of tile diffusion and blending mentioned in the paper are very similar to those of MultiDiffusion [1], and I haven't seen any effective strategies specifically designed for 3D.

[1] MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation

### Questions
1. The performance of TRELLIS in text-to-3D geneartion is worse than in image-to-3D generation. Therefore, I'm wondering if using text as the conditioned condition might have a lower performance, as I understand that generalization and controllability should be worse than models conditioned on images.
2. Quality of figures in the paper should be improved.
3. What is the maximum number of tiles that Trellisworld can generate simultaneously? How does the model's performance change as the number of simultaneously generated tiles increases?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose TRELLISWorld, a training-free framework for text-driven 3D scene generation, which composes complex scenes by leveraging pre-trained text-to-3D object diffusion models.
Instead of training an end-to-end scene generator, TRELLISWorld decomposes the scene noise into multiple object-level subregions (“chunks”) and employs a cosine-weighted re-aggregation strategy to efficiently synthesize large-scale 3D environments.
Compared to the state-of-the-art method SynCity, TRELLISWorld achieves superior visual quality and significantly faster inference, demonstrating the effectiveness of its modular and scalable generation approach.

### Strengths
1. State-of-the-Art Results
The proposed TRELLISWorld achieves superior CLIP score performance compared to the recent state-of-the-art method SynCity, while also requiring less computational resources and delivering faster inference speed. This demonstrates the efficiency and scalability of the training-free design.

2. Comprehensive Ablation Studies
The authors present comprehensive qualitative ablation studies on key components—Tiled Diffusion, Blending, and Tiled Decoder—clearly illustrating the contribution of each to the final scene generation quality. These studies effectively highlight how each module enhances visual coherence and overall realism.

### Weaknesses
1. Heavy Reliance on the Base Model
As acknowledged in the manuscript, the proposed method—being training-free—is inherently limited by the capabilities of its underlying base model, TRELLIS. Consequently, the overall performance and generalization ability are closely tied to the pretrained model’s strengths and weaknesses, which may restrict the method’s applicability across diverse domains.

2. Lack of Quantitative Ablation Studies
While the qualitative ablation studies provide valuable insights, the paper would benefit from quantitative analyses to numerically assess the contribution of each component. Such evaluations would help clarify how elements like Tiled Diffusion, Blending, and Tiled Decoder quantitatively influence the final output quality and performance.

### Questions
1. Could the authors investigate how the performance changes when the base model is replaced with alternatives to TRELLIS? Such an analysis would help assess the generality and adaptability of the proposed framework.

2. It is assumed that the stride size used in the tiled generation process may influence the final performance. Could the authors conduct additional experiments with varying stride sizes to analyze its impact on scene quality and consistency?

3. Could the authors provide quantitative ablation results (e.g., CLIP score) without the proposed components to clarify the contribution of each module to the overall performance?

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
3

### Summary
The paper targets the training-free goal for 3d scene generation. The authors propose TRELLISWorld, which reframes text-to-3D scene synthesis as a multi-tile denoising problem. The overlapping 3D regions are generated by a pretrained object-level model and blended with cosine-weighted averaging. The experiments show qualitative results with advantages over SynCity.

### Strengths
The core idea of using tiled diffusion with cosine blending to smoothen the inter-tile transition is straightforward with easy-to-understand intuition.

The method description is clear and the implementation provides some details, though it's doubtful if it's sufficient for readers to reperform w/o open-sourced codes. 

The results show clear advantages over the peering work Syncity.

The limitation section acknowledges its base-model dependence and lack of object disentanglement.

### Weaknesses
As mentioned in the strength, the method is quite straightforward, therefore the impact heavily lies in the provision of the tool as opensourced code to the community, as SynCity has done.

The innovative contribution is more an incremental improvement of Trellis, thus whether it meets the standard as a standalone paper in ICLR may need further discussion.

The work is heavily depending on the base object generator, which limits the contribution. 

The comparison is mainly against SynCity while other recent 3Descene generation works referenced in the related work sections are largely missing.

The computation cost analysis is too simple, without showing any memory/runtime tests scaling with tile counts or comparisons to optimized SDS/LRM pipelines.

### Questions
1. Could the authors provide complete implemenation details with full metric setups?
2. Could the authors broaden the tests addressing the comments in the weakness, e.g. scale up with tile count to test computation, add other 3d scene baselines etc.?
3. Does the proposal only work for static scene? Any idea how to make it work on dynamic scene?
4. Can this proposal function as well on other base generator besides Trellis? Could the authors test the performance impact across a few other base generators?

### Soundness
3

### Presentation
3

### Contribution
3
