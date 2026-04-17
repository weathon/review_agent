# SplatFont3D: Structure-Aware Text-to-3D Artistic Font Generation with Part-Level Style Control

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Artistic font generation (AFG) can assist human designers in creating innovative artistic fonts. However, most previous studies primarily focus on 2D artistic fonts in flat design, leaving personalized 3D-AFG largely underexplored. 3D-AFG not only enables applications in immersive 3D environments such as video games and animations, but also may enhance 2D-AFG by rendering 2D fonts of novel views. Moreover, unlike general 3D objects, 3D fonts exhibit precise semantics with strong structural constraints and also demand fine-grained part-level style control.  To address these challenges, we propose SplatFont3D,  a novel structure-aware text-to-3D AFG framework with 3D Gaussian splatting, which enables the creation of 3D artistic fonts from diverse style text prompts with precise part-level style control. Specifically, we first introduce a Glyph2Cloud module, which progressively enhances both the shapes and styles of 2D glyphs (or components) and produces their corresponding 3D point clouds for Gaussian initialization. The initialized 3D Gaussians are further optimized through interaction with a pretrained 2D diffusion model using score distillation sampling. To enable part-level control, we present a dynamic component assignment strategy that exploits the geometric priors of 3D Gaussians to partition components, while alleviating drift-induced entanglement during 3D Gaussian optimization. Our SplatFont3D provides more explicit and effective part-level style control than NeRF, attaining faster rendering efficiency. Experiments show that our SplatFont3D outperforms existing 3D models for 3D-AFG in style–text consistency, visual quality, and rendering efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SplatFont3D, a model for structure-aware text-to-3D artistic font generation with precise component-level style control. The method combines 3D Gaussian Splatting with a 2D diffusion prior-driven Score Distillation Sampling (SDS). It includes a Glyph2Cloud module for geometry-aware 3D Gaussian initialization from 2D glyphs and introduces a dynamic component assignment strategy to address part disentanglement issues during optimization.

### Strengths
- The paper clearly articulates the need for and importance of 3D artistic fonts with fine-grained style control, as well as the limitations of previous NeRF/3DGS-based models in handling font semantic constraints.

- It proposes Glyph2Cloud, a method for initializing global 3D representations, and a dynamic component assignment strategy for local component editing, achieving geometric information grouping and iterative optimization for disentanglement.

- Experiments demonstrate that SplatFont3D has significant advantages in generation quality, controllability, and training efficiency.

### Weaknesses
- The introduction is overly verbose, with Lines 54-78 and Lines 79-96 containing repetitive content. Figure 1 lacks formulaic symbol annotations, making it difficult to understand the formulaic symbols in the method section.

- The multi-view consistency is insufficient in Table 4.2. For example, in Figure 6, the side view of the leaves still appears wide. Regarding efficiency, while using 3DGS for geometric representation is an obvious way to improve efficiency, the three-stage training strategy and component-wise independent SDS optimization further reduce efficiency.

- Technically, the paper primarily includes mask-guided 3D artistic font generation and an additional 3D editing framework. The main contribution should lie in the latter, rather than the 3DGS architecture and its advantages. Therefore:
1. The paper lacks references related to 3D editing, such as GaussianEditor, Control3D, 3DStyleGLIP, SketchDream, TIP-Editor, etc.
2. Comparative methods are missing. While the authors compare a large number of text-to-3D works, which is reasonable for evaluating 3D artistic font generation capability, they should comprehensively discuss the advantages of the proposed 3D editing method.

- Key technical details are missing. For example: How are the 2D font masks decomposed? What 2D prior model is used? How are the generated artistic font images initialized into 3D Gaussians? The authors only use frontal view constraints; how is consistency ensured for other views, such as side views?

### Questions
- In Section 3.2, many details are unclear. How are 3D Gaussians obtained from the generated images? What technology is used? Text-to-artistic font image generation technologies are now quite mature, such as SeeDream4.0, Hunyuan, or ControlNet. What advantages does Glyph2Cloud have over these methods?

- In the parameterized trade-off experiment for shape and style, why is there a clear gradient in the 2D generation results but not in the 3D generation?

- What are the requirements for the input text prompts? The paper does not seem to include any complete examples of prompt phrases.

- Reference the questions raised in the Weaknesses.

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
The paper addresses the unexplored direction of structured 3D artistic font generation (3D-AFG) by proposing SplatFont3D. Using 3D Gaussian Splatting (3DGS) as the representation and guided by Score Distillation Sampling (SDS) from 2D diffusion models, the method enables the generation of 3D artistic fonts from glyphs and textual style prompts, while also supporting part-level style control.

To tackle three core challenges—point cloud initialization, the lack of real 3D font data, and part drift and entanglement during training—the paper introduces three key designs:
	1.	Glyph2Cloud: Balances shape and style in the latent space of 2D diffusion to generate stylized glyphs. Foreground extraction is performed via text-guided segmentation, followed by point sampling to obtain a 3D point cloud for initializing 3DGS.
	2.	3DGS Optimization via SDS: Optimizes 3DGS by distilling supervision from a 2D diffusion model through the SDS paradigm, enabling the generation of high-quality 3D artistic fonts.
	3.	Dynamic Component Assignment (DCA): Dynamically updates the Gaussian points’ component labels during training based on component heatmaps from stylized 2D results, alleviating part entanglement caused by Gaussian drift and enabling explicit part-level style control.

Experiments and Results: The proposed method is compared with multiple baselines (DreamFusion, Latent-NeRF, MVDream, Wonder3D, Fantasia3D, GaussianDreamerPro, GsGen, DreamFont3D, etc.) under three settings—Global, Part-Level, and Global + Part-Level. Quantitative results demonstrate clear superiority in terms of quality and consistency metrics. Ablation studies further validate the effectiveness of the Glyph2Cloud and DCA designs.

### Strengths
By designing Glyph2Cloud and DCA specifically for font—a highly structured object—the paper effectively leverages the explicit representation advantages of 3D Gaussian Splatting (3DGS), such as high rendering efficiency and structural decomposability, making them well-suited for this task.
The work addresses a clear yet underexplored need in structured 3D artistic font generation, filling an important research gap in applying 3DGS to font modeling and generation.
The experiments are comprehensive, including multiple settings (Global, Part-Level, and Combined), extensive comparisons with baselines and ablations, and diverse qualitative and quantitative evaluations, demonstrating the robustness and effectiveness of the proposed approach.

### Weaknesses
Data and Generalization:
The evaluation dataset remains limited in scale and linguistic diversity. It is recommended to extend the experiments to fonts with higher stroke density, more complex structures, and additional languages. Moreover, it would be beneficial to report the control effectiveness and computational overhead under finer-grained component segmentation settings (e.g., more than three components).

Relation to Feed-Forward 3D Generation:
Recent feed-forward 3D generation methods (e.g., Trellis, UniLat3D) can efficiently produce 3DGS-based objects from 2D images. Although these approaches primarily target general object domains, it is suggested that the authors discuss their relevance—for instance, by comparing the efficiency and quality differences between generating 3D fonts from 2D artistic glyphs (x_g) using feed-forward methods and the proposed approach.

### Questions
1. Compared to directly generating 3D fonts from stylized 2D glyphs (x_g) using feed-forward methods such as Trellis, what specific advantages does the proposed approach offer?
2. Since the Dynamic Component Assignment (DCA) module relies on 2D label maps for guidance, how accurate are the reassigned component labels when severe occlusion or part overlap occurs from certain viewpoints?
3. What is the feasibility of extending the method to finer-grained (stroke-level) style control, and how would such extension impact optimization complexity and computational overhead?

### Soundness
3

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
This paper introduces SplatFont3D, a novel framework for generating 3D artistic fonts from text prompts and 2D font image, emphasizing structure-aware synthesis and precise part-level style control. Key challenges include maintaining semantic and structural constraints of fonts, achieving fine-grained part-level stylization, and overcoming the scarcity of 3D font data. They also propose a dynamic component assignment strategy to handle Gaussian drift and enable part-level control.

### Strengths
1. **Reasonable Solutions towards the 3D fonts challenge**: They use 2D diffusion priors to initialize 3D point clouds, balancing shape preservation and stylistic fidelity through denoising interventions and segmentation. The dynamic component assignment is a smart solution to Gaussian drift, enabling explicit part decomposition superior to implicit representations like NeRF.
2. **Extensive evaluation results**: The evaluation is comprehensive, covering global and part-level scenarios across diverse characters and styles. Diverse metrics like CLIP score, Alignment, Quality, V-LPIPS, and V-CLIP. Comparisons with state-of-the-art baselines demonstrate empirical superiority.

### Weaknesses
1. **Limited Scope of Evaluation Data**: The dataset comprises only 44 characters with 2 styles and modes each (1760 pairs total), which may not fully represent the diversity of fonts or languages. While including Chinese characters adds some breadth, the focus on limited categories (e.g., fruits, foods) could bias results toward simpler styles, potentially limiting generalizability to more complex or abstract prompts.
2. **Lack of ablation studies about each component**. The paper lacks several important ablation studies to prove the effectiveness and necessity of the proposed components. For example how to prove the dynamic component assignment is necessary. Also the reviewer thinks the most critical part is the initialization module. Can the authors show how is the comparison if the initializations of all the methods are the same?
3. While the challenges remain, the qualitative results could still be improved.
4. The method avoids collecting real 3D data, which is a strength, but how does it compare to supervised fine-tuning on synthetic 3D fonts? Would incorporating even a small 3D dataset further improve performance, and if so, under what conditions?

### Questions
Please refer to Weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
2
