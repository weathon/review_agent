# FoR-SALE: Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Frame of Reference (FoR) is a fundamental concept in spatial reasoning that humans utilize to comprehend and describe space. 
With the rapid progress in Multimodal Language models, the moment has come to integrate this long-overlooked dimension into these models.  In particular, in text-to-image (T2I) generation, even state-of-the-art models exhibit a significant performance gap when spatial descriptions are provided from perspectives other than the camera. To address this limitation, we propose Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing (FoR-SALE), an extension of the Self-correcting LLM-controlled Diffusion (SLD) framework for T2I. 
For-Sale evaluates the alignment between a given text and an initially generated image, and refines the image based on the Frame of Reference specified in the spatial expressions. It employs vision modules to extract the spatial configuration of the image, while simultaneously mapping the spatial expression to a corresponding camera perspective. This unified perspective enables direct evaluation of alignment between language and vision. When misalignment is detected, the required editing operations are generated and applied. FoR-SALE applies novel latent-space operations to adjust the facing direction and depth of the generated images.
We evaluate FoR-SALE on two benchmarks specifically designed to assess spatial understanding with FoR. 
Our framework improves the performance of state-of-the-art T2I models by up to 5.3\% using only a single round of correction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a critical limitation in Text-to-Image (T2I) generation: the inability of current methods to correctly render spatial relationships described from perspectives other than the default camera view. The authors propose FoR-SALE, a self-correction framework, including visual perception modules for spatial analysis, LLM-based reasoning for perspective interpretation and layout planning, and diffusion-based operations for iterative refinement.

### Strengths
*   The investigation of spatial relations from diverse perspectives is compelling. The proposed vision perception modules and LLM-based interpretation seem reasonable, as they provide 3D prior information.
*   The experimental results show the method's efficacy across multiple T2I baselines. Furthermore, the proposed progressive refinement strategy can further enhance the generated spatial accuracy.

### Weaknesses
*   The core components of the self-correction strategy and the LLM-based parser appear to build directly upon established paradigms [1], which may limit the methodological novelty. Furthermore, could current Spatial-LLMs [2,3] be potentially applied for enhanced spatial reasoning in this framework? If so, more discussion and comparison with these approaches should be included.

*   It remains unclear whether the proposed method can effectively handle complex scenes involving multiple objects with intricate relations or cases where objects overlap. If such scenarios significantly impact the method's performance, this should be acknowledged and discussed as part of the limitations.

*   The computational efficiency of the approach requires further clarification. What is the specific time overhead of the proposed framework? Does multi-round refinement lead to substantially increased inference time? 

References

[1] L. Lian et al., "LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models," arXiv, 2023.

[2] J. Yang et al., "Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces,"  CVPR, 2025.

[3] Y. Mao et al., "SpatialLM: Training Large Language Models for Structured Indoor Modeling," arXiv, 2025.

### Questions
Please address my concerns in the **Weaknesses** section. I would change my rating if my concerns could be addressed.

### Soundness
3

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
3

### Summary
This paper introduces FoR-SALE, a framework for improving spatial consistency in text-to-image generation when the textual description uses a non-camera frame of reference (FoR). Built upon the SLD pipeline, FoR-SALE integrates a rule-based FoR-Interpreter for linguistic normalization, orientation, and depth estimation for visual grounding, and two latent editing modules for iterative correction. Experiments on two small benchmarks (FoR-LMD and FoREST) show consistent but moderate performance improvements.

### Strengths
The paper targets a clear and practically relevant problem that has received little prior attention: spatial reasoning under varying frames of reference.
1. The paper frames a well-motivated problem clearly and proposes a reasonable solution path.

2. The pipeline demonstrates robust engineering design, combining rule-based linguistic grounding with structured latent-space edits.

3. The presentation is clear, figures are intuitive, and the authors provide sufficient implementation details to facilitate reproducibility.

### Weaknesses
1. This method is largely an integration of existing components, such as SLD, SAM, ControlNet, and depth/orientation estimators.

2. The reported gains, though consistent, are relatively modest and require multiple editing rounds.

3. The FoR-Interpreter relies on 32 handcrafted rules, limiting generalization to complex or ambiguous spatial relations, and the benchmarks used are small and partly synthetic.

4. The ablation results suggest that genuine 3D structural correction is still only partially achieved.

### Questions
1. How can the 32-rule FoR-Interpreter scale to more complex or compositional spatial relations?

2. Have you considered replacing the rule-based mapping with a learnable FoR normalization module for better generalization?

3. Please include a detailed efficiency analysis to clarify the computational cost and practicality of the proposed iterative pipeline.

4. See in W4.

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
This paper tackles a significant limitation in modern Text-to-Image (T2I) models: their poor understanding of spatial relations described from a non-camera (intrinsic) Frame of Reference (FoR). While models can often place objects correctly relative to the camera (e.g., "a horse to the left of a cow"), they fail when the perspective is object-centric (e.g., "a horse to the left of a cow *from the cow's perspective*").

To address this, the authors propose **FoR-SALE** (Frame of Reference-guided Spatial Adjustment in LLM-based Diffusion Editing), a framework that extends the Self-correcting LLM-controlled Diffusion (SLD) pipeline.

### Strengths
- The paper clearly identifies and addresses the failure of SOTA T2I models on non-camera (intrinsic) Frames of Reference, a key component of spatial reasoning.
- The FoR-SALE framework's design is logical. The **FoR Interpreter** is a particularly strong contribution, as it cleverly simplifies the reasoning problem by translating intrinsic prompts into a canonical camera-centric view *before* layout planning.
- The paper introduces `Facing Direction Modification` and `Depth Modification`, two new latent-space operations that are necessary for correcting the 3D spatial errors (orientation, depth) inherent in FoR tasks.

### Weaknesses
- The framework's performance is heavily capped by the quality of its sub-modules, which the paper acknowledges. The failure analysis (Appendix F) shows that "Incorrect orientation generation" (28.33%) and "Object Detection failure" (18.33%) are the dominant error sources. The framework can't fix what it can't "see" or what the underlying diffusion model can't "draw" correctly.
- The ablation study (RQ4, Table 3) shows that removing the `Depth Modification` operation results in only a minor 0.3% accuracy drop. This suggests the current implementation (using ControlNet) has a very limited positive impact, and correcting 3D depth remains an unsolved challenge.
- The FoR-Interpreter relies on 32 manually-defined "perspective conversion rules" (Appendix G). This approach, while effective for the 4 relations (L/R/F/B) and 8 directions tested, seems brittle. It's unclear how this would scale to more complex spatial prepositions (e.g., "between," "alongside," "diagonally behind") or more nuanced descriptions without a combinatorial explosion of rules.
- The evaluation is entirely automated. While the authors state the protocol is based on prior work aligned with human judgment, the complexity of spatial relations (especially FoR) can be subtle. A small-scale human study to validate that the automated "correct" images are indeed perceptually correct would strengthen the paper's claims.

### Questions
1. The ablation study (RQ4) shows that `Depth Modification` has a minimal (0.3%) impact on performance. Why is this contribution so small? Is it a limitation of the underlying ControlNet-based operation, or do you believe depth is simply less critical than orientation for the spatial relations in your benchmarks?
2. he FoR-Interpreter's 32 manual rules are a core component. How do you see this approach scaling to more complex spatial language, such as relations involving three or more objects (e.g., "A is between B and C") or more nuanced prepositions ("A is just to the left of B")? Does this manual-rule approach present a fundamental bottleneck?
3. Your failure analysis (Appendix F) identifies "Incorrect orientation generation" (28.33%) as the single largest source of error after one round of correction. Since your `Facing Direction Modification` operation directly targets this, could you provide more insight into why this operation fails so often? Is the issue with the DiffEdit model used, or is it a problem of composing the new latent representation back into the scene?

### Soundness
4

### Presentation
4

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
This paper focuses on the reasoning deficiencies of multimodal large models (MLLMs) in handling Frames of Reference (FoR), particularly in text-to-image (T2I) generation, where spatial misalignment frequently occurs when the described scene adopts a non-camera viewpoint. To address this issue, the authors propose FoR-SALE, a self-correction approach that explicitly models FoR within an LLM-guided diffusion editing framework. The method performs edits in the latent space of diffusion models and supports multi-round iterative refinement. The authors evaluate their approach on the FoR-LMD and FoREST benchmarks, achieving state-of-the-art performance.

### Strengths
1. The FoR plays a crucial role in spatial reasoning of MLLMs, yet it has not been widely explored in prior research.
2.The authors conducted various analyses in the experimental section, such as error analysis,which enhances the rigor of the paper.

### Weaknesses
1. In Table 1, the performance of GraPE and SLD declines compared to the vanilla model. Does this indicate that these compared methods are relatively weak?
 2. In Line 313, The authors seem to have reused the image understanding component of their method in the evaluation criteria. Could this introduce an evaluation bias and ultimately lead to higher scores for their proposed approach?
 3. Although I am very satisfied with the series of analyses conducted by the authors, is there a lack of additional generation latency metrics to assess whether the improvements are worthwhile? After all, image generation time is also an important factor to consider.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2
