# Agentic 3D Scene Generation with Spatially Contextualized VLMs

- Decision: Reject
- Scores: 4, 4, 6, 4, 4

## Abstract
Vision-language models (VLMs) have made rapid progress in multimodal generation, but extending them to structured 3D scene construction requires addressing three key challenges: (1) unifying diverse inputs into a semantic representation that captures global layout, environment setup, object-level appearance, and constraints for 3D object reconstruction, (2) encoding object-object and object-environment interactions, and (3) enabling accurate control over object placement, appearance, and reconstruction. We introduce an agentic framework for 3D world generation that tackles these challenges through a spatially contextualized design. A scene portrait integrates multimodal inputs into a semantic blueprint with sub-descriptions and reconstruction constraints, consisting of text, image appearance, and partial 3D point clouds. A scene hypergraph encodes rich spatial relations, capturing both object-object interactions and object-environment interactions that guide object placement. Finally, geometric reconstruction with ergonomic adjustment delivers accurate 3D reconstruction while refining object shape, scale, and placement through optimization. These components are supported by an auto-verification agent that ensures the generated reconstruction satisfies the input constraints. Together, they form a structured spatial context that the VLM iteratively reads and updates, enabling coherent, editable, and semantically aligned 3D environments. Experiments demonstrate strong generalization across diverse inputs and show that spatial context injection empowers VLMs with downstream capabilities such as interactive scene editing and path planning, advancing spatially intelligent systems in graphics and 3D vision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an agentic 3D scene generation framework that augments Vision-Language Models (VLMs) with a structured spatial context, enabling multimodal-to-3D generation that is both semantically coherent and geometrically plausible.

The framework introduces two core components:
1. Scene Portrait: A multimodal semantic blueprint integrating text, image, and geometry.
2. Scene Hypergraph: A relational structure encoding unary, binary, and higher-order spatial relations such as contact, alignment, and symmetry.

Injected with this structured context, the VLM performs iterative, agentic operations. Applications include asset synthesis, environment setup, layout optimization, and ergonomic adjustment, while auto-verification mechanism continuously evaluates and corrects inconsistencies.

### Strengths
1. Originality: Introduces a well-structured spatial context for VLMs, unifying symbolic reasoning and 3D geometry.

2. Clarity: Clear methodology, figures, and well-structured narrative; high presentation quality.

3. Significance: Advances the state of the art in spatially grounded 3D generation and suggests a foundation for agentic scene understanding.

### Weaknesses
1. Limited Structural Granularity: The reliance on a scene-graph-based representation may constrain the method's ability to achieve finer-grained or part-level control. Extending beyond object-level relations toward detailed component or articulation modeling could further push the boundary of structured generation.

2. Unclear Novelty over Prior Work: The paper should clarify the main differences an concrete contributions relative to prior scene-graph-based and layout-driven approaches. The proposed scene portrait and scene hypergraph abstractions, while well-defined, appear conceptually similar to structures used in earlier scene representation works, and their novelty is not fully evident.

3. Scalability Concerns: As acknowledged by the authors, the framework may struggle when handling dense scenes containing numerous small or heavily occluded objects, where the hypergraph construction and relation inference become less reliable. More discussion on scalability or potential solutions would strengthen the work.

4. Incomplete Ablations: The ablation studies could be expanded, demonstrating the respective contributions of each component to overall generation quality.

### Questions
1. Improving 3D Understanding in VLMs. More as a discussion-oriented curiosity: beyond introducing structural-level context (e.g., scene graphs or hypergraphs), how might we fundamentally enhance the 3D reasoning capability of vision-language models? Would training or pretraining on large-scale 3D datasets rather than relying on 2D-projected structures, offer a more direct route toward genuine spatial understanding?

2. Handling Open-Vocabulary or Abstract Inputs. How does the proposed system deal with open-ended or highly abstract text prompts (e.g., "a dreamlike city of glass") where semantic grounding is inherently ambiguous? Is there any mechanism to disambiguate such concepts or to gracefully degrade toward plausible visual interpretations?

### Soundness
2

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
3

### Summary
This paper proposes spatially contextualized vision–language models (VLMs) that act as agents for structured 3D scene generation. The key idea is to maintain a continually updatable spatial context that guides the model’s understanding and generation of 3D environments.

### Strengths
1. The method can generate high quality result from somehow ambiguous text or image prompt.
2. The method outperform prior works in terms of diversity and out-door scene generation.
3. It can generate scenes with some details and the object placement seems to be physically plausible (due to the hypergraph constraints).

### Weaknesses
1. Fundamentally, it still rely on VLM's ability and depth estimation foundation models' ability for the overall scene quality. It might sometimes not enough. 
2. Is there a solver to do "Hypergraph-based ergonomic adjustment"? It is unclear of the efficiency of this part (time-cost) and of the whole pipeline.
3. It's unclear that how is the background generated, like the water body and the walls. Can you provide some full scene graphs and scene portraits for generated scenes?
4. It's not so meaningful to compare with prior indoor scene generation works in out door scenes. Also, it seems that you are using different renderer for Holodeck and your method. Which brings up concerns about the fairness of the comparison in Table 1.
5. Need more qualitative results to demonstrate the robustness of this fully automatic pipeline.

### Questions
See Weakness.

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
3

### Summary
This paper presents a novel framework for agentic 3D scene generation by introducing spatially contextualized Vision-Language Model. The method enables structured and coherent 3D scene generation from diverse multimodal inputs such as text prompts, single images, by equipping VLMs with a dynamic, updatable spatial context. This context consists of two key components: a scene portrait, which captures high-level semantic blueprints including layout, object descriptions, visual references, and initial geometric grounding; and a scene hypergraph, which explicitly models unary, binary, and higher-order spatial relations among objects and environmental elements.

The designed pipeline operates in an agentic manner. the VLM reads and updates the spatial context throughout the generation process, performing iterative steps such as asset synthesis, coarse alignment via point cloud fitting, environment setup, layout optimization, and ergonomic adjustment using relation-specific differentiable losses.  A closed-loop auto-verification mechanism further ensures fidelity to both semantic constraints and geometric plausibility by continuously monitoring the evolving scene and triggering corrections when discrepancies are detected.

Key contributions include:
- The formulation of spatially contextualized VLMs that act as agents for structured 3D scene generation through a continually maintained and updated spatial memory.
- The design of a dual-component spatial context (scene portrait + scene hypergraph) that supports multimodal integration and relational reasoning for layout planning and ergonomic refinement.
- An agentic generation workflow combining multiple stages—from asset creation to layout optimization—with built-in feedback via auto-verification.
- Demonstrations of strong generalization across input modalities, producing semantically aligned, editable, and physically plausible 3D scenes, while enabling downstream tasks such as interactive editing and path planning.

Experiments show favorable performance compared to state-of-the-art methods in terms of consistency, aesthetic quality, and geometric plausibility, particularly excelling in preserving spatial structure and style from reference images.

### Strengths
The core idea of augmenting VLMs with a structured and persistent spatial context is conceptually compelling. This context comprises two key component: the scene portrait and the scene hypergrap, which jointly capture both the semantic content and spatial relationships within a scene in a coherent and interpretable manner. The integration of partial point clouds as geometric priors for individual assets provides a principled mechanism for grounding abstract multimodal inputs into 3D space, effectively bridging the gap between perceptual understanding and generative modeling. Furthermore, the agent-based design, featuring iterative layout optimization and an auto-verification mechanism, ensures that the generated scenes evolve in a semantically consistent and geometrically plausible manner, thereby mitigating severe spatial or functional irrationalities.

### Weaknesses
## Problems on Writing:
1. The authors state in Section 4.2: *Due to space limitations, we provide only the ablation study on environment setup in the main paper;additional ablations are included in the supplementary material.* However, the follow-up content includes the ablation study on environment setup and layout planning.
2. Table 1 reports AQ (aesthetic quality) and FP (functional plausibility) under columns labeled “(4o/User)(↑)”, suggesting scores from both GPT-4o and human users. However, the paper does not specify whether these values represent ordinal rankings or normalized scores. Given that lower numerical values indicate better performance (e.g., Ours: 1.00), they appear to be relative ranks—but this must be explicitly stated. Without clarification, quantitative comparisons are ambiguous and potentially misleading.

## Over-Reliance on External Generators Introduces Confounding Factors
The framework heavily depends on external APIs (e.g., Meshy.ai) for 3D asset generation. While practical, this reliance means that aspects of appearance fidelity, texture quality, and stylistic consistency are largely determined by third-party models rather than the proposed method itself. As a result, it becomes difficult to disentangle the contribution of the spatial context mechanism from the inherent capabilities of the black-box generator. For instance, if the final scene lacks style consistency with a reference image, is this due to failure in context modeling or limitations of the mesh generator?

## Insufficient Ablation Study
The paper lacks ablation studies directly evaluating the proposed spatial context. In the ablation study, the effect of the overall replacement with other methods was compared only by graphical comparison. It is unclear how essential each component is to the final performance. This weakens the argument for the necessity of the full spatial context design.

## Limited Discussion of Scalability and Failure Cases
The method assumes accurate Initial point cloud reconstruction and segmentation via existing methods. However, performance likely degrades in cluttered scenes or with small/occluded objects, as briefly acknowledged in the appendix (Limitation A). Yet, there is no empirical analysis of how robust the pipeline is under such conditions. Including failure case such as misaligned furniture due to noisy depth estimation or incorrect hyperedges would strengthen the paper’s credibility and guide future improvements.

### Questions
**1. Ablation Study Discrepancy:**  
The paper claims only the environment setup ablation is included due to space limits, yet Figure 9 presents a full layout planning ablation. Please clarify this inconsistency and confirm whether additional ablations should exist in the supplement.

**2. Metric Definition:**  
Table 1 reports AQ and FP scores (e.g., 1.00 for Ours). Please explicitly define the scoring protocol.

**3. Component Contribution:**  
Can you provide evidence isolating the impact of the spatial context (e.g., w/o hypergraph or portrait)?

**4. Failure Cases:**  
Could you share one or two representative failure cases to better illustrate current limitations?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a pipeline that integrates Vision-Language Models (VLMs) with 3D scene generation technology. By leveraging multimodal inputs, such as images, text, and unstructured image collections, it enables the creation of 3D scenes that align with textual semantics or image content. Additionally, the paper presents an automatic verification mechanism that utilizes VLMs for point cloud restoration throughout the scene generation process. The paper is well-motivated, logically organized, and effectively demonstrates 3D reconstruction results.

### Strengths
1. The paper proposes using a structured Spatial Context to provide detailed descriptions of the environment to be generated. This structured data serves as input to the VLM, guiding the 3D scene generation process.
2. During the 3D scene generation, the VLM continuously reads from the Spatial Context, ensuring that the scene remains semantically coherent and geometrically accurate.

### Weaknesses
1. In the initialization stage, the pipeline employs Fast3R to construct an initial 3D scene from an image. However, the generated scene contains occlusion relationships between instance assets and the background. Although the paper mentions repairing and generating the invisible regions of the assets, no corresponding repair is conducted for the background. How is the reconstruction completeness of the occluded background areas ensured? According to the supplementary videos, this method can produce a complete scene, especially in cases with only text input. Nevertheless, it remains unclear how the scenes and assets observed from novel viewpoints are generated and aligned with the initialized scene.
2. During instance asset generation, the pipeline first evaluates the completeness of the initialized asset point clouds, but does not present the specific evaluation algorithm or its implementation details. After repairing the assets into complete point clouds, they render front-view images of the assets and combine them with textual descriptions to jointly guide 3D asset generation. In this multimodal guidance process, is priority given to image consistency or semantic consistency? Furthermore, does the incorporation of textual descriptions introduce inconsistencies between the generated 3D assets and the rendered views?
3. In the quantitative comparison experiments, the paper does not disclose the specific settings of the experimental data. For instance, how many scenes were generated in the experiments? How were these scenes rendered, and how many images were rendered to compute metrics such as CLIP?
4. The authors also perform ablation studies on the environment setup, but the paper lacks explanations regarding the specific environmental configuration methods or experimental details. In the supplementary video, the water flow in the environment is dynamic; however, according to the paper, the proposed method does not support the generation of dynamic point cloud data. How is this dynamic scene achieved?
5. In the qualitative experiments and supplementary materials, the authors present only a limited number of results. It would be desirable to include more qualitative examples and report what is the success rate of the proposed method.
6. In Section 3.2, the authors claim that “our framework performs agentic scene generation, where the VLM synthesizes 3D assets.” Given that the output of a VLM is typically textual, the authors should clearly explain the process of asset synthesis.

### Questions
The pipeline utilizes a Vision-Language Model (VLM) agent to facilitate 3D scene generation and supports multiple input modalities (text, single image, and image group). However, it does not specify the experimental protocol adopted when using an image set as input. Based on the qualitative examples in Figure 8, this image set appears to include both asset and scene images. If multi-view scene images are used instead, would the experimental protocol differ?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a framework for agentic 3D scene generation by augmenting vision-language models (VLMs) with a spatial context composed of two key components: a scene portrait (semantic layout blueprint) and a scene hypergraph (object–object and object–environment relations). The method enables a VLM (GPT-4o) to act as an autonomous agent that iteratively constructs, verifies, and refines 3D environments from text, images, or unstructured image sets. The system performs asset synthesis, layout planning, ergonomic adjustment, and environment setup, while an auto-verification module ensures geometric and semantic consistency. Experiments demonstrate strong performance compared to Holodeck, DreamScene, and ACDC, achieving better alignment and plausibility across text-, image-, and multi-view inputs.

### Strengths
1. the paper’s notion of “spatially contextualized VLMs” is original and well-motivated. It reinterprets VLMs as reasoning agents operating over structured 3D contexts.
2. The ability to handle both single images and unstructured image collections is impressive and practically relevant.

### Weaknesses
1. reported metrics (CLIP, BLIP, LPIPS) evaluate rendered 2D projections. There is no quantitative evaluation of 3D accuracy, geometry consistency, or spatial relation correctness—critical aspects for a 3D generation paper.
2. while conceptually coherent, the pipeline involves multiple submodules (Fast3R, Point-M2AE, Meshy, Blender), which could make it hard to scale or analyze systematically.
3. while the conceptual framing of “spatially contextualized VLMs” is interesting, many subcomponents: scene graphs, point-cloud restoration, and ergonomic layout optimization, build directly on existing methods (e.g., ATISS, Point-M2AE, Fast3R). The integration is thoughtful but primarily a system-level composition rather than a new algorithmic breakthrough.
4. the paper does not clearly define what datasets or prompts were used for quantitative evaluation or how they were standardized across methods. Without standardized benchmarks, numerical comparisons (especially involving user studies) may be subjective.

### Questions
1. do you provide an object level promp (an electric guitar), or you extract it from the main promps? can you do better when given more detailed object level prompt?
2. How does StructMap handle self-occlusion when rendered to 2D?
3 the first image can be generated using the prompt? if there are any missing object between the prompt and the generated image, will your approach work?

### Soundness
3

### Presentation
2

### Contribution
2
