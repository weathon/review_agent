# Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Generating interactive 3D scenes from text requires not only synthesizing assets but arranging them with spatial intelligence—support, affordances, and plausibility. However, training data for interactive scenes is dominated by a few indoor datasets, so learning-based methods overfit to in-distribution layouts and struggle to compose diverse arrangements (e.g., outdoor settings and small-on-large relations). Meanwhile, LLM-based layout planners can propose diverse arrangements, but the lack of visual grounding often yields implausible placements that violate commonsense physics. We propose Scenethesis, a training-free, agentic framework that couples LLM-based scene planning with vision-guided layout refinement. Given a text prompt, Scenethesis first drafts a coarse layout with an LLM; a vision module refines the layout and extracts scene structure to capture inter-object relations. A novel optimization stage enforces pose alignment and physical plausibility, and a final judge verifies spatial coherence and triggers targeted repair when needed. Across indoor and outdoor prompts,  Scenethesis produces  realistic, relation-rich, and physically plausible 3D interactive scenes, reducing collisions and stability failures compared to SOTA methods, making it practical for virtual content creation, simulation, and embodied AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Scenethesis, a framework that couples LLM-based scene planning with vision-guided spatial refinement and physics-aware optimization for the task of 3D scene generation. Its physics-aware optimization for mesh-level contact and stability improves physical plausibility.

Scenethesis first drafts a coarse layout via an LLM; a vision module then produces a guidance image, constructs a scene graph with 3D bounding boxes, and retrieves assets. A physics-aware optimization module refines object poses using semantic correspondences and SDF-based contact/support constraints, and a final scene judge verifies spatial consistency.

### Strengths
1) The pipeline couples LLM-based scene planning with vision-guided spatial refinement to capture real-world spatial complexity.
2) The paper introduces a physics-based optimization that further adjusts asset placement via semantic feature matching and SDF-constraints, which is novel and completes the full scene-generation cycle.
3) The quantitative and qualitative results are strong.
4) The paper is written in an easy-to-follow manner.

### Weaknesses
1) Missing results: Table 1 lacks the spatial quality preference results for Scenethesis.
2) Comparison with baselines: 

     a. L395-397: SceneTeller and LayoutGPT are training-free in-context learning methods.

     b. Comparing against these methods on 3D-FRONT, which covers primarily bedrooms and living rooms with corresponding assets, is not fair, since many prompt objects are absent. In these baselines, retrieval is typically constrained by predicted category and then nearest dimensions (3DBBs). It is unclear how missing objects in 3D-FRONT are handled in Table 1’s qualitative examples (e.g., retrieving a “double bed” in place of a “piano,” or what substitutes for a “wheelchair”).

     c. To fairly compare with training-free in-context learning methods, the baselines should also be evaluated using Objaverse. As not having the right assets could significantly affect metrics like CLIP score. A small set of sample layouts can be curated using Objaverse and provided to the baselines as in-context samples during evaluation.

3) Lack of comparison with recent baselines (e.g., LayoutVLM [1])

[1] Sun, F. Y., Liu, W., Gu, S., Lim, D., Bhat, G., Tombari, F., ... & Wu, J. (2025). Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 29469-29478).

### Questions
All of my questions are listed in the weaknesses section, and I may adjust the rating if they are well addressed.

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
This paper proposes Scenethesis, an agentic framework that can generate 3D interactive scenes from language inputs. Scenethesis is a multi-stage framework that first leverages language models to plan the overall design and potential objects, which will be used for generating a 2D image as guidance. Based on the image guidance, the vision module will generate the scene graph and retrieve 3D assets.  Then, the optimization module will adjust the placement of objects to ensure pose alignment and physical plausibility. Finally, the judge module uses GPT-5 to evaluate the quality of the generated scene. The experimental results demonstrate Scenethesis outperforms previous methods both qualitatively and quantitatively.

### Strengths
1. The proposed approach is technically sound, and the qualitative examples clearly show that Scenethesis can generate more realistic scenes than previous methods, especially for outdoor scenes and small objects placement.
2. The ablation studies show that each component in spatial and physical constrain can improve the performance.

### Weaknesses
1. My major concern is the lack of evidence on downstream applications. Although this work demonstrates better realism and diversity in generating scenes, it is unclear whether such improvement can transfer to downstream tasks like embodied AI, robot navigation/manipulation, or enhance the spatial understanding of vision-language models, etc. Including such downstream experiments will enhance this paper's contributions and validate the effectiveness of the generated scenes.
2. The rendering qualities vary a lot across baselines and the proposed methods, which may affect human judgments. Controlling this variance can help us understand whether the improvement comes from better layout or just higher fidelity.

### Questions
1. How often does the Judge Module get triggered, and how much performance gain can it bring? How long will the re-planning step take, and will it significantly affect efficiency?
2. Have you quantified the benefits of the environment map? This may affect the CLIP/BLIP scores a lot. It is better to do some ablations to disentangle the contributions of layout, assets, and map.

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
This paper proposes Scenethesis, an agentic framework for generating physically consistent, interactive 3D scenes from textual prompts.

The authors design the following procedures to achieve a physically realistic 3D scene:
- Coarse scene planning: using an LLM that first generates a rough layout plan based on the text prompt, objects, and spatial relationships.
- Layout visual refinement: using Image-guidance, Sceen-Graph generation and asset retrieval for to handle the uncommon co-occurrence and physical inconsistencies
- Optimization module: First align the pose and then refine(optimize) the pose with self-defined physical constrraints.
- Judgment with VLMs, and then decide whether to replan or not.

In summary, the framework is training-free, using different existing and pretrained models for a physically plausible 3D scene generation.

### Strengths
- The paper pipeline is intuitive, reasonable, and easy to understand. The whole paper is well-written in presenting the methodology.
- The visualization results are realistic and visually plausible
- The quantitative results seem to be much better

### Weaknesses
- Some failure cases can better show the limitations of this work,
- This work seems to be relying heavily on the existing modules and database. The contribution may be limited. For example, both [1], [2], and [3] mentioned similar physical constraints. Prior works also used LLMs as planners and judges.

[1]: Cast: Component-aligned 3d scene reconstruction from an rgb image
[2]: WorldCraft: Photo-realistic 3D world creation and customization via LLM agents
[3]: Layout-your-3d: Controllable and precise 3d generation with 2d blueprint

### Questions
- I am wondering why the visualization results are much better than previous works. In this work, there are certain illumination conditions during rendering, while in previous works, they are weak. Though the layouts of this work is better, are there any other rendering setings that make the visualization better?
- How much time does it take to synthesize one 3D scene?
- What is the success rate of this work? Since the integration of different modules may result in an accumulative error. Some of the modules are not that stable, e.g., Image generation and Grounded SAM.
- When the LLM plans the layout, does it explicitly model hierarchical or relational structures (e.g., “a cup on a table next to a book”)? How does it ensure that such relations are spatially consistent?
- In my understanding, this work relies on the generated image, can this work support some methods to extrapolate the range of the scene?

### Soundness
3

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
3

### Summary
This paper presents Scenethesis, a training-free, agentic framework for text-to-3D scene generation that combines an LLM-based planner with vision-based layout refinement and a physics-aware optimization loop. The system decomposes the task into four modules: (1) coarse scene planning using an LLM, (2) layout visual refinement using visual foundation models (e.g., Grounded-SAM, DepthPro), (3) physics-aware optimization enforcing SDF-based collision and stability constraints, and (4) a GPT-5-based scene judge for coherence verification and targeted repair. Scenethesis outperforms several text-to-3D baselines such as DiffuScene, LayoutGPT, Holodeck, and SceneTeller on metrics of layout realism, physical plausibility, and spatial coherence, across both indoor and outdoor prompts.

### Strengths
1. physics-aware optimization, the use of SDF constraints for collision avoidance and stability is more principled than bounding-box–based heuristics in prior work (e.g., LayoutGPT, Holodeck).
2. generality, unlike most prior works limited to indoor scenes, Scenethesis generalizes to outdoor prompts and long-tail object relations (on-top-of, inside, behind).
3. despite not requiring new model training, Scenethesis achieves or exceeds the quality of trained baselines, highlighting the power of coordination across existing large models.

### Weaknesses
1. most image generation has problems with creating several objects, but i cant find how did you challenge it?
2. many qualitative examples lack strict spatial or functional order. For instance, in “A living room with many reading materials”, books are scattered arbitrarily on the floor — any random placement would satisfy the prompt. In another case, the bookshelf is positioned behind the sofa, making it unreachable and violating functional affordance. This suggests that while Scenethesis achieves physical plausibility, its understanding of semantic utility and human-centered spatial reasoning remains limited.
3. several components (LLM-based coarse planning, VLM grounding, Collision avoidance) closely follow prior work like Holodeck, Lay-A-Scene and LayoutGPT. The main new element, in-loop SDF-based optimization, while effective, is an incremental rather than novel.

### Questions
you might want to cite "Lay-A-Scene" who also made a collision free and many aspects as your paper
@article{rahamim2024lay,
  title={Lay-a-scene: Personalized 3d object arrangement using text-to-image priors},
  author={Rahamim, Ohad and Segev, Hilit and Achituve, Idan and Atzmon, Yuval and Kasten, Yoni and Chechik, Gal},
  journal={arXiv preprint arXiv:2406.00687},
  year={2024}
}

1. How many judge repair iterations are typically performed, and is convergence guaranteed?

### Soundness
3

### Presentation
2

### Contribution
2
