# CHOrD: Synthesizing Spatially Coherent, House-Scale, Organized, and Diverse 3D Indoor Scenes via Image-Based Layout Guidance

- Decision: Reject
- Scores: 8, 2, 4, 2

## Abstract
We introduce CHOrD, a generative framework for synthesizing spatially coherent, house-scale, hierarchically organized, and diverse 3D indoor scenes. At the core of CHOrD is a two-stage generation paradigm: given a floor plan, CHOrD first synthesizes an intermediate, image-based 2D layout representation, which is subsequently transformed into a graph-based scene structure. In contrast to existing tabular-based or LLM-based generative models, the enhanced spatial capabilities of CHOrD substantially reduce long-standing artifacts frequently observed in prior work—such as physically implausible collisions, out-of-bound objects, inconsistent orientations, and incomplete layouts missing essential object placements. Furthermore, unlike existing methods, CHOrD can be conditioned on complex, irregular room shapes and is robust in synthesizing house-wide layouts that adhere to both geometric and semantic floor plan structures. We also introduce a novel layout dataset with expanded coverage of object categories and room configurations, as well as significantly improved data quality. CHOrD achieves state-of-the-art performance on both the 3D-FRONT dataset and our proposed dataset, excelling in spatial coherence, quality, and diversity, without relying on collision detection, iterative re-generation for self-correction, or predefined rules.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduce CHOrD, a two-stage generative framework for indoor house-level scene synthesis and digital twin creation. The main novel contribution of this work stems from the fact that instead of directly predicting a 3D scene graph or object list, CHOrD first generates a 2D image-based furnished scene layout conditioned on a given floor plan image using a diffusion-based image-to-image model. This output is then parsed by a fine-tuned YOLOv8 object detector and segmenter to extract a hierarchical scene graph. 3D meshes are retrieved and rendered according the scene graph to produce photorealistic, simulation-ready environments.

The core insight of the authors is that an intermediate 2D layout produced by image encoders and decoders with strong spatial priors enhances spatial reasoning, allowing CHOrD to effectively avoid common artifacts such as object collisions, misalignment, and out-of-bound placements—without the need for costly post-processing of collision checks or iterative self-correction. In addition, CHOrD can support multi-level autoregressive layout generation, enabling fine-grained spatial composition, e.g., objects on tables, and multi-modal conditioning, e.g., text-guided and open-plan floor planning.

The authors also present a new CHOrD dataset containing 9,706 clean, fully furnished scenes covering 26 furniture super-categories, including kitchens, bathrooms, and balconies - areas underrepresented in prior datasets. The quantitative results provided in the paper on both 3D-FRONT and CHOrD datasets show state-of-the-art performance compared to baselines, while qualitative comparisons confirm CHOrD’s spatial coherence and robustness to irregular room shapes.

### Strengths
The paper's core innovation is the introduction of an image-based intermediate layout representation as a key insight, which significantly enhances spatial reasoning and coherence. This approach to scene graph generation effectively reduces spatial artifacts, while being robust to implementation in various other pipelines, enabling streamlined adoption to the industry. 

The paper’s central claim—the use of a generative model to produce an image-based intermediary representation leveraging strong spatial priors—is clearly articulated and well supported. The intuition behind this design choice is effectively explained and further substantiated through both experimental analysis and qualitative comparisons to baseline methods. The authors provide quantitative results across multiple datasets and evaluation metrics (FID, KID, POR, PIoU), demonstrating consistent superiority over baselines. 

The paper presents a clear motivation and problem definition, addressing the challenge of spatial incoherence in 3D indoor scene synthesis. Its core insight—introducing a 2D image-based layout as an intermediate representation— enhances spatial reasoning and coherence. CHOrD is fully data-driven, avoiding handcrafted rules, collision detection, or iterative regeneration. Moreover, CHOrD supports hierarchical and fine-grained layout generation, enabling realistic multi-level spatial relationships. 

The authors also contribute a high-quality dataset that expands room and object coverage beyond 3D-FRONT, and provide a comprehensive evaluation demonstrating consistent state-of-the-art performance across multiple metrics and datasets. 

Finally, the model shows robustness to out-of-distribution spatial artifacts, supported by both theoretical justification and empirical validation.

### Weaknesses
The paper would benefit from comparisons to a broader range of benchmarks. CHOrD is effectively compared only to InstructScene and DiffuScene, with the comparison to PhyScene being quite limited. Although other baselines did not release training code, it would still strengthen the paper to include reported metrics from those works, even with appropriate caveats, to better contextualize CHOrD’s performance within the broader literature.

The paper lacks ablation studies. For instance, the necessity and impact of training the diffusion model are not analyzed or discussed, nor is the contribution of the segmentation component examined in detail.

The discussion of CHOrD’s limitations is somewhat superficial. The section primarily lists unaddressed directions—such as stylistic control or segmentation accuracy—but does not critically analyze inherent limitations of the proposed method itself, nor does it present or reflect on any observed failure cases.

Minor issues:

Some phrases are repeated unnecessarily — for instance, the example “such as placing objects on a coffee table” appears three times throughout the paper.

Figure 4: The highlighted squares do not accurately correspond to the visible regions, and it would be helpful to indicate the camera orientations. Additionally, the ordering of the images on the right side seems arbitrary and would benefit from a clearer logical structure.

Section 3.2 (Fine-grained Layout): The paragraphs might flow better if presented in reverse order.

Tables 1 and 2: The method names should be consistent across both tables—currently, one lists reference names while the other uses method names. Moreover, Table 2 is missing a “Method” column title, which should be added for clarity.

Section 3.1: the sentence "In 2D images, implausible spatial artifacts are instantly
visible and flagged as OOD samples, enabling the model to generate coherent, realistic layouts." requires further explanation or citation.

### Questions
Have you evaluated how much each component (e.g., diffusion model, YOLO segmentation) contributes to performance? How was the amount of training determined?

Were any stability or convergence issues observed during diffusion training?

Have you evaluated the performance of CHOrD across datasets (e.g., trained on 3D-FRONT and tested on CHOrD or vice-verse) to validate robustness of training?

What types of room layouts or object arrangements provide a challenge for CHOrD?

How does CHOrD handle non-orthogonal geometries such as circular rooms? Some examples on this would be good.
Additionally, is CHOrD able to position furniture such that it is not aligned with any of the walls?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces CHOrD, a generative framework for synthesizing spatially coherent, house-scale, and hierarchically organized 3D indoor scenes. The core innovation is a two-stage process that first generates an intermediate 2D image-based layout representation from a floor plan, which is then converted into a scene graph. This approach leverages the spatial reasoning capabilities of image-based models to mitigate common artifacts like collisions and incomplete layouts observed in prior tabular or LLM-based methods. The authors also introduce a new, high-quality dataset, the CHOrD dataset.

### Strengths
- Unlike many prior methods restricted to simplistic rectangular rooms or single-room layouts, CHOrD can handle house-scale layouts with complex, irregular room shapes and floor plan structures.

- The paper introduces a large, clean dataset (9,706 scenes, $\approx 1.4\times$ larger than 3D-FRONT). This dataset offers expanded coverage (including fully furnished kitchens, bathrooms, and balconies) and is artifact-free.

- CHOrD achieves superior quantitative results across all key metrics (FID, KID, POR, PIoU) on both the 3D-FRONT and the proposed CHOrD datasets, demonstrating its ability to generate high-quality, diverse, and coherent layouts.

### Weaknesses
- The pipeline's second stage relies on fine-tuned YOLOv8 to detect objects and extract the structured scene graph from the generated 2D image. How to ensure the object orientation within the scene graph, since the orientation of object is very important for a reasonable scene structure. Can you quantify the failure rate of YOLOv8 detection and describe the specific training strategy used to ensure the YOLO model accurately maps the colored, top-down 2D layout image into precise 3D bounding boxes and orientations?

- The baselines (DiffuScene, InstructScene, PhyScene) are limited to synthesizing individual rooms, while CHOrD can synthesize house-scale layouts. However, the quantitative evaluation in Table 2 includes single-room results (Bedroom, Living Room) and one "Entire House" column. The comparison is unfair; there are some work related on whole house layout generation, such as HouseGAN and HouseGAN++, and its follow-up works. I think some comparisons with this kind of work are more essential.

- How are the inter-room dependencies (e.g., door placements, connectivity) implicitly encoded and maintained throughout the conditional diffusion process for the overall floor plan image? And if the input room boundaries are rotated by random angles, what about the robustness of the proposed model?

- The Empty Room Rate for the CHOrD dataset is 0.2902 (Table 8). Since the dataset is described as "artifact-free and ready to use" and containing "fully furnished kitchens, bathrooms, and balconies", could you clarify what constitutes an "empty room" in the CHOrD dataset? Does this mean some rooms, like small utility rooms or hallways, are intentionally unfurnished in the ground truth, or is this still considered an unavoidable artifact?

- Given that the CHOrD dataset is much cleaner than 3D-FRONT (Table 8 shows baseline PIoU of 0.2547 for 3D-FRONT vs. 0.0018 for CHOrD dataset), were the baseline models (DiffuScene, InstructScene) re-trained on the raw, uncleaned 3D-FRONT dataset or on the cleaned subset (4,847 scenes) used by previous work?

### Questions
See weaknesses

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
4

### Summary
This work presents a method for synthesising furniture layouts conditioned on floorplans. In contrast to existing approaches that use generative models in a 'symbolic' space of scene descriptions, the proposed method instead uses an image-to-image latent diffusion model to map from a floorplan (containing just walls, doors, windows, etc.) to a furnished layout (using colored boxes to denote furniture). The resulting image is then 'interpreted' by a standard object detection pipeline, in order to convert it back to a more conventional scene representation indicating where furniture instances are to be placed. Results on 3D-FRONT and a custom dataset show better performance (in terms of realism and interpenetrations) than several recent baseline methods.

### Strengths
The proposed approach to furniture layout generation is novel. The idea of directly generating in plan-view image space, conditioned on a plan-view image of the empty room is straightforward but elegant, and presumably easier for models to learn than other representations (e.g. predicting bounding-box coordinates).

As an extra contribution beyond the technical approach, the paper introduces a new dataset of layouts, apparently of higher quality than the widely-used 3D-FRONT, and sufficiently large for training generative models from.

Empirical comparisons against three fairly recent baseline methods (DiffuScene, InstructScene, PhyScene) show improvements over those baselines on both datasets (3D-FRONT and the proposed dataset). This improvement is uniform across metrics including distributional similarity to ground-truth layouts (FID and KID), as well as object penetration rates (POR and PIoU).

There is an additional experiment showing that the model can also be used to generate 'fine grained' layouts, i.e. arrangements of adornments such as objects placed on tables etc (as opposed to just large furniture items).

There is a brief but informative analysis of why the models tend not to generate out-of-distribution (intersecting) furniture elements so often compared with prior methods.

The paper is clear, well-structured, and pleasant to read.

### Weaknesses
The paper title, abstract and introduction strongly emphasise "house-scale" generation, i.e. jointly modelling several rooms together. However nothing in the method is specific to this setting, and there is no quantitative evaluation of how well this works – in particular how well inter-room dependencies are captured, i.e. whether a truly accurate joint distribution across all the furniture in a house is learnt, or just per-room marginals.

The method only seems to support floorplan-conditioned generation. While conditioning on floorplans is vital, for a layout generation model to be useful it must also be possible to provide text conditioning or other guidance to ensure the layout meets other requirements for the target domain. This is now standard for methods in this area, including the baselines DiffuScene and InstructScene.

The proposed dataset only specifies object classes and bounding-boxes. It does not incorporate any information on style, shape, etc. This greatly limits its usefulness in the task of generating plausible layouts, since haphazard choice of furniture styles is a common failure mode and hallmark of automated layout generation methods.

It is unclear how the proposed dataset of layouts was collected. It is stated they were prepared by "experts" but there is a lack of information on who these experts were, how they were instructed, and how the quality of the resulting layouts was verified. This is problematic given the history of somewhat dubious datasets (SUN-CG and 3D-FRONT) in this area that have tended to contain large proportions of low-quality scenes (as the authors themselves note in the case of 3D-FRONT).

The section on fine-grained generation is rather minimal; in particular it is not clear how large the dataset was nor how it was collected; it is also not clear whether overfitting might have occurred.

Using an object detector to 'interpret' the diffusion-generated furniture layout plan-view image and convert back to a symbolic bounding-box representation feels somewhat hacky, and borderline strong enough as the main technical contribution for an ICLR paper. Indeed overall the pipeline is very much an engineered system built out of standard well-understood components, albeit combined in a novel and effective way.

### Questions
Most relevant issues are discussed under "Weaknesses" above. In particular…

Please provide evidence or argumentation to properly support the "house-scale" claim, beyond a small set of visual examples?

Please provide more details on both the main CHoRD dataset and the smaller dataset used for fine-grained object layout, in particular the protocol that was used to ensure the layouts are of high quality.

What does "2D bounding boxes defined by … 3D coordinates" (L295) mean? Are the furniture and room elements represented in 2D or 3D?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a scene synthesis method and an indoor dataset. The method first generates a floor plan using a diffusion model. Based on this floor plan, a detector detects large objects. Based on these objects, a hierarchical scene graph is extracted, which maintains the relationships between objects and rooms. For large objects that can be regarded as a platform, this again can be used for generating small object layouts using a diffusion model, so the method iteratively finishes the indoor synthesis.

### Strengths
The paper is very easy to understand and has a very clear pipeline.
The paper contributes a large-scale dataset to the community for further research. The dataset has many unique characters that 3D-FRONT does not have.

### Weaknesses
1. The method appears to be a straightforward pipeline that chains existing components (diffusion → detection → diffusion) in a loop. The contribution seems incremental, as each module has prior art and the combination does not clearly yield a novel algorithmic insight.

2. The paper says fine-grained object generation is autoregressive in L241-242, but the description (and Fig. 10) looks like one-shot generation of all small items conditioned on a parent anchor. That’s hierarchical/conditional, not autoregressive. If it is truly AR, please provide a formula how you model the problem. Please spell out the factorization and decoding order and show that each item conditions on previously placed siblings.

3. The manuscript groups DiffuScene under “graph-based” methods in L247-249, but DiffuScene doesn’t actually use explicit edges during generation. Meanwhile, the community has some scene-graph-based generation methods, like CommonScenes, GraphDreamer, EchoScene, Planner3D, and MMGDreamer. Please I wonder why the authors neglect them in the baselines and references?

4. What exactly is the hierarchical scene graph in this paper? The paper references it often but never really defines it. Please provide node/edge types, attributes, hierarchy rules, and how constraints are enforced. Without a precise definition, it’s hard to judge the claimed benefits.

5. The experiments lack of evaluation of dinning rooms, where fine and cluttered objects matter. 

6. After carefully inspecting the supplenmentary materials, I found that the rooms in the dataset only provide names (`roomName`) in Chinese. For an international venue, please provide English (or bilingual) labels.

### Questions
The bounding boxes are generated from a BEV floor plan where all objects are clearly separated. However, how are the clutter situations handled? Typically, a chair is inserted into a table slot; thus, the bounding boxes have overlaps. This would affect the performance of the object detector.

The paper only researches BEV renderings, as far as I understand. If I am right, the teaser is a bit confusing. If I am wrong and the paper can actually provide 3D rooms, how is the physical simulation handled? For example, how are objects naturally placed on the floor or on the table without any penetrations?

### Soundness
2

### Presentation
3

### Contribution
2
