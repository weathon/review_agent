# CAPNAV: TOWARDS ROBUST INDOOR NAVIGATION WITH DESCRIPTION-FIRST MAPS

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 6, 0

## Abstract
Humans naturally form mental maps of their surroundings: they picture what a destination looks like, relate it to nearby objects, and implicitly plan a route before moving. We seek a similar capability for embodied agents: given a free-form description such as ``go to the white sofa with curved edges'', the agent should pick the correct 3D instance among many lookalikes and navigate to it safely. We propose CapNav, a description-first navigation framework that builds an instance-centric 3D map from RGB-D streams and uses natural-language object descriptions as the primary interface for goal selection. CapNav maintains a dense semantic voxel map for global geometry and, in parallel, constructs persistent 3D object tracks by aggregating Detic based open-vocabulary detections and LSeg features over time. For each stabilized track, a small set of views is captioned with a vision-language model and embedded with BGE-M3, yielding a caption-enriched representation that links language, semantics, and 3D pose. At test time, free-form instructions are encoded in the same text space, matched against object captions to select a target instance, and then converted into a metric-space waypoint followed by A* planning. CapNav shows consistent improvements over category-only and map-based baselines (ZSON, LM-Nav, CoW, VLMaps) in multi-object navigation tasks, and its instance-level captions make retrieval decisions transparent and easy to interpret.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces CapNav, a description-first indoor navigation framework designed to handle attribute-based natural language instructions such as “go to the white sofa with curved edges.” CapNav integrates multi-view captioning, attribute-based 3D mapping, and a multimodal LLM planner to improve object localization and navigation robustness. The system constructs instance-level 3D maps annotated with attributes (color, shape, texture) and verifies attribute consistency before executing navigation. Experiments in simulated Matterport3D environments demonstrate improvements over VLMaps on multi-object navigation tasks.

### Strengths
1. The paper addresses an important problem—bridging fine-grained visual grounding and natural-language navigation. This direction aligns well with current ICLR interests in multimodal reasoning and embodied AI.

2. The pipeline combines several components (object detection, segmentation, 3D reconstruction, multimodal retrieval) into a unified framework. The end-to-end system demonstrates practical functionality in simulated environments.

### Weaknesses
1. Lack of originality and algorithmic contribution. The proposed CapNav mainly combines existing modules—YOLO/SAM for detection, CLIP/DINOv2 for embeddings, and LLMs for captioning and retrieval—under a new name.

2. Writing and structure issues. The Introduction and Related Work sections are overly long and not clearly separated into paragraphs, making it difficult to follow the narrative flow. 

3. Experimental evaluation is insufficient. The experiments are conducted only in one Matterport3D scene, with a small number of instructions (9 episodes). Comparisons are limited to VLMaps without any other baselines; no ablation studies, robustness tests, or cross-domain evaluations are provided. Reported gains (71% vs. 52%) are not statistically analyzed, and the small scale of experiments limits generalization.

4. The motivation for the paper is not sufficiently robust. The paper claim that "People describe indoor destinations with attributes rather than names alone", I haven't found any evidence to support this.

### Questions
no

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CapNav, a "description-first" indoor navigation framework to address the limitation of existing VLN methods in handling fine-grained attribute-aware natural language instructions. CapNav constructs a 3D instance-level map with explicit attribute fields (color, shape, texture) via multi-view captioning and feature clustering (DINOv2, LongCLIP), parses natural language into "category-attribute-relation" constraints using a multimodal LLM, and verifies attribute consistency before navigation. Experiments on Matterport3D (Habitat simulator) show CapNav achieves 71.0% subgoal success rate, outperforming VLMaps (51.6%) by 37.6%.

### Strengths
1. The paper clearly identifies the three core challenges of attribute combinatoriality, view dependence, and perception-planning decoupling in VLN, which are highly relevant to practical applications (e.g., home service robots).

2. The multi-view fusion and pre-goal verification modules directly address the limitations of prior methods—qualitative results show CapNav can disambiguate similar objects (e.g., sofas with different edges) and provide interpretable decisions (traceable to captions).

3. By leveraging LongCLIP and YOLO11x, CapNav handles unseen categories/attributes, which is more practical than closed-vocabulary VLN methods.

### Weaknesses
1. Limited to Simulated Environments: All experiments are conducted on Matterport3D (simulated) without real-robot tests. Real-world noise (e.g., depth sensor error, dynamic objects, uneven lighting) may significantly degrade performance, but the paper does not discuss "sim-to-real" transfer—this contradicts the title "towards robust indoor navigation".

2. Long-Horizon Navigation Ignored: The full-chain completion rate is only 20%, but the paper does not analyze the root cause (e.g., whether subgoal localization errors accumulate, or the planner fails to handle subgoal transitions). It also does not propose any improvements, treating this as "future work" without justification.

3. Fixed Attribute Thresholds: Clustering and verification thresholds (e.g., τ_sem=0.65, τ_vol=0.15) are manually set. The paper does not evaluate the sensitivity of these thresholds to different scene clutter levels, nor does it implement the "online calibration" mentioned in future work—this makes the framework less adaptive to diverse environments.

4. The paper only briefly mentions FindAnything (2025) and LERF (2023) but does not provide quantitative comparisons. For example, it claims FindAnything "lacks attribute verification", but no data is given to show how much CapNav outperforms it—this weakens the argument for CapNav’s superiority.

5. The framework assumes static scenes but does not discuss how to update the 3D map when objects move (e.g., a pillow moved from the sofa to the floor). This is a critical limitation for real indoor navigation, where environments are often dynamic.

6. Some related and important works are missing citations: [1] Weakly-Supervised Multi-Granularity Map Learning for Vision-and-Language Navigation [2] Bevbert: Multimodal map pre-training for language-guided navigation [3] Instruction-guided path planning with 3D semantic maps for vision-language navigation

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes a navigation system that enhances object localization by integrating attribute-based language grounding with visual perception. The system constructs a 3D instance-level map where each object is annotated with semantic embeddings and explicit attribute fields (e.g., color, shape, texture). Using foundation models like DINOv2 and CLIP, the method performs multi-view captioning and feature clustering to create consistent per-instance descriptors. A multimodal LLM is used to parse natural language queries into structured constraints, and a verification stage ensures attribute consistency before navigation. Experiments in simulated environments show that CapNav outperforms a baseline (VLMaps) on subgoal success rate (71.0% vs. 51.6%) in multi-object navigation tasks.

### Strengths
1. Unlike prior methods, CapNav treats attributes as first-class, queryable fields, improving fine-grained object disambiguation.
2. Navigation choices are traceable to structured captions, aiding transparency and error analysis.

### Weaknesses
1. The entire evaluation is conducted on only a single Matterport3D scene . This extremely narrow scope fails to demonstrate the system's performance and robustness across diverse indoor environments with varying layouts, lighting, object types, and clutter levels.

2. The proposed method is compared against only one baseline, VLMaps. This is a critical flaw.

3. The paper fails to justify design choices (e.g., weighting α=0.4/0.6 for embeddings) or isolate the contribution of each component.

### Questions
See Weakness.

### Soundness
2

### Presentation
1

### Contribution
1
