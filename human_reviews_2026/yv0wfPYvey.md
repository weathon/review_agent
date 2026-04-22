# Spatio-Temporal LLM: Reasoning about Environments and Actions

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Despite significant recent progress of Multimodal Large Language Models (MLLMs), current MLLMs are challenged by "spatio-temporal" prompts, i.e., prompts that refer to 1) the entirety of an environment encoded in a point cloud that the MLLM should consider; and simultaneously also refer to 2) actions that happened in part of the environment and are encoded in a short ego-centric video clip. However, such a holistic spatio-temporal understanding is important for agents operating in the real world. To address this challenge, we first develop a framework to collect a large-scale dataset. Using the collected "Reasoning about Environments and Actions" (REA) dataset, we show that recent MLLMs indeed struggle to correctly answer "spatio-temporal" prompts. Building on this dataset, we study two spatio-temporal LLM (STLLM) baselines: 1) STLLM-3D, which directly fuses point cloud, video, and text representations as inputs to the LLM; and 2) STLLM-Aligner, which aligns spatial context with video and text before LLM decoding. Both baselines aim to enhance spatial understanding of environments and temporal grounding of egocentric observations. On REA, the STLLM baselines outperform existing models, demonstrating the effectiveness of our designs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of enabling multimodal large language models to reason about both the spatial layout of an entire 3D environment and the temporal dynamics of actions within it. To study this capability, the authors introduce the REA dataset, a collection of question-answer pairs built from point clouds and egocentric videos. They also propose two model architectures, STLLM-3D and STLLM-Aligner, which are designed to integrate these distinct spatio-temporal inputs. The experiments show that existing generalist MLLM struggle with these prompts, while the proposed STLLM baselines achieve a better understanding of the integrated scene and action sequences.

### Strengths
- The paper overall is easy to follow. The proposed new dataset is well-motivated. It is natural and reasonable to develop 3D understanding dataset for MLLMs by improve their capabilities about reasoning of the 3D enviroment.

### Weaknesses
- The solution to collect the dataset is quite straightforward and simple, where the idea has considerable overlap with existing solutions to develop 3D MLLMs like [r1, r2] and more recent work on foundation embodied models like [r3]. These models and new benchmarks like VSI-Bench are not considered. 

[r1] Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence

[r2] 3UR-LLM: An End-to-End Multimodal Large Language Model for 3D Scene Understanding, TMM

[r3] RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete, CVPR 2025.

### Questions
Please refer to my comments above. Overall, although the basic idea of the paper is well-motivated, the contribution of the proposed dataset and models is a bit limited. There is no clear advantage in size or quality compared to existing efforts, and the source has an overlap with existing ones. The performance of the models is also not stronger than existing 3D/embodied models. Therefore, I cannot recommend acceptance for this paper.

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
This paper identifies a gap in Multimodal Large Language Models (MLLMs), which currently struggle with "spatio-temporal" prompts. The paper makes two main contributions: A new dataset, "Reasoning about Environments and Actions" (REA). This dataset is collected using a detailed pipeline built upon existing assets like EPIC-KITCHENS, VISOR, and EPIC-FIELDS. REA features 24,371 training samples across five tasks: Relative Direction, Relative Distance, Find My Item, Furniture Affordance Prediction, and Action Planning. Two baseline models, STLLM-3D and STLLM-Aligner. These models extend a video-language model (LLaVA-Video-Qwen2 ) to incorporate 3D point cloud information.

### Strengths
1. The primary strength is the introduction of the REA dataset. It identifies a clear weakness in existing MLLMs and provides a new, challenging benchmark with a detailed data collection pipeline (Sec 3.1) to spur further research.
2. The paper proposes two simple but effective baseline architectures (STLLM-3D and STLLM-Aligner) that demonstrate the clear benefit of fusing point cloud data with video-language models.

### Weaknesses
1. The paper's definition of "spatio-temporal" reasoning—understanding an "entirety of an environment encoded in a point cloud" and "actions... encoded in a short ego-centric video" —feels like an overclaim. This formulation primarily models a single agent's movement and interaction within an otherwise static 3D space. The essence of spatio-temporal reasoning, which would distinguish it from spatial reasoning, should arguably involve understanding the dynamics of other moving objects, agents, or complex environmental changes that are not just a consequence of the egocentric agent's actions. The current work is more of a "3D-Aware Egocentric Action Reasoning" model than a general "Spatio-Temporal LLM."
2. Related to the first point, most of the proposed tasks (e.g., Relative Direction, Relative Distance, Find My Item, Furniture Affordance Prediction ) do not seem to be sufficiently distinct from existing 3D spatial reasoning tasks. These tasks appear to require identifying objects in a 3D point cloud and then relating the agent's position (derived from the video) to them. While the agent moves, the core reasoning is still heavily spatial. Only the "Action Planning" task seems to robustly require temporal logic. The benchmark would be significantly stronger if it focused more on tasks that are impossible to solve with spatial-only reasoning.

### Questions
1. The paper's definition of "spatio-temporal" is central to its contribution. Could the authors comment on the decision to limit the scope to a static point cloud and a single egocentric video, rather than a more complex definition involving multiple dynamic objects or agents? Is the current REA dataset seen as a first step, or is the single-agent perspective the intended final scope?
2. For tasks like "Relative Direction" and "Relative Distance," how did the authors ensure that these require genuine temporal reasoning, as opposed to simply comparing two static agent-object spatial relationships extracted from the video and point cloud?
3. Even with the STLLM-Aligner design, the model does not explicitly leverage frame-to-frame geometric correspondences (e.g., camera trajectories or object motion paths). Thus, its capacity to maintain temporal-spatial consistency in dynamic scenes remains uncertain. Have you evaluated the model separately on static versus dynamic video subsets to assess robustness to motion and viewpoint changes?
4. The Aligner module performs cross-attention between video and point-cloud features, but it is not shown that this attention actually captures meaningful geometric relations. Could you visualize the attention maps between video and point-cloud tokens to demonstrate that the model focuses on geometrically consistent regions rather than redundant appearance cues?
5. Your evaluation relies mainly on LLM judges (ChatGPT-4o, Gemini Flash), which assess linguistic correctness rather than geometric consistency. Have you considered introducing explicit geometric metrics—such as spatial relation accuracy or direction prediction error to demonstrate that the model learns true 3D reasoning rather than textual alignment?

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
The paper introduces a new dataset, called REA, which incorporates spatial and temporal relations in scenes. REA is built on the EPIC-Kitchens dataset, and consists of ~5K unique actions, 300 objects, and 30K samples. The paper also trains two models on REA: STLLM-3D (a LLaVa-style fused model) and STLLM-Aligner (which uses cross-attention). On the REA dataset, these fine-tuned models outperform zero-shot and linearly probed LLMs.

### Strengths
1. The paper empirically demonstrates that existing MLLMs perform poorly at spatio-temporal tasks (23.85% to 31.46% across tasks), whereas the REA-trained baselines achieve 41.89% overall accuracy. 

1. The generation of rich, fine-grained question-answer pairs from the EPIC-Kitchens dataset is well thought through; in particular, the point-cloud reconstruction (Section 3.1.2) describes the generation of 3D views from a variety of scenes, which are then used to produce the REA questions. Prior work (see 2, 4 in Weaknesses] has mostly dealt with either 2D-spatial-only, or coarse-grained 3D-spatio-temporal views, which makes the REA dataset novel.

### Weaknesses
1. The REA dataset is not compared against existing benchmarks for embodied agents,  such as Embodied-Bench [1], which covers both high-level and low-level motor tasks in 4 environments (unlike the REA benchmark, which focuses on the EPIC kitchen environment). It is thus unclear what the novelty of the new benchmark is: is it the 3D-based point cloud reconstruction (which is missing in Embodied-Bench)? More concretely: a good benchmark should ideally demonstrate that models trained on it can transfer to other, unseen, benchmarks (e.g. such as BLINK). 
 
1. The paper does not address key prior work, in particular the technique of using intermediate scene graphs for robot planning [2-4]. For example, ConceptGraphs [2] produces _open-vocabulary_ 3D scene graphs for robot manipulation, albeit on spatial relations only; other work [4] has extended it to temporal relations too. Thus, statements such as "the study of integration of both spatial and temporal data into LLMs is in its infancy" (Line 209) should be used with caveats, since this paper attempts to learn these scene graphs implicitly within latent space, as opposed to generating intermediate graphs explicitly.

References

[1] embodied-bench.github.io

[2] Gu et al, ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning. https://arxiv.org/abs/2309.16650

[3] Hsu et al, What’s Left? Concept Grounding with Logic-Enhanced Foundation Models. https://arxiv.org/abs/2310.16035

[4] Huang et al, LASER: A neuro-symbolic framework for learning spatio-temporal scene graphs with weak supervision. https://arxiv.org/abs/2304.07647

### Questions
See weaknesses above.

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
2

### Summary
This paper addresses the limitations of current Multimodal Large Language Models (MLLMs) in handling spatio-temporal prompts, where both environmental context (from 3D point clouds) and temporal actions (from short egocentric videos) must be jointly understood. Existing MLLMs are typically trained only on static image-text pairs, which leads to poor performance when faced with tasks requiring integrated spatial and temporal reasoning.

To study this challenge, the authors introduce a large-scale dataset, Reasoning about Environments and Actions (REA), which features five spatio-temporal question-answering tasks. These tasks require models to reason about relative direction, distance, object localization, furniture affordances, and future action planning, based on a combination of sparse point clouds and densely annotated egocentric videos.

The authors propose two baseline models:

* STLLM-3D, which directly fuses 3D point cloud, video, and text inputs into a unified representation for LLM decoding.
* STLLM-Aligner, which first aligns spatial information with video and language features before decoding.

Both models significantly outperform existing MLLMs on REA, demonstrating that integrating structured 3D context and temporal dynamics leads to more effective spatio-temporal reasoning.

### Strengths
* Addresses an important and underexplored problem in MLLMs: joint spatial and temporal reasoning.
* Introduces a new dataset (REA) that is built with careful attention to grounding and scene structure, using real-world environments and realistic tasks.

### Weaknesses
* All evaluations are performed in kitchen scenarios. The models may not generalize to other environments such as offices or outdoor scenes.
* The analysis does not deeply examine what the models actually learn. Are they truly reasoning over space and time, or simply picking up on surface correlations?
* The data pipeline relies on noisy pose and segmentation estimates, but the paper lacks analysis of annotation errors or their impact on dataset quality.

### Questions
* Can you provide a human baseline on the REA dataset to help contextualize model performance?
* What is the effect of removing point cloud information? How much of the gain comes from 3D input versus the egocentric video?

### Soundness
3

### Presentation
2

### Contribution
2
