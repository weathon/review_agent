# TS-Attn: Temporal-wise Separable Attention for Multi-Event Video Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Generating high-quality videos from complex temporal descriptions, which refer to prompts containing multiple sequential actions, remains a significant challenge. Existing methods are constrained by an inherent trade-off: using multiple short prompts fed sequentially into the model improves action fidelity but compromises temporal consistency, while a single complex prompt preserves consistency at the cost of prompt following capability. We attribute this problem to two primary causes: temporal misalignment between video content and the prompt, and conflicting attention coupling between motion-related visual objects and their associated text conditions. To address these challenges, we propose a novel, training-free attention mechanism, Temporal-wise Separable Attention (TS-Attn), which dynamically rearranges attention distribution to ensure temporal awareness and global coherence in multi-event scenarios. TS-Attn can be seamlessly integrated into various pre-trained text-to-video models, boosting StoryEval-Bench scores by 33.5% and 16.4% on Wan2.1-T2V-14B and Wan2.2-T2V-A14B with only a 2% increase in inference time. It also supports plug-and-play usage across models for multi-event image-to-video generation. The source code and video demos are available in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents advancements in a typical issue presented by video generation methods, which often produce sequences of events that present temporal anomalies like overlapping events and incorrect ordering. This problem is also discussed in detail, to better understand its causes and justify the approach.

The solution presented is TS-Attn, a method that requires a modification to the cross-attention layer such that it makes use of event ordering information presented in the prompt itself. This expanded attention layer  requires additional input of temporal segmentation information, which is generated using either an external API, human data, or a simpler segmentation method.

Finally, the work does very complete experiments which (except for the caveats in the weaknesses section) appear to produce an important improvement to the provided baselines, while being a relatively simple to implement addition to the video generation systems.

### Strengths
The paper has substantial strengths:

1) the problem is worth solving, and the root cause analysis is excellent
2) the presented method is (to the extent of my knowledge) novel and interesting; event-aware attention modulation in particular seems like an interesting approach to me
3) the benchmark is very complete and required the integration to 3 different models, which is impressive
4) the benchmark results that are indicated in the paper are very strong, except for the points raised in the weaknesses section which will hopefully be easy to fix
5) finally, the paper is well written, and the explanation is very clear

### Weaknesses
This work presents a very complete analysis of an original method.

The soundness and contribution scores of this paper are diminished by ambiguity on the impact of latency in the full system, as explained below.

Specifically, the work would benefit from a more detailed explanation of how the uniform segmentation method works. When explaining this method, the paper refers to a number of events in the prompt, but it's unclear how the events in the prompt are parsed themselves. To me this seems like a critical point because if the event segmentation in the prompt has to be provided by a model for example, the time to run such segmentation should be accounted for in the latency analysis.

This applies generally to claims about latency, for example in figure 1c, and in other tables in the paper. It's unclear to me which segmentation method was used for each figure and table, whether segmentation itself was included in the latency analysis or not, and whether TS-Attn required additional human input that other methods didn't require (e.g. if uniform segmentation required manual selection of prompt events).

In summary, the paper could benefit from:

1) more clearly specifying how the uniform segmentation method works
2) providing details on which segmentation method was used for each figure and table
3) providing details on the impact of segmentation on latency

My understanding is that in most figures the LLM API approach is used, and it only adds about 2-3 seconds to latency (changing the +2% latency claim to 2.2% or 2.3% perhaps), but it would be good to be more explicit about this.

More minor points:

4) including the prompts used in the segmentation methods in the appendix could be an interesting addition.
5) that figure 2 makes use of 3 colors, 2 of which can be easily confused with each other.

### Questions
My main questions are about the segmentation pipeline and full impact on latency, which are discussed in detail in the weaknesses section.

### Soundness
2

### Presentation
4

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
This paper proposes Temporal-wise Separable Attention, a training-free method to improve video generation with complex multi-event prompts. The proposed attention mechanism enables this by dynamically restructuring cross-attention distributions to ensure motion-related regions in each frame primarily attend to temporally aligned events, demonstrating superior results across various video generation models.

### Strengths
- The problem is well-defined and the solution is intuitively designed. The visualization in Figure 2 makes this more convincing.
- The proposed Attention Rearrangement and Attention Reinforcement are carefully designed to better inject multi-prompts while maintaining pre-trained generative priors in a training-free manner.
- The experimental results are extensive. The proposed method has been implemented on various pre-trained models to demonstrate its effectiveness and is also compared with other methods for multi-prompts.

### Weaknesses
- There are some heuristics arising from the training-free design. For example, the erosion function and the accompanying spatial separation of subject tokens limit the applicable scenarios of this method to simple ones. For instance, cases where the subject is the style of the video or cannot be clearly distinguished in 2D spatial aspects fall outside this premise.
- While the attached video results look good, there is an absence of video comparisons with other multi-prompt methods. More qualitative comparisons in the paper would also be beneficial.

### Questions
Are there any issues with sudden scene changes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TS-Attn, a training-free attention mechanism that improves multi-event video generation by dynamically separating and modulating cross-attention between motion regions and multi-event textual conditions. By introducing motion region extraction and event-aware attention modulation, the method reduces temporal misalignment and cross-event coupling, achieving better temporal coherence and event accuracy. TS-Attn can be plugged into existing diffusion-based video models without retraining, yielding substantial performance gains on StoryEval-Bench with only ~2% extra inference cost.

### Strengths
- Well-motivated and intuitive idea that directly targets temporal attention entanglement in multi-event video generation.

- Training-free and plug-and-play design makes it broadly applicable across existing diffusion models.

- Extensive experiments and ablations demonstrate consistent improvements and robustness across architectures and benchmarks.

- Clear presentation and visualizations that effectively explain both the mechanism and empirical benefits.

### Weaknesses
**Lack of comparison with prior methods**
The paper omits several highly relevant works that also manipulate cross-attention maps to achieve fine-grained event grounding without retraining, such as DreamRunner [1], VideoTetris [2], and TALC [3].
These methods similarly align textual tokens with corresponding visual regions through attention reweighting, making them conceptually close to TS-Attn. However, the authors neither cite nor compare with them. Including these approaches as baselines or at least discussing their differences would strengthen the paper’s positioning and contribution clarity.

**Unclear generalization to multi-subject scenarios**
The proposed motion-region extraction appears to assume a single dominant subject, computing masks for the entire video latents.
In cases involving subject transitions (e.g., a person leaves and a cat enters), this design may fail to isolate subject-specific motion regions, leading to incorrect or conflicting event-to-visual grounding.
Clarifying how TS-Attn handles such cases, or showing examples involving multiple subjects, would improve the completeness of the work.

---
[1] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation, 2024.

[2] VideoTetris: Towards Compositional Text-to-Video Generation, 2024.

[3] TALC: Time-Aligned Captions for Multi-Scene Text-to-Video Generation, 2024.

### Questions
See weakness. Overall, I lean towards borderline for the current version and am happy to update my rating if my questions are well answered.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
the paper proposes a training-free temporal-wise separable attention for multi-event conditioned video generation

### Strengths
- the motivation of the work is clear and well-explained. the proposed motion region extraction and event-based attention modulation are intuitive and visualizations are reasonable
- extensive experiments demonstrate the effectiveness of the proposed method. the visual results are convincing and clearly show the multi-event coherence
- the method achieves reasonable performance improvements with negligible overhead
- the paper provided source codes in the supplementary material

### Weaknesses
- the benchmark protocol and the evaluation metrics are not well-justified. while I understand there is no reasonable benchmark framework in the current field, the reliability of the vlm-based evaluation is still questionable, especially when evaluated against a commercial endpoint. in that case, it is hard to determine the actual performance gain based on the reported scores. a user study is highly recommended to validate the effectiveness of the proposed method considering the human evaluation is still the most reliable metric for video generation tasks
- it is unclear what is the max possible number of events the proposed method can handle
- what is the success rate of a given prompt? what are the typical failure cases?
- while the proposed framework provides a solution for multi-subjects, it is unclear whether the proposed framework can faithfully handle multiple subjects with same/similar actions

### Questions
please refer to the weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3
