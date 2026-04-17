# Step-by-Step Video-to-Audio Synthesis via Negative Audio Guidance

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
We propose a step-by-step video-to-audio (V2A) generation method for finer controllability over the generation process and more realistic audio synthesis.
Inspired by traditional Foley workflows, our approach aims to provide better controllability by enabling incremental generation of desired sound, thus enabling users to produce multiple sound events induced by a video comprehensively.
To avoid the need for costly multi-reference video–audio datasets, each generation step is formulated as a negatively guided V2A process that discourages duplication of existing sounds.
The guidance model is trained by finetuning a pre-trained V2A model on audio pairs from adjacent segments of the same video, allowing training with standard single-reference audiovisual datasets that are easily accessible.
Objective and subjective evaluations demonstrate that our method enhances the separability of generated sounds at each step and improves the overall quality of the final composite audio, outperforming existing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Inspired by traditional Foley workflows, this paper proposes a step-by-step video-to-audio generation method to overcome controllability limitations of single-pass models. The core method, Negative Audio Guidance (NAG), conditions the generation on previously created audio tracks. This negatively guided process, formulated via concept negation, discourages the duplication of existing sounds and steers the model to incrementally synthesize missing audio events. Critically, the authors introduce the practical contribution of being trainable without costly multi-reference datasets, instead leveraging audio pairs from adjacent segments of the same video.

### Strengths
1. To the best of my knowledge, generating audio with separated concepts using audio-based negative guidance is novel in this field. The proposed NAG method is well-justified and, based on the audio separability metrics in Table 2, appears to offer significant improvement. Further, Figure 4 visually demonstrates that the model is capable of the incremental refinement that was challenging for previous V2A models. 

2. To overcome the data scarcity problem for this new task, the authors utilized audio pairs from adjacent segments of the same video. This allows the framework to be trained using standard, widely accessible single-reference datasets (e.g., VGGSound).

3. The authors compare their results against strong baselines, showing that step-by-step generation with NAG demonstrates better or comparable results across diverse key metrics, as well as in the user study. Furthermore, the authors constructed the Multi-Caps VGGSound dataset using a VLM to generate multiple, distinct captions per video, which was necessary to evaluate this novel task.

### Weaknesses
1. The authors explicitly state this limitation in Appendix G; the overall quality and the ability to generate specific sounds are heavily dependent on the underlying V2A model (MMAudio). NAG is a guidance mechanism that primarily prevents duplication; it seems to struggle with generating more diverse audio outputs or creating subtle or rare sounds if the base model was not adequately trained on them. This creates a hard ceiling on the final audio fidelity.

2. The ablation study in Appendix F (Table A3) reveals that the method's performance is sensitive to the order of generation. This adds a procedural constraint to achieve optimal results, which requires using the ImageBind score as a proxy to determine the generation order.

### Questions
1. Some improvements for the composite audio in Table 1 (e.g., IB-score) appear marginal compared to the baseline (MMAudio-S-16k). Please provide the standard deviation for both 'Ours' and 'MMAudio-S-16k' across all objective metrics in Table 1 and Table 2. This would help to ascertain the statistical significance of the observed improvements.

2. As stated in Appendix G, the naive addition is a limitation. Did the authors experiment with any alternative mixing strategies, such as learning the mixing weights or applying techniques like convex optimization, to create the final composite audio?

3. There appear to be discrepancies between the main results for 'Ours' (Tables 1 and 2) and the default setting ($\alpha=4.5, \beta=1.5$) in the ablation study (Table A2). For example, the $FD_{PANNs}$ is 6.47 in Table 1 but 7.32 in Table A2. Please clarify about this discrepancy.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a step-by-step video-to-audio (V2A) generation framework that mimics the Foley workflow, where missing sound events are incrementally added.
To prevent duplication of already-generated sounds, the authors propose Negative Audio Guidance (NAG) that pushes new audio away from previously generated audio features.
The method fine-tunes MMAudio with a ControlNet.
For evaluation, they build Multi-Caps VGGSound, a dataset with multiple captions per video to represent multiple sound events.
Experiments show improved audio separability across generated tracks and slightly better audio–video alignment compared to MMAudio and negative-prompt baselines.

### Strengths
1.	First to explicitly tackle incremental, Foley-style video-to-audio generation via negative conditioning.
2.	The method is compatible with existing flow matching diffusion frameworks.
3.	The authors propose a new dataset (Multi-Caps VGGSound).

### Weaknesses
1.	The number of generation steps is fixed to 5 for all 8-second clips, regardless of actual sound density. Adaptive step selection or early stopping could make the framework more efficient and realistic.
2.	The gain over the MMAudio baseline is marginal. Since MMAudio already produces high-quality, semantically aligned sounds, it remains unclear why a multi-step process is needed beyond marginal improvements in separability. More compelling examples (e.g., complex or dense soundscapes) would strengthen. Or a direct comparison with fine-tuned MMAudio trained on the multi-caption dataset would make the advantage clearer.

### Questions
1.	In the demos, temporal alignment appears less critical than semantic correctness (e.g., when a pigeon appears, the model simply adds wing-flapping sounds). This is also reflected in Figure 5 and Table 1, where temporal alignment shows the least improvement. Have the authors considered how it might be improved?
2.	Since Qwen cannot handle audio input, have the authors thought of other evaluation benchmarks (e.g., VGGSounder (ICCV 2025), which include multiple audio/video labels per clip)?
3.	Given that the proposed method requires roughly 10 seconds of additional generation time compared to MMAudio, do the authors believe this trade-off is worthwhile in practice?
4.	The paper could discuss and compare with recent ControlNet-style or additive generative approaches such as:
- SonicVisionLM:  Playing Sound with Vision Language Models (CVPR 2024)
- Read, watch and scream! sound generation from text and video (AAAI 2025)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a step-by-step video-to-audio(V2A) generation method that avoids duplicating previously generated sounds through Negative Audio Guidance (NAG). Additionally, the authors construct a new audiovisual benchmark dataset, named Multi-Caps VGGSound, which supports evaluation of compositional and incremental sound generation in V2A tasks. Experiments on the proposed benchmark across diverse aspects show that the proposed method improves audio separability, text fidelity, and overall audio quality compared to strong baselines. In summary, this work offers a practical framework for controllable and compositional Foley-like sound generation.

### Strengths
Unlike conventional V2A models that generate an entire audio tracks in a single pass, this paper introduces an incremental refinement framework inspired by real-world Foley workflows. The authors provide a strong theoretical foundation for the effectiveness of Negative Audio Guidance (NAG) by leveraging well-established techniques such as flow matching and classifier-free guidance, and empirically validate it thorough experiments. The proposed framework consistently outperforms state-of-the-art baselines across multiple metrics, including audio quality, semantic and temporal alignment.
The paper is clearly written and fairly easy to follow.

### Weaknesses
1.	While the proposed NAG framework introduces a promising way to suppress previously generated audio, it is unclear whether the model can effectively disentangle composited audio to understand which events have already been generated and should be suppressed in subsequent steps.

2.	The experimental evaluation relies solely on the VGGSound-based constructed dataset (Multi-Caps VGGSound), which may limit the generalizability of the proposed method. Although the authors proposed a method to construct evaluation benchmarks by generating multiple captions using a vision-language model, this procedure could have been applied to other datasets as well (e.g., AudioCaps) to better assess the robustness and applicability of the method in diverse audiovisual contexts.

3.	While the method is compared with several variants of MMAudio and negative prompting strategies, the evaluation lacks diversity in terms of architecture and training paradigms. Demonstrating the application of NAG to other types of V2A models would strengthen the claim of its generalizability beyond a single base model.

4.	According to the paper, the user study for composite audio was conducted with only 10 participants. This limited scale raises concerns about the statistical significance of the results. A larger and more diverse user study would provide stronger evidence of the method’s perceptual effectiveness.

### Questions
-	How effectively can the model distinguish individual audio events from the composited audio used in the NAG condition, especially in cases with overlapping or complex sound mixtures? Is there any analysis on the model’s capacity for selective suppression in such scenarios?

-	Given that the authors construct the evaluation set (Multi-Caps VGGSound) by generating multiple captions using a vision-language model, have authors considered applying the same procedure to other standard audiovisual datasets such as AudioCaps to better validate the generalizability and robustness of their method?

-	Is there any reason authors used Qwen2.5-VL model instead of other vision language model?

-	Have the authors explored integrating NAG with other V2A architectures beyond MMAudio, particularly those that support classifier-free guidance, to better validate the generalizability of the proposed method?

-	To improve the reliability of the user study for the composite audio, would the authors consider conducting a larger-scale user study and providing more detailed statistical analysis?

### Soundness
2

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
3

### Summary
The paper introduces a method to sequentially generate multiple audio tracks from video. To achieve this, the authors derive a technique called "Negative Audio Guidance" (NAG). During inference, NAG uses a reference audio-based model to prevent the generation of sounds that already exist in the audio track. The effectiveness of this method is demonstrated through quantitative and qualitative results on the VGGSound dataset.

### Strengths
1. This paper is well-written and it is easy to follow. 

2. The step-by-step generation process guided by negative audio is a novel and intuitive approach to audio synthesis, offering more control over the final output.

3. The method's ability to train without requiring complex, multi-reference datasets makes it more practical and accessible for researchers.

4. The paper provides a comprehensive evaluation, including both objective metrics and a subjective user study, to validate the effectiveness of the proposed method.

### Weaknesses
1. The motivation of generating multiple audio tracks step by step is not well-verified. The authors should compare such an step-by-step method with those methods with single-step inference with postprocessing (audio tracks decomposition). 

2. The dataset used for evaluation ("Multi-Caps VGGSound") was created using a vision-language model without access to the original audio, which could introduce a discrepancy between the text captions and the actual sound events. The authors should provide detailed quality analysis of such a dataest. 

3. The central idea, "Negative Audio Guidance" (NAG), is presented as a key contribution. However, this is functionally an application of negative prompting or classifier-free guidance, techniques that are well-established in the broader field of conditional diffusion models, particularly in the image domain. The paper's novelty lies in applying this to a sequential audio generation framework, but it would be stronger if it were positioned as an application and adaptation of existing principles rather than the invention of a fundamentally new guidance method. The contribution is more in the "how" (the sequential process) than the "what" (the negative guidance itself).

4. The paper's stated goal is to mimic the Foley process to create a complete, realistic audio track. However, the evaluation primarily measures the separability and individual quality of the generated sounds (stems), not the coherence and realism of the final composite audio. The simple summation used for mixing is a critical point of failure for realism. A realistic soundscape is not just a sum of its parts; it involves complex interactions, occlusion, and environmental acoustics. The current evaluation does not adequately measure whether the final output sounds like a plausible, unified audio scene.

### Questions
1. Could you elaborate on the relationship between NAG and existing negative prompting techniques in diffusion models? A more explicit positioning of your work within this broader context would help clarify whether the primary contribution is the guidance mechanism itself or its application within your proposed sequential generation framework.

2. We suggest the authors also compare single audio-track generation with completed audio track generation with post-processing (e.g., audio tracks decomposition) to evaluate the performance of a single audio track.

### Soundness
2

### Presentation
3

### Contribution
3
