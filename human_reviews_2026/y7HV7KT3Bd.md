# JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
This paper introduces JavisDiT, a novel Joint Audio-Video Diffusion Trans- former designed for synchronized audio-video generation (JAVG). Based on the powerful Diffusion Transformer (DiT) architecture, JavisDiT simultaneously generates high-quality audio and video content from open-ended user prompts in a unified framework. To ensure audio-video synchronization, we introduce a fine-grained spatio-temporal alignment mechanism through a Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) Estimator. This module extracts both global and fine-grained spatio-temporal priors, guiding the synchronization between the visual and auditory components. Furthermore, we propose a new benchmark, JavisBench, which consists of 10,140 high-quality text-captioned sounding videos and focuses on synchronization evaluation in diverse and complex real-world scenarios. Further, we specifically devise a robust metric for measuring the synchrony between generated audio-video pairs in real-world content. Experimental results demonstrate that JavisDiT significantly outperforms existing methods by ensuring both high-quality generation and precise synchronization, setting a new standard for JAVG tasks. Our code, model, and data are available at https://javisverse.github.io/JavisDiT-page/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents JavisDiT, an end-to-end audio-video synchronized generation model based on the Diffusion Transformer architecture. By introducing a Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo), the model achieves fine-grained alignment between visual and auditory signals, significantly improving spatial-temporal consistency and overall generation quality. The authors also construct a large-scale dataset, JavisBench, and propose a new synchronization evaluation metric, JavisScore, which enhances the systematicity and objectivity of the study. Experimental results show that JavisDiT outperforms existing methods across multiple benchmarks. Overall, this work is solid with new innovations and extensive experiments, demonstrating potential application in multimodal generation.

### Strengths
- The research topic in this paper is meaningful and valuable. Audio-video joint generation (JAVG) is a promising emerging field, and an open-sourced JAVG method is appreciated.
- The proposed model exhibits some novel ideas. In particular, the proposed Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) enables fine-grained modeling of spatial-temporal synchronization relationships in generation, which is a new and promising attempt.
- The data and evaluation contributions are also great. The proposed JavisBench benchmark and JavisScore metric may provide valuable resources and evaluation standards for future research in the community.

### Weaknesses
Given the clear motivation and solid contributions in method, data, and benchmark, I have no major concerns. I would like to share some questions or suggestions for further improvement. Specifically,

- The model structure may be complex and resource-intensive. The dual-branch Diffusion Transformer architecture for audio and video involves a large number of parameters, leading to potentially high training and inference costs. The proposed HiST-Sypo requires additional pre-training. It is recommended to include a complexity and resource consumption analysis in both training and inference.

- The paper lacks failure case analysis. Although the proposed method is effective, it would be better to include typical failure cases and corresponding discussions, which would help reveal the model’s limitations and potential directions for improvement.

- There is a slight overclaim in some parts of the writing. The current work mainly focuses on sound effects generation and has not yet been extended to speech audio. It is suggested to avoid overextended descriptions of timbre or speech-like attributes in the Introduction section to maintain precision and academic rigor.

- Minor issues: 1) The appendix contains comprehensive details. But, it may be better to move the ablation table 4 into the appendix while adding more details back to the main paper (eg, to further explain or validate the key design of HiST-Sypo).  2) Typo: '240P4s' in the Caption of Table A1 at Line 810 should be '240P 4s'?

### Questions
- An open question is whether it would be possible to use much less data to achieve comparable performance?

- Regarding the text semantic understanding module: In related works such as Tango [1] (text-to-audio model) and MultiTalk [2] (speech-based audio-visual generation model), the text encoder is often based on T5-like large language models [3,4], which can capture richer semantic representations. Has the author considered adopting similar LLM architectures to further enhance text semantic modeling capability?

[1]Text-to-audio generation using instruction guided latent diffusion model. ACM MM.

[2]  Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation. arXiv preprint arXiv:2505.22647.

[3] Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70), pp.1-53.

[4] UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining. ICLR.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses joint audio-video generation, focusing on two key objectives: (1) achieving strong generative performance for each individual modality, and (2) ensuring fine-grained synchronization between audio and video. To meet these goals, the authors propose JavisDiT, a diffusion transformer (DiT)-based model capable of generating high-quality audio and video. To enhance cross-modal alignment, they further introduce a hierarchical spatial-temporal synchronized prior estimation module. In addition, the paper presents a new benchmark, JavisBench, consisting of high-quality text-annotated sounding videos for evaluating joint audio-video generation.

### Strengths
1. The two identified challenges—developing a strong backbone for joint audio-video generation and achieving fine-grained audio-video synchronization—are well-motivated and reasonable.

2. Both the proposed JavisDiT model and the accompanying benchmark are valuable contributions that can provide meaningful insights to the research community.

3. The experimental evaluation is comprehensive and well-conducted.

### Weaknesses
1. Although the JavisDiT model is clearly described, its architectural design is not entirely intuitive. For example, the rationale behind the specific designs of the spatial-temporal self-attention and spatial-temporal cross-attention modules is not well explained. Providing more insights into the design choices would make the model structure easier to understand.

2. The proposed JavisScore is reasonable. It might be interesting to explore whether it could be extended in a hierarchical manner—for example, by incorporating multiple window sizes for evaluation.

3. Given the crucial role of the training dataset in determining both model accuracy and generalization, it would be helpful to clarify the main criteria used to ensure dataset validity. Additionally, it would be valuable to discuss how JavisDiT performs when potential biases arise from the heavy reliance on VLM-generated video captions.

### Questions
1. Training such a model requires substantial computational resources. Therefore, it would be valuable to further explore the broader potential of JavisDiT, for example, by examining its applicability to other related tasks.

2. The robustness of the model with respect to the quality of the training dataset should also be discussed, as dataset noise or imbalance could significantly affect performance and generalization.

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
1. The paper proposes a joint audio-video generation model named JavisDiT, which is based on the Diffusion Transformer (DiT) architecture and aims to simultaneously generate high-quality and synchronized audio-video content from text prompts.
2. The paper designs a core module, the Hierarchical Spatial-Temporal Synchronized Prior Estimator, which extracts both global coarse-grained semantic and local fine-grained spatio-temporal priors from text. These priors are then injected into the DiT module to guide the precise spatial and temporal synchronization of the audio and video.
3. Addressing the lack of scene diversity and complexity in existing benchmarks, the authors constructed a new and more challenging benchmark dataset, JavisBench. If this dataset were to be made open-source, it would be beneficial to the community's development.
4. To address the deficiencies of existing metrics (such as AV-Align) , the paper introduces JavisScore. This new metric uses a temporal-aware semantic alignment mechanism to robustly measure synchronization in complex real-world scenarios.

### Strengths
The paper's motivation is clear and precisely targets the two core challenges in the field of Joint Audio-Video Generation (JAVG): ensuring high-quality generation of audio and video, and maintaining perfect synchronization between the two modalities. It is rare for a single work to contribute a model, dataset, and metric simultaneously. This represents a significant potential to advance the field and, as an academic work, is sufficiently compelling.

### Weaknesses
1. There is a clear disconnect between the paper's core claims and the qualitative results provided in the supplementary demo. The paper claims to solve high-quality generation and perfect synchronization, but the demos fail on both counts. The visual quality is significantly lower than the level of current video generation models, and audio-video synchronization is not well demonstrated in these demos, often appearing as coarse "scene-ambience" rather than the claimed "fine-grained" alignment. Furthermore, the number of samples is insufficient to cover the breadth of capabilities claimed. Critically, the demos lack the most essential evidence: direct comparisons against baseline methods (e.g., "T2V + V2A" or "T2A + A2V") for the same prompt. While the authors provide objective metrics in Table A8, the corresponding qualitative examples to visually substantiate these claims are missing.
2. The paper lacks sufficient subjective (human) evaluation. This is a necessary component for measuring video quality, audio quality, and audio-video alignment, and its absence is a significant omission.
3. The authors' justification for their model's weaker audio performance (e.g., in line 1538) is unconvincing. Attributing this weakness to supporting "variable-length audio generation" is a flawed argument, as the baseline AudioLDM 2 also supports this feature. Additionally, while the authors' concern about AudioLDM 2 training on AudioCaps is valid, their use of the term "data leakage" is imprecise.
4. The multi-stage training strategy is overly complex. Does the complex multi-stage training strategy require significant manual intervention and tuning when applied to new datasets.
5. The paper's readability is poor due to dense prose. For example, the term "Fine-Grained Spatial-Temporal Self-Attention Cross-Attention" (line 89) is structurally confusing. This confusion is amplified by the inconsistent naming in Figure 2 (e.g., "Fine-Grained ST-CrossAttn" and "ST-SelfAttn"), which hinders a clear understanding of the architecture. While the authors state in the Appendix (A.4) that LLMs were used "solely as writing assistants," we believe the paper's over-reliance on such tools has resulted in these convoluted descriptions and reduced overall readability.

### Questions
1. The HiST-Sypo estimator uses ImageBind's text encoder, while the proposed JavisScore metric also relies on ImageBind to measure audio-visual synchronization. Does this not create a 'circular' evaluation, where the model is effectively trained to optimize for the feature space of the metric that is supposed to be judging it?
2. Regarding the audio autoencoder, why did the authors opt for a Mel-spectrogram based VAE (which necessitates a vocoder) instead of a waveform-based VAE (e.g., Wave-VAE)? The choice of a Mel-VAE introduces an inherent quality loss from the vocoder, which could be a critical bottleneck for the final audio generation quality.
3. The paper claims to support variable-length audio generation, possibly via dynamic temporal masking. How is this practically implemented during inference? Given that the main evaluations are on 4-second videos, what evidence is there that this mechanism can robustly scale to longer durations (e.g., 10 seconds) while maintaining quality and synchronization?
4. In the ablation studies (Sec 5.3, line 795), the paper introduces normalized scores ($S_{AVQ}$, $S_{AVC}$, $S_{AVS}$) to simplify the results. What is the justification for the specific weighting factors used in these formulas? These equations appear arbitrary and lack a clear theoretical or empirical basis, making the ablation results difficult to interpret.

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
4

### Summary
JavisDiT is an end-to-end Diffusion Transformer ($\text{DiT}$) architecture that simultaneously generates high-quality video and audio content. The core innovation is the Hierarchical Spatial-Temporal Synchronized Prior ($\text{HiST-Sypo}$) Estimator. This module extracts both coarse-grained semantic priors (overall framework) and fine-grained spatio-temporal priors (timing and location of events) from the input text to guide the synchronization of the generated video and audio.

Key Contributions:

Novel JAVG Architecture (JavisDiT): A unified $\text{DiT}$-based system that leverages spatio-temporal self-attention, cross-attention, and a Multi-Modality Bidirectional Cross-Attention ($\text{MM-BiCrossAttn}$) module to achieve high-quality joint generation.

Hierarchical Synchronization Mechanism ($\text{HiST-Sypo}$): A fine-grained alignment module that estimates and injects text-derived spatial priors (where an event occurs) and temporal priors (when an event starts/stops) into the generation process.

New Benchmark ($\text{JavisBench}$) and Metric ($\text{JavisScore}$): The paper introduces $\text{JavisBench}$ ($\sim$10,140 samples), a challenging dataset focused on complex, multi-event, real-world scenarios (including sequential and simultaneous events). It also proposes $\text{JavisScore}$, a temporal-aware semantic metric designed to more robustly evaluate fine-grained synchronization than previous methods.

### Strengths
The combination of $\text{DiT}$ as a backbone for joint audio-video generation with the Hierarchical Spatio-Temporal Prior ($\text{HiST-Sypo}$) Estimator is highly novel. Previous JAVG methods lacked explicit fine-grained spatio-temporal modeling. The $\text{JavisScore}$ metric, which targets the least synchronized frames to increase sensitivity to local desynchronization, is also an original contribution to evaluation methodology.

The system design, including the detailed module structure (Figure 2) and the three-stage training strategy (pretraining, $\text{ST-Prior}$ training with contrastive loss, $\text{JAVG}$ training), is robust. The ablations demonstrate that the $\text{HiST-Sypo}$ estimator provides a significant and consistent gain in $\text{AV-Consistency}$ and $\text{AV-Synchrony}$ compared to simple bidirectional cross-attention ($\text{BICA}$). Furthermore, the empirical verification confirms that $\text{JavisScore}$ is substantially more reliable ($\sim$23% higher accuracy) than $\text{AV-Align}$ in distinguishing synchronous from asynchronous pairs.

The paper clearly articulates the dual challenges of high-quality generation and fine-grained synchronization. The distinction between $\text{Spatial Alignment}$ and $\text{Temporal Alignment}$ is precisely defined. The framework is well-diagrammed, and the appendix provides ample technical detail on data curation, augmentation for negative samples, and loss functions.

This work significantly elevates the bar for $\text{JAVG}$ research. $\text{JavisDiT}$ achieves $\text{SOTA}$ performance across both established and the new, challenging $\text{JavisBench}$ dataset, setting a new standard for generation quality and synchronization coherence. The introduction of $\text{JavisBench}$, focused on multi-event complexity (75% multiple events, 57% simultaneous events), directly addresses the "out-of-domain" issue that hinders the practical applicability of current models.

### Weaknesses
Computational Cost and Efficiency: The paper acknowledges the high computational overhead of the $\text{DiT}$-based architecture and diffusion process9. The latency analysis in Table $\text{A1}$ indicates that generating a small $240\text{P}, 4\text{s}$ clip takes $\mathbf{30}$ seconds on an $\text{H}100 \text{ GPU}$, which is prohibitive for real-time or fast creative applications. The training of the full JAVG model takes $\mathbf{256}$ GPU days on $\text{H}100$10. Suggestion: Discuss explicit strategies for accelerating inference (e.g., consistency models or distilled sampling methods) as a crucial next step to transition the model from research to practical use.

Dataset Leakage and Bias: While the authors took steps to avoid leakage by collecting videos between June and December 2024 and filtering 11, they trained the audio branch on vast public datasets ($\sim 788\text{K}$ entries) including $\text{AudioSet}$ and $\text{VGGSound}$12. These datasets are known to contain biases (e.g., $\text{off-screen}$ sounds) and may overlap non-trivially with the test sets of cascaded models (which often use V2A models trained on these very datasets). Suggestion: Explicitly discuss how the $\text{HiST-Sypo}$ training (Stage 2) mitigates biases learned by the audio branch during pretraining (Stage 1).

Ambiguity in Prior Estimation: The fine-grained $\text{ST-Prior}$ estimation adopts a $\text{VAE}$-like sampling strategy to model the variability of events for the same text prompt: $(p_{s},p_{t})\leftarrow\mathcal{P}_{\phi}(s;\epsilon)$13131313. However, it's unclear how much of the final generation is controlled by the sampled noise ($\epsilon$) vs. the core semantics. Suggestion: Discuss or ablate the influence of the sampling variance ($\mathcal{P}_{\phi}$'s variance output) on the final generated synchronization, specifically for multi-event or off-sc

### Questions
Robustness to New Language (Zero-Shot/Few-Shot): The HiST-Sypo relies on an ImageBind encoder trained on text-image pairs. How robust is the HiST-Sypo estimator to novel descriptive phrases (e.g., neologisms, highly contextual language) that were not present in its training corpus? Does the learned spatio-temporal prior generalize effectively when the text is complex and abstract?

Necessity of Two-Stage Prior Training: The ST-Prior Estimator is trained separately (Stage 2) using a complex contrastive learning pipeline (four loss functions, negative sampling strategies). Was a simpler end-to-end training strategy (e.g., training the estimator jointly with the full JAVG model using only the JAVG loss and minimizing a simpler feature distance) attempted? If so, why did it perform worse than this multi-stage approach?

Visualization of Fine-Grained Alignment: Figure 7 is a compelling visualization of the Spatial and Temporal Attention maps, confirming the model focuses on the correct object ("bubbles" vs. "diver"). Could the authors provide a similar visualization for a multi-event, sequential or simultaneous case (e.g., like the one in Figure 1, robot and dog tussle, then aliens talk) to demonstrate how the attention maps shift over time to capture the different spatial and temporal priors injected?

JavisScore Parameter Selection Justification: The authors selected Topk-min-40% as the synchronization estimation strategy (focusing on the 40% least synchronized frames). The ablation (Fig. A6) shows only minor variations in AUROC across most Topk thresholds. Why was 40% chosen over 50% or 60%? Is there a theoretical or perceptual reason for focusing on the hardest 40% of the clip, and did human evaluation confirm that desynchronization is most noticeable at these specific thresholds?

### Soundness
3

### Presentation
3

### Contribution
3
