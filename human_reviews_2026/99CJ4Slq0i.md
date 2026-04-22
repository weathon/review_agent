# FoleyGenEx: Unified Video-to-Audio Generation with Multi-Modal Control, Temporal Alignment, and Semantic Precision

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We introduce FoleyGenEx, a unified framework for video-to-audio (VTA) generation that integrates multi-modal control, frame-level temporal alignment, and fine-grained semantic expressivity, enabling synchronized, versatile, and expressive audio synthesis across diverse tasks. Existing VTA methods either offer multi-modal control with weak temporal alignment or achieve strong alignment while lacking reference audio conditioning and semantic precision. FoleyGenEx bridges this gap through three key innovations: a conditional injection mechanism enabling audio-controlled VTA and Foley extension, a multi-modal dynamic masking strategy preserving synchronization during multi-modal training, and an adverb-based data augmentation algorithm leveraging signal processing and large language models to enrich audio representations and textual supervision with nuanced semantic cues. Experiments on AudioCaps, VGGSound, and Greatest Hits show that FoleyGenEx delivers competitive performance in controllable VTA generation, achieving strong temporal fidelity, versatile multi-modal control, and fine-grained semantic precision compared to existing methods. Demo samples are available at https://foleygenex.github.io/FoleyGenEx.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new video-to-audio model that can handle multi-modal controls such as text prompts and reference audio. The proposed model is based on MMAudio and is modified to accept reference audio as input for audio-conditioned video-to-audio generation. For training, the authors also propose a new data augmentation strategy to enhance the controllability of the model. This strategy focuses on several specific types of adverbs appearing in the text prompt and augments both the text and the corresponding audio so that the augmentation process creates another pair of text and audio with different characteristics in terms of the semantics specified by the target adverb. The experimental results show that the proposed model achieves both high audio-visual alignment and controllability compared with prior methods.

### Strengths
- The motivation for this work is clear and convincing. Simultaneously achieving both high audio-visual alignment and multi-modal controllability is critically important for practical applications of video-to-audio generation.
- The design of the proposed model is reasonable. It is based on the MMAudio architecture, but includes several modifications: proper handling of video tokens for potentially misaligned video conditions and the addition of conditional audio input for audio-controlled video-to-audio generation.
- The quality of the generated audio shown on the demo page is excellent.

### Weaknesses
- It would be helpful if the authors could provide more empirical evidence on the importance of adverbs in the text prompt for achieving higher controllability.
  - While the idea of adverb-based data augmentation is interesting, it contributes little to improving the quality of the generated audio in general cases, as shown in Tables 2–4. Maybe this is due to an insufficient amount of samples containing such adverbs in the text prompt.
  - The proposed augmentation boosts the model’s performance on the dedicated test set (Table 6), but this is expected since the model is trained to perform well on such datasets. I'm not that convinced that this test set is represetative of real-world scenarios for controllable video-to-audio.
  - Therefore, to demonstrate the significance of the proposed augmentation, providing empirical evidence on the importance of adverbs is essential.
- Since the architecture of the proposed model is largely based on MMAudio, it would be beneficial to include a list that explicitly describes the modifications made to MMAudio.

### Questions
- Why do the authors focus on adverbs for higher controllability?
  - How did the authors choose the specific set of adverbs shown in Figure 4?
- Could you provide a list of modifications from MMAudio?

### Soundness
2

### Presentation
2

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
This paper introduces FoleyGenEx, a model that extends MMAudio to incorporate audio conditioning for reference-based video-to-audio synthesis and foley extention. To achieve multi-modal control and synchronization, the approach employs a conditional injection mechanism and a dynamic masking strategy. Furthermore, an adverb-based data augmentation method is proposed to augment the training data, enabling more fine-grained foley control. The authors conducted experiments on the AudioCaps, VggSound, and Greatest Hits datasets to validate their approach.

### Strengths
1. This paper aims to achieve temporally synchronized, multi-modal, and fine-grained foley generation for videos, a task with significant practical value.

### Weaknesses
1. Lack of Clarity in Model Architecture and Training: The paper's illustrations and descriptions of the model are insufficient for reproducibility and full comprehension. Figure 2, which should clarify the training process, is confusing; the data flow is not apparent, and the purpose of the block labeled "Flow" is ambiguous. It is also unclear whether the method involves full fine-tuning or a parameter-efficient approach with frozen components. Figure 3 suffers from similar clarity issues, as the input configurations for different conditional modes (e.g., (2) vs. (3)) are visually indistinguishable. The precise definitions and differences between key feature notations, such as F_S and F_{RS}, are not adequately explained.

2. Insufficient Technical Detail and Unclear Novelty: The paper's primary contributions—the conditional injection mechanism and the dynamic masking strategy—are not described in enough technical detail. The current descriptions are too high-level, and it is difficult to understand their precise implementation. Established techniques like concatenation with latent variables or cross-attention are widely used for conditional control. The paper fails to articulate what makes its proposed mechanisms novel compared to this extensive body of prior work. This ambiguity fundamentally undermines the paper's claimed contributions.

3. Questionable Validity of Data Augmentation: The proposed data augmentation method, which involves speeding up audio clips, is not convincingly justified. Altering the speed of an audio signal can easily introduce significant artifacts and degrade acoustic quality, potentially providing a noisy or misleading training signal. The authors provide no analysis to demonstrate that this technique preserves the essential foley characteristics or that the augmented data is of sufficient quality to benefit the model.

4. Limited Perceptible Improvement in Audio Quality: A subjective listening comparison of the audio samples provided for FoleyGenEx and the baseline (MultiFoley) reveals no clear or significant improvement. The qualitative results are highly similar, which calls into question the practical effectiveness of the proposed method. This lack of a discernible advantage in the final output fails to support the paper's claims of superiority.

### Questions
1. Could you please provide a revised diagram or a more detailed description of the end-to-end training process? Specifically, please clarify the training strategy: is the entire model fine-tuned, or are some components (e.g., the MMAudio backbone) frozen?

2. In Figure 3, could you explicitly define the feature representations F_S and F_{RS} and explain how they differ in each of the illustrated modes? For example, what is the exact input difference between mode (2) and mode (3)?

3. I strongly recommend a thorough revision of Figures 2 and 3 to ensure they are self-contained and clearly illustrate the data flow, the state of inputs (e.g., zeroed-out, masked), and the role of each component.

4. Could you provide a detailed architectural description of the "conditional injection mechanism"? How does it fundamentally differ from standard, widely-used conditioning techniques like FiLM layers or cross-attention mechanisms?

5. To substantiate your claims of novelty, please add a subsection to the paper that explicitly compares your proposed mechanisms to prior work, highlighting the specific architectural or functional innovations.

6. Have you performed any analysis to measure the audio quality of the augmented data? How can you ensure that speed alteration does not introduce artifacts that would impair training? The paper would be strengthened by including evidence of the augmentation's validity. This could be in the form of objective quality metrics or a small perceptual study comparing original and augmented audio samples.

7. Given that the audio samples sound qualitatively similar to the baseline, could you point to specific acoustic characteristics (e.g., temporal alignment, texture, clarity) in your results that demonstrate a clear improvement? When quantitative gains are marginal, strong qualitative evidence is essential. A formal user study (e.g., an A/B preference test) would be required to make a convincing case for the perceptual superiority of your method.

### Soundness
3

### Presentation
2

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
This paper proposes FoleyGenEx, which aims to achieve controllable and well-synchronized video-to-audio generation. Primarily, this method combines the synchronization features from MMAudio and the audio control features from MultiFoley. The authors have also collected an “adverb-augmented” dataset training fine-grained control. The proposed method achieves comparable performance to both MMAudio and MultiFoley in their respective strong domains.

### Strengths
- Overall, the method is well-executed and achieves good performance without bells and whistles. Fine-grained semantic control and temporal alignment are often at odds with each other because the visual information often conflicts with the controls. This paper seems to have handled that well, and the qualitative results are promising.
- The newly collected adverb-based data should be helpful for the community for training more fine-grained models, if this dataset could be released.

### Weaknesses
- The technical contributions are not very clear. Specifically, it is unclear what makes the proposed method work. This is exemplified by the fact that there are no ablation studies in this paper. For example, how important is the concatenation of the conditional latent
- The implementation details of some of the applications are not very clear. For example, in AC-VTA, what do the authors mean by “The latent extracted from the reference audio is not only concatenated with the initial noise, along the channel dimension, but is also prepended to the initial noise.”? Does this mean that the (un-noised) latent is in the same token sequence as the initial noise? Was the network trained this way? Additionally, how is “inversion” used for editing? Can the authors provide more details?

### Questions
If the authors can provide more details and evidence (i.e., ablations) on the components that contribute to the good performance of this method, I will be happy to raise my score.

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
This work proposes FoleyGenEx a unified video-to-audio generation framework. It is built upon the multi-modal diffusion transformer (MMDiT). It combines multi-modal control, frame-level temporal alignment, and fine-grained semantic expressivity within a single system. FoleyGenEx introduces a conditional injection mechanism for reference-audio conditioning, a multi-modal dynamic masking strategy to preserve synchronization, and an adverb-based data augmentation algorithm leveraging signal processing and large language models to enhance semantic precision.

### Strengths
- The paper provides a unified framework. The framework effectively consolidates multiple previously fragmented VTA tasks into a single model with shared mechanisms. 
- The conditional injection mechanism and dynamic masking are well-motivated, addressing specific weaknesses of MultiFoley (poor synchronization) and MMAudio (lack of reference conditioning).
- It provides an adverb-based augmentation pipeline, combining LLM-based caption generation with signal-level perturbations.
- The experiments cover a wide range of tasks, datasets, and metrics (FD, IS, CLAPT, IB-score, DeSync, etc.), providing a thorough performance comparison.

### Weaknesses
### Major
- Despite solid engineering, FoleyGenEx is heavily built on existing architectures (MMAudio + MMDiT + Synchformer). The contributions—masking and conditional injection—feel like moderate extensions rather than a conceptual breakthrough. Meanwhile "adverb augmentation"  is from the prior text-based data augmentation frameworks in audio-language modeling. Its novelty is limited.

- Some improvements are minor. For example, for Adverb-Augmented(+AA), table-2 on VGGSound it shows FDVGG 0.74 → 0.73, table-3 CLAP_T 33.53 → 34.20. Without ablation on training diversity or robustness (e.g., out-of-domain videos), it’s uncertain whether the improvements are meaningful beyond numerical artifacts.

### Others
- The paper does not isolate the effects of each proposed component (e.g., masking vs. conditional injection vs. adverb augmentation). The results in Table 6 touch on adverb augmentation, but the interaction among components is unclear.
- The paper does not use human study or perceptual analysis verifying whether “semantic precision” is actually perceptible beyond automated metrics.

### Questions
- Could you provide detailed ablation results isolating the effects of the conditional injection mechanism, the multi-modal dynamic masking strategy, and the adverb-augmented data respectively? It is currently unclear which component contributes most to the improvements in Tables 2–6.
- Some improvements are minor. For example, for Adverb-Augmented(+AA), table-2 on VGGSound it shows FDVGG 0.74 → 0.73, table-3 CLAP_T 33.53 → 34.20. Without ablation on training diversity or robustness (e.g., out-of-domain videos), it’s uncertain whether the improvements are meaningful beyond numerical artifacts.Could you clarify how significant these differences are, and whether the augmented data improves controllability in qualitative user tests?
- Since the framework integrates masking, conditional injection, and flow matching, how sensitive is the model to hyperparameter settings (e.g., mask ratio, conditioning dropout)? Did you observe any trade-offs between synchronization and style fidelity?

### Soundness
3

### Presentation
3

### Contribution
3
