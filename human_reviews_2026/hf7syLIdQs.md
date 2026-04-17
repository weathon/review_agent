# How Should Corruption Be Used in SSL? Empirical Insights for Effective Pretraining

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
We study how corruption design—masking and additive noise—affects self-supervised pretraining of vision models. Although denoising diffusion models succeed in generation, noise-driven extensions of masked image modeling (MIM) achieve only marginal gains on recognition tasks, including fine-grained benchmarks. We thus investigate why this would be the case, seeking effective ways to combine masking and noising within the corruption-to-reconstruction (C2R) paradigm. We begin by analyzing prior noise-based MIM approaches, categorizing them into Substitutive Corruption (masked tokens replaced by noised ones) and Conjunctive Corruption (masked and noised tokens coexist), and further into Encoder- or Decoder-style depending on where corruption and restoration occur. Our study reveals that the literature trends toward a Decoder-style design. In contrast, we evaluate an Encoder-style alternative with a focus on transfer. Building on these analyses, we propose three principles for effective C2R pretraining: corruption and restoration should occur within the encoder, noise is most effective when injected at the feature level, and mask reconstruction and de-noising must be explicitly disentangled to avoid interference. By implementing these findings, we propose a framework that captures a broader frequency spectrum of representations and improves transferability, surpassing MIM by up to 8.1% and recent noise-driven pretraining methods by 8.0% across diverse recognition benchmarks. Code is available in the Supplementary Material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper empirically investigated the how masking and corruption should be used in visual SSL pretraining. The author proposed three principles: apply corruption/restoration in the encoder (encoder-style), inject feature-level noise early, and disentangle masked token reconstruction from de-noising. Following these principles, they built an encoder-style C2R framework using feature-space noise and a disruption loss to suppress mask–noise interference. Experiments show up to ~8% transfer gains over MIM on fine-grained tasks.

### Strengths
1. This paper systematically derives a framework based on a clear definition of encoder/decoder styled and conjunctive vs substitutive corruption, which provides concrete grid of classifying different denoising and MIM based pretraining objectives. 
2. The authors provided controlled comparisons that isolate specific factors (corruption placement, noise injection stage, etc), which supports causal conclusions. 
3. The empirical findings are summarized into three clean design principles that are easy to understand.

### Weaknesses
1. All experiments use a ViT-B backbone, pretrains only on ImageNet-1K for 400 epochs, and then fine-tunes on downstream tasks, which limits evidence that the conclusions hold at larger model scales or longer pretraining epochs (1600 epochs to have a fair comparison with result reported by DiffMAE and MAE). 
2. The motivation behind principle 1 needs to be expanded. Principle 1 favors encoder-style architecture for downstream transfer tasks and  was mainly described in section 4.1. However, the result of figure 4b only shows modest gains comparing to the decoder style, and the paper acknowledges that this alone “does not fully reveal its potential,” and immediately moved on to the principle 2 and 3. Without theoretical, empirical analysis, or reference to works that directly discusses the choices, it is unknown whether the following two principles will also help the decoder-styled C2R. Decoder-styled C2R could benefit from the efficiency (fewer token will be consumed by the encoder) and enable longer pretraining in the same computation budget. I encourage the author to provide more comprehensive investigations on the benefits of choosing encoder-style approach.
3. An extension to weakness 2, the author should consider to show isolated gain from each principle as a separate ablation study.
4. The paper does not discuss added compute cost using the encoder-style C2R. 
5. Appendix C discusses longer pretraining schedule but switched to FGVC as pretraining dataset. The author should consider to report on ImageNet1K in consistent with the result reported elsewhere in the paper. 
6. While I understand in this paper the author is trying to investigate how to make effective usage of denoising in SSL for vision tasks, it is not clear to me, both empirically and theoretically, why this is an important application. If the idea is to make C2R effective in both image understanding and generation tasks, the author should make this motivation clear in the paper. Otherwise, the current result is behind SOTA MIM, and showing the motivation somewhat ill-grounded.

### Questions
1. The C2R model is pretrained with 400 epochs and the reported result for MAE is not matching up with the performance reported in MAE paper which is pretrained on 800/1600 epochs. I'm wondering whether the longer of the pretraining will also help C2R on downstream transfer. 
2. I'm wondering how quantitatively C2R performs on image generation/reconstoration tasks beyond qualitative examples presented in figure 11.

### Soundness
3

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
This paper comprehensively studies the corruption issues used in self-supervised learning (SSL), especially for the masked image modeling (MIM) methods. Corruption is a common manipulation in SSL, instantiated as masking the input image in MIM methods. This paper analyzes the noising strategy (substitutive or conjunctive) and the location for corruption (encoder-style or decoder-style). With the analysis results, the paper proposes to perform training with the encoder style and feature-level noise, where the masked token reconstruction and denoising are disentangled.

### Strengths
1. How to design the masking strategy is an interesting and important problem in MIM methods. This paper presents a systematic study that guides the design of the final training method based on the obtained conclusions.
2. The proposed MIM method is evaluated on several tasks and benchmarks, presenting notable improvements.

### Weaknesses
1. The biggest concern lies in the impact of the paper. Currently, the community primarily focuses on encoder pretraining for multimodal data and settings, such as CLIP and its successors. There also emerge other stronger pretrained encoders like DINOv3. How will this method benefit the self-supervised learning field? I recognize that the obtained conclusions can be useful for MIM methods, but MIM ones may be somewhat limited in current competitions. The main methods or baselines in this paper are from more than two years ago.
2. Discussing the MIM methods with diffusion models may be inappropriate. Though some works (e.g. MaskDiT) do similar attempts, there indeed exist inherent differences, where sampling and (multi-step) denoising steps in diffusion models are not applicable in MIM.

### Questions
1. An explanation about the potential impact of the paper is needed, considering MIM methods may be limited in the current SSL field.
2. A better demonstration of the relationship between MIM and diffusion models needs to be provided. It is suggested to distinguish the two methodologies/tasks explicitly.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper analyzes how to combine masking and additive noise for self-supervised pretraining in vision models. The authors introduce a unified corruption-to-reconstruction (C2R) framework that categorizes existing methods and show that encoder-style and conjunctive corruption lead to better transfer than the common decoder-style setups. Building on this analysis, the authors propose three principles: perform corruption and restoration within the encoder, add noise in feature space (early blocks), and disentangle de-masking from de-noising via a proposed disruption loss. The proposed method shows consistent gains over standard MIM and recent noise-based baselines across a range of downstream tasks.

### Strengths
- The design choices are carefully motivated, each supported by clear hypotheses and empirical validation.
- The proposed approach achieves consistent improvements over standard MIM and noise-based baselines across different downstream tasks.
- The paper is well-written and structured, making the ideas easy to follow and the experimental results clear.

### Weaknesses
- The main novelty seems to lie in the disentanglement (disruption) loss, as encoder-style and feature-level noise have been explored in prior works; the overall contribution feels more exploratory and incremental than conceptually new.
- The paper lacks comparisons with some recent MIM baselines, for example, ColorMAE [A], HPM [B], and MixedAE [C], which would strengthen the empirical evaluation.
- A comparison of the learned feature visualizations between the proposed method and baseline models would help highlight what new information or structure the encoder captures under the proposed framework.
- The paper does not report computational cost or training time. Since the method combines masking and additive noise, an analysis of efficiency and resource requirements would be useful.
- Main quantitative comparisons are mostly shown in figures; presenting them in a summary table would make it easier to assess the actual performance gains.

Minor comments:
- Some figures (e.g., Fig. 8) are pixelated and should be improved for clarity.

[A] Carlos Hinojosa, Shuming Liu, and Bernard Ghanem. "ColorMAE: Exploring data-independent masking strategies in Masked AutoEncoders." European Conference on Computer Vision (ECCV) 2024.

[B] Kai Chen, et al. "Mixed autoencoder for self-supervised visual representation learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR) 2023.

[C] Haochen Wang, et al. "Hard patches mining for masked image modeling." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023.

### Questions
- Could the authors provide a comparison of their method with more recent MIM baselines such as ColorMAE, HPM, or MixedAE?
- Could the authors provide comparative visualizations of the learned features to better illustrate what the encoder captures under the proposed framework?
- What is the computational cost of the proposed approach compared to standard MAE or MIM baselines?
- Could the authors clarify what is the final training objective and how the disruption loss is combined with others, if any?
- Could the authors provide visualizations showing how the disruption loss actually changes the attention or affinity patterns in practice?

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
This paper investigates how different corruption strategies—specifically masking and additive noise—impact self-supervised pretraining of vision models within the corruption-to-reconstruction (C2R) paradigm. While denoising diffusion models have been highly successful in generative tasks, their noise-driven extensions to masked image modeling (MIM) have not yielded significant improvements for recognition tasks. The authors systematically analyze why this is the case, categorizing prior approaches into Substitutive Corruption (masked tokens replaced by noised ones) and Conjunctive Corruption (masked and noised tokens coexist), and further into Encoder- or Decoder-style frameworks depending on where corruption and restoration occur.
Through extensive empirical study, the paper proposes three key principles for effective C2R pretraining; Corruption and restoration should occur within the encoder (since the encoder is transferred to downstream tasks). Noise is most effective when injected at the feature level (especially in lower encoder layers).Mask reconstruction and de-noising must be explicitly disentangled to avoid interference, which is achieved by suppressing attention between masked and noised tokens.
Implementing these principles, the authors design a new pretraining framework that captures a broader frequency spectrum of representations, leading to improved transferability. Their method outperforms standard MIM by up to 8.1% and recent noise-driven pretraining methods by 8.0% across a variety of recognition benchmarks, including fine-grained visual categorization, image classification, semantic segmentation, object detection, and instance segmentation.

### Strengths
1) The paper provides a thorough empirical study and a clear taxonomy of corruption strategies (Substitutive vs. Conjunctive, Encoder- vs. Decoder-style), clarifying why previous noise-based C2R methods have limited effectiveness for recognition tasks.
2) The authors distill their findings into three actionable principles for effective C2R pretraining, offering concrete guidance for the community on how to combine masking and noising for better transfer learning.
3) By advocating for encoder-style corruption/restoration, feature-level noise injection, and explicit disentanglement of de-masking and de-noising objectives, the proposed method captures richer, more transferable representations—demonstrated both theoretically and empirically.
4) The proposed approach achieves substantial improvements over both standard MIM and recent noise-based methods across a wide range of tasks and datasets, including challenging fine-grained recognition benchmarks. The results are robust, with statistical significance established through multiple trials.

### Weaknesses
1) How will this method extend to the AiM-v2 method? AiM-v2 does unfined decoding for image and text, can similar ablations be shown for AiM-v2 as well?
2) Comparison with CAN, in CAN contrastive loss resulted in good improvement in features and denoising loss also helped in representation learning. Comparison with CAN would also suggest what was the correlation between MIM loss, Contrastive loss and Denoising loss.
3) In terms of gains, how much did it come from Disruption loss and vs other choices? We need systematic comparison of each component and where did the gains come from.
4) The noise in different layers helps different datasets in different manners, which layer to finally apply noise to is not very clear. 

References.
[1]Multimodal Autoregressive Pre-training of Large Vision Encoders.
[2] A SIMPLE, EFFICIENT AND SCALABLE CONTRASTIVE MASKED AUTOENCODER FOR LEARNING VISUAL REPRESENTATIONS

### Questions
The implementation details for the final model are not very clear in the paper. Also there are other Contrastive MIM based methods that are not included in the paper which should be discussed.

### Soundness
3

### Presentation
2

### Contribution
2
