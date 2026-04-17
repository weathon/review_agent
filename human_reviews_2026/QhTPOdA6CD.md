# KeySync: A Robust Approach for Leakage-free Lip Synchronization in High Resolution

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Lip synchronization, known as the task of aligning lip movements in an existing video with new input audio, is typically framed as a simpler variant of audio-driven facial animation. However, as well as suffering from the usual issues in talking head generation (e.g., temporal consistency), lip synchronization presents significant new challenges such as expression leakage from the input video and facial occlusions, which can severely impact real-world applications like automated dubbing, but are largely neglected by existing works. To address these shortcomings, we present KeySync, a two-stage framework that succeeds in solving the issue of temporal consistency, while also incorporating solutions for leakage and occlusions using a carefully designed masking strategy. We show that KeySync achieves state-of-the-art results in lip reconstruction and cross-synchronization, improving visual quality and reducing expression leakage according to LipLeak, our novel leakage metric. Furthermore, we demonstrate the effectiveness of our new masking approach in handling occlusions and validate our architectural choices through several ablation studies. Our code and models will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents KeySync, a two-stage diffusion-based framework for high-resolution lip synchronization that addresses key challenges in existing methods: temporal consistency, expression leakage, and occlusion handling. The approach uses a keyframe-interpolation strategy adapted from facial animation, implements a carefully designed masking strategy to minimize expression leakage from input videos, and introduces an inference-time occlusion handling mechanism using video segmentation. The authors also propose LipLeak, a novel metric to quantify expression leakage. KeySync achieves state-of-the-art performance at 512×512 resolution, particularly excelling in cross-synchronization scenarios where audio and video are mismatched.

### Strengths
1. The paper addresses several overlooked but practically important challenges in lip synchronization. The identification and formalization of "expression leakage" as a key problem is valuable, as is the introduction of the LipLeak metric to quantify this phenomenon. 
2. The paper is well-written. The problem formulation effectively highlights the limitations of existing approaches.

### Weaknesses
1. The evaluation is conducted on a relatively small test set (100 videos), which may limit the statistical significance of results. The cross-sync evaluation uses random audio swapping, which may not fully capture the complexity of real-world dubbing scenarios where semantic content matters.
2. The occlusion handling approach relies on SAM2 for segmentation, which may not handle all types of occlusions effectively (e.g., partial occlusions, transparent objects, or complex lighting conditions). The evaluation of occlusion handling is limited to a few qualitative examples.
3. The paper does not thoroughly analyze the computational overhead of the two-stage approach compared to single-stage methods. The additional RGB loss (Equation 5) requires VAE decoding during training, which likely increases computational cost significantly, but this trade-off is not quantified.

### Questions
1. How sensitive is the proposed masking strategy to different face shapes, poses, and video qualities? Have you tested the approach on more diverse datasets beyond the three used in training?
2. While you claim improved temporal consistency, could you provide more quantitative analysis of temporal artifacts? How does performance degrade with longer sequences?
3. Have you tested the approach on real-world dubbing scenarios with professional content? How does it handle challenges like multiple speakers, rapid speech, or non-frontal face poses?

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
This paper presents a well-structured and technically sound approach to high-resolution lip synchronization through a two-stage diffusion-based framework. It identifies and attempts to address important issues such as expression leakage and occlusion handling, both of which are often overlooked in prior work. The proposed methods and evaluation are thorough and demonstrate improvements over existing baselines.  The paper's primary strength lies in its focused and effective approach to solving critical real-world problems，specifically expression leakage and occlusion handling.  The proposal of the LipLeak metric is a valuable and original contribution that enhances the field's evaluation toolkit. On the other hand, a notable weakness is that the paper's core architectural framework is an adaptation of existing models rather than a fundamentally new design; its novelty is therefore more in the application and refinement than in foundational methodology.

### Strengths
1.The LipLeak metric is a unique addition to the field, enabling more precise measurement of expression leakage, which has been an under-explored issue in previous methods.
2.The method is robust and well-executed, with careful attention to both theoretical design and empirical validation. The ablation studies  provide clear evidence of the method's superiority over existing approaches.
3.Unlike prior masks that either only cover the mouth or the entire lower face, KeySync’s mask extends from nose level to the image bottom. This balances leakage suppression and visual quality, as validated by ablation.

### Weaknesses
1.Architectural innovation is limited: the two-stage keyframe-interpolation diffusion paradigm is borrowed from KeyFace and other works, and the core contributions are more reflected in  adaptation and module integration.
2. KeySync’s inference speed is faster than other diffusion models but far below real-time requirements for applications like live dubbing. The authors only mention future acceleration without testing current optimizations. 
3. The authors note “Table 1 results were generated from a single run” due to computational cost. Without 3+ repeated runs (to report mean±std) or t-tests, it is unclear if KeySync’s performance gains are statistically significant or due to randomness .

### Questions
1. Can the authors provide further justification for why LipLeak is superior to existing metrics for leakage detection? Additionally, how does the metric perform across a range of video qualities (e.g., lower-quality videos, videos with noise)?
2. Table 8 shows that while KeySync is fast for a diffusion model, it is still far from the real-time performance of GANs like TalkLip. Given your experience, what do you see as the primary technical bottleneck for deploying such high-quality diffusion-based lip-sync models in interactive applications? Beyond the distillation methods you mentioned, are there other promising avenues (e.g., architectural changes, quantization) you believe are critical for closing this performance gap?

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
The paper proposes KeySync, a lip-synchronization framework aimed at three pain points the authors argue are under-addressed in current works: (i) expression leakage from the source video during cross-synchronization, (ii) temporal consistency at higher resolutions, and (iii) robustness to facial occlusions. The method is a two-stage latent video diffusion pipeline (keyframe → interpolation) adapted from recent keyframe-style video diffusion (notably SVD / KeyFace-style setups) but specialized for lip-sync.

### Strengths
- High-resolution (512×512) results with temporal consistency: relevant for dubbing and for real deployment.
- Ablations in the right spots: audio encoder, keyframe generator design, mask choice. This makes the paper feel engineered, not just “we tried a big model.”

### Weaknesses
1. Lack of Novelty： The framework basically follows the existing two-stage / keyframe→interpolation video diffusion paradigm, with only engineering modifications to the masking strategy and mask correction during inference. Related improvement is incremental and lacks a unique technical contribution.
2. “Leakage-free” conclusion depends on specific metrics： This paper heavily relies on the proposed LipLeak, but this indicator itself is based on the stability of the mouth landmarks/MAR. The reliability of this indicator decreases in the presence of occlusion, side profile, low light, or large facial expressions, thus affecting the paper's core argument of "leakage-free." The paper does not provide a sensitivity analysis of this indicator under different mismatch scenarios.
3. The Robustness of the obscured scene is exaggerated： The so-called "robust occlusion" essentially relies on an external segmentation/tracking model to produce a cleaner inpaint mask, rather than the generative model itself learning occlusion-invariant lip-sync synthesis. Moreover, the mask generation pipeline is very trivial. In other words, it introduces a stronger front-end, rather than improving the generalization ability of the lip-sync model itself.

### Questions
1. Does the current LipLeak evaluate the "minor mouth movement" model as optimal? Could you provide an example of the opposite (greater mouth movement but no leakage) to demonstrate that the metric does not monotonically favor "minor movement"?
2. Could you provide a comparison: using the same two-stage video diffusion architecture plus your mask, but without your "leakage analysis/new metrics," how much performance difference is there? In other words, how much improvement does the metric itself actually bring?
3. How does your LipLeak handle invalid frames when there is occlusion, side view, or landmark glitches? Does it directly include them, discard them, interpolate, or perform smoothing? Are the current metrics still valid?

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
4

### Summary
This paper presents KeySync, a two-stage latent diffusion framework for audio-driven lip synchronization. The method employs a keyframe-interpolation strategy to enhance temporal coherence, introduces a leakage-proof masking scheme to prevent expression leakage, and incorporates an occlusion-aware inference mechanism based on SAM2. In addition, a new metric LipLeak, is proposed to quantitatively evaluate expression leakage.

### Strengths
The paper highlights an underexplored issue, expression leakage, in audio-driven lip synchronization and introduces KeySync as a solution. The proposed LipLeak metric quantitatively evaluates the severity of expression leakage.

### Weaknesses
The main concern lies in the limited technical novelty. The proposed pipeline is conceptually modest and largely incremental, introducing no new architecture or learning formulation beyond standard latent diffusion. Most components, such as masking, interpolation, and SAM2-based occlusion handling, are heuristic and externally attached, rather than being jointly optimized within the generative framework.

Another limitation is the insufficient temporal modeling. Temporal consistency is achieved only through latent interpolation, without explicit mechanisms for capturing temporal dependencies (e.g., diffusion transformers or cross-frame attention). As a result, the model may fail to handle rapid phoneme transitions or complex lip dynamics.

The occlusion-aware inference further presupposes access to a large external segmentation model, which introduces additional computational latency and limits deployability.

The proposed LipLeak metric is a reasonable attempt to quantify expression leakage, but its formulation remains limited and lacks validation against perceptual judgments.

The paper also overclaims that “most models are constrained to low-resolution (256×256) outputs.” In fact, many recent methods already generate 512×512 LatentSync 1.6 [1].

[1] LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision

### Questions
SAM2 does not produce semantic class labels or fine-grained facial parsing. How to prevent it from misclassifying mouth interiors (teeth, tongue) or shadows as separate foreground objects? Was the accuracy of SAM2 segmentation evaluated？

### Soundness
3

### Presentation
3

### Contribution
2
