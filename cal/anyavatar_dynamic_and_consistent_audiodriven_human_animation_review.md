=== CALIBRATION EXAMPLE 24 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title promises "dynamic and consistent audio-driven human animation for **multiple characters**," yet Section A.5 (Appendix) quietly reveals that the multi-character capability cannot support "different characters speak[ing] different lines simultaneously or where interruptions occur." The title and abstract overstate what is actually delivered. An honest characterization would be "sequential turn-taking audio-driven animation," not general multi-character dialogue. The abstract's claim that AnyAvatar "simultaneously generating dynamic, emotion-controllable, and multi-character dialogue videos" is thus misleading.

Additionally, the abstract describes the AEM as extracting "emotional cues from an emotion reference image," but presents it as an audio-emotion alignment module. The emotion source is actually a *separate input image*, not the audio itself — the audio merely provides synchronization cues, not the emotional content. This conflation obscures what the system actually does.

### Introduction & Motivation

The three challenges are genuinely present in the literature and the motivation is reasonable. However, the introduction over-claims novelty: the character image injection idea (conditioning via latent-space addition rather than padding frames) is a natural next step explored in image-to-video customization, and the face-mask-based audio routing bears close resemblance to masked inpainting in diffusion models. The paper would benefit from being more precise about what is truly novel versus what is a natural combination of existing ideas.

The related work section is unusually thin and purely descriptive — it summarizes prior methods but makes no argument about what gaps they leave that this paper fills. This structure makes the motivation feel disconnected from the method.

### Method

**Section 3.1 – Character Image Injection Module (CIIM):**

The paper explores three mechanisms and finds that mechanism (c) — adding a repeated image latent element-wise to the video latent, then concatenating with the reference token — works best. However, the rationale is entirely post-hoc empirical observation; there is no theoretical or intuitive justification for why element-wise addition in latent space resolves the dynamism-consistency trade-off better than channel concatenation. The reader is left to trust a three-way ablation on subjective metrics only.

Critically, the ablation in Table 3 uses only subjective user scores (LS, VQ, IP, MD). No objective metrics (Sync-C, FID, FVD) are reported for this comparison, which makes it impossible to know whether the qualitative benefit translates to measurable gains.

**Section 3.2 – Face-Aware Audio Adapter (FAA):**

The core idea is to apply a face bounding-box mask ($g_M$) to gate the audio cross-attention update, restricting audio influence to specific facial regions. While functional, this is essentially masked conditioned generation — a well-known technique in inpainting and controllable generation — applied to audio injection. The paper compares against MultiTalk (Kong et al., 2025), which uses a more principled Label Rotary Position Embedding (Label RoPE) for audio-person binding, but does not discuss why the mask approach is preferable or how it compares conceptually.

The multi-character limitation (Section A.5) deserves prominent mention in the main text: the model splits audio sequentially (first half = character A, second half = character B), which means characters cannot overlap, interrupt, or truly converse. This is a substantial restriction on the stated goal of "multi-character dialogue generation."

**Section 3.3 – Audio Emotion Module (AEM):**

The module injects emotion by attending to a **user-supplied emotion reference image**, not by parsing emotional content from the audio signal. This design choice creates two problems acknowledged in Section A.8 but insufficiently foregrounded: (1) users must supply an extra input (significant practical burden), and (2) the system cannot handle dynamic within-utterance emotion changes. For a system claimed to "align video characters' emotions with those conveyed in the audio," the actual emotion source is not the audio at all. The naming is confusing and the design falls short of the stated goal.

The observation that the AEM must be inserted into **Double Blocks** (not Single Blocks) to work well is interesting but unexplained. No analysis of why Double Blocks' joint text-video attention is necessary for emotion transfer is provided.

**Section 3.4 – Long Video Generation:**

The authors directly adopt the Time-aware Position Shift Fusion method from Sonic (Ji et al., 2024), adapted to HunyuanVideo. This is explicitly acknowledged: "We use the Time-aware Position Shift Fusion method from Sonic." As a result, long video generation is not a contribution of this paper, yet it is presented as a feature and used in comparisons.

**Equation-level concern:** Equation (1) shows audio rearranging but the notation `Rearrange(gA, 0)` is undefined — the second argument "0" is unexplained. Similarly, the transition between Equations (2) and (3) is abrupt; the relationship between the face-mask-gated audio update (Eq. 2) and the emotion injection (Eq. 3) is not articulated clearly. It is unclear whether both operate on the same latent sequentially, and in what order.

### Experiments & Results

**Quantitative results — misleading "best" claims:**

The paper asserts "our method achieves the best performance" (Section 4.2) but a careful reading of Tables 1 and 2 reveals selective wins:

- *Table 1 (Portrait)*: AnyAvatar loses to Sonic on Sync-C for both datasets (4.92 vs. 5.58 on CelebV-HQ; 5.30 vs. 5.81 on HDTF). It also loses to Sonic on FVD for CelebV-HQ (445.02 vs. 415.04). Sync-C is arguably the most important metric for a lip-sync paper.
- *Table 2 (Full-body)*: AnyAvatar loses to WanS2V on IQA (4.668 vs. 4.812), FVD (650.541 vs. 613.213), Sync-D (8.535 vs. 7.957), and HKV (0.390 vs. 0.413). It loses to OmniHuman-1 and WanS2V on FCN and FBN user scores, and to OmniHuman-1 on FCN. The paper's own text admits OmniHuman-1 uses super-resolution in its online service, making the subjective comparison unfair — yet OmniHuman-1 is still included in the quantitative table.

**Evaluation benchmark for full-body:**

The full-body evaluation uses a self-constructed test set (250 videos, 200 identities) with no external validation. This raises an obvious concern about potential bias in benchmark construction. No established public full-body benchmark is used as a cross-check. Given that all competing methods are evaluated on this dataset, the authors' ability to curate it to favor their model's strengths cannot be ruled out.

**Emotion alignment — no quantitative evaluation:**

The AEM is claimed as a key contribution, yet its evaluation consists solely of a qualitative comparison in Figure 7(a). No established emotion recognition metrics (e.g., emotion classification accuracy on generated frames, FER scores, or perceptual alignment ratings) are reported. This is a significant gap for what is presented as a core technical contribution.

**Multi-character evaluation:**

Similarly, the multi-character capability is evaluated only qualitatively (Figure 6), with no quantitative metrics specifically targeting per-character lip sync accuracy or audio-character binding.

**User study:**

30 participants is a small cohort for a study across 7+ methods, with 30 videos each. The paper does not report inter-annotator agreement, confidence intervals, or any statistical significance tests. Given fatigue effects from evaluating large numbers of videos, the user study results have limited statistical reliability.

**Ablation on CIIM (Table 3):**

The ablation compares three injection mechanisms but does so *subjectively only*. There is no objective counterpart. Furthermore, "Motion Diversity (MD)" is an informal subjective label with no standard definition — it is unclear what the annotators were told to evaluate.

**Ablation on FAA (Table 4):**

The FID results show that the "w mask" condition *increases* FID relative to "w/o mask" (74.087 vs. 72.124), which is a worse score on a key metric. The paper does not address this discrepancy — instead, it focuses on IQA and AES while glossing over the FID result.

### Writing & Clarity

The Methods section has a structural issue: Section 3.3 (AEM) appears after Section 3.4 (Long Video Generation) in the parsed text, suggesting either a section ordering error or a parsing artifact. Regardless, the logical flow of the methods presentation is non-linear and harder to follow than necessary.

The distinction between the backbone model (HunyuanVideo-I2V) and the authors' contributions is not always clearly delineated. For instance, the 3D-RoPE spatial shift trick (Eq. 6) cites Omnicontrol (Tan et al., 2024) but it is unclear whether this is the authors' novel application or an existing feature of HunyuanVideo-I2V.

### Limitations & Broader Impact

Section A.8 honestly acknowledges the three key limitations: (1) emotion requires a reference image rather than being inferred from audio, (2) inference is extremely slow (60 min for a 10s video), and (3) multi-character simultaneous speech is not supported. These limitations are appropriate and well-stated. However, they are consigned to the appendix rather than the main paper, which is problematic because they directly pertain to the three core contributions. At a minimum, the multi-character limitation should be stated clearly in Section 3.2 or 4.2.

The deepfake and misuse risks are taken seriously in Section A.3 and A.9, with watermarking, consent requirements, and public figure restrictions discussed. This is commendable and among the more thorough ethics discussions in this space.

---

### Overall Assessment

AnyAvatar is a well-motivated engineering system built on the strong HunyuanVideo-I2V backbone, combining three components (CIIM, AEM, FAA) to address dynamism, emotion alignment, and multi-character scenarios in audio-driven human animation. The qualitative results look compelling, and the quantitative results show competitive performance. However, the paper has several substantive weaknesses that undermine its ICLR acceptance case. First, the novelty of each individual component is incremental: CIIM is a variant of known latent conditioning strategies, FAA is essentially masked inpainting applied to audio injection, and the long video method is directly borrowed from Sonic. Second, the evaluation has methodological problems: results are described as "best" when AnyAvatar clearly loses on important metrics (Sync-C vs. Sonic, IQA/FVD vs. WanS2V); the full-body benchmark is self-constructed and potentially biased; the AEM has no quantitative evaluation despite being a central contribution; and multi-character performance is assessed only qualitatively. Third, the AEM's core design — using a static reference image rather than inferring emotion from audio — is a fundamental mismatch with the claimed goal of audio-emotion alignment, and this limitation is insufficiently prominent. The multi-character capability is also considerably more restricted than the title implies. In its current form, the paper reads more as a strong system paper suitable for an application or vision conference than an ICLR paper, which expects either significant methodological novelty or rigorous scientific evaluation. Substantial revisions are needed to either sharpen the technical novelty, fix the evaluation gaps, or honestly recalibrate the claims.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces AnyAvatar, a Multimodal Diffusion Transformer framework designed for audio-driven human animation that simultaneously targets motion dynamism, identity consistency, emotion alignment, and multi-character control. It proposes three architectural components: an additive character image injection module to decouple reference conditions from latent spaces, an Audio Emotion Module that transfers affective cues via reference image cross-attention, and a Face-Aware Audio Adapter that uses latent-level facial masking for per-speaker audio binding. Comprehensive experiments on portrait, full-body, and custom wild datasets demonstrate that the approach achieves competitive or superior performance across synchronization, video quality, and identity preservation metrics relative to recent state-of-the-art baselines.

### Strengths
1. **Clear Problem Formulation & Unified Pipeline:** The paper directly addresses three persistent bottlenecks in audio-driven generation (dynamism-consistency trade-off, emotion misalignment, and single-speaker limitations) and integrates solutions into a single HunyuanVideo-based architecture, demonstrating strong systems engineering.
2. **Extensive & Multi-Faceted Evaluation:** The authors evaluate on standard benchmarks (CelebV-HQ, HDTF) and a newly constructed 250-video full-body test set, employing a wide array of objective metrics (Sync-C/D, FID/FVD, HKC/HKV, IQA/AES) alongside a structured user study. Quantitative results in Tables 1–3 consistently show strong competitive performance.
3. **Empirically Grounded Design Choices:** The ablation studies in Section 4.2 (e.g., Table 3 comparing token concat, channel concat, and additive injection; Table 4 evaluating mask injection strategies) provide concrete, evidence-backed justification for the selected architectures and hyperparameters, reinforcing methodological soundness.
4. **Commitment to Reproducibility & Ethics:** The paper explicitly details dataset composition (500k samples, 1250 hours, OpenhumanVid + self-collected), training infrastructure (160 GPUs, batch size 40, 1e-5 LR), and ethical safeguards (consent forms, bias mitigation, watermarking, abuse prevention). The promise to release code and weights aligns well with ICLR reproducibility standards.

### Weaknesses
1. **Incremental Technical Novelty:** The core mechanisms—additive feature projection, region-specific latent masking, and cross-attention injection—are established paradigms in controllable diffusion literature. While the integration is effective, the paper lacks algorithmic or theoretical innovation that meaningfully advances the foundational understanding of DiT conditioning or audio-video representation learning.
2. **Overstated Emotion Alignment Claims:** Section 3.3 and the abstract emphasize "precise emotion alignment between characters and audio," yet the AEM relies entirely on an *external reference image* to dictate emotion. As explicitly acknowledged in Appendix A.8, this prevents dynamic intra-audio emotion tracking and increases user burden, making the "audio-driven emotion" claim misleading without an audio-native emotion encoder.
3. **Gaps in Evaluation Rigor & Dataset Transparency:** The custom full-body dataset lacks defined train/test splits, synchronization ground-truth verification protocols, and demographic/diversity quantification. Additionally, the subjective study (N=30) reports inferior scores for Facial/Full-Body Naturalness compared to OmniHuman-1 but attributes this to HunyuanVideo's baseline limitations and baseline super-resolution advantages without statistical testing or controlled ablation.
4. **Ambiguous Training & Implementation Details:** Key training specifications are underspecified: stage-by-stage loss formulations, gradient accumulation/clipping, exact GPU memory distribution, and tokenizer initialization strategies are missing or vaguely described. Equation numbering/formatting inconsistencies (e.g., Equations 1–4 in Section 3) and undefined tensor dimensions hinder strict reproducibility independent of the code release.

### Novelty & Significance
**Novelty:** Moderate. The work does not propose new diffusion objectives or transformer primitives but competently orchestrates existing conditioning strategies (additive injection, spatial masking, reference-guided cross-attention) into a cohesive pipeline tailored for multi-character audio animation. The engineering integration is robust but conceptually incremental.
**Clarity:** Good, though marred by minor structural ambiguities. The high-level architecture is well-illustrated, but precise mathematical formulations of tensor transformations, RoPE modifications, and attention masking operations lack consistent dimensionality definitions and clean formatting, which can impede reader comprehension.
**Reproducibility:** Promising but currently incomplete. The inclusion of dataset statistics, compute resources, and a commitment to open-source code are strong positives. However, the absence of exact per-stage hyperparameters, loss weighting schedules, and formal dataset filtering pipelines means full reproduction currently depends heavily on the forthcoming codebase.
**Significance:** High. The paper tackles highly practical, industry-relevant challenges in digital human synthesis, multi-speaker dialogue generation, and controllable video animation. Given the rapid adoption of DiT-based video models, AnyAvatar provides a strong, reproducible baseline and practical framework that will likely see substantial community utilization and extension.

### Suggestions for Improvement
1. **Reframe or Extend Emotion Conditioning:** Either explicitly rename the contribution to "reference-guided emotion transfer" to accurately reflect the input modality, or conduct a pilot experiment injecting audio-derived emotion logits/features (e.g., from wav2vec2 or HuBERT emotion heads) to demonstrate true end-to-end audio-to-emotion mapping, particularly handling intra-clip emotion shifts.
2. **Standardize Dataset Reporting & Statistical Validation:** Release the exact curation/filtering pipeline and specify train/validation/test splits for the custom full-body set. Add statistical significance testing (e.g., paired t-tests or bootstrap confidence intervals) to quantitative comparisons, and control for super-resolution effects when interpreting subjective naturalness scores.
3. **Quantify Multi-Character Binding Accuracy:** Beyond qualitative mask ablations, introduce explicit metrics for audio-to-speaker binding (e.g., cross-talk leakage rate, per-character lip-sync accuracy, identity drift during simultaneous speech) to rigorously validate the Face-Aware Audio Adapter in multi-speaker scenarios.
4. **Clarify Training Protocol & Computational Profiling:** Provide a complete training recipe (stage durations, loss weights, optimizer settings, gradient clipping, exact batch-per-GPU mapping). Include an inference benchmark table reporting VRAM usage, wall-clock time per second of generated video, and throughput relative to key baselines to contextualize the 60-minute/10s generation latency and discuss practical deployment trade-offs transparently.
5. **Improve Mathematical Rigor & Notation:** Ensure all equations are sequentially numbered, cleanly formatted, and cross-referenced. Explicitly define tensor shapes for `gA`, `gM`, and all intermediate attention inputs/outputs in Sections 3.1–3.3, and clarify how RoPE is shifted for the injected image latent to prevent implementation ambiguity.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Independent Multi-Character Audio:** The abstract claims "multi-character dialogue," but Appendix A.5 admits characters must speak the *same* lines. An experiment driving different characters with *distinct* audio tracks is missing and essential to validate the core multi-character claim.
2. **Objective Identity Metrics:** The ablation for character consistency (Table 3) relies on subjective user scores. Add objective face recognition similarity scores (e.g., ArcFace) to quantitatively prove identity preservation without human bias.
3. **Dynamic Emotion Audio Test:** The AEM requires a static emotion reference image. An experiment testing audio with *changing* emotions (e.g., happy to sad) is needed to prove the method fails or succeeds in dynamic scenarios, as current claims imply general emotion alignment.

### Deeper Analysis Needed (top 3-5 only)
1. **Masking Artifacts & Occlusion:** The FAA relies on face masks. Analyze performance when faces overlap or turn away, as latent masking likely causes artifacts in these common multi-character interactions.
2. **Temporal Consistency Drift:** Long video generation uses segment shifting. Provide quantitative drift metrics (e.g., feature variance over time) to prove consistency doesn't degrade over the 50-second clips mentioned.
3. **Metric Discrepancy Explanation:** Table 2 shows WanS2V outperforming Ours on IQA, yet the text claims overall superiority. Analyze why Ours wins on Sync-C but loses on quality metrics to avoid misleading SOTA claims.

### Visualizations & Case Studies
1. **Emotion Transition Failure:** Visualize a case where audio emotion shifts but the reference image remains static, exposing the limitation of the AEM module.
2. **Multi-Character Occlusion:** Show a case study where two characters interact closely (overlap), revealing whether the face-aware masking causes background or body distortion.
3. **Full-Body Motion Comparison:** Side-by-side visualizations of body motion against full-body baselines to verify the "dynamic motion" claim vs. static body baselines.

### Obvious Next Steps
1. **Support for Distinct Audio Tracks:** The current FAA architecture binds audio globally before masking. The architecture must be revised to inject distinct audio features per masked region to support true dialogue.
2. **Automated Emotion Extraction:** The reliance on manual emotion reference images limits usability. Integrate an audio-emotion encoder to remove the need for external image guidance.
3. **Rigorous Objective Ablation:** Subjective ablation scores are insufficient for ICLR. Replace user studies in ablation sections with standardized objective metrics to validate module contributions.

# Final Consolidated Review
## Summary

AnyAvatar proposes a multimodal diffusion transformer (MM-DiT) based framework for audio-driven human animation that addresses three challenges: maintaining character consistency while enabling dynamic motion, aligning emotions between audio and video, and supporting multi-character scenarios. The method introduces three components: a Character Image Injection Module that uses additive latent-space conditioning instead of conventional padding frames, an Audio Emotion Module that transfers emotional cues from a reference image to generated video, and a Face-Aware Audio Adapter that uses latent-level face masks to enable per-character audio control. Experiments on portrait datasets (CelebV-HQ, HDTF) and a newly constructed full-body dataset demonstrate competitive or superior performance on most metrics compared to recent baselines.

## Strengths

- **Unified multi-objective framework:** The paper explicitly targets three well-documented bottlenecks in audio-driven animation—the dynamism-consistency trade-off, emotion misalignment, and single-speaker limitations—and integrates coherent architectural solutions into a single system built on HunyuanVideo, demonstrating strong systems engineering.
- **Comprehensive empirical evaluation:** The authors evaluate on both established benchmarks (CelebV-HQ, HDTF) and a custom 250-video full-body test set, employing a wide range of objective metrics (Sync-C/D, FID/FVD, HKC/HKV, IQA/AES) alongside a structured user study with 30 participants. Tables 1-2 show competitive or leading performance on most metrics including Sync-C, IP, and LS.
- **Transparent ethics and reproducibility commitment:** The paper provides detailed dataset composition (500k samples, ~1,250 hours from OpenHumanVid and self-collection), training infrastructure (160 GPUs, batch size 40, learning rate 1e-5), and thorough discussion of consent, bias mitigation, watermarking, and misuse prevention (Sections A.3, A.9), with explicit commitment to release code and model weights.

## Weaknesses

- **Multi-character capability is significantly more restricted than claimed:** The title and abstract promise "multi-character dialogue," but Appendix A.5 explicitly states that different characters cannot speak different lines simultaneously and interruptions are not supported. The current implementation handles turn-taking (sequential audio segments mapped to different characters via masks), not true simultaneous multi-speaker dialogue. This limitation should be stated prominently in the main text, not buried in the appendix.

- **Audio Emotion Module naming and design mismatch:** The module is named "Audio Emotion Module" and described as aligning "video characters' emotions with those conveyed in the audio," but the emotion source is actually a user-supplied *reference image*—not extracted from the audio signal. Appendix A.8 acknowledges this prevents handling dynamic within-audio emotion changes. The naming is misleading, and the design imposes additional user burden while falling short of true audio-to-emotion alignment.

- **Selective reporting of "best" claims:** The paper states "our method achieves the best performance on most evaluation metrics," but Tables 1-2 show losses on important ones: Sync-C is worse than Sonic on both portrait datasets (4.92 vs 5.58; 5.30 vs 5.81); FVD is worse than Sonic on CelebV-HQ and worse than MultiTalk on full-body; IQA, FVD, Sync-D, and HKV are worse than WanS2V on full-body. The claims should accurately reflect these trade-offs.

- **No quantitative evaluation of emotion alignment:** The AEM is presented as a core contribution, yet Figure 7(a) provides only qualitative comparison. No emotion recognition metrics (e.g., classification accuracy on generated faces, emotion consistency scores) are reported to validate that emotions are actually transferred correctly.

- **Self-constructed full-body test set without external validation:** The 250-video full-body dataset is author-curated with no independent benchmark cross-check. While reasonable for an emerging task, this raises concerns about potential inadvertent bias toward the proposed method's strengths.

- **Key ablation uses only subjective metrics:** Table 3 (CIIM ablation) relies exclusively on subjective user scores (VQ, MD, IP, LS). No objective metrics such as FID, FVD, or identity similarity scores are reported, making it difficult to independently verify the claimed trade-off resolution.

- **Unexplained metric discrepancy in FAA ablation:** Table 4 shows FID is *worse* with the proposed mask (74.087) than without (72.124), yet the paper claims the masked approach "achieves higher overall consistency and better performance." This discrepancy deserves discussion.

## Nice-to-Haves

- **Automated emotion extraction:** Integrating an audio-native emotion encoder (e.g., from wav2vec2 or HuBERT) to derive emotion directly from speech would eliminate the need for external reference images and enable handling of dynamic emotion shifts within utterances.

- **Quantitative multi-character binding metrics:** Introduce metrics measuring cross-talk leakage (wrong character responding to audio), per-character lip-sync accuracy in multi-speaker settings, and identity drift during simultaneous speech to rigorously validate FAA.

- **Occlusion analysis for face masking:** The FAA relies on face bounding boxes; an analysis of performance when characters overlap or faces turn away would clarify robustness in realistic multi-character interactions.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Equation notation nitpicks:** The harsh critic claimed Equation 1's `Rearrange(gA, 0)` is undefined. The "0" is standard PyTorch-style axis specification. This is a formatting nitpick, not a substantive issue.

- **Section ordering complaint:** The critic noted AEM appears after Long Video in the "parsed text," but the paper correctly orders them as Sections 3.3 and 3.4. This was a parsing artifact.

- **"Natural next step" dismissals:** Claims that CIIM is "just a natural next step" or "bears close resemblance to masked inpainting" are vague dismissals without counter-evidence. The additive injection with RoPE spatial shift is a meaningful design choice validated empirically.

- **"More principled" MultiTalk comparison:** The assertion that MultiTalk's Label RoPE is "more principled" than the masking approach is opinion stated as fact. Both are design choices; the paper demonstrates strong empirical results.

- **Related work thickness complaint:** Section 2 covers 9 relevant methods across portrait and full-body animation. Calling this "unusually thin" is overly critical without specifying what's missing.

## Novel Insights

The architecture reveals a fundamental tension in multi-character audio-driven animation: current DiT-based video models inject audio globally via cross-attention, which works for single-speaker settings but creates binding conflicts when multiple characters must respond to distinct audio streams. The paper's latent-space masking solution (FAA) is a practical workaround but requires sequential audio allocation (first half → character A, second half → character B), which fundamentally cannot support overlapping dialogue or interruptions. True simultaneous multi-speaker control would require architectural changes—either per-token audio routing, separate cross-attention branches per character, or audio-person binding at the conditioning level before the diffusion process. Additionally, the paper inadvertently exposes that current video generation benchmarks lack standardized emotion evaluation protocols, leaving authors to rely on qualitative figures for emotion-related claims.

## Suggestions

- **Revise abstract and title to accurately reflect multi-character scope:** Change phrasing from "multi-character dialogue videos" to "multi-character turn-taking animation" and clarify in the main text (Section 3.2 or 4.2) that simultaneous different-speech and interruption scenarios are not currently supported.

- **Rename Audio Emotion Module to reflect its actual function:** Consider "Reference-Guided Emotion Transfer Module" and clarify that emotion is specified via image, not extracted from audio. Add a discussion of this design choice's trade-offs in the main text.

- **Add objective metrics for emotion alignment:** Evaluate generated frames with an off-the-shelf facial expression recognition model (e.g., AffectNet classifier) to quantitatively assess whether emotions from the reference image transfer correctly.

- **Report all metric trade-offs honestly in the main results discussion:** Acknowledge where baselines outperform (Sync-C vs. Sonic, IQA/FVD vs. WanS2V) and explain what AnyAvatar gains versus what it trades off.

- **Add objective identity metrics to CIIM ablation:** Include face recognition similarity (e.g., ArcFace cosine distance) alongside subjective scores to strengthen the identity preservation claim.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 6.0]
Average score: 4.5
Binary outcome: Reject
