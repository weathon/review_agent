=== CALIBRATION EXAMPLE 44 ===

# Harsh Critic Review
Now I have a thorough read of the paper. Let me write the full review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "AnyAvatar: Dynamic and Consistent Audio-Driven Human Animation for Multiple Characters" is broadly accurate but somewhat marketing-driven. "AnyAvatar" implies broad generality while the system still requires face masks, emotion reference images, and per-character audio segments split by time slot. The abstract claims the method "surpasses state-of-the-art methods on benchmark datasets," which is not fully supported by the actual results in Tables 1 and 2 (the method loses on several individual metrics to different baselines). The abstract is otherwise informative and clearly states the three technical contributions.

---

### Introduction & Motivation

The problem motivation is genuine and well-organized around three challenges. The survey of prior limitations—static backgrounds in portrait methods, unnatural movement in full-body methods, and inability to drive multiple characters independently—is legitimate. However, **the contributions are somewhat oversold relative to their actual scope**:

- The introduction implies "multi-character dialogue" is fully addressed, but Appendix A.5 reveals a critical restriction: characters cannot speak different content simultaneously. Two characters can only be driven independently by time-sliced audio. This is a fundamental limitation that should be disclosed upfront, not buried in the appendix.
- The claim in Section 1 that the approach resolves "the trade-off between dynamism and consistency" via the CIIM is presented as a solved problem, but Table 3 shows that the chosen approach (Token + Add) loses on IP and LS metrics to "Token + Channel," indicating a real trade-off still exists—not an elimination of it.

---

### Method / Approach

**Section 3.1 — Character Image Injection Module (CIIM):**
The paper explores three conditioning mechanisms and selects mechanism (c): the reference image is repeated T times, encoded by Tokenizer2, and *added* to video latents via projection layers before token-concatenating with the reference image from Tokenizer1. This is a reasonable design choice, but the justification is thin. There is no discussion of *why* addition in latent space is preferable to channel concatenation beyond citing empirical ablation results. The paper also does not explain why using the backbone tokenizer weights to initialize Tokenizer2 (Section 3.1) accelerates convergence—a potentially interesting finding is left unexplained. The full formal description of this module is deferred to Appendix A.4, leaving the main method section underspecified.

**Section 3.2 — Face-Aware Audio Adapter (FAA):**
The spatial cross-attention formulation (Eq. 2) is reasonable. However, there are significant underspecification and correctness concerns:

- Equation 2 shows `yt,A'' = yt,A' + αA × CrossAttn(gA, yt') × gM`. The face mask `gM` is multiplied element-wise with the cross-attention output. This means audio features are *zeroed* outside the face region but audio features from *other characters' masks* still freely influence the whole video latent in regions not zeroed. The paper does not explain how the mask interaction between two characters is managed when both are in the same frame simultaneously.
- The temporal alignment procedure (Eq. 1 and surrounding text) is adequately described, but the claim that a face mask value of 1 is set for the initial (identity) frame implies audio influence leaks into the identity frame, which could cause identity drift.

**Section 3.3 — Audio Emotion Module (AEM):**
This module requires an **emotion reference image** as an additional mandatory input. This is a substantial practical constraint: users must source or generate an appropriate emotion reference image for each segment. The paper does not discuss how sensitive the output is to the choice of this image, whether the same emotion image can be used for different audio segments with varying emotional content, or how it interacts with the audio's own implicit emotional cues (Whisper features). The ablation (Figure 7a) only compares against text-only guidance, not against extracting emotion directly from audio (e.g., via speech emotion recognition), leaving the design choice unjustified.

Furthermore, the observation that AEM only works in Double Blocks and not Single Blocks (last paragraph of Section 3.3) is an interesting and potentially publishable finding, but it is presented as a single sentence without explanation or analysis.

**Section 3.4 — Long Video Generation:**
The algorithm (Algorithm 1 in Appendix) uses circular padding when segment indices exceed the audio sequence length (`s ← s % l`, `e ← e % l`). This means the algorithm wraps around and reuses earlier audio frames to fill the end of the video. This is conceptually problematic for maintaining temporal coherence and would cause a character to appear to "repeat" earlier speech patterns. No analysis or ablation of this behavior is provided.

Additionally, this section is described both in Section 3.4 and again in Appendix A.5 with substantial redundancy. The method itself is directly adopted from Sonic (Ji et al., 2024), with the only adaptation being to HunyuanVideo's MM-DiT architecture. This should be clearly labeled as an engineering adaptation rather than a novel contribution.

---

### Experiments & Results

**Dataset and Evaluation Setup (Section 4.1):**
The full-body animation test set is **self-constructed** (250 videos, 200 identities). This introduces a significant concern: all competing methods are evaluated on a test set the authors curated. Even with good intentions, there is no guarantee that the chosen distribution of content, identity diversity, or action types does not inadvertently favor the proposed method. A public benchmark would provide more credible evaluation.

**Table 1 (Portrait Animation):**
The paper claims "our method achieves the best performance" in portrait animation. Examining Table 1 specifically:
- On CelebV-HQ: AnyAvatar does **not** win on FVD (445.02 vs. Sonic's **415.04**) and Sync-D (8.55 vs. Sonic's **8.31**).
- On HDTF: AnyAvatar does **not** win on Sync-C (5.30 vs. Sonic's **5.81**).
  
These are consequential failures for a system whose primary purpose is audio synchronization—Sync-C directly measures audio-visual synchronization confidence. Sonic outperforms on this metric, which raises the question of whether AnyAvatar is actually superior for the core task.

**Table 2 (Full-Body Animation):**
The claim that AnyAvatar achieves "best performance on most evaluation metrics" is questionable on scrutiny:
- WanS2V outperforms on IQA (**4.812** vs. 4.668), Sync-D (**7.957** vs. 8.535), HKV (**0.413** vs. 0.390), and FBN (**4.50** vs. 3.88).
- Multitalk outperforms on FVD (**613.213** vs. 650.541).
- OmniHuman-1 outperforms on FCN (**4.11** vs. 3.91).
- AnyAvatar wins on AES, Sync-C, FID, HKC, IP, and LS—six of twelve metrics.

The authors partially explain the FCN/FBN gap by noting OmniHuman-1 uses super-resolution at inference (Section 4.2), but make no such explanation for FVD losses to MultiTalk or FBN/Sync-D losses to WanS2V. These are real weaknesses that should be acknowledged clearly rather than glossed over.

**Table 3 (CIIM Ablation):**
Critically, this ablation uses **subjective user ratings** (VQ, MD, IP, LS on 1–5 scales) rather than objective metrics. There is no indication of how many participants rated these ablation variants, or whether the rating was blinded. For ablation studies at a venue like ICLR, this is non-standard and insufficient. Objective metrics (FID, FVD, Sync-C) should accompany subjective scores.

**Table 4 (Face Mask Injection Ablation):**
The FID score for "w mask" (74.087) is **worse** than "w/o mask" (72.124) and "w token" (73.124). The paper states "our method outperforms the other two approaches in both background evaluation metrics and overall video quality," which is factually incorrect for FID. While the authors highlight IQA, AES, and FVD improvements, FID is a standard and important metric and the unexplained degradation deserves acknowledgment.

**Missing Ablations:**
1. No quantitative evaluation of the **emotion alignment** quality—there is no emotion classification accuracy or perceptual alignment metric, only a side-by-side qualitative figure (Figure 7a).
2. No quantitative evaluation of **multi-character animation**—the most novel claimed contribution is only demonstrated qualitatively (Figures 5, 6, 8).
3. No analysis of the **emotion reference image sensitivity**: what happens when the reference image emotion mismatches or conflicts with the audio emotion?
4. No evaluation of **long video quality degradation** over time: the circular padding issue and coherence across segments are not quantitatively assessed.

**User Study:**
30 participants evaluating 30 videos per method across 4 dimensions is a reasonable but small study. Statistical significance tests (e.g., Wilcoxon signed-rank) are absent. Given that AnyAvatar loses on FCN and FBN (full-body and facial naturalness) vs. OmniHuman-1, and the authors attribute this to super-resolution rather than model quality, an ablation where OmniHuman-1 output is downsampled to match resolution would clarify whether this explanation holds.

---

### Writing & Clarity

There are a few structural issues that impede understanding:

1. **Section ordering**: Section 3.4 (Long Video Generation) appears before Section 3.3 (Audio Emotion Module) in the text file (though this may be a PDF parsing artifact). If genuinely misplaced, it disrupts the logical flow.
2. **Self-promotion**: Phrases like "extensive qualitative and quantitative experiments demonstrate that AnyAvatar outperforms existing methods" (Section 5) are not consistently supported by the data and should be tempered.
3. The multi-character limitation (only time-split audio, no simultaneous different speech) is mentioned in Appendix A.5 but not in the main paper or Limitations section. Given this is the paper's most distinctive contribution, the architectural constraint must be disclosed in Section 3 or 4.

---

### Limitations & Broader Impact

**Appendix A.8** (referenced but truncated in available text) presumably addresses limitations. The acknowledged limitation on multi-character simultaneous speech is important and should appear in the main paper. Additional missing limitations:

1. **Emotion reference image dependency**: Requiring a separate emotion reference image at inference time is a significant usability barrier not discussed as a limitation.
2. **Computational cost**: Training on 160 GPUs × 96GB is extremely resource-intensive. While code will be released, reproduction of training is practically infeasible for most researchers.
3. **Circular padding in long video generation**: The wrap-around behavior in Algorithm 1 can produce incoherent outputs for audio longer than the generation segment length; this failure mode is not discussed.
4. **Backbone inheritance**: The paper acknowledges "inherent issues of HunyuanVideo" affect FCN and FBN scores, but does not characterize these issues (e.g., specific motion artifacts, generation failures).
5. **Deepfake risk**: The ethics section discusses safeguards at a high level, but deepfake generation of real individuals is directly enabled by this system and deserves more concrete mitigation analysis.

---

### Overall Assessment

AnyAvatar addresses a timely and practically relevant problem—audio-driven full-body animation with character consistency, emotion alignment, and multi-character support—and presents three technically coherent modules built atop HunyuanVideo (MM-DiT). The engineering is competent and the generated results appear visually strong based on the qualitative figures. However, the paper has substantive weaknesses that would give ICLR reviewers pause. The quantitative evidence for "state-of-the-art" performance is inconsistent: the method loses on Sync-C (its primary task) to Sonic in portrait animation and loses on multiple metrics to WanS2V and MultiTalk in full-body animation. The ablations for the paper's most novel contributions (AEM and FAA) are purely qualitative, and the full-body test set being self-constructed without a public release introduces evaluation bias. The multi-character contribution—arguably the most novel aspect—is limited to time-sliced audio and cannot support simultaneous different-content speech, a critical constraint buried in the appendix rather than disclosed prominently. The intellectual novelty of each individual module is incremental (latent masking for multi-person, cross-attention for emotion, addition-based conditioning), and the long video generation method is directly borrowed from Sonic. As submitted, the paper falls below the acceptance bar for ICLR, primarily due to insufficient and inconsistently reported quantitative support for its claims, inadequate ablation rigor, and a significant practical limitation on the headline multi-character feature that is not adequately surfaced.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes AnyAvatar, a multimodal diffusion transformer (MM-DiT)-based framework for audio-driven human animation that addresses challenges in motion dynamics, character consistency, and multi-character support. It introduces three key innovations: a Character Image Injection Module to decouple identity and motion, an Audio Emotion Module for style control, and a Face-Aware Audio Adapter for independent character driving. Experimental results on portrait and full-body benchmarks suggest the method outperforms existing state-of-the-art approaches in synchronization and visual quality.

### Strengths
1.  **Effective Architecture Modification for Consistency:** The Character Image Injection Module (CIIM) demonstrates a thoughtful analysis of the "dynamism-consistency" trade-off common in DiT video generation. The ablation study (Table 3, Fig 3) provides concrete evidence that adding reference latents directly in the channel dimension preserves identity better than token concatenation while maintaining higher motion diversity compared to padding strategies.
2.  **Novel Multi-Character Implementation:** The Face-Aware Audio Adapter (FAA) represents a significant practical contribution by enabling the generation of multi-character scenes in a single forward pass. By using latent-space masking to decouple audio influence, the method allows distinct audio inputs to drive specific face regions, which is a capability currently limited in most portrait or full-body animation systems.
3.  **Comprehensive Evaluation Pipeline:** The authors validate their method across both portrait (CelebV-HQ, HDTF) and custom full-body datasets. The inclusion of subjective user studies alongside objective metrics (Sync-D, FVD, HKV) and detailed ablation studies on all proposed modules (Sections 4.2, A.7) adds credibility to the claimed advancements.

### Weaknesses
1.  **Inconsistent Terminology and Capabilities:** There is a discrepancy between the Abstract claiming "multi-character dialogue videos" and the Limitations section (Appendix A.5), which states: "it is currently not possible to support scenarios where different characters speak different lines simultaneously." While "crosstalk" is supported, true dialogue with independent speech streams is not yet achieved, potentially overstating the contribution in the summary.
2.  **Dependency on Reference Images for Emotion:** The Audio Emotion Module (AEM) (Section 3.3) relies on a specific *emotion reference image* to set the emotional tone, rather than inferring emotion directly from the audio content (e.g., sentiment in text/voice). This adds operational complexity and reduces the "end-to-end" utility of the system (Appendix A.8).
3.  **Computational Efficiency:** The inference cost is reported as high (approx. 60 minutes to generate 10 seconds of video at 720p, Section A.5). While acceleration techniques (4x speedup mentioned) are noted, significant speed improvements are still required for real-time or consumer applications, which is a critical factor for ICLR's interest in scalable generative models.
4.  **Baseline Comparisons:** Some comparisons rely on closed-source services (e.g., OmniHuman-1) or services with post-processing (super-resolution), as noted in the text. This can bias subjective scores (Table 2 discussion) and make the comparison less rigorous than evaluating against open-weight models with identical evaluation pipelines.

### Novelty & Significance
**Novelty:** The core novelty lies in the specific architectural adaptations to the MM-DiT backbone to handle audio conditioning for multi-character consistency. While the underlying models (HunyuanVideo, Whisper) are existing, the CIIM and FAA modules offer distinct technical solutions to problems specific to multi-subject generation. However, the reliance on pre-existing large-scale video foundations reduces the theoretical novelty compared to proposing a wholly new generative architecture.
**Significance:** The significance is high for practical applications. Achieving consistent, dynamic full-body animation with multi-character support addresses a major gap in current generative video tools (e.g., for virtual production or digital assistants). If the limitations regarding speed and emotion inference are addressed in future work, this framework could set a new standard for audio-driven human animation.

### Suggestions for Improvement
1.  **Clarify Multi-Character Scope:** Revise the Abstract and Contribution list to accurately reflect the current limitation regarding independent speech for different characters. Consider adding a quantitative metric or qualitative example specifically showcasing the "crosstalk" generation to justify the "dialogue" claim.
2.  **Explore Audio-Only Emotion Extraction:** Investigate integrating Large Language Models or Audio Classification modules to infer emotional intensity directly from the audio/text input, removing the need for a separate emotion reference image input in the AEM.
3.  **Standardize Baseline Evaluation:** Where possible, re-evaluate closed-source baselines using the same inference pipeline and resolution settings used for AnyAvatar to ensure fair comparison in FID/FVD metrics, or explicitly discuss the normalization factors applied to these metrics.
4.  **Detail Inference Optimization:** While acceleration (4x speedup) is mentioned briefly, provide a more detailed breakdown of inference time scaling with video length and resolution to better contextualize the "60 minutes" claim for potential users.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **True Multi-Character Dialogue Evaluation:** Evaluate with distinct audio tracks per character instead of shared audio; the current setup (Appendix A.5) forces characters to speak the same lines, which directly contradicts the Abstract's claim of "multi-character dialogue generation."
2. **Quantitative Emotion Alignment Metrics:** Include emotion recognition accuracy (e.g., via FACET or pretrained classifiers) on generated videos; currently, claims of "accurate emotion alignment" rely solely on qualitative visuals (Figure 7) without numerical verification.
3. **Public Benchmark or Dataset Release:** Evaluate on a public full-body dataset or release the constructed test set; relying on a private "wild dataset" prevents verification of the claimed SOTA full-body performance and violates reproducibility standards.
4. **Controlled Baseline Comparison:** Re-evaluate comparisons against OmniHuman-1 without its super-resolution post-processing; the paper admits this gives baselines a "natural visual advantage" (Section 4.2), making the claimed superiority in video quality unreliable.

### Deeper Analysis Needed (top 3-5 only)
1. **Emotion Reference Dependency:** Quantify performance degradation when the emotion reference image is removed or mismatched; the method shifts the task from audio-driven to audio+image-driven without analyzing how much the audio actually contributes to emotion.
2. **Identity-Motion Trade-off Contradiction:** Analyze why the proposed injection method yields lower Identity Preservation (IP) than "Token + Channel" in Table 3; this undermines the core claim of resolving the consistency-dynamics trade-off if identity scores drop.
3. **Computational Feasibility vs. Claims:** Analyze the inference cost relative to the claimed "practical application scenarios"; admitting 60 minutes for a 10s video (Appendix A.8) contradicts the Abstract's promise of real-world applicability.

### Visualizations & Case Studies
1. **Audio-Image Emotion Conflict:** Show cases where audio emotion contradicts the reference image to expose whether the model prioritizes audio or the image; this reveals if the AEM actually learns audio-emotion mapping or just copies the reference.
2. **Multi-Character Failure Modes:** Visualize attempts to drive characters with different audio streams to demonstrate the current limitation explicitly; this clarifies the actual capability versus the claimed "dialogue" generation.
3. **Long-Term Identity Drift:** Show frame-by-frame identity consistency over >50 seconds; segment-wise generation (Algorithm 1) risks accumulation of drift not visible in short clips, challenging the "infinite-length" stability claims.

### Obvious Next Steps
1. **Clarify Task Definition:** Explicitly state that emotion control requires an extra image input rather than implying pure audio-driven emotion alignment in the Abstract and Introduction.
2. **Release Test Set:** Publicly release the constructed full-body test set to ensure reproducibility and fair comparison against future methods, as required for ICLR acceptance.
3. **Integrate Acceleration:** Include distillation or sampling acceleration in the main method rather than relegating real-time performance to future work, given the claim of "practical applicability."

# Final Consolidated Review
## Summary

AnyAvatar proposes an MM-DiT-based framework for audio-driven human animation addressing three challenges: motion dynamics with character consistency, emotion alignment, and multi-character generation. The method introduces a Character Image Injection Module (CIIM) that adds reference latents to video latents via projection layers, an Audio Emotion Module (AEM) requiring emotion reference images for expression control, and a Face-Aware Audio Adapter (FAA) using latent-space masking to enable per-character audio driving.

## Strengths

- **Novel Multi-Character Architecture:** The Face-Aware Audio Adapter (FAA) enables independent audio control for different characters within a single scene by masking face regions in latent space and applying spatial cross-attention. This is a practical and technically sound solution to a real limitation in existing audio-driven animation systems, and the ablation (Table 4) demonstrates that masking does not harm background preservation (90% success rate on static backgrounds).

- **Systematic Exploration of Conditioning Mechanisms:** The Character Image Injection Module ablation (Table 3, Figure 3) compares three conditioning strategies—token concatenation, token+channel concatenation, and token+addition—with clear trade-offs: "Token" achieves highest motion diversity (MD=3.585) but lowest identity preservation (IP=4.402), while "Token+Channel" preserves identity best (IP=4.576) but restricts dynamics (MD=2.336). The chosen "Token+Add" approach balances both (MD=4.127, IP=4.289), providing empirical grounding for the design choice.

- **Strong Quantitative Performance on Key Metrics:** On portrait animation (Table 1), AnyAvatar achieves best FID (38.01 on HDTF), IQA (3.99 on HDTF), and AES scores. On full-body animation (Table 2), it achieves best Sync-C (7.534), FID (49.380), HKC (hand quality, 0.849), IP (identity preservation, 4.84), and LS (lip sync, 4.65) among baselines.

## Weaknesses

- **Overstated "State-of-the-Art" Claims:** The Abstract claims AnyAvatar "surpasses state-of-the-art methods," but Tables 1 and 2 show losses on several key metrics. On portrait animation, AnyAvatar loses FVD to Sonic on CelebV-HQ (445.02 vs 415.04) and Sync-C on HDTF (5.30 vs 5.81). On full-body animation, WanS2V achieves better IQA (4.812 vs 4.668), Sync-D (7.957 vs 8.535), and FBN (4.50 vs 3.88); MultiTalk achieves better FVD (613.21 vs 650.54). The claims should be qualified rather than absolute.

- **Multi-Character Capability Is Limited to Same-Line Speech:** The Abstract and Introduction claim "multi-character dialogue videos," but Appendix A.5 discloses that "it is currently not possible to support scenarios where different characters speak different lines simultaneously." Characters can only be driven independently when audio is time-sliced—two characters cannot speak different content at the same time. This is a fundamental architectural constraint that should be prominently disclosed in the main paper, not buried in the appendix.

- **Emotion Reference Image Dependency:** The Audio Emotion Module requires an additional input—an emotion reference image—rather than inferring emotion directly from audio. This shifts the task from purely audio-driven to audio+image-driven, adding operational complexity without analysis of how performance degrades when this input is absent or mismatched. The ablation (Figure 7a) only compares against text-only guidance, not against audio-only emotion extraction.

- **Self-Constructed Test Set for Full-Body Evaluation:** The full-body animation benchmark is a private dataset of 250 videos (200 identities) curated by the authors. While there is no existing public full-body benchmark, the lack of independent verification makes the claimed superiority on full-body animation difficult to assess objectively. Releasing this test set publicly would strengthen reproducibility.

- **Ablation Studies Lack Quantitative Rigor:** The CIIM ablation (Table 3) uses subjective user ratings (VQ, MD, IP, LS on 1-5 scales) without specifying sample size, participant selection, or blinding procedure. Standard objective metrics (FID, FVD, Sync-C) should accompany subjective scores for a venue like ICLR. Furthermore, there is no quantitative evaluation for the two most novel contributions—emotion alignment quality and multi-character animation quality.

## Nice-to-Haves

- **Quantitative Emotion Alignment Evaluation:** Include emotion classification accuracy (e.g., via pretrained classifiers or human emotion recognition metrics) to substantiate claims of "accurate emotion alignment."

- **Audio-Only Emotion Extraction:** Investigate integrating audio-based emotion recognition to remove the requirement for emotion reference images, making the system more end-to-end.

- **Public Test Set Release:** Release the constructed full-body test set to enable reproducible comparison for future work.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Section Ordering Complaint:** The harsh critic claimed Section 3.4 appears before 3.3, but this is a PDF parsing artifact—the paper correctly presents Section 3.3 (AEM) before Section 3.4 (Long Video Generation).

- **Computational Cost as Core Weakness:** The inference time (60 minutes for 10s video) is acknowledged in Appendix A.8 as a limitation with ongoing acceleration efforts. Large computational requirements are standard for video diffusion models and not unique to this work.

- **Circular Padding Criticism:** The circular padding in Algorithm 1 is a standard technique for handling sequence boundaries during segment-wise denoising; it does not cause characters to "repeat" earlier speech patterns as claimed.

- **Long Video Method Non-Novelty:** The time-aware position shift fusion is correctly attributed to Sonic (Ji et al., 2024) in Appendix A.5. Building on prior work with proper attribution is acceptable, not a weakness.

## Novel Insights

The observation that the Audio Emotion Module works effectively in Double Blocks but not Single Blocks (Section 3.3) is an interesting architectural finding: "integrating it into a Double Block enables the model to better drive character emotions. This suggests that the Double Block plays a crucial role in capturing and representing emotional details." This suggests Double Blocks in MM-DiT architectures may have different functional specializations worth investigating further.

The face-aware latent masking strategy for multi-character control is technically elegant—it avoids complex architectural modifications and enables independent audio driving by simply masking the cross-attention output. The ablation (Table 4) confirms this does not introduce background distortion (BD improves from 0.0027 to 0.0028, DB improves from 87% to 90% success).

## Suggestions

- **Revise the Abstract and Introduction** to accurately reflect multi-character capabilities (e.g., "multi-character animation with time-sliced audio" rather than "multi-character dialogue") and qualify SOTA claims to acknowledge metrics where baselines outperform.

- **Move the Multi-Character Limitation from Appendix A.5 to the Main Paper** (Section 3.2 or 4) so readers understand the architectural constraint upfront.

- **Add Quantitative Metrics for Emotion and Multi-Character Evaluation:** Even a simple metric such as per-character sync accuracy or emotion classification rate would substantiate these key contributions beyond qualitative figures.

- **Clarify CIIM Ablation Methodology:** Specify the number of participants, rating protocol, and ideally include standard objective metrics (FID, FVD, Sync-C) alongside subjective scores in Table 3.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 6.0]
Average score: 4.5
Binary outcome: Reject
