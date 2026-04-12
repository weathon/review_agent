=== CALIBRATION EXAMPLE 25 ===

# Final Consolidated Review
## Summary
This paper presents AnyAvatar, an MM-DiT-based audio-driven human animation system built on HunyuanVideo-I2V. The work combines three main components: a character image injection mechanism aimed at improving the dynamics/identity trade-off, a face-aware audio adapter for localized audio control in multi-character scenes, and an emotion-control module that uses a reference image to guide facial expression style. Empirically, the system is competitive on portrait benchmarks and performs strongly on lip sync and identity-related metrics on the authors’ full-body test set.

## Strengths
- **The character image injection design targets a real and important failure mode in image-conditioned video generation: excessive copying of the reference image at the expense of motion.** The paper does more than claim this abstractly: it compares multiple injection schemes in Table 3 and shows a nontrivial trade-off between identity preservation and motion diversity, with the proposed “Token + Add” variant yielding the best motion diversity while keeping video quality competitive.
- **The paper tackles multi-character audio-driven animation through localized conditioning rather than global audio injection.** Section 3.2 explicitly restructures cross-attention to operate per temporally aligned frame and modulates the audio effect with a face mask, which is a concrete architectural choice tailored to the multi-character setting rather than a generic conditioning add-on.
- **The method appears genuinely strong on lip sync / identity-oriented metrics, especially for full-body animation.** In Table 2, the method attains the best Sync-C, FID, HKC, IP, and LS among compared methods, which supports the claim that the proposed conditioning scheme helps preserve character identity while maintaining strong audio-video synchronization.
- **The paper’s scope extends beyond photorealistic humans to diverse character styles.** The visual results and appendix examples indicate the model is intended to generalize across stylized characters (e.g., anime, sketch-like, LEGO-like styles), which is a meaningful practical extension beyond many audio-driven portrait papers.
- **The paper is unusually explicit about limitations and ethics.** In particular, Appendix A.8 candidly states that emotion is not inferred directly from audio and that simultaneous different utterances are not yet supported, which helps separate what the system does from what it does not yet do.

## Weaknesses

### Fatal
- **The paper’s headline claim of “precise emotion alignment between characters and audio” is not technically supported by the actual method.**  
  Section 3.3 describes the Audio Emotion Module as injecting features from an **emotion reference image** into the video latent through cross-attention. There is no mechanism in that section that extracts emotion from audio or maps time-varying affective cues from speech into expressions. The paper itself acknowledges this directly in Appendix A.8:  
  > “our current approach relies on emotion reference images to drive the character’s emotions, rather than allowing the model to infer and generate emotions directly from the audio.”  
  and further notes:  
  > “Since each reference image corresponds to only one emotion, multiple emotions in a single audio segment may result in generation errors.”  
  This does not invalidate the whole paper, but it **does** invalidate the strongest version of the emotion-alignment claim as written in the abstract, introduction, and conclusion. What the paper demonstrates is reference-image-guided emotion style control in an audio-driven animation system, not true audio-to-emotion alignment.

### Major:
- **The empirical evaluation does not quantitatively validate two of the paper’s central claims: emotion alignment and multi-character audio-person binding.**  
  The reported metrics in Tables 1 and 2 focus on overall quality, FID/FVD, smoothness-related proxies, lip sync, hand quality, and subjective preference. None directly evaluate whether the generated expression matches audio emotion, nor whether the correct character is driven by the correct audio stream in multi-character settings. For emotion and multi-character control, the evidence is almost entirely qualitative (Figures 5, 7, 11) plus indirect user scores. For a paper whose abstract foregrounds these two capabilities, the absence of dedicated quantitative validation is a substantial weakness.
- **The “multi-character dialogue” claim is overstated relative to the actual capability described in the appendix.**  
  The abstract states that the model enables “multi-character dialogue videos” and Section 1 suggests realistic multi-character dialogue generation. However, Appendix A.5 substantially narrows this claim. The described mechanism is to feed a single audio clip whose temporal segments correspond to different speakers and apply temporally varying masks to different faces. The paper explicitly states:  
  > “it is currently not possible to support scenarios where different characters speak different lines simultaneously or where interruptions occur during speech.”  
  So the system does support **sequential speaker turns with mask/audio assignment in one pass**, but not general multi-speaker dialogue in the sense of overlapping or independently simultaneous speech. This limitation should be reflected much more clearly in the main claims.
- **The main text overstates overall superiority on the full-body benchmark.**  
  The paper repeatedly says it “surpass[es] state-of-the-art methods” and “achieves the best performance on most evaluation metrics.” On Table 2, the picture is mixed: WanS2V is better on IQA, Sync-D, HKV, and FBN; OmniHuman-1 is better on FCN; MultiTalk is better on FVD. AnyAvatar is clearly competitive and strong on several metrics, but the narrative of broad dominance is not fully aligned with the actual table.
- **The long-video component is not a core contribution, yet the presentation sometimes blurs this boundary.**  
  Section 3.4 presents long-video generation as part of the framework, but Appendix A.5 states that the paper uses the “Time-aware Position Shift Fusion method from Sonic” and adapts it to HunyuanVideo13B. Using an adapted existing strategy is perfectly fine, but it should be more clearly distinguished from the paper’s novel contributions.

### Minor
- **The methodological explanation of the “training/inference mismatch” solved by the character image injection module is not fully crisp.**  
  The intuition is understandable from Section 3.1 and the appendix discussion—padding frames and direct latent reuse can reduce motion dynamics—but the claim that the module “eliminat[es] the inherent condition mismatch between training and inference” is stronger and more formal-sounding than what is actually demonstrated. The evidence is mainly ablation-based and subjective, rather than a clear formal characterization of the mismatch.
- **Some ablations remain narrower than ideal for isolating what specifically matters.**  
  Table 3 compares three image injection designs, which is useful, but does not fully disentangle whether the gains come from the added tokenizer branch, the elementwise addition, the token concatenation structure, or the positional treatment described in the appendix. Similarly, the emotion and multi-character analyses remain mostly qualitative.
- **The user study methodology is only briefly described.**  
  The paper states that 30 users rated four dimensions on 30 videos per method, but gives little detail on presentation protocol, randomization/blinding, or uncertainty/statistical significance. This does not nullify the findings, but it reduces how much weight can be placed on the subjective results.

### Trivial
- **Inference is very slow for practical deployment.**  
  Appendix A.8 reports roughly 60 minutes for a 10-second 720×1216 video with 50 steps on the base setup. This is a serious deployment limitation, though the paper does acknowledge it clearly and does not claim real-time performance.

## Nice-to-Haves
- Add dedicated quantitative evaluation for **emotion alignment**, ideally including cases where audio affect changes over time and cases where the emotion reference conflicts with the spoken affect.
- Add a controlled **multi-character benchmark** with per-character lip-sync / identity / audio-binding metrics, especially for speaker turns and crowded scenes.
- Clarify in the main paper that the long-video inference scheme is an **adapted prior method**, not a new algorithmic contribution.
- Provide stronger analysis of failure cases for mask inaccuracies, head pose changes, occlusions, and audio leakage between adjacent characters.
- Temper the “state-of-the-art” wording to match the mixed-but-strong results in Table 2.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair baseline comparisons because the authors use more/proprietary data.”**  
  Removed under the review rules. The paper indeed trains on a large curated dataset and compares against existing methods, but this criticism is not sufficient on its own here, and asymmetries in training setup do not automatically invalidate the comparison.
- **Reproducibility complaints about missing low-level implementation details.**  
  The paper gives substantial implementation detail in Section 4.1 and Appendix A.5/A.6. More detail would help, but this is not a substantive weakness for the final review.
- **Pure novelty dismissal of FAA as ‘just masked cross-attention’.**  
  Overly reductive. The face-aware, temporally aligned spatial cross-attention is indeed incremental rather than foundational, but it is still a concrete design contribution tailored to the problem setting.
- **Criticism questioning practical value because the system is not real-time.**  
  We keep the efficiency limitation as a weakness, but remove any implication that the paper is invalid because it is offline. The paper explicitly scopes itself as an offline, high-quality generation system.
- **Any concern about the existence/release/availability of cited systems or services.**  
  Removed per instruction.

## Novel Insights
The paper is strongest when interpreted not as a complete solution to “audio-driven emotion-aware multi-speaker dialogue,” but as a high-quality audio-driven animation system with two meaningful extensions: localized face-masked audio conditioning for speaker-targeted control, and reference-image-based emotion style steering layered on top of speech-driven motion. Read this way, the work has a clearer and more defensible contribution profile: it advances identity preservation and localized control in full-body generation, but currently overclaims on fully audio-grounded emotion understanding and general multi-party dialogue.

## Suggestions
- **Rewrite the emotion claim throughout the paper** to accurately reflect what is implemented: reference-image-guided emotion control in an audio-driven animation model, not audio-derived emotion inference.
- **Quantify the two flagship capabilities** with dedicated metrics: emotion alignment and multi-character audio-person binding.
- **Revise the abstract/introduction wording on multi-character dialogue** to state clearly that the current method supports temporally segmented speaker control, but not overlapping independent speech.
- **Separate core contributions from borrowed infrastructure** in the presentation, especially for long-video generation.
- **Align the result narrative with Table 2** by highlighting where the method is strongest (sync, identity, HKC, FID) and where baselines remain better (e.g., IQA/FBN/FCN/FVD depending on method).

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 6.0]
Average score: 4.5
Binary outcome: Reject
