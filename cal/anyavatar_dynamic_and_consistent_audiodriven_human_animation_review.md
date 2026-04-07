=== CALIBRATION EXAMPLE 42 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title and abstract clearly state the paper's contribution: a method for dynamic, emotion-aligned, multi-character audio-driven animation. The claims are specific and correspond to the three proposed innovations. However, the abstract's final claim of surpassing SOTA on a "newly proposed wild dataset" requires careful scrutiny in the experiments section to ensure the comparison is fair and the dataset is valid.

**Introduction & Motivation:** The introduction effectively motivates the problem, identifying key gaps in existing work (dynamism vs. consistency trade-off, emotion misalignment, lack of multi-character support). The three key objectives are clearly stated and logically lead to the proposed contributions. The transition from problem to solution is coherent.

**Related Work:** The section covers relevant prior work in portrait and full-body animation. A minor weakness is that it is somewhat descriptive (listing what each method does) rather than synthesizing a clearer narrative about how the specific limitations of prior art (e.g., how they handle conditioning, why they fail at multi-character) directly inform the design of AnyAvatar. This is not a major flaw, but a more critical analysis would strengthen the positioning.

**Method / Approach:**
*   **Character Image Injection Module (CIIM):** The motivation is sound (resolving train-inference mismatch of padding frames). However, the explanation of why mechanism (c) "Token concat + add" works best is somewhat superficial. The claim is that it "improves dynamics while ensuring consistency," but the underlying reason—why adding a repeated image latent via a projection works better than concatenation—is not explored theoretically or with feature-level analysis. The statement that adding a second tokenizer "accelerates model convergence" is an observation, not an explanation. More insight into the mechanism would strengthen this core contribution.
*   **Face-Aware Audio Adapter (FAA):** This is a clear and pragmatic solution for multi-character animation. The use of a face mask to spatially localize cross-attention is intuitive. A key question for reproducibility is the exact process: How is the face mask `g_M` generated and aligned? The mention of InsightFace for bounding boxes is good, but what about videos where the face moves? Is the mask static or tracked per frame? This detail is crucial.
*   **Audio Emotion Module (AEM):** This module's design raises significant questions. It requires an *emotion reference image* as input to guide the emotion. This contradicts the paper's goal of "achieving precise emotion alignment between characters and audio," as the emotion is now sourced from an image, not the audio. The audio's role in emotion becomes unclear. Is the AEM merely transferring expression from a reference image, with audio only driving lip sync? The claim of aligning "audio's emotional tone" (Sec. 5) seems misleading if the emotion source is a separate image. This is a major conceptual weakness that needs clarification or re-framing.
*   **Long Video Generation & Multi-Character Drive:** The segment-wise generation algorithm (Alg. 1) is a standard sliding window approach adapted from Sonic. The description for driving multiple characters (Sec. A.5) is clear for sequential speaking but correctly identifies the limitation for simultaneous, overlapping speech as future work.
*   **Reproducibility Concern:** The method is built upon "HunyuanVideo," a large, non-open-source model. While the proposed adapters are described, the core backbone's weights and architecture are proprietary. This severely limits reproducibility and accessibility for the research community, which is a critical issue for ICLR.

**Experiments & Results:**
*   **Datasets:** The construction of a new "wild" full-body test set (250 videos) is necessary due to the lack of public benchmarks. However, details on its composition, diversity, and how it avoids bias are minimal (relegated to the appendix). More importantly, using a custom dataset for the main full-body comparison (Table 2) while comparing on public datasets for portrait (Table 1) is acceptable but requires extra care to ensure the custom dataset doesn't inadvertently favor the proposed method.
*   **Baselines:** The selection of baselines is comprehensive for both portrait and full-body animation. A notable omission is a direct comparison of the **multi-character** capability. While MultiTalk and WanS2V are included in Table 2, the metrics (IQA, FID, etc.) do not specifically measure multi-character performance (e.g., binding accuracy, cross-talk interference). A dedicated quantitative evaluation for the multi-character scenario is missing.
*   **Metrics:** The use of a broad set of metrics (IQA, AES, FID, FVD, sync, user study) is good. The justification for using HKC/HKV for hand quality is appropriate for full-body animation.
*   **Results Interpretation:**
    *   Table 1 (Portrait): Shows strong performance, leading on many metrics. This is convincing.
    *   Table 2 (Full-body): The results are mixed. AnyAvatar leads in Sync-C and HKC, is competitive in IQA/AES, but is not the best in several other metrics (e.g., FVD, Sync-D, HKV, FCN, FBN). The claim that it "achieves the best performance on most evaluation metrics" is an overstatement; it achieves the best on *some* key metrics. The user study shows OmniHuman-1 leading in FCN/FBN, which the authors attribute to super-resolution in its online service—this is a valid point but highlights that subjective scores can be influenced by external factors.
*   **Ablation Studies:**
    *   CIIM Ablation (Table 3): Subjective evaluations (VQ, MD, IP, LS) clearly show the proposed mechanism's superiority. This is effective.
    *   AEM Ablation (Fig 7a): The qualitative figure shows a difference, but without the reference emotion image, it's just a comparison between text-only and (text + audio + emotion image). It doesn't isolate the contribution of *audio-driven* emotion. A proper ablation would compare using AEM with a *neutral* reference image vs. an *emotional* one.
    *   FAA Ablation (Fig 7b, Table 4): The qualitative figure is clear. Table 4 effectively shows that masking doesn't degrade background quality.
*   **Overall Experimental Rigor:** The experiments are extensive but have gaps. The most critical missing experiment is one that quantifies the **emotion alignment** between audio and video output, given the AEM module's central role. How well does the generated expression match the emotion in the *audio*, not just the reference image?

**Writing & Clarity:** The paper is generally well-written and logically structured. Some technical passages could be clearer (e.g., the derivation of `n` in Sec. 3.2: "with n = \lfloor n' / 4 \rfloor + 1" is likely intended). Figures are helpful. The significant confusion around the Audio Emotion Module's functionality (audio-driven vs. image-driven emotion) is the primary clarity issue.

**Limitations & Broader Impact:** The limitations section (A.8) is excellent and honest. It correctly identifies the major weakness of the AEM (reliance on reference images, inability to handle dynamic emotions), the slow inference speed, and the challenge of real-time interaction. The societal impact section (A.9) is thorough, discussing potential misuse, mitigation strategies (watermarks, filters), and employment impacts. It is responsibly written.

### Overall Assessment

This paper presents a technically sound system, AnyAvatar, that makes tangible progress on audio-driven animation, particularly in enabling multi-character scenarios and improving motion dynamism. The core innovations (CIIM, FAA) are well-motivated and demonstrate clear benefits via ablations. However, the work is significantly hampered by two major issues for an ICLR audience: 1) **Reproducibility:** Dependence on the proprietary HunyuanVideo backbone limits the ability of the community to build upon this work. 2) **Misleading Contribution:** The Audio Emotion Module (AEM) is framed as aligning audio emotion with video, but its design fundamentally relies on an emotion reference *image*, shifting the source of emotion control. This muddles the paper's narrative and is not adequately justified or evaluated. The experiments, while broad, lack a decisive multi-character benchmark and a proper evaluation of audio-emotion alignment. The contribution stands as a competent engineering framework built on a powerful base model, but the conceptual clarity and openness required for a top-tier conference are not fully met in its current form.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes AnyAvatar, a diffusion transformer-based model for audio-driven human animation that addresses three key challenges: generating dynamic motions while preserving character consistency, aligning facial expressions with audio emotion, and enabling multi-character dialogue generation. The core innovations are a Character Image Injection Module (CIIM) to enhance dynamics and identity preservation, an Audio Emotion Module (AEM) for emotion control via reference images, and a Face-Aware Audio Adapter (FAA) for localized audio driving of multiple characters. Extensive experiments on portrait and full-body benchmarks demonstrate superior performance over state-of-the-art methods in both quantitative metrics and user studies.

### Strengths
1. **Comprehensive Problem Formulation and Solution:** The paper clearly identifies three persistent limitations in audio-driven human animation (dynamics-consistency trade-off, emotion misalignment, and lack of multi-character support) and proposes three corresponding, well-motivated technical components to address them. The modules are integrated into a unified framework, and their individual contributions are validated through ablations (Tables 3, 4, Figure 7).

2. **Rigorous and Extensive Evaluation:** The experimental section is thorough, comparing against numerous SOTA methods on both standard portrait datasets (CelebV-HQ, HDTF) and a newly curated wild full-body dataset. Evaluation includes a wide range of objective metrics (IQA, AES, FID, FVD, Sync-C/D, HKC, HKV) and a detailed user study across four key dimensions (LS, IP, FBN, FCN). Results consistently show AnyAvatar outperforms or matches competitors across most metrics (Tables 1, 2).

3. **Effective Multi-Character Generation:** The Face-Aware Audio Adapter (FAA) is a practical solution for a largely unexplored problem. By applying latent-space face masks and independent cross-attention for audio injection, the method enables convincing multi-character dialogue generation from a single audio track, as demonstrated in qualitative results (Figures 5, 8).

### Weaknesses
1. **Indirect and Cumbersome Emotion Control:** The Audio Emotion Module (AEM) requires an external emotion reference image to guide expression generation, rather than inferring emotion directly from the audio signal. This increases user burden and limits the model's ability to handle dynamic emotional shifts within a single audio clip, a limitation the authors acknowledge in Sec. A.8.

2. **High Computational Cost and Dependency on Large Backbone:** The method is built upon the HunyuanVideo-13B model, a massive pre-trained foundation model. Training requires 160 high-memory GPUs, and inference is slow (e.g., 60 minutes for a 10-second video). This severely impacts accessibility, reproducibility, and potential for real-time applications. While acceleration techniques are mentioned, they are not a core contribution.

3. **Incremental Nature of Some Components:** The core technical ideas, while effectively combined, may not be fundamentally novel. The CIIM's "additive" conditioning is a common alternative to concatenation. The FAA's use of masks for localized control is reminiscent of inpainting techniques. The paper could do more to delineate the specific novelty of each mechanism compared to prior conditioning strategies in diffusion models.

### Novelty & Significance
**Novelty:** The work's primary novelty lies in the integrated design targeting the three identified challenges simultaneously, particularly the multi-character audio-driven animation via the FAA. The CIIM's specific implementation (token concat + add with spatial shift) and its application to resolve the dynamism-consistency trade-off in this context is a non-trivial contribution. The AEM, while relying on reference images, provides a pathway for emotion control.

**Significance:** The ability to generate dynamic, identity-consistent, and emotionally aligned animations for multiple characters is a significant step toward practical digital human and cinematic content creation. The paper benchmarks a challenging new task (multi-character audio-driven animation) and sets a strong baseline. The comprehensive evaluation and promising results make it a valuable contribution to the field of generative models for human animation.

### Suggestions for Improvement
1. **Direct Audio-Emotion Modeling:** A major advancement would be to modify the AEM to extract emotional cues directly from the audio input (e.g., using a pre-trained audio emotion recognition model) instead of relying on a reference image. This would make the system more usable and capable of handling complex, time-varying emotions.

2. **Efficiency and Reproducibility Analysis:** Include a dedicated section on computational requirements (FLOPs, memory, inference time) and model size. To aid reproducibility, provide more details on the training data pipeline (e.g., exact filtering criteria from LatentSync/Koala-36M) and consider releasing a smaller-scale version of the model or training script that is feasible with moderate resources.

3. **Deeper Ablation and Comparison:** Conduct an ablation study on the wild full-body dataset (similar to Table 3) to quantitatively disentangle the contributions of CIIM, AEM, and FAA to final performance metrics. Furthermore, compare the FAA more directly with alternative multi-character conditioning approaches (e.g., separate cross-attention layers per character without masking) to better justify the design choice.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the necessity of an emotion reference image for the Audio Emotion Module (AEM).** The paper requires an external emotion reference image, but does not compare to a baseline where emotion is inferred directly from the audio features (e.g., via an emotion classifier). Without this, the core claim of "audio emotion alignment" is only partially validated, as the model is merely copying expression from a reference image.
2. **Quantitative evaluation of multi-character generation.** The paper only provides qualitative visuals for multi-character scenarios. There are no metrics (e.g., individual character lip-sync accuracy, cross-talk interference scores) comparing AnyAvatar's FAA to other multi-character methods (MultiTalk, WanS2V) on a standardized multi-speaker test set. This undermines the claim of enabling realistic multi-character dialogue.
3. **Systematic evaluation of the claimed dynamism-consistency trade-off.** The paper claims its character injection module resolves this trade-off, but provides only a small, subjective ablation (Table 3). No objective metrics for motion diversity (e.g., optical flow magnitude variance) and identity preservation (e.g., face embedding cosine similarity over time) are reported across all compared methods. This leaves the central claim unsupported by rigorous evidence.
4. **Long-video generation comparison.** The paper mentions generating 50-second videos and claims stronger stability, but provides no quantitative comparison (e.g., temporal flicker metrics, consistency scores over time) against methods like StableAvatar or WanS2V that also target long sequences. The claim of superior long-video capability is therefore anecdotal.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of emotion alignment accuracy.** The paper does not measure how well the generated facial expressions match the actual emotion in the audio. An analysis using an off-the-shelf audio emotion classifier and a corresponding video expression classifier is needed to quantify the "accurate emotion style control" claim.
2. **Failure mode analysis for the Face-Aware Audio Adapter.** The paper shows successful cases but does not analyze when and why the method fails (e.g., with overlapping speech, when face detection is inaccurate, or with extreme poses). Understanding these limits is critical for assessing the robustness of the multi-character approach.
3. **Disentanglement analysis for multi-character control.** It is unclear if audio intended for one character ever "leaks" and influences another character's face, especially when masks are close. A quantitative analysis of cross-character influence (e.g., measuring lip motion of a non-speaking character) is necessary to trust the isolation claimed by the FAA.

### Visualizations & Case Studies
1. **Side-by-side video comparisons highlighting dynamism vs. consistency.** Show a short clip generated by AnyAvatar versus a top baseline (e.g., StableAvatar, WanS2V) for the same input, focusing on how natural the body motion is while the face identity remains stable. This would directly showcase the paper's core improvement.
2. **Visualization of emotion transfer failure cases.** Show examples where the provided emotion reference image does not match the audio's emotional content, leading to conflicting or unnatural expressions. This would clarify the limitations of the current AEM design and the need for direct audio-emotion modeling.
3. **Grid visualization of multi-character scenarios with individual and combined audio drives.** Show a single frame with multiple characters, and then show how each character's lip movements change when driven by different audio inputs (individual and mixed). This would concretely demonstrate the FAA's capability for independent control.

### Obvious Next Steps
1. **Integrate direct audio-emotion understanding.** Given the acknowledged limitation of requiring an emotion reference image, the paper should have included a baseline or variant where emotion features are extracted directly from the audio signal (e.g., using a pretrained emotion recognition model) to drive the AEM. This is a logical and critical step for true audio-emotion alignment.
2. **Benchmark on a public multi-character audio-visual dataset.** The authors created their own test set. To allow direct comparison and foster future research, they should have evaluated on an existing multi-speaker talking-head dataset (even if adapted) or made their test set publicly available with clear evaluation protocols.
3. **Conduct a user study specifically on emotion perception.** The existing user study does not evaluate emotion alignment. A subjective evaluation where participants rate how well the generated video's expression matches the emotion conveyed by the audio is essential for the AEM claim.

# Final Consolidated Review
## Summary
AnyAvatar is a multimodal diffusion transformer model for audio-driven human animation. It introduces three key components: a Character Image Injection Module to enhance motion dynamics while preserving identity, an Audio Emotion Module that uses a reference image to guide facial expressions, and a Face-Aware Audio Adapter that enables multi-character animation via latent-space masking. The method is evaluated on portrait and full-body benchmarks, showing competitive performance.

## Strengths
- **Comprehensive integration targeting three distinct challenges:** The paper clearly formulates and addresses the trade-off between dynamism and consistency, emotion alignment, and multi-character generation within a single framework. Ablation studies (Tables 3, 4, Figure 7) validate the contribution of each proposed module.
- **Effective enabling of multi-character animation:** The Face-Aware Audio Adapter provides a practical, mask-based solution for localizing audio control to specific characters, a capability underexplored in prior work. Qualitative results (Figures 5, 8) demonstrate convincing multi-character dialogue generation.

## Weaknesses
- **Audio Emotion Module does not align audio with emotion; it copies expression from an image:** The module requires an external emotion reference *image* as input (Sec. 3.3, A.8). The emotion is transferred from this image, not inferred from the audio signal. This contradicts the paper's claim of "precise emotion alignment between characters and audio" and makes the system cumbersome, as acknowledged in the limitations.
- **Dependence on a large, proprietary backbone limits reproducibility and accessibility:** The method is built upon HunyuanVideo, a non-open-source model. Training requires 160 GPUs, and inference is slow (60 minutes for a 10-second video per A.8). This severely hinders community verification, extension, and practical deployment.
- **Lack of quantitative evaluation for core claimed capabilities:** There is no dedicated quantitative benchmark for multi-character performance (e.g., character-audio binding accuracy, cross-talk interference) to substantiate claims against other multi-character methods. Similarly, no metric evaluates how well the generated expressions align with the emotion in the *audio*, only that they match a provided reference image.

## Nice-to-Haves
- A baseline or variant where emotional features are extracted directly from the audio signal (e.g., via a pre-trained emotion recognition model) would strengthen the Audio Emotion Module's claim and usability.
- Releasing the curated "wild" full-body test set with evaluation protocols would facilitate direct comparison and future research in multi-character animation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "The explanation of why mechanism (c) works best is superficial." The paper provides an empirical ablation (Table 3) showing clear superiority across subjective metrics (VQ, MD, IP, LS), which is a standard and sufficient justification for an architectural choice in this field.
- **Weakness:** "The related work is descriptive rather than synthesizing a narrative." While a more analytical presentation could be beneficial, this is a presentational preference, not a substantive flaw in the technical contribution or evaluation.
- **Weakness:** "Incremental nature of some components." The novelty lies in the integrated design and application to solve the specific, identified problems in audio-driven animation, which is a valid contribution.
- **Strength:** "The paper is well-written and logically structured." This is a generic strength that does not highlight what this specific paper does uniquely well.

## Novel Insights
The paper's primary novel insight is the integrated framework that simultaneously tackles dynamism-consistency, emotion guidance, and multi-character generation—a combination not addressed by prior work. Specifically, the Face-Aware Audio Adapter demonstrates that latent-space masking combined with spatially localized cross-attention is an effective and relatively simple mechanism for achieving independent audio-driven control of multiple characters in a single diffusion transformer forward pass.

## Suggestions
- Reframe the contribution of the Audio Emotion Module to accurately reflect that it provides *expression control via a reference image*, not direct audio-emotion alignment. The text in the abstract, introduction, and conclusion should be adjusted accordingly.
- Include a quantitative evaluation for the multi-character scenario, such as measuring lip-sync accuracy per character and cross-influence between characters when driven by separate audio segments.
- In the limitations or future work, propose a concrete research direction for replacing the emotion reference image with features directly extracted from the audio signal.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 6.0]
Average score: 4.5
Binary outcome: Reject
