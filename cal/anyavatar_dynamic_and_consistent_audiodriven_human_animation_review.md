=== CALIBRATION EXAMPLE 48 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately captures the paper's core contributions: dynamic, consistent, audio-driven animation for multiple characters. The abstract clearly states the three key challenges and the corresponding proposed innovations (Character Image Injection Module, Audio Emotion Module, Face-Aware Audio Adapter). The claims of surpassing state-of-the-art methods and generating realistic avatars are directly supported by the experiments section. The abstract is well-structured and aligns with the paper's content.

### Introduction & Motivation
The introduction effectively contextualizes the work within the broader field of audio-driven human animation and video generation. It clearly identifies the persistent gaps in current methods: the trade-off between dynamism and consistency, poor emotion alignment, and the inability to handle multi-character scenes. The three stated objectives directly address these gaps. The contributions are reiterated clearly. The related work section is comprehensive but could be more succinctly integrated to maintain narrative flow.

### Method / Approach
The overall framework is built upon the HunyuanVideo (MM-DiT) backbone, which is a reasonable and strong foundation. The three proposed modules are described in detail.

*   **Character Image Injection Module (CIIM):** The motivation—resolving the train-inference mismatch of using reference images—is sound. The ablation comparing three mechanisms (Fig. 3, Table 3) is a strong point. However, the description of mechanism (c) ("Token concat + add") is somewhat confusing. It states the reference image is repeated T times, fed into `tokenizer2`, then added to the video latent via an FC layer. The equation in Appendix A.4 (`p = TokenCat({K1(t_r) + K2(t_noise)}, t_R)`) seems to describe a concatenation, not an addition. This apparent inconsistency between the main text and appendix needs clarification for reproducibility. The claim that this method "improves the dynamics of motion while ensuring the consistency" is supported by the ablation study.

*   **Face-Aware Audio Adapter (FAA):** This is a clever and well-motivated solution for multi-character control. The use of a face mask in the latent space to isolate audio influence is clearly explained. The temporal alignment procedure for audio features and masks is detailed. The ablation (Fig. 7b, Table 4) effectively demonstrates its necessity and that it doesn't degrade background quality. A minor point: Equation 2 uses `g_M` (the mask) as a multiplier for the cross-attention output, which makes sense, but the description preceding it states the mask is used to generate "face-masked video latents" that are fused with audio. This could be phrased more precisely.

*   **Audio Emotion Module (AEM):** The goal of aligning audio emotion with facial expression is important. The method of using an emotion reference image injected via cross-attention into the "Double Block" is clear. The ablation (Fig. 7a) shows its effectiveness. However, a **significant conceptual concern** is raised: the module transfers emotion from a *reference image*, not directly from the *audio*. This means the emotional content must be provided visually by the user, which is a major operational limitation and somewhat contradicts the claim of "audio emotion" alignment. The paper acknowledges this as a limitation in Appendix A.8, but this critical caveat should be more prominently featured in the main method description to avoid overclaiming.

*   **Long Video Generation:** The segment-wise denoising with position shift (Algorithm 1) is a standard adaptation for long-sequence generation and is sufficiently described. The statement that it "does not add any extra inference or training costs" is accurate.

*   **Reproducibility:** The method relies on several external components (Whisper, InsightFace, HunyuanVideo's pretrained 3D VAE and LLaVA). The training stages, hardware (160 GPUs), and key hyperparameters are provided. The appendix offers further implementation details. While the scale is large, the description is sufficiently detailed for the approach to be reproduced in principle.

### Experiments & Results
The experimental setup is thorough and appropriate for ICLR.

*   **Datasets:** Using established benchmarks (CelebV-HQ, HDTF) for portrait animation is good. The construction of a new "wild" full-body test set is necessary and well-justified, given the lack of such a public benchmark. Details on its composition (250 videos, 200 identities) are provided.
*   **Metrics:** The use of a comprehensive suite of metrics (IQA, AES, FID, FVD, Sync metrics, HKC/HKV, VBench smoothness) and a user study (LS, IP, FBN, FCN) is commendable and covers quality, fidelity, synchronization, and specific animation aspects.
*   **Baselines:** Comparisons against a wide array of recent SOTA methods for both portrait (Sonic, EchoMimic, Hallo-3) and full-body animation (OmniHuman-1, StableAvatar, WanS2V, etc.) are comprehensive and fair.
*   **Results:**
    *   Quantitative results (Tables 1 & 2) show that AnyAvatar achieves top or competitive performance across nearly all metrics on both portrait and full-body tasks. This strongly supports the paper's claims.
    *   The user study results (Table 2) show AnyAvatar leading in IP and LS, which are core to its contributions. The lower scores in FCN/FBN compared to OmniHuman-1 are honestly discussed (attributed to super-resolution in the online service and inherent model issues).
    *   The qualitative figures (Figs. 4, 5, 6, and appendix visuals) effectively demonstrate the model's capabilities in multi-character scenarios, diverse styles, and emotion control. The improvement in dynamism and consistency is visually apparent.
*   **Ablation Studies:** The ablations for CIIM (Table 3), AEM (Fig. 7a), and FAA (Fig. 7b, Table 4) are crucial and well-presented. They provide clear evidence for the design choices. A minor note: the metrics in Table 3 (VQ, MD, IP, LS) should be explicitly defined in the caption or nearby text.

### Writing & Clarity
The paper is generally well-written and logically structured. The technical descriptions are clear, though the aforementioned minor confusion in the CIIM description and the relationship between the main text and appendix equations should be resolved. Figures 2 and 3 are referenced but not included in the provided text (a known parser issue), which hinders full assessment of clarity, but the descriptions are adequate. The flow from problem statement to method to results is coherent.

### Limitations & Broader Impact
The appendix thoroughly addresses limitations and societal impact, meeting ICLR's expectations.
*   **Limitations (A.8):** The primary limitation—reliance on emotion reference images rather than direct audio-to-emotion inference—is correctly identified and discussed. The computational cost and lack of real-time capability are also honestly stated, along with notes on ongoing acceleration efforts. This is a strong and necessary section.
*   **Societal Impact (A.9 & A.3):** The discussion is extensive and responsible. It covers data sourcing, consent, bias mitigation (including cultural bias), anti-abuse measures (watermarking, filters, public figure review), and potential impacts on employment and democratic institutions. The ethics statement is detailed. This goes beyond many submissions and is a positive aspect.

### Overall Assessment
This is a strong paper with a clear, valuable contribution. It addresses well-defined gaps in audio-driven human animation through three innovative and well-motivated technical modules (CIIM, FAA, AEM). The experimental evaluation is extensive, using multiple datasets, a comprehensive set of metrics, and comparisons against numerous SOTA baselines. The results robustly demonstrate superior or competitive performance. The main weakness is the design of the Audio Emotion Module, which requires an external emotion reference image rather than inferring emotion directly from audio, limiting its practicality and slightly misaligning with the stated goal. This limitation is, however, openly acknowledged. The paper is well-written, reproducible in design, and includes a thorough discussion of limitations and societal impact. With minor clarifications in the method description, this paper represents a solid contribution that likely meets the acceptance bar for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes AnyAvatar, a multimodal diffusion transformer (MM-DiT) model for generating dynamic, emotion-aligned, and multi-character audio-driven human animation videos. The core innovations include: (1) a character image injection module to enhance motion dynamics while preserving identity consistency, (2) an Audio Emotion Module (AEM) that uses an emotion reference image to align facial expressions with audio affect, and (3) a Face-Aware Audio Adapter (FAA) that employs latent-space face masking to enable independent audio-driven animation of multiple characters. The method is evaluated on portrait and full-body animation benchmarks, showing improvements in video quality, lip-sync, and multi-character generation over existing state-of-the-art methods.

### Strengths
1. **Addresses Multiple Key Challenges**: The paper explicitly targets three significant, unsolved problems in audio-driven animation: the trade-off between dynamism and identity consistency, emotion-audio alignment, and multi-character generation. The proposed modules offer clear, motivated solutions to each.
2. **Comprehensive Experimental Validation**: The paper provides extensive quantitative comparisons (using IQA, AES, Sync-C/D, FID, FVD, HKC, HKV) on multiple datasets (CelebV-HQ, HDTF, and a custom wild dataset) and includes a user study (30 participants). Results show superior or competitive performance across most metrics against strong baselines like Hallo-3, OmniAvatar, and WanS2V.
3. **Detailed Ablation Studies**: Ablation experiments (Tables 3, 4, Figure 7) convincingly justify the design choices for the Character Image Injection Module (CIIM), the injection of face masks, and the necessity of the Audio Emotion Module (AEM). The analysis of different CIIM mechanisms (Token, Token+Channel, Token+Add) is particularly insightful.
4. **Clear Technical Exposition**: The methodology is described in detail with equations and architectural diagrams (Figures 2, 3). The use of established components (HunyuanVideo backbone, Whisper, InsightFace) provides a solid foundation.

### Weaknesses
1. **Limited Emotion Modeling**: The Audio Emotion Module (AEM) requires an *emotion reference image* as input, rather than extracting emotion directly from the audio signal. This is a significant practical limitation, as noted in the appendix (A.8), increasing user burden and preventing the modeling of dynamic emotional shifts within a single audio clip. The novelty of using a reference image for emotion transfer is somewhat diminished.
2. **Computational Cost and Speed**: The model is built on the large HunyuanVideo-13B backbone. The paper acknowledges that generating a 10-second video takes about 60 minutes (Appendix A.8), which is far from real-time and limits practical applicability. While training-free acceleration is mentioned, it is not a core contribution or thoroughly evaluated.
3. **Incomplete Multi-Character Scenarios**: The Face-Aware Audio Adapter enables multi-character generation but has a key restriction: within a single forward pass, different characters cannot speak different lines simultaneously or interrupt each other (Appendix A.5). The paper demonstrates "multi-character" scenarios often by driving characters with the same audio or sequentially. This limits the realism of true conversational dialogue.
4. **Comparative Analysis Depth**: While comparisons are extensive, the analysis of *why* certain competitors fail (e.g., why others have less dynamic motion or worse consistency) is sometimes surface-level. A more detailed failure mode analysis would strengthen the claims.
5. **Potential Data Bias**: The training data (500k samples, 1250 hours) is filtered but its composition is only briefly described (Appendix A.6). Given the focus on emotion and diverse characters, a deeper discussion of dataset diversity and potential cultural biases (briefly mentioned in A.3) would be warranted for ICLR.

### Novelty & Significance
**Novelty**: The work is incrementally novel. The core ideas—injecting character reference features via addition, using a reference image for emotion style transfer, and applying latent masking for multi-character control—are logical extensions of existing conditioning and inpainting techniques in diffusion models. However, their specific integration and application to the simultaneous problems of audio-driven human animation constitute a meaningful synthesis.
**Significance**: The problem area is highly relevant and timely. Achieving high-dynamism, consistent, emotionally aligned, and multi-character animation from audio is a critical step towards practical digital avatars for film, gaming, and virtual communication. The paper demonstrates measurable progress over a strong set of recent baselines, which is significant for the field.

### Suggestions for Improvement
1. **Direct Audio-Emotion Modeling**: A major advance would be to replace or supplement the emotion reference image with a module that infers emotion directly from the audio prosody/features. This could be explored as a fusion approach or as a future direction with preliminary experiments.
2. **In-depth Efficiency Analysis**: Include a dedicated section or experiments on inference speed (FPS), parameter counts, and memory usage compared to key baselines. Explore or propose a more efficient architecture variant to address the real-time limitation.
3. **Enhanced Multi-Character Evaluation**: Design and report metrics for true interactive dialogue scenarios (e.g., turn-taking, overlapping speech). A qualitative analysis of current limitations in such settings would provide a clearer roadmap.
4. **Deeper Qualitative Analysis**: Provide more visual examples of failure cases (e.g., when emotion reference is mismatched, when background distorts, limitations in long sequences). This would help the community understand the model's boundaries.
5. **Strengthen Societal Impact Discussion**: While the appendix includes a thorough ethics statement, the main paper could briefly summarize key ethical considerations (e.g., deepfake potential, consent for training data) to meet ICLR's emphasis on this aspect.

**Overall Recommendation**: This is a solid, well-executed paper that makes clear technical contributions and shows strong empirical results. It addresses important, well-motivated problems and is mostly well-written. The primary concerns are the practicality of the emotion module and the computational cost. With the suggested improvements, particularly a more critical discussion of limitations and a roadmap for audio-based emotion, it would be a strong candidate for ICLR.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Ablation study on the core "dynamism-consistency trade-off" claim.** The paper argues the Character Image Injection Module (CIIM) solves a key trade-off, but only compares three injection mechanisms against each other (Table 3). It does not ablate the module entirely (i.e., remove it) against a baseline using simple padding frames or standard conditioning, making it impossible to verify that CIIM is the critical driver of the claimed improvement over prior art.
2.  **Benchmark for multi-character generation.** A core claim is enabling "realistic multi-character dialogue generation." However, quantitative results (Table 2) compare against multi-character methods (Multitalk, WanS2V) only on single-character metrics. There is no evaluation on multi-character-specific metrics (e.g., binding accuracy, cross-talk interference, individual lip-sync scores per character) using a dedicated multi-person test set. This gap directly undermines the multi-character contribution.
3.  **Comparison of emotion control against audio-driven methods.** The Audio Emotion Module (AEM) uses an emotion *reference image*, not the audio, to control expression. The paper does not compare this paradigm against existing methods that infer emotion directly from audio (e.g., via classifier guidance or latent modulation). Without this, the claim of "precise emotion alignment between characters and audio" is not substantiated; the method merely copies expression from an image.
4.  **Long video generation quantitative comparison.** The paper mentions 50-second generation and includes a qualitative figure (Fig 6), but provides no quantitative metrics (FVD, smoothness, consistency) for long sequences compared to baselines like StableAvatar or WanS2V that also feature long generation. The claim of "stronger stability" is thus unsupported.

### Deeper Analysis Needed (top 3-5 only)
1.  **Analysis of what the FAA actually learns.** The Face-Aware Audio Adapter (FAA) uses a pre-computed face mask to gate audio injection. The paper needs to analyze whether the model learns to associate audio with the masked region *semantically*, or if it's just a crude spatial gating. For example, does audio incorrectly affect other moving facial features (hair, glasses) outside the mask? Failure case analysis is missing.
2.  **Breakdown of performance by data/identity type.** The model is trained on a large, filtered dataset. Performance analysis should dissect results by factors present in the training data: race, gender, age, speaking style (e.g., energetic vs. calm), and presence of accessories (beards, glasses). Without this, it's unclear if gains are uniform or driven by overfitting to majority groups in the data.
3.  **Sensitivity analysis of the emotion reference image.** The AEM requires an emotion reference image. How sensitive are results to the choice of image (e.g., same identity vs. different identity, intensity of expression)? Does the module simply copy the reference expression statically, or does it adapt it temporally to the audio's prosody? The current analysis (Fig 7a) is superficial.

### Visualizations & Case Studies
1.  **Side-by-side video comparisons of failure modes.** The paper shows successes. To build trust, it must visualize common failure cases: loss of identity in profile views, artifacts during rapid head motion, "jumps" during the long video segment fusion, and, crucially, failures in multi-character scenarios when characters move or occlude each other.
2.  **Case study on multi-character overlapping/interrupting speech.** The appendix states the model cannot handle different characters speaking simultaneously. A detailed case study showing what *does* happen in this scenario (e.g., blended faces, one character dominating) is essential to understand the method's limits and validate the masking approach.
3.  **Visual ablation of the FAA's effect on the background.** The paper claims (Table 4) that masking doesn't degrade backgrounds. This should be demonstrated visually with side-by-side comparisons of videos generated with and without the mask, highlighting the background regions to show no introduced flicker or distortion.

### Obvious Next Steps
1.  **Implement and benchmark direct audio-to-emotion mapping.** The major limitation acknowledged is the need for an emotion reference image. A clear next step within the scope of this paper would be to implement a baseline module that extracts emotion embedding from audio (e.g., using Wav2Vec or an emotion classifier) and injects it, comparing results directly to the image-based AEM. This would strongly address the "emotion alignment" claim.
2.  **Conduct a user study focused on multi-character dialogue.** The existing user study (Table 2) lumps all methods together. A dedicated study where raters compare the multi-character output of AnyAvatar, Multitalk, and WanS2V on dimensions like speaker distinctness and conversation naturalness is necessary to prove the core contribution.
3.  **Profile and report inference speed.** For a practical contribution, reporting the inference time (seconds per frame) on standard hardware is crucial, especially since the method uses a large 13B parameter backbone. Comparing it to lighter baselines (e.g., OmniAvatar uses LoRA) contextualizes its practicality.

# Final Consolidated Review
## Summary
AnyAvatar is a method for audio-driven human animation that generates dynamic videos with consistent character identity, aligns facial expressions with audio emotion via a reference image, and enables multi-character scenarios through latent-space face masking. It introduces three modules: a Character Image Injection Module to balance dynamism and consistency, an Audio Emotion Module that transfers emotion from a reference image, and a Face-Aware Audio Adapter for multi-character control.

## Strengths
- The paper tackles three well-defined challenges in audio-driven animation with motivated solutions, supported by extensive experiments on multiple datasets (CelebV-HQ, HDTF, and a new wild dataset) and a comprehensive set of metrics (IQA, AES, FID, FVD, sync metrics, HKC/HKV, and user study), showing superior or competitive performance against state-of-the-art methods.
- Detailed ablation studies convincingly justify design choices, particularly for the Character Image Injection Module (comparing three injection mechanisms) and the Face-Aware Audio Adapter (demonstrating that masking does not degrade background quality).
- The method is built on a strong foundation (HunyuanVideo backbone) and is described with sufficient detail for reproducibility, including training stages, hyperparameters, and ethical considerations.

## Weaknesses
- The Audio Emotion Module requires an emotion reference image rather than inferring emotion directly from the audio, which limits practicality and partially contradicts the claim of "audio emotion alignment." Although acknowledged in the appendix, this critical limitation is underemphasized in the main text.
- The multi-character generation cannot handle different characters speaking different lines simultaneously or interruptions, and the paper lacks quantitative evaluation specific to multi-character scenarios (e.g., binding accuracy, individual lip-sync scores). This restricts the realism of true conversational dialogue, a core contribution.
- Inference speed is slow (60 minutes for a 10-second video at 720×1216 resolution), hindering real-time application. While acceleration efforts are mentioned, they are not thoroughly evaluated or compared to baselines.

## Nice-to-Haves
- A more detailed ablation comparing the Character Image Injection Module against a baseline without it (e.g., standard padding frames) to isolate its impact.
- Quantitative evaluation of long video generation (e.g., FVD, smoothness over 50-second sequences) compared to methods like StableAvatar and WanS2V.
- Sensitivity analysis of the emotion reference image (e.g., effect of identity mismatch, expression intensity) and failure case visualization for multi-character masking.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Clarify the description of the Character Image Injection Module in the main text to resolve any apparent inconsistency with the appendix regarding addition versus concatenation.
- Include a quantitative evaluation of multi-character generation using dedicated metrics (e.g., speaker binding accuracy) and a user study focused on dialogue naturalness.
- Discuss the limitation of the emotion module more prominently in the method section, and consider adding a baseline for direct audio-to-emotion mapping in the ablation or as a future direction.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 2.0, 6.0]
Average score: 4.5
Binary outcome: Reject
