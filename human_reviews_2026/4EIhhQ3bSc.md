# MANGO: Natural Multi-speaker 3D Talking Head Generation via 2D-Lifted Enhancement

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Current audio-driven 3D head generation methods mainly focus on single-speaker scenarios, lacking natural, bidirectional listen-and-speak interaction. Achieving seamless conversational behavior, where speaking and listening states transition fluidly remains a key challenge. 
Existing 3D conversational avatar approaches rely on error-prone pseudo-3D labels that fail to capture fine-grained facial dynamics.
To address these limitations, we introduce a novel two-stage framework \textit{\textbf{MANGO}}, which leveraging pure image-level supervision by alternately training to mitigate the noise introduced by pseudo-3D labels, thereby achieving better alignment with real-world conversational behaviors. Specifically, in the first stage, a diffusion-based transformer with a dual-audio interaction module models natural 3D motion from multi-speaker audio. In the second stage, we use a fast 3D Gaussian Renderer to generate high-fidelity images and provide 2D-level photometric supervision for the 3D motions through alternate training. Additionally, we introduce MANGO-Dialog, a high-quality dataset with over 50 hours of aligned 2D-3D conversational data across 500+ identities.  Extensive experiments demonstrate that our method achieves exceptional accuracy and realism in modeling two-person 3D dialogue motion, significantly advancing the fidelity and controllability of audio-driven talking heads.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MANGO, a two-stage framework for multi-speaker 3D talking head generation. The first stage uses a diffusion-based dual-audio motion generator to produce 3D motion parameters conditioned on both speakers’ audio. The second stage employs a 3D Gaussian Renderer (MG-Renderer) to synthesize high-fidelity images using 2D photometric supervision, which the authors refer to as a 2D-lifted enhancement. A new dataset, MANGO-Dialog, is also introduced, consisting of over 50 hours of synchronized 2D-3D conversational data from 500+ speakers. Experimental results demonstrate quantitative and qualitative improvements over prior 3D talking head methods such as DualTalk and DiffPoseTalk, particularly in lip-sync precision and overall mesh–image alignment.

### Strengths
- Novel Dual-Audio Diffusion Framework : The combination of a dual-audio fusion module and a diffusion-based motion generator allows modeling of bidirectional conversational dynamics, distinguishing speaking and listening states more effectively than single-speaker systems.
- Two-Stage 2D-Lifted Enhancement Strategy : The introduction of 2D image-level supervision through the Gaussian Renderer effectively refines 3D mesh predictions, mitigating the noise of pseudo-3D labels obtained from tracking.
- New Dataset (MANGO-Dialog) : A large-scale, 2D–3D aligned multi-speaker dialogue dataset is a valuable contribution that could benefit future research in multi-person conversational synthesis.
- Comprehensive Evaluation : Extensive comparisons with single- and multi-speaker baselines (FaceFormer, DualTalk, SadTalker, etc.) show improved 3D accuracy (LVE, MVE) and 2D fidelity (PSNR, SSIM, LPIPS).

### Weaknesses
- Limited Conceptual Novelty : The framework is primarily a combination of existing paradigms. 3D talking head generation and speech-based motion diffusion—with modest architectural novelty. The overall system resembles a combination of talking head generation and speech separation rather than a fundamentally new paradigm.
- Scalability Concerns : The method generates videos at a head level, which limits scalability to full-scene multi-speaker synthesis. Extending the approach to simultaneous multi-agent scenes or long-turn interactions would be challenging given the per-head rendering design.
- Problem Scope Overlap : The targeted issue of over-smoothed mouth motion is not unique to multi-speaker setups; numerous single-speaker 3D talking head works (e.g., DiffPoseTalk, FaceFormer) already address similar issues with comparable diffusion or transformer-based solutions.
- Relatively Lower Visual Quality : Compared to high-quality 3D generative systems such as HALLO3 or LivePortrait, the generated outputs in Fig. 6 appear less photorealistic and expressive. Leveraging stronger generative priors might substantially improve realism and lip-sync fidelity.
- Terminological Ambiguity.The term “2D-lifted enhancement” is not clearly justified : It appears to describe the process of applying 2D photometric loss to refine 3D motion, but the phrasing could mislead readers into thinking it’s a new geometric transformation rather than a training strategy.

### Questions
- Advantage over Task Composition : Since the proposed setup essentially combines speech-driven motion synthesis with conversational context modeling, what specific benefits does MANGO achieve beyond simply combining existing talking head and speech separation modules?
- Scene-Level Generation : Could the method be extended to generate an entire two-speaker video scene simultaneously, rather than per-head? If so, what architectural or computational challenges arise due to the current 3D representation?
- Relation to Single-Speaker 3D Methods : How does the proposed system differ from prior works that already tackled over-smoothing using diffusion or Gaussian renderers (e.g., DiffPoseTalk, ARTalk)? Are the improvements mainly empirical or conceptual?
- Quality Gap vs. MANGO-Dialog Baselines : The paper shows improved quantitative metrics, but the generated samples from MANGO-Dialog still look coarse compared to prior generative 3D methods. Have you considered integrating more advanced generative models like LivePortrait or HALLO3 to enhance appearance realism?
- Clarification on “2D-Lifted Enhancement : ”Could you clearly define this term? Is it equivalent to the two-stage alternated supervision process (3D→2D refinement), or does it imply a structural connection between 2D features and 3D geometry?

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
3

### Summary
This paper introduces MANGO, a 3D conversational multi-speaker talking-head generation framework that unifies the synthesis of both speaking and listening behaviors.
The framework consists of two stages:
1. A diffusion-based multi-audio fusion model that models motion distributions across speakers;
2. A 3D Gaussian Splatting (3DGS) renderer that converts predicted motion sequences into videos, with additional 2D image supervision to mitigate inaccuracies from 3D tracking.
The authors also provide a new 3D conversational dataset for training and evaluation.

### Strengths
1. Presents a unified framework for generating both speaking and listening 3D talking heads. It is a novel and ambitious direction.

2. Incorporating 2D image-level loss after 3DGS rendering helps partially alleviate the errors caused by 3DMM estimation, providing additional supervision for the 3D talking head generation.

### Weaknesses
1. (Major) Limited effectiveness in both speaking and listening states: From the demo videos, while the model can roughly switch between speaking and listening modes, neither mode performs convincingly.

    - Speaking: The lip motions are not accurate and clearly worse than single-speaker baselines such as CodeTalker or DiffPoseTalk. Even though MANGO separates speaking and listening audio inputs and introduces an indicator for speaking status, the generated lips remain unsatisfactory. This raises the question of whether the dual-audio module introduces interference between the two states.
    - Listening: The listening behaviors appear almost random or static, lacking clear correlation with the interlocutor’s speech (e.g., at 00:38 in the demo, when hearing “luckily”, DualTalk shows a smile but MANGO does not). The dataset examples contain rich listening behaviors — nodding, smiling, eyebrow raises, or thoughtful blinking — yet these are not reflected in the results. Quantitative or qualitative evidence showing the correlation between listening behavior and input audio would strengthen the claim.
    - In conclusion, the framework currently fails to convincingly capture both accurate lip articulation and expressive listening dynamics.

2. (Major) Questionable benefit of 2D image loss after 3D reconstruction: While introducing a 2D image loss after rendering is presented as a core contribution, such image-space supervision has long been standard in 3D face reconstruction pipelines.
Here, applying the 2D loss after generation introduces compounded errors — both from inaccurate expression estimation and imperfect 3D rendering.
The actual effectiveness of this design is unclear and requires visual ablation evidence.
Moreover, the rendered frames in the demo show strong 3DGS artifacts, raising doubts about gradient stability and potential negative impacts on 3DMM coefficient learning.
As an alternative, will it be more stable and effective to optimize the predicted 3DMM (pGT) directly through differentiable rendering and computing the image loss on the rendered image of pGT?

3. (Major) Relation to INFP remains underspecified: Although the authors claim their task differs from INFP (which generates 2D talking heads), MANGO ultimately renders to 2D and mainly differs in that it uses 3DMM as the intermediate representation instead of INFP’s motion features.

    - The two tasks and formulations are thus highly similar, and a visual comparison with INFP would be essential to demonstrate advantages in motion controllability.

    - Both methods employ a dual-audio module to link speech and motion features. What's the difference and strength of MANGO's audio2motion model against the INFP's?

4. (Minor) Presentation and clarity issues:
    - The two claimed contributions, conversational talking-head generation and 2D image loss, appear weakly connected and seem like two independent ideas. And the statement “in conversational scenarios, lip movements become more complex due to the dynamics of interaction” (L80) lacks empirical justification; single-speaker data can exhibit similar complexity.
    - The naming of Stage 1/2 and Training Phase 1/2 is confusing. For example, does Training Phase 1 (Stage 2 training) refer to only training the MG-Renderer on pGT meshes with 2D image loss?
    - In Equation (3), both $H_{self}$ and $H_{other}$  pass through the Transformer jointly, which seems inconsistent with the schematic in Figure 4.

### Questions
1. In our understanding, a listener’s expressions during conversation should depend not only on the other party’s speech content but also on the speaker’s facial expressions. Will the speaking state (or visual features of the speaker) be considered as an additional input when modeling the listening behavior?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes MANGO, a two-stage framework for generating natural, bidirectional 3D talking heads in multi-speaker conversational settings. Unlike prior work that focuses on either speaking or listening, MANGO aims to model fluid transitions between these states using dual-audio inputs and 2D photometric supervision to refine 3D motion. The authors also introduce MANGO-Dialog, a new dataset of 50+ hours of aligned 2D–3D conversational videos across 500+ identities. The core idea is to bypass the inaccuracies of pseudo-3D labels (from 3D face trackers) by using 2D image-level losses to guide 3D motion learning through a 3D Gaussian renderer.

### Strengths
1. Dual-audio fusion module enables speaker–listener disentanglement
The paper introduces a Dual-audio Interaction Module (DIM) that explicitly models conversational dynamics by fusing self and other speaker audio through a Transformer, followed by a residual connection with the self-audio. This design helps preserve speaker-specific lip-sync fidelity while allowing listener behaviors (e.g., subtle smiles, head nods) to be conditioned on the interlocutor’s speech. As shown in Fig. 7(a) and Table 2, removing this module leads to cross-contamination—e.g., the listener exhibits speaking-like mouth movements. This is a non-trivial contribution, as prior multi-speaker methods (e.g., DualTalk) do not explicitly model such asymmetric audio roles.

2. MANGO-Dialog: A large-scale, temporally aligned 2D–3D conversational dataset
The authors release MANGO-Dialog, comprising 50+ hours of dual-speaker videos across 500+ identities, with synchronized audio, pseudo-3D FLAME parameters (via Spectre), and refined camera poses. Crucially, clips are 30–120 seconds long, ensuring natural speaking–listening transitions—a rarity in existing datasets (e.g., VoxCeleb, HDTF are mostly single-speaker). The dataset also includes speaker diarization labels (via TackNet), enabling training of the speaking indicator. While the 3D labels are pseudo-ground truth (see Cons), the 2D–3D alignment and scale make this a valuable resource for future research in conversational avatars.

### Weaknesses
1. Inadequate 2D SOTA comparison

The paper compares its 2D output against SadTalker (2023), AniTalker (2024), and ARTalk (2025)—all of which are 3D-parameter-driven 2D renderers, not end-to-end 2D diffusion or neural rendering pipelines. It omits recent high-fidelity 2D talking head methods that achieve near-photorealism and strong lip-sync, such as:

VASA-1 (Microsoft, 2024): generates real-time, high-resolution, emotionally expressive talking faces from audio + single image.

OmniHuman-1: supports full-body, multi-view, and expressive control.

IF-MDM (2024): uses masked diffusion for coherent long-term 2D animation.

GaussianTalker / FlashAvatar: pure 3D Gaussian-based pipelines that may share architectural similarities with MANGO’s renderer but are not discussed or compared.

Without these comparisons, the claim of “superior 2D realism” is not convincingly supported—especially since MANGO’s 2D results (Fig. 6) show limited texture fidelity (e.g., blurry teeth, flat skin shading) compared to VASA-style outputs.

2. Missing comparison with industry-grade 3D pipelines

The 3D evaluation (Table 1) only includes academic methods (FaceFormer, CodeTalker, DualTalk, etc.). It excludes NVIDIA Audio2Face, which is:

*Widely used in production,

*Trained on high-quality 3D scans,

*Capable of real-time inference,

*Supports expression and viseme controls.

Given that MANGO claims “exceptional accuracy,” a comparison with Audio2Face on the same test set (even via qualitative side-by-side) would be essential to validate industrial relevance.

3. No explicit modeling of head pose dynamics or eye blinking

While the FLAME model includes head pose, the paper does not evaluate or visualize head motion quality. In Fig. 5–6, heads appear mostly static, suggesting the model may underutilize head pose variation—a key aspect of natural listening (e.g., nodding, tilting). Similarly, eye blinking is absent: FLAME does not model eyelids, and the renderer does not synthesize blinking. This leads to unnaturally fixed gazes, reducing perceived realism—especially in listening mode, where blink rate and gaze shifts are critical social signals. Previous methods such as DiffPoseTalk, Media2Face, already include the head-pose prediction and some also deliver natural eye blinking. 

4. Limited expression control and variation

The method uses FLAME’s expression parameters (ψ ), but the paper provides no analysis of non-mouth expressions (e.g., brow raises, smiles, frowns). While Fig. 6 shows some smiling, it’s unclear whether this is audio-driven or coincidental. There is no user control over expression intensity or type, and no disentanglement between speech-driven and emotion-driven motion. This limits applications requiring emotional or stylistic control.

5. 3D labels derived from noisy 2D-to-3D lifting

The dataset’s 3D labels come from Spectre, which the paper itself critiques (Fig. 2) for over-smoothing or exaggerating lip motion. This creates a fundamental supervision bottleneck: even with 2D-lifted refinement, the initial motion prior is biased. The authors claim their output sometimes exceeds the pseudo-GT mesh (Fig. 5, 9), but this is not quantified (e.g., via human preference or 2D re-projection error vs. GT image). Without ground-truth 3D scans (e.g., from multi-view capture), the true 3D accuracy remains unverifiable.

6. Ablation studies are missing from the demo video

The paper includes strong ablation results (Table 2, Fig. 7), but the supplementary demo video (presumably linked in submission) does not visualize these variants (e.g., w/o DIM, w/o two-stage). This makes it hard for reviewers/users to perceptually validate the claimed improvements. For a method relying on subtle conversational cues, visual ablation is essential.

### Questions
1. How is Fig. 2 generated?

Fig. 2 shows “over-smoothed” (orange) and “exaggerated/noisy” (blue) 3D lip motion curves compared to a “real” red curve, with visual insets of misaligned meshes. However, the paper does not specify:

What is the ground-truth reference for the red curve? Is it manually annotated lip keypoints, or derived from high-fidelity 3D scans?

Which 3D reconstruction methods produced the orange and blue curves? Are they Spectre, DECA, 3DDFA-v3, etc.?

Are these curves from real conversational data (like MANGO-Dialog) or from single-speaker datasets?

Without this, the figure risks being illustrative rather than empirical, weakening the motivation for 2D-lifted supervision.

2. The paper claims MANGO sometimes outperforms pseudo-GT meshes (e.g., Fig. 5, 9). But how is this quantified?

The visual examples in Fig. 5 and Fig. 9 suggest that MANGO’s mesh aligns better with the 2D ground-truth image than the pseudo-GT mesh from Spectre. However:

Is there a 2D re-projection error (e.g., L1 distance between rendered mesh and GT image) comparing MANGO vs. Spectre?

Have you conducted a user study where humans judge which mesh (Spectre vs. MANGO) better matches the GT video?

If MANGO is “better than pseudo-GT,” does that imply the pseudo-GT is a poor training target—and if so, why not use 2D-only supervision from the start?

This is central to the paper’s core claim but remains anecdotal.

3. The speaking indicator I_self is assumed to be perfectly known. How does performance degrade under realistic diarization errors?
The method uses a binary speaking indicator derived from TackNet (Sec 3.4), which is likely near-perfect on curated clips. But in real-world deployment:

What happens if I_self flips state 10% or 20% of the time (common in overlapping speech)?

Is the model robust to missing or delayed indicators?

Could the model infer the speaking state from audio alone, removing reliance on external diarization?

This affects practical applicability, yet no ablation on indicator noise is provided.

4. The ablation in Table 2 shows “Ours (+two stage)” has higher LVE/MVE than the “+jaw pose” variant. Why does adding 2D supervision increase 3D vertex error?

In Table 2:

The “+jaw pose” row: LVE = 0.235,

The full “Ours (+two stage)” row: LVE = 0.122,

But in the MANGO-Dialog column of Table 1, the full model reports LVE = 1.741, which is much higher than the ablation’s 0.122. This suggests a unit or normalization inconsistency.

Are the ablation metrics computed on mouth vertices only (as in LVE definition), while Table 1 uses full mesh?

Or is there a scaling difference (e.g., mm vs. normalized units)?

Please clarify the metric definitions and scales across tables to ensure comparability.

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
The paper proposes MANGO, a two-stage framework for natural, multi-speaker 3D talking head generation. Stage-1 predicts FLAME parameters (facial expression and articulated head/jaw pose) directly from dual-speaker audio, explicitly modeling speaking/listening interaction. Stage-2 renders the predicted 3D motion with a 3D-Gaussian splatting–based image generator and uses 2D photometric/perceptual losses to lift supervision back to the 3D motion, alternating training between the motion and the renderer. On a new multi-speaker conversational dataset, the method reports improved 3D motion accuracy and 2D visual/lip-sync scores versus recent baselines.

### Strengths
- Modeling both speakers’ audio and the role switch (speaking vs. listening) is well motivated and aligns with emerging conversation-aware talking-head research. This is a non-trivial step beyond speaker-only driving. 

- Alternating 2D-lifted supervision is elegant and plausible. Using a fast differentiable renderer (3D Gaussians) to refine motion predicted in Stage-1 is technically sound and explains the observed improvement in mouth articulation/expressiveness. 

- Evaluation with community-recognized metrics. Reporting LSE-C/LSE-D alongside image-space metrics aligns with established practice in audio-visual lip-sync evaluation.

### Weaknesses
- Limited analysis of listening behaviors. The qualitative figures suggest better non-verbal feedback (nodding, smiles), but there is little targeted measurement of listener' realism beyond global metrics including more diverse non-verbal signals. Consider role conditioned metrics or human studies that separately score speaking vs. listening segments.

- Ablations could isolate Stage-2’s contribution more sharply. It would help to report identical Stage-1 models trained (a) without any 2D-lifted refinement, (b) with only photometric vs. only perceptual losses, and (c) with/without Gaussian renderer fine-tuning.

### Questions
1. How sensitive is Stage-1 to errors in active-speaker detection and speech overlaps? Any quantitative robustness test (e.g., synthetic noise or mis-attribution)?

2. Can the model generalize to unseen speakers and to diverse emotions (e.g., laughter, surprise)? A small cross-emotion test would be informative. Please refer relevant work [1].

[1] LaughTalk: Expressive 3D Talking Head Generation with Laughter, https://arxiv.org/pdf/2311.00994

### Soundness
3

### Presentation
3

### Contribution
3
