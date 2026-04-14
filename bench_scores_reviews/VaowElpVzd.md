## Summary
Co³Gesture addresses concurrent two-speaker co-speech 3D gesture generation, a genuinely underexplored problem distinct from single-speaker gesture synthesis. The authors contribute GES-Inter, a 70-hour, 7M-frame pseudo-labeled dataset of dyadic conversations with SMPL-X mesh, facial, phoneme, and text annotations, and propose a bilateral cooperative diffusion framework that uses a Temporal Interaction Module (TIM) and mutual attention to model asymmetric interaction dynamics between the two speakers simultaneously.

## Strengths

- **Fills a concrete dataset gap with scale and modality breadth.** GES-Inter is the only concurrent co-speech gesture dataset with SMPL-X mesh-based whole-body annotation, facial expression, phoneme, and text, at 70 hours — dwarfing the only prior concurrent dataset TWH16.2 (17 hours, joint-only, no facial). This directly enables downstream tasks beyond gesture generation (talking face, behavior analysis).

- **Bilateral asymmetric architecture is principled and well-motivated.** The observation that conversational motion is asymmetric (one speaker active, one passive/responding) is not merely asserted but drives a concrete design choice: separate denoising branches conditioned on individual separated audio, rather than a holistic shared model. The ablation in Table 4 (FGD: 0.769 vs. 1.669 for single-branch) provides strong quantitative validation of this insight.

- **Ablation suite is comprehensive and diagnostic.** Tables 3–5 systematically ablate TIM, mutual attention, bilateral branches, mixed audio, audio separation, and foot contact loss, with each component producing a meaningful performance gap. This is a genuine strength — each ablation corresponds to a stated design decision rather than a generic sweep.

- **Performance margins over adapted baselines are substantial.** A >24% FGD improvement over the best competitor (InterGen, which already uses bilateral branches but lacks interaction modeling) specifically isolates the value of the TIM and mutual attention, not merely the bilateral structure.

## Weaknesses

### Fatal
None.

### Major

- **No quantitative dataset quality assessment undermines the foundation of all evaluations.** All 3D poses are pseudo-labeled by PyMAF-X on in-the-wild video. The paper reports going from 20M raw frames to 7M after filtering but gives no filtering criteria, no estimated pose error rate, no analysis of hand articulation quality (critical for gestures), and no reprojection error or comparison to a manually corrected subset. Since FGD is computed using an autoencoder trained on GES-Inter, if the pseudo-labels contain systematic bias or noise, the reported FGD improvements could partly reflect that the model learns the biases of the estimator, not true gesture quality.

- **No interaction-specific metric is a critical evaluation gap.** The paper's core contribution is interaction coherency, but all quantitative evaluation uses FGD (distribution realism), BC (speech-rhythm alignment), and Diversity — none of which measures inter-speaker coherency. The paper acknowledges this in limitations ("we will put more effort into designing specific interaction metrics"), but this is the central claim and it is only supported by an underpowered user study. Metrics like cross-speaker motion correlation, reaction latency, or turn-taking synchrony should accompany the submission.

- **The most important ablation baseline is missing: two independent single-speaker models.** The paper ablates bilateral vs. single-branch (Table 4), but the single-branch ablation fuses both speakers into one model holistically. What is absent is the baseline of running a single-speaker gesture model (e.g., the best baseline DiffSHEG or TalkSHOW) twice in parallel on separated audio with no interaction module at all. If this independent-generation baseline performs comparably on FGD and BC, the paper's core claim that explicit interaction modeling (TIM, mutual attention) is necessary collapses. This is the single most critical missing experiment.

- **Method reproducibility is insufficient.** Key implementation details are either missing or ambiguous: (a) In Eq. (1), the same projection matrix **W** is used for Q, K, and V across the cross-attention, which conflicts with standard multi-head attention using separate W_Q, W_K, W_V — this is either a notation shortcut or a real design, but it is unexplained. (b) The shape/dimensionality of σ from Eq. (2) is never specified — it is called a "learnable weight parameter" but could be scalar, per-time-step (N×1), or per-channel (1×D), which dramatically changes what TIM actually does. (c) The denoiser backbone is described only as "transformer-based diffusion branches with 8 blocks, 8 heads" — how conditioning signals (C_a, C_mix, t) are injected is unspecified. The paper cannot be reproduced from the main text alone.

### Minor

- **User study is underpowered.** 15 volunteers rating 2 videos per method (≈30 ratings per method) with no statistical significance tests reported. Differences of 0.3–0.4 points on a 0–5 scale with this sample size are unlikely to be statistically robust. This is not sufficient evidence for broad perceptual claims.

- **Per-speaker-role breakdown absent.** The paper emphasizes that conversational dynamics are asymmetric (active speaker vs. listener), but FGD and BC are averaged across both speakers. Reporting these metrics separately for the speaking role vs. the listening role would directly validate whether the model captures asymmetry or merely averages it out. Without this, the asymmetry argument is design motivation without quantitative validation.

- **Baseline adaptation fairness is partially documented.** DiffSHEG and TalkSHOW use their original audio encoders (HuBERT and Wav2vec respectively), while all other methods use the paper's audio encoder. Although both DiffSHEG and TalkSHOW are retrained from scratch on GES-Inter, the encoder difference is a confound. The paper should report ablations showing that the encoder choice alone does not explain the gap.

- **Audio separation quality is unquantified.** The entire pipeline depends on pyannote-audio correctly separating and assigning speakers. No diarization error rate (DER) or signal-level separation quality (SDR) is reported, and no analysis of how separation errors propagate to gesture generation quality is provided.

### Tiny

- **Problem formulation inconsistency.** §3.2 defines the task as "given C_mix, generate x," but §3.3 makes clear the actual inputs are C_mix, C_a, and C_b. The formal problem statement should match the actual inference inputs.

- **Train/test split protocol not fully specified.** The paper splits 27,390 clips at 85/7.5/7.5% "following criteria" from Liu et al., but does not state whether the split is by clip, video, speaker identity, or show. A clip-level split from the same video/speaker across train and test would inflate generalization estimates.

- **The paper does not explain whether the foot contact loss provides a meaningful gradient for upper-body generation** when the lower body is imputed as T-pose. The ablation shows a benefit, but the mechanism (whether FK propagates a useful gradient into upper-body joints from a fixed lower-body) should be clarified.

## Nice-to-Haves

- Visualize the learned σ values from TIM across time for speaking vs. listening states. If σ is near 0.5 uniformly, the gating mechanism is uninformative. If it is state-dependent, showing this would substantially strengthen the paper's interpretability.
- Report audio separation quality metrics (DER, speaker-level SDR) and correlate separation quality with gesture generation quality to address the most fragile step in the pipeline.
- Add a speaker-swapped input experiment: swapping Speaker A and B audio should produce appropriately swapped gesture dynamics if asymmetric modeling works. This would directly validate the bilateral asymmetric design.
- Include failure case visualization (jittery motions, wrong-speaker reactions, misaligned interactions). This strengthens credibility and helps the community understand failure modes.
- Discuss dataset split by speaker identity to avoid train/test leakage, and add an ethics/data governance note regarding in-the-wild video usage and annotation redistribution rights.
- Shared vs. separate denoiser weights ablation: the marginal-distribution invariance argument for shared weights is stated but untested. If separate weights perform similarly, the symmetry assumption holds; if not, it may reveal speaker-role imbalance in the data.

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **"Related work is catalog-like"** (Harsh Critic): Removed as a generic writing quality critique without specific factual errors.
- **"Table 1 incomplete — missing speaker count, scene count, overlap statistics"** (Harsh Critic): Removed as a pure formatting/completeness nitpick; the relevant comparative dimensions (duration, modalities, concurrent gestures) are present.
- **"Evaluation on TWH16.2 required"** (Spark Finder): Removed; the paper explicitly notes that TWH16.2 uses joint-based representation incompatible with SMPL-X mesh, which is a legitimate technical barrier. If compatible, this would be valuable — retained as a nice-to-have.
- **"The first large-scale claim should be qualified"** (Harsh Critic): Removed; under the specific criteria of mesh-based, concurrent, multimodal co-speech, the claim appears technically defensible and Table 1 supports it.
- **"Forward process notation differs from standard DDPM"** (Harsh Critic): Removed; EDM-style and other simplified forward-process parameterizations are valid and published alternatives.
- **"Foot contact loss conceptually odd for upper-body model"** (Harsh Critic): Removed; the paper directly addresses this: lower-body joints are completed as T-pose via FK during loss computation. The ablation result in Table 5 empirically validates the benefit.
- **"No ethics/data governance section"** (Harsh Critic): Downgraded to nice-to-have. While important for camera-ready, it is not substantive to technical evaluation.
- **"The claim 'we introduce the new task' is somewhat overstated"** (Harsh Critic): Removed; TWH16.2 exists but at 17 hours with incompatible representation. The mesh-based concurrent formulation with multi-modal annotation is sufficiently distinct.
- **"Unfair comparison from DiffSHEG/TalkSHOW using original encoders"** (multiple reviewers flag this): Per rules, this asymmetry is unfavorable to the proposed method (using potentially weaker encoders for baselines retrained from scratch), hence such unfair comparisons favoring baselines are excluded. Retained as a minor documentation request only.

## Novel Insights

The most genuinely novel insight emerging from synthesis of the three reviews goes beyond what the paper itself highlights: the paper's asymmetric bilateral design tacitly distinguishes between two different notions of "interaction coherency" — (1) each speaker's individual gesture being rhythmically aligned with their own speech (captured by BC), and (2) the cross-speaker temporal coherency of one speaker's response to the other's movement (not captured by any reported metric). This distinction is important because it is entirely possible for a model to achieve strong BC on both branches independently by simply ignoring the partner, thus appearing to do well on all reported metrics without ever modeling genuine social interaction dynamics. The missing "two independent models" baseline and the absence of cross-speaker metrics are not merely experimental oversights — they are the precise gap between the paper's claimed contribution and what the current evaluation actually demonstrates. An explicit per-role BC decomposition and a cross-speaker motion correlation metric could be defined without new data collection and would either strongly validate or fundamentally qualify the paper's central claim.

## Suggestions

1. **Add the independent two-speaker baseline immediately.** Run the best-performing single-speaker baseline (DiffSHEG or TalkSHOW) twice with separated audio, no cross-branch communication. Report FGD, BC, and Diversity. This is the paper's most critical missing experiment and is feasible to run.

2. **Define and report at least one interaction-specific metric** before claiming interaction coherency quantitatively. Cross-speaker motion beat correlation (adapting BC to measure synchronized beat events across both speakers) or inter-speaker motion energy correlation are feasible without user studies.

3. **Report filtered-frame statistics and pose quality proxies in the appendix.** At minimum: proportion of frames discarded, which failure criteria trigger filtering (e.g., PyMAF-X confidence threshold), and re-projection error distribution on a held-out subset. This is necessary for the dataset to be trusted as ground truth for FGD.

4. **Clarify Eq. (1)** — specify whether W is the same across Q, K, V (unusual), or whether this is simplified notation for three separate learned projections. Specify the shape of σ in Eq. (2) (N×D, N×1, or scalar).

5. **Specify the train/test split criterion** (clip-level vs. video-level vs. speaker-identity-level) and report whether the FGD autoencoder is trained on the training split only or the full dataset.

6. **Report per-speaker-role BC** (active speaker vs. listener) to empirically validate the asymmetric design claim, rather than only reporting the average.