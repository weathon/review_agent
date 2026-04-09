## Summary

WithAnyone addresses the "copy-paste" artifact in identity-consistent image generation, where models over-replicate reference images rather than synthesizing identities under new conditions. The paper introduces three contributions: (1) MultiID-2M, a large-scale paired multi-identity dataset with 500k group photos and ~1M reference images; (2) MultiID-Bench, a benchmark with a novel Copy-Paste metric ($M_{CP}$) that quantifies the trade-off between identity fidelity and copy-paste artifacts; and (3) the WithAnyone model built on FLUX, using a GT-aligned ID loss, an ID contrastive loss with extended negatives, and a four-phase training pipeline that transitions from reconstruction to paired identity-conditioned generation.

## Strengths

- **Formalizing and measuring the copy-paste artifact**: The $M_{CP}$ metric (Eq. 2) is a genuine contribution—it operationalizes a widely recognized but poorly measured failure mode. Unlike raw Sim(Ref) which rewards trivial copying, $M_{CP}$ captures the *relative* bias toward the reference versus ground truth. The metric shows moderate positive correlation with human judgments (Pearson $r=0.44$, Table 7), and the GPT-4o anomaly on TV-series identities provides natural validation of its discriminative power (Sec. F.3).

- **MultiID-2M fills a real data gap**: The dataset provides 500k paired group photos with ~400 reference images per identity across ~3k identities, directly enabling training paradigms (paired supervision, extended negative pools) that reconstruction-only datasets cannot support. Table 4 shows the scale advantage over existing multi-ID datasets (PIPA: 40k, MHP: 5k).

- **GT-aligned ID loss is a clean technical innovation**: Using GT landmarks for face alignment during training (Eq. 14) avoids noisy landmark extraction from intermediate denoised images, enables ID supervision at *all* noise levels (not just $t < 0.25$ as in PortraitBooth), and leverages FLUX's single-step velocity prediction for efficiency. Fig. 7 provides empirical support that this yields lower error at low noise and more informative gradients at high noise.

- **Compelling trade-off breaking**: Fig. 5 is a strong result—WithAnyone visibly deviates from the regression curve that all other methods lie on, achieving the highest Sim(GT) while maintaining lower copy-paste than methods with comparable identity fidelity. This is the paper's central claim and it is well-supported.

## Weaknesses

### Major:

- **Confounding between data scale and methodological contribution**: The paper attributes performance gains to the training paradigm (paired data, contrastive loss, GT-aligned loss), but the ablation in Table 3 compares "FFHQ only" (70k images, no paired data, no contrastive loss) versus the full system (2M images with all components). This confounds dataset scale with loss design. A cleaner isolation—training with the same MultiID-2M data but using only reconstruction loss, or applying the proposed losses on a public dataset like FaceID-6M—would strengthen the claim that the *loss functions and training strategy* (not just data scale) are responsible for breaking the trade-off.

- **Benchmark's dependence on ground-truth limits evaluation scope**: The $M_{CP}$ metric and Sim(GT) both require a ground-truth image, restricting evaluation to reconstruction-adjacent scenarios where a target image exists. This does not fully test the model's capability for open-ended, prompt-driven generation (e.g., "put this person in a cyberpunk city" with no GT). While CLIP-T is reported for prompt adherence, the paper's central quantitative claims rest on GT-dependent metrics. The authors should clarify the scope of what MultiID-Bench evaluates and discuss its limitations for assessing truly controllable generation beyond reconstruction.

### Minor:

- **User study is underpowered**: With only 10 participants ranking 230 image groups, the study lacks sufficient statistical power for the strong claims about human preference across four dimensions. No inter-annotator agreement metric (e.g., Krippendorff's $\alpha$) is reported, making it difficult to assess the reliability of the human evaluation. The correlation analysis (Table 7) helps, but the study design itself is a limitation.

- **Limited quantitative evaluation on non-celebrity identities**: Generalization beyond celebrities is shown only qualitatively in Fig. 16 (3 examples from OmniContext). Given the model's training on celebrity data and the practical importance of generating non-public identities, quantitative results on a standard non-celebrity test set would strengthen the generalization claim.

- **Inconsistency in similarity threshold reporting**: The main text (Sec. 3) states the identity assignment threshold as 0.4, while the appendix (Sec. C.1) reports 0.5. This inconsistency, while minor, raises questions about the dataset construction's precision and should be clarified.

### Trivial:

- The negative pool ablation (Table 3, "w/o Ext. Neg.") shows a clear drop from 4096 to 63 negatives, but a graded analysis (e.g., 256, 1024, 4096) would better characterize the scaling behavior of this component.

## Nice-to-Haves

- Apply the proposed loss functions and training pipeline to a public dataset (e.g., FaceID-6M) to cleanly isolate the methodological contribution from data scale.
- Evaluate identity blending and similarity scaling as the number of subjects increases beyond 4 (e.g., 5–8 people), to stress-test the multi-ID capability.
- Report inference speed and VRAM usage relative to baselines to contextualize practical deployment trade-offs.
- Expand the user study to ≥30 participants with inter-annotator agreement reporting.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Ethical/legal concerns about Right of Publicity superseding CC licenses**: While a legitimate societal concern, the paper includes a comprehensive Ethics Statement (Sec. 7 + Appendix) addressing data sourcing (CC-licensed, publicly known figures only), anonymization (no names or identity labels in training), non-commercial release, and recommended mitigations (consent verification, watermarking, abuse monitoring). The concern as raised overstates the paper's oversight—the authors have addressed this substantially, even if not perfectly.

- **Unfair baseline comparison (baselines not tuned on MultiID-Bench)**: The paper uses "official implementations and checkpoints (or API) with default settings" (Sec. F.1). Using published configurations is standard practice and, if anything, favors the baselines by giving them their best-published performance. Per review rules, this is removed.

- **Architecture agnosticism (testing on UNet/SDXL)**: The paper explicitly builds on FLUX. Demanding verification across other architectures is scope creep—the contribution is the training paradigm and data strategy, not architecture-specific modifications. Applying to other architectures would strengthen the paper but is not a core flaw.

- **Training cost as a weakness**: The 4-phase pipeline on 8 H100s is resource-intensive but standard for large-scale diffusion model training. This is a practical consideration, not a methodological flaw, and is more appropriately a nice-to-have discussion point.

- **Demand for error bars / multiple seeds on Fig. 5**: Large-scale benchmark evaluation with single runs is standard practice in this area. While variance reporting would be ideal, its absence does not undermine the results, especially given the complementary user study and ablation support.

- **GT-aligned loss constraining pose controllability at inference**: This concern misunderstands the mechanism. The GT-aligned loss is only applied during training; at inference, no loss is computed. The paired training (Phase 3) explicitly uses different reference/GT pairs to prevent pose locking. The loss teaches the model to focus on identity features in a consistently aligned space, which should improve rather than hinder pose flexibility.

## Novel Insights

The paper reveals a fundamental tension in ID-preserving generation that has been hiding in plain sight: the community's primary evaluation metric (Sim(Ref)) inadvertently *rewards* the failure mode it should penalize. By formalizing this as the $M_{CP}$ metric and demonstrating that most SOTA methods lie on a fidelity-artifact trade-off curve, the paper exposes how optimization for the wrong signal has shaped the field's progress. The insight that paired data (where reference and target are different images of the same person) naturally breaks the reconstruction shortcut is simple but powerful—it reframes copy-paste not as a model capacity issue but as a *data objective* issue.

## Suggestions

- Add a "same data, different loss" ablation: train a model on MultiID-2M using only reconstruction loss for the same number of steps, then compare. This would cleanly demonstrate that the proposed losses (not just data scale) drive the trade-off breaking.
- Clarify the similarity threshold inconsistency (0.4 vs 0.5) between main text and appendix with a single consistent value and justification.
- Expand non-celebrity evaluation to include quantitative metrics on a holdout set of non-public identities (e.g., from OmniContext's real-human subset), not just the 3 qualitative examples in Fig. 16.