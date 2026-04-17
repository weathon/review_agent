Now I have enough context from the paper itself and calibration papers. Let me compile the final review.

## Summary

WithAnyone formalizes the "copy-paste" failure mode in identity-consistent image generation (where models replicate reference faces rather than preserving identity under natural variation), and addresses it through three contributions: (1) MultiID-2M, a large-scale paired dataset of 500k labeled group photos with hundreds of references per identity; (2) MultiID-Bench, a benchmark with a novel copy-paste metric (M_CP) measuring bias toward reference vs. ground truth; and (3) WithAnyone, a FLUX-based model using GT-aligned ID loss, ID contrastive loss with extended negatives, and a four-phase training pipeline. Experiments on MultiID-Bench across 14 baselines show WithAnyone achieves the highest Sim(GT) among identity-specific methods while maintaining lower copy-paste scores.

## Strengths

- **Problem articulation is excellent.** The copy-paste artifact is a real, prevalent, and under-discussed failure mode. Fig. 2 convincingly illustrates the gap between natural identity variation and model over-adherence to references. Formalizing this as a measurable quantity rather than an anecdotal observation is a genuine service to the community.

- **MultiID-2M fills a real data gap.** Providing ~3k identities with ~400 references each, plus 500k labeled group photos and 1.5M unpaired group photos, is a substantial and well-motivated resource. Paired multi-reference data is genuinely scarce, and reconstruction-only training is the root cause of copy-paste behavior.

- **The GT-aligned ID loss is a clean, practical innovation.** Using ground-truth landmarks for ArcFace alignment during training (instead of noisy landmark extraction from generated images) enables applying identity supervision at all noise levels with negligible overhead. This directly addresses a known limitation of prior work (e.g., PuLID's expensive full denoising, PortraitBooth's limited supervision window).

- **Comprehensive baseline comparison.** Evaluating against 14 methods spanning both general customization and face-specific models, on both single-person and multi-person subsets, provides a broad empirical picture. The four-phase training pipeline and extended-negative contrastive loss are conceptually well-motivated, and the ablation (Table 3) shows each component contributes.

- **The paired-tuning idea (Phase 3) directly targets copy-paste.** Replacing 50% of reconstruction samples with distinct images of the same identity is a straightforward but effective strategy that breaks the direct-copy shortcut.

## Weaknesses

### Major:

- **The copy-paste metric (M_CP) is under-validated and partially entangled with Sim(GT).** By construction, M_CP = (θ_gt − θ_gr)/max(θ_tr, ε). Any method that improves Sim(GT) will tend to receive a lower (better) CP score, since being closer to GT automatically reduces θ_gt. This means the two primary headline metrics (high Sim(GT), low CP) are not fully independent. The paper reports only "moderate positive correlation" with human judgments on copy-paste (Sec. 6.3) without reporting correlation coefficients, significance tests, or failure cases in the main text. For a metric that is the cornerstone of the paper's conceptual framing, this is insufficient validation. A stronger paper would show: (a) quantitative correlation with human ratings (with confidence intervals), (b) examples where M_CP agrees/disagrees with human perception, and (c) evaluation using an alternative face embedding (e.g., AdaFace) to decouple train-eval alignment.

- **"Breaking the trade-off" is overclaimed.** The abstract and introduction state the method "breaks the long-observed trade-off between fidelity and copy-paste." What the experiments actually show is that WithAnyone achieves a better operating point in the Sim(GT)–CP plane on one benchmark (Fig. 5). This is an *improvement*, not evidence of a qualitative structural change in the trade-off. The scatter plot is shown without confidence intervals, without varying λ_ID and λ_CL to produce a Pareto curve, and without statistical testing of whether the improvement over the regression trend is significant. Demoting this to "substantially mitigating" or "achieving a favorable trade-off" would be more honest.

- **ArcFace is used for both training and evaluation, creating metric alignment.** ArcFace embeddings provide the clustering/labeling for MultiID-2M, the ID loss (L_ID), the contrastive loss (L_CL), and all evaluation metrics (Sim(GT), Sim(Ref), CP). A model explicitly trained to optimize ArcFace distances can be expected to score well on ArcFace-based metrics relative to methods not trained with this objective. The paper does not include evaluation under an alternative face embedding to verify that the improvements generalize beyond ArcFace.

- **Benchmark fairness concern: WithAnyone trains on data closely aligned with MultiID-Bench's domain.** MultiID-2M consists of celebrity group photos, and MultiID-Bench evaluates on similar celebrity images with "no overlap to training data." However, several baselines (FLUX-Kontext, OmniGen, GPT-4o, etc.) are general-purpose models not trained on this curated identity-dense data. While this comparison is informative, it structurally favors WithAnyone in the same way that evaluating a medical model on in-domain medical data would favor it over a general model. The paper should acknowledge this limitation more explicitly and, ideally, include comparison with at least one face-customization baseline that is retrained/fine-tuned on comparable data.

### Minor:

- **Incomplete ablation of the four-phase pipeline.** Table 3 ablates Phase 3 (paired tuning) and the two losses, but not Phases 1, 2, or 4 independently. Since the four-phase pipeline is a central design choice, showing that each phase matters (or that some can be skipped) would strengthen the contribution. Similarly, the contrastive loss is ablated only at 63 vs. 4096 negatives—no sweep over intermediate sizes is provided.

- **Small user study.** Only 10 participants evaluated 230 image groups. While the results are consistent with the quantitative metrics, this sample size is below typical standards for perceptual evaluation. No inter-rater agreement or confidence intervals are reported in the main text.

- **Lower aesthetic scores.** WithAnyone achieves Aes = 4.783 in Table 1, the lowest among all compared methods. While identity fidelity and controllability are the primary goals, the degradation in aesthetics is not discussed or analyzed. The quality-tuning phase (Phase 4) was intended to address this, yet the model still lags behind.

- **Train-test identity separation is underspecified.** The paper states "no overlap to training data" but does not detail how identities were deduplicated between MultiID-2M and MultiID-Bench, or whether identities in CelebA-HQ/FFHQ/FaceID-6M (used in Phases 1–2) overlap with test identities. Given that ArcFace clustering can mis-group near-duplicates, a more rigorous deduplication protocol should be described.

## Nice-to-Haves

- Evaluate identity similarity using an alternative face embedding (e.g., AdaFace) to verify that improvements are not artifacts of ArcFace alignment.

- Sweep negative pool sizes (128, 256, 512, 1024, 4096) to justify the 4096 design choice.

- Report Pareto curves by varying λ_ID and λ_CL to characterize the actual trade-off frontier, rather than presenting a single operating point.

- Include evaluation on at least one established single-ID benchmark (e.g., PhotoMaker or IP-Adapter test sets) to demonstrate backward compatibility.

- Expand the user study to 20–30 participants and report Krippendorff's alpha and correlation coefficients between human ratings and M_CP.

## Removed Points

- **"The CP metric requires GT images, limiting its practical applicability."** (Neutral reviewer) — This is inherent to any ground-truth-based benchmark evaluation. All standardized benchmarks (ImageNet accuracy, COCO mAP, etc.) require ground truth. The metric is for evaluation, not deployment monitoring. Removing this as a weakness.

- **"Dataset bias toward celebrities inflates identity similarity."** (Human finder, referencing PersonalVideo review) — While celebrity bias is a legitimate generalizability concern, the paper explicitly states its domain is celebrity images and the task is multi-ID generation. This is a scope limitation, not a flaw per se. Demoting to a discussion point.

- **"CC-licensing of celebrity images is questionable."** (Spark) — This questions the existence/availability of the dataset under claimed licensing. Per hard rules, if the paper cites/claims it, I assume it exists and the licensing claims are accurate. Removing.

- **"DynamicID excluded due to unavailability."** — The paper explains why it's excluded (footnote 1). This is a standard exclusion, not a weakness of the paper under review.

- **"The GT-aligned loss creates a train-inference gap since GT landmarks are unavailable at inference."** (Spark) — This misunderstands the method. The GT landmarks are used only during *training* to align the generated face for ArcFace extraction. At inference, no landmark alignment is needed; the model simply generates the image. The point of using GT landmarks during training is precisely to avoid the noisy landmark extraction problem at inference. Removing as a strawman.

- **"No failure case analysis."** (Spark) — While always helpful, the paper does show qualitative comparisons where other methods fail (Fig 6) and discusses copy-paste failure modes extensively. This is a nice-to-have, not a weakness.

- **"Computational cost not reported."** (Spark) — Per soft rules, reproducibility nitpicks about training cost are not core weaknesses for a methods paper. Training cost for diffusion models is inherently large and reporting specific GPU hours is nice-to-have, not required.

## Novel Insights

The paper's most novel insight is that reconstruction-based training is not merely sufficient but actively *harmful* for controllable identity generation—it builds in a shortcut that makes copy-paste the path of least resistance. The formalization of copy-paste as a *relative* phenomenon (closer to reference than to GT, normalized by their natural separation) is conceptually interesting even if metric validation remains incomplete. The paired-training insight (using distinct images of the same identity as input and target) is simple but powerful and should influence future work on ID-consistent generation.

## Suggestions

- Replace "breaks the trade-off" with "achieves a substantially more favorable trade-off" throughout the paper, and provide statistical evidence (e.g., Pareto analysis across λ settings) if the stronger claim is to be kept.

- Add evaluation under at least one alternative face recognition model (e.g., AdaFace or a different ArcFace checkpoint) to demonstrate that improvements are not artifacts of embedding alignment.

- Report quantitative user study results in the main text: correlation coefficient between M_CP and human copy-paste ratings, inter-rater agreement, and confidence intervals.

- Include at least one face-customization baseline (e.g., PuLID) retrained on MultiID-2M or comparable data, to separate the data advantage from the methodological contribution.

## Score and Decision

**Calibration comparison:**

- **DreamBench++** (6,6,6,6 → Poster): A benchmark+dataset paper with moderate validation. This paper is comparable in contribution type (dataset + benchmark + method) but has stronger methodological novelty and more baselines.
- **ID-Booth** (3,3,3,3 → Rejected): Weak quantitative results, limited novelty. This paper is clearly stronger—its contributions are more substantial and its results more convincing.
- **FUSION IS ALL YOU NEED** (3,6,3,3 → Rejected): Limited novelty, weak identity fidelity. This paper has more novelty (copy-paste formalization, new metric, new dataset) but similar overclaiming issues.
- **UIFace** (6,6,6 → Poster): Decent contribution to face generation diversity with ablation concerns. Similar profile—meaningful but not revolutionary.
- **MGFR** (6,8,8 → Spotlight): Face restoration + dataset contribution, stronger validation. This paper is weaker in validation of its central metric.
- **Progressive Compositionality** (6,8,8,8 → Spotlight): Novel loss + curriculum + dataset, strong results. This paper is weaker in novelty of the loss design and in validation rigor.

This paper sits in the 5–6 range. It makes real and significant contributions (the dataset alone is valuable, the copy-paste formalization is novel and important), but the main metric is under-validated, the "breaking trade-off" claim is overstated, and evaluation is primarily on the authors' own benchmark with an architecture-aligned embedding space. These are significant but not fatal. The paper would be substantially strengthened by metric validation using alternative embeddings and more honest framing of the trade-off claim.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>