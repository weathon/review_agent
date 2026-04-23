Now I have all the information I need. Let me write the final consolidated review.

## Summary

The paper introduces Free-Form HOI Generation, extending hand-object interaction synthesis beyond grasp-centric paradigms to diverse interactions like pushing, poking, and rotating. To support this task, the authors construct WildO2, a 4.4k-sample in-the-wild 3D HOI dataset built from internet videos using an automated reconstruction pipeline (O2HOI pairing + image-to-3D + differentiable rendering alignment + hand-object refinement), and propose TOUCH, a three-stage framework: contact map prediction via CVAEs, a multi-level diffusion model with coarse-to-fine text/geometry conditioning, and a physical refinement module with cycle-consistency loss.

## Strengths

- **Novel and well-motivated task formulation.** Extending HOI generation beyond grasping to free-form interactions (pushing, poking, rotating) addresses a genuine limitation in the field. The paper correctly identifies that existing inductive biases toward force-closure grasps limit diversity (Sec. 1, Fig. 1).

- **O2HOI pairing strategy for dataset construction.** The idea of pairing an object-only reference frame with a hand-object interaction frame from the same video clip, then using dense matching to transfer object masks, is a practical solution to the occlusion problem that avoids the geometric inconsistencies of inpainting-based approaches (Sec. 3.1). This is likely the dataset construction contribution with the most reuse potential.

- **Strong ablation evidence for core design choices.** Table 2 provides compelling evidence that the multi-level conditioning and contact map guidance are essential: removing multi-level conditioning ("✗ mul.") drops P-IoU from 0.728 to 0.525, and removing contact maps ("✗ hoc.") drops it further to 0.492. The cycle-consistency loss also shows a measurable P-IoU improvement (0.728 vs. 0.702).

- **Principled coarse-to-fine diffusion design.** The hierarchical injection of global conditions (SSC + global geometry) in early Transformer blocks and local conditions (DSC + contact-point features) in later blocks aligns naturally with the denoising process's progression from global structure to local detail (Eqs. 4–5).

- **Quantitative improvements over existing baselines.** Table 1 shows TOUCH outperforms ContactGen and Text2HOI across nearly all metrics, with notable gains in P-IoU (0.776 vs. 0.711/0.620), MPVPE (2.97 vs. 4.69/5.46), and P-FID (4.13 vs. 15.72/6.08).

- **Fine-grained hand part segmentation beyond grasping assumptions.** The 17-part hand segmentation including dorsal regions, nails, and knuckles (Sec. 3.3) is necessary for non-grasping interactions and goes beyond the coarse inner-hand focus of prior work.

## Weaknesses

### Fatal
None.

### Major

- **Limited baseline comparisons.** Only two baselines are compared — ContactGen (an object-conditioned CVAE using coarse hand part labels, not designed for text conditioning) and Text2HOI (a temporal diffusion model whose temporal axis was removed and which was "adapted for our setting," Sec. 5.2). Both baselines require significant adaptation, and the paper acknowledges they exhibit "noticeable overall hand drift." Notably, DiffH2O (Christen et al., 2024) — a diffusion-based method for synthesizing hand-object interactions from textual descriptions — is cited in the related work section (Sec. 2.3, line 30) but is not used as a baseline despite being directly relevant. With only two baselines, both substantially disadvantaged by the domain shift, the headline improvements in Table 1 cannot be interpreted as strong evidence for TOUCH's superiority over methods designed for related tasks.

- **Fine-grained semantic controllability is the central claim but lacks rigorous evaluation.** The paper's core thesis is enabling "fine-grained semantic controllability" beyond coarse verb-noun pairs (Abstract, Sec. 1). However, the evaluation of this claim relies on: (a) P-FID, which measures distributional similarity, not controllability; (b) a VLM-assisted score whose protocol is entirely undescribed; and (c) a perceptual score from only 10 users. None of these measure whether specific fine-grained textual attributes (e.g., "pushing with the index fingertip" vs. "pushing with the palm") are correctly reflected in the output. A controlled experiment — varying one textual attribute while holding others fixed, then measuring the corresponding change in generated contact patterns or hand pose — would directly test this core claim. The qualitative results (Fig. 8, Fig. 9) are suggestive but not a substitute for quantitative evaluation of the paper's central contribution.

- **Evaluation metrics are measured against unvalidated reconstructed 3D "ground truth."** The entire WildO2 dataset is produced by a multi-stage reconstruction pipeline from single-view internet videos (image-to-3D, single-image hand estimation, differentiable rendering alignment, iterative refinement; Sec. 3.2). While manual inspection removed many failures (8k clips → 4,414 samples), the pipeline has never been quantitatively validated against any independent 3D ground truth (e.g., mocap or multi-view data). Since the model is trained on this data and evaluation metrics (MPVPE, P-IoU, P-F1, PD, PV) all compare against this same reconstructed reference, systematic reconstruction biases would be invisible in the reported numbers. This is a common challenge in the field, and the paper's perceptual study partially mitigates it, but the paper should at minimum discuss how reconstruction errors might propagate into the reported metrics.

### Minor

- **Data split could allow object-instance leakage.** The split is performed within each "hand part contact category" (Sec. 5.1), meaning the same object instance could potentially appear in both train and test sets if it appears across different categories. The paper does not address whether object-level separation was enforced, which could inflate object-conditioned metrics.

- **Baseline post-processing details are unspecified.** The paper states that baselines are augmented with "an optimization-based post-processing module to correct hand poses" (Sec. 5.2) but provides no details about what this module is or how it works, making it difficult to assess the fairness of the comparison.

- **The "large-scale" characterization of WildO2 is generous.** With 4,414 samples across 610 object categories, the average is approximately 7 samples per category, which is quite sparse. While the paper is transparent about these statistics, calling this "large-scale" (Abstract) sets an expectation that the dataset does not fully meet.

- **PD/PV trade-off deserves more honest discussion.** The ablation's argument that PD/PV are "misleading" when hands drift away (Sec. 5.3) is technically valid, but the paper should acknowledge that the refiner module improves contact accuracy at the cost of increased penetration (PD rises from 0.723 without refiner to 1.093 with refiner in Table 2) — this is a genuine trade-off, not just a metric artifact.

### Trivial
None.

## Nice-to-Haves

- A controlled controllability experiment: systematically vary individual textual attributes (contact region, interaction verb, force descriptor) while holding others fixed, and measure the corresponding change in generated outputs — this would directly validate the paper's central claim.
- Validation of a subset of WildO2 reconstructions against multi-view or mocap ground truth, or at minimum a discussion of expected reconstruction error magnitudes.
- Comparison with DiffH2O, which is directly relevant as a text-conditioned HOI generation method.
- Per-intent performance breakdown (with 92 intents, aggregate metrics may obscure performance on rare interaction types).
- Failure mode analysis: what fraction of generated results are physically implausible, and which interaction types fail most?

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic: "Structural: Unvalidated 3D ground truth undermines ALL quantitative claims."** While the concern about unvalidated 3D reconstructions is valid (moved to Major), the framing that it "undermines all quantitative claims" is too strong. The paper does include manual inspection, physical constraints, and a perceptual user study that partially mitigate this. The concern is real but not fatal.

- **Harsh critic: "45% rejection rate highlights how error-prone the pipeline is."** The 55% acceptance rate after manual inspection actually shows quality control is working; it doesn't inherently mean the remaining samples are poor. The real concern is systematic biases in accepted samples, not the rejection rate itself.

- **Harsh critic: "A model that perfectly reproduces the pipeline's systematic errors would score perfectly."** This is theoretically true but speculative — the paper does include perceptual evaluation (PS from 10 users) that would catch egregious errors. The concern is better framed as "metrics should be interpreted with caution" rather than "all metrics are meaningless."

- **Harsh critic: "Calling 4.4k samples 'large-scale' is misleading."** Downgraded to Minor. The paper is transparent about the statistics. Whether "large-scale" is appropriate depends on the reference class — relative to other in-the-wild 3D HOI datasets, this is not small.

- **Harsh critic: "How often does differentiable rendering alignment fail? How sensitive is the result to the initial hand estimate?"** These are reasonable questions but are engineering-level concerns about the pipeline, not scientific weaknesses of the paper's contribution.

- **Harsh critic: "Several design choices are under-justified: the hard switch at block 4, the 10% random dropout rate, the specific choice of N_loc = 128/64."** These are standard hyperparameter choices; the ablation in Table 2 validates the overall design, and individual hyperparameter sensitivity analysis would be nice-to-have, not a weakness.

- **Harsh critic: "The self-supervised cycle consistency loss (Eq. 7) is interesting but the paper doesn't analyze how well it works in isolation from the other physical losses."** The ablation in Table 2 does show the effect of removing Lcyc (P-IoU drops from 0.728 to 0.702), which is a reasonable analysis. Isolating it from all other physical losses would be a nice-to-have but is not required.

- **Harsh critic: "The claim that existing methods 'fail to capture the rich diversity of daily HOI' even with elaborate language is stated without evidence."** This is a reasonable motivation claim; the paper's entire contribution is to demonstrate that existing methods are limited. The qualitative comparison in Fig. 5 and quantitative results in Table 1 provide some evidence.

- **Strength finder: "Out-of-domain generalization to novel objects."** This is based on Fig. 7, which is qualitative only. Without quantitative metrics on held-out object categories, this is an unsupported claim, not a strength. Moved to Nice-to-Have.

- **Strength finder: "Demonstrated semantic controllability including force expression."** The 22–25% larger contact area finding is interesting but is presented in the Discussion section (5.4.3) without formal experimental protocol. This is suggestive but not rigorous enough to list as a standalone strength. The qualitative evidence (Fig. 9) supports it partially.

## Novel Insights

The paper reveals an important tension specific to free-form HOI evaluation: traditional penetration metrics (PD, PV) become "deceptively" favorable when a generated hand simply drifts away from the object, since no contact means no penetration. This insight — that metrics designed for grasping scenarios can be actively misleading for non-grasping interactions — is a genuine methodological contribution that should inform future evaluation design in this space. The paper's proposed prioritization of contact accuracy metrics over penetration metrics for free-form HOI is a useful, if preliminary, step toward appropriate evaluation.

## Suggestions

- Design a controlled controllability evaluation: for a fixed object, generate interactions with systematically varied text prompts (e.g., change only the hand part or interaction verb), then measure whether the generated contact maps and hand poses change accordingly. This single experiment would substantiate the paper's central claim.
- Add DiffH2O as a baseline comparison. Since it generates HOI from text descriptions and is already cited in the paper, the comparison would be natural and would significantly strengthen the empirical evaluation.
- Discuss the limitations of the evaluation framework arising from reconstructed ground truth, including an estimate of expected reconstruction error magnitudes and how they might propagate into the reported metrics.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| CLUTCH | /home/wg25r/review_agent/human_reviews_2026/W7YRskO47j.md | 5.0 | Closest analog: hand motion from internet videos, 3D reconstruction pipeline, dataset quality concerns, limited baselines. TOUCH has a more novel task formulation but weaker baselines (2 vs 3+) and less supported central claim. |
| HOIGS | /home/wg25r/review_agent/human_reviews_2026/JWBcEPzM89.md | 4.5 | Monocular HOI reconstruction, limited baselines, quality concerns. TOUCH is stronger due to a generation task with more complete experiments. |
| SynHLMA | /home/wg25r/review_agent/human_reviews_2026/EzJowEZ1UJ.md | 5.5 | Hand-object interaction generation with discrete representation, overclaimed controllability. Similar weakness profile. |
| ROBOWHEEL | /home/wg25r/review_agent/human_reviews_2026/VBVCqm2t1J.md | 4.0 | Data engine from HOI videos with multi-stage pipeline, limited baselines. TOUCH has clearer task formulation and stronger ablations. |
| HOI via VLM-guided RL | /home/wg25r/review_agent/human_reviews_2026/LfkPlFTfe0.md | 7.0 | High-scoring HOI paper with comprehensive ablations and thorough empirical validation. TOUCH falls short of this bar due to baseline and evaluation gaps. |
| WetBench | /home/wg25r/review_agent/human_reviews_2026/SxFOEwQLMT.md | 2.0 | Fundamentally flawed circular evaluation. TOUCH is much stronger — its ground truth, while noisy, is not circular. |
| MACEval | /home/wg25r/review_agent/human_reviews_2026/gOQ4x4Ykyg.md | 2.0 | Unvalidated ground truth with overclaimed benchmark. TOUCH is significantly stronger with real reconstruction and manual inspection. |

TOUCH has genuine contributions — a novel task formulation, a well-designed method with strong ablation evidence, and a practical dataset construction pipeline — but these are undermined by insufficient baselines, an unsupported central claim about fine-grained controllability, and evaluation against unvalidated reconstructed ground truth. Compared to CLUTCH (5.0, the closest analog), TOUCH has a more novel task but weaker empirical evidence for its core claims. It sits above the HOIGS/ROBOWHEEL tier (4.0–4.5) due to stronger ablations and a more clearly novel contribution, but below the 6+ tier due to the evaluation gaps. A score of 5.0 reflects a paper with real contributions that are not yet convincingly established.

**Evaluation on axes:**
- **Originality:** Good. The free-form HOI task formulation is novel and the O2HOI pairing strategy is creative.
- **Importance of research question:** Good. Moving beyond grasping is important for applications.
- **Claims well supported:** Moderate. Ablations are strong, but the central controllability claim lacks direct evidence, and baselines are thin.
- **Soundness of experiments:** Moderate. The unvalidated ground truth and limited baselines are significant concerns.
- **Clarity of writing:** Good. The paper is well-structured and clearly written.
- **Value to community:** Good. The dataset and task formulation have reuse potential.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>