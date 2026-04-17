Now I have enough calibration data. Let me synthesize my final review.

Key calibration anchors:
- **GeneOH Diffusion**: Accepted poster (6,6,8,6) — strong new representation + generalizable denoising + 4 benchmarks, but limited baselines and no user study. 
- **HOI-Diff**: Rejected (6,5,5,5) — new dataset annotation + modular diffusion design, but weak metrics and missing baselines.
- **SIGHT**: Rejected (6,3,3,6) — new task formulation + trajectory generation, but limited generality and weak evaluation.
- **IHDiff**: Rejected (5,6,5,6) — first generative model for interacting hands, but limited novelty and baselines.
- **ResDex**: Accepted poster (6,8,6,8) — strong SOTA results on established benchmark, clear generalization.

The paper under review has a meaningful task formulation (free-form HOI) and a new dataset, but significant evaluation gaps. Let me now carefully verify the harsh critic's key claims against the paper text.

The harsh critic claims that in Table 1, TOUCH has lower diversity than baselines. Let me verify: Tab 1 shows Entropy: ContactGen=2.85, Text2HOI=2.85, Ours=2.93; CS: ContactGen=4.93, Text2HOI=5.20, Ours=5.40. Wait — let me re-read the table carefully from the paper text:

| Method | P-IoU↑ | P-F1↑ | MPVPE↓ | PD↓ | PV↓ | Ent.↑ | CS↑ | P-FID↓ | VLM↑ | PS↑ |
| ContactGen | 0.620 | 0.730 | 5.46 | 1.296 | 7.37 | 2.85 | 4.93 | 6.08 | 4.8 | 6.3 |
| Text2HOI | 0.711 | 0.795 | 4.69 | 1.239 | 4.93 | 2.85 | 5.20 | 15.72 | 6.5 | 7.5 |
| Ours | 0.776 | 0.844 | 2.97 | 0.932 | 2.67 | 2.93 | 5.40 | 4.13 | 7.1 | 8.8 |

So Entropy: Ours (2.93) > ContactGen (2.85) = Text2HOI (2.85) — this is HIGHER for TOUCH.
CS: Ours (5.40) > Text2HOI (5.20) > ContactGen (4.93) — this is also HIGHER for TOUCH.
PV: Ours (2.67) < Text2HOI (4.93) < ContactGen (7.37) — this is LOWER for Ours (better since PV↓).

The harsh critic was WRONG about diversity metrics! They claimed TOUCH has "much lower Entropy and CS than ContactGen and Text2HOI" but the actual numbers show TOUCH is HIGHER on both Ent (2.93 vs 2.85, 2.85) and CS (5.40 vs 5.20, 4.93). This is a factual error. The PV column is what the critic seems to have confused with something else.

Let me re-examine the harsh critic's claim more carefully. They wrote: "TOUCH has much lower Entropy and CS than ContactGen and Text2HOI (2.67 vs 7.37 and 4.93; CS 2.93 vs 2.85–2.85), yet the text claims 'our method's ability to generate controllable, diverse… interactions.' By their own metric, TOUCH is less diverse than baselines."

Looking at the numbers: they wrote "2.67 vs 7.37 and 4.93" for entropy — 2.67 is the PV (penetration volume) of Ours, not Entropy! And "CS 2.93 vs 2.85-2.85" — 2.93 is actually the Entropy of Ours, not CS. The critic clearly mixed up which columns correspond to which metrics. The actual diversity metrics (Ent. and CS) both favor TOUCH. This is a major factual error in the harsh critic's review, and this weakness should be REMOVED.

Now let me carefully go through all the weaknesses:

## Summary

The paper introduces the task of Free-Form HOI Generation, extending beyond grasp-centric paradigms to diverse daily interactions (pushing, poking, rotating). It constructs WildO2, a 3D HOI dataset from internet videos (4.4k samples, 92 intents, 610 object categories), and proposes TOUCH, a three-stage framework: contact map prediction via dual CVAEs, multi-level conditioned diffusion (coarse-to-fine text+geometry conditioning), and physical constraints refinement with cycle-consistency loss.

## Strengths

1. **Well-motivated and impactful task formulation**: The paper convincingly argues that existing HOI generation is confined to grasp-centric paradigms and that free-form interactions (pushing, poking, rotating) represent an important and underexplored direction (Sec. 1). This is a genuine conceptual advance.

2. **Thoughtful dataset construction pipeline**: The O2HOI frame pairing strategy using SAM2+RoMa mask transfer is a practical and scalable approach that avoids geometric inconsistencies of diffusion-based inpainting while addressing the occlusion challenge (Sec. 3.1). The multi-level annotation system (SSCs, DSCs, contact maps, 17-part hand segmentation) adds substantial value.

3. **Principled multi-level diffusion design**: The coarse-to-fine conditioning (SSC+global geometry in early blocks; DSC+local contact features in later blocks) is technically clean and aligned with the diffusion process structure (Sec. 4.2, Eq. 4–5). The ablation confirms its importance (Tab. 2, "✗ mul." row).

4. **Elegant self-supervised refinement**: The cycle-consistency loss (Eq. 7) for bidirectional contact mapping is a novel regularizer, and the qualitative improvements in Fig. 6 are convincing. The insight that PD/PV can be deceptively low when the hand drifts away from the object is important and distinguishes free-form HOI from grasp generation (Sec. 5.3).

5. **Comprehensive ablation**: Table 2 ablates all major components (contact guidance, refiner, cycle loss, multi-level conditioning, text levels, text encoders), providing good evidence for design choices.

6. **Demonstrated semantic controllability**: The force-expression analysis (22-25% larger contact area for "firm/tight" interactions, Fig. 9) provides evidence that the model captures fine-grained semantic nuances, not just coarse intent.

## Weaknesses

### Major:

1. **Evaluation does not adequately test generalization to unseen intents or objects** — The train/test split (4:1 by hand part contact category, Sec. 5.1) does not explicitly ensure disjointness on intents (92 categories), object categories (610), or video clips. Since the conditioning signals are text and object geometry, the split should ideally test generalization to novel intent-object combinations. Without this, the metrics largely reflect in-distribution conditional regression rather than the claimed "free-form generation." The out-of-domain experiment (Sec. 5.4.2, Fig. 7) is purely qualitative with no quantitative metrics. This matters because the paper's headline claim is about controllable generation under fine-grained text, but the evaluation does not probe whether the model can generalize to novel combinations or paraphrased instructions beyond training distribution.

2. **Baseline comparisons are limited and asymmetric** — Only two baselines are compared: ContactGen (designed for object-conditioned grasp generation without text) and Text2HOI (a temporal motion model adapted by removing the temporal axis, which is a nontrivial architectural change left underspecified). Neither baseline was designed for static, text-conditioned, free-form HOI generation. While the paper acknowledges adapting them ("To ensure fair comparison, we also augment them with an optimization-based post-processing module to correct hand poses"), the details of this post-processing are absent — losses, iterations, and whether it matches TOUCH's refinement pipeline are not specified. More recent and relevant methods like DiffH2O (Christen et al., 2024) and SemGrasp (Li et al., 2024b) are cited in related work but not included as baselines. This undermines the claim of superiority. However, I note that the authors do augment baselines with post-processing and the setting is new, making exact baseline matching inherently difficult.

3. **Dataset quality is not quantitatively validated** — WildO2 is a primary contribution, but the reconstruction pipeline (O2HOI pairing → image-to-3D → camera alignment → hand-object refinement, Sec. 3.1–3.2) is a long chain of optimization steps with multiple potential failure modes. While the paper mentions "manual inspection and refinement" producing 4,414 samples from 8k clips (~55% acceptance rate), no quantitative validation is provided: no reconstruction error statistics, no comparison with lab-captured ground truth, no inter-annotator agreement on contact maps or DSC quality. Since all training and evaluation uses this data as ground truth, systematic reconstruction errors could undermine claimed performance. The contact map computation ("relative and absolute distance thresholds with bidirectional nearest-neighbor filtering," Sec. 3.3) also lacks validation.

4. **Diversity and semantic consistency metrics are under-specified** — The diversity metrics (entropy, cluster size) in Sec. 5.1 lack clear definitions: what features are clustered, what algorithm is used, and how many clusters. The P-FID metric references Nichol et al. (2022) but does not describe the point-cloud encoder, sample size, or reference distribution. The VLM-assisted evaluation lacks details on prompts and scoring protocol. The user study has only 10 participants with no variance or inter-rater agreement reported. These gaps make key numbers in Table 1 difficult to interpret independently.

### Minor:

1. **Key implementation details are missing**: The text-to-hand-part-mask mapping (Sec. 4.1: "hand-part mask initialized from the fine-grained text TDSC") is critical for the contact CVAE but unexplained. The block split threshold (i<4 vs i≥4) for coarse-to-fine conditioning has no ablation beyond the single "✗ mul." switch. The refiner's TTA iteration count N_tta and computational cost are unspecified.

2. **Contact CVAE independence assumption untested**: The dual CVAEs for hand and object contact maps (Sec. 4.1) model contacts independently, but hand-object contact is inherently coupled. No ablation or justification for this independence assumption is provided.

3. **Removing DSC or SSC text only modestly degrades physical metrics**: In Table 2, removing TDSC or TSSC causes small drops in contact accuracy (0.776→0.698/0.687 P-IoU) and sometimes slightly improves MPVPE/PD/PV. This suggests geometry and contact priors carry most of the capacity for physical plausibility, while the narrative emphasizes fine-grained language as a central driver. The authors should discuss this.

### Trivial:

- The x0-prediction parameterization (vs. noise prediction) for the diffusion model is non-standard and not motivated with a comparison experiment, though this is a well-known alternative.

## Nice-to-Haves

- Quantitative out-of-domain evaluation on Objaverse objects (success/failure rates on unseen objects with standard metrics)
- Cross-dataset evaluation on an established HOI benchmark with real 3D ground truth (e.g., HO3D, DexYCB) to validate that WildO2-based training generalizes
- Computational cost analysis (training/inference time per stage)
- Ablation on joint vs. independent contact map prediction
- More baselines from recent relevant work (DiffH2O, SemGrasp)

## Removed Points

These points are flagged to be removed; treat them with caution.

1. **"TOUCH has much lower diversity than baselines"** (Harsh Critic weakness #4): This is factually wrong. Table 1 clearly shows Ours has HIGHER Entropy (2.93 vs 2.85, 2.85) and HIGHER CS (5.40 vs 5.20, 4.93) than both baselines. The critic confused PV (penetration volume, 2.67 for Ours) with Entropy, and Entropy (2.93) with CS. The diversity metrics actually favor TOUCH, contradicting the critic's claim.

2. **"Baselines are misaligned and likely unfair, undermining claimed superiority"** — While the baseline comparison has issues (limited baselines, adaptation details missing), per the hard rules, I should not flag "unfair comparison with other methods if the asymmetry favors the baseline and not the author's method." The baselines ContactGen and Text2HOI are adapted from different settings, and the asymmetry (they lack TOUCH's contact priors and fine-grained text conditioning) actually favors the baselines in that they get a simpler task structure. The concern about missing methodological details for the baseline augmentation remains valid and is captured in Major weakness #2 above.

3. **"Overclaim about prior work being grasp-only"** (Section notes from harsh critic): The paper does cite specific works and their limitations. Whether some recent methods model non-grasping interactions is a matter of degree, and the paper's characterization is defensible as the dominant paradigm is indeed grasp-centric. This is not a substantive factual error.

4. **Reproducibility concerns about unspecified hyperparameters, latent dimensions, and KL weight β** — These are implementation details that fall under the rule about removing nitpicks on reproducibility.

5. **Demands for confidence intervals on benchmarks** — Single-run evaluation is the norm in this field; this is a nice-to-have at best.

## Novel Insights

The paper's observation that penetration depth/volume metrics are fundamentally deceptive for free-form HOI generation (because hands drifting away from objects score well on PD/PV by avoiding contact entirely) is a genuinely important insight that the community should adopt. This creates a meaningful distinction between the evaluation of grasping tasks (where contact is structurally guaranteed by force closure priors) and free-form tasks (where contact must be actively established and evaluated first). The authors' argument for primacy of contact metrics over penetration metrics is well-supported by their ablation data.

## Suggestions

1. Add a held-out evaluation split based on unseen interaction intents or object categories to directly test generalization, which is the core of the "free-form" claim.
2. Include at least one more recent text-conditioned HOI generation baseline (e.g., DiffH2O, SemGrasp) to strengthen the comparison.
3. Report reconstruction quality statistics for WildO2 (e.g., reprojection error, mesh quality scores) and failure rates at each pipeline stage to establish dataset credibility.
4. Define diversity metrics explicitly and consider adding per-intent-category breakdowns to demonstrate coverage of non-grasping interactions.

## Score and Decision

**Calibration comparison:**
- **GeneOH Diffusion** (Accept poster, avg 6.5): Novel representation + 4 benchmark evaluation + generalization evidence. TOUCH has a similarly novel task formulation and dataset contribution, but weaker evaluation (single in-distribution dataset, limited baselines).
- **HOI-Diff** (Reject, avg 5.25): New dataset annotation + modular diffusion design, but weak metrics and missing baselines. TOUCH is comparable in contribution (new task + dataset + method) but also comparable in evaluation gaps.
- **SIGHT** (Reject, avg 4.5): New task formulation but limited generality. TOUCH has a stronger dataset contribution and method, but similar evaluation concerns.
- **IHDiff** (Reject, avg 5.5): First of its kind but limited novelty and baselines. TOUCH is comparable.

The paper has genuinely novel contributions (task formulation, dataset, method architecture) that advance the field. However, the evaluation protocol does not adequately test the core claims — particularly generalization to unseen intents/objects and text controllability. The dataset quality is not validated, and baselines are limited. These are significant but not fatal gaps: the paper makes real contributions that could be verified with stronger evaluation. It sits between HOI-Diff (rejected with similar gaps) and GeneOH (accepted with stronger evaluation).

**Score: 5.0** — The paper makes a meaningful conceptual and resource contribution but the evaluation does not convincingly substantiate the central claims about free-form generation and text controllability. With stronger evaluation (cross-dataset, held-out intents, more baselines, dataset validation), this could be a strong paper.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>