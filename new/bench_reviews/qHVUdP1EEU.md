## Summary

Jigsaw++ proposes a generative framework that learns complete shape priors from partially assembled 3D objects to aid reassembly. It maps point clouds to RGB images via coordinate encoding, leverages a pretrained image-to-3D model (LEAP/DINOv2) with rectified flow for shape generation, and introduces a "retargeting" strategy that fine-tunes the model to handle the distribution shift between biased partial inputs and complete shapes. Experiments on Breaking Bad and PartNet show reductions in Chamfer distance and improvements in precision/recall over raw assembly baselines.

## Strengths

- **Novel problem formulation:** Identifying that reassembly methods lack complete-object priors and proposing a generative completion layer orthogonal to existing assembly algorithms is a well-motivated and conceptually valuable contribution. The problem statement in Section 3.1 is clear and articulates a genuine gap.

- **Interesting engineering of the 2D-3D bridge:** Mapping point cloud coordinates to RGB channels and leveraging LEAP's pretrained features is a creative solution to the limited 3D training data and fixed-point-count problems. This is a practical and non-obvious design choice.

- **Substantial empirical improvements on Breaking Bad:** Table 1 shows consistent CD reductions (e.g., 10.5→4.5 for Jigsaw baseline) and precision/recall gains. The missing-pieces experiment (Table 2 left) demonstrates robustness when 20% of fragments are removed.

- **Honest limitations discussion:** Section 6.3 candidly identifies failure modes (size limitations, unseen categories, topology constraints), which is unusual and adds credibility.

- **Ablation studies on key parameters:** Figure 5 provides intuition for the reverse sampling steps (k) and noise injection (α), showing meaningful trade-offs between input fidelity and completion quality.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed "category-agnostic" capability:** The abstract and introduction prominently state Jigsaw++ learns a "category-agnostic shape prior," but on PartNet (Section 6.1), the model is trained independently on three category subsets (chairs, tables, lamps). This is literally per-category training, not category-agnostic. On Breaking Bad, the everyday subset has limited diversity, and no cross-category generalization experiment is conducted. The failure cases in Section 6.3 explicitly acknowledge struggles with unseen object types. The "category-agnostic" claim is not supported by the evidence; what is demonstrated is a per-dataset or per-category generative prior.

- **The central claim—shape priors improve reassembly—is not demonstrated realistically:** The entire motivation (Section 1, 3.1) frames Jigsaw++ as providing shape priors that guide reassembly. Yet the only experiment linking the prior back to assembly (Table 2 right) uses ground-truth closest-point matching between the original object surface and the generated shape—an oracle that would never be available in practice. The paper itself concedes: "we have yet to devise methods to effectively leverage our outputs as guidance for further reconstructions" (Section 7). Without a realistic pipeline that takes partial assembly + generated prior and produces improved poses, the core selling point is unsubstantiated. What is shown is that Jigsaw++ produces shapes closer (in CD) to ground-truth complete objects, which is shape completion quality, not assembly improvement.

- **No quantitative comparison against 3D shape completion baselines:** Table 1 compares "baseline assembly method" versus "baseline + Jigsaw++" but never against any 3D shape completion or generative model adapted to the same task. Figure 2 shows qualitative failures of AdaPointTr and LION+SDEdit, but no quantitative metrics are provided for these alternatives. This makes it impossible to determine whether Jigsaw++'s specific retargeted rectified-flow design is necessary or whether any decent shape prior would yield similar gains. Since the comparison is against assembly algorithms (not completion methods), the improvements may simply reflect the benefit of having any complete shape reference.

### Minor

- **Misleading "Langevin dynamics" terminology:** Equation (5) is described as applying Langevin dynamics, but it is simply linear interpolation with Gaussian noise: x₀ = αx̂₀ + √(1-α²)ξ. Actual Langevin dynamics involves gradients of log-density and step-size schedules. The ablations show the approach works empirically, so this is a presentation issue, but it misrepresents the method theoretically.

- **The bi-directional point-cloud-to-RGB mapping is lossy despite "high fidelity" claims:** The coordinate-to-color mapping f(o_i)=⌊255o_i⌋ introduces 8-bit quantization per axis, and multi-view rendering suffers from self-occlusion. The paper claims this yields "high fidelity" cyclic reconstruction, but fidelity depends heavily on LEAP's learned 3D priors to compensate for information loss. The method works empirically, but the theoretical framing of the mapping as geometry-preserving is overstated. This is partially acknowledged in Section 6.3's limitations.

- **No analysis of multi-modality:** For partially assembled inputs, multiple valid complete shapes may exist, but only a single output is evaluated against a single ground truth. Whether the model collapses to a mean shape or can generate diverse plausible completions is not discussed.

- **Per-category PartNet training contradicts scope claims:** Training independently on chairs, tables, and lamps means three separate generative models rather than one unified model. This limits the scope of conclusions about generalization.

### Trivial
- The threshold η for precision/recall metrics (Section 6.1) is not specified. (Standard in the field but worth noting.)

## Nice-to-Haves

- A simple correspondence module (even ICP-based) to demonstrate realistic assembly improvement from the generated priors—this would dramatically strengthen the paper's core narrative.
- Cross-category generalization experiments (train on chairs+lamps, test on tables) to validate the "category-agnostic" claim.
- Quantitative comparison against shape completion baselines (AdaPointTr, PoinTr, etc.) under the same partial-assembly conditioning.

## Removed Points

- **Reproducibility of hyperparameters and training details:** Demanding complete training schedules, latent dimensionalities, etc. is impractical for this format; the key architectural choices are described.

- **Unfair comparison with baselines (assembly methods):** The harsh critic argued that comparing "baseline vs. baseline+Jigsaw++" is unfair because it doesn't isolate Jigsaw++ against alternative completion methods. This is valid and kept as a major weakness. However, the comparison *does* demonstrate that Jigsaw++ adds value orthogonal to existing assembly methods, which is a legitimate finding—it just doesn't prove Jigsaw++ is better than other completion approaches.

- **Information loss in mapping is "not fixable":** The harsh critic claims this is a structural flaw. However, the method empirically works and the limitations are acknowledged. This is a real limitation but not fatal; it's kept as a minor weakness.

- **Demand for theoretical proofs of rectified flow correctness:** Not standard for an empirical generative methods paper. The ablations demonstrate the approach works; requiring formal proofs is scope creep.

- **Inference time/computational cost:** Not reported, which is a gap, but not standard for the field. Kept as a trivial note only.

- **Missing ablation on joint generation of g and r:** The paper states g is generated but only r is decoded. An ablation would strengthen but not invalidate; kept as implicit in "nice to have" rather than a standalone weakness.

## Novel Insights

The most insightful observation across the reviews is the fundamental disconnect between what Jigsaw++ demonstrates (shape completion quality measured by CD) and what it claims (a shape prior that improves reassembly). This is distinct from merely having weak experiments—the paper has good shape completion results, but the entire framing around "reassembly" is aspirational rather than demonstrated. A shape prior that reduces CD to ground truth is necessary but not sufficient for assembly guidance; the hard problem is establishing correspondences between fragments and the generated prior, which the paper leaves entirely to future work. Additionally, the retargeting strategy is a reasonable adaptation of inversion-based editing to 3D, and the empirical results suggest it works, but without comparison to simpler conditioning alternatives, it remains unclear whether rectified flow retargeting is key or whether any fine-tuned generative model would suffice.

## Suggestions

1. **Either demonstrate realistic assembly improvement or reframe the contribution.** If the paper is fundamentally about shape completion for partial assemblies (not reassembly guidance), frame it accordingly and title it accordingly. The current title ("Object Reassemble") overpromises.
2. **Replace "category-agnostic" with honest language** such as "category-agnostic architecture" (the architecture doesn't require category labels as input) while acknowledging per-category training in experiments, or add a cross-category experiment to substantiate the claim.
3. **Add quantitative comparison with at least one shape completion baseline** (e.g., AdaPointTr, PoinTr) under the same conditioning setup to establish that the retargeted rectified flow approach is necessary.

## Score and Decision

Calibration against similar papers:
- **PuzzleFusion++** (assembly on Breaking Bad, scores 6-8, Accept): Strong end-to-end assembly results on the same dataset, with actual assembly metrics. Jigsaw++ shows good shape completion but doesn't close the loop on assembly.
- **ComPC** (point cloud completion with 2D priors, scores 6-8, Accept): Similar 2D→3D strategy, but with stronger baseline comparisons and validated on unseen categories.
- **BiAssemble** (bimanual assembly, scores 6-8, Reject): Interesting problem but low practical success; reviewers noted the gap between stated claims and demonstrated results—parallel to Jigsaw++.
- **Completion Consistency for PCC** (scores 3-5, Reject): Incremental, limited empirical validation.

Jigsaw++ has a genuinely novel and well-motivated problem formulation with interesting engineering. However, three major weaknesses substantially undermine the paper: (1) the "category-agnostic" claim is contradicted by per-category experiments, (2) the core narrative about improving reassembly is not demonstrated (only shape completion quality is shown), and (3) no shape completion baselines are compared. These are not minor gaps—they concern the paper's central claims. The paper is below the acceptance bar for a venue like ICLR/NeurIPS but has potential if the claims are restructured and proper baselines are added.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>