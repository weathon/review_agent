## Summary

This paper presents a large-scale empirical ablation of normalization layers in Mamba architectures, evaluating five normalization types (BN, LN, GN, IN, RMSN) across different positions (before/after the selective SSM module) and in 25 pairwise combinations. The study spans video sequence modeling (Breakfast) and image classification (ImageNet-100), with additional validation on LRA ListOps and ImageNet-1k. The authors find that post-SSM normalization generally outperforms pre-SSM placement, that heterogeneous combinations can exceed homogeneous defaults, and provide an L2-norm-based intuition for why certain combinations stabilize training.

## Strengths

- **Exhaustive controlled ablation.** The paper delivers a systematic grid search (Tables 1–4) over five normalization types, two positions, and all pairwise combinations that the Mamba literature has lacked. This produces a concrete reference table showing, for example, that IN→SSM→LN reaches 72.5% on Breakfast while RMSN→SSM→BN reaches 87.3% on ImageNet-100, both well above their homogeneous defaults (58.9% and 84.1% respectively).
- **Cross-dataset validation.** The top-performing configurations are tested on LRA ListOps and ImageNet-1k (Table 5), and the ListOps result shows a large 15.6 percentage-point gain over the original baseline.
- **Mechanistic diagnostics.** Figure 4 demonstrates that post-SSM normalization prevents layer-wise explosion of weight L2 norms across depth, grounding the empirical recommendations in observable training dynamics rather than accuracy alone.
- **Taxonomy of prior practices.** Figure 1 and Section 2 systematically organize over 20 existing Mamba models by normalization strategy, rigorously documenting the lack of consensus and strengthening the motivation for the study.

## Weaknesses

### Fatal
None.

### Major

- **Weak vision validation undermines generalization claims.** The ImageNet-1k validation in Table 5 shows only a 0.3 percentage-point improvement (70.8% → 71.1%) over the VMamba baseline. The paper reports single-run estimates with no variance, and a margin this small is well within typical run-to-run variation for ImageNet training. Because the abstract and introduction claim findings are “validated on other datasets” and yield “practical recommendations,” this unreliable vision validation significantly weakens the central practical contribution for the image-classification community. (The ListOps sequence validation is much stronger, but the vision side cannot carry comparable weight.)
- **Task-specific lookup tables vs. promised general principles.** The paper’s abstract and contribution bullets promise “practical recommendations” and “general guidelines” for selecting normalization in Mamba architectures. Yet the “best” combination changes completely across tasks (IN→LN for sequence, RMSN→BN for vision), and the final recommendation that “LN emerges as a versatile … performer” (Section 4.4) is difficult to reconcile with the tables: LN alone is 58.9% on Breakfast (beaten by GN at 68.8%) and 86.6% on ImageNet-100 (matched or beaten by several mixed combinations). The paper delivers an empirical grid whose winner is task-dependent, but stops short of a principled, task-agnostic rule for predicting which combination will work for a new problem.

### Minor

- **No variance estimates across random seeds.** Every table reports single-point accuracies without standard deviations. This is especially problematic for the near-zero ImageNet-1k margin, but it also means the reader cannot assess whether the 7.0% “no normalization” Breakfast baseline—a 10-class accuracy below random chance—reflects a single failed run, a best-of-N selection, or a consistent outcome.
- **Sloppy contradictory text in Section 4.5.** The paragraph describing Table 5 contains an internal contradiction. The first sentence incorrectly states: “For vision tasks, RMSN→SSM→RMSN represents the original Mamba's normalization configuration, while IN→SSM→IN represents our proposed normalization configuration.” The very next sentence corrects this (“LN→SSM→LN represents the original VMamba's normalization configuration … while RMSN→SSM→BN represents our proposed normalization configuration”), and the table itself is internally consistent. Nevertheless, the copy-paste error makes the validation section confusing and erodes confidence in the presentation.

### Trivial

- The justification for removing the FFN module from VMamba “for fair comparison” (Section 4.5) is mentioned but not explained in enough detail to reproduce.
- The L2-norm “harmonic structure” analysis (Figure 5) is limited to a single combination (BN→IN); the paper does not test whether the observed norm-balancing predicts performance across the remaining 24 combinations in Table 4.

## Nice-to-Have

- Learning-curve plots (loss, gradient norm, or update norm over steps) for the best, worst, and original normalization schemes would substantiate the repeated claim that post-SSM normalization improves “training stability.”
- A predictive test of the L2-norm intuition—using the “harmonic principle” to rank a held-out subset of combinations before running the full grid—would strengthen the explanatory contribution.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **“L2-norm harmonic structure is an unfalsifiable after-the-fact description.”** The paper explicitly frames this as a tentative intuition: “We made an intuitive inference … but this is not intended as an essential explanation” (Section 4.6). The critic misread this framing as a claim of predictive theory.
- **“Breakfast conflates spatial and temporal modeling / missing LRA benchmarks.”** The paper uses Breakfast for sequence modeling and LRA ListOps for validation. While additional long-range benchmarks would strengthen the work, demanding them is scope creep; the paper does not claim to isolate pure long-range sequence ability.
- **“Section 3 is essentially a notation guide.”** For an empirical study, clearly defining the normalization types, positions, and notation is standard and necessary.
- **Any formatting, grammar, or typo criticisms.** These are parser artifacts, not author errors.

## Novel Insights

None beyond the paper's own contributions. The exhaustive grid and the observation that heterogeneous combinations consistently outperform homogeneous defaults are the paper's primary novel deliverables.

## Suggestions

1. Report mean and standard deviation across multiple seeds for all main results, especially ImageNet-1k.
2. Either collect stronger vision validation (e.g., larger margin or multiple seeds showing consistency) or honestly frame the ImageNet-1k result as inconclusive rather than validation.
3. Fix the contradictory text in Section 4.5 so the table caption accurately describes the vision baseline and proposed configuration.
4. Derive and state a more precise selection heuristic—e.g., pairing a statistics-dependent norm before SSM with a rescaling-only norm after SSM—rather than the vague “balance” observation.

## Score and Decision

**Score: 4.5**

**Calibration papers compared:**
- **High anchor:** `d8w0pmvXbZ.md` (avg 8.0, Accept Oral) — small-scale transformer instability proxies. Far more careful methodology, variance reporting, and predictive extrapolation. The paper under review is well below this.
- **High anchor:** `BChpQU64RG.md` (avg 6.2, Accept Poster) — Mix-LN. Proposes a novel method with solid experiments and clear, consistent results. The paper under review lacks a novel algorithm and has weaker validation.
- **Medium anchor:** `yqAToOgxgf.md` (avg 5.0, Reject) — Pi-Sigma rejuvenation. Similar systematic ablation structure, but cleaner presentation and clearer baselines, though on a less timely topic. The paper under review is slightly below this due to its validation and presentation issues.
- **Medium anchor:** `j5EbZEyK9I.md` (avg 4.5, Reject) — data composition observational study. Similar profile: extensive empirical effort, limited novelty, and weak generalization claims. Comparable in quality.
- **Low anchor:** `g4VGwNqzpB.md` (avg 3.0, Reject) — HENP pruning. Limited novelty, weak evaluation, unclear writing. The paper under review is above this because its topic is timely, its experiments are extensive, and its writing is generally clear.

The paper under review sits between the 3.0 low anchors and the 5.0 medium anchor. Its extensive empirical sweep and timely focus on Mamba training stability are real assets, but the weak ImageNet-1k validation, lack of variance estimates, internal text contradiction, and gap between promised principles and task-specific results place it below the acceptance threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>