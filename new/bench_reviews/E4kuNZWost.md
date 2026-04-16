Now let me read the human review files for calibration:Now I have enough context to produce the final review. Let me synthesize everything carefully.

---

## Summary
TULiP proposes a post-hoc OOD detector grounded in linearized training dynamics (NTK theory). The core idea is to bound training fluctuations from hypothetical pre-convergence perturbations, and translate this bound into a practical test-time uncertainty score via weight perturbation and surrogate ensemble construction. The paper is evaluated on OpenOOD benchmarks, showing strong near-OOD performance.

---

## Strengths

- **Novel theoretical framing**: Theorem 3.1 provides a meaningful, if imperfect, connection between epistemic uncertainty (Eq. 1) and gradient-space distance from the test point to the training set. The NTK-based derivation is a genuine theoretical contribution — most post-hoc methods lack any such connection to the training process.
- **Strong near-OOD performance**: TULiP achieves top-1 or top-2 AUROC on near-OOD across all SS-OOD benchmarks in Table 1 (e.g., CIFAR-10 near-OOD AUROC 89.67 vs. next-best 88.68). This is consistent with the paper's own theoretical prediction from the closeness assumption (Eq. 8), lending modest credibility to the theory.
- **No training data required**: Unlike ViM and MDS (which require training set statistics and significant runtime), TULiP operates purely post-hoc, a practical advantage for deployment scenarios.
- **Transparency about heuristics**: The paper explicitly acknowledges that the layer-wise scaling is "highly heuristic," that E_x[Θ(x,x)] is omitted for tractability, and that λ is a proxy for an unspecified constant — more honest than typical submissions.
- **Plug-and-play compatibility**: TULiP+GEN demonstrates that the framework is modular and composable with existing scoring methods.
- **Architecture generalization**: Fig. 3 shows consistent improvement over MLS and ODIN across MobileNet, VGG, and RegNet.

---

## Weaknesses

### Fatal
*None. The paper has real contributions and transparent limitations. The FUNDAMENTAL ISSUES rule is not triggered.*

### Major

- **Substantial theory-practice gap that weakens the "theoretically-driven" positioning.** The practical algorithm (Algorithm 1) departs from the theoretical bound at nearly every critical step: it sets t_s=0 and θ_{t_s}=0 (replacing the theoretical initialization parameter), uses a heuristic layer-wise empirical Jacobian in place of the NTK (Eq. 11–12), drops E_x[Θ(x,x)] entirely (Sec. 4.3: "intractable and irrelevant to z"), absorbs unspecified constants into a tuned hyperparameter λ, and constructs the final score via a variance-squeezing heuristic (Eq. 15). The paper explicitly labels the scaling as "highly heuristic" (Sec. 4.1). The result is that S in line 13 of Algorithm 1 is not a rigorous estimator of the bound in Proposition 3.3 — it is a heuristic motivated by it. The paper can stand as an empirically-motivated heuristic with theoretical inspiration, but framing it as "theoretically-driven" with "state-of-the-art performance" is an overclaim that will mislead readers about the strength of the theoretical guarantee.

- **Empirical claims are overstated in several key places.** The abstract states TULiP "consistently improves previous state-of-the-art methods across various settings." Table 1 contradicts this: on ImageNet-1K far-OOD, ASH achieves AUROC 95.74 vs. TULiP's 88.03; ViM achieves FPR95 24.67 vs. TULiP's 48.01. Table 2 has a more serious problem: TULiP's row has all four values bolded, yet ViM scores 83.93 on ImageNet-C (vs. TULiP's 82.91) and 87.92 on ImageNet-R (vs. TULiP's 82.07). The paper does acknowledge ViM uses training data (†), but bolds TULiP as best regardless — this presentation is misleading. Figure 3 compares TULiP only against MLS and ODIN, excluding stronger baselines (ASH, GEN, ViM, EBO), yet the accompanying text claims TULiP "outperforms baseline methods consistently across the board" for architecture generalization. This is unsupported given the restricted comparison.

- **The closeness assumption (Eq. 8) is the single most critical yet least-validated step.** Moving from the inf over training data (Eq. 6) to the expectation over training data (Eq. 7–8) is what makes the bound tractable and train-data-free. This is also precisely where the OOD problem lives: for far-OOD points, there is no a priori reason the expectation upper-bounds the infimum in a way useful for OOD ranking. The empirical verification in Fig. 1d is limited to one model (ResNet18) on one ID dataset (ImageNet-1K) using 256 samples. This insufficient validation of a load-bearing assumption is consistent with TULiP's well-documented failure on far-OOD scenarios.

### Minor

- **Hyperparameter sensitivity and tuning methodology**: The method requires tuning ε, δ, λ, and M. Figure 4 shows a clear near/far tradeoff with hyperparameter choices. The paper acknowledges tuning on a Blur-ImageNet validation set for CS-OOD, then explains poor ImageNet-R performance as a consequence of that tuning strategy — making part of the reported CS-OOD pattern a selection artifact rather than an intrinsic method property.

- **Computational cost understated**: TULiP uses M=10 forward passes plus a finite-difference estimate (line 12), making it ~10× slower per input than single-pass methods like EBO, MLS, and ASH. The paper only reports a comparison to ViM's training-data extraction time (3× faster) without providing wall-clock times across all baselines.

- **Abstract claim about non-classification problems is unsubstantiated empirically**: The abstract explicitly says TULiP "is not limited to classification problems," and Algorithm 1 line 18 uses softmax for classification. The only non-classification evidence is Fig. 2 (synthetic Splines regression using an infinite-width NTK network, not finite CNN). All real-world experiments are classification. This claim needs either real regression/detection experiments or should be qualified.

### Trivial

- There is a minor notation inconsistency between Eq. 13 (which uses δ→0 for the finite-difference limit) and Algorithm 1 line 12 (which uses ε). These likely refer to different quantities — ε as the practical finite step size, δ as the theoretical infinitesimal — but the conflation is unexplained and potentially confusing.

---

## Nice-to-Haves

- **Deep ensemble comparison**: Since Eq. 1 defines epistemic uncertainty as ensemble variance, a comparison to deep ensembles (even on small-scale experiments) would directly test whether TULiP's surrogate samples approximate true ensemble behavior. This is the most natural validation of the mechanistic claim and is currently absent.

- **Transformer/ViT experiments**: The conclusion acknowledges ViTs as a limitation. Even a brief appendix experiment would strengthen the versatility claim, especially given ViT's dominance in modern practice.

- **Ablation of theoretical vs. heuristic components in isolation**: A comparison of TULiP without layer-wise scaling and without the λ-correction (using raw perturbed logits directly) would clarify how much of the performance comes from the theoretical insight versus engineering choices.

- **Rigorous analysis of when Eq. 8 holds**: Connecting the closeness condition to NTK spectral properties or data geometry, rather than a single empirical check, would significantly strengthen the theoretical narrative.

- **M sensitivity analysis**: How does performance scale with M=1,5,10,20? This would help practitioners understand the compute-accuracy tradeoff.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Assumes epistemic uncertainty is conflated with OOD"** (Human Finder reviewer): The paper explicitly engages with the relationship between epistemic uncertainty and OOD distance (Sec. 3.2, Eq. 1 discussion), and cites relevant literature. The framework is internally consistent on this point even if the broader philosophical question remains open. This is too generic to be actionable as a paper-specific weakness.

- **Missing RankFeat, DICE, ReAct as primary comparisons** (Neutral reviewer): The OpenOOD benchmark used by the paper defines a standard set of baselines, and the paper uses this standard set faithfully. Criticizing the absence of specific baselines when the benchmark choice already standardizes the comparison set is scope creep.

- **Hyperparameter sensitivity as undermining the "post-hoc" claim** (Neutral reviewer): All post-hoc methods require some tuning. The use of a validation set is explicitly acknowledged and consistent with standard OpenOOD evaluation protocol.

- **Demanding theoretical proofs for finite-network validity** (multiple reviewers): This is an empirical systems paper grounded in theoretical motivation. Demanding formal proofs for the heuristic steps exceeds community standards for this type of work.

---

## Novel Insights

The most genuinely novel contribution is the observation that a hypothetical pre-convergence perturbation, propagated through linearized training dynamics, generates a bound on epistemic uncertainty that is dominated by gradient-space distance to the training set — and that this distance can be estimated at test time without the training set by leveraging the weight trajectory (θ_T − θ_{t_s}). The surrogate ensemble construction via variance-squeezing is a practically clever way to translate a scalar bound into ensemble-like predictions usable with entropy-based scoring. These insights, even if implemented heuristically, open a new direction for post-hoc uncertainty estimation that is more principled than score-engineering approaches.

---

## Suggestions

1. **Reframe the abstract and introduction**: Replace "theoretically-driven" with "theoretically-motivated" and qualify SOTA claims to be specific about near-OOD and the no-training-data setting. The current framing invites criticism that the paper cannot fully defend.

2. **Fix Table 2 formatting**: Do not bold TULiP where ViM outperforms it in absolute terms, or add a separate bolding scheme for "best without training data" and "best overall."

3. **Expand Figure 3 to include at least ASH and GEN** for architecture generalization comparisons. These are included in Table 1 and compatible with the architecture-agnostic requirement.

4. **Multi-dataset validation of Eq. 8**: Test the closeness assumption on CIFAR-10 and ImageNet-200 using the practical NTK estimate, not only ImageNet-1K with ResNet18.

5. **Add a wall-clock time table** comparing all methods per test image on the same GPU.

---

## Score and Decision

**Calibration papers reviewed:**

| Paper | Scores | Decision | Comparison |
|---|---|---|---|
| VBLL (Sx7BIiPzys) | 8, 10, 6, 8 | Spotlight | Clean theory + strong empirics; TULiP is substantially weaker on both axes |
| CDR Score (fsEzHMqbkf) | 3, 6, 8, 6 | Reject | Post-hoc OOD on non-standard benchmarks; TULiP uses standard OpenOOD, stronger evaluation |
| Iterative Linearization (lIYxAcxY1B) | 3, 5, 5 | Reject | NTK-based analysis with poor empirical contribution; TULiP has stronger empirics |
| Lazy Regime (XgAKt7rbXk) | 3, 5, 3, 3 | Reject | Unsupported claims, weak experiments; TULiP is stronger on evidence quality |
| Pathologies of OOD (hlijRgXTDK) | 5, 6, 5, 3 | Reject | OOD detection critique with limited experiments; comparable positioning but TULiP is more constructive |

**Assessment**: TULiP is meaningfully above the 3–4 range papers (Iterative Linearization, Lazy Regime) that had weak empirics. It is clearly below VBLL (8–10 range), which had tight theory and comprehensive empirics. The CDR Score rejection (3–8 split) is the closest analog — a post-hoc OOD paper with real empirical contribution but overclaiming and benchmark concerns. TULiP is somewhat stronger (standard OpenOOD benchmark, genuine near-OOD advantage, more transparent about limitations) but shares the overclaiming and theory-practice gap issues.

**Axes evaluation:**
- *Originality*: Moderate-high. The NTK-to-post-hoc-OOD link is novel.
- *Importance of research question*: High. Post-hoc OOD without training data is practically important.
- *Claims well-supported*: Moderate. Near-OOD claims are supported; SOTA and theoretical claims are overstated.
- *Soundness of experiments*: Moderate. OpenOOD is a good benchmark, but Figure 3 and Table 2 comparisons are selectively presented.
- *Clarity of writing*: Good. The paper is generally well-organized and honest about heuristics.
- *Value to community*: Moderate. A working heuristic with theoretical motivation and strong near-OOD results has clear value, but the overclaiming reduces credibility.

**Final score: 5.0** — marginally below acceptance. The paper presents a genuinely interesting idea, uses an appropriate benchmark, and shows real near-OOD gains. However, the gap between the theoretical narrative and the implemented algorithm, the overstated empirical claims (particularly in Table 2 and Figure 3), and the limited validation of the critical closeness assumption keep it below acceptance threshold in its current form. Revision to reframe the theoretical contribution honestly, fix the empirical overclaiming, and expand the closeness assumption validation would bring this to an acceptable level.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>