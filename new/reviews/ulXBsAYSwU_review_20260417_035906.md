# Review of /home/wg25r/review_agent/iclr2026_cspaper/papers/ulXBsAYSwU.txt

Now I have a good picture of the landscape. Let me compile the final review. The key calibration points:

- **MolMiner earlier version (SDjCRmuaDS)**: Scores 3, 6, 3, 5 → rejected. Main issues: no baselines, novelty concerns, 3D contribution unclear, limited evaluation.
- **STGG+ (26kgSlMmhA)**: Multi-property conditional generation, scores 5,5,3,3,5,6,6 → rejected. Overclaiming, limited novelty, weak evaluation.
- **MAGNet (5FXKgOxmb2)**: Fragment-based molecular generation, scores 5,8,8,8 → accepted (Spotlight). Novel factorization, strong empirical results, good baselines.
- **Training-free multi-objective diffusion (X41c4uB4k0)**: Scores 8,6,6,6 → accepted (poster).

This current version of MolMiner has substantially improved from the earlier version (SDjCRmuaDS) - it now has HierVAE baseline, more properties (12 vs 3), more evaluation, and better articulation. However, it still has significant weaknesses: limited baselines especially for conditional generation, qualitative-only conditional evaluation, and claims that exceed the evidence.

Let me now write the final review.</think>Now I have a comprehensive understanding of both the paper and the calibration landscape. Let me compose the final review.

## Summary

MolMiner is a fragment-based, order-agnostic autoregressive model for molecular generation that integrates dynamic 3D geometry (via UFF forcefield relaxation), symmetry-aware fragment attachment, and multi-property conditional generation over twelve physicochemical/structural properties. A GMM-based mechanism allows users to condition on any subset of properties while sampling the rest from the training distribution. The paper also proposes Wasserstein distance and calibration plots as evaluation protocols.

## Strengths

- **Unified framework with multiple well-integrated components**: The combination of order-agnostic rollouts, symmetry-aware fragment attachment, geometry-biased attention, and GMM-based partial conditioning into a single coherent model is technically solid. Each component addresses a real challenge (fragment symmetries, rollout rigidity, spatial awareness, flexible conditioning), and their integration is non-trivial.

- **Symmetry-aware fragment attachment**: The procedure for standardizing attachment configurations across symmetry-equivalent positions using Morgan fingerprints and cyclic permutation matching is a concrete, practical contribution that addresses an under-discussed problem in fragment-based models (Section 3.2). This is more than a minor engineering fix—it affects generation quality directly.

- **Flexible partial conditioning via GMM**: The ability to condition on arbitrary subsets of properties while completing missing values from a learned GMM prior is a practical and user-friendly design. This goes beyond single-target conditioning and enables real-world use cases where only some targets are known.

- **Evaluation methodology advances**: The use of Wasserstein distance for unconditional distributional comparison and calibration plots for conditional generation — rather than single-point metrics — pushes evaluation beyond what is standard in the field and provides richer diagnostic information.

- **Honest limitation discussion**: The paper candidly acknowledges the early-termination bias producing smaller molecules and its impact on molWt, TPSA, and MR. This is a refreshing contrast to the typical tendency to obscure weaknesses.

## Weaknesses

### Major:

- **No baselines for conditional generation — the paper's core contribution**: The central claim is calibrated multi-property conditional generation, yet Section 4.3 provides no comparison to any conditional generation baseline. Not even a simple conditional VAE, classifier-free guided model, or property-conditioned SMILES generator is included. Without context, it is impossible to assess whether MolMiner's conditional performance is competitive, trivially achievable, or subpar. The paper states "to our knowledge, this is the first model to support simultaneous conditioning across as many as twelve molecular properties," but without baselines, being first does not establish being good. This is the most consequential gap because the entire paper is positioned around conditional generation.

- **Conditional generation claims are not quantitatively supported**: Section 4.3 asserts "calibrated conditional generation for most of the twelve properties" based entirely on visual inspection of Figure 2 (mean trend lines with ±1 SD bands and confusion matrices). No numerical metrics — RMSE, R², mean absolute error, expected calibration error, or success rate under tolerance bands — are reported for any of the 12 properties. The calibration failures on QED, molWt, and MR are acknowledged qualitatively but not measured. For a paper whose headline claim is "calibrated conditional generation," this level of evidence is insufficient.

- **Evaluation limited to simple RDKit descriptors on a single dataset with no out-of-distribution testing**: All 12 properties (logP, QED, SAS, FractionCSP3, molWt, TPSA, MR, HBD, HBA, ring count, rotatable bonds, chiral centers) are straightforward 2D descriptors computable by RDKit, many of which correlate strongly with molecular size. The introduction motivates the model for "structure-dependent properties" and "quantum chemical calculations," but no evaluation touches these harder targets. No external dataset, scaffold split, or out-of-distribution test probes generalization. The practical relevance to drug discovery or materials design — explicitly claimed in the conclusion — is therefore unsupported by evidence.

- **Unconditional performance is weaker than HierVAE, the only baseline**: Table 1 shows MolMinerD (sampling conditions from data) and MolMinerS (sampling from GMM) have higher Wasserstein distances than HierVAE across many properties, with notable gaps in molWt (0.31 vs 0.26 for MolMinerD; 0.46 vs 0.26 for MolMinerS), TPSA (0.36 vs 0.15 for MolMinerD), and MR (0.56 vs 0.20 for MolMinerD). Since MolMiner is an inherently conditional model being evaluated unconditionally, some degradation is expected, but HierVAE is the sole unconditional baseline and MolMiner does not clearly match or exceed it.

### Minor:

- **Ablation results are only qualitative**: Section 4.1 states that geometry-aware attention "aids performance when initialized with positive bias" and that rollout resampling "serves as effective regularization," but no quantitative ablation table appears in the main text. For architectural choices as important as geometry integration, readers need effect sizes — not just directional claims. The earlier-review finding that the no-geometry variant "could still perform very well" raises the question of how much 3D information actually helps.

- **GMM prior strongly constrains the conditional evaluation regime**: The GMM fills in unspecified properties from the training distribution's learned manifold. This means the conditional evaluation (sampling one property uniform over μ±2σ, completing the rest from GMM) never tests truly novel or adversarial property combinations. The gap between MolMinerD and MolMinerS already shows the GMM introduces errors; the paper does not characterize which property regions the GMM fails to cover or how often the GMM produces infeasible completions.

- **Jensen's inequality lower bound tightness not analyzed**: Equation (3) trains on a lower bound of the log-likelihood. Whether this bound is tight — which affects the quality of the learned distribution — depends on the variance across rollout orders, which is never analyzed or estimated.

- **Forcefield (UFF) limitations not discussed**: UFF is a classical forcefield with known accuracy limitations. No discussion of FF convergence failures, unrealistic conformers, or computational cost per molecule appears. The claim of "dynamic geometry" depends on UFF's reliability, which is not examined.

### Trivial:

- The "first to unify" phrasing in the abstract and conclusion overstates the contribution relative to the evidence. The claim could be reframed to highlight that this is a first exploration of combining these techniques, rather than asserting a solved integration.

## Nice-to-Haves

- A quantitative conditional evaluation table with RMSE/MAE/R² for each of the 12 properties, and ideally a joint-conditioning experiment fixing 2–4 properties simultaneously to test scalability of multi-property control.
- At least one conditional generation baseline (e.g., conditional VAE, classifier-free guided model) to contextualize conditional performance.
- Evaluation on at least one out-of-distribution split or a dataset with more challenging 3D-dependent properties (e.g., QM9 dipole moments, HOMO-LUMO gaps) to justify the "geometry-aware" and "3D-aware" claims.
- A quantitative ablation table (in the main text) showing Wasserstein distances and conditional accuracy with and without geometry-aware attention, with and without order-agnostic rollouts, and with and without symmetry-aware attachment.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **MoLeR exclusion criticism**: The harsh critic and human finder note the exclusion of MoLeR from comparisons. The paper explains that MoLeR's VAE prior-posterior mismatch caused poor sampling (consistent with documented issues), and results are included in Appendix A.9. Excluding a broken baseline from the main table is reasonable — it would be misleading to compare against a baseline that is known to malfunction. The exclusion is well-justified. *However*, the absence of *any* other conditional baseline remains a valid concern (see Major weaknesses above).

- **Formatting nitpicks**: Minor issues like garbled table rendering (parser artifact), figure caption complaints, and reference formatting are removed per rules.

- **Missing related works**: The human finder suggests missing related works (Podda et al. 2020, Bengio et al. 2021). Per rules, we do not flag missing citations as we cannot confirm their relevance and existence from this context.

- **Demand for synthesizability evaluation**: Generating "synthesizable" molecules or evaluating retrosynthetic accessibility goes beyond the paper's stated scope of property-conditional generation. This is a nice-to-have at best.

- **Concerns about UFF validation with DFT**: Demanding quantum-chemical validation of the forcefield goes beyond the paper's scope. The paper makes a legitimate architectural choice (UFF for intermediate geometry); whether UFF is "accurate enough" for the model's purposes is a valid question but not a requirement for a generative model paper.

## Novel Insights

The GMM-based partial conditioning mechanism creates an implicit constraint that needs more scrutiny: by completing unspecified properties from the training manifold, the conditional evaluation regime favors in-distribution combinations. This means the paper's calibration results — even if numerically quantified — would primarily reflect the model's ability to stay within learned property correlations rather than truly disentangling and independently controlling properties. The "tomographic effect" (conditioning on more properties improves generation quality) that the paper cites in its ablation actually supports this concern: if more conditioning always helps because it constrains the generation to a narrower region of the data manifold, the real question is what happens when the conditioned properties specify an *out-of-manifold* combination, which is never tested.

## Suggestions

1. **Add a quantitative conditional evaluation table** with RMSE or MAE for each property at minimum. This is the single most impactful improvement — it converts the central claim from "visually calibrated" to empirically verifiable.

2. **Include at least one conditional baseline** — even a simple property-conditioned SMILES VAE or a random forest regressor+generator pipeline would provide crucial context for whether MolMiner's conditional performance is non-trivial.

3. **Test joint conditioning on 2–4 properties simultaneously** to demonstrate the key selling point of multi-property control, not just single-property sweeps.

4. **Tone down the "first to unify" and practical relevance claims** to match the evidence. The model's value lies in its architectural innovations and flexible conditioning design; claims about "accelerating discovery" and "next-generation materials" are speculative without evaluation on domain-relevant tasks.

5. **Report the ablation results quantitatively** in the main text, particularly the geometry-aware attention ablation, given that it is a claimed contribution.

## Score and Decision Calibration

- **MolMiner earlier version (SDjCRmuaDS)**: Rejected with scores 3, 6, 3, 5 (avg ~4.25). Key weaknesses: no baselines, unclear 3D contribution, limited novelty, small evaluation.
- **STGG+ (26kgSlMmhA)**: Multi-property conditional generation, rejected with scores 5,5,3,3,5,6,6 (avg ~4.7). Weak baselines, limited novelty.
- **MAGNet (5FXKgOxmb2)**: Fragment-based generation, accepted (Spotlight) with scores 5,8,8,8 (avg ~7.25). Strong novelty, good baselines, solid evaluation.

This version of MolMiner has improved substantially from the earlier version: more properties (12 vs 3), HierVAE baseline, clearer methodology, Wasserstein/calibration evaluation, and a focused conditional generation story. However, the fundamental weaknesses remain similar: insufficient baselines (especially for conditional generation), qualitative-only evidence for the core claim, and evaluation on easy descriptors. The "first to unify" framing overreaches relative to the evidence. Compared to STGG+ (rejected, avg ~4.7), which also claimed multi-property conditioning but lacked baselines, this paper is in a similar position — arguably slightly better due to the architectural contributions and honest limitations, but held back by the same core gap of no conditional baselines and no quantitative conditional metrics.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>

**Predicted score: 4.5**
