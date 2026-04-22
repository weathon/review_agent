Now I have enough information to write the final review. Let me compile everything.

## Summary

SIGMADOCK introduces a fragment-based SE(3) Riemannian diffusion model for molecular docking that decomposes ligands into rigid-body fragments and learns to reassemble them within a binding pocket, bypassing the ill-conditioned dynamics of torsional diffusion models. The paper provides a principled theoretical critique of torsional parameterizations (Theorem 1), a fragmentation reduction scheme (FR3D) with soft triangulation constraints (Lemma 1), and an SO(3)-equivariant architecture with proven invariance to local coordinate choice (Theorem 2), achieving 79.9% Top-1 PB-valid on PoseBusters—dramatically exceeding prior DL methods (12.7–32.8%) and classical docking (Vina ~57%).

## Strengths

- **Theoretical critique of torsional models is novel and substantive.** Theorem 1 proves that torsional models produce highly entangled, non-product induced measures in Cartesian space, while disjoint rigid fragments yield a factorised product of Haar measures on SE(3)^m. The gauge ambiguity and "lever effect" arguments (Section 2.2.2) provide a principled geometric explanation for why torsional diffusion underperforms—this goes beyond empirical observation and is a genuine conceptual contribution to the field.

- **PB-validity results represent a real advance over prior DL docking.** Achieving ~80% PB-valid Top-1 on PoseBusters—where Buttenschoen et al. (2024) demonstrated DL methods consistently fail to generate chemically plausible poses—directly addresses the most damaging critique of DL docking. Even with N_seeds=10 and reduced ranking (72.2% PB-valid, Table 1 row H), SIGMADOCK substantially exceeds the best prior DL method (32.8%).

- **Proven invariance to local coordinate orientation (Theorem 2).** The paper rigorously proves that the training objective and sampling procedure are invariant to the arbitrary choice of local coordinate axes for fragments, and that the score model is SO(3)-equivariant. This resolves a fundamental ambiguity in the fragment parameterization that could otherwise undermine the approach.

- **Co-factor stratification analysis provides evidence against memorization.** Table 2 shows failure rates are highest for natural ligands (41.2%) and lowest for no co-factors (16.2%), consistent with the model learning physical interactions rather than memorizing, since SIGMADOCK deliberately excludes co-factors from its input.

- **Strong generalization to unseen proteins.** Table 4 shows 72% PB-validity for proteins with <30% sequence similarity to training data, and the comparison with Vina (~57%) is under controlled conditions and genuinely impressive.

- **Data efficiency vs. co-folding models.** SIGMADOCK achieves competitive performance with AlphaFold3 (~80% vs. ~84% PB-valid) using only 19k training complexes and 50× faster sampling (Section 3.2, Table 4), with lower train-test leakage (Appendix J).

## Weaknesses

### Fatal
None.

### Major

- **Headline comparisons with DL baselines lack fully controlled evaluation conditions.** The paper's most prominent claim—79.9% vs. 12.7–32.8% for "recent deep learning approaches"—compares SIGMADOCK (trained on PB temporal split, 40 samples, energy+PB ranking) against published numbers from models evaluated under potentially different conditions (different training splits, sample counts, ranking schemes). The paper notes "Under fair comparison with models trained on the PoseBusters train-test split" (footnote 1), but does not clarify whether DiffDock and other baselines were actually retrained on the PB temporal split. The 6.3× improvement over DiffDock is the paper's most quoted figure but could partially reflect evaluation-protocol differences rather than purely methodological superiority. That said, the gap is large enough (even N_seeds=10 at 72.2% exceeds the best DL baseline at 32.8%) that the core finding would almost certainly persist under controlled conditions. The controlled comparison with Vina (~80% vs. ~57%) is strong but tells a less dramatic story than the abstract's framing.

### Minor

- **The ranking/scoring heuristic contributes substantially to headline performance without transparent disentanglement from generative quality.** Table 1 shows removing energy scoring drops PB-valid from 79.9% to 66.1% (−13.8pp) and removing PB scoring drops it to 70.8% (−9.1pp). While ranking is standard practice in docking (Vina uses its own scoring; DiffDock uses a confidence model), the paper frames SIGMADOCK as not requiring "a separately trained confidence model" (Section 2.5), which understates the role of the pseudo-binding-energy scorer. The N_seeds=10 result (72.2% PB-valid) partially addresses this by showing the generative model itself produces high-quality samples, but a direct "first-sample without ranking" condition would more cleanly isolate generative quality.

- **Quantitative evidence for the conformer alignment assumption is deferred to the appendix.** The entire method rests on the claim (Section 2.2.1) that bound poses lie approximately on the SE(3)/torsional orbit of RDKit-generated conformers (RMSD ≪ 2Å after alignment). While Figure 2b provides a qualitative example, the paper states the result qualitatively ("substantially below...2Å") without providing median, percentile, or worst-case RMSD numbers in the main text. For the theoretical load-bearing assumption of the approach, a compact quantitative summary (e.g., a 1-line table of alignment RMSD statistics) in Section 2.2.1 would strengthen confidence—particularly for ligands with significant induced strain or macrocyclic ring puckering where the approximation may break down.

- **The AF3 comparison, while acknowledged as indirect, could be more carefully framed.** The paper states "we cannot directly compare SIGMADOCK to co-folding methods" but then claims "AF3-level performance." SIGMADOCK operates in the simpler re-docking setting (fixed holo receptor, known pocket) while AF3 solves the harder co-folding problem (joint protein-ligand structure prediction). Calling this "AF3-level" is like comparing a specialized tool against a general-purpose one on the specialized task—it is informative but the framing should be more precise.

### Trivial
None.

## Nice-to-Haves

- Evaluate on cross-docking (apo-to-holo), which is the practically relevant setting for drug discovery. The paper explicitly restricts to re-docking, which is the standard benchmarking setting, but cross-docking would substantially increase practical impact.
- Report single-sample (no re-ranking, N_seeds=1) performance to cleanly isolate the generative model's intrinsic quality from the scoring heuristic.
- Analyze failure modes by ligand flexibility/size to reveal whether the fragment approach scales or merely shifts the curse of dimensionality as m increases.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The 'first deep learning approach to surpass classical physics-based docking' is trivially true / vacuous."** — This misreads the paper. The PoseBusters paper (Buttenschoen et al., 2024) specifically evaluated DL methods under the PB split and showed they fail (12.7–32.8% PB-valid). SIGMADOCK surpassing classical methods under the same evaluation is a substantive, non-vacuous finding.

- **"FR3D algorithm provides insufficient detail about merging criterion."** — The paper references Algorithm 1 in Appendix D.4 for the full algorithm description. Appendix content is stripped by the parser; this is not a missing description.

- **"The AF3 comparison is misleading because AF3 solves the harder co-folding problem."** — The paper explicitly acknowledges this limitation ("Although we cannot directly compare SIGMADOCK to co-folding methods"). The comparison is provided as context for performance level, not as a claim of superiority. Downgraded to Minor (framing concern) rather than removed entirely.

- **"Missing retrained DL baselines undermines the entire contribution."** — While the comparison is not fully controlled, the gap is so large (72.2% with N_seeds=10 vs. 32.8% best DL baseline) that the core finding is robust. Additionally, DiffDock was likely trained with MORE test-set overlap (non-temporal split), making the comparison potentially biased against SIGMADOCK. The concern warrants Major (not Fatal) status.

- **Strength dropped: "Open-source code availability"** — Generic strength without specific section/table citation; code availability is expected, not a distinguishing contribution.

- **Strength dropped: "No separate confidence model or energy minimization required"** — This conflicts with the verified weakness that the pseudo-binding-energy scorer contributes 13+ pp. While technically not a "separately trained" model, it functionally serves a similar role and should not be listed as an unqualified strength.

- **"Request for confidence intervals / multi-seed runs."** — Single-run evaluation is the standard in large-scale docking benchmarks; demanding confidence intervals is a nice-to-have, not a core flaw.

- **"Missing related works"** — Cannot verify existence of specific missing references without external sources.

## Novel Insights

The fragment-based SE(3) parameterization reveals a fundamental tension in molecular docking: torsional models reduce dimensionality but introduce ill-conditioned dynamics through geometric entanglement (Theorem 1), while fragment models maintain well-conditioned product measures but increase dimensionality. SIGMADOCK's approach—reducing DoFs through FR3D merging and soft triangulation constraints rather than explicit torsional parameterization—suggests that the path forward for generative molecular modeling may lie in structured dimensionality reduction within well-conditioned spaces, rather than operating directly on the low-dimensional but poorly-conditioned torsional manifold. The co-factor failure analysis (Table 2) also provides a template for honest evaluation of DL docking methods that future work should adopt.

## Suggestions

- Add a 1–2 line quantitative summary of conformer alignment RMSD statistics (median, 90th percentile) directly in Section 2.2.1 to substantiate the foundational assumption without requiring the reader to consult the appendix.
- Report a "first-sample" (N_seeds=1, no ranking) condition in the ablation table to cleanly disentangle generative model quality from scoring heuristic contributions.
- Tone down the "6.3× improvement" framing in the abstract and Section 3.2 to acknowledge that this compares under potentially different evaluation conditions; the controlled comparison with Vina (1.4×) is more defensible and still impressive.
- Clarify in Section 3.2 whether the DL baseline numbers come from retrained models on the PB temporal split or from the original PoseBusters paper evaluations.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Quotient-Space Diffusion | /home/wg25r/review_agent/human_reviews_2026/3JPAkwSVc4.md | 7.50 | Similar theoretical contribution (SE(3) diffusion, principled framework), SIGMADOCK has stronger domain-specific empirical results but less general theoretical scope |
| La-Proteina | /home/wg25r/review_agent/human_reviews_2026/RDerF20JYT.md | 8.00 | Similar domain (molecular structure generation), SIGMADOCK has comparable empirical strength but slightly more comparison fairness concerns |
| Bures-Wasserstein Flow Matching | /home/wg25r/review_agent/human_reviews_2026/5Bl5qf3fON.md | 7.00 | Similar (theory-grounded generative model for drug discovery), SIGMADOCK is comparable in quality |
| SynCoGen | /home/wg25r/review_agent/human_reviews_2026/24QKU4iqft.md | 5.00 | Fragment-based 3D generation, SIGMADOCK is clearly stronger with more convincing results and better theoretical grounding |
| PSDNorm | /home/wg25r/review_agent/human_reviews_2026/BZMQotjBwW.md | 5.20 | Overclaimed SOTA, SIGMADOCK's improvements are far more substantial and real |
| MotifScreen | /home/wg25r/review_agent/human_reviews_2026/1EGfgGkHFY.md | 2.50 | Unfair comparison in drug discovery, SIGMADOCK's gap is enormous and more defensible |
| Perturbed Flow Matching | /home/wg25r/review_agent/human_reviews_2026/CEuzrRs613.md | 2.67 | Unfair comparison in structure-based drug design, SIGMADOCK is far stronger with genuine theoretical contributions |
| EdGr | /home/wg25r/review_agent/human_reviews_2026/m9RFvTGh2t.md | 2.00 | Fragment-based graph diffusion for drug design, SIGMADOCK has far more practical utility and stronger results |

SIGMADOCK sits firmly in the high-scoring range. Its theoretical contributions (Theorems 1–2, Lemma 1) and empirical results (79.9% PB-valid) are comparable to papers scoring 7–8. The comparison fairness concern is real but the gap is large enough that the core finding is robust. The paper is slightly below the Quotient-Space Diffusion (7.50) and La-Proteina (8.00) anchors due to the less controlled DL baseline comparison and the ranking heuristic's substantial contribution being underemphasized, but clearly above the medium-scoring anchors.

**Evaluation across axes:**
- **Originality:** High — the theoretical critique of torsional models and the fragment-based SE(3) alternative are genuinely novel
- **Importance of research question:** Very high — PB-validity is the central challenge for DL docking identified by the community
- **Claims well supported:** Mostly — the core PB-validity result is strong, but the 6.3× framing overclaims relative to the controlled evidence
- **Soundness of experiments:** Good — transparent ablations, generalization analysis, and co-factor stratification, but DL baseline comparison lacks full control
- **Clarity:** Good — the paper is well-structured with clear theoretical development, though some key evidence is appendix-deferred
- **Value to community:** High — this is likely to influence how the field thinks about parameterizing diffusion models for molecular tasks

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>