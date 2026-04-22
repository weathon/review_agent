Now I have all the information I need. Let me write the final consolidated review.

## Summary

SIGMADOCK introduces a fragment-based SE(3) Riemannian diffusion model for molecular docking that decomposes ligands into rigid-body fragments and learns to reassemble them within a binding pocket, bypassing the ill-conditioned torsional-to-Cartesian mappings of prior approaches. The paper contributes FR3D (a fragmentation reduction scheme), soft triangulation constraints, and an SO(3)-equivariant architecture, with formal theoretical justification (Theorem 1 on product vs. entangled measures, Lemma 1 on triangulation, Theorem 2 on equivariance). Empirically, SIGMADOCK achieves 79.9% Top-1 success (RMSD < 2 & PB-valid) on the PoseBusters benchmark, substantially outperforming prior deep learning methods.

## Strengths

- **Principled theoretical motivation**: Theorem 1 formally proves that torsional models induce entangled, non-product measures in Cartesian space, while fragment-based SE(3)^m diffusion yields a factorised product of Haar measures. This provides a clean, rigorous foundation for the fragment-based approach and is one of the paper's distinguishing contributions (Section 2.2.2).

- **Strong empirical performance even without PB-based selection**: After removing PB-scoring from the ranking heuristic, SIGMADOCK still achieves 70.8% PB-valid Top-1 (Table 1, Config E), which is ~2.2× higher than the best prior DL method (32.8%). This confirms the fragment-based inductive biases genuinely enforce chemical validity by construction, not just via the selection heuristic.

- **FR3D fragmentation scheme is a practical, novel contribution**: Reducing fragments from m̂ = k+1 to m ≈ (2/3)m̂ while removing over-constrained dummy atoms addresses a real DoF inflation problem. The ablation (Table 1, Config C) shows ~6–7% absolute improvement from fragment merging.

- **Triangulation constraints provide effective geometric priors**: Lemma 1 proves cross-fragment triangulation distances uniquely determine bond angles without restricting dihedrals. The ablation (Table 1, Config A) shows removing triangulation drops PB-validity from 79.9% to 67.1% — a ~13pp reduction — confirming these constraints are critical for managing the higher DoFs of fragment space.

- **Comprehensive ablation study**: Table 1 systematically isolates contributions of each component (triangulation, PL interactions, fragment merging, energy scoring, PB scoring, Nseeds), each showing meaningful 4–13% relative improvements.

- **Co-factor failure analysis provides mechanistic insight**: Table 2 shows failures correlate with excluded co-factors (e.g., 41.2% failure rate with natural ligands vs. 16.2% without), supporting the claim that the model learns genuine physical interactions rather than memorizing.

- **SO(3)-equivariance guarantees**: Theorem 2 proves invariance to the choice of local coordinate orientations, resolving a genuine ambiguity in the fragment parameterization (Section 2.4).

## Weaknesses

### Fatal
None.

### Major

- **PB-validity headline inflated by evaluation-overlapping sample selection**: The paper uses physicochemical checks (bond angles, bond lengths, internal energy — Section 2.5) as part of the sample ranking heuristic, then reports PB-validity as the headline metric. Since PoseBusters validates substantially overlapping properties, this creates circularity between selection and evaluation. Table 1 (Config E) makes this explicit: removing PB-scoring drops PB-validity from 79.9% to 70.8% — a 9.1pp inflation directly attributable to using the evaluation criterion for selection. While using scoring heuristics for ranking is standard in docking, the overlap with the specific evaluation metric is problematic for the headline claim. The paper does transparently report this ablation, which is commendable, but the abstract's primary framing ("79.9% Top-1") buries this qualification. This matters because the 79.9% number does not purely measure the generative model's quality; it measures model + selection-heuristic where the heuristic has privileged access to the evaluation criterion.

- **AF3 comparison framing is misleading**: The paper claims "AF3-level performance" (abstract, Section 3.2) based on comparing SIGMADOCK's 79.9% to AF3's reported 84%. However, SIGMADOCK performs rigid-receptor re-docking (known pocket, holo protein), while AF3 is a co-folding model predicting both protein and ligand structure de novo — a strictly harder problem with different inputs and evaluation protocols. The paper partially acknowledges this ("Although we cannot directly compare SIGMADOCK to co-folding methods," Section 3.2), but the abstract and results section prominently feature the "AF3-level" and "50× faster" claims without this qualification. The speed comparison is similarly asymmetric: AF3 predicts full complex structure while SIGMADOCK places a ligand in a known pocket. This matters because claiming parity with AF3 while solving a different (easier) problem inflates the perceived significance of the contribution.

### Minor

- **No variance or confidence intervals reported**: All results appear to be from a single training run, despite the model depending on stochastic fragmentation (FR3D) and random conformer sampling. This leaves the reproducibility and robustness of the exact numbers uncertain (Table 1, Section 3.2).

- **Conformer quality impact on docking accuracy is not analyzed**: Section 2.2.1 shows that conformers from πMc can be aligned to bound poses with negligible RMSD, but this alignment uses the ground-truth bound pose. During inference, conformers are sampled without such access. The paper does not analyze how conformer quality or diversity affects downstream docking accuracy — a gap since the entire forward process initializes from a random conformer.

- **Product-structure benefit vs. dimensionality cost is not fully disentangled**: The high PB-validity could stem primarily from fragment rigidity (trivially enforcing intra-fragment validity) rather than from the SE(3) product measure. An ablation using torsional parameterization with the same fragment-based chemical constraints would isolate the diffusion-space contribution from the chemical-prior contribution.

### Trivial
None.

## Nice-to-Haves

- Evaluate on cross-docking (apo protein structures) to test generalization beyond the holo-receptor setting, which is the more practically relevant scenario for drug discovery.

- Report results without any sample selection heuristic (random sample from Nseeds) to establish the model's standalone generation quality, cleanly separating it from the selection procedure's contribution.

- Include at least one retrained baseline (e.g., DiffDock) on the identical PDBBind-only training split to strengthen the controlled comparison, even though the current asymmetry actually favors the baselines.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Uncontrolled baseline comparison" as a Major weakness**: The harsh critic argues that the 79.9% vs. 12.7–32.8% comparison is uncontrolled because baselines were not retrained on the same split. However, the paper deliberately restricts training to PDBBind(v2020) only, while DiffDock-L was trained on the significantly larger PDBBind(v2020) ∪ BindingMOAD corpus. The asymmetry in training data actually *favors the baselines*, not SIGMADOCK. Per the rule that weaknesses about unfair comparison where asymmetry favors the baseline should be removed, this does not qualify as a Major weakness. The paper's footnote also qualifies the comparison. It remains a nice-to-have to retrain baselines for a fully controlled comparison, but it is not a flaw in the current work.

- **"Conformational manifold circularity" claim**: The harsh critic suggests Section 2.2.1's alignment is circular because it uses the ground-truth bound pose. However, this alignment is used for *justification* (showing the conformational manifold is close to the bound manifold), not for inference. This is standard practice in the field and is not circular in the problematic sense. During inference, conformers are sampled from πMc without access to ground truth.

- **"Underspecified FR3D algorithm" claim**: The harsh critic says the stochastic search procedure is underspecified in the main text. The algorithm is described in Appendix D.4 (Algorithm 1), and the empirical finding (m ≈ (2/3)m̂) is clearly stated. This is an appendix-deferred detail, which is standard for space-limited submissions. Per the rules about missing appendix details, this is removed.

- **"Missing variance across training runs" as a Major weakness**: This is a standard concern in ML papers but not a Major issue for a paper with strong ablations and a 9pp performance gap over prior work. Downgraded to Minor.

- **Generic formatting/presentation nitpicks** from the harsh critic (e.g., "buried in a brief paragraph") are removed per rules against style nitpicks.

## Novel Insights

The interplay between the PB-scoring circularity and the underlying model quality reveals a nuanced picture: SIGMADOCK's fragment-based approach genuinely enforces intra-fragment chemical validity by construction (as shown by the 70.8% PB-validity without PB-scoring), but the headline 79.9% conflates two distinct contributions — the generative model's structural inductive biases and the selection heuristic's metric-specific optimization. The paper would be stronger if it treated these as complementary but distinct claims, reporting both numbers prominently and framing the PB-scoring as a cheap alternative to confidence models (which is how DiffDock handles sample selection), rather than letting the combined number dominate the narrative.

## Suggestions

- Report both 70.8% (without PB-scoring) and 79.9% (with PB-scoring) prominently in the abstract and results, explicitly framing the PB-scoring as a lightweight confidence model substitute. This would preserve the paper's strong narrative while being transparent about the selection heuristic's contribution.

- Qualify the AF3 comparison explicitly in the abstract itself (not just in the results section), noting that SIGMADOCK solves rigid-receptor re-docking while AF3 solves full co-folding — a fundamentally harder task.

## Score and Decision

**Calibration anchors compared:**

| Paper | Score | Comparison |
|-------|-------|------------|
| ShEPhERD (8.0, Oral) | 8.0 | Novel SE(3)-equivariant diffusion for drug design, no significant overclaiming. SIGMADOCK has comparable novelty but more overclaiming. |
| MOFDiff (8.0, Poster) | 8.0 | Fragment-based diffusion with E(3)-equivariance, comprehensive evaluation. SIGMADOCK is comparable in approach but has the PB-scoring circularity issue. |
| GroupBind (6.75, Poster) | 6.75 | Docking SOTA but overclaimed novelty (ComBind existed), missing head-to-head comparison. SIGMADOCK is stronger in theory and results. |
| EBMDock (5.75, Poster) | 5.75 | Docking with inflated metrics (ground-truth interface info). SIGMADOCK's PB-scoring issue is less severe than EBMDock's ground-truth access. |
| DrugFlow (6.67, Poster) | 6.67 | Strong empirical results but missing statistical variation. SIGMADOCK is stronger overall. |
| FreeLM (2.0, Reject) | 2.0 | Overclaimed outperforming GPT-3 with 0.3B model, uncontrolled comparison. SIGMADOCK's overclaiming is much milder — the underlying method is genuinely strong. |

SIGMADOCK sits above GroupBind (6.75) and DrugFlow (6.67) in terms of methodological novelty and theoretical grounding, but below the 8.0-scoring papers due to the PB-scoring circularity and AF3 comparison framing. The underlying contribution is genuinely strong (70.8% PB-validity without PB-scoring still massively outperforms prior work), but the overclaiming is a real concern that prevents a higher score.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>