Now I have enough information to write the review. Let me carefully assess the key claims and weaknesses.

Let me verify the harsh critic's claim about DoFs more carefully by re-reading the relevant text:

The paper says:
- Torsional models have k+6 DoFs
- Naïve fragmentation yields m̂ = k+1 fragments with 6m̂ DoFs
- FR3D reduces m̂ to m (empirically m ≈ ⅔m̂)
- Effective DoFs concentrate between k+6 (triangulation lower bound) and 6m (unconstrained upper bound)

So with m ≈ ⅔(k+1), the upper bound becomes 6m ≈ 4(k+1) ≈ 4k+4, which is indeed larger than k+6. The paper acknowledges this by saying the *effective* DoFs concentrate *between* k+6 and 6m, and that triangulation provides soft constraints that reduce effective DoFs. The paper's claim is about reducing from 6m̂ → 6m (via FR3D) and that triangulation pushes effective DoFs toward k+6. The harsh critic is right that the raw DoFs of the fragment model (6m) exceed those of the torsional model (k+6), but the paper doesn't claim it reduces DoFs below the torsional model. The paper says "How can we reduce the DoFs of the system" referring to reducing from the naïve 6m̂ to the effective DoFs with triangulation, and then it discusses the effective DoFs being between k+6 and 6m. The paper's language about "reducing DoFs" is somewhat imprecise but it's about reducing from the naïve fragmentation, not about claiming the final DoFs are below the torsional model. The effective DoFs claim (via soft triangulation) is at least partially addressable.

Regarding the energy scoring ablation: The ablation (Config D) shows removing energy scoring drops from 87.4→67.2 RMSD<2 and 80.5→66.1 PB-valid. Wait, that doesn't match - let me re-read:
- Config D removes energy scoring: 67.2, 66.1
- Config * (full with N=40): 80.5, 79.9

So energy scoring contributes about 13-14% absolute improvement in PB-valid (66.1 to 79.9). The harsh critic said 20.3% which is 87.4-67.2 for RMSD<2, but those numbers don't appear in the table. Let me re-check...

Actually, looking at the table again:
- Config D (- Energy Scoring): RMSD<2 = 67.2, PB Val. = 66.1
- Config * (SIGMADOCK N=40): RMSD<2 = 80.5, PB Val. = 79.9

So energy scoring contributes 80.5-67.2 = 13.3% for RMSD<2 and 79.9-66.1 = 13.8% for PB-valid. Not the 20.3% the harsh critic claimed. Unless they were looking at a different pair. Let me check: perhaps there's a row I'm missing. The Config H with N=10 achieves 74.7, 72.2 - so between N=10 and N=40 with energy scoring we see additional gains.

Wait, let me re-read: the config * has Nseeds=40 with energy scoring, and Config D removes energy scoring. But does Config D also have N=40? That's unclear. If Config D removes energy scoring but also has N seeds different, then the comparison is not clean. But the ablation table seems to indicate these are single-ablation changes from the full model.

Actually looking at the harsh critic's numbers: 87.4 → 67.2 for RMSD<2. The 87.4 doesn't appear in the table provided. That's odd. Let me check again - maybe there's a different version of the table or I'm missing something.

Looking at the table more carefully:
| D | (-) Energy Scoring | 67.2 | 66.1 |
| * | SIGMADOCK (Nseeds = 40) | 80.5 | 79.9 |

So the drop from full model to removing energy scoring is: 80.5 - 67.2 = 13.3% (RMSD<2) and 79.9 - 66.1 = 13.8% (PB-valid). Not 20.3%.

The harsh critic may have misread the table or was looking at a different comparison. The 13.3% and 13.8% figures are still significant but not as dramatic as claimed.

Now let me also check the paper's claim about surpassing Vina. Looking at Figure 4 description (which I can't see as an image), but the text says 79.9% PB-valid is the headline number. Vina is noted at ~76% RMSD<2 in the paper's description of Figure 4. Since Vina produces chemically valid poses by construction, its PB-valid ≈ RMSD<2 (as the harsh critic notes). So Vina's PB-valid could be ~76%, meaning SigmaDock's 79.9% margin over Vina is only ~4%.

Let me also check the DiffDock comparison details more carefully. The paper says DiffDock achieves 12.7-32.8% PB-validity, and that DiffDock-L achieves 50% Top-1 RMSD only on PB. So the comparison is between different metrics (RMSD only vs RMSD & PB-valid) and different training data sizes.

Now I can write the final review.

## Summary

SIGMADOCK introduces a fragment-based SE(3) Riemannian diffusion model for molecular docking that decomposes ligands into rigid-body fragments and learns to reassemble them within a binding pocket. The approach avoids the entangled, non-product induced measures of torsional parameterizations (Theorem 1) and enforces chemical validity by construction through the fragment decomposition, achieving 79.9% Top-1 success rate (RMSD < 2Å & PB-valid) on the PoseBusters temporal split—the first DL method to surpass classical physics-based docking on this benchmark.

## Strengths

- **Theoretical grounding for fragment-over-torsional parameterization (Theorem 1):** The paper provides a formal argument that torsional models produce entangled, non-product induced measures on Cartesian coordinates, whereas fragment-based models yield a factorized product of Haar measures on SE(3)^m. This identifies a genuine and underappreciated problem with torsional diffusion formulations, giving principled motivation for the design choice.

- **Strong empirical leap over prior DL methods on PB-validity:** SIGMADOCK achieves 79.9% Top-1 PB-validity on PoseBusters, compared to 12.7–32.8% for prior DL methods evaluated under the same split—a substantial improvement, particularly on the metric (chemical plausibility) that DL methods previously failed on.

- **Effective ablation design demonstrating component contributions:** The ablation study (Table 1) systematically shows the impact of triangulation conditioning (∅→67.1% without, 79.9% with), protein-ligand interactions, FR3D merging, and energy scoring. The co-factor analysis (Table 2) provides mechanistic insight into failure modes, showing failures correlate with excluded co-factors rather than memorization.

- **Natural enforcement of chemical validity:** By decomposing ligands into rigid fragments whose internal geometry is preserved by construction, SIGMADOCK directly addresses the major failure mode of DL docking (chemically implausible outputs), eliminating the need for post-hoc minimization hacks.

- **Data efficiency and controlled evaluation:** Trained on only 19,443 complexes from PDBBind(v2020) with deliberate temporal split, achieving strong generalization to unseen proteins (72% PB-valid even at [0,30)% sequence similarity), with lower train-test leakage than many competitors.

## Weaknesses

### Fatal
None.

### Major

- **Energy-based reranking accounts for a substantial portion of headline performance, but baselines are not equivalently reranked:** Removing energy scoring drops PB-valid from 79.9% to 66.1% (13.8% absolute) and RMSD<2 from 80.5% to 67.2% (13.3% absolute) according to Table 1 (Config D). This means ~17% of the gap to the next-best DL method, and a non-trivial fraction of the margin over Vina, comes from the scoring heuristic rather than the generative model itself. While the paper transparently reports this ablation, the abstract and introduction frame the results as evidence for the *fragment-based SE(3) diffusion framework* surpassing classical methods, without clarifying the role of this post-hoc scoring step. The paper does not show whether DiffDock or Vina benefit similarly from equivalent reranking of 40 samples. This matters because: (1) classical methods like Vina already use energy functions internally; (2) DiffDock uses a separately trained confidence model for ranking. The comparison is thus not purely between generative architectures. The contribution of the diffusion model itself is best represented by the no-scoring ablation (66.1% PB-valid), which still surpasses DiffDock but by a smaller margin than the headline number suggests.

- **The "first DL method to surpass classical methods" claim is accurate but the margin over Vina is modest and context-dependent:** Vina achieves ~76% PB-valid (by construction, its PB-valid ≈ RMSD<2 rate), while SIGMADOCK's margin is only ~4%. This margin could shift with different pocket definitions, different Vina configurations, or different evaluation protocols. The paper's Table 3 shows reducing pocket size doesn't help Vina, which partially addresses this, but the claim of "surpassing classical methods" rests on a narrow gap in an evaluation setting that still uses the holo (bound) protein structure—a setting favorable to DL methods.

### Minor

- **DoF framing is somewhat imprecise but not misleading upon close reading:** The paper's claim that FR3D "reduces the DoFs of the system" refers to reducing from the naïve 6m̂ to an effective range between k+6 and 6m, not to reducing below the torsional model's k+6. When m ≈ ⅔(k+1), the unconstrained upper bound 6m ≈ 4k+4 still exceeds k+6. The paper acknowledges this range (Section 2.2.3: "the effective DoFs concentrate between k+6 and 6m"), and the triangulation constraints provide soft but learned inductive biases that push effective DoFs closer to the lower bound. However, the introductory framing ("To reduce the additional degrees of freedom introduced from fragmentation") could mislead casual readers into thinking the fragment model operates in fewer dimensions than the torsional model. The genuine advantage is the factorized prior structure (Theorem 1), not dimensionality reduction per se.

- **No head-to-head comparison with torsional diffusion using the same architecture and training data:** The paper compares against DiffDock's published numbers, which differ in architecture, training data size, and evaluation protocol. While the comparison is useful, it cannot isolate whether the gains come from the fragment parameterization or from other design choices (architecture, training data, etc.). A direct ablation—same backbone and training pipeline, torsional vs. fragment diffusion—would more convincingly demonstrate the advantage of the fragment approach.

- **Conformational manifold assumption validation is relegated to the appendix:** The key assumption that bound poses can be approximately recovered from vacuum conformers via SE(3) + torsional alignment underpins the entire method. The paper states "RMSD ≪ 2Å" but defers the quantitative distribution of alignment errors to Appendix D.3. Including summary statistics in the main text would strengthen confidence.

- **Co-factor analysis sample sizes are very small:** Table 2 has only 17 natural ligand complexes and 37 with crystallisation aids. Percentage comparisons from such small samples have large confidence intervals, limiting the strength of conclusions drawn.

### Trivial
None worth listing.

## Nice-to-Haves

- **Cross-docking evaluation:** The paper evaluates on re-docking (holo structure provided) but the introduction motivates HTVS, where cross-docking (different protein conformations) is more realistic. Testing on cross-docking would strengthen practical relevance.

- **Energy-based reranking applied to baselines:** Showing what happens when DiffDock/Vina samples are reranked with equivalent energy scoring would clarify how much of SIGMADOCK's advantage is generative vs. scoring-based.

- **AF3 comparison under identical conditions:** The AF3 comparison (Table 4) cites numbers from a different paper under potentially different evaluation settings. A direct comparison under the same protocol would be more convincing.

## Removed Points

- **Harsh critic's claim that DoF reduction claim "reverses the facts" (structural):** The paper explicitly states "the effective DoFs concentrate between k+6 and 6m" and positions FR3D as reducing from the naïve 6m̂ to a range bounded below by k+6. While the introductory framing could be clearer, the paper does not claim the fragment model has fewer DoFs than the torsional model—it claims FR3D reduces DoFs relative to naïve fragmentation and that soft constraints push effective DoFs toward k+6. The harsh critic overstates this as "reversing the facts." Demoted to minor.

- **Harsh critic's claim of "20.3% absolute reduction" from energy scoring:** The actual numbers from Table 1 show Config D (−Energy Scoring) drops PB-valid from 79.9% to 66.1% (13.8% absolute) and RMSD<2 from 80.5% to 67.2% (13.3% absolute). The 20.3% figure appears incorrect based on the provided table. Corrected to 13.3–13.8% in the review.

- **Formatting/typo nitpicks (harsh critic's section-by-section notes on appendix-deferred proofs):** The parser strips appendices; proofs and analyses exist in the original submission. Removed per instructions.

- **Missing related works:** Cannot verify existence of missing references. Removed per instructions.

- **Strawman criticism about "energy scoring accounts for ~20% of accuracy" and should be the "primary comparison point":** The energy scoring is a legitimate component of the system (not a hack), and using it is standard practice (DiffDock uses a confidence model, Vina uses energy internally). The no-scoring ablation is informative but calling it the "primary comparison point" would be equally misleading since the architecture includes this scoring by design.

- **Strengths about "dramatic improvement" and "6.3× higher PB-validity than DiffDock":** Retained as genuine but context-dependent—the gap to DiffDock is large on PB-validity, but the comparison is not purely apples-to-apples due to different training sizes and evaluation conditions.

## Novel Insights

The key insight that emerges from the collective review is that SIGMADOCK's advantage is not primarily about dimensionality reduction (despite the paper's framing), but about the *structure* of the prior: a factorized product of Haar measures on SE(3)^m provides better-conditioned diffusion dynamics than the entangled, non-product induced measures from torsional parameterizations. The fragment decomposition is a means to this structural end, not an end in itself. The energy scoring component is substantial and integral to the system—but importantly, it replaces a separately trained confidence model with a simple physics-based heuristic, which is itself a design contribution that eliminates a training dependency. The tension in the paper is that its strongest theoretical claim (prior factorization) is distinct from its strongest practical selling point (PB-validity and chemical plausibility by construction), and neither alone fully explains the headline results, which arise from the combination of fragment decomposition, triangulation conditioning, and energy-based reranking.

## Suggestions

- Report results both with and without energy scoring as the primary comparison, and discuss what happens when baselines receive equivalent reranking over their sample pools.
- Add a same-architecture ablation comparing fragment vs. torsional parameterization to isolate the contribution of the fragment formulation from other design choices.
- Include the distribution of alignment errors (RMSD between conformers and bound poses) and fragment counts (m/m̂ ratio) in the main text, not just in the appendix.
- Soften the "first DL method to surpass classical methods" language to acknowledge the modest margin over Vina (~4% PB-valid) and the role of energy scoring in closing the gap.

## Evaluation Axis Summary

- **Originality:** High. The fragment-based SE(3)^m formulation, FR3D merging, triangulation conditioning, and Theorem 1 on prior factorization represent genuine conceptual contributions that rethink the diffusion parameterization for docking.
- **Importance of research question:** Very high. Chemical plausibility of DL-generated poses is the central problem identified by the PoseBusters benchmark, and SIGMADOCK directly addresses it.
- **Claims support:** Partially supported. The 79.9% headline number conflates generative model quality with energy-based reranking (13.8% absolute contribution). The "surpassing classical methods" claim holds but with a narrow margin. The theoretical contribution (Theorem 1) is solid.
- **Soundness of experiments:** Good ablations, but missing head-to-head same-architecture comparison with torsional diffusion, and no equivalent reranking applied to baselines.
- **Clarity:** Generally clear, though some imprecise DoF framing and energy scoring's contribution is underemphasized in the abstract/introduction.
- **Value to community:** High. The approach opens a new direction for molecular docking diffusion models and achieves a genuine milestone on PB-validity.

## Calibration

I compared this paper against the following anchors:

1. **ShEPhERD** (avg 8.0, Accept Oral): SE(3)-equivariant diffusion for 3D molecular generation with strong empirical results and theoretical grounding. SIGMADOCK is comparable in novelty and motivation but has a clearer real-world benchmark milestone (first DL method to surpass classical docking on PB-validity). However, SIGMADOCK's headline number partially depends on post-hoc scoring.

2. **GroupBind** (avg 6.75, Accept Poster): Multi-ligand docking framework achieving SOTA on PDBBind. Reviewers noted incomplete baselines and overclaimed "first" status. SIGMADOCK has stronger benchmark results and more rigorous evaluation, but similar concerns about fairness of comparisons.

3. **EBMDock** (avg 5.75, Accept Poster): Energy-based docking framework. Relevant comparison because it also uses energy scoring as a core component. SIGMADOCK achieves much stronger empirical results.

4. **PoseBench** (avg 4.38, Withdrawn): Benchmark paper questioning DL vs classical docking. SIGMADOCK directly addresses the concerns raised in this benchmark.

5. **dIK7GpOwNY** (avg 3.0, Reject): Claims about dimensionality that reviewers found misleading/tautological. SIGMADOCK's DoF framing is imprecise but not tautological—the actual contribution (factorized prior) is substantive.

6. **NExT-Mol** (avg 5.5, Accept Poster): Strong empirical results but overclaimed validity and missing baselines. Similar pattern to SIGMADOCK.

Considering these anchors, SIGMADOCK has genuine substantive contributions (fragment formulation, triangulation constraints, Theorem 1) and achieves a clear milestone on an important benchmark, but the headline number overstates the generative model's contribution by conflatting it with energy-based reranking, and the DoF framing is imprecise. The paper is stronger than GroupBind (6.75) and NExT-Mol (5.5) in terms of empirical achievement and theoretical grounding, but weaker than ShEPhERD (8.0) in terms of clean attribution of gains. A score in the 6.5-7.0 range seems appropriate.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>