Now I have sufficient calibration data. Let me write the final review.

## Summary

SIGMADOCK introduces a fragment-based SE(3) Riemannian diffusion model for molecular docking that decomposes ligands into rigid-body fragments via a novel FR3D fragmentation scheme, replacing torsional parameterization with independent SE(3) transforms per fragment. The method achieves 79.9% Top-1 PB-valid on the PoseBusters benchmark, claiming to be the first deep learning approach to surpass classical physics-based docking under the PB train-test split, while demonstrating competitive performance with AlphaFold3 using far less training data.

## Strengths

- **Novel and well-motivated formulation**: The fragment-based SE(3)^m approach is a genuine departure from torsional diffusion. The theoretical critique in Theorem 1 (Section 2.2.2) correctly identifies that torsional models induce non-product measures in Cartesian space, while the fragment formulation yields a factorized product of Haar measures. This is a principled motivation supported by the practical issues of gauge ambiguity and the lever effect (Section 2.2.2).

- **Soft triangulation constraints (Lemma 1)**: An elegant mechanism that injects bond-length/angle priors via cross-fragment distances without constraining dihedrals, effectively shrinking the DOF gap between fragment and torsional models. The ablation (Table 1, Config A) confirms a 12.8pp drop in PB-valid when removed, making it the single largest contributor.

- **Near-parity between RMSD<2Å (80.5%) and PB-valid (79.9%)**: Table 1 Config I shows SIGMADOCK almost never generates geometrically close but chemically implausible poses—directly addressing the key failure mode identified by Butenschön et al. (2024) for prior DL methods.

- **Interpretable failure analysis**: Table 2 shows failure rates correlate systematically with co-factor absence (41.2% when natural ligands present vs. 16.2% with none), providing evidence that the model learns genuine physicochemical interactions rather than memorizing poses.

- **Data efficiency and principled evaluation scope**: SIGMADOCK deliberately restricts to PDBBind v2020 for fair comparison (Section 3.1), acknowledges train-test leakage concerns, and reaches AF3-competitive results with only 19k training complexes (Table 4).

- **SO(3)-equivariant architecture with coordinate ambiguity resolution**: Theorem 2 (Section 2.4) establishes invariance to local coordinate orientation via pseudo-force prediction, resolving a real design challenge.

## Weaknesses

### Fatal
None.

### Major

- **Metric reporting inconsistency across comparisons makes the headline improvement claim partially unverifiable**: The abstract compares SIGMADOCK's "RMSD < 2Å PB-valid" (79.9%) against "12.7–32.8% reported by recent deep learning approaches," but Figure 4 does not clearly indicate whether baseline numbers are RMSD<2Å or PB-valid. For example, G2G and Vibe2 report 58.1% in Figure 4 left under "Top-1 (%)" without specifying which metric. The 12.7–32.8% range appears to come from Butenschön et al. (2024) reporting PB-valid rates for DL methods in holo-specified mode, while SIGMADOCK operates in pocket-specified mode—a different evaluation condition not clearly flagged. The paper does not report both RMSD<2Å and PB-valid for every baseline in one table, making it impossible for the reader to determine the true gap on the same metric under the same conditions. This doesn't invalidate the results but undermines the precision of the headline claims.

- **Contradictory generalization numbers between Figure 4 (right) and Table 4**: Figure 4 right reports Top-1 percentages of 51%, 53%, 53% across the three sequence similarity bins (counts 109, 76, 123), while Table 4 reports PB-valid percentages of 72%, 79%, 87% for the same bins with the same counts. The Figure 4 right numbers average to ~52%, wildly inconsistent with the ~80% overall rate. These cannot both be the same metric on the same data. The paper does not clarify what metric Figure 4 right uses. Since the paper's generalization claim ("consistent generalisation to unseen proteins") is central to its contribution, this ambiguity significantly weakens the evidence for that claim.

- **Opaque energy scoring component undermines the "surpassing classical methods" claim**: Table 1 Config D shows that removing "(pseudo) binding energy" scoring drops performance from 80.5%/79.9% to 67.2%/66.1%—a 13.3pp absolute contribution. The energy function is described only as a "simple and cheap heuristic" (Section 2.5, line 180) with details deferred to the appendix (which is stripped). If this scoring incorporates classical force field terms, then the claim of being "the first deep learning approach to surpass classical physics-based docking" is misleading: the method surpasses classical docking partly *by using* classical scoring to rank its own samples. The paper does not present the comparison of SIGMADOCK-without-energy-scoring vs. classical methods on PB-valid, which would be the fair test for this claim. Note: even without energy scoring, 66.1% PB-valid may still exceed some classical baselines on RMSD<2Å, but the paper doesn't show this comparison.

### Minor

- **Cross-setting DiffDock comparison**: The "6.3× higher PB-validity than DiffDock" claim (Section 3.2) compares SIGMADOCK (pocket-specified, PB-valid) against DiffDock (holo-specified, PB-valid from an external source). These are different evaluation conditions. The paper does compare fairly against pocket-specified baselines (G2G, Vibe2) showing a 79.9% vs. 58.1% gap, but buries this more modest (though still substantial) improvement.

- **AF3 comparison inconsistency**: The paper states "we cannot directly compare SIGMADOCK to co-folding methods" (line 260) but then provides Table 4 comparing against AF3 on PB-valid, calling it "competitive performance." These two statements are in tension. Since AF3 does co-folding (jointly predicting protein structure) on a different task, the comparison is informative but should not be characterized as competitive on the same task.

- **Vina PB-valid rate not reported**: Table 3 shows Vina's RMSD<2Å rates (~57%) but not PB-valid. Since PB-valid is the paper's chosen primary metric and the basis for the "surpassing classical methods" claim, reporting Vina's PB-valid rate is important.

- **Table 1 ablation training vs. inference-time confound**: Rows A–C are retrained from scratch while rows D–H appear to be inference-time ablations. The paper notes "A–C are re-trained from scratch" (line 237) but does not explicitly flag D–H as inference-only, which affects how readers interpret the contributions.

- **Stochastic fragmentation variance uncharacterized**: FR3D (Section 2.2.3) uses stochastic search, meaning the same ligand can receive different fragmentations. The paper does not analyze how much performance varies across different fragmentation outcomes, which affects reproducibility and reliability.

### Trivial
None.

## Nice-to-Haves

- Report both RMSD<2Å and PB-valid for all baselines in one unified table, clearly specifying evaluation conditions (pocket-specified vs. holo-specified).
- Define the pseudo binding energy function in the main text, and show SIGMADOCK-without-energy-scoring vs. Vina on PB-valid to isolate the DL contribution from the classical scoring contribution.
- Clarify what metric Figure 4 right reports and reconcile the numbers with Table 4.
- Analyze FR3D fragmentation variance and its impact on stability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic claim that the abstract's 12.7–32.8% range is "cherry-picking" or "dishonest"**: The range 12.7–32.8% appears to come from Butenschön et al. (2024) reporting PB-valid rates for DL methods. This is not fabricated; the issue is metric/condition clarity, not dishonesty.

- **Harsh critic claim that this is "not a presentation gap—it makes it impossible to verify"**, phrased as a fatal issue: While the metric inconsistency is a genuine major concern, the numbers are present in the paper and can be cross-referenced by a careful reader. The issue is ambiguity, not absence of data.

- **Harsh critic claim that the energy scoring dependency is "fundamentally misleading"**: This is an important concern but calling it "fundamentally misleading" goes too far. The paper does disclose using energy scoring (Section 2.5) and shows the ablation (Table 1). The issue is about the framing of the "surpassing classical methods" claim, not deception.

- **Strength Finder claim that "SIGMADOCK is the first DL method to surpass classical physics-based docking on the PB benchmark"**: This claim is partially undermined by the energy scoring dependency (Major weakness #3). While the method does surpass classical baselines on PB-valid, the contribution of classical scoring to this result is substantial (13.3pp). The claim is not fully supported as stated but the result is still genuine.

- **Harsh critic claim about DiffDock comparison being a "category error"**: This overstates the case. DiffDock was evaluated in holo-specified mode because that was its original setting. The comparison is imperfect but not a category error—both are re-docking tasks, just with different receptor specification methods.

- **Harsh critic note about "symmetry correction implementation matters"**: Specific implementation details of Meli & Biggin (2020) symmetry correction are a reasonable concern but standard practice. The paper cites the method used, which is the community standard.

- **Request for confidence intervals / statistical tests on Table 2**: Not standard in this field for docking benchmarks and would be a nice-to-have, not a weakness.

- **Harsh critic assertion that Theorem 1 is "essentially a definitional observation"**: While the mathematical content may seem definitional, the practical implications (stiff dynamics, gauge ambiguity) are the real contribution, and the theorem formally justifies the design choices. This is appropriate for the paper's scope.

## Novel Insights

The fragment-based SE(3)^m formulation represents a genuine conceptual advance over torsional diffusion for molecular docking. The key insight—that independent rigid-body fragments yield a factorized product measure avoiding the entangled induced measures of torsional models—is both theoretically clean and empirically impactful. However, the paper's evaluation framework has a notable gap: the near-parity between RMSD<2Å and PB-valid suggests the fragment structure + triangulation constraints are doing the heavy lifting for chemical validity, while the pseudo binding energy scoring does most of the work for ranking accuracy. Disentangling these two contributions more carefully would clarify whether the core advance is in generation quality (the SE(3)^m diffusion itself) or in the evaluation/ranking pipeline.

## Suggestions

- Create a single comprehensive table with all methods, both metrics (RMSD<2Å and PB-valid), clearly labeled evaluation conditions (pocket-specified vs. holo-specified), and sample counts (N_seeds). This would immediately resolve the metric ambiguity concern.
- Run the experiment: SIGMADOCK (no energy scoring) vs. Vina on PB-valid with the same N_samples. This single comparison would either validate or invalidate the "surpassing classical methods" claim independently of classical scoring contributions.
- Add a caption or footnote to Figure 4 right specifying which metric is plotted (RMSD<2Å or PB-valid) and reconcile with Table 4 numbers.

## Score and Decision

**Calibration anchors:**

1. **High-scoring**: Quotient-Space Diffusion Models (3JPAkwSVc4, avg 7.5, Accept Oral) — similar SE(3) theoretical framework for molecular generation, principled formal contribution. SIGMADOCK has comparable theoretical novelty but weaker evaluation transparency.

2. **High-scoring**: La-Proteina (RDerF20JYT, avg 8.0, Accept Poster) — strong empirical results with novel flow matching for protein generation. SIGMADOCK's results are strong but evaluation issues hold it below this level.

3. **Medium-scoring**: Matcha (r9Uw9kKjUy, avg 4.0, Reject) — Riemannian flow matching for docking, flagged for unfair metric conflation on PoseBusters, missing baselines. SIGMADOCK has similar metric conflation issues but much stronger core results and a more novel formulation.

4. **Medium-scoring**: PoseX (qqzxKudD4T, avg 5.0, Accept Poster) — benchmark with cross-docking metric conflation concerns. SIGMADOCK has more substantive technical contribution.

5. **Medium-scoring**: Bento (kIxAQxUZHq, avg 5.0, Reject) — benchmarking paper, limited novelty. SIGMADOCK is well above this in technical contribution.

6. **Low-scoring**: MagicDock (nVnPqLtO8r, avg 3.5, Reject) — claims surpassing classical docking but evaluation methodology is weak. SIGMADOCK is clearly above this.

SIGMADOCK's core contribution (fragment SE(3)^m diffusion with triangulation constraints) is genuinely novel and the empirical performance is strong. However, the evaluation has significant transparency issues (metric inconsistency, contradictory Figure/Table numbers, large dependence on an undefined classical scoring component). These are not fatal but they meaningfully undermine the precision of the central claims. Compared to Matcha (4.0, similar metric issues but weaker method), SIGMADOCK is clearly superior. Compared to Quotient-Space Diffusion (7.5, cleaner evaluation), SIGMADOCK falls short on evaluation rigor but has comparable technical contribution. The paper sits above the medium anchors but below the high anchors primarily due to evaluation transparency.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>