## Summary

This paper proposes Boltzmann Alignment (BA-DDG), a method for predicting ΔΔG in protein-protein interactions by connecting inverse folding model log-likelihoods to binding free energy through a thermodynamic cycle framework. The approach outperforms existing supervised (per-structure Spearman 0.5134) and unsupervised (0.3201) methods on SKEMPI v2, and demonstrates utility in binding energy prediction, protein-protein docking selection, and antibody optimization.

## Strengths

- **Strong empirical performance across multiple benchmarks**: BA-DDG achieves clear improvements over prior methods on SKEMPI v2 (Table 1), with per-structure Spearman 0.5134 vs. previous SoTA 0.4324. The improvement is consistent across all 7 evaluation metrics, and the ablation in Table 2 validates the benefit of the thermodynamic cycle formulation (BA-Cycle 0.3201 vs. ProteinMPNN 0.2741).

- **Effective unsupervised scoring heuristic**: BA-Cycle (Eq. 10) provides a fast, training-free method for ranking mutations that outperforms all unsupervised learning baselines and performs comparably to traditional empirical energy functions (Table 1).

- **Demonstrated downstream utility**: Sections 4.4.2 and 4.4.3 show practical value for docking selection (Fig. 4 right) and antibody maturation (Table 5), indicating usefulness for computational design pipelines.

- **Comprehensive baseline comparison**: The evaluation spans traditional forcefields, sequence-only models, structure-informed pre-trained models, and supervised baselines across multiple metrics, providing a clear empirical picture.

## Weaknesses

### Fatal

None.

### Major

- **The KL divergence term in the loss (Eq. 12) has the wrong sign**: The paper defines $\mathcal{L}_{\text{Boltzmann}} = \|\Delta\Delta G - \widehat{\Delta\Delta G}\| - \beta D_{\text{KL}}(p_\theta \| p_{\text{ref}})$. Minimizing this loss *maximizes* the KL divergence from the reference model, directly contradicting the stated goal of "maintaining the distribution of the original pre-trained model" (Sec. 3.3). This is a genuine mathematical error that undermines the "Boltzmann Alignment" training objective as presented. If corrected to $+\beta$, the mechanism becomes standard KL-regularized fine-tuning, reducing the novelty over established approaches.

- **The physical derivation's core premise conflates model likelihoods with thermodynamic probabilities**: Eq. 1 treats $p_{\text{bind}}$ and $p_{\text{unbind}}$ (defined as conditional probabilities of a specific backbone conformation given a sequence) as equivalents of Gibbs distribution probabilities. In statistical mechanics, free energy relates to partition functions summing over microstates, not single-conformation inverse folding likelihoods $p(S|X)$. The subsequent derivations (Eqs. 2-8) are internally consistent but built on this approximation, which the paper frames as "physical inductive bias" without adequately acknowledging it is a physically-motivated heuristic rather than a rigorous derivation. The authors' own admission in Sec. 4.4.1 that they "lack a comprehensive explanation for canceling out $p(\mathcal{X}_{\text{bnd}})$ and $p(\mathcal{X}_{\text{unbnd}})$" further weakens the physical grounding claim.

- **Treating $k_B T$ as a learnable parameter undermines physical claims**: Sec. 3.3 explicitly states $k_B T$ is optimized alongside model weights. Thermodynamically, $k_B T$ is a fixed physical constant (~0.593 kcal/mol at 298K). Learning it reduces the method to supervised regression with an empirical scaling factor, stripping the claimed thermodynamic grounding. While this may improve performance, it contradicts the paper's framing of physics-informed alignment.

- **Performance claims lack uncertainty quantification**: Tables 1-4 report only mean values across 3-fold cross-validation without standard deviations, confidence intervals, or paired significance tests. Given SKEMPI v2's heterogeneity and the modest absolute improvement (~0.08 Spearman over Surface-VQMAE), it is unclear whether the headline SoTA claim is robust or within fold-to-fold variance.

### Minor

- **The unbound state approximation (Eq. 9) assumes complete chain independence**, which ignores interfacial solvation entropy and conformational coupling between chains. While this is a standard approximation, the paper should more clearly delineate the conditions under which it is likely to fail (e.g., large conformational changes upon binding).

- **No stratified analysis by mutation type**: The rigid-body assumption and independent-chain unbound approximation likely perform poorly for mutations that induce significant conformational changes. A stratified error analysis (single vs. multi-point, interface core vs. rim, hydrophobic vs. charged) would clarify the method's boundary conditions.

### Trivial

None identified.

## Nice-to-Haves

- Report the learned value of $k_B T$ post-training to assess whether it deviates substantially from the physical constant, which would inform whether the model truly learns a physical relationship or merely an empirical scaling.
- Provide scatter plots of predicted vs. experimental ΔΔG colored by complex family or PDB ID to reveal whether improvements are driven by broad generalization or overfitting to specific complex families.
- Clarify whether the AF3 structures used in Table 4 were energy-minimized post-prediction, as this affects interpretability of the "predicted structure robustness" claim.

## Removed Points

- **Criticism that the alignment terminology is misleading for a continuous energy task**: The paper explicitly scopes Boltzmann Alignment to ΔΔG prediction and uses the term internally. While "alignment" typically refers to LLMs, the paper provides sufficient context for its usage in this domain. This is a terminological preference, not a substantive flaw.
- **Criticism that the unbound state approximation is presented as novel**: The paper (Sec. 3.2) frames this as a "reasonable estimation approach," not as a novel physical insight. This point was overstated.
- **Criticism about missing Appendix proofs or references**: Per submission rules, the Appendix was stripped from the reviewed paper. These sections exist in the original submission and should be evaluated based on their presence there.

These points are flagged to be removed, treat them with caution.

## Novel Insights

The paper's core insight — that inverse folding models implicitly capture energy landscapes through their sequence-likelihood outputs — has been observed empirically by prior work (Bennett et al., 2023; Widatalla et al., 2024) but remains underexplained. The paper's contribution is providing a thermodynamically-motivated derivation for this correlation, even if the derivation is a heuristic rather than rigorous proof. The explicit modeling of the unbound state via independent-chain factorization (Eq. 9) is a practical contribution that improves empirical performance. However, the "Boltzmann Alignment" framing overstates the method's physical grounding relative to what the equations actually implement.

## Suggestions

1. **Correct Eq. 12's sign**: Change the loss to $\mathcal{L} = \|\Delta\Delta G - \widehat{\Delta\Delta G}\| + \beta D_{\text{KL}}(p_\theta \| p_{\text{ref}})$ to match the stated regularization objective. Retrain with the corrected sign and verify that performance is maintained (which would support the empirical utility claim regardless of theoretical framing).

2. **Add standard deviations or confidence intervals** to all multi-fold results in Tables 1-4. For the 3-fold CV, this is readily computable and would substantiate or contextualize the SoTA claims.

3. **Reframe the theoretical section**: Instead of presenting the derivation as establishing true thermodynamic equivalence, frame it as a physically-motivated heuristic. This honest framing would still credit the method's empirical success while avoiding overclaiming physical rigor.

4. **Report the learned $k_B T$ values**: Showing whether the optimized $k_B T$ converges near the physical constant would be informative for interpreting what the model actually learns.

## Score and Decision

I calibrated this paper against several anchors:

- **Light-DDG** (IxmWIkcKs5.md): Accepted as Poster with scores 6, 8, 5, 8 (avg ~6.8). Similar ΔΔG prediction domain but with fewer theoretical overclaims and broader downstream validation.
- **clawNOs** (KEpR8hFzvO.md): Rejected with scores 5, 6, 3, 6, 5 (avg 5.0). Similar pattern — strong empirical results with weak theoretical justification and missing theoretical novelty.
- **HEi70bBquo** (Score 5,3,5 avg 4.3): Strong experiments but incremental methodology.
- **uL1H29dM0c** (Score 6,8,8,6 avg 7.0): Accepted with solid physics-informed derivation.

This paper's empirical results are genuinely strong and broadly validated (multiple baselines, multiple metrics, multiple downstream tasks, predicted structure robustness). However, the sign error in Eq. 12, the overclaiming of physical derivation rigor, and the learned $k_B T$ parameter collectively undermine the "Boltzmann Alignment" central claim. The paper functions as an effective empirical method, but its theoretical framing needs correction. Compared to the calibration anchors, it lies between clawNOs (avg 5.0, rejected) and Light-DDG (avg 6.8, accepted). It is stronger empirically than clawNOs but has similar theoretical overclaim issues; it is weaker than Light-DDG in comprehensiveness and clarity of contribution. On balance, the paper's strong empirical validation outweighs its theoretical framing issues, which are addressable through reframing.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>