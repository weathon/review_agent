## Summary
This paper proposes Boltzmann Alignment, a method that integrates thermodynamic cycles and Boltzmann distributions into inverse folding models to predict mutational effects on protein-protein interactions (ΔΔG). The key innovation is explicit modeling of the unbound state in the thermodynamic cycle, which addresses a known limitation in prior inverse folding-based approaches. The method achieves state-of-the-art results on SKEMPI v2 with Spearman correlations of 0.3201 (unsupervised) and 0.5134 (supervised).

## Strengths
- **Novel unbound state modeling**: The explicit approximation of unbound state likelihood as the product of independent chain likelihoods (Eq. 9) is a methodologically sound improvement over prior inverse folding methods that only consider the bound complex. Table 2 validates this design choice, showing BA-Cycle achieves 0.3201 Spearman vs. 0.2741 for standard ProteinMPNN usage.
- **Strong empirical performance**: BA-DDG achieves state-of-the-art results across all 7 evaluation metrics on SKEMPI v2 (Table 1), with particularly notable improvements in per-structure correlations (0.5453 Pearson, 0.5134 Spearman) compared to previous best methods like Surface-VQMAE (0.4694/0.4324).
- **Comprehensive ablation studies**: The paper includes thorough ablations on the thermodynamic cycle (Table 2), supervision methods (Table 3 showing Boltzmann supervision outperforms SFT and DPO), and structure quality robustness (Table 4 showing minimal degradation with AlphaFold3 predicted structures).

## Weaknesses

### Fatal
None

### Major
- **Ambiguous baseline comparison protocol**: The Abstract claims to surpass "previously reported SoTA results" (0.2632 and 0.4324), while Table 1 caption states "Mean results of 3-fold cross-validation." The baseline values in Table 1 exactly match literature values, but SKEMPI structure-based splits are not unique—different clustering seeds yield different test sets. If baseline numbers are cited from literature (different splits) while BA-DDG uses new splits, the comparison is invalid. If baselines were re-run, the text should explicitly state "we re-implemented" rather than "previously reported." This ambiguity directly undermines the SoTA claims and needs clarification.

### Minor
- **Sign error in supervised loss formulation (Eq. 12)**: Equation 12 defines the loss as `||ΔΔG - ΔΔG_hat|| - β D_KL(p_θ || p_ref)`. A negative KL coefficient mathematically encourages *maximizing* divergence from the reference model, contradicting the stated goal of "maintaining the distribution of the original pre-trained model." This appears to be a typographical error (the implementation likely uses +β), but as written, the equation is mathematically incorrect. Similar formula typos in accepted papers (e.g., 0r1KU3dlps.md, accepted at 6.50) suggest this is correctable, but it needs fixing.
- **Overstated physical grounding via learnable temperature**: Section 3.3 treats k_B T as a learnable parameter, yet the paper frames the method as having "physical inductive bias" from thermodynamic principles. In physical thermodynamics, k_B T is a fixed environmental constant. Making it learnable absorbs scaling mismatches into this parameter, reducing the "Boltzmann" derivation to a heuristic scaling layer. The performance gains likely stem from the unbound state approximation (Eq. 9) rather than thermodynamic alignment itself. The physical framing should be tempered accordingly.

### Trivial
- **Per-Structure vs. Overall metric emphasis**: The Abstract highlights Per-Structure Spearman coefficients (0.3201/0.5134), but Table 1 shows Overall Spearman coefficients are significantly higher (0.4097/0.6346). Highlighting the lower metric without explanation is confusing and suggests potential cherry-picking to match specific baseline numbers from literature.

## Nice-to-Haves
- Report standard deviations for the 3-fold cross-validation results in Table 1 to enable statistical significance testing of claimed improvements.
- Analyze the learned value of k_B T and discuss whether it deviates from physical room temperature (~0.59 kcal/mol), explicitly acknowledging if it acts as a scaling factor.
- Expand the antibody design case study (Section 4.4.3, Table 5) beyond five single-point mutations to better substantiate generative design capabilities.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic's "Fundamental Error" claim**: While Eq. 12 does have a negative KL sign, calibration papers like 0r1KU3dlps.md (accepted at 6.50) and JRlNrcTllN.md (accepted at 6.00) show that formula typos don't necessarily lead to rejection when empirical results are strong. This is a correctable typo, not a fundamental methodological flaw.
- **Harsh Critic's claim about rigid backbone approximation**: The paper does acknowledge this limitation in Section 5 ("we assume that the mutant backbone structure is identical to that of the wild-type protein"), so criticizing it as unacknowledged is unfair.
- **Harsh Critic's AUROC class balance concern**: This is a minor presentation issue, not a substantive weakness—the primary correlation metrics are appropriate for regression.
- **Strength Finder's generic claims**: Removed strengths like "this paper addressed an important problem" and "this paper targeted an interesting question" as they are generic and not specific to this paper's contributions.
- **Human Finder's similar weaknesses from other papers**: Removed weaknesses about missing related works (e.g., TopNetTree, GearBind) as I cannot verify their existence or relevance without external sources.

## Novel Insights
The paper's core insight—that explicitly modeling the unbound state in the thermodynamic cycle improves ΔΔG prediction—is genuinely novel and well-supported by the ablation study (Table 2). However, the "Boltzmann Alignment" framing is somewhat overstated given the learnable temperature parameter. The empirical contribution (SoTA results with proper unbound state modeling) is more substantial than the theoretical contribution (thermodynamic derivation with heuristic modifications).

## Suggestions
1. **Clarify baseline protocol immediately**: State explicitly whether baselines were re-run on the same 3-fold splits or cited from literature. If re-run, report the new numbers; if cited, acknowledge the split mismatch limitation.
2. **Correct Equation 12**: Change the sign to `+ β D_KL` to match the stated optimization goal, and add an ablation on β to demonstrate regularization strength effects.
3. **Temper physical claims**: Acknowledge that the learnable k_B T parameter acts as a scaling factor rather than a physical constant, and clarify that performance gains primarily stem from unbound state modeling rather than thermodynamic alignment per se.

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Decision | Comparison to this paper |
|-------|-----------|----------|-------------------------|
| o0Qfsq1fK8.md (ADiT) | 6.50 | Accept | Strong empirical results on binding affinity, some overclaim concerns—similar profile but ADiT has more architectural novelty |
| 1bJN1EQByS.md (WT-ASBS) | 6.50 | Accept | Physics-informed sampling with strong empirical results, clear methodology—this paper has comparable empirical strength |
| fvCHkWbdgX.md (ProTDyn) | 6.00 | Accept | Strong empirical results with some clarity issues—very similar profile to this paper |
| 0r1KU3dlps.md (Block Lewis) | 6.50 | Accept | Accepted despite confirmed formula typo—supports that Eq. 12 sign error is correctable |
| 50gXwRjWKl.md (TEL) | 5.50 | Reject | Physics-inspired with learnable temperature, but weaker empirical results—this paper has stronger empirical validation |
| Dp1RM3gPg8.md (RaftPPI) | 5.00 | Accept | Strong engineering contribution with incremental novelty—this paper has more methodological novelty |
| QNcrdCKNa5.md (TopoScorer) | 4.00 | Reject | Competitive but unclear SoTA claims, baseline issues—similar baseline ambiguity concern but weaker results |
| 16MJNrz0xV.md (EnerBridge-DPO) | 3.33 | Reject | Energy-guided inverse folding with ΔΔG prediction, but significant methodological concerns—this paper is cleaner |

**Scoring rationale:** This paper has genuinely strong empirical results (SoTA across all 7 metrics) and a novel methodological contribution (explicit unbound state modeling). The baseline comparison ambiguity is a significant concern but potentially addressable in rebuttal. The Eq. 12 sign error is a clear typo similar to accepted papers (0r1KU3dlps.md). Compared to calibration anchors, this paper's empirical strength and novelty are comparable to ProTDyn (6.00) and WT-ASBS (6.50), while the issues are less severe than rejected papers like TopoScorer (4.00) or EnerBridge-DPO (3.33). The paper warrants acceptance with revisions to clarify the baseline protocol and correct the formula typo.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>