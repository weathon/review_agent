Now I have a thorough understanding of the paper and the key issues. Let me write the consolidated review.

## Summary

The paper proposes Boltzmann Alignment, a technique that transfers knowledge from pre-trained inverse folding models (specifically ProteinMPNN) to ΔΔG prediction for protein-protein interactions. By deriving a connection from the Boltzmann distribution and thermodynamic cycle, the method explicitly incorporates both bound and unbound state likelihoods—unlike prior inverse folding-based methods that only use bound-state probabilities. The approach achieves state-of-the-art results on SKEMPI v2 (per-structure Spearman 0.3201 unsupervised, 0.5134 supervised), with additional demonstrations on binding energy prediction, docking, and antibody optimization.

## Strengths

- **Substantial empirical improvements across all metrics on SKEMPI v2**: Table 1 shows BA-DDG achieving per-structure Spearman of 0.5134 vs. prior supervised SoTA of 0.4324 (Surface-VQMAE), and BA-Cycle achieving 0.3201 vs. prior unsupervised SoTA of 0.2632 (RDE-Linear). Improvements are consistent across all 7 metrics, not just one.

- **Clean ablation validating the thermodynamic cycle**: Table 2 shows that incorporating the unbound state (BA-Cycle) improves per-structure Spearman from 0.2741 (ProteinMPNN alone) to 0.3201. This directly validates the core practical insight that accounting for the unbound state improves predictions.

- **Controlled comparison of supervision strategies**: Table 3 compares SFT, DPO, and Boltzmann supervision on the same base model (ProteinMPNN), isolating the contribution of the Boltzmann loss. Boltzmann supervision outperforms both SFT (0.5134 vs. 0.4769 per-structure Spearman) and DPO (0.5134 vs. 0.3913).

- **Robustness to predicted structures**: Table 4 shows that using AlphaFold3 predicted structures instead of crystal structures causes only marginal performance drops (overall Spearman 0.7821 vs. 0.7946), confirming practical applicability.

- **Appropriate evaluation protocol**: 3-fold cross-validation with structure-based splitting prevents data leakage, and per-structure metrics complement overall metrics for practical relevance.

## Weaknesses

### Fatal
None.

### Major

- **The sign of the KL divergence term in Eq. 12 is incorrect relative to the stated objective.** The loss is written as $\mathcal{L} = \|\Delta\Delta G - \widehat{\Delta\Delta G}\| - \beta D_{\text{KL}}(p_\theta \| p_{\text{ref}})$, and the text states the goal is to "minimize the discrepancy… while maintaining the distribution of the original pre-trained model." For a minimization objective, the KL penalty must have a *positive* sign ($+\beta D_{\text{KL}}$) to penalize deviation from the reference. With the negative sign as written, minimizing $\mathcal{L}$ *maximizes* the KL divergence—actively pushing the model away from the pre-trained distribution. This is the core method equation of the paper and must be correct. If this is a typo and the implementation uses the correct sign, it still undermines reproducibility and theoretical clarity; if implemented as written, the method works despite rather than because of the penalty, changing the interpretation.

- **The derivation from Eq. 1 to Eq. 2 conflates macroscopic state probabilities with point-evaluated probability densities, an approximation whose severity is understated.** $p_{\text{bind}}$ in Eq. 1 is the total probability of occupying the bound conformational basin—an integral over all bound conformations. In Eq. 2, this is replaced by $p(\mathcal{X}_{\text{bind}}|S_{AB})$, a probability density evaluated at a single conformation (the crystal structure). The paper acknowledges "we simplify and approximate $\mathcal{X}_{\text{bind}}$ and $\mathcal{X}_{\text{unbind}}$ as the backbone structure," but this language understates what is happening: the ratio of integrated partition functions that gives the true binding free energy is not equivalent to the ratio of point-evaluated probability densities. This substitution implicitly assumes that conformational entropy contributions of the bound and unbound basins cancel, which is a strong and unacknowledged approximation. The paper frames its contribution as providing rigorous "insight into the high correlation previously observed between binding energy and log-likelihood," but this insight depends on the rigor of the connection, which is compromised at this step. The practical method still works, but the theoretical claim is overstrong.

### Minor

- **Application demonstrations have small sample sizes that don't support the "broader applicability" framing**: The antibody optimization (Table 5) evaluates only 5 favorable single-point mutations; the docking experiment (Fig. 4, right) covers only 13 antibody complex generation tasks. These are suggestive but insufficient to substantiate claims about "broader applicability" or that "fine-tuning for ΔΔG prediction can enhance the design of more effective antibody sequences" as a general principle.

- **No standard deviations reported across the 3 folds**: Given SKEMPI v2 has only 348 complexes and 3-fold cross-validation, fold-to-fold variability could be meaningful. Without variance, it is difficult to assess whether improvements over prior work are statistically significant.

- **Missing ESM-IF + cycle ablation**: Table 2 compares BA-Cycle (ProteinMPNN + cycle) against ESM-IF and ProteinMPNN alone, but does not show ESM-IF + cycle. This makes it impossible to determine whether the improvement comes from the thermodynamic cycle insight or from ProteinMPNN being a better base model than ESM-IF for this purpose.

- **DPO's dramatic underperformance is unexplained**: Table 3 shows DPO achieving per-structure Spearman of 0.3913 vs. SFT's 0.4769 and Boltzmann's 0.5134. Since one of the paper's claims is that physics-informed alignment outperforms generic alignment, understanding why DPO fails so badly would strengthen this claim—is it data scale, noise, or incompatibility with log-likelihood computation?

- **The unbound state approximation (Eq. 9) uses complex-derived backbone coordinates for individual chains**: The paper approximates $p(S_{AB}|\mathcal{X}_{\text{unbnd}}) \approx p_\theta(S_A|\mathcal{X}_A) \cdot p_\theta(S_B|\mathcal{X}_B)$, where $\mathcal{X}_A$ and $\mathcal{X}_B$ are extracted from the complex structure. Real unbound structures often have different backbone conformations. The paper acknowledges this as a limitation in the discussion, but does not quantify its impact.

- **Learnable $k_BT$ disconnects predictions from physical units**: In the physical derivation, $k_BT$ is a fixed constant; making it learnable effectively introduces a scaling parameter with no physical meaning. The paper does not discuss this tension explicitly.

### Trivial
None.

## Nice-to-Haves

- Per-mutation analysis correlating the magnitude of the unbound-state correction with mutation location (interface vs. non-interface) would reveal whether the method's gains come from physically meaningful corrections.
- Evaluation on an independent ΔΔG benchmark beyond SKEMPI v2 would strengthen generalizability claims.
- Per-structure scatter plots showing when BA-Cycle corrects ProteinMPNN would provide interpretable case studies.

## Removed Points

These points are flagged to be removed, treated with caution:

- **"Not yet released / cannot be independently verified" concerns about AlphaFold3 or other cited tools**: The paper cites AlphaFold3 and other tools as existing; per review rules, cited entities are assumed to exist.
- **Formatting/style nitpicks**: Removed per rules. The parser artifacts are not paper errors.
- **Missing related works**: Removed per rules—cannot confirm their existence without external sources.
- **Reproducibility concerns about undisclosed hyperparameters or implementation details**: Removed as nitpicks about reproducibility per rules.
- **Criticism about lacking a comprehensive explanation for canceling $p(\mathcal{X}_{\text{bnd}})$ and $p(\mathcal{X}_{\text{unbnd}})$ in the ΔG case**: The paper honestly acknowledges this limitation (Section 4.4.1: "we lack a comprehensive explanation for canceling out $p(\mathcal{X}_{\text{bnd}})$ and $p(\mathcal{X}_{\text{unbnd}})$"). While a valid observation, the paper is transparent about it and ΔG prediction is a secondary application, not the core claim.
- **Strength claim about "principled theoretical derivation connecting thermodynamics to inverse folding likelihoods"**: This strength from the Strength Finder conflicts with the verified Major weakness about the Eq. 1→2 conflation. Moved here because the theoretical derivation is not as rigorous as claimed, undermining this as a strength.
- **Generic weaknesses about requesting larger datasets or more models**: The SKEMPI v2 benchmark is the standard in the field, and the model comparison is already comprehensive.
- **Criticism about the rigid backbone assumption for multi-point mutations**: The paper explicitly acknowledges this limitation in the Discussion section (line 274). While valid to note, it is already addressed by the authors.

## Novel Insights

The most interesting observation is that the paper's theoretical and practical contributions are somewhat decoupled: the practical finding—that incorporating the unbound state into the thermodynamic cycle improves inverse folding-based ΔΔG prediction—is cleanly validated by the BA-Cycle ablation (Table 2) regardless of the rigor of the derivation. However, the theoretical contribution (explaining *why* inverse folding log-likelihoods correlate with binding energy) depends on a derivation with a significant unacknowledged approximation at the Eq. 1→2 step. This creates an asymmetry: the method is empirically sound, but the theoretical insight that frames the contribution is less rigorous than presented. The field would benefit from an honest assessment of where the physics-informed derivation is exact versus approximate, as this would better guide future work on whether the Boltzmann framework can be tightened or whether the practical gains come from a different mechanism.

## Suggestions

- **Correct the sign of the KL penalty in Eq. 12** to $+\beta D_{\text{KL}}$ (or clarify if the formulation is intentionally a maximization converted to minimization and show the conversion explicitly). This is the single most important revision.
- **Explicitly acknowledge the probability vs. probability density approximation** at the Eq. 1→2 step and discuss its physical implications (conformational entropy cancellation), rather than framing it as "simplifying and approximating the backbone structure."
- **Soften the claim** about providing "insight into the high correlation previously observed between binding energy and log-likelihood" to reflect that the derivation provides a qualitative motivation rather than a rigorous proof.
- **Report standard deviations** across folds in the main results table.
- **Add ESM-IF + cycle** to Table 2 to disentangle the cycle contribution from the base model choice.

## Evaluation

**Originality**: The thermodynamic cycle insight for incorporating the unbound state in inverse folding-based ΔΔG prediction is novel and well-motivated. The Boltzmann supervision loss is a reasonable extension of physics-informed alignment, though the KL sign error undermines confidence in the formulation.

**Importance of research question**: ΔΔG prediction is an important problem in protein engineering and drug design. The scarcity of labeled data makes transfer from pre-trained models highly relevant.

**Claims support**: The empirical claims are well-supported by strong results on SKEMPI v2. The theoretical claims are partially undermined by the approximation at Eq. 1→2 and the sign error in Eq. 12.

**Soundness of experiments**: The main experiments are sound—3-fold cross-validation with structure-based splitting, comprehensive baselines, and controlled ablations. The application demonstrations are too small to support the broad claims made.

**Clarity**: The paper is generally well-written and the method is clearly described. The theoretical derivation is easy to follow, though the severity of its approximations is understated.

**Value to community**: The practical method and the code release provide value. The thermodynamic cycle insight is actionable even if the theoretical framework needs refinement.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LED-GFN (GFlowNets/Boltzmann) | /home/wg25r/review_agent/human_reviews/P15CHILQlg.md | 8.0 | Much stronger theoretical rigor than BA-DDG; BA-DDG has weaker theory |
| Light-DDG | /home/wg25r/review_agent/human_reviews/IxmWIkcKs5.md | 6.75 | Same domain (ΔΔG/SKEMPI), similar empirical strength, cleaner methodology; BA-DDG has novel theoretical angle but with issues |
| P-Align | /home/wg25r/review_agent/human_reviews/AgTSjXh7vl.md | 4.25 | Also physics-informed alignment with empirical gains but weak theory; BA-DDG is much stronger empirically |
| Refine-PPI | /home/wg25r/review_agent/human_reviews/jPrKs5rOWw.md | 3.5 | Same ΔΔG/SKEMPI domain but with fundamental clarity and scope issues; BA-DDG is clearly stronger |
| Energy landscape global minima | /home/wg25r/review_agent/human_reviews/OcTUquFXfx.md | 2.6 | Weak theory + limited experiments; BA-DDG is far above this |

BA-DDG sits between Light-DDG (6.75, clean methodology, strong empirical) and P-Align (4.25, weak theory, moderate empirical). BA-DDG has stronger empirical results than both and a genuine practical insight, but the theoretical framework has two substantive issues (the Eq. 1→2 approximation and the Eq. 12 sign error). The empirical contribution alone—demonstrating that the unbound state matters and achieving new SoTA—would merit a solid score, but the theoretical overclaim and equation error are significant enough to pull it below Light-DDG.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>