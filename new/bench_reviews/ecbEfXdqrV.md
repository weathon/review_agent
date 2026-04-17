Now I have thoroughly read the paper and compared against calibration examples. Let me write the final consolidated review.

## Summary

This paper investigates whether the "counterintuitive phenomenon"—where out-of-distribution samples receive higher likelihood than in-distribution samples under normalizing flows—also occurs in tabular anomaly detection. The authors propose a domain-agnostic formal definition (Definition 3.3) based on relative AUROC performance against comparison models, conduct extensive experiments across 47 tabular and 10 embedding datasets from ADBench against 12 baselines, and provide theoretical (Theorem 5.4, Corollary 5.6) and empirical (intrinsic dimension ratio analysis) explanations linking the phenomenon's rarity in tabular data to lower dimensionality and weaker feature correlations.

## Strengths

- **Important and timely research question.** Whether the well-documented likelihood inversion pathology in image flows extends to tabular data is practically significant for the anomaly detection community. The paper systematizes a question that has been discussed informally (e.g., Kirichenko et al., 2020 showed limited evidence on two tabular datasets) but never rigorously evaluated at scale.

- **Comprehensive benchmarking without selection bias.** Using all 47 tabular and 10 embedding datasets from ADBench—rather than cherry-picking favorable datasets—is a commendable design choice. The 12 baseline models (6 shallow, 6 deep) provide a reasonable comparison pool, and NF-SLT achieving the lowest fail ratio (0.02) and best average AUROC (0.8575) is a clear empirical signal.

- **Principled attempt at formalization.** Definition 3.3 attempts to address a real gap: the literature has no precise, domain-agnostic definition of "counterintuitive phenomenon." The two-condition formulation (proportion of outperforming models exceeds β AND minimum gap exceeds γ) avoids the trivial "any AUROC < 100% is counterintuitive" interpretation and the CIFAR-10 vs. SVHN example demonstrates it recovers the known pathological case.

- **Useful empirical observation for practitioners.** That simple flow-based likelihood scoring is competitive (often best) for tabular AD benchmarks is a valuable finding that contrasts with the image-domain narrative and provides a principled, easy-to-implement baseline.

- **Dimensionality and correlation analysis provides interpretable intuition.** The d Ratio framework, the synthetic Gaussian experiments linking correlation to intrinsic dimension, and the comparison between image (~1% d Ratio) and tabular (much higher) datasets offer plausible explanatory narratives, even if not rigorously causal.

## Weaknesses

### Major

- **Definition 3.3 reframes "counterintuitive phenomenon" as a relative benchmarking property, creating a conceptual gap from the original likelihood-inversion notion.** The image literature's phenomenon is a distribution-level observation: OOD samples receive systematically higher likelihoods than ID samples, verifiable without any comparison model. Definition 3.3 instead defines the phenomenon as occurring when (1) most comparison models outperform the flow AND (2) the performance gap is large. This makes the definition sensitive to the choice of baseline pool, their hyperparameters/tuning budgets, and the unspecified thresholds β and γ. While the paper argues that any-AUROC-<1 is too inclusive, the result is that the "negative finding" (phenomenon is rare) is primarily about NF-SLT's benchmark rank, not about whether likelihood inversions occur in tabular data. The paper should more clearly acknowledge this definitional shift and its implications for interpreting the results. (See paper lines 61–70 for Definition 3.3 and the CIFAR/SVHN validation.)

- **The key thresholds β and γ are not specified in the main text.** These parameters control when a "counterintuitive phenomenon" is declared to occur, yet readers cannot assess how stringent or lenient the criterion is. The "fully rigorous formulation" is relegated to an appendix. Without stated values or sensitivity analysis, the claim that the phenomenon is "rare" (fail ratio 0.02) cannot be independently verified, and the concern that thresholds were chosen post-hoc to support the conclusion cannot be dismissed. This directly affects the paper's central claim.

- **No direct analysis of likelihood distributions.** The original phenomenon is formulated in terms of likelihood overlap/inversion between normal and anomalous samples. This paper never shows likelihood histograms, mean log-likelihoods, or per-dataset distribution comparisons on tabular data. All conclusions are mediated by AUROC and relative rank. A reader could reasonably expect the paper to test whether anomalies actually receive higher likelihoods than normal samples in the tabular datasets where NF-SLT underperforms, to verify whether the failure mode aligns with the likelihood-inversion mechanism. Without this, the paper shows NF-SLT is competitive on AUROC but cannot confirm that the underlying likelihood behavior is fundamentally different from the image case.

### Minor

- **Single flow architecture (NICE) in main experiments.** NICE is a volume-preserving flow with constant Jacobian determinant, which means log-likelihood is driven almost entirely by the latent term—a specific architectural property that may inherently avoid the volume-term issues contributing to likelihood inversion in images. Results with other flow families (RealNVP, Neural Spline) are mentioned in Appendix G but not discussed in the main text. The claim "likelihood-only detection with normalizing flows offers a practical and reliable approach" (Abstract, Conclusion) is too broad for the narrow architectural scope of the main experiments.

- **Theoretical results rely on strong independence assumptions.** Theorem 5.4 assumes P and Q are product measures (independent coordinates), which does not hold for real tabular data with correlated features. The paper acknowledges this for images (Table 3 footnote) but the same limitation applies to tabular data. The theorem provides conceptual insight but the link from "independent coordinates + entropy conditions" to "flows on real tabular data with correlated, heterogeneous features" is asserted rather than established. The value is primarily as qualitative guidance rather than quantitative prediction.

- **Uniform hyperparameter selection across all datasets.** Selecting a single hyperparameter combination maximizing average AUROC across all 47 datasets is an unusual protocol that could disadvantage models whose optimal settings differ across datasets. No ablation comparing uniform vs. per-dataset tuning is provided, making it unclear whether NF-SLT's advantage persists under alternative evaluation protocols.

- **d Ratio analysis is observational and one-directional.** The paper shows that among the 25 datasets where NF-SLT ranks ≥3, most have low d Ratio. But no converse analysis is presented—do high-d-Ratio datasets systematically see better NF-SLT performance? The ID estimators (MLE, TwoNN) are acknowledged to underestimate at high true dimension, yet absolute values and trends are interpreted fairly strongly. The causal link from correlation → low ID → likelihood behavior is plausible but not rigorously established.

### Trivial

- No standard deviations or confidence intervals reported despite 10 repeated runs, which limits assessment of statistical significance for small AUROC gaps (e.g., the 0.02 gap cited for the "yeast" dataset).

## Nice-to-Haves

- Apply Definition 3.3 to standard image OOD benchmarks using the same evaluation protocol to verify the definition recovers well-documented counterintuitive cases beyond CIFAR-10/SVHN.
- Provide likelihood histograms for normal vs. anomalous samples on representative tabular datasets, particularly those where NF-SLT underperforms.
- Test on high-dimensional tabular datasets (genomics, text BoW) that the paper acknowledges as exceptions but does not evaluate—these are precisely the cases where the theory predicts the phenomenon should emerge.
- Include results with additional flow architectures (RealNVP, Neural Spline) in the main text rather than only in appendices.

## Novel Insights

The paper's most interesting insight is the empirical observation that the tabular domain effectively acts as a "natural intervention" on the dimension/correlation conditions that drive likelihood inversion in images. By quantifying this via the d Ratio, the paper provides a unifying framework that also explains why CV/NLP embeddings (with higher d Ratios than raw pixels) are less susceptible—a finding consistent with Kirichenko et al. (2020) but now connected to an explicit dimensional analysis. However, the insight remains tentative because the d Ratio–performance relationship is only demonstrated in one direction (low d Ratio associated with underperformance) and is not tested with controlled interventions on actual tabular data.

## Suggestions

- Report the values of β and γ used in the evaluation, and provide a sensitivity analysis showing how the "rare" conclusion changes across reasonable threshold choices.
- Add per-dataset reporting of whether Definition 3.3 is satisfied, with the specific proportion of outperforming models and minimum AUROC gap for each dataset, enabling readers to verify the "rare" claim at the individual-dataset level.
- Include likelihood distribution visualizations (histograms or density plots) for at least a few representative tabular datasets (both where NF-SLT succeeds and where it underperforms), to bridge the gap between the AUROC-based definition and the likelihood-inversion phenomenon as understood in prior work.

## Evaluation

**Originality:** The formalization attempt (Definition 3.3) is novel though its design choices create the issues noted above. The systematic empirical evaluation of NF-based AD on ADBench is new and valuable. The theoretical extension of Caterini & Loaiza-Ganem to incorporate dimension effects is incremental but meaningful.

**Importance:** The question is practically significant. Showing that NF-SLT is competitive for tabular AD addresses a misconception that likelihood-based detection is fundamentally flawed.

**Claims support:** The core claim that the "counterintuitive phenomenon is rare" is weakened by the definitional choices and lack of direct likelihood analysis. The claim that NF-SLT is "practical and reliable" is better supported empirically but overclaimed in light of the single-architecture limitation and the missing β/γ specification.

**Experiments:** Broad and well-designed in terms of dataset coverage, but the evaluation protocol (uniform hyperparameters, no variance reporting) and the indirect evidence (AUROC without likelihood analysis) limit the strength of conclusions.

**Clarity:** The paper is generally clear, but the critical definitional shift from likelihood-inversion to relative-AUROC could be made much more explicit, and the β/γ omission is an unnecessary transparency gap.

**Community value:** The empirical finding that NF-SLT works well for tabular AD is useful for practitioners, and the dimensional/correlation analysis provides interpretable intuition even if not rigorously established.

### Score Calibration

I compared against:
- **ReTabAD** (Accept Poster, scores 8/4/6/4, avg ~5.5): Comprehensive tabular AD benchmark with strong dataset curation. Our paper is weaker in the precision of its formalization and less novel in the benchmark contribution (uses existing ADBench rather than curating new resources), but makes a distinctive analytical contribution.
- **Likelihood Paradox Mitigation** (Reject, scores 2/2/8): Related NF/OOD work with theoretical analysis but limited novelty. Our paper has broader empirical scope but similar concerns about the gap between theory and practice.
- **NRDE** (Reject, scores 2/4/4): Tabular AD with normalizing flows. Our paper provides more comprehensive benchmarking but introduces definitional issues not present in NRDE.
- **Rethinking Diffusion Model in High Dimension** (Reject, scores 0/2/0/2): Fundamental issues with theoretical claims unsupported by evidence. Our paper is substantially better than this—the empirical work is genuine and the definition, while imperfect, is defensible.
- **AdaSCALE** (Reject, scores 4/6/8/2/4/4): OOD detection with strong empirical results but concerns about scope and novelty. Our paper's conceptual contribution is more distinctive but suffers from the definitional concerns.

Our paper sits between the clearly rejected papers (which had either fundamental unsupported claims or too-narrow contributions) and the borderline-accept benchmarking papers. The definitional issues and the gap between the paper's claims and what the evidence actually supports are significant but not fatal—the paper does make a genuine empirical contribution and the observation that NF-SLT works well for tabular AD is valuable. However, the overclaiming (from "NF-SLT is competitive" to "counterintuitive phenomenon is rare" to "practical and reliable approach") and the missing β/γ specification push it below the accept threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>