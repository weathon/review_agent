Now I have sufficient information to write the final review. Let me compile everything.

## Summary

This paper investigates whether the well-known counterintuitive likelihood phenomenon in image-domain anomaly detection (where OOD data receives higher likelihood than in-distribution data) also occurs in tabular data. The authors propose a domain-agnostic definition of the counterintuitive phenomenon based on relative model performance (Definition 3.3), conduct extensive experiments across all 47 ADBench tabular datasets and 10 embedding datasets with 12 baselines, and provide theoretical and empirical analysis linking the phenomenon's rarity in tabular data to lower dimensionality and weaker feature correlation compared to images.

## Strengths

- **Comprehensive, unbiased benchmarking on all 47 ADBench tabular datasets and 10 embedding datasets** against 12 baselines, motivated by the selection-bias critique of Shwartz-Ziv & Armon (2022). NF-SLT achieves the best average AUROC (0.8575), best average rank (3.43), highest Top2 Ratio (0.45), and lowest Fail Ratio (0.02) among 13 models (Table 1). This is a genuine empirical contribution that many prior works on this topic lack.

- **The intrinsic dimension (d Ratio) analysis (Section 5.2, Table 4, Figure 1)** provides a useful quantitative comparison of feature correlation structure between image and tabular domains. The finding that image datasets have d Ratios of ~1–3% while tabular datasets range from ~0.39–0.81, and that NF-SLT tends to underperform on tabular datasets with lower d Ratios, is clearly presented and offers a concrete explanation for the domain difference.

- **The dimensionality-reduction experiment (Table 2)** provides clean evidence that AUROC improves as dimensionality decreases under the independence assumption, directly supporting the theoretical claim that higher dimensionality exacerbates the counterintuitive phenomenon.

- **The embedding analysis** (estimating intrinsic dimensions of CIFAR-10 and SVHN embeddings as 23 and 18 vs. ambient 1000, yielding higher d Ratios than raw pixels) connects the correlation story to practical settings and is consistent with Kirichenko et al. (2020)'s findings about semantic embeddings alleviating the phenomenon.

- **The bilinear interpolation finding (Table 3)** — that resizing images can push AUROC above 0.5 for challenging pairs (e.g., CelebA/SVHN from 0.1207 to 0.7037) — is a practically interesting and somewhat surprising result.

## Weaknesses

### Fatal
None.

### Major

- **Definition 3.3 measures relative model performance, not likelihood inversion.** The original counterintuitive phenomenon (Nalisnick et al., 2019a) is that the *expected* likelihood of OOD data exceeds that of in-distribution data — a specific property of the learned density. Definition 3.3 replaces this with a criterion about whether comparison models outperform the generative model by a sufficient margin. These are fundamentally different: likelihood inversion can occur without poor relative performance (if overlap is moderate, AUROC may still be reasonable), and poor relative performance can occur without likelihood inversion (e.g., the likelihood ordering is correct but not discriminative enough). The paper's argument against the "naive" definition — that "the argument would consider any result outside 100% AUROC as counterintuitive" (Section 1) — is based on a strawman interpretation of likelihood inversion as "any overlap in likelihood distributions." The original phenomenon refers to the *expected* log-likelihood ordering being inverted, which is a much stronger condition that typically manifests as AUROC well below 50%. Consequently, the paper's central claim that "the counterintuitive phenomenon is rare in tabular data" is supported only under its redefined (and different) notion. What Table 1 actually demonstrates is that NF-SLT is competitive on tabular data — a valuable but weaker claim.

- **Theorem 5.4 assumes P and Q are product distributions (independent features), which severely limits its applicability.** Real tabular data — indeed, any data where anomaly detection is meaningful — typically violates this assumption, as feature dependencies are precisely what make anomaly detection non-trivial. The supporting dimensionality experiment (Table 2) applies ICA before the flow, which enforces independence and thus directly matches the theorem's assumption, making the experiment partially circular as a test of the theory's real-world relevance. The paper acknowledges that Table 3's raw-image results "conflict with the theorems" due to pixel correlation, but does not fully grapple with the implication that the independence assumption — rather than dimensionality per se — may be the primary driver of the theorem's predictions. No analysis addresses how conclusions change under realistic dependence structures, even for simple cases (e.g., a Gaussian model with structured covariance).

### Minor

- **Only NICE is used for the main NF-SLT benchmark results (Table 1).** NICE is a volume-preserving flow with a constant Jacobian determinant — among the least expressive normalizing flow architectures. The counterintuitive phenomenon in images was observed with more expressive flows (Glow, RealNVP). While other architectures appear in Appendix G and in Tables 2–3, integrating them into the main benchmark analysis would strengthen the claim that the phenomenon is "rare" regardless of flow architecture.

- **The feature correlation analysis is correlational rather than causal.** Table 4 shows that datasets where NF-SLT ranks below 3 tend to have lower d Ratios, but this could be confounded by other factors (e.g., these datasets may simply be harder for all methods). No controlled experiment varies correlation while holding other factors constant, even though the paper itself introduces the AR(1) covariance model (Equation 5) that would enable such an experiment.

- **The entropy condition H(P) − H(Q) > DKL(Q||P) in Theorem 5.4 is an unverified assumption.** Whether normal data has higher entropy than anomalous data is dataset-dependent. The theorem's conclusion is conditional on this assumption, but the paper presents it as a general explanation without empirical verification across the 47 datasets.

- **The paper dismisses near-random NF-SLT performance on the imdb dataset (AUROC 0.5013)** because "the difference in performance with the comparison model is very small" and thus doesn't satisfy Definition 3.3's second condition. However, an AUROC of 0.5013 indicates that the likelihood-based test is essentially non-informative on this dataset, which is precisely the kind of failure worth investigating — not dismissing through a definitional technicality.

- **Different preprocessing for different models** (NeuTraLAD excluded from RobustScaler due to performance decrease) introduces a potential fairness concern in the comparison, as preprocessing can significantly affect anomaly detection performance.

### Trivial
None.

## Nice-to-Haves

- A direct visualization of log-likelihood distributions for normal vs. anomalous data on representative tabular datasets would show whether likelihood inversion occurs in its original form, independent of Definition 3.3.
- A controlled synthetic experiment varying correlation (using the AR(1) model from Equation 5) while holding dimension and anomaly type constant would provide causal evidence for the correlation claim.
- Specifying and justifying the β and γ thresholds (currently in Appendix B) in the main text would allow readers to assess what "rare" means quantitatively without consulting the appendix.
- Running the full 47-dataset benchmark with more expressive flows (RealNVP, Glow) as the primary NF-SLT would verify the main conclusion beyond volume-preserving flows.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Missing appendix/proofs" complaints**: The harsh critic notes that β and γ values and the rigorous formulation of Definition 3.3 are in Appendix B, and proofs are in Appendix D. Per rules, missing appendix content is a parser artifact — the original submission includes these sections.

- **Overclaimed practical reliability**: The strength finder claims NF-SLT "offers a practical and reliable approach for anomaly detection in tabular domains." The harsh critic notes average rank 3.43 is not dominance. This is partially valid — the claim could be softened to "competitive" — but the empirical results do show NF-SLT outperforming all other methods on aggregate metrics. This is already captured in the Major weakness about Definition 3.3 overclaiming.

- **Formatting/style nitpicks**: The harsh critic's concern about "the bottom portion of Table 1" formatting is a presentation issue and removed per rules.

- **Dismissing high-dimensional tabular datasets as "atypical"**: The harsh critic argues that excluding genomics datasets as "very different characteristics from typical tabular datasets" is post hoc. This is partially valid but the paper does address these in Appendix C.4, and defining scope boundaries is a reasonable authorial choice. Weakened to minor significance.

- **Single hyperparameter setting as a major concern**: The paper selects "the hyperparameter combination with the highest average AUROC for all datasets." While non-standard, this applies the same protocol to all methods equally, and sensitivity analysis is reported. This is a minor rather than major concern.

## Novel Insights

The most novel observation in this paper is the connection between intrinsic dimension (d Ratio) and the domain-dependence of the counterintuitive phenomenon. While prior work (Kirichenko et al., 2020; Serrà et al., 2020) discussed feature correlation qualitatively in the image domain, this paper provides the first quantitative comparison showing that tabular datasets consistently have d Ratios an order of magnitude higher than image datasets (~0.39–0.81 vs. ~0.002–0.019), and that this metric also predicts within-domain variation in NF-SLT performance. The bilinear interpolation finding — that simple image resizing can partially mitigate likelihood inversion by strengthening pixel correlations and reducing entropy — is an unexpected and potentially practical insight that hasn't been noted in prior work.

## Suggestions

- Rename or qualify the central claim: state that "practical failure of likelihood-based anomaly detection is rare in tabular data" rather than "the counterintuitive phenomenon is rare," since the two are not equivalent under the paper's own definition.
- Add a direct measurement of likelihood distributions (histograms of log p(x) for normal vs. anomalous samples) on a few representative tabular datasets to verify whether actual likelihood inversion occurs, even if it doesn't lead to practical failure.
- Relax the independence assumption even partially — e.g., a theoretical result for Gaussian P, Q with structured covariance, or an experiment using synthetic data with known correlation structure from the AR(1) model already defined in Equation 5.
- Report results with at least one more expressive flow architecture (RealNVP or Glow) on the main 47-dataset benchmark, not just in the appendix, to verify that the finding is not an artifact of NICE's limited expressiveness.

## Evaluation

**Originality**: The question of whether the likelihood inversion phenomenon extends to tabular data is important and underexplored. The d Ratio analysis is a novel quantitative contribution. Definition 3.3 is a new operationalization but measures something different from the original phenomenon.

**Importance of research question**: High — likelihood-based anomaly detection with normalizing flows is widely used, and understanding when it fails is practically important.

**Claims support**: The empirical claim that NF-SLT is competitive on tabular data is well-supported. The definitional claim that "the counterintuitive phenomenon is rare" is overclaimed relative to what Definition 3.3 measures. The theoretical explanation is limited by the independence assumption.

**Soundness of experiments**: The benchmarking is thorough and unbiased. The dimensionality and correlation experiments are well-designed within their assumptions but have the circularity concern (ICA enforcing independence) and are correlational (d Ratio analysis).

**Clarity**: Generally clear, though the definitional argument in Section 1 is somewhat muddled (conflating "any overlap" with "expected likelihood inversion").

**Value to community**: The comprehensive benchmark and the finding that NF-SLT works well on tabular data are valuable. The theoretical contribution is limited by its assumptions.

## Calibration

Anchors used for scoring:
- **High (avg > 7)**: Deep Orthogonal Hypersphere Compression for AD (avg 8.0, Accept spotlight) — novel method, strong theory, clean experiments. The paper under review is clearly below this.
- **Medium (avg 4–6)**: GNN spectral benchmark (avg 5.25, Accept Poster) — comprehensive benchmark with overclaimed theoretical links; Re-evaluating graph classification benchmarks (avg 6.0, Reject) — extensive benchmark, new metric, limited theoretical applicability; Re-evaluating unseen-class impact in SSL (avg 6.0, Accept Poster) — redefines evaluation methodology with comprehensive experiments. The paper under review is comparable to these: strong empirical contribution, weaker theoretical foundation, overclaimed scope.
- **Low (avg < 3)**: Pan for Gold (avg 2.2, Withdrawn) — grand paradigm claims with vague definitions; NF homogeneous mapping (avg 4.25, Reject) — addresses likelihood inversion but limited scope. The paper under review is clearly above these.

The paper under review sits in the 5–6 range: its comprehensive benchmarking and practical finding are genuine contributions, but the definitional mismatch and theoretical limitations prevent it from scoring higher. It is slightly below the re-evaluating SSL paper (which had a clearer methodological contribution) and comparable to the GNN spectral benchmark.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>