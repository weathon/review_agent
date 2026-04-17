Now I have a comprehensive understanding of the paper. Let me write my final consolidated review.

## Summary

This paper investigates why the well-known "counterintuitive phenomenon"—where generative models assign higher likelihood to OOD/anomalous data than in-distribution data, as commonly observed in image domains—rarely occurs in tabular anomaly detection. The authors propose a domain-agnostic definition of this phenomenon based on relative model performance (Definition 3.3), conduct experiments across all 47 tabular and 10 embedding datasets from ADBench benchmarked against 12 baselines, and provide theoretical and empirical analyses attributing the phenomenon's absence to lower dimensionality and weaker feature correlations in tabular data.

## Strengths

- **Comprehensive, unbiased benchmarking.** Using all 47 tabular + 10 embedding datasets from ADBench without cherry-picking, compared against 12 baselines (6 shallow, 6 deep), is a genuine empirical strength. NF-SLT's best average AUROC (0.8575), best average rank (3.43), lowest fail ratio (0.02), and highest top-2 ratio (0.45) form a clear and compelling result.

- **Tackles an important and underexplored question.** Whether the likelihood paradox observed in image domains occurs in tabular anomaly detection is a genuinely interesting question with practical implications. The finding that a simple likelihood test with a normalizing flow is actually quite competitive on tabular data is a useful and somewhat surprising empirical observation.

- **Multi-faceted explanatory approach.** Connecting the empirical results to both dimensionality (Theorem 5.4) and feature correlation (d-Ratio analysis) provides complementary perspectives for understanding *why* tabular data behaves differently. The ICA experiments (Table 2) and resize experiments (Table 3) offer controlled empirical support for the dimensional perspective.

- **Reasonable formalization effort.** While the specific definition has issues (discussed below), the paper correctly identifies that the prior informal notion of "counterintuitive phenomenon" (any case where likelihood inversion occurs) is too inclusive—trivially, any model that doesn't achieve perfect AUROC could be deemed "counterintuitive." Attempting to formalize a more meaningful definition is a worthwhile contribution.

## Weaknesses

### Major

- **Definition 3.3 conflates "likelihood inversion" with "poor relative performance," creating a conceptual mismatch with the motivating literature.** The original "counterintuitive phenomenon" in Nalisnick et al. (2019) and Kirichenko et al. (2020) concerns likelihood *inversion*—OOD data receiving higher likelihood than ID data—and near-random AUROC despite a trained model. Definition 3.3 replaces this with: (i) most baselines outperform NF-SLT, AND (ii) the minimum gap exceeds γ. A model can have severe likelihood inversion yet not be deemed "counterintuitive" if baselines are also weak or the gap is below γ. Conversely, a model with no likelihood inversion but slightly worse than all baselines could be labeled "counterintuitive." The paper's abstract claims this "rarely occurs in tabular data," but this conclusion depends on the specific definition chosen—a definition that measures something fundamentally different from what prior work measured. The thresholds β and γ are not specified in the main text, and no sensitivity analysis is provided, making it impossible to assess whether the "rarity" conclusion is robust to these choices.

- **Theorem 5.4 requires independent coordinates (P = ∏pᵢ, Q = ∏qᵢ), which contradicts the paper's own thesis about feature correlation.** The paper's central theoretical result shows that the likelihood gap's lower bound decreases linearly with dimension—*under the assumption that features are independent*. Yet Section 5.2 argues that tabular data has *weaker* feature correlation, making it more "independent-like." This creates a circularity: the theorem's assumptions are most plausible precisely where the paper argues they are most relevant, but real tabular data still has correlated features. The paper applies the theorem verbally to "general tabular data" without empirically verifying the independence or entropy conditions (H(P) − H(Q) > D_KL(Q||P)) on actual datasets. The paper itself acknowledges (line 135) that Table 3's resizing experiments "cannot be applied" under the theorem's assumptions, yet still draws qualitative conclusions about dimensionality's role from these results.

- **Intrinsic dimension / d-Ratio analysis is correlational and conditionally reported.** Table 4 (bottom) reports d-Ratio thresholds only for the 25 datasets where NF-SLT ranks ≥ 3, not across all datasets. No statistical test of association between d-Ratio and NF-SLT performance is provided. Without showing how d-Ratio relates to performance on successful datasets, the claim that "heterogeneous features" drive NF-SLT's success remains suggestive rather than demonstrated. Furthermore, intrinsic dimension estimators (MLE, TwoNN) are known to be noisy and biased (the paper acknowledges they are "lower bounds"), yet these point estimates are used directly as quantitative evidence without uncertainty quantification.

### Minor

- **Single hyperparameter configuration selected globally.** The paper selects one hyperparameter combination per model based on the highest *average* AUROC across all datasets. This may advantage models whose performance is more stable across datasets (like NF-SLT) and disadvantage models that benefit from per-dataset tuning. While this is a defensible protocol for a standardized benchmark, it limits the strength of claims about NF-SLT's superiority.

- **Only NICE architecture for main results.** The primary experiments use NICE (a volume-preserving flow with constant Jacobian determinant). Other architectures appear only in Appendix G. Since NICE's fixed Jacobian determinant means the likelihood is determined solely by the latent term, the results may not generalize to more expressive flows. This matters because the volume term's interaction with dimensionality is precisely part of the theoretical story.

- **The Corollary 5.6 moment assumption (O(d^k) scaling) is stated without verification.** The assumption that the n-th absolute central moment of the log-likelihood difference scales as O(d^k) for k < n drives the AUROC upper bound result, but no empirical or theoretical justification is provided beyond the statement.

- **The paper's claim about the 'imdb' dataset undermines its own definition.** On 'imdb', NF-SLT achieves AUROC 0.5013—essentially random guessing. The paper dismisses this because the gap to other models is small (γ condition not met). But random-guessing performance is *exactly* what the original "counterintuitive phenomenon" literature is about: the model is no better than chance. The definition's second condition causes the paper to not recognize a clear instance of the phenomenon it seeks to study.

### Trivial

- The abstract and introduction conflate the original "likelihood inversion" notion with the new relative-AUROC definition, which can mislead readers familiar with the prior literature.

## Nice-to-Haves

- Report β and γ values used, and conduct sensitivity analysis showing how the proportion of datasets exhibiting the "counterintuitive phenomenon" varies across threshold choices.
- Include additional flow architectures (RealNVP, Glow) in the main results table, not just in an appendix.
- Add per-dataset scatter plots of d-Ratio vs. NF-SLT AUROC across all 57 datasets (not just the 25 failure cases), with statistical tests of association.
- Verify the entropy/entropy-gap condition H(P) − H(Q) > D_KL(Q||P) on several real ADBench datasets to ground the theorem's assumptions.
- Provide confidence intervals or standard deviations across the 10 repeated runs, at least for the main metrics.

## Novel Insights

The most interesting empirical finding—that NF-SLT is surprisingly competitive for tabular anomaly detection—stands on its own regardless of the definitional and theoretical issues. This challenges the prevailing narrative from the image domain that likelihood-based OOD detection is fundamentally flawed. The observation that tabular and image data differ in intrinsic dimension ratio (d-Ratio) by two orders of magnitude (~0.002–0.019 for images vs. 0.389–0.810 for tabular) is a striking quantification of a qualitative intuition, even if the causal link to NF-SLT performance is not fully established.

## Suggestions

- Redefine "counterintuitive phenomenon" to include a direct likelihood-based criterion (e.g., fraction of anomalous samples with higher likelihood than the median normal sample) alongside the relative-AUROC definition, and show both converge in their conclusions.
- Run NF-SLT with at least 2–3 flow architectures (NICE, RealNVP, and either Glow or neural spline flows) on a representative subset of datasets and report whether the findings are architecture-dependent.
- Test the independence assumption by running flows on ADBench datasets after applying ICA (removing correlations) and comparing performance to the original correlated data, directly testing whether correlation hurts NF-SLT as claimed.
- For the 'imdb' result: rather than dismissing it via Definition 3.3, provide likelihood distribution plots showing actual overlap between normal and anomalous samples, and discuss what this means for the original (not redefined) notion of counterintuitive behavior.

## Score and Decision

**Calibration anchor papers:**

- *On Uniformly Scaling Flows* (0eEtTsnmyo): scores 4/2/6, average ~4. Rejected. Similar domain (normalizing flow + anomaly detection), but with weaker empirical coverage and a narrower theoretical contribution. This paper has much broader benchmarking.

- *Noise-Robust Density Estimation for Tabular Data* (JdbqDiguyO): scores 2/4/4, average ~3.3. Rejected. Directly comparable domain (tabular AD on ADBench with normalizing flows), but with unverified core assumptions and poor writing. Our paper has significantly stronger empirical coverage and writing quality.

- *Likelihood Paradox Mitigation* (jCQVjd4vrX): scores 2/2/8, average ~4. Rejected. Directly related topic (likelihood paradox in OOD detection with flows), but limited novelty and insufficient empirical validation.

- *How Many Code and Test Cases Are Enough* (RomWar2kVN): scores 6/8/6/4, average ~6. Accepted poster. Strong empirical contribution with a novel formalization; analogous in structure to this paper. This paper's formalization is weaker, but its empirical contribution is comparably broad.

- *Ads that Stick* (A8wfXZkoMs): scores 8/6/6, average ~6.7. Accepted poster. Clean theoretical contribution with experimental support. This paper's theory is much weaker.

This paper has genuinely valuable empirical results (broad benchmarking showing NF-SLT works well on tabular data) and tackles an important question. However, the central formalization (Definition 3.3) conflates likelihood inversion with relative model performance, the theoretical analysis rests on strong independence assumptions that conflict with its own thesis, and the d-Ratio analysis is correlational and partially reported. The empirical finding that NF-SLT is competitive on tabular data is robust and useful, but the explanatory claims overreach what the theory and analysis can support. The paper's contribution is primarily empirical; the theoretical and definitional contributions are weaker than claimed.

Compared to rejected papers in the same domain (NRDE at ~3.3, Likelihood Paradox Mitigation at ~4), this paper has stronger empirical coverage and better writing. Compared to accepted papers with clean but modest theoretical contributions (~6-7), this paper's theory is messier and its claims are overreaching. I place this paper in the 4–5 range: a borderline paper with a useful empirical finding that is undermined by overclaiming on the theoretical and definitional fronts.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>