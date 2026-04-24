Now I will synthesize the final review based on my direct reading of the paper and calibration against the human-review corpus.

## Summary

This paper identifies *multimodal wild TTA*—a test-time adaptation setting where unlabeled target data contain mixed weak OOD, strong OOD, and missing modalities—and proposes SuMi to address it. SuMi combines three strategies: (1) interquartile-range sample selection with a time-varying threshold to gradually include more outlying samples during adaptation; (2) unimodal entropy-assisted filtering to select low-multimodal-entropy samples that nevertheless exhibit moderate unimodal entropy; and (3) a KL-divergence-based cross-modal consensus loss. Extensive experiments on Kinetics50-C and VGGSound-C show SuMi outperforming seven baselines across corruption types, severity levels, and mixed-domain ratios.

## Strengths

- **Novel and practically important problem.** The paper is among the first to systematically study test-time adaptation when target domains contain simultaneous corruptions, missing modalities, and mixed distribution shifts. The constructed Kinetics50-C and VGGSound-C benchmarks are useful community contributions.
- **Strong motivating observation.** Figure 3(a) shows that adapting first on weak OOD then on strong OOD (“Weak→Strong”) yields substantially higher accuracy than direct strong-OOD adaptation on Kinetics50-C, providing intuitive empirical support for gradual sample inclusion.
- **Dominant empirical performance.** SuMi consistently outperforms existing methods—often by large margins—on strong OOD scenarios (e.g., Table 2: 33.4% average on Kinetics50-C strong OOD vs. 29.1% for the next-best method, READ, and ≤16% for all others). The mixed-ratio curves in Figure 5 further demonstrate robustness as the proportion of strong OOD samples increases to 100%.
- **Thorough ablations.** Table 5 provides a full combination matrix of the three proposed components across datasets and severity levels, and Table 6 isolates the impact of different sample-selection regions.

## Weaknesses

### Fatal

None.

### Major

- **“Mutual information sharing” is a misnomer that misrepresents the mechanism.** Section 3.3 labels Equation 6 as “mutual information sharing,” but the loss is a sum of KL divergences between each unimodal prediction and a mixture of its complementary unimodal prediction and the multimodal prediction. Mutual information is the well-defined quantity *I(X;Y)*; Equation 6 is a consensus/distillation term. Using established information-theoretic terminology incorrectly breaks the claimed theoretical connection and is misleading.
- **Experimental evaluation is incomplete and undermined by suspect baseline behavior.** (1) Tables 1–4 omit source-model accuracy, making it impossible to verify the repeated claim that existing methods “perform even worse than the source model” (§4.2). (2) On VGGSound audio corruptions (Table 4), SAR, SoTTA, DeYO, and CEMA collapse to single-digit or near-single-digit averages (3.6%, 7.7%, 4.2%, and 3.7%, respectively). The paper offers no explanation for why methods that perform adequately on Kinetics50 audio (Table 2) or VGGSound video (Table 3) would fail catastrophically here. These results look like implementation artifacts rather than inherent methodological limitations, which compromises the fairness of the comparison. (3) Figure 1(b)–(d) reports numbers that are inconsistent with the detailed results in Tables 2–4 (e.g., Figure 1(b) lists Tent strong-OOD bars around 15%, 10%, 20%, and 5%, whereas Table 2 reports Tent at 30.8%, 27.9%, 44.5%, and 16.9% for the same scenarios), with no explanation of the discrepancy.
- **Unimodal entropy reasoning is confounded and thinly supported.** The paper argues that high *unimodal* entropy indicates “rich multimodal information” (§3.2.2), selecting samples via Equation 4 when the weighted sum of unimodal entropies exceeds a threshold. This inverts the standard interpretation of entropy as uncertainty. The evidence is limited to a single t-SNE-style bar chart (Figure 3c) showing video entropy only on Kinetics50-C. Because Equation 4 uses a *summed* threshold, a sample with one clean modality (low entropy) and one heavily corrupted modality (high entropy) can still be selected; conversely, a sample with both modalities moderately corrupted might have low multimodal entropy and high summed unimodal entropy, passing both filters. The paper does not quantitatively disentangle “multimodal informativeness” from noise or sample difficulty.

### Minor

- **IQR smoothing justification is strained for the evaluated setting.** The paper frames IQR expansion as “smoothing the adaptation process” to avoid abrupt distribution shifts. However, the mixed-domain experiments (Figure 5, §4.2) evaluate stationary mixtures of weak and strong OOD samples, not temporally ordered streams. While the time-varying threshold does gradually include more samples over optimization iterations, the paper assumes—without quantitative evidence beyond a single t-SNE plot (Figure 3b)—that representational outlier-ness under the source model reliably correlates with ground-truth OOD strength.
- **Ablation lacks a fixed-threshold control.** Table 5 ablates the three components but does not compare the time-varying IQR schedule against a fixed outlier threshold (e.g., a constant Tukey fence). Without this control, one cannot determine whether gains stem from the *schedule* or merely from filtering outliers at any threshold.
- **Algorithm 1 is ambiguous about Q1/Q3 computation.** Lines 5–6 compute quantiles over the representation vector **h**, but it is unclear whether these statistics are computed per mini-batch, over a running buffer, or across the full target set. This ambiguity impacts reproducibility.

### Trivial

- The abstract claim that “Existing TTA methods always fail” is hyperbolic. Tables 1–4 show that EATA and READ remain competitive with SuMi on weak OOD, and the gaps on some strong-OOD settings are modest (e.g., EATA 17.4 vs. SuMi 19.7 on VGGSound strong OOD in Table 4).

## Nice-to-Haves

- Evaluate on a true temporally-ordered wild TTA stream (e.g., clean → weak → strong) to better match the “smoothing” framing in the abstract and introduction.
- Provide annotated scatter plots of video entropy versus audio entropy (colored by corruption type) to validate whether Equation 4 actually isolates the intended high-multimodal-information samples.

## Removed Points

These points are flagged to be removed; treat them with caution.
- **Criticism questioning model/dataset existence or release status.** All cited benchmarks, models, and prior methods are properly referenced and assumed to exist.
- **Claim that IQR expansion “does not smooth a temporal trajectory.”** The smoothing in the paper refers to the adaptation trajectory over optimization iterations, not necessarily over a temporal data stream. While the critic’s concern about conflating optimization smoothing with stream smoothing has merit, the mechanism does gradually include samples during adaptation.
- **Claim that Equation 4 would select samples where “both modalities are corrupted.”** If both modalities are severely corrupted, multimodal entropy is likely to exceed *γ_m* and the sample would be filtered out by the first condition of Equation 4. The critic overstates this particular confound.
- **Formatting, typo, and grammar nitpicks.** These are parser artifacts, not author errors.
- **Reproducibility nitpicks about undisclosed hyperparameters or missing training logs.** The paper provides core hyperparameters in §4.1 and states code is available.

## Novel Insights

None beyond the paper's own contributions. The identification of multimodal wild TTA as a distinct problem and the empirical observation that curriculum-like adaptation (weak→strong) outperforms direct strong-OOD adaptation are the paper’s main novel insights.

## Suggestions

1. **Rename Equation 6** to “cross-modal consensus loss” or “KL-alignment loss” and remove all references to “mutual information sharing.”
2. **Report source-model accuracy** in every results table to validate claims about TTA destroying prior knowledge.
3. **Investigate and explain the VGGSound audio baseline collapse.** If the implementations are correct, provide a diagnosis; if not, rerun and update the results.
4. **Reconcile Figure 1 with Tables 2–4** or clearly state what metric/split Figure 1 uses.
5. **Add a fixed-IQR-threshold ablation** to isolate the benefit of the time-varying schedule.

## Score and Decision

**Calibration papers compared:**
- `/home/wg25r/review_agent/human_reviews/TPZRq4FALB.md` (READ, avg 8.00, Accept): Same topic (multimodal TTA) and benchmarks. READ is cleaner in terminology, baselines, and presentation. SuMi is clearly below this anchor.
- `/home/wg25r/review_agent/human_reviews/SIzjhS9kEF.md` (avg 5.75, Reject): Strong empirical results but overselling claims and contradictory reasoning. Comparable quality level—SuMi has stronger empirical gains but similar reasoning flaws.
- `/home/wg25r/review_agent/human_reviews/Chq4OQ3p18.md` (Intransigent Teachers, avg 5.25, Reject): Simple TTA method with baseline comparison issues and limited theoretical depth. SuMi has a more novel problem and larger empirical margins, but similar baseline integrity concerns.
- `/home/wg25r/review_agent/human_reviews/H65sp7ztys.md` (TTA for OOD detection, avg 3.67, Reject): Hyperbolic claims and theoretically shallow. SuMi is above this anchor due to stronger problem formulation and more comprehensive experiments.
- `/home/wg25r/review_agent/human_reviews/KZZbdJ4wff.md` (PRO, avg 3.75, Reject): Combination of existing methods with insufficient novelty. SuMi is above this due to genuine novelty in problem and method.

SuMi sits between the rejected 5.25 anchor (simple method, baseline issues) and the rejected 5.75 anchor (strong experiments, overclaiming). The paper introduces a valuable problem and demonstrates strong empirical results, but the terminology misuse, missing source baselines, suspicious baseline collapse on VGGSound audio, and thin theoretical justification for the entropy criterion significantly undermine credibility. These issues are addressable in revision but are too severe for acceptance in the current form.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>