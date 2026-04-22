Now I have a good understanding of the paper and calibration anchors. Let me synthesize the final review.

Key verification of harsh critic's claims:

1. **PVI with softmax normalization reduces to prior-corrected second model outputs**: MATHEMATICALLY CORRECT. After softmax, PVI_k ∝ (f'[x](y_k)/f[∅](y_k))^(1/T). However, calling it "vacuous" is too strong — it's still a meaningful prior-correction mechanism. The more important point is the asymmetry in comparison.

2. **Missing deep ensemble baseline**: VALID. PVI requires training a second full model. The natural comparison is a 2-model ensemble, which is conspicuously absent.

3. **Asymmetric model capacity across PMI/PSI/PVI**: VALID. PVI uses a full architecture copy while PMI uses a shallow 2-layer network and PSI uses projections. This confounds the comparison.

4. **Overlapping confidence intervals**: PARTIALLY VALID. The paper itself acknowledges this in the results section: "After considering the margin of error, while the performance improvement of PVI is less pronounced for AUROC_f and AUPR_{f,success}, it remains notably significant for AUPR_{f,failure} and AURC." So the paper partially addresses this.

5. **Limited calibration baselines**: VALID. Only MSP and SM in Table 3, justified only by "confidence calibration requires the confidence estimates to be between 0 and 1." This is a reasonable methodological constraint but still limits the claim.

6. **Temperature scaling ablation missing**: VALID — this is a meaningful gap.

Let me write the final review now.

## Summary

This paper proposes using three pointwise information (PI) measures—PMI, PVI, and PSI—as post-hoc confidence estimators for deep neural networks. It provides theoretical analysis of their invariance properties, margin sensitivity, and convergence rates, and empirically compares them across failure prediction and confidence calibration tasks, finding that PVI outperforms baselines on most metrics.

## Strengths

- **Novel invariance hierarchy (Propositions 1–3, Section 3.1):** The paper establishes that PMI is invariant to homeomorphisms (potentially counter-productive per Remark 9), PSI lacks invariance to invertible linear transformations, and PVI strikes an intermediate balance by being invariant to invertible linear but not all homeomorphic transforms. This hierarchy provides genuine theoretical insight into why these measures behave differently and directly supports T2/T5.

- **Proposition 4 (PMI fails on non-overlapping distributions, Section 3.2):** This non-obvious result shows pmi(x;y)=1 for any non-overlapping class-conditional distributions with equal priors, meaning PMI cannot distinguish margin in well-separated settings. This is a crisp, novel theoretical deficiency of PMI.

- **Consistent empirical advantage of PVI on key metrics (Tables 2–3):** PVI achieves the best AUPR_{f,error} and AURC across all four model-dataset combinations (e.g., 56.07 vs. next-best 48.54 on ResNet50/CIFAR-10 for AUPR_{f,error}), and the lowest ECE in all settings. These are the more informative metrics per Jaeger et al. (2023).

- **Insightful tension between margin correlation and confidence estimation (Section 5, Table 1):** PSI has the highest margin correlation (0.758–0.846) yet PVI outperforms it on accuracy-based tasks. The paper's explanation—that margin sensitivity measures boundary proximity while confidence estimation requires predictive reliability—is nuanced and practically important.

- **Honest reflections on limitations (Section 5):** The paper explicitly acknowledges that PI measures require additional model training and that higher margin correlation does not guarantee better confidence estimation, which shows intellectual honesty.

## Weaknesses

### Fatal
None.

### Major

- **PVI after softmax normalization functionally reduces to prior-corrected outputs of a separately trained classifier, undermining the "information-theoretic" framing of the empirical advantage.** After softmax normalization across classes, PVI_k = exp((-log f[∅](y_k) + log f'[x](y_k))/T) / Z is proportional to (f'[x](y_k)/f[∅](y_k))^(1/T)—i.e., a temperature-scaled, prior-corrected probability from the *separately trained* model f'. The f[∅] terms encoding class marginals are absorbed into the softmax normalization. This means PVI's advantage over baselines that use only the original model's outputs (MSP, SM, LM, etc.) may stem from having access to a second trained model's capacity rather than from any information-theoretic property of PVI. The paper does not acknowledge this collapse or decompose PVI's performance into the contribution of (a) the second model, (b) prior correction, and (c) temperature scaling. (Section 2, Definition 3; Section 4 normalization paragraph.)

- **Missing deep ensemble baseline invalidates the headline claim of outperforming "all existing baselines."** Since PVI requires training a second full model with the same architecture (Section 2: "using the same network but with different initialization"), the direct fair comparison is a 2-model deep ensemble—identical computational cost (two full training runs). Deep ensembles are among the strongest known post-hoc uncertainty methods. If a 2-model ensemble matches or outperforms PVI, the paper's contribution reduces to "training two models is better than one," which is not novel. The paper's claim to "outperform all existing baselines" is unsupported without this comparison. (Abstract, Section 4.1 baseline list.)

- **Asymmetric estimator capacity across PMI, PSI, and PVI confounds the comparison.** PVI uses a full copy of the original architecture; PMI uses a "shallow 2-layer neural network" (Section 2); PSI uses Gaussian/binning estimators on 1D projections (Section 2). When PVI outperforms PMI and PSI, it is impossible to attribute this to the information-theoretic properties rather than to the simple fact that a full model is a much more powerful density estimator than a 2-layer network. The theoretical analysis (Section 3) studies the *idealized quantities*, not the finite-capacity estimators, so it cannot explain differences driven by model capacity. The paper should either match estimator capacities or acknowledge this confound explicitly. (Section 2; Section 3.3 T5.)

### Minor

- **Calibration evaluation (Table 3, Section 4.2) includes only two baselines (MSP and SM),** justified on the grounds that "confidence calibration requires the confidence estimates to be between 0 and 1." While this is methodologically reasonable, it significantly limits the strength of the "outperforms all baselines" claim for calibration, since methods like Platt scaling or Dirichlet calibration also produce calibrated probabilities in [0,1] and could serve as additional baselines.

- **No temperature scaling ablation on the original model.** The paper applies softmax + temperature scaling to PI values. As the harsh analysis notes, if temperature scaling alone on the original model's outputs (without PI measures) achieves comparable results, the PI framework adds no value beyond the standard post-hoc calibration already widely used. This ablation is missing and would significantly strengthen or weaken the paper's claims. (Section 4.)

- **Many claimed improvements fall within standard deviation margins.** While PVI shows clear advantages on AUPR_{f,error} and AURC, improvements on AUROC_f are often within noise (e.g., ResNet50/CIFAR-10: PVI 86.50±1.02 vs. SM 85.14±0.38). The sweeping "outperforms all baselines" claim in the abstract is stronger than the data support, though the paper partially acknowledges this in Section 4.1's results paragraph. (Table 2, Abstract.)

### Trivial
None.

## Nice-to-Haves

- Decompose PVI's performance into the contribution of (a) having a second model, (b) prior correction via f[∅], (c) temperature scaling, and (d) the information-theoretic formulation itself. This would clarify what drives the empirical advantage.
- Run all PI measures with matched estimator capacity (e.g., all using the original architecture) to disentangle the effect of the information measure from the estimator's power.
- Compare PVI against a 2-model deep ensemble, which uses identical computational resources.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"PVI is not a post-hoc confidence estimator in the same class as MSP"** — This is an interpretive framing issue, not a flaw. PVI *is* applied post-hoc to a trained model; it just requires an additional training step. The paper is transparent about this in Section 2 and the Limitations paragraph. The real concern is the asymmetry of comparison, which is captured in Major weaknesses above.

- **"Post-hoc without modifying architecture is misleading because PVI trains a full second model"** — The paper does say "without needing to modify their architecture or training process" (abstract), which refers to the *original* model. PVI's additional model is a separate estimator. While the framing could be clearer, the paper does disclose the additional training in Section 2.

- **Proposition 2 invariance may not hold for finite networks** — The proposition is stated as a mathematical fact about the idealized PVI quantity, which is standard practice. The gap between theoretical and estimated quantities is a universal issue in information theory, not unique to this paper. The Limitations section partially addresses this.

- **Proposition 5 only gives an upper bound, not a tracking relationship** — This is a valid observation but mischaracterized as a fatal flaw. An upper bound is still informative—it tells us PVI cannot exceed a margin-dependent quantity, which constrains its behavior. The paper doesn't overclaim this result.

- **PSI having higher margin correlation contradicts the theory's prediction that PVI is "most well-rounded"** — The paper explicitly discusses this in Section 5 and explains the distinction between margin sensitivity and predictive reliability. This is an honest finding, not a contradiction that undermines the paper.

- **Missing related works** — Not verifiable without external sources; removed per rules.

- **Formatting/style/appendix concerns** — Removed per rules (parser artifacts, missing appendix).

- **Missing statistical significance testing** — Single-run evaluation with standard deviations is the norm in this area; requesting paired significance tests is a nice-to-have, not a weakness.

- **Reproducibility concerns about hyperparameters or implementation details** — Removed per rules.

## Novel Insights

The most novel insight from this review is that the information-theoretic framing of PVI, while theoretically elegant, functionally collapses into prior-corrected second-model outputs after the softmax normalization the paper applies. This means the paper's central empirical finding—PVI outperforming baselines—is most parsimoniously explained by the trivial advantage of having a second trained model, not by the information-theoretic properties the theoretical analysis studies. The disconnect between theory (which predicts PVI is "most well-rounded") and the likely driver of empirical success (model capacity) is a deeper problem than the paper acknowledges.

## Suggestions

- Add a 2-model deep ensemble baseline. This is the single most important experiment: if the ensemble matches PVI, the paper's contribution shifts; if PVI beats the ensemble, it's a much stronger result.
- Run an ablation where temperature scaling is applied directly to the original model's logits (without PI measures) to isolate the value added by the PI framework.
- Match estimator capacities across PMI, PSI, and PVI (e.g., all using the same architecture) in at least one experiment to disentangle measure properties from estimator power.

## Evaluation on Axes

- **Originality:** Moderate. The systematic comparison of PI measures with theoretical analysis is novel, but the empirical advantage may reduce to model capacity rather than the information-theoretic framework itself.
- **Importance of research question:** High. Confidence estimation in post-hoc settings is a practically important problem, especially given that calibration methods can harm failure prediction.
- **Claims well supported:** Moderately. The theoretical claims are well-supported, but the central empirical claim ("outperforms all baselines") is undermined by the absence of the most natural baseline (2-model ensemble) and the asymmetry in model capacity.
- **Soundness of experiments:** Weakened by missing deep ensemble baseline, no temperature-scaling ablation, and asymmetric estimator capacity across PI measures.
- **Clarity of writing:** Generally good. The T1–T5 takeaways structure is helpful. The framing of PVI as "post-hoc without modifying architecture" could be more transparent about the second-model requirement.
- **Value to community:** Moderate. The theoretical invariance hierarchy and margin analysis are useful contributions regardless of the empirical confound, but the practical value is diminished until the confounds are addressed.

## Calibration

**Anchors used:**

1. **ta26LtNq2r.md** (avg 8.0, Accept Spotlight) — "Learning to Reject Meets Long-tail Learning": Strong theory + clear empirical contribution. Much cleaner causal chain from theory to experiments than this paper. This paper is clearly below this anchor due to the fundamental comparison asymmetry.

2. **TId1SHe8JG.md** (avg 7.5, Accept Spotlight) — "Provable Uncertainty Decomposition via Higher-Order Calibration": Principled uncertainty method with formal guarantees. Stronger theoretical contribution with more rigorous empirical validation. This paper is below this anchor.

3. **YUefWMfPoc.md** (avg 5.75, Reject) — "How to fix a broken confidence estimator": Comprehensive evaluation of post-hoc confidence estimators with strong empirical scope but limited novelty. This paper has more theoretical contribution but worse empirical methodology (missing ensemble baseline, asymmetric capacity). Roughly comparable or slightly below this anchor.

4. **ohHtdp3jDi.md** (avg 4.0, Reject) — "Implicit Functional Bayesian Deep Learning": Missing deep ensemble baseline, overclaimed results, sub-par performance. Similar pattern of missing key baselines. This paper has better theory but similar empirical gaps. Comparable or slightly above.

5. **jTnHyyGYy2.md** (avg 4.5, Reject) — "LoRA-Ensemble": Mixed results, deep ensembles outperform proposed method on some benchmarks, missing comparisons to compute-efficient approaches. Directly relevant comparison pattern (method vs. ensemble). This paper is comparable to this anchor.

6. **63r6HyqyRm.md** (avg 2.33, Reject) — "Vision-free Baseline for Multimodal Grammar Induction": Unfair comparison where proposed method uses pre-trained model importing billions of training data. More severe version of the asymmetry problem in this paper. This paper is clearly above this anchor since the asymmetry is less extreme.

The paper sits in the 4–5 range: it has genuine theoretical contributions (invariance hierarchy, PMI's failure on non-overlapping distributions) but the central empirical claim is undermined by the missing ensemble baseline and the effectively trivial nature of PVI's advantage after softmax normalization. This is worse than the 5.75 anchor (which had stronger empirical methodology) and comparable to the 4.0–4.5 anchors that shared similar weaknesses around missing baselines and overclaimed results.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>