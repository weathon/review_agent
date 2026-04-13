=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary

This paper proposes an approach to adversarial robustness through multi-resolution input representations and a novel aggregation mechanism called CrossMax, inspired by Vickrey auctions. By stacking multiple downsampled versions of an input image channel-wise and ensembling predictions from intermediate network layers, the authors achieve strong adversarial accuracy on CIFAR-10 (~72%) and CIFAR-100 (~48%) without adversarial training, using only ImageNet-pretrained models fine-tuned with standard training.

## Strengths

- **Strong empirical results without adversarial training**: The paper achieves 71.88% adversarial accuracy on CIFAR-10 and 48.16% on CIFAR-100 under AutoAttack ($L_\infty = 8/255$) without any adversarial training (Table 1). This is notable because leading RobustBench entries typically require extensive adversarial training.
- **Novel aggregation mechanism (CrossMax)**: Algorithm 1 introduces a principled ensembling approach that subtracts per-predictor and per-class maxima before median/top-k selection. This provides defense against any single predictor or class dominating the ensemble—a concrete contribution to ensemble theory for robustness.
- **Empirical insight on intermediate layer de-correlation**: Figure 5 demonstrates that adversarial attacks targeting specific layers do not transfer well to layers in different depth bands (early/middle/late). This finding justifies the self-ensembling approach and is independently interesting.
- **Visual evidence of interpretable adversarial examples**: Figures 7-10 show that attacks against the proposed model produce semantically meaningful changes (e.g., pear→apple by adding a dividing edge), supporting the authors' Interpretability-Robustness Hypothesis.

## Weaknesses

- **Evaluation conducted on small subsets rather than full test sets**: Table 1 reports results on 128 examples (CIFAR-10) and 512 examples (CIFAR-100), while RobustBench standard protocol uses all 10,000 test images. At n=128, a 95% confidence interval around 71.88% adversarial accuracy spans approximately ±8 percentage points, making the claimed comparison to SOTA (73.71%) statistically inconclusive. Full test set evaluation is essential for claims of matching or exceeding RobustBench leaders.

- **Potential gradient masking not ruled out**: The method combines stochastic test-time transformations (jitter, noise), non-differentiable aggregation operations (topk, median), and multi-resolution channel stacking—all hallmarks of obfuscated gradient defenses. While the authors use AutoAttack with the `rand` flag, no BPDA (Backward Pass Differentiable Approximation) attack or adaptive attack specifically designed for CrossMax is provided. For a defense centered on non-differentiable aggregation, this is a serious omission that leaves genuine robustness unverified.

- **No ablation comparing CrossMax to standard aggregation**: The paper does not present a controlled comparison of CrossMax versus mean logit, mean softmax, or geometric mean ensembling under the same attack suite. This is the most critical ablation for evaluating whether CrossMax provides benefit over simpler alternatives.

- **Compute-unfair comparison with SOTA**: The best results (71.88% on CIFAR-10) use "3-ensemble of self-ensembles"—three independent ResNet152 models, each with 12-channel multi-resolution input and intermediate-layer predictions. This inference budget is roughly 10-15× that of single-model SOTA baselines being compared against. A fair comparison would report results for a 3-ensemble of adversarially trained SOTA models, or report single-model performance.

- **Inconsistent clean accuracy reporting**: Table 1 reports clean test accuracy of 89.17% for the "Multires backbone" ResNet152 on CIFAR-10, while Figure 6 reports 73.7% for the same configuration. Similarly, Table 1 shows 87.14% for "Self-ensemble" while Figure 6 shows 68.9%. These discrepancies are unexplained and undermine confidence in the experimental reporting.

- **Linear probes trained for only one epoch**: The intermediate-layer linear probes used to demonstrate layer de-correlation (Section 2.3) are trained for a single epoch. Undertrained probes may appear "robust" simply because they haven't learned to be vulnerable, rather than reflecting genuine representational properties.

- **VLM attack and image generation claims are unsubstantiated in main paper**: The abstract and conclusion claim "successful transferable attacks on large vision language models" and "turn pre-trained classifiers...into controllable image generators," but these results appear only in the appendix (which was not included). Major claims must be substantiated in the main submission.

## Nice-to-Haves

- **ImageNet evaluation**: The method uses ImageNet-pretrained models but evaluates only on CIFAR-10/100. Testing on ImageNet would strengthen claims about scalability and representation quality.
- **Ablation over resolution choices**: The resolution set $\rho = \{32, 16, 8, 4\}$ is described as "arbitrary" (Section 3). An ablation varying N and specific scales would show whether robustness gains generalize beyond this specific configuration.
- **Quantitative interpretability metrics**: The Interpretability-Robustness Hypothesis is supported only by qualitative visual examples. Human evaluation scores or similarity metrics would strengthen this claim.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Biological motivation is loose"**: The analogy to microsaccades and foveal vision is indeed imprecise, but this is a motivation critique, not a weakness of the technical contribution. The method stands on its empirical merits regardless of whether the biological analogy is tight.

- **"Interpretability-Robustness Hypothesis is philosophically vague"**: While not rigorously quantified, the hypothesis is empirically supported through visual evidence and serves as a framing device rather than a formal theorem. Requiring formal quantification would be scope creep.

- **"No L2 results"**: The paper consistently uses $L_\infty = 8/255$ and compares to RobustBench entries using the same threat model. Demanding L2 evaluation is asking for additional experiments outside the stated scope.

- **"ResNet18 from scratch uses different sample size (1024 vs 128)"**: The paper reports 1024 samples for ResNet18 and 128 for ResNet152—different but internally consistent for each configuration. This is not a fatal flaw.

## Novel Insights

The intermediate layer de-correlation finding (Figure 5) is genuinely novel: adversarial attacks designed to fool early layers do not transfer to middle/late layers, and vice versa. This empirical "three-way split" suggests that different network depths encode fundamentally different vulnerability surfaces, and that self-ensembling across layers is not merely a trick but exploits genuine structure in how adversarial perturbations propagate through deep networks. If verified on other architectures, this finding could inform new defense strategies beyond the specific method proposed.

## Suggestions

- **Report full test set results** with standard AutoAttack (no subset sampling) to enable fair comparison with RobustBench entries.
- **Provide BPDA or adaptive attack evaluation** specifically designed for CrossMax to rule out gradient masking as the source of robustness gains.
- **Include compute-matched baselines**: either report single-model CrossMax results, or compare against a 3-ensemble of standard adversarially trained models.
- **Clarify clean accuracy discrepancies** between Table 1 and Figure 6—these appear to report incompatible numbers for the same experimental conditions.
- **Substantiate or remove appendix-only claims** (VLM attacks, image generation) from the main paper if space permits, or clearly note them as preliminary findings requiring separate investigation.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 5.0]
Average score: 6.8
Binary outcome: Reject
