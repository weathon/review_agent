=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary

This paper proposes forecasting whole-brain neuronal activity in larval zebrafish directly from raw 4D volumetric fluorescence video, bypassing the traditional pipeline of segmenting recordings into 1D activity traces. The authors adapt a 3D UNet with temporal frames encoded as input channels, design a spatially sharded data loading system to handle trillion-voxel inputs across 16 A100 GPUs, and conduct extensive ablations on temporal context, spatial context (receptive field), input resolution, and pre-training strategies. On ZAPBench, the video model is the only multivariate approach that consistently outperforms univariate baselines, achieving a ~10 percentage-point error reduction at step 1 for short temporal context (C=4), while offering no statistically significant MAE gain over trace-based models for long context (C=256).

---

## Strengths

- **Novel problem formulation with demonstrated payoff**: The specific claim that spatial structure discarded during trace extraction is *predictively recoverable* is directly validated. The video model is the only multivariate approach that consistently beats univariate trace models on this benchmark—a concrete, surprising, and domain-significant result that contradicts the standard neuroscience pipeline assumption.

- **Engineering scalability with near-linear compute scaling**: Processing 16× more data (full vs. 4× downsampled resolution) at near-linear scaling via spatially sharded JAX data loading and model parallelism is a non-trivial systems contribution. The FLOPS-controlled receptive field ablation design (replacing four blocks at lowest resolution with one block at higher resolution) is a clean and reproducible methodology.

- **Principled spatial–temporal trade-off characterization**: Figure 5, with error bars over three seeds, reveals a genuine crossover between C=16 and C=64 where larger spatial context transitions from beneficial to harmful due to overfitting. The alignment of this crossover with the ~64-step periodicity of stimulus conditions is an interesting scientific observation.

- **Honest reporting of null and counterintuitive results**: The paper transparently reports that (1) cross-specimen pre-training hurts rather than helps, (2) higher input resolution does not improve and actually slightly harms performance at full resolution, and (3) there is no significant MAE advantage for C=256 on the test set. Papers that suppress inconvenient results are common; this one does not.

- **Informative masking experiment (Appendix A.2)**: The finding that masking out all unsegmented voxels does not reduce performance is a clean negative control that substantially narrows the mechanism of improvement to spatial correlations between segmented cells, rather than hypothesized inter-cell or sub-cellular signal.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Limited and partially misleading gains for C=4**: While the abstract states the model "outperforms trace-based forecasting approaches," Figure 6 (C=4, Test set) shows the trace model catching up around step 10 and being comparable at later horizons. The aggregate MAE improvement is real, but driven predominantly by early forecasting steps. The abstract and introduction should be more qualified about this step-dependent pattern—claiming general outperformance obscures a non-trivial limitation.

- **C=256 is essentially a null result for the primary metric**: The paper explicitly acknowledges "no significant difference between the univariate trace-based model and the video model on the test set when evaluated with MAE" for C=256. For a benchmark covering both C=4 and C=256, half the benchmark settings show no benefit. The correlation metric improvement is relegated to the appendix. This asymmetry substantially weakens the paper's overall claim of video-model superiority.

- **Pre-training failure is under-analyzed**: Section 4.1.2 reports that pre-training on 8 additional specimens (8× data) actively *harms* performance versus training from scratch, yet the only explanation offered is "distribution shifts between specimens." No quantification of these shifts (signal-to-noise ratios, imaging depth, behavioral protocol differences, mean ΔF/F levels) is provided. The failure of cross-specimen transfer is one of the paper's most scientifically important findings and deserves substantially deeper investigation than a one-sentence hypothesis.

- **Table 2 resolution comparison lacks statistical power**: Only the 4× downsampled condition is run with three seeds (±0.0002 SE). The 2× and full-resolution conditions are single runs. The claimed difference between 4× (0.0267) and full resolution (0.0273) is only 3× the observed seed-to-seed variability. The paper acknowledges this is due to compute costs, but the absence of error bars makes the directional conclusion about full resolution underperformance statistically fragile.

### Minor

- **Temporal-channel design introduces an ordering invariance**: Representing C input frames as channels means any permutation of input frames produces identical features, since the UNet has no temporal convolution. The paper acknowledges this design and justifies it on scalability grounds, but does not discuss whether it materially limits forecasting quality for dynamics with strong causal asymmetry. An experiment comparing to even a lightweight recurrent layer in the bottleneck would help characterize this trade-off.

- **Missing training wall-clock time**: Section 4 states "most individual training experiments use 16 A100 40GB GPUs" but provides no wall-clock training time. For a paper with stated reproducibility aspirations and the claim of 2–3 orders of magnitude more compute than trace models, concrete training durations are essential for practitioners to assess feasibility.

- **Equation 2 potential off-by-one in notation**: $\hat{\mathbf{Y}}(t, h) = f_h(\mathbf{Y}(t), \dots, \mathbf{Y}(t+C), \mathbf{w})$ lists C+1 frames (t through t+C inclusive), while C is described as context length throughout. If this is intentional inclusive indexing, it should be stated; otherwise it is a notation error.

- **Validation–test discrepancy in Figure 5**: The validation and test performance curves for S=21 diverge in a way that suggests the model selection procedure may have overfit to the validation set for some configurations. This should be discussed, since the conclusions drawn from Figure 5 depend on the curves being representative.

- **Single-specimen generalizability**: All main results derive from one larval zebrafish. The failure of cross-specimen pre-training suggests the learned representations may be animal-specific. The paper does not discuss whether conclusions about the utility of spatial context would hold for other animals or imaging conditions.

### Tiny

- The receptive field formula is presented without a derivation; the implicit assumptions (unit stride, same padding, specific kernel sizes) should be stated explicitly for reproducibility.

- Section 4.2's conclusion that "correlations between cells… drives improvements" is presented confidently but rests on two indirect experiments (masking + resolution). This is the best available evidence given experimental constraints, but the confidence of the claim somewhat exceeds the directness of the evidence.

---

## Nice-to-Haves

- **2D slice-by-slice baseline**: A natural intermediate between full 3D video and 1D traces would be processing each Z-plane independently. Its inclusion would help attribute gains specifically to cross-plane 3D spatial context versus simpler within-plane 2D modeling.

- **Feature/error visualization**: Spatial heatmaps of per-neuron prediction error (video vs. trace) would allow readers to assess whether improvements cluster in anatomically coherent brain regions and whether the model is capturing biologically meaningful spatial structure.

- **Per-neuron analysis of benefited vs. harmed cells**: For C=4, the video model is worse in 2/9 conditions. Analyzing *which* neurons or brain regions drive these failures would provide scientific insight into when spatial modeling helps or hurts.

- **Compute-accuracy tradeoff guide**: A brief quantitative framing (e.g., for what scientific questions does a 10% step-1 MAE reduction justify 100× compute) would help the neuroscience community calibrate when to adopt this approach.

- **Alternative video architecture evaluation**: Testing even one alternative (e.g., a factorized spatial-temporal transformer at the bottleneck) would help establish whether the UNet design is near-optimal or merely adequate. This is not expected for a first paper in a new domain, but would strengthen the architectural contribution claim.

- **Probabilistic outputs**: The stochastic nature of neural activity makes probabilistic evaluation appealing. The authors note this as future work, and given the early-stage nature of the benchmark, it is not a current requirement.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **ZAPBench circular dependency (Harsh Critic)**: The benchmark is a concurrent ICLR submission (explicitly cited as "Anonymous, 2024") and the relationship is fully disclosed in the paper. Co-development of benchmark and method is not inherently problematic and is common in new domains. The paper even notes that ZAPBench includes its own comparisons of video vs. trace models. The speculative concern that ZAPBench's design choices might advantage video models is unsubstantiated and should not penalize the authors.

- **No comparison to transformer-based video models (Harsh Critic)**: The paper's architectural choice of UNet is explicitly motivated by scalability to trillion-voxel inputs—a constraint that makes transformers with global attention intractable. Demanding a Video Swin or VideoMAE comparison is scope creep given the engineering context. (Kept as nice-to-have above.)

- **Lack of probabilistic evaluation (Harsh Critic)**: The paper explicitly notes that the ZAPBench experimental design did not allow sufficient trial repetitions for proper probabilistic evaluation and cites this as future work. Criticizing the absence of probabilistic evaluation given this constraint is unfair.

- **Lead-time conditioning overstated (Harsh Critic)**: Figure 4 shows nearly identical final MAE between direct prediction and lead-time conditioned variants; the paper's stated advantage is training stability and loading efficiency. The paper does not substantially overstate the accuracy benefit.

- **Extended pre-training evaluation: 8 specimens is insufficient (Spark Finder)**: 8 additional specimens of the same species is a non-trivial data quantity. The claim that this fails is already supported as a finding; demanding 20+ specimens before drawing any conclusions is not a standard requirement for an ablation study.

- **Spatial shuffling ablation to confirm inter-cell correlations drive gains**: Interesting but non-standard for a systems-level paper. The masking experiment in Appendix A.2 already provides triangulating evidence.

---

## Novel Insights

The most novel and potentially impactful insight—one not obvious from the paper's self-description—is the *informational geometry* finding that emerges from combining the masking experiment and the resolution study: improvements come not from sub-cellular voxel detail, not from unsegmented inter-cell voxels, but specifically from the *spatial correlation structure between segmented neurons as seen through the raw video grid*. This means that the segmentation mask correctly identifies *which* voxels matter, but the trace-extraction step destroys the *joint spatial layout* of those signals that enables prediction. This suggests that future neuroscience recording pipelines might benefit from retaining low-resolution spatial summaries (rather than fully segmented 1D traces) as a computationally cheap intermediate that preserves much of the predictive signal. The observation that this spatial advantage disappears for long temporal context (C≥64), aligned with the ~64-step stimulus periodicity, further suggests that temporal redundancy and spatial redundancy are interchangeable in this system—an insight with implications for how calcium imaging data should be represented for downstream predictive modeling.

---

## Suggestions

1. **Qualify the abstract and introduction**: Clarify that the video model's advantage over trace-based models for C=4 is concentrated in early forecasting steps (approximately steps 1–10) and diminishes thereafter, and that for C=256 there is no MAE advantage on the test set. The current framing reads as broader than the results support.

2. **Quantify distribution shifts for pre-training failure**: Add at minimum a table reporting per-specimen SNR, mean ΔF/F, and imaging depth for the 8 pre-training specimens vs. the ZAPBench specimen. This would transform a speculative hypothesis into a testable and publishable mechanistic claim.

3. **Report confidence intervals for Table 2 or acknowledge the statistical limitation explicitly in the main text**: Either run 2–3 seeds for the 2× downsampled condition (less expensive than full resolution) or add a formal caveat in the main text that the full-resolution comparison is statistically underpowered and directional only.

4. **Promote correlation results for C=256 to the main text**: If MAE shows no advantage but correlation does, reporting only MAE in Figure 6 gives a misleadingly negative picture of C=256. The correlation metric should appear in Figure 6 or in a companion figure.

5. **Report training wall-clock time**: Even a rough estimate (e.g., "X hours on 16 A100s") would allow practitioners to assess feasibility and is needed for a paper emphasizing engineering scalability.

6. **Analyze conditions where video model underperforms**: For the 2/9 stimulus conditions where the trace model wins, a short analysis of what distinguishes those conditions would be more scientifically informative than aggregate averaging.

---

**Overall evaluation:**

- **Novelty**: Moderate — the idea of applying video models to volumetric neural recordings is domain-novel and the problem formulation is fresh, but the ML techniques (UNet, temporal-as-channels, lead-time conditioning) are established
- **Technical soundness**: Good — the engineering is rigorous and the ablation design is careful, though Table 2's statistical weakness and the pre-training analysis's shallowness are real gaps
- **Empirical support**: Adequate for C=4 short-horizon claims; weak for C=256; single-specimen scope is a genuine limitation
- **Significance**: Moderate-high for computational neuroscience; the finding that spatial trace layout carries recoverable predictive signal is a meaningful result for the field
- **Clarity**: Good — the paper is well-organized and the key results are honestly reported, though the abstract and framing of main claims would benefit from more precise qualification

# Actual Human Scores
Individual reviewer scores: [5.0, 6.0, 8.0]
Average score: 6.3
Binary outcome: Reject
