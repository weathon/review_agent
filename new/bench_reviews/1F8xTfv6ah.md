Now I have sufficient context to write a well-calibrated final review.

---

## Summary
This paper proposes a novel OOD detection method that exploits the local neuroplasticity of Kolmogorov-Arnold Networks (KANs): an untrained KAN is kept as a reference, a copy is trained on InD data, and at inference time the difference in spline activation responses between the two networks serves as an InD score. To address the fundamental limitation that KANs process each feature independently (capturing only marginal, not joint, distributions), the authors propose partitioning the InD dataset and training separate KAN detectors per partition. Experiments on seven benchmarks spanning image classification (CIFAR-10/100, ImageNet-200/1K full-spectrum) and tabular medical data demonstrate top-ranked or near-top AUROC performance, with notably strong robustness to reduced training set sizes.

---

## Strengths

- **Conceptually novel OOD mechanism grounded in KAN architecture**: The use of spline-coefficient locality—where training on InD data only modifies grid coefficients near InD inputs—as the detection signal is architecturally principled and distinct from all existing OOD methods. Unlike NAC (neuron activation coverage) or histogram-based activation methods, the mechanism is uniquely tied to the parametric structure of KANs (Eq. 5), not generic activation statistics.

- **Strong and broad empirical validation on challenging benchmarks**: The KAN detector achieves the best overall average AUROC on CIFAR-10 (94.12 vs 93.37 for NAC), ImageNet-200 FS (71.46 vs ~67 for competing methods), and ImageNet-1K FS (78.52 vs 76.28 for NAC/ASH)—these are large, meaningful margins on the harder full-spectrum benchmarks. The medical domain evaluation on eICU adds cross-domain credibility.

- **Compelling robustness to training set size (Table 6)**: On CIFAR-10, the KAN detector degrades only ~1 AUROC point from 100% to 0.1% of training data, while KNN collapses to 8.15% and VIM drops significantly. This practically important property is demonstrated with a clear ablation and plausible explanation rooted in the spline-locality mechanism.

- **Honest acknowledgment and fix of the marginal-distribution limitation**: Section 2.3 transparently identifies the core architectural weakness (KANs process features independently) and proposes a partitioning strategy backed by both a toy demonstration (Fig. 3) and a comprehensive ablation (Table 7). This demonstrates intellectual integrity and methodological depth.

---

## Weaknesses

### Fatal
*None.* The paper has genuine issues but they do not rise to the level of invalidating the core contribution.

### Major

- **Partitioning is essential, not supplementary, yet is framed as a "limitation workaround"**: Table 7 shows that P=1 (the base detector) achieves only 46.08 ± 15.58 AUROC on CIFAR-10—essentially random performance. The reported headline results depend entirely on P=10. This is a structural framing mismatch: the paper presents "local neuroplasticity" as the core mechanism, but the core mechanism without partitioning is nonfunctional on real benchmarks. The actual contribution is better described as a *partitioned ensemble of KAN detectors*. The paper should reframe its contribution accordingly or provide a rigorous analysis of why the base detector fails and how partitioning specifically recovers the local plasticity signal (rather than primarily acting as an ensemble).

- **The mechanistic explanation is validated only on toy data, not on real experiments**: The intuition that OOD samples activate untrained spline regions (Eq. 5, Fig. 2) is compelling for the 1D regression toy. However, the paper never demonstrates on real benchmarks (a) that spline coefficient changes remain spatially localized, or (b) that OOD samples systematically land in regions of different spline support than InD samples in high-dimensional backbone feature spaces. Fig. 4 shows distributions for three specific examples, which is insufficient. Without this validation, the mechanism claim is a plausible hypothesis rather than an established explanation.

- **Performance claims are slightly overstated in key places**: The abstract and introduction claim "superior performance… across all seven benchmarks." The paper's own tables show this is not accurate: on the Age benchmark (Table 4), KLM achieves 51.0 ± 0.7 vs. KAN's 50.5 ± 0.5 (KAN is not best); and on CIFAR-100 near-OOD average, KAN (77.17) is substantially behind RMDS (80.15), GEN (81.31), and KNN (82.40). The overall average win on CIFAR-100 is driven by far-OOD and is within variance of NAC. The claim should be narrowed to "best or near-best on most image benchmarks and competitive on medical benchmarks."

- **No computational cost analysis**: The method requires training P separate KAN models (P=10 in best CIFAR-10 configuration) plus one untrained reference, performing P+1 forward passes at inference, and storing P models. KANs are known to be significantly slower than MLPs (roughly 10× in training per the original KAN paper). The paper provides no comparison of training time, inference latency, or memory footprint against any baseline—neither zero-shot methods like Energy/MSP nor training-required ones like KNN or NAC. This omission is a significant gap for a post-hoc method claiming practical utility.

### Minor

- **Medical benchmark absolute performance is very low**: On the Age benchmark (Table 4), all methods cluster near 50% AUROC—essentially random. While the paper correctly notes these are hard near-OOD settings, claiming "strong cross-domain effectiveness" in this regime is an overclaim. The synthetic OOD benchmark (Table 5) is more favorable, but a discussion of *why* the method struggles at AUROC ≈ 50 would strengthen the analysis.

- **The MLP-comparison baseline is missing**: The most critical ablation to isolate the KAN-specific benefit would be a trained-vs-untrained MLP comparison using the same scoring framework. The histogram baseline (Sec. 3.3) partially addresses this but replaces only the spline with histograms, not testing whether an MLP trained on InD data versus an untrained MLP also separates InD/OOD. Without this, the claim that the KAN architecture (not just the trained-vs-untrained comparison paradigm) is essential remains unsupported.

- **Unclear partitioning strategy across benchmarks**: The paper discusses both class-label partitioning (for classification) and k-means clustering but does not clearly specify which was used for each of the seven benchmarks in the main results. This affects reproducibility and interpretation—class-label partitioning uses label information unavailable to most other post-hoc methods.

- **Near-OOD performance asymmetry is underemphasized**: The overall AUROC average conflates near and far OOD. On some benchmarks (CIFAR-100 near-OOD, ImageNet-200 FS near-OOD ≈ 59.74%), the method is materially weaker than some baselines. The paper mentions this only briefly; a more prominent discussion would set appropriate expectations.

### Trivial

- The training task ablation (regression vs. classification) shows only 0.2% improvement for images but 3% degradation for tabular data, yet the conclusion ("any training task yields a valid detector") is stated without qualification—this should be hedged.

---

## Nice-to-Haves

- **MLP-based trained-vs-untrained comparison**: Train an MLP on InD features with the same partitioning and scoring logic, and compare against KAN. This would directly validate whether the KAN architecture (local splines) is necessary or whether the paradigm itself is what drives performance.
- **Spline visualization on real backbone features**: Show the actual learned vs. untrained spline activation differences for a few representative features in the CIFAR or ImageNet setting. This would make the mechanistic claim much more credible.
- **Feature-space analysis**: A UMAP/t-SNE of backbone features colored by KAN InD scores would help readers understand when and why the detector succeeds or fails, especially for near-OOD.
- **FPR@95 in main tables**: This metric is often more operationally relevant than AUROC for OOD detection and is currently appendix-only.
- **Impact of histogram normalization ablation**: The paper introduces this preprocessing step without ablating it—including it as a proper ablation would strengthen the main claims.

---

## Removed Points

*These points were flagged for removal; treat with caution:*

**Removed – "Seamless integration with any pre-trained classifier"**: The harsh critic flagged this as an overclaim requiring evidence across many architectures. The paper uses two backbone families (ResNet and FT-Transformer) and is explicit that it uses latent features. This is a reasonable claim scope for a post-hoc method paper; the "any" is standard convention in the OOD literature and not a falsifiable empirical claim. **Removed as scope-appropriate.**

**Removed – "Regression-based experiments unsupported"**: The harsh critic noted regression results are in the appendix (removed from the review copy). Since appendices exist and the results are referenced explicitly, this is not an evidential failing—the claim is supported in the full paper. **Removed per hard rule on assuming cited materials exist.**

**Removed – Sensitivity to KAN initialization**: The paper explicitly addresses this in Appendix A.4, reporting that initialization-based variance is lower than backbone-based variance. The neutral reviewer's point about this is already addressed in the paper. **Removed as already addressed.**

**Removed – "Unclear contribution of partitioning vs. ensemble"**: The spark reviewer's suggestion to compare against an MLP ensemble is a valid *missing experiment* but does not constitute a proven weakness—it is a nice-to-have ablation. Moved to Nice-to-Haves.

**Removed – "Limited evaluation on diverse backbone architectures" (ViT etc.)**: This is a reasonable suggestion but standard criticism applied to virtually every OOD detection paper (e.g., NAC itself only uses ResNet). Evaluating the paper against its own community's norms, this is a nice-to-have. Moved to Nice-to-Haves.

---

## Novel Insights

The most genuinely novel insight is the use of **spline-coefficient locality as a geometric memory structure for OOD detection**: unlike MLP weights that are globally coupled, KAN spline coefficients record InD sample density in specific grid intervals, making the trained-vs-untrained comparison a form of implicit density estimation over piecewise-defined regions of feature space. The ablation showing an ~9% AUROC gap between KAN splines and a matched histogram baseline (85.29 vs. 94.12 on CIFAR-10) suggests that continuous spline interpolation (smoothing adjacent regions) provides substantially better generalization than binary occupancy counting—an insight that connects OOD detection performance to the functional regularity of the learned activation functions rather than just their support coverage. This is a conceptually fresh angle in OOD detection that goes beyond existing activation-pattern or feature-density approaches.

---

## Suggestions

1. **Reframe Section 2 and contributions**: Present the method as a "partitioned KAN ensemble detector" where the core unit is a single trained-vs-untrained KAN comparison, and partitioning is an essential architectural component (not a workaround). This aligns the framing with what Table 7 shows actually works.

2. **Add a trained-vs-untrained MLP ablation**: Use the same pipeline (same partitions, same scoring function, same aggregation) but replace KAN with a standard MLP. This is the single most important missing experiment.

3. **Add computational cost table**: Report training time (per partition, total), inference latency per sample, and memory for at least the CIFAR-10 and ImageNet-200 settings, alongside representative baselines (KNN, NAC, VIM).

4. **Show spline locality on real data**: Visualize the distribution of |c_trained - c_untrained| across grid points for the CIFAR setting, and show that OOD samples overlap less with high-delta regions than InD samples (quantitatively, not just for three exemplars).

5. **Narrow overclaim language**: Replace "across all seven benchmarks" with "on most image benchmarks and competitively on medical benchmarks," and qualify the data-size robustness claim to CIFAR experiments specifically.

---

## Score Calibration

**Comparison papers consulted:**
- **NAC (SNGXbZtK6Q)** – Accept (spotlight), scores 5/8/6/8 ≈ 6.75. Strong neuron-activation OOD paper with extensive baselines on three benchmarks. The current paper matches NAC's evaluation scope and has competitive or superior results against it.
- **HAct (Oo5spZRpH6)** – Reject, scores 3/5/3 ≈ 3.67. Similar concept (activation histograms for OOD), rejected due to missing baselines, unclear evaluation, and single-backbone evaluation.
- **Feature Map Matters (ZrY38sUYWs)** – Reject, scores 5/6/6/5 ≈ 5.5. OOD method with state-of-the-art claims but unclear disentanglement of contributions.
- **SCALE (RDSTjtnqCg)** – Accept (poster), scores 5/8/6/6 ≈ 6.25. Clean analysis-driven OOD method with strong ImageNet results.
- **Neural Collapse OOD (mUXdysoxEP)** – Accept (poster), scores 8/8/6/5 ≈ 6.75. Strong theoretical grounding + comprehensive empirical results on CIFAR+ImageNet.
- **Hyperspherical Energy (6sfRRcynDy)** – Reject, scores 3/6/5/5 ≈ 4.75. Rejected partly for missing baselines and marginal improvement over CE-trained methods.

**Assessment**: This paper is stronger than HAct (which it essentially beats methodologically—the KAN approach outperforms a histogram baseline by ~9%) and stronger than Hyperspherical Energy (which had weaker baselines and marginal gains). It is comparable to SCALE and Feature Map Matters in breadth, but materially stronger on hard ImageNet benchmarks. The major weaknesses (framing mismatch between base mechanism and partitioned method; no computational cost analysis; mechanistic claims on toy data only) are similar to the issues that pushed SCALE to 6.25 and Feature Map Matters to rejection at 5.5. However, the empirical results here are stronger and more broadly validated than Feature Map Matters, and the conceptual novelty (KAN-specific mechanism) is more interesting than SCALE (incremental over ASH). The robustness-to-dataset-size finding is a genuine novel empirical contribution absent from comparable papers.

Weighing: strong empirical results on image benchmarks (+), genuine conceptual novelty (+), cross-domain evaluation (+), serious framing/mechanistic gap (-), no compute analysis (-), overstated claims (-). I place this paper above Feature Map Matters (rejected at ~5.5) and comparable to SCALE (accepted at ~6.25), but below NAC/Neural Collapse (accepted at ~6.75) due to the unresolved framing and mechanistic validation gap. 

**Final score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>