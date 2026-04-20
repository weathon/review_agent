Now let me write the final review.

## Summary

The paper proposes a novel out-of-distribution (OOD) detection method that exploits the local neuroplasticity property of Kolmogorov-Arnold Networks (KANs). By comparing activation differences between a trained KAN and its untrained counterpart, the method detects whether a sample activates regions that were adapted during training (InD) or unadapted regions (OOD). The approach operates post-hoc on latent features from any pretrained backbone, partitions the training data to approximate joint distributions via k-means clustering, and achieves state-of-the-art AUROC across seven diverse benchmarks, including large-scale ImageNet and tabular medical datasets. A distinctive property is near-invariant performance even when training data is reduced to 0.1%.

## Strengths

- **Novel mechanism leveraging KAN architecture for OOD detection**: The core idea—using the difference between trained and untrained KAN spline activations as an InD score—is a creative and well-motivated application of KAN local neuroplasticity. Equation 5 provides a clear decomposition separating "where InD information is stored" (coefficient differences) from "which regions a sample activates" (B-spline mask), making the mechanism interpretable.

- **Strong empirical results across diverse benchmarks**: The KAN detector achieves the best overall AUROC on every tested benchmark: CIFAR-10 (94.12%), CIFAR-100 (83.44%), ImageNet-200 FS (71.46%), ImageNet-1K FS (78.52%), and three tabular medical datasets. On ImageNet-200 FS, the margin over the next-best method is substantial (7.83 AUROC points). The evaluation follows the OpenOOD protocol with appropriate baselines including NAC, the current best post-hoc method on the CIFAR leaderboard (Section 3.2, Tables 1–2).

- **Distinctive robustness to training dataset size**: Table 6 shows the KAN detector maintains 93.21% AUROC at 0.1% of CIFAR-10 training data (~50 samples), versus 94.12% at full data, while competing methods degrade sharply (KNN drops to 8.15%, VIM to 76.38%). This is a practically valuable property for deployment in data-limited settings, and it stems naturally from the local plasticity mechanism rather than being a post-hoc engineered robustness.

- **Self-identified limitation with an empirically validated workaround**: The paper correctly identifies that individual KAN activations process features independently (Section 2.3), capturing only marginal distributions. The k-means partitioning solution is validated on the L-shaped toy dataset (Figure 3) and quantified in Table 7, where increasing partitions from $\mathcal{P}=1$ (46.08% AUROC) to $\mathcal{P}=10$ (94.12%) recovers the joint distribution signal. The paper also explores feature-augmentation alternatives (PCA, autoencoders) in Section 3.3 and reports that partitioning outperforms them on high-dimensional benchmarks.

## Weaknesses

### Fatal
None.

### Major

- **The necessity of the KAN architecture over standard density estimators is not sufficiently established.** The paper demonstrates that KAN splines (94.12% AUROC) outperform a binary histogram baseline (85.29%) on the same features (Section 3.3), which is a useful result. However, this histogram baseline is a simple binary presence counter—not a smoothed, bandwidth-tuned histogram, let alone KDE or Gaussian Mixture Models. The method's partitioning strategy (k-means on features) and its median-of-differences scoring are architectural choices that could plausibly help *any* density estimator. Without showing that KAN-specific spline smoothing is necessary for the reported gains—as opposed to the partitioning scheme, the aggregation strategy, or the histogram normalization preprocessing—it remains unclear whether the KAN architecture is the primary driver of performance or whether a well-tuned KDE/GMM on partitioned features would achieve comparable results. This gap weakens the paper's core claim that the KAN architecture uniquely enables this detection mechanism.

- **The claim that learned spline plasticity drives OOD discrimination is partially undermined by the data-efficiency results.** Table 6 shows near-identical performance at 0.1% (93.21%) and 100% (94.12%) of CIFAR-10 training data. ~50 training samples can only modify a tiny fraction of the spline grid across a 512/2048-dimensional feature space. If the method truly relies on the trained network's adapted spline coefficients $\Delta_{p,q}$ distinguishing InD regions, this near-flat curve suggests the untrained initialization and the scoring pipeline (histogram normalization + median aggregation) contribute a dominant share of the detection signal. This is not necessarily a flaw—it may indicate the method is inherently initialization-heavy, which could be a feature—but it contradicts the framing that "training-induced local plasticity" is the primary mechanism, and this tension is not discussed.

### Minor

- **Computational cost of partitioning is not analyzed.** The method trains $\mathcal{P}$ separate KAN models (one per partition) and runs inference through each. While Section 2.3 notes the untrained KAN can be shared across partitions, the number of forward passes per sample still scales linearly with $\mathcal{P}$. No training time, inference latency, or GPU memory comparisons are provided against training-free baselines like KNN, VIM, or NAC. Given that the method is proposed as a deployment-ready post-hoc detector, the practical viability depends on this trade-off. For context, KNN operates on the backbone features alone without any additional model training.

- **The regression-to-constant training task ablation is under-discussed despite its significance for mechanistic interpretation.** Section 3.3 reports that training the KAN with a regression-to-constant task yields ~0.2% *improvement* on image benchmarks and ~3% *decrease* on tabular data compared to the original classification task. This result suggests the specific training task has limited influence—the detector primarily registers feature distributions rather than learning task semantics. This finding should be foregrounded in the mechanism discussion (Section 2.2) rather than presented as an afterthought, as it clarifies what "learning" the KAN is actually doing.

### Trivial

- **Table 7 uses "$k$" for grid size in the column header, while the paper uses "$G$" throughout the text (e.g., Section 2.1: "grid size  $G$ ").** This notation inconsistency is minor but worth correcting.

## Nice-to-Haves

- Report inference latency and parameter counts per partition $\mathcal{P}$ to give practitioners a sense of the deployment cost.
- Provide t-SNE/UMAP visualizations of backbone features colored by the KAN InD score to confirm that high scores correlate with data density rather than artifact regions from histogram normalization.
- Briefly clarify the histogram normalization range mapping and how grid boundaries are set—this is an important preprocessing step that affects downstream performance.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Removed (misreads the mechanism)**: The harsh critic argues that "near-identical performance [at 0.1% vs 100%] directly invalidates the proposed learning mechanism and its causal link to the results." However, the KAN architecture's local neuroplasticity means only spline coefficients near training samples are modified—this is a *feature*, not a bug. With 50 samples, the network still registers the InD regions that those samples activate. The stability across dataset sizes is an empirical strength (as reflected in the strength section), not a contradiction. The critic's assumption that "0.1% should underfit" incorrectly applies MLP-style global learning intuition to a locally plastic architecture.

- **Removed (partially addressed in paper)**: "The paper compares against a single histogram baseline but does not compare against KDE, GMM, or smoothed histograms." The paper *does* include a histogram baseline (85.29% vs 94.12% AUROC in Sec 3.3), which validates that the spline smoothing operation contributes meaningfully. While comparisons with KDE/GMM would strengthen the claim (and this concern is retained in Major), the criticism that "no non-neural baselines" were tested overstates the gap and ignores the histogram ablation.

- **Removed (strawman / misreads)**: "The method fundamentally reduces to univariate marginal feature matching... the partitioning workaround is mathematically insufficient for high-dimensional feature spaces." The paper acknowledges the marginal-only property (Section 2.3, line 96), proposes partitioning, and empirically validates it achieves $\mathcal{P}=10$ AUROC matching the full benchmark performance on CIFAR-10 (Table 7: 94.12%). The curse-of-dimensionality concern is theoretical but the empirical evidence contradicts it for the evaluated feature dimensions. The critique that partitioning "inherently creates a union-of-detectors bias" is a design trade-off the authors already address: "Since the partitions are non-overlapping, for InD samples, there will be only one model that recognises the sample as InD" (Section 2.3, line 110).

- **Removed (overstated scope concern)**: "Max-aggregation across partitions increases false negatives for OOD samples that align with any single partition's marginal regions." This is a valid precision-recall trade-off inherent to any max-based ensemble. However, the paper's results show strong performance on near-OOD datasets (e.g., CIFAR-10 near-OOD avg: 91.64% AUROC), indicating the max operator does not systematically fail in practice. Requesting a precision-recall analysis is a nice-to-have but does not undermine the paper's claims.

- **Removed (nitpic about median computation details)**: The critic's request for "how the median is computed across a matrix of size $n_{in} \times n_{out} \times \mathcal{P}$" is answered implicitly: $\Delta$ is a 3D array, flattening it and taking the median produces a scalar (Section 2.2, Eq. 4). The paper also notes this choice is motivated in Appendix A.1.

- **Removed (unfounded reproducibility nitpick)**: "Without explicit details (range mapping, bin/spline boundaries), the results are not reproducible." The paper states that hyperparameters are tuned on the validation set per OpenOOD guidelines (Section 3.1, line 148). Histogram normalization is a standard technique. This is a minor implementation detail, not a reproducibility blocker.

## Novel Insights

The paper's most genuinely novel observation is that KANs' local neuroplasticity—a property that makes them resistant to catastrophic forgetting in continual learning—can be repurposed as a mechanism for OOD detection without any architectural modifications to the KAN itself. The insight that the *difference* between a trained and untrained network's activations (rather than the trained network's raw outputs) serves as the detection signal is non-obvious and distinguishes this approach from standard post-hoc methods. The data-efficiency finding (Table 6) further suggests that local plasticity may be a more natural and low-data mechanism for distribution registration than the global distance-to-centroid or density-estimation approaches used by competing methods.

## Suggestions

1. Add a direct baseline comparing KAN-based detection with a properly tuned KDE or GMM density estimator applied to the same partitioned features. This would isolate the contribution of the KAN spline architecture from the partitioning/aggregation strategy.
2. Include an ablation that replaces the trained KAN with its untrained initialization alone (i.e., compute the score without any training) to quantify how much of the detection signal comes from learning versus the initial spline configuration and normalization pipeline.
3. Expand the discussion in Section 2.2 to explicitly address the data-efficiency property observed in Table 6 and clarify the relative contribution of initialization vs. learned plasticity to the final score.

## Score and Decision

I calibrated against the following human-review anchors:

- **High-scoring accept (8,8,6,8 avg ~7.5)**: NegLabel (xUO1HXz4an.md) — novel post-hoc OOD detection with VLMs, theoretical grounding, extensive experiments. This paper is below NegLabel because it lacks theoretical analysis and has the unresolved question of KAN necessity vs. standard density estimation.
- **Medium-scoring accept (6,5,6,8 avg ~6.3)**: AROS (GrDne4055L.md) — OOD detection with NODEs and Lyapunov stability, comprehensive benchmarks, ablation studies. This paper is comparable in empirical strength but lacks the theoretical grounding that AROS provides.
- **Borderline accept/reject (5-6 avg)**: Papers like UTnq6hJJYa (avg 5.3, rejected) and RDSTjtnqCg (avg 6.3, mostly accepted) — OOD detection methods with good results but missing baselines or unclear mechanisms. This paper is stronger than those in terms of benchmark comprehensiveness and empirical consistency.
- **Low-scored reject (3,3,3,3 avg 3)**: KAE (K9xuqsaP0R.md) — KAN applied to autoencoders without clear benefit over baselines, missing comparisons. This paper is substantially stronger, with genuine SOTA results across 7 benchmarks.

This paper sits between the medium and high-scoring OOD papers. It has genuinely strong empirical results that are hard to dismiss, a genuinely novel mechanism, and a distinctive data-efficiency property. However, the missing KDE/GMM baseline comparison and the under-discussed tension between the claimed "learned plasticity" mechanism and the near-invariant performance at 0.1% data size prevent it from reaching the level of the highest-scoring papers. It is a solid contribution that advances the state of the art empirically, even if the theoretical and ablation depth could be improved. Compared to the borderline papers that were rejected, this paper has stronger results and a clearer mechanism.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>