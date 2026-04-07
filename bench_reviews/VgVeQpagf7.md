## Summary
This paper introduces SPS and SPS+, algorithms for generating differentially private synthetic datasets by distilling summary statistics from a sensitive dataset using a public pre-trained model. SPS+ achieves higher accuracy than DP-SGD on CIFAR-10 and CIFAR-100 at strict privacy budgets (ε=1), marking the first time a generation-based method surpasses gradient-based approaches in image classification, while enabling flexible downstream use like ensembling and federated learning without additional privacy cost.

## Strengths
- **State-of-the-art accuracy on standard benchmarks**: SPS+ outperforms DP-SGD on CIFAR-10/100 across multiple ε values, e.g., 96.2% vs. 94.8% on CIFAR-10 at ε=1 (Table 1), validating its core claim.
- **Demonstrated practical flexibility**: The synthetic dataset can be reused without extra privacy loss, enabling ensembling, federated learning (asynchronous, no synchronization rounds), and continual learning—capabilities often infeasible with DP-SGD due to composition constraints, as shown in Figures 2, 5, and Table 1.
- **Novel algorithmic contributions**: Key innovations include adapting D3S to DP by removing reliance on a privately trained model, privatizing only summary statistics, and introducing multistage clipping and grouped pseudo-classes that significantly boost performance in high-privacy regimes (Table 8).

## Weaknesses
- **Dependence on public pre-trained models**: Performance hinges on the availability of a relevant public model; while tested on domain-shifted data (CAMELYON17), extreme mismatches or lack of public data could limit applicability, and the method does not provide a fallback, affecting real-world deployment.
- **Incomplete justification for grouped pseudo-classes (GPC)**: The claim that GPC helps only through "optimization dynamics" (Section 4.2) is not substantiated with analysis or experiments, leaving a key component poorly understood and raising questions about its robustness.
- **Hyperparameter tuning under DP constraints**: Critical parameters like projection dimensions \(D_G, D_C\) are chosen arbitrarily for different settings (Table 10), and the paper does not discuss how to select them in a privacy-preserving way, which is essential for practical use where tuning consumes privacy budget.
- **Federated learning with overlapping client data**: The federated experiment assumes disjoint data partitions (Section 5.5); if client data overlaps, privacy guarantees would require composition across clients, an unaddressed scenario that impacts real-world applicability.

## Nice-to-Haves
- More detailed efficiency comparison with DP-SGD, including wall-clock time and memory usage on comparable hardware, to better contextualize trade-offs.
- Extension to additional image benchmarks like downsampled ImageNet to further validate scalability beyond CIFAR.
- Analysis of synthetic data quality beyond FID, e.g., per-class accuracy or diversity metrics, to characterize limitations more thoroughly.

## Removed Points
These points are flagged to be removed, treat them with caution:
- Formatting issues in figures (e.g., "Col1" in captions) are parser artifacts, not paper flaws.
- Criticism about missing experiments on non-image modalities is outside the paper's stated scope on image classification.
- Demand for deeper privacy-utility trade-off analysis using distributional metrics (e.g., Wasserstein distance) is not standard in DP image classification literature where accuracy is the primary metric.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Provide an intuitive explanation or simple experiment to illustrate why grouped pseudo-classes improve optimization but not direct mean estimation, enhancing methodological clarity.
- Discuss strategies for hyperparameter selection under DP, such as using public validation data or allocating a small privacy budget for tuning, to guide practitioners.
- Clarify the privacy implications for federated learning when client data may overlap, and suggest how SPS could be adapted (e.g., via centralized generation with composed guarantees).