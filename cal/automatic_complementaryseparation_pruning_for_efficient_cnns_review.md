=== CALIBRATION EXAMPLE 17 ===

# Harsh Critic Review
Now I have a comprehensive view of the paper. Let me compose a thorough critical review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is appropriately descriptive. The abstract's headline claims are broadly consistent with the experimental results: 2.25× speedup on ResNet-50, 1.5–2.5× FLOPs reduction, automation without manual tuning. However, the abstract conflates **FLOPs-based speedup** with **actual inference-time speedup**. Table 2 reveals wall-clock reductions of only 6–20% (i.e., roughly 1.06–1.25× actual speedup), not 2.25×. This is a significant misrepresentation that should be corrected or at least clearly flagged in the abstract.

---

### Introduction & Motivation

The motivation is clearly articulated: manual pruning ratio specification is impractical, and most methods impose it. The claim that ACSP "selects the pruning extent automatically in a single pass per layer using a data-driven knee-finding approach" is misleading. In reality, Algorithm 1 runs k-Medoids for *every* k from 2 to Nᵢ before applying Kneedle—that is O(Nᵢ) full clustering runs, not a single pass. The paper's own complexity analysis (O(Nᵢ²) for Kneedle) omits the dominant cost of the k-Medoids sweeps.

The four stated contributions are a fair description of what the paper does, though Contribution 1 ("fully automates...without manual intervention") oversells novelty slightly—AutoPrune (Xiao et al., 2019), AMC (He et al., 2018b), and MetaPruning (Liu et al., 2019) also automate the pruning ratio.

---

### Method (Section 3)

**Graph Space Construction (§3.3.1)**

The core idea—encoding each channel's separability across all class pairs into a vector and then clustering channels to enforce diversity—is intuitive and has an analogy in feature selection (GB-AFS, Levin & Singer 2024). However, several technical concerns arise:

1. **Scalability for large C is a critical unaddressed problem.** The graph-space vector for each component has dimensionality p² × C(C−1)/2. For ImageNet with C = 1,000 classes, this yields 499,500 class pairs. For a mid-level convolutional layer with spatial dimension p = 7, each component's representation has ~24.5 million dimensions. The paper claims to have run experiments on ImageNet-1K (ResNet-50, MobileNet-V2), but *does not explain how this combinatorial explosion is handled*. The limitations section acknowledges scaling issues but offers no solution used in practice. This is a critical methodological gap—the reviewer cannot verify that the ImageNet experiments are computationally feasible as described.

2. **Gaussian assumption for activations.** Equations (1) and (2) use the Bhattacharyya/JM distance formula that assumes activation values are *Gaussian-distributed* within each class. This is never validated. ReLU activations are non-negative and typically right-skewed or bi-modal (mass at zero + positive tail). Applying a Gaussian-parametric distance to such distributions is questionable and could systematically misestimate separability.

3. **Per-pixel treatment of convolutional feature maps.** The paper computes JM distances pixel-by-pixel across the p × p activation map (§3.3.1, convolutional case). This ignores spatial correlations within the feature map. Two channels that activate different spatial regions but have similar per-pixel distributions would be treated as identical in graph space. A pooling or summary statistic would be more principled.

**Graph Space Evaluation and k-Medoids (§3.3.2)**

4. **Medoid vs. highest-weight component inconsistency (§3.4.2).** Algorithm 1, line 12, selects "top-k' components by weight" (L1 norm within each cluster). But the MSS-based k optimization in lines 7–10 uses cluster medoids to score coverage. These are different components: the MSS score reflects the medoid set's coverage, while the final selected set uses weight-dominant components from each cluster. There is no guarantee that weight-dominant components maintain the same inter-cluster diversity that justified the MSS score. An ablation comparing medoid selection vs. weight-based selection is needed to establish whether this "hybrid" approach is justified.

5. **Running k-Medoids for all k ∈ {2, …, Nᵢ} is O(Nᵢ) full clustering runs.** For a layer with Nᵢ = 512 channels (common in VGG/ResNet), this means 510 independent k-Medoids runs in a high-dimensional space. The paper dismisses overhead by saying "Kneedle runs in O(Nᵢ²)" and takes under 0.1 s, but k-Medoids is not O(1)—its cost dominates. *What is the actual wall-clock time for the graph-space construction and clustering sweep per layer?* This should be reported.

**Automatic Pruning Volume (§3.4.1)**

6. **Sensitivity of Kneedle.** The Kneedle algorithm requires a polynomial degree parameter ("second-degree polynomial" is mentioned for experiments). How sensitive are results to this choice? Is there guidance on when a different degree should be used?

---

### Experiments & Results (Section 4)

**Baseline Fairness**

7. **Inconsistent fine-tuning regimes across baselines.** ACSP fine-tunes after *each* pruned layer (many rounds of fine-tuning), while most compared methods (DepGraph, HRank, Network Slimming, etc.) fine-tune once at the end. This makes the comparison unfair in ACSP's favor, as iterative layer-wise fine-tuning amounts to a significantly larger optimization budget. Total training epochs consumed by ACSP are never reported.

**Accuracy Improvements**

8. **Unexplained accuracy gains after pruning.** ACSP repeatedly shows accuracy *improvements* (e.g., +0.66% on ResNet-50, +0.62% on VGG-19 CIFAR-100). This could simply reflect the regularization benefit of fine-tuning—not the pruning criterion. Without a control experiment comparing "fine-tuning the same model *without* any pruning" or "pruning randomly with the same amount of fine-tuning," it is impossible to attribute these gains to the ACSP selection principle. This is a significant missing ablation.

**FLOPs vs. Latency Discrepancy**

9. **The gap between reported FLOPs speedup and actual latency improvement is enormous.** Table 1 reports 2.25× FLOPs speedup for ResNet-50, but Table 2 shows only 6.32% (batch) and 8.07% (single) latency reduction—approximately 1.07×. The paper notes "hardware utilization is not perfectly linear with FLOP count," but a 30× discrepancy between FLOPs and wall-clock speedup is not a rounding issue. This suggests the pruned models are not truly achieving dense computation in hardware—possibly because of small residual channel counts that still trigger the same CUDA kernel launches, or because the pruning operates on layers where FLOPs are not the bottleneck. This fundamentally undermines the practical deployment claim. At ICLR's bar, reporting 2.25× speedup in the abstract and conclusions without acknowledging that *actual* speedup is ~1.07× is a serious accuracy-of-claims issue.

**Missing Ablations**

10. The paper claims JM is superior to Hellinger and Wasserstein distances (§3.3.1), but the metric comparison results are only alluded to ("detailed in the experiments section") yet never shown in a proper table. Where is this ablation?

11. There is no ablation on the "complementary" selection principle vs. simple top-k by weight or top-k by separability. The central claim—that complementary diversity improves pruned accuracy—is never directly tested.

12. The effect of the fine-tuning fraction (25% of data, 2 epochs) is not ablated. How sensitive is performance to this choice?

**Statistical Significance**

13. Results are reported as single runs with no error bars or standard deviations over multiple trials. Differences of ±0.1–0.3% in accuracy are being used to rank methods, but without confidence intervals, these rankings are not statistically reliable.

---

### Writing & Clarity

14. **"Graph space" terminology is misleading.** Throughout the paper, "graph space" refers to the Euclidean embedding of components by their separability vectors. No graph structure (nodes, edges, adjacency) is defined or used. The clustering and Kneedle algorithms operate purely on this vector space. Calling it a "graph space" borrows credibility from graph-theoretic machinery without actually using it; "feature space" or "separability embedding space" would be more accurate.

15. Section 3.4.2 introduces weight-based selection only after the MSS analysis that implicitly assumes medoid selection. The relationship between these two steps is not clearly motivated until several paragraphs in, creating confusion about the actual algorithm. Figure 2's caption helps but the text logic is circuitous.

---

### Limitations & Broader Impact (Section 5)

16. The paper appropriately identifies scaling with large C as a limitation. However, it does not quantify the current computational cost: How long does pruning ResNet-50 on ImageNet take? How does this compare to one-shot methods like DepGraph?

17. There is no discussion of applicability to transformers, attention heads, or non-CNN architectures, which are increasingly the deployment target. The method is explicitly "tailored for supervised learning tasks" with class-pair structure—this is a meaningful architectural constraint that limits the paper's reach.

18. The method requires the full training dataset during pruning (activation computation for all samples). This could be a significant practical barrier in privacy-sensitive settings; data-free or synthetic-data approaches (DFPC etc.) are not discussed as an alternative regime.

---

### Overall Assessment

ACSP proposes a technically interesting pruning framework that combines separability-based graph-space clustering with an automatic knee-finding procedure, drawing legitimately on GB-AFS ideas (Levin & Singer, 2024). The coverage of architectures and datasets is broad, and the results are generally competitive. However, the paper has several serious problems that prevent acceptance at ICLR in its current form. The most critical is the **scalability gap**: the ImageNet experiments require computing 499,500 class-pair JM distances per component per layer, and the paper offers no explanation of how this is made tractable—leaving a core result unreproducible. Second, the **FLOPs-vs.-latency discrepancy** (2.25× claimed vs. ~1.07× actual for ResNet-50) is not adequately addressed and undermines the practical justification. Third, the **fairness of comparisons** is compromised by unequal fine-tuning budgets, and the **accuracy-gain claims** lack a crucial fine-tuning-only baseline. Finally, the paper is missing key ablations (complementary selection vs. weight-only, metric comparison table, fine-tuning sensitivity) that are essential to verify that the proposed components are individually contributing. These gaps collectively make it difficult to assess the true source of the method's gains and whether they generalize beyond the specific architectures tested.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Automatic Complementary Separation Pruning (ACSP), a method for automatically pruning convolutional neural networks by constructing a graph space encoding the separability of network components across class pairs. ACSP leverages clustering techniques and a knee-finding algorithm to select a complementary subset of channels or neurons, eliminating the need for manual pruning ratio specification. Extensive experiments on CIFAR and ImageNet datasets demonstrate that ACSP achieves significant inference speedups with minimal accuracy loss, often outperforming manual-tuning-based baselines in computational efficiency.

### Strengths
1.  **Fully Automated Pruning Ratio:** ACSP effectively addresses the labor-intensive manual tuning of pruning ratios common in existing structured pruning methods. By utilizing the Mean Simplified Silhouette (MSS) index and a knee-finding algorithm (Kneedle), the method automatically determines the optimal subset size for each layer without user intervention. This is a significant practical contribution for practitioners who lack hyperparameter tuning resources.
2.  **Complementary Selection Mechanism:** The approach intelligently targets redundancy by evaluating components based on their "separation capability" using Jeffries-Matusita (JM) distance across all class pairs. By selecting components from diverse regions of the graph space (via k-Medoids clustering), the method ensures the retained channels offer distinct and complementary information, rather than simply dropping weights with the lowest magnitude. This theoretical grounding in feature diversity is compelling.
3.  **Strong Empirical Performance:** The paper provides comprehensive evaluations across diverse architectures (VGG, ResNet, DenseNet, MobileNet) and datasets (CIFAR-10/100, ImageNet-1K). Results consistently show that ACSP matches or exceeds the accuracy of baselines like DepGraph and AMC while achieving competitive FLOP reductions (e.g., 2.15× speedup on ResNet-56). The inclusion of inference latency measurements (Table 2) further solidifies the practical utility beyond just FLOP estimates.

### Weaknesses
1.  **Scalability with Class Count:** The methodology relies on computing separability for all class pairs, which implies a complexity scaling quadratically with the number of classes ($O(C^2)$). While the authors mention this in the Conclusion, they claim the overhead is "negligible." For large-scale classification tasks like ImageNet ($C=1,000$), computing the separability matrix for every channel might introduce significant overhead compared to magnitude-based heuristics. The claim that pruning "deep" networks costs <0.1s per layer warrants more robust verification in the experimental section.
2.  **Comparison Granularity:** In Table 1, the baselines are compared based on their "best" speedup/accuracy trade-offs rather than at a fixed FLOP budget. For instance, DepGraph on ResNet-56 achieves 2.11× speedup with +0.24% accuracy, while ACSP achieves 2.15× with +0.13%. Without a Pareto frontier analysis or fixed FLOP setting, it remains unclear if ACSP is strictly superior or simply pruned to a different density.
3.  **Fine-Tuning Cost Justification:** The method requires a "short fine-tuning process" (e.g., 2-3 epochs) for each layer (or potentially the whole model per layer pruning). While the authors state this adds negligible cost, fine-tuning is the dominant computational overhead in pruning pipelines. The paper should quantify this total pipeline cost (graph construction + fine-tuning) versus the inference gain, particularly for scenarios where the inference gain is modest.
4.  **Robustness of Knee-Finding:** Relying on the Kneedle algorithm to determine the pruning volume automatically assumes the MSS curve possesses a distinct "knee." In cases where the graph clustering quality degrades smoothly without a sharp transition, the automatic selection might fail to identify the optimal point, potentially leading to under-pruning or over-pruning without user oversight to adjust the MSS metric weights.

### Novelty & Significance
**Novelty:** The integration of graph-based feature selection (specifically complementary selection via separability metrics) into structured pruning is a novel angle. While graph-based dependency pruning exists (e.g., DepGraph), ACSP’s focus on *class-pair separability diversity* distinguishes it from standard magnitude-based or dependency-based approaches. The automatic determination of the pruning level via clustering indices adds a further layer of distinctness.

**Significance:** The significance is high for the application domain of automated model compression. Reducing the need for manual hyperparameter tuning significantly lowers the barrier to deploying optimized models on resource-constrained devices. The method’s hardware-friendly structure (channels/neurons) ensures real-world applicability. However, the scalability concerns regarding the class count limit its immediate significance for extremely large-scale classification problems unless approximations are adopted.

### Suggestions for Improvement
1.  **Address Class Scaling:** Explicitly analyze the pruning time overhead relative to $C$. If the overhead is prohibitive for $C > 100$, propose or validate the sampling strategy mentioned in the conclusion (class-pair sampling) to ensure the method remains practical for large-scale tasks.
2.  **Standardize Comparison Baselines:** Provide a comparison where all methods are constrained to the same FLOP reduction target (e.g., 50% or 70% removal). This would clarify whether ACSP’s superior speedup is due to more aggressive pruning or better retention of accurate components.
3.  **Quantify End-to-End Pipeline:** Report the total computational time and energy cost required to apply ACSP (including graph construction and fine-tuning) compared to the baseline gain. This contextualizes the "negligible overhead" claim against a fixed performance budget.
4.  **Ablation on Separation Metrics:** The paper mentions JM, Hellinger, and Wasserstein distances were evaluated. A dedicated ablation study showing the impact of the specific separability metric choice on the final pruning performance would strengthen the empirical contribution beyond just the final accuracy claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Pruning Process Runtime:** Quantify the wall-clock time required to construct the separation graph for ImageNet, as the $O(C^2)$ class-pair calculation is computationally prohibitive for 1000 classes. Without this data, the claim that ACSP is "practical for real-world applications" is unverifiable and likely misleading.
2. **Matched FLOPs/Sparsity Comparisons:** Compare accuracy against baselines at identical FLOP reduction ratios rather than different speedups (e.g., 1.55x vs 1.41x in Table 1). Current tables conflate efficiency gains with accuracy gains, making it impossible to determine if ACSP is truly superior or simply less pruned.
3. **Metric Ablation Study:** Provide a quantitative table comparing JM distance against Hellinger and Wasserstein distances as claimed in Section 3.3.1. Without evidence that JM is statistically superior, the core graph construction design appears arbitrary.

### Deeper Analysis Needed (top 3-5 only)
1. **FLOPs vs. Latency Discrepancy:** Explain why ~50% FLOP reduction (Table 1) yields only ~5-20% latency reduction (Table 2). This gap suggests memory-bound bottlenecks or inefficient layer shapes that directly contradict the paper's primary contribution of inference efficiency.
2. **Hyperparameter Sensitivity for "Automation":** Analyze sensitivity to Kneedle parameters and fine-tuning schedules (epochs, LR) to validate the "no manual tuning" claim. If performance collapses under slight variations of these hidden hyperparameters, the method is not truly automated.
3. **Component Diversity Verification:** Quantify the correlation between kept channels to prove the "complementary selection" claim. There is currently no metric demonstrating that the selected subset is less redundant than a standard magnitude-based pruning subset.

### Visualizations & Case Studies
1. **MSS Knee-Point Curves:** Plot the MSS scores vs. subset size $k$ for representative layers to show the knee point. If the curve is flat or the knee is ambiguous, the automatic volume selection is unstable and unreliable.
2. **Channel Selection Heatmaps:** Visualize the indices of kept channels across layers compared to L1-norm pruning. This would reveal if ACSP truly selects diverse regions or simply converges to high-magnitude weights despite the graph mechanism.
3. **Graph Space Projection:** Provide a t-SNE projection of the component graph space to verify if components actually cluster by separability capabilities. This is necessary to validate the fundamental assumption that the graph space encodes meaningful complementary information.

### Obvious Next Steps
1. **Total Compute Cost Report:** Include the GPU hours required for the pruning process plus fine-tuning versus training from scratch. Efficiency claims must account for the cost of finding the efficient model, not just the final inference cost.
2. **Downstream Task Evaluation:** Evaluate pruned models on object detection (COCO) or segmentation to verify utility beyond classification. Channel diversity often impacts dense prediction tasks more severely than image classification accuracy.
3. **Class-Pair Approximation Analysis:** Experiment with class-pair sampling rates to determine the minimum sampling required to maintain performance given the admitted $O(C^2)$ bottleneck. This is essential for scaling the method to datasets with more classes than ImageNet.

# Final Consolidated Review
## Summary
ACSP introduces a structured pruning method that automatically determines pruning ratios by constructing a separability embedding for each network component across class pairs, then selecting complementary subsets via k-Medoids clustering and knee-finding. The method eliminates manual pruning ratio specification while maintaining accuracy across CIFAR and ImageNet benchmarks on VGG, ResNet, DenseNet, and MobileNet architectures.

## Strengths
- **Fully automated pruning ratio determination.** ACSP removes the need for manual tuning of pruning ratios through its MSS-based knee-finding approach, addressing a genuine practical pain point in existing pruning methods that require user-specified compression targets.
- **Principled complementary selection mechanism.** Unlike magnitude-based pruning, ACSP explicitly enforces diversity among retained components by selecting from different regions of the separability embedding. This theoretical grounding in feature diversity distinguishes it from standard filter importance heuristics.
- **Comprehensive empirical coverage.** The paper evaluates across 7 architecture-dataset combinations (CIFAR-10/100 with VGG-16/19, ResNet-56, DenseNet-40, MobileNet-V2; ImageNet with ResNet-50, MobileNet-V2) and includes actual latency measurements in Table 2, going beyond FLOP-only reporting.

## Weaknesses
- **Critical scalability gap for large class counts.** The separability matrix dimension scales as O(C² × p²) per component. For ImageNet (C=1000), this requires computing ~499,500 class-pair distances per component per layer. The paper claims ImageNet experiments were conducted but provides no explanation of how this computational explosion was handled—whether via sampling, approximation, or brute-force computation. This leaves a core experimental result unverifiable.
- **FLOPs-latency discrepancy undermines practical claims.** Table 1 reports 2.25× FLOPs reduction for ResNet-50, but Table 2 shows only 6–8% latency improvement (~1.07× speedup). The abstract and conclusion emphasize "faster inference time" and "significant speed-ups" without adequately explaining this 15–30× gap between theoretical and realized gains. For deployment-focused work, actual throughput matters more than FLOP counts.
- **Missing core ablations on method components.** Three critical ablations are absent: (1) comparison of JM distance against Hellinger and Wasserstein, which Section 3.3.1 claims was evaluated but never tabulates; (2) complementary selection vs. simple top-k by weight or separability score, which is the central claim of the paper; (3) accuracy gains from fine-tuning alone without pruning, to isolate whether the +0.5–0.6% improvements come from ACSP's selection or simply from the regularization effect of the fine-tuning schedule.
- **Unequal fine-tuning budgets across compared methods.** ACSP applies fine-tuning after pruning each layer (iteratively), while baselines like DepGraph, HRank, and Network Slimming typically fine-tune once after all pruning. The total fine-tuning budget consumed by ACSP across all layers is never reported, making efficiency comparisons potentially unfair.
- **No fixed-FLOP baseline comparisons.** Table 1 compares methods at their respective "best" speedup/accuracy trade-offs rather than at fixed FLOP budgets. For example, MobileNet-V2 shows ACSP at 1.55× and SANP at 1.41× speedup—these are different compression levels. Without Pareto frontier analysis or fixed-sparsity comparisons, it remains unclear whether ACSP's accuracy advantage stems from better component selection or simply from pruning less aggressively.
- **k-Medoids sweep computational cost not reported.** The method runs k-Medoids clustering for all k ∈ {2, …, Nᵢ} before applying Kneedle. The paper states Kneedle takes <0.1s but omits the dominant k-Medoids sweep cost. For layers with Nᵢ = 512 channels, this is 510 independent clustering runs in a high-dimensional space—wall-clock time should be quantified.

## Nice-to-Haves
- **Total pipeline cost quantification.** Report GPU hours for the complete pruning pipeline (graph construction + k-Medoids sweeps + fine-tuning across all layers) to contextualize whether the automation overhead is acceptable.
- **Downstream task evaluation.** Testing pruned models on object detection or segmentation would verify whether the complementary selection principle transfers beyond classification.
- **Class-pair sampling experiments.** Given the acknowledged O(C²) bottleneck, empirical validation of how much class-pair subsampling is viable would strengthen claims of practical applicability.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Gaussian distribution assumption for activations.** The JM distance formula assumes Gaussian distributions, and ReLU activations are typically non-Gaussian. However, JM is commonly used as a robust separability metric regardless of exact distribution, and the empirical results suggest this works adequately in practice. This is a theoretical concern that doesn't invalidate the method.
- **Per-pixel treatment ignores spatial correlation.** Computing JM distances pixel-by-pixel on convolutional feature maps ignores spatial structure. While true, the flattened representation still captures separability information, and the empirical success suggests this simplification is acceptable.
- **"Graph space" terminology is misleading.** The paper uses "graph space" but defines only a Euclidean embedding—no graph structure (nodes, edges, adjacency) exists. This is a terminology nitpick; the method remains valid as a clustering-based selection approach.
- **Kneedle polynomial degree sensitivity.** Requiring a polynomial degree parameter slightly undermines "fully automated" claims. However, this is a minor hyperparameter compared to manual pruning ratio specification, and reasonable defaults exist.
- **Statistical significance not reported.** Results lack error bars across multiple runs. While good practice, this is common in the pruning literature and doesn't invalidate the results given the consistent trends across architectures.
- **No discussion of transformer applicability.** The method is designed for CNNs with class-pair separability structure. Criticizing lack of transformer coverage is scope creep—the paper explicitly targets supervised CNN pruning.

## Novel Insights
The complementary selection principle—ensuring retained components cover diverse regions of the separability embedding rather than simply selecting highest-magnitude channels—is an underexplored idea in structured pruning. Most existing methods rank by importance (magnitude, gradient, or task-specific metrics) and cut at a threshold; ACSP's clustering-based diversity mechanism could theoretically better preserve model capacity at equivalent compression levels. However, the paper never directly validates this mechanism against simpler baselines (e.g., top-k by separability score without clustering), leaving the core theoretical contribution empirically unsubstantiated.

## Suggestions
1. **Add fixed-FLOP comparisons.** Re-run baselines or report accuracy at matched FLOP reduction targets (e.g., 50%, 70%) to enable fair comparison.
2. **Report the missing ablations.** Include tables for: (a) JM vs. Hellinger vs. Wasserstein; (b) complementary selection vs. top-k by weight vs. top-k by separability; (c) fine-tuning-only baseline (same epochs, no pruning).
3. **Clarify ImageNet computational tractability.** Explain how the C² class-pair computation was handled—whether full computation was feasible, or whether approximations were used.
4. **Report total pruning time.** Include wall-clock time for the complete ACSP pipeline (not just per-layer Kneedle) to contextualize the automation benefit against its cost.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0, 2.0]
Average score: 1.3
Binary outcome: Reject
