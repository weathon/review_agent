## Summary
This paper proposes TTVD, a geometric framework for test-time adaptation that reformulates neighbor-based methods using Voronoi Diagrams and extends them with Cluster-induced Voronoi Diagrams (CIVD) and Power Diagrams (CIPD). The method achieves state-of-the-art classification error and calibration (ECE) across CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R under the standardized TTAB evaluation protocol, with ablation studies demonstrating the contribution of each geometric component.

## Strengths
- **Consistent empirical improvements across multiple benchmarks**: TTVD achieves the lowest classification error on all four datasets (20.5% on CIFAR-10-C, 49.1% on CIFAR-100-C, 59.8% on ImageNet-C, 67.5% on ImageNet-R), outperforming eight strong baselines including TENT, SAR, and SHOT (Table 1). The gains are particularly notable for calibration, with ECE reductions of 3.4-4.3 percentage points.

- **Validated contribution of geometric components via ablation**: Table 2 demonstrates that the specific geometric extensions drive improvements: VD (28.4%) → CIVD (22.7%) → CIPD (20.5%) on CIFAR-10-C. This confirms the utility of multi-site influence and weighted boundaries rather than attributing gains to generic factors.

- **Standardized evaluation under TTAB**: The paper uses the peer-reviewed TTAB toolkit, which constrains hyperparameters, metrics, and baselines to a common standard. This reduces the risk of cherry-picked protocols and increases confidence in the comparisons, including the calibration metrics that are often omitted in TTA work.

- **Adaptation dynamics show sustained improvement**: Figure 4 illustrates that TTVD maintains a downward error trend across 750 online batches on ImageNet-C corruptions, while TENT and SAR plateau earlier. This suggests better long-term stability and reduced overfitting during adaptation.

## Weaknesses

### Fatal
None

### Major
- **Sample filtering mechanism ("diagram subtraction") lacks algorithmic specification**: The paper claims CIPD's 2.2% improvement over CIVD comes from Power Diagram-based noisy sample filtering: "By subtracting the PD from the VD, we can extract a larger region from the resulting differences...Noisy samples in these regions are excluded during adaptation" (Section 3.3). However, the main text provides no explicit criterion for which samples are filtered (e.g., $\{z : r_{VD}(z) \neq r_{PD}(z)\}$?), no description of how PD weights $v_k$ are instantiated or updated when only normalization parameters are adapted, and no pseudocode for the filtering step. Algorithm 3 is referenced as being in Appendix H (stripped by parser). Without these details in the main text, the key robustness claim attributed to CIPD cannot be fully evaluated, and it is unclear whether the gains arise from principled geometric filtering or simpler regularization effects.

- **Claims about "avoiding negative transfer" lack direct evidence**: The paper repeatedly emphasizes that CIVD's joint influence structure "avoids the negative transfer since the objective is now unified" (Section 3.2) and "seamlessly integrates self-supervision and entropy minimization" (Abstract). Prior work (Gandelsman et al., 2022; Niu et al., 2023) documents negative interactions between these objectives, making this a significant claim. However, the paper provides no gradient analysis (e.g., cosine similarity between objectives), no loss-surface characterization, and no comparison to a direct multi-task baseline combining rotation prediction and entropy minimization under the same TTAB protocol. The ablation (VD→CIVD→CIPD) shows performance gains but does not isolate whether they stem from reduced gradient conflict versus simply adding more augmented prototypes. This weakens the central narrative that the geometric framework solves a known failure mode of TTA.

### Minor
- **Limited statistical characterization of results**: All experimental results are reported as single runs without standard deviations, confidence intervals, or statistical tests. Given that gains over baselines are 0.7-1.6% on most datasets, it is unclear whether these differences are robust across random seeds, corruption orderings, or batch sampling. This is common in TTA literature due to computational cost, but it limits confidence in claims of consistent state-of-the-art performance, especially when combined with under-specified methodological details.

- **CIPD ablation limited to CIFAR-10-C**: Table 2's VD/CIVD/CIPD comparison is only shown for CIFAR-10-C. Given that CIPD is claimed as a major contributor to robustness (particularly for ImageNet-C where the largest absolute gains occur), similar ablations for CIFAR-100-C or ImageNet-C would help establish whether the geometric benefits scale to larger datasets. The paper notes CIPD improves by 2.2% on CIFAR-10-C but does not quantify the relative contribution of noise filtering on other benchmarks.

### Trivial
- **Figure 2 caption lacks detail**: The caption for Figure 2 states "Reliable samples can be identified by subtracting Voronoi cells, marked in deeper colors" but does not explain what the color gradient in part (a) represents numerically or how the boundary differences in part (b) relate to the filtering criterion. The figure is illustrative but would benefit from more precise labeling.

## Nice-to-Haves
- **Gradient conflict analysis**: Including empirical measurements of gradient cosine similarity between the self-supervision and entropy components (with and without CIVD's joint influence) would strengthen the "avoiding negative transfer" claim.

- **Runtime/overhead comparison**: Since diagram-based guidance implies precomputation of class means and potentially per-batch influence calculations, reporting adaptation cost (seconds per batch, FLOPs) relative to baselines like TENT or SAR would help assess practical deployment feasibility.

- **Visualization in higher-dimensional embeddings**: Extending the 2D MNIST-C visualizations to t-SNE/UMAP projections of ImageNet-C features would help demonstrate that the geometric partitioning corresponds to meaningful structure in realistic high-dimensional settings.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Critic's claim about "no explicit loss function for CIVD/CIPD"**: This is factually incorrect. The paper explicitly defines the VD loss in Equation 3 ($\mathcal{L}_{VD}(\tilde{y}_k) = -\sum_k \tilde{y}_k \log \tilde{y}_k$) and states "Similar to Equation 3, the soft label given by CIVD can be calculated from the influence function" (Section 3.2), with Equation 4 defining the CIVD influence function and Equation 6 defining CIPD. The entropy minimization form applies to the influence-based scores, which is a standard pattern in geometric ML papers.

- **Critic's claim about "unclear which parameters are updated"**: The paper explicitly states in Section 3: "Commonly, only the channel-wise affine parameters in normalization layers are updated during TTA, while the rest of the model remains unchanged." This is the standard TTAB protocol. While the interaction between frozen classifier weights and PD weights could be clarified, the core update rule is specified.

- **Critic's claim about exponent 7 and γ lacking motivation**: These are borrowed from prior computational geometry work on CIVD (Chen et al., 2013; 2017; Huang et al., 2021), which established these as effective influence function parameters. Re-deriving them for TTA is not necessary; the paper appropriately cites the source.

- **Critic's characterization of MNIST-C toy example as overclaiming**: The paper uses the 3-class MNIST-C visualization explicitly for intuition ("Figure 1: Visualization of space partitions and adaptation performance on MNIST-C...See Appendix C for details") and does not claim this represents multi-class real settings. The main results are on CIFAR and ImageNet benchmarks.

## Novel Insights
The paper's core insight—that neighbor-based TTA methods implicitly implement Voronoi space partitioning—is a useful unifying perspective that connects computational geometry to adaptation. However, this observation alone does not constitute a major theoretical advance; it primarily serves as motivation for applying existing geometric structures (CIVD, PD) to TTA. The empirical demonstration that these structures improve both accuracy and calibration is more substantial than the conceptual reframing. No genuinely novel insights emerge from the reviews beyond the paper's own contributions.

## Suggestions
1. **Add explicit sample filtering algorithm to main text**: Even if full pseudocode remains in the appendix, include a concise mathematical definition of the filtering criterion (e.g., "samples $x$ are excluded if $|r_{VD}(x) - r_{PD}(x)| > \delta$" or similar) and clarify how PD weights are instantiated when the classifier is frozen.

2. **Include a direct multi-task baseline**: Add a comparison to a method that jointly optimizes rotation prediction and entropy minimization under TTAB (without the geometric framing) to isolate whether gains come from objective unification versus additional capacity/augmentation.

3. **Report variance for key results**: Even for one dataset (e.g., CIFAR-10-C), running 3-5 seeds and reporting mean ± std for TTVD and the strongest baseline (SAR or TENT) would help establish whether the 0.8-1.6% gains are statistically meaningful.

4. **Extend CIPD ablation to ImageNet-C**: Show the VD/CIVD/CIPD breakdown for at least ImageNet-C to demonstrate scaling of geometric benefits.

## Score and Decision

**Calibration process:**
- **Topic anchors**: Searched for TTA papers with geometric frameworks. Found no direct Voronoi-based TTA papers in the human review corpus. Closest matches were TTA papers with geometric elements (NGTTA: 6,5,6,5 rejected; AdapTable: 5,5,3,5 rejected) and non-TTA geometry papers (GeoRCG: 8,3,5,5,6).

- **Quality-based anchors**: 
  - Papers rejected for underspecified methods: PRO (3,3,3,6) criticized for "Details of algorithm (PRO) are missing. It is unclear how the loss functions are combined"; G-TIGRE (3,3,3,3) for missing definitions. These had weaker empirical cases than TTVD.
  - Papers accepted with some missing details but strong results: DeYO (8,6,6,8 spotlight) despite missing baselines; Diffusion Bridge (6,8,6,8,5 poster) despite minor equation errors. These had clearer empirical advantages.
  - Papers rejected for modest gains: Continual TTA (3,5,5,6,5) for "<0.5% accuracy" improvements; similar TTA papers with 1-2% gains often land in 5-6 range.

- **Deliberate range anchoring**:
  - High-scored TTA papers (7-8): Often have clear theoretical contributions or large gains (>3%).
  - Borderline papers (5-6): Typically have solid results but underspecified methods or modest gains without statistical analysis.
  - Low-scored papers (3-4): Usually have fundamental methodological flaws or very weak empirical case.

**Positioning**: TTVD has a stronger empirical case than most borderline papers (consistent SOTA across 4 datasets, 3.4-4.3% ECE improvements, ablation validation), which pushes it above the typical 5-6 range for modest-gain TTA papers. However, the underspecified "diagram subtraction" mechanism and lack of direct evidence for "avoiding negative transfer" claims prevent it from reaching 7-8 territory where theoretical clarity or transformative empirical gains are expected. The paper aligns most closely with DeYO (8,6,6,8) in empirical strength but lacks DeYO's clearer theoretical motivation, suggesting a score in the **6.5** range.

**Comparative assessment**:
- **Originality**: Moderate—the geometric framing is useful but builds on established CIVD/PD theory.
- **Importance**: High—TTA is a practical problem, and calibration improvements matter for deployment.
- **Claim support**: Partial—empirical claims are well-supported; mechanistic claims ("avoiding negative transfer") are not.
- **Soundness**: Fair—method is implementable but key filtering details are deferred.
- **Clarity**: Good—writing is clear, figures are illustrative, but some algorithmic details are missing.
- **Value**: Moderate-high—the method works and the geometric perspective may inspire future work.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>