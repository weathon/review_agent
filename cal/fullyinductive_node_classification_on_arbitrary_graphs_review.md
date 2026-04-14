=== CALIBRATION EXAMPLE 52 ===

# Final Consolidated Review
## Summary
GraphAny introduces the *fully-inductive* node classification setting, where a model must generalize to test graphs with entirely new structures, feature dimensions, and label spaces—without any additional training. The method combines LinearGNNs (linear graph convolutions whose weights are solved analytically via pseudo-inverse) with a learned inductive attention module parameterized by entropy-normalized pairwise distance features between LinearGNN predictions, ensuring permutation invariance and dimensional robustness. Trained on a single small graph (Wisconsin, 120 labeled nodes), GraphAny achieves competitive average accuracy across 30 held-out graphs while offering substantial computational savings over per-graph transductive training.

---

## Strengths

- **Principled formalization of a new and genuinely harder setup**: The fully-inductive setup—arbitrary feature AND label spaces, no shared embedding space, no fine-tuning—is clearly and formally defined (Eq. 6, Figure 2). This goes meaningfully beyond standard inductive learning and addresses the core bottleneck for cross-domain graph foundation models.

- **Elegant permutation-invariant attention design**: The observation that pairwise Euclidean distances between LinearGNN predictions cancel the label-permutation matrix Q (since ‖ŷQ − ŷ′Q‖² = ‖ŷ − ŷ′‖² for orthogonal Q) is clean and correct. Building the attention MLP on these distance features as in Eq. (9) is a principled, minimal solution that sidesteps the need for equivariant architectures.

- **Entropy normalization as a principled fix for the curse of dimensionality across label spaces**: Figure 5 compellingly demonstrates that raw Euclidean distance distributions collapse to near-zero for datasets with large label spaces (e.g., FullCora with 70 classes), while entropy-normalized features maintain consistent scale. The adaptation of entropy normalization from SNE/t-SNE to this cross-graph setting is original and well-motivated.

- **Striking data efficiency**: A model trained on 120 labeled nodes generalizes to match or exceed the average performance of transductive GCN/GAT models collectively trained on 511k labeled nodes across 31 datasets. Even acknowledging the small aggregate margin, the operating regime (cross-graph zero-retraining transfer from 120 labels) is genuinely new.

- **Concrete wall-time efficiency**: The 2.95× total wall-time speedup over optimized DGL-GCN across 31 graphs, and 15× for LinearGNN alone, are measured and reproducible — not just asymptotic claims.

- **Ablation clearly establishes the value of entropy normalization**: Figure 8 shows that unnormalized features (Euclidean, Jensen-Shannon) exhibit improving transductive accuracy but deteriorating inductive accuracy as training progresses (a clear overfitting signature), while EntNorm variants converge stably in both settings. This is a compelling demonstration of the mechanism's role.

---

## Weaknesses

### Fatal
None that would invalidate the core contribution.

### Major

- **Equation (10) appears to contain a formula error** — In Eq. (10), the denominator uses σ_u^(k) (a distinct bandwidth per summation index k) while the numerator uses σ_u^(i) (fixed for a given i). Additionally, the factor of 2 appears in the numerator exponent but is absent from the denominator. Standard SNE (Hinton & Roweis 2002; van der Maaten & Hinton 2008), which the paper cites explicitly, uses σ_u^(i) uniformly across all denominator terms:
  $$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2/2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2/2\sigma_i^2)}$$
  If the denominator in Eq. (10) uses σ_u^(k) per k, the entropy constraint that determines σ_u^(i) becomes ill-defined, since σ_u^(k) values for k≠i influence the normalizing sum in a circular fashion. The paper must either (a) confirm this is a typographic error and supply the corrected formula, or (b) if intentional, provide an explicit derivation showing that σ_u^(i) is still uniquely determined and the resulting p_u(j|i) remains a valid probability distribution.

- **Complexity table omits dominant cost of pseudo-inverse** — Table 1 lists LinearGNN's optimization complexity as O(|V_L|), but computing the pseudo-inverse F_L^+ via SVD of F_L ∈ ℝ^{|V_L|×d} costs O(min(|V_L|, d)² · max(|V_L|, d)), which depends critically on feature dimension d. For modern graph datasets with high-dimensional features (e.g., OGB-Arxiv, d=128; text-attributed graphs, d≫1000), this factor is non-negligible. Similarly, the inference complexity O(|V_U|) omits the per-node iterative binary search required to determine σ_u^(i) in the entropy normalization step. The wall-time results show GraphAny remains fast in practice, but the complexity table as written is misleading and should be corrected.

- **The headline claim "surpassing transductive methods" is fragile and not adequately supported** — The margin of GraphAny-Wisconsin (67.26%) over GAT (67.03%) is 0.23 percentage points in 31-graph average accuracy. This comparison is made without any statistical significance test, despite high per-dataset variance (e.g., Wisconsin: ±5.98%). More critically, GraphAny badly underperforms on large, practically important graphs: Arxiv (57.79% vs 73.65%, −15.86 pp) and Products (60.28% vs 79.45%, −19.17 pp). The aggregate average obscures these large gaps. The abstract and main text present "surpassing transductive methods" as a clean win; this claim needs either a significance test for the aggregate comparison or explicit qualification of the large-graph regime where GraphAny currently falls far short.

- **Test-time labeled-node sensitivity is unanalyzed** — GraphAny requires a set of labeled nodes V_L on each test graph to compute W* = F_L^+ Y_L. Yet the paper never ablates the size of this test-time label set. If performance degrades sharply with fewer test labels, the advantage over transductive SGD methods (which also require labeled nodes) is diminished. Since transductive methods also benefit from more labeled nodes via backpropagation, the comparison is only fair if GraphAny is robust to small |V_L|. This is a central empirical gap.

### Minor

- **Domain coverage does not support "arbitrary graphs" claim** — All 31 datasets are citation networks, social networks, and e-commerce graphs with similar structural properties. Generalization to structurally distinct domains (molecular graphs, protein interaction networks, spatiotemporal graphs) is entirely untested. The "arbitrary" framing in the title and abstract should be qualified to match the scope of the evaluation.

- **MSE approximation for cross-entropy not justified** — Equation (2) replaces cross-entropy with MSE to enable the analytical solution. This is a non-trivial modeling choice: MSE treats one-hot labels as continuous regression targets, and predictions can fall outside [0, 1] before the softmax in Eq. (1). The paper does not analyze when this approximation is appropriate or how it interacts with class imbalance. A brief discussion would strengthen the motivation.

- **Attention MLP architecture not reported in main text** — The learned module f_θ: ℝ^{t(t-1)} → ℝ^t (mapping 20 features to 5 attention weights for t=5) is the only trainable component of GraphAny and thus critical for reproducibility, yet its architecture (layers, width, activation) is absent from the main text. It is unclear how much of the performance gain is attributable to learning vs. simple ensemble averaging.

- **Figure 6 applies "Inductive" label to training-graph evaluation** — Wisconsin appears in Figure 6's x-axis despite being GraphAny-Wisconsin's training graph. The y-axis label "Inductive test accuracy" is misleading for Wisconsin, since evaluating on Wisconsin's held-out test split with a model trained on Wisconsin's labeled split is the transductive setting. This should be labeled or noted separately.

### Tiny

- **Hits@2 random baseline not reported** — Section 4.3 reports Hits@2 of 0.65 and 0.77 for the attention module identifying the best-performing LinearGNN. For a 5-choice task, random Hits@2 = 2/5 = 0.40. Reporting this baseline would better calibrate the magnitude of the learned attention's benefit.

- **Figure 8 convergence description overstates stability** — The paper describes EntNorm variants as achieving "stable convergence" in both settings, but the figure caption and description note a visible downward trend in inductive accuracy for some EntNorm configurations as training continues. A more precise characterization (e.g., best checkpoint vs. final checkpoint) would be appropriate.

---

## Nice-to-Haves

- **Ablation on number of LinearGNNs t**: The paper fixes t=5 throughout. An oracle uniform-ensemble baseline (unweighted average of all t LinearGNNs) would clarify whether the learned attention provides improvement beyond simple ensemble averaging, isolating the contribution of the attention mechanism.

- **Test-time label sensitivity curves**: Plot GraphAny accuracy vs. |V_L| on test graphs (e.g., 10, 50, 100, 200 labels) to characterize the operating regime in which GraphAny maintains its advantage over transductive baselines.

- **Efficiency crossover analysis**: Plot wall-clock inference time vs. |V_L| to show at what labeled-node scale the pseudo-inverse computation becomes the bottleneck, and whether iterative least-squares solvers (e.g., LSQR) or low-rank approximations would help at large scale.

- **Numerical stability of pseudo-inverse**: Reporting condition numbers of F_L across datasets in the appendix would validate the reliability of the analytical solution on diverse graphs, where ill-conditioned feature matrices (common in sparse, high-dimensional settings) are a practical concern.

- **Attention weight vs. homophily visualization**: Plotting averaged attention weights per LinearGNN channel against the homophily ratio of the test graph would reveal whether the attention mechanism is genuinely capturing graph spectral structure (high homophily → low-pass preference) or acting as a dataset-agnostic ensemble.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Missing meta-learning / few-shot learning baselines** (Harsh Critic): The fully-inductive setup requires handling arbitrary feature and label spaces — a fundamental constraint that existing meta-learning methods for graphs (G-Meta, ProtoNet-style) cannot satisfy without modification, as they assume shared feature spaces. Their absence is not a meaningful gap.

- **Cross-dataset leakage in Figure 6** (Harsh Critic): GraphAny-Wisconsin uses Wisconsin's *training* labeled nodes for W*; the evaluation in Figure 6 uses held-out *test* nodes. There is no leakage. The Wisconsin bar in Figure 6 represents a standard transductive evaluation on the training graph's test split, which is valid and disclosed in Table 2.

- **Weak transductive baselines inflate competitiveness** (Harsh Critic): The paper uses GCN and GAT as "cheating" baselines because they represent standard models that can train separately on each test graph with full backpropagation. Adding stronger specialized methods (GCNII, GPR-GNN, etc.) would only make GraphAny's aggregate comparison harder — the asymmetry favors the baseline, not GraphAny, strengthening rather than inflating the point.

- **LinearGNN novelty is insufficient on its own** (Harsh Critic): The paper explicitly positions LinearGNN as a known-type building block (citing SGC, Sato 2024) and makes no claim that LinearGNN alone is novel. The novelty lies in the full system design. This is not a weakness.

- **Non-standard "fully-inductive" terminology** (Harsh Critic): The paper clearly defines the term. Stylistic terminology preferences are not substantive concerns.

- **Statistical significance for large-scale benchmarks** (concerning 31-graph aggregate): Single-run aggregate comparisons over 31 datasets are standard practice in the graph learning community. The small margin IS a genuine concern (kept in Major), but requesting confidence intervals on aggregate benchmarks is somewhat non-standard. The core concern is retained as a Major weakness framed around the fragility of the headline claim.

---

## Novel Insights

The paper surfaces an underappreciated interaction between the curse of dimensionality and cross-graph knowledge transfer: attention modules trained to route between graph filters fail to generalize not because of structural domain shift, but because the raw distance distributions between filter predictions collapse to near-zero scale as the number of classes grows—making the learned decision boundaries in one label-space regime useless in another. The entropy normalization solution (importing σ-search from SNE) is conceptually simple but non-obvious in this context. More interestingly, Figure 5 reveals an unexpected finding: structurally dissimilar graphs (e.g., citation network Cora with 7 classes and e-commerce Products with 47 classes) produce nearly identical entropy-normalized distance feature distributions, hinting that the learned attention captures genuinely transferable structural priors about how spectral filters relate to each other—rather than memorizing dataset-specific statistics. This empirical observation deserves deeper theoretical investigation in follow-up work.

---

## Suggestions

1. **Fix Equation (10)**: Verify whether the denominator should use σ_u^(i) uniformly (standard SNE) and include the factor-of-2. If the formula is intentional, provide an explicit derivation showing the entropy constraint remains well-posed and that σ_u^(i) is uniquely solvable.

2. **Correct Table 1**: Report the pseudo-inverse computation cost (O(min(|V_L|,d)²·max(|V_L|,d))) explicitly, move it to the preprocessing or optimization row, and note the entropy normalization binary-search constant in the inference column.

3. **Qualify the headline claim**: Add explicit caveats in the abstract and Section 4.2 noting the large-graph performance gaps (Arxiv: −15.86 pp, Products: −19.17 pp vs. GAT) and provide either a paired significance test or confidence interval for the 31-graph aggregate comparison.

4. **Add test-time |V_L| sensitivity experiments**: Run GraphAny with varying numbers of test-time labeled nodes to establish the regime of reliable inductive generalization.

5. **Report the attention MLP architecture in the main text** for reproducibility.

6. **Qualify "arbitrary graphs"**: Replace or qualify the claim to reflect that evaluation covers citation/social/e-commerce graphs; note molecular, biological, and spatiotemporal domains as future work.

---

**Overall assessment across axes:**
- *Novelty*: High — the fully-inductive setup and its principled solution via permutation-invariant entropy-normalized attention are original contributions.
- *Technical soundness*: Moderate — the architectural design is principled, but a likely formula error in Eq. (10) and incomplete complexity analysis are material issues that need resolution.
- *Empirical support*: Moderate — the 31-graph evaluation is broad, but the headline claim rests on a marginal, untested gap; critical ablations (test-time label count, t sensitivity) are missing; and underperformance on large graphs is under-examined.
- *Significance*: High — establishing that cross-graph, zero-retraining generalization is possible without shared feature spaces is an important result for the graph foundation model agenda.
- *Clarity*: Good overall, but Section 3.2 is dense, Eq. (10) requires correction, and Figure 6's labeling is misleading.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 8.0]
Average score: 7.0
Binary outcome: Accept
