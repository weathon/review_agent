## Summary

The paper proposes a multi-view consensus clustering framework for text embeddings that generates multiple transformed views of SBERT embeddings, aggregates cluster assignments via spectral consensus on a co-occurrence matrix, and refines representations through iterative co-training with a hybrid InfoNCE + GMM loss. The authors derive a theoretical bound showing exponential error decay with the number of views under independence and informativeness conditions, and demonstrate improvements on DBPedia and Reuters R8 datasets.

## Strengths

- **Theoretical grounding for consensus clustering:** The paper derives a bound on the expected misclustering fraction for multi-view consensus versus single-view clustering (Appendix B), explicitly linking error reduction to the number of views $m$ and the advantage parameter $\delta$. This provides formal justification for why aggregating multiple views can reduce error, going beyond purely empirical claims common in clustering literature.

- **Generalization to unseen data:** Table 5 shows that models trained on small subsets (10% training) achieve strong performance on held-out data (NMI ~79, ARI ~70), with minimal degradation across train/test splits. This suggests the learned latent space captures cluster structure rather than overfitting to training documents.

- **Empirical improvement over classical baselines:** Tables 2-4 demonstrate consistent improvements in NMI and ARI over k-means, GMM, and spectral clustering on static embeddings, with gains of 5-15 points on DBPedia when using multi-view consensus with diverse transformations.

## Weaknesses

- **Mismatch between theoretical analysis and algorithm:** The proof in Appendix B analyzes majority voting across views, but Algorithm 1 implements spectral clustering on a co-occurrence matrix. The Hoeffding inequality argument is specific to vote counting; its extension to eigenvector-based consensus is not established. This disconnect means the theoretical guarantee does not formally apply to the implemented method.

- **Independence assumption violated in practice:** Condition 1 requires mutually independent views, yet all proposed transformations (PCA, WPT, Gaussian noise, multiple BERT models) operate on the same underlying text embeddings. The views share the input signal and are inherently correlated. The paper acknowledges that "weakly uncorrelated views contribute proportionally" but does not quantify how correlation degrades the bound, leaving a theory-practice gap unaddressed.

- **Outdated baseline comparisons:** The paper compares only against k-means, GMM, and spectral clustering—methods predating modern deep representation learning. No comparison to recent deep text clustering methods (e.g., DEC, VaDE, SCCL, SCAN) is provided. For ICLR, demonstrating superiority over current methods that jointly learn representations and cluster assignments is essential.

- **Insufficient dataset coverage:** Evaluation uses only two datasets (DBPedia with k=8/14 and Reuters R8 with k=6). Both are relatively clean English corpora. No multilingual, domain-shift, or larger-scale datasets are included despite the introduction highlighting multilingual streams and distributional shifts as motivating challenges.

- **Missing architectural and hyperparameter details:** The MLP encoder $q_\phi$ is never described—number of layers, hidden dimensions, activations, and output dimensionality are absent. Hyperparameters $\alpha, \beta, \tau, e$ appear in Algorithm 2 but are not specified or ablated in experiments, impeding reproducibility.

- **High variance in some configurations:** Table 4 shows standard deviations up to ±14.7 ARI for certain single-view settings on Reuters R8. While the mean multi-view results improve, the high baseline variance raises questions about whether improvements exceed noise levels on this dataset.

- **No scalability analysis:** The co-occurrence matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$ requires $O(n^2)$ memory, and spectral decomposition scales poorly. The introduction emphasizes RAG systems and large corpora, but computational cost is never discussed—this is a practical limitation for real-world deployment.

- **K assumed known:** The method requires the number of clusters $K$ as input with no mechanism for estimation. This practical limitation is not acknowledged or discussed.

## Nice-to-Haves

- Comparison to modern deep clustering baselines (DEC, VaDE, SCCL, or recent contrastive clustering approaches)
- Evaluation on 4-5 additional datasets including multilingual or cross-domain settings
- Ablation study on view correlation: measure empirical correlation between generated views and quantify performance degradation
- Analysis of computational complexity and wall-clock runtime
- Methods for estimating $K$ or robustness analysis to mismatched $K$

## Removed Points

These points are flagged to be removed; treat them with caution:

- **Claim that Table 4 shows multi-view fails to improve on Reuters R8:** Upon verification, multi-view consensus (PCA + Multiple Models) achieves NMI=80.8 versus single-view GMM NMI=73.9, showing improvement. The ARI values are similar (~70 for both), but NMI clearly improves. The critic misread the table.

- **Proof labeling swap (H(c) vs H(c|h)):** Equations 7-8 in the appendix correctly assign H(c) as marginal entropy and H(c|h) as conditional entropy. The critic's claim of label swapping appears incorrect.

- **Formatting/style nitpicks:** Comments about Section 2.1 wasting space are editorial; the GMM overview, while standard, provides necessary notation for the method section.

## Novel Insights

The exponential error bound based on the advantage parameter $\delta$ provides a useful conceptual framework: even weakly informative views can collectively achieve strong clustering when aggregated in sufficient numbers, provided they satisfy diversity. However, the critical insight is that the bound's reliance on independence creates a fundamental tension—the most diverse transformations in practice (different BERT models, PCA projections) still share substantial mutual information because they derive from the same source documents. This suggests a productive direction: explicitly measuring view correlation and developing theoretical bounds that incorporate correlation structure, rather than assuming independence. The empirical finding that PCA+WPT+Multiple Models (the most diverse combination) yields the best performance partially validates this, but the correlation between these views remains unquantified.

## Suggestions

1. **Add deep clustering baselines:** Include at least one recent deep text clustering method (e.g., SCCL, DCC) trained on the same embeddings to establish whether multi-view consensus provides gains over learned representations.

2. **Conduct view correlation analysis:** Measure pairwise ARI or correlation between view-level clusterings and correlate with consensus performance to empirically validate (or correct) the theoretical assumptions.

3. **Specify all hyperparameters and architecture:** Provide MLP layer counts, hidden dimensions, activation functions, and all training hyperparameters ($\alpha, \beta, \tau, e$, learning rate, batch size) for reproducibility.

4. **Expand datasets:** Add at least 2-3 diverse datasets (e.g., AGNews, 20Newsgroups, or a multilingual corpus) to support generalization claims.

5. **Add computational analysis:** Report training time, memory usage, and discuss scalability limitations or potential approximations (e.g., sparse affinity matrices, mini-batch variants).