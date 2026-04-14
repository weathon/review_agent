---

## Summary

FGL_AC proposes a federated graph learning framework for graph classification that combines two components: (1) spectral clustering applied client-side as a data preprocessing step to reduce training burden, and (2) an attention mechanism applied server-side during parameter aggregation to assign different weights to clients based on their training performance. Experiments on three small TU benchmark datasets (MUTAG, ENZYMES, PROTEINS) with 3 clients and 4 baseline variants are presented.

---

## Strengths

- **Four heterogeneity scenarios**: The paper evaluates across balance/unbalance × overlap/no-overlap data splits, which meaningfully tests robustness under different non-IID conditions and is a more thorough partition design than simply reporting one setting.
- **Component-level ablation**: Section 4.2 explicitly isolates the contribution of the clustering component (FGL_AC−C) and the attention component (FGL_AC−A) against the full model, providing direct evidence that both components contribute rather than one dominating. This is more structured than typical ablation studies in the space.
- **Predominant empirical wins**: Across the majority of dataset/partition combinations in Table 2, FGL_AC achieves the best accuracy, demonstrating the overall soundness of the idea at small scale.

---

## Weaknesses

1. **Graph vectorization is undefined — makes the method unreproducible.** Section 3.2 states "each sub-graph in the dataset is regarded as a point in the space" and computes Euclidean distance ‖g_i − g_j‖² (Eq. 1), but graphs are not Euclidean objects. The paper never specifies what vector representation of each graph is used. Without this, the entire spectral clustering preprocessing step is mathematically ill-defined and cannot be reproduced. This is a critical gap in the core methodology.

2. **Central update equation missing from Algorithm 1.** The formula Z_{G+1} = Z_G − η ∑_k α_k(Z_G − z_k), which governs how the global model is updated, appears only in the Figure 2 caption. It is absent from Algorithm 1 (which merely says "Distribute the global parameter Z_g") and from the main text. Its derivation, relationship to Eq. (9), and dependence on η are never discussed. For a method paper, this is a fundamental presentation failure.

3. **The bridge from pairwise attention (Eq. 8) to scalar aggregation weights (Eq. 9) is never derived.** Eq. (8) yields a pairwise coefficient Attention(c_i, c_j), yet Eq. (9) introduces scalar weights α, β, δ. The paper states only that these are "calculated by different clients acting as target clients through formula 8," offering no formal derivation. It is unclear how per-pair coefficients reduce to per-client scalars.

4. **Ambiguity of client feature vectors c_i in the attention mechanism.** Eq. (8) uses "feature vectors of the current client," and Figure 2 shows clients transmitting a "Local Representation" to the server. However, the paper never defines exactly what c_i is (parameter vector, learned embedding from the MLP, loss values, etc.). This fundamentally changes the attention mechanism's behavior and privacy implications.

5. **Privacy claim is vacuous.** Figure 2 prominently labels the communication channels with "Differential Privacy," yet the main text contains no discussion of DP whatsoever: no noise mechanism, no ε values, no sensitivity analysis, no privacy budget. The DP mention appears to be cosmetic. Furthermore, if Gaussian/Laplace noise is added to c_i before sending to the server, the attention computation in Eq. (8) is corrupted; if it is not added, the privacy claim is false.

6. **Notation conflicts throughout.** Table 1 defines L as both "quantity of local iterations" and "Laplacian matrix," and D as both "degree matrix" and "Training data collection." These collisions persist into Algorithm 1 and the formulas, creating systematic ambiguity.

7. **Selective and misleading abstract.** The abstract claims FGL_AC achieves "2.63%–4.03% improvement" over other frameworks, but Table 2 shows: FGL_AC F1 of 83.55 vs. GCN-FedAvg F1 of 84.41 on MUTAG balance-no-overlap; accuracy tied with GCN-FedProx at 44.17% on ENZYMES unbalance-no-overlap; and SAGE-FedProx F1 of 36.73 beating FGL_AC F1 of 33.50 on PROTEINS unbalance-overlap. These failure cases are not acknowledged.

8. **Ablation figures have identical descriptions, raising validity concerns.** The textual descriptions of Figures 3 and 4 are word-for-word identical (identical numerical ranges ~0.65 to ~0.85, identical sentence structure). If the figures themselves are visually indistinct across the two different partition conditions (unbalance-no-overlap vs. balance-overlap), this requires explanation. If they are accidentally duplicated, this is a serious error.

9. **Only 3 clients, 3 small datasets, 4 closely related baselines.** The experimental scale is minimal. With only 3 clients, it is impossible to assess how the attention mechanism performs as the number of clients grows. Only FedAvg- and FedProx-based GNN variants are compared; no baselines for attention-based FL aggregation or dedicated FGL methods are included. Improvements of 0.36%–2.6% accuracy on MUTAG (188 graphs) carry little weight without statistical significance testing (no variance or confidence intervals reported anywhere).

10. **Communication overhead and efficiency claims never measured.** The abstract and introduction repeatedly claim FGL_AC reduces "communication overhead" and "training burden," yet no wall-clock time, FLOPs, or communication bytes per round are reported. These claims are entirely unsubstantiated.

11. **Section 4.3 presents a trivial comparison.** Showing that two federated clients outperform one isolated client is an expected and uninteresting result. This is not a meaningful evaluation of the method against centralized training (which would pool all data jointly).

12. **"Degeneration to FedAvg" claim is unproven.** The paper asserts FGL_AC degenerates to FedAvg "in the worst case" (equal client performance), but provides no proof. Under the softmax in Eq. (8), equal inputs do produce equal weights — but the paper does not show this formally or empirically.

---

## Nice-to-Haves

- Define a graph embedding or kernel (e.g., graph-level features, degree sequences, or a pre-trained encoder) to formally ground the Euclidean distance in Eq. (1), and conduct a sensitivity analysis over the number of clusters k and KNN neighbors.
- Visualize how attention weights α_k evolve across communication rounds to verify that the server learns to differentiate client contributions rather than converging to uniform weights.
- Provide at least empirical convergence curves comparing FGL_AC against its ablated variants at larger client counts (e.g., 5–10) to support generalizability.
- Quantify actual communication cost per round to support efficiency claims.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"FedAvg is referred to as 'he'"** (Reviewer 2/3): Pure proofreading nitpick.
- **O(n³) spectral complexity as a practical barrier** (Harsh Critic): At the paper's dataset sizes (n=188, 600, 1113 graphs), eigendecomposition of an n×n matrix is trivially fast. The concern is valid in principle for large-scale IIoT but not at the tested scale; it was presented as an immediate blocking issue, which overstates it.
- **Demanding theoretical convergence proofs** (Harsh Critic): Convergence proofs for FL aggregation are not universally expected for empirical systems papers in this sub-community. This is a nice-to-have, not a blocking weakness.
- **Missing related work on FGL-specific methods / personalized FL / FedAtt** (Harsh Critic): Per policy, missing related works are not cited as weaknesses since external sources cannot be verified.
- **Section 4.3 "odd experimental design"** framed as comparing 2 clients doing FL vs. 1 doing "centralized" training (Harsh Critic's framing): It is somewhat unusual, but the "centralized" here means isolated local training, which is a common FL baseline; the design is weak but not "odd." Captured under the main weakness about trivial comparison.

---

## Novel Insights

None beyond the paper's own contributions. The combining of spectral clustering as preprocessing with GAT-style attention aggregation in federated learning is the paper's stated contribution, and no reviewer surfaces an insight about the combination that goes beyond what the paper itself claims. The reviewers collectively surface that the ill-defined graph-to-vector mapping for spectral clustering is a deeper problem than it might first appear: since the spectral clustering operates on the Laplacian of a *meta-graph* built over graph instances (not a single graph), the entire preprocessing step implicitly assumes a fixed-dimensional representation of individual graphs — making it equivalent to a graph embedding method that is entirely unspecified. This matters because the quality of the clustering (and hence the claimed training burden reduction) is entirely contingent on this hidden representation choice.

---

## Suggestions

1. **Define graph representation explicitly**: Either use graph-level hand-crafted features (degree histogram, clustering coefficient, etc.) or a pre-trained Graph2Vec/kernel embedding, and state this clearly in Section 3.2. Without this, the preprocessing step is not reproducible.
2. **Integrate the update rule into Algorithm 1**: Move Z_{G+1} = Z_G − η ∑_k α_k(Z_G − z_k) into the algorithm box with a formal derivation from Eq. (9).
3. **Clarify or remove the DP claim**: Either provide formal DP accounting (mechanism, ε, δ, sensitivity) or remove the DP label from Figure 2 to avoid overstating privacy guarantees.
4. **Report variance across seeds**: Run at least 3–5 random seeds and report mean ± std in Table 2. Several improvements are sub-1% and may not be statistically significant.
5. **Fix notation conflicts**: Use distinct symbols for local iteration count and Laplacian, and for degree matrix and dataset collection.
6. **Expand client scale**: Run experiments with at least 5–10 clients to test whether the attention mechanism remains beneficial as the federation grows; 3 clients is too small to demonstrate generality.

---

**Evaluation:**

- **Novelty**: Low. The combination of spectral clustering preprocessing and GAT-style attention aggregation in federated graph learning is a straightforward engineering combination of existing techniques, without novel theoretical insight.
- **Technical soundness**: Weak. Two critical gaps — undefined graph vectorization and missing derivation of the aggregation rule — undermine reproducibility and correctness of the core method.
- **Empirical support**: Poor. Three-client, three small-dataset experiments with no statistical significance and missing efficiency measurements do not substantiate the paper's claims, including several cases where baselines match or beat the proposed method.
- **Significance**: Limited. The performance gains are modest, unverified statistically, and obtained on datasets small enough that the paper's IIoT motivation is not meaningfully addressed.
- **Clarity**: Poor. Multiple notation conflicts, a central equation absent from the algorithm, vague contribution statements, and unexplained DP claims reflect insufficient preparation for a top venue.

MY FINAL SCORE: <pineapple>3.2</pineapple>