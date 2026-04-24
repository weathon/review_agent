## Summary

This paper proposes Multimodal Structure Preservation Learning (MSPL), a representation learning framework that aligns the pairwise distances of a primary modality’s learned representations with the clustering structure of a secondary modality, rather than performing traditional feature-level fusion. The authors demonstrate the approach on synthetic time series and two epidemiology datasets (MALDI-to-WGS and MALDI-to-AMR), introducing a domain-specific SNP threshold loss and a cluster F1 metric for imbalanced cluster evaluation.

## Strengths

- **Novel structure-level alignment approach.** MSPL deviates from standard multimodal connectors by preserving external clustering structure through pairwise distance alignment (Eq. 7) rather than feature-level fusion. This allows the method to leverage structural information even when raw secondary-modality data are unavailable—a condition satisfied in the proprietary dataset where only SNP distance matrices, not raw WGS reads, were accessible (Section 4).
- **Domain-informed custom loss for epidemiological thresholds.** The paper derives an application-specific SNP loss (Eq. 15–16) that respects the clinical outbreak-detection threshold by penalizing feature distances only when the ground-truth SNP distance is below the threshold or when the feature distance underestimates a large SNP distance. This encodes real-world epidemiological decision criteria into the learning objective.
- **Modular and adaptable framework.** Separating reconstruction, pretext, and structure preservation into explicit terms (Eq. 8) makes the method easy to adapt to other modalities and tasks.

## Weaknesses

### Fatal
None.

### Major

- **Proprietary dataset results contradict the central application claim.** On the proprietary dataset, MSPL$_{\text{thr}}$ achieves an ARI of **0.001 ± 0.002** and an NMI of **0.137 ± 0.040** (Table 1), indicating essentially chance-level structural agreement with the ground truth. At the same time, the model predicts only **38 clusters** versus **544** true WGS clusters (Figure 6). The paper nevertheless claims this configuration "effectively mimics the decision-making criterion of epidemiologists" (Section 5) and that MSPL makes MALDI a "viable substitute for WGS in practice" (Section 1). A near-zero ARI means the partition is structurally random; the high cluster F1 of 0.962 is an artifact of the metric’s exclusion of singleton ground-truth clusters (Section 3: "we only evaluate them on the subset ... where the ground truth cluster has more than 1 element"), which in this highly skewed distribution excludes the vast majority of clusters. Without validation that the 38 predicted clusters map to actual outbreak events of interest, the core epidemiological claim is unsupported.
- **The clusCLS baseline is evaluated with data leakage, invalidating key comparative claims.** The authors derive clusCLS’s target cluster labels $\mathcal{C}_T$ "beforehand using the external dissimilarity matrix $\mathbf{d}$ on the **full dataset**" (Section 3). Because cross-validation is used in evaluation (Section 5), these globally derived labels embed test-set structure into the training targets for every fold. This leakage artificially inflates clusCLS’s performance in a way that is not comparable to MSPL. Reported conclusions such as "MSPL has a marked advantage over the classification approach on sparser datasets" (Section 5) are therefore unreliable, particularly on Synth-TS where clusCLS otherwise outperforms MSPL on the densest setting (Table 1).

### Minor

- **Cost-reduction and deployment claims remain unsubstantiated.** The abstract and introduction assert that MSPL "substantially reduce[s] data acquisition cost" and enables a cheap modality to substitute for an expensive one. However, every experiment is conducted on fully paired data with standard cross-validation, and MSPL requires the expensive modality’s full pairwise dissimilarity matrix during training (Eq. 7). There is no experiment showing that MSPL trained on a paired dataset can then be deployed on new, MALDI-only samples to detect outbreaks. Because the evidence stops at structure replication on paired data, the cost-reduction claim is speculation.
- **DRIAMS performance is more mixed than presented.** While the text states that "MSPL outperforms all baseline models in precision, recall, and F1 score" on DRIAMS, Table 1 shows that onlyCLS$_{\text{thr}}$ achieves a higher cluster F1 (0.818) than at least one of the MSPL$_{\text{thr}}$ entries (0.667), and Figure 5(e,f) shows numerous DRIAMS species where MSPL underperforms both baselines (lift < 1). This variance is downplayed rather than analyzed.

### Trivial
None.

## Nice-to-Haves

- Deployment validation: train MSPL on a paired training set, then evaluate exclusively on held-out MALDI-only samples to test real-world outbreak detection utility.
- Leakage-free baseline: re-run clusCLS deriving cluster labels from the training fold only.
- Ablations isolating the pure distance-matching component (autoencoder + $\mathcal{L}_{\text{struct}}$ only, no pretext) to verify whether the full MSPL framework is necessary relative to simpler distance-regression baselines.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Missing related work (metric learning, MDS, Isomap, knowledge distillation):** Per instructions, missing related works should not be flagged as we do not have external sources to confirm their existence.
- **"The novelty of Eq. 7 is limited":** This is a subjective opinion about novelty degree rather than a verifiable flaw. The paper's contribution lies in the overall framework, not Eq. 7 in isolation.
- **Formatting/style nitpicks and typo complaints:** These are parser artifacts, not author errors.
- **Request for confidence intervals for large-scale benchmarks:** Not standard practice in this field and the paper already reports confidence intervals where appropriate.
- **Request for larger datasets / more models:** Generic scope-creep criticism; the current model zoo and dataset sizes are adequate to illustrate the method.

## Novel Insights

The tension between the proposed cluster F1 metric (which rewards purity on non-singleton clusters and can reward coarse lumping) and ARI (which penalizes under-clustering) reveals a genuine methodological challenge in evaluating structure-preservation methods on highly imbalanced cluster distributions. The proprietary dataset failure mode—collapsing 544 clusters into 38 while achieving high cluster F1—suggests that distance-matching objectives may naturally merge many small clusters into a few large ones when the feature space has limited capacity. Future work in this area should explicitly guard against this collapse, for example by incorporating anti-collapse regularization or differentiable cluster-count estimation, particularly when the application domain (e.g., outbreak detection) places high value on detecting small clusters.

## Suggestions

- Revise the claims about epidemiological utility to accurately reflect the proprietary dataset limitations. If the method cannot reliably recover small clusters, it should not be presented as "effectively mimicking epidemiologists."
- Re-run the clusCLS baseline with per-fold label derivation to ensure a fair comparison, or remove/relegate the leaked comparison.
- Add a deployment-style experiment (even a simple train/held-out-test split where only MALDI is available at test time) to substantiate the cost-reduction narrative.

## Score and Decision

**Calibration comparison:**
- **High anchor:** `/home/wg25r/review_agent/human_reviews/ttXg3SKAg5.md` (C3, avg 7.00) — strong theory + experiments, clean methodology. MSPL is well below this; its core real-world result has near-random ARI and its key baseline comparison is compromised.
- **Medium anchor:** `/home/wg25r/review_agent/human_reviews/DL7JWbdGr3.md` (PEM, avg 4.75) — epidemiology application with missing baselines but solid core results. MSPL is below this because its primary real-world result is structurally flawed (ARI ≈ 0, 38 vs 544 clusters) and the leakage taints the main baseline comparison.
- **Low anchor:** `/home/wg25r/review_agent/human_reviews/a4O528mek9.md` (Mul2vec, avg 3.00) — confused methodology and poor presentation. MSPL is substantially above this; it has a clear framework, reasonable presentation, and works on synthetic and DRIAMS data.
- **Additional anchor:** `/home/wg25r/review_agent/human_reviews/ul1cjLB98Y.md` (Unimodal bias, avg 5.25) — restricted setting and questioned novelty, but clean evaluation. MSPL has broader scope but more severe empirical flaws on its primary dataset.

MSPL sits between the low and medium anchors: it has a genuinely novel framing and works on some datasets, but the proprietary dataset failure and the data leakage are significant enough to weigh against acceptance. Relative to the anchor cluster, **4.5** is appropriate.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>