## Summary
This paper studies offline change point localization and inference in dynamic multilayer random dot product graphs, proposing a two-stage procedure that combines seeded binary segmentation with tensor-based low-rank refinement. Its main technical claims are consistency for estimating both the number and locations of change points, plus limiting distributions and a data-driven confidence interval procedure for refined estimators.

## Strengths
- The paper tackles a genuinely specific and underexplored problem: **offline** change point localization and inference for **dynamic multilayer** latent-position network models, rather than either single-layer settings or online detection. The contribution is clearly scoped in Section 1.1 and is not just a minor variant of prior single-layer results.
- The methodological design is nontrivial and well integrated: Stage I uses seeded binary segmentation to obtain coarse candidates, and Stage II uses localized TH-PCA-based refinement. This is more than a generic pipeline; the tensor refinement is tailored to the multilayer low-rank structure induced by the D-MRDPG model.
- The paper provides substantial theory beyond consistency. In particular, it derives asymptotic distributions for refined estimators in both vanishing and non-vanishing jump regimes (Theorems 2 and 3), which is a meaningful advance over papers that stop at rate bounds.
- The experiments include several useful robustness checks beyond the main table: sensitivity to threshold and rank choices, performance under random change-point locations, temporal dependence stress tests, and some out-of-model scenarios. These help characterize where the method works well and where it degrades.
- The real-data analyses are not purely decorative: the method identifies interpretable change points in agricultural trade and U.S. air transportation data, and the paper attempts to connect detected changes to plausible domain events rather than reporting dates without interpretation.

## Weaknesses

### Major:
- **Theory–implementation mismatch for the independence assumptions.**  
  The theoretical analysis of Algorithm 1 assumes four mutually independent sequences \(\{A\}, \{A'\}, \{B\}, \{B'\}\), and this independence is used explicitly in the proofs. For example, Section 2.2 states: “**The assumption of mutual independence among the four sequences in Algorithm 1 is imposed for theoretical convenience. In practice ... Stage I and Stage II are implemented using the same two split tensor sequences via the odd–even splitting approach.**” The proof of Theorem 1 then conditions on the Stage I event and uses independence to justify that the distribution of the refinement sample is unaffected. As written, the main guarantees therefore apply to an idealized sample-split version, not directly to the exact implementation used in experiments. This is a substantive gap because the paper presents the empirical method and the theorem-backed method too closely, without a formal reconciliation.
- **The confidence-interval procedure is only partially justified by the theory provided.**  
  Section 3.1 presents a fully data-driven CI construction using plug-in estimates of the jump tensor and variances, but the paper does not provide a theorem proving that these plug-in estimates preserve the limiting law or yield asymptotically valid coverage. Theorems 2 and 3 establish limiting distributions for refined estimators, but the leap from those population-level limits to the practical plug-in CI algorithm is not fully closed. The empirical coverage study helps, but it does not replace a validity argument for the proposed interval construction.
- **The paper’s strongest empirical claim is somewhat overstated relative to the main-table comparisons.**  
  The abstract and contribution list claim superiority over “existing state-of-the-art algorithms,” but the main text primarily compares against gSeg and kerSeg, which are generic change-point methods adapted to network-derived inputs. The more directly relevant comparison to a dynamic multilayer network method (CPDonline from Wang et al., 2025) is deferred to the appendix. Since the paper’s core selling point is being the first offline method in this setting, the empirical evidence would be stronger if the most relevant adapted network comparator were featured centrally rather than peripherally.

### Minor
- **The theoretical regime is restrictive in a way that matters for the claimed scope.**  
  Model 1 assumes \(\Delta=\Theta(T)\), which effectively keeps the number of changes bounded. The paper is transparent about this and discusses it in Section 5, but it remains a real limitation because many dynamic-network applications involve more frequent changes. The appendix includes some experiments with larger numbers of change points, which is useful, but those experiments sit outside the main theory.
- **The low-rank assumptions are somewhat abstract and not especially interpretable from the network model itself.**  
  Assumption 1(ii)–(iii) imposes rank conditions on transformed \(Q\)-matrices. The authors themselves note: “**this low-rank structure may not directly or transparently reflect the explicit model structure**.” That honesty is appreciated, but it also means the assumptions are not especially natural from a modeling standpoint, which weakens the practical interpretability of the theory.
- **Finite-sample reliability of the inference procedure looks uneven outside the model class.**  
  Table 2 shows 76.67% coverage for a nominal 95% CI in Scenario 3 when \(n=100\), and the paper notes that this scenario violates Model 1 and involves smaller layer-specific changes. This does not invalidate the theory—since the model is violated—but it does indicate that the practical CI procedure can be fragile, and the paper could do more to state when users should distrust the intervals.
- **Scalability is a practical concern.**  
  The complexity is quadratic in \(n\), and Appendix G reports about 10 hours for \(n=100, T=200\) over 100 Monte Carlo trials on a CPU. This does not make the method unusable, but it does suggest that the current implementation may be heavy for larger multilayer networks, especially given that Stage II uses repeated tensor estimation.

### Trivial
- The real-data confidence intervals can look extremely sharp relative to the small time horizons (e.g., agricultural trade with \(T=35\)), which invites skepticism about finite-sample calibration even if not a formal contradiction of the asymptotic theory. A short cautionary discussion would help.

## Nice-to-Haves
- Include an ablation directly showing how much Stage II refinement improves over Stage I, e.g., by plotting raw CUSUM versus refined scan statistics around true changes.
- Add an experiment on rank misspecification beyond the limited sensitivity table, since Stage II depends on low-rank tensor estimation.
- Clarify practical guidance for threshold calibration, since the paper tunes \(c_{\tau,1}\) via null simulations in the appendix rather than giving a fully automatic selection rule.
- Expand the discussion of how one should construct confidence intervals in the non-vanishing jump regime, since the main text focuses on the vanishing regime.
- Provide a stronger computational discussion, including where the time is spent and whether sparsity or warm-starting TH-PCA could reduce cost.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing stronger offline multilayer baselines” as a criticism of related work coverage.**  
  It is fair to say the main empirical comparison could foreground more relevant comparators, which I keep above. But claims framed as “missing related methods” or assuming unspecified baselines should exist are not reliable to include as core weaknesses without external verification.
- **Criticism that the paper should not evaluate on scenarios violating Model 1.**  
  The paper explicitly says Scenarios 2 and 3 do **not** follow Model 1 and uses them to assess robustness: “**The changes in Scenarios 1 and 4 follow Model 1, while those in Scenarios 2 and 3 do not, allowing us to assess the robustness of our methods.**” So treating this as a flaw would misunderstand the purpose of those experiments.
- **Complaint that frequent-change experiments invalidate the paper because \(\Delta=\Theta(T)\) is assumed.**  
  The theory indeed does not cover those settings, but the appendix experiments are clearly presented as robustness studies outside the theory, not as evidence for the theorem. The right criticism is that the theory is restrictive, not that these experiments are illegitimate.
- **Reproducibility nitpicks about releasing exact seeds/code/preprocessing.**  
  These are not substantive enough for the main review under the stated rubric.
- **Purely generic strengths such as “the paper is well written” or “the experiments are extensive.”**  
  Omitted because they are not specific enough.

## Novel Insights
The most important synthesis point is that this is a paper with **real theoretical ambition and genuine novelty**, but its strongest vulnerability is not the mathematics itself—it is the gap between the **idealized sample-splitting device used to obtain clean proofs** and the **practical odd–even implementation actually evaluated**. A second key insight is that the paper is strongest on localization theory and weakest on inference calibration: the asymptotic limit laws are interesting and likely valuable, but the practical CI procedure currently feels one theorem short of being fully justified. So this is not a weak paper in the usual sense; rather, it is a strong technical paper whose final inferential and empirical claims are somewhat ahead of what is rigorously established.

## Suggestions
- Prove a version of the main localization and inference results for the actual odd–even implementation, or clearly restate theorems as applying only to the idealized split-sample algorithm and discuss the gap explicitly.
- Add a theorem or proposition establishing asymptotic validity of the plug-in CI procedure, including consistency of \(\hat\kappa_k\), \(\hat\Psi_k\), and the variance estimators.
- Move the CPDonline comparison from the appendix into the main experimental section, and moderate “state-of-the-art” language unless the main text supports it directly.
- Add a concise warning box or proposition describing when CI coverage may degrade in practice, especially under model misspecification or short time horizons.
- Include an ablation isolating the gain from TH-PCA refinement over Stage I alone, both in localization error and computational cost.

