Now let me look at some of the human reviews for calibration.---

## Summary

This paper proposes an Optimal Transport (OT)-based preprocessing algorithm for federated learning (FL) that aligns local client datasets by computing channel-wise Wasserstein barycenters locally, aggregating them at a server to produce a global target distribution, and projecting each client's images toward that target. The goal is to reduce inter-client distributional discrepancy without sharing raw data. The method is evaluated exclusively on CIFAR-10 with FedAvg, demonstrating dramatically higher accuracy relative to the unaligned FedAvg baseline.

---

## Claims and Support

**Claim 1: OT preprocessing minimizes distributional discrepancy across clients.**
The paper asserts this in the abstract and Section 4, but it is *never directly measured*. No Wasserstein distance, MMD, FID, or any distributional divergence metric is reported before vs. after preprocessing. Only downstream classification accuracy is used as a proxy. **Partially supported** at best.

**Claim 2: The method is model- and algorithm-agnostic.**
The preprocessing is architecturally decoupled from the learning loop, which is true by construction. However, empirically only FedAvg is tested in the main paper (ResNet is in the appendix). The method is also RGB-image-specific as explicitly acknowledged in Section 8. The broad "any FL algorithm" framing is unsupported empirically. **Partially supported.**

**Claim 3: The method improves convergence speed and generalization.**
Table 1 shows improvements of ~25–34 percentage points over the FedAvg baseline. However, this claim is severely weakened by the fact that the baseline FedAvg accuracy (65–71%) is anomalously low for CIFAR-10 even under strong non-IID conditions (well-known reference implementations achieve 85%+), while the OT-aligned accuracy (93–99%) is extraordinarily high for a ~1M parameter CNN. This combination strongly suggests either a flawed experimental setup (e.g., incorrect train/test split), data leakage through the global preprocessing step, or that the transformation changes the effective difficulty of the task. **Severely inadequately supported due to experimental validity concerns.**

**Claim 4: The method outperforms prior FL work and achieves "best results."**
Table 2 compares across papers using different models, splits, and hyperparameters; FedAvg alone is reported at 66.3% (Li et al., 2021) and 86.29% (Wang et al., 2020) simultaneously in the same table, making the comparison incoherent. The paper acknowledges "not using the exact same hyperparameters" but still asserts "undoubtedly comparable." **Unsupported — methodological error.**

**Claim 5: WBs preserve privacy because they are "irreversible."**
No threat model, reconstruction attack, or formal privacy bound is given in the visible submission. The claim is asserted but not demonstrated. **Unsupported as stated.**

**Claim 6: Lower computational cost than FedOT.**
Only asymptotic preprocessing complexity is given for the proposed method alone; no empirical runtime comparison against FedOT or any other baseline is provided. **Unsupported as a comparative claim.**

---

## Strengths

- **Novel positioning as a one-shot preprocessing step**: Separating distribution alignment from the training loop is a clean modular design that could in principle be paired with any FL algorithm. This contrasts with FedOT's iterative approach and is a conceptually interesting direction.
- **Clarity of algorithm presentation**: Algorithm 1 and Figures 2–3 clearly communicate the two-step barycenter workflow. The high-level idea is easy to follow.
- **Complexity analysis**: Section 6 provides explicit time complexity for barycenter computation and projection, which is useful for practitioners.

---

## Weaknesses

### Fatal

*(None that single-handedly invalidate the entire contribution, but the combination of the two major experimental issues below comes close.)*

### Major

1. **Suspicious accuracy numbers render the main experimental result uninterpretable.** The FedAvg baseline of 65–71% on CIFAR-10 is anomalously low—standard FedAvg implementations achieve 85%+ even under moderate non-IID conditions. The OT-aligned method reaching 99.62% with a ~1M parameter CNN is extraordinary for CIFAR-10 (near state-of-the-art ResNets achieve ~93–95%). This combination strongly suggests either: (a) the train/test splitting is non-standard, (b) the global barycenter leaks information from the test set or effectively standardizes the problem away, or (c) the data partition is so extreme as to constitute a different task. The paper provides no sanity check (centralized upper bound, local-only baseline, description of per-class imbalance severity) to rule out these explanations. Without resolving this, the magnitude of improvement reported in Table 1 cannot be trusted.

2. **Invalid cross-paper comparison used as the primary evidence of SOTA superiority.** Table 2 conflates results from papers with different architectures, non-IID partitions, hyperparameter budgets, and evaluation protocols. The same algorithm (FedAvg) appears at 66.3% and 86.29% in the same table. The paper acknowledges the mismatch and still claims "undoubtedly comparable" superiority. This is a methodological error. The conclusion that the method achieves "superior results than … other comparable work" (Section 7) is not supported.

3. **No ablation against simpler preprocessing baselines.** The proposed method essentially matches 1D marginal pixel intensity histograms per RGB channel independently—a process mathematically very close to histogram matching or histogram equalization per channel (e.g., Reinhard et al. color transfer). The paper claims novelty over simple normalization by citing a lack of cross-agent alignment, but provides no experiment comparing against: per-channel histogram equalization, Reinhard color transfer, per-channel mean/variance normalization, or local-only barycenter (no global sharing). Without these, it is impossible to attribute any gain to OT specifically.

4. **The core alignment claim is never measured.** The entire paper is motivated by "minimizing the distributional discrepancy" across agents, yet no distributional metric (Wasserstein distance, MMD, FID) is reported before vs. after preprocessing to verify alignment actually occurs. The gap between the paper's thesis and what is actually measured is fundamental.

5. **Evaluation is limited to a single dataset, single algorithm, and one non-IID partition type.** CIFAR-10 with FedAvg and uniform-without-replacement sampling is not representative of the FL heterogeneity literature. Standard benchmarks use Dirichlet-based label skew (α=0.1 or 0.5), pathological non-IID partitions, or cross-silo domain datasets. The method operates on pixel color statistics and has no obvious mechanism to address the dominant form of FL heterogeneity—label distribution skew—yet this is never acknowledged or tested.

### Minor

- **Technically incorrect OT terminology.** Section 3 states: "Intuitively, one is looking for a *permutation matrix* P that determines how to distribute mass in a cost-minimizing fashion." The coupling P in the Kantorovich problem is a transportation matrix whose marginals match a and b—not a permutation matrix (which has exactly one 1 per row/column). Permutation matrices correspond to the degenerate Monge case. This mischaracterization suggests some imprecision in the paper's OT foundations.

- **Informal privacy claim.** The statement "WBs obfuscate the data in an irreversible fashion" is asserted without a threat model or reconstruction analysis. The paper's appendix reportedly elaborates, but even the weaker claim "no raw data is transmitted" would be more accurate and appropriate in the main body.

### Trivial

- The "faster convergence" claim ignores preprocessing cost. Communication round count is used to claim faster convergence, but preprocessing adds a non-trivial one-time cost not included in the comparison.

---

## Nice-to-Haves

- Report results with multiple seeds and standard deviations, since results depend on stochastic client partitioning.
- Visualize original vs. OT-aligned images and the inter-client distributions before/after to give readers intuition for what the transformation does.
- Test whether gains persist when paired with FedProx or SCAFFOLD to substantiate the algorithm-agnostic claim.
- Provide a formal or empirical privacy analysis (e.g., demonstrating that barycenters cannot be used to reconstruct individual samples via membership inference) if privacy is intended as a substantive contribution.
- Evaluate on at least one additional dataset (CIFAR-100, FEMNIST) and one additional heterogeneity setting (Dirichlet partitioning) to assess generality.

---

## Removed Points

*These points were flagged by reviewers but are removed or weakened for the reasons given:*

- **Reviewer claim that the method cannot be independent of the training algorithm because it is RGB-specific.** This is technically a limitation the paper openly acknowledges (Section 8, Future Work) and scopes the current work to colored image datasets. It is a limitation, already addressed as future work—kept only as a minor scope note, not as a hidden flaw.

- **Reviewer claim about missing related works (e.g., Reinhard color transfer, histogram equalization papers).** Per the rules, we do not cite missing related works, as we cannot verify their existence from the submission alone. The *experimental* absence of these as ablation baselines is kept as a major weakness; the *citation* absence is removed.

- **Demand for confidence intervals across all Table 1 results.** Moved to Nice-to-Haves. Single-run evaluations are common in FL papers, and the concern about variance is already subsumed by the more substantive issue of experimental validity raised in the major weakness section.

- **Claim that FedOT runtime comparison is required.** The paper claims conceptual simplicity over FedOT, not measured wall-clock superiority. Requesting a full runtime benchmark is beyond the paper's stated scope; toned down to a trivial note.

- **Cross-paper comparison being "unfair to the paper's method."** Table 2 is not unfair *to* the proposed method—it is being used to *support* the authors' own superiority claim. The REMOVE rule applies when the asymmetry favors the baseline; here the authors are claiming superiority based on a flawed comparison, which is a valid weakness to keep.

---

## Novel Insights

The modular, one-shot preprocessing framing—compute global barycenter once, project locally, then train with any FL algorithm—is a genuinely clean architectural design pattern. The insight that distributional summaries (barycenters) can serve as a privacy-respecting information bottleneck between clients and a server is worth pursuing. However, none of the insights are validated sufficiently in the current submission. The fundamental limitation is that channel-wise 1D color histogram alignment is unlikely to address the dominant FL heterogeneity concern (label skew), making the empirical scope mismatch a conceptual problem, not just an experimental gap.

---

## Suggestions

1. **Verify and explain the accuracy numbers**: Add a centralized upper bound (all data pooled), a local-only lower bound, and confirm the exact train/test protocol. Reproduce the FedAvg baseline using a reference implementation to confirm the 65–71% figure is intentional.
2. **Add ablation against histogram matching and per-channel standardization**: This is the single most important experiment to determine whether OT adds value beyond simple color normalization.
3. **Measure distributional alignment directly**: Report inter-client Wasserstein distance or MMD before and after preprocessing to verify the paper's core claim.
4. **Reframe Table 2**: Replace the cross-paper comparison with a self-contained controlled comparison—re-run FedProx, SCAFFOLD, or MOON under exactly the same partition, architecture, and budget.
5. **Test on label-skew heterogeneity**: Use Dirichlet partitioning (α=0.1, 0.5) to determine whether channel-level color alignment helps when the main source of heterogeneity is label distribution rather than color statistics.

---

## Score and Decision

**Calibration:**

I compared against five FL papers from human reviews:
- **FLea** (LGzTtvisL3): Scores 3/6/5/5 → Rejected. Single dataset/model limitation, but had multi-round evaluation and privacy analysis.
- **FedPALS** (nwETBpOPiC): Scores 3/6/3/5/3 → Rejected. Has theoretical proof and more experiments but weak baselines.
- **FedDualMatch** (In0phMs7BK): Scores 3/5/3/5/3 → Rejected. More experiments (multiple datasets), formal theory, still rejected.
- **ANFR** (uBEl8DMA8K): Scores 5/5/5/6 → Withdrawn. Multiple datasets, comprehensive ablation, formal DP analysis.

**Assessment relative to anchors:** This paper is *weaker* than all four comparison papers. FLea, FedPALS, and FedDualMatch were rejected despite having more thorough evaluations, multiple datasets, and theoretical backing. ANFR, which scored 5–6, had comprehensive experiments and ablations. The paper under review has one dataset, one algorithm, no ablations, suspicious accuracy numbers that undermine experimental validity, and an invalid SOTA comparison. The core claim (alignment reduces discrepancy) is never directly measured.

**Axes:**
- *Originality*: Moderate idea (one-shot barycenter preprocessing); method is incremental given existing OT color transfer literature.
- *Importance of research question*: FL heterogeneity is an important problem, but the paper targets the wrong form (color statistics vs. label skew).
- *Claims vs. support*: Very weak; headline accuracy numbers are suspicious, SOTA comparison is methodologically flawed.
- *Soundness of experiments*: Poor; single dataset, single algorithm, no controls, no ablations, anomalous numbers.
- *Clarity of writing*: Acceptable structure but overclaims throughout.
- *Value to research community*: Low in current form; the idea has some merit but is unvalidated.

**Final Score: 2.5 — Strong Reject**

MY FINAL SCORE: <pineapple>2.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>