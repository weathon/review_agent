=== CALIBRATION EXAMPLE 17 ===

# Final Consolidated Review
Now I have a thorough understanding of the paper. Let me produce the consolidated final review.

---

## Summary

This paper proposes an Optimal Transport (OT)-based preprocessing algorithm for Federated Learning (FL) that aligns local data distributions before training begins. The method computes channel-wise Wasserstein barycenters of RGB pixel intensity histograms locally, aggregates them at a central server to form a global target color space, and then projects each agent's local images to this target space. The preprocessing is claimed to be model- and algorithm-agnostic, privacy-preserving, and to substantially improve convergence speed and test accuracy on CIFAR-10 when paired with FedAvg.

---

## Strengths

- **Decoupled, plug-in preprocessing design:** The method operates entirely before FL training begins and is decoupled from the optimization algorithm. This is architecturally cleaner than iterative alignment methods like FedOT (Farnia et al., 2022), which couple alignment with model training and add overhead to every communication round. A one-shot preprocessing step that does not touch the learning algorithm is a genuinely practical design choice.

- **Complexity analysis with parallel-aware decomposition:** Section 6 provides a concrete breakdown: O(Md²/ε²) for local barycenters (parallel across agents), O(Nd²/ε²) for the global barycenter, and O(Md²) for projection. The observation that local barycenter computation and local projection both parallelize across agents is non-trivial and demonstrates awareness of the edge-compute setting.

- **Clear positioning against FedOT:** The paper explicitly explains the conceptual distinction from Farnia et al. (2022): rather than iteratively learning both the transport map and the target space jointly, this method precomputes the target space in one shot. The two claimed benefits (unified alignment target, lower computational cost) are clearly stated.

---

## Weaknesses

### Fatal

*None that are individually fatal in isolation, but the combination of (1) an ambiguous and likely near-IID experimental partition producing implausible accuracy numbers and (2) methodologically invalid cross-paper comparisons collectively make the empirical claims of the paper unsupportable in their current form.*

### Major

- **Fundamentally ambiguous and likely near-IID experimental partition.** Section 5 states data is distributed by "uniformly sampling, without replacement." Uniform sampling from a pooled dataset of 50,000 images concentrates per-class fractions near the global proportions by the law of large numbers — this is approximately IID, the easiest regime for FedAvg. Yet the reported FedAvg baseline is 65–71% accuracy, which is dramatically below the 85–90%+ that FedAvg routinely achieves on CIFAR-10 even under substantial non-IID skew in the established literature. The paper provides no Dirichlet α value, no shard partition description, and no per-client class histogram, making it impossible to assess the actual degree of heterogeneity. This is not a nitpick: the entire empirical contribution depends on understanding whether the experimental setting is genuinely non-IID.

- **Implausible accuracy figures.** Table 1 reports 99.62% test accuracy on CIFAR-10 with 5 agents using a small two-conv-layer CNN (~1M parameters) under the proposed method, and values above 93% across nearly all configurations. State-of-the-art centralized models on CIFAR-10 with heavily tuned large architectures achieve ~99%. A small CNN in any FL setting — which is supposed to be harder than centralized — reaching or exceeding centralized SOTA is extraordinary and requires a clear explanation. These numbers are either a consequence of the partition being effectively IID (making the task trivially easy) or an experimental artifact. Neither possibility supports the paper's core claim of solving a hard non-IID problem.

- **Methodologically invalid Table 2 comparisons.** The paper compares its 93.34% (N=100, P=10) against FedMA (87.53%), FedProx (85.32%), and FedAvg (86.29%) from Wang et al. (2020) — all obtained under different model architectures, data partitioning schemes, and hyperparameters. The paper explicitly acknowledges "not using the exact same hyperparameters" while simultaneously claiming its results are "undoubtedly comparable." These are contradictory positions. If the author's partition is approximately IID (as argued above), the proposed method is operating in a qualitatively easier regime. Claiming to surpass FedMA under these conditions is not a valid empirical finding.

- **Conceptual mismatch between stated problem and proposed solution.** The Introduction and Abstract frame the problem as "dataset imbalance" where agents "do not have equal representation of the labels." Label/class imbalance is the canonical non-IID challenge in FL. However, the proposed solution aligns marginal per-channel pixel intensity histograms. Matching RGB histograms does not redistribute class labels: if Agent A holds mostly "airplane" images and Agent B holds mostly "automobile" images, their channel histograms might be made identical by OT projection while their label distributions remain completely different. The paper provides no theoretical argument or empirical evidence that RGB histogram alignment reduces the gradient divergence caused by label skew. This gap between the motivation and the mechanism is never addressed.

- **No ablation against trivially simple baselines.** The core operation is 1D distribution matching per color channel. Per-channel histogram equalization, min-max normalization to a shared range, or per-channel mean/variance standardization to a shared target are all far simpler operations with similar intent. Without ablation showing the Wasserstein barycenter specifically is necessary — as opposed to any distribution-matching procedure — the contribution of the OT formulation is unverified.

### Minor

- **No empirical comparison with FedOT.** The paper devotes significant text to positioning itself against FedOT (Farnia et al., 2022), describing it as "the most relatable work." Despite this positioning, no empirical comparison is provided. The claimed computational advantage over FedOT is stated but not measured. A single head-to-head experiment on CIFAR-10 would directly substantiate the claim.

- **Key implementation step deferred to missing appendix.** Algorithm 1, Step 4 states "Project image i → WB^G" without specifying how the projection is computed. The actual transport map computation is deferred to Appendix A.2, which is not present in the submission. The projection is the mechanism of the method; it should be described in the main text at ICLR.

- **Privacy claims are informal and deferred to missing appendix.** Section 4 asserts that "WBs obfuscate the data in an irreversible fashion" and forwards the reader to Appendix A.1 (also not present). A channel-wise pixel intensity histogram is a compact but informative summary of local data. Whether sharing it satisfies any standard privacy criterion is a substantive open question, not a self-evident claim.

- **Single dataset, single architecture.** All reported experiments use CIFAR-10 with one CNN. The ResNet results are in Appendix A.3 (not provided). For a method claimed to be "model- and learning algorithm-agnostic," evaluation on a single dataset with a single algorithm and architecture provides very limited evidence of generality.

### Tiny

- **No statistical significance reporting.** All entries in Table 1 appear to be single runs. Given stochastic data partitioning and optimization, variance across seeds could be non-trivial. Reporting multiple runs is especially important here because the claimed gains (65% → 99%) are so large that a single anomalous run would inflate them.

- **Communication round counts are very large.** The method requires 500–1000 communication rounds for larger networks. While the paper attributes this to FedAvg's simplicity, the large round count partially offsets the claimed communication efficiency advantage shown in Figure 4.

---

## Nice-to-Haves

- Evaluate under standard Dirichlet partitioning (α = 0.1, 0.5) to produce a properly calibrated non-IID baseline that aligns with FL literature benchmarks. This would also make the FedAvg baseline credible.
- Report wall-clock preprocessing time vs. training time to demonstrate the net efficiency gain; theoretical complexity does not capture practical cost.
- Provide inter-client Wasserstein distance before and after preprocessing to directly verify the alignment claim.
- Derive a convergence bound connecting the reduction in Wasserstein distance between client distributions to a reduction in the gradient divergence term in the FL convergence analysis.
- Integrate with a variance-reduction method (e.g., SCAFFOLD) to determine if the benefits are complementary.
- Visualize pre- and post-alignment images to verify that OT projection does not introduce visual artifacts or distort semantic content.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Domain alignment" overstates the method (terminology):** The critic argues the title should say "color histogram alignment." While technically precise, this is a scope/framing complaint rather than a factual error. The method does perform a form of marginal distributional alignment; the naming is conventional in the color transfer literature. Removed as a pure style nitpick.

- **Missing color transfer literature:** The critic notes that the method is "mathematically equivalent to Wasserstein-based color transfer (Reinhard et al., 2001 and extensions)" and flags the omission of this literature. Per review policy, missing related works are not cited as weaknesses, as external sources cannot be confirmed. Removed.

- **"Zero-shot" is undefined and the claim is not novel:** The critic argues that histogram equalization satisfies the same zero-shot criterion. This may be correct, but comparing to unnamed unlisted prior methods constitutes a missing-related-works complaint. Removed.

- **No statistical significance for large-scale benchmarks:** The critic demands confidence intervals. For the scale of experiments here (CIFAR-10 with a small CNN), multi-seed statistics would be valuable — this has been retained as a Tiny weakness. The argument that single-run evaluation is the norm for large-scale benchmarks does not apply at this scale, so this point is partially retained rather than fully removed.

---

## Novel Insights

The most genuinely insightful observation that cuts across all three reviews is the **internal inconsistency between the problem framing and the proposed mechanism**: the paper motivates itself with label distribution imbalance (the canonical FL non-IID problem) but implements a color histogram alignment that is orthogonal to class label distributions. This is not a criticism that reviewers manufactured — it is visible in the paper itself, which in the introduction explicitly defines imbalance as unequal label representation and then in Section 4 computes a global RGB barycenter over all images regardless of class. If the experimental partition is approximately IID (uniform sampling without replacement), the color histograms across agents are already similar, and the large accuracy gain attributed to OT alignment is therefore explained not by the alignment itself but by the near-IID regime making FedAvg nearly optimal — which in turn makes the baselines look bad and the proposed method look good. The real empirical question — does RGB histogram alignment help in a truly label-skewed, color-style-shifted setting (e.g., satellite vs. camera imagery of the same classes) — is never tested.

---

## Suggestions

1. **Re-run all experiments with an explicit, reproducible non-IID partition** (e.g., Dirichlet α = 0.1 and α = 0.5 with published per-client class histograms). This is the single most important fix: it either validates or invalidates the core claim.

2. **Re-implement FedMA, FedProx, and SCAFFOLD within your own codebase** under the same partition, architecture, and hyperparameters before claiming superiority over them.

3. **Add an ablation table** comparing OT barycenter projection against: (a) per-channel mean/variance standardization to a shared target, (b) per-channel histogram equalization, and (c) no alignment. This isolates the contribution of the Wasserstein formulation.

4. **Move the projection mechanism (Appendix A.2) into the main text** — it is the operative step of the method and cannot be left undefined in Algorithm 1.

5. **Provide a formal or empirical privacy analysis** of sharing channel-wise Wasserstein barycenters (e.g., a reconstruction or membership-inference experiment), or explicitly scope the privacy claim to "raw images are not shared" and remove the stronger "irreversible obfuscation" language.

6. **Address the label-skew/color-alignment gap** either theoretically (show that color alignment implies reduced gradient divergence under some distributional assumption) or empirically (test on a dataset where domain shift co-occurs with label shift, such as DomainNet or Office-Home federated splits).

---

**Overall evaluation:** The core idea — one-shot OT-based preprocessing to align client data distributions before FL training — is conceptually interesting and architecturally appealing. However, the paper's empirical foundation is currently unreliable: the non-IID partition is insufficiently described and is likely near-IID, the resulting accuracy figures are implausible, and the cross-paper comparisons are uncontrolled. The theoretical justification for why color histogram alignment addresses the label skew problem that motivates the paper is absent. In its current form, the paper does not meet the empirical or theoretical standards expected at ICLR. A substantially revised version with a rigorously defined non-IID setup, controlled comparisons, and ablations against simpler baselines would be needed to support its claims.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 5.0, 5.0]
Average score: 4.5
Binary outcome: Reject
