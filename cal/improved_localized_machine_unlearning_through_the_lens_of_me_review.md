=== CALIBRATION EXAMPLE 18 ===

# Final Consolidated Review
## Summary

This paper investigates localized machine unlearning by drawing inspiration from the memorization literature. The authors systematically evaluate four localization strategies (Deepest, Shallowest, CritMem, SalLoc) under a unified budget-controlled framework, yielding insights about which design choices matter. From these investigations, they derive a new non-iterative, channel-level, weighted-gradient localization strategy that, when paired with a Reset+Finetune algorithm (forming DEL), outperforms both localized and full-parameter state-of-the-art methods on unlearning efficacy and utility across CIFAR-10/ResNet-18 and SVHN/ViT experiments.

---

## Strengths

- **Modular and principled framework.** The explicit separation of the localization strategy $\mathcal{L}$ from the unlearning algorithm $\mathcal{U}$ (Figure 1) is both conceptually clarifying and practically useful — it enables rigorous mix-and-match ablation (Figure 3) and will help future work isolate where gains come from.

- **Shallowest as an underappreciated strong baseline.** The finding that shallowest-layer masking achieves near-oracle forget accuracy and MIA efficacy at moderate budgets (Figure 2), while sacrificing test accuracy, is a concrete contribution to the community's benchmarking vocabulary. The authors explicitly recommend its inclusion alongside EU-k in future benchmarks, a recommendation backed by evidence rather than opinion.

- **Ablation rigorously validates design choices.** Table 1 evaluates all four combinations of {parameter, channel} granularity × {gradients, weighted gradients}, clearly demonstrating that channel + weighted gradients is the best combination. This is not post-hoc rationalization; the ablation is a first-class result.

- **Control experiment (Table 3) cleanly isolates parameter selection quality.** The random-vs-standard masking experiment, where random masks match the per-layer parameter count distribution of the structured mask, provides strong, direct evidence that *which* parameters are chosen drives performance, not merely *how many* — a non-trivial result.

- **Forget-set-agnostic localization insight (Table 4).** The finding that for IID forget sets, replacing the forget-set-specific weighted gradient criterion with a training-set-computed variant yields similar performance is a practically actionable insight: practitioners could precompute masks before receiving deletion requests when the forget set is IID with the training distribution.

- **Intellectually honest discussion.** Section 7 explicitly acknowledges that CritMem — the most direct translation of memorization hypotheses — fails to outperform simple baselines, and situates this alongside Hase et al. (2024) and Guo et al.'s parallel findings in LLMs. This calibrated framing is rare and commendable.

---

## Weaknesses

### Fatal
None.

### Major

- **SCRUB is absent from Table 2 without explanation.** SCRUB (Kurmanji et al., 2024) is explicitly introduced in Section 3 as building on NegGrad+ with knowledge distillation and presented as a stronger full-parameter method. Yet it does not appear in Table 2 or any other comparison table, with no justification given. The paper's central claim — "DEL sets a new state-of-the-art…against both localized and full-parameter methods" — is incomplete without it. If SCRUB is omitted because it underperformed NegGrad+ in preliminary experiments, that should be stated; if it was inadvertently excluded, it should be added.

- **The hyperparameter $h$ is never specified or analyzed.** Section 5.2 defines neuron criticality as the average of the top-$h$ criticality scores among a channel's parameters ($c_{o_i} = \frac{1}{h}\sum_{j=1}^{h}\tilde{s}_i[j]$), but the value of $h$ used in experiments is never stated in the main text. No sensitivity analysis is provided, and no guidance is given for choosing $h$ for new architectures (e.g., ViT attention heads vs. ResNet convolutional channels, which differ substantially in channel size). This omission prevents faithful reproduction and undermines claims about the method's generality.

- **Membership inference evaluation is weak.** The paper uses the Fan et al. (2023) confidence-threshold MIA throughout, while itself acknowledging that state-of-the-art evaluation methods exist (Hayes et al., 2024; Triantafillou et al., 2024). The adopted MIA applies a fixed threshold on confidence scores rather than a likelihood-ratio test, and is known to be a weak adversary. If a stronger single-model attack (e.g., LiRA-style) reveals residual membership signal that the confidence threshold misses, the unlearning efficacy rankings could change. The paper's honesty in flagging this does not mitigate the risk that the empirical claims are evaluated against an insufficiently challenging adversary.

- **Experimental scope is narrow for an ICLR-level "state-of-the-art" claim.** All experiments use exactly 10% of the training set as the forget set. All main CIFAR-10 results use ResNet-18; SVHN/ViT results are relegated to Figure 4 (radar only) and "Table 8 in the Appendix" (not visible in the submitted manuscript). Neither dataset is large-scale. The combination of two small datasets, a single forget-set size regime, and two architectures limits confidence in the generality of the findings, especially given the method's architectural sensitivity (channels in CNNs vs. attention heads in ViTs is handled differently but not analyzed).

### Minor

- **The rationale for choosing Reset+Finetune (RFT) is purely empirical.** The paper states "we find we obtain strongest results by pairing it with the simple Reset + Finetune algorithm" without a principled explanation. Why reinitializing the most critical parameters (the ones most responsible for memorizing the forget set) and then retraining from a random initialization outperforms gradient-based alternatives like NegGrad+ is left as an open question. The intuition around "disruption" is suggestive but underdeveloped.

- **No wall-clock runtime comparison against SalLoc.** The paper argues that the proposed strategy is more efficient than CritMem due to being non-iterative and batched, which is clearly true. However, since both the proposed method and SalLoc are one-shot gradient-based approaches, there is a non-trivial overhead from channel-level aggregation versus parameter-wise sorting. A brief runtime table comparing the localization step across CritMem, SalLoc, and the proposed method would substantiate the efficiency claims for the most relevant comparison.

- **Performance margins over baselines on IID forget sets are modest.** In Table 2 for the IID setting, DEL achieves $\Delta_{\text{forget}} = 0.97 \pm 0.42$ and $\Delta_{\text{MIA}} = -0.97 \pm 0.40$, while Random Label achieves $1.69 \pm 0.46$ and $1.69 \pm 0.47$ respectively. DEL is better in magnitude across both metrics, but the gap is moderate. The Non-IID setting shows a clearer advantage; the paper should be careful not to over-claim for the IID case.

### Tiny

- **The "lens of memorization" narrative is partially undermined by its own findings.** The most direct memorization-motivated strategies (Deepest, CritMem) fail, and the paper's actual contribution borrows only the granularity and criticality criterion from memorization work, not a direct translation of memorization hypotheses. The Discussion section handles this honestly, but the title and introduction set up an expectation that the paper partially contradicts. This is not a scientific flaw but creates a slight narrative inconsistency that could be smoothed.

---

## Nice-to-Haves

- **Stronger MIA evaluation (LiRA or shadow-model attacks).** Even a single supplementary experiment with a stronger attack would substantially reinforce the unlearning efficacy claims.

- **Forget set size sensitivity.** Testing with smaller forget sets (e.g., 0.1%–5% of training data) would connect the work to the primary privacy-motivated use case (GDPR individual deletion) and characterize when localized unlearning breaks down.

- **Larger-scale validation.** Even a subset of ImageNet with ResNet-50 would increase confidence that channel-level selection and the weighted gradient criterion scale to architectures used in practice.

- **Theoretical grounding for weighted gradients.** Connecting $s_j = |\theta_j \cdot g_j|$ to influence functions or second-order Taylor approximations of the loss change upon parameter deletion would elevate the contribution beyond an empirically-validated heuristic.

- **Visualization of mask spatial distribution** across layers, compared to SalLoc, to verify that the proposed strategy genuinely concentrates on specific functional regions rather than spreading uniformly.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"30% of ResNet-18 is not genuinely small"** (Harsh Critic): Subjective and does not undermine any claim. The paper clearly states performance holds across multiple budget levels (20%–30%) and provides the absolute percentages throughout.

- **Hyperparameter tuning contamination concern** (Harsh Critic): The paper states hyperparameters are tuned separately for each budget and strategy; this is standard practice. Without direct evidence of test-set leakage in the tuning procedure, this speculation should not be included as a weakness.

- **Shallowest test accuracy drop — absolute values not given** (Harsh Critic): This is a valid question but Figure 2 makes the trend clear (test accuracy ~83.5% → ~81% at large budgets), and the paper explicitly discusses the tradeoff. Demanding precise absolute numbers for what is already a labeled control baseline is excessive.

- **Requesting confidence intervals for large-scale benchmark comparisons** (implied by multiple reviewers calling for more statistical testing): Single-run evaluations at this scale are standard in the unlearning community; the paper already reports standard deviations in all tables.

- **Requesting user studies or theoretical proofs for the unlearning algorithm**: This is an empirical systems paper; neither is expected.

---

## Novel Insights

The most genuinely novel observation — emerging from the interplay of the three reviews and the paper's own findings — is the **structural disconnect between memorization localization and unlearning localization**. The paper empirically shows that a direct transplant from memorization research (CritMem) fails, yet the *design vocabulary* from memorization research (channel granularity, weighted-gradient criticality) yields the best unlearning performance when combined with an efficient, non-iterative selection scheme. This suggests that the memorization and unlearning research programs share useful *inductive biases* (coarser granularity is more robust, parameter magnitude matters alongside gradient magnitude) without sharing the same *mechanistic answers* about where in the network to intervene. The parallel finding in Table 4 — that forget-set-agnostic criteria suffice for IID settings — further implies that the channel+weighted-gradient combination may succeed not because it localizes memorization precisely, but because it finds parameters that are both influential for the training data broadly and disproportionately loaded with forget-set signal, a hybrid "general criticality with forget-set amplification" effect that the paper does not yet fully characterize.

---

## Suggestions

1. **Add SCRUB to Table 2**, or explicitly explain why it was excluded (e.g., a brief appendix note with a result or citation showing it performs similarly to NegGrad+ on these benchmarks).

2. **Specify the value of $h$ used in all experiments** and add a sensitivity table (e.g., $h \in \{1, 5, 10, \text{all}\}$ on CIFAR-10/ResNet-18) so the hyperparameter's role is transparent and reproducible.

3. **Report wall-clock time** for the localization step (and optionally the full pipeline) for CritMem, SalLoc, and the proposed method in a table or paragraph in Section 6 or the appendix.

4. **Include at least one experiment with a smaller forget set** (e.g., 1% random, or a single class) to characterize where DEL's advantage holds and where it may degrade.

5. **Clarify hyperparameter selection procedure** — specifically, whether the best budget was selected on a held-out validation split or post-hoc on the reported evaluation data, and ensure Table 2 comparisons are based on the same selection protocol for all methods.

6. **Expand the SVHN/ViT results** into the main paper (at minimum a table analogous to Table 2), since currently Figure 4 alone (a radar chart without absolute values) is insufficient to assess statistical significance of the SVHN improvements.

---

**Axis assessments:**

- **Novelty:** Moderate. The combination of channel granularity + weighted gradients + non-iterative batched selection is new and ablation-validated, but the conceptual gap from SalUn is incremental. The memorization-lens framing and the benchmark insights (Shallowest baseline, Table 3 control, Table 4 IID analysis) add broader scientific value.
- **Technical soundness:** Moderate-to-good. The method is well-motivated and ablated. The missing $h$ specification and the absence of SCRUB are concrete gaps in rigor.
- **Empirical support:** Moderate. Results on CIFAR-10 are solid and well-structured; SVHN/ViT results are suggestive but underreported. The single forget-set-size regime and weak MIA limit the strength of unlearning efficacy claims.
- **Significance:** Moderate. Localized unlearning is practically important; the specific improvements demonstrated here are meaningful for the community. Broader impact is constrained by the small-scale experimental setting.
- **Clarity:** Good. The paper is well-organized, the framework is clearly explained, and the Discussion section is unusually self-critical and informative.

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 6.0, 5.0]
Average score: 5.0
Binary outcome: Reject
