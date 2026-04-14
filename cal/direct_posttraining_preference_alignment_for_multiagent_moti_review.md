=== CALIBRATION EXAMPLE 19 ===

# Final Consolidated Review
## Summary
DPA-OMF proposes to use Optimal Transport (OT) over occupancy measures to rank a pre-trained multi-agent motion generation model's own rollouts relative to expert demonstrations, constructing preference pairs without any human annotation. These pairs are then used to fine-tune the model via a multi-agent extension of Contrastive Preference Learning. Applied to large-scale traffic simulation (up to 128 agents) using a 1M-parameter MotionLM model on the Waymo Open Motion Dataset, the method achieves a composite realism of 0.739 — significantly above the reference model's 0.721 and within reach of models 30–100× larger — while offering a clear conceptual improvement over adversarial AFD approaches.

---

## Strengths

- **Principled distinction from adversarial AFD with mechanistic evidence.** Rather than treating all generated samples as negative (which suppresses nuanced signal), DPA-OMF ranks model generations against each other via OT distance. Figure 6 provides a clear mechanistic explanation of *why* adversarial AFD fails: during training, AFD simply pushes up the likelihood of expert demonstrations while leaving the preferred/unpreferred generated samples indistinguishable. This causal analysis, not just performance numbers, distinguishes the paper from a pure empirical comparison.

- **Directly addresses the heterogeneity problem in AFD.** By constructing preference pairs entirely from the reference model's own generations, DPA-OMF avoids the distribution mismatch between human demonstrations and model outputs that plagues AFD methods such as Chen et al. (2024b). The paper identifies and names this problem precisely and shows that it matters in practice (Table 3: classification accuracy 0.84 vs 0.52 for adversarial AFD, realism 0.739 vs 0.720).

- **Significant parameter-efficiency result.** Achieving 0.739 composite realism with a 1M-parameter model versus BehaviorGPT (3M, 0.747), Trajeglish (35M, 0.721), and SMART (102M, 0.761) under alignment constraints that add no human cost is a practically meaningful finding for the community, even if the gap with the strongest baseline is non-negligible.

- **Informative over-optimization and scaling analysis.** Figure 7 (right) reveals that scaling preference data shifts the Pareto frontier between KL divergence and realism, and that insufficient data causes performance-degrading over-optimization. This analysis mirrors analogous studies in LLMs (Tang et al., 2024) and transfers insights to the embodied domain in a concrete, empirically grounded way.

---

## Weaknesses

### Fatal
None.

### Major

- **Shared feature space between OT metric and evaluation metric raises a circularity concern.** The OT preference distance is computed using features {collision status, distance to road boundary, minimum clearance, control effort, speed}, and the paper explicitly states: *"These features are also used to encode the agent's state in the realism metric."* The WOSAC realism metric and the OT preference distance are mathematically distinct (likelihood of expert trajectories under 32 rollouts vs. OT transport cost between empirical occupancy measures), and the paper correctly notes they are not identical. However, the shared feature basis means that optimizing the OT-based ranking objective is not independent of the evaluation criterion in a meaningful sense: a model that better matches these specific features will, by construction, tend to score higher on both. This does not invalidate the method, but the paper should either (a) demonstrate that improvement persists when evaluating on sub-metrics that do *not* directly correspond to OT features (e.g., Kinematic vs. Map compliance separately), or (b) provide a more rigorous argument about why feature sharing does not constitute metric gaming. As it stands, this weakens the claim that DPA-OMF improves *genuine human preference alignment* rather than a surrogate metric aligned to the benchmark.

- **No statistical significance for the key results.** Table 1 reports DPA-OMF at 0.739 vs. SFT-bestOA at 0.723, a delta of 0.016, with no error bars, confidence intervals, or multiple-seed results. For preference alignment methods known to be sensitive to sampling and initialization, this omission makes it impossible to assess whether the gain is reliable. While single-run evaluation is norm for WOSAC leaderboard submission, the internal comparisons (Table 1, Table 2, Table 3) are not leaderboard evaluations and should carry variance estimates.

- **Computational cost of OT not quantified.** The paper claims DPA-OMF is "computationally efficient" relative to RLHF and adversarial methods, but provides no wall-clock time or FLOPs breakdown for the OT computation, which must solve the transport problem over 64 rollouts × up to 128 agents × T timesteps per training example. Without this data, the efficiency claim is unsubstantiated and reproducibility is hindered.

### Minor

- **Single dataset and single architecture.** All experiments use MotionLM on the Waymo Open Motion Dataset. It is unclear whether the gains from DPA-OMF generalize to other architectures or datasets. The pre-training and evaluation data coincide (Waymo), and alternative datasets (e.g., nuScenes) or architectures are not explored. This limits confidence in the method's generality.

- **Notation inconsistencies in the key equations.** In Eq. (2), the optimal coupling is written as $\mu_i^{i,*}$, where both the subscript and superscript refer to agent $i$, creating ambiguity. In Eq. (3), the cost function appears as $\bar{c}(\phi(o_{i,t}^i), \phi(o_{i,t'}^i))$, again using $i$ redundantly as both subscript and superscript. These should be disambiguated (e.g., by using a separate rollout index).

- **"Scaling laws" framing overclaims.** The paper uses the term "preference data scaling laws" (abstract, Section 5.4) for an analysis that shows a monotone performance curve as a function of data size. A scaling law in the formal sense requires fitting a power-law relationship with characterized exponents. The finding is useful and interesting, but should be framed as a "scaling study" or "scaling analysis" to avoid misleading readers.

- **Ablation over $K$ (rollout count) and margin selection not provided.** The paper fixes $K=64$ rollouts, selecting top-16 as preferred and bottom-16 as unpreferred (16 pairwise comparisons). No ablation over this margin or count is provided. Given that Table 2 shows significant sensitivity to feature choice, sensitivity to the selection margin is equally plausible and should be checked.

- **Temporal ordering discarded in occupancy measure.** The empirical occupancy measure is defined as a uniform mixture $\frac{1}{T}\sum_t \delta_{\phi(o_t)}$, which explicitly discards temporal structure. For traffic simulation, the order of events matters (e.g., yielding *before* a conflict versus after). The paper does not acknowledge this limitation.

### Tiny

- **"Comparable" in the abstract slightly overstates the gap.** DPA-OMF achieves 0.739 vs. SMART's 0.761 — a ~3% gap — which is noteworthy given the 102× parameter difference but is more accurately described as "approaching" or "competitive with" rather than "comparable to."
- **Section heading inconsistency:** Section 5.5 is titled "Preference Over Exploitation" but discusses "preference over-optimization" throughout — minor but worth fixing.

---

## Nice-to-Haves

- **Learned feature encoder.** The OT distance relies on hand-crafted features. An ablation or discussion comparing hand-crafted versus learned embeddings would strengthen the method's generality claims and address reviewer concerns about feature engineering sensitivity.
- **Explicit safety metric reporting.** Reporting collision rate and/or min time-to-collision as standalone metrics (alongside WOSAC) would give a clearer picture of whether realism gains correspond to genuine safety-relevant behavioral improvement.
- **Computational cost breakdown.** Even a rough comparison of OT curation time vs. adversarial AFD discriminator training would substantiate the efficiency claim.
- **Release of the constructed preference dataset** or curation code, given that the data pipeline is itself a core contribution.
- **Visualization of OT coupling for failure cases**, as already shown in Figure 2 for successes, to expose limitations of the distance function (e.g., cases where low OT cost does not correspond to semantic alignment).

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **Human evaluation requirement (Reviewer 3):** For WOSAC, the WOSAC realism metric is the accepted community standard. Demanding a human preference study for a systems paper benchmarked on a standard leaderboard is not consistent with the norms of this community and would apply equally to every prior work on the benchmark.
- **OOD generalization evaluation (Reviewer 3):** The paper's stated scope is improving pre-training-data-consistent traffic simulation. OOD generalization to unseen scene configurations is a reasonable future direction but is outside the paper's claimed contributions.
- **Unfair baseline comparison criticism (implicit in Reviewer 1):** The comparisons with BehaviorGPT, Trajeglish, and SMART differ in pre-training data scale, input features, and sampling strategy, but these differences all favor the baselines (larger models, more parameters). The asymmetry therefore makes a stronger point, not a weaker one, and the comparison is valid.
- **Classification accuracy "tautological" (Reviewer 1):** The critic argues that DPA-OMF's 0.84 classification accuracy is circular because the objective incentivizes it. However, the comparison to AFD's 0.52 is meaningful and shows that AFD fundamentally fails at this task. Framing this as tautological ignores that AFD is trained with a similar objective and still fails.
- **Missing related works criticism:** Not assessed — external references cannot be verified.
- **Request for theoretical proofs for over-optimization threshold (Reviewer 3):** This is an empirical systems paper. Requiring formal theoretical grounding for the over-optimization behavior is not standard for this venue and type of work.

---

## Novel Insights

The most genuinely novel insight synthesized from the reviews — beyond the paper's own stated contributions — is the **mechanistic explanation of why adversarial AFD fails in the on-policy setting** (Figure 6): because expert demonstrations are the reference model's pre-training targets, their log-likelihood is already much higher than that of any model generation. When adversarial AFD is applied, the alignment signal degenerates to further increasing expert likelihood while leaving the pair-wise discrimination between generated samples nearly unchanged. This points to a broader principle — that *heterogeneity between the positive and negative data sources in preference alignment can cause the model to learn distributional identity rather than behavioral quality* — which may have implications for AFD methods in other embodied domains beyond traffic simulation. The paper hints at this in Section 2 but does not fully articulate this as a general failure mode.

---

## Suggestions

1. **Address the feature-sharing circularity directly:** Report whether improvement in WOSAC sub-metrics that are *least* represented in the OT feature set (e.g., Kinematic, which captures speed profiles vs. Map compliance, which reflects road boundary adherence) follows a different pattern. This would help disaggregate metric gaming from genuine alignment.
2. **Add variance estimates for Table 1–3 results:** Even reporting results over 2–3 runs for the key internal comparisons would substantially increase confidence in the gains over SFT-bestOA and adversarial AFD.
3. **Quantify OT curation time:** Report the wall-clock time for constructing the preference dataset relative to the pre-training or SFT-bestOA baseline. This is necessary to substantiate the efficiency claim.
4. **Revise "scaling laws" to "scaling study"** throughout to match the actual analysis.
5. **Acknowledge the temporal-order-discarding property** of the occupancy measure formulation as a limitation, and discuss whether it matters empirically (e.g., are there scene types where temporal ordering is critical for preference ranking?).
6. **Fix notation in Eq. (2) and (3):** Use distinct indices for agent identity and rollout identity to eliminate ambiguity.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
