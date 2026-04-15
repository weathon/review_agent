Now I have sufficient information to write the consolidated review. Let me produce it.

---

## Summary

This paper proposes using manifold capacity and associated geometric measures—radius, dimension, alignment—as a representation-based framework to quantify and characterize feature learning in neural networks. The core thesis is that task-relevant manifolds untangle during rich learning, and that manifold capacity (borrowed from Chung et al., 2018 and Chou et al., 2024) tracks the degree of this richness. The authors extend Ba et al. (2022) from regression to classification in a one-step theoretical result, demonstrate capacity tracks the scale-factor interpolation between lazy and rich regimes empirically, and apply the framework to RNN neuroscience tasks and an OOD transfer setting.

---

## Strengths

- **Representation-based diagnostic for lazy/rich regimes, with direct neuroscience relevance.** Weight- and NTK-based measures are inaccessible in experimental neuroscience where only neural activity can be observed. The manifold capacity framework operates entirely on representations, which is a genuine gap the paper correctly identifies and fills. Section 5.1 provides a concrete demonstration of this utility—recovering the same finding as Liu et al. (2024) from representations alone.

- **Non-trivial theoretical extension from regression to classification.** Theorem 1 extends Ba et al. (2022) to a classification setting by analyzing the margin of the Gaussian-equivalent model after one gradient step (using tools from Montanari et al., 2019). The authors acknowledge this is a specific asymptotic, one-step result, and the proof requires technical additions beyond prior work. The monotonicity of capacity in η and the invertible capacity–accuracy link are clean and useful within the stated scope.

- **The geometric decomposition yields non-trivially distinct findings for equal-capacity networks.** Section 5.1 shows that RNNs with different initial weight ranks reach approximately the same manifold capacity at final epoch but display qualitatively different geometric organizations (radius vs. dimension trade-off). This is the paper's sharpest and most concrete finding: the scalar capacity does not fully characterize the learned representation, and geometry adds genuine information.

- **Empirical OOD finding adds new geometric framing.** The observation that ultra-rich training degrades CIFAR-100 linear-probe accuracy, coinciding with manifold radius expansion and center-axis alignment increase, provides a geometric lens on a practically important phenomenon. The noted architectural difference (radius vs. dimension expansion for VGG-11 vs. ResNet-18) is a specific novel observation.

---

## Weaknesses

### Fatal
*None that unambiguously invalidate the core claim.*

### Major

- **The "OOD generalization" framing in Section 5.2 is misleading and non-standard.** The paper defines "OOD generalization" as the case where the label set differs between train and test, implements it as a linear probe trained on CIFAR-100 representations learned from CIFAR-10, and frames the results as explaining OOD generalization failure. In the machine learning community, "OOD generalization" refers to the generalization of the *trained* model under distributional shift—not transfer of representations to an entirely new labeled dataset via a fresh linear classifier. What is being measured is **representation transferability to a different task**, which is a legitimate and interesting quantity but is not the same phenomenon. The paper labels CIFAR-100 as "OOD" in contrast to CIFAR-10C as "domain adaptation," but the CIFAR-100 experiment tests a different *label set*, not a distribution shift over the same label set. This framing mismatch propagates through the abstract and introduction ("providing geometric insights into out-of-distribution generalization") and overstates the scope of the contribution.

- **Theoretical scope is narrow, and the theory–empirical gap is unaddressed.** Theorem 1 covers one gradient step, a 2-layer network, asymptotic proportional limits, Gaussian data, and a fixed readout. Footnote 6 acknowledges that the Gaussian equivalence likely does not hold for multiple steps. Yet the empirical claims throughout are about multi-epoch training in deep networks and RNNs. The paper offers no theoretical justification—even partial or heuristic—for why capacity should track richness in these more complex settings. The gap is acknowledged but not bridged.

- **"Learning stages" in Section 4.2 are identified visually from a single run without principled criteria or robustness checks.** The four stages (clustering, structuring, separating, stabilizing) are read off from a normalized heatmap of geometric measure dynamics for a single VGG-11/CIFAR-10 run. There is no formal stage-boundary criterion, no analysis across random seeds, no demonstration that the same structure appears in other architectures or datasets, and no assessment of whether the stage boundaries predict anything. The paper presents these as "previously unreported subtypes" but the evidence is exploratory visualization, not a discovered phenomenon. The claim that manifold geometry "reveals" these stages implies more than is demonstrated.

- **Methodological novelty is limited: the core framework is borrowed.** The manifold capacity measure, the effective geometric measures (radius, dimension, center/axis alignment), and their analytical connections are all drawn from Chung et al. (2018) and Chou et al. (2024). The paper's contribution is in applying these existing tools to the lazy-rich learning setting. This is not inherently disqualifying, but the paper should be clearer that the novelty is primarily observational, not methodological.

### Minor

- **The "better than conventional measures" claim in Section 3.2 rests on a selective comparison.** Weight change is designed to measure parameter-space movement and is structurally blind to initialization quality—making it a poor reference for the wealthy/poor distinction by construction. The comparison should include stronger representation-based baselines beyond rep-label alignment to credibly support the claim of superiority.

- **"Structural inductive biases" is too causal a label for the RNN finding.** Section 5.1 shows an association between initial weight rank and final geometric organization, but "structural inductive bias" implies a clean causal account disentangled from optimization differences. The evidence supports "an observed geometric difference correlated with initial connectivity rank" but not a causal identification.

- **Scalability and computational cost of manifold capacity are unaddressed.** Computing mean-field capacity involves solving quadratic programs and estimating expectations over random projections and dichotomies. No wall-clock times or scaling analysis are given. Practitioners cannot assess feasibility for larger models or datasets.

### Trivial

- **The scale factor is called "ground truth for the degree of feature learning."** It is a regime-control parameter that interpolates between lazy and rich behavior in the specific Chizat-style setup. This is a useful proxy but calling it "ground truth" is overconfident outside that setting.

---

## Nice-to-Haves

- A **random label control** (training with shuffled labels and computing capacity dynamics) would provide a basic sanity check that capacity increases reflect task-relevant structure and not arbitrary representation change.
- **Formal stage-detection criterion** (e.g., change-point detection on the geometric measure trajectories) with multi-seed validation would elevate Section 4.2 from visualization to finding.
- **Quantitative comparison** of capacity vs. baseline measures (e.g., rank correlation with the ground-truth scale factor) would make the superiority claim in Section 3.2 precise and falsifiable.
- **Even a preliminary geometric intervention** (e.g., regularizing manifold radius during training to improve CIFAR-100 transfer) would close the loop on the OOD finding and demonstrate practical value.
- **Application to a modern architecture** (transformer or larger ResNet on ImageNet) would demonstrate the scalability and generality of the framework beyond VGG-11 and small RNNs.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic's claim that the OOD experiment is "structurally misframed" to the point that the "application claim does not stand"**: Partially retained but weakened. The paper does explicitly define its OOD protocol (linear probe on CIFAR-100 with a different label set), so it is not deceptive—but the terminology mismatch with standard ML usage is a real problem and is kept as a major weakness rather than a fatal one.

- **Request for confidence intervals across seeds on all figures**: Moved to nice-to-have per soft rules; single-run evaluation is common for geometric/representational analyses in this field.

- **Criticisms that manifold capacity and geometric measures are "not novel"**: The paper explicitly and correctly cites Chung et al. (2018) and Chou et al. (2024) as the source of the framework. This is not a hidden borrowing; it is proper attribution. The criticism of "limited novelty" is kept but weakened—it does not disqualify the contribution, it scopes it.

- **Request for missing related works**: Removed per hard rules.

- **Spark reviewer's demand for a CIFAR-100 vs. DomainNet OOD comparison ("CIFAR-100 is a poor OOD test")**: Removed—the paper's OOD design is for linear-probe transfer to a new label set, which is internally consistent. Requesting a specific alternative benchmark is out of scope for the stated design.

---

## Novel Insights

The paper's most genuinely novel observation—substantiated by data—is that networks with different initial weight ranks can converge to approximately the same scalar manifold capacity but with qualitatively distinct geometric organizations (radius-heavy vs. dimension-heavy). This demonstrates that capacity alone is insufficient to characterize learned representations and motivates the geometric decomposition as a necessary complement. This echoes the paper's own thesis but is illustrated more sharply in the RNN section than anywhere else. The geometric lens on ultra-rich OOD failure (radius expansion as the dominant geometric correlate for VGG-11, dimension expansion for ResNet-18) is also a specific, architectural-level observation that goes beyond prior work. The learning-stages claim in Section 4.2 could, if robustified, constitute a genuinely novel contribution but currently remains an illustrative visualization.

---

## Suggestions

1. **Reframe Section 5.2 as "cross-dataset representation transferability" rather than "out-of-distribution generalization."** Define the experiment clearly as linear-probe transfer to a new label set, not OOD generalization of the trained classifier. This is a real finding and does not need inflated framing.
2. **Add a robustness analysis for the learning stages** (Figure 4c): run at least 3 seeds, show the same stage structure appears, and define a quantitative criterion for stage boundary detection.
3. **Narrow the abstract and introduction claims** to match the scope of the theorem: the theoretical result justifies capacity as a richness measure in the one-step, 2-layer asymptotic setting; the empirical results extend this plausibly but non-rigorously to deeper networks.
4. **Add a quantitative comparison** in Section 3.2 (e.g., rank correlation between each measure and the scale factor ground truth) to replace the visual "better than" argument.

---

## Evaluation

| Axis | Assessment |
|---|---|
| **Novelty** | Moderate. The framework (manifold capacity + geometric measures) is borrowed; the contribution is in application. The specific findings in Sec. 5.1 and 5.2 are novel observations, but individually modest. |
| **Technical soundness** | Moderate. Theorem 1 is technically sound for its narrow scope. Empirical analyses are largely qualitative; some overclaim in framing. |
| **Empirical support** | Weak-to-moderate. Most key claims are supported by single-run, single-architecture experiments. Learning stages lack any robustness analysis. OOD finding is correlational. |
| **Significance** | Moderate. The representation-based toolkit has real value for neuroscience. The ML contributions are incremental given existing diagnostic tools. |
| **Clarity** | Good for the method; the framing of OOD and the scope of the theoretical result are misleading in places. |

---

## Score and Decision

**Calibration rationale:**
- `slSmYGc8ee.md` (8,6,8,5 → Accept, poster): Studies RNN connectivity and rich/lazy learning with theory and solid empirical backing—more focused and deeper than Section 5.1 of this paper. Score: ~7.
- `iEfdvDTcZg.md` (6,6,5,8 → Accept, poster): Empirical investigation of the feature-learning strength hyperparameter, similar in spirit to this paper's scale-factor analysis. Accepted despite limited theory. Score: ~6.25.
- `MY8SBpUece.md` (5,5,6,6 → Reject): One-step theory for feature learning, limited empirical scope. Rejected at average ~5.5.
- `oKglS1cFdb.md` (6,6,5 → Reject): OOD paper with geometric insights but no actionable interventions. Rejected at average ~5.7.

The paper under review is **broader in scope than `MY8SBpUece`** but the individual contributions are shallower than the accepted papers. It is comparable in quality to `iEfdvDTcZg` and `oKglS1cFdb`—the former was accepted, the latter rejected—but the misleading OOD framing and lack of robustness for the learning-stages claim are real weaknesses that the accepted comparable papers did not have. The borrowed methodology (no methodological innovation) and limited experimental scale (VGG-11, ResNet-18, CIFAR-10/100, 300-unit RNNs) align it with the borderline cluster. The specific finding in Section 5.1 (equal capacity, different geometry) is genuinely insightful and is the paper's best contribution.

Overall: this is a paper with a coherent research question and real empirical observations, but with overclaiming in key sections (OOD, learning stages) and a significant theory–empirical gap. It sits at the low end of the marginally-acceptable range.

**Score: 5.0**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>