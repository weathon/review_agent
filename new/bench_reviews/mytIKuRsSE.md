Now I have enough information to write the final consolidated review.

---

## Summary

The paper introduces **Dual-level Noisy Correspondence (DNC)** — a practically grounded problem in multi-modal entity alignment (MMEA) where both intra-entity (entity-attribute) and inter-graph (entity-entity, attribute-attribute) correspondences are noisy in real-world knowledge graphs. To address DNC, the authors propose **RULE**, which estimates reliability via a two-fold principle (Dempster-Shafer uncertainty + consensus), applies it to both a dually robust training loss (DRL) and a dually robust fusion module (DRF), and adds a test-time correspondence reasoning (TTR) module using a large MLLM with chain-of-thought. Experiments across five benchmarks with three noise levels consistently outperform seven baselines.

---

## Strengths

- **Novel and well-motivated problem formulation (DNC)**: The paper empirically shows that real MMEA benchmarks (ICEWS) already contain >50% noisy correspondences without any artificial injection (Fig. 1, Appendix B), and demonstrates quantitatively that this degrades both fusion and alignment performance (Fig. 1(b)). This grounds the paper's contributions in a genuine real-world gap rather than a synthetic toy problem.

- **Theoretically motivated reliability estimation (Theorem 1, Eq. 2–7)**: The paper proves that low Dirichlet uncertainty alone is insufficient to guarantee correct correspondence (Theorem 1), motivating the combination of uncertainty *and* consensus. Fig. 3(b) confirms the two-fold reliability cleanly separates clean/noisy pairs in a converged model, and Fig. 4 validates that the three-way partition (S_U, S_I, S_C) is empirically meaningful.

- **Strong and consistent experimental gains, especially under high noise (Tables 1–2)**: At 50% DNC in the non-name protocol, RULE achieves 64.3% avg. H@1 vs. 54.0% for the next-best baseline (MEAformer), a 10.3-point margin across five diverse datasets. The robustness curve (Fig. 3(a)) shows RULE degrades far more gracefully than all baselines as noise increases from 0–70%.

- **Comprehensive ablation isolating each component (Table 3)**: The ablation systematically removes DRL, DRF, TTR, and tests uncertainty-only and consensus-only variants — confirming all three modules and both reliability principles contribute positively. Critically, removing DRL causes the most severe drop (31.6% H@1), establishing it as the backbone contribution.

- **Self-adaptive thresholds (Eq. 8)**: The thresholds β_u and β_c are derived from true-positive training statistics rather than hand-tuned, which is a practical strength for generalization across different noise levels and datasets.

---

## Weaknesses

### Fatal
None.

### Major

- **Test-time TTR module uses Qwen2.5-VL-72B-Instruct, creating an asymmetric comparison vs. all baselines** — The paper explicitly states "for fair comparisons, we adopt the same backbone (i.e., CLIP) for all baselines" (Section 3.2), but this claim covers only the training-time encoder. The TTR module deployed at inference uses Qwen2.5-VL-72B-Instruct, a model orders of magnitude larger than any baseline encoder. The ablation (Table 3) shows removing TTR drops ICEWS-WIKI 50% DNC non-name H@1 from 58.2 to 56.5 — a meaningful 1.7 pp. For the all-attributes setting, the drop is 97.7 → 94.0 (3.7 pp). Neither the headline results nor the narrative separates RULE's training-time design from the inference-time MLLM advantage. Without testing whether adding the same 72B model to leading baselines (e.g., MEAformer, PMF) at inference yields comparable gains, any claim that the performance advantage is specifically attributable to the DNC-robustness design remains partially unvalidated. The ablation's "w/o TTR" variant does show RULE *without TTR* still exceeds all baselines (56.5% vs ~54.0% best baseline at 50% DNC), which demonstrates the training-time contributions are real — but the headline comparison remains structurally asymmetric and this should be directly acknowledged and addressed.

### Minor

- **Assumption 1 (consensus greedy strategy) is stated without empirical validation across training stages** — The consensus-based reliability estimation assumes that correctly associated attributes always produce ∆ ≥ 0 and irrelevant ones produce ∆ < 0 under the greedy marginal contribution criterion (Eq. 6–7). This drives the estimated ŷ_i used in DRL (Eq. 12). However, in early training epochs when representations are poor, this assumption may frequently be violated, feeding incorrect soft labels into the very loss meant to provide robustness. The paper does not analyze how often Assumption 1 holds as a function of training epoch or provide a curriculum analysis showing early-epoch violations are bounded. This circular dependency (reliability estimation ↔ representation quality) is well known in self-training literature and should be at least acknowledged as a limitation.

- **HHEA baseline performs anomalously on DBP15K datasets** — In Table 1 (Non-name, Inherent DNC), HHEA achieves ~48.7% avg. H@1 while all other baselines achieve 75–84%. No explanation for this large discrepancy is offered. If HHEA was misconfigured or run under different settings, this could artificially inflate RULE's average rank on these benchmarks. The authors should clarify the implementation details for HHEA or note any known configuration issues.

- **Ablation is conducted only on ICEWS-WIKI at 50% DNC** — ICEWS-WIKI is the dataset where RULE shows its largest absolute gains and the one with the highest claimed inherent noise. Ablating only on the most favorable dataset limits the generalizability of conclusions about individual components, particularly for DRF, whose contribution may differ on DBP15K-style benchmarks.

- **No statistical variance reported across any table** — No standard deviations or multi-run results appear. Given that 2–5 pp improvements on DBP15K at high noise levels constitute the core empirical claims, unreported training variance makes it difficult to assess the reliability of the reported gains.

### Trivial

- The β = 0.3 threshold is described as fixed across all experiments with sensitivity analysis deferred to an appendix; even a one-row sensitivity table in the main text (e.g., β ∈ {0.2, 0.3, 0.4}) would reassure readers that the clean/noisy division is stable.

---

## Nice-to-Haves

- **Computational cost characterization**: The paper is entirely silent on inference latency and GPU memory. A 72B-parameter MLLM changes the deployment profile by orders of magnitude vs. CLIP-scale baselines. Even rough wall-clock figures and the number of MLLM calls per test query would help practitioners assess feasibility.

- **Failure-case analysis**: Figure 5 shows successful attribute reliability weighting, but showing cases where the DRF conflates inter-graph and intra-entity noise (e.g., suppressing a correctly fused modality because of a noisy cross-graph correspondence) would sharpen the paper's honest scope.

- **Controlled experiment: add TTR to a representative baseline**: Adding Qwen2.5-VL-72B-Instruct at test time to MEAformer would directly isolate how much gain comes from the MLLM alone vs. the DNC-specific training design. This would strongly bolster the training-time contribution claims.

- **Early-epoch reliability curve**: Tracking Assumption 1's empirical validity (fraction of attributes correctly identified as clean/noisy) across training epochs would provide evidence that the curriculum effect is stable rather than coincidentally working in this experimental setting.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic Issue 2 — Two directly competing published methods omitted from baselines** (Chen et al. NeurIPS 2024; Wang et al. ICDE 2024): Removed per the hard rule that missing related works cannot be mentioned without external sources to confirm their existence. Even if cited in the reference list, the meta-reviewer cannot verify these papers exist or that they are directly comparable in setting.

- **Strength Finder "Fair comparison protocol: All baselines use the same CLIP backbone"**: Dropped as a standalone strength because it is contradicted by the verified major weakness (TTR uses a 72B model); listing it as a strength would mislead readers.

- **Harsh Critic — Section 3.3 reliability visualization is "post-hoc"**: Removed as a weakness. Confirming that the trained model separates clean/noisy pairs is exactly what a good visualization should demonstrate; demanding that this visualization represent something other than a converged model is a strawman.

- **Harsh Critic — "No Section 1.x or separate related work section"**: Removed as a pure presentation/structural nitpick with no bearing on scientific validity.

---

## Novel Insights

The two-fold reliability estimator (uncertainty + consensus, motivated by Theorem 1 showing each is individually insufficient) is a principled synthesis that could be applied broadly in self-supervised multi-modal learning beyond entity alignment. The "inherent DNC" framing — showing that benchmark noise exists without any artificial injection and quantifying it as >50% in ICEWS — is a methodologically important observation that motivates the entire research direction and could change how future MMEA systems are evaluated. The combination of training-time robustness and test-time MLLM reasoning is a structurally sound idea even if the current comparison does not yet fully disentangle their contributions.

---

## Evaluation on Key Axes

- **Originality**: Good. The DNC problem formulation is novel and unifies scattered observations about MMEA noise. The Dempster-Shafer + consensus combination is principled, though individual components draw from prior work.
- **Importance of research question**: High. Real MMEA benchmarks are demonstrably noisy; this is not a hypothetical scenario.
- **Whether claims are well supported**: Partially. The training-time claims are well-supported by ablation; the headline claims mixing TTR are overstated relative to what is controlled.
- **Soundness of experiments**: Moderate. Five datasets and seven baselines is commendable; single-run results and ablation scope limitations are genuine gaps.
- **Clarity of writing**: Good. Motivation and formulation are clear; the TTR fairness issue is obscured by the backbone-comparison claim.
- **Value to research community**: Moderate-to-high. Practical problem, publicly released code, substantial gains that partially survive the fairness concern.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg Score | Relevance to this paper |
|---|---|---|
| `/human_reviews/9Cu8MRmhq2.md` | 8.0 | Same problem area (noisy correspondence learning, multi-modal); cleaner comparisons, oral quality — this paper falls below it due to TTR fairness gap and missing variance |
| `/human_reviews/wfgZc3IMqo.md` | 6.0 | Similar: robust training under noisy labels, some comparison credibility concerns, accepted as poster — this paper's problem framing is stronger but has a bigger structural fairness issue |
| `/human_reviews/OZdr2mV5EI.md` | 4.25 | Closest analog to the TTR fairness concern (MLLM vs. small-model baselines, unfair comparison); but OZdr2mV5EI's MLLM was the *whole* system, not a separable test-time module, and it had data leakage — this paper is moderately above it |
| `/human_reviews/fACNPcPcrs.md` | 5.5 | Unfair comparison (175B LLM vs. small encoders) — the RULE paper's TTR fairness issue is similarly serious but the ablation mitigates it |
| `/human_reviews/a4O528mek9.md` | 3.0 | Rejected multi-modal paper with fundamental clarity and methodology issues — this paper is clearly above it |

**Positioning:** The paper sits between wfgZc3IMqo (6.0, accepted) and OZdr2mV5EI (4.25, rejected). The training-time contributions (DRL+DRF) demonstrably work and beat all baselines even without TTR — this distinguishes it from OZdr2mV5EI where the MLLM was the core system. However, the headline results include TTR and the paper's "fair comparison" claim is factually misleading. The absent variance reporting and single-dataset ablation are compounding but not fatal. Relative to 9Cu8MRmhq2 (oral, 8.0), this paper lacks the clean methodology and presentation rigor needed for a top-tier placement.

**Score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>