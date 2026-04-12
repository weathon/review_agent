=== CALIBRATION EXAMPLE 17 ===

# Final Consolidated Review
## Summary
This paper proposes SURE, a test-time adaptation method for vision-language models that builds a dynamic class-level Prototype-Reliability Graph (PRG) from text-prototype similarity and temporally estimated class reliability. Predictions are regularized via graph propagation, and the graph co-evolves with prototype updates over the test stream. Empirically, the method shows consistent gains over several recent VLM-TTA baselines on ImageNet distribution shifts and cross-dataset transfer, with a favorable runtime/accuracy tradeoff.

## Strengths
- **The paper contributes a concrete and distinctive structured TTA mechanism for VLMs, rather than another per-instance entropy or prompt-tuning variant.** The core design couples semantic affinity from text prototypes with temporal reliability statistics into a class-level graph, then uses that graph for logit regularization and subsequent graph evolution. This is a more specific and interesting design choice than generic confidence-thresholding or prototype updates.
- **The empirical gains are consistent across two CLIP backbones and two evaluation regimes.** In Table 1, SURE achieves the best average performance on natural shifts for both RN50 (51.12 average / 47.88 OOD average) and ViT-B (66.23 / 64.99). In Table 2, it also has the best average cross-dataset performance for both backbones (62.79 RN50, 70.04 ViT-B), outperforming strong recent baselines such as DPE, BCA, and ZERO on the reported averages.
- **The ablation results support that reliability-aware graph construction matters, not just prototype adaptation alone.** Table 4 shows a progression from CLIP → ProtoOnly → graph without reliability → graph with reliability → full logit propagation, with the reliability-aware graph and final propagation each providing additional gains, especially on OOD average.
- **The method appears practically lightweight relative to strong alternatives.** Table 3 reports 0.067s/sample for SURE, much faster than TPT and in the same rough regime as other efficient methods, while still improving average accuracy over BCA and matching/exceeding ZERO on the reported benchmark.
- **The appendix provides useful supporting evidence on stability and calibration.** Tables 7–8 show low run-to-run variance across three seeds, and Table 10 indicates that SURE substantially improves over the naive prototype-only variant in ECE while retaining stronger accuracy, which is relevant given the paper’s emphasis on suppressing error amplification.

## Weaknesses

### Fatal
- None.

### Major:
- **The claimed efficiency/scaling story is not fully supported by the method description.** The paper states in the efficiency discussion that “graph updates scale linearly with class count \(C\),” but Algorithm 1 explicitly recomputes the full prototype similarity matrix each test step (“Compute \(S_{ij} = \cos(t_i,t_j)\)”, then form \(W\), then top-\(k\) sparsify). Since \(S \in \mathbb{R}^{C\times C}\), this is at least quadratic in the number of classes if done as written. This matters because the deployment-friendliness claim is one of the paper’s selling points, and the current text leaves a mismatch between the algorithm and the stated complexity/runtime intuition.
- **The initialization of prototype counts is unusually strong and insufficiently justified in the main paper.** Section 5.1 states: “Following (Zhou et al., 2025), each class prototype is initialized with \(N_i^{proto}=30000\) confident samples and updated via normalized moving averages.” This does not mean the model is pre-adapted on 30,000 target samples per class—the paper does not say that—but it does impose a very strong inertia/prior on prototype updates. Because this choice can materially stabilize adaptation and reduce early drift, the paper should explain more clearly what this initialization means operationally, why this large value is appropriate, and how sensitive results are to it. As written, readers are left unsure how much of the robustness comes from the graph mechanism versus simply making prototype updates extremely conservative.
- **Order dependence and streaming robustness are under-analyzed for a method built around temporal statistics.** Reliability is estimated from sliding windows \(Q_i\), graph smoothing uses a buffer of recent adjacency matrices, and the whole method is explicitly framed as a streaming closed-loop system. Yet the paper does not clearly specify the test-stream ordering used in the main experiments, nor does it evaluate different permutations or non-stationary stream patterns. This is important because temporal TTA methods can behave quite differently under shuffled, class-clustered, or abrupt-shift streams, and the current evidence is mostly on static benchmark splits rather than genuinely sequential shift scenarios.
- **The paper does not compare directly against the graph-based baseline it highlights as closest in spirit.** The related work explicitly discusses PROGRAM as a graph-based adaptation method and argues SURE differs by reliability-driven topology and VLM-specific design, but no direct empirical comparison is provided. Since one of the main claims is that reliability-weighted graph construction is the key improvement over more generic graph propagation, this omission weakens the empirical case for the paper’s specific novelty.

### Minor
- **The reliability metric remains heuristic and only lightly justified.** The score \(R_j=\mu_j(1-\sigma_j/\sigma_{max})\) with \(\sigma_{max}=0.5\) is motivated intuitively, but not very deeply. The paper clips \(R_j\) to \([0,1]\) “for numerical stability,” which further underscores that this is a pragmatic gating function rather than a principled uncertainty estimator. A stronger sensitivity analysis or explanation for \(\sigma_{max}\) would improve confidence in the method.
- **Early-step dynamics are not discussed clearly enough.** Reliability is initialized to \(\mu_j=1,\sigma_j=0\) for all classes, i.e., maximal reliability before any evidence has been observed. This is a reasonable bootstrap choice, but it means the graph initially behaves as if all classes are fully trustworthy. Given the method’s emphasis on suppressing unreliable propagation, a brief analysis of the first few adaptation steps would be useful.
- **Some of the paper’s broader claims are stronger than the evidence shown.** The introduction and conclusion use language such as “principled” and “generalizable,” but the experiments are all on CLIP backbones and static benchmark suites. The evidence supports a promising CLIP-based VLM TTA method; it does not yet fully establish broader architecture-agnostic generality.
- **The visualization evidence is illustrative rather than strongly diagnostic.** Figure 4’s 5-class “micro-universe” is helpful for intuition, but it is not very informative about behavior in dense 200- or 1000-class settings where graph crowding, false semantic neighbors, and competition for top-\(k\) edges are more consequential.

### Trivial
- **A few implementation and presentation details would benefit from clarification.** In particular, the paper should be clearer about whether similarities are fully recomputed every step in practice, whether timings include all graph-maintenance overhead, and what exact stream ordering is used in the benchmark protocol.

## Nice-to-Haves
- Add experiments with multiple test-stream orderings (random, class-clustered, alternating domain shifts) to validate the temporal reliability mechanism.
- Report sensitivity to the prototype-count initialization \(N_i^{proto}\), since this hyperparameter likely interacts strongly with adaptation stability.
- Include a direct comparison to PROGRAM or another graph-structured adaptation baseline to isolate the value of reliability-aware topology.
- Show a static-vs-dynamic graph ablation (freeze adjacency after initialization versus continuous updates) to demonstrate whether graph evolution is necessary.
- Provide temporal plots of class reliability \(R_j\), prototype drift, and calibration over the stream, which would better substantiate the claimed closed-loop behavior.
- Evaluate at least one additional non-CLIP VLM backbone if the goal is to claim method-level generality beyond CLIP.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Massive offline prototype initialization violates online TTA because it requires 30,000 confident target samples per class.”** This criticism overreaches and misreads the paper. The text says “each class prototype is initialized with \(N_i^{proto}=30000\) confident samples,” but the update equation uses this as a running-count prior/inertia term, not as a claim that 30,000 target samples per class are collected before adaptation. The real issue is lack of justification/sensitivity for this large prior, which is retained above.
- **“The method fundamentally cannot suppress errors because the reliability update creates a mathematically guaranteed confirmation-bias loop.”** The paper does indeed use adapted prediction confidence to update reliability buffers, so there is a self-reinforcing pathway. However, calling this a fatal contradiction is too strong: the method also gates updates by confidence thresholding, multiplies reliabilities across class pairs, uses sparse top-\(k\) neighbors, smooths adjacency over time, and empirically improves over a prototype-only baseline. This is better framed as a potential mechanism-level concern that would benefit from deeper analysis, not as a proven invalidation of the method.
- **Claims about missing related work or omitted baselines such as SAR/MEMO/other recent methods.** Per instruction, unsupported missing-related-work complaints are removed. The retained concern is narrower and paper-grounded: the paper explicitly discusses PROGRAM as a close conceptual comparison but does not evaluate against it.
- **“ProtoOnly lacks confidence thresholds.”** This is factually wrong. Section 4.3 states that prototype updates are performed only for “a high-confidence pseudo-label … (confidence \(>\theta\)),” and the ablation builds on this adaptation framework.
- **Pure reproducibility complaints about unspecified minor details.** The paper actually provides many implementation details (e.g., \(L=5\), \(k=3\log C\), \(\theta=0.3\), prompt ensemble, seed-variance tables). The remaining substantive concern is specifically about stream ordering because it directly affects a temporal TTA method’s validity.

## Novel Insights
The most important synthesis is that this paper’s true contribution is not merely “graph regularization,” but **using temporal class reliability to decide which semantic relations deserve to remain active during adaptation**. That is a meaningful idea and is empirically supported to some extent by the ablations. At the same time, the paper somewhat undersells a subtle tension: several of its strongest robustness gains may derive from *conservatism*—large prototype-count initialization, thresholded updates, adjacency smoothing, and single-step propagation—rather than from aggressive semantic reasoning. Framed positively, this suggests SURE works because it introduces structured caution into TTA, not just richer structure.

## Suggestions
- Clarify exactly how \(N_i^{proto}=30000\) is used, and add an ablation over this value to disentangle graph benefits from conservative prototype inertia.
- Reconcile the algorithm with the efficiency claim: either provide the actual per-step complexity including similarity recomputation, or explain the optimized implementation used in timing.
- Explicitly specify test-stream ordering and add permutation/stream-shift experiments to validate the temporal reliability mechanism.
- Compare against PROGRAM or another graph-based adaptation baseline to directly support the paper’s claimed novelty.
- Expand the analysis of the reliability score, especially the role of \(\sigma_{max}\), clipping, and early adaptation behavior before buffers are populated.
- Tone down claims of broad generality unless additional non-CLIP evidence is added.

# Actual Human Scores
Individual reviewer scores: [4.0, 4.0, 6.0, 4.0, 4.0]
Average score: 4.4
Binary outcome: Reject
