## Summary

SPARC proposes a rehearsal-free continual learning method that maps biological working-memory/semantic-memory distinctions onto depthwise-separable convolutions: task-specific depthwise filters act as working memories, while half of the pointwise filters are shared across tasks as semantic memory and updated via exponential moving average. The paper reports strong Class-IL accuracy on Seq-CIFAR100 and Seq-TinyImageNet with high parameter efficiency and linear parameter growth, and it includes ablations validating the split-memory design.

## Strengths

- **Elegant architectural insight.** The mapping of DSC’s natural depthwise/pointwise split onto task-specific working memory and task-agnostic semantic memory is clean and biologically motivated (Figure 1, Sections 3.1–3.2).
- **Strong empirical results on complex benchmarks.** Table 1 shows SPARC achieves the highest reported Class-IL accuracy on Seq-CIFAR100 (49.03%) and Seq-TinyImageNet (32.29%) among the compared methods, and Table 4 demonstrates far more modest parameter growth than prior parameter-isolation approaches (3.62M vs. 2645M for PNNs at 20 tasks).
- **Evidence for the core split-memory mechanism.** Table 5 shows that SPARC’s semantic consolidation (49.13%) closely approaches fully separate filters (51.57%) at roughly half the parameter cost, and Figure 4 (right) links the EMA rate α directly to stability, validating the design in Eqns. 3–4.

## Weaknesses

### Fatal

None.

### Major

- **Weight re-normalization is claimed as a main contribution but has no supporting ablation or quantitative evidence in the main text.** The paper lists “task-specific biases” as a key challenge and weight re-normalization (Eqn. 5) as a technical remedy (Introduction, third bullet; Section 3.3). However, no table, figure, or sensitivity analysis isolates its effect. Because this is explicitly sold as a core contribution, its complete lack of empirical validation in the submitted paper is a significant evidential gap.

### Minor

- **Anomalously high variance on the easiest benchmark.** On Seq-CIFAR10 Class-IL, SPARC reports a standard deviation of ±4.81 (Table 1), whereas Seq-CIFAR100 and Seq-TinyImageNet show ±0.05 and ±0.01. This inverted variance pattern—highest instability on the simplest dataset—suggests hyperparameter sensitivity or training instability that the paper does not explain.
- **Main-text efficiency comparisons are confounded by backbone differences.** The headline comparisons in Tables 1 and 4 juxtapose SPARC’s DSC backbone (with halved filter counts: 32/64/128/256) against ResNet-18 baselines (64/128/256/512). While DSC is an explicit design choice and the paper references backbone-controlled comparisons in Appendix D.2, the main text does not disentangle backbone effects from the proposed memory architecture when presenting the “6% parameters” narrative.
- **Class-IL inference cost is not characterized.** Section 3.4 states that Class-IL inference requires forwarding each image through all task-specific sub-networks, implying linear growth in forward passes with task count. Despite the paper’s emphasis on computational efficiency (Table 1 reports training F/B counts), this deployment cost is never quantified.
- **Selective baseline inclusion in Figure 3.** The right panel (Task-IL on Seq-TinyImageNet) omits PNNs, which Table 1 shows outperform SPARC (67.84% vs. 65.66%). This omits a relevant baseline and slightly misrepresents relative standing.
- **Figure 2 highlights Task-IL rather than Class-IL for the longest sequence.** The 20-task scalability result in Figure 2 is shown for Task-IL, whereas Class-IL is the harder setting central to the paper’s claims. The paper notes that longer-sequence Class-IL results are in Appendix E.3, but the main text omits them.

### Trivial

- The abstract claims SPARC “matches rehearsal-based methods on various CL benchmarks.” On Seq-CIFAR10 Class-IL (Table 1), SPARC trails DER++, ER-ACE, Co²L, CLS-ER, and OCDNet, so the claim is slightly overstated, though the paper itself later acknowledges that SPARC lags on this simpler benchmark.

## Nice-to-Haves

- Report per-image Class-IL inference FLOPs/latency as a function of task count to complete the efficiency narrative.
- Include activation-distribution visualizations before/after weight re-normalization to substantiate the bias-mitigation claim.
- Report 20-task Class-IL accuracies in the main text, since scalability in the harder setting is a central claim.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **“No controlled backbone experiments.”** The paper explicitly states that “performance of competing approaches with SPARC-like backbone” is provided in Appendix D.2 (line 269). Criticizing its absence is factually wrong.
- **“Missing 20-task Class-IL results.”** The paper references Appendix E.3 for “performance under longer task sequences.” Per hard rules, criticisms about appendix-deferred content are removed.
- **“Table 2 shows default architecture is under-capacity.”** Observing that wider/deeper models perform better is a standard capacity ablation, not evidence of a flaw.
- **“Notation in Eqn. 4 is ambiguous.”** While the update rule could be phrased more precisely, the intent (EMA of shared semantic memory) is recoverable from context.

## Novel Insights

The paper’s core insight—exploiting the natural structural split in depthwise-separable convolutions to implement a biologically inspired working-memory/semantic-memory architecture—is genuinely elegant and well-motivated. Table 5 demonstrates that this split meaningfully approximates hard parameter isolation at much lower cost, suggesting the idea has practical merit beyond the specific benchmarks tested.

## Suggestions

- Add a weight re-normalization ablation (with/without Eqn. 5, and sensitivity to κ) to the main text, or at minimum reference Appendix evidence prominently if it exists.
- Discuss the high Seq-CIFAR10 variance explicitly—whether it reflects hyperparameter sensitivity, random seed instability, or dataset-specific behavior.
- In Table 1 or Figure 3, add a footnote clarifying that baseline architectures use standard ResNet-18 while SPARC uses a DSC-based variant, so readers do not conflate backbone and mechanism.

## Score and Decision

**Calibration comparison:**
- **High anchors:** *SD-LoRA* (5U1rlpX68A, 7.50) and *OVOR* (FbuyDzZTPt, 6.00) both provide thorough ablations and theoretical/empirical justification for their mechanisms. SPARC falls short of this bar because a claimed contribution lacks any ablation.
- **Medium anchors:** *Dual-Arch* (YFdopzmpdr, 5.20) and *Auxiliary Classifiers* (1nHQRsb3Ze, 5.00) have extensive experiments but gaps in ablations or novelty substantiation. SPARC is slightly above these because its core split-memory mechanism is better validated (Table 5, Figure 4) and its benchmark coverage is stronger.
- **Low anchors:** *DIRAD* (ZHTYtXijEn, 2.33) and *MambaCL* (1TXDtnDIsV, 4.67) suffer from weak baselines, limited scale, or insufficient analysis. SPARC is clearly above this cluster.

Relative to these anchors, SPARC sits in the upper-medium range: it has real contributions and strong results, but the missing ablation for weight re-normalization—a claimed main contribution—is a material weakness that prevents it from reaching the 6+ accept threshold.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>