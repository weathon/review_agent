## Summary

This paper introduces and studies the problem of Dual-level Noisy Correspondence (DNC) in multi-modal entity alignment (MMEA), where both intra-entity (entity-attribute) and inter-graph (entity-entity, attribute-attribute) correspondences can be corrupted. The authors propose RULE, a robust framework that estimates correspondence reliability via a two-fold principle (uncertainty and consensus), mitigates noise impact during attribute fusion and graph alignment, and incorporates a novel test-time reasoning module using multi-modal large language models (MLLMs) to uncover latent attribute connections. Extensive experiments on five benchmarks under varying noise levels demonstrate significant improvements over seven existing MMEA methods.

## Strengths

- **Novel and well-motivated problem formulation:** The paper clearly identifies and formalizes DNC, a practical issue in real-world MMKGs supported by statistics (e.g., >50% inherent noise in ICEWS benchmarks) and illustrative examples, addressing a critical gap in the field.
- **Comprehensive and cohesive method design:** RULE integrates a principled reliability estimation (evidential learning for uncertainty and a greedy consensus strategy), robust training modules (dually robust fusion and discrepancy elimination with tailored loss strategies), and an innovative test-time reasoning (TTR) module using MLLMs, offering a holistic solution.
- **Extensive and rigorous empirical validation:** Experiments across five benchmarks under two evaluation protocols, three noise types, and noise levels up to 70% show consistent and substantial gains over seven strong baselines. Thorough ablation studies, parameter analyses, and visualizations validate each component’s contribution and the method’s robustness.

## Weaknesses

### Major:
- **Heuristic components in consensus estimation lack theoretical grounding and thorough analysis:** The greedy strategy for estimating correct correspondences (Eq. 7) relies on Assumption 1 (marginal contribution sign indicates correct association), which is presented without theoretical justification. The paper does not analyze how often this assumption holds or the sensitivity of the method to its failures, making the consensus principle somewhat arbitrary.
- **Computational and practical concerns for the test-time MLLM module:** Although ablation studies show performance gains with smaller MLLMs (3B, 7B), the default use of a 72B parameter model incurs significant inference time (∼10k seconds on ICEWS-WIKI) and resource overhead, limiting real‑world deployment. The paper does not propose efficient alternatives (e.g., dynamic triggering, distillation) or thoroughly discuss the trade‑offs between accuracy and cost.
- **Insufficient quantitative evaluation of reliability estimation and error analysis:** While reliability scores are visualized (Figs. 3‑5), there is no quantitative measure (e.g., AUC, precision/recall) of how accurately the method distinguishes noisy from clean correspondences. Similarly, the test‑time reasoning module lacks a systematic analysis of failure cases (beyond anecdotal examples in Appendix I), leaving its limitations unclear.

### Minor:
- **Limited comparison with general noise‑robust techniques:** The paper compares only with MMEA‑specific methods. Including state‑of‑the‑art noisy‑label or noisy‑correspondence methods from related fields (adapted to MMEA) would better contextualize the robustness claim, though this may be considered beyond the immediate scope.

### Trivial:
- None.

## Nice-to-Haves
- Extend noise‑injection experiments to even higher levels (e.g., 90%) to test the method’s breaking point, though 70% is already explored.
- Include a more detailed case study in the main paper showing step‑by‑step how the test‑time reasoning resolves ambiguous alignments.

## Removed Points
*These points are flagged to be removed, treat them with caution:*  
- No points were removed; all criticisms were factually grounded and did not violate the hard rules (e.g., none questioned the existence of cited models or misunderstood the paper’s content).

## Suggestions
- Provide a quantitative evaluation of the reliability estimation module (e.g., AUC for noisy/clean classification) and a systematic error analysis for the test‑time reasoning module, documenting common failure patterns.
- Explore and discuss efficiency improvements for the test‑time module, such as dynamically triggering MLLM reasoning only for low‑confidence predictions or distilling the MLLM into a smaller model.
- Conduct a sensitivity analysis of the greedy consensus strategy, showing how violations of Assumption 1 affect performance and under what noise conditions it remains reliable.