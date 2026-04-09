## Summary

This paper systematically investigates memory-accuracy trade-offs for deploying reasoning models under fixed memory budgets, studying over 1,700 configurations across the Qwen3, DeepSeek-R1-Distill, and OpenReasoning-Nemotron families. It identifies a scale-dependent threshold (around 8-bit 4B effective size) below which allocating memory to larger/higher-precision weights outperforms longer generation, and above which the opposite holds. It further shows that 4-bit quantization is memory-optimal for knowledge-intensive tasks but suboptimal for mathematical reasoning and code generation, and that KV cache eviction outperforms KV cache quantization for small models while both are competitive for larger ones.

## Strengths

- **Comprehensive empirical design covering multiple interacting dimensions.** The study varies five key factors (model size, weight precision, token budget, parallel scaling, KV cache compression) across 1,700+ configurations, three model families, and four benchmarks. This scope enables principled conclusions about when different strategies dominate, rather than single-point comparisons. The Pareto frontier framing makes the trade-offs immediately interpretable.

- **Actionable scale-dependent threshold that refines prior prescriptions.** The finding that models below ~8-bit 4B effective size should prioritize weight capacity over test-time compute directly challenges the prevailing assumption that 4-bit quantization is universally memory-optimal (Dettmers & Zettlemoyer, 2023). This threshold is supported by consistent evidence across Qwen3 (Figure 1–2) and validated on DeepSeek-R1-Distill (Figure 6) and OpenReasoning-Nemotron (Figure 16).

- **Task-specific precision insights that go beyond prior work.** The demonstration that 4-bit weights are memory-optimal for GPQA-Diamond (knowledge-intensive) but 8-/16-bit weights dominate for AIME25 and LiveCodeBench (math/code) provides a nuanced correction to scale-agnostic quantization guidelines. This aligns with and extends concurrent findings (Li et al., 2025a; Liu et al., 2025b) by embedding them in a memory-budget framework.

- **KV cache compression analysis integrated into the deployment trade-off.** Rather than treating weight quantization and KV cache compression independently, the paper shows that both eviction and quantification advance the Pareto frontier (Figure 8), and identifies that eviction dominates for small effective sizes while quantization becomes competitive for larger models (Figure 9).

## Weaknesses

### Major:

- **The "effective size" metric conflates parameter count and precision, obscuring the mechanism behind the threshold.** The key finding is organized around "effective size" (parameters × bits per weight), but this single aggregate hides whether the threshold is driven by having more parameters at lower precision or fewer parameters at higher precision. For instance, a 32B model at 4-bit and an 8B model at 16-bit have similar effective sizes but very different properties. Without disentangling these factors, the 8-bit 4B threshold could be an artifact of how these two dimensions interact in the specific model families tested, rather than a principled boundary. The paper would be significantly stronger with an ablation that varies parameter count and bit-width independently while holding effective size constant.

- **The choice of HQQ for KV cache quantization may not represent the true Pareto frontier, potentially biasing the eviction-vs-quantization comparison.** HQQ is primarily a weight quantization method; specialized KV cache quantization methods like KIVI (Liu et al., 2024, cited in the paper) are designed to handle the asymmetric importance of keys versus values and the online nature of KV caching. If a more capable KV quantizer narrows the gap with eviction, then finding #5 ("eviction is more effective than quantization for small models") may partially reflect the choice of suboptimal KV quantization rather than a fundamental trade-off. The paper should either justify this methodological choice or acknowledge it as a limitation that affects the strength of this particular conclusion.

### Minor:

- **Lack of mechanistic explanation for the threshold.** The 8-bit 4B threshold is empirically identified but not explained. Is it related to attention head capacity, activation magnitude distributions, or numerical properties of specific layers? Without understanding why this threshold exists, it remains unclear whether it will shift with architectural improvements or apply to future model families with different designs.

- **The "reasoning-specific" nature of the findings is not empirically isolated from generation length.** The paper claims that reasoning models require different memory strategies than non-reasoning models, but the key differentiator may simply be long generation length rather than reasoning per se. A long-context summarization or document QA task with similar token budgets might exhibit comparable KV cache dominance. The comparison to prior non-reasoning work (e.g., Dettmers & Zettlemoyer, 2023) uses different evaluation protocols and shorter contexts, making it difficult to attribute the difference to reasoning versus generation length. The paper acknowledges testing only on "challenging benchmarks representing complementary difficulty profiles" (Section 3) but does not include a long-context non-reasoning control.

- **Budget forcing may introduce artifacts that vary across precision levels.** The paper uses "Wait" prompt injection to extend generation beyond natural stopping points (Section 3). While standard practice (Muennighoff et al., 2025), this technique can cause looping or hallucination, and the paper itself notes non-monotonic accuracy on MATH500 (Appendix C.4). If forced generation degrades more severely at lower precision, the serial scaling comparisons could systematically understate the value of longer generation for 4-bit models, biasing the threshold identification. An analysis of whether budget forcing artifacts interact with precision would strengthen confidence in the threshold.

- **Generalizability beyond 32B parameters is untested.** The largest model evaluated is Qwen3-32B. At 70B+ scales, KV cache dominance becomes even more pronounced, and it is unclear whether the 8-bit 4B threshold shifts, whether 4-bit quantization remains suboptimal for math, or whether the eviction-vs-quantization trade-offs change qualitatively.

- **Latency and throughput trade-offs are relegated to the appendix.** For a paper offering deployment guidelines, the latency analysis (Appendix C.1) is directly relevant—particularly for parallel scaling, where the memory savings come at a substantial time cost. The finding that 4-bit precision is "never on the Pareto frontier for any model size" in latency-accuracy trade-offs (Appendix C.1) is an important qualifier to the main paper's memory-centric conclusions and deserves more prominent discussion.

### Trivial:

- The term "scale" in the subtitle is ambiguous—it could refer to model size, memory budget, or generation length. Given that the paper's central thesis is about the interaction of these different notions of scale, slightly more precise language would aid initial comprehension.

## Nice-to-Haves

- A practitioner-oriented decision table mapping (task type, memory budget) → recommended configuration, distilling the 1,700+ configurations into actionable rules.
- Qualitative error analysis showing what specifically breaks under 4-bit quantization for mathematical reasoning (e.g., arithmetic errors vs. logical errors vs. planning failures), which would provide mechanistic insight into the precision sensitivity finding.
- Evaluation on long-context non-reasoning tasks at similar token budgets to isolate whether the observed effects are reasoning-specific or generation-length-specific.
- Comparison with a specialized KV cache quantization method (e.g., KIVI) to validate the eviction-vs-quantization findings.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Table 1 memory discrepancy (Qwen3-4B at 16-bit ≈ 4.19 GB vs. expected ~8 GB):** The reviewer estimated 4B parameters × 2 bytes ≈ 8 GB, but model names are approximate and modern architectures commonly use tied embeddings or parameter sharing, reducing stored parameters. The KV cache values in Table 1 are internally consistent with the architectural details in Table 2 (e.g., Qwen3-4B: 144 KB/token × 2000 tokens ≈ 0.27 GB). The reported weight sizes likely reflect actual stored parameters rather than a computation error.

- **Missing non-reasoning baseline experiments (e.g., MMLU at long context):** The paper explicitly scopes its contribution to reasoning models and compares against established findings for non-reasoning models (citing Dettmers & Zettlemoyer, 2023; Chee et al., 2023). Running full non-reasoning baselines at matched token budgets would be valuable but is scope creep for this paper.

- **Reproducibility concerns about undisclosed hyperparameters:** The paper provides detailed inference specifications (temperature, budget forcing protocol, quantization settings) and links to code. The concern about missing implementation details is not substantiated.

- **Confidence intervals on Pareto frontier plots:** The paper reports accuracy averaged over 32 generations per instance. Adding confidence intervals to Pareto frontier curves is not standard practice for this type of analysis and would add visual clutter without changing the qualitative conclusions.

- **Hardware specificity of thresholds:** The memory-accuracy trade-offs studied are fundamentally about memory allocation, which is hardware-independent. While latency/throughput would vary across hardware, the core memory findings are not GPU-specific.

- **Missing related works:** Per the hard rules, I cannot confirm the existence of specific uncited works.

- **Formatting/style nitpicks:** Terminology consistency and figure presentation issues are minor and do not affect the core contributions.

- **QAT (Quantization-Aware Training) baselines:** The paper explicitly scopes to post-training quantization methods, which is the standard deployment scenario. Including QAT is outside the stated scope.

## Novel Insights

The paper's most striking finding is the *task-dependent inversion* of the 4-bit optimality rule: for knowledge-intensive tasks, the established wisdom that 4-bit is memory-optimal still holds, but for mathematical reasoning and code generation, 8-/16-bit weights are more memory-efficient even when accounting for the reduced KV cache capacity that comes with higher precision. This suggests that quantization noise selectively degrades the computational reasoning capabilities that chain-of-thought amplification relies on, while leaving knowledge retrieval relatively intact—a distinction that has significant implications for how we should think about deploying reasoning models versus standard LLMs.

## Suggestions

- Disentangle the effective size threshold by running experiments where parameter count and precision are varied independently at matched memory budgets (e.g., compare a hypothetical 8B model at 4-bit vs. a 4B model at 8-bit, both at similar total memory), to determine whether the threshold is driven by capacity (parameter count) or numerical fidelity (precision).

- Add a brief discussion or footnote acknowledging that the eviction-vs-quantization comparison uses HQQ rather than a KV-specific quantizer, and note this as a limitation that could affect the strength of finding #5.

- Move the key latency/throughput findings from Appendix C.1 into the main paper, at minimum as a paragraph noting that 4-bit precision is never latency-optimal, which qualifies the memory-only recommendations.

- Include a long-context non-reasoning task (e.g., document QA with 10k+ tokens) as a control to isolate whether the observed KV cache dominance is reasoning-specific or simply a function of generation length.