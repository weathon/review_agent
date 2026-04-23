Now I have all the information I need. Let me synthesize the final review.

## Summary

This paper investigates memory-accuracy trade-offs for deploying reasoning models, where KV cache (rather than model weights) can dominate memory. Through systematic Pareto frontier analysis across 1,700+ configurations spanning model sizes (0.6B–32B), weight precisions (4/8/16-bit), token budgets, parallel scaling, and KV cache compression strategies, the paper identifies a scale-dependent threshold (~8-bit 4B effective size) governing when to allocate memory to model capacity versus test-time compute. It also shows that the universal 4-bit quantization prescription fails for mathematical reasoning tasks, and that KV cache compression (both eviction and quantization) is essential beyond weight quantization alone.

## Strengths

- **Timely and important problem framing**: As reasoning models generate far more tokens, KV cache becomes a first-class memory concern. The paper provides concrete evidence (Table 1: Qwen3-4B at 4-bit uses 2.49 GB for weights vs. 4.42 GB for a 32k KV cache) that prior work's focus on weight compression alone is insufficient for this regime.

- **Extensive empirical scope**: Over 1,700 configurations across three model families (Qwen3, DeepSeek-R1-Distill, OpenReasoning-Nemotron), four benchmarks, multiple quantization schemes, and KV cache compression strategies provide a rich reference for practitioners.

- **Challenges established quantization wisdom with task-specific evidence**: The finding that 4-bit is memory-inefficient for mathematical reasoning (Figure 2: 32B-4bit is strictly dominated by 14B-8bit and 8B-16bit on AIME25) while remaining optimal for knowledge-intensive tasks (Figure 4: GPQA-Diamond) is a concrete, practically important corrective to Dettmers & Zettlemoyer (2023).

- **Cross-model-family validation**: The scale-dependent parallel scaling pattern is confirmed on DeepSeek-R1-Distill (Figure 6) and OpenReasoning-Nemotron (Figure 16), strengthening the generalizability of core findings.

- **Quantization scheme robustness**: Appendix C.2 verifies that AWQ and FP8 produce nearly identical memory-accuracy curves to GPTQ, confirming findings are not artifacts of a specific quantization method.

- **Clean negative result on external verifiers**: Figure 7 shows PRM-based Best-of-N (13.28 GB overhead from ActPRM-X) is consistently outperformed by self-contained majority voting on the Pareto frontier—a practical and counterintuitive deployment guideline.

- **KV cache compression shown essential beyond weight quantization**: Finding 4 (Figure 8) demonstrates both eviction and quantization advance the Pareto frontier across all weight precisions, establishing that reasoning model deployment requires combined strategies.

## Weaknesses

### Fatal
None.

### Major

- **Internal inconsistency in the central threshold**: The paper's organizing principle is a scale-dependent threshold, but Finding 5 (line 36) states "models with an effective size smaller than an **8-bit 4B** model" while Section 5's detailed analysis (line 136) states "models with an effective size smaller than an **8-bit 8B** model." The conclusion also refers to "models under the 8B size." This inconsistency is never acknowledged or explained. If the threshold is "8-bit 4B" as stated in Findings 1–3 and 5's summary, the Section 5 analysis contradicts it; if it is "8-bit 8B," then Findings 1–3 overclaim. Either way, the precision of the paper's central organizing concept is undermined. This matters because the threshold is used to derive concrete deployment prescriptions, and practitioners relying on the wrong threshold would make incorrect allocation decisions.

- **No uncertainty quantification on Pareto frontier differences**: The entire argument rests on which configurations lie on the Pareto frontier, but no confidence intervals, variance, or statistical tests are reported. Accuracies are averages over 32 generations (8 for KV cache experiments), and frontier differences between adjacent configurations often appear to be a few percentage points—well within plausible variance. A small perturbation in accuracy could reshape the frontier and alter qualitative conclusions (e.g., whether 4-bit or 8-bit lies on the frontier at a given budget). Without uncertainty quantification, it is impossible to assess whether the reported frontier differences are signal or noise. This is particularly important given that the paper makes specific numeric prescriptions (the "8-bit 4B" threshold) from these frontier differences.

### Minor

- **The "8-bit 4B" threshold is derived from a limited set of discrete model sizes**: The threshold falls between the 1.7B and 4B Qwen3 models at 8-bit, but "effective size" is not an independently varied continuous variable—it is derived from two discrete factors yielding a small number of distinct configurations. With different model size granularity (e.g., a 2.5B or 3B variant), the threshold could shift. The paper does not discuss this sensitivity. However, the cross-model-family validation on DeepSeek-R1-Distill and OpenReasoning-Nemotron provides some robustness, and the qualitative direction of the finding (smaller models benefit from weight capacity, larger models from test-time compute) is likely robust to the exact threshold value.

- **Task-conditional findings limit the universality of guidelines**: The paper promises "principled guidelines" but the optimal strategy depends heavily on task type (4-bit for GPQA, 8/16-bit for AIME). The paper does not provide a framework for mixed workloads, which is the typical deployment scenario. The paper does acknowledge this dependency (Finding 2), so this is more about the gap between the abstract's framing ("principled guidelines") and the actual conditional nature of the findings.

### Trivial
None.

## Nice-to-Haves

- Bootstrapped confidence intervals or error bars on Pareto frontier plots would immediately clarify whether frontier differences are statistically meaningful.
- Experiments with intermediate model sizes (e.g., 2B, 3B) to test threshold robustness.
- Analysis of how architectural choices (e.g., GQA with fewer KV heads) mediate the weight-to-KV-cache ratio and thus the trade-offs studied.
- Mixed-workload evaluation to bridge the gap between task-conditional findings and practical deployment.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Batched inference assumption buried"** (Harsh Critic): The paper explicitly states "Assuming a batched inference setting" in the parallel scaling section (line 96) and analyzes non-batched settings in Appendix C.3 (line 74). The assumption is clearly disclosed, not buried.

- **"Only a single PRM verifier evaluated"** (Harsh Critic): The paper explicitly scopes this as "a limited evaluation of an external verifier rather than a comprehensive comparison of verifier-based methods" in the Limitations section (line 155). This is acknowledged, not overlooked.

- **"Missing appendix, missing proofs in appendix, or absent references"**: Parser artifacts—the original submission includes these.

- **"Architectural analysis of weight-to-KV-cache ratio" as a Major weakness**: The paper chose Qwen3 specifically for its "broad size range and fixed architecture" (line 155), which is a deliberate experimental choice to isolate the effect of model size. Requesting architectural variation is reasonable as a nice-to-have but not a core flaw.

- **"Report pass@k with different k values"**: This is a reasonable suggestion but not a weakness of the current analysis, which is consistently framed around Pareto-optimal memory-accuracy trade-offs.

- **Generic strengths without specific citations**: Several of the Strength Finder's items were too generic (e.g., "well-structured findings") and have been filtered.

## Novel Insights

The most insightful observation across the reviews is that the paper's threshold inconsistency (8-bit 4B vs. 8-bit 8B) may reflect a genuine phenomenon rather than a simple error: the crossover point for KV cache compression strategy (eviction vs. quantization) may genuinely differ from the crossover for serial vs. parallel scaling, because the two trade-offs involve different mechanisms. If the paper explicitly acknowledged and investigated this possibility—that the "effective size" threshold is context-dependent rather than universal—it would strengthen rather than weaken the contribution by adding nuance to the framework.

## Suggestions

- Fix the threshold inconsistency immediately: either reconcile "8-bit 4B" and "8-bit 8B" with an explanation of why the KV cache compression threshold differs from the scaling threshold, or correct one of them if it is an error.
- Add bootstrap confidence intervals on the Pareto frontier—this would be low-effort (the 32 samples per instance are already collected) and would dramatically increase the credibility of the frontier comparisons.
- Reframe the abstract to acknowledge the task-conditional nature of the guidelines alongside the scale-dependence, e.g., "scale- and task-dependent" rather than just "scale-dependent."

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| Reasoning with Sampling (Vsgq2ldr4K) | 7.5 | Accept (Oral) | Novel method with broad applicability; this paper is weaker—no novel method, only empirical observations |
| ParoQuant (1USeVjsKau) | 7.0 | Accept (Poster) | Novel quantization method with hardware co-design; this paper has broader empirical scope but no algorithmic contribution |
| Is Finer Better? (3jDSqfTSrn) | 5.5 | Accept (Poster) | Similar scale-dependent threshold finding with theoretical backing; this paper has broader experiments but no theory and has the threshold inconsistency |
| Scaling Law for QAT (dcPH77OVgN) | 5.0 | Reject | Empirical scaling law limited to one model family; this paper has much broader scope and more practical value |
| Reliability Scaling Laws (QhkW8xPH1v) | 5.0 | Reject | Comprehensive empirical study lacking theoretical depth; this paper provides more actionable findings |
| GSR-Guided Quantization (mUB2N8L0vD) | 4.5 | Reject | Overclaimed generalizability; this paper is stronger with broader validation |
| ILRe (GiI6tPrPAG) | 2.0 | Reject | Fundamentally flawed; this paper is clearly superior |

This paper sits above the rejected empirical studies (5.0) due to its broader scope, cross-model-family validation, and practical importance, but below the accepted method papers (7.0+) that offer novel algorithms. It is comparable to "Is Finer Better?" (5.5) but trades theoretical backing for broader experiments, while also carrying the threshold inconsistency. The lack of uncertainty quantification and the threshold inconsistency are meaningful but do not invalidate the directionally correct qualitative observations. The paper provides genuine practical value for practitioners deploying reasoning models under memory constraints.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>