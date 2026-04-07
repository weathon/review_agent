## Summary
The paper investigates memory-accuracy trade-offs for reasoning models under fixed memory budgets, challenging the universal applicability of 4-bit quantization for LLMs. Through systematic experiments across 1,700+ configurations (Qwen3, DeepSeek-R1-Distill, OpenReasoning-Nemotron on AIME25, GPQA-Diamond, LiveCodeBench, MATH500), the authors identify a scale-dependent threshold: models with effective size below ~8-bit 4B parameters benefit more from allocating memory to larger weights, while larger models benefit more from test-time compute (longer generations). The paper further shows that weight precision preferences are task-dependent (4-bit for knowledge-intensive tasks, 8-bit+ for math/code) and that KV cache eviction outperforms quantization for smaller models.

## Strengths
- **Comprehensive empirical scope**: The study systematically explores a vast configuration space—six model scales (0.6B–32B), three weight precisions, token budgets from 2k–30k, parallel scaling group sizes up to 16, and multiple KV compression strategies. This breadth provides unusually strong empirical grounding for practical deployment guidelines.

- **Task-specific precision insights**: The finding that 4-bit weights are memory-optimal for knowledge-intensive tasks (GPQA-Diamond) but suboptimal for mathematical reasoning (AIME25) and code generation (LiveCodeBench) is a substantive contribution. This contradicts the "4-bit is universally optimal" heuristic and provides actionable guidance for practitioners.

- **KV cache compression comparison**: The systematic comparison of eviction (R-KV, StreamingLLM) vs. quantization (HQQ) across model scales fills a gap in the literature. The finding that eviction is preferable for smaller models while quantization becomes competitive for larger models is novel and practical.

## Weaknesses
- **No uncertainty quantification despite small benchmark sizes**: The paper's central claims rest on accuracy differences measured on AIME25, which contains only 30 problems. Even with 32 generations per instance averaged to estimate pass@1, the benchmark-level variance is substantial. The paper states "the 8B model in 8-bit consistently outperforms the 14B model in 4-bit"—but these differences are often 3-5 percentage points on a 30-problem set. Without confidence intervals, error bars, or statistical tests, it is impossible to assess whether observed differences are meaningful or within noise.

- **Inconsistent threshold values across analyses**: Section 4 identifies "8-bit 4B" as the threshold for weight-vs-KV allocation, while Section 5 states that for KV eviction-vs-quantization, "models with an effective size smaller than an 8-bit 8B" benefit from eviction. The paper does not explain why these thresholds differ or whether this reflects genuine task/phenomenon differences versus experimental artifacts. This inconsistency undermines the narrative of a clean, unified threshold.

- **Threshold precision overstates empirical resolution**: The "8-bit 4B" threshold is stated with precision that is not earned. The tested model sizes (0.6B, 1.7B, 4B, 8B, 14B, 32B) are discrete jumps, and the threshold could lie anywhere between 1.7B and 4B. The paper acknowledges MATH500 shifts the threshold to smaller sizes, further showing this is not a fixed constant. The framing should be more appropriately hedged.

- **Potential budget forcing confound**: The budget forcing methodology (injecting "Wait" when the model tries to stop) may disadvantage small models disproportionately. Smaller models may have weaker ability to recover coherent generation from forced continuation, which could partially explain why allocating more tokens to small models is memory-inefficient. This confound is not discussed.

## Nice-to-Haves
- **Latency/throughput integration in main figures**: While Appendix C.1 analyzes latency and throughput, these metrics are critical for deployment decisions. Integrating latency into the main Pareto frontier figures would make the practical implications more accessible.

- **Hardware diversity**: All experiments use a single A100 80GB GPU. The thresholds may shift on hardware with different memory bandwidths or on smaller consumer GPUs. A brief discussion or validation on alternative hardware would strengthen generalizability.

- **Non-reasoning model baseline**: Including a non-reasoning model (e.g., a standard LLM without extended chain-of-thought training) would more directly validate the paper's core claim that reasoning models require fundamentally different memory strategies.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"No theoretical grounding"**: The paper is explicitly empirical and does not claim theoretical contributions. Demanding theory for an empirical systems paper is scope creep.
- **"Limited PRM/verifier comparison"**: The paper already acknowledges this limitation in Section 7. The critique that only one PRM variant was tested is valid but addressed in the limitations.
- **"Missing QAT baselines"**: Quantization-aware training is a fundamentally different approach requiring training, while the paper focuses on post-training quantization for inference-only deployment. This is outside scope.
- **"Definition of effective size conflates memory and capability"**: The paper clearly defines effective size as the memory footprint of weights (N × PW), not as a capability proxy. This is a consistent definition throughout.

## Novel Insights
Beyond the paper's own contributions, a novel synthesis emerges: **reasoning models exhibit a "memory-compute inversion" that does not exist for non-reasoning models.** For non-reasoning models, memory optimization has centered on model weights because KV cache was negligible. For reasoning models, the KV cache can exceed weight memory by 1.8× or more, fundamentally changing the optimization landscape. This shift suggests that as reasoning models become longer-horizon (e.g., agents, multi-step planners), memory optimization will increasingly diverge from the established 4-bit orthodoxy, and techniques developed for KV cache-heavy workloads (eviction, compression) will become first-class citizens alongside weight quantization. The paper's scale-dependent threshold suggests we should think of a "reasoning capacity frontier"—below a certain model scale, reasoning quality is capacity-limited, and memory should be allocated to model weights; above this scale, reasoning quality is compute-limited, and memory should be allocated to longer inference.

## Suggestions
1. **Add confidence intervals or bootstrap error bars** to key figures (especially Figures 1, 5, 8) and report whether accuracy differences exceed statistical significance thresholds given benchmark sizes.
2. **Clarify the relationship between the two thresholds**: Either explain why the weight-vs-KV threshold (8-bit 4B) differs from the eviction-vs-quantization threshold (8-bit 8B), or acknowledge that the threshold phenomenon is approximate and context-dependent.
3. **Discuss budget forcing artifacts**: Add a brief discussion of whether the memory-inefficiency of small models with long generations could be partially attributable to budget forcing disruptions rather than pure memory-accuracy trade-offs.