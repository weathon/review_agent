## Summary
This paper systematically investigates memory-accuracy trade-offs for deploying reasoning-focused large language models under fixed memory budgets. It demonstrates that optimal memory allocation is scale-dependent: for models with an effective size below approximately that of an 8-bit 4B model, memory is better spent on higher-precision weights, while larger models benefit more from longer generations (larger KV cache). The work also analyzes how parallel scaling and KV cache compression strategies are governed by model scale and task type.

## Strengths
- **Comprehensive empirical study:** The paper evaluates over 1,700 configurations across multiple model families (Qwen3, DeepSeek-R1, OpenReasoning-Nemotron), tasks (mathematical, coding, knowledge-intensive), and optimization axes (weight precision, token budget, group size, KV cache compression). This extensive experimentation strongly supports the identified trends.
- **Clear, actionable guidelines:** The core finding of a scale-dependent threshold for allocating memory between weights and KV cache is well-supported by Pareto frontier analysis and distilled into practical recommendations for practitioners. The distinction between task types (mathematical/code vs. knowledge-intensive) regarding optimal weight precision is particularly valuable.
- **Problem reformulation:** The work successfully reframes test-time scaling from a FLOPs-centric view to a memory-constrained deployment perspective, highlighting the growing dominance of the KV cache in reasoning workloads—a crucial insight for real-world serving.

## Weaknesses
- **Lack of statistical uncertainty reporting:** Accuracy metrics are averaged over 32 generations per instance, but no error bars, standard deviations, or confidence intervals are provided. This omission makes it difficult to assess the robustness of comparisons, especially for configurations near the Pareto frontier.
- **Threshold generality and presentation:** The specific threshold of "8-bit 4B" is derived primarily from the Qwen3 family. While similar scale-dependent behavior is shown for other model families, the paper occasionally presents this threshold as a fixed point rather than an observed trend that may vary across architectures. A more nuanced presentation would strengthen the claims.
- **Limited architectural diversity in primary analysis:** The detailed analysis is centered on the Qwen3 family; validation on DeepSeek-R1 and OpenReasoning-Nemotron is provided but less exhaustive. A broader variety of architectures (e.g., different attention mechanisms, MoE models) would increase confidence in the generalizability of the findings.
- **Latency/throughput analysis is not integrated into core guidelines:** While Appendix C.1 analyzes latency and throughput, these critical deployment metrics are not incorporated into the main memory-accuracy trade-off framework or the final recommendations. For practical deployment, a joint consideration of memory, accuracy, and speed is often necessary.

## Nice-to-Haves
- Include an additional knowledge-intensive benchmark (beyond GPQA-Diamond) to further substantiate the claim that 4-bit weights are broadly memory-optimal for such tasks.
- Conduct a more precise mapping of the scale threshold by testing intermediate model sizes (e.g., 2B, 6B) to better characterize the transition region.
- Measure and report the latency and throughput implications of KV cache compression methods (eviction and quantization), as these can affect real-world performance.
- Test longer generation lengths (e.g., >30k tokens) to validate the claim that for large models, memory should be allocated to the KV cache "until saturation," as saturation points may be task-dependent.
- Provide a theoretical intuition or hypothesis for why the scale-dependent threshold exists, linking it to concepts like model capacity or task complexity, to give a deeper foundation for the empirical results.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- **Criticism about budget forcing artifacts:** The paper follows established prior work (Muennighoff et al., 2025) for budget forcing. A discussion of potential unnatural continuations is not required for the core contribution.
- **Demand for deeper mechanistic explanations** (e.g., why small and large models differ, root cause of precision sensitivity): These are interesting but go beyond the paper's empirical scope and are not necessary to validate the main findings.
- **Request for hyperparameter sensitivity analysis:** The paper uses standard settings for quantization and compression methods; a full sensitivity study is not expected in a broad empirical survey.
- **Claim that direct iso-memory comparisons are missing:** The Pareto frontier analysis inherently compares strategies across memory budgets; direct fixed-memory comparisons are implied by the curves.
- **Formatting and writing nitpicks** (e.g., informal phrase "false economy", verbose figure captions): These do not affect the technical content.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Report variance estimates (e.g., standard deviation across instances or bootstrap confidence intervals) for key accuracy measurements to strengthen the comparative claims.
- Clarify in the abstract and main text that the "8-bit 4B" threshold is an observed trend based on the studied models, not a universal constant, and note that the exact inflection point may shift with architecture and task.
- Elevate the latency/throughput discussion from the appendix to the main text, and incorporate latency considerations into the memory-optimization guidelines where appropriate (e.g., noting that weight quantization can reduce latency for large models).