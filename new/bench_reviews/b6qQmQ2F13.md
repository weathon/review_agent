Now I have enough calibration context. Let me synthesize the final review.

## Summary

This paper systematically investigates memory–accuracy trade-offs for deploying reasoning models, studying how to optimally allocate a fixed memory budget across model size, weight precision, KV cache length, parallel scaling group size, and KV cache compression. Through over 1,700 experimental configurations primarily on the Qwen3 family (0.6B–32B), with validation on DeepSeek-R1-Distill and OpenReasoning-Nemotron, the authors identify a scale-dependent effective-size threshold (~8-bit 4B) below which memory is better spent on larger weights rather than longer generation, and above which the opposite holds. They also find that math/code reasoning is more sensitive to weight precision than knowledge-intensive tasks, and that KV cache eviction outperforms quantization for small models while they become competitive for large ones.

## Strengths

1. **Practically important and timely problem formulation.** The insight that reasoning models fundamentally change the memory landscape (KV cache can exceed weight memory) and the consequent reframing of test-time scaling from FLOPs to memory constraints is valuable and directly relevant to deployment. The concrete example (Qwen3-4B at 4-bit: 2.49 GB weights vs. 4.42 GB KV cache) effectively motivates the work.

2. **Comprehensive empirical exploration at scale.** Over 1,700 configurations across five axes (model size, weight precision, token budget, group size, KV compression), six model sizes, multiple quantization schemes (GPTQ, AWQ, FP8), and four benchmarks provide a rich empirical map. The scope of experimentation is a genuine contribution.

3. **Task-dependent precision finding is novel and useful.** The discovery that 4-bit quantization is memory-optimal for knowledge-intensive tasks (GPQA) but suboptimal for mathematical reasoning (AIME25) and code generation (LiveCodeBench) is an important and actionable insight that contradicts the prevailing wisdom from Dettmers & Zettlemoyer (2023). This finding aligns with practitioner experience and is well-supported by the data.

4. **Good Pareto frontier methodology.** The approach of plotting memory vs. accuracy across configurations provides a clean, principled framework for comparing fundamentally different allocation strategies on equal footing, making the trade-offs visually and analytically clear.

5. **Honest limitations section.** The paper explicitly acknowledges its scope constraints (Qwen3 focus, inference-only, limited quantization schemes, two main benchmarks), which is appropriate.

## Weaknesses

### Major

1. **The "8-bit 4B" threshold is presented as a sharp, general organizing principle but is weakly supported as such.** This is the paper's central finding and it appears in Findings 1, 3, and 5, yet the empirical basis is notably coarse around the alleged boundary. Qwen3 model sizes are {0.6B, 1.7B, 4B, 8B, 14B, 32B}, so the transition region near ~4.2 GB is probed by only a few configurations (4-bit 4B ≈ 2.49 GB, 8-bit 4B ≈ 4.19 GB, 8-bit 1.7B ≈ 1.93 GB, 16-bit 0.6B ≈ 1.40 GB). The data supports a smooth regime change in preferences as effective size increases, but promoting it as a specific "8-bit 4B" threshold that governs multiple aspects of memory optimization overstates what can be concluded from six model sizes at three precisions per task. The paper acknowledges that "our findings do not provide specific prescriptions for all tasks or models" and that "the inflection point may change as models become more sophisticated," but these caveats are not reflected in the Finding statements or the conclusion, which repeatedly assert the threshold as a principled guidepost.

2. **No statistical uncertainty quantification on Pareto frontier claims.** Many key arguments depend on one configuration "strictly dominating" or being "consistently memory-inefficient" compared to another (e.g., "the 32B model in 4-bit is strictly dominated by both the 14B model in 8-bit and the 8B model in 16-bit"). AIME25 has only 30 problems; GPQA-Diamond has ~198. With 32 generations per instance and the resulting variance in pass@1 estimates, it is unclear whether these dominance relationships are robust or within noise. For a paper whose core contribution is identifying specific allocation boundaries on Pareto frontiers, the absence of error bars, confidence intervals, or bootstrap analyses substantially weakens confidence in the claimed threshold locations and frontier shapes. This concern is partially mitigated by the large number of configurations and the fact that some trends are visually clear, but it remains important for claims about specific crossover points.

3. **Claims of generality outpace cross-architecture validation.** While the paper validates on DeepSeek-R1-Distill and OpenReasoning-Nemotron for parallel scaling (Figures 6, 16), the core serial scaling findings (Findings 1 and 2) and the KV compression findings (Finding 5) are demonstrated primarily on Qwen3. The Qwen3 family, with its fixed architecture and broad size range, is a reasonable primary object of study, but the paper's language—speaking of "reasoning models" generically and presenting "general principles"—extends beyond what two supplementary model families for one sub-analysis can establish. The extent to which the 8-bit 4B threshold and precision preferences transfer across architectures, training procedures, and future model generations remains an open question that the paper does not adequately address for the breadth of its claims.

4. **The effective-size metric conflates architecturally distinct models.** A 4-bit 8B model (≈5.68 GB, 8B parameters, reduced precision) and a 16-bit 1.7B model (≈3.78 GB, 1.7B parameters, full precision) have different depths, attention heads, and hidden dimensions, yet are treated as belonging to the same "effective size" continuum. The paper treats effective size as the organizing variable, but architectural structure could matter as much as raw memory footprint, particularly for reasoning tasks that depend on chain-of-thought depth. The paper does not unpack this conflation or investigate when models of similar effective size but different architectures diverge in behavior.

### Minor

5. **External verifier comparison is too narrow for the breadth of the claim.** Section 4.1 evaluates a single PRM (ActPRM-X, 7B/13.28 GB) and concludes that "using an external verifier such as PRM is memory-inefficient." One large verifier at one size, tested on one benchmark, does not establish this general conclusion. Smaller or distilled PRMs, different verification strategies (e.g., partial verification, early pruning), or multi-turn verification could change this trade-off. This is a secondary finding but the claim is stated too broadly.

6. **Budget forcing as the sole serial scaling mechanism limits generalizability.** All serial scaling results depend on the "Wait" injection protocol for budget forcing. Whether the same memory-accuracy trade-offs hold under natural reasoning lengths, different continuation prompts, or with models trained for longer reasoning is not explored. The paper acknowledges this implicitly through its scope constraints, but the findings are presented as general memory-allocation principles for reasoning models when they are specifically tied to budget forcing.

7. **KV compression experiments reduce evaluation depth.** The KV compression section (Section 5) uses 8 generations per instance instead of 32 used elsewhere, apparently for computational cost reasons. This is a notable reduction in sampling depth for the section that makes strong claims about Pareto frontiers and eviction-vs.-quantization dominance. The impact of reduced sampling on variance and frontier stability is not analyzed.

### Trivial

- The distinction between "8-bit 4B" as a threshold and "models under the 8B size" in the conclusion is slightly inconsistent—the abstract and Findings use effective size (parameters × bits), while the conclusion summarizes in terms of parameter count alone.

## Nice-to-Haves

- **Confidence intervals or bootstrap resampling** on Pareto frontier points, particularly around threshold regions, would substantially strengthen the quantitative claims.
- **Mechanistic analysis of why the transition occurs near 8-bit 4B**—e.g., analyzing error propagation through multi-step reasoning chains at different precisions and scales—would elevate this from an empirical observation to a principled guideline.
- **Testing mixed-precision weight assignments** (e.g., 8-bit for attention layers, 4-bit for MLP) would be the natural deployment implication of Finding 2 and would significantly increase practical impact.
- **A concise summary table** mapping (effective size range, task type) → recommended (precision, serial budget, parallel group size, KV strategy) would make the "principled guidelines" more actionable than they currently are, scattered across five separate Findings.

## Removed Points

- **"Budget forcing could shift curves"**: While the reviewers raised this as a concern about generalizability, the paper is explicitly scoped to budget forcing as a serial scaling mechanism. I've kept a milder version under Minor weakness (6) since the paper does present the findings as general principles for reasoning models, but removed the stronger version that demands testing alternative serial scaling methods.

- **"Weights amortization makes main results unrealistic"**: The reviewer claimed the main results assume batch size 1 and are thus not deployment-relevant. However, the paper explicitly acknowledges this limitation (Section 4 mentions it, and Appendix C.3 analyzes batched settings). The batch-size-1 analysis is a valid and common simplifying assumption for studying memory allocation per-query, and the paper does not hide this. I've weakened this to a minor note rather than treating it as a major flaw.

- **"Only one eviction method (R-KV) and one quantizer (HQQ)"**: The paper explicitly acknowledges this in the Limitations section. The choice of one representative per method category (eviction vs. quantization) is a reasonable scoping decision for a study that already spans 1,700+ configurations. This is a nice-to-have for future work, not a core flaw.

- **"No theoretical justification for the threshold"**: While a theoretical account would strengthen the paper, this is an empirical systems paper, and the community standard for such work does not require theoretical proofs. The demand for a mechanistic explanation is reasonable as a nice-to-have but not a required contribution.

- **"The paper overclaims failure of prior 4-bit prescriptions"**: The reviewer suggested the paper implies Dettmers & Zettlemoyer's findings "fail." The paper actually correctly notes that prior work focused on non-reasoning, zero-shot tasks, and its findings apply to reasoning models with long generations. This is a fair contrast, not an overclaim.

- **"Latency claims depend on hardware/batching regime"**: The paper states that for small models, larger-effective-size configurations "are also faster because end-to-end latency is dominated by the token budget." This is a well-known property of autoregressive generation and is clearly documented in Appendix C.1. It is a factual observation, not an unsupported claim.

- **"Requesting additional PRMs/pruning strategies"**: The reviewer demanded exploring smaller PRMs or hybrid designs. The PRM comparison (Section 4.1) is a sub-analysis, not the core contribution. One data point is sufficient to suggest that a 7B verifier adds significant memory overhead; the paper's claim that it is "memory-inefficient" under tight budgets is reasonable for a verifier of that size.

## Novel Insights

The most distinctive contribution is the demonstration that the memory-optimal precision for reasoning models is task-dependent in a way that diverges from non-reasoning model prescriptions: 4-bit weights are broadly optimal for knowledge-intensive tasks, but 8-bit or 16-bit weights are memory-efficient for mathematical reasoning and code generation. This directly challenges the prevailing "4-bit is universally best for inference" wisdom established for non-reasoning models. The second notable insight is that KV cache eviction outperforms KV cache quantization for effectively small reasoning models—a reversal from what might be expected given the general preference for quantization in weight compression. These findings suggest that the reasoning model deployment landscape requires fundamentally different optimization strategies than what has been established for standard LLMs.

## Suggestions

1. Add error bars or bootstrap confidence intervals to the Pareto frontier plots, especially around the claimed 8-bit 4B transition region, to establish whether the threshold location is robust to estimation noise.
2. Reframe the Findings to present the 8-bit 4B threshold as an empirically observed transition region rather than a universal law—e.g., "for the model families and benchmarks studied, the memory-allocation preference shifts around an effective size of 8-bit 4B" rather than "for models effectively smaller than an 8-bit 4B."
3. Add a concise decision-making table or flowchart mapping (effective size, task type) to recommended configurations, making the guidelines immediately actionable for practitioners.
4. Investigate when models of similar effective size but different architectures diverge in behavior (e.g., 4-bit 8B vs. 8-bit 4B), even in a small appendix analysis, to address the effective-size conflation concern.

## Evaluation on Key Axes

- **Originality**: Moderate. The problem formulation (memory-constrained test-time scaling) is novel and important. The task-dependent precision finding is original. However, the empirical methodology is standard.
- **Importance of research question**: High. Reasoning model deployment under memory constraints is timely and significant.
- **Claims well supported**: Partially. The directional trends are well-supported, but the specific threshold and some "strict dominance" claims lack statistical backing.
- **Soundness of experiments**: Moderate. Very large scope but concentrated on one model family, one serial scaling strategy, and limited near-threshold granularity. No error bars.
- **Clarity**: Good. Well-organized with clear Findings and visualizations.
- **Value to community**: Moderate to high. Useful practical guidelines with important caveats about generalizability.

## Score and Decision

Calibration against reviewed papers:

- **Scaling Laws for Precision (Oral, avg 8.0)**: Has deep theoretical grounding, predictive equations, validated on 465+ runs, much stronger theoretical contribution. This paper is clearly below this.
- **Test-Time Compute (Oral, avg 7.5)**: Novel framework, extensive experiments, but some overclaiming about matching larger models. This paper has similar overclaiming issues but less theoretical depth.
- **Inference Scaling Laws (Poster, avg 5.75)**: Empirical study with limited models/tasks, mixed reviews, concerns about generalizability. This paper has a similar profile—empirical study making broader claims than fully supported.
- **Inference Optimal VLMs (Poster, avg 5.8)**: Similar trade-off study, some concerns about generality and limited baselines. Most directly comparable.
- **Empirical Guidelines for Deploying LLMs (Reject, avg 4.75)**: Empirical guidelines without statistical rigor, vague conclusions. This paper is significantly stronger—more rigorous methodology, more configurations, clearer framework.

This paper falls between the "Inference Scaling Laws / Inference Optimal VLMs" tier and the rejected "Empirical Guidelines" paper. It has genuinely useful empirical contributions and an important problem formulation, but the central threshold claim is over-precise for the data, and no statistical uncertainty analysis is provided. The task-dependent precision finding alone is a worthwhile contribution. The paper would be stronger with more measured claims and statistical rigor.

Score: 5.5

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>