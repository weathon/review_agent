## Summary
This paper investigates how foundational training strategies can adapt generalist MLLMs for deeper chart comprehension, specifically targeting the common reliance on OCR shortcuts for annotated numeric labels. The authors propose CHOPINLLM, trained via a three-stage pipeline: JSON-data alignment pre-training, end-to-end fine-tuning with a structured three-level QA taxonomy (literal, inferential, reasoning) augmented by text-only and data-driven QA pairs, and downstream LoRA fine-tuning. An efficient orthogonal data generation pipeline yields ~5M synthetic chart images. The paper also introduces a new benchmark spanning 20 chart types and multiple QA complexity levels. Experiments demonstrate improved robustness on unannotated charts and diverse chart types, though some empirical claims are overstated relative to reported tables.

## Strengths
- **Efficient orthogonal data generation pipeline** (Sec. 3.1, Fig. 2a): Decoupling code and data generation via shared JSON/README templates enables quadratic scaling of chart images without repeated multimodal LLM calls, offering a practical, low-cost alternative to linear iterative generation pipelines.
- **Addresses a genuine distribution shift** (Sec. 1, Fig. 1): The paper correctly identifies and empirically tackles the "annotation dependency" problem. Results on PlotQA (~33-34% vs. ~30% for baselines) and qualitative examples (Fig. 4, Fig. 5) validate that JSON alignment during pre-training reduces reliance on explicit numeric labels.
- **Structured three-level QA taxonomy** (Sec. 3.2, Fig. 3): Breaking chart comprehension into literal, inferential, and reasoning tiers provides a granular evaluation framework that moves beyond simple value retrieval, offering better diagnostic signals for MLLM capability assessment.
- **Comprehensive new benchmark** (Sec. 3.3, Table 1, Table 5): The evaluation covers 20 chart types and verifies generalization to non-overlapping, complex charts (e.g., Gantt, Heatmap, Funnel) where prior models struggle, expanding the evaluation landscape for chart-specific MLLMs.

## Weaknesses

### Fatal
None.

### Major
- **Overclaimed state-of-the-art performance in conclusion:** Section 5 concludes that *"CHOPINLLM surpasses the previous state-of-the-art across four benchmarks,"* a statement directly contradicted by the authors' own Table 4. ChartAst achieves a higher average on ChartQA (79.9 vs. 71.39) and outperforms CHOPINLLM on both Chart-to-Text splits (Pew: 15.5 vs. 12.66; Statista: 41.0 vs. 40.81). CHOPINLLM only clearly leads on PlotQA and the RNSS metric. Framing the results as universally superior when the table shows defeats on two of the four primary evaluation axes is a fundamental structural flaw that misrepresents the model's empirical standing.
- **Marginal empirical impact of novel training steps compared to standard instruction tuning:** The paper positions JSON-only QAs and data-driven QAs (Findings 2 & 3) as pivotal innovations. However, Table 3 reveals that adding standard chart instruction data (`+ Literal / infer. / reasoning QAs`) drives the overwhelming majority of performance gains (+24% literal, +17% inferential, +14% reasoning from baseline). The proposed novel components add only ~1–2% on top of this. This strongly suggests the observed improvements stem primarily from the scale and variety of domain-specific instruction tuning, rather than the specific architectural/training novelties claimed. The ablation design does not sufficiently isolate methodological contribution from data volume effects.

### Minor
- **Train-test leakage risk in the synthetic benchmark:** The new benchmark (Sec. 3.3) is drawn from the exact same generative pipeline used for the 5M training images. The paper does not specify the partitioning mechanism (e.g., disjoint Python script/JSON template splits, hash-based isolation). Given that the pipeline pairs orthogonal code variants with orthogonal data templates, structural or stylistic overlaps between train and test images are highly probable. Without explicit isolation guarantees, the benchmark results may partially reflect template memorization rather than true generalization.
- **Low absolute performance on complex chart types without error analysis:** Table 5 shows CHOPINLLM achieving 40.9% on Line charts, 25.8% on Heatmaps, and 15.8% on Gantt charts. While the paper frames this as "consistent outperformance" over baselines, the absolute scores indicate significant headroom. The absence of an error analysis (e.g., axis misalignment vs. scale hallucination vs. value inversion) makes it difficult to assess whether the model is genuinely extracting geometric/value signals or relying on brittle synthetic priors.

### Trivial
- None warranting inclusion. (Parser artifacts, minor formatting inconsistencies, and omitted appendix details per the review rules are excluded.)

## Nice-to-Haves
- **Controlled data-volume ablation:** Run a `Chart QA Only (Large)` vs. `Chart QA + JSON Alignment (Small)` comparison to definitively prove whether the proposed pre-training steps provide genuine marginal benefits beyond simply scaling instruction data.
- **Leakage control documentation:** Explicitly state the train/test separation protocol for the synthetic benchmark in the main text to bolster confidence in the evaluation's validity.
- **Inference vs. Training breakdown for Data-Driven QAs:** Clarify how much of the `+ Data Prompting †` gain (Table 3) stems from the fine-tuning on data-driven QAs versus the explicit chain-of-thought inference prompt, as the current conflation slightly obscures the true training contribution.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- **Criticism that Finding 3 is purely an inference-time technique misattributed to fine-tuning:** The paper explicitly trains with "Data-driven QAs" (multi-turn extraction prompts), which improves ChartQA human split from 49.60 to 52.28. The `†` denotes an *additional* inference-time prompting strategy that leverages this trained behavior, yielding 56.96. The training contribution is real; the critic conflates the fine-tuning step with the inference extension that builds upon it. [Moved to Nice-to-Have]
- **Criticism regarding missing hyperparameters (learning rate, batch size, epochs) in Sec. 4.1:** Standard reproducibility nitpick. Core architectural choices and pipeline details are sufficiently covered; full hyperparameter sweeps are typically relegated to appendices and do not affect the validity of the claimed contributions. [Removed]
- **Criticism regarding unreported attrition rates in data filtering:** While reporting filtering rates (code execution errors, OCR failures) is good practice, omitting it does not invalidate the quadratic scaling claim or the downstream results. [Removed]
- **Criticism regarding ChartQA noisy ground truth affecting SOTA claims:** The paper explicitly acknowledges ChartQA ground truth noise (Sec. 4.5, footnote on Fig. 4). This is a known dataset limitation, not an author error, and does not excuse the textual overclaim in the conclusion, but is not itself a standalone weakness of the paper. [Removed]

## Novel Insights
The paper effectively isolates the OCR-dependency bottleneck in current chart MLLMs and demonstrates that aligning visual features with raw JSON data during pre-training, coupled with text-only reasoning fine-tuning, mitigates this shortcut. However, the marginal gains of these specific multi-stage tweaks over standard instruction tuning suggest a broader community insight: for domain-specific MLLMs, scaling diverse, reasoning-heavy instruction data often yields the primary performance driver, while sophisticated pre-training alignment tricks contribute diminishing returns. Future work may benefit more from curating high-complexity reasoning datasets than from designing intricate multi-stage alignment pipelines.

## Suggestions
1. **Correct the SOTA claim in the Conclusion and Abstract:** Amend the phrasing to accurately reflect that CHOPINLLM achieves competitive or leading results on unannotated charts and the new benchmark, but acknowledge ChartAst's superior performance on ChartQA and Chart-to-Text. Scientific precision here is critical.
2. **Isolate methodological novelty from data scale:** Add an ablation row matching the exact cardinality of the `Chart QA + JSON Alignment` setting with a `Chart QA Only (matched size)` variant. If the novel steps still yield +3-5% gains even at equal data volume, the methodological claim stands much stronger.
3. **Add a preliminary error analysis table for the new benchmark:** Categorize failures on the ~10 most common chart types (e.g., "Axis misread," "Value inversion," "Reasoning failure on correct extraction"). This would contextualize the low absolute scores (Table 5) and demonstrate genuine geometric understanding.

## Score and Decision
**Calibration anchors:**
- **Low/Borderline Reject (~4.7):** `UU9Icwbhin` rejected due to misleading tables and overclaiming. This paper shares a structural overclaim in its conclusion, warranting a similar downward pull.
- **Borderline Reject (~5.5-5.8):** `WVBzN1HIFS` (Reject, ~5.5) and `dd2CABUZaw` (Reject, ~5.8) were scored in this range for weak contributions/novelty and marginal ablation insights despite solid empirical setups. This paper's novel components show similarly marginal gains (1-2%) over standard instruction tuning.
- **Borderline Accept (~6.0-6.75):** `yaQbTAD2JJ` (Accept, 6.0) and `o5TsWTUSeF` (Accept/Oral, ~6.75) feature stronger ablation isolation, clearer novelty, or more compelling empirical gaps. This paper falls below them due to the factual overclaim and the inability to disentangle training method novelty from data scaling.

The paper sits squarely in the borderline-reject range. The orthogonal pipeline and unannotated chart focus are valuable, and the new benchmark is a solid contribution. However, the explicit overclaiming of results in the conclusion and the marginal empirical justification for the specific novel training steps significantly weaken the scientific rigor. I position it at 5.0.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>