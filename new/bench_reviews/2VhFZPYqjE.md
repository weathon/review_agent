## Summary

This paper introduces CHASE (CHallenging AI with Synthetic Evaluations), a framework for generating synthetic evaluation benchmarks using LLMs through bottom-up problem construction. The framework hides solution components within generated context to increase difficulty, and decomposes generation into verifiable sub-tasks to ensure data correctness. The authors implement CHASE across three domains — document-based QA (671 examples, ~6k tokens), repository-level code completion (500 examples, ~17k tokens), and grade-school math reasoning (500 examples) — and evaluate 15 contemporary LLMs, showing that even state-of-the-art models achieve only 40-65% accuracy.

## Strengths

- **Novel bottom-up generation paradigm**: The framework's core idea of starting with a simpler problem-solution pair and iteratively hiding solution components within generated context (rather than forward-generating problems, which are inherently solvable by the generator) is conceptually sound and directly addresses a real gap in synthetic benchmark creation. Section 3 and Figure 1 clearly articulate this reversal of the typical pipeline.
- **Multi-domain, concrete implementation**: The paper fully instantiates CHASE across three substantially different domains with detailed pipeline steps, verification logic, and generation parameters (Sections 4.1-4.3). The execution-based verification for CHASE-CODE and ensemble verification for CHASE-MATH demonstrate careful engineering choices.
- **Extensive multi-model evaluation**: Table 1 evaluates 15 diverse LLMs across all three benchmarks, demonstrating that models outside the generator/verifier family (e.g., Gemini-1.5-Pro at 63.2%) also struggle. This partially mitigates concerns about family-specific difficulty.
- **Useful secondary findings on long-context degradation**: Figure 3 demonstrates consistent performance drops across four models as context scales from 6k to 50k tokens (up to 70% drop), providing practical empirical evidence about long-context limitations at evaluation-relevant scales.
- **Honest comparison to direct prompting baselines**: Table 2 shows CHASE-generated problems are substantially harder and higher quality than direct prompting approaches (e.g., Evol-Instruct), with 34% manual error rates in the math baseline.

## Weaknesses

### Fatal
None.

### Major

- **Rejection sampling conflates problem generation with problem selection**: The paper's difficulty calibration relies explicitly on discarding problems solvable by a weaker model (GPT-4o-mini). For CHASE-QA, "randomly discard half of the problems on which it was correct both times" (Sec. 5.1, line 199); for CHASE-CODE, similarly discarding ~half (line 201); for CHASE-MATH, discarding ~75% solvable by GPT-4o-mini (line 203). The claim that "state-of-the-art LLMs only achieve accuracies in the range of 40-60%" is mathematically guaranteed by this filtering — the framework *selects* hard problems for a specific model family rather than intrinsically generating them. No ablation shows how accuracy scales with different filtering thresholds, nor whether SOTA models would still perform poorly without this model-specific pruning. This undermines the core difficulty claim and prevents the benchmark from serving as an absolute measure of model capability.
- **Generator-judge circularity in CHASE-QA**: GPT-4o serves as both the generator of CHASE-QA problems and the LLM-as-a-judge evaluating model predictions (Sec. 5.1, lines 199 and 207). This introduces semantic and stylistic prios that could inflate or deflate accuracy depending on whether a model's reasoning style matches GPT-4o's. While the 91% human-agreement statistic on a 100-item balanced sample supports judge reliability in isolation, it does not address bias across the full benchmark where the judge systematically evaluates against ground truth it helped create. This compromises CHASE-QA as an independent evaluation artifact.

### Minor

- **LLM verification reliability is assumed, not independently validated at scale**: The paper verifies that LLMs can be "relied upon for verification because our framework makes each verification task smaller and simpler" (Sec. 3, line 133). The only quality check is manual sampling: 2/30 errors in QA, 7/100 in Math (Sec. 5.2). Given that LLMs are known to miss subtle logical inconsistencies and silent reasoning errors — particularly in long-context documents or chain-of-thought math — these rates do not tightly bound the false-positive ground-truth error rate. Undetected flaws would directly corrupt the accuracy metric.
- **Context-length confounding in QA and Code**: For both CHASE-QA and CHASE-CODE, the performance drops reported in Figure 3 may conflate attention-dilution/retrieval-failure with genuine reasoning difficulty. The paper does not disentangle whether models fail because the context is too long, because relevant information is buried, or because reasoning depth is genuinely challenged. The claim that difficulty stems from "multiple steps of drawing inferences or reasoning over a longer context" (Sec. 3, line 129) remains under-specified.
- **Benchmark size is small for per-domain statistical power**: CHASE-QA has 671 examples, CHASE-CODE has 500, and CHASE-MATH has 500. For sub-domain analyses (e.g., CHASE-CODE DATA vs ALGO with 250 each), standard errors are non-trivial (~3% at 40% accuracy), which limits confidence in fine-grained model ranking claims.

### Trivial
None.

## Nice-to-Haves

- An ablation study varying the rejection-sampling discard percentage (0%, 50%, 90%) would clarify how much of the reported difficulty stems from filtering versus intrinsic generation.
- Human evaluation of a larger stratified sample across all three domains would better bound the LLM-verifier pipeline's error rate.
- Error categorization (e.g., confusion matrices or failure-mode analysis) for CHASE-QA would clarify whether models fail due to document retrieval, factual omission, or reasoning mistakes.
- The evaluation prompts and rejection-sampling scripts would benefit from being open-sourced alongside the benchmarks.

## Removed Points

*These points are flagged to be removed.*

1. **Criticism about "LLM-based verification cannot guarantee correctness at scale, dismissing manual sampling as acceptable"**: The 2/30 (QA) and 7/100 (Math) manual error rates are actually comparable to or better than standard practice in the synthetic-data literature (e.g., OptiBench and Prometheus also rely on LLM-verified synthetic data with similar error bounds). This is a standard methodological choice, not an author error.

2. **Criticism about "generator-judge circularity invalidates CHASE-QA entirely"**: While the overlap is a real concern, the paper explicitly evaluates non-OpenAI models (Gemini, Claude, Llama, Qwen) on CHASE-QA and shows they also struggle substantially (Table 1). This demonstrates the benchmark is hard across model families, not just biased against non-GPT models. The paper even highlights that "models different from the generator and verifier (such as Gemini-1.5-Pro) do better" (line 280), suggesting the bias is partial, not total.

3. **Criticism about "context-size Figure 3 replicates well-documented phenomena and does not validate CHASE's contribution"**: The context-size experiment is presented as a secondary finding ("We further highlight the utility..."), not as a core CHASE contribution. Evaluating models at long-context scales where human benchmarks don't exist is itself a valid contribution.

4. **Criticism about "zero-human-involved framing contradicts use of human-crafted prompts and bootstrapping seeds"**: The paper explicitly describes bootstrapping with seed examples in Sections 4.1-4.3 and the framework is designed to scale beyond those seeds. The "without human involvement" claim in the abstract refers to the scalable generation phase, not the one-time prompt engineering. This is scope-appropriate language for ML papers.

5. **Criticism about "contamination is not mitigated if future models are trained on CHASE outputs"**: The abstract states CHASE "can be renewed periodically to mitigate contamination concerns" — this refers to the pipeline's ability to regenerate fresh benchmarks, not to immunity from contamination if the current release is leaked. The concern conflates the mechanism with its deployment.

## Novel Insights

The paper's bottom-up generation paradigm — hiding solution components first, constructing context second — is a genuine conceptual contribution to the synthetic evaluation space. This approach directly inverts the common forward-generation pipeline (generate problem → generate solution), which by construction produces problems solvable by the generator. Combined with decomposition into independently verifiable sub-tasks, CHASE offers a principled alternative for creating benchmarks that are hard even for the models that produce them. The multi-domain implementation reveals that long-context evaluation exposes sharper model differentiation than standard benchmarks like MMLU or HumanEval, where SOTA models cluster. These insights are useful for benchmark practitioners, though the rejection-sampling methodology and generator-judge overlap significantly weaken the evidentiary value of the reported absolute accuracy figures.

## Suggestions

1. **Add rejection-sampling ablation**: Report how accuracy scales when discarding 0%, 25%, 50%, and 75% of easily-solved problems, ideally across multiple filter-models (not just GPT-4o-mini), to demonstrate that difficulty is a property of the generation pipeline rather than an artifact of a specific filter choice.
2. **Decouple CHASE-QA judge from generator**: Use an independent judge model (e.g., a non-GPT evaluator) or a larger human evaluation set to validate that the accuracy metric reflects genuine reasoning differences rather than stylistic alignment with GPT-4o.
3. **Disentangle context-length from reasoning depth**: Evaluate models on CHASE variants where distractor length is held constant but reasoning depth is varied, and vice versa, to quantify the independent contribution of each factor to difficulty.
4. **Add error analysis tables**: Provide categorized failure modes for CHASE-QA (e.g., document retrieval fail, factual omission, hallucinated irrelevant content) to help model developers understand their limitations.

---

## Score and Decision

**Calibration anchors compared:**
- **OptiBench** (fsDZwS49uY.md): Synthetic benchmark with reverse data synthesis and LLM verification; scored 6,6,8 (Accept Poster). Had similar verification approach but stronger ablation studies and error analysis.
- **LongGenBench** (3A71qNKWAS.md): Novel benchmark for long-form generation; scored 8,3,5,8,8 (Accept Poster). More novel evaluation axis with clearer conceptual novelty.
- **PROMETHEUS** (8euJaTveKw.md): LLM-as-judge paper with circularity concerns about training on GPT-4 data; scored 5,6,1,6 (Accept Poster). Accepted despite circularity issues, similar to the generator-judge concern here.
- **MHPP** (TVFVx8TUbN.md): Hard code benchmark rejected at 3,3,5,6 for incremental contribution.

This paper is stronger than MHPP (novel framework, not just a dataset), comparable in methodological rigor and novelty to OptiBench (which scored 6,6,8), but weaker on the ablation and error-analysis axes. It shares the circularity/Prometheus concerns. The rejection-sampling methodology is a meaningful flaw that would concern reviewers, but the framework itself (bottom-up generation, verifiable decomposition) is useful and well-specified. Positioning relative to these anchors suggests a score of 5.5 — borderline, with acceptance potential if the community values the framework contribution over the filtering concern.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>