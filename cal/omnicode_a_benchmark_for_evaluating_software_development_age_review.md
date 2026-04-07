=== CALIBRATION EXAMPLE 22 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the contribution: a benchmark (OmniCode) for evaluating software development agents. The abstract clearly states the problem (narrow scope of existing benchmarks) and the solution (a diverse, multi-task, multi-language benchmark with synthetic generation to avoid leakage). The key claims—diverse tasks (4 categories), 1,794 tasks across 3 languages, manual validation, and challenging results for current agents—are all substantiated in the paper. The abstract is well-written and sets appropriate expectations.

### Introduction & Motivation
The introduction effectively motivates the need for a broader evaluation of coding agents beyond bug fixing. It clearly outlines the four task categories and positions OmniCode as a step toward automating the full software development lifecycle. The contributions are explicitly listed. The motivation is compelling and well-aligned with the community's direction.

### Benchmark Construction (Section 3)
This is the core of the paper. The process of collecting base instances (real GitHub pull requests) and generating synthetic tasks is described in reasonable detail. However, several important concerns arise:

1.  **Reproducibility & Methodology Gaps:** The prompts for generating bad patches, reviews, and the task-specific instructions for agents are stated to be in the appendix, but the appendix sections (B, etc.) are largely empty in the provided text. This is a critical omission for reproducibility. The exact evaluation prompts and LLM instructions for synthetic data generation must be provided.
2.  **Quality of Synthetic Data:** The methods for generating "bad patches" (via Agentless failures and LLM perturbation) and "code reviews" (via Gemini) are sensible but lack validation. The paper does not assess whether these synthetic artifacts (bad patches, reviews) are realistic, diverse, or of consistent quality. For instance, are LLM-generated reviews comparable to human code reviews in style and content? A qualitative analysis or validation (even on a sample) is needed to establish trust in the benchmark's construct validity.
3.  **Task Design Details:**
    *   **Test Generation:** The requirement that a generated test must pass on the gold patch *and fail on all bad patches* is a robust and valuable criterion. However, the number of bad patches per instance and their selection process are unclear. The paper mentions using a "subset" of instances for Java and C++ due to generation difficulties, which could introduce a selection bias. The benchmark's difficulty may be inconsistent across languages because of this.
    *   **Style Fixing:** The evaluation metric `score = max((resolved - new)/original, 0)` is introduced but not sufficiently motivated or explained in the main text. The relationship between this "score" and the percentages reported in results tables (e.g., 72.2%) is ambiguous. Are these scores averaged? Is there a threshold for "passing"? This needs clarification.
4.  **Complexity Metric:** The ad-hoc `complexity` metric (∆Files + Hunks + (AddedLines + RemovedLines)/10) is used to analyze task difficulty. The choice of weights (especially the divisor 10) is arbitrary and not justified. While the relative trends may hold, the metric itself lacks grounding.

### Experimental Setup & Results (Sections 3.3, 4, 5)
The experimental setup is appropriate, evaluating strong baseline agents (SWE-Agent, Aider) and a range of recent LLMs. The results are comprehensive, spanning tasks and languages.

1.  **Missing Baseline Comparison:** A significant omission is the lack of a direct comparison on the **bug-fixing** subset against established benchmarks like SWE-Bench or Multi-SWE-Bench. This would help calibrate the difficulty of OmniCode's instances and verify that the collected data is comparable. Without this, it's hard to know if lower performance on C++/Java is due to inherent language difficulty or dataset curation differences.
2.  **Statistical Rigor:** The paper reports point estimates (percentages) but provides no measures of uncertainty (e.g., standard errors, confidence intervals). Given the varying and sometimes modest instance counts (e.g., 44 C++ test generation instances), this is important for interpreting differences between models and agents.
3.  **Analysis Depth:** The analysis in Sections 5.1-5.5 is a strength of the paper. The findings—weak correlation between bug-fixing and style-fixing, the effectiveness of bad patches for rigorous test evaluation, the high complexity of unresolved instances—are insightful and well-presented. The comparison between SWE-Agent and Aider is useful.
4.  **Presentation of Results:** Tables 2 and 3 are referenced, but their formatting is broken in the provided text, making it difficult to parse the results. The authors must ensure all tables and figures are clearly presented in the final version.

### Related Work
The related work section adequately covers major coding benchmarks (HumanEval, SWE-Bench, Multi-SWE-Bench, SWT-Bench) and positions OmniCode's contributions (multiple task types, synthetic generation, multi-language focus). It could be slightly more comprehensive (e.g., mentioning APPS, CodeContests, or RepoBench for broader context), but it is sufficient.

### Limitations & Future Work
The limitations section appropriately acknowledges the narrow scope relative to real-world software engineering and outlines valuable future directions (more languages, security, migration). It could also explicitly mention the limitations discussed in this review: the lack of validation for synthetic data, the ad-hoc complexity metric, and the potential selection bias in bad patch generation.

### Writing & Clarity
Aside from the formatting artifacts (broken tables, empty appendix sections, misplaced figure references—which we attribute to the PDF parser), the paper is generally well-written and logically structured. Some technical details need clarification (as noted above), but the core ideas are communicated effectively.

### Overall Assessment
OmniCode is a timely and valuable contribution that addresses a real need in the community: a more holistic benchmark for evaluating coding agents. The design of multiple, diverse tasks (especially test generation with bad patches and code review response) is innovative and pushes evaluation beyond functional correctness. The empirical results reveal important weaknesses in current agents, particularly in test generation and on C++.

The primary weaknesses are methodological: the lack of validation for synthetic data generation, missing prompts/reproducibility details, an uncalibrated complexity metric, and insufficient statistical reporting. The absence of a direct bug-fixing comparison to prior benchmarks is also a notable omission.

Despite these issues, the core idea and initial execution are strong. The paper provides a solid foundation for a benchmark that could become a standard. For ICLR, which values novel, impactful ideas and rigorous evaluation, the paper is promising but requires significant revisions to address the methodological concerns and ensure reproducibility. **I recommend a weak accept, contingent on the authors satisfactorily addressing the major concerns around validation, reproducibility, and statistical reporting.**

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces OmniCode, a new benchmark designed to evaluate LLM-powered software development agents across a broader spectrum of real-world software engineering tasks beyond existing benchmarks like HumanEval and SWE-Bench. OmniCode comprises 1,794 tasks spanning three programming languages (Python, Java, C++) and four categories: bug fixing, test generation, code review response, and style fixing. A key contribution is a framework for synthetically generating diverse tasks (e.g., bad patches, reviews) from a curated set of real-world GitHub issues to mitigate data leakage. The paper presents an empirical evaluation of several state-of-the-art agents (e.g., SWE-Agent, Aider) and models, revealing significant performance gaps, particularly in test generation and C++ tasks.

### Strengths
1. **Addresses a Clear Gap in Evaluation:** The paper convincingly argues that existing benchmarks are too narrow (focused on bug-fixing or competition programming) and proposes a more holistic evaluation covering four distinct, realistic software engineering activities. The inclusion of tasks like test generation and code review response is timely and valuable for the field.
2.  **Novel Synthetic Generation Framework:** The methodology for creating tasks like test generation (using "bad patches") and review response (synthetically generating reviews for bad patches) is a creative and thoughtful approach to expanding benchmark scope from limited real-world data while actively combating data contamination. The analysis of bad patch and review categories (Appendix A) adds depth.
3.  **Rigorous and Multi-Faceted Evaluation:** The evaluation is extensive, covering multiple languages, state-of-the-art models (Gemini 2.5 Flash, DeepSeek-V3.1, etc.), and agent frameworks (SWE-Agent vs. Aider). The analysis goes beyond aggregate scores to examine correlations between tasks (Section 5.1, Appendix E), patch complexity (Section 5.4), and the critical impact of including bad patches for robust test evaluation (Section 5.5). The finding that agent performance is strong on Python style fixing but weak on Java/C++ provides nuanced insights.

### Weaknesses
1. **Limited Scale and Potential Selection Bias:** With 494 base instances (issues), the benchmark is smaller in scale compared to SWE-Bench (~2,300 issues) or Multi-SWE-Bench. While manual validation ensures quality, the process of selecting "sane and reliable" instances from existing benchmarks and hand-picking others (Section 3.1) may introduce selection bias, potentially making the dataset less representative of the full difficulty distribution in open-source software. The paper acknowledges this as a limitation but does not quantify its impact.
2. **Synthetic Generation's Fidelity to Reality:** Although the synthetic generation of bad patches and reviews is innovative, its realism is not thoroughly validated. The bad patches come from weaker agents or LLM perturbations, and reviews are generated by an LLM (Gemini 2.0 Flash). It remains unclear how well these synthetic artifacts mirror the distribution and quality of human-generated bad patches and code reviews in real development cycles. The paper would benefit from a small-scale human evaluation of this fidelity.
3. **Incomplete Baseline Comparison and Ablation:** The evaluation focuses on SWE-Agent and Aider but does not include a comparison with performance on the original tasks from the source benchmarks (e.g., SWE-Bench bug-fixing scores). This makes it hard to contextualize the reported performance (e.g., is 38.1% on Python bug-fixing good or expected?). Furthermore, there is no ablation study on the synthetic components (e.g., evaluating test generation with vs. without the requirement to fail on bad patches) to isolate their contribution to the difficulty.

### Novelty & Significance
**Novelty:** The paper's core novelty lies in its **multi-task, multi-language benchmark** constructed via a **synthetic generation framework** from real-world data. While individual tasks (bug-fixing, test generation) have been explored in isolation, combining them into a unified benchmark with a methodology to create interrelated synthetic tasks (bad patches → reviews) is a significant advance over prior work.
**Significance:** For the ICLR community, OmniCode is highly significant as it pushes the evaluation of coding agents towards more realistic, holistic software engineering capabilities. It provides a platform to diagnose agent weaknesses (e.g., in test generation or C++), spur development of more robust systems, and study task correlations. The synthetic framework also offers a blueprint for creating future benchmarks while mitigating data leakage concerns.

### Suggestions for Improvement
1. **Expand Dataset and Analysis of Representativeness:** Actively work to scale the number of base instances and provide a more systematic analysis of the selected issues' characteristics (e.g., complexity distribution) compared to the broader population of GitHub issues. This would strengthen claims about the benchmark's realism and challenge level.
2. **Validate Synthetic Artifacts:** Conduct a human evaluation to assess the realism of a sample of generated bad patches and code reviews. For instance, ask software engineers to categorize or judge the plausibility of these synthetic elements. This would add crucial credibility to the benchmark construction methodology.
3. **Strengthen Experimental Section:** (a) Include baseline results from the source benchmarks (e.g., SWE-Bench scores for the overlapping bug-fixing instances) to anchor OmniCode's performance numbers. (b) Perform an ablation study for the test generation task, showing the pass rate if evaluated only with the gold patch (Fail-to-Pass) versus the stricter criterion (Fail-to-Pass & Fail-to-Fail on bad patches). This would quantitatively demonstrate the importance of the proposed robust evaluation.
4. **Improve Clarity on Evaluation Metrics and Accessibility:** (a) The style-fixing score formula in Section 3.2.4 (`(resolved - new)/original`) could produce negative values; clarify how the `max(...,0)` is applied and discuss the interpretation of a score of 0. (b) Explicitly state in the main text (or a reproducibility statement) whether the benchmark code, data, and exact prompts will be released publicly. For ICLR, easy reproducibility is a major asset.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Lack of established baseline comparisons.** The paper only evaluates SWE-Agent and Aider. It must include state-of-the-art baselines specifically designed for tasks like test generation (e.g., CodeT, ChatUnitest) and program repair (e.g., SWE-Lancer, RepairBench). Without these, it's impossible to tell if the low scores reflect a hard benchmark or just underperforming agent frameworks.
2. **Missing prompt and configuration ablation.** The evaluation uses default settings for agents. The paper must ablate different prompting strategies, instructions, and in-context examples for each task type. Performance is highly sensitive to these choices, so the results as presented do not isolate the agent's capability from prompt engineering.
3. **Incomplete multi-language evaluation for all tasks.** The test generation and review-response tasks use a subset of instances for Java (77) and C++ (44), while bug-fixing uses the full set. This inconsistent coverage makes cross-language and cross-task comparisons misleading. The authors should either generate bad patches for all instances or explicitly account for this sampling bias.
4. **No experiment on task interdependence.** A core claim is evaluating "different aspects of software development." The paper should test if an agent trained or fine-tuned on one task (e.g., style-fixing) improves performance on another (e.g., bug-fixing) to validate that these are distinct but related capabilities, rather than just separate prompts on the same data.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of synthetic data quality and realism.** The bad patches and code reviews are generated by LLMs. The paper must analyze their quality: e.g., compare the distribution of errors in synthetic bad patches to real, rejected PRs; or have human evaluators judge the realism of generated reviews. Without this, the benchmark's validity is questionable.
2. **Failure mode analysis per task and language.** The paper reports aggregate scores but lacks a qualitative breakdown of *why* agents fail. For example, for C++ test generation, are failures due to compilation errors, misunderstanding complex types, or inability to write assertions? This diagnostic analysis is crucial for guiding future research.
3. **Statistical significance testing.** The paper makes comparative claims (e.g., "SWE-Agent consistently outperforms Aider") but provides no statistical tests (e.g., paired bootstrap tests). Given the relatively low instance counts per language/task, differences could be due to chance. This must be addressed for credible claims.
4. **Correlation analysis is superficial.** The paper notes correlations between task performances but doesn't investigate the underlying causes. For instance, why is style-fixing weakly correlated with bug-fixing? Is it because style errors are trivial, or because it requires a different skill? A deeper factor analysis is needed.

### Visualizations & Case Studies
1. **Side-by-side examples of successful vs. failed task instances.** For each task type, show a concrete example: the original code, the agent's output, the gold standard, and the evaluation outcome. This is especially critical for test generation and review-response to show what a "good" vs. "bad" generated test or patch looks like in this benchmark.
2. **Visualization of agent trajectories for complex tasks.** For a few representative C++ bug-fixing or test generation instances, show the sequence of commands (e.g., file edits, test runs) the agent took. This would reveal whether failures are due to flawed reasoning, getting lost in the repo, or tool misuse.
3. **Error heatmaps for style-fixing.** Instead of just aggregate scores, show which specific style rules (e.g., "avoid-magic-numbers") are most often fixed or introduced as new errors by each agent/model, visualized across files or projects. This would pinpoint specific weaknesses.

### Obvious Next Steps
1. **Human evaluation of generated artifacts.** At least a small-scale human assessment (e.g., by software engineers) of the generated tests, style fixes, and review responses is necessary to confirm that the automatic metrics (pass/fail on bad patches, linter scores) align with human judgment of quality and usefulness.
2. **Benchmark leakage analysis.** The authors mention avoiding data leakage but do not test it. They should report the performance of evaluated models on the training/validation split of the data they were trained on (e.g., CodeXGLUE, The Stack) to check if high performance is due to memorization.
3. **Standardized, public evaluation harness.** The paper should release not just the dataset but an easy-to-run evaluation script that reproduces the exact container setup, command execution, and scoring logic. Without this, the community cannot reliably compare new methods to the reported baselines, undermining the benchmark's utility.
4. **Analysis of computational cost and efficiency.** The paper is silent on the practical cost of running these agents (e.g., average number of LLM calls, total tokens, runtime per task). For a benchmark aiming at real-world utility, reporting these metrics alongside performance is essential for a complete picture.

# Final Consolidated Review
## Summary
OmniCode introduces a multi-task, multi-language benchmark for evaluating software development agents, extending beyond standard bug-fixing to include test generation, code review response, and style fixing. It presents a framework for synthetically generating diverse tasks (e.g., "bad patches" and LLM-generated reviews) from a curated set of real-world GitHub issues to mitigate data leakage. An evaluation of modern agents and models reveals significant performance gaps, particularly in test generation and on C++ tasks.

## Strengths
- **Addresses a Clear and Impactful Gap:** The benchmark moves beyond the narrow scope of existing benchmarks (e.g., SWE-Bench, HumanEval) by holistically evaluating four distinct, realistic software engineering activities. This is a timely and valuable contribution for steering research toward more capable, general-purpose coding agents.
- **Innovative Synthetic Generation Framework:** The methodology for creating challenging tasks like test generation (using "bad patches" to require discriminative tests) and review response from a limited set of base issues is creative. It provides a blueprint for expanding benchmark scope while actively combating data contamination concerns.
- **Revealing and Nuanced Empirical Analysis:** The evaluation is extensive, covering multiple languages, models, and agent frameworks. The analysis goes beyond aggregate scores to provide insights such as the weak correlation between bug-fixing and style-fixing performance, the critical role of bad patches in robust test evaluation, and the disproportionate difficulty of C++ tasks.

## Weaknesses
- **Insufficient Validation of Synthetic Data Fidelity:** The realism and quality of the LLM-generated "bad patches" and code reviews are not validated (e.g., via comparison to human artifacts or qualitative assessment). This is a core methodological gap that undermines confidence in the benchmark's construct validity. The paper states categories were analyzed via LLM prompting (Appendix A), but this does not establish that the synthetic data mirrors real-world distributions.
- **Methodological and Reproducibility Gaps:** Key details are missing, making reproduction difficult. The prompts for generating synthetic data and for task-specific agent evaluation are referenced as being in the appendix, but the provided content shows these sections are largely empty. Furthermore, the evaluation metric for style fixing (`score = max((resolved - new)/original, 0)`) and its aggregation into reported percentages are not clearly explained in the main text.
- **Ad-hoc Complexity Metric and Limited Statistical Reporting:** The paper uses an ungrounded, ad-hoc complexity metric (∆Files + Hunks + (AddedLines + RemovedLines)/10) to analyze difficulty. While the relative trends may hold, the metric's arbitrary weights are not justified. Additionally, results are presented as point estimates without measures of uncertainty (e.g., confidence intervals), which is important given the modest and variable instance counts per language/task (e.g., 44 for C++ test generation).

## Nice-to-Haves
- A direct comparison of bug-fixing performance on the overlapping instances with the source benchmarks (e.g., SWE-Bench) would help calibrate OmniCode's difficulty and contextualize the reported scores.
- An ablation study for the test generation task, showing performance with and without the requirement to fail on "bad patches," would quantify the contribution of this stricter evaluation criterion.
- Analysis of task interdependence (e.g., whether performance on one task predicts another) could provide deeper insights into the skills being measured.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism about missing prompts for agent evaluation:** The paper states these are in the appendix (Section B). The fact they are not visible in the provided text is assumed to be a parser/extraction issue, not an author omission. However, their absence in the submission is a genuine reproducibility concern, so it is addressed in the weaknesses.
- **Criticism about the lack of comparison to specialized baselines (e.g., CodeT for test generation):** OmniCode is an agent *benchmark*, not a new method. It is evaluated using general-purpose agent frameworks (SWE-Agent, Aider). Requiring comparisons to specialized, non-agent tools is outside its stated scope.
- **Criticism about formatting artifacts and broken tables:** These are attributed to the PDF parser in the review instructions and are not the authors' fault.
- **Strengths like "the paper is well-written" or "the topic is important":** These are generic and do not highlight what this specific paper does well that others do not.

## Novel Insights
The benchmark reveals that agent performance on style fixing is weakly correlated with performance on functional tasks like bug fixing (e.g., Pearson ~0.512), suggesting these require different capabilities. Furthermore, it demonstrates that evaluating test generation without "bad patches" dramatically overestimates performance (e.g., Qwen C++ pass rate would be 22.7% instead of 4.55%), highlighting that many generated tests capture superficial behaviors rather than core program semantics. This provides a crucial, more rigorous evaluation standard for automated testing.

## Suggestions
- **Add a validation study:** Perform a human evaluation or analysis to assess the realism of a sample of generated bad patches and code reviews (e.g., by asking software engineers to judge plausibility or by comparing error distributions to real rejected patches). This is essential for establishing benchmark validity.
- **Improve reproducibility:** Ensure all prompts for synthetic data generation and agent task instructions are included in a final appendix or released code.
- **Clarify metrics and report uncertainty:** Explicitly define how the style-fixing score is aggregated into the percentages shown in results tables. Consider reporting confidence intervals or standard errors for key results, especially where instance counts are lower.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Reject
