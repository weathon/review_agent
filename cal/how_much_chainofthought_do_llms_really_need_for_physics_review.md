=== CALIBRATION EXAMPLE 41 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and directly reflects the paper's core question. The abstract succinctly outlines the problem, method, key findings, and implications. Claims are specific (e.g., "accuracy remains stable under heavy deletions (40–60%)") and appear to be supported by the experiments described later. The connection to AI-for-Science is well-motivated.

### Introduction & Motivation
The introduction effectively frames the "faithfulness gap" in CoT reasoning and justifies physics as a stringent, structured testbed. The three stated contributions are clear and map well to the subsequent sections. However, the distinction between "faithfulness" and mere "necessity" of the CoT text could be sharper from the outset. The claim that physics is "methodologically revealing" for broader AI-for-Science is plausible but not deeply argued; a stronger connection to general principles of structured reasoning (equations, units, symbolic manipulation) would strengthen this point.

### Problem Setup (Section 2)
- **Tasks and Datasets:** The description of the three benchmarks is adequate, but the asserted difficulty ordering (UG Physics easiest, PhyBench hardest) is not justified with evidence (e.g., baseline performance, human solve rates). This weakens interpretations of cross-dataset comparisons.
- **Models:** The choice of open-source models to enable token-level intervention is well-justified. However, the models differ significantly in size, architecture (e.g., MoE vs. dense), and training pipelines. While the paper aims to show general patterns, the lack of controlled comparison (e.g., similar-scale models) makes it difficult to disentangle model-specific properties from universal behaviors.
- **Calibrating Chain-of-Thought:** The three prompting styles (Full/Medium/Low) are a sensible operationalization. The calibration for sample size (5 prompts) is reasonable, but the description is brief; more detail on the bootstrapping procedure and the "relative error bar below 10%" criterion would aid reproducibility.
- **Metrics:** The use of an LLM (Claude-4 Sonnet) as a primary judge for correctness is a potential weakness. No validation of the judge's accuracy or agreement with human experts is provided, which is critical for a scientific domain where precise correctness matters. The information overlap metrics (Jaccard, Manhattan distance on BoW) are simplistic for physics content; they may conflate superficial lexical overlap with meaningful recovery of equations and logical steps. Domain-specific matching (e.g., equation equivalence) is mentioned but not detailed, leaving the rigor of the faithfulness analysis in question.

### Experimental Results (Section 3)
- **Prompting and Calibration (3.1):** The finding that more explicit reasoning improves scores is expected but provides a necessary baseline. The reliance on an unvalidated LLM judge remains a concern.
- **CoT Deletion Sweeps (3.2):** This is the core methodological contribution. Several significant issues arise:
    1. **Methodological Ambiguity:** The exact procedure for "intercepting CoT mid-generation" and "removing tokens before decoding" is not technically detailed. Does the model generate a full CoT, then the experimenter deletes tokens from that text, and then the model is prompted to continue from the truncated context? Or is generation paused, tokens deleted from the internal context window, and then generation resumed? The former seems more likely but is not explicitly described. This lack of clarity hinders reproducibility.
    2. **Information Persistence:** A critical conceptual flaw is that deleting tokens from the *generated text* does not necessarily erase the information from the model's *internal state*. The model may have already computed and used the reasoning represented by those tokens. Therefore, robustness to deletion does not conclusively demonstrate that the CoT was "not needed" or "bypassed"; it may simply show that the information was already encoded in the model's activations. The paper's claims about "shallow reliance" and "opportunistic" use are overstated given this confound.
    3. **Presentation of Results:** Figures are referenced (e.g., Figures 3, 4, 5, 6) but the text often describes trends without providing key quantitative results (e.g., baseline accuracy scores, magnitude of length increase). The narrative is sometimes hard to follow without the visual aids, though the figures are included in the text.
    4. **Cramming Evidence:** The increase in final answer length is presented as evidence of "cramming." While plausible, increased verbosity could also stem from uncertainty or attempts to hedge. Qualitative examples showing the regeneration of specific deleted equations would strengthen this claim considerably.

### Analysis and Discussion (Section 4)
- **Cramming and Overlap:** The analysis correlates increased answer length with deletion and shows lexical overlap increases. However, the BoW overlap metrics are insufficient to demonstrate faithful *reasoning* recovery. The discussion acknowledges variability and "opportunistic" reconstruction but does not adequately address the fundamental limitation that token deletion may not remove the underlying computed information.
- **Implications for Faithfulness:** The discussion correctly notes that CoT appears both informative and redundant. However, the leap from observing compensatory lengthening and lexical overlap to questioning the "faithfulness of CoT traces as evidence of underlying reasoning" is too strong given the methodological confound. The paper would benefit from a more nuanced definition of "faithfulness" in the context of its intervention (perhaps "textual necessity" rather than "computational dependence").
- **Practical Implications:** Suggestions like "early stopping of CoT generation" are interesting but speculative without cost-accuracy trade-off analysis.

### Limitations
The listed limitations are appropriate (scope, observational nature, need for mechanistic studies). However, the most critical limitation—that token deletion does not equate to erasing internal computations—is **not mentioned**. This omission significantly undermines the paper's central conclusions.

### Writing & Clarity
The writing is generally clear and well-structured. Some sections (e.g., the deletion procedure) lack precise technical detail, which impacts clarity and reproducibility. The flow from results to analysis is logical.

### Overall Assessment
The paper asks a timely and important question about the role of CoT in scientific reasoning and introduces a creative deletion framework for probing it. The empirical work is substantial, covering multiple models and benchmarks. However, a fundamental methodological confound—deleting output tokens does not erase internal state information—severely weakens the interpretation that models "do not really need" the CoT. The primary claim about shallow reliance and limited faithfulness is therefore not sufficiently supported. Additional concerns include the unvalidated LLM-based evaluation and simplistic overlap metrics. For ICLR, where methodological rigor and well-supported claims are paramount, these issues are significant. The paper's contribution is potentially valuable but requires major revisions to address the core confound, strengthen the evaluation, and temper the conclusions. In its current form, it is likely below the acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates the faithfulness of chain-of-thought (CoT) reasoning in large language models (LLMs) applied to physics problem-solving. It introduces a systematic deletion framework that intercepts and removes tokens from CoT traces mid-generation to probe whether models genuinely depend on their own reasoning. The main findings are that models maintain accuracy under substantial deletions (40-60%) by "cramming" (reconstructing steps in final answers), but this recovery is shallow and opportunistic, exposing a gap between CoT generation and genuine reasoning dependence.

### Strengths
1. **Methodological Innovation**: The paper introduces a novel and systematic deletion framework (end, random, and physics-aware) to actively probe CoT dependence. This is a concrete, interventionist approach that moves beyond correlational studies of faithfulness.
2. **Rigorous and Multi-faceted Evaluation**: The study is thorough, evaluating three diverse open-source models (Phi-4, Qwen-A3B, Magistral) on three physics benchmarks of varying difficulty. It employs multiple metrics: accuracy (judged by Claude-4), answer length, and information overlap (Jaccard, Manhattan distance), providing a comprehensive characterization.
3. **Strategic Domain Choice**: Physics is an excellent testbed due to its structured nature (equations, units, precise terminology), which allows for clear annotation and quantification of reasoning elements. This strengthens the claim that the findings are critical for "AI-for-Science".
4. **Clear Empirical Patterns**: The results consistently show the "cramming" phenomenon (increasing answer length under deletion) and quantify the redundancy of CoT traces. The finding that accuracy is robust to moderate deletions before collapsing is a robust and important insight.

### Weaknesses
1. **Limited Generalizability**: The empirical scope is confined to three models and one scientific domain (physics). While the authors argue physics is representative, the core phenomena (cramming, overlap) may not generalize to other reasoning domains (e.g., commonsense, social reasoning) without further validation. The claim that patterns "may generalize" is speculative.
2. **Mechanistic Understanding is Superficial**: The study is observational, analyzing model outputs. It does not probe *why* cramming occurs (e.g., via analysis of internal representations, attention patterns, or decoding dynamics). The term "cramming" describes a behavior but does not explain its underlying mechanism, limiting insights for model design.
3. **Overlap Metrics are Surface-Level**: The use of Bag-of-Words metrics (Jaccard, Manhattan) to measure "information recovery" is quite lexical and may not capture semantic faithfulness or logical consistency. A deleted equation reappearing in the final answer might be correct but used in a different logical step, which these metrics would miss.
4. **Missing Baseline Comparison**: The paper does not quantitatively compare its deletion framework against existing metrics for CoT faithfulness (e.g., from Lanham et al., 2023 or Lyu et al., 2023). Positioning its methodological contribution relative to this established literature would strengthen the paper.
5. **Judgment Model Reliance**: Using Claude-4 as a sole judge for accuracy scoring introduces potential biases and dependencies on another proprietary model's capabilities. A more robust evaluation would include human evaluation or ground-truth checking for at least a subset.

### Novelty & Significance
**Novelty**: The core idea of *systematic mid-generation deletion* to probe reasoning dependence is novel and clever. While the question of CoT faithfulness is actively studied, applying this interventionist lens to structured scientific reasoning is a fresh contribution. The empirical characterization of "cramming" is also new.
**Significance**: The work is highly significant for the AI-for-Science community and the broader reasoning evaluation field. It compellingly argues that standard accuracy benchmarks are insufficient and that *faithfulness* must be a core evaluation criterion. The findings caution against treating CoT as a transparent window into model reasoning, with direct implications for interpretability and the design of reliable scientific AI systems. It meets ICLR's bar for a solid empirical study with clear implications.

### Suggestions for Improvement
1. **Expand Scope for Generalizability**: Include a small-scale experiment on a non-physics reasoning dataset (e.g., a math or logical reasoning benchmark) to test whether the cramming and robustness phenomena hold. This would strengthen claims about the broader implications.
2. **Deepen the Analysis**: Conduct a targeted analysis to better understand the "cramming" mechanism. For example, analyze whether the increase in answer length is due to regurgitation of training templates, expansion of remaining context, or a shift in decoding dynamics. Even a hypothesis-driven discussion would add depth.
3. **Refine Overlap and Faithfulness Metrics**: Supplement lexical overlap with metrics that assess the *functional role* of recovered content (e.g., whether a recovered equation is used in the correct step of the derivation). Consider using task-specific structured parsing or entailment checks.
4. **Compare to Existing Faithfulness Metrics**: Explicitly compare the diagnostic power of the deletion framework against established faithfulness scores (e.g., based on answer prediction from intermediate steps). Discuss what new insights the deletion method provides that these others do not.
5. **Address Judgment Reliability**: To mitigate concerns about using Claude-4, report inter-annotator agreement with human experts on a subset of problems, or use a simple exact-match/equation-check metric for the subset of problems where it's feasible. Acknowledge this as a limitation more directly.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Include a no-CoT (direct answer) baseline.** Without comparing performance when the model is prevented from generating any reasoning trace, it is impossible to quantify the actual utility of CoT or to calibrate the severity of the deletion effects.
2. **Test on state-of-the-art closed-source models (e.g., GPT-4, Claude).** The claims about CoT faithfulness in LLMs are limited by only evaluating open-source models, which may not reflect the capabilities of the most advanced reasoning systems.
3. **Ablate CoT content by type (e.g., equations vs. text).** The paper deletes tokens by position or physics annotation but does not systematically vary the semantic role of deleted content to identify which reasoning components are critical for final answer generation.
4. **Experiment with erroneous or misleading CoT.** To test if models truly rely on the reasoning provided, the CoT should be manipulated (e.g., by inserting incorrect steps) before deletion to see if the model propagates errors or ignores them.

### Deeper Analysis Needed (top 3-5 only)
1. **Conduct statistical significance testing for key claims.** The paper reports trends (e.g., accuracy drop at 40% deletion) with error bars but lacks statistical tests (e.g., pairwise comparisons) to confirm these thresholds are meaningful and not due to chance.
2. **Correlate cramming (answer length increase) with answer correctness.** It is unclear if longer final answers actually help recover accuracy or are merely verbose and incorrect. Analyzing the relationship between length change and score change per instance is necessary.
3. **Analyze the predictive power of overlap metrics for faithfulness.** The paper computes Jaccard similarity and Manhattan distance but does not assess whether higher overlap with deleted content leads to more correct answers, which is central to the claim of shallow recovery.
4. **Break down results by problem type and difficulty within benchmarks.** The aggregate results may mask variability; analyzing performance by problem category (e.g., conceptual vs. calculation) would show whether CoT dependence is task-dependent.

### Visualizations & Case Studies
1. **Show side-by-side examples of original CoT, deleted CoT, and final answers.** Concrete instances are needed to illustrate what "cramming" looks like—how deleted equations or facts reappear (or don't) in the final output.
2. **Visualize token-level deletion patterns for each strategy.** Highlighting which specific tokens (e.g., equations, units, explanations) are removed in physics-aware vs. random deletions would clarify the intervention and its effects.
3. **Present case studies of successful vs. failed reconstructions.** Select problems where the model maintains accuracy after deletion and where it fails, then analyze the differences in CoT structure and final answer to identify failure modes.

### Obvious Next Steps
1. **Perform statistical testing on the reported deletion thresholds (e.g., 40%, 60%).** This is a basic requirement for making credible claims about robustness boundaries.
2. **Include a direct answer (no CoT) condition as a baseline.** This should have been a fundamental comparison point in the initial performance evaluation.
3. **Analyze the relationship between CoT length and deletion robustness.** The effect of deletions may depend on the initial length and redundancy of the CoT; this analysis is straightforward and informative.
4. **Test the deletion framework on a non-physics reasoning benchmark (e.g., math).** To argue for general applicability, a minimal extension to another structured domain is necessary.

# Final Consolidated Review
## Summary
This paper introduces a systematic deletion framework to probe chain-of-thought (CoT) reasoning dependence in LLMs applied to physics problem-solving. By intercepting and deleting tokens from CoT traces mid-generation, the authors show that models maintain accuracy under moderate deletions (40–60%) and compensate by producing longer final answers ("cramming"). Lexical overlap analysis suggests shallow, opportunistic recovery of deleted content, indicating that CoT traces are informative but not strictly necessary, raising concerns about faithfulness in scientific domains.

## Strengths
- **Novel and interventionist methodology:** The deletion framework (end, random, physics-aware) provides an active, controlled probe of CoT dependence, moving beyond correlational studies of faithfulness.
- **Comprehensive empirical evaluation:** The study rigorously tests three diverse open-source models across three physics benchmarks, employing multiple metrics (accuracy, answer length, lexical overlap) to characterize behavior consistently.
- **Strategic domain choice:** Physics, with its structured equations, units, and terminology, enables precise annotation and quantification of reasoning elements, making it a stringent and relevant testbed for AI-for-Science.

## Weaknesses
- **Internal state confound:** Deleting tokens from the generated CoT does not erase the corresponding information from the model's internal activations. Robustness to deletion therefore does not conclusively demonstrate that the CoT was computationally unnecessary; it may simply indicate the information was already encoded and used. This fundamental limitation undermines stronger claims about "shallow reliance" and "bypassing" CoT.
- **Lack of a direct-answer (no CoT) baseline:** The paper does not include a condition where the model is prevented from generating any CoT, making it impossible to quantify the absolute utility of CoT or to calibrate the severity of deletion effects relative to a zero-reasoning baseline.
- **Over-simplified overlap metrics:** The use of Bag-of-Words metrics (Jaccard, Manhattan distance) captures only lexical overlap, not semantic or logical faithfulness. A deleted equation might reappear but be used in an incorrect step, which these metrics would not detect, weakening the analysis of "recovery."
- **Unvalidated LLM-based evaluation:** The primary accuracy metric relies solely on Claude-4 Sonnet without validation against human experts or ground-truth checking for a scientific domain where precise correctness is critical, introducing potential bias.

## Nice-to-Haves
- **Statistical significance testing:** Formal tests for the claimed deletion thresholds (e.g., 40%, 60%) would strengthen the evidence for these boundaries.
- **Correlation between cramming and correctness:** Analyzing whether increased answer length actually leads to more correct answers would clarify if cramming is beneficial or merely verbose.
- **Breakdown by problem type:** Analyzing results by problem category (e.g., conceptual vs. calculation) within benchmarks could reveal task-dependent variations in CoT dependence.
- **Case studies:** Concrete examples of original CoT, deleted CoT, and final answers would vividly illustrate the cramming phenomenon.

## Novel Insights
The paper demonstrates that models can maintain accuracy despite significant CoT deletions by compensatory lengthening of final answers ("cramming"), with lexical overlap increasing as more content is removed. This suggests CoT traces are partially redundant and that models may opportunistically reconstruct surface content rather than faithfully relying on the intermediate reasoning. These findings challenge the assumption that CoT provides a transparent window into model reasoning, especially in structured domains like physics.

## Suggestions
- **Address the internal state confound:** Revise the interpretation to acknowledge that token deletion does not erase internal computations, and temper claims about CoT being unnecessary. Consider discussing this limitation explicitly.
- **Include a direct-answer baseline:** Add a condition where the model is prompted to give an answer without any CoT to establish baseline performance and better contextualize deletion effects.
- **Enhance overlap metrics:** Supplement lexical metrics with domain-specific measures, such as checking equation equivalence or logical step consistency, to better assess faithful recovery.
- **Validate the judge model:** Provide human evaluation on a subset of problems to verify Claude-4's scoring reliability, or use exact-match metrics for problems with clear answers.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
