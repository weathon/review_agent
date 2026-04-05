=== CALIBRATION EXAMPLE 20 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is directionally accurate, but somewhat underspecified about the actual contribution. The paper is not proposing a new tool-calling architecture or training method; it is comparing two prompting/output-formats for generation: schema-constrained vs. template-based. “Improving Tool Calling Accuracy” reflects the goal, but “template-based prompting for tool calling” would be more precise.
- The abstract states the problem and the general method, but it is extremely thin on the key experimental findings. It says the method “improves F1 scores ... on most tests,” which is true only in a qualified sense, since Table 2 and Table 3 show several losses, including statistically significant ones for GPT-5 and DeepSeek on some datasets.
- The abstract does not mention that the gains are model-dependent and sometimes reverse for reasoning-oriented models, which is one of the paper’s central findings. That omission makes the takeaway feel stronger than the evidence supports.

### Introduction & Motivation
- The motivation is plausible: tool calling is usually evaluated with rigid schema outputs, while LLMs are pretrained primarily on natural language. That gap is worth investigating.
- However, the introduction does not clearly establish why this is a new research question beyond prompting-format engineering. The related work gap is stated broadly, but the paper does not convincingly show prior work has not already explored natural-language tool-call formats or constrained decoding alternatives.
- The contributions are somewhat overstated. The paper presents an empirical comparison across datasets and models, but it does not introduce a new learning algorithm or a new benchmark. In ICLR terms, this is potentially publishable if the empirical insight is strong and general, but the introduction currently makes it sound more foundational than it is.
- The claim that this approach is broadly beneficial is underqualified early on, especially given the later evidence of mixed results for GPT-5 and DeepSeek-Coder.

### Method / Approach
- The method description is understandable at a high level, but not yet fully reproducible. The core comparison is between schema-constrained generation and a template-based natural-language output that is later regex-parsed into the same schema.
- The key unresolved methodological issue is that the two prompting styles may differ in more than just surface form. A natural-language template can provide different inductive biases, different tokenization properties, and potentially different error tolerance. The paper treats this as a format comparison, but does not isolate which aspects matter.
- The parsing step is important but underspecified. Regular expressions are mentioned, but the paper does not explain how robustly outputs are mapped back to structured calls, what happens when the template is partially matched, or how malformed outputs are handled before scoring.
- The evaluation of parameter values using embedding cosine similarity above 0.9 is a major methodological choice that needs much more justification. This threshold is arbitrary without sensitivity analysis, and it could materially affect parameter F1. The paper does not show whether results are robust to the threshold or to the embedding model used.
- Fine-tuning Mistral and DeepSeek for five epochs with early stopping is mentioned, but the training details are insufficient: learning rates, batch sizes, validation splits, prompt templates, and whether both methods were trained/evaluated under exactly matched conditions are missing. For ICLR, this would be a reproducibility concern.
- The assumption that template-based generation is “closer to natural language” is plausible but not operationalized. The paper later hypothesizes this, but the method section does not define how to test or measure it.
- There are important edge cases not discussed: multi-tool calls, argument ordering, optional/omitted parameters, nested structures, and conversation-state dependence. Since ToolACE and When2Call include complex cases, the paper’s simplification to first two turns or one-request formatting may limit the claim substantially.

### Experiments & Results
- The experiments do test the paper’s main claim: whether template-based prompting improves tool-call accuracy relative to schema-constrained prompting across models and datasets.
- The choice of datasets is reasonable for tool calling, but the experimental setup mixes very different regimes: API-Bank L1/L2, ToolACE, and When2Call are not directly comparable, and the paper sometimes draws broad conclusions from them as if they were.
- A major concern is the consistency of the evaluation protocol across models. GPT-4o and GPT-5 use prompts “provided directly in the datasets,” whereas Mistral and DeepSeek are fine-tuned. This means the comparison is not purely about prompting format; it also mixes zero-shot/proprietary models with fine-tuned open-source models. That makes cross-model conclusions harder to interpret.
- The paper reports macro-F1 and p-values from paired permutation tests, which is good. However, the results tables do not include confidence intervals or effect-size estimates beyond raw deltas, and some p-values are reported as 0.00 rather than to a meaningful precision.
- The biggest missing ablation is a control that separates “template wording” from “schema constraint removal.” For example, is the gain due to natural-language phrasing, to easier token prediction, or simply to a better prompt? Without alternate templates or multiple template styles, the causal claim is weak.
- Another important missing analysis is robustness to prompt sensitivity. Since the method is prompt-format dependent, ICLR reviewers would expect multiple templates or prompt variants, especially because the paper makes a general claim about template-based generation.
- The results are mixed rather than uniformly positive. Table 2 and Table 3 show that GPT-5 loses on API-Bank L1 and When2Call for both tool-name and parameter F1, and DeepSeek also has several negative or near-zero deltas. The paper does acknowledge this later, but the main narrative still overemphasizes gains.
- The qualitative error analysis is helpful, especially Tables 4–7. It gives concrete failure modes such as schema violations, incorrect tool names, and reasoning failures. However, these analyses are limited to selected datasets/models where the method looked favorable. A balanced error analysis would also examine the negative cases, especially GPT-5 and DeepSeek on datasets where template-based generation underperforms.
- The reasoning-effort study in Section 4.5 is interesting, but the figure is garbled in the extracted text and, more importantly, the analysis remains descriptive. It suggests instability with template-based patterns under higher reasoning effort, but does not establish why. This is important because it directly complicates the paper’s core hypothesis.
- Table 1 is useful for dataset scale, but “Expected tool calling” and “Expected parameter” need clearer interpretation. It is not obvious how these counts relate to instances when ToolACE has multiple calls per instance and When2Call includes no-tool cases.
- For ICLR standards, the experimental evidence is promising but not yet fully convincing as a general principle. The results look more like a useful empirical observation than a broadly established method.

### Writing & Clarity
- The paper is mostly understandable, but there are several places where clarity affects the contribution itself rather than just presentation.
- Section 2 would benefit from a clearer statement of what exactly changes between the two conditions and what remains constant. Right now, readers must infer whether the only difference is the output format or whether the prompt instructions also materially differ.
- Section 4’s categorization of errors is useful, but the definitions overlap somewhat in spirit. For example, “Reasoning Failure” vs. “Incorrect Tool Name” could be hard to separate in borderline cases. The paper does not explain adjudication rules.
- Section 4.5 is difficult to interpret from the extracted figure, and even setting parser artifacts aside, the discussion would be stronger with a concise numeric summary of the reasoning-effort trends.
- The tables are informative overall, especially Tables 2–8, but the analysis would be clearer if the authors explicitly linked each table to a concrete claim rather than narrating results piecemeal.
- The paper’s central message is not always stable: sometimes it argues template-based generation helps because it is more natural language-like; elsewhere it says improvements come from reduced schema violations; elsewhere it says reasoning models may not align with template patterns. These are plausible but not yet integrated into a coherent explanation.

### Limitations & Broader Impact
- The limitations section is better than many papers’ and does acknowledge some important constraints: limited models/datasets, one prompt structure, and the need for deeper mechanistic understanding.
- That said, it misses a few key limitations that matter for the paper’s claim:
  - The evaluation of parameter values via embedding similarity is potentially consequential and should be treated as a limitation or validated more carefully.
  - The method may be brittle to prompt wording and parsing conventions, but this sensitivity is not framed as a limitation.
  - The study does not assess actual end-to-end tool execution success; it measures structured output accuracy, which is necessary but not sufficient for real tool use.
- The broader impact discussion is minimal. For a tool-calling paper, the main broader impact is practical rather than societal: this could improve agent reliability, but it could also encourage overconfidence in tool-call generation without testing downstream execution correctness.
- There is no discussion of risks from malformed tool calls in deployment, although that is a natural concern in this domain. The paper does not need an extensive societal-impact section, but a brief note on operational failure modes would be appropriate.

### Overall Assessment
This paper studies a genuinely relevant question for tool-calling in LLMs: whether natural-language template outputs can outperform rigid schema-constrained outputs. The empirical results are interesting and there are clear gains in several settings, especially for Mistral and some GPT-4o/GPT-5 ToolACE cases. However, at ICLR’s bar, the work currently feels more like a well-motivated prompting comparison than a fully established method. The main concerns are methodological: the evaluation mixes very different model regimes, the template-vs-schema comparison is not isolated enough, parameter scoring relies on an arbitrary similarity threshold without robustness checks, and the results are meaningfully mixed rather than uniformly positive. I think the contribution is promising and worth discussing, but in its current form it does not yet make a sufficiently general or mechanistically grounded case for template-based generation as a broadly superior tool-calling approach.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies whether LLM tool-calling performance can be improved by replacing schema-constrained generation (e.g., JSON-like outputs) with template-based natural language generation, followed by regex parsing into the same tool schema. Across API-Bank, ToolACE, and When2Call, and with four models (GPT-4o, GPT-5, Mistral-7B, DeepSeek-Coder), the authors report that template-based prompting improves tool-name and parameter F1 in many cases, though gains are inconsistent and some statistically significant regressions occur, especially for GPT-5 and code-centric models.

### Strengths
1. **Simple, testable idea with practical relevance.**  
   The paper targets a concrete bottleneck in tool calling: brittle schema-constrained output. The proposed template-based alternative is easy to understand and could be adopted without changing model weights or tool schemas.

2. **Cross-benchmark and cross-model evaluation.**  
   The study compares two prompting formats on three public benchmarks and four different LLMs, which is better than a single-benchmark or single-model ablation. This breadth is useful for ICLR readers interested in generalization.

3. **Uses statistical testing rather than only raw scores.**  
   Reporting paired permutation-test p-values is a positive methodological choice and helps distinguish real changes from noise.

4. **Attempts to analyze failure modes, not just aggregate metrics.**  
   The paper includes categorical error analysis (schema violation, incorrect tool name, etc.) and a GPT-5 reasoning-effort study. This is aligned with ICLR’s preference for understanding *why* methods work or fail.

5. **The method is potentially low-cost.**  
   Since the approach is prompt-level rather than architectural, it has a plausible deployment path if it proves robust.

### Weaknesses
1. **The core novelty is modest relative to ICLR’s bar.**  
   At a high level, the paper mainly compares two prompt/output formats for the same task, without introducing a new learning algorithm, model, decoding method, or theoretical insight. For ICLR, a strong paper typically needs either a deeper methodological contribution or a more substantial conceptual advance than “natural language templates may work better than JSON-style outputs.”

2. **The evaluation protocol is not fully convincing as a fair comparison.**  
   The template-based method is parsed with regular expressions and converted into schema form, but the paper does not clearly demonstrate that both methods receive equally optimized prompting or decoding support. The authors themselves note that current models are biased toward schema-constrained formats, which makes the baseline comparison somewhat entangled with prompt engineering rather than isolating a principled algorithmic effect.

3. **Some results are inconsistent and weaken the general claim.**  
   Template-based generation does not uniformly help: GPT-5 shows significant regressions on API-Bank L1 and When2Call for both tool-name and parameter F1, and DeepSeek-Coder also has several losses. This makes the paper’s headline claim “improving accuracy” too broad; the evidence supports only a conditional improvement, not a general one.

4. **The analysis does not fully explain the mechanism.**  
   The paper hypothesizes that natural-language alignment helps, but this remains speculative. The error analysis identifies categories where outputs differ, yet it does not isolate whether gains come from better instruction following, reduced formatting brittleness, altered decoding entropy, or dataset-specific prompt matching.

5. **Reproducibility is incomplete.**  
   The paper says code will be released after acceptance, but that is not sufficient for a strong reproducibility claim at submission time. Important details are missing or underspecified, including exact prompts for all settings, fine-tuning hyperparameters, regex parsing rules, embedding model used for similarity, and how ambiguous parameter-value equivalence was handled across datasets.

6. **Metric design may blur semantic and exact-match correctness.**  
   Parameter values are judged equivalent when embedding cosine similarity exceeds 0.9. This may be reasonable, but the paper does not justify the threshold or analyze sensitivity to it. For tool calling, especially in API settings, small value changes can be materially wrong; conversely, embedding similarity may over-credit near-synonyms that are not operationally identical.

7. **Dataset/setting choices limit the strength of the conclusion.**  
   ToolACE is subsampled to 1,000 instances and only the first two turns are evaluated, while API-Bank Level 3 and ToolACE’s multi-step aspects are explicitly excluded. This narrows the scope substantially and makes it harder to claim broad progress on tool-use. The work therefore addresses a restricted slice of the problem.

### Novelty & Significance
**Novelty:** Moderate to low for ICLR standards. The paper offers an interesting prompt-format comparison, but not a fundamentally new learning paradigm.  
**Significance:** Moderate. The findings could matter in practice for tool-calling pipelines, especially if template-style outputs are easier for some models to produce. However, the impact is limited by inconsistent gains, restricted settings, and the absence of a deeper mechanistic or algorithmic contribution.  
**Clarity:** Generally understandable, with a clear high-level narrative, though several experimental details are missing or underspecified.  
**Reproducibility:** Partial. The overall setup is described, but key implementation specifics are not complete enough for confident reproduction.  
**ICLR acceptance bar:** This paper seems below the typical ICLR bar for a strong acceptance because it is more of an empirical prompt-format study than a substantial method paper, and its claims are stronger than its mixed evidence supports.

### Suggestions for Improvement
1. **Strengthen the methodological contribution beyond a prompt-format swap.**  
   For example, develop a principled template-generation objective, constrained decoding strategy, or hybrid method that adapts between schema and template formats based on model confidence or task type.

2. **Add stronger ablations to isolate the cause of gains.**  
   Compare against: (a) schema prompts with more natural-language phrasing, (b) template prompts with stricter syntax, (c) hybrid prompts, and (d) alternative parsing methods. This would help determine whether gains come from natural language form, reduced rigidness, or prompt verbosity.

3. **Report full implementation details.**  
   Include exact prompts, regex rules, fine-tuning hyperparameters, train/test splits, early-stopping criteria, token limits, and the embedding model/threshold used for value equivalence.

4. **Analyze the failures of GPT-5 more deeply.**  
   Since GPT-5 sometimes gets worse under template-based generation, the paper should investigate whether reasoning effort, overthinking, or formatting drift is responsible, and whether this is model-specific or task-specific.

5. **Evaluate on broader and harder tool-use settings.**  
   To support stronger claims, include multi-step tool invocation, longer dialogues, unseen tools, and end-to-end task success metrics rather than only tool-name/parameter F1.

6. **Justify and stress-test the semantic-matching criterion.**  
   Provide a sensitivity analysis over the cosine threshold and, ideally, human validation on a sample of matched/unmatched parameter values.

7. **Tone down the generality of the conclusion.**  
   The results support “template-based generation can help in some settings,” not “template-based generation generally improves tool calling accuracy.” A more precise claim would better match the evidence and make the paper more credible.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a stronger baseline that keeps the same semantic content but changes only the output format, e.g., XML / plaintext with the same level of structure, plus a “JSON but with more natural-language instructions” control. Right now the claim that “natural language-like templates” help is not isolated from a simple prompt-formatting effect.

2. Evaluate on more tool-calling benchmarks that reflect current ICLR expectations for generality, especially broader function-calling datasets beyond API-Bank/ToolACE/When2Call. The current evidence is too narrow to support a general claim that template-based generation improves tool calling accuracy.

3. Add a direct baseline where the schema-constrained output is post-processed with a parser/repair step or constrained decoding that removes formatting failures. Since many gains come from reducing schema violations, the paper must show whether the method still helps after giving the schema baseline standard robustness tricks.

4. Compare against prompt engineering and output-format baselines for the same models, including chain-of-thought vs no-CoT and few-shot exemplars for both formats. The GPT-5 results especially suggest the effect may depend more on prompting/reasoning setup than on the template format itself.

5. Run an ablation on template design: remove quotation marks, backticks, and fixed phrasing separately to see which components drive gains. Without this, it is unclear whether the improvement comes from natural-language alignment or from incidental cues in the template.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze performance by error type in a way that explains the net F1 gains, not just counts. The paper needs to show whether template-based generation improves the critical failure modes that matter in practice, or merely shifts errors between tool-name and parameter fields.

2. Report variance across multiple random seeds, sampling runs, or repeated fine-tuning runs. The current significance claims are not enough for ICLR; without stability analysis, it is unclear whether the gains are robust or prompt/run-specific.

3. Explain why GPT-5 sometimes gets worse under the template format, especially on API-Bank L1 and When2Call. This undermines the central claim of a broadly beneficial method and needs a principled analysis rather than a post hoc explanation.

4. Separate effects of model capability from training-data mismatch. The paper hypothesizes natural-language alignment, but it never tests whether gains correlate with pretraining mixture, instruction tuning style, or tool-use fine-tuning history.

5. Clarify whether the cosine-similarity matching for parameter values changes rankings or inflates scores. Since semantic equivalence is approximated with a hard 0.9 threshold, the paper needs sensitivity analysis to show results are not an artifact of this matcher.

### Visualizations & Case Studies
1. Add side-by-side qualitative examples of success and failure for each dataset/model pair, not just one cherry-picked example. ICLR reviewers will want to see whether the method fixes real tool-selection errors or just exploits formatting quirks.

2. Show confusion matrices or error-transition plots from schema-constrained to template-based outputs. This would reveal whether gains come mainly from fewer schema violations, improved tool selection, or better parameter extraction.

3. Provide a plot of performance versus reasoning effort with confidence intervals and exact values, not the broken figure. The current discussion suggests strong interaction effects, and a clean visualization is needed to verify whether template-based generation is genuinely unstable under stronger reasoning.

4. Include examples where template-based generation hurts performance, especially for GPT-5. These failure cases are essential to understanding the method’s limits and to avoid overclaiming broad applicability.

### Obvious Next Steps
1. Test whether the method still helps when the schema baseline is made equally “natural-language-friendly” through prompt rewriting or instruction tuning. Without this, the paper does not establish that the template form itself is the source of the gain.

2. Fine-tune or instruction-tune models directly on the template format and compare against schema-trained models. The paper’s own stated hypothesis depends on format familiarity, so this is the most obvious follow-up experiment.

3. Evaluate end-to-end tool-use success in an actual agent loop, not just F1 on predicted calls. ICLR would expect evidence that higher call-level F1 translates into better task completion, not only better string matching.

4. Test larger and more diverse open-source models, including non-code and tool-specialized models, to see whether the effect scales with model type. The current set is too limited to support a general method claim.

5. Add a controlled study of prompt length and verbosity. Since template prompts are more descriptive, the improvement could come from extra context rather than format, and that distinction is central to the paper’s contribution.

# Final Consolidated Review
## Summary
This paper asks a simple but relevant question: can tool-calling accuracy improve if LLMs generate natural-language templates instead of rigid schema-constrained outputs, with the same outputs later parsed into a tool schema? The authors evaluate this across API-Bank, ToolACE, and When2Call using GPT-4o, GPT-5, Mistral-7B, and DeepSeek-Coder, and find that template-based generation often helps, but the effect is clearly model- and dataset-dependent, with some substantial regressions.

## Strengths
- The paper studies a concrete and practically relevant bottleneck in tool calling: brittle schema-constrained generation. The proposed template format is simple and deployable without changing model weights or tool schemas, which makes the idea easy to act on if it were robust.
- The evaluation is broader than a toy ablation: it spans three public benchmarks and four models, and the paper includes statistical testing plus an error-category analysis. The tables show that the gains are real in several settings, especially for Mistral and ToolACE, rather than being just noise.

## Weaknesses
- The contribution is modest: this is primarily a prompt/output-format comparison, not a new tool-calling algorithm, training method, or decoding method. For ICLR, the paper’s empirical question is interesting, but the method itself is not especially deep.
- The central claim is too broad relative to the evidence. Template-based generation helps in many cases, but it also causes statistically significant regressions for GPT-5 and DeepSeek-Coder on some datasets, including clear drops on API-Bank L1 and When2Call. The paper acknowledges this, but the headline framing still overstates generality.
- The comparison is not clean enough to isolate why the gains happen. The two conditions differ in surface form, prompt style, parsing behavior, and potentially tokenization/robustness effects; the paper does not include strong ablations to separate “natural-language alignment” from simpler explanations like prompt verbosity or easier decoding.
- The parameter-value metric is underspecified and potentially brittle. Treating values as equivalent when embedding cosine similarity exceeds 0.9 is a consequential choice, but there is no sensitivity analysis or justification of the threshold, so parameter F1 may be less trustworthy than presented.
- Reproducibility is incomplete. Important implementation details are missing or only lightly described, especially the regex parsing rules, fine-tuning hyperparameters, embedding model used for semantic matching, and exact prompt variants. That limits confidence in the reported numbers and makes the study hard to replicate.

## Nice-to-Haves
- A stronger ablation suite would be very helpful: alternative templates, a “more natural-language” schema baseline, and a hybrid format would clarify whether the gain is due to the template itself or just a better prompt.
- A sensitivity analysis for the semantic-matching threshold on parameter values would improve confidence in the reported parameter F1.
- The reasoning-effort study would be easier to interpret with a clean numeric table and confidence intervals rather than the current figure-heavy presentation.

## Novel Insights
The most interesting insight is not that template-based generation is universally better—it clearly is not—but that the effect seems to interact strongly with model family and reasoning style. In particular, the results suggest that models with stronger alignment to natural-language generation can benefit from a descriptive tool-call format, while reasoning-oriented or code-centric models may not reliably gain from it and can even degrade. The error analyses support a plausible mechanism: template format can reduce schema violations and sometimes improve contextual tool selection, but the same format may also introduce instability when the model’s reasoning or formatting habits are mismatched to the template.

## Potentially Missed Related Work
- Tool-learning and function-calling surveys / benchmarks such as the cited Tool Learning with LLMs survey and API-Bank/ToolACE/When2Call — relevant as benchmark and problem context, though the paper already cites them.
- Constrained decoding and structured output repair work — relevant because a key open question is whether template gains persist once the schema baseline is given equivalent robustness treatment. The paper does not situate itself deeply in this line.

## Suggestions
- Add controlled ablations that keep semantic content fixed while varying only output format and verbosity, so the paper can separate template effects from prompt-engineering effects.
- Report the full parsing and semantic-matching pipeline, including regex rules, embedding model, and threshold sensitivity, so the evaluation is auditable.
- Expand the analysis of negative cases, especially GPT-5 and DeepSeek-Coder failures, because these are central to understanding the method’s limits and preventing overclaiming.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 2.0, 0.0]
Average score: 1.0
Binary outcome: Reject
