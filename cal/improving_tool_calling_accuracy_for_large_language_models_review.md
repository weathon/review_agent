=== CALIBRATION EXAMPLE 22 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title accuracy:** The title accurately describes the empirical focus but is generic and does not distinguish the proposed approach (template-based natural language generation vs. schema-constrained output).
- **Abstract clarity:** The abstract outlines the problem, method, and experimental setup, but over-generalizes results by stating the method "improves F1 scores... on most tests" without acknowledging the documented degradations (e.g., GPT-5 on API-Bank L1 and When2Call, DeepSeek-Coder on L1). This creates an inaccurate first impression of consistent gains.
- **Unsupported claims:** The abstract implies the method is broadly superior, but the paper itself shows highly model-dependent and dataset-dependent outcomes. The claim that template-based generation is inherently better is not established; it is contingent on model architecture and post-training alignment.

### Introduction & Motivation
- **Problem motivation & gap:** The core motivation—that LLMs are pretrained on natural language and thus schema-constrained (JSON-like) generation creates a distribution shift—is intuitively plausible but presented as fact without empirical grounding. The introduction does not engage with the extensive literature on constrained decoding, grammar-guided generation, or modern function-calling alignment, which explicitly bridge this gap during RLHF/post-training. Consequently, the identified "gap" is overstated relative to current practice.
- **Contributions:** The contributions are essentially a prompt format change and an empirical evaluation. For ICLR, this leans heavily toward engineering/prompt engineering rather than a novel algorithmic or representational advance. The contributions are stated clearly but under-qualify the simplicity of the intervention.
- **Over/under-selling:** The introduction slightly over-claims novelty ("novel method") while under-specifying why existing schema-constrained pipelines fail systematically. It also lacks discussion of why a simple regex-parsable NL template would be robust in production agent loops, which weakens the practical motivation.

### Method / Approach
- **Reproducibility:** Several critical details are missing. Fine-tuning for Mistral and DeepSeek-Coder mentions "five epochs with early stopping" but omits learning rate, batch size, optimizer, hardware, and whether parameter-efficient fine-tuning (LoRA/QLoRA) was used. The exact regex patterns used to parse the template outputs are not provided. These omissions materially hinder reproducibility.
- **Assumptions & justification:** The method assumes that natural language phrasing aligns better with LLM pretraining distributions and that regex parsing is sufficiently robust. Neither is justified or analyzed. For instance, natural language templates may introduce higher token counts and altered attention patterns compared to compact JSON, potentially shifting latency/cost trade-offs that are not measured.
- **Logical gaps:** The causal link between the NL format and improved accuracy is hypothesized in Section 5 ("closer alignment with natural language") but never isolated from confounders like prompt length, explicit instruction phrasing, or tokenization boundaries. Without controlling for these, it is unclear if gains stem from the format itself or incidental prompt improvements.
- **Edge cases/failure modes:** The paper does not address what happens when parameter values contain template keywords (e.g., `Call`, `parameter`, quotation marks), which would break the regex parser. The handling of multi-value parameters, nested structures, or special characters is also absent.
- **Theoretical/mechanistic grounding:** The paper lacks any formal or mechanistic account of why template-based generation should outperform schema constraints, beyond a high-level distributional mismatch argument. No analysis of attention, token probability distributions, or logit shifts under the two formats is provided.

### Experiments & Results
- **Testing claims:** The experiments test accuracy but not the full scope of tool-calling utility. ICLR expects evaluation that measures downstream impact (e.g., successful API execution, agent loop stability) or at least controls for major confounders. The paper measures only macro-F1 on names/parameters.
- **Baselines:** The baseline is the "original method used in all datasets," which typically relies on standard instruction prompting for JSON. This critically misses modern constrained decoding techniques (e.g., `outlines`, `guidance`, regex/logit-bias generation) that guarantee syntactically valid schema outputs. Comparing a free-form template against a baseline prone to syntax errors without enforcing valid JSON via decoding unfairly advantages the template approach. This is a major methodological gap.
- **Missing ablations:** There is no ablation over template phrasing, instruction clarity, token count, or the cosine similarity threshold (0.9). The choice of 0.9 for semantic equivalence is arbitrary; different embedding models or thresholds could drastically alter parameter F1, yet the embedding model used is never specified. Fine-tuning data splits and sizes are also not rigorously documented.
- **Statistical rigor:** The use of paired permutation tests (9,999 permutations) is appropriate and reported consistently. However, confidence intervals or variance estimates across multiple seeds are absent, which limits interpretability for models with smaller sample sizes.
- **Results vs. claims:** The tables show mixed results: GPT-5 and DeepSeek-Coder experience statistically significant drops on several subsets. The text acknowledges this but frames it as marginal or dataset-specific. The claim of broad improvement is therefore only conditionally true and relies on post-hoc selection of positive cases.
- **Datasets/metrics:** The datasets are well-known and relevant. However, ToolACE and API-Bank contain multi-turn interactions; restricting analysis to the first two turns is justified but excludes evaluation of how template generation scales in longer agent trajectories. The parameter F1 metric relies on exact match for names and embedding similarity for values, which is reasonable but under-documented.

### Writing & Clarity
- **Confusing sections:** Section 4.5 discusses reasoning effort vs. performance, but Figure 1's textual description and axis labels are garbled (parser artifact). The underlying trend (schema-constrained benefits from reasoning, template-based does not) is explained clearly enough, but the lack of a readable plot reduces confidence in the quantitative trend. The error categorization in Section 4 (types 1–6) claims mutual exclusivity but provides no mechanism for how samples with multiple error modes are assigned to a single category.
- **Figures/tables:** Tables 1–3 and 4–8 are clear and support the narrative. The error breakdown tables are informative but would benefit from proportional normalization (percent of total errors) rather than raw counts, given the varying dataset sizes.

### Limitations & Broader Impact
- **Acknowledged limitations:** The authors correctly identify evaluation suboptimality (models biased toward JSON), limited scope, and the need for mechanistic understanding.
- **Missed limitations:** 
  1. **Parsing robustness:** Regex-based extraction is brittle to minor lexical variations or adversarial inputs, which is not discussed.
  2. **Downstream execution:** High F1 does not guarantee successful tool invocation or agent loop convergence. Errors in template phrasing could cause API clients to fail even if a human evaluator marks it "correct."
  3. **Safety/Security:** Natural language templates may be more susceptible to prompt injection or value poisoning compared to strictly validated JSON schemas. This is unaddressed.
- **Broader impact:** Minimal discussion of broader impact is expected for this type of methodological paper, but a brief note on deployment stability, parsing overhead, or safety trade-offs would align with ICLR's expectations for responsible deployment considerations.

### Overall Assessment
This paper investigates a straightforward shift from schema-constrained (JSON-like) to template-based natural language output for LLM tool calling. While the empirical study is well-structured and reports statistically sound permutation tests, the contribution falls short of ICLR's bar for methodological novelty and rigor. The core idea is essentially a prompt engineering variant, and the claimed gains are not isolated from confounding factors such as prompt phrasing, token length, or the absence of modern constrained-decoding baselines. Critical reproducibility details (fine-tuning hyperparameters, embedding model for parameter similarity, regex extraction logic) are missing, and the analysis of failure modes lacks systematic coverage across all model/dataset pairs. The mixed results on GPT-5 and DeepSeek-Coder appropriately temper the enthusiasm, but the paper does not develop a mechanistic or theoretical framework to explain when and why template-based generation succeeds or fails. In its current form, the work is better suited for an applied NLP venue (e.g., ACL/EMNLP findings or an agent engineering workshop) unless significantly strengthened with rigorous constrained-decoding baselines, ablation studies, full reproducibility details, and a deeper investigation into the underlying model behavior.

# Neutral Reviewer
## Balanced Review

### Summary
The paper proposes replacing conventional schema-constrained (e.g., JSON) tool-calling prompts with natural language-like templates to better align with the pretraining data of large language models. Empirical evaluations across four models and three benchmarks show that template-based generation generally improves tool name and parameter F1 scores, with error analysis indicating reductions in schema violations and improved contextual adherence. The authors conclude that NL-aligned templates offer a simple, effective alternative for tool invocation, though results vary by model architecture and reasoning capabilities.

### Strengths
1. **Rigorous statistical evaluation:** The report of absolute deltas alongside paired permutation tests (9,999 permutations) provides statistically grounded evidence for observed gains. Results are clearly tabulated with significance markers (Tables 2 & 3).
2. **Systematic error categorization and analysis:** Section 4 breaks down failures into six mutually exclusive categories. This fine-grained analysis (e.g., Tables 4–7) reveals that template-based prompting primarily reduces "Schema Violation" and "Incorrect Tool Name" errors, offering actionable debugging insights for agent pipelines.
3. **Thoughtful model selection:** Evaluating a diverse set of models (general-purpose, reasoning-enhanced, and code-specialized) allows the authors to contextualize performance differences. The observation that code-centric models see marginal gains due to limited NL training data demonstrates careful empirical reasoning (Section 3 & 4.6).

### Weaknesses
1. **Limited baseline comparison and methodological gap:** The paper positions schema-constrained JSON as the sole incumbent baseline but ignores modern constrained decoding techniques (e.g., Outlines, JSONMode, grammar-guided generation) that fundamentally eliminate the parsing/schema-violation issues the paper aims to solve. Without comparing against these stronger baselines, the claim of "improvement" overestimates the practical contribution.
2. **Missing experimental details hindering reproducibility:** Key implementation details are omitted. The fine-tuning setup for Mistral and DeepSeek lacks optimizer, learning rate, batch size, and data-mixing specifications (Section 2). Additionally, the embedding model and prompt used for the 0.9 cosine similarity threshold (Section 3) are unspecified, introducing unquantified variance in parameter F1 scoring.
3. **Overgeneralized conclusions despite mixed empirical signals:** While the abstract and introduction frame template-based generation as consistently beneficial, Tables 2 and 3 show statistically significant *degrades* for GPT-5 on API-Bank L1 and When2Call, as well as inconsistent trends with reasoning effort (Section 4.5, Table 8). The paper acknowledges this but does not sufficiently integrate these contradictions into its core claims.

### Novelty & Significance
**Novelty:** Low to moderate for ICLR. The core insight that LLMs generate natural-language-like templates more reliably than rigid JSON schemas is an established observation in the LLM agent community. The paper does not introduce a new training objective, decoding algorithm, or theoretical framework; rather, it empirically validates a straightforward prompting heuristic.
**Clarity:** High. The paper is well-structured, the experimental pipeline is logically laid out, and the error taxonomy is easy to follow. Figures and tables (despite parser artifacts) effectively communicate results.
**Reproducibility:** Moderate to low. While code is promised post-acceptance, critical hyperparameters, the regex parsing logic, and the semantic equivalence pipeline are not detailed. The reliance on proprietary API calls (GPT-4o/5) further limits independent verification.
**Significance:** Moderate practical value but limited for ICLR's methodological bar. The findings are useful for practitioners building lightweight tool-use pipelines without grammar-constrained libraries, but the lack of comparison to state-of-the-art structured generation methods and the absence of deeper mechanistic insights place this work outside ICLR's typical acceptance threshold for full papers. It aligns more closely with an empirical benchmark or workshop contribution.

### Suggestions for Improvement
1. **Incorporate structured decoding baselines:** Compare template-based generation against modern constrained decoding libraries (e.g., Outlines, guidance, or JSON schema validators) to isolate whether template benefits stem from NL alignment or merely from avoiding brittle regex/parsing on malformed JSON.
2. **Complete the reproducibility checklist:** Provide exact fine-tuning hyperparameters (optimizer, LR schedule, batch size, epochs/early stopping criteria), specify the embedding model used for cosine similarity, and justify or ablate the 0.9 threshold. Release an anonymized repository with run scripts prior to or alongside revision submission.
3. **Deepen the reasoning model analysis:** The GPT-5 results show instability with increased reasoning effort (Section 4.5, Table 8). Provide qualitative examples of how intermediate reasoning chains amplify template deviations versus how they stabilize JSON outputs. This would transform an observed inconsistency into a mechanistic insight.
4. **Calibrate claims to empirical evidence:** Temper conclusions to explicitly acknowledge that template-based generation is model- and dataset-dependent. Frame it as a complementary prompting strategy rather than a universal replacement for schema-constrained generation, and discuss scenarios where JSON remains strictly preferable (e.g., programmatic downstream parsers, strict schema validation).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Evaluate against modern schema-constrained decoding (e.g., grammar-guided generation via `outlines` or `guidance`, or model-native `json_schema`) instead of prompt-only formats. Without this baseline, the claim that templates inherently outperform schema-constrained generation is unconvincing, as it may simply reflect weak prompt engineering for the baseline.
2. Align training regimes across all models. Closed models are evaluated zero-shot while open models are fine-tuned on dataset-specific splits; this confounds the results because observed gains could stem from fine-tuning dataset overlap rather than the generation format. Run controlled zero-shot evaluations for all models or consistently fine-tune all on identical data.
3. Validate the parameter F1 heuristic with a threshold ablation on the 0.9 cosine similarity cutoff, or replace it with an LLM-as-a-judge semantic equivalence evaluator. The arbitrary threshold inflates or deflates parameter accuracy depending on minor embedding shifts, directly undermining the reported F1 scores.

### Deeper Analysis Needed (top 3-5 only)
1. Quantitatively test the "natural language alignment" hypothesis by measuring the perplexity or embedding shift distance of template outputs vs. schema outputs against typical LLM pretraining corpora. The paper attributes all gains to this hypothesis but never measures it, leaving the core mechanism speculative.
2. Decouple LLM generation errors from regex parsing failures. Template outputs are inherently brittle to LLM verbosity and stylistic drift; report the exact parse failure rate separately from reasoning/logic errors to prove the method improves generation rather than just masking parsing brittleness with different prompt wording.
3. Diagnose why reasoning models (GPT-5) exhibit inconsistent or degraded performance under the template format. Analyze whether chain-of-thought or internal planning steps naturally conflict with the rigid template structure; without this, the claim that templates broadly improve tool calling ignores a critical architectural mismatch.

### Visualizations & Case Studies
1. Provide side-by-side trace examples where template-based generation introduces novel failure modes absent in schema generation (e.g., conversational filler breaking regex, omitted backticks, or hallucinated template fields). This reveals whether the method actually reduces errors or merely exchanges schema violations for structural drift.
2. Visualize the specific conversational contexts where "Incorrect Tool Name" errors were resolved by templates (Tables 4 & 5) with explicit highlighting of how the natural language prompt forced better context grounding versus JSON generation. Without concrete traces, the claimed improvement in contextual understanding remains an unverified assertion.
3. Show output stability distributions across multiple sampling runs or temperature settings for both methods. This exposes whether template generation actually reduces variance or simply shifts failure concentrations to different parts of the output string.

### Obvious Next Steps
1. Supervised fine-tune a base model on template-generated data to test the hypothesis that domain alignment yields compounding accuracy gains. Relying solely on prompt engineering at inference time cannot prove the format is superior for future model development.
2. Integrate the template approach with regex-constrained decoding or constrained generation frameworks to eliminate parsing brittleness. This isolates pure generative performance and demonstrates the practical viability of the method before it undermines reliability.
3. Systematically vary template strictness and phrasing to quantify prompt sensitivity. If minor instruction tweaks cause large accuracy swings, the method lacks the robustness required for a credible contribution.

# Final Consolidated Review
## Summary
This paper investigates a simple but practical shift in LLM tool calling: replacing rigid, schema-constrained (e.g., JSON-like) output formats with natural language-like templates. Evaluated across four diverse models (general-purpose, reasoning-enhanced, code-specialized) and three benchmarks, the approach generally improves macro-F1 scores for tool names and parameters, primarily by reducing syntactic schema violations and improving contextual tool selection. However, the gains are highly model-dependent, and the study leaves several methodological gaps regarding baseline adequacy and evaluation consistency.

## Strengths
- **Rigorous, statistically backed evaluation:** The paper employs paired permutation tests (9,999 permutations) across all model-dataset combinations and systematically reports significance levels for performance deltas. This provides robust quantitative grounding that avoids overclaiming based on marginal raw differences.
- **Granular error taxonomy revealing failure mode shifts:** The six-category error breakdown (Tables 4–7) clearly demonstrates that template-based generation specifically mitigates "Schema Violation" and "Incorrect Tool Name" errors in conversational settings. This pinpoint analysis offers actionable debugging insights for agent pipeline engineers.
- **Nuanced characterization across model training distributions:** By evaluating models with distinct pretraining/fine-tuning backgrounds, the paper correctly identifies that the benefits of NL-aligned templates are not universal. The documented performance drops for code-specialized models and reasoning-heavy setups provide a realistic boundary for when the method is appropriate.

## Weaknesses
- **Inadequate baseline comparison undermines the core motivation:** The paper compares template prompting against prompt-constrained JSON baselines, but omits modern constrained decoding techniques (e.g., grammar-guided generation, structured output enforcement libraries like `outlines` or model-native `json_schema`). These modern methods deterministically prevent schema violations at the decoding level. Without this baseline, it is unclear whether template gains stem from better "natural language alignment" or simply from comparing a flexible format against an artificially brittle prompt-based constraint.
- **Confounded training and evaluation regimes hinder cross-model attribution:** Proprietary models (GPT-4o/GPT-5) are evaluated zero-shot, while open models (Mistral, DeepSeek-Coder) are fine-tuned on dataset-specific splits for five epochs. This mixing confounds the observed format-dependent improvements, as gains on open models could partially result from fine-tuning data distribution rather than the output format itself. Consistent zero-shot or consistent fine-tuning protocols are required to isolate the format's effect.
- **Undocumented parameter evaluation pipeline limits reproducibility:** Parameter F1 relies on a 0.9 cosine similarity threshold between value embeddings, yet the specific embedding model, prompt, or threshold justification is never specified. This introduces unquantified variance into a primary metric, making exact replication and cross-study comparison impossible.

## Nice-to-Haves
- Ablate template strictness and phrasing to quantify prompt sensitivity, which would clarify whether the method is robust to minor instruction variations or highly brittle.
- Separate regex parsing failure rates from logical/reasoning errors to prove that the format actually improves generation quality rather than merely masking parsing brittleness with different prompt wording.
- Probe the "natural language alignment" hypothesis mechanistically (e.g., via perplexity shifts or embedding distance to pretraining corpora) to move the explanation from post-hoc intuition to measurable evidence.

## Novel Insights
The core takeaway transcends the specific prompt format: optimal tool calling is not a one-size-fits-all interface but a model-alignment problem. Template-based generation successfully leverages a model's native linguistic priors to bypass syntactic constraints, but this advantage actively backfires for reasoning-enhanced models, where intermediate chain-of-thought steps conflict with rigid output templates, causing instability as reasoning effort scales. Similarly, code-specialized models lack the natural language fluency to exploit the format. This reveals a clear design rule: the decoding interface must be matched to the model's primary training distribution, with NL templates benefiting general-purpose language models but offering diminishing returns for specialized or heavily reason-oriented architectures.

## Suggestions
- Integrate at least one modern constrained decoding baseline (grammar-guided or schema-enforced) to rigorously isolate whether template gains stem from format alignment or from avoiding weak prompt-based constraints.
- Standardize the evaluation protocol across all models (e.g., evaluate all zero-shot using identical prompt structures, or fine-tune all models on the same held-out training split) to decouple format effects from fine-tuning benefits.
- Disclose the exact embedding model and evaluation pipeline for parameter F1, and add a brief sensitivity analysis around the 0.9 similarity threshold to ensure metric transparency and reproducibility.

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 2.0, 0.0]
Average score: 1.0
Binary outcome: Reject
