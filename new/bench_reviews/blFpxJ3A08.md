## Summary

The paper introduces LPFQA, a 505-question benchmark derived from professional technical forums across 20 academic fields, designed to evaluate LLMs on long-tail professional knowledge. LPFQA claims four key innovations: fine-grained evaluation dimensions (knowledge depth, reasoning, terminology comprehension, contextual analysis), hierarchical difficulty structure, authentic professional scenario modeling, and interdisciplinary knowledge integration. Twelve frontier LLMs are evaluated, and ablation studies examine the effects of code interpreter and web search augmentation.

## Strengths

- **Novel data sourcing idea**: Mining professional forums for LLM evaluation questions addresses a genuine gap—forum questions involve practical, specialized knowledge that textbook-derived benchmarks miss. The 8-step pipeline from forum scraping through expert verification (Section 3.2) is a reasonable approach to sourcing such content.

- **Interesting ablation finding on tool augmentation**: The observation that both code interpreter (Table 3, average −10.64%) and web search (Table 4) *decrease* performance on LPFQA raises a practically significant finding: long-tail professional knowledge is inherently difficult to retrieve online, and retrieval augmentation can introduce misleading information. While the inference drawn is overstrong (see Weaknesses), the raw finding is valuable.

- **Systematic filtering for discriminative power**: Introducing LPFQA⁻ (436 items, removing unanswerable questions) and LPFQA= (421 items, additionally removing universally solvable questions) with recalculated scores (Table 2) demonstrates attention to whether the benchmark actually differentiates models—a step many benchmarks skip.

- **Broad model coverage**: Evaluating 12 frontier models including GPT-5, Claude-4, Grok-4, and DeepSeek-R1 (Table 1) provides a useful snapshot of current capabilities on specialized content, with meaningful score separation (32.40–47.28).

## Weaknesses

### Fatal

None.

### Major

- **Claimed evaluation dimensions are never used in results reporting**: The paper prominently claims "fine-grained evaluation dimensions" targeting "knowledge depth, reasoning, terminology comprehension, and contextual analysis" as its first innovation (Abstract, Section 1, Section 3.1). Yet results in Section 4 are reported exclusively by academic field (Figures 3–4), with no per-dimension breakdown for any model. Without dimension-level scores, the central claim that LPFQA provides "comprehensiveness in evaluating LLMs' capabilities" through fine-grained dimensions is entirely unsupported—the dimensions function as labels, not as measurable evaluation axes. This undermines the paper's first and most emphasized contribution.

- **The "authentic professional scenario modeling" claim is overstated**: The paper repeatedly claims "authentic professional scenario modeling with realistic user personas" (Abstract) and that tasks are "derived from real discussions in technical forums" (Section 3.1). However, the actual pipeline (step ❹, Section 3.2.2) discards original forum questions and instead has an MLLM examine screenshots to *generate new* question–answer pairs. The original questions are never used directly. The benchmark thus measures how models handle an MLLM's interpretation of screenshots of forum posts—not how they handle real user queries. The authenticity and "user personas" claims are significantly overstated relative to what the pipeline delivers.

- **The knowledge-vs-reasoning conclusion is overdrawn**: Section 4.2.2 concludes that "LPFQA primarily reflects a model's mastery of domain knowledge rather than its reasoning ability" because adding a code interpreter decreased scores (Table 3). But a code interpreter specifically assists computational reasoning—it is irrelevant to logical, analogical, verbal, or domain-specific reasoning that dominates professional Q&A. The code interpreter's failure to help is equally consistent with: (a) the benchmark testing non-computational reasoning, (b) the code interpreter introducing errors on knowledge-heavy items, or (c) reasoning that doesn't benefit from code execution. The inference from "code interpreter didn't help" to "LPFQA measures knowledge not reasoning" is a non sequitur that invalidates a major claimed finding.

### Minor

- **Very small per-field sample sizes undermine field-level analysis**: With 505 questions across 20 fields, some fields have as few as 3 items (Data Science). Claims like "DeepSeek-R1 attains leading scores in DS" based on 3 items (Section 4.1) are statistically unreliable. The per-field rankings in Figures 3–4 should be interpreted with caution, but the paper presents them as substantive findings without acknowledging this limitation.

- **Selective model reporting in ablation tables**: Table 3 reports 10 of 12 models; Table 4 reports only 3 models. No explanation is given for the omissions (Claude-4 and Kimi-K2 are missing from Table 3; 9 models are missing from Table 4). This makes it impossible to assess whether the reported patterns generalize across all evaluated models.

- **Misleading "overall best-performing model" claim**: The paper states "DeepSeek-V3 demonstrates the most balanced and consistent performance across disciplines... and can thus be regarded as the overall best-performing model" (Section 4.1), but GPT-5 has the highest score (47.28 vs. 32.60). Redefining "best" as "most balanced" without explicit acknowledgment or a quantitative balance metric is misleading.

- **"Interdisciplinary knowledge integration" overclaimed**: Listed as the fourth innovation, this appears to mean nothing more than "the benchmark contains questions from multiple fields"—which MMLU, BIG-bench, and many others already do. No question in the benchmark demonstrably requires cross-disciplinary reasoning.

### Trivial

- **Inconsistent task count**: The abstract states "covering 502 tasks" while Section 3.1 and 3.3 state 505 questions.

## Nice-to-Haves

- Report per-dimension scores for all 12 models, making the fine-grained evaluation dimensions operational rather than aspirational.
- Use original forum questions directly rather than MLLM regenerations, which would genuinely deliver on the authenticity claim.
- Expand fields with very few items (e.g., Data Science with 3 items) to at least 20–30 questions per field to make per-field analysis defensible.
- Include a data contamination check—since forum content is public web data, probing whether evaluated models have memorized source material is important for a benchmark claiming to test long-tail knowledge.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Difficulty structure circularly defined by evaluated models" (Harsh Critic #4)**: While step ❽ does use model performance to classify difficulty, this is standard practice in benchmark construction (many widely-used benchmarks calibrate difficulty similarly). The concern about benchmark instability is noted but is not a distinctive flaw of this paper relative to its peers. Downgraded to acknowledged practice rather than a weakness.

- **"Questions are MLLM-generated, not authentic, undermining authenticity claim" as a fatal flaw**: The critic presented this as a fatal structural issue. While the authenticity claim IS overstated (and this is kept as a Major weakness), the content does originate from real forum screenshots—this is not purely synthetic data. The criticism is valid but its severity is Major, not Fatal.

- **"Appendix B example is a generic MCQ" (Harsh Critic)**: This is a presentation nitpick about the example question. The format of a single example doesn't invalidate the benchmark.

- **"What MLLM/LLM was used for generation/quality control?" (Harsh Critic)**: The paper states these details are in the appendix. Per our rules, missing appendix content is not a valid criticism.

- **"Why not extract text directly instead of screenshots?" (Harsh Critic)**: The paper explains this is "to facilitate later multi-modal content analysis" and to "preserve contextual and visual information" (Section 3.2.1). This is a design choice with stated rationale.

- **"Missing experiments: confidence intervals, significance tests" (Harsh Critic)**: Reporting confidence intervals is not standard practice for LLM benchmark papers of this type; this is a nice-to-have, not a core flaw.

- **"Test data contamination" (Harsh Critic)**: While important for a long-tail knowledge benchmark, contamination checks are not yet standard in the community. This is a nice-to-have.

- **Strength Finder's "Broad interdisciplinary coverage" as a standalone strength**: This is weakened by the very uneven distribution (3 items in DS). Kept the point but with qualification in the Minor weaknesses.

- **Strength Finder's "Authentic data sourcing from real professional forums with a reproducible pipeline"**: The "authentic" part is directly contradicted by the verified Major weakness about MLLM generation. Removed as a standalone strength; the pipeline's existence is noted in the "Novel data sourcing idea" strength.

## Novel Insights

The most interesting empirical finding—overlooked by both reviewers—is that web search augmentation *decreases* performance on LPFQA (Table 4). This suggests a concrete boundary condition for RAG systems: for genuinely specialized, long-tail professional knowledge, retrieval from the open web may actively harm rather than help. This is practically significant and underexplored in the literature. However, the paper's interpretation (that this proves LPFQA measures knowledge not reasoning) is a misattribution—the finding is about the limitations of web retrieval for niche topics, not about what LPFQA measures per se.

## Suggestions

- **Operationalize the evaluation dimensions**: Tag each question with its primary dimension and report per-dimension accuracy. This would transform an unsupported claim into the paper's strongest contribution.
- **Provide a quantitative "balance" metric**: If claiming DeepSeek-V3 is "most balanced," define and report a variance or entropy metric across fields rather than relying on visual inspection.
- **Explain model omissions in ablation tables**: State why some models are missing from Tables 3 and 4 (cost? API limitations? compatibility issues?).
- **Soften the knowledge-vs-reasoning claim**: Replace "LPFQA primarily reflects domain knowledge mastery rather than reasoning ability" with a more carefully worded conclusion that the code interpreter ablation suggests computational tool augmentation does not aid LPFQA performance, consistent with (but not proving) a knowledge-heavy benchmark.

## Score and Decision

**Calibration anchors:**
- /home/wg25r/review_agent/human_reviews_2026/ZfdnZhOP0k.md (Hubble, avg 7.5, Oral): Far above LPFQA—rigorous methodology, fully open-source, extensive controlled experiments, no overclaims. LPFQA is clearly below this.
- /home/wg25r/review_agent/human_reviews_2026/Q5QLu7XTWx.md (PCB-Bench, avg 6.0, Poster): More comprehensive (3700+ instances), better validated, cleaner claims. LPFQA is below this due to overclaims and smaller scale.
- /home/wg25r/review_agent/human_reviews_2026/7TlCUD2tQI.md (DiagnosticIQ, avg 4.0, Reject): Similar profile—domain-specific MCQA benchmark with automated pipeline, evaluated on 15 LLMs. LPFQA has comparable overclaim issues but adds the interesting ablation finding. Roughly comparable.
- /home/wg25r/review_agent/human_reviews_2026/Fj7adDEWm4.md (TRIDENT, avg 4.0, Reject): Similar—domain-specific LLM benchmark with overclaimed contributions. LPFQA is comparable.
- /home/wg25r/review_agent/human_reviews_2026/5OXvNX9LWh.md (CML-Bench, avg 3.0, Reject): Very small dataset (100 samples), overclaimed novelty, confusing evaluation dimensions. LPFQA is somewhat above this given its larger scale and useful ablation findings.
- /home/wg25r/review_agent/human_reviews_2026/u72E1za8dU.md (S³-Bench, avg 2.5, Reject): Overclaimed title/benchmark, fundamentally a survey paper. LPFQA is above this.

LPFQA sits in the 3.5–4.0 range: it has a reasonable core idea and one genuinely interesting ablation finding, but three of its four claimed innovations are substantially unsupported or overstated. This is comparable to DiagnosticIQ and TRIDENT (both scored 4.0 and rejected), with similar profiles of domain-specific benchmarks with overclaimed novelty.

**Evaluation on axes:**
- Originality: Moderate—forum mining is a reasonable idea but the execution doesn't distinguish from prior work
- Importance of research question: Moderate—long-tail knowledge evaluation matters
- Claims well supported: Weak—three of four innovations unsupported or overclaimed
- Soundness of experiments: Moderate—core evaluation is sound but ablation inference is flawed and reporting is selective
- Clarity: Moderate—writing is readable but claims outpace evidence
- Value to community: Low to moderate—small benchmark with overstated features; the search-augmentation finding is the most transferable insight

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>