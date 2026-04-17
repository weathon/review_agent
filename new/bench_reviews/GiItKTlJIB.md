Now I have enough information to compose the final review. Let me synthesize everything carefully.

**Key verification of harsh critic claims against the paper:**

1. **"Faithfulness claims not supported by deletion protocol"** — This is the most serious criticism. The paper says "we intercept the scratchpad and remove k% of CoT tokens before the final answer." This *does* suggest the model re-decodes from the post-deletion prefix. However, the description is ambiguous. The paper doesn't explicitly clarify whether the model sees only the truncated prefix when generating the final answer, or whether deletions are applied post-hoc. This ambiguity is real. However, the claim that "if deletion is applied only to what is shown to the judge, the main claim is invalid" is speculative — the paper's description ("intercepting CoT mid-generation and removes tokens before decoding") suggests it's a real intervention, not post-hoc. This is a valid concern about under-specification but likely not a fatal flaw.

2. **"Evaluation judge under-specified"** — Valid. No inter-rater reliability, no validation against human grader, no error analysis.

3. **"Overlap metrics don't capture faithful recovery"** — Valid. Bag-of-words metrics are coarse for physics equations.

4. **"Ambiguity of deletion mechanism"** — Valid, as discussed above.

5. **"Lack of direct comparison to ground-truth answers"** — Partially valid. The scoring is 0-1 by Claude-4 Sonnet, not based on exact numerical answers.

**Spark reviewer claims:**
- "Zero-CoT baseline" — The paper has Low/Medium/High reasoning but doesn't test a true zero-CoT condition. This is a meaningful gap.
- "Non-physics domain control" — Valid suggestion but paper explicitly scopes to physics.
- "Control for CoT length rather than content" — Interesting but not critical.
- "Judge validation" — Valid.
- "Correctness of recovered content" — Valid, overlaps with overlap metric criticism.
- "Per-problem difficulty stratification" — Valid.
- "Statistical significance" — Partially valid; calibration study uses 5 runs but no CIs in main results.

**Human finder's synthesized weaknesses:**
- Limited models / generalizability — Valid but acknowledged in limitations.
- Faithfulness metric surface-level — Valid.
- Using Claude-4 Sonnet as judge — Valid.
- Limited domain scope — Acknowledged.
- No mechanistic explanation — Valid.
- Incremental methodology — Partially valid; the physics-aware deletion is somewhat novel.
- Overlap metrics don't distinguish recovery from plausible regeneration — Valid and important.
- Conflates stability with dispensability — Valid.

Let me now write the final consolidated review.</think>## Summary

This paper introduces a systematic CoT deletion framework for probing how much LLMs genuinely depend on their chain-of-thought reasoning traces when solving physics problems. By intercepting CoT mid-generation, deleting tokens via three strategies (from-the-end, random, physics-aware), and measuring downstream effects on answer quality, answer length, and information overlap, the authors find that models maintain accuracy under 40–60% CoT deletions and exhibit compensatory "cramming" behavior—reconstructing missing reasoning in their final answers. The paper argues this indicates shallow and opportunistic reliance on CoT, raising concerns about CoT faithfulness in AI-for-Science contexts.

## Strengths

1. **Novel and intuitive probing methodology.** The deletion-based framework—intercepting scratchpads mid-generation and systematically removing tokens—is a creative and straightforward intervention for studying CoT dependence. The three complementary deletion strategies (end, random, physics-aware) provide converging perspectives on how different types of CoT content matter.

2. **Well-motivated domain choice.** Physics is an excellent testbed for CoT faithfulness because its structured reasoning (equations, units, terminology) enables more precise quantification of information recovery than open-ended tasks. The paper makes a compelling case for why AI-for-Science domains deserve specialized CoT evaluation.

3. **Interesting empirical finding: "cramming."** The consistent X-shaped pattern—where deleting CoT tokens causes final answer length to increase while accuracy remains relatively stable—is a nontrivial behavioral observation documented across three models and three benchmarks. This phenomenon is worth reporting and could catalyze further investigation.

4. **Important and timely research question.** Whether CoT traces faithfully reflect underlying computation is critical for LLM interpretability and safety, and evaluating this specifically in scientific domains (where reliability matters most) adds practical significance. The paper's framing connects CoT faithfulness to broader concerns about evaluation in AI-for-Science.

5. **Calibration study.** The convergence analysis (§3.1) determining that ~5 prompt completions reduce relative error below 10% adds methodological rigor and aids reproducibility.

## Weaknesses

### Major

1. **Overlap metrics are structurally misaligned with the faithfulness claims.** The paper uses bag-of-words Jaccard similarity and Manhattan distance to measure whether "deleted content reappears" in final answers and to draw conclusions about whether recovery is "surface-level" or "heuristic." However, these metrics cannot distinguish between: (a) an equation being correctly reconstructed from parametric knowledge versus (b) a superficially similar but incorrect equation sharing vocabulary tokens, or (c) a genuinely equivalent derivation using different variable names. Physics reasoning is inherently structural—equations, algebraic relations, and step order matter—yet these metrics discard all structure. The paper explicitly notes physics is valuable because of its "equations, units, and structured terminology," then uses metrics almost maximally insensitive to that structure. The strong interpretive claims about "shallow recovery," "heuristic reconstruction," and "not faithfully recovered" (§4.2–4.3) are not justified by these metrics alone.

2. **The deletion mechanism and decoding setup are under-specified, undermining the causal claims.** The paper's central causal narrative is that "intercepting" CoT and deleting tokens before the final answer allows measurement of downstream effects. However, it is never explicitly stated whether: (a) the model is *re-run* from the truncated prefix (so it genuinely cannot access the deleted tokens during generation of the final answer), or (b) the model generates the full CoT+answer in one pass, with deletions applied only post-hoc to the scored output. Only (a) would constitute a genuine intervention on the model's conditioning context and support claims about whether the model *needs* the CoT tokens it produced. The paper says "we intercept the scratchpad and remove k% of CoT tokens before the final answer" and "these CoT deletion experiments allow us to assess whether scratchpads are faithfully consumed," but does not describe the implementation. This ambiguity matters because if deletions are post-hoc (only affecting what the judge sees or what is counted), the main claim about CoT bypassability is invalid. At minimum, clarification is essential; if the weaker interpretation is correct, the core contribution is substantially undermined.

3. **Conflation of answer stability under deletion with CoT dispensability/unfaithfulness.** The paper interprets stable accuracy under moderate deletions as evidence that "models remain accurate under heavy deletions" and "not all intermediate steps in the scratchpad are faithfully required." However, an alternative explanation—that models have learned redundant internal representations and can regenerate key information from parametric knowledge—is not adequately ruled out. If the model faithfully used its CoT during generation but can also recover from its removal (due to learned redundancy), that is a property of *robustness*, not necessarily *unfaithfulness*. The paper briefly acknowledges without investigation that models "may draw on internalized physics knowledge or learned solution templates" (§4.1), but this rival explanation is never tested. The stronger claims about "shallow and opportunistic reliance" and CoT being "not a transparent window into model reasoning" go beyond what the deletion experiments alone demonstrate. A true no-CoT baseline (zero-shot direct answer generation on the same problems) would help distinguish between "CoT isn't needed because the model doesn't use it" versus "CoT isn't needed because the model can solve without it"—but no such baseline is reported.

4. **LLM-as-judge scoring without validation is a significant evidential gap.** All quantitative results hinge on Claude-4 Sonnet scoring answers 0–1 on "correctness, derivation accuracy, logic, formatting, and clarity." No inter-rater reliability, comparison to human ground-truth scoring, or error analysis is provided. LLM judges are known to suffer from position bias, verbosity preference, and domain-specific inaccuracies. This is especially concerning because: (a) formatting and clarity scores can change under deletion independently of correctness, potentially creating the appearance of stable accuracy when only style is preserved; (b) the judge may be more lenient toward longer answers, which could conflate "cramming" with genuine correctness; and (c) physics evaluation requires numerical precision that LLMs may not reliably assess. Without any validation, the headline findings about "stable accuracy up to 40–60% deletion" are built on an unvalidated measurement instrument.

### Minor

5. **No qualitative examples or content-level analysis of "cramming."** Despite the behavioral observation that models produce longer final answers under CoT deletion, no examples are provided showing what these lengthened answers actually contain. Are they genuine reconstructions of deleted equations? Disfluent repetition? Alternative (possibly incorrect) solution paths? Without content-level inspection, the "cramming" phenomenon remains a descriptive observation rather than an understood mechanism.

6. **Claude-4 Sonnet is used for both annotation and evaluation.** The physics-aware deletion strategy relies on Claude-4 Sonnet to tag physics-relevant spans, and Claude-4 Sonnet also scores the answers. This creates potential circularity: the same model family identifies what to delete and evaluates the consequences.

7. **No breakdown by problem difficulty or type within benchmarks.** The paper describes benchmarks as varying in difficulty (UG Physics = easiest, PhyBench = hardest) but reports only aggregate trends. If easy factual-recall problems dominate UG Physics and are inherently robust to CoT deletion (because they require little reasoning), this could explain the observed stability patterns without invoking "cramming" as a compensatory mechanism.

### Trivial

8. **Overlap is measured against the entire original CoT, not just the deleted portion.** The information-overlap metrics (§4.2) compare the final answer against the *entire* pre-deletion CoT, not specifically the deleted spans. This confounds baseline similarity (content that was never deleted) with genuine "recovery" (content that was deleted and then reappeared), inflating overlap scores.

## Nice-to-Haves

- **Equation-level or semantic overlap metrics** rather than bag-of-words: parsing and comparing LaTeX equations, or using embedding-based similarity on derivation steps, would substantially strengthen the faithfulness claims and align the metrics with the domain's structure.

- **A true zero-CoT baseline** on the same problems, to distinguish "the model doesn't need the CoT" from "the model can recover from CoT removal."

- **Qualitative case studies** showing original CoT → deleted version → final answer with highlighted recovery/substitution, to move beyond aggregate metrics.

- **At least one larger or closed-source model** to address the generalizability concern, even on a subset of problems.

- **Judge validation** against human annotators on a subsample, with inter-annotator agreement reported.

## Removed Points

1. **"The intervention doesn't actually prevent the model from using its earlier reasoning"** (Harsh Critic, Critical Issue #1 at its strongest). As phrased, this claimed the paper might only be deleting visible text without affecting internal computation. While the deletion mechanism *is* under-specified (see Major Weakness #2), the paper's description—"intercepting CoT mid-generation, removing tokens, and measuring downstream impact"—and "we intercept the scratchpad and remove k% of CoT tokens before the final answer" most naturally suggests re-decoding from the truncated prefix. The criticism about ambiguity is valid and retained, but the strongest version of this claim—that the intervention *definitely* doesn't work—overreaches what we can determine from the text.

2. **"Lack of related work on faithfulness diagnostics"** (Harsh Critic, §6). The related work section does reference Lanham et al. (2023), Turpin et al. (2023), and Barez et al. (2025) on CoT faithfulness. While a more detailed contrast would be welcome, claiming the section is missing these works is inaccurate.

3. **"No numeric results or variance bars in text"** (Harsh Critic, §3.1). This is a formatting/presentation nitpick. The figures contain the relevant data and are referenced appropriately.

4. **"Stopword/punctuation normalization not specified for overlap metrics"** (Harsh Critic, §2.4). This is a minor implementation detail that could affect absolute overlap scores but not the qualitative trends the paper reports.

5. **"Missing experiments across other reasoning domains"** (Spark, Missing Experiment #2). The paper explicitly scopes itself to physics as a structured, high-stakes testbed and acknowledges this limitation (§4.4). Demanding additional domains is scope creep.

6. **"Test early stopping as a strategy"** (Spark, Obvious Next Steps; Neutral Reviewer, Weakness #6). The suggestion that the paper should validate the practical implication about early stopping is reasonable but goes beyond the paper's stated contribution, which is about evaluation methodology rather than inference optimization.

7. **"No confidence intervals on main results"** (Spark, Deeper Analysis #4). The calibration study (§3.1) addresses statistical stability, and single-run evaluations with bootstrap are common in LLM evaluation. This is a nice-to-have, not a core flaw.

## Novel Insights

The "cramming" observation—where models systematically lengthen their final answers as CoT is deleted—is a genuinely novel behavioral finding. While prior work (Lanham et al., 2023) has examined whether models need their CoT traces via early-answering interventions, the specific compensatory pattern of lengthening answer content has not been documented. This suggests that models maintain an internal representation of key reasoning steps that can be externally re-expressed, which has implications for both interpretability (CoT traces may be genuine but redundant outputs of that representation) and efficiency (CoT tokens may be compressible). However, the paper's current metrics cannot determine whether this re-expression constitutes faithful recovery or merely plausible regeneration—this remains an open and important question.

## Suggestions

1. **Clarify the deletion implementation explicitly.** State whether the model re-decodes from the truncated prefix (and if so, describe the generation protocol in detail) or whether deletions are applied only post-hoc. This is critical for the paper's causal claims.

2. **Add equation-level or semantic overlap analysis** as a complement to bag-of-words metrics. For a physics domain, this would leverage the very structure the paper claims makes physics a good testbed.

3. **Report human-validated accuracy on a subsample** alongside the LLM judge scores, to ground the observed stability trends in correctness rather than judge-awarded quality.

4. **Include a true zero-shot (no CoT) baseline** to distinguish between CoT dispensability and CoT recoverability.

5. **Provide qualitative examples** of cramming behavior—side-by-side comparisons of original CoT, deleted CoT, and regenerated final answer—to move beyond aggregate metrics.

## Evaluation

- **Originality**: Moderate. The deletion-sweep methodology extends prior interventions (Lanham et al., 2023; Turpin et al., 2023) with physics-aware deletion and systematic sweep across deletion percentages. The cramming finding is novel. The faithfulness claims based on BoW overlap metrics are incremental relative to prior work.

- **Importance of research question**: High. CoT faithfulness in scientific domains is important and underexplored.

- **Claims well-supported**: Partially. The behavioral observations (stability under deletion, cramming) are well-documented. The faithfulness conclusions are overclaimed relative to what the metrics can show, and the deletion mechanism under-specification creates interpretive ambiguity.

- **Soundness of experiments**: Moderate. The experimental design is reasonable, but the measurement instruments (LLM judge without validation, BoW overlap) and the unclear causal intervention weaken the evidence.

- **Clarity**: Good. The paper is well-structured and clearly written, though the deletion mechanism description needs more detail.

- **Value to community**: Moderate. The cramming observation and deletion framework are useful contributions, but the overclaimed faithfulness conclusions may mislead.

## Score and Decision

**Calibration anchors:**
- *On the Hardness of Faithful CoT Reasoning in LLMs* (1OyE9IK0kx): Similar topic (CoT faithfulness), similar issues (limited models, surface-level metrics, incremental methodology). Scores: 8,3,3,5,5,6 → Reject. This paper has stronger data (3 models × 3 benchmarks) but similarly overclaimed conclusions.
- *A Causal Lens for Evaluating Faithfulness Metrics* (yDICgRUj5s): Similar topic (evaluating faithfulness), similar issue (oversimplified settings for faithfulness claims). Scores: 3,5,3,5,6 → Reject. This paper has a clearer methodology but similar metric limitations.
- *To CoT or not to CoT?* (w6nlcS8Kkn): Accepted poster (6,8,6). Comprehensive empirical study with clear, well-supported findings about CoT scope. Our paper shares the empirical ambition but overclaims relative to its evidence.
- *How LLMs Implement CoT* (b2XfOm3RJa): Similar issue (limited models, broad claims). Scores: 10,6,3,6 → Reject.

This paper shares weaknesses with the rejected CoT faithfulness papers (surface-level metrics, limited models, overclaimed conclusions about internal reasoning from behavioral observations) but makes a genuine empirical contribution (cramming observation, multiple deletion strategies, physics domain focus). It is stronger than the bottom-tier CoT faithfulness papers but falls short of the accepted CoT analysis paper due to overclaimed faithfulness conclusions and measurement gaps.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>