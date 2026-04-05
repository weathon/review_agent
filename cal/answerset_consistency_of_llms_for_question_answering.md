=== CALIBRATION EXAMPLE 42 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately captures the core topic: answer-set consistency for LLM question answering.
- The abstract clearly states the problem, the benchmark idea, the evaluation of multiple LLMs, and the prompting-based mitigation.
- One concern is that the abstract may slightly overstate the strength of the mitigation claim. It says the prompting strategy “leads to improved answer-set consistency,” which is supported in the paper, but the gains are uneven and sometimes come with higher IDK rates and/or tradeoffs (e.g., Table 3). The abstract would be stronger if it acknowledged those tradeoffs.

### Introduction & Motivation
- The problem is well-motivated and timely for ICLR: consistency is a meaningful reliability concern, and enumeration QA is a concrete setting where inconsistency is observable and practically important.
- The gap in prior work is reasonably identified. The comparison to cloze-style consistency, boolean logical consistency, and SQL equivalence work helps situate the contribution.
- The contributions are clearly stated, but one point needs sharpening: the paper frames “answer-set consistency” as a novel notion, yet much of the underlying structure is close to query containment/equivalence and self-consistency ideas from adjacent areas. The novelty is mostly in the benchmark and empirical study, not the formal relation itself.
- The intro does not over-claim overall, but it would benefit from more careful framing around benchmark construction: much of the dataset is manually curated and partly synthetic, so claims about “systematic insights” should remain bounded to the specific benchmark distribution.

### Method / Approach
- The formalization in Section 3.1 is clear at a high level: the paper defines answer-set consistency for equivalence, containment, disjointness, overlap, and a ternary difference relation.
- The reproducibility story is mixed. The high-level pipeline is described, but several critical details are missing or only partially specified:
  - How exactly the dataset tuples were selected from the source KGQA datasets.
  - How ground-truth relations were verified after heavy manual modification.
  - How the LLM-generated questions were validated for objective answer sets.
  - How answer sets were canonicalized before computing equality/Jaccard. This matters greatly for entity surface-form variation, pluralization, aliases, and spelling variants.
- There are logical gaps in the discussion of the consistency/contradiction distinction:
  - Section 3.1 defines answer-set consistency relative to gold relations.
  - Then it defines answer-set contradiction relative to the model’s predicted relation.
  - These are related but different notions, and later results sometimes blend them conceptually. The paper should more explicitly separate “gold-relation consistency,” “prediction accuracy,” and “self-contradiction.”
- The biggest methodological concern is the construction of the benchmark itself:
  - In Section 3.2 and Appendix B, many quadruples are “heavily modified,” with LLM suggestions often not satisfying the intended relations and then being manually fixed.
  - This is acceptable for creating a curated benchmark, but it raises a validity concern: the benchmark may over-represent “clean,” semantically crisp cases that are easier for humans to label than real user queries.
  - Because the questions are manually crafted from inspirations, the dataset may also encode a narrow style of questions, potentially limiting generalization.
- The assumptions behind the ternary relation \(E_{4,1\setminus 3}\) are under-explained. For example, the paper says this relation is an example of a more complex task, but it is unclear whether the output relation is evaluated as a set difference over full answer sets or only over canonicalized entity lists.
- The mitigation strategies are conceptually simple, but the “Oracle” is somewhat synthetic: it conditions on the true relation and then re-asks enumeration. That is fine as an upper bound, but it should be framed explicitly as such rather than as an actionable method.
- On theoretical claims: the paper is not proving new theorems, so there is no proof obligation, but some definitions are somewhat informal and would benefit from tighter notation. The most important issue is not mathematical correctness but operational clarity.

### Experiments & Results
- The experiments largely do test the paper’s claims: classification accuracy assesses whether models can identify relations, enumeration consistency tests whether answer sets satisfy those relations, and the CtE/Oracle variants probe mitigation.
- The choice of baselines is reasonable given the task. Comparing Base, CtE, and Oracle is appropriate for isolating the effect of prompting and relation awareness.
- The main weakness is that the evaluation does not fully establish whether inconsistency is due to the model’s reasoning versus surface-form/entity normalization issues. Since the paper relies on exact set comparison and Jaccard similarity, the metric is highly sensitive to aliasing and partial-name variation. Appendix H mentions terminology variation, but the paper does not quantify how much of the inconsistency is merely canonicalization noise.
- The results support the main qualitative claims:
  - Relation classification can be strong for larger models (Section 4.1).
  - Answer-set consistency is substantially worse than classification accuracy might suggest (Section 4.2).
  - CtE often improves over Base, and Oracle often improves further, though not always.
- However, several result claims need more caution:
  - The statement that CtE “even outperforms the Oracle in many cases” is interesting, but the explanation given is speculative. This could reflect prompt leakage, better self-conditioning, or evaluation artifacts; the paper does not isolate the cause.
  - The claim that newer/bigger models do not universally outperform smaller/older ones is plausible, but the evidence is descriptive rather than controlled. It would be useful to analyze whether differences remain after accounting for IDK rates and model family.
- Missing ablations that would materially strengthen the paper:
  - A normalization ablation to quantify alias/surface-form effects.
  - A prompt-order or prompt-verbosity ablation for CtE.
  - An ablation comparing “classify relation then answer” versus “classify relation and explicitly restate constraints” to see what part matters.
  - A subset analysis by source dataset (LC-QuAD, QALD, QAWiki, Synthetic), since these likely differ in difficulty and cleanliness.
- Error bars / statistical significance:
  - The paper does report McNemar tests, which is good.
  - But the main tables report only point estimates; there are no confidence intervals or variance estimates across runs/seeds. Given the emphasis on stochasticity, that omission is notable.
- The datasets and metrics are appropriate in spirit, but the metrics only partly capture the phenomenon:
  - Jaccard similarity is useful for equivalence and difference relations, but less informative for containment unless interpreted carefully.
  - The binary consistency rate is crisp but hides degrees of near-miss behavior.
- ICLR-level standard: the benchmark and empirical study are solid, but the paper would be stronger if it more convincingly demonstrated that the observed failures are not primarily due to alias normalization or benchmark curation artifacts.

### Writing & Clarity
- The main narrative is understandable, and the paper does a good job of motivating the topic and defining the high-level tasks.
- That said, some sections are difficult to follow because the notation and task definitions accumulate quickly. The reader has to reconstruct the experimental protocol across Sections 3.1–3.4, Table 2, and Appendix A/B.
- The biggest clarity issue is not grammar but operational ambiguity:
  - What exactly constitutes an “answer set” after parsing model output?
  - How are duplicates, aliases, order, and punctuation handled?
  - How are “empty answer” and “idk” treated in the metrics versus the consistency judgments?
- Tables 3 and 4 are informative, but the interpretation of the columns would benefit from a more explicit walkthrough, especially for the control relation \(E_{1,*}\) and the ternary relation \(E_{4,1\setminus 3}\).
- The discussion of why \(N_{4,1}\) is harder than \(N_{3,1}\) is plausible but somewhat underdeveloped; it would help to tie this more concretely to the construction of \(Q_4\) versus \(Q_3\).

### Limitations & Broader Impact
- The paper does acknowledge several limitations in Section 5 and the appendices: restricted single-turn setting, English-only data, static factual domains, and the possibility of extending to cardinality.
- The most important limitation that is acknowledged only partially is benchmark generality. Because the dataset is heavily curated and often manually adapted from KGQA sources, it may not represent the diversity of natural user enumeration questions.
- A key missed limitation is entity normalization and aliasing. For enumeration tasks, this is central: many apparent inconsistencies may come from different acceptable surface realizations of the same entity, not true semantic inconsistency.
- Another limitation is that the benchmark tests only a narrow family of set relations, and mainly on knowledge-base-style factual questions. It remains unclear how well the findings transfer to open-domain, subjective, or temporally evolving domains.
- Broader impact is mostly positive: the paper provides a useful diagnostic benchmark and highlights reliability issues. But the broader implications for deployment should mention that encouraging models to “reason about relations” may improve apparent consistency without improving factual accuracy. That tradeoff matters for users relying on answer completeness.
- The paper does not discuss possible negative impacts, such as overconfidence in prompting-based mitigation or misuse of consistency metrics as proxies for truthfulness.

### Overall Assessment
This is a timely and genuinely interesting paper that identifies an underexplored failure mode of LLMs: inconsistency across enumeration questions whose answers should obey simple set relations. The benchmark and empirical analysis are likely to be of interest to ICLR readers, and the paper’s core finding—that strong relation classification does not guarantee consistent enumeration—is persuasive. The main concerns are methodological rather than conceptual: the benchmark is heavily curated, the evaluation appears sensitive to aliasing/canonicalization issues, and the mitigation results would be more convincing with stronger ablations and clearer operational details. Despite these limitations, the contribution stands as a useful empirical study, but to meet ICLR’s stronger bar the paper should more rigorously separate true logical inconsistency from surface-form and evaluation artifacts, and it should better quantify generalization beyond the curated benchmark.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces answer-set consistency as a new evaluation dimension for LLMs on enumeration question answering: if two questions have a known set-theoretic relation between their true answer sets, do the model’s generated answer sets respect that relation? The authors build ASCB, a handcrafted benchmark of 600 quadruples (2,400 questions) drawn from KGQA sources and synthetic generation, and evaluate 18 LLMs under base, classify-then-enumerate, and oracle prompting variants. The main empirical finding is that even strong frontier models often produce inconsistent answer sets, especially for containment and ternary relations, while prompting the model to reason about the relation first substantially improves consistency.

### Strengths
1. **Clearly articulated and timely problem formulation.**  
   The paper identifies a real failure mode of LLM QA that is distinct from answer accuracy: two answers can each be plausible in isolation yet violate obvious set relations across questions. This is a meaningful and underexplored axis of reliability, and it fits ICLR’s interest in model behavior, robustness, and reasoning.

2. **Useful formalization of answer-set consistency and contradiction.**  
   The paper provides a concrete definition of consistency for equivalence, containment, disjointness, overlap, and a ternary set-difference relation. This makes the phenomenon measurable rather than anecdotal and helps distinguish “wrong answer” from “internally inconsistent answers.”

3. **Benchmark creation is substantial and manually curated.**  
   ASCB contains 600 quadruples / 2,400 questions from multiple sources, with manual revision by three authors. For ICLR, a benchmark contribution can be valuable if it is well-motivated and carefully curated, and the paper does put effort into filtering for objective, non-empty enumeration questions with controlled cardinalities.

4. **Empirical evaluation is broad across models and prompting strategies.**  
   The paper evaluates 18 LLMs spanning several families and sizes, which provides a reasonably wide snapshot of current behavior. The inclusion of a control condition (re-asking the same question) is also a nice touch for disentangling stochasticity from semantic inconsistency.

5. **Mitigation is simple but effective.**  
   The classify-then-enumerate strategy is easy to understand, practically implementable, and often improves consistency significantly; the Oracle variant gives an upper-bound-style comparison. This makes the paper more than just a benchmark paper, since it also offers an actionable prompting idea.

### Weaknesses
1. **The benchmark construction appears heavily manual and may limit scalability and generality.**  
   The dataset is carefully curated, but much of the process relies on manual modification of LLM-generated candidates and author judgment. While this is not inherently bad, the paper does not fully quantify inter-annotator agreement, error rates, or how often LLM suggestions were rejected. For an ICLR benchmark paper, stronger evidence of dataset reliability and construction robustness would be expected.

2. **Ground-truth relation labeling is partly derived from the authors’ judgment rather than a fully formal source.**  
   Many relations are inferred from question semantics, and the paper explicitly says the dataset is handcrafted and heavily revised. This raises concern about borderline cases where “equivalence,” “containment,” or “disjointness” may be debatable, especially across knowledge sources and paraphrases. The paper would benefit from stronger verification of the relation labels, ideally via structured queries or external validation.

3. **Evaluation of consistency depends on exact-set matching, which may be brittle.**  
   The metric seems to treat answer sets as literal sets of surface forms, while the paper itself notes terminology variation such as “Spain” vs. “Kingdom of Spain.” This can conflate genuine semantic consistency with formatting or lexical variation. The paper does discuss this qualitatively, but the metric design still appears vulnerable to over-penalizing correct answers expressed differently.

4. **Some reported results are hard to interpret because performance is entangled with abstention/IDK behavior.**  
   The paper notes that stronger consistency can come with higher %IDK, and that CtE may outperform Oracle partly because it induces safer abstention. This complicates the interpretation of improvements: better consistency does not necessarily mean better QA behavior. The paper would benefit from a more explicit joint analysis of consistency, completeness, and usefulness.

5. **Statistical analysis is not fully convincing.**  
   The use of McNemar tests is reasonable for paired binary outcomes, but the paper reports many comparisons across models and relations without clearly discussing multiple-comparison correction. It also relies on aggregate rates where confidence intervals or effect sizes would help interpret practical significance.

6. **The paper’s novelty is solid but not transformative for ICLR.**  
   The idea is interesting and useful, but it is primarily an evaluation/benchmark paper plus a prompting baseline. The mitigation method is relatively modest, and the conceptual leap beyond known consistency studies is incremental rather than a new learning method or a deeper theoretical framework. That may still be acceptable at ICLR if the benchmark is strong and the empirical insight is compelling, but the acceptance bar is high.

7. **Reproducibility is only moderate from the paper alone.**  
   The paper mentions code and data availability, which is good, but the dataset is partially handcrafted and prompt-driven, and the exact selection/modification workflow is complex. Without a very detailed release, reproducing ASCB exactly may be difficult. The paper also evaluates many proprietary and changing model versions, which reduces long-term reproducibility of the reported scores.

### Novelty & Significance
The paper’s novelty is moderate to good: it introduces a new, clearly defined consistency notion for enumeration QA and a corresponding benchmark targeting relations beyond simple equivalence. This is a meaningful extension over prior consistency work on cloze facts, boolean logic, or paraphrase consistency, and the empirical findings are interesting in showing that even frontier models can be highly inconsistent despite good relation-classification performance.

In terms of significance, the work is relevant to ICLR because it studies a practical reliability failure in LLMs and proposes a simple mitigation that improves behavior. However, the contribution is more diagnostic than fundamentally methodological, and the benchmark’s handcrafted nature plus the modest mitigation strategy may make it somewhat narrower than the strongest ICLR papers. Overall, I would judge it as a valuable but not clearly top-tier breakthrough contribution.

### Suggestions for Improvement
1. **Add stronger validation of the benchmark construction.**  
   Report annotation guidelines, inter-annotator agreement, rejection rates for candidate questions, and examples of ambiguous cases. If possible, include external verification of the intended set relations using structured queries or database-backed checks.

2. **Introduce semantic normalization or fuzzy matching in evaluation.**  
   To reduce brittleness, evaluate not only exact string-set equality but also normalized forms, aliases, and possibly entity-linked answers. At minimum, report how often “inconsistent” cases are due purely to lexical variation.

3. **Separate consistency from usefulness more explicitly.**  
   Report joint metrics that combine consistency, completeness, and abstention. For example, analyze whether a strategy improves consistency by producing better answers or merely by returning more “idk” responses.

4. **Strengthen the statistical reporting.**  
   Include confidence intervals, effect sizes, and multiple-comparison correction across the many model/relation tests. This would make the significance claims more robust and ICLR-reviewer-friendly.

5. **Analyze error categories more systematically.**  
   The appendix offers qualitative causes like terminology mismatch and incompleteness, but a structured breakdown of failure modes by relation and model family would better support the paper’s causal claims.

6. **Compare against stronger baselines beyond prompting.**  
   Since the paper is about consistency, it would be useful to compare against constrained decoding, self-consistency-style aggregation, retrieval augmentation, or symbolic post-processing. Even a lightweight baseline that normalizes outputs or enforces set constraints could sharpen the contribution.

7. **Clarify the scope and limitations of the relation taxonomy.**  
   Explain more precisely which kinds of enumeration questions are covered and which are excluded, and discuss whether the benchmark generalizes to open-domain, ambiguous, or multi-hop questions. This would help readers understand when the results apply.

8. **Make the reproducibility package extremely explicit.**  
   Provide exact prompts, model versions, query timestamps, random seeds if available, and a script for reconstructing every reported metric. Given the reliance on external model APIs and handcrafted data, this is especially important.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a strong human-annotated or independently verified validation set for the question relations and answer-set correctness. Right now the benchmark is heavily hand-curated with LLM assistance, so the core claim that these are well-defined set relations is not yet trustworthy enough for ICLR’s bar.

2. Compare against non-LLM baselines and extraction-based systems, not just other LLMs. The paper claims practical reliability issues for enumeration QA, but without baselines like KG/SPARQL execution, retrieval-augmented extraction, or simple symbolic set operators, it is unclear how much of the error is inherent to LLMs versus the task setup.

3. Run ablations isolating what actually causes the improvement from CtE and Oracle: relation classification, extra context, self-reasoning, and “idk” usage. Without these ablations, the paper cannot support the claim that prompting the model to reason about relations is the mechanism behind the gain.

4. Evaluate robustness across decoding settings and repeated runs per prompt, especially for the same input under fixed seeds and multiple seeds. Since the paper attributes much inconsistency to stochasticity, it needs direct evidence quantifying variance across runs; otherwise the control relation is not a convincing diagnostic.

5. Test transfer beyond the current KGQA-derived English benchmark, such as on other enumeration QA datasets or multilingual settings. ICLR will expect evidence that the phenomenon and mitigation are not specific to this handcrafted benchmark.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify benchmark contamination and memorization risk. Many questions are derived from popular KGQA datasets and LLM-generated variants, so the paper must analyze whether models are recalling training data or dataset artifacts rather than exhibiting genuine reasoning failures.

2. Provide error analysis broken down by relation difficulty, answer set size, and lexical overlap between paired questions. The paper currently asserts that containment and ternary relations are harder, but does not explain whether this is due to logical complexity, surface-form similarity, or answer cardinality.

3. Separate inconsistency caused by answer normalization from true semantic mismatch. Different surface forms like “Spain” vs. “Kingdom of Spain” can inflate inconsistency, so the paper needs normalization and equivalence-checking analysis before claiming logical failure.

4. Analyze how often classification is correct but enumeration still fails, versus classification failure driving enumeration failure. This distinction is central to the paper’s story, but the current evidence is too coarse to trust the causal interpretation.

5. Report uncertainty calibration for both relation classification and enumeration refusal. The high %IDK rates suggest the models may be gaming the metric by abstaining, so the paper needs analysis of whether improved consistency is coming from better reasoning or more conservative behavior.

### Visualizations & Case Studies
1. Add side-by-side examples of cases where Base fails but CtE or Oracle succeeds, and where CtE fails despite correct classification. These are necessary to show whether the mitigation really changes reasoning or just changes refusal behavior.

2. Show per-question-set traces with the predicted relation, generated answers, normalized answer sets, and the resulting set relation. This would expose whether the model is internally consistent, semantically correct, or merely producing superficially similar outputs.

3. Include plots of consistency versus answer-set size and versus lexical overlap between Q1/Q2/Q3/Q4. These would reveal whether the reported gains are driven by easier small sets or by genuine relational reasoning.

4. Provide repeated-run variance plots for the control relation E1,* across models. This would directly reveal how much inconsistency is due to stochastic decoding versus prompt sensitivity or model instability.

### Obvious Next Steps
1. Add a fully automated evaluation pipeline that uses an external verifier to check answer-set relations from the model outputs. ICLR readers need to see that the benchmark can be evaluated independently of model self-reporting.

2. Extend the benchmark to more relations and harder composition patterns, especially nested containment, partial overlap, and multi-hop set operations. The current paper is still too narrow to support broad claims about “answer-set consistency” in general.

3. Test whether structured outputs or constrained decoding outperform prompting-only methods. If the goal is reliability, the next obvious step is to compare prompt-based mitigation against methods that directly enforce set-formatted outputs.

4. Provide a contamination-aware benchmark release protocol with held-out construction rules. Without this, the dataset risks becoming another curated QA benchmark that models can memorize rather than a reliable diagnostic of reasoning behavior.

# Final Consolidated Review
## Summary
This paper studies a concrete and underexplored failure mode of LLM question answering: when two enumeration questions should stand in a simple set relation, the model’s generated answer sets often violate that relation. The authors formalize this as answer-set consistency, construct ASCB with 600 curated quadruples from KGQA sources plus synthetic data, and evaluate 18 LLMs under base, classify-then-enumerate, and oracle prompting.

## Strengths
- The paper identifies a real and practically relevant reliability issue that is different from ordinary QA accuracy: models can answer two related questions plausibly yet still violate equivalence, containment, or disjointness between their answer sets.
- The benchmark and analysis are broad in scope for this topic: ASCB covers 600 quadruples, multiple source datasets, 18 LLMs, a control condition for repeated-question stochasticity, and a mitigation study showing that relation-aware prompting often improves consistency.

## Weaknesses
- The benchmark construction is heavily hand-curated and partially LLM-assisted, but the paper does not provide enough validation to fully trust the intended relations at ICLR standard. Many candidate questions were manually revised or discarded, and the paper gives no strong independently verified validation set, inter-annotator agreement, or formal confirmation that the final set relations are unambiguous across all items.
- The evaluation is brittle with respect to answer normalization and surface-form variation. The paper itself notes terminology inconsistencies like “Spain” versus “Kingdom of Spain,” yet the reported consistency metrics appear to rely on exact set matching; this can inflate apparent inconsistency and makes it hard to tell how much of the failure is true relational reasoning versus aliasing, formatting, or canonicalization noise.
- The mitigation story is not fully convincing mechanistically. CtE and Oracle do improve consistency, but the paper does not isolate whether the gain comes from relation reasoning, extra context, safer abstention, or simply higher IDK rates. In particular, the reported improvement can come with more refusals, so “better consistency” does not necessarily mean better QA behavior.
- The statistical reporting is incomplete for such a comparison-heavy study. McNemar testing is appropriate, but the paper reports many model/relation results without confidence intervals, effect sizes, or any correction discussion for the large number of comparisons.

## Nice-to-Haves
- A normalization ablation that quantifies how much inconsistency disappears after entity linking, alias resolution, and surface-form canonicalization.
- A more systematic breakdown of failures by relation type, answer-set size, and lexical overlap between paired questions.
- Side-by-side case studies showing where Base fails but CtE succeeds, and where CtE fails despite correct relation classification.

## Novel Insights
The most interesting insight is that relation classification and answer-set consistency are meaningfully different abilities: several models can identify the correct set-theoretic relation with high accuracy yet still generate answer lists that violate that same relation. This suggests that “knowing the relation” and “producing outputs that obey the relation” are not the same capability, and that prompting the model to reason about the relation first can partially bridge the gap, though often by shifting behavior toward abstention rather than guaranteeing genuinely better enumeration.

## Potentially Missed Related Work
- Zhao et al. 2023, LLMSQL-Solver — relevant because it studies whether LLMs can determine SQL equivalence, which is closely related to the paper’s question-relations setting.
- Wei et al. 2025, Equibench — relevant because it evaluates reasoning about program semantics via equivalence checking, a nearby formal-relational benchmark.
- Liu et al. 2024c, Aligning with Logic — relevant because it studies logical consistency in LLMs across structured relations between predictions, offering a closer conceptual comparison than paraphrase-only consistency work.

## Suggestions
- Add a human-verified or independently audited subset of ASCB and report validation details, including ambiguous cases and rejection criteria.
- Re-evaluate consistency with canonicalized answers and alias-aware matching, then report how much of the error remains after normalization.
- Separate “consistency,” “completeness,” and “abstention” in the analysis so that improvement from prompting cannot be mistaken for improved answer quality.
- Include ablations that decouple relation classification, self-reasoning, and the effect of allowing “idk,” since these are currently conflated.


# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 6.0, 4.0]
Average score: 4.0
Binary outcome: Reject
