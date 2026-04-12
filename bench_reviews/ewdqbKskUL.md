## Summary
This paper formalizes **answer-set consistency** for LLMs answering enumeration questions: if two questions are known to stand in relations such as equivalence, containment, disjointness, or set difference, then the model’s returned answer sets should respect those relations. To study this, the authors build a 600-quadruple benchmark (ASCB), evaluate 18 LLMs, and test prompting-based mitigations, finding that inconsistency is common even for strong models and that relation-aware prompting can improve measured consistency.

## Strengths
- **The paper identifies a genuinely distinct failure mode and formalizes it cleanly.** The distinction between (i) violating the gold relation between question pairs and (ii) **self-contradicting** a relation the model itself predicts is useful and sharper than prior “consistency” notions built around single-answer QA or boolean statements. Section 3.1 gives a clear set-theoretic formulation, and Appendix F extends this to internally contradiction-free behavior.
- **The benchmark design is more structured than typical paraphrase-consistency probes.** ASCB is built around quadruples \((Q_1,Q_2,Q_3,Q_4)\) that induce multiple relations at once—equivalence, narrower containment, disjointness, and a ternary set-difference relation \(E_{4,1\setminus 3}\). This gives the paper leverage to compare relation types rather than only paraphrase equivalence.
- **The empirical picture is richer than a simple leaderboard.** The paper does not just report one aggregate metric; it separates classification accuracy, relation-specific consistency, Jaccard similarity, refusal/empty-response rates, and a repeated-query control \(E_{1,*}\). This supports the central claim that answer-set inconsistency is pervasive and relation-dependent.
- **One specific finding is quite interesting:** models can often **recognize** the intended relation much better than they can **produce answer sets that satisfy it**. This gap between relation classification and consistent enumeration is one of the more compelling takeaways of the study.
- **The mitigation experiments are useful diagnostically, even if not fully convincing as a deployable solution.** The CtE and Oracle settings provide evidence that prompting the model to reason about relations before answering can alter behavior substantially, especially on harder relations.

## Weaknesses

### Major:
- **The main mitigation claim is substantially confounded by the evaluation protocol excluding refusals/empty answers from consistency metrics.**  
  This is the most important issue. In Section 3.4, the paper defines consistency rates while explicitly excluding empty answers and `"idk"`: “**Here we exclude empty answer sets and responses of ‘idk’, which are reported separately.**” In Section 4.2, the authors themselves note that under CtE, “**LLMs tend to adopt a safer approach by answering ‘idk’ when uncertain, which may explain why CtE outperforms the other two strategies.**”  
  This means CtE can improve reported consistency by declining a larger share of difficult cases rather than by actually producing more logically coherent answers on the full benchmark. Table 3 shows this tradeoff clearly for several models (e.g., GPT-5, GPT-4o, Mistral-small, GPT-oss-20b). As a result, the headline conclusion that CtE “mitigates” inconsistency is only partially supported: it improves **conditional consistency on answered cases**, but the paper does not establish improvement in a refusal-aware end-to-end metric over the full evaluation set. For a reliability paper, this is a significant weakness.

- **The answer-set extraction/evaluation appears too brittle to surface-form variation, and the paper does not convincingly quantify how much this inflates inconsistency.**  
  The benchmark treats outputs as sets and computes exact set-based relations plus Jaccard similarity over extracted answers. But the paper’s own error analysis in Appendix H acknowledges that models often use different names for the same entity, e.g. “**Spain**” versus “**Kingdom of Spain**.” That is not a minor edge case; it directly affects both exact consistency and Jaccard-based evaluation.  
  While the prompt tries to reduce variability by asking for full names and a pipe-delimited exhaustive list, the paper does not describe a robust entity normalization or semantic matching stage before scoring. Appendix A.4 describes storing responses and computing metrics, but not a serious canonicalization pipeline. Without this, some portion of the reported inconsistency is likely due to lexical variation rather than genuine logical failure. This does not invalidate the phenomenon, but it weakens the absolute interpretation of the reported inconsistency rates and especially cross-model comparisons.

- **The paper over-interprets the \(E_{1,*}\) control as evidence about causes (“stochasticity” vs “semantic misunderstanding”).**  
  The control is useful, but the causal claims are stronger than what the design cleanly supports. The paper defines \(E_{1,*}\) as asking “**the same question \(Q_1\) posed in a different context at a different time**” (Table 2), then in Section 3.4 states that comparisons involving \(E_{1,*}\) can help assess the role of stochasticity versus semantic misunderstanding. However, “different context at a different time” changes more than just generation stochasticity; it also allows differences due to conversational context effects, backend nondeterminism, serving changes, or other environmental factors.  
  More importantly, the gap between \(E_{1,*}\) and \(E_{1,2}\) does not isolate “semantic misunderstanding” of set-theoretic relations, because \(E_{1,2}\) also introduces paraphrase sensitivity and retrieval/recall differences between phrasings. The paper’s analysis here is suggestive, not conclusive, and should be framed more cautiously.

- **A factual-accuracy/completeness baseline is missing, which limits how to interpret consistency as a reliability measure.**  
  The paper explicitly states in Section 3.1 that “**We do not need ground-truth answer sets for questions in order to analyze answer-set consistency.**” That is true for measuring internal consistency, but it also means a model can be consistently wrong, consistently incomplete, or consistently overcautious. The current results show that consistency is low, but they do not show whether more consistent models are actually better at correct exhaustive enumeration. Since the paper motivates consistency as improving reliability for QA, some factual accuracy/completeness anchor would materially strengthen the claims.

### Minor
- **The benchmark scope is intentionally narrow, which is acceptable, but it limits generality.**  
  The dataset focuses on English, factual, relatively “crisp” enumeration questions with 2–100 answers, largely inspired by KGQA sources and substantial manual curation. This is appropriate for a first benchmark, but the conclusions should remain scoped to this regime rather than broader “LLM question answering” generally.

- **The mitigation comparison is incomplete without a more standard reasoning baseline.**  
  CtE is compared to Base and Oracle, but not to a simpler generic reasoning prompt (e.g., “think step by step before listing answers”). Without such a baseline, it is unclear how much of the gain is specific to relation classification versus simply eliciting more deliberate answering behavior.

- **There is limited analysis of how difficulty scales with answer-set size or domain.**  
  Since the dataset spans answer sets from 2 to 100 entities, a stratified analysis by cardinality could reveal whether failures are mainly set-reasoning failures or simply list-completeness degradation on larger answer sets. Likewise, domain-wise analysis could help distinguish knowledge gaps from relation reasoning failures.

- **The statistical significance analysis is stronger on p-values than on practical effect characterization.**  
  The McNemar tests are reasonable for paired binary outcomes, but given the dataset size, very small effects can become highly significant. The paper would benefit from fuller reporting of refusal-aware effect sizes and practical deltas, not only p-values.

### Trivial
- **There is a correctness issue in the description of disjointness/Jaccard.**  
  In Section 4.2 the text says, “**for \(D_{3,4}[SIM]\), a score lower than 0 for Jaccard similarity is better**,” which is mathematically impossible since Jaccard similarity is in \([0,1]\). The intended meaning is clearly that **closer to 0 is better** for disjointness. This is easy to fix, but it should be corrected.

## Nice-to-Haves
- Add a **refusal-aware primary metric** over all benchmark instances, e.g., treating `idk`/empty as failures for end-to-end consistency, or reporting coverage-consistency curves.
- Add a **canonicalization/normalization layer** (entity matching, alias resolution, or semantic equivalence adjudication) and re-run a sensitivity analysis to estimate how much inconsistency comes from lexical variation.
- Include a **generic CoT/reasoning baseline** to determine whether CtE’s gains are specific to explicit relation classification.
- Provide analysis by **answer-set cardinality**, **domain**, and **relation misclassification vs enumeration failure**, which would make the causal story much sharper.
- Add a modest **accuracy/completeness check** on a subset of the benchmark to connect internal consistency to actual QA reliability.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Dataset construction quality concerns because LLMs were used during curation, implying contamination/unverifiability.”**  
  The paper is explicit that LLMs were used for suggestion and filtering, but also repeatedly states that the final dataset was **heavily manually revised and curated by three authors**. This is a valid scope/generalization concern, but not evidence of contamination or invalidity by itself.

- **“The Oracle strategy is unrealistic, therefore a weakness of the paper.”**  
  The paper already presents Oracle as an **ideal diagnostic upper bound**, not as a deployable method: “**This task is an ideal version of Task 2... This will give us insights into what the model could achieve**.” Criticizing it for not being directly deployable misunderstands its role.

- **“Prompt formatting strictness is itself a core flaw.”**  
  The strict output format could interact with parsing, and the lack of a detailed normalization/parsing description is a fair concern. But simply objecting that models may struggle with pipe-delimited outputs is not, by itself, a substantive weakness.

- **“Unfair comparison because some baselines are treated asymmetrically.”**  
  No concrete unfairness of this kind was substantiated in the paper text.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s strongest contribution is probably **diagnostic rather than mitigative**. The benchmark exposes a real gap between (a) recognizing set relations between questions and (b) actually producing answer sets that obey those relations. That gap suggests LLM failures here are not reducible to simple ignorance: models can often state the right relational structure yet fail to realize it in generated enumerations. At the same time, the current mitigation results indicate that “improving consistency” is entangled with abstention behavior, so the paper is most compelling as a study of a new failure mode and benchmark, less so as evidence that the proposed prompting strategy solves it.

## Suggestions
- Redefine the main mitigation evaluation around a **full-coverage, refusal-aware metric** and move the current exclusion-based CON metric to a secondary analysis.
- Implement and report an **entity normalization sensitivity study**; even a manually validated subset would help establish how much of the inconsistency is semantic vs lexical.
- Soften the causal claims around **stochasticity vs semantic misunderstanding** unless supported by tighter controls.
- Add at least one **generic reasoning prompt baseline** to contextualize CtE.
- Report **cardinality-stratified** and, if feasible, **domain-stratified** results.
- Include a **small factual correctness/completeness audit** so that consistency can be tied to practical reliability rather than only internal coherence.

