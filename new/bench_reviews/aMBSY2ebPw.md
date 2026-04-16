## Summary
This paper revisits the claim that long-context LLMs can learn translation for extremely low-resource languages from a grammar book, and asks what in the book is actually useful. Its core contribution is a clean ablation separating parallel examples from non-parallel grammatical exposition, showing across Kalamang, Nepali, and Guarani that the translation gains come mainly from the parallel material, while grammatical prose contributes little for MT; it also shows that small MT fine-tuning is competitive and explores typological prompting on more grammar-aligned tasks.

## Strengths
- **The central ablation is exactly the right experiment and is well executed.** Splitting grammar books into `BOOK_para` and `BOOK_non-para` directly tests the paper’s main question rather than accepting the appealing “learn from a grammar book” narrative at face value. The Kalamang results are especially compelling: `BOOK_non-para` is far worse than `BOOK_para`, and `BOOK_all` is only marginally better than `BOOK_para` despite using far more context.
- **The paper makes a substantive and important negative result.** For translation, the evidence supports the narrower claim that long-context prompting benefits primarily from parallel examples and bilingual lexical content, not prose grammatical explanations. This is an important correction to prior interpretation and practically useful for XLR MT.
- **The NMT comparison materially strengthens the paper.** Table 4 shows that fine-tuning NLLB on extracted parallel data is competitive with or better than Gemini prompting on the same data, which turns the paper from a prompt-analysis exercise into a stronger methodological statement about what practitioners should do.
- **The work goes beyond criticism and asks where grammar actually helps.** The shift to grammaticality judgment and IGT prediction is thoughtful and aligned with the nature of grammar books. Even if the positive results there are modest, the paper is stronger for testing a more appropriate use case for grammatical knowledge.
- **The paper is generally clear and well argued.** The experimental setup, motivation, and limitations are mostly easy to follow, and the authors are appropriately explicit in several places about what their results do and do not show.

## Weaknesses
###: Fatal

### Major:
- **The broad resource-allocation recommendation is stronger than the evidence supports.** The paper repeatedly concludes that XLR MT data collection is “best focused on parallel data over linguistic description.” The experiments strongly support this for the tested *prompting paradigm* and settings, especially Kalamang, but the empirical basis is still limited: one truly unseen-language case (`kgv`) with a 100-example test set, plus two seen low-resource languages (`npi`, `gug`) where 0-shot competence is already substantial in some directions. This supports a narrower conclusion about long-context prompting with raw grammar books much more cleanly than a general prescription for multilingual XLR MT resource prioritization.
- **The positive claim that typology helps on linguistic tasks is only modestly supported.** The improvements over the strongest non-typology prompt are small: in Figure 1, `TYP + BOOK_p` is only a few points above `BOOK_p`, and in Table 5 it improves morpheme accuracy from 45.4 to 46.1 over `BOOK_para`. The paper sometimes frames this as “leading” performance or as strong evidence that LLMs can exploit grammar for relevant tasks; that is directionally fair, but the evidence is relatively thin for a strong positive claim.
- **The vocabulary-coverage regression is somewhat overinterpreted.** Section 5.1 shows a meaningful correlation between test-set type coverage and CHRf++ across prompt settings, but the causal language goes too far when claiming performance is “directly modelled” by coverage and that explanations provide no significant advantage beyond it. The prompt settings vary in more than vocabulary coverage: structure, glosses, retrieval characteristics, and example format also differ. The regression is useful supporting analysis, but it does not by itself explain away any possible contribution of grammatical information.
- **The Kalamang test set is still small for some of the paper’s finer-grained claims.** The paper improves over the prior 50-example directional setup by combining to 100 examples, but this is still limited for reading much into sub-point CHRf++ differences or into strong generalization claims. This matters especially where the paper contrasts small margins such as explanation-added vs. parallel-only settings.

### Minor
- **The grammaticality judgment task is only a proxy and has limited interpretability.** The paper explicitly acknowledges: “we cannot guarantee all corruptions are ungrammatical.” That is a reasonable caveat, but it does limit how strongly Figure 1 can be interpreted as evidence of grammar acquisition rather than sensitivity to sentence perturbations.
- **Some conclusions rely on single-run API prompting without uncertainty estimates on the relevant task comparisons.** This is not a fatal reproducibility complaint, and single-run reporting is common, but it weakens confidence in very small gains on the linguistic tasks and in close prompt comparisons, particularly on the 97-example IGT test set.
- **The paper does not fully disentangle whether grammar is unhelpful for MT, or whether raw descriptive book format is a poor interface for LLMs.** The typology prompt partly addresses this, but because it is a high-level structured summary rather than extracted rules from the book, the paper more securely shows that *raw grammar-book prose* is ineffective for translation than that grammatical content in any presentation would be.

### Trivial

## Nice-to-Haves
- A more explicit cost comparison between long-context Gemini prompting and NLLB fine-tuning would strengthen the practical takeaway that fine-tuning is cheaper.
- A small targeted error analysis comparing `BOOK_all`, `BOOK_para`, and `BOOK_non-para` could reveal whether grammatical prose changes error types even when it does not improve overall CHRf++.
- Testing one additional long-context model or a compressed/structured grammar-summary variant would help separate “grammar content does not help MT” from “this presentation of grammar does not help MT.”

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claims about model/tool/dataset existence or release status.** Per instruction, any concern implying cited resources may not exist or are not verifiable is removed.
- **Pure reproducibility complaints rooted in Gemini being API-only.** The paper clearly states the model choice and context-window rationale. While single-model reliance is a legitimate scope/generalization weakness, doubting reproducibility because the model is API-only is not an acceptable core criticism here.
- **Formatting issues around the duplicated `Book_p` label in the extracted Figure 1/table text.** The user explicitly noted PDF extraction artifacts; this is not a paper flaw.
- **Complaints that the paper lacks NMT baselines.** This would be factually wrong: the paper includes NLLB fine-tuning and Llama fine-tuning, and Table 4 is one of its strengths.
- **Claims that evaluation is invalid because there is no human evaluation.** The paper explicitly states why Kalamang human evaluation is infeasible and uses CHRf++ for that reason; this is a limitation, but not a standalone fatal methodological flaw.

## Novel Insights
The paper’s strongest contribution is not merely that grammar books help less than expected, but that it cleanly separates *resource content* from *resource packaging*. The evidence suggests that for MT, long-context LLM prompting behaves more like retrieval over parallel exemplars than like induced learning from linguistic exposition; the same paper resource becomes useful or useless depending on whether it is represented as aligned examples, glossed triples, or prose. That framing is more valuable than the narrower benchmark dispute, because it points toward a broader lesson for XLR NLP: the key question is not whether a language has a grammar book, but whether its information can be transformed into task-aligned supervision.

## Suggestions
- Narrow the main conclusion from a general recommendation about XLR MT data collection to a claim about the tested prompting paradigm: raw grammar-book prose is much less effective than extracted parallel examples for MT.
- Soften the positive typology claims to reflect that the gains on grammaticality judgment and IGT are promising but small.
- Rephrase Section 5.1 so the regression is presented as supporting descriptive evidence rather than a causal explanation that rules out any contribution from grammatical information.
- Add uncertainty estimates or bootstrap confidence intervals for the close comparisons, especially on the 100-example Kalamang MT test set and the 97-example IGT test set.
- Clarify more explicitly that the grammaticality task is a proxy perturbation test rather than a validated benchmark of Kalamang grammatical competence.
- If space permits, include one structured-grammar condition (e.g., compressed rules or extracted paradigms) to better isolate whether the issue is grammatical content or book format.

## Score and Decision
**Originality:** High. The ablation of grammar-book content into parallel vs. non-parallel material is a genuinely useful and novel experimental intervention.  
**Importance of research question:** High. The paper addresses an influential and practically relevant claim about XLR translation and resource usage.  
**Whether the claims are well supported:** Mixed but overall good. The core negative MT claim is well supported; the broader policy conclusion and the positive typology story are somewhat overstated.  
**Soundness of experiments:** Good overall. The design is thoughtful and the NMT baseline is important, though some analyses are overinterpreted and the smallest gains are underpowered.  
**Clarity of writing:** Good to very good.  
**Value to the research community:** High, especially as a corrective to overclaiming about grammar books for XLR MT.

**Calibration against human-reviewed papers:**  
- Compared with **MTOB** (`tbVWug9f2h.md`, scores 6/8/8, accepted spotlight), this paper is less benchmark-creating and less broadly ambitious, but more incisive in causal analysis and stronger in its ablation logic. It is not as obviously high-upside as MTOB, but it is methodologically sharper on the specific question it asks.  
- Compared with **Mufu** (`0eMsrRMmCw.md`, scores 6/6/8, accept poster), this paper feels similar in overall quality: both have meaningful low-resource MT contributions and practical takeaways, but also limits in breadth and some overstatement.  
- Compared with **IntGrad MT** (`SmxM4POTBk.md`, scores 6/3/8/6/3, reject), this submission is stronger because its baselines are more appropriate, its core claim is cleaner, and its practical message is better grounded.  
- Compared with **ASROB** (`sjvz40tazX.md`, scores 8/3/5/6, reject), this paper is stronger and more self-contained: it has a sharper central experiment and a clearer main takeaway, even if its generalization is still limited.

Overall, this paper is above the reject-range anchors because it has a real, well-supported central result and useful practical implications, but below strong-accept papers because it overextends its conclusions and its positive auxiliary claims are modest.

**Score: 7.0**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>