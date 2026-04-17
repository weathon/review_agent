---
job_id: 66f83f67-2d0b-4144-9f0c-5315deed280a
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: IU4rqTlpRb.pdf
paper: Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on LLM unlearning, benign relearning, and syntactic properties of data, which fits squarely under representation learning, safety/privacy, and evaluation/benchmarks for large-scale language models.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method/Approach, Experiments, Results, Discussion/Conclusion) are present and reasonably detailed. The work is technically coherent, clearly written, and supported by nontrivial experiments; I see no fatal methodological or theoretical flaws that would warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no evidence of prompt injection, hidden instructions to LLM reviewers, or other manipulative content in the manuscript.

---

# Expected Review Outcome:

## Summary

The paper studies *benign relearning* for LLM unlearning: the phenomenon that forgotten content can resurface when an unlearned model is later fine-tuned on ostensibly benign data. Re-examining the BLUR benchmark and conducting controlled experiments on TOFU and other settings, the authors argue that *syntactic similarity* between the relearn and target sets, rather than topical relevance, is the primary driver of relearning. Based on this diagnosis, they propose *syntactic diversification* of the forget set, implemented via paraphrasing queries into diverse surface forms, which is shown to reduce relearning, speed up forgetting, and alleviate the utility–forgetting trade-off.

## Strengths

1. **Clear, focused conceptual contribution (syntax vs topicality).**  
   The main conceptual claim is crisp: benign relearning is largely driven by syntactic similarity rather than topical relevance. This is not the default intuition in the current unlearning literature, which has focused on topical overlap. The paper builds a consistent empirical story for this alternative explanation across several setups (BLUR’s WMDP/WHP/RWKU; TOFU; WHP “Who’s Harry Potter?”; WMDP ACE example).

2. **Careful re-analysis of BLUR benchmark with a key protocol correction.**  
   Section 4 and **Figure 3** dissect BLUR’s experimental protocol, noting that earlier conclusions about topical relevance are confounded by unequal dataset sizes and reporting at a single epoch. The authors standardize the step budget and report the *best* ROUGE-L over steps, showing in **Figure 2** that recovery under \(D_{\text{hi}}, D_{\text{mid}}, D_{\text{low}}\) becomes similar, especially on WHP and RWKU. This is a concrete and important correction that others in the area will care about.

3. **Nice use of controlled synthetic design on TOFU to separate topical and syntactic effects.**  
   The construction in Section 5.2 of \(D_{\text{relearn}}^{\text{topic}}\) vs \(D_{\text{relearn}}^{\text{syntactic}}\) is well thought out: same target authors vs different authors, matched QA format, with explicit Levenshtein similarity scores. **Figure 4** shows a stark contrast, where syntactic relearn sets consistently cause much higher relearn success rates than topical ones across GA, NPO, and SCRUB and across unlearning steps. This is a strong empirical backbone for the paper’s central claim.

4. **Representation and gradient alignment analysis is insightful.**  
   Section 6’s analysis with **Figure 5** directly links syntactic similarity to model internals: syntactically similar relearn sets have higher cosine similarity in last-token hidden states and gradient directions relative to the target set. The triple-bar plots (representation similarity, gradient similarity, relearn success) per dataset and method help connect the observed behavior to optimization geometry instead of leaving it as a purely phenomenological observation.

5. **Template vs keyword decomposition and loss-ratio diagnostic.**  
   The template/keyword separation and the loss ratio  
   \[
   \text{Loss Ratio} = \frac{\mathcal{L}_{\text{template}}}{\mathcal{L}_{\text{keyword}}}
   \]  
   are clever diagnostics. **Figure 6** (and **Figure 9 (Top)**) clearly show that, under standard unlearning on TOFU, the loss ratio grows, meaning templates get disproportionately suppressed compared to the name keywords. The causal “template injection” experiment in Appendix F, with **Figure 17**, further corroborates that the forgotten names are still easily recalled if you manually provide the template. This is a sharp, non-obvious insight about how current forget losses behave at the token level.

6. **Simple but effective mitigation via syntactic diversification.**  
   The proposed solution, syntactic diversification, is technically simple (paraphrase the forget queries into diverse syntactic forms) but directly addresses the diagnosed failure mode. **Figure 7** illustrates the transformation from homogeneous TOFU queries (“What is the full name of the author born in …?”) to varied templates. **Figure 8** shows that, after diversification, relearning success rate under syntactic relearn sets is drastically reduced across relearning steps. **Figure 9 (Bottom)** and **Figure 18** further show reduced relearning and similar effectiveness even when using a cheaper generator (Llama-3-8B). The intervention is appealingly practical.

7. **Utility–forgetting trade-off improvement is supported by quantitative results.**  
   **Table 2** quantitatively compares model utility on Real Authors, World Facts, and Retain sets, using ROUGE, probability, and truth ratio. The diversified forget set \(D_{\text{forget}}'\) improves average scores substantially (e.g., Retain avg 0.1607 → 0.3128), suggesting that diversification lets you forget more robustly with *fewer* unlearning steps, limiting collateral damage.

8. **Breadth of empirical evidence and cross-metric checks.**  
   The paper does not restrict itself to a single base model or evaluation metric. There are experiments with Zephyr-7B, Llama-2-7B, Llama-3-8B, Phi-1.5B, and both full-parameter and LoRA unlearning/relearning, as well as comparisons of keyword, cosine similarity, and LLM-judge leakage metrics (**Table 6, Table 7**). This breadth makes the central observation about syntactic similarity more convincing.

9. **Discussion sections on safety training and PEFT are useful.**  
   Appendix E’s comparison between IDK/DPO vs GA/NPO under syntactic relearning (**Figure 16**) is informative: safety training looks particularly brittle, which is important for practitioners conflating “refusal training” with true unlearning. The observation that LoRA-based relearning recovers forgotten data faster than full fine-tuning (**Figures 10–12**) is also a notable practical warning.

## Weaknesses

1. **Heavy dependence on synthetic / stylized benchmarks, especially TOFU, to support the central claim and proposed method.**  
   Although the authors include BLUR, WHP, and WMDP experiments, the strongest evidence for “syntax dominates topicality” and for the effectiveness of syntactic diversification is based on TOFU’s highly templated, synthetic QA. The template–keyword analysis, loss ratio, and diversification experiments are all on that setting. The WHP “Who’s Harry Potter?” experiment in Appendix C and WMDP ACE setup in Appendix D are valuable, but they are small-scale and somewhat hand-crafted, and they do not test diversification. This raises questions about external validity:  
   - How does syntactic diversification behave on more natural, long-form forget requests like copyrighted passages (e.g., Harry Potter paragraphs) or real-world personal data?  
   - Would the same “template predominant suppression” pattern in **Figure 6** and **Figure 17(a)** hold up when the forget set is not dominated by a single rigid QA template?  
   The paper would be stronger with at least one large, real-data scenario where diversification is shown to work, not only diagnosis.

2. **Syntactic similarity measurement is simplistic yet central, and its limitations are under-discussed.**  
   The formal definition in Section 5.1 is normalized Levenshtein distance at the character level. While the authors do cross-check with POS-based template mining and parse-tree similarity in Appendix I (**Table 8, Table 9**), there is no clear analysis of when Levenshtein is misleading (e.g., in long or multi-sentence passages, or under synonym substitutions that change length/characters strongly but preserve syntax). Since **Table 1** and multiple arguments rely on averages of this score across datasets, a deeper discussion is needed on:  
   - Whether token-level or syntactic-structure metrics might better capture the relevant notion than character-level edits.  
   - How sensitive the conclusions in **Table 1** are to the similarity metric used; right now this is addressed only indirectly in the appendix.

3. **Some experimental designs and evaluation choices risk overstating the causal role of syntax.**  
   A recurring pattern is: construct a topically relevant set and a syntactically similar set so that only one dimension differs. In practice, though, the manipulations do not fully isolate syntax from semantics. For example:  
   - In TOFU, \(D_{\text{relearn}}^{\text{topic}}\) asks about target authors’ awards, motivation, genres, etc., while \(D_{\text{relearn}}^{\text{syntactic}}\) asks for names of *different* authors. This not only changes syntax but also the *type of label* (named entity vs description). That may inherently drive different gradient patterns, independent of “surface syntax” per se.  
   - In the WHP syntactic-relearn setup (Appendix C), the syntactically similar set uses trivia-style questions about other franchises. These differ not only in syntax but also in question type and difficulty relative to the unlearned trivia.  
   The representation and gradient similarities in **Figure 5** indeed show that \(D_{\text{relearn}}^{\text{syntactic}}\) is closer to \(D_{\text{target}}\), but because these datasets differ in label structure, this does not cleanly disentangle “syntax” from “task / answer-type similarity.” The paper should be more cautious in language, or add experiments where only word order / function words are altered while keeping label type fixed.

4. **Mathematical formulations of the key diagnostics are minimal and leave operational details underspecified.**  
   The syntactic similarity metric (Section 5.1) and loss ratio (Section 6) are the main quantitative constructs; however several implementation choices are under-specified and could materially affect conclusions:  
   - For the syntactic similarity score  
     \[
     \mathrm{Sim}(s_1,s_2)=1-\frac{d_{\text{Lev}}(s_1,s_2)}{\max(|s_1|,|s_2|)},
     \]  
     the paper says “we compute similarity at the sentence level and report dataset-level similarity as the average across all sentence pairs between \(D_{\text{relearn}}\) and \(D_{\text{target}}\).” Averaging over *all* pairs scales as \(O(|D_{\text{relearn}}||D_{\text{target}}|)\), which for large datasets is expensive and strongly influenced by set size. Do the authors instead average per-target over nearest neighbors, or subsample? Are similarity scores dominated by a few very-close pairs or by many weakly related pairs? This matters for interpreting **Table 1** and **Table 8**.  
   - For the loss ratio, it is not clearly defined how “template tokens” vs “keyword tokens” are determined. Are keywords always exact author names (subword spans), and are templates everything else in both question and answer? Are tokens in the date/locations treated as template or keyword? A stricter mathematical description (e.g., a mapping \(T: \text{token} \to \{\text{template},\text{keyword}\}\)) would make **Figure 6, Figure 9 (Top), Figure 17** more interpretable and reproducible.

5. **Syntactic diversification procedure and its cost / trade-offs are not fully characterized.**  
   Section 7.1 describes using GPT‑4o to paraphrase each query and then applying filtering with Levenshtein similarity thresholds. While some details are in Appendix G, several practical concerns remain:  
   - How many paraphrases per query are needed to get the reported effect in **Figure 8**? What is the sensitivity of performance to that number?  
   - The method assumes access to a strong external LLM (GPT‑4o or Llama‑3‑8B) and human-in-the-loop filtering for semantic fidelity. This can be costly and nontrivial for millions of forget examples. The paper hints in **Figure 18** that smaller models work, but does not quantify the overhead relative to naive unlearning.  
   - There may be edge cases where diversification *introduces* new unwanted content or privacy-sensitive variants; how are such risks mitigated?  
   Without a clearer cost–benefit discussion, it is hard to gauge whether syntactic diversification is ready as a practical deployed tool, or mainly a conceptual proof-of-concept.

6. **Limited comparison against other unlearning strategies and mitigation baselines at the forget-data level.**  
   The main comparison axes are GA, NPO, and SCRUB, all using the same forget set \(D_{\text{forget}}\), and then “ours” uses \(D_{\text{forget}}'\). However, it would be informative to see whether simpler changes to the forget set that also reduce syntactic homogeneity can close some of the gap, such as:  
   - Random perturbations or minor word-order shuffling of the existing templates.  
   - Mixing in diverse negative queries (e.g., “Who is not the author born in X?”).  
   Essentially, is there something special about LLM-generated paraphrases, or is any diversity enough? **Figure 7** and the examples in Appendix G.2 suggest fairly rich paraphrasing, but there is no ablation isolating this design space.

7. **Theoretical framing remains largely empirical / descriptive.**  
   While the empirical narrative is cohesive, claims like “syntactic similarity is the primary driver of benign relearning” are quite strong but not theoretically justified. The gradient and representation similarities in **Figure 5** are suggestive but do not amount to a formal argument, and the paper stops short of specifying a formal causal model of syntax → gradient alignment → relearning. Given how declarative the claims are, even a simple theoretical toy model (e.g., linear decoder with separate template and keyword embeddings) could help to clarify under what conditions such behavior must arise.

8. **Some analyses rely heavily on cherry-picked or small evaluation sets.**  
   For instance, the WHP “Who’s Harry Potter?” experiment in Appendix C uses 10 manually selected trivia questions with identical starting bigrams (“What is …”). **Figure 14** then plots LLM-judge scores for each question separately. While illustrative, this is very small scale; conclusions like “topically relevant set exhibits little relearning effect” are partly due to the fact that the topically relevant set actually contains direct or partial answers to 5 of those questions, which the authors also note. Similarly, WMDP’s ACE target in Appendix D is a single paragraph and 5 evaluation questions. These small-n cases are useful but should be qualified more clearly as anecdotal support rather than definitive evidence.

9. **Positioning relative to broader unlearning theory / evaluation work is thin.**  
   The related work section focuses mostly on LLM-specific unlearning efforts and BLUR. Given the broader literature on data deletion and certified or approximate unlearning, the paper could contextualize better how its “syntax as a driver” perspective interacts with theoretical frameworks. For example, works on certified deletion or “deep unlearning” benchmarks could provide a lens for understanding whether syntactic diversification moves models closer to the retrained-from-scratch ideal. I list some missing references below.

## Potentially Missing Related Work

1. **Ginart et al., “Making AI Forget You: Data Deletion in Machine Learning”, 2019.**  
   Foundational work on efficient data deletion and certified unlearning in simpler models. While not LLM-specific, it frames what it means for a model to approximate retraining-on-\(\mathcal{D}\setminus D_{\text{forget}}\). The paper should briefly connect its empirical notion of benign relearning and robustness to this theoretical baseline in Section 2.1 or the discussion.

2. **Bourtoule et al., “Deep Unlearn: Benchmarking Machine Unlearning”, 2024.**  
   Proposes benchmarks and protocols for unlearning in deep networks. It is directly relevant to the paper’s argument that existing benchmarks (e.g., BLUR) have confounds. Citing and comparing to Deep Unlearn in Section 2.1 and Section 4 would help position the contribution.

3. **Ebrahimpour & Boroojeny, “Toward Reliable Machine Unlearning: Theory, Algorithms, and Evaluation”, 2025 (and Ali, “Toward Reliable Machine Unlearning: Theory, Algorithms, and Evaluation”, 2025).**  
   These works provide a more systematic and theoretical treatment of unlearning algorithms and evaluation. Discussing them in Section 2 would help anchor the paper’s empirical findings in the larger reliability/evaluation discourse.

4. **Goel, “Corrective Machine Unlearning”, 2024.**  
   Introduces corrective unlearning mechanisms. It would be useful to mention in Section 2.1 and perhaps in the discussion, as such corrective strategies might interact with or compensate for syntactic relearning effects.

5. **Hine et al., “Supporting Trustworthy AI Through Machine Unlearning”, 2024.**  
   Discusses the ethical and trust implications of unlearning. Since Section 8 touches on deployment risks (“threat of syntactic homogeneity”), a short connection to this line of work would enrich the broader implications.

6. **Ali, “Evaluating Machine Unlearning: Applications, Approaches, and Accuracy”, 2025.**  
   Surveys evaluation methodologies for unlearning. Given that this paper emphasizes evaluation protocol corrections (e.g., best-step ROUGE, multiple leakage metrics in **Table 6–7**), it should cite and position itself relative to evaluation-oriented work.

7. **United States Artificial Intelligence Institute, “Machine Unlearning: The New Wave of Artificial Intelligence in 2024”, 2024.**  
   A broader overview piece on the state of unlearning. A brief mention in the introduction or related work (Section 2.1) would contextualize the work within trends in industry and policy.

## Questions

1. **Clarification on syntactic similarity computation and pairing strategy.**  
   When computing the average syntactic similarity between \(D_{\text{relearn}}\) and \(D_{\text{target}}\) (e.g., **Table 1**, **Table 8**), do you average over *all* pairwise sentence combinations, or over nearest neighbors per target, or a sampled subset? Please provide a precise sampling/aggregation formula and comment on how sensitive the values and conclusions are to this choice.

2. **Exact definition of template vs keyword tokens and robustness of the loss ratio.**  
   How exactly are keywords identified in TOFU (and in other benchmarks, if at all)? Are they fixed spans (e.g., author proper names) or do you include dates/locations? Have you tried altering this definition (e.g., treating dates as keywords) and checking whether the loss ratio curves in **Figure 6** and **Figure 9 (Top)** qualitatively persist?

3. **Generalization of syntactic diversification beyond TOFU.**  
   Have you attempted to apply diversification in a non-templated benchmark, such as WHP’s Harry Potter questions or RWKU’s celebrity facts? Even a smaller-scale experiment (e.g., paraphrasing 50 forget queries and comparing relearning curves) would be informative. If you tried and it performed poorly, that would also be useful to understand the boundary of applicability.

4. **Ablations on the nature and degree of diversification.**  
   How many paraphrases per query are ultimately kept in \(D_{\text{forget}}'\)? Have you examined how relearning robustness and utility in **Figure 8** and **Table 2** vary as you change that number? Also, have you tried simpler diversification baselines such as random word swaps or prompt-based paraphrasing without manual filtering?

5. **Interaction with PEFT-based unlearning.**  
   You show in Appendix B.3.1 (**Figures 10–12**) that LoRA-based relearning recovers knowledge faster than full-parameter relearning, even when the unlearning is full. If you perform unlearning itself with LoRA (as in Section B.1), then apply syntactic diversification, does LoRA unlearning remain more vulnerable to relearning than full unlearning, or does diversification close this gap?

6. **Scalability and cost of diversification in realistic settings.**  
   For a realistic deployment where \(D_{\text{forget}}\) might be tens of thousands of passages or QA pairs, what is your estimate of the extra computational and annotation overhead to synthesize and filter \(D_{\text{forget}}'\)? Could you quantify paraphrase generation cost and human filtering effort in your TOFU experiments as a baseline?

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The experimental methodology is generally sound, with multiple benchmarks, models, and leakage metrics; the central insights are well supported. Some aspects of the constructs (syntactic similarity metric, template/keyword split, and diversification ablation) are under-specified or under-explored, but I do not see fatal flaws.

## Presentation Rating

3: good.  
The paper is overall clear and well organized, with helpful diagrams (e.g., **Figure 1**, **Figure 7**) and carefully explained setups. Some methodological details are relegated to appendices or remain informal, and the discussion could be more explicit about limitations, but the narrative is coherent and readable.

## Contribution Rating

3: good.  
The work offers a meaningful reframing of benign relearning around syntactic similarity, backed by systematic experiments, and proposes a simple mitigation that appears effective in stylized settings. The lack of evaluation on more realistic, non-templated forget data slightly limits impact, but the insight is valuable for the unlearning community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper provides a compelling and well-supported empirical diagnosis that syntactic similarity is a key driver of benign relearning, correcting a widely held topicality-centric view, and offers an intuitive mitigation via syntactic diversification that improves robustness and utility on TOFU. However, the heaviest evidence and the proposed method are confined to synthetic, strongly templated tasks, and several methodological details (similarity metrics, token categorization, diversification ablations) are not explored as deeply as they could be. I lean positive because the central insight seems robust and likely to stimulate further work, but I would like to see clearer discussion of limitations and, ideally, some broader evaluations.

## Reviewer Confidence

4: confident.  
I am familiar with LLM unlearning and related evaluation work and carefully examined the methodology, including the equations and figures. Some application-specific details (e.g., broader legal/privacy implications) are outside my expertise, but they are not central to the technical claims.