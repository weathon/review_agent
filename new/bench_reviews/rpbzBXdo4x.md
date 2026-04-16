## Summary
This paper studies when chain-of-thought (CoT) prompting hurts rather than helps model performance, using cognitive psychology as a heuristic for selecting tasks where human verbal deliberation is known to impair performance. Across six adapted tasks, the paper finds large and consistent CoT-related degradations in three settings—artificial grammar learning, face recognition, and learning exception-laden categories—while three other psychology-inspired tasks show neutral or positive effects, which the paper discusses in terms of human/model differences.

## Strengths
- The paper asks an important and timely question. As inference-time reasoning is increasingly used by default, identifying settings where it degrades performance is practically valuable.
- The framing is original and intellectually meaningful: instead of searching for failure cases ad hoc, the paper uses a psychology-inspired hypothesis to guide task selection.
- The empirical findings in the strongest sections are substantial. In particular, the within-model drops on artificial grammar learning are large (e.g., GPT-4o 87.5% to 64.4%), and the exception-based category learning task shows very large slowdowns (e.g., GPT-4o 2.9 to 12.5 rounds).
- The paper is commendably not one-sided: it includes tasks where the human analogy does not transfer and attempts to explain those boundary cases rather than hiding them.
- The evaluation covers a reasonably broad set of contemporary models across the main tasks, lending weight to the narrower claim that CoT can materially hurt on some nonstandard task families.
- The writing is clear overall, and the paper is explicit in several places that it is proposing a heuristic rather than claiming exact cognitive equivalence between humans and models.

## Weaknesses

###: Fatal
- None.

### Major:
- **The headline 36.3% result is not a valid same-model CoT comparison and is over-emphasized.**  
  The abstract and introduction foreground “up to 36.3% absolute accuracy for OpenAI o1-preview compared to GPT-4o” on artificial grammar learning. But this compares different models, not the same model with and without CoT. That cannot isolate the effect of reasoning. The paper does provide valid within-model comparisons in Table 1, so the core empirical claim survives, but the most dramatic number should not be used as primary evidence for “thinking hurts.”
- **The paper overstates the strength of its central heuristic claim.**  
  The evidence supports an interesting set of psychology-motivated case studies, but not a validated predictive framework. The paper states that it can “help us identify risky cases” and later that it “successfully identify[ies] three settings,” yet the actual evidence is six handpicked task types with 3 successes and 3 non-transfers, plus largely post hoc explanations for the misses. There is no comparison against alternative heuristics, no estimate of predictive precision/recall, and no prospective transfer criterion beyond qualitative judgment. This makes the broad framing stronger than the empirical support.
- **Several experiments do not cleanly separate “reasoning hurts” from prompt/intervention effects.**  
  The core comparison is usually direct/zero-shot versus CoT, but this changes more than just “reasoning.” In §4.3, for example, the paper explicitly changes the human manipulation from explanation after feedback to CoT before each prediction. That may still be a reasonable LLM adaptation, but it is a different intervention and one that directly encourages rule-search behavior. More generally, longer CoT generations can alter instruction following, context use, or answer formatting. Since there is no non-reasoning filler control or prompt-variation study in the main paper, the mechanism-level claims about verbal reasoning remain weaker than the performance-level claims.
- **Task adaptations raise construct-validity concerns for the broader cognitive interpretation.**  
  The paper often substantially modifies human tasks to make them scalable for models: e.g., removing distractors in face recognition, converting the spatial task to multiple choice, and changing the temporal placement of explanation in the exceptions task. These are understandable engineering choices, and they do not negate the benchmark findings, but they weaken the stronger claim that the same underlying cognitive phenomenon has been preserved across human and model versions. This matters mainly for the psychology-to-LLM interpretation, not for the narrower empirical observation that CoT can hurt performance on these adapted tasks.

### Minor
- **The facial-recognition section is somewhat harder to interpret mechanistically because some models are already in a degenerate regime.**  
  The paper notes that weaker models often answered that “all images are of the same person,” leading to below-chance performance. In such cases, a further drop under CoT is real but less informative about verbal-overshadowing-like mechanisms than it is for stronger models.
- **The logical inconsistency section is described a bit too cleanly as a mismatch case.**  
  Table 4 is more mixed than the narrative suggests: CoT strongly helps some models, but decreases others (e.g., Claude 3 Opus and Gemini 1.5 Pro on MNLI/SNLI). The paper does acknowledge this by saying “mixed effects,” but the subsection framing could better reflect that this is not purely a non-transfer case.
- **The apartment-selection benchmark is somewhat model-anchored.**  
  The desirability scores used to generate apartment instances were obtained from GPT-4o. That is a practical way to operationalize utility, but it introduces some dependence between task construction and at least one evaluated model’s preferences, which slightly weakens neutrality.
- **Mechanistic evidence is limited.**  
  The paper offers plausible interpretations—e.g., CoT induces verbalizable but misleading rules in grammar learning and exception-laden classification—but provides little direct analysis of the actual generated chains or systematic error types. Stronger qualitative or quantitative error analysis would better support the proposed parallels to human verbal overshadowing or overgeneralization.
- **Model coverage is thinner on the exceptions task.**  
  The CDE result is strong, but only three models are reported because others were not workable in the multi-turn setup. This is understandable, but it does limit generality relative to the other sections.

### Trivial
- None.

## Nice-to-Haves
- Add prompt-ablation studies to test whether the observed drops are robust across multiple CoT phrasings and structures.
- Add a non-reasoning verbosity control to distinguish “must produce extra text” from “engages in harmful reasoning.”
- Provide representative CoT outputs and error analyses for the three strongest failure cases.
- Clarify more explicitly that the contribution is a useful task-selection heuristic and a set of empirical case studies, not a fully validated predictive theory.
- Discuss practical mitigation strategies: e.g., when to avoid CoT, how to detect risky task structures, or whether adaptive prompting could help.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper lacks variance details / exact statistical testing methodology.”**  
  The paper reports p-values and large sample sizes for several tasks, and the criticism as stated drifts toward reproducibility nitpicking without showing that the omission undermines the core claims. It could be useful for revision, but it is not a central weakness from the provided text alone.
- **“The implicit statistical learning task may favor memorization or similarity heuristics over genuine grammar induction.”**  
  The negatives are indeed constructed by letter replacement, so this is a plausible caveat. However, the paper’s main claim in that section is about CoT harming performance on this adapted task, not about proving a pure grammar-induction mechanism. Without appendix details, this point should be treated cautiously rather than elevated.
- **“The paper should collect fresh human data on the exact adapted tasks.”**  
  This would strengthen the cognitive-comparison aspect, but it is scope-expanding rather than essential for the paper’s stated empirical contribution about model behavior on adapted tasks.
- **“The task set is too small because there are only six tasks.”**  
  On its own this is too generic. The more precise and valid criticism is that six curated tasks are insufficient to validate the heuristic as a predictive framework; that point is already kept above.

## Novel Insights
The clearest synthesis is that the paper is strongest when interpreted as demonstrating a **representation-mismatch failure mode** for CoT rather than a general “thinking is bad” claim. The successful cases share a pattern: CoT seems most harmful when the task can be solved relatively well without explicit verbal decomposition, but verbalization encourages compression into oversimplified, linguistically convenient rules or features that are misaligned with the real discriminative signal. By contrast, the non-transfer cases are exactly those where either zero-shot competence is too weak for degradation to be visible, or the model’s architecture/resources fundamentally differ from the human bottleneck that made deliberation harmful. This is a useful conceptual refinement, but the paper does not yet operationalize it into a predictive theory.

## Suggestions
- Remove or strongly de-emphasize the o1-preview vs GPT-4o 36.3% comparison as headline evidence; lead instead with within-model results such as GPT-4o, Claude, Gemini, and Llama on the artificial grammar task.
- Narrow the main claim: present the paper as psychology-inspired case studies that reveal several robust CoT failure modes, rather than as a validated general heuristic for predicting them.
- Add at least one control that disentangles reasoning from verbosity/intervention effects, such as a filler-text condition or alternate non-explanatory structured output.
- Include error analyses of generated CoTs for ISL, face recognition, and CDE to show whether the model is actually over-verbalizing misleading features/rules.
- Make the narrative around the mismatch tasks more precise, especially for logical inconsistency, where the results are mixed rather than uniformly non-transferring.

## Score and Decision
**Assessment across axes:**  
- **Originality:** High. The psychology-guided framing is novel and more thoughtful than standard CoT benchmarking.  
- **Importance of question:** High. Understanding when CoT hurts is important and practically relevant.  
- **Support for claims:** Moderate. The narrower empirical claim is well supported; the broader heuristic/predictive claim is overclaimed relative to the evidence.  
- **Experimental soundness:** Moderate to good. Several experiments are convincing, but some interventions confound reasoning with prompt/interface changes, and task adaptations weaken strong cognitive interpretations.  
- **Clarity:** Good. The paper is generally clear and appropriately discusses limitations.  
- **Value to the community:** Good. Even with narrowed claims, the paper surfaces real and practically relevant failure modes.

**Calibration against human-reviewed papers:**  
- Compared to **“To CoT or not to CoT?”** (scores 6/8/6), this submission is similarly interesting and empirically useful, but somewhat weaker in claim discipline because its broad heuristic framing is not fully validated.  
- Compared to psychology/LLM analogy papers like **“MIND SCRAMBLE”** (all 3s) and **“Tracking Cognitive Development of LLMs”** (3/1/3/3), this paper is clearly stronger: it has real empirical signal, multiple robust effects, and better acknowledgment of non-transfer cases.  
- Compared to stronger cognitive-evaluation papers such as **“Does Spatial Cognition Emerge in Frontier Models?”** (6/8/8/5), this paper is somewhat weaker on validation of the overarching framework, though stronger on the sharpness of some individual failure effects.  
- Relative to these anchors, this paper lands in the **weak accept / strong borderline** range: stronger than rejected analogy papers, but not as airtight as the better accepted empirical studies.

**Final score:** 6.5 / 10  
**Decision:** Accept

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>