## Summary
This paper proposes **BlackDAN**, a black-box jailbreak framework that uses **NSGA-II** to optimize jailbreak prompts under multiple objectives, primarily unsafe-response likelihood and semantic consistency with the harmful query. The paper’s main empirical message is that adding a second objective beyond nominal attack success can improve jailbreak effectiveness across several LLMs and multimodal LLMs, but the submission overstates what is actually implemented and validated.

## Strengths
- **Well-motivated problem framing:** The paper correctly identifies a real limitation in jailbreak evaluation: optimizing only for nominal success can produce refusals, irrelevant outputs, or otherwise impractical attacks. Elevating **semantic consistency** as an explicit optimization target is a meaningful contribution, and Section 3 does concretely define a second fitness function \(f_2\) based on response-query similarity.
- **Reasonable use of multi-objective search:** Using **NSGA-II** for prompt search is a plausible and interpretable design choice. The paper clearly explains dominance and crowding-distance selection, and the evolutionary framework is naturally compatible with trading off multiple objectives rather than collapsing everything into a single scalar reward.
- **Broad empirical coverage:** The paper evaluates on multiple text LLMs and includes multimodal experiments on MM-SafetyBench, which is broader than many narrower jailbreak papers. If the protocol were stronger, this breadth could make the study useful to the community.
- **Some effort toward analysis beyond leaderboard numbers:** The embedding-space visualizations in Figures 5–6 are at least an attempt to understand how Pareto ranks relate to prompt/response structure, rather than presenting only attack rates.

## Weaknesses

###: Fatal

### Major:
- **The paper’s headline “multi-objective including stealthiness” claim is not actually supported by the implemented method or evaluation.**  
  The abstract and introduction repeatedly claim optimization over **ASR, stealthiness/detectability, and semantic relevance**, and the conclusion again says the method includes “ASR, stealthiness, and semantic consistency.” However, Section 3.1 defines only **two** fitness functions:  
  1. unsafe token probability via Llama Guard 2, and  
  2. semantic consistency via MiniLM cosine similarity.  
  No stealthiness objective is formalized, and Section 4 defines only keyword-ASR and a GPT-4 judge metric. Figure 2 mentions additional objectives such as “conversion,” “diversity,” “length of text,” and “number of steps,” but these are not mathematically specified or empirically studied. So the paper demonstrates at most a **two-objective proxy optimization**, not the broader stealthy multi-objective framework it advertises.
- **The primary evaluation metric is too weak to support the paper’s strongest claims about effective jailbreaking.**  
  Section 4.1 defines keyword-based ASR as success whenever the output omits a list of refusal phrases like “I’m sorry” or “I cannot.” This does **not** ensure the model actually produced harmful assistance. A non-refusal response can still be irrelevant, evasive, hedged, or only weakly aligned to the harmful request. This matters especially here because the method explicitly optimizes semantic similarity, so responses that paraphrase or loosely echo the harmful query may be counted as successes without genuinely providing actionable harmful content. The GPT-4-based metric is a better complement, but it is secondary, not reported everywhere, and the success threshold \(g \ge 5\) is not convincingly justified.
- **The superiority claims over prior methods are not adequately supported by controlled comparisons.**  
  Table 1 mixes **white-box, gray-box, and black-box** methods, which operate under different assumptions. That can still be informative descriptively, but it is not enough to support a clean “outperforms prior methods” claim. The paper does not clearly state whether baselines were rerun under matched prompts, budgets, query counts, judges, datasets, and target model versions, or whether some numbers came from incompatible prior protocols. Table 2 is stronger, but even there the experimental controls remain under-described, with no variance estimates, no run counts, and little budget information. Given how sensitive jailbreak results are to judges and prompting details, this weakens the central empirical conclusion.
- **The “Rank Boundary Hypothesis” / interpretability narrative is substantially underdeveloped relative to the claims.**  
  The contributions section claims a “Rank Boundary Hypothesis” with better differentiation between toxic and non-toxic prompts, and the introduction says the paper will “verify and analyze” prompt boundaries. But the method section never formalizes this hypothesis in a testable way, and the evidence is limited to post hoc embedding visualizations. Figures 5 and 6 may be suggestive, but they do not establish that these boundaries are meaningful, causal, or useful for jailbreak generation. As written, the interpretability and boundary claims are speculative and much stronger than the evidence warrants.

### Minor
- **The optimization/evaluation alignment is not clearly validated.**  
  The method optimizes proxy objectives from **Llama Guard 2** and **MiniLM**, but the final evaluation uses **keyword ASR** and sometimes **GPT-4 judgment**. The paper does not analyze how well the optimized proxies correlate with the final success criteria, leaving uncertainty about whether the optimization is truly targeting what the evaluation claims to measure.
- **The evolutionary algorithm is under-specified for a stochastic black-box attack setting.**  
  Important details such as population size, number of generations, mutation/crossover rates, stopping criteria, initialization details, and especially **query budget** are not given in the main text. Since this is a query-based black-box attack, query efficiency is a substantive issue rather than a mere implementation nitpick.
- **No ablation cleanly isolates the value of the multi-objective design itself.**  
  The paper compares against prior methods and does show a “single-objective” discussion, but there is no careful same-framework ablation holding the optimizer fixed while varying only the objectives. As a result, it remains unclear how much of the gain comes from NSGA-II / search design versus the added semantic objective.
- **The genetic operators are simplistic and insufficiently justified.**  
  Section 3.3 uses random sentence swapping for crossover and synonym replacement for mutation. These are plausible baseline operators, but the paper does not show that they are effective, nor which components matter. Given that prompt jailbreaks can be fragile, stronger empirical justification is needed.
- **Multimodal setup is not described in enough detail.**  
  The paper claims extension to multimodal jailbreaks and reports Figure 4, but it does not explain clearly how images are incorporated into the optimization loop, whether the attack is text-only prompt evolution over fixed images, or how multimodal harmfulness is judged.

### Trivial
- **Writing and presentation are sometimes internally inconsistent.**  
  Some sections conflate objectives (ASR, harmfulness, stealthiness), and the explanation around figures/tables is occasionally confusing. This is not just style: it contributes to uncertainty about what exactly was optimized and evaluated.

## Nice-to-Haves
- Add a true **stealthiness/detectability** evaluation, since stealthiness is a core claimed contribution.
- Include **Pareto-front plots** and trade-off analyses to make the multi-objective story concrete.
- Report **variance across runs** and sensitivity to NSGA-II hyperparameters.
- Provide qualitative **same-query SO vs. MO examples** showing that the multi-objective version is more semantically aligned, not merely higher-scoring under proxy metrics.
- Expand the discussion of **defensive implications** and safer red-teaming framing for a safety-oriented venue.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related work” complaints.** Removed per instruction. While some reviewers requested more baselines or more prior methods, I do not frame this as a related-work omission.
- **Pure reproducibility nitpicks about appendix placement or omitted trivial details.** I retained only the omissions that materially affect soundness in a stochastic black-box setting (e.g., query budget, generations, population size). Pure “algorithm is in appendix” complaints were not kept as standalone weaknesses.
- **Criticism that baseline comparison is unfair simply because settings differ and the asymmetry may favor baselines.** I kept only the valid concern that the paper uses mixed settings to support superiority claims without enough protocol detail. I did not keep a generic “comparison is unfair” complaint.
- **Any complaint questioning existence/release/availability of cited models, tools, or datasets.** Not applicable and removed by rule.
- **A claimed contradiction that Table 2 disproves the paper’s superiority because GPT-4 GPT4-Metric is 28.0 vs PAIR 30.0.** This point is factually valid for that cell, but the paper’s stronger issue is broader overclaiming and loose evaluation. I therefore did not elevate this isolated number as a central criticism; it is better treated as one symptom of overclaiming rather than a standalone decisive flaw.

## Novel Insights
The strongest synthesis here is that the paper’s core idea is **better than its evidentiary framing**. There is a genuinely worthwhile contribution in treating jailbreak generation as a trade-off problem between harmful compliance and semantic relevance, and NSGA-II is a sensible mechanism for exploring that trade-off. However, the submission repeatedly markets itself as a broader framework for stealthiness, interpretability, and rank-boundary analysis without ever closing the loop from objective definition to rigorous measurement. In other words, the paper’s likely publishable nucleus is a **two-objective black-box jailbreak search method**, while much of the surrounding narrative—stealthiness, prompt boundaries, interpretability—currently reads as aspirational rather than demonstrated.

## Suggestions
- **Narrow the claims to what is actually shown.** If the paper only optimizes unsafe-score and semantic consistency, say so explicitly and stop claiming demonstrated stealthiness unless it is measured.
- **Replace or substantially supplement keyword-ASR.** Use a stronger harmful-compliance evaluation throughout, ideally with judged harmfulness tied to the specific query and clearer threshold justification.
- **Run a same-framework ablation:** BlackDAN with only harmfulness proxy vs. BlackDAN with harmfulness + semantic consistency, keeping optimizer, initialization, and budget fixed.
- **Report budgets and stochasticity details:** population size, generations, mutation/crossover rates, query counts to target and judge models, and results over multiple runs.
- **Formalize the Rank Boundary Hypothesis** and test it quantitatively if it is to remain a contribution; otherwise, present Figures 5–6 as exploratory analysis rather than validated theory.
- **Clarify multimodal optimization** with a precise description of how images, prompts, and judges interact.

## Score and Decision
**Originality:** Moderate. The combination of NSGA-II with jailbreak objectives is a reasonable and somewhat novel framing, though the components themselves are standard.  
**Importance:** High. Better evaluation and optimization of jailbreaks is important for red-teaming and safety.  
**Claims support:** Weak-to-moderate. The paper overclaims stealthiness, interpretability, and boundary analysis relative to what is implemented and measured.  
**Experimental soundness:** Moderate at best. Breadth is good, but metrics, controls, and protocol detail are not strong enough for the claimed superiority.  
**Clarity:** Mixed. The high-level idea is understandable, but the paper is internally inconsistent about its objectives and conclusions.  
**Community value:** Potentially useful if reframed more narrowly and evaluated more rigorously.

For calibration, I compared this paper against:
- **AutoDAN** (`/home/wg25r/review_agent/human_reviews/ZuZujQ9LJV.md`, scores 5/10/5/5, reject): that paper also drew concern about weak ASR measurement and incomplete evaluation, but some reviewers saw a clearer technical contribution. BlackDAN is similar in spirit and suffers from the same metric issue, with additional overclaiming about stealthiness and interpretability.
- **AutoDAN (poster accept)** (`/home/wg25r/review_agent/human_reviews/7Jwpw4qKkb.md`, scores 6/8/6/8): the accepted version appears to have stronger evidence directly tied to its stealthiness claim, including defense-oriented evaluation. BlackDAN is below this anchor because it claims stealthiness without actually evaluating it.
- **JAILJUDGE** (`/home/wg25r/review_agent/human_reviews/cLYvhd0pDY.md`, reject): like that paper, this submission has a meaningful problem setting but insufficiently convincing empirical support for its broad claims.
- **AutoDAN-Turbo** (`/home/wg25r/review_agent/human_reviews/bhK7U37VW8.md`, accept spotlight): far stronger empirical support and more convincing performance/novelty than the present paper.

Overall, this paper contains a plausible and potentially useful idea, but the current submission overclaims beyond its demonstrated evidence and does not meet the bar for acceptance in this form.

**Score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>