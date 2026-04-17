The paper content is already in the prompt. Let me carefully verify the claims against the paper text I have.

Key things to verify:
1. The o1-preview vs GPT-4o comparison claim
2. Whether the paper mentions output parsing
3. Whether it tests prompt variations
4. The CDE task details

From the paper text provided:

**On o1-preview vs GPT-4o (Section 4.1, Table 1):**
- The paper says: "OpenAI o1-preview, which has a form of CoT built into its responses, has a 36.3% absolute accuracy decrease compared to GPT-4o zero-shot on the same subset."
- Table 1 clearly shows: GPT-4o zero-shot = 94.00%, o1-preview = 57.70%, on the same subset of problems
- The paper also reports within-model comparisons: GPT-4o 87.50% → 64.40%, Claude 3 Opus 70.70% → 62.70%, etc.
- So the 36.3% is a cross-model comparison indeed.

**On prompt variations:**
- The paper says "We considered two prompts, zero-shot and CoT (see Appendix A.2)" for ISL
- In the Discussion: "We tested the effectiveness of ToT on GPT-4o for the implicit statistical learning task (see Appendix A.4). While ToT improved accuracy (64.55% vs. 62.52%), this was still far from GPT-4o's zero-shot performance of 94.00%"
- So there is one additional CoT variant tested (ToT) on one task

**On output parsing:**
- Not explicitly discussed in the main text
- The paper doesn't detail how answers were extracted from CoT outputs

**On the CDE task:**
- "We also investigated the per-round accuracy of GPT-4o and found that direct prompting resulted in the model attaining perfect classification on the second or third iteration, while with CoT, the model was only able to correctly classify around 8/10 objects after 5 iterations"
- The paper explicitly notes: "likely due to CoT biasing the model to rely on the seemingly generalizable rules from the exemplars, while down-weighing the usefulness of contextual tokens that explicitly contained all of the correct answers."

**On the apartment task and GPT-4o circularity:**
- Table 6 shows GPT-4o is both the model being evaluated AND the model used to generate ground truth ratings. This is noted.

Now let me write the final review.

## Summary

This paper investigates settings where chain-of-thought (CoT) prompting systematically *reduces* model performance by drawing on the cognitive psychology literature about tasks where verbal thinking or deliberation hurts human performance. The authors test six tasks from this literature, finding that CoT significantly decreases performance on three—implicit statistical learning (artificial grammar learning), facial recognition (verbal overshadowing), and classification with exceptions—while showing neutral or positive effects on the other three (logical inconsistency, spatial intuitions, and multi-feature decision aggregation), which they explain through differences between human and model capabilities. Across the three failure cases, they document substantial performance drops using a diverse set of state-of-the-art LLMs and VLMs.

## Strengths

1. **Novel and well-motivated research question with practical significance.** While extensive work has studied when CoT helps, systematically identifying when CoT *hurts* is underexplored and practically important given the increasing default use of inference-time reasoning in deployed models (e.g., o1, Claude). The psychology-inspired heuristic is a creative and principled way to navigate the vast space of potential failure cases.

2. **Strong empirical evidence for the three failure cases with large effect sizes.** The performance drops are large, consistent across models, and statistically significant. For implicit statistical learning, GPT-4o drops 23.1% (87.5% → 64.4%); for face recognition, all 6 VLMs tested show drops including 14.4% absolute for Claude 3 Opus; for classification with exceptions, GPT-4o requires 331% more rounds (2.9 → 12.5). These are not marginal effects.

3. **Honest and nuanced analysis of the mismatch cases.** Rather than hiding the three tasks where human–model parallels break down, the paper carefully provides coherent explanations: low zero-shot base rates for logical inconsistency, absence of motor simulation priors for spatial intuitions, and longer context windows overcoming human working memory limitations for feature aggregation. This prevents overclaiming and deepens the contribution.

4. **Thoughtful connection to the cognitive psychology literature.** The paper accurately and thoroughly reviews the verbal overshadowing, implicit learning, and explanation-based effects literature, situating the work within both the psychology and ML communities in a way that could bridge these fields.

5. **Broad model coverage.** Testing multiple open- and closed-source models (9 LLMs, 6 VLMs) provides evidence that the effects are not idiosyncratic to a single system.

## Weaknesses

### Major:

1. **The o1-preview vs GPT-4o cross-model comparison is methodologically invalid for isolating the effect of CoT, yet is the paper's headline result.** The abstract prominently states "up to 36.3% absolute accuracy for OpenAI o1-preview compared to GPT-4o." This compares two fundamentally different models with different architectures, training data, and RLHF policies. The 36.3% gap could be entirely attributable to model-level differences rather than CoT. The within-model comparisons (e.g., GPT-4o zero-shot vs. CoT: 23.1% drop) are the actual evidence for a CoT effect; the cross-model comparison is not interpretable as such and should not be the headline. The paper's own Table 1 structure reveals this issue: for o1-preview, no zero-shot condition is even reported, making it impossible to do a within-model comparison.

2. **Prompt design and evaluation do not convincingly isolate "reasoning hurts" from prompt framing effects.** The central claim is that engaging in chain-of-thought reduces performance, but the experiments confound several variables: (a) the CoT instruction explicitly changes the task framing (e.g., "think step-by-step" may redirect attention to rule-seeking rather than pattern recognition); (b) output parsing differences between verbose CoT responses and terse zero-shot responses are not discussed, which is a standard pitfall in CoT vs. direct comparisons; (c) there is no "long-but-non-reasoning" control prompt (e.g., "read carefully and respond with only your answer") to disentangle length/framing effects from reasoning effects. Without such controls, it is ambiguous whether it is *the act of reasoning* or *a specific instruction formulation* that causes harm.

3. **Generalization from three narrow, synthetic task instantiations to broad task categories is overstated.** Each "category" is represented by a single instantiation: one specific finite-state grammar construction with one negative-sampling strategy (replace-one-letter); one synthetic face recognition benchmark from a single generator; one specific multi-pass vehicle classification protocol. The strong typological framing ("tasks with implicit statistical learning," "tasks where language is ill-suited to represent stimuli") implies that CoT is systematically harmful across these broad categories, but the empirical evidence covers only single instances. For example, the artificial grammar results depend on the specific grammar structure and negative-example construction used; different FSG complexities or noise models could change the pattern.

4. **The heuristic's predictive value is only partially demonstrated and not rigorously evaluated.** The psychology-to-model heuristic works for 3/6 tasks (50% success rate). The three failures are explained post-hoc (different priors, missing motor simulation, different memory constraints), which is reasonable but makes the framework sound more predictively successful than it is. Without pre-registered predictions on additional tasks, or a formal specification of boundary conditions, it is hard to assess whether the heuristic is genuinely useful for prospectively identifying CoT failure cases, or whether these three successes are cherry-picked.

### Minor:

1. **Statistical significance is used to inflate the impression of universal CoT harm for near-chance models.** For example, Llama 3.1 8B drops 53.52% → 51.54% on ISL (both near chance at 50%) with p < 0.0001; InternVL2 26B drops 9.2% → 6.0% on FR (both far below the 20% random baseline). These "significant" effects are driven by large sample sizes and reflect near-meaningless differences. The core argument rests on the large effects for capable models; including marginal effects for weak models as "consistent decreases" is misleading.

2. **The CDE task measures learning speed (rounds to convergence) rather than accuracy, making cross-task comparison of effect sizes difficult.** While rounds-to-convergence is meaningful, it is a fundamentally different metric from the accuracy drops reported for ISL and FR, making it hard to assess whether the "CoT harm" is of comparable magnitude across all three failure cases.

3. **GPT-4o serves as both the evaluator and the evaluated model in the apartment selection task.** The paper uses GPT-4o to rate feature impact scores that define the ground truth, and then evaluates GPT-4o on the same task. This circularity is minor since the paper's main finding here is that CoT *helps* rather than hurts, but it should be acknowledged.

4. **The apartment selection task reveals a severe failure mode (Llama 3.1 70B: 42% → 6% with CoT) that appears to be a generation truncation issue rather than a cognitive effect.** The paper notes Llama "was often unable to return an answer after deliberating in the CoT condition." This conflates an implementation artifact with a genuine reasoning-vs-no-reasoning comparison.

### Trivial:

- The SI (spatial intuitions) task essentially shows null results across all models (all near chance), yet is included as a full task section. While the null result is interpretable in context, it adds little empirical insight.

## Nice-to-Haves

- Analysis of actual CoT outputs (e.g., what rules do models articulate for ISL grammars? How do they describe faces in FR?) to provide mechanistic evidence for the proposed explanations, rather than relying solely on behavioral outcomes.
- Testing multiple CoT prompt phrasings per task to demonstrate robustness to prompt wording.
- A "direct-but-long" or "detailed-but-non-reasoning" control prompt to isolate the reasoning component from length/framing effects.
- Pre-registered predictions on additional psychology tasks to prospectively validate the heuristic.

## Removed Points

- **Demand for within-model comparison specifically for o1-preview (forced to not reason).** This is a nice-to-have, not a critical flaw per se. While the cross-model comparison is methodologically weak as a causal claim about CoT, the paper does provide ample within-model evidence from other models. The o1-preview result can be reframed as supplemental, not primary, evidence. *Removed as a fatal concern because the paper's core argument does not solely rest on the o1 result.*

- **Demand for "few-shot non-CoT baseline" or "output-length/token-position control" as strict requirements.** These are nice-to-have controls for further isolating the mechanism, but the paper already demonstrates the basic zero-shot vs. CoT contrast. The absence of these controls is a methodological limitation, not a fatal flaw.

- **Criticism that the psychology literature on verbal overshadowing has replication debates.** The paper's contribution is about using psychology as a *heuristic* for identifying CoT failure cases in models, not about verifying the psychology itself. Whether the original human studies replicate perfectly is orthogonal to whether the model results hold.

- **Demand for the paper to provide "a formal model predicting when the heuristic will/won't apply."** This is beyond the paper's stated scope. The paper explicitly positions its contribution as identifying promising case studies, not producing a closed-form theoretical framework.

- **Criticism that the FR task uses synthetic faces from textual attribute descriptions, making it "non-naturalistic."** This is actually a *strength* of the experimental design: by controlling the textual attributes, the paper ensures that verbal overshadowing cannot be circumvented by models simply reading the face descriptions. The synthetic nature is deliberate and appropriate for the research question.

- **Request for multiple comparisons correction.** The primary results feature large effect sizes that would survive any reasonable correction. The marginal effects on near-chance models are a minor issue (addressed above), not a multiple comparisons problem.

## Novel Insights

The most insightful finding is not the parallel between human and model failures, but the *divergence*: the CDE task reveals that CoT biases LLMs toward *generalizable rule-seeking* in a way that paradoxically makes them worse at exploiting the full written history available in context—including the explicit correct answers from prior passes. In humans, the analogous effect arises from limited working memory making simple rules attractive; in LLMs, it appears to arise from the CoT instruction shifting probability mass away from simply pattern-matching the transcript toward explicit verbal reasoning. This suggests that CoT harm in LLMs may often be an *attention allocation* problem rather than a *representational capacity* problem, which is a genuinely different mechanism than what drives the human effect.

## Suggestions

1. **De-emphasize the o1-preview vs. GPT-4o cross-model comparison** in the abstract and introduction. Lead with the within-model GPT-4o result (23.1% drop), which is still striking and scientifically clean. The o1 result can be mentioned as supporting evidence for the broader trend of inference-time reasoning being harmful in certain settings, but not as the headline.

2. **Add analysis of model CoT outputs** for at least one task (e.g., ISL). Show what kinds of rules or patterns models articulate in their chains, and correlate error types with reasoning content. This would transform the current mechanism (which is speculative) into empirically grounded analysis.

3. **Qualify the categorical claims.** Replace "three categories where CoT reduces performance" with "three task types where we observe CoT-related performance reductions" or similar, acknowledging that broader generalization within each category is not yet established.

4. **Separate near-chance model results from above-chance results in the main tables** or explicitly note in the text that the CoT harm is meaningfully demonstrated only for models performing well above chance, to avoid inflating the sense of universal harm.

## Evaluation Axes

- **Originality:** High. The idea of using cognitive psychology's "thinking hurts" literature to predict CoT failure modes is novel and creative. No prior work systematically bridges these literatures for this purpose.
- **Importance of research question:** High. With inference-time reasoning becoming default in deployed models, understanding when it backfires is critical and timely.
- **Claims support:** Moderate. The within-model evidence for the three failure cases is solid, but the causal attribution to "reasoning" (vs. prompt framing) and the generalization to broad task categories are under-supported.
- **Experimental soundness:** Moderate. The experiments are well-designed in their basic structure but lack key controls (non-reasoning long prompts, CoT output analysis, prompt variation robustness).
- **Clarity:** Good. The paper is well-written, well-organized, and honest about limitations.
- **Value to research community:** Moderate-to-high. The findings are practically useful and could shift how practitioners think about default CoT deployment, though the heuristic needs further validation.

## Score and Decision

**Calibration comparison:**

- **"To CoT or not to CoT" (w6nlcS8Kkn):** Scores 6, 8, 6 (Accept Poster). This paper did a broad meta-analysis of when CoT helps, with solid methodology and clean claims. The current paper has a more novel framing (psychology-to-ML bridge) but weaker methodological controls and more overclaim.

- **"Turning LLMs into Cognitive Models" (eiC4BKypf1):** Scores 8, 8, 8, 8 (Accept Poster). This paper had tight methodology, clean empirical evidence, and appropriate scope. The current paper is below this level due to the confounds and overclaiming.

- **"Tracking Cognitive Development of LLMs" (fI6TkT050a):** Scores 3, 1, 3, 3 (Reject). This paper made strong anthropomorphic claims with weak methodology. The current paper is significantly better—it is more careful, does not anthropomorphize, and has real empirical findings—but shares some of the overclaiming tendency.

- **"MIND SCRAMBLE" (KBixkDNE8p):** Scores 3, 3, 3, 3, 3 (Reject). Overclaimed psychology-to-LLM analogy without rigorous connection. The current paper is well above this level with careful task design and honest failure reporting.

- **"LLMs Assume People Are More Rational" (dAeET8gxqg):** Scores 8, 6, 6, 5 (Accept Poster). This paper connected psychology experiments to LLM behavior with solid methodology. Comparable in quality, though the current paper has more significant methodological gaps.

The current paper sits in the range of 5-6. The core idea is genuinely novel and the three failure cases are real and important. However, the headline claim rests partly on a confounded cross-model comparison; the experiments lack controls to isolate reasoning from prompt framing; and the generalization from narrow synthetic tasks to broad categories is overstated. These are significant weaknesses but not fatal—the paper makes a real contribution that the community should see, even if the claims need tempering.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>