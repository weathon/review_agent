## Summary
This paper investigates whether example selection algorithms for in-context learning (ICL) amplify social biases in large language models, constructing a paraphrased sentiment classification dataset (EEC-paraphrase) for evaluation. Across eight LLMs and four selection methods, the authors find that while mean bias often decreases, maximum bias across random seeds increases relative to zero-shot, and that example selection contributes to what they term "spurious correlations." They propose ReBE, a prompt-tuning method with a bias-contrastive loss, which reduces bias metrics while preserving accuracy.

## Strengths
- **Important and novel problem framing.** The intersection of ICL example selection and fairness is genuinely underexplored. Demonstrating that accuracy-optimal example selection can yield high worst-case bias (Figure 1) is a valuable empirical observation that challenges a common implicit assumption.
- **Broad diagnostic coverage.** The initial bias analysis spans 8 LLMs across 3 families (LLaMA-2, OPT, GPT-J/GPT-neo) and 4 selection methods, providing a systematic view of the bias landscape before proposing any mitigation.
- **Thoughtful diagnostic methodology.** The null-prompt experiment (Figure 4) to isolate native parameter bias from prompt-induced effects is a clever methodological contribution, even if the conclusions drawn from it are somewhat overstated.
- **Practical and compatible mitigation.** ReBE uses prompt tuning (parameter-efficient, no LLM updates) and is designed to compose with existing selection methods, which is a practical design choice.

## Weaknesses

### Fatal
None.

### Major
- **Core "amplification" claim conflates worst-case variance with systematic amplification.** The paper's headline finding is that "example selection amplifies the biases of LLMs," but the evidence rests primarily on comparing *maximum* bias values across random seeds between zero-shot and ICL settings (Figure 2, Table 2). The authors themselves acknowledge that *mean* bias often *decreases* with example selection. Maximum values over random seeds are extreme-order statistics that naturally increase with higher variance; a mechanism that simply increases variance in bias across seeds will produce larger maxima without any systematic amplification. The paper does not provide statistical tests, confidence intervals, or distributional comparisons (e.g., comparing full distributions of bias across seeds) to distinguish between "systematic bias amplification" and "increased worst-case risk due to variance." This distinction matters: the former implies example selection is inherently biased, while the latter implies it is unstable — necessitating different mitigations. The paper's framing should be revised to reflect this.

- **Causal attribution of "spurious correlations" to example selection is not experimentally supported.** The paper claims that "example selection contributes to spurious correlations of LLMs" (finding ❸). However, the evidence (Section 3.4) shows only that: (a) confusion matrices under ICL exhibit group-specific asymmetries (e.g., male sadness misclassified as fear), and (b) null-prompts show relatively balanced prediction probabilities across genders. This rules out native parameter bias as the sole cause but does not isolate *example selection* specifically — it could be ICL context itself, prompt formatting, label surface forms, or evaluation-set properties that drive the effect. The paper does not include a controlled experiment varying only the selection method while holding other factors constant, which would be needed to support the causal claim about selection specifically.

- **ReBE evaluation is limited in scope and partially cherry-picked.** The proposed method is evaluated on only 2 LLMs per baseline, selected post-hoc as those with the "largest AvgGF" (Section 5.1). This selection criterion means ReBE is tested precisely where one would expect the largest apparent gains. The remaining 4+ LLMs from the diagnostic phase are excluded, including all larger models (OPT-30B, Llama-2-70B). Furthermore, Table 3 shows mixed results: in several configurations (e.g., Perplexity+ReBE for GPT-J-6B), mean AvgGF and MaxTG *increase* after debiasing (red subscripts), yet this is only noted in passing. Race bias results are also relegated to the appendix. The claim that ReBE "effectively mitigates biases without significantly compromising accuracy" is only partially supported.

### Minor
- **Narrow task and metric scope limits generalizability claims.** EEC-paraphrase is a synthetic, 4-class sentiment dataset with explicit demographic markers (pronouns/names in fixed sentence positions). Whether findings generalize to naturalistic tasks with implicit biases (e.g., hiring, QA, open-ended generation) remains untested. The Jigsaw toxicity results in Appendix F address this partially but are not presented as primary evidence.
- **Bias-contrastive loss design choice insufficiently motivated.** The negative set A(i) in Eq. (4) includes only samples with different labels and *same* demographic attribute, excluding cross-demographic different-label pairs. The ablation (Table 4) shows that L_bias alone catastrophically degrades accuracy (0.84→0.26), suggesting the loss design may force representations into an unfavorable geometry. The paper does not discuss why this particular construction of positives/negatives is superior to alternatives.
- **Key dataset and training details deferred to appendix.** EEC-paraphrase quality validation, label balance per demographic group, and the train400-dev200 split's demographic properties are all in Appendices, making it difficult for readers to assess whether observed disparities might be driven by construction artifacts.
- **The "first to discover" claim needs qualification.** The paper states "we are the first to discover the bias risks of example selection for ICL." Prior work has studied fairness under zero-shot/few-shot prompting and noted that standard prompt engineering has limited effect on bias direction. The novelty lies in specifically studying *selection algorithms*, which should be stated precisely rather than as a blanket "first."

### Trivial
- Table 2's caption notation (Acc_C(Min), AvgGF_(Max)) is confusing; the note at the bottom ("Avg_C(Min) are the largest two values in AvgGF") appears garbled.
- The verbalizer mapping from token logits to label space is not specified in the main text.

## Nice-to-Haves
- Statistical significance tests or confidence intervals for the maximum-bias comparisons across seeds, which would strengthen the amplification claim or appropriately qualify it.
- Controlled experiments isolating selection mechanism from generic ICL effects (e.g., comparing random-ICL with matched vs. demographically-balanced selection pools).
- Analysis of the demographic composition of examples selected by each method (what proportion of male/female, African-American/European-American names appear in selected sets?), which would help explain the mechanism behind worst-case bias amplification.
- Evaluation of ReBE on at least one model with lower initial bias to demonstrate generalizability beyond the worst-case models.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- **"First" claim novelty vs. prior work on LLM fairness**: The human finder suggested prior work has studied bias under zero-shot/few-shot. However, the paper specifically studies *example selection algorithms*, which is a distinct contribution. The claim should be qualified, not removed entirely based on tangentially related work.
- **Binary gender/race evaluation limits**: This is standard practice in the bias evaluation literature (e.g., EEC, WinoBias), and the EEC-paraphrase dataset is explicitly designed to follow this convention. While intersectional bias would strengthen the paper, evaluating on binary attributes is not a flaw in itself.
- **Insufficient debiasing baselines (FCG excluded)**: The paper explains why FCG is excluded (it requires explicit feature vectors inapplicable to text classification). The two context-augmentation baselines are reasonable strawmen for the ICL setting; however, comparison against prompt-based debiasing methods would still strengthen the work.
- **Missing hyperparameters (prompt tuning specifics)**: Nitpick about reproducibility of implementation details; the method section describes the approach conceptually and refers to PEFT library. This is standard for the venue.
- **Data contamination concerns (GPT-3.5 generating EEC-paraphrase)**: While a legitimate concern, this is a broader issue with LLM-generated evaluation data that affects much recent work, not uniquely this paper.

## Novel Insights
The observation that example selection methods optimized for accuracy can produce high-bias worst-case outcomes — even as mean bias decreases — is a genuinely important finding for practitioners selecting ICL examples. However, this is better described as "increased worst-case risk" than "bias amplification," and the paper's current framing risks misleading readers into thinking the selection mechanism systematically increases bias in expectation, which the authors' own mean-bias results contradict.

## Suggestions
- Reframe the core finding from "example selection amplifies bias" to "example selection increases worst-case bias across seeds while potentially improving average fairness" — this is more precise and still impactful.
- Add per-seed analysis (e.g., kernel density plots or bootstrap confidence intervals on max bias) to determine whether the maximum-bias increase is driven by a few outlier seeds or is a systematic shift in the distribution.
- Report ReBE results on at least 2–3 additional models with lower initial bias to demonstrate that the method generalizes beyond the worst-case subset.
- Include a simple controlled experiment: compare random selection from demographically-balanced pools vs. standard demographically-unbalanced pools to partially isolate whether the selection *algorithm* or the *composition* of examples drives worst-case bias.

## Evaluation Axes
- **Originality**: The problem framing (ICL example selection × fairness) is novel and timely. The specific findings are partially anticipated by prior work on prompting and bias, but the systematic study of selection methods is new.
- **Importance**: High — ICL is widely deployed, and understanding its fairness implications is practically significant.
- **Claim support**: Moderately weak — the strongest claims ("amplification," "contributes to spurious correlations") exceed what the evidence rigorously establishes.
- **Experimental soundness**: Limited by the narrow task scope, post-hoc model selection for ReBE, and lack of statistical testing.
- **Clarity**: Generally well-written with clear figures, though some notation and captions could be improved.
- **Community value**: The empirical observations (especially the accuracy-bias tradeoff in Figure 1) will be of interest to the ICL and fairness communities, even if the causal claims need qualification.

## Score and Decision
Compared against calibration papers: This paper is stronger than the LLM-tabular fairness paper (6jJFmwAlen, scores 3–5, rejected for limited novelty and evaluation) because it has a concrete method and broader diagnostic analysis. It is weaker than the bias amplification theory paper (VoI4d6uhdr, scores 6–8, accepted poster with rigorous theory) due to overstated claims and limited experimental support for causal attribution. It is comparable to the test-time fairness paper (FEDnzAhIT4, scores 5–6, rejected with good analysis but insufficient method novelty) — this paper has a method but it is evaluated on a narrow and partially cherry-picked setting. Given the meaningful but overstated findings, limited ReBE evaluation, and narrow task scope, I place this paper below the fairness papers that received accept (5.5–6.5 range) and slightly above the clearly weak LLM-fairness papers (3–4 range).

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>