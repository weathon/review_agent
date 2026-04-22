Now I have read and analyzed the paper thoroughly and calibrated against relevant anchors. Let me compose the final review.

## Summary

The paper addresses length volatility—the inconsistency of output length across multiple generations—in long-form LLM generation. It makes a three-stage contribution: (1) VOLTBench, a multi-dimensional benchmark quantifying length volatility across structured/unstructured tasks, languages, and difficulty levels; (2) an attention-trace analysis identifying "Attention Collapse" and "Attention Instability" as internal precursors of generation failure; and (3) SELB (Structural Enforcement via Logits Boosting), a training-free decoding strategy that boosts section-title logits, bans EOS tokens before the final section, and suppresses conversational filler phrases to stabilize generation.

## Strengths

- **VOLTBench is a genuine and well-designed contribution.** It is the first benchmark to systematically introduce multi-sample stability evaluation for long-form generation (Table 1 shows no prior benchmark includes this). The four-dimensional design (task type, complexity, language, length) with both structured and unstructured tasks, automated quality metrics (SCA, fine-grained constraint verification), and scales up to 100 sections fills a real gap. The observation that structured tasks exhibit less volatility than unstructured tasks (Figure 3, c.1 vs. c.2) is interesting and plausibly explained by structural constraints providing stronger generation guidance.

- **The problem formulation is important and under-studied.** Figure 1 vividly demonstrates that output standard deviation can reach 103% of mean length, and the failure-mode taxonomy (Incomplete Generation, Section Skipping) in Section 4.3 clearly describes phenomena familiar to practitioners.

- **SELB produces substantial empirical improvements.** On the 100-section task, mean length increases from 6,320 to 15,651 words, LVC drops from 45.4% to 14.02%, and SCA reaches 100% vs. LongWriter-8B's 32.6%. Figure 5 visually confirms the improvement across three base models.

## Weaknesses

### Fatal
None.

### Major

- **The probe→mitigate narrative is broken: the attention analysis does not inform SELB.** The paper's central framing is a three-stage pipeline (benchmark → probe → mitigate), and Section 5 explicitly identifies "Attention Collapse" and "Attention Instability" as the root causes to target. However, SELB (Section 6) does not intervene on attention dynamics at all. Its three mechanisms are: (1) boosting section-title logits at a target word count (Eq. 2), (2) banning the EOS token before the final section (Eq. 3), and (3) banning filler phrases (Eq. 3). None of these modify, stabilize, or rescue attention to constraint tokens. The method addresses symptoms (premature termination, section skipping) through hard output constraints, not the identified root cause. The paper acknowledges "Based on our analysis of generation patterns" (Section 6.2), but the banned tokens and EOS suppression are directly observable from output failures, not from the attention analysis. The probing section could be removed entirely without affecting the method. This disconnect undermines the paper's claimed contribution pipeline.

- **SELB's headline improvements are trivially expected consequences of hard decoding constraints.** Banning the EOS token directly forces the model to keep generating, trivially increasing mean output length and reducing length volatility (a model that cannot stop early will, by definition, produce longer and less variable outputs). Forcing section transitions at target word counts directly enforces structural completeness, trivially improving SCA from 32.6% to 100%. The paper presents no ablation separating these effects—there is no experiment showing what happens if only EOS-banning is applied, or only section-forcing, or only filler-banning. There is also no comparison against simple baselines (e.g., "continue generating until N tokens are reached"). Without such decompositions, the 148% length increase and 69% volatility reduction cannot be attributed to principled intervention rather than to the obvious effects of preventing early termination.

- **The attention analysis is limited to visual inspection of two models on a single task, with no quantitative evaluation.** "Attention Collapse" and "Attention Instability" are identified from attention traces of Qwen2.5-3B and Qwen2.5-7B on one diary-generation task (Figure 4). There is no quantitative criterion for what constitutes "collapse" or "instability," no statistical test linking attention drops to output failures, no evaluation across multiple tasks or models, and no predictive experiment. Calling these "common internal patterns" overstates the evidence significantly.

### Minor

- **N=5 samples per instruction may be insufficient for reliable volatility estimation.** All volatility metrics (LSD, LVC, FAD) are computed from 5 generation runs. Under normality assumptions, the relative standard error of the sample standard deviation is approximately 0.35, giving confidence intervals spanning roughly ±70% of the estimate. This affects benchmark rankings and comparison claims, though it is common practice for expensive LLM generation benchmarks.

- **SELB requires known section titles, limiting generality.** Equation 2 requires $V_{title}^{(p+1)}$, the set of tokens for the next section title, which assumes a known template (e.g., "Chapter 5"). The free-form generalization (SELB-Hybrid, Section 6.4) is described only at a high level in the main text with key details and evaluation relegated to the appendix. The core experiments only validate SELB on template-structured tasks.

- **Including Claude-3.5-Sonnet in the volatility comparison is misleading.** Table 2 shows Claude generating only 176 words on a 100-section task. Its LVC of 1.9% is trivially low because a model that barely generates anything cannot exhibit high variance in generation length. The paper excludes Claude from quality analysis but still includes it in the volatility comparison.

## Nice-to-Haves

- **Ablate each component of SELB independently** (EOS-banning alone, section-forcing alone, filler-banning alone) and compare against a simple "generate until N tokens" baseline. This would clarify whether SELB's gains come from principled intervention or from forcing non-termination.

- **Quantify the attention patterns** (e.g., define a numerical criterion for "collapse," compute its prevalence across models/tasks, correlate with failure rates). This could transform an interesting qualitative observation into a falsifiable claim.

- **Provide attention traces with and without SELB** to show whether SELB actually affects attention dynamics (which it claims to address) or merely constrains the output distribution without changing internal representations.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Claim that the paper does not deliver on "lacking in-depth investigation" (Introduction's third limitation):** The paper does provide an in-depth investigation in the form of the attention trace analysis in Section 5; the issue is not that it is absent but that it is shallow and does not connect to the method. This is already captured in the Major weakness above.

- **Claim that Table 2 mixes unfair comparisons (API models vs. open-source):** The paper explicitly notes excluding Claude from quality analysis due to its low output. Comparing diverse models on a benchmark is standard practice and the table provides useful context about model capabilities.

- **Complaint about the paper criticizing LLM-as-a-Judge while using it for UCA:** The paper criticizes over-reliance on LLM-as-a-Judge as the sole metric for prior benchmarks, not its use per se. VOLTBench addresses this by introducing structured tasks with objective SCA and fine-grained constraint automation; UCA is used only as a complement for unstructured tasks where no fully objective metric exists.

- **Concern about averaging attention across all layers and heads:** This is a design choice; many interpretability papers average across layers/heads to obtain a global signal. The paper is transparent about this. It is a minor methodological choice, not a fundamental flaw.

- **Demand for confidence intervals on volatility metrics:** While desirable, single-run benchmarking without CIs is the norm in this community. Making this a strict requirement would be unfair relative to community standards.

- **Claim that the appendix-only treatment of SELB-Hybrid is a weakness:** The parser strips appendices. The original submission likely contains these details, and the main text provides a reasonable summary of the approach.

- **Concern about SELB generating superficial or incoherent content:** This is speculative. The paper reports 86.7% UCA and 100% SCA on structured tasks, and the lexical diversity analysis in Appendix G addresses repetition concerns. Without evidence that generated content is hollow, this remains speculation.

## Novel Insights

The observation that structured tasks yield far more stable outputs than unstructured tasks (Figure 3, c.1 vs. c.2) suggests that explicit structural scaffolding—rather than raw model scale or training data—is the key lever for reliable long-form generation. This is consistent with SELB's success precisely because SELB imposes structural scaffolding at decoding time. However, this insight also reveals an ironic tension: SELB's effectiveness stems from external structural enforcement, not from understanding or fixing the internal attention dynamics the paper claims to identify. The real lesson may be that for long-form generation, controllable output structure matters more than controllable internal representations.

## Score and Decision

**Calibration anchors:**

1. **TAME (avg 7.0, Accept Poster):** Attention-mechanism analysis + training-free decoding strategy for hallucination. Closes the loop: intervention directly targets the identified attention failure mode (eigenvalue variance). This paper *does not* close that loop—SELB doesn't touch attention.

2. **Instructive Decoding (avg 7.5, Accept Spotlight):** Training-free logit adjustment with extensive experiments and ablations. Simpler idea but thoroughly validated and honest about scope.

3. **KinDEL (avg 5.5, Reject):** Valuable benchmark/dataset contribution with weak method novelty and limited analysis. Closest structural analog to this paper (good benchmark + weak method).

4. **AttentionDefense (avg 2.33, Reject):** Attention-based analysis with weak benchmark comparisons and disconnect between problem framing and solution. Similar disconnect problem but weaker overall contribution.

This paper is better than AttentionDefense (stronger benchmark, real problem) but worse than both TAME and Instructive Decoding (broken probe→mitigate narrative, no ablation, trivially expected method improvements). It sits near KinDEL in quality: a useful benchmark contribution, but a method contribution that is oversold relative to its actual substance. The benchmark alone is a solid contribution; the probing and mitigation pillars are significantly overclaimed.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>