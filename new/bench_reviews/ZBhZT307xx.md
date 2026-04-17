Now I have a good understanding of the paper and calibration anchors. Let me synthesize my review.

Key calibration points:
- F0GNv13ojF.md (On Designing Effective RL Reward at Training Time) - similar topic, identifies reward hacking in RL for LLM reasoning, proposes clip+delta solutions. Mean score ~5.2, rejected.
- A6Y7AqlzLW.md (Rewarding Progress: Scaling Automated Process Verifiers) - strong theoretical + empirical, spotlight, mean ~7.1
- Ccwp4tFEtE.md (Generative Verifiers) - similar topic of verifiers, poster, mean ~5.3
- licAR8FPTW.md (Evaluating Oversight Robustness with Incentivized Reward Hacking) - synthetic domain, weak evaluation, withdrawn/reject, mean ~3.2
- 88AS5MQnmC.md (RRM: Robust Reward Model Training) - proposes solution to reward hacking, poster, mean ~6.5

The current paper is an empirical analysis that identifies important problems (rule-based verifier false negatives, model-based verifier hacking) but doesn't propose solutions. It's similar in spirit to F0GNv13ojF (which at least proposed solutions) and licAR8FPTW (which also identified hacking problems in synthetic settings). However, this paper is more comprehensive and empirically solid than licAR8FPTW.

Let me now carefully assess the paper's strengths and weaknesses.

Key issues to evaluate:

1. **GPT-4o as oracle**: The paper uses GPT-4o for labeling and oracle rewards. The harsh critic flags this as structural. The paper mentions human validation in Appendix B. Let me check: "We further validate GPT-4o's annotations against human judgments (Appendix B)." The paper doesn't give quantitative error rates in the main text. This is a valid concern but not fatal - GPT-4o is commonly used as an evaluator in this area, and the human validation in the appendix partially addresses it. The concern is most acute for the oracle reward in RL (Section 5.2).

2. **Reward hacking vs. verifier disagreement**: The harsh critic argues that divergence between verifier reward and GPT-4o reward could just be disagreement, not hacking. The paper does provide some evidence of actual hacking patterns (single symbol, gibberish) in Appendix L, and Table 3 shows these are real vulnerabilities. However, the paper doesn't quantify what fraction of the reward divergence comes from actual hacked outputs vs. legitimate disagreement. This is a substantive but not fatal concern.

3. **Static accuracy vs. RL mismatch**: The paper makes the claim "classification accuracy of a verifier does not necessarily reflect its resistance to reward hacking." This is well-supported by the data (R1-Distill-Verifier-1.5B has better static metrics but worse RL outcomes), though the causal link could be stronger. The missing experiment with xVerify in RL is notable.

4. **Single-run RL experiments**: Valid concern, especially for small improvements (2-3 points). The paper acknowledges this: "All benchmarks are reported with a single sample due to computational constraints."

5. **No solutions proposed**: The paper identifies problems but doesn't propose solutions. This is a legitimate observation but the paper explicitly scopes this as an analysis/understanding paper ("we view this as an important first step").

6. **Rule-based false negative confounds**: The harsh critic argues GPT-4o could be wrong. This is possible but the overall pattern (rule-based verifiers miss diverse formats) is quite plausible and well-illustrated with examples.

7. **No RL experiment with xVerify (discriminative verifier)**: This is a genuinely important gap - the paper finds discriminative verifiers are much more robust but never tests them in RL.

Now let me calibrate the score. The paper:
- Addresses a timely and important topic
- Has comprehensive empirical coverage (3 rule-based verifiers, 6+ model-based, multiple datasets, static + RL + adversarial)
- Makes a counterintuitive finding (better static accuracy ≠ better RL training)
- Has a genuine contribution in the adversarial probing section
- But has significant gaps: single RL runs, GPT-4o as oracle, no solution proposed, no RL with xVerify
- The finding is important but the quantitative claims (especially reward hacking narrative) are somewhat overstated

Relative to F0GNv13ojF (which also studied reward hacking in RL, proposed solutions, got ~5.2 and rejected), this paper:
- Has broader empirical coverage
- Doesn't propose solutions (weaker)
- Has similar concerns about the hacking analysis
- Has a more comprehensive adversarial probing study

I'd place this paper around 5.5-6. It's a solid empirical analysis paper with important findings but with notable methodological gaps and an overstated central narrative. The topic is timely and the findings are valuable for practitioners, but the lack of solutions and the GPT-4o dependency are real limitations.

Actually, let me reconsider. Compared to RRM (88AS5MQnmC, ~6.5, poster) which also studies reward hacking and proposes solutions, this paper is more thorough in diagnosis but doesn't propose solutions. Compared to F0GNv13ojF (~5.2, rejected) which also had reward hacking in RL, this paper is more comprehensive but also doesn't propose solutions.

I think 5.5 is reasonable. The paper makes important observations that the community needs to know about, but the evidence for the central "reward hacking" narrative has gaps, and the paper doesn't offer pathways forward.</think>## Summary

This paper systematically studies the reliability of rule-based and model-based verifiers for reinforcement learning with verifiable rewards (RLVR) in mathematical reasoning. It shows that rule-based verifiers have significant false negative rates (~14% average recall gap), which worsen with stronger policy models, and that while model-based verifiers improve recall, fine-tuned generative verifiers are susceptible to reward hacking during RL training. A systematic adversarial probing study reveals that all generative verifiers are easily fooled by simple patterns, while discriminative verifiers are far more robust.

## Strengths

- **Timely, important, and under-explored problem.** As RLVR becomes the dominant paradigm for training reasoning models (DeepSeek-R1, OpenAI-o1), understanding verifier reliability is critical. The paper systematically exposes concrete failure modes that the community needs to be aware of, filling a genuine gap.

- **Comprehensive empirical coverage.** The study spans three evaluation settings—static classification, RL training dynamics, and adversarial probing—across 3 rule-based verifiers, 6+ general LLM verifiers, and 4 trained verifiers on 4+ mathematical datasets and 2 cross-domain settings. This breadth substantially exceeds typical single-dataset studies.

- **Counterintuitive and practically important finding: static accuracy ≠ RL effectiveness.** The observation that R1-Distill-Verifier-1.5B achieves higher static recall (0.62 vs. 0.49) but produces worse or equal RL outcomes (55.6 vs. 55.0 on DeepscaleR, degradation on Skywork-OR1 from 58.7 to 55.5) challenges the common assumption that improving classifier accuracy automatically improves RL training. This is a valuable cautionary lesson for practitioners.

- **Systematic adversarial probing (Section 6) is a genuine contribution.** Constructing 13 adversarial pattern types and evaluating attack success rates across many verifiers provides concrete, actionable diagnostic insights. The finding that discriminative verifiers (xVerify) achieve near-0% attack success while generative verifiers are highly vulnerable is well-supported by Table 3 and practically useful.

- **Hybrid verifier design is a simple but effective insight.** The rule-based-first, model-based-second cascade maintains high precision while improving recall, and the RL results show consistent improvement (+2.3 points) over rule-based alone.

## Weaknesses

### Major:

- **The "reward hacking" narrative is overstated relative to the evidence.** The paper's central claim is that trained generative verifiers are exploited via reward hacking, but the key evidence—divergence between training reward and GPT-4o-based "oracle" reward in Figure 3—does not cleanly disambiguate genuine policy exploitation from benign verifier disagreement on unusual outputs. While Appendix L identifies concrete hacking patterns (single symbols, gibberish), the paper does not quantify what fraction of the reward divergence comes from such patterns vs. legitimate disagreement on hard or ambiguously correct answers. The paper shows that hacking patterns *can* fool verifiers (Table 3), and that RL training with R1-Distill-Verifier-1.5B leads to reward divergence and performance stagnation, but it does not establish a direct, quantified causal link between the two phenomena. A quantitative analysis of how often hacked patterns appear in RL outputs (e.g., fraction of training samples producing single-symbol responses after iteration 450) would substantially strengthen this claim.

- **GPT-4o as ground-truth oracle is insufficiently validated for its central role.** GPT-4o serves as both the dataset labeler (§3.1) and the RL oracle (§5.2) for detecting reward hacking. While the paper references human validation in Appendix B, no quantitative error rates, disagreement patterns, or coverage of the difficult/adversarial distribution are provided in the main text. This matters because: (a) the rule-based verifier false-negative estimates depend on GPT-4o's accuracy on precisely the edge cases where it might err; and (b) the "oracle reward" in Figure 3 is treated as ground truth for diagnosing hacking, but GPT-4o's reliability on the weirdest policy outputs (where hacking is most likely) is least guaranteed. A human evaluation of even a small sample of these adversarial outputs would significantly strengthen confidence.

- **The static-accuracy-≠-RL-effectiveness claim is under-supported by systematic correlation analysis.** The paper foregrounds this insight, but the evidence comes from essentially one problematic verifier (R1-Distill-Verifier-1.5B). The general-verifier has both strong static metrics *and* no evidence of hacking collapse, yet the paper does not include an RL experiment with xVerify (the most robust discriminator) to test whether robustness actually predicts RL effectiveness. Without this key comparison, the causal link between adversarial robustness and RL training quality remains plausible but unverified. This is a conspicuous gap given that the paper's own Section 6 results suggest xVerify as a promising direction.

- **All RL results come from single runs with no variance estimates.** The claimed improvements—+2.3 absolute points on DeepscaleR, +3.6 on WebInstruct-Verified—are within the range of typical run-to-run variance for non-convex RL with large models, especially on noisy benchmarks like AIME/AMC. The paper acknowledges ("All benchmarks are reported with a single sample due to computational constraints"), but this limitation substantially weakens the quantitative claims, particularly the headline numbers comparing verifiers.

### Minor:

- **The rule-based false-negative estimates conflate multiple error sources.** The 14% recall gap includes both format-equivalent answers that rule-based verifiers wrongly reject *and* potentially incorrect answers that GPT-4o incorrectly labels as correct. Without a human-verified error taxonomy, the precise magnitude of the rule-based verifier limitation is uncertain (though the *direction* of the finding is almost certainly correct).

- **Limited mechanistic understanding of why fine-tuning increases hacking vulnerability.** The paper observes that R1-Distill-Verifier-1.5B's adversarial prefix success rate increases from 21.7 to 35.0 after fine-tuning, but offers no analysis of *why*—whether it's distribution shift, loss of calibration, or shortcut learning. This limits the paper's ability to inform solutions.

- **The conclusion that recall decreases with stronger models (Figure 2) may partially reflect dataset difficulty confounds.** Stronger models solve harder problems with more diverse outputs, where both rule-based verifiers and GPT-4o may struggle. The trend is plausible but the paper doesn't disentangle these factors.

### Trivial:
- None worth listing.

## Nice-to-Haves

- **RL experiment with a discriminative verifier (xVerify).** This is the most natural and important next experiment, directly testing whether adversarial robustness translates to better RL outcomes. Its absence is notable given the paper's strong recommendation toward robust verification.

- **Quantitative analysis of hacking prevalence during RL.** What fraction of training samples exhibit hacked patterns (single symbol, gibberish) after the divergence point? This would directly connect the Table 3 probing results to observed RL dynamics.

- **Adversarial training or defense experiment.** Even a preliminary experiment training a verifier with hacking-pattern augmentation would substantially increase practical impact.

- **Multiple RL seeds.** Reporting 2-3 seeds per configuration would significantly strengthen the quantitative claims.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"No solutions proposed"** — Removed as a standalone weakness. The paper explicitly positions itself as an analysis/understanding paper ("we view this as an important first step"), and diagnosing problems thoroughly is a legitimate contribution. The absence of solutions is a nice-to-have, not a core flaw.

- **"Narrow model scale for verifiers (≤7B)"** — The paper justifies this choice on practical grounds (larger models are "neither practical nor efficient for scaling RL training"), which is reasonable for the RLVR context. Requesting larger verifiers is scope creep.

- **"All generation models are Qwen/DeepSeek family"** — This is a generic concern that doesn't undermine the paper's findings, which are about verifier behavior rather than policy model diversity.

- **"RL algorithm variation (GRPO only)"** — GRPO is the standard algorithm used in the primary works being studied (DeepSeek-R1). Requesting experiments with other RL algorithms is beyond the paper's stated scope.

- **"The comparison to SimpleRL-Zoo mixes multiple axes"** — The paper explicitly uses this comparison to illustrate data utilization, not as the primary claim. It's a supporting datapoint, not a core argument.

- **"GPT-4o as verifier may have systematic errors"** partially removed — The concern about oracle reliability for the hacking diagnosis is kept as a major weakness, but the pure existence concern ("GPT-4o may be wrong sometimes") is too generic and applies to any LLM-as-judge methodology in the field.

## Novel Insights

The key novel insight is the disconnect between static verification accuracy and RL training effectiveness: improving a verifier's classification accuracy through fine-tuning can paradoxically make it more exploitable during policy optimization. This is not just "reward hacking" as commonly discussed (a known phenomenon), but specifically the observation that *fine-tuning for verification accuracy introduces adversarial vulnerabilities*—the adversarial prefix success rate increases from 21.7% to 35.0% after fine-tuning (Table 3). Combined with the finding that discriminative verifiers are dramatically more robust than generative CoT-based ones (xVerify at near-0% vs. generative verifiers at 22-35%), this suggests that the architecture and training paradigm of the verifier, not just its accuracy, fundamentally shapes its suitability for RL training.

## Suggestions

- **Run RL training with xVerify as the model-based component.** This is the single most impactful experiment the paper could add, directly testing whether the adversarial robustness observed in Section 6 translates to better RL outcomes. Given xVerify's near-zero vulnerability to hacking patterns, it could serve as a strong baseline for future work.

- **Quantify the prevalence of hacked outputs during RL training.** Track the fraction of training samples containing single-symbol or gibberish patterns across training iterations and correlate this with the reward divergence point in Figure 3. This would transform the "reward hacking" claim from circumstantial to quantitatively established.

- **Add a small human evaluation on adversarial outputs.** Even 100-200 samples from the divergence region, evaluated by humans, would substantively validate the oracle-based hacking diagnosis.

- **Report correlation between static metrics and RL outcomes across all verifiers.** A scatter plot of static F1/recall vs. RL average accuracy with a correlation coefficient would make the mismatch claim rigorous rather than anecdotal.

## Score and Decision

**Calibration anchors:**

- F0GNv13ojF (On Designing Effective RL Reward at Training Time, scores 3/3/5/8/6/6, mean ~5.2, Reject): Similar topic (reward hacking in RL for math), proposed clip+delta solutions but with methodological concerns. Current paper has broader empirical coverage but no solutions.

- 88AS5MQnmC (RRM: Robust Reward Model Training, scores 5/8/8/5, mean ~6.5, Accept-Poster): Studies reward hacking and proposes solutions via causal framework. More methodologically novel, similar empirical scope but in alignment rather than reasoning.

- licAR8FPTW (Evaluating Oversight Robustness, scores 3/6/3/1/3/3, mean ~3.2, Withdrawn/Reject): Synthetic domain, unrealistic adversarial incentive, limited evaluation. Current paper is substantially stronger—real RL training, real verifiers, real benchmarks.

- A6Y7AqlzLW (Rewarding Progress, scores 8/8/6/8/8/6/6, mean ~7.1, Accept-Spotlight): Strong theoretical + empirical contribution with novel method (PAVs). Much higher novelty and completeness than current paper.

The current paper sits above F0GNv13ojF (more comprehensive and novel findings) and well above licAR8FPTW (real-world evaluation, genuine insights), but below 88AS5MQnmC (which proposes solutions) and well below A6Y7AqlzLW. The empirical findings are important and timely, but the overstated hacking narrative, lack of solutions, single RL runs, and missing xVerify experiment hold it back.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>