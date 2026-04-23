Now I have all the information needed. Let me synthesize the final review.

## Summary

This paper investigates what constitutes a suitable reward function for self-improving chain-of-thought (CoT) reasoning on unstructured text at pretraining scale. It introduces a "what"/"where" analytical framework to diagnose failures of common reward functions—showing that standard loss is worse than random at identifying useful CoT placement locations—and proposes Reasoning Advantage (RA), a clipped and normalized loss-based reward function. RA is demonstrated to be the only reward function enabling self-improvement on the authors' new MMLU-FREE-FORM benchmark, while an exploratory experiment on OpenWebMath fails, which the authors attribute to insufficient CoT diversity.

## Strengths

- **The "what"/"where" analytical framework is a genuine conceptual contribution.** Decomposing reward function quality into (1) what reasoning is rewarded and (2) where reasoning is rewarded provides principled criteria that prior work (including Quiet-STaR) lacks. The finding that standard loss performs *worse than random* at identifying optimal CoT placement locations (39.4 AUC, Table 2) is striking and important.

- **RA is simple, well-motivated, and empirically validated in the diagnostic experiments.** Table 2 shows RA achieving 66.3% what-accuracy and 77.0 where-AUC versus 44.6/39.4 for standard loss. The histogram in Figure 1 effectively communicates why clipping matters—loss conflates incorrect and random CoTs, while RA separates them.

- **MMLU-FREE-FORM is a useful intermediate benchmark.** By removing MMLU's multiple-choice format, it creates a dataset that mirrors the challenges of unstructured text (no exact-match reward, variance in answer format) while maintaining a higher density of reasoning opportunities than typical pretraining corpora (Section 5.2, lines 179–181).

- **The paper is commendably honest about the OpenWebMath failure** and provides insightful analysis: RA succeeded at identifying the best available CoTs, but only 0.01% exceeded the threshold, and those were overly conservative (Section 6.1). The diagnosis that the bottleneck is CoT diversity rather than reward quality is specific and actionable.

- **RA's design is driven by analysis rather than brute-force search.** Clipping addresses the "what" problem (distinguishing random from incorrect CoTs); baseline incorporation addresses the "where" problem (avoiding trivially predictable suffixes). Each design choice is motivated by a specific diagnosed failure mode.

## Weaknesses

### Fatal
None.

### Major

- **The self-improvement experiment (Section 5.2) relies on expected accuracy rather than actual generation accuracy.** The paper evaluates using "answer probability" (P(correct answer | question, CoT)), which is a proxy for—but not identical to—generation accuracy. While the paper is transparent about this (line 187: "This metric is also known as 'expected accuracy'"), the headline claim of "improving zero-shot transfer accuracy on GSM8K by nearly 7%" (line 38) rests on this proxy. A model can assign higher probability to the correct answer while still generating incorrect completions more often due to miscalibration or mode coverage issues. The absence of any actual generation accuracy report makes it impossible to verify whether the proxy metric translates to real output improvements. This matters because the paper's central empirical claim—RA enables self-improvement—is only supported through this indirect measure.

- **Different filtering thresholds across methods yield different training data quantities, confounding the comparison.** RA filters much more aggressively (only ~1,000 training steps of data above threshold, line 201), while other methods train for up to 4,000 steps. This makes it impossible to disentangle whether RA's benefit comes from better filtering quality or from using less but higher-quality data (a data quantity vs. quality confound). A controlled comparison matching the number of training samples across methods would significantly strengthen the conclusions.

### Minor

- **The severe degradation of the random filtering baseline on GSM8K transfer (Figure 2b, dropping from ~0.25 to ~0.10) is not discussed.** This is an important observation: training on randomly-filtered CoT-inserted data *actively harms* out-of-distribution performance. While the loss-based methods don't degrade (they slightly improve to ~0.26–0.27), the random baseline's collapse suggests that naive CoT insertion can introduce harmful distribution shift. The paper should acknowledge and investigate this, as it provides important context: part of RA's advantage may be its aggressive filtering that removes potentially harmful training samples, not just its selection of beneficial ones.

- **The OpenWebMath experiment provides limited evidence for the diversity hypothesis.** The diagnosis is based primarily on a single statistic (0.01% of CoTs above threshold) and qualitative examples. Quantitative diversity metrics (e.g., distinct n-gram ratios, embedding-space spread, reward distribution statistics across the generated pool) would make the diagnosis more compelling.

- **The "what" experiment's CoT categories (correct vs. incorrect) are defined by generation procedure rather than reasoning quality.** "Correct" CoTs are generated with suffix access (post-rationalization) and "incorrect" without (Section 5.1, line 145). While the paper acknowledges that incorrect CoTs "often exhibit sophisticated reasoning" but don't predict the suffix as well, the categories conflate generation method with reasoning quality. A post-rationalized CoT could contain flawed reasoning despite predicting the suffix; a non-post-rationalized CoT could contain valid partial reasoning. This limits interpretability of the "what" results.

### Trivial

- **The RA sign convention could be stated more explicitly.** Since R_clipped values are negative (log probabilities), RA = (R_with_CoT − R_without_CoT) / R_without_CoT yields a *negative* value when CoT helps (positive numerator, negative denominator). The filtering threshold "< -0.2" is consistent with this, but the paper never explicitly addresses the sign convention, making the formulation harder to follow than necessary.

- **Potential instability when the denominator (empty-CoT baseline) approaches zero is not discussed.** For highly predictable suffixes, R_clipped(p, " ", s) approaches 0, making the RA ratio unstable. While clipping constrains the range, the edge case deserves acknowledgment.

## Nice-to-Haves

- A controlled experiment where all reward functions filter to the same number of training samples, separating filtering quality from data quantity effects.
- Actual generation accuracy (sampling completions) on MMLU-FREE-FORM and GSM8K, in addition to expected accuracy.
- Testing RA with an online RL method (e.g., REINFORCE with RA as reward), not just offline filtering. The paper motivates RA as a reward function for RL but only evaluates it in an SFT-with-filtering pipeline.
- Quantitative diversity analysis of generated CoTs in the OpenWebMath experiment (distinct n-gram ratios, embedding-space spread).

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Compute-only scaling" framing is aspirational rather than supported** (Harsh Critic): The paper explicitly acknowledges it does not solve the full unstructured pretraining setting (lines 40, 51, 207). The framing is about the *direction* of research, not a claim of achievement. The abstract and conclusion are measured in their claims.

- **LLM-as-judge "Yes" with footnote is misleading** (Harsh Critic): Table 1 accurately reflects that LLM-as-judge doesn't require *external* intelligence if the same model is used. The footnote explains a practical difficulty, which is additional information rather than a contradiction of the criterion.

- **The "what" experiment doesn't validate transfer to truly unstructured text** (Harsh Critic): This is scope creep. The paper explicitly designs MMLU-FREE-FORM as an intermediate benchmark and separately attempts OpenWebMath. Criticizing the "what" experiment for not using unstructured text misses its stated purpose.

- **The paper claims "RA avoids catastrophic degradation rather than enables improvement"** (Harsh Critic): This is factually misleading. In Figure 2b, the loss-based methods (loss, clipped loss, delta loss) *do* improve GSM8K transfer (from ~0.25 to ~0.26–0.27). Only the random baseline degrades. RA's improvement (~0.25 to ~0.28) is clearly beyond what other loss-based methods achieve, not merely avoidance of harm.

- **Strength Finder's claim that RA "substantially outperforms existing reward functions on both criteria"**: The margin over delta loss (66.3 vs 58.3 on what, 77.0 vs 64.4 on where) is meaningful but not overwhelming. "Substantially" overstates the gap, especially since non-normalized RA variants perform similarly on "what" (as noted in line 149 and Appendix B.1).

- **Strength Finder's garbled text about RA being the only function with both what/where**: The Strength Finder's output degraded into incoherent text for this point; removed as unusable.

## Novel Insights

The paper reveals an underappreciated asymmetry in reward function design: the same modification (clipping) that helps with "what" reasoning is rewarded is largely orthogonal to the modification (baseline incorporation) that helps with "where" reasoning is rewarded. This decomposition suggests that future reward function improvements may benefit from independently targeting these two axes rather than optimizing a single scalar. Additionally, the random baseline's catastrophic degradation on GSM8K transfer suggests that naive CoT insertion during fine-tuning can actively harm out-of-distribution reasoning—a risk that the community should be more aware of.

## Suggestions

- Report actual generation accuracy on at least GSM8K to validate that expected accuracy improvements translate to real output improvements.
- Run a controlled comparison with matched training data quantities across reward functions to isolate the quality-of-filtering effect from the data-quantity effect.
- Discuss why random CoT insertion harms GSM8K transfer and what this implies about the distribution shift introduced by CoT-augmented fine-tuning.

## Evaluation

**Originality:** The "what"/"where" framework and the RA reward function are genuine contributions. While the individual components (clipping, baseline subtraction, normalization) are not novel in isolation, their specific combination and the analytical justification for each is original. The finding that loss performs worse than random at "where" is novel and important.

**Importance of research question:** The question of suitable reward functions for self-improving reasoning on unstructured text is timely and significant, as it directly addresses a key bottleneck for compute-only scaling of reasoning.

**Claim support:** The diagnostic claims are well-supported (Tables 2–4, Figure 1). The self-improvement claim relies on a proxy metric and has a data-quantity confound, which weakens—but does not invalidate—the evidence. The OpenWebMath failure is honestly reported.

**Experimental soundness:** The "what" and "where" experiments are well-designed. The self-improvement experiment has the limitations noted above. The OpenWebMath experiment is exploratory and appropriately scoped.

**Clarity:** The paper is generally well-written with clear motivation and structure. The RA sign convention could be clearer.

**Community value:** The what/where framework, RA reward function, MMLU-FREE-FORM benchmark, and the honest failure analysis on OpenWebMath all provide value for future research in this direction.

## Score and Decision

**Calibration comparison:**

- **High anchors:** Rewarding Progress (avg 7.14, Spotlight): similar topic (reward design for reasoning) but stronger theoretical grounding and more comprehensive empirical validation. Lean-STaR (avg 7.50, Spotlight): extends STaR with clear positive end-to-end results. This paper is below these due to the proxy metric limitation and the OpenWebMath failure.
- **Medium anchors:** Language Imbalance Driven Rewarding (avg 5.0, Accept Poster): interesting self-improvement direction but with limited evaluation and unclear generalizability. This paper has a clearer analytical framework and more thorough diagnosis. RFT/Scaling (avg 5.25, Reject): empirical study of scaling for math reasoning; this paper is more novel methodologically. Meta-Rewarding (avg 5.0, Reject): self-improvement with more fundamental issues (evaluation reliability, model coverage). This paper is above these.
- **Low anchors:** SALA (avg 2.50, Reject): minimal novelty, poor experiments. Language of Thoughts (avg 4.75, Reject): unclear claims, poor presentation. This paper is far above these.

This paper sits above the medium anchors due to its genuine analytical contributions (what/where framework, the striking loss-worse-than-random finding) and honest reporting, but below the high anchors due to the proxy metric in the self-improvement experiment and the OpenWebMath failure. The contributions are primarily analytical/diagnostic rather than a strong end-to-end empirical demonstration, but the analysis itself is valuable and well-executed.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>