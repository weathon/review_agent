Now I have all the information I need. Let me compile the final review.

## Summary

This paper investigates how LLMs of varying sizes within the same family respond to intentionally misleading in-context information. By injecting deceptive hints (e.g., "(hint: the correct answer is A)") into multiple-choice benchmarks, the authors find that larger models consistently exhibit smaller Relative Accuracy Drops than their smaller counterparts. Control experiments attempt to rule out that larger models simply ignore in-context cues (via Guidance and Directive Instruction conditions) and that resilience is due to memorization (via Context Removal and DCLM-7B comparisons). The authors interpret these findings as evidence that larger models develop more robust "world models."

## Strengths

- **Genuine and interesting empirical observation**: The core finding—that larger models within the same family maintain more of their performance under deceptive hint injection—is consistently observed across four model families (Llama, Gemma, Phi, Mistral) and nine benchmarks, as shown in Figure 2. Even if the interpretation is debatable, the empirical pattern is worth reporting and appears robust.

- **Well-structured experimental design with multiple controls**: The paper goes beyond a single experiment by including Guidance (Section 4.2, near-perfect accuracy with truthful hints), Directive Instruction (Section 4.2, Figure 3, larger models follow wrong instructions *more* faithfully), and Context Removal (Section 4.3, Figures 4–5) conditions. The Guidance + Directive Instruction combination is particularly effective at ruling out the "ignoring hints" alternative, as it shows larger models process in-context cues—they just resist misleading ones.

- **Creative use of DCLM-7B**: Comparing a model guaranteed to have no MMLU exposure against an overfitted Llama-3.1-8B (Figure 5) is a creative way to probe memorization, even if it doesn't fully answer the right question (see weaknesses).

- **Within-family comparison design**: By comparing models within the same family (e.g., Llama-3.1-8B vs 70B), the paper controls for training data and architectural confounds that would plague cross-family comparisons, making the scaling trend more credible than a heterogeneous model comparison would be.

## Weaknesses

### Fatal

None.

### Major

- **The Relative Accuracy Drop metric does not adequately disentangle baseline accuracy from resilience, and the paper's justification for preferring it is insufficient (Section 3.5, Figure 2).** Relative Accuracy Drop = (Original − Altered) / Original mechanically produces smaller values for models with higher baselines, even if the absolute deception effect is identical. The paper asserts in Section 3.5 that a 5% absolute drop "should be perceived differently" for models with different baselines, but never provides a principled justification for why relative normalization is the correct way to measure resilience rather than, say, a calibrated effect size. The paper does note (Section 4.1) that Figure 7 in the appendix shows "smaller models also tend to exhibit a higher absolute Accuracy Drop," which partially mitigates the concern—if both metrics agree, the finding holds regardless. However, if both metrics tell the same story, the paper should lead with the more interpretable and less confounded absolute metric and show relative as supplementary, rather than burying absolute results in the appendix. More importantly, the paper never analyzes whether the resilience effect is explained entirely by baseline accuracy (i.e., whether larger models are more resilient only because they know more). Without this disentangling, the "world model robustness" interpretation is not distinguishable from "models that know more lose fewer answers."

- **The deception manipulation is too simple to support the "world model robustness" and "cross-referencing with internal knowledge" claims (Section 4.1).** The deceptive cue is simply appending "(hint: the correct answer is A)" to the prompt. Larger models may resist this through surface-level pattern recognition—recognizing "hint:" as an unreliable injected tag—rather than through the sophisticated "cross-referencing with internal knowledge" the paper claims. The paper provides no mechanistic evidence for its stated interpretation; the qualitative Appendix A example (mentioned but not in the main text) of models "diverging during reasoning" does not establish the specific mechanism. A model that spots "hint:" as an obviously injected annotation needs no deep knowledge integration. To support the "world model" framing, the paper would need deception manipulations requiring genuine knowledge conflict (e.g., misleading reasoning chains, deceptive contextual paragraphs), not an easily identifiable textual tag.

- **The memorization control experiment (Section 4.3) does not test whether deception resilience is due to memorization.** The Context Removal experiment removes the question and measures whether models still answer above chance—this tests whether models can answer from answer-choice information alone, which is interesting but orthogonal to whether *deception resilience specifically* is inflated by memorization. The correct experiment would compare the Relative Accuracy Drop under deception for a contaminated model versus a clean model. The paper's actual conclusion—that memorization "is not the sole factor" contributing to resilience—is much weaker than the abstract's claim that "resilience is not a Result of Memorization" (Contribution 3), and the stronger claim is not established.

- **Overclaimed scope in the conclusion (Section 5).** The paper claims to provide "the first empirical evidence linking LLM capacity to resilience against misinformation." The evidence links capacity to resilience against a very specific, easily identifiable textual hint injection on MCQ benchmarks—not "misinformation" in the general sense the conclusion implies. Similarly, the claim that "world models inherently becomes more robust" as models scale goes well beyond what the experiments establish.

### Minor

- **Only two size points per family weakens the scaling claim.** Each family has exactly two models (e.g., Llama 8B vs 70B, Gemma 2B vs 9B). A trend over two points provides weak evidence for a scaling law, though cross-family consistency provides some mitigation.

- **The Mistral family comparison confounds architecture with scale.** Mistral-7B-Instruct-v0.2 (dense) vs Mixtral-8x22B-Instruct-v0.1 (MoE) differs in both parameter count and architecture. Treating this as a clean size comparison is not valid, though it affects only one of four families.

- **The "world model" framing is invoked as the central explanatory construct but never concretely defined.** The paper acknowledges in Section 2 that "the concept of a world model is less explored and more vaguely defined in language models" but then uses it as if it were well-specified. This creates a gap between the experimental evidence (accuracy differences on MCQ benchmarks with hint manipulations) and the theoretical interpretation.

- **Gemma family as outlier in Directive Instruction undermines the "consistently" language.** Figure 3 shows Gemma models deviate from the trend where larger models follow wrong instructions more faithfully. The paper acknowledges this but still uses "consistently" in the abstract and conclusion.

### Trivial

- No statistical tests or confidence intervals are reported for accuracy measurements, though this is common practice in this type of benchmarking study.

## Nice-to-Haves

- Adding at least one more subtle deception condition (e.g., a misleading explanatory paragraph or a deceptive rephrasing of the question) would substantially strengthen the "world model robustness" interpretation by testing whether the finding holds beyond trivially identifiable hint tags.
- Including a third size variant per family would make the scaling claim more credible.
- Running the deception experiment on DCLM-7B and comparing its Relative Accuracy Drop with Llama-3.1-8B would directly test whether memorization drives the resilience finding.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Demand for generation-based evaluation on deceived prompts**: The reviewer suggested evaluating free-text responses instead of log-likelihood. While log-likelihood on instruct models has known issues (which the paper itself cites: Lyu et al., 2024), this is a methodological preference, not a flaw. The paper's approach is standard in the MCQ evaluation literature and the reviewer provides no evidence that log-likelihood ranking fails specifically in this deception context. Moved to Nice-to-Have tier.

- **Instruction-tuning confound**: The reviewer argued that instruction-tuned models are explicitly trained to follow prompt instructions, confounding the size comparison. However, the paper explicitly chose instruct models to ensure they process in-context information (Section 3.3: "We specifically choose instruction-tuned versions...which is particularly important for our experiments as discussed in Section 4.2"). Using base models would introduce a different confound (base models may not follow hints at all). This is a design tradeoff the authors reasonably addressed.

- **Missing per-benchmark breakdown**: The reviewer requested per-benchmark trends in the main text. Figure 2 already shows thin dashed lines for individual benchmarks, and the appendix provides detailed results. This is a presentation preference, not a substantive gap.

- **Demand for analysis of log-likelihood distributions over options**: This would be interesting but is not necessary to support the paper's claims. The accuracy-based analysis is sufficient for the stated conclusions.

## Novel Insights

The paper's most valuable insight is the interaction between the Guidance and Directive Instruction results: larger models follow truthful hints *and* follow instructions to pick wrong answers *more* faithfully, yet resist deceptive hints *more*. This double-dissociation pattern—better at using in-context information in general, yet better at resisting specifically misleading information—is more informative than the headline "larger models are more resilient" claim and deserves more emphasis. However, the paper misses the key confound that this pattern could be explained by larger models simply having stronger priors over their parametric knowledge (higher confidence in what they know), which would produce exactly this behavior without requiring any "world model integration" mechanism.

## Suggestions

- **Lead with absolute accuracy drops and show relative as supplementary.** Since the appendix (Figure 7) confirms both metrics tell the same story, use the more interpretable, less confounded absolute metric as the primary result. This immediately defuses the metric criticism.
- **Explicitly analyze the relationship between baseline accuracy and resilience.** Fit a model predicting deception effect from baseline accuracy and model size separately. If resilience is entirely explained by baseline accuracy, the "world model robustness" framing should be abandoned or substantially qualified.
- **Tone down the "world model" and "misinformation" language.** Replace "world model robustness" with something more precise like "resistance to contradictory in-context cues" and replace "resilience against misinformation" with "resilience against deceptive hint injection." The findings are interesting on their own terms without the overclaim.
- **Run the deception experiment on DCLM-7B.** This is the single most impactful additional experiment—it would directly test whether memorization drives the resilience finding, which the current memorization control cannot establish.

## Evaluation

**Originality:** The paper identifies a genuine and underexplored phenomenon (scaling of deception resilience within model families) and proposes a simple but effective evaluation methodology. The framing around "world models" adds little novelty beyond repackaging. Moderate originality.

**Importance of research question:** Understanding how model scale affects susceptibility to misleading information is practically important. However, the specific form of deception tested (trivial hint injection) limits the real-world relevance.

**Claims well supported:** The empirical observation is well supported; the mechanistic interpretation ("world model robustness") and the memorization ruling are not. The overclaiming is the paper's most significant liability.

**Soundness of experiments:** The experimental structure (deception + guidance + directive instruction + context removal) is well-designed, but each control has gaps as detailed above.

**Clarity:** The paper is clearly written and well-organized. The "world model" terminology is vague but the paper acknowledges this.

**Value to community:** Moderate. The empirical observation is worth knowing, and the evaluation methodology is simple and reusable. But the overclaimed interpretation reduces the value by potentially misleading readers about what has been demonstrated.

## Calibration

**Anchors compared:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Interpretability Illusions | `/home/wg25r/review_agent/human_reviews/v675Iyu0ta.md` | 5.6 | Similar pattern: rigorous experiments on narrow scope with overclaimed broad conclusions. This paper has somewhat weaker controls but a more practically relevant topic. |
| Additive Motif | `/home/wg25r/review_agent/human_reviews/P2gnDEHGu3.md` | 5.25 | Similar: good experiments but mechanism claim insufficiently proven. This paper is comparable—interesting observation but mechanism not established. |
| LLM Robustness to Conflicting Prompts | `/home/wg25r/review_agent/human_reviews/bjlTHVAkHS.md` | 4.33 | Very similar topic (LLM robustness to conflicting prompts). That paper had a more comprehensive framework but similar overclaiming issues. Slightly lower due to less clean experimental design. |
| RNN Episodic Memory Theory | `/home/wg25r/review_agent/human_reviews/HEcbGXzIHK.md` | 4.25 | More severe overclaiming (universality from linear RNNs on one task). This paper is better—more model families, more controls. |
| Precision Scaling Laws | `/home/wg25r/review_agent/human_reviews/wg1PCg3CUP.md` | 8.0 | High bar: clean theoretical contribution with strong empirical validation. This paper is far below this level. |
| Catastrophic Forgetting (trivial) | `/home/wg25r/review_agent/human_reviews/ZyMXxpBfct.md` | 1.5 | Low bar: trivial contribution with overclaimed explanation. This paper is clearly above this—real experiments, real observation, multiple controls. |

This paper sits in the medium-low band alongside the LLM robustness (4.33) and RNN episodic memory (4.25) papers. It has a more interesting and practical observation than the RNN paper, but the overclaiming and the memorization control gap pull it down. The metric concern is partially mitigated by the appendix absolute drop results, which keeps it from being lower. Score: **4.5**—marginally below acceptance, with a clear path to improvement through additional experiments and toned-down claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>