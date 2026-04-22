Now I have a comprehensive understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

The paper introduces a three-stage framework for studying output volatility in long-form LLM generation: (1) VOLTBench, a multi-dimensional benchmark that systematically quantifies length volatility across structured/unstructured tasks, languages, and instruction complexities; (2) attention trace analysis identifying "Attention Collapse" and "Attention Instability" as internal patterns correlated with generation failures; and (3) SELB (Structural Enforcement via Logits Boosting), a training-free decoding strategy that forces section transitions and suppresses failure-mode tokens to improve output stability.

## Strengths

- **VOLTBench is a genuinely novel and well-designed benchmark.** Table 1 shows it is the only benchmark among those compared that includes both "Multiple Sampling" and "Stability Eval" while covering unstructured and structured tasks with scaling to ~100k words. The chapter-based format enabling controlled scaling from 5 to 500 sections is a practical and extensible design choice.

- **The problem framing is important and underappreciated.** Identifying output volatility (inconsistency across multiple generations) as a distinct evaluation dimension is a meaningful contribution. The finding that LongWriter-8B — a model specifically trained for long-form generation — has the highest LVC (45.4%) is striking and counterintuitive (Table 2).

- **Cross-dimensional analysis yields actionable insights.** Figure 3 shows that structured tasks yield longer and more stable outputs, and Section 4.3.1 documents sharp constraint-following degradation beyond 100 sections, quantifying a previously underappreciated limitation. These findings have practical implications for prompt design.

- **SELB is training-free and operates at decoding time.** Unlike data-centric approaches (e.g., LongWriter's specialized SFT) or RL methods (LongWriter-Zero), SELB requires no additional training, making it immediately applicable to any autoregressive LLM.

- **Fine-grained constraint evaluation provides objective quality signals.** The character-level, keyword, and theme constraints (Section 4.2) allow automated quality assessment even for unstructured tasks, and the SCA metric uses execution-based verification for code tasks.

## Weaknesses

### Fatal
None.

### Major

- **SELB's gains are largely artifacts of hard decoding constraints, not evidence of addressing the underlying attention failures.** Equations 2–3 reveal that SELB works by (a) forcibly injecting section-title tokens when a section reaches target length, and (b) banning EOS tokens and filler phrases until the target section count is reached. These are hard constraints on the output space — if you ban the stop token and force structural transitions, longer and more stable outputs are guaranteed by construction. The paper's own probing identifies attention collapse and attention instability as root causes (Section 5), yet SELB does nothing to stabilize or restore attention; it simply forces the model to continue generating past the point where attention collapses. The "69% volatility reduction" and "148% length improvement" are direct consequences of overriding the model's decoding, not evidence that the generation process has improved. While constrained decoding is a legitimate technique, the paper's framing obscures the distinction between symptom suppression and root-cause intervention.

- **The three-stage pipeline (benchmark → probe → mitigate) implies a causal connection between the attention analysis and SELB's design that is not validated.** The paper states "Based on these insights, we propose SELB" (Section 6), but SELB's two components — forcing section transitions and banning failure tokens — are generic strategies that could have been designed without any attention analysis. The attention analysis identifies attention collapse → premature termination and attention instability → section skipping, which do loosely motivate SELB's two components, but no ablation shows that attention insights specifically improved SELB's design, no experiment directly stabilizes attention to test the causal claim, and no evidence shows that models with more stable attention traces produce less volatile outputs under standard decoding. The correlation between attention patterns and output volatility (Section 5) is presented as if it establishes causation, but the mitigation does not test this causality. This disconnect undermines the paper's central narrative.

### Minor

- **The headline "148%" improvement claim is ambiguously phrased.** The abstract states SELB "improves the mean output length of the base model by 148%," where "base model" most naturally refers to Qwen2.5-7B (the model SELB is built on). However, the 148% figure matches the comparison with LongWriter-8B ((15,651 − 6,320) / 6,320 ≈ 148%), not Qwen2.5-7B ((15,651 − 445) / 445 ≈ 3,417%). If the 148% is computed across all task configurations (not just the Story task shown in Table 2), this should be clearly specified. The current phrasing is misleading.

- **N=5 samples per prompt is statistically thin for volatility estimation.** The benchmark's core contribution is measuring volatility across multiple generations, but with N=5 (Section 3.2), the standard error of the estimated standard deviation is approximately 0.35σ, giving confidence intervals spanning roughly ±70% of reported values. While cost constraints for long-form generation make larger N impractical, the paper should acknowledge this limitation explicitly.

- **Quality evaluation under extreme length increase deserves more rigor.** SELB increases output from ~445 to 15,651 words (~35x). While SCA 100% on execution-verified code tasks is meaningful, UCA relies on LLM-as-a-Judge (known bias toward longer outputs), and the content quality analysis beyond n-gram metrics is deferred to appendices. With such dramatic length changes, demonstrating that the additional content is coherent and non-repetitive in the main paper would strengthen the quality claims.

### Trivial
None.

## Nice-to-Haves

- An ablation testing whether a version of SELB designed *without* attention insights performs differently would validate or falsify the claimed analysis-method connection, which would be far more impactful than simply demonstrating SELB's effectiveness.

- Directly testing whether attention-stabilizing interventions (e.g., attention sink mechanisms, positional encoding adjustments) reduce volatility under standard decoding would validate the causal claim linking attention dynamics to output stability.

- A comparison with a simple "Length Constraint + Structure Prompting" baseline (combining the existing Length Constraint baseline from Table 2 with explicit section-structuring prompts) would help isolate SELB's contribution beyond what simpler constraints achieve.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"SCA achieving a perfect 100% is suspicious"** (Harsh Critic #4): SCA is defined as execution-based verification (Number of Correct Chapters / Number of Required Chapters, Section 3.2). Achieving 100% means every generated code chapter executes correctly, which is meaningful and not suspicious — it reflects that SELB's forced structural transitions produce structurally valid code, which is precisely what the metric measures. The critic's claim that "forced section transitions producing compilable code templates does not guarantee semantic correctness" conflates compilability with execution correctness; SCA measures the latter.

- **"Claude-3.5-Sonnet exclusion is not pre-registered and conveniently removes a strong model"** (Harsh Critic Section-by-Section Notes): The paper excludes Claude-3.5-Sonnet because it generates only 176 words (Table 2), which is insufficient for evaluating long-text generation quality. This is a reasonable exclusion criterion, not a convenient removal — a model that generates 176 words when asked for 100 sections cannot meaningfully participate in a long-form generation evaluation.

- **"The 'attention summits' interpretation is subjective"** (Harsh Critic Section-by-Section Notes): While the visual identification of attention summits and collapses could benefit from more quantitative definition, the paper provides a formal definition of the constraint attention measure α^(t) (Section 5) and the qualitative patterns are consistent across examples. This is a presentation limitation, not a scientific error.

- **"No human evaluation is reported"** (Harsh Critor #4): Human evaluation is not standard for LLM generation papers at this scale, and the paper uses execution-based verification (SCA) for structured tasks, which is more objective than human judgment. This is a nice-to-have, not a weakness.

- **"Missing comparison with recent work"** (Harsh Critic — related work): Removed per the rule against flagging missing related works.

- **Strength Finder's claim that "SELB demonstrates that the identified attention failure patterns can be effectively mitigated through lightweight decoding-time intervention"**: This strength conflicts with the verified Major weakness that SELB does not directly intervene on attention patterns. Removed from strengths.

- **Strength Finder's claim about "Generalization to free-form generation"** (SELB-Hybrid, Section 6.4): While mentioned in the main text, the empirical details for SELB-Hybrid are deferred to Appendix I. The claim of MLA 97% and LVC 12.1% on 20,000-word novel writing lacks accessible evidence in the main paper. Moved to Nice-to-Have.

## Novel Insights

The paper's most insightful finding is not the method but the benchmark result: the observation that LongWriter-8B — a model specifically fine-tuned for long-form generation — has the *highest* length volatility (LVC 45.4%) among all evaluated models, while producing the longest mean output. This suggests a fundamental tension between training for length and training for stability, which has significant implications for future long-form generation research. The cross-dimensional finding that structured tasks consistently yield more stable outputs (Figure 3) also points to a practical design principle: structured prompting may be a low-cost way to improve generation reliability, independent of any decoding strategy.

## Suggestions

- Be transparent that SELB is a constrained decoding approach. Reframe the contribution as "we show that constrained decoding can effectively enforce long-form output stability" rather than implying that the attention analysis causally drives the mitigation design. This is more honest and still valuable.

- Replace or supplement the "improves the mean output length of the base model by 148%" with the explicit comparison: "produces 2.5× the output length of LongWriter-8B (15,651 vs. 6,320 words)." This eliminates the ambiguity about "base model."

- Add a simple ablation: compare SELB against a naive baseline that applies only Length Constraint + forced section-structuring prompts on Qwen2.5-7B. If SELB outperforms this, it demonstrates value beyond obvious constraints; if not, the contribution needs rethinking.

## Evaluation Axis Assessment

- **Originality**: The problem framing (output volatility as a distinct evaluation dimension) is genuinely novel. The benchmark design is original. The attention analysis is interesting but not highly original in methodology. SELB is a straightforward constrained decoding approach with limited technical novelty.
- **Importance of research question**: High — output volatility in long-form generation is a real and underappreciated problem with practical implications.
- **Claims well supported**: Partially — the benchmark claims are well supported, but the claim that SELB "mitigates" the identified attention patterns is overstated, and the "148%" claim is ambiguously phrased.
- **Soundness of experiments**: Reasonable for the benchmark evaluation, but the method evaluation lacks critical ablations and the N=5 sample size is thin for volatility estimation.
- **Clarity of writing**: Generally clear, though the three-stage narrative creates a misleading impression of deep connection between the stages.
- **Value to research community**: The benchmark is the primary value — it provides a systematic tool for evaluating a previously overlooked dimension of LLM generation.

## Calibration

Papers compared against:
- **VKGTGGcwl6** (avg 8.0, Oral): Novel problem framing (multi-turn degradation) with rigorous large-scale analysis. The paper under review has a similarly novel problem framing but weaker analysis-method connection and more overclaiming.
- **sSfep4udCb** (avg 7.0, Oral): Clean formalization of long-form generation + tool-use with theory-practice alignment. The paper under review lacks this theoretical grounding and the analysis-method causal chain is weaker.
- **7Sph4KyeYO** (avg 5.5, Poster): Constrained decoding for diffusion LLMs — novel technical problem, some theoretical concerns. The paper under review has a stronger benchmark contribution but a less technically novel method.
- **qlf6y1A4Zu** (avg 5.0, Poster): Attention head identification + suppression with more principled causal connection between analysis and intervention. The paper under review has broader benchmarking but less rigorous analysis-method linkage.
- **yz4NpKm7Yx** (avg 4.0, Reject): Attention Smoothing — attention analysis + decoding-time mitigation with unprincipled approximations. The paper under review is clearly stronger with a solid benchmark and more effective method.
- **XbVMiW0jTM** (avg 2.5, Reject): PROBE benchmark with overclaimed paradigm overfitting conclusions. The paper under review has better empirical support but shares some overclaiming tendencies.
- **lJxRCW0UOI** (avg 1.33, Reject): LLM sentiment analysis with unjustified claims. The paper under review is far stronger — it has genuine benchmark contributions and real experimental infrastructure.

The paper sits above the medium-rejection anchors (yz4NpKm7Yx at 4.0, XbVMiW0jTM at 2.5) due to its genuine benchmark contribution and novel problem framing. It sits below the high-accept anchors (VKGTGGcwl6 at 8.0, sSfep4udCb at 7.0) due to the analysis-method disconnect and constrained decoding overclaiming. It is comparable to the 5.0–5.5 anchors, with a stronger benchmark than qlf6y1A4Zu (5.0) but a weaker causal chain.

**Score: 5.5** — The benchmark contribution and problem framing are genuine and important, but the method is essentially constrained decoding that is oversold as "mitigating" attention patterns, and the three-stage narrative creates a misleading impression of scientific connection between the probing insights and the mitigation design.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>