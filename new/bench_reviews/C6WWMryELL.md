Now I have enough context to synthesize my final review. Let me organize my thoughts:

1. **The paper's core contributions**: (a) VOLTBench - a benchmark for measuring volatility in long-form generation, (b) attention trace analysis identifying "Attention Collapse" and "Attention Instability," (c) SELB - a decoding-stage method for mitigating volatility.

2. **Key issues to evaluate**:
   - Volatility definition confounded with sampling hyperparameters (harsh critic point 1)
   - SELB comparisons not fair/underspecified (harsh critic point 2)
   - SELB relies on pre-specified section structure (harsh critic point 3)
   - Attention trace analysis is correlational not causal (harsh critic point 4)
   - LLM-as-a-judge quality concerns (harsh critic point 5 / human finder point 4)
   - N=5 sample size for volatility estimation (human finder point 5)
   - Quality-length tradeoff underexplored (neutral reviewer / human finder)
   - Missing ablations (spark reviewer)

Let me verify some specific claims against the paper:

For the harsh critic's point 1 about decoding configuration: The paper says in Section 4.1 it describes the models but does NOT specify temperature/top-p settings. This is a valid concern - the paper doesn't detail decoding parameters.

For point 2 about what base model SELB is applied to: Looking at the paper, Section 4.1 mentions "common training-free decoding strategies for comparison, implemented on Qwen2.5-7B-Instruction." And Section 6.3 says "our model" but doesn't explicitly state in the main results section what the base model is for SELB. This is a real gap.

For point 3 about section structure: The paper acknowledges this in Section 6.4 and addresses it with SELB-Hybrid, but the main results are on structured tasks. Valid concern.

For point 4 about correlational analysis: Valid - the attention patterns are identified qualitatively and SELB doesn't directly modify attention.

For point 5 about LLM-as-judge: Valid but somewhat standard in the field for unstructured tasks.

Now let me calibrate my score. Looking at similar papers:
- LongGenBench: 8, 3, 5, 8, 8 → accepted as poster (mixed but overall positive)
- LongWriter: 6, 6, 6, 6 → accepted as poster
- HelloBench: 3, 5, 6, 5 → withdrawn/rejected (limited benchmark contribution)
- Attention Satisfies: 6, 6, 6 → accepted as poster (attention analysis, limited practical impact)

This paper has more substance than HelloBench (which was rejected) - it introduces a benchmark, does analysis, AND proposes a method. But it has significant methodological issues with the SELB evaluation (unfair comparisons, reliance on structural priors, no ablations). The benchmarking contribution is solid but the method claims overreach the evidence.

Given the calibration:
- It's clearly above HelloBench (rejected, score ~4.5 avg) which had very limited contribution
- It's below LongWriter (accepted poster, score 6) which had cleaner if limited contribution
- It's below the Attention Satisfies paper (6) which had similar correlational analysis but was cleaner

I'd rate this around a 4.5-5: the benchmarking contribution is real and useful, but the method claims are overstated and the evaluation has significant gaps. The paper reads more like a benchmark + analysis paper with a method appendix, yet the method is presented as a primary contribution.

Actually, let me reconsider. The paper does make three clear contributions: (1) benchmark with volatility metrics, (2) attention analysis, (3) SELB method. The benchmark contribution alone is reasonable - similar to LongGenBench which got accepted. But the method has serious issues: it's essentially hard-coded structural enforcement masquerading as "logits boosting," and the comparisons don't clearly isolate the contribution. The attention analysis is correlational.

I think a fair score is around 4.5-5: the benchmarking and analysis are reasonable contributions, but the method is significantly oversold relative to its actual mechanism and experimental validation.</think>Now I have all the information I need to write the final meta-review. Let me synthesize the evidence carefully.

## Summary

This paper investigates the understudied problem of output volatility (length and content inconsistency across multiple generations) in long-form LLM generation. The authors construct VOLTBench, a heterogeneous-task benchmark covering structured and unstructured tasks across languages and instruction complexities, with multi-sample evaluation to quantify length volatility. Through attention trace analysis, they identify "Attention Collapse" and "Attention Instability" as patterns correlated with generation failures. Based on these observations, they propose SELB, a decoding-stage method that structurally enforces section generation and suppresses failure-mode tokens, reporting a 148% increase in mean output length and 69% reduction in length volatility.

## Strengths

1. **Novel and practical problem formulation.** Systematically studying multi-run volatility in long-form generation—rather than just single-generation quality—is a genuine and timely contribution. The observation that even models fine-tuned for long output (e.g., LongWriter-8B) exhibit extreme variance (standard deviation of 103% of mean length) across runs is striking and practically relevant for deployment cost estimation.

2. **Comprehensive and well-designed benchmark.** VOLTBench covers both structured (code, LaTeX) and unstructured (story, diary) tasks, bilingual settings (English/Chinese), multiple difficulty levels, and fine-grained per-section constraint evaluation. The chapter-based scaling to 500 sections (~100k words) pushes evaluation boundaries. Table 1 clearly shows how VOLTBench fills gaps left by prior benchmarks that lack multi-sampling and volatility metrics.

3. **Attention trace analysis provides useful descriptive insights.** Identifying "Attention Collapse" (attention dropping to near-zero before premature termination) and "Attention Instability" (attention spikes preceding section skipping) is a valuable observation that moves beyond purely behavioral characterization toward mechanistic understanding, even if the causal link remains incomplete.

4. **Fine-grained constraint tasks are well designed.** The character-level, keyword, and thematic constraints embedded in specific sections offer a rigorous way to probe localized instruction-following in very long contexts—a genuine addition over prior work.

## Weaknesses

### Major:

1. **SELB's comparison setup is underspecified and potentially unfair, undermining the headline claims.** The paper does not clearly state which base model SELB is applied to for the main 148%/69% claims. Section 4.1 mentions Qwen2.5-7B for training-free baselines, but Section 6.3 compares "our model" primarily against LongWriter-8B (a different architecture trained for long output). Without a clean ablation (same model, same prompts, same decoding parameters, with and without SELB), the headline percentage improvements conflate the effect of the decoding intervention with model architecture and training differences. Additionally, decoding parameters (temperature, top-p) are not specified, making it impossible to assess whether volatility differences stem from SELB or from sampling configuration. This is critical because the paper frames volatility as a fundamental model limitation, yet standard production practice (deterministic or low-temperature decoding) trivially reduces variance.

2. **SELB is tightly coupled to the benchmark's structured format and does not generalize as claimed.** The method's core mechanism (Eq. 2) requires advance knowledge of section title tokens V^{(p+1)}_{title}, making it only applicable when the output structure is fully specified a priori. This is essentially hard-coded structural enforcement—a template grammar layered over the model—rather than a general decoding strategy. The SELB-Hybrid generalization to free-form tasks (Section 6.4) is entirely relegated to Appendix I with no main-paper evaluation, making the claim that SELB "can be effectively generalized beyond structured tasks" unsupported by the presented evidence. The paper acknowledges this in passing but the abstract and conclusion overstate generality.

3. **No ablation study isolating SELB's components.** SELB combines two interventions: M_struct (section title boosting) and M_fail (EOS/filler token suppression). There is no experiment showing what happens with just M_struct, just M_fail, or how much of the 148% length improvement and 69% volatility reduction comes from each component. Without this, it is impossible to determine whether the method's gains come from the "attention-informed" intervention (as motivated by the probing stage) or from brute-force suppression of EOS tokens and section-title injection—mechanisms that require no internal analysis to design.

4. **The attention trace analysis is descriptive and correlational, not causally linked to the mitigation.** The paper identifies Attention Collapse and Instability patterns qualitatively through illustrative traces but provides no quantitative statistics (e.g., how often these patterns predict failure, across how many prompts/models). More critically, SELB does not directly modify attention—it modifies output logits. The narrative "attention patterns cause failures → SELB fixes them" is asserted but not demonstrated; showing that SELB changes attention traces (e.g., preventing collapse) would close this loop, but no such experiment is provided.

### Minor:

1. **N=5 generations per prompt is a small sample for estimating variance metrics (LSD, LVC).** With only 5 runs per prompt, the standard deviation of the estimated standard deviation itself is large, making volatility comparisons between models noisy. No confidence intervals or bootstrap analyses are reported.

2. **Quality evaluation relies on LLM-as-a-Judge without calibration.** UCA (unstructured content accuracy) uses an external LLM judge, but the paper does not specify the judge model, rubric, or calibration against human judgment in the main text. Given that SELB forces longer outputs, there is a well-known length bias in LLM judging, which is not controlled for. The paper reports SCA=100% for SELB on code tasks, but this may reflect structural compliance rather than content quality.

3. **The benchmark's chapter-based template format may conflate instruction-tracking failure with genuine long-form generation difficulty.** As also observed by reviewers of similar benchmarks (e.g., LongGenBench), tasks requiring enumeration of many semi-independent chapters primarily test whether the model remembers the instruction—rather than whether it can sustain coherent, interconnected long-form content. The failure mode (losing track of instructions in long context) is related to but distinct from the claimed problem of long-form generation instability.

### Trivial:

- The main results (Table 2, Fig. 6) focus on the "100-section, English, simple difficulty" setting; results across the full difficulty and language spectrum are deferred to the appendix, making it hard to assess robustness.

## Nice-to-Haves

- Ablation experiments separating M_struct, M_fail, and the τ_max threshold, plus comparison against a trivial templating baseline that simply inserts section headers at fixed intervals without any attention analysis.
- Evaluation of SELB on free-form tasks in the main paper body, not just the appendix.
- Application of SELB to at least one additional model architecture (beyond Qwen2.5-7B) to demonstrate generality.
- Attention trace overlays (baseline vs. SELB) to demonstrate that SELB actually addresses the identified internal patterns rather than bypassing them.
- Statistical tests or confidence intervals on volatility metrics given the small N=5 sample size.

## Removed Points

- **"Decoding parameters not disclosed → volatility metric is mis-specified."** While it's true that decoding parameters aren't specified, the paper is benchmarking models under their default or comparable configurations, which is standard practice. The concern that users can "just lower temperature" is a practical mitigation, not a reason the volatility phenomenon isn't real—high-temperature/stochastic settings are the norm for creative generation tasks. The concern is valid that the paper should specify parameters and acknowledge this dependency, but it doesn't undermine the existence of the phenomenon. Demoted to a minor mention rather than a fatal flaw.

- **"MLA symmetrically penalizes over- and under-generation."** This is a minor design choice in the metric; in many generation scenarios, overshooting is as undesirable as undershooting (e.g., cost predictability), and the paper's context makes a symmetric metric reasonable. Not a substantive weakness.

- **"V_channel vs V_banned typo in Eq. 3."** This is an OCR/formatting artifact, not a methodological issue. Removed as a nitpick per instructions.

- **"No comparison with LongWriter-Zero, SURI, Temp-Lora."** These are training-based methods operating in a fundamentally different paradigm (model-level interventions). The paper explicitly positions SELB as a lightweight, training-free approach (Sec. 2), so demanding comparison with methods requiring retraining or RL is scope creep. However, acknowledging this distinction more prominently would help.

- **"VOLTBench tasks are heavily templated / not realistic."** This is inherent to any controllable, scalable benchmark for long-form generation. Evaluating open-ended creative writing at scale requires some structure for automated metrics. The paper does include unstructured tasks with LLM judging. This is a design trade-off, not a flaw.

- **"Reproducibility concerns about undisclosed hyperparameters."** The paper commits to releasing code and data on acceptance, and the SELB method is straightforward to describe. This is standard for the field and not a meaningful weakness.

## Novel Insights

The paper makes a genuine contribution by formalizing output volatility as a first-class metric for long-form generation evaluation, moving beyond single-generation quality assessment. The finding that even specialized models (LongWriter-8B) exhibit 103% coefficient of variation is striking and underscores that length adherence is not merely a mean performance issue but a reliability one. The attention trace analysis, while not causally validated, identifies interpretable failure signatures (collapse and instability patterns) that could guide future mechanistic interventions. However, the disconnect between the probing insights (attention-level failures) and the proposed solution (logit-level hard-coded enforcement) represents a missed opportunity—a method that actually repairs attention collapse rather than overriding it would be more scientifically compelling and practically general.

## Suggestions

1. **Add an ablation table** showing SELB with only M_struct, only M_fail, and both, plus a trivial "section header injection" baseline that doesn't use attention analysis, to isolate the real contribution.
2. **Specify decoding parameters** (temperature, top-p, top-k) for all models in the main experiments, and run a small sensitivity analysis showing how volatility changes with temperature (even for 1–2 models).
3. **Move key SELB-Hybrid free-form results into the main paper**, or scale back the generality claims in the abstract and conclusion to match what is actually demonstrated.
4. **Report results across the full VOLTBench spectrum** (complex and fine-grained settings, both languages) in the main paper, not just the simplest 100-section English configuration.

## Score and Decision

**Calibration:**
- LongGenBench (similar long-form generation benchmark, accepted poster): scores 8, 3, 5, 8, 8 — benchmark contribution with limited analysis depth
- LongWriter (long-form generation method, accepted poster): scores 6, 6, 6, 6 — data-centric approach, evaluation concerns but clearly novel
- HelloBench (long-form generation benchmark, withdrawn/rejected): scores 3, 5, 6, 5 — limited contribution, insufficient analysis
- Attention Satisfies (attention analysis for LLM failures, accepted poster): scores 6, 6, 6 — correlational but informative, limited practical impact

This paper is above HelloBench (which was rejected for limited contribution and shallow analysis) because it provides a benchmark, analysis, AND a method. It is below LongWriter (score ~6) because LongWriter's method was cleanly evaluated and clearly novel, while SELB's evaluation has significant fairness and generality gaps. It is below the Attention Satisfies paper (score ~6) because although this paper also presents correlational attention analysis, it makes much stronger causal and generalizability claims that the evidence doesn't support.

The benchmark contribution is solid and the volatility framing is important, but the method is oversold relative to its actual mechanism (hard-coded structural enforcement that works only on templated tasks), and the experimental validation for SELB lacks critical controls. The paper reads as a benchmark paper (which could be strong) trying to also be a methods paper (where it is weak).

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>