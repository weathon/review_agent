Now I have all the information I need. Let me compile the final review.

## Summary

This paper identifies and systematically characterizes the "priming vulnerability" in Masked Diffusion Language Models (MDLMs): when affirmative tokens for a harmful query appear at intermediate denoising steps, subsequent generation is steered toward harmful responses even in aligned models. The authors demonstrate this under two threat models—a hypothetical attacker who can intervene in the denoising process (anchoring attack), and a more realistic attacker who cannot intervene but uses optimization-based attacks (First-Step GCG, enabled by Theorem 4.1's tractable lower bound). To address the vulnerability, they propose Recovery Alignment (RA), which trains models to generate safe responses from intentionally contaminated intermediate states using an RLHF-style objective with a curriculum schedule.

## Strengths

- **Clean identification of a genuine, MDLM-specific vulnerability**: The observation that standard alignment trains only from fully masked sequences, leaving models undefended against contaminated intermediate states, is precise, well-motivated, and non-obvious. The ablation "RA w/o inter" in Table 2 compellingly confirms this—without contaminated-state training, ASR at t_inter=4 is 22.0% vs. 1.3% for full RA on LLaDA Instruct.

- **First-Step GCG is a practical algorithmic contribution**: Table 1 shows First-Step GCG achieves 58% ASR on LLaDA Instruct (vs. 20% for MC-GCG) while being ~20× faster (0.2h vs. 4.3h per prompt). This is useful for red-teaming MDLMs and also serves as a strong realistic attack for evaluation.

- **Theoretical contribution enabling tractable optimization**: Theorem 4.1 derives a tractable lower bound on the full denoising log-likelihood, bypassing the intractable gradient caused by stochastic re-masking. This provides principled grounding for First-Step GCG beyond just an engineering trick.

- **Comprehensive evaluation**: The paper evaluates across 3 MDLMs, 7+ attack types (intervention-based: anchoring, PAD, DiJA; optimization-based: GCG; conversational: PAIR, ReNeLLM, Crescendo), and 11 capability benchmarks. This gives a broad picture of both the vulnerability landscape and the defense's effectiveness.

- **Strong capability preservation**: Table 4 shows negligible degradation across 11 benchmarks (52.6 vs. 52.2 average for LLaDA, 52.8 vs. 52.7 for LLaDA 1.5), directly addressing the most common concern about safety training.

- **Principled curriculum ablation**: Figure 3b validates that linear scheduling outperforms both constant and uniform scheduling, providing practical guidance for practitioners.

## Weaknesses

### Fatal
None.

### Major

- **The monotonicity assumption underpinning Theorem 4.1 is under-validated in the main text**: The theorem requires log π_θ(r̃_{t+1} = r | q, r_t) ≥ log π_θ(r̃_1 = r | q, r_0) for all t, but validation is deferred to Appendix C.2 with only a brief paragraph of informal justification in the main text (Section 4.2). This assumption is not obviously true: early in denoising when r_t contains few unmasked tokens, the model's distribution may become more concentrated on completions diverging from the specific target r. Since Theorem 4.1 is the sole theoretical support for First-Step GCG (the paper's primary realistic attack), the assumption deserves direct, prominent validation—especially because the 1/T prefactor in the bound (Equation 3) already signals potential looseness. The paper does state that the assumption holds "across a broad range of models" in Appendix C.2, but this should be in the main text.

- **"Superior robustness" claim against conventional attacks is partially contradicted by data**: Section 6.2 states "RA achieves superior robustness against such attacks and outperforms baselines." While RA does outperform baselines in 8 of 9 conventional attack scenarios across 3 models, the MMaDA ReNeLLM case shows RA (81.7%) performing worse than MOSA (75.7%) and only marginally better than the original (79.3%). Additionally, ReNeLLM ASR remains above 70% across all models even with RA (72.3% on LLaDA, 71.7% on LLaDA 1.5, 81.7% on MMaDA). The paper acknowledges RA "remains imperfect against strong attacks, such as ReNeLLM" but does not explicitly note the one case where RA underperforms a baseline. The claim should be qualified to reflect that RA is the best overall method but not uniformly superior, particularly for strongly-aligned attacks on weakly-aligned models.

### Minor

- **RA's effectiveness degrades at high intervention steps**: At t_inter=32 (25% of tokens unmasked), RA still yields 50.7% ASR on LLaDA Instruct and 43.0% on LLaDA 1.5 (Table 2). The paper acknowledges this honestly ("many anchors make recovery impossible"), but does not quantify what fraction of the vulnerability space RA actually covers or discuss at what intervention depth the defense becomes impractical.

- **The t_min and t_max hyperparameters for the linear schedule are not specified in the main text**: The algorithm defines the schedule over [t_min, t_max] but the specific values used in experiments are deferred to the appendix. These are critical hyperparameters that directly control training difficulty and defense strength.

- **RA struggles on weakly-aligned models**: MMaDA results are significantly worse across all metrics (e.g., First-Step GCG ASR remains 45.7% after RA, No Attack ASR is 3.3% vs. 0.0% on LLaDA). The paper attributes this to MMaDA's weak baseline alignment but does not analyze whether RA fundamentally requires a certain level of baseline alignment to be effective.

- **The training procedure uses oracle knowledge of harmful responses**: Equation 7 constructs contaminated intermediate states from the known harmful response r. In deployment, an attacker would not have access to the exact harmful response. The evaluation uses attacks that don't require this oracle (GCG, PAIR, etc.), so the train-test mismatch is partially addressed, but the paper should discuss whether this creates a systematic bias in RA's measured effectiveness.

### Trivial
None.

## Nice-to-Haves

- Comparison of First-Step GCG ASR against GCG on a comparably-sized ARM would help calibrate whether the 58% ASR is an MDLM-specific weakness or a standard jailbreak result, directly supporting the paper's claim that "DLM-specific safety research" is needed.

- Characterizing which single tokens at t_inter=1 drive the 2%→21% ASR increase (affirmative vs. content tokens) would reveal the mechanism behind the paper's most surprising finding.

- A qualitative example showing a denoising trajectory before and after RA would make the priming mechanism and recovery behavior more tangible.

- Testing RA with a DPO-style training instantiation, as mentioned in the limitations, would strengthen generalizability claims.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh critic's claim that "MMaDA under PAIR: original ASR is 79.3% and RA achieves 81.7%—slightly worse"**: This is factually incorrect. The reviewer confused the PAIR and ReNeLLM columns. MMaDA PAIR goes from 98.0% → 46.3% with RA (major improvement). The slight worsening is on ReNeLLM (79.3% → 81.7%), not PAIR.

- **Criticism that the anchoring attack threat model is "too strong to justify urgency claims"**: The paper explicitly labels the anchoring attacker as "hypothetical" (Section 4.1, line 60) and provides the more realistic First-Step GCG as a separate threat model. The paper's structure clearly separates these, and the urgency claim is supported by the realistic First-Step GCG results (58% ASR) as well.

- **Criticism that the abstract's "simply injecting" obscures the intervention requirement**: The abstract discusses the vulnerability broadly and also mentions optimization-based attacks that don't require intervention. The "simply injecting" phrasing refers to the conceptual simplicity of the vulnerability mechanism, not the ease of execution.

- **Request to compare against ARM prefilling attacks more deeply**: The paper already draws the analogy with ARM prefilling (Section 1, citing Wei et al., 2023a) and identifies the structural difference (parallel iterative vs. causal sequential). Demanding deeper comparison with ARMs is scope creep for a paper focused on MDLM-specific safety.

- **Criticism about missing denoising trajectory visualizations**: This is a nice-to-have, not a weakness.

- **Nitpick about unspecified t_min/t_max**: Promoted to minor weakness above as it is a genuinely important hyperparameter, but it's not a major issue.

- **Request for DPO-style RA experiments**: This is a nice-to-have; the paper already mentions it as future work and the GRPO instantiation is sufficient.

- **Formatting/notation nitpicks**: Removed per rules.

## Novel Insights

The paper reveals a fundamental insight about MDLM safety alignment: standard RLHF/DPO-style training implicitly conditions on a single initial state (fully masked), creating a distributional gap at test time when the model encounters contaminated intermediate states. This is conceptually distinct from ARM prefilling vulnerabilities—where the issue is causal conditioning on harmful prefixes—because MDLMs face the problem at *every* denoising step and in *parallel* across all token positions. The ablation (RA w/o inter) cleanly isolates this mechanism: training on contaminated states is the critical ingredient, not just more safety training. This suggests a general principle for non-autoregressive generative models: safety alignment must cover the space of intermediate states the model might encounter, not just the canonical starting condition.

## Suggestions

- Move the monotonicity assumption validation (Appendix C.2) to the main text or provide a prominent figure/table showing the assumption holds for the actual attack targets used, to solidify the theoretical grounding.

- Qualify the "superior robustness" claim against conventional attacks to acknowledge the MMaDA ReNeLLM exception, e.g., "RA achieves the best or near-best robustness in most conventional attack settings, though it remains imperfect against particularly strong attacks on weakly-aligned models."

- Report the specific t_min and t_max values used in experiments in the main text alongside Algorithm 1, since these are critical hyperparameters.

- Add a brief analysis of what intervention depths RA can reliably defend against (e.g., "RA effectively mitigates attacks up to t_inter=X") to give practitioners concrete guidance on the method's operational limits.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| DiffuGuard | /home/wg25r/review_agent/human_reviews_2026/zBPzxhso8M.md | 5.20 | Very similar topic (dLLM safety, denoising-path dependence). This paper is stronger: adds theoretical contribution (Theorem 4.1), more comprehensive evaluation (3 models, 7+ attacks, 11 capability benchmarks), and a training-based defense with strong ablation rather than a training-free approach. |
| A2D | /home/wg25r/review_agent/human_reviews_2026/URTnuyQJI1.md | 5.50 | Very similar (dLLM safety alignment via contaminated states). This paper has broader scope (intervention + non-intervention attacks, theoretical grounding) but A2D achieves near-zero ASR on some settings. Roughly comparable quality. |
| DIJA | /home/wg25r/review_agent/human_reviews_2026/rIPeatvPy3.md | 5.00 | dLLM attack paper, no defense proposed. This paper goes further by proposing and validating a defense. |
| AlphaSteer | /home/wg25r/review_agent/human_reviews_2026/1vvbzAqdTe.md | 7.00 | Novel defense with principled approach, similar overclaiming issue. This paper is narrower in scope (MDLM-specific) but more comprehensive in evaluation. |
| Persona Features | /home/wg25r/review_agent/human_reviews_2026/yjrVOxjkDR.md | 7.50 | High-impact mechanistic finding with broad implications. This paper is more domain-specific but still makes a clean, important observation. |
| Emoji Game | /home/wg25r/review_agent/human_reviews_2026/e8b3fRYeDt.md | 0.67 | Low anchor: fundamentally flawed methodology. This paper is far above this bar. |
| SEAM | /home/wg25r/review_agent/human_reviews_2026/ERNpUGr8M5.md | 6.50 | Defense with contaminated data optimization. Similar defense philosophy, higher score due to broader applicability. |

This paper is solidly above DiffuGuard (5.20) and A2D (5.50) due to its theoretical contribution (Theorem 4.1), more comprehensive evaluation, and the clean ablation isolating the contaminated-state training mechanism. It is below AlphaSteer (7.00) and SEAM (6.50) because of the narrower scope (MDLM-specific only), the monotonicity assumption validation gap, and the partially overstated conventional attack robustness. The core contribution—identifying the training-from-clean-states gap and proposing contaminated-state training—is sound, well-motivated, and well-supported by evidence. The weaknesses are real but do not undermine the central claims.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>