Now let me search for calibration papers.Now I have enough information to write the final consolidated review.

---

## Summary

Zer0-Jack proposes using zeroth-order gradient estimation combined with a patch coordinate descent strategy to generate adversarial image inputs for jailbreaking MLLMs, targeting the scenario where the attacker has query access to the model's output logits/probabilities but not its parameters. The method achieves 90–98% ASR on three open-source MLLMs, substantially outperforms all transfer-based baselines, and significantly reduces GPU memory usage (enabling attacks on 70B models on a single A100). A GPT-4o demonstration achieves 69% ASR.

---

## Strengths

- **Memory efficiency with concrete evidence** (Table 1): Zer0-Jack reduces GPU memory from 31G → 22G for 13B MiniGPT-4 and eliminates OOM for 70B models (white-box OOM, Zer0-Jack: 63G). This is a genuine and practically useful contribution.
- **Substantial empirical improvement over transfer attacks** (Tables 2 & 3): On nearly every model-dataset combination, Zer0-Jack surpasses all transfer-based baselines by very large margins—e.g., 95% vs. 16% (AutoDAN) on MiniGPT-4 (Harmful Behaviors), 95.8% vs. 41.7% on LLaVA1.5 (MM-SafetyBench-T).
- **Patch coordinate descent validated by ablation** (Figure 4): Without patch updating, Zer0-Jack achieves only ~30–45% ASR; with it, 95–98%. The patch-size sensitivity analysis further supports the design rationale.
- **Comparable ASR to white-box methods**: Zer0-Jack matches or nearly matches white-box performance on all three models without gradient access.
- **GPT-4o demonstration** (Table 5): Shows a non-trivial path to attacking commercial MLLMs, reaching 69% ASR vs. 18–30% for non-optimized baselines.

---

## Weaknesses

### Fatal
None.

### Major

- **Imprecise "direct black-box" framing throughout**: Zer0-Jack requires access to output logits or log-probabilities at every optimization step (Eq. 4, 6; Section 3.3). In the adversarial ML literature, this constitutes a *score-based* (grey-box) access model, not a "black-box" setting (which conventionally refers to hard-label-only access). The paper itself acknowledges this in Section 5: "there are some commercial MLLMs' API that do not support return logits (Anthropic, 2024)." The claim to be "the first method that aims at jailbreaking black-box MLLMs directly" is only valid under a non-standard interpretation of "black-box." This framing pervades the abstract, introduction, and contributions, and is consequential because it determines what the appropriate baseline class is.

- **Missing score-based adversarial attack baselines**: Given that the method is a score-based zeroth-order attack, the natural comparison class includes established score-based adversarial example methods (NES, SPSA, SimBA, Bandits, etc.) applied to the jailbreak objective. These are relegated to "Appendix B" and absent from the main experimental tables (Tables 2 and 3). Without these comparisons, it is unclear whether the performance gains stem from the patch coordinate descent innovation, the jailbreak-specific loss function, or simply from using any score-based method at all. This is the single most important missing comparison.

- **Query budget never reported**: Zeroth-order optimization is fundamentally query-expensive. Each gradient estimate (Eq. 6) requires two forward passes per patch, and the 7×7 patch grid means 49 patches per iteration. The number of queries needed to reach the reported ASR figures is entirely absent from Tables 2 and 3. The paper only mentions "$0.8 per sample" for GPT-4o, implying non-trivial cost. Without query counts, the claim of "reasonable queries" in the abstract is unverifiable, and fair comparison with zero-query transfer attacks is impossible.

### Minor

- **GPT-4o attack relies on OpenAI-specific logit-bias API feature**: The attack requires adding an artificially high logit bias to force "sure" to appear, then reading back its log-probability. This is not a general mechanism — it exploits a feature specific to OpenAI's API unavailable on other platforms. The paper does not analyze whether the loss landscape under forced token generation correlates with the landscape during free inference, which is the core assumption enabling the 69% ASR. This limits the generalizability of this result.

- **The strongest white-box baseline (WB + patch) outperforms Zer0-Jack**: Figure 4 shows WB + patch achieves ~100% (MM-SafetyBench-T) and ~98% (Harmful Behaviors), while Zer0-Jack achieves 98.2% and 95%, respectively. The paper acknowledges this in Section 4.4 but the framing in the main results ("comparable with white-box approaches") uses the weaker WB baseline (without patch). This is honest but the nuance should be stated more clearly in main results.

- **Evaluation metric lacks calibration details**: The paper uses GPT-4 as a judge. The specific GPT-4 version, the exact judge prompt, and any inter-rater reliability checks are not reported in the main text. This matters particularly for the headline 95–98% ASR figures.

- **All tested open-source models have weak safety alignment**: MiniGPT-4, LLaVA1.5, and INF-MLLM1 are acknowledged to be among the more easily attacked models. The high ASR values may partially reflect this, and it is unclear whether results would hold on models with stronger RLHF-based alignment.

### Trivial
None.

---

## Nice-to-Haves

- Reporting query-count vs. ASR curves alongside main results would greatly strengthen the practical evaluation.
- A failure-mode analysis showing which categories of harmful queries resist the attack, and whether failure patterns are consistent across models.
- Visualization of perturbed patches to show whether the attack exploits semantic or statistical artifacts in the vision encoder.
- A preliminary exploration of hard-label (decision-only) extension, which would genuinely justify the "black-box" claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Two-point estimator presented as novel"** (Harsh Critic): The paper cites Spall (1992) and explicitly states it is using the standard zeroth-order estimator. The novelty claim is for the patch coordinate descent + jailbreak loss combination, not the estimator itself. The critic's framing is a misread. Removed.

- **"Block-coordinate zeroth-order attacks are not compared"** (Harsh Critic): This is subsumed by the broader "missing score-based baselines" point already captured at Major level. Deduplication.

- **"Transferability results don't argue against transfer attacks"** (Harsh Critic): This is a misreading. The paper uses low transfer ASR (~51–54%) to motivate direct black-box attacks — a fair and reasonable argument. The paper never overclaims what this shows. Removed as a strawman.

- **"Patch updating improves white-box attacks = broader applicability claim"** (Strength Finder): True but generic; subsumed by ablation strength already captured. Not independently worth listing as a strength.

- **"Practical attack on commercial MLLMs is an innovative contribution"** (Strength Finder describing logit-bias as innovative): The logit-bias workaround is clever engineering, but its non-generalizability is a documented weakness. Dropped as a standalone strength since the corresponding weakness partially negates it.

---

## Novel Insights

The patch coordinate descent framing — decomposing the image into 32×32 patches and applying zeroth-order gradient estimation per-patch sequentially — is a sensible and practically validated solution to the high-dimensional estimation noise problem in zeroth-order optimization for jailbreaking. The ablation demonstrating that patch updating improves even white-box attacks (Fig. 4, left) is a genuinely interesting finding that extends the technique beyond the black-box setting. The observation that transfer attacks from MiniGPT-4 to LLaVA1.5/GPT-4o achieve only ~51–54% ASR (vs. 90–98% for direct optimization) provides concrete quantitative evidence for the value of direct score-based attack methods over transfer methods, corroborating theoretical intuition with empirical measurement.

---

## Suggestions

1. Rename the threat model explicitly as "score-based" or "logit-access black-box" throughout, distinguishing from hard-label black-box access. This would align framing with conventions in the adversarial ML literature and make the contribution more precisely scoped.
2. Add NES/SPSA/SimBA-based baselines to the main tables, even as simple adaptations of these methods to the jailbreak loss, to anchor the contribution of patch coordinate descent specifically.
3. Report queries-per-sample (or queries-to-90%-ASR) in Tables 2 and 3 for Zer0-Jack and any applicable baselines.
4. Expand the GPT-4o section to explicitly address the correlation between logit-bias-forced generation and free generation, either empirically (measure correlation) or theoretically.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison to paper under review |
|---|---|---|
| `/human_reviews/wNg0LibmQt.md` | 5.0 (Reject) | Most similar topic (gradient jailbreak for multimodal models); also criticized for limited baselines and narrow scope, but uses white-box access. Zer0-Jack is more technically novel but shares the baseline gap weakness. |
| `/human_reviews/wvFnqVVUhN.md` | 6.25 (Accept) | Rigorous empirical study on 40+ VLMs, accepted; much larger scope than Zer0-Jack, with comprehensive controls. Zer0-Jack's evaluation is narrower and misses its natural baselines. |
| `/human_reviews/htX7AoHyln.md` | 6.5 (Accept) | Score-based black-box attack paper with proper query budget analysis and comparison to related score-based methods — exactly what Zer0-Jack lacks. |
| `/human_reviews/9kR4MREN9E.md` | 3.5 (Reject) | Transfer attack on fine-tuned LLMs; more limited contribution with missing baselines, rejected. Zer0-Jack is more original but shares the incomplete evaluation problem. |
| `/human_reviews/rgiIZ3pcZY.md` | 4.75 (Reject) | Jailbreaking MLLMs via out-of-distribution strategies; weak evaluation and baselines, rejected. Zer0-Jack is stronger but comparably limited in comparison class. |
| `/human_reviews/djcciHhCrt.md` | 4.25 (Reject) | Misusing tools in LLMs via visual adversarial examples; gradient-based attack that lacks comprehensive evaluation. |

Zer0-Jack sits between the rejected papers (avg 4–5) and the accepted ones (avg 6–6.5). Its core technical contribution (patch coordinate descent for score-based MLLM jailbreaking + memory efficiency) is real and empirically validated, and the empirical results are striking. However, the missing score-based baselines (which is the primary comparison class for a score-based attack paper) and the absent query budget analysis are the same class of weakness that causes borderline papers to be rejected. The imprecise threat model framing also weakens the contribution's positioning. Compared to htX7AoHyln (avg 6.5, proper score-based paper with query analysis) and wvFnqVVUhN (avg 6.25, rigorous evaluation), this paper falls short of the bar those set. I place it at **4.5**.

**Originality**: Moderate — applying zeroth-order optimization to MLLM jailbreaking with patch decomposition is novel in application, though both the estimator and coordinate descent idea are known tools.  
**Importance**: Real problem; MLLM jailbreaking in non-white-box settings is relevant.  
**Claims vs. support**: Claims are mostly supported within a score-based access regime, but the "black-box" framing overstates generality.  
**Experimental soundness**: Solid internal experiments; critically missing the natural comparison class (score-based adversarial attacks) and query budget reporting.  
**Clarity**: Generally clear; Figure 2 and Algorithm overview are helpful.  
**Value to community**: Moderate — the patch coordinate descent idea and memory efficiency results are useful contributions, but the incomplete evaluation limits confidence in the claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>