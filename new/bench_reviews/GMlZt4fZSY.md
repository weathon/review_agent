Now let me run the calibration searches in parallel.Now I have enough information to write the final consolidated review. Let me compile it.

---

## Summary

MobileLLM-R1 is a series of sub-billion parameter reasoning language models (140M, 360M, 950M) trained using a three-stage pipeline (pre-training → mid-training → post-training) with a data-centric framework. The core methodological contributions are: (1) a leave-one-out (LOO) analysis to identify high-value pre-training datasets via NLL on capability-probing sets, (2) a cross-capability influence-based data mixture weighting scheme, and (3) a data–model co-evolution strategy for mid-training compression. The 950M model claims to match or surpass Qwen3-0.6B on multiple benchmarks while using only 11.7% of Qwen3's 36T training tokens, and achieves an AIME score of 15.5 versus 0.6 for OLMo-2-1.48B.

---

## Strengths

- **State-of-the-art results among fully open-source sub-1B models (Table 2, Figure 8, Figure 9):** MobileLLM-R1-950M achieves MATH 57.8, GSM8K 68.5, AIME 15.5 — substantially ahead of OLMo-2-1.48B (53.0 / 58.8 / 0.6) and SmolLM2-1.7B (41.4 / 50.5 / 0.3) under identical reasoning SFT. These margins are large and represent a genuine advance for the fully open-source small model community.

- **Full reproducibility commitment:** Models at all three scales, training code, data sources, and mixing ratios are publicly released. This is meaningful for the community building on sub-1B models.

- **Controlled post-training ablation (Table 1, Table 2):** The staged alignment-then-reasoning SFT approach is demonstrated to outperform joint training (e.g., 57.8 vs. 56.2 MATH on 950M), and the contribution of pre/mid-training quality is isolated by holding the reasoning SFT corpus constant across all baselines.

- **Principled LOO dataset analysis with an unexpected cross-domain finding (Figure 3, Section 2.1.2):** The leave-one-out methodology is a rigorous diagnostic for dataset importance. The finding that StarCoder benefits math more than OpenWebMath benefits code is a genuinely interesting reversal of conventional wisdom and worth further investigation.

- **Influence-based pre-training mixture shows NLL improvement over uniform sampling (Figure 4):** The influence-derived mixture consistently outperforms uniform sampling on held-out Code, Math, and Knowledge probing sets, providing evidence that the method captures meaningful cross-domain signals.

---

## Weaknesses

### Fatal
None.

### Major

- **The central "11.7% token efficiency" claim is structurally misleading.** The paper's headline comparison pits MobileLLM-R1-950M (~950M parameters) against Qwen3-0.6B (~600M parameters) — roughly a 58% parameter advantage for MobileLLM-R1 — and uses the *full Qwen3 series training corpus* (36T tokens, spanning models from 0.6B to 235B) as the denominator. These two factors together undermine the framing: (a) a larger model is expected to need fewer tokens than a smaller model to reach a given performance level, so matching the smaller model is not surprising; (b) the compute-optimal token count for a 0.6B model under Chinchilla-style scaling would be far less than 36T, making the 11.7% ratio look more impressive than it is in practice. The paper never provides a controlled parameter-matched comparison (e.g., MobileLLM-R1 at 600M vs. Qwen3-0.6B), nor does it acknowledge this confound. This framing pervades the abstract, introduction, Figure 1, and conclusion. The actual results (best-in-class fully open-source sub-1B model) are genuine, but the headline claim overstates what has been demonstrated.

- **Influence-based data mixture method is never validated at the final benchmark level.** The LOO analysis (Section 2.1) and influence weighting (Section 2.2) are evaluated solely via NLL on capability-probing datasets. Figure 4 compares influence-weighted vs. uniform sampling on *probing NLL* — not on downstream task accuracy. No ablation in the paper compares influence-weighted vs. uniform-weighted vs. heuristic-weighted mixtures directly on MATH, GSM8K, AIME, or LCBv6 final benchmarks. The gap between "better probing NLL" and "better benchmark accuracy" is not bridged. As stated in Section 2.2, the probing sets are themselves constructed from the training corpus via high-scoring sub-samples, raising the question of whether probing NLL is a sufficiently independent proxy. Without this benchmark-level ablation, the influence method's contribution to the empirical results cannot be established; the gains in Table 2 may be attributable to dataset selection (LOO analysis) rather than the weighting methodology.

### Minor

- **Data repetition is not clearly distinguished from unique token efficiency.** The abstract states "~2T tokens of high-quality data are sufficient" but training uses "4.2T tokens resampled from these ~2T tokens," implying approximately 2× repetition. The comparison to Qwen3's 36T corpus (presumably largely unique tokens) is not like-for-like. No ablation tests whether a single pass over 2T tokens achieves equivalent results, or whether the specific repetition schedule is what matters. The paper should clearly distinguish unique tokens from training tokens in its efficiency narrative.

- **Baseline comparison asymmetry in Table 2 is acknowledged but not fully analyzed.** Baseline models use their full "instruct checkpoints" (which include proprietary post-training beyond SFT), while MobileLLM-R1 uses intermediate Tulu3-SFT checkpoints (marked with *). The paper notes this with the asterisk but does not discuss whether differences in prior post-training explain part of the gap or favor/disadvantage MobileLLM-R1.

- **The mid-training subsampling benefit is shown only on MMLU (Figure 6), not on the final benchmark suite.** The claim that data–model co-evolution improves training stability is supported only by MMLU training curves. It is possible that MMLU is an outlier or that the effect size is different on MATH, GSM8K, or LCBv6.

- **LOO conclusions are based purely on probing NLL, not benchmark accuracy.** The claim "FineWeb-Edu is the most important dataset" rests entirely on NLL differences. While NLL is a reasonable proxy, the relationship to final benchmark accuracy is assumed, not validated. The interesting finding about StarCoder benefiting math deserves corroboration via direct benchmark comparison.

### Trivial
None worth noting beyond parser artifacts.

---

## Nice-to-Haves

- A scatter plot of probing-NLL vs. downstream benchmark accuracy across the LOO runs would either validate or falsify the proxy assumption and substantially strengthen the methodological contribution.
- A parameter-matched comparison (MobileLLM-R1 trained at 600M) or explicit acknowledgment of the parameter asymmetry in the Qwen3 comparison would make the efficiency claims defensible.
- Since all compared baselines (Qwen3-0.6B, DeepSeek-R1-Distill variants) use RL-based post-training, an RL post-training experiment or explicit scoping to SFT-only settings would clarify the contribution's limits.
- A direct ablation of influence-weighted vs. uniform-weighted mixing on final benchmark scores would be the most important addition to validate the core methodology.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "The paradigm shift framing challenges assumptions already addressed by prior work"** — Removed as scope-creep criticism. The paper explicitly cites the works that challenge the first assumption and positions itself against the second. Noting that prior work exists is not a weakness.

- **Harsh Critic: Circularity in capability-probing dataset construction** — Removed as overstated. The paper uses high-quality filtering (FineWeb-Edu score ≥4, Ask-LLM top 10%) for *probing* sets, which is standard practice for constructing representative evaluation data. The "benchmark-free" claim is specifically about not using downstream *task* benchmarks during training, which is accurate.

- **Harsh Critic: Domain-specialized checkpoints (θC, θM, θK) as surrogates** — Kept in modified form as the minor point about the method's approximation quality. However, the suggestion that using domain-specialized models "introduces an approximation whose fidelity is never validated" is too strong — this is explicitly described as a scalability trade-off following AutoMixer's protocol.

- **Harsh Critic: HumanEval/GSM8K contamination concern regarding Figure 7 perplexity diagnostics** — Removed. Figure 7 tracks *perplexity* as a training diagnostic, not evaluation accuracy. Perplexity on HumanEval during training is a proxy for coding capability development, not a benchmark evaluation. Contamination concerns apply to final accuracy evaluation, not training monitoring.

- **Strength Finder: "Full reproducibility" as a generic strength** — Retained because the paper provides an unusually complete release (all three model scales, code, data sources, and mixing ratios), which is above average for this type of work.

- **Strength Finder: "Controlled ablation isolating pre/mid-training from post-training effects (Table 2)"** — Retained as a core strength, though slightly modified to acknowledge the instruct vs. SFT checkpoint asymmetry.

---

## Novel Insights

The most genuinely novel observation in this paper is the cross-domain influence finding that StarCoder (code data) benefits mathematical reasoning more than OpenWebMath (math data) benefits coding — a reversal of the conventional direction of transfer assumed since Lewkowycz et al. (2022). This has implications for dataset curation decisions beyond small-model training. The paper also demonstrates concretely that staged alignment-first, reasoning-second post-training substantially outperforms joint training for small models, which is an actionable insight for practitioners working in resource-constrained settings.

---

## Suggestions

1. **Add a benchmark-level ablation comparing influence-weighted vs. uniform-weighted vs. heuristic-weighted pre-training data mixture** evaluated on MATH, GSM8K, and LCBv6. This is the single most important missing experiment to validate the methodological contribution.
2. **Revise the central efficiency claim** to either: (a) provide a parameter-matched comparison at 600M, or (b) explicitly acknowledge the parameter asymmetry and reframe as "our 950M model surpasses Qwen3-0.6B using only X% of Qwen3-series training tokens, despite the size advantage."
3. **Clarify unique vs. total tokens** in the abstract and throughout. Report "4.2T total training tokens drawn from ~2T unique tokens" consistently to make the efficiency claim precise.
4. **Validate the probing-NLL proxy assumption** with a scatter plot across LOO runs relating probing NLL changes to final benchmark accuracy changes.

---

## Score and Decision

**Calibration anchors consulted:**

| Paper | Avg Score | Decision | Comparison |
|---|---|---|---|
| `/home/wg25r/review_agent/human_reviews/3OyaXFQuDl.md` | 7.0 | Accept Poster | Sub-billion reasoners via compute-optimal sampling; similar topic, clear counterintuitive finding, good experimental design. Paper under review has stronger empirical breadth but weaker methodology validation. |
| `/home/wg25r/review_agent/human_reviews/eENHKMTOfW.md` | 6.0 | Accept Poster | Practical guide for SFT of small LLMs; similar practical contribution level, comparable empirical rigor. Paper under review has more novel methodology (influence scoring) and stronger results. |
| `/home/wg25r/review_agent/human_reviews/54KcduuYeG.md` | 5.5 | Reject | Data mixture prediction for LLM pretraining; similar theme of data composition optimization. Rejected partly for insufficient scale validation. Paper under review has similar issue (NLL proxy not validated at benchmark level) but stronger empirical contribution. |
| `/home/wg25r/review_agent/human_reviews/1fwZJzGdKj.md` | 5.5 | Reject | Multi-agent data selection for LLM pretraining; similar topic. |
| `/home/wg25r/review_agent/human_reviews/yiPtWSrBrN.md` | 3.0 | Reject | Small LM training, very weak contribution, much weaker than this paper. |

**Scoring rationale:** The paper is clearly above the low anchor (~3.0, weak contributions, no surprising findings). It is comparable to the 5.5 anchors (data mixture papers that have the NLL-proxy issue), but slightly stronger due to: (a) stronger empirical results (SOTA in a competitive, practically important space), (b) more comprehensive open-source release, (c) the cross-domain transfer finding, and (d) the concrete post-training ablation. Compared to the 7.0 anchor (compute-optimal reasoner paper), the paper under review has a more extensive empirical evaluation but a weaker methodological claim (the influence method is not validated at benchmark level). The misleading framing of the central efficiency claim is a real issue that prevents a score in the 7+ range. Positioning at **5.5** — above the data-composition rejects, comparable to practical small-model-training contributions at the accept borderline.

**Axis summary:**
- *Originality*: Moderate — builds on AutoMixer and established LOO analysis; the cross-domain insight is novel.
- *Importance*: High — on-device/small-model reasoning is practically important and the SOTA results are real.
- *Claims support*: Weak on the central efficiency claim (parameter asymmetry, 36T denominator issue); strong on the open-source SOTA claim.
- *Experimental soundness*: Moderate — good ablations on post-training; missing the key ablation connecting influence method to final accuracy.
- *Clarity*: Good overall; misleading in the efficiency framing.
- *Community value*: High — full open release, detailed recipe, strong baseline results.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>