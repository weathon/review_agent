Now let me search for calibration anchors.Now I have enough information to write the final meta-review.

---

## Summary
This paper presents the first demonstration that training language models to win debates (via self-play DPO) improves a calibrated judge's accuracy on reading comprehension questions (QuALITY-HARD), producing a 4% absolute accuracy gain. The authors contribute a modified DPO objective that exploits continuous judge-probability rewards, a calibrated finetuned judge (addressing sycophancy in prior work), and two novel baseline evaluation protocols—ensembled and double consultancy—that allow systematic decomposition of *why* debate outperforms consultancy. An important mechanistic finding is that the theorized refutation mechanism does not appear to be operative, and the gains are instead attributable to balanced information exposure and adversarial training dynamics.

---

## Strengths

- **First positive result for trained debate as scalable oversight.** Section 4.2 shows a 4% absolute increase in judge accuracy from SFT to fully DPO-trained debate (p < 10⁻⁶). Prior trained debate work (Radhakrishnan, 2023) failed to find this effect; this is a meaningful first. The absence of any ground-truth supervision signal during debater training strengthens the significance.

- **Novel ensembled and double consultancy baselines enable principled mechanistic decomposition.** The four-way comparison (single consultancy 68%, ensembled consultancy 72%, double consultancy 75%, debate 77%) precisely isolates how much of debate's advantage comes from asymmetric evidence exposure, simultaneous comparison, and adversarial training pressure. This goes substantially beyond prior work.

- **Modified DPO objective with continuous reward signal.** The cross-entropy formulation against a Bradley–Terry-converted reward probability (Equation 2) is well-motivated over binary-preference DPO, and the ablation against logit/log transforms is reported. The concurrent Reward-aware Preference Optimization (Nvidia et al., 2024) is properly acknowledged.

- **Calibrated judge training mitigates a key confound in prior work.** Section 3.1 documents that untrained GPT-4-Turbo agreed with consultants 72% before training and Figure 3 shows the finetuned judge is better calibrated on both debate and consultancy, directly addressing the sycophancy issue that inflated consultancy numbers in Khan et al. (2024) (>90% agreement).

- **Evidence-use and transfer analyses provide concrete mechanistic texture.** Debate models increase quoted evidence by 96% while consultancy models decrease it by 70% and become 98% repetitive (Section 4.4). The Pearson correlation between trained GPT-4T and untrained GPT-4o win rates is 0.98 for debate vs. 0.51 for consultancy, suggesting debate training learns generalizable rather than judge-specific strategies.

---

## Weaknesses

### Fatal
None.

### Major

- **The GPT-4o transfer analysis measures win rates, not judge accuracy.** Section 4.4 shows a 0.98 correlation between debate win rates under the trained GPT-4T judge and an untrained GPT-4o judge, but this correlation concerns *which model wins*, not *whether the correct answer is chosen*. The central accuracy claim (Figure 5) is measured exclusively against the trained judge. It is plausible that stronger debaters are more convincing to any judge without that being tied to ground truth. Directly measuring whether GPT-4o also shows a positive skill-accuracy relationship for debate checkpoints would close this gap. As it stands, the positive accuracy trend and the GPT-4o transfer result are parallel measurements that do not jointly establish generalization of the accuracy gains.

- **Framing tension between introduction/conclusion and actual findings.** The paper motivates debate as scalable oversight via Irving et al.'s refutation mechanism, and the conclusion states the results "suggest that debate training has unique properties that make it well suited for supervising more sophisticated models." However, Section 4.3 and Appendix G (single-turn ablation) directly show that refutation is not operative — double consultancy nearly matches debate without any opponent exposure at generation time. Section 5.1 addresses this honestly, but the introduction (claiming debate "takes another crucial step" in implementing scalable oversight) and conclusion do not update their framing accordingly. The gains stem from information balancing and training-time adversarial pressure preventing judge exploitation — not from the refutation-based truthfulness guarantee that motivates the scalable oversight argument.

### Minor

- **Asymmetric hyperparameter tuning between debate and consultancy conditions.** Section 3.2.2 reports that the second DPO iteration uses a lower learning rate for debate (5e-5 vs 1e-5) and different γ values (7 for debate, 10 for consultancy), each the result of separate sweeps. The headline comparison is the positive skill-accuracy trend for debate vs. the flat trend for consultancy, so any asymmetry in optimization effort is a potential confound. The paper is transparent about this but does not ablate sensitivity to these choices for each condition.

- **Single training run with no variance across seeds.** The skill-accuracy relationship (Figure 5) is derived from checkpoints of a single training run. With only ~10 evaluation checkpoints spanning one DPO epoch, the slope of the accuracy curve could be sensitive to initialization. No seed variance is reported.

- **Single domain with artificially information-based expertise gap.** The paper honestly discusses (Section 5.2) that the information-asymmetry proxy may not generalize to capability-based expertise gaps (e.g., reasoning), and that Kenton et al. (2024) found debate helps reading comprehension more than reasoning tasks. The results are thus informative but limited in scope.

### Trivial
None identified (formatting artifacts in the extracted PDF are parser issues, not paper problems).

---

## Nice-to-Haves
- A direct judge accuracy measurement using GPT-4o or another independent judge on the debate checkpoints would be the single most impactful extension, directly addressing the major weakness above.
- Analysis of transcripts to distinguish "refutation absent" from "refutation present but ignored" would clarify whether future progress requires better debater training or better judge prompting.
- Multiple training seeds would make the skill-accuracy slope more robust.
- Testing a "double consultancy trained" condition (trained knowing both sides' speeches appear at evaluation) would more cleanly isolate whether the adversarial training signal itself, rather than just the adversarial evaluation structure, drives the debate/double-consultancy gap.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: p-value should be interpreted as a single training trajectory, not robust trend.** The p < 10⁻⁶ significance is correctly applied to the comparison of checkpoint accuracies on 433 questions. This is standard evaluation practice. The paper does not use the p-value to claim the trend is cross-seed robust; that concern is captured under the minor weakness on single training runs.

- **Harsh Critic: Judge distribution shift — DPO models improve partly by producing GPT-4-like transcripts.** This is speculative and not grounded in evidence from the paper. No data supports the claim that DPO-trained 8B models' outputs are mistaken for GPT-4 transcripts by the fine-tuned judge; the paper's Section 4.1 shows that even the initial SFT model was treated with skepticism.

- **Harsh Critic: Single-turn ablation deserves prominence in main text, not appendix.** The paper's Section 4.3 explicitly states: "We also run experiments with single-turn debate and consultancy which yield a similar conclusion, as the one-turn debates where no explicit refutation could occur are judged just as accurately as two-turn debates (see Appendix G)." The main text does incorporate this finding; putting the full experiment in the appendix is appropriate.

- **Strength Finder: "This paper addressed an important problem."** Generic strength dropped per instructions.

- **Strength Finder: One-turn debate control condition listed as a strength.** Reclassified as a methodological element that supports the mechanism analysis (already captured in the mechanistic decomposition strength) rather than an independent strength.

---

## Novel Insights
The most genuinely novel observation synthesized across the reviews is the behavioral divergence revealed by the consultancy policy analysis: consultancy training *decreases* evidence use and increases repetition, while debate training increases evidence use — suggesting that the adversarial dynamic in debate serves as an implicit regularizer against cheap argumentative strategies. The mechanism is well-specified in Section 5.1: if a strategy's persuasiveness is independent of truth (like repetition or unsupported assertion), adversarial training eliminates it because an opponent can reveal the absence of supporting evidence. This is a more concrete and testable version of Irving et al.'s intuition, and it reorients the scalable oversight argument away from refutation and toward training-time adversarial pressure as the active ingredient.

---

## Calibration Anchors

| Path | Avg Score | Comparison |
|---|---|---|
| `/home/wg25r/human_reviews/ChNy95ovpF.md` (DebateGPT) | 4.33 | Debate-related but uses multi-agent debate only for SFT data generation, no novel baselines, no mechanism analysis. Clearly weaker than the paper under review. |
| `/home/wg25r/human_reviews/tCfvktlrHI.md` (Self-Play Non-Zero-Sum) | 4.75 | Self-play LM training in games, single task, no mechanistic decomposition, no alignment framing. Weaker methodology. |
| `/home/wg25r/human_reviews/QAwaaLJNCk.md` (Multiagent Debate Factuality) | 6.00 | Prompting-only debate, no training, weaker baselines. Paper under review has better methodology and clearer framing. |
| `/home/wg25r/human_reviews/FQepisCUWu.md` (ChatEval) | 5.60 | Multi-agent debate for LLM evaluation, accepted, similar accuracy theme but less rigorous mechanistic analysis. |
| `/home/wg25r/human_reviews/licAR8FPTW.md` (Oversight Robustness) | 3.17 | Scalable oversight evaluation on synthetic domain, weaker empirical grounding and significant methodological gaps. |
| `/home/wg25r/human_reviews/xsELpEPn4A.md` (JudgeLM) | 7.50 | Fine-tuning LLMs as judges, accepted spotlight; stronger generalization across tasks and scales. Paper under review has narrower scope. |
| `/home/wg25r/human_reviews/Cnwz9jONi5.md` (Rethinking Reward Model Eval) | 7.25 | Reward model evaluation, accepted spotlight; broader task coverage, stronger analysis. Comparable methodological care. |

**Score rationale:** The paper under review clearly surpasses the low-scoring anchors (ChNy95ovpF at 4.33, tCfvktlrHI at 4.75, licAR8FPTW at 3.17) in originality, methodological rigor, and honest mechanism analysis. It is meaningfully better than the 5.6–6.0 borderline papers (ChatEval, Multiagent Debate Factuality) due to its training-based approach, novel baselines, and the first positive result in a previously unsuccessful space. It falls short of the 7.25–7.5 spotlight papers primarily due to single-domain scope and lack of direct accuracy validation with an independent judge. A score of **6.5** is appropriate — solidly above borderline, reflecting genuine contributions on an important problem, but not at spotlight level given the generalization gaps.

## Score and Decision

**Originality:** High within its niche — first trained-debate positive result, novel baselines, novel DPO variant.
**Importance of research question:** High — scalable oversight is a central AI safety question.
**Claims supported:** Mostly well-supported; the core accuracy claim is backed by strong statistics, but the scalable oversight framing is slightly ahead of what the experiments show.
**Soundness of experiments:** Good, with the noted caveat about independent judge validation and single-run results.
**Clarity of writing:** Very good — clear structure, honest limitations section.
**Value to research community:** Meaningful — provides the first empirical grounding for trained debate and a decomposition framework for future work.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>