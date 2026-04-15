Now let me search for calibration papers to compare scores.Now I have enough information to write the consolidated review.

---

## Summary

This paper investigates whether training language models to win debates via self-play improves an AI judge's ability to determine correct answers in an information-asymmetric reading comprehension setting (QuALITY-HARD). Using a modified DPO objective (DPO+) that leverages continuous judge probabilities and branching self-play rollouts, the authors show for the first time that stronger trained debaters yield a statistically significant 4% absolute improvement in judge accuracy—a result that prior training-based work (Radhakrishnan 2023) failed to observe. By contrast, analogous optimization on consultancy baselines produces no such trend. The paper introduces two novel consultancy evaluation protocols (ensembled and double consultancy) to more rigorously situate the debate-vs-consultancy comparison.

---

## Claims and Support

| Claim | Verified? | Notes |
|---|---|---|
| Training LMs to win debates produces more accurate judge evaluations | **Partially supported** | The 4-point gain and p<10⁻⁶ are real, but evaluation and reward share the same judge family. GPT-4o transfer (r=0.98) partially addresses this. |
| First demonstration of training-time debate improving evaluator accuracy | **Plausible, not disputable** | Paper cites Radhakrishnan 2023 as the only prior training attempt, which showed null results. |
| No analogous positive trend for consultancy | **Partially supported** | Consultancy baselines are tested but the strongest ones (ensembled, double) are evaluation-only protocols, not training objectives. |
| Debate training encourages more informative arguments (96% more quotes, lower repetition) | **Well-supported** | Figure 6 directly shows this. Confirmed in paper. |
| Win rates of debate checkpoints transfer to untrained GPT-4o (r=0.98 vs. 0.51 for consultancy) | **Well-supported** | Section 4.4, Figure 6 right panel. |
| Explicit refutation does not yet play a material role | **Well-supported and honestly stated** | Section 4.3 and Appendix G (single-turn debates equally accurate as two-turn). The paper presents this as a finding, not an overclaim. |
| Results imply debate is well-suited for supervising more sophisticated models | **Weakly supported** | Single task domain, LM judges only. Conclusion language is appropriately cautious ("suggest"). |

---

## Strengths

- **First empirical evidence that trained (not just prompted) debaters improve judge accuracy**: The distinction between inference-time prompting and actual training-time optimization is crucial for scalable oversight. Prior work (Radhakrishnan 2023) failed to show this trend. The positive result here directly advances the literature on a concrete open question.

- **Thoughtful baseline design with ensembled and double consultancy**: Rather than relying on a strawman consultancy baseline, the paper introduces two meaningful variants that progressively close the gap with debate. The insight that double consultancy (75%) nearly matches debate (77%) isolates the specific contribution of adversarial *training* vs. dual-sided *evaluation*, which is a genuine methodological contribution.

- **Mechanistic analysis beyond aggregate accuracy**: The paper quantifies policy changes (96% increase in quote usage for debate, 70% decrease for consultancy; increasing quote repetitiveness for consultancy), and validates strategies against an out-of-training-loop judge (GPT-4o), making directionally useful claims about *why* debate training produces different behavior.

- **Calibrated judge construction is non-trivial and properly motivated**: Reducing default sycophancy from 72% agreement to near-chance is a necessary prerequisite for the experiment to be meaningful, and the paper handles this carefully (Section 3.1, Figure 3).

---

## Weaknesses

### Fatal
*None.*

### Major

- **Judge entanglement between reward signal and evaluation**: The debaters are optimized to win under a finetuned GPT-4T judge, and evaluation of judge accuracy is also performed by that same judge (family) on self-play transcripts. The paper notes that "at no point is the judge trained on transcripts from our debater models," which is a partial mitigation. However, the training objective directly targets "convince this judge," and the evaluation metric is "does this judge get it right?" The GPT-4o transfer test (Section 4.4, r=0.98) is a useful partial check, but it tests only correlation of *win rates across checkpoints*, not direct judge accuracy on ground truth. A robust test would show that the accuracy trend persists using multiple held-out judges (including humans) not involved in reward construction. This is the main interpretive gap: the result is consistent with, but not conclusively distinguished from, debate models learning to produce transcripts that happen to suit this judge's style in a way that correlates with truth on this domain.

- **Consultancy training-evaluation mismatch limits the paradigm comparison**: The paper's core contention is that adversarial (debate) training has a unique property that non-adversarial (consultancy) training lacks—namely a positive skill-accuracy relationship. But ensembled and double consultancy, the most competitive baselines, are explicitly **evaluation protocols only**, not training objectives (Section 2.3: "Ensembled and double consultancy are different *evaluation* methods, not training procedures"). The consultancy model was trained only to maximize single-consultancy score. It remains undemonstrated that a consultant trained to maximize double-consultancy or pairwise objectives would lack the positive trend. The paper's strongest causal claim—that adversarial training uniquely prevents judge exploitation—cannot be cleanly established under this design asymmetry.

### Minor

- **Modest absolute effect size and apparent saturation**: The 4% absolute gain (from SFT to fully trained DPO) is statistically significant but small, and Figure 5 shows accuracy appearing to plateau around 77–78% despite continued win rate gains. Whether continued training would yield further improvements, or whether this represents a ceiling effect of the current setup, is left open. This doesn't invalidate the result but does limit strong claims about the "scalability" of the positive trend.

- **DPO+ contribution is not fully ablated in the main paper**: The DPO+ modification is presented as a methodological contribution (using continuous probability rewards rather than binary preferences), but the main paper doesn't include a direct apples-to-apples comparison between vanilla DPO and DPO+ under matched compute. Appendix C reports reward-transformation comparisons (logit/log vs. direct confidence), but not DPO vs. DPO+. The innovation's specific contribution is therefore unclear.

### Trivial

- **Learning rate asymmetry across debate and consultancy training**: A lower learning rate is used exclusively for the second round of debate training (noted to produce better head-to-head performance), but not for consultancy. The paper acknowledges this explicitly and ran hyperparameter sweeps for both. This is a minor transparency point and not a meaningful confound.

---

## Nice-to-Haves

- **Train a double-consultancy model** as a direct training objective (not post-hoc evaluation) to cleanly test whether adversarial training is necessary, or whether simply presenting two sides at training time (without opponent visibility) suffices.
- **Cross-play evaluation**: Evaluating a stronger debater against a weaker opponent (and vice versa) rather than relying solely on self-play would help establish whether accuracy gains reflect transcript quality improvements or self-play-specific dynamics.
- **Human evaluation of at least a subset of transcripts** to test whether the gains transfer to the ultimate target evaluator (humans), particularly given that both the training reward and evaluation metric use LM judges.
- **Quote-controlled ablation**: Verify that the accuracy gains are not simply a function of evidence volume by testing with quote-count-matched debate and consultancy transcripts.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[REMOVED] Harsh Critic: "Refutation mechanistic claims are overinterpreted"**: This criticism misreads the paper. Section 4.3 and the Discussion (5.1) explicitly state "explicit refutation does not yet seem to play a role" and "we find little evidence that this refutation materially affects the judge's decision making." The paper presents the *absence* of refutation's impact as a finding, with appropriate hedging. The harsh reviewer claimed this was an overclaim—it is the opposite.

- **[REMOVED] Harsh Critic: "The broader oversight claim (Claim 10) is unsupported"**: The paper's Conclusion says explicitly "our conclusions are limited to one particular domain and set of model capabilities" and uses "suggest" rather than "prove" throughout. The abstract says "shows promise that it can help." This is appropriately scoped language; the harsh reviewer applies a standard that the paper itself does not claim.

- **[REMOVED] Spark: "Explicitly document how quote extraction was verified"**: This is a reproducibility/implementation nitpick without evidence of a methodological problem.

- **[REMOVED] Neutral Reviewer, Weakness 3: "Under-specification of DPO+ hyperparameter interactions"**: Requesting full ablations of α, β, and γ interactions is beyond the standard for an empirical systems paper of this type. The hyperparameters are reported and the key reward-function comparison is in Appendix C. The lack of a complete factorial sweep is not a meaningful flaw. (Kept a weakened version of this as a minor weakness about vanilla DPO comparison.)

- **[REMOVED] Spark: "Report variance across training seeds"**: The core statistical claim rests on a regression across 433 questions with p<10⁻⁶; the statistical power is not the concern. Multi-seed training variance reporting is not standard in this literature.

- **[REMOVED] Neutral/Spark: "Potential distributional overlap between judge and debater SFT"**: Both are trained on prior transcripts by Michael et al. and Khan et al., but this is unavoidable given the task domain and is not a specific experimental flaw—it applies equally to all work in this area. The paper also uses distinct debater and judge data splits.

---

## Novel Insights

The most genuinely novel observation in the paper—supported by the double-consultancy baseline analysis—is that the adversarial *training* condition (not just adversarial *evaluation*) appears necessary to prevent non-truth-seeking strategies from emerging. The double-consultancy evaluation format achieves nearly the same one-shot accuracy as full debate (75% vs 77%), yet the underlying models are less reliable because they were trained without an adversary and learned to exploit judge weaknesses. This implies that the value of debate is not merely in providing dual-sided information to the judge, but in shaping the *training dynamics* of the models themselves: the presence of an opponent at training time appears to suppress judge-idiosyncratic strategy learning. This is a distinct and practically important finding that goes beyond the inference-time comparisons in prior work.

---

## Suggestions

1. **Evaluate with at least one human judge (or panel) on a subset**: Even 50–100 questions with human evaluation would substantially strengthen the core claim that debate training improves oversight for the ultimate target evaluator. This is the single highest-impact improvement.
2. **Train a double-consultancy model directly under its evaluation format** and report whether its skill-accuracy relationship becomes positive. If not, this strongly validates adversarial training as the key ingredient. If yes, it meaningfully revises the paper's core conclusion—either outcome is valuable.
3. **Report cross-play accuracy trends** alongside self-play: does a stronger debater improve accuracy when paired against a fixed opponent (e.g., the SFT model), and does the effect persist there? This would decouple transcript quality from self-play dynamics.

---

## Calibration and Score

**Comparison papers:**

1. **49ZYkhEGmv** (Scalable AI Safety via Doubly-Efficient Debate): Rejected, avg score ~6.5. Theoretical contribution to debate, no empirical LM training. The paper under review is more empirically grounded with a novel training-time result, but narrower in scope.

2. **licAR8FPTW** (Evaluating Oversight Robustness, withdrawn): Avg score ~3.2. Uses a synthetic toy game (CodeNames), no LM training, simple setting. The paper under review is substantially stronger in experimental ambition and empirical quality.

3. **OUkZXbbwQr** (Reward Design for Justifiable Sequential Decision-Making, accepted poster): Scores 6,6,8,8. Uses debate-based reward model in a real healthcare RL setting. Has a concrete deployed application but narrower contribution to LLM oversight specifically.

**Assessment**: The paper under review is clearly above `licAR8FPTW` (solid empirical contribution, real LMs, honest methodology). It is comparable to the lower end of accepted posters like `OUkZXbbwQr`, with a meaningful novel contribution (first training-time evidence for debate) but with real methodological limitations (judge entanglement, consultancy mismatch) that are less present in the accepted paper. It is stronger than the rejected theoretical debate paper on empirical grounding but narrower in theoretical scope.

**Axis evaluations:**
- **Novelty**: Moderate-high. First training-time evidence; novel consultancy baselines; DPO+ for multi-turn debate.
- **Technical soundness**: Moderate. Methodology is reasonable, but judge entanglement and consultancy mismatch are genuine gaps.
- **Empirical support**: Moderate. Clear positive signal with good statistical treatment; limited to one domain/judge.
- **Significance**: Moderate. Scalable oversight is important; this is a meaningful empirical step forward, not a complete answer.
- **Clarity**: High. The paper is well-organized, honest about limitations, and transparent about what it claims.

**Final score: 6.0** — Marginally above the acceptance threshold. The contribution fills a concrete gap in the debate-as-oversight literature with a real empirical result, but the scope is narrow and the two major methodological concerns (judge entanglement and training-evaluation asymmetry for consultancy) prevent stronger confidence in the headline interpretation.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>