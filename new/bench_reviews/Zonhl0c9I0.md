Good. Now let me read some calibration papers to score appropriately.Now I have enough information to write the consolidated review. Let me also quickly check one more calibration paper.---

## Summary

Self-Evolved Reward Learning (SER) is a self-training framework for reward models (RMs) that reduces dependency on expensive human preference annotations. Starting from a seed RM trained on only 15% of human-labeled data, SER iteratively self-labels unlabeled response pairs, filters high-confidence examples via a curriculum-style "learning status" mechanism (Status 1: easy good-vs-bad distinctions; Status 2: subtle quality differences), and retrains the RM. The paper reports consistent empirical improvements across four datasets (HH-RLHF, UltraFeedback, Summarize, StackOverflow) and four model families/sizes (Mistral 7B, Llama 3 8B, Llama 2 13B, Llama 3 70B), with average accuracy gains of 7.88% over the seed model and performance approaching or exceeding full-dataset training. PPO results on two datasets using GPT-4 win-rate evaluation provide additional downstream validation.

---

## Strengths

- **Practical relevance and consistent empirical signal.** The paper tackles a real and important bottleneck in RLHF: the cost of human preference labels. The core empirical finding—that self-training with confidence-based filtering can close the gap to full-data RM training—is replicated across 4 datasets and 4 model families, lending credibility to the result. The average improvement of 7.88% over the seed model is meaningful.

- **Intuitive curriculum design.** The two-status decomposition (easy clear-preference pairs first, hard similar-quality pairs later) is well-motivated and supported by the loop-wise breakdown in Figure 2. Loop 1 consistently provides the largest gains (~4.54% on average), and the Status 2 strategy for later loops demonstrably rescues models from stagnation on easy data, as seen in the Llama 13B / UltraFeedback case.

- **Broad experimental scope.** Testing across four distinct task domains, three model sizes, and including PPO downstream validation is more thorough than much prior work on self-training for reward models.

- **Honest discussion.** The Discussion section appropriately acknowledges that data filtering strategies are empirical, that convergence theory remains incomplete, and that later iterations show diminishing returns—a sign of intellectual honesty uncommon in papers that overclaim.

---

## Weaknesses

### Fatal
None. The core empirical pattern is plausible and consistent.

### Major

1. **PPO evidence is too thin to support the headline claim that SER "boosts the capabilities of large language models."** The abstract and introduction frame downstream LLM improvement as a key contribution, but the PPO evaluation covers only 2 datasets, uses only GPT-4 win rate as judge (no standard benchmarks such as AlpacaEval or RewardBench, no human evaluation), reports no sample sizes, and shows very high tie rates (54–71% on HH-RLHF). Margins against the Full baseline are within the noise given these tie rates. The claim that performance gains in PPO are "positively correlated" with RM accuracy (Section 4.2) is asserted across just two data points—this is not a correlation analysis. The evidence supports that SER-trained RMs are competitive, but falls well short of establishing that SER reliably boosts LLM capabilities.

2. **The self-labeling quality is never directly measured, leaving confirmation bias unaddressed.** SER trains the RM on its own predictions and evaluates on the same task family. The paper never measures the fraction of self-labeled pairs that agree with held-out human ground truth. The claim that high-confidence filtering selects reliable pseudo-labels is plausible but unvalidated. Without showing that filtered pairs are more accurate than unfiltered ones, the iterative improvement could be explained by regularization effects or data-volume effects rather than the proposed curriculum mechanism. This is a significant gap for a self-training paper.

3. **Core design choices lack ablation, making the "only 15% of human data needed" claim underspecified.** The 15% seed ratio is the central headline claim, but the paper provides no experiments at 5%, 10%, 25%, or 50% seed data. It is unknown whether 15% is robustly optimal, a lower bound, or a somewhat arbitrary choice. Similarly, the thresholds τ_high = 0.55, τ_low = 0.45, τ_Δ = 0.3 are set by noting they "provided the most consistent improvements" with no sensitivity analysis, even though these are central to the method's behavior. The "sufficient number" of predictions (e.g., "600 in the HH dataset") that determines status is introduced informally and varies with dataset size—this aggregation rule is never formalized.

4. **Evaluation protocol is underspecified for the small differences cited as victories.** The main text says full details are in Appendix C, and the metric appears to be pairwise accuracy. Several claimed wins over the Full baseline are 0.13%–1.93%. Without variance estimates, repeated runs, or significance testing, these small differences cannot be reliably interpreted as wins. This matters because the paper explicitly claims to "surpass" the full-data baseline in multiple settings.

### Minor

- **Notation/formalization gaps in the method.** The relationship between r_i (reward score, Eq. 1) and p_i^1, p_i^2 (probabilities, Eq. 2) is never stated—does the RM output a probability directly, or is r_i mapped through sigmoid? Since thresholding at 0.55/0.45 is central to the method, this is not cosmetic. For Status 2 in Eq. (4), the filtering selects pairs where |RM(A1) − RM(A2)| > δ, but the preferred/rejected ordering within those pairs is implicit—the pairwise loss in Eq. (5) needs a consistent ordering to be well-defined.

- **Error accumulation across iterations is not discussed.** The paper notes D_filtered accumulates across loops (D_filtered^n + D_filtered^{n-1}), but earlier pseudo-labels are never revised or reweighted. Error accumulation is a well-known failure mode of self-training, and at least a qualitative discussion or empirical check (e.g., reward score distributions across loops) is warranted.

- **The theoretical section is too weak to carry evidential weight.** Section 3.2 states convergence "under reasonable assumptions" but does not state those assumptions in the main text. The claim that Loop 1 improvement "verifies our Theory 1" (Section 4.1.2) is rhetorical—an empirical gain does not verify a theorem, especially when the theorem's assumptions remain unspecified in the main text. The theory should either be presented rigorously or not invoked as evidence.

- **PPO Figure 4 is confusing to parse.** "Left represents SER method" requires the reader to cross-reference the caption to interpret the table, and the comparison rows differ between panels (a) and (b) without clear explanation. High tie rates (54–71%) make it difficult to draw conclusions from small left-vs-right differences.

### Trivial

- The final sentence of the Discussion ("break through the performance ceiling of those strongest LLMs") is aspirational framing in a Discussion section, not a core claim, but it overshoots what the experiments demonstrate.

---

## Nice-to-Haves

- **RLAIF baseline comparison.** The paper motivates SER partly against RLAIF but never experimentally compares the two. Adding a baseline that uses GPT-4 (or another strong model) to label the same unlabeled data would directly quantify when self-evolution is better or worse than AI labeling.

- **Comparison with existing self-rewarding or self-training RM methods.** The paper cites Self-Rewarding Language Models (Yuan et al.) and Math-Shepherd but doesn't benchmark against them. At least a qualitative comparison of design choices would help readers calibrate SER's position.

- **Reward score distribution plots across iterations.** Showing how the RM's output distribution evolves loop by loop on a held-out set would reveal whether the model is becoming calibrated or just more confidently wrong—a key diagnostic for self-training methods.

- **Qualitative case studies.** Examples of pairs where the self-label agrees or disagrees with human judgment, especially in later loops, would illuminate whether the RM learns genuine quality signals or systematic shortcuts.

- **Computational cost comparison.** The paper claims "human-labor efficiency" but does not report GPU hours or training steps relative to full-data training. Given that SER involves iterative RM retraining, total compute may not be dramatically lower.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "SER still depends on initial RM quality, making the RLAIF distinction overstated."** The paper accurately characterizes RLAIF as requiring "stronger LLMs to provide feedback" while SER uses a same-scale RM as the labeler. This is a real and accurate distinction; RLAIF systems like Constitutional AI do indeed rely on larger models. Removed as factually disputable.

- **Harsh Critic: "The final sentence about breaking the performance ceiling of strongest LLMs is far beyond experiments."** This is a sentence in the Discussion, clearly framed as future potential. Discussions routinely include aspirational claims. Removed as nitpick.

- **Spark / Harsh Critic: "Reproducibility concern about undisclosed hyperparameters and implementation details."** The paper provides thresholds, model names, and dataset details. Removal of full training logs and implementation minutiae is standard. Removed under the hard rule on trivial reproducibility concerns.

- **Spark: "No human evaluation for PPO outputs."** GPT-4-as-judge is the current norm in RLHF papers at major venues; requesting human evaluation is a methodology request not standard for this setting. Moved to Nice-to-Have status rather than a weakness.

- **Human Finder Weakness 6: "Comparison with weak LLM feedback"** — This is essentially the RLAIF comparison suggestion, moved to Nice-to-Haves above.

---

## Novel Insights

Across the three reviewers, one genuinely useful observation emerges beyond the paper's own framing: the **loop-wise decomposition of self-training gains** provides evidence that curriculum staging (easy then hard) is not just a nice-to-have but is arguably necessary to prevent the RM from stagnating on simple data. The fact that Loop 2 (still on Status 1 data) consistently shows the smallest gains—and can even decrease performance—while Loop 3 (Status 2 data) recovers improvement, gives empirical grounding to the curriculum hypothesis that generalizes beyond this paper. This pattern is consistent across models and datasets and represents a transferable insight for self-training of evaluators more broadly. However, this remains observation rather than causal proof, since no direct measurement of self-label correctness is provided.

---

## Suggestions

1. **Run ablations on the seed data ratio** (5%, 10%, 15%, 25%) to show that 15% is not a cherry-picked sweet spot and to characterize the method's behavior as a function of human annotation budget.
2. **Add RLAIF as a baseline** using GPT-4 labeling of the same unlabeled pool, even for a single model/dataset, to directly quantify SER's advantage over AI-labeled alternatives.
3. **Measure self-label quality directly**: take a random sample of self-labeled pairs from each loop and compare to held-out human labels. Report the fraction correct per loop to validate (or challenge) the confirmation bias concern.
4. **Clarify the mapping from reward scores to probabilities** (Eq. 1 → Eqs. 2–3) and formalize the aggregation rule for status determination (what constitutes "a sufficient number"—make this dataset-agnostic or explicitly parameterized).
5. **Expand PPO evaluation**: add at least one standard benchmark (e.g., AlpacaEval 2.0), report sample sizes, and consider length-controlled win rates to reduce GPT-4 judge length bias.
6. **Perform threshold sensitivity analysis**: sweep τ_high and τ_low over [0.50, 0.65] and τ_Δ over [0.2, 0.5] to demonstrate that the reported thresholds are not uniquely special and that the method works for a range of values.

---

## Score and Decision

**Calibration:**

| Paper | Decision | Scores |
|---|---|---|
| Self-Taught Evaluators (I7uCwGxVnl) | **Reject** | 6, 5, 5, 5, 5 (avg ~5.2) |
| SemiReward (dnqPvUjyRI) | **Accept/Poster** | 6, 6, 6 (avg 6.0) |
| Progress or Regress (RFqeoVfLHa) | **Accept/Poster** | 6, 6, 8, 6 (avg 6.5) |
| Iterative Label Refinement (q5EZ7gKcnW) | **Accept/Spotlight** | 8, 8, 5, 8 (avg 7.25) |

**Positioning reasoning:**
- SER is more empirically broad than Self-Taught Evaluators (more datasets/models) but shares many of the same weaknesses (degradation after iterations, no analysis of label quality, thin theoretical grounding). STE was rejected; this is a meaningful anchor for the low end.
- SER is roughly comparable to SemiReward in practical scope, methodological rigor, and contribution level—SemiReward received poster at 6/6/6.
- SER falls clearly below Iterative Label Refinement, which had stronger theoretical motivation, cleaner ablations, and a more surprising finding.
- The PPO weakness, missing ablations on the central 15% claim, and unvalidated self-label quality are material gaps, but the multi-dataset/multi-model empirical signal is genuine.

**Verdict:** Borderline, leaning toward a marginal accept at poster level. The core empirical contribution is real and consistently demonstrated; the paper's main weaknesses (PPO thinness, threshold sensitivity, missing ablations) are significant but not fatal—they are the kind of issues that would be addressed in a revision. The paper is more thoroughly validated than STE (rejected) and comparable to SemiReward (accepted poster).

**Originality:** Moderate — curriculum self-training for RMs is novel in combination, though components are individually known.  
**Importance:** High — reducing human annotation cost for reward models is a central RLHF bottleneck.  
**Claim support:** Adequate for reward modeling; weak for PPO/LLM improvement.  
**Experimental soundness:** Good breadth, notable gaps in ablations and PPO depth.  
**Writing clarity:** Generally clear; notation gaps in methodology section.  
**Community value:** Positive; the empirical findings on loop-wise behavior are practically informative.

**Score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>