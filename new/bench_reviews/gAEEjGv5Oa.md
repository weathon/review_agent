Now let me read the calibration papers referenced by the human finder reviewer.Now let me read the calibration papers to anchor my score.---

## Summary

This paper investigates whether **training** language models to win debates—using self-play and a modified DPO objective—improves the ability of an AI judge to identify correct answers, as opposed to merely optimizing at inference time. On the QuALITY-HARD reading-comprehension benchmark with an information-asymmetry setup, the authors find that stronger self-play debaters (by checkpoint) are associated with 4% higher judge accuracy (p < 10⁻⁶), while trained consultancy models show no such positive trend. The paper introduces three consultancy variants to decompose the mechanisms behind debate's advantage, and finds honest null results: explicit refutation does not appear to be operative.

---

## Strengths

- **First positive training-based result for debate accuracy.** Prior work (Radhakrishnan 2023) trained models to debate but failed to show judge-accuracy improvement. This paper breaks that negative result, establishing that the positive skill–accuracy trend seen at inference time (Khan et al., Kenton et al.) also holds under training. This is a real milestone for scalable oversight.

- **Thoughtful consultancy baseline decomposition.** The three consultancy variants (single, ensembled, double) isolate distinct mechanisms—information asymmetry, side-by-side comparison, refutation—and provide genuine insight into *why* debate helps. Double consultancy reaching 75% vs. debate's 77% is itself an informative finding.

- **Careful judge design.** Addressing sycophancy bias (initial GPT-4T agreement 72%) via finetuning and using token-level probabilities for calibrated rewards is a practically important methodological contribution that strengthens the experimental foundation.

- **Modified DPO with continuous rewards.** The reward-aware DPO formulation that exploits the judge's continuous confidence rather than binary preferences is well-motivated and cleanly described.

- **Scientific honesty about refutation.** The paper forthrightly reports that single-turn debates are judged as accurately as two-turn debates (Appendix G), and that double consultancy nearly matches debate—openly undercutting the theoretical refutation story while providing a more nuanced explanation. This is commendable.

- **Compelling policy analysis.** The finding that debate models use 96% more quoted words while consultancy models become more repetitive, plus the 0.98 vs. 0.51 Pearson correlation for transfer to GPT-4o, provides a mechanistic account of the differing trends.

---

## Weaknesses

### Fatal
*(None. The paper's core empirical claim—that training for debate produces a positive skill–accuracy trend in self-play—is supported in the described setting.)*

### Major

- **Cross-play evaluation is entirely absent, yet it is the critical test for the scalable oversight claim.** The paper exclusively uses self-play (model vs. itself) to measure judge accuracy. But the motivating threat model for scalable oversight is adversarial: a *stronger* model defending the *wrong* answer competes against a weaker model defending the *right* one. Without cross-play experiments where a fully-trained wrong-side debater faces an SFT-level right-side debater, the paper cannot establish that debate actually helps judges find truth in the adversarial setting that matters. The positive self-play trend could reflect that both debaters improving symmetrically, with truth-correlated evidence accumulating naturally; it does not demonstrate that debate resists adversarial deception.

- **The trained judge is used simultaneously as the reward source, model-selection criterion, and primary evaluator.** Debater models are trained on rewards from the finetuned GPT-4T judge, skill (win rate) is measured under this same judge, and the headline judge accuracy figures are produced by this same judge. The GPT-4o transfer check in Section 4.4 measures win-rate *correlation*, not judge *accuracy on truth*—which is the paper's headline claim. Without showing that the same checkpoint ordering yields improved accuracy under an independent judge, the result is compatible with "training produces debates that this particular judge handles better," which is a weaker claim than the paper makes.

- **The debate vs. double consultancy gap is 2 percentage points (77% vs. 75%), and the paper never tests whether this difference is statistically significant.** Since double consultancy is the strongest baseline and the closest to debate, this gap is the primary evidence that adversarial interaction per se provides unique value. If the 2pp gap is not significant, the paper's central contrastive claim—that debate uniquely benefits judges beyond simply presenting both sides—becomes unsupported.

### Minor

- **Consultancy training–evaluation mismatch weakens the comparative conclusion.** Consultancy models are trained for single consultancy, while ensembled and double consultancy are evaluation-only wrappers (explicitly noted in Section 2.3). A model trained specifically for double consultancy (aware that both sides will be presented simultaneously) might narrow the gap further, making it unclear how much of the advantage is attributable to adversarial training as opposed to training–evaluation alignment.

- **Single model size (Llama3-8B) and only two DPO iterations leave open scalability questions.** Whether the positive trend continues, plateaus, or degrades with larger models or more training is unknown. The paper does not track accuracy across enough training iterations to characterize the asymptotic behavior.

- **The modified DPO loss is not ablated against standard DPO.** Three reward transforms (raw confidence, log, logit) are compared, but the additional SFT auxiliary loss term and the choice to use continuous vs. binary preferences are not systematically ablated. It is unclear whether the debate gains stem from the debate paradigm or partly from the specific loss engineering.

- **Statistical independence of the p < 10⁻⁶ estimate is questionable.** The same 433 questions are reused across all checkpoints, and checkpoints from the same training run are not independent. The reported significance level may be inflated.

### Trivial

- The base rate of QuALITY-HARD (4-way multiple choice → 25%) is not prominently reported, making it harder to contextualize the 68–77% absolute accuracy figures.

---

## Nice-to-Haves

- **Train a double-consultancy-specific model** to cleanly isolate adversarial training from evaluation-protocol effects, and test its skill–accuracy trend.
- **Include at least one non-reading-comprehension task** (e.g., math, code) in a preliminary capacity, even if only to confirm the domain boundary.
- **Conduct human judge experiments** on a small subset to validate that the observed trends are not artifacts of the AI judge's training distribution.
- **Analyze failure modes**: characterize the questions where debate fails (wrong side wins) to understand what drives the remaining 23% error rate and whether further training will shrink it.
- **Provide a quote-coverage visualization**: show what fraction of the story's relevant passages are quoted in debate vs. consultancy vs. double consultancy, to test whether debate's advantage is purely informational.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Overinterpretation of the scalable oversight implication."** While the abstract/intro do use broad language, the paper's conclusion explicitly states: *"Although our conclusions are limited to one particular domain and set of model capabilities, these results nonetheless suggest…"* Section 5.2 also enumerates three concrete limitations. The paper is more carefully scoped than this criticism implies. The phrasing in Section 2.4 ("provides evidence that the oversight protocol will continue to produce truth-seeking behavior with more advanced AI systems") follows Khan et al. (2024)'s own framing and is a plausible interpretation of the observed trend, not a dramatic overclaim. Weakened to context rather than retained as a standalone weakness.

- **Human Finder: "Post-hoc analysis of why debate outperforms consultancy."** The paper labels Section 5.1 explicitly as "Analysis" and does not claim a priori hypothesis testing. Post-hoc mechanistic analysis of observed results is standard scientific practice and not a methodological flaw.

- **Human Finder: "Dependency on judge calibration introduces reliability concerns about GPT-4's limitations."** This is a generic "LLM-as-judge" criticism that does not speak to a specific failure in the paper's design. The judge finetuning specifically addresses sycophancy and calibration, which are the relevant failure modes for this setting.

- **Neutral Reviewer: "AI judge limits conclusions about human oversight."** While human studies would strengthen the work, using an AI judge is entirely standard in this literature (Khan et al., Kenton et al., Radhakrishnan 2023 all use similar setups). This is a limitation inherent to the research program, not a specific flaw of this paper. Moved to nice-to-have.

---

## Novel Insights

The most genuinely novel finding is the **mechanistic contrast between debate and consultancy training dynamics**: debate models learn to deploy *more* evidence (96% more quoted words) while consultancy models become *more repetitive* and learn strategies idiosyncratic to the trained judge (0.51 vs. 0.98 transfer correlation). This suggests that adversarial competition at training time, rather than at evaluation time, suppresses the emergence of judge-exploiting shortcuts. The complementary finding—that refutation is not yet operative despite two-turn debates—is also novel and important: it implies that current debate gains derive from information revelation and side-by-side comparison rather than the argumentation dynamics that originally motivated Irving et al.'s proposal. Together these suggest debate's current empirical benefits are not yet the ones the theory predicts, which is a productive and honest observation for the field.

---

## Suggestions

1. **Add cross-play experiments** (stronger wrong-side debater vs. weaker right-side debater) to validate the scalable oversight threat model directly. This is the single most important missing experiment.
2. **Report statistical significance for the 2pp debate vs. double consultancy gap**, and discuss implications if non-significant.
3. **Evaluate judge accuracy (not just win-rate correlation) under an independent judge** (e.g., untrained GPT-4o) to disentangle training-to-judge adaptation from truth-seeking behavior.
4. **Clarify the statistical independence assumption** underlying the p < 10⁻⁶ value.
5. **Ablate the modified DPO loss** against standard DPO to quantify how much of the gain is due to continuous rewards versus adversarial training structure.

---

## Score and Decision

**Calibration:**

- `tCfvktlrHI` (self-play in cooperative language games, DoND): Scores 3, 5, 6, 5 → avg ~4.75 → **Rejected**. That paper also showed positive self-play results but on a simpler task with unclear generalizability. The paper under review is substantially stronger: clearer research question, larger positive result, and directly relevant to scalable oversight.

- `licAR8FPTW` (reward hacking in scalable oversight, synthetic domain): Scores 3, 6, 3, 1, 3, 3 → avg ~3.2 → **Withdrawn/Rejected**. That paper had largely preliminary and negative results. The paper under review is clearly above this level.

- `49ZYkhEGmv` (doubly-efficient debate, theoretical): Scores 6, 8, 6, 6 → avg ~6.5 → **Rejected** despite strong theoretical contribution. The paper under review makes a narrower but empirically grounded contribution; comparable in ambition but less technically deep.

- `tvhaxkMKAn` (sycophancy in RLHF): Scores 6, 6, 8, 6 → avg 6.5 → **Accepted**. Broader empirical coverage (5 models, 4 tasks), clearer story. The paper under review is somewhat narrower.

**Assessment:** This paper contributes a real first result (training-based positive trend for debate judge accuracy) and is well-executed within its scope. The missing cross-play experiments and judge-specificity concern are genuine gaps but do not invalidate the core empirical finding. The paper is above the rejected self-play paper (`tCfvktlrHI`) and the scalable oversight negative-result paper, comparable in quality to theoretical debate work, but below the broader, more tightly validated `tvhaxkMKAn`.

**Originality:** Good — first training-based positive result in this literature  
**Importance of research question:** High — scalable oversight is central to AI safety  
**Claim support:** Moderate — the narrow empirical claim is well-supported; the broader implications outrun the evidence  
**Soundness of experiments:** Moderate — careful within scope, but missing the critical cross-play validation  
**Clarity of writing:** Good — well-organized, honest about limitations  
**Value to research community:** Moderate-to-high — fills a real gap, provides useful mechanistic analysis  

**Final score: 5.5 — Borderline, leaning toward weak reject without cross-play and independent-judge accuracy results, but publishable if those gaps are addressed.**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>