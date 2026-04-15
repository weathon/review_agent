Now I have sufficient calibration information. Let me synthesize the final review.

---

## Summary
This paper investigates whether training language models to win debates via self-play can improve the accuracy of a judge on a reading comprehension task where the judge lacks access to the source text. The core finding is a positive relationship between debate model skill and judge accuracy (~4 percentage point absolute gain, p < 10⁻⁶), while analogous consultancy-style baselines show no such trend. The paper contributes a novel multi-turn DPO variant with soft reward targets, a finetuned calibrated judge, and two new consultancy baseline protocols (ensembled and double consultancy) that sharpen the comparison with debate.

---

## Strengths

- **First positive training-time result for the skill→accuracy relationship in debate.** Prior work by Radhakrishnan (2023) failed to show this with training, while Khan et al. (2024) and Kenton et al. (2024) showed it only for inference-time optimization. This paper demonstrates the trend survives model training, which is an important step for scalable oversight.

- **Genuinely informative consultancy baselines.** The addition of *ensembled consultancy* and especially *double consultancy* (reaching 75% accuracy vs. debate's 77%) goes substantially beyond Michael et al. (2023)'s single-consultancy baseline. These designs allow the paper to decompose debate's advantage into (a) access to both sides, (b) in-context comparison, and (c) training-time adversarial dynamics—a principled decomposition missing from prior work.

- **Judge calibration design addresses a real confound.** The paper explicitly identifies that GPT-4T without finetuning is sycophantic (72% agreement with consultant before any training) and produces overconfident, uncalibrated probabilities. The finetuned judge is trained on prior human transcript data and shows improved Brier scores and calibration—directly addressing a known weakness in prior consultancy comparisons.

- **Reward-aware soft-label DPO for multi-turn debate.** Using the judge's continuous probability to construct soft Bradley-Terry targets for DPO, applied via branching rollouts in self-play, is a technically coherent and practically useful extension that enables richer training signals than binary preference data alone.

- **Evidence of divergent learned policies.** The finding that debate models use 96% more quoted evidence and transfer to an untrained GPT-4o judge with r=0.98 correlation (vs. 0.51 for consultancy) provides concrete, quantitative signals distinguishing the two training regimes beyond headline accuracy.

---

## Weaknesses

### Fatal
None. The paper has no single fatal flaw that invalidates its core claim. The core result—a positive training-time skill→accuracy relationship for debate—is robustly demonstrated within this setting.

### Major

- **Reward model and evaluator are the same artifact.** The most significant limitation is structural: the finetuned GPT-4T judge serves simultaneously as the reward signal for debater training and as the primary measure of judge accuracy. The paper shows that the GPT-4o transfer correlation for debate is r=0.98, which is promising, but this is reported only as a win-rate ranking correlation, not as a full independent judge-accuracy measurement. Without a direct end-to-end accuracy comparison under at least one fully independent judge (different architecture/training lineage), the central claim cannot cleanly distinguish "debate training produces genuinely more truth-revealing arguments" from "debate training produces arguments that interact more productively with this specific GPT-4-family judge."

- **Self-play-only evaluation narrows the scope of the main claim.** Judge accuracy is measured exclusively on self-play transcripts, where both debaters are drawn from the same checkpoint. This creates a specific equilibrium condition (same stylistic biases, same failure modes on both sides) that may not generalize to cross-play settings relevant for scaling oversight. The paper's central framing—that stronger debaters will yield better oversight of stronger AI systems—implies cross-play generalization that is not empirically tested.

- **Narrow single-domain scope vs. broad framing.** All results are on QuALITY-HARD reading comprehension with a specific hidden-text information asymmetry. The paper itself acknowledges that Kenton et al. (2024) found debate less helpful on non-reading-comprehension tasks. Given this acknowledged fragility in the literature, the broader framing around scalable oversight would benefit from at least one additional domain or information-gap type. As it stands, the paper establishes a valid proof of concept, but the broader framing slightly outpaces the evidence.

### Minor

- **Double consultancy's training-evaluation asymmetry.** Debate is trained in the same adversarial interaction format it is evaluated in (both sides see each other's arguments). Double consultancy is a stronger evaluation protocol applied to models trained for single consultancy. The authors acknowledge this (Section 2.3), but it means the comparison underestimates what a model trained for double consultancy-style pairwise comparison might achieve. The residual 2-point gap between double consultancy and debate may therefore partly reflect this asymmetry rather than an intrinsic advantage of debate's adversarial training.

- **Mechanistic claims about refutation are underspecified.** The paper concludes "explicit refutation does not yet seem to play a role" based on (a) double consultancy nearly matching debate, and (b) single-turn debates matching two-turn debates. However, this could equally reflect underdeveloped refutation in current models at this scale rather than an inherent unimportance of refutation. The two-turn simultaneous format also limits meaningful refutation by design. The paper appropriately hedges in the discussion, but the abstract's framing edges toward overstatement.

- **Modest effect size.** The headline 4-point absolute accuracy gain, while statistically significant (p < 10⁻⁶ on 433 questions), is modest in practical terms, especially given that double consultancy already closes most of the gap. The paper presents this honestly, but it affects how strongly the oversight conclusions can be drawn at current scale.

- **Hyperparameter asymmetry between debate and consultancy training.** Different γ values (7 vs. 10), and a different learning rate for debate's second DPO round, were used. The paper notes a hyperparameter sweep was run, but the differing choices make strict apples-to-apples comparison of the training procedures slightly difficult to interpret.

### Trivial

- The claim that training yields improvements "without the requirement of a ground truth supervision signal" is technically accurate for the debaters but slightly misleading: the judge itself is finetuned on labeled human judgments, and the evaluation labels are used to measure accuracy. A small clarification would improve precision.

---

## Nice-to-Haves

- Report judge accuracy (not just win-rate correlation) under the untrained GPT-4o judge for all checkpoints, elevated as a primary result rather than an auxiliary analysis.
- Add cross-play evaluation: strong vs. weak checkpoint matchups to verify whether the skill→accuracy trend holds outside the specific same-checkpoint self-play equilibrium.
- Add variance across training seeds and confidence intervals on the key trend plots in Figure 5, since the core claim rests on trend shapes over checkpoints.
- Analyze quote fidelity and false-paraphrase rates across checkpoints to confirm that higher quote counts reflect genuinely informative evidence rather than selective misquotation.
- Probe refutation more directly by annotating whether second-turn rebuttals successfully identify factual errors in the opponent's first turn, and whether judges cite them in their verdicts.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Cannot verify release status/availability of cited models"** (raised implicitly in reproducibility concerns about GPT-4T finetuning API): Per the hard rules, all cited models and tools are assumed real and available.

- **Criticism that single-consultancy training vs. double-consultancy evaluation is an unfair comparison disadvantaging the author's method:** The asymmetry here slightly *disadvantages* the consultancy baseline (the baseline would presumably be stronger if trained for the pairwise setting). This asymmetry favors the authors' argument that debate has a genuine advantage, so by the hard rules this cannot be treated as a flaw in the paper's comparison.

- **Criticism about undisclosed hyperparameters / request for full training logs:** The paper discloses β, γ, α, learning rates, batch size, LoRA rank, and iteration counts. Requests for further exhaustive implementation details fall under the reproducibility nitpick rule.

- **Request for theoretical proofs / formal guarantees:** This is an empirical systems paper; demanding theoretical proofs is not standard in this field and has been moved to nice-to-haves.

- **"The paper is well-written" / "The topic is important":** Removed as generic strengths.

---

## Novel Insights

The paper's most genuinely novel insight is the mechanistic decomposition using the consultancy protocol ladder: by showing that single → ensembled → double consultancy accounts for most of the accuracy gap between debate and plain consultancy, the paper establishes that the bulk of debate's benefit comes from *information availability and in-context comparison* rather than from adversarial refutation. This partially inverts the original theoretical motivation for debate (Irving et al., 2018), which emphasized refutation as the key mechanism. Simultaneously, the finding that consultant models converge on judge-idiosyncratic strategies (low GPT-4o transfer, increasing repetition) while debate models adopt more general evidence-based strategies (high GPT-4o transfer, more quoting) is a concrete behavioral signature of the alignment risk posed by non-adversarial RLHF-like training, and provides a new empirical fingerprint for distinguishing truth-seeking from judge-exploiting optimization.

---

## Suggestions

1. **Elevate the GPT-4o evaluation to a primary result.** Report full judge accuracy (not just win-rate correlation) under untrained GPT-4o across debate and consultancy checkpoints. If the accuracy trend persists under this independent judge, it substantially strengthens the main claim.

2. **Add cross-play accuracy measurements.** Evaluate judge accuracy on transcripts where strong checkpoints debate against weak checkpoints (and vice versa), not only same-checkpoint self-play. This tests whether the skill→accuracy relationship is a genuine property of stronger models or an artifact of a particular equilibrium.

3. **Train a consultancy model for the double-consultancy evaluation protocol.** This would close the methodological gap flagged in the training-evaluation asymmetry and sharpen the remaining difference attributable to debate's adversarial dynamics.

4. **Narrow the abstract claim.** Replace "training language models to win debates with self-play improves judge accuracy" with a qualified version such as "in a hidden-context reading comprehension setting, training to win debates…" to precisely reflect the evidential scope.

---

## Score and Decision

**Calibration:**

- **licAR8FPTW** (Evaluating Oversight Robustness with Incentivized Reward Hacking, similar domain): Scores 3, 6, 3, 1, 3, 3 → Rejected. That paper was weaker: toy domain, no training results, poor writing. This paper is substantially stronger.
- **OUkZXbbwQr** (Reward Design for Justifiable Sequential Decision-Making, debate-based RL, one domain): Scores 6, 6, 8, 8 → Accepted. Single domain, interesting method, good analysis. That paper was considered proof-of-concept at similar empirical scope.
- **49ZYkhEGmv** (Doubly-Efficient Debate, theoretical debate work): Scores 6, 8, 6, 6 → Rejected. Strong theoretical contribution but no empirical grounding. The current paper is *more* empirically grounded but *less* theoretically novel.
- **ChNy95ovpF** (DebateGPT, using multi-agent debate for finetuning): Scores 3, 5, 5 → Rejected. That paper lacked clear contribution and comparison rigor. The current paper is substantially better designed and more clearly motivated.

**Assessment:** This paper sits clearly above the withdrawn/rejected scalable oversight papers (licAR8FPTW, ChNy95ovpF) and is roughly comparable to OUkZXbbwQr, which was accepted at 6-8. The current paper's core contribution—the first training-time positive result for the skill→accuracy relationship in debate, with novel baselines and a careful judge design—is meaningful and well-executed within its scope. The main weaknesses (single domain, shared reward/eval judge, self-play-only evaluation) are real but do not invalidate the result; they limit its scope. The paper is a solid, carefully done empirical paper that advances an important line of research. I calibrate this as **6.5**: above the borderline-accept band, but not a standout paper due to the limited empirical scope and the shared reward/evaluator concern. Given that OUkZXbbwQr was accepted at 6-8 with a single domain and no ground-truth judge independence, and that the current paper's contribution is more tightly motivated and better designed, a 6.5 is appropriate.

**Axes:**
- *Novelty*: Moderate-high. First positive training-time result; novel baselines; reward-aware DPO variant.
- *Technical soundness*: Good. Method is coherent and carefully implemented.
- *Empirical support*: Moderate. Convincing within this domain/judge; limited cross-domain and cross-judge validation.
- *Significance*: Moderate-high for the scalable oversight community.
- *Clarity*: Good. Well-structured, honest about limitations.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>