Now I have enough information to write a thorough final review. Let me synthesize everything.

---

## Summary

This paper trains LLMs (Llama3-8B) to win debates via self-play using supervised finetuning followed by a modified DPO objective (DPO⁺) with a calibrated GPT-4-Turbo judge, on information-asymmetric reading comprehension questions from QuALITY-HARD. The central finding is a positive and statistically significant skill–accuracy relationship for debate-trained models (4% absolute improvement, p < 10⁻⁶), while consultancy-trained models exhibit no such trend. The paper also introduces two novel consultancy baseline variants (ensembled and double consultancy) to decompose where debate's advantage comes from, and finds evidence that debate training encourages more evidence-based argumentation while consultancy training leads to judge-specific exploitation strategies.

---

## Claims and Support

**Claim 1: Training to win debates via self-play improves judge accuracy.**
*Partially supported.* The within-setup empirical result is real and statistically significant. However, optimization, skill measurement (Elo/win rate), and final accuracy evaluation all use the same finetuned GPT-4T judge — creating an evaluator-coupling loop. The GPT-4o transfer result in Section 4.4 measures cross-judge *win-rate correlation*, not judge accuracy on the underlying task under an independent evaluator, so this does not fully decouple the result. The narrow claim is well-supported; the broader "truth-tracking" interpretation is weaker.

**Claim 2: This is the first demonstration that training LMs to win debates produces more accurate evaluator judgments.**
*Supported as a novel result.* The paper correctly distinguishes its training-time result from prior inference-time work (Khan et al., Kenton et al.) and from the failed training attempt in Radhakrishnan (2023). Framing as "to our knowledge" is appropriate and used appropriately (footnote 1 and introduction).

**Claim 3: No positive relationship exists between optimization pressure and judge accuracy for consultancy models.**
*Partially supported.* The narrow claim — that single-consultancy-trained models show no skill–accuracy trend — is supported. However, ensembled and double consultancy are *evaluation-only* protocols (explicitly stated in Section 2.3: "Ensembled and double consultancy are different evaluation methods, not training procedures"). The absence of a positive trend in these protocols cannot be strongly attributed to "consultancy as a paradigm"; it simply reflects training/evaluation mismatch. The paper is transparent about this, but the comparative conclusions sometimes outrun the evidence.

**Claim 4: Debate training encourages stronger and more informative arguments.**
*Partially supported.* The evidence is proxy-level: 96% more quoted words for debate, 70% fewer for consultancy (with high variance), 98% quote repetition in consultancy's second speech, and high cross-judge win-rate correlation for debate (0.98 vs. 0.51 for consultancy). These are informative but do not directly establish causal argument quality improvement. More quoting is not necessarily better evidence.

**Claim 5: Debate's advantage comes primarily from two-sided exposure and side-by-side comparison; explicit refutation does not materially help.**
*Supported as an empirical observation but mechanistically underdetermined.* The protocol comparison (single 68% → ensembled 72% → double 75% → debate 77%) and one-turn vs. two-turn parity (Appendix G) support the claim. However, the refutation conclusion could equally be explained by poor refutation capability in current 8B models or judge insensitivity, not the inherent unimportance of refutation. The paper acknowledges this ambiguity (Section 4.3: "either (a) the debaters are failing to engage in meaningful refutation, or (b) the judge does not benefit from reading the models' refutations").

**Claim 6: Gains occur without ground-truth supervision signal.**
*Overstated as written.* The self-play DPO stage does not use direct answer labels as debater training signal — this narrower claim is supported. But the full system relies on: (a) correct-answer labels in QuALITY for side assignment and accuracy measurement, (b) SFT on curated human/GPT-4 debate transcripts (Section 3.2.1), and (c) a judge finetuned on human correctness judgments (Section 3.1). The framing in the abstract that "gains occur without the requirement of a ground truth supervision signal" is misleading as a global characterization.

**Claim 7: DPO⁺ is effective for multi-turn debate training.**
*Supported as an engineering contribution.* The formulation is reasonable, performance exceeds SFT baseline and vanilla DPO. Concurrent work (Nvidia RPO) confirms the approach is principled. Not established as a strong standalone algorithmic contribution, but not overclaimed as one.

---

## Strengths

- **First training-time demonstration of the skill–accuracy relationship in debate.** Prior work (Radhakrishnan 2023) explicitly failed on this. Showing it holds with a properly calibrated judge and training-time optimization — not just inference-time prompting — is a genuine and important advance for the scalable oversight literature.

- **Calibrated judge construction addressing sycophancy.** The paper identifies a concrete failure mode (GPT-4T agreeing 72% with consultants out-of-the-box) and fixes it via finetuning, which makes the consultancy baseline meaningfully harder and the comparison more credible than prior work.

- **Novel ensembled and double consultancy baselines.** These are not cosmetic additions. They allow the paper to decompose the debate advantage into (a) seeing both sides, (b) seeing both sides in context, and (c) adversarial training effects — a much richer analysis than a single consultancy baseline permits.

- **Honest mechanistic reporting.** The paper explicitly acknowledges that explicit refutation does not appear to contribute, reports the result in the main text (not just appendix), and discusses alternative explanations rather than selectively reporting favorable results.

- **Behavioral policy analysis.** The evidence use and cross-judge transfer analysis provide concrete, falsifiable characterizations of what changes during training, going well beyond topline accuracy comparisons.

---

## Weaknesses

### Fatal
*None.* The core empirical finding is real and the paper's limitations, while significant, do not invalidate the contribution outright.

### Major

- **Evaluator coupling undermines the generality of the headline claim.** The finetuned GPT-4T judge is used to generate rewards during DPO training, compute win rates for skill measurement, and evaluate final judge accuracy. This closed loop means the paper demonstrates optimization against one judge yields transcripts that same judge answers more accurately — a meaningful but narrower result than "debate training improves truth-seeking oversight." The GPT-4o transfer experiment (Section 4.4) measures cross-judge *win-rate correlation*, not independent *task accuracy* evaluation. An independent judge evaluating task accuracy — even a separate GPT-4 variant not used in training — would substantially strengthen the main claim. This is the single largest gap between what is shown and what is claimed.

- **"Without ground-truth supervision signal" framing is materially overstated.** The abstract and introduction both highlight this property, but it applies only to the self-play DPO stage. The system depends on labeled correct answers for side assignments and accuracy measurement, human-judged training data for the judge, and supervised transcript data for SFT. For the scalable oversight framing to be compelling, this should be clarified: only the debater self-play optimization step avoids direct answer labels, and the significance of that should be stated precisely rather than globally.

- **No human judge evaluation despite human-oversight motivation.** The entire scalable oversight motivation concerns helping weaker supervisors — ultimately humans — evaluate capable AI. All evaluations use a finetuned GPT-4T model. It is unknown whether the skill–accuracy trend would hold for human judges, which is the setting where it matters most. Even a small-scale human study (100–200 judgments) on debate vs. double consultancy transcripts would substantially change the evidentiary picture.

### Minor

- **Consultancy comparison is not protocol-matched, limiting comparative causal claims.** Consultancy models are trained only to maximize single-consultancy reward. Double and ensembled consultancy are evaluation-only. The paper is transparent about this (Section 2.3), but the discussion in Sections 4.3 and 5.1 sometimes draws comparative causal conclusions (e.g., about the role of refutation, or the unique "truth-seeking" properties of adversarial training) that require matched training to support. Training a separate consultancy model for double-consultancy would cleanly isolate whether the skill-accuracy divergence is about training objective or protocol.

- **Mechanism for accuracy gains is potentially confounded by evidence quantity.** Debate models use 96% more quoted words after training. The paper does not test whether equalized quote budgets would close the accuracy gap between debate and double consultancy. If the accuracy gain is largely driven by evidence quantity rather than debate structure, the theoretical significance of the result changes. A simple control — giving consultants an equal quote budget, or giving the judge debate transcripts with rebuttal removed — would address this.

- **The 77% vs. 75% debate–double consultancy gap is the core adversarial benefit, and its statistical significance is never reported.** The entire argument that adversarial training provides unique value rests on this 2-point gap (and the policy differences in Section 4.4). Yet no significance test or confidence interval is reported for this comparison. Given the modest effect size, the gap could be consistent with zero.

- **Single domain and model family limits generalizability.** The paper honestly acknowledges (Section 5.2) that Kenton et al. (2024) failed to replicate debate benefits on non-reading-comprehension tasks. With experiments restricted to one task type (hidden-context reading comprehension), one base model (Llama3-8B), and one judge family (GPT-4T), the paper cannot resolve whether its positive result is domain-specific or broadly applicable.

### Trivial

- **Refutation irrelevance finding is buried in Appendix G rather than foregrounded.** Given that Irving et al. (2018)'s theoretical motivation for debate centers on refutation, this finding deserves prominent placement and extended discussion in the main text.

---

## Nice-to-Haves

- **Additional DPO iterations or larger base models** to test whether the positive skill–accuracy trend continues, plateaus, or reverses with further optimization. This is critical for the scalability thesis but may be compute-prohibitive.
- **Scatter plot of quote density vs. judge accuracy** across all checkpoints and conditions to directly visualize whether evidence quantity mediates the accuracy effect.
- **Paired qualitative case studies** comparing debate and double consultancy transcripts on the same question to give the reader intuition for when the 2% difference arises.
- **Significance test and confidence interval for the debate vs. double consultancy accuracy gap (77% vs. 75%).**

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**Removed: "No comparison to other scalable oversight methods (RRM, constitutional AI, etc.)"** — This is scope creep. The paper explicitly scopes to comparing debate with consultancy variants. Demanding comparison against all scalable oversight paradigms would apply to essentially any paper in this sub-field and is not a fair criterion.

**Removed (from Neutral): "Limited training scale (only 2 DPO iterations)" as a weakness** — The paper reports this honestly and notes it as a limitation. Demanding more compute-intensive experiments is not a criticism of what was done. Kept as a Nice-to-Have.

**Removed (from Harsh): "The claim that DPO⁺ outperforms vanilla DPO is not established as a distinct algorithmic contribution"** — The paper explicitly frames DPO⁺ as an engineering adaptation and flags concurrent identical work (Nvidia RPO). It does not claim algorithmic novelty beyond that framing.

**Removed (from Neutral/Spark): "Risk that debate-trained models could learn judge-specific exploits with more training"** — Speculative; not a weakness of the current paper.

**Removed (from Harsh): "Win rate defined by training judge blurs 'becoming more persuasive' vs. 'becoming a better debater in a truth-tracking sense'"** — This is partially captured in the judge coupling weakness, but the framing as a separate standalone criticism is somewhat circular. The paper explicitly defines "skill" as win rate and never conflates it with truth-tracking independently of judge accuracy. Merged into the evaluator coupling weakness above.

---

## Novel Insights

The most genuinely novel contribution of this paper is the decomposition of debate's advantage into three separable components via the consultancy baseline suite: (1) seeing both sides increases asymmetric-evidence resolution, (2) seeing both sides in one context enables direct comparison, and (3) adversarial training discourages judge-exploiting strategies. This decomposition — supported by the policy analysis showing consultancy devolves into repetitive judge exploitation while debate increases evidence use — provides the most original mechanistic insight into *why* debate might work as an oversight protocol. The finding that explicit refutation appears not to contribute, while initially disappointing, is itself scientifically valuable and points toward the relative importance of presentation format and training incentive structure over the adversarial interaction mechanism that theoretically motivated the debate paradigm.

---

## Suggestions

1. **Evaluate judge accuracy using at least one independent judge** (e.g., a GPT-4 variant not used in training, or a model from a different family) on the same transcripts, and report the skill–accuracy trend under that independent evaluator. This is the single most impactful experiment for validating the headline claim.

2. **Narrow the "no ground-truth supervision signal" language** to: "the self-play DPO stage does not use direct answer labels as a training signal for debaters," and add a sentence explaining what supervised signals do exist elsewhere in the pipeline.

3. **Report statistical significance for the debate vs. double consultancy gap (77% vs. 75%)** and discuss what the confidence interval implies for the causal role of adversarial training.

4. **Move the one-turn vs. two-turn finding (Appendix G) into the main text** and expand the discussion of what this implies for the theoretical mechanism of debate. This is a substantive finding about refutation's role, not supplementary material.

5. **Add an evidence-quantity control**: Run a condition where consultants are given the same word budget for quotes as debate models, or evaluate debate transcripts with rebuttal turns masked, to test whether quote volume rather than debate structure drives accuracy gains.

6. **Conduct even a small-scale human judge study** (100–200 judgments) on a stratified sample of debate vs. double consultancy transcripts to provide at least preliminary evidence relevant to the core scalable oversight motivation.

---

## Score and Decision

**Originality:** Moderate-high. The first training-time demonstration of a positive skill–accuracy relationship in debate is genuinely novel. The consultancy decomposition and calibrated judge construction are non-obvious methodological contributions.

**Importance:** High in principle. Validating debate as a scalable oversight mechanism is consequential. However, the narrow domain and unresolved judge coupling question limit how much this paper advances the broader program.

**Claims vs. support:** Mixed. The core empirical finding is real. Several interpretive and framing claims (ground-truth-free oversight, mechanism attribution, causal superiority of debate training) outrun the evidence.

**Soundness:** Moderate. The training procedure and evaluation are carefully executed within their scope. The evaluator coupling is a genuine methodological gap, not a procedural error. The consultancy mismatch is acknowledged but the conclusions drawn from it are sometimes too strong.

**Clarity:** Good. The paper is well-organized, and the limitations section is notably honest for the sub-field.

**Value to community:** Positive. Despite its limitations, this is an important data point for the debate-as-scalable-oversight literature, and the consultancy baseline variants will likely be adopted by follow-up work.

Overall, this is a meaningful incremental contribution with a real finding, but the generality of the conclusions is constrained by the single-domain, single-judge-family setup and the evaluator coupling concern. It falls solidly in borderline territory: publishable in its contribution, but with gaps that limit the scope of its claims.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>