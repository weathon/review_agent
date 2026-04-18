Now I have sufficient context to synthesize the final review. Let me carefully evaluate all points against the paper.

## Summary

This paper investigates whether training language models to win debates via self-play improves judge accuracy as a scalable oversight method. Using an information-asymmetric reading comprehension setup (QuALITY-HARD), the authors train Llama3-8B debaters and consultants using a modified DPO objective with reward-aware preferences from a calibrated GPT-4-Turbo judge. They find a positive relationship between debater skill (win rate) and judge accuracy (4% absolute increase, p < 10⁻⁶), but no such relationship for consultancy baselines, including two novel baselines (ensembled and double consultancy) that help isolate why debate helps.

## Strengths

- **First demonstration that debate *training* improves judge accuracy.** Prior work by Radhakrishnan (2023) failed to show this relationship with trained debaters; the current paper succeeds, providing the first evidence that optimizing debaters via self-play DPO yields a positive skill-accuracy trend—a meaningful milestone for scalable oversight research. The result is statistically significant (p < 10⁻⁶) and replicable.

- **Thoughtful baseline design isolates mechanisms.** The introduction of ensembled consultancy (averaging two single-consultancy evaluations) and double consultancy (presenting both sides' arguments without adversarial interaction) cleanly decomposes the sources of debate's advantage: ensembled consultancy shows that seeing both sides helps; double consultancy shows that side-by-side comparison helps more; and the residual gap with debate (75% → 77%) quantifies the contribution of adversarial interaction at training time. This is a genuine methodological contribution.

- **Well-motivated technical contribution in DPO variant.** The reward-aware preference optimization (effectively weighted DPO using continuous judge confidence rather than binary preferences) is a natural and appropriate adaptation for the debate setting, cleanly formalized with the Bradley–Terry model. The concurrent discovery of essentially the same approach (Nvidia et al., 2024) validates the formulation.

- **Careful judge calibration and anti-sycophancy efforts.** The paper identifies and addresses two key issues with using GPT-4-Turbo as a judge—poor calibration and sycophantic bias (72% agreement with consultants pre-training)—by finetuning on human judgment data. Figure 3 demonstrates meaningful improvements in both calibration and accuracy.

- **Insightful analysis of learned policies (Section 4.4).** The finding that debate models increase quote usage by 96% while consultancy models decrease by 70%, combined with the differential transfer to an untrained GPT-4o judge (r=0.98 for debate vs. r=0.51 for consultancy), provides concrete mechanistic evidence that debate training encourages more genuinely informative argumentation rather than judge-specific exploit.

## Weaknesses

### Fatal
None.

### Major

- **Circularity in the core skill-accuracy relationship.** The paper's central claim—that better debaters yield more accurate judgments—relies on measuring both "skill" (win rate under the finetuned GPT-4T judge) and "judge accuracy" (how often that same judge picks the correct answer) using the same model. Since the debaters are explicitly trained to optimize win probability under this judge, the positive correlation between skill and accuracy is partially tautological: as models become more optimized for what the judge finds convincing, they also produce transcripts where the judge more often agrees with ground truth. The GPT-4o transfer analysis (Section 4.4) partially addresses this by showing that debater strategies transfer across judges (r=0.98), but this only shows *persuasiveness* transfers, not that the judge accuracy improvement is robust to evaluator change. A stronger test would measure judge accuracy under an independent evaluator (e.g., a different model or human raters). Without this decoupling, the headline "better debaters yield more accurate judgments" conflates genuine truth-seeking with alignment to a particular judge's decision boundaries. The paper's scalable oversight framing makes this circularity more consequential than it would be in a purely empirical paper.

- **Single domain and model scale significantly limits generality.** All experiments use one task (QuALITY-HARD reading comprehension), one model size (Llama3-8B), and one judge (finetuned GPT-4-Turbo). The paper itself acknowledges that Kenton et al. (2024) found debate less helpful for non-reading-comprehension tasks, and the information-asymmetry setup—where debaters have privileged access to text the judge cannot see—may be especially favorable for debate. The paper's claims about debate being "well suited for supervising more sophisticated models" (Section 6) are not supported by evidence beyond this single, specific configuration. Whether the positive skill-accuracy trend persists with larger models, different task types, or reasoning-based expertise gaps remains an open question that the paper's framing does not adequately caveat.

### Minor

- **The consultancy comparison is partially unfair due to training-evaluation mismatch.** Consultancy models are trained to optimize single-consultancy reward but evaluated under ensembled and double consultancy protocols. Since debate models are trained in the multi-agent setting in which they are evaluated, this asymmetry means the consultancy baselines never receive training signal about how their arguments appear when juxtaposed with counterarguments. The paper is transparent about this, and double consultancy still provides a meaningful upper bound, but the "no positive trend" result for consultancy should be interpreted with this mismatch in mind.

- **Refutation's role is empirically undermined but understatedly framed.** The findings that one-turn debates (without refutation) are judged as accurately as two-turn debates, and that double consultancy nearly matches debate (75% vs. 77%), significantly weaken the theoretical motivation for debate as originally proposed by Irving et al. (2018). The paper's discussion (Section 5.1) acknowledges this, but the framing still emphasizes debate's "unique properties"—when the data suggests the primary mechanism is simply presenting both sides' evidence simultaneously, not adversarial refutation. This is not fatal to the contribution (the baseline analysis itself is valuable), but the narrative should more directly grapple with the implication that debate per se may be less important than information symmetry.

- **Modest effect size with uncertain scaling trajectory.** The 4% absolute improvement (approximately 73% → 77%) is statistically significant but modest in practical terms, and Figure 5 (left panel) shows judge accuracy plateauing in later training epochs rather than clearly increasing. Whether further optimization would yield continued gains is unclear, making the claim that "further optimization should yield more accurate outcomes" (Section 1) moderately speculative.

- **Only two DPO iterations with no error bars or seed variation.** The entire training pipeline runs for just two DPO iterations with no reported variance across random seeds. This makes it difficult to assess the robustness of the observed trends and whether the positive skill-accuracy relationship is stable or could shift with different initializations.

### Trivial
- The phrase "another crucial step" in the abstract overstates the contribution of what is an initial proof-of-concept in a constrained setting.

## Nice-to-Haves

- Evaluate judge accuracy under an independent model or human raters to decouple the skill-accuracy circularity.
- Test on at least one additional task domain (e.g., reasoning or code verification) to assess generalizability beyond reading comprehension.
- Control for quote quantity to determine whether the accuracy improvement reflects more evidence or better argumentation.
- Include a weaker judge condition to test whether debate advantages scale with the expertise gap between debaters and judges—the setting most relevant to future scalable oversight.
- Run multiple random seeds and report variance/error bars on the main results.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Judge accuracy measured against ground truth, not human standard."** The harsh critic argues all results are "just" ground-truth MCQ accuracy. However, measuring against ground truth is the standard evaluation protocol for this benchmark and task. Human evaluation of new transcripts would strengthen the paper, but the absence of fresh human evaluation is a limitation (noted above in the circularity concern), not a fatal flaw that invalidates the results. MCQ ground truth is an appropriate and objective correctness measure for this task.

- **"Consultancy baselines are weakened because the judge was trained to resist sycophancy."** This inverts the logic. A sycophantic judge would make consultancy artificially easy to exploit, inflating consultancy's apparent accuracy. Training the judge to resist sycophancy actually makes the *consultancy* evaluation more challenging and the comparison *fairer*, not less fair. The harsh critic's framing incorrectly treats anti-sycophancy judge training as stacking the deck against consultancy, when in fact it removes a confound that would otherwise make consultancy look better than it is.

- **"Overclaimed safety/alignment interpretations."** While the paper's framing in terms of scalable oversight is clearly motivated, the paper also contains meaningful caveats in Section 5.2 about the limitations of reading comprehension tasks, the possibility of obfuscated arguments, and the narrow scope. The claims about debate being "well suited for supervising more sophisticated models" are somewhat overreaching but are presented as suggestive rather than definitive. The main concern (single domain, circular evaluation) is captured in the Major weaknesses above.

- **"Demand for larger model architectures or more DPO iterations."** These are reasonable suggestions for future work but are beyond the scope of a single paper. The 8B model and two DPO iterations are sufficient for an initial demonstration; scaling up is clearly important follow-up work.

- **"Missing ablation for controlling quote quantity."** This is a valid experiment that would strengthen the paper, but the differential quote usage (debaters increase 96%, consultants decrease 70%) is itself an important *result* showing that debate training encourages more evidence-based argumentation. Treating quote quantity as purely a confound misses that it's part of the mechanism the paper identifies.

## Novel Insights

The paper's most valuable insight is the decomposition of debate's advantage via the ensembled/double consultancy baselines. Rather than treating debate as a monolithic intervention, the paper shows that the majority of debate's accuracy gain over single consultancy comes from (1) seeing both sides' evidence (ensembled consultancy captures part of this), (2) presenting both sides simultaneously for direct comparison (double consultancy captures most of the remaining gain), and (3) adversarial training-time pressure preventing exploitative strategies (the residual 2% gap between double consultancy and debate). This decomposition reveals that the mechanism behind debate's success in this setting is primarily information-symmetry at evaluation time rather than refutation—the latter being the mechanism most theorized in the original debate proposal. This finding, while potentially disappointing for debate proponents, is an honest and important empirical contribution.

## Suggestions

- Decouple skill measurement from accuracy measurement by evaluating checkpoints under a different judge (e.g., untrained GPT-4o or a separate Llama-based judge) and report the cross-judge accuracy trends as a complementary analysis.
- Directly test whether quote-quantity-matched debate transcripts yield higher accuracy than consultancy transcripts to separate the "more evidence" from "better argumentation" mechanisms.
- Add a brief human evaluation (even on just 50-100 transcripts) to validate that the finetuned judge's accuracy trends align with human judgments on model-generated outputs.

## Evaluation Summary

**Originality:** The paper makes a genuine contribution as the first to show that *training* debate models (as opposed to inference-time optimization) improves judge accuracy. The baseline design (ensembled and double consultancy) and the modified DPO formulation are novel and well-motivated. The finding that refutation is not the primary mechanism is an important empirical contribution, even if it complicates the narrative.

**Research question importance:** Scalable oversight is an important problem for AI safety. Validating whether debate training can produce truth-seeking behavior is a significant research direction. The narrow empirical scope limits but does not negate the importance.

**Claim support:** The core empirical result (positive skill-accuracy trend for debate, flat for consultancy) is supported, but the circularity of skill and accuracy measurements under the same judge weakens the inference that this reflects genuine truth-seeking. The double-consultancy near-parity result is honestly reported but somewhat at odds with the paper's pro-debate framing.

**Experimental soundness:** Experiments are well-designed within their scope. The main limitations are single domain, single model scale, and circular evaluation. The analysis of learned policies (quote usage, GPT-4o transfer) goes beyond bare accuracy numbers and provides useful mechanistic insight.

**Clarity:** The paper is well-written, clearly structured, and transparent about limitations. The baseline definitions are precise and the training procedure is described in sufficient detail.

**Community value:** The baseline design and DPO variant will be useful for follow-up work. The finding that refutation may not be the key mechanism is an important negative result for the community.

## Score and Decision Calibration

Comparing against calibration papers:
- **tCfvktlrHI** (Self-Play in Non-Zero-Sum Games): Scores 3-6, avg ~4.75, rejected. Similar concerns about single task/model and limited training iterations, but that paper had weaker methodology and more overclaiming.
- **49ZYkhEGmv** (Doubly-Efficient Debate): Scores 6-8, avg ~6.5, rejected. Similar topic area (debate for oversight), but theoretical rather than empirical.
- **dcjtMYkpXx** (Reward Model Ensembles): Scores 6-8, avg ~6.5, accepted poster. More systematic empirical evaluation, similar concerns about overoptimization but better addressed.
- **OUkZXbbwQr** (Debate-based rewards for decision-making): Scores 6-8, avg ~7, accepted poster. Smaller scope but clean execution.

This paper has a meaningful empirical contribution (first demonstration that debate training improves judge accuracy) with thoughtful baseline design, but is limited by the circularity of skill-accuracy measurement, single-domain evaluation, and the somewhat awkward finding that refutation doesn't appear to be the key mechanism. It falls in a similar quality range to the accepted poster papers (6-7 range) but would benefit from addressing the circularity concern. The contribution is real but narrower than claimed.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Reject</orange>