Now I have enough information from the paper. Let me write the final consolidated review.

---

## Summary

This paper investigates whether **training** language models to win debates via self-play improves a judge's ability to identify correct answers, using an information-asymmetric reading comprehension setup (QuALITY-HARD). The core finding is that debate training yields a statistically significant 4% absolute improvement in judge accuracy, while consultancy training shows no such relationship. The authors develop a modified DPO objective using continuous judge confidence as reward, introduce two novel consultancy baselines (ensembled and double consultancy) that help decompose the sources of debate's advantage, and present behavioral evidence that debate training drives more evidence-based argumentation while consultancy training encourages judge-exploitation.

---

## Strengths

- **First training-based demonstration that debate improves judge accuracy.** Prior work (Radhakrishnan, 2023) failed to find this effect. The paper cleanly distinguishes inference-time optimization from training-time optimization and closes that gap empirically. The p < 10⁻⁶ result on 433 questions is at least statistically credible within its setting.

- **Decomposition of debate's advantage via novel consultancy baselines.** The introduction of ensembled (72%) and double (75%) consultancy baselines is a genuinely useful methodological contribution. The triplet—single (68%), ensembled (72%), double (75%), debate (77%)—enables a principled decomposition into asymmetric evidence, side-by-side comparison, and adversarial training effects. This kind of structured ablation is rare in the scalable oversight literature.

- **Modified DPO with continuous reward signal.** Converting judge confidence into soft preference targets via the Bradley-Terry model (DPO⁺) is a concrete and implementable technical contribution that outperforms both standard DPO (71% win rate) and the SFT baseline (31%). The design rationale is clearly articulated.

- **Honest reporting of the refutation non-finding.** The paper explicitly notes that single-turn debates match two-turn debates (Appendix G) and that double consultancy nearly closes the gap to debate, directly contradicting the canonical Irving et al. (2018) motivation based on adversarial refutation. Reporting this honestly, and retheorizing debate's benefit around information exposure and anti-exploitation, is intellectually honest and improves the field's understanding.

- **GPT-4o transfer analysis as partial cross-validation.** The Pearson correlation of 0.98 (debate) vs 0.51 (consultancy) between trained GPT-4T and untrained GPT-4o win rates provides suggestive evidence that debate training learns more judge-general strategies, partially mitigating the evaluator-coupling concern.

---

## Weaknesses

### Fatal
None. The methodological concerns below are significant but do not fully invalidate the empirical signal.

### Major

- **Judge-evaluator coupling undermines the headline claim's generality.** The same finetuned GPT-4T judge both supplies the reward signal during DPO training (via rollout-based judge confidence) and serves as the evaluator for the main judge accuracy metric. This creates a non-trivial confound: training models against this judge produces transcripts that this judge finds compelling, and if that judge is also the evaluator, the reported accuracy gain may partly reflect transcripts being stylistically adapted to a specific evaluator rather than genuinely more truth-revealing debate. The paper notes the debaters are "not trained to maximize judge accuracy" and the judge was not finetuned on the current debaters' transcripts—which partially addresses the concern—but the coupling through the reward mechanism remains. The GPT-4o transfer analysis provides partial reassurance only for win rates, not for the main judge accuracy outcome. **What is needed:** judge accuracy evaluation under at least one held-out judge model that never supplied reward signals.

- **Information-asymmetry proxy does not match the scalable oversight setting.** The paper artificially creates expertise asymmetry by giving debaters access to text the judge cannot read. However, the judge (GPT-4T) has substantially stronger general reasoning capabilities than the debaters (Llama3-8B). In actual scalable oversight, we need to verify whether a *weaker* judge can supervise *stronger* debaters—the opposite configuration from this paper. The paper acknowledges this ("the debater's expertise comes from superior access to information") but does not fully confront the implications for scalability claims. A judge stronger than both debaters will succeed for reasons unrelated to debate's oversight-theoretic properties.

- **Self-play-only evaluation is too narrow for the oversight claim.** Judge accuracy is measured only on self-play transcripts (a model debating itself). This leaves open the possibility that training changes transcript style in ways that make self-vs-self interactions easier for this particular judge, without improving truth-revealing properties in more realistic cross-play settings (e.g., trained debater vs. untrained, or early vs. late checkpoint debates). The paper frames its contribution as validating debate as a scalable oversight method; self-play evaluation alone does not support that claim.

- **Consultancy optimization asymmetry confounds the key comparison.** Debate reaches ~67% win rate while consultancy reaches only ~52%. The "no positive skill-accuracy trend for consultancy" could partly reflect that the consultancy DPO training was simply weaker or less converged, rather than a fundamental protocol difference. The paper notes "whether due to differences in skill or distribution shift, the judge treats the initial SFT model's outputs with skepticism" (Sec. 4.1) but does not satisfactorily rule out that further consultancy training would begin to show a positive trend. This confound weakens the core debate-vs-consultancy comparison.

### Minor

- **Missing significance test for the debate vs. double-consultancy gap.** The central claim that debate uniquely benefits from adversarial format rests substantially on a 77% vs 75% accuracy difference. No significance test for this comparison is reported, despite it being directly policy-relevant. If this gap is not statistically distinguishable, the claim that adversarial training per se (beyond information exposure) matters collapses.

- **Mechanistic conclusions (evidence use, helpfulness) outrun the data.** Quote count is a weak proxy for argument quality—more quoting could reflect verbosity or quote-stuffing rather than evidentiary strength. The consultancy repetition finding is noted as "barely significant due to high variance" (Sec. 4.4). These are useful exploratory analyses but are presented with more confidence than their evidential basis warrants.

- **Limited scale:** All results use Llama3-8B as debater. Whether the positive skill-accuracy trend persists or degrades with stronger debater models (which are the models we ultimately need oversight for) is entirely open and directly pertinent to the paper's central motivation.

### Trivial

- The "without ground truth supervision signal" framing in the abstract slightly overstates independence from ground truth: the judge was finetuned on human-labeled transcripts, and the task itself requires labeled correct answers. The accurate claim is that *debater policy optimization* does not use direct answer labels.

---

## Nice-to-Haves

- A baseline where the judge is given a random sample of quotes matching the debate's total quote volume, which would clarify how much accuracy gain stems from intelligent quote selection vs. debate structure.
- Experiments on at least one reasoning-type task (math, code) to bound the scope of generalization, even if negative.
- A "consultant vs. consultant with cross-visibility" condition (double consultancy with models specifically trained for that format), which would more cleanly isolate adversarial training effects from information exposure effects.
- Bootstrap confidence intervals on all main accuracy comparisons, especially the 77% vs 75% debate-double-consultancy gap.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "No ground truth supervision" is a fundamental misrepresentation.** The paper specifies this refers to the debater training signal, which is accurate. The claim is slightly imprecise but not dishonest and the paper's body makes the nuance clear.
- **Harsh Critic: The modified DPO objective is not isolated from gains.** The paper's claim is about debate training, not DPO⁺ per se, and comparing against the SFT baseline is appropriate. This is not a fatal gap.
- **Spark Reviewer: Demand for task-domain experiments as missing for acceptance.** The paper explicitly scopes to reading comprehension and honestly cites evidence both for and against generalizability. Demanding domain coverage the paper explicitly scopes out is scope creep.
- **Harsh Critic: p-value is uninformative without clustering correction.** This is a reasonable statistical concern but rises to a minor precision issue, not a fundamental problem. Moved to minor rather than major.
- **Human Finder: Concern about "whether LLM judge perturbations can be trusted."** This is a generic concern about LLM-as-judge methodology not specific to this paper's claimed contribution.
- **Spark Reviewer: Computational cost and scalability.** Reasonable as a discussion point but not a weakness of the paper's claims; the paper does not claim its pipeline is computationally cheap.

---

## Novel Insights

The most genuinely novel observation—one that the paper surfaces but does not fully theorize—is the mechanism by which adversarial training prevents judge exploitation: the *presence of a competing argument at training time* acts as an implicit regularizer against cheap rhetorical strategies. A consultant learning to repeat quotes and assert without evidence is never penalized because there is no opponent to expose the weakness; a debater faces that opponent directly, and quote-stuffing is surfaced as empty if the opponent provides stronger-contextualized evidence. This suggests that the debate format's key contribution to scalable oversight may be less about truth discovery at inference time and more about **policy shaping at training time**—making the reward landscape for debate training more truth-correlated than for consultancy training, even with an imperfect judge. This reframing has implications beyond debate: any multi-agent training setup where one model can expose another's rhetorical weaknesses might have similar anti-exploitation properties.

---

## Suggestions

1. **Independent judge evaluation for the main accuracy metric**: Run judge accuracy under a model that never supplied reward during training (e.g., untrained GPT-4o or a separate judge family) and report this alongside the trained GPT-4T results.
2. **Cross-play accuracy matrix**: Evaluate judge accuracy not just on self-play but on pairings of different checkpoints (early vs. late training), to rule out style-adaptation to self-play transcripts.
3. **Significance test for debate vs. double consultancy**: Report a paired test on the 77% vs. 75% gap.
4. **Report consultant win rate with matched compute**: Run consultancy training to convergence with the same DPO budget, or explicitly show the convergence curve, before concluding there is "no positive trend."

---

## Evaluation

- **Novelty:** Moderate-high. Being the first to demonstrate training-based debate improvement is meaningful. The modified DPO and baseline decomposition are incremental but useful. The refutation non-finding is a genuinely surprising result.
- **Technical soundness:** Moderate. The DPO variant is well-motivated. The main methodological concerns (judge coupling, self-play evaluation, consultancy optimization asymmetry) are real and partially unaddressed.
- **Empirical support:** Moderate. The 4% gain is statistically significant and the analysis is mostly honest, but the narrow scope (one task, one debater size, self-play only) and evaluation confounds limit the strength of the evidence for the broader oversight claim.
- **Significance:** Moderate-high for the debate/oversight subfield; moderate for ICLR overall. The paper takes a real step forward but stops short of the evidence needed to strongly validate debate as a scalable oversight mechanism.
- **Clarity:** Good. The paper is honest about its limitations, the baselines are well-explained, and the analysis is structured clearly.

## Score and Decision

No prior reviews exist in this run, so I calibrate against ICLR training knowledge. This paper:
- Makes the first positive training-based result in a contested area of alignment research
- Is methodologically careful but has real and partially unresolved validity concerns
- Is limited to one domain and one scale
- Is honest about limitations and surprising findings

A strong accept at ICLR (8+) would require broader scope, independent evaluation, or a much larger effect. A clear reject (<5) would require fundamental invalidity. This paper occupies the borderline accept/weak-accept space: the contribution is real, the concerns are real, and the work is worth building on. I place it at **6.5**—above the ICLR mean, reflecting a meaningful and honest empirical contribution with acknowledged limitations that do not fully invalidate the core finding.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>