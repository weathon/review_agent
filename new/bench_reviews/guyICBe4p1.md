## Summary

This paper investigates how contextual information affects belief-direction probing in LLMs, adapting truth-value judgment (TVJ) tasks to evaluate whether probes capture prior, conditional, or marginal beliefs. The authors introduce four error scores (E1–E4) to quantify different consistency failures, propose a stable variant of CCS called CCR, and conduct a causal intervention experiment shifting premise representations along belief directions. The key findings are that belief probes are sensitive to premises (including irrelevant ones), and that belief directions causally mediate how hypotheses are positioned during in-context inference.

## Strengths

- **Formal error score framework (E1–E4, Table 1)**: The paper provides a principled, philosophically-grounded error taxonomy that distinguishes whether probes reflect prior, conditional, or marginal beliefs—a genuine advance over prior work that evaluated only probe accuracy.

- **Causal intervention experiment (Section 4.2, Figure 4)**: Demonstrates that shifting premise representations along belief directions causes entailed hypothesis probabilities to decrease and contradicted hypothesis probabilities to increase in the expected direction. This goes beyond correlational probing to show that belief directions are (partial) causal mediators of in-context inference.

- **Important negative finding: probes are sensitive to irrelevant context**: Table 2 shows E1 and E2 error scores near 1.0 for no-prem probes, indicating corrupted and unrelated premises affect probe outputs with approximately the same magnitude as relevant premises. This is a meaningful empirical result about probe fragility.

- **CCR probing method (Equation 2)**: The proposed Contrast Consistent Reflection objective eliminates the degenerate solution of CCS and achieves stable convergence without training multiple probes. This is a useful methodological contribution with a mathematically principled geometric constraint.

- **Careful experimental design**: The meta-linguistic "[in]correct" framing (Section 4) avoids presupposition failures from natural language negation, and the SNLI picture-A/B setup provides a clean way to construct genuinely neutral premise-hypothesis pairs.

- **Systematic evaluation**: Testing across four probing methods (LR, CCS, MMP, CCR + LM-head baseline), two model scales (7B/13B), and instruction-tuned vs. base variants establishes a useful empirical baseline for how different recovery techniques behave in context-rich settings.

## Weaknesses

### Major

- **PE-normalized error scores are noisy in low-sensitivity regimes, weakening quantitative claims**: The error scores E1–E4 are expressed as multiples of the Premise Effect (PE = p(h; q⁺) − p(h)), which is intentional to compare methods with different premise sensitivities (Section 3.3). However, for no-prem probes, PE is very small (~0.05–0.06 per Table 2), making the normalized ratios highly sensitive to small absolute fluctuations. An E1 score near 1.0 in this regime means the corrupted premise has a comparable *relative* effect to the relevant premise—but the *absolute* probability change is tiny (e.g., 0.49 vs. 0.50 in the p(h) column). The paper interprets high normalized scores as "belief directions are sensitive to irrelevant information" (Section 4.1), which is directionally correct, but the normalization inflates the perceived severity of the inconsistency. The qualitative claim holds; the quantitative severity comparison across methods and layers is less trustworthy than presented.

- **Causal intervention experiment overclaims belief mediation without verifying the premise belief shift**: Section 4.2 intervenes by shifting the premise representation q by |θ_mm| along the belief direction and measures the downstream effect on p(h). However, the paper does not verify that this intervention actually moved the premise's own probed belief p(q) toward the intended value. Without this verification, the observed ~0.1 probability shift in p(h) could partially result from general representational disruption or alignment with non-belief features (e.g., attention pattern changes) rather than from toggling the premise's truth representation. The fact that the directionality is correct (entailed decrease, contradicted increase) provides indirect support, but the causal mediation claim is stronger than what this single measurement justifies.

### Minor

- **Fixed intervention magnitude confounds layer-wise causal effect strength**: The intervention uses a constant Euclidean shift magnitude |θ_mm| across layers 8–14 (Section 4.2). As transformer activation norms vary significantly across depth, a fixed norm represents a proportionally larger perturbation in deeper layers. This means the increasing trend in Figure 4 could partially reflect geometric scaling rather than increasing causal mediation strength—a confound that complicates interpretation of the layer-wise results.

- **Linear probe limitations in assessing belief–context entanglement**: The paper concludes that "LLMs do not represent prior beliefs fully independently" and "belief directions are not orthogonal to context" (Section 4.1). However, in transformer models, attention naturally pools premise information into hypothesis representations, and a linear probe on concatenated premise-hypothesis representations will capture this pooling regardless of whether the model's truth-evaluation circuits are entangled or separate. The no-prem vs pos-prem comparison mitigates this somewhat (no-prem probes still show some sensitivity), and the intervention experiment provides additional support, but the specific claim about representational entanglement relies on an assumption about linear separability of the underlying beliefs.

### Trivial

- **Notation PE⁻¹ in Table 1**: The paper writes scores as multiplied by "PE⁻¹", which represents scalar division, not matrix inversion. This is slightly misleading notationally but does not affect the methodology.

## Nice-to-Have

- **Report p(q) before and after intervention**: Showing that the intervention actually shifts the premise's probed belief would substantially strengthen the causal mediation narrative.
  
- **Normalize intervention magnitude per layer**: Using a fixed standard-deviation shift per layer (relative to layer-wise activation norms) would remove the depth-dependent scaling confound.

- **Evaluate on a dataset without hypothesis-label shortcut**: While the paper's defense of SNLI (Section 4, SNLI data) is reasonable, running probes on a dataset without the Poliak et al. (2018) hypothesis-only leakage would more cleanly isolate true NLI sensitivity.

- **Integrate the 2D belief subspace structure**: The paper acknowledges Bürger et al. (2024) in the limitations but leaves it entirely as future work. Even a brief analysis of how much measured error might stem from polarity-axis misalignment rather than truth-judgment failure would strengthen the evaluation.

- **Statistical significance for Table 2**: Given noted variance (especially for CCS), bootstrapped or confidence-interval estimates for error scores would help support the fine-grained claims about relative method performance.

## Removed Points

> These points are flagged for removal — treat them with caution:

- **"Variance calibration is under-justified and potentially harmful" (Section 3.1 note)**: The paper explicitly states the calibration is "to make the results from different probing methods comparable." The cross-method comparison purpose is reasonable, and while the interaction with PE normalization is a consideration, it is not "potentially harmful" in the way described. *Moved to minor scope: the calibration purpose is clear.*

- **"SNLI hypothesis-only leakage response is insufficient" (Section 3.2 note)**: The paper directly addresses this in Section 4 (SNLI data), arguing that if a probe trained only on hypotheses still responds coherently to premise presence, it must be capturing something beyond spurious hypothesis statistics. This is a reasonable theoretical defense. The suggestion to test on a dataset without shortcuts is moved to Nice-to-Haves.

- **"Max{·, 0} in E3 discards information and biases the metric" (Section 3.3 note)**: The paper explains this design choice explicitly: "we can isolate those cases where the numerator and denominator have the same sign, which are the errors we want to capture in E3." The opposing cases are captured by E4, which the paper notes: "Because E3 and E4 measure deviations for two different types of beliefs, they are opposing." The critic's concern misunderstands the paper's own explanation.

- **"Paper treats 2D belief subspace as a post-hoc excuse rather than integrating it" (Section 5 note)**: The paper cites Bürger et al. (2024) transparently in the Limitations section and clearly states this as future work. The paper was written before the 2D finding was fully established; using it to explain limitations is appropriate scientific practice, not a "post-hoc excuse."

- **"The linear probe confound is a structural flaw" (Criticism 3)**: This is partially addressed by the no-prem/pos-prem comparison and by the causal intervention in Section 4.2. The concern is retained as a minor weakness rather than removed entirely.

## Novel Insights

The paper makes a genuinely useful contribution to the belief-probing literature by demonstrating that *the context-sensitivity of belief directions is not clean* — probes designed to detect prior truth-judgments remain sensitive to irrelevant and corrupted context, and this sensitivity pattern varies systematically across layers and model types. The intervention experiment further extends the line of work initiated by Marks & Tegmark (2023) by targeting premise-level belief representations rather than isolated-sentence representations and showing downstream effects on related hypotheses. The CCR variant addresses a practical convergence problem in CCS that has plagued downstream analyses in the probing literature. Together, these results refine our understanding of belief-direction probes from tools that cleanly extract "the model's knowledge" to methods that reveal context-embedded representational structures with nuanced failure modes.

## Suggestions

1. **Address the PE normalization concern in the paper text**: Acknowledge that E1/E2 scores near 1.0 for no-prem probes reflect relative (not absolute) inconsistency, and report the raw absolute probability differences alongside normalized scores so readers can assess the actual effect magnitude.

2. **Add a p(q) verification for the intervention**: Report the premise's own probed belief before and after intervention to confirm the manipulation successfully shifted it along the intended axis. This would strengthen the causal claim significantly.

3. **Clarify the layer-wise intervention scaling**: Either normalize the intervention magnitude per layer or add a discussion of how activation norm variation across depth might affect the observed trend.

4. **Move Figure 4 description to the main text more prominently**: The key finding about causal mediation deserves more emphasis in the Results section, not just "we can see the effect."

## Score and Decision

I calibrated this paper against several anchors:

- **High-scoring interpretability papers**: WCRQFlji2q (scores 8,10,8,10) — clear causal evidence linking discovered mechanisms to downstream behavior with thorough experimental grounding. This paper under review lacks the same level of mechanistic depth and causal verification, so it falls below this tier.

- **Moderate-scoring interpretability papers**: jE8xbmvFin / "Language Models Represent Space and Time" (scores 8,5,6,8; avg ~6.75) — solid empirical findings with overclaimed scope ("world model") noted by reviewers. This paper is comparable: solid experiments with partially over-supported causal claims. af2ztLTFqe / TA-ITI (scores 5,6,5,6,8; avg ~6.0) — incremental but technically sound intervention work; similar in being empirically solid but with limited verification depth.

- **Low-scoring papers**: InWaCoIMMN (all 3s) — fundamental methodological gaps preventing useful insights; this paper is clearly above this tier as its methods are sound and findings are meaningful.

The paper under review sits firmly in the 6 range. The methodological concerns (PE normalization noise, incomplete intervention verification) are real but do not invalidate the qualitative findings — the core claims about context sensitivity and causal mediation survive even when the strongest quantifications are weakened. This is comparable to the af2ztLTFqe and jE8xbmvFin borderline anchors. The paper provides useful empirical results for the probing community, with identified weaknesses that the community will need to account for when interpreting belief-direction probes.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>