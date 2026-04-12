=== CALIBRATION EXAMPLE 32 ===

# Final Consolidated Review
## Summary
This paper introduces **HUBBLE**, a fully open-source suite of pretrained LLMs designed specifically for controlled study of memorization in pretraining. The core contribution is not just model release, but a perturbation framework spanning copyright, privacy, and benchmark contamination, with known duplication counts and insertion timing, enabling causal analyses that are hard to perform on existing pretrained models. The paper’s main empirical findings are that memorization weakens when sensitive data is diluted in a larger corpus, and that data seen only early in training can later be forgotten; it also demonstrates HUBBLE as a benchmark for membership inference and unlearning.

## Strengths
- **A genuinely useful controlled-pretraining artifact, not merely another checkpoint release.** The paper creates standard and perturbed models with explicit randomized insertions at duplication levels \{0,1,4,16,64,256\}, after decontaminating the base corpus (§3.1, App. A.3). This gives researchers ground-truth membership and dosage information across multiple realistic risk domains, which is unusually valuable for causal memorization studies.
- **The perturbation design is unusually broad and policy-relevant.** Rather than studying only synthetic canaries or a single domain, the paper covers book/news passages, paraphrases, biographies, chats, and contaminated benchmarks across copyright, privacy, and evaluation contamination (§2). This breadth makes the suite much more useful than narrowly scoped memorization testbeds.
- **The dilution result is empirically well supported within the paper’s setup.** The core 2×2×2 design over model size, corpus size, and perturbation condition is a strong experimental scaffold (§3.2). Across many tasks, the 500B-token runs show weaker memorization at a fixed duplicate count than the 100B-token runs (§4, Fig. 2, App. plots), which is a meaningful empirical regularity even if its mechanistic interpretation is not fully pinned down.
- **The paper is commendably transparent about where benchmark design breaks.** A notable strength is that the authors explicitly identify ELLie as invalid for dilution analysis because minimal-pair structure leaks across duplication bins: “**This invalidates the use of ELLie for studying dilution**” (§D.3). Likewise, they show that metric choice materially changes copyright conclusions (§D.1, Fig. 4). This kind of self-audit increases trust in the resource.
- **The MIA benchmark construction appears sounder than prior confounded setups.** The paper directly motivates HUBBLEMIA by avoiding temporal or other spurious splits and instead using randomized insertions with true non-members at duplication 0 (§6.1). This is a concrete methodological contribution, even if some resulting empirical trends are consistent with prior work.
- **The unlearning benchmark is well chosen to test specificity rather than only gross forgetting.** Splitting duplicated examples into Unlearn and Keep subsets from the same distribution (§6.2) forces methods to erase targeted items without simply erasing nearby content. That design is stronger than many coarse unlearning evaluations.
- **Several domain-specific negative results are scientifically interesting.** In particular, the paper shows that contamination does not reliably improve generalization, that WinoGrande contamination can hurt performance on paired unseen examples, and that format mismatch can negate or reverse contamination benefits (§5, §D.3). These are more interesting than a simplistic “more contamination raises benchmark accuracy” story.

## Weaknesses
###: Fatal
- None.

### Major:
- **The “ordering” claim is overstated as a general best practice relative to the evidence provided.** The paper shows that if perturbations are inserted only in early phases of training and then never seen again, they can be forgotten (§4; Fig. 13–14). This is a real empirical result in the authors’ setup, and the harsh review is wrong to call it invalid. However, elevating this to a broad practitioner recommendation—“**sensitive data can be ordered to appear early in training**” (§4)—goes beyond what is demonstrated. The timing experiments are only on **1B models trained on 100B tokens** (§3.2), and they depend on a training regime where sensitive examples are present in restricted windows rather than throughout the shuffled training stream. So the phenomenon is credible, but the “best practice” framing should be narrowed and qualified.
- **The interference claim is supported only at a coarse level and only for a limited setting.** The paper argues that perturbations from different domains “minimally interfere” because a 1B/100B joint perturbed model matches single-domain models on corresponding domain evaluations (§4, Fig. 20). This is useful evidence, but it is not strong enough to justify a broad claim of minimal interference for the suite as a whole. It does not test the larger 8B or 500B settings, nor does it assess finer-grained per-example or representational interference. Since the suite’s utility partly rests on treating jointly trained perturbations as separable probes, this limitation matters.
- **The dilution conclusion is empirically compelling but not fully disentangled mechanistically.** The paper interprets weaker memorization in the 500B runs as showing that “memorization risks are determined by the frequency of sensitive data relative to size of the training corpus” (Abstract) and recommends dilution as a best practice. The empirical trend is clearly shown, but the design changes multiple coupled quantities at once: relative frequency, spacing between repeats, and exposure rate per total training horizon. The paper does not isolate whether the effect is fundamentally about corpus-scale density versus optimization dynamics induced by rarer replay over more tokens. This does not invalidate the result, but it weakens the strength of the causal interpretation.
- **Some of the paper’s strongest-use-case sections are more benchmark instantiations than new scientific findings.** The MIA and unlearning sections convincingly demonstrate that HUBBLE is a useful testbed (§6), but the concrete findings themselves—e.g., MIAs work much better at high duplicate counts than at single-duplicate counts; current unlearning methods struggle to erase targets without collateral damage—are not, by themselves, major new scientific discoveries. The main novelty here is the benchmark/resource quality and cleaner experimental control, not a fundamentally new conclusion about MIA or unlearning behavior. The paper would be stronger if it framed these sections more explicitly as validation of HUBBLE’s utility.

### Minor
- **External validity to frontier-scale models remains limited.** The suite tops out at 8B parameters and 500B tokens (§3), and the authors themselves note the gap to much larger commercial models. This does not negate the contribution—the paper is explicitly about an academic-scale open suite—but it does limit how confidently one should transfer quantitative memorization thresholds or best practices to frontier systems.
- **The privacy perturbations are only partially realistic.** The paper wisely includes both synthetic YAGO biographies and natural ECtHR cases (§2.2), but many privacy analyses rely heavily on the templated YAGO biographies. Those are useful for control and attack standardization, yet real-world PII leakage is often messier, more fragmented, and less templated. The conclusions about privacy leakage mechanisms should therefore be framed as controlled evidence rather than a full characterization of web-scale PII risk.
- **The paper does not deeply analyze which properties of perturbations drive memorization differences.** It spans many data types, but there is limited systematic disentangling of whether length, templaticity, ambiguity, format, semantic uniqueness, or contextual predictability are the main drivers of memorization thresholds. This is understandable given the paper’s breadth, but it leaves scientific insight on the table.
- **The unlearning study, while useful as a benchmark demonstration, remains somewhat narrow.** Only three methods are evaluated (§6.2), and although the appendix includes a grid search (Table 13), the paper does not show full Pareto frontiers or more diagnostic analysis of why Keep-set degradation occurs. This is enough to establish the benchmark’s challenge level, but not enough to make strong comparative claims about the unlearning landscape.

### Trivial
- **Claims about no capability degradation should be stated with more caution.** The paper says it finds “no degradation in the perturbed models” (§3.1), but some task-specific perturbation effects and benchmark-specific degradations are visible elsewhere, especially in contaminated-task settings (§5, §D.3). This is mostly a matter of careful phrasing: the models do not show broad catastrophic degradation, but “no degradation” is too absolute.

## Nice-to-Haves
- Validate the timing/ordering effect on at least one 8B model to show it is not only a 1B phenomenon.
- Add an ablation better separating dilution-as-relative-frequency from dilution-as-changed-replay/optimization schedule.
- Extend interference analysis beyond domain-average outcomes, e.g., with per-example comparisons or representational divergence measures.
- Show hyperparameter trade-off frontiers for unlearning methods rather than representative points only.
- Provide more synthesis across perturbation properties to explain why some data types memorize much more readily than others.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Reproducibility criticism about the GPT-4.1-mini paraphrase dependency lacking enough detail.** The paper already states the model used, the general generation procedure, fallback model, and validation process in Appendix E.2. More detail could help, but this is not a substantive weakness by ICLR standards.
- **Claim that the ordering result is “invalidated” because it is incompatible with standard pretraining pipelines.** This is too strong and not supported by the paper. The experiments do show forgetting under restricted-time exposure. The right criticism is overgeneralization, not invalidity.
- **Criticism that the paper should compare its recommended mitigation strategies against differential privacy, regularization, etc.** This is largely scope creep. The paper studies memorization dynamics and provides empirical practices within that framing; it is not positioned as a head-to-head mitigation-method paper.
- **Requests for exact insertion seeds/scheduling scripts or full statistical significance testing everywhere.** These are not core scientific flaws here and go beyond normal expectations for this kind of systems/benchmark paper.
- **Generic strength statements such as “the paper is well-written” or “the experiments are extensive.”** These were removed in favor of specific strengths.

## Novel Insights
The most interesting synthesis across the paper is that HUBBLE’s real value is not primarily any single empirical claim, but the combination of (i) randomized, dosage-controlled pretraining perturbations, (ii) cross-domain risk coverage, and (iii) explicit self-auditing of benchmark failure modes. That combination makes the suite unusually suitable for studying where memorization behaves unlike naive intuition: contamination can fail to improve generalization, format-specific contamination may not transfer across formats, and stronger semantic training signal (e.g., paraphrased biographies) can sometimes make certain attacks easier even when verbatim memorization is weaker. In other words, the paper suggests that “memorization risk” is not a single scalar property but an interaction among duplication, timing, format, and extraction metric—and HUBBLE is valuable precisely because it exposes those interactions under controlled conditions.

## Suggestions
- Narrow the headline framing of **ordering** from a broadly actionable best practice to a more carefully scoped empirical observation about finite-horizon pretraining with restricted exposure windows.
- Soften the claim of **minimal interference** and explicitly state that the current validation is domain-level and only on 1B/100B models.
- Add one targeted ablation that better isolates why the 500B runs memorize less at fixed duplicate count.
- Reframe the MIA and unlearning sections as **benchmark validation plus illustrative findings**, rather than as major novel scientific conclusions on those topics.
- Expand discussion of external validity: which conclusions likely transfer to larger models, and which should be treated as hypotheses enabled by HUBBLE rather than settled facts.


# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
