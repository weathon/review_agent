## Summary
This paper proposes MARINE, a training-free inference-time method for reducing object hallucination in LVLMs by extracting object cues from external vision models (DETR and RAM++) and using them as textual guidance in a classifier-free-guidance-style decoding scheme. Across five LVLMs and standard object-hallucination benchmarks (CHAIR and POPE), the empirical results are generally strong, showing sizable reductions in hallucination and favorable latency relative to several prior mitigation methods.

## Strengths
- **Addresses an important practical problem with a simple, deployable idea.** Object hallucination is a major failure mode in LVLMs, and MARINE tackles it without retraining the target LVLM. The basic recipe—use image-grounded detectors/taggers to provide object-level guidance, then combine guided and unguided logits at inference time—is easy to understand and broadly applicable.
- **Strong empirical gains on the paper’s main target metrics.** On average over five LVLMs, Table 1 shows improved CHAIR scores and Table 2 shows the best average POPE accuracy/F1 among the listed baselines. These gains are not merely cosmetic; on several models/datasets the improvements are substantial.
- **Good model coverage for the main hallucination evaluations.** The paper tests on five LVLMs (LLaVA, LLaVA-v1.5, MiniGPTv2, mPLUG-Owl2, InstructBLIP), which is stronger than many contemporaneous hallucination-mitigation papers that evaluate on only one or two.
- **Useful ablations on multi-source guidance.** Table 6 supports that combining DETR and RAM++ helps versus either alone, and Table 7 usefully probes intersection vs. union aggregation. Including the oracle variant MARINE-Truth is also informative, as it reveals the headroom available with better guidance.
- **Reasonable efficiency story for the evaluated setup.** Table 5 suggests that, for LLaVA-7B, MARINE incurs about 2x decoding overhead, which is materially better than some stronger-cost baselines such as OPERA and LURE as reported here.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper overclaims preservation of general task quality and detailedness relative to the evidence shown.** The abstract claims MARINE reduces hallucinations “while maintaining the detailedness of LVLMs’ generations,” and Section 5.2 says it maintains overall performance on broader tasks. But the supporting evidence is limited: Table 3 covers only **two** models on **90 QA examples** and **50 caption examples**, and one reported detailedness score slightly decreases (LLaVA captioning: 4.39 → 4.36). Figure 2 is only a radar plot for two models and does not provide the level of quantitative support needed for a broad cross-model “no trade-off” claim. The paper supports “strong hallucination reduction with some evidence of limited quality preservation,” not the stronger blanket claim.
- **The mechanism is only partially isolated: it is unclear how much of the gain comes from CFG-style logit interpolation versus simply injecting detector text into the prompt.** MARINE combines several components at once: external detectors/taggers, object-to-text conversion, a fixed prompt template (“focusing on the visible objects in this image:”), aggregation by intersection, and logit interpolation between guided and unguided branches. The ablations study detector combinations, aggregation strategy, and guidance strength, but they do **not** include a crucial control such as “append the aggregated object list as a normal prompt without MARINE guidance” or a simpler prompt-only baseline. Without this, the paper does not convincingly establish that the proposed decoding formulation itself is responsible for the gains, rather than the extra grounded text alone.
- **Some key results are less uniformly positive than the narrative suggests, especially on MiniGPTv2 and on recall/detail trade-offs.** In Table 1, MARINE is not uniformly best across all models/metrics: for MiniGPTv2, its CHAIR\_S (11.8) is worse than Greedy (8.2), Woodpecker (7.5), VCD (6.8), and OPERA (9.2), though recall is higher. This is important because it suggests MARINE may induce a different precision/recall or verbosity trade-off depending on the base LVLM. Similarly, LURE has much higher average recall than MARINE (55.2 vs. 44.5), showing MARINE’s stronger hallucination suppression may come with meaningful conservatism. The paper should discuss these trade-offs directly rather than repeatedly framing the method as consistently superior.
- **Dependence on the external guidance models is real, but failure analysis is thin.** The paper’s own oracle gap shows this clearly: MARINE-Truth improves recall substantially over MARINE (Table 1 average recall 57.5 vs. 44.5), indicating that guidance quality is a major bottleneck. Yet the paper offers little systematic analysis of cases where DETR/RAM++ are incomplete or wrong and MARINE correspondingly suppresses correct content or inherits detector bias. Given that the method’s core contribution is to trust external object guidance, this missing error analysis matters.

### Minor
- **The “API-free / low-overhead” framing is somewhat imprecise because a central aggregation component is underspecified in the method description.** Section 4.1 says aggregation “can be done by the language model … or rule based algorithm,” and Algorithm 1 leaves this as a generic `Aggr.` function. The experiments later appear to use intersection/union-style rule-based aggregation (Table 7), which is compatible with the API-free claim, but the main method description should make the evaluated implementation explicit rather than presenting an open-ended aggregation mechanism.
- **The theoretical exposition is sloppier than it should be.** In Section 4.2, the presentation mixes token-level and sequence-level notation in the logit-space formula, making the math less careful than the implementation likely is. This does not invalidate the method, but it weakens the clarity and rigor of the formulation.
- **The claim that MARINE addresses the “root causes” of hallucination is overstated.** The paper provides a useful inference-time mitigation strategy, but it does not actually establish causality about hallucination origins beyond plausible motivation. The wording should be softened to reflect mitigation rather than explanation of root causes.
- **Guidance-strength analysis is limited.** Figure 3 studies only two models and only CHAIR/recall, yet the paper recommends a global range of \(\gamma \in (0.3, 0.7)\) and uses a fixed value across all tasks/models. This is acceptable for a practical method, but the robustness claim is broader than the evidence.
- **Latency claims are only partially substantiated.** Table 5 reports latency on LLaVA-7B only. That is a useful point measurement, but not enough to fully justify broad claims about “lowest computational overhead” across the whole evaluated setting.

### Trivial
- The statement in Section 4.2 that \(\gamma=1\) produces generation “entirely based on the control” is imprecise: the conditional branch still includes the original visual tokens and task prompt, not only control text.

## Nice-to-Haves
- Add a prompt-only control: append the detector-derived object text to the prompt and decode normally, to isolate the value of the CFG-style interpolation.
- Provide failure-case analysis where DETR/RAM++ miss or falsely predict objects, and show how often MARINE harms an otherwise correct answer.
- Expand quality-preservation evaluation to more than two models and provide direct quantitative tables instead of only radar plots.
- Analyze generation length/verbosity changes, since reductions in hallucination can sometimes come from more conservative outputs.
- Include broader sensitivity analysis of \(\gamma\) across all five LVLMs.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Concerns about whether cited models/baselines/benchmarks are available or verifiable.** Per instruction, these are removed.
- **Missing related work / requests to cite additional methods.** Removed because external completeness cannot be verified here.
- **Complaints about omitted trivial implementation details or generic reproducibility nitpicks.** The paper gives the main hyperparameters and evaluated setup; such points are not core weaknesses here.
- **Criticism that baseline asymmetry is unfair when it favors the baseline.** For example, LURE being specialized or not broadly transferable does not weaken MARINE’s positive result; if anything, asymmetry favoring baselines is acceptable.
- **Pure style/formatting issues.** Parser artifacts and minor writing issues are not substantive grounds.
- **Claim that the method is not actually API-free because the paper allows an LLM-based aggregator.** The evaluated paper explicitly defines “API-free” as “elimination of any need for API calls to OpenAI,” and the experiments appear to use rule-based intersection/union aggregation. So the strong version of this criticism is not supported by the paper text; the remaining valid concern is only that the aggregation implementation should be described more concretely.

## Novel Insights
The most interesting synthesis is that this paper is stronger as an **engineering paper about object-level hallucination mitigation** than as a broader claim about preserving general LVLM utility. The empirical evidence is good enough to support a practical message—external object grounding plus guided decoding can substantially reduce object hallucination across several LVLMs—but the paper’s rhetoric repeatedly stretches beyond that into claims about “root causes,” maintained detailedness, and broad utility preservation that are not equally well supported. In other words, the central contribution is useful and likely publishable, but it should be framed more narrowly and more honestly.

## Suggestions
- Explicitly add a **prompt-only baseline** using the same aggregated object text without logit interpolation.
- Discuss the **MiniGPTv2 anomaly** and other non-uniform cases instead of summarizing results as uniformly superior.
- Provide a **failure analysis** tied to detector/tagger errors; the MARINE vs. MARINE-Truth gap makes this especially important.
- Soften claims from “maintains detailedness / overall performance” to a narrower statement unless stronger evidence is added.
- Clarify in the main paper that the evaluated aggregator is **rule-based intersection/union**, and give the exact textual form of the guidance prompt.
- Tighten the mathematical exposition in Section 4.2 to consistently use token-level next-step logits/probabilities.
- Report more direct quantitative quality metrics across more models if the paper wants to maintain its broader utility claims.

## Score and Decision
**Assessment on the main axes:**  
- **Originality:** Moderate. The individual ingredients are not new, but their combination into a simple, training-free object-grounded guidance scheme for LVLM decoding is useful and reasonably distinct.  
- **Importance:** High. Object hallucination in LVLMs is a real and active problem.  
- **Support for claims:** Mixed. The core claim about reducing object hallucination is well supported; the broader claims about maintained detailedness and general task quality are not.  
- **Soundness of experiments:** Good for the main object-hallucination target, but with notable gaps in mechanism isolation and trade-off analysis.  
- **Clarity:** Generally clear at the high level, though the method description and math could be tighter.  
- **Value to the community:** Good. Even with limitations, this is a practical mitigation strategy with strong empirical utility.

**Calibration against human-reviewed anchors:**  
- Compared with **LURE** (`oZDJKTlOUe`, accept poster; scores 6/8/6/5), this paper has similarly practical value and broad empirical coverage, but less convincing support for its broader claims about quality preservation. I place it in a similar but slightly more cautious range.  
- Compared with **Visual Evidence Prompting** (`xh3XUaB8M9`, reject; scores 5/6/6/5), MARINE is empirically stronger: more models, more standardized benchmarks, better ablations, and a clearer method. So it should score above that reject line.  
- Compared with **PATCH** (`ZPTHI3X9y8`, mixed 5/8/5/6, overall reject), MARINE is less methodologically ambitious but also cleaner and more plug-and-play; its evidence for practical gains is somewhat stronger.  
- Compared with stronger accepted papers like **VTI** (`LBl7Hez0fF`, accept spotlight; 6/8/8) and **Modular Attribution and Intervention** (`Bjq4W7P2Us`, accept poster; 8/8/6/6), MARINE feels less complete scientifically: weaker mechanism validation, weaker broader-task evidence, and more overclaiming. Thus it should score below those stronger accepts.

Overall, this paper is **above clear reject** because the core empirical contribution is real and useful, but **below strong accept** because the paper overstates what it has demonstrated and leaves important analysis gaps. My calibrated recommendation is a **weak accept / borderline accept**.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>