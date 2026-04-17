Now I have sufficient calibration data. Let me synthesize the final review.

Key calibration anchors:
- **0koPj0cJV6** (Black-Box Watermarking): Scores 5,6,5,3,1 → avg ~4, rejected. Weaknesses: limited model evaluation, lack of theoretical soundness, comparison issues with existing work.
- **NQZImD0VGP** (Universally Optimal Watermarking): Scores 6,3,3,6 → avg ~4.5, rejected. Weaknesses: flawed theoretical framework, limited practical experiments.
- **eKGEsFdpin** (Sampling-Based Watermarking): Scores 5,3,3 → avg ~3.7, rejected. Limited attacks evaluation, dependency on model capability.
- **DEJIDCmWOz** (Reliability of Watermarks): Scores 6,6,6 → avg 6, accepted poster. Comprehensive evaluation, important topic.
- **RKQcJ1lXNT** (Adaptive Attacks on Watermarks): Scores 6,6,5,5 → avg ~5.5, rejected. Interesting but specific limitations.
- **SsmT8aO45L** (Provable Robust Watermarking): Scores 8,8,6,6 → avg 7, accepted poster. Strong theory, good experiments.
- **ujpAYpFDEA** (Watermarked LLMs Identified by Crafted Prompts): Scores 8,6,8,8 → avg 7.5, accepted spotlight. Novel framing, good experiments.

Now let me think about where this paper falls:

The paper has a genuinely novel idea (ICW via prompt engineering), demonstrates it works on strong proprietary models, and has a compelling motivating scenario (IPI for peer review). However, it has significant limitations:
1. Only 2 models tested, both proprietary OpenAI models
2. Unrealistic threat model for IPI
3. No adaptive adversarial evaluation
4. Overclaiming of "model-agnostic" and "practical"
5. Most ICW methods fail on the weaker model (GPT-4o-mini)

The paper's strengths are real (novel framing, systematic exploration, strong results on capable models), but the weaknesses are structural: the overclaiming, the narrow evaluation, and the unrealistic IPI threat model.

Comparing to the calibration anchors:
- Similar to eKGEsFdpin (rejected, score ~3.7) in terms of limited model diversity and attack evaluation
- Similar to 0koPj0cJV6 (rejected, avg ~4) in terms of overclaiming relative to experimental evidence
- More novel than eKGEsFdpin but with similar practical limitations
- Better executed than NQZImD0VGP but with more overclaiming

I think a fair score would be around 4.5-5. The paper has a genuinely interesting and novel idea, but the overclaiming, narrow evaluation, and unrealistic threat model for the IPI scenario are substantial weaknesses. It's above the weakest papers (3-4 range) because the idea is genuinely novel and interesting, but below the accepted watermarking papers (6+) because the experimental support doesn't match the claims.

Let me write the final review.</think>## Summary

This paper introduces In-Context Watermarking (ICW), a paradigm that embeds watermarks into LLM-generated text solely through prompt engineering—leveraging LLMs' instruction-following capabilities rather than requiring access to the decoding process. The authors propose four ICW strategies at different linguistic granularities (Unicode, Initials, Lexical, Acrostics), each with a tailored detection method, and evaluate them in two settings: Direct Text Stamp (DTS, where users directly provide watermarking instructions) and Indirect Prompt Injection (IPI, where instructions are covertly embedded in input documents to detect AI-misused peer reviews).

## Strengths

- **Novel problem formulation**: The core insight—that watermarking can be achieved through prompts alone, enabling third parties (not just model owners) to embed watermarks—is a genuine and important contribution. No prior work has systematically explored this direction, and the IPI setting for detecting AI-misused peer reviews is a timely and compelling motivation.

- **Systematic exploration of the design space**: The four strategies (Unicode → Initials → Lexical → Acrostics) span different linguistic granularities with clear trade-off analysis (Table 1) across LLM requirements, detectability, robustness, and text quality. Each method has a principled detection scheme (z-statistic, Levenshtein distance, etc.) with theoretical guarantees for Initials and Lexical ICW (Appendix B).

- **Strong results on capable models**: With GPT-o3-mini, all four ICW methods achieve near-perfect detection (ROC-AUC ≥ 0.995 in DTS, ≥ 0.997 in IPI). Most methods maintain good text quality (Table 3), and robustness to paraphrasing is demonstrated for Initials and Acrostics ICW (Figure 3).

- **Honest presentation of failure modes**: The paper clearly reports that GPT-4o-mini largely fails for Initials and Acrostics ICW (e.g., ROC-AUC 0.572 and 0.590 in DTS), acknowledging the capability dependence. The open discussion of trade-offs and limitations (Section 6) is commendable.

## Weaknesses

### Major:

- **Overclaiming "model-agnostic" and "practical" based on extremely narrow experimental support**: The abstract claims ICW is a "model-agnostic, practical watermarking approach," but all experiments use only two proprietary models from the same provider (GPT-4o-mini and GPT-o3-mini). Three of four ICW methods essentially fail on GPT-4o-mini (a still-strong model), and no open-source models (LLaMA, Mistral, etc.) are tested. The claim of model-agnosticism is unsupported by the evidence—what the experiments actually show is that ICW works reliably on one specific strong proprietary model under controlled conditions. This is an interesting preliminary finding, not proof of generality.

- **Unrealistic and under-specified IPI threat model**: The motivating peer-review scenario assumes reviewers paste entire PDFs (including hidden/invisible text) directly into LLMs. This is operationally fragile: (a) reviewers may use only visible text, copy selected sections, or use PDF-to-text converters that strip invisible content; (b) once aware of the scheme, reviewers can trivially strip hidden instructions; (c) the paper defers attack/defense analysis for the IPI setting to future work, which is precisely where the primary threat lies. The paper acknowledges that "a detailed investigation of attack and defense methods is left for future work" (Section 3.2), yet the IPI results are presented as validating the approach's effectiveness. This gap between the claimed application and the experimental support is significant.

- **Insufficient robustness evaluation against adaptive adversaries**: Robustness is evaluated only against non-adaptive attacks (random deletion, synonym replacement, non-adversarial paraphrasing). The paper acknowledges that Initials ICW is vulnerable to spoofing (an adversary can infer the green letter set) and that ICW instructions can be removed, but neither is tested empirically. Given that the IPI setting is explicitly adversarial (dishonest reviewers), this is a critical gap. The "ignore prior prompts" attack is mentioned only briefly in Appendix D.1, and more sophisticated attacks (instruction stripping, targeted rewriting to break specific patterns) are unexplored. The paper's own citation of adaptive attack work (Sadasivan et al., 2023) underscores this concern.

### Minor:

- **No baselines in the IPI setting**: The paper's most novel and practically motivated scenario (IPI) has no baseline comparisons. While post-hoc watermarking methods cannot be applied in this setting (as the authors note), comparing against AI-text detectors (e.g., DetectGPT, GPTZero) would establish that ICW improves upon the best available alternatives.

- **IPI robustness evaluation is incomplete**: Robustness results for the IPI setting are relegated to Appendix Table 6 rather than the main text, and only cover output-text modifications. The input-side attack surface (modifying the stamped document before LLM processing) is not evaluated.

- **Text quality evaluation concerns**: Table 3 shows unwatermarked GPT-o3-mini text scoring 4.992 overall—noticeably higher than human text (4.235)—indicating LLM-as-a-Judge systematic bias favoring LLM-generated text. Additionally, quality is only evaluated for GPT-o3-mini; for GPT-4o-mini (where instructions are poorly followed), quality impacts are unreported.

- **Limited evaluation breadth**: Only ELI5 (DTS) and ICLR papers (IPI) are used, both English-only. No evaluation on shorter texts, different domains (code, creative writing), or multi-turn conversations. The paper also provides no analysis of detection performance as a function of output length, which is critical for practical deployment.

## Nice-to-Haves

- Test on at least one open-weight model to substantiate the "model-agnostic" claim and identify minimum capability thresholds.
- Evaluate adaptive attacks specifically designed for each ICW method (e.g., frequency analysis to infer green word lists, sentence-boundary manipulation for Acrostics).
- Report detection performance at very low FPR thresholds (0.1%, 0.01%), which are more realistic for policy applications like peer-review policing.
- Provide concrete qualitative examples of watermarked text alongside unwatermarked text so readers can assess perceptibility.
- Analyze performance as a function of output length to establish operational boundaries.

## Removed Points

- **Harsh Critic Point 5 ("as LLMs improve" narrative vs. alignment/safety constraints)**: While the tension between ICW and anti-prompt-injection defenses is worth noting, this is speculative reasoning about future model behaviors. The paper's forward-looking claim is explicitly hedged ("suggests that as LLMs become more capable, ICW offers a promising direction"). This is a reasonable conjecture, not an overclaim requiring correction. Moved to Nice-to-Have as a discussion point.

- **Harsh Critic Point 4 (detection metrics not aligned with deployment needs)**: While very-low FPR analysis would strengthen the paper, T@1%F and T@10%F are standard metrics in the watermarking literature. Requesting 0.01% FPR analysis is a Nice-to-Have enhancement rather than a core flaw, especially given the field's norms.

- **Harsh Critic Critique of Sec. 2 (comparison with Bahri et al. not "honest")**: The paper does discuss black-box watermarking and differentiates ICW by its prompt-based nature. The comparison to post-hoc baselines is appropriate given the different threat models. This is not a dishonest comparison.

- **Neutral Reviewer Point 6 ("incremental technical contribution of individual methods")**: The framing and systematic exploration of ICW as a paradigm is the primary novelty. The individual methods being straightforward adaptations is acknowledged and does not diminish the contribution; the contribution is the framing and the IPI application, not each individual technique.

- **Human Finder Point 4 ("limited theoretical analysis")**: The paper provides theoretical guarantees for Initials and Lexical ICW in Appendix B, which is reasonable for an empirical exploration paper. Requesting additional formal analysis for all methods is a Nice-to-Have.

- **Human Finder Point 3 ("LLM-as-Judge text quality concerns")**: Partially valid (unwatermarked scoring above human), but the paper also uses perplexity as a complementary metric. This is a minor concern, not a major flaw.

## Novel Insights

The paper's most striking contribution is the reversal of the traditional watermarking threat model: rather than model owners embedding watermarks, ICW enables *third parties* (e.g., conference organizers) to embed watermarks via document modification. The finding that ICW effectiveness scales sharply with model capability (near-random on GPT-4o-mini vs. near-perfect on GPT-o3-mini) is both a strength and a caution—suggesting ICW is currently viable only for top-tier models but will become increasingly powerful as models improve. This creates an interesting dynamic where advances in LLM capabilities and instruction-following directly benefit ICW, but the same advances may also improve adversaries' ability to detect and strip embedded instructions. The four-method taxonomy (Unicode → Initials → Lexical → Acrostics) nicely maps text granularity to trade-offs, with Unicode being most fragile but least demanding, and Acrostics being most robust but requiring the strongest models.

## Suggestions

- Narrow the scope claims: replace "model-agnostic, practical" with "applicable to strong instruction-following LLMs in controlled settings" and acknowledge the capability threshold explicitly in the abstract.
- Add at least 1-2 open-weight model experiments (e.g., LLaMA-3.1-70B, Qwen-2.5-72B) to establish capability requirements more concretely.
- Conduct even a small-scale adversarial evaluation: test whether an adversary who knows the ICW scheme (but not the key) can strip or evade watermarks—for example, test sentence restructuring attacks against Acrostics ICW.
- Move the "ignore prior prompts" attack results from Appendix D.1 to the main text, as this is the most practically relevant attack on the IPI pipeline.

## Score and Decision

Calibration comparison:

- **SsmT8aO45L** (Provable Robust Watermarking, accepted poster, avg 7): Strong theory + solid experiments; more rigorous but less novel framing.
- **ujpAYpFDEA** (Can Watermarked LLMs be Identified?, accepted spotlight, avg 7.5): Novel framing, good experiments, similar overclaiming concerns but better validated.
- **eKGEsFdpin** (Sampling-Based Watermarking, rejected, avg ~3.7): Limited experiments, model-specific results, incremental novelty.
- **0koPj0cJV6** (Black-Box Watermarking, rejected, avg ~4): Overclaimed generality relative to experimental support, theoretical concerns.
- **DEJIDCmWOz** (Reliability of Watermarks, accepted poster, avg 6): Timely topic, thorough evaluation, but limited theoretical depth.

This paper has a genuinely novel and important idea (ICW via prompt engineering + IPI application) that is more creative than most watermarking papers. However, the experimental validation is narrow (2 models, both from one provider), the IPI threat model is unrealistic, adaptive adversarial evaluation is absent, and the claims of "model-agnostic" and "practical" are overstated. On the positive side, it demonstrates strong results on a capable model, has a systematic design-space exploration, and honestly reports failure modes. The paper falls between the weak rejected watermarking papers (3-4) and the solidly accepted ones (6-7): the novelty and timeliness of the idea push it above pure incremental work, but the gap between claims and evidence pulls it below acceptance-level rigor.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>