## Summary

The paper introduces In-Context Watermarking (ICW), which embeds detectable watermarks into LLM-generated text solely through prompt engineering—no access to model decoding required. Four strategies are explored at different linguistic granularities (Unicode character insertion, initial-letter biasing, green-word lexical biasing, acrostic sentence-initial constraints), each paired with a tailored detection method. The paper further studies the Indirect Prompt Injection (IPI) setting as a case study, where watermarking instructions are covertly embedded in documents (e.g., academic manuscripts) to detect AI-generated content such as fraudulent peer reviews. Experiments on GPT-4o-mini and GPT-o3-mini show that ICW effectiveness scales strongly with model capability, with all four strategies achieving ROC-AUC ≥ 0.995 on GPT-o3-mini in both the Direct Text Stamp (DTS) and IPI settings.

## Strengths

- **Identifies a real gap in the watermarking literature**: Existing LLM watermarking methods require access to the decoding process. ICW addresses scenarios (e.g., detecting AI-generated peer reviews) where only prompt-level control is available. This is a legitimate and timely problem (Section 1, Section 3.2).

- **Demonstrates strong detection performance on capable models**: With GPT-o3-mini, all four ICW strategies achieve ROC-AUC ≥ 0.995 in the DTS setting and ≥ 0.997 in the IPI setting (Table 2), comparable to baselines YCZ+23 (0.998) and PostMark (0.977). Critically, ICW methods operate in the IPI setting where no baseline can (shown as "−" in IPI columns).

- **Honest reporting of capability dependence**: The paper transparently reports poor GPT-4o-mini results (e.g., Initials: 0.572, Acrostics: 0.590 ROC-AUC), framing this as evidence that ICW scales with model capability rather than hiding it. This is scientifically responsible (Table 2, Section 5.2.1).

- **Systematic design space exploration**: The four strategies span character → letter → word → sentence granularity, and Table 1's trade-off summary (LLM requirements, detectability, robustness, text quality) provides a useful organizing framework validated by experimental results.

- **Good text quality preservation**: ICW methods with GPT-o3-mini maintain high quality scores (Lexical: 4.808, Acrostics: 4.813 overall) compared to human text (4.235), substantially outperforming PostMark (2.997) (Table 3).

## Weaknesses

### Fatal
None.

### Major

- **The IPI setting—the paper's primary motivating scenario—has a fundamental adversarial vulnerability**: The introduction and abstract prominently motivate ICW as a way to detect dishonest reviewers who use LLMs to write reviews. This is definitionally an adversarial setting where the reviewer has strong incentive to evade detection. Yet ICW's security in the IPI setting relies on the adversary being unaware of the watermarking instructions. Once the technique is public, an informed adversary can trivially defeat all four strategies: stripping zero-width spaces (Unicode), inferring the green letter set from a single output (Initials), prepending "ignore prior instructions about word selection" (Lexical), or counter-instructing against sentence-initial constraints (Acrostics). The paper acknowledges this in one sentence (Section 3.2: "the adversary may also employ defensive strategies, such as detecting and removing the embedded instruction. [...] a detailed investigation of attack and defense methods is left for future work") and mentions investigating the "ignore prior prompts" attack in Appendix D.1. However, this is not a minor limitation to be deferred—it is a Kerckhoffs-type failure for the stated application. Publishing the technique undermines its utility for its own motivating use case. This does not invalidate the broader ICW concept or the DTS setting, but it does create a fundamental tension between the paper's primary motivation and its solution.

- **False positive rate calibration for Initials ICW depends on a genre-mismatched corpus**: The γ parameter for Initials ICW detection (Section 4.2.2, Eq. for D(y|k_c, τ_c)) is estimated from the Canterbury Corpus—a collection of literary and technical texts from the early 1990s. The distribution of initial letters varies substantially by topic and genre; academic reviews about neural networks will have very different initial-letter frequencies than Canterbury Corpus texts. This means the claimed false positive rate control (Appendix B guarantees) may not hold for the actual application domain. Similarly, for Lexical ICW, γ = |VG|/|V| assumes a uniform word distribution under H₀, but natural language follows heavily skewed (Zipf) distributions, meaning the z-statistic's variance estimate may be inaccurate in practice. These calibration issues undermine the practical reliability of the detection framework for real-world deployment.

### Minor

- **Unclear negative class for IPI evaluation**: The paper specifies ELI5 answers as human-generated negatives for the DTS setting, but does not clearly state what serves as the negative class for the IPI detection results in Table 2. If ELI5 answers (long-form question answers) are reused as negatives for IPI (which detects AI-generated *reviews*), this is a genre mismatch that could distort false positive estimates. The paper should explicitly document what human-generated texts are used as negatives for the IPI setting (Section 5.1).

- **Acrostics ICW detection lacks formal false-alarm guarantees**: The bootstrap-based detection for Acrostics ICW (Section 4.2.4) derives µ and σ by resampling sentence-initial letters from the suspect text itself, rather than from a clean reference distribution. The paper's own Appendix B theoretical guarantees are stated to apply only to Initials and Lexical ICWs, not Acrostics. The statistical properties (Type I error control) of this self-referential approach are not analyzed, which is a gap given that Acrostics is one of the strongest-performing methods (ROC-AUC = 1.000 with GPT-o3-mini).

- **Overclaim about "practical" applicability**: The abstract states ICW is a "practical watermarking approach," and Section 6 suggests ICW "empowers third parties" and will become "more powerful" as LLMs advance. These claims are speculative for the IPI setting given the adversarial vulnerability above, and for current widely-deployed models (GPT-4o-mini) where most ICW strategies perform near chance level. The paper would benefit from more tempered language.

### Trivial
None.

## Nice-to-Haves

- Evaluation on at least one open-source model (e.g., Llama-3.1-70B) would clarify the capability threshold needed for ICW and improve reproducibility beyond proprietary APIs.
- A small human study on text quality perceptibility (even 10–20 judgments) would strengthen claims about watermark imperceptibility beyond LLM-as-a-Judge.
- Showing concrete watermarked text examples side-by-side with unwatermarked text for each ICW method would help readers qualitatively assess quality impact, especially for Acrostics ICW where sentence structure is constrained.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic claim that GPT-4o-mini poor performance is a weakness**: The paper honestly reports this and frames it correctly as a capability-dependent result. This is a property of the method, not a flaw in the paper's presentation. The paper does not overclaim GPT-4o-mini performance.

- **Demand for human evaluation as a Major weakness**: While human evaluation would strengthen quality claims, LLM-as-a-Judge is a widely accepted methodology in the current field. This is a nice-to-have, not a substantive flaw.

- **Demand for open-source model testing as a Major weakness**: Testing on proprietary models is standard practice in current LLM research. The capability-scaling finding from GPT-4o-mini to GPT-o3-mini already demonstrates the key insight. This is a nice-to-have.

- **Critic's claim that ICW strategies are "straightforward adaptations" with "limited technical novelty"**: This undervalues the contribution. The framing of watermarking as a prompt engineering problem, the systematic exploration across granularities, and the paired detection schemes with theoretical guarantees represent genuine design and analysis work. Novelty is not only in the individual strategies but in the conceptual framing and systematic evaluation.

- **Critic's claim about LLM-as-a-Judge limitations (verbosity bias, self-preference)**: These are known general limitations of LLM-as-a-Judge, not specific issues with this paper's evaluation. The paper also supplements with perplexity evaluation using LLaMA-3.1-70B (Section 5.1).

- **Critic's speculation that "more capable models may develop better instruction-following guardrails that resist following hidden watermarking instructions"**: This is equally speculative in the opposite direction. The paper's empirical evidence shows instruction-following works for watermarking; the critic's counter-claim has no evidence.

- **Critic's complaint that "no post-hoc detection baseline (GPTZero) is compared in IPI"**: The paper explicitly states (Section 5.1) that baselines "are not applicable in the IPI setting, as the dishonest reviewer has no incentive to add a watermark by themselves." GPTZero is a post-hoc detector, not a watermarking method, and its comparison in IPI would be a separate research question.

## Novel Insights

The most interesting tension in this paper is the fundamental paradox of the IPI setting: the more successful ICW becomes at watermarking via prompt injection, the more it relies on the LLM's instruction-following capability—which is the same capability that an adversary can exploit via counter-instructions ("ignore prior prompts"). This creates an asymmetric arms race where the defense (watermark embedding) requires the LLM to faithfully follow specific instructions embedded in long context, while the offense (watermark evasion) only requires the LLM to override those instructions with a simpler, more recent counter-instruction. The paper's Appendix D.1 investigation of the "ignore prior prompts" attack is therefore not just an ancillary experiment but the central question determining whether IPI-based watermarking can ever work against informed adversaries.

## Suggestions

- Reframe the paper to position the DTS setting as the primary contribution and the IPI setting as an exploratory case study with explicitly acknowledged adversarial limitations. This would align the paper's claims with its evidence and make the IPI vulnerability read as an honest boundary rather than a structural flaw.
- For the IPI setting, analyze the "ignore prior prompts" attack as a main-result experiment rather than an appendix item. Report success rates of evasion and discuss whether instruction ordering, repetition, or obfuscation strategies can mitigate it.
- For the Initials ICW, compute γ from multiple text genres (academic reviews, news, fiction) and report how much it varies. If γ is relatively stable across genres, the Canterbury Corpus concern is mitigated; if it varies substantially, the paper should discuss adaptive calibration.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| PMark | /home/wg25r/review_agent/human_reviews_2026/EhDgP69DJG.md | 7.0 | Solid theory + complete evaluation; our paper has comparable empirical results but significant gaps in adversarial analysis and false positive calibration |
| CoDeC | /home/wg25r/review_agent/human_reviews_2026/YlpaaYxx4t.md | 5.33 | Novel ICL-based detection idea with extensive experiments; our paper has similar novelty but more significant methodological gaps |
| BIRA | /home/wg25r/review_agent/human_reviews_2026/dZY5t9ZrUB.md | 4.5 | Attack paper showing watermark vulnerabilities; our paper is more constructive but shares the vulnerability theme |
| SaFT | /home/wg25r/review_agent/human_reviews_2026/0pVKknV9nM.md | 4.0 | Prompt-based detection with questionable evaluation fairness; comparable concerns about practical effectiveness |
| Prob. Automata | /home/wg25r/review_agent/human_reviews_2026/nDSTDq0uDK.md | 2.0 | Serious scheme flaws; our paper is clearly better with honest reporting and real empirical validation |

This paper sits between the medium-scoring anchors (BIRA at 4.5, SaFT at 4.0) and CoDeC (5.33). It has a genuinely creative and well-motivated idea with strong empirical results on capable models, but the IPI adversarial vulnerability is a significant structural concern for its primary motivating application. The paper is clearly above the low-scoring anchors (which have fundamental scheme flaws) but below the high-scoring anchors (which have rigorous theory and complete evaluation). I place it slightly above the pure attack/vulnerability papers because it makes a constructive contribution with the ICW concept and DTS setting, but the gap between the IPI motivation and its practical security prevents a higher score.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>