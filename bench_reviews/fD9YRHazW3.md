## Summary

This paper introduces In-Context Watermarking (ICW), a novel approach for embedding watermarks in LLM-generated text through prompt engineering rather than access to the decoding process. The authors propose four strategies (Unicode, Initials, Lexical, Acrostics) and evaluate them in two settings: Direct Text Stamp (DTS), where users provide watermarking instructions directly, and Indirect Prompt Injection (IPI), where hidden instructions are embedded into documents to detect AI misuse (e.g., AI-generated peer reviews). The work targets a practical gap: third parties who need provenance verification without model control.

## Strengths

- **Novel problem formulation for watermarking without model access**: The paper correctly identifies that existing watermarking methods require controlling the decoding process, limiting deployment to model owners. The IPI setting—where conference organizers could embed hidden instructions in submitted papers to detect AI-generated reviews—addresses a genuine and timely problem. The formalization in Section 3 is clean, using a standard hypothesis testing framework.

- **Multiple complementary strategies with trade-offs**: The four strategies (Unicode, Initials, Lexical, Acrostics) operate at different linguistic granularities with varying requirements. Table 1 usefully summarizes trade-offs: Unicode has minimal LLM requirements but low robustness; Acrostics preserves text quality but demands strong instruction-following. This systematic exploration provides practical guidance.

- **Strong detection performance on capable models**: With GPT-o3-mini, all four methods achieve ROC-AUC ≥ 0.995 in the DTS setting and ≥ 0.997 in the IPI setting (Table 2). Initials ICW achieves 0.999 AUC while maintaining robustness against paraphrasing (0.887 AUC after ChatGPT paraphrase attack, Table 5).

- **Robustness evaluation against realistic attacks**: Section 5.2.2 and Table 5-6 evaluate robustness under word replacement (30%), deletion (30%), and LLM paraphrasing. The paper also tests an "ignore prior prompts" attack (Table 11), demonstrating that the watermark persists even when an adversary attempts to neutralize hidden instructions.

## Weaknesses

- **Limited experimental scope undermines "model-agnostic" claim**: All experiments use only GPT-4o-mini and GPT-o3-mini from OpenAI. No open-source models (LLaMA, Mistral, Qwen) are evaluated, despite these being commonly used by budget-conscious bad actors. The paper frames ICW as "model-agnostic," but three of four methods (Initials, Lexical, Acrostics) show near-random performance on GPT-4o-mini (ROC-AUC 0.57–0.91). The claim that "as LLMs continue to advance, ICW will become correspondingly more powerful" is supported by only two data points from the same provider family—whether this reflects general capability scaling or provider-specific instruction-following alignment remains unknown.

- **Unicode ICW completely broken by adaptive attack**: Table 10 shows that after the adaptive attack (paraphrasing with watermark detection attempt), Unicode ICW achieves ROC-AUC = 0.000—meaning the watermark is not just degraded but inverted. This fundamental vulnerability is buried in Appendix D.2 and receives insufficient emphasis in the main text. For a security-focused paper, this failure mode deserves prominent discussion.

- **Acrostics detection procedure bootstraps null distribution from test text**: Section 4.2.4 describes estimating μ and σ for detection by "randomly resampl[ing] N sequences of sentence initial letters... from the suspect text." This estimates the null distribution from within the potentially-watermarked text itself, which may not accurately characterize the distribution under H₀. The paper acknowledges the lack of formal false-alarm analysis for Acrostics, but the proposed procedure's validity remains unestablished.

- **No validation that hidden instructions survive PDF ingestion**: The IPI setting assumes hidden text (white font, zero-size font) survives PDF parsing into LLM context. The paper provides no empirical validation that modern LLM file-upload interfaces actually preserve such hidden characters. If parsers strip invisible formatting, the core IPI application becomes physically infeasible—yet this foundational assumption is not tested.

- **γ estimation from Canterbury Corpus may not match modern LLM output distributions**: For Initials ICW, the probability γ of naturally beginning words with "green" letters is estimated from the Canterbury Corpus (classic literary and technical documents). Whether this baseline matches modern LLM outputs or academic review text is not validated. Mismatched γ could inflate false positive rates in practice.

## Nice-to-Haves

- **Open-source model validation**: Testing ICW on LLaMA-3, Mistral, or Qwen would strengthen the model-agnostic claim and clarify whether performance scales with general capability or depends on provider-specific training.

- **Instruction compliance rate beyond aggregate AUC**: Reporting the percentage of generations where models completely fail to follow watermarking instructions would help assess reliability for high-stakes deployments.

- **Key recovery analysis**: Evaluating whether adversaries can statistically infer the secret key (green letter list or word list) from sufficient watermarked outputs would inform spoofing risks.

- **Cost analysis for practical deployment**: Lexical ICW passes ~2,000 green words to the model; the token overhead and latency impact on large-scale deployment (e.g., thousands of paper submissions) is not analyzed.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"LLM-as-a-Judge bias toward LLM-generated text"**: The reviewer criticizes that unwatermarked GPT text scores 4.992 while human text scores 4.235, suggesting judge bias. However, the relevant comparison is ICW vs. unwatermarked LLM output (the baseline for quality degradation), not ICW vs. human text. The paper correctly uses unwatermarked LLM output as the reference (Table 3), and ICW methods achieve comparable scores (4.28–4.81), properly demonstrating minimal quality degradation.

- **"Ethical tension with modifying authors' manuscripts"**: While valid concerns about consent exist, the paper explicitly states in footnote 1 that conference organizers (not authors) should implement this to avoid conflict of interest, and the Ethics Statement addresses responsible deployment. This is a deployment consideration rather than a research flaw.

- **"Qualitative table circles not visible"**: Table 1 uses visual circles to indicate trade-offs; if these don't render in extraction, it's a format issue, not a paper problem. The criteria (LLM requirements, detectability, robustness, text quality) are described in the text.

- **"Confusion between Initials and Letter ICW naming"**: Section 5.2.2 uses "Initials ICW" while Table 5 uses "Letter ICW"—this appears to be the same method with inconsistent naming. While a minor clarity issue, it does not affect the paper's core claims.

## Novel Insights

The paper's key insight is reframing prompt injection—typically a security vulnerability—into a constructive tool for provenance tracking. The IPI setting elegantly addresses the "motivated third party" problem in watermarking: stakeholders who need detection but lack model control. The finding that watermark effectiveness correlates with model capability (comparing GPT-4o-mini to GPT-o3-mini) suggests a threshold effect that warrants deeper investigation—whether this reflects general instruction-following ability or specific alignment behaviors could inform both watermarking research and capability evaluation.

## Suggestions

- **Add at least one open-source model experiment** to validate whether ICW scales with general capability or is specific to proprietary instruction-following training.

- **Test PDF parsing explicitly**: Include a simple experiment uploading a PDF with hidden text through an LLM file interface to verify the instruction survives ingestion.

- **Emphasize Unicode ICW adaptive vulnerability in the main text**: Move the ROC-AUC = 0.000 result from Appendix D.2 to Section 5 or the limitations discussion, with explicit guidance on when (not) to use this method.

- **Provide confidence intervals for detection metrics**: Given the high-stakes application (flagging academic misconduct), reporting statistical uncertainty would strengthen reliability claims.

- **Clarify the Acrostics detection assumption**: Either provide theoretical bounds on the bootstrapped detection procedure or acknowledge that this method is empirical-only pending future analysis.