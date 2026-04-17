---
job_id: 49645e63-75a4-44f4-8b33-53da7071a0ed
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: fD9YRHazW3.pdf
paper: In-Context Watermarks for Large Language Models
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about LLM watermarking, in‑process vs post‑hoc methods, and prompt‑based mechanisms in both standard and security‑sensitive settings (DTS and IPI). This fits ICLR’s core areas of representation learning, generative models, and safety/privacy.

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion/Discussion). The work is technically coherent, with nontrivial methodology and reasonably extensive experiments. No obvious fatal theoretical or experimental flaw is apparent from the text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
There are no hidden instructions targeting the reviewer. Any “prompt injection” phrasing is part of the research content itself, not an attempt to manipulate the review.

---

# Expected Review Outcome:

## Summary

The paper introduces **In‑Context Watermarking (ICW)**, a family of watermarking schemes for LLM‑generated text that operate purely via prompt engineering, without requiring access to the model’s decoding process.  

Four strategies are proposed at different linguistic granularities (Unicode, Initials, Lexical, Acrostics), each with a tailored detector and, for Initials/Lexical, a statistical false‑alarm guarantee. The authors evaluate these methods in a **Direct Text Stamp (DTS)** setting and in an **Indirect Prompt Injection (IPI)** case study aimed at detecting LLM‑written peer reviews by embedding hidden instructions inside the manuscript.

## Strengths

1. **Clear, well‑motivated problem setting (DTS + IPI)**  
   - The paper tackles a genuinely important gap: watermarking when one has *no* access to the generation process and cannot rely on provider‑side watermarks. The DTS setting captures generic user‑side watermarking, while the IPI case study in Section 3.2 and **Figure 2** is concrete and timely (conference organizers embedding invisible instructions in PDFs to detect LLM‑generated reviews). This is a meaningful extension of the current watermarking landscape from “provider‑controlled” to “third‑party‑controlled”.

2. **Conceptually simple but systematically explored ICW family**  
   - The four strategies (Unicode, Initials, Lexical, Acrostics; Section 4.2) cover character‑, word‑, and sentence‑level constraints. The paper does a good job discussing tradeoffs across LLM requirements, detectability, robustness, and text quality, summarized in **Table 1**.  
   - The acrostic design (Section 4.2.4), with secret strings and Levenshtein‑distance‑based detection, is an interesting twist that leverages sentence structure instead of token distribution.

3. **Substantive empirical evaluation across multiple axes**  
   - Detection performance is reported for two proprietary models (GPT‑4o‑mini and GPT‑o3‑mini), two settings (DTS, IPI), and several ICW strategies, with and without attacks. **Table 2** is a strong central piece: it clearly shows that ICWs with GPT‑o3‑mini reach ROC‑AUC ≈ 1.0 in both DTS and IPI for most methods, and highlights that weaker models (GPT‑4o‑mini) simply fail to follow more complex instructions (Initials/Acrostics).  
   - Robustness to editing/paraphrasing is tested via word deletion, word replacement, and LLM paraphrasing attacks (Section 5.2.2). **Figure 3** shows ROC curves under these attacks and makes it visually clear that Initials, Lexical, and Acrostics ICWs maintain high ROC‑AUC under paraphrasing, roughly matching or exceeding baselines in the paraphrasing regime. Tables 5 and 6 in Appendix D.1 further break this down per model and setting.
   - Text quality is evaluated using both perplexity and an LLM‑as‑a‑judge rubric (Section 5.2.3). **Table 3** demonstrates that, for GPT‑o3‑mini, ICW variants have overall quality scores close to unwatermarked outputs and often *better* than some post‑hoc baselines (e.g., PostMark).

4. **Some theoretical grounding for statistical detectors**  
   - For the Initials and Lexical ICWs, the paper provides a false‑alarm guarantee via adaptation of the green/red list analysis of Zhao et al. (2023a). **Theorem B.1** on Page 20 (Appendix B) gives a bound on the probability of seeing an unusually high count of green words under the null, and the discussion explains how to select an input‑dependent threshold $\eta$ to control the false‑alarm rate $\alpha$. Even though the result is imported rather than fundamentally new, it strengthens the credibility of the detection rule based on $z_{\mathbf y}$ in Sections 4.2.2 and 4.2.3.

5. **Insightful observation about dependence on model capability**  
   - The empirical results make a convincing point that ICW effectiveness scales with instruction‑following/in‑context‑learning strength. In **Table 2**, Initials ICW on GPT‑4o‑mini has ROC‑AUC ≈ 0.57–0.62, essentially near random, whereas on GPT‑o3‑mini it jumps to ≈ 0.997–0.999. This observation is discussed explicitly in Section 5.2.1 and in the concluding remarks, giving a realistic picture of when ICW is likely to work in practice.

6. **Useful qualitative examples and visualizations**  
   - **Figure 1** provides an intuitive end‑to‑end schematic of ICW in the DTS setting, showing how a single system‑prompt instruction can watermark all subsequent answers and later be detected.  
   - The examples in Appendix F (e.g., Tables 13‑16) illustrate how ICWs manifest in natural‑looking text. Particularly, the Initials ICW example (Table 14) makes it clear that the watermark is fairly invisible while still introducing an unusual bias in initial letters, which is relevant for assessing detectability vs. stealthiness.

## Weaknesses

1. **Security / robustness analysis against informed adversaries is quite shallow**  
   - The paper acknowledges spoofing/removal risks but stops at a light empirical “adaptive attack” (Appendix D.2, Table 10) where another LLM is asked to detect and remove watermarks without prior knowledge of the scheme. This is a weak adversary model:  
     - If the attacker knows ICW is used and has a guess about the scheme (e.g., “lots of words starting with a particular subset of letters”), **Initials ICW** is trivially spoofable or removable via simple post‑processing. The authors hint at this (Section 4.2.2 “vulnerable to spoofing”) but do not quantify any spoofing false‑positive rates or show how easy it is to *imitate* the watermark on human text.  
     - For Unicode ICW, an attacker who simply normalizes whitespace or strips zero‑width characters defeats the watermark completely. While Section 5.2.2 mentions fragility under cross‑platform transformations, there is no experiment that systematically applies simple sanitization pipelines to confirm that ROC‑AUC collapses, which is important for realistic deployments.  
   - In IPI, the reviewer is explicitly assumed to be “potentially malicious” (Section 3.2) yet the attack space analyzed under this setting is minimal: one “ignore prior prompts” string (Table 11) and generic paraphrasing. No exploration of simple, practical defenses the attacker might use (e.g., copy‑pasting only part of the paper, stripping metadata, converting PDFs to plain text using tools that drop hidden text) is provided. This significantly weakens the security story for the core motivating use case.

2. **Limited empirical diversity: only two proprietary models and no open‑weight models**  
   - All watermark generation relies on GPT‑4o‑mini and GPT‑o3‑mini. There is no evaluation on open models (e.g., Llama 2/3, Qwen2.5, etc.) nor any discussion of potential issues such as differing tokenization, instruction‑following behavior, or context handling. Given that the selling point is *model‑agnostic* watermarking, readers will reasonably ask how portable ICW is across the model ecosystem.  
   - Moreover, the IPI case study is evaluated only on a single downstream task (paper reviews) and only assuming the reviewer uses exactly those two OpenAI models. It is unclear whether similar detection performance would hold if reviewers used a weaker or differently aligned model, or a fine‑tuned local LLM.

3. **Somewhat incremental technical novelty relative to existing watermarking & prompt‑injection work**  
   - At a conceptual level, ICW is “use natural‑language instructions to bias certain observable linguistic statistics, then detect them”. Initials ICW and Lexical ICW are essentially green/red list watermarking instantiated through prompts instead of decoding‑time logit biasing. Acrostics ICW is an acrostic‑style constraint similar in spirit to classic text steganography (e.g., Topkara et al., 2006; Meral et al., 2009). Unicode‑based schemes have been explored by Sato et al. (2023) and Por et al. (2012).  
   - What the paper adds is the *systematic exploration under in‑context learning*, plus the IPI peer‑review application. This is still a meaningful step, but technically the algorithms themselves are simple and mostly adaptations. The theoretical component (Theorem B.1) is imported almost verbatim from Zhao et al. (2023a). The paper could benefit from more ambitious design, e.g., ICWs that fundamentally tie into attention patterns or reasoning steps, not just surface statistics.

4. **Detection methodology for Acrostics ICW is underspecified / heuristic**  
   - Section 4.2.4 defines detection via Levenshtein distance between the sequence of sentence‑initial letters $\boldsymbol \ell$ and a secret key sequence $\boldsymbol \zeta$, then normalizing via an empirical z‑statistic where $\mu$ and $\sigma$ are estimated by resampling subsequences from the suspect text. Several details are unclear or potentially problematic:
     - How exactly are “sentences” defined when parsing arbitrary text? Are they splitting on '.', '!' and '?' only, and how do they handle abbreviations, lists, or bullet points (especially in reviews / scientific prose)? Errors in segmentation can strongly affect $\boldsymbol \ell$ and therefore detection.  
     - The resampling procedure for estimating $\mu$ and $\sigma$ implicitly assumes something like an exchangeability or independence structure for sentence initials which is not justified. If the suspect text has topical or stylistic biases, the null distribution of Levenshtein distance to $\boldsymbol \zeta$ may deviate substantially from the surrogate distribution built from $\widehat{\boldsymbol \ell}_j$.  
     - There is no theoretical false‑alarm bound for this detector (Appendix B explicitly notes this), and the empirical Section 5 does not explore calibration of thresholds for desired FPRs beyond the reported ROC curves. Given that Acrostics ICW can show perfect ROC‑AUC in **Table 2**, the lack of a principled threshold selection method is a gap for high‑stakes scenarios.

5. **Statistical assumptions and practicality of the z‑test for Initials/Lexical ICW are not fully discussed**  
   - The detector for Initials and Lexical ICWs (Sections 4.2.2 & 4.2.3) is \( D(\mathbf y) = (|\mathbf y|_G - \gamma|\mathbf y|) / \sqrt{\gamma(1-\gamma)|\mathbf y|} \), which approximates a standard normal under a binomial model where each word independently falls into $\mathcal V_G$ with probability $\gamma$.  
   - While Theorem B.1 gives a more careful bound in terms of $V(\mathbf y)$ and $C_{\max}(\mathbf y)$, the main text both uses simple z‑scores and treats $\gamma$ at times as a corpus prior (Canterbury Corpus) and at times as $|\mathcal V_G|/|\mathcal V|$. The distinction between these two calibration regimes is not fully clarified. For Initials ICW, estimating $P_\mathcal A$ from Canterbury may not match the domain distribution of peer reviews or ELI5 answers, which can affect FPR control.  
   - There is no discussion of how sensitive performance is to errors in $\gamma$ or to correlation between word choices. The ablations in Tables 7 and 8 focus on context/output length, but not on mis‑specification of letter/word priors, which matters for deploying the claimed false‑alarm guarantees.

6. **Positioning and related work omits several directly related watermarking efforts**  
   - The related work section is lengthy but still misses some relevant, recent efforts that are close in spirit:
     - *SynthID‑Text: AI Text Watermarking Tool* (DeepMind, 2024) introduces a production watermarking tool targeting AI text provenance, which would be natural to mention alongside other in‑process and post‑hoc methods in Section 2.  
     - *Visual Pattern‑Based Watermarking for Large Language Model Generated Text* (Zhang et al., 2026) proposes watermarking via visual formatting patterns. Since Unicode ICW also exploits invisible or near‑invisible formatting, this is close conceptually and should be discussed, particularly around Table 1 where visual/Unicode methods are compared.  
   - Including and contrasting with these would help clarify where ICW sits relative to state‑of‑practice tools and conceptually similar pattern‑based approaches.

7. **Practical deployment issues in the IPI peer‑review scenario are underexplored**  
   - **Figure 2** portrays a neat pipeline where conference organizers stamp PDFs with instructions that survive whatever pipeline the reviewer uses to feed the paper to an LLM. In practice, there are many brittle points:
     - Many reviewers paste only sections of the paper (e.g., abstract + conclusion). If the hidden instruction is at the end (as in the experiments; Appendix C), will it ever be seen by the LLM? No partial‑input analysis is provided.  
     - OCRed or re‑saved PDFs (e.g., via screenshot → PDF or print‑to‑PDF) may drop hidden text. The Unicode ICW especially seems sensitive to such transformations, but there is no experiment for the instructions themselves getting lost.  
     - Detectors assume access to the *entire review text* and use 300‑word chunks in experiments. Real peer reviews vary widely in length, and shorter ones may fall below the regime where the z‑statistic is reliable. The impact of review length on IPI detection is not analyzed separately (Tables 7 and 8 are for DTS).  
   - These practical issues matter because the central selling point is “scalable and accessible content attribution in realistic peer‑review workflows.”

8. **Evaluation of text quality relies heavily on one LLM‑as‑a‑judge and perplexity; no human subjective study**  
   - **Table 3** and **Figure 4** (perplexity boxplots) suggest ICWs retain quality comparable to unwatermarked outputs. However, all quality scores come from a single judge model (Gemini‑2.0‑flash) and a single perplexity model (LLaMA‑3.1‑70B). There is no indication of inter‑rater reliability or robustness across judges.  
   - Given that some methods (Initials ICW) explicitly bias lexical choice and may introduce subtle unnaturalness, a small human study or at least cross‑judge evaluation would strengthen the claim that the watermarks are imperceptible “in practice”, especially in the context of expert reviews.

Overall, the paper is solid and thought‑provoking, but the security and deployment aspects are not as deeply developed as the empirical detection plots might suggest.

## Potentially Missing Related Work

1. **DeepMind, “SynthID‑Text: AI Text Watermarking Tool”, 2024.**  
   - Directly relevant as a deployed watermarking system for AI‑generated text, built for provenance and detection. It should be cited in Section 2 when discussing in‑process watermarking methods, and briefly contrasted with ICW in terms of required access to the generation process and black‑box applicability.

2. **Zhang, H., Wang, J., Li, T., “Visual Pattern‑Based Watermarking for Large Language Model Generated Text”, 2026.**  
   - Proposes watermarking based on visual patterns in rendered text, conceptually close to Unicode ICW and other formatting‑based schemes. It would fit naturally into the “Post‑hoc LLM Watermarking” discussion in Section 2 and also be a useful comparison point in Section 4.2.1 when discussing Unicode ICW’s strengths and fragilities.

## Questions

1. **Robustness under more realistic IPI reviewer behavior**  
   - In the IPI experiments, are reviewers (LLMs) always given the *entire* PDF including the embedded instruction? Could you provide detection results when only the abstract + introduction, or random sections of the paper, are used as the LLM input? This would help assess how brittle ICW is to partial copy‑pasting, which is common in practice.

2. **Sentence segmentation and parsing for Acrostics ICW**  
   - How exactly do you define and extract sentences when computing $\boldsymbol \ell$ for Acrostics detection (Section 4.2.4)? How are edge cases handled (enumerated lists, abbreviations, section headings)? Can you provide empirical sensitivity analysis showing how often segmentation errors change the detection outcome?

3. **Attacker aware of the scheme**  
   - For Initials and Lexical ICWs, have you tried spoofing attacks where an attacker explicitly overuses the same green letters/words on *human* text to falsely trigger the detector? How does ROC‑AUC change when such adversarial positives are added? This seems critical for understanding whether these schemes can be gamed to incriminate humans.

4. **Generalization to open models and different tokenizations**  
   - Have you conducted any preliminary experiments on open‑weight LLMs (e.g., LLaMA/Qwen) to verify that the same instructions produce similar watermark strengths? If not, can you comment on expected difficulties, such as tokenization mismatches (especially for Unicode ICW) or weaker instruction‑following capabilities?

5. **Threshold selection and calibration for high‑stakes deployment**  
   - For methods with statistical detectors (Initials/Lexical/Acrostics), how would you recommend practitioners pick thresholds $\eta$ in practice? Is there an empirical procedure (e.g., using a small held‑out corpus of known human reviews) to calibrate TPR/FPR, and have you tested its stability?

Clarifications along these lines could strengthen the security and practicality story and might shift my assessment more positively.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core ideas are simple but correctly instantiated; experiments are broad and largely well‑executed. Security modeling and some detection details (especially for Acrostics) are somewhat heuristic and under‑analyzed, but not blatantly incorrect.

## Presentation Rating

3: good.  
The paper is generally well‑written, well‑structured, and supported by clear figures and tables (e.g., Figures 1–3, Tables 1–3). A few technical aspects (sentence parsing, calibration details) could be explained more rigorously.

## Contribution Rating

3: good.  
The main contribution is bringing watermarking into a purely prompt‑based, black‑box setting and showing that this works reasonably well given strong LLMs, plus a timely IPI peer‑review application. Technically the schemes are adaptations rather than deeply new, and the security depth is moderate, but the work is still a meaningful and useful addition to the field.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper offers a solid and well‑evaluated exploration of in‑context watermarking methods and convincingly demonstrates that prompt‑only watermarking is feasible with modern LLMs, including in an IPI peer‑review scenario. At the same time, the security analysis is relatively shallow for an accountability tool, and much of the technical machinery closely parallels existing watermarking or steganographic ideas. Overall, the strengths slightly outweigh the weaknesses, and I see this as a worthwhile, if not definitive, contribution.

## Reviewer Confidence

4: confident.  
I am familiar with LLM watermarking and prompt‑injection literature and have carefully checked the main equations, algorithms, and experimental tables/figures, though I did not independently re‑implement the methods.