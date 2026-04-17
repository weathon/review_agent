---
job_id: ef24b49f-7263-449c-807c-2b5f8864522e
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: VbTLgEUocp.pdf
paper: LLMs Can Hide Text in Other Text of the Same Length
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies steganography using large language models, including generative modeling, representation learning for language, and AI safety considerations, which are all squarely within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The submission includes an abstract, introduction, related work, a clear description of the method, empirical analysis (including quantitative results and multiple qualitative case studies), a discussion section, and conclusions. The work is technically coherent and in English.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find hidden prompts, instructions to automated reviewers, or other manipulative content in the submission.

---

# Expected Review Outcome:

## Summary

The paper introduces **Calgacus**, a protocol that uses an LLM to hide a “secret” text \(e\) inside a different, coherent text \(s\) of the *same token length*. The core idea is to record, for each token of \(e\), its **rank** in the next-token distribution of a chosen LLM given the preceding context, then generate \(s\) under a separate prompt \(k\) by forcing the LLM at each step to emit the token with the corresponding rank. The authors show qualitatively that many stegotexts are fluent and on-topic, and quantitatively that their log-probabilities fall within the distribution of real Reddit posts (Figure 4), while discussing security, limitations, and implications for AI safety and the notion of “hallucination.”

## Strengths

1. **Simple but striking core mechanism with full-capacity property.**  
   The protocol is conceptually very clean: the only “channel” between \(e\) and \(s\) is the rank sequence \(\{r_i\}\). This directly yields the full-capacity property that stegotext and hidden text share token length. The illustration in **Figure 3** nicely clarifies the rank-recording and rank-following steps and makes the method easy to reason about, also showing why the same LLM must be used for encoding and decoding.

2. **Strong qualitative demonstrations grounded in figures.**  
   - **Figure 1** is compelling: it shows a political critique of Caesar hidden inside both a boar recipe and a laudatory speech. All three texts read as coherent and human-written, making the “decoupling” of surface semantics from hidden content very vivid.  
   - **Figures 7, 8, 9, 11, 12, and 13** cover a diverse set of domains (chess notation, Python code, dialect poetry, VR game review, safety-critical instructions), which effectively argue that the procedure is not a toy example but can reliably operate in many realistic text domains, given a capable model and reasonable prompts.

3. **Quantitative plausibility analysis using LLM log-probabilities.**  
   **Figure 4** is a useful diagnostic: comparing the cumulative log-probability of 1000 real Reddit posts with random ASCII strings, random English-word sequences, and stegotexts reveals that Calgacus stegotexts mostly fall *within* the real-text band, unlike random baselines. This supports the claim that the hidden-text-carrying outputs would typically pass as normal text to an LLM-based plausibility checker. **Figure 14** mirrors this using a *different* LLM (Phi-3), showing that the lower probability of stegotext relative to its original \(e\) is not just an artifact of using the same generator for evaluation.

4. **Insightful analysis of why stegotexts are less probable than originals.**  
   The “Low entropy token choices” section, backed by **Figure 5**, teasing apart the *frequency* of rank 1 tokens and their *probability mass*, is a nice piece of analysis. It makes clear that although ranks are preserved, rank-1 positions in the secret are often “wasted” on higher-entropy spots in the stegotext, which explains the consistent probability drop for \(s\) relative to \(e\).

5. **Security properties including deniability are at least qualitatively addressed.**  
   The security section discusses the requirement of knowing the exact LLM and prompt-key \(k\), points out that naive brute-force search on \(k\) is intractable in vocabulary-space, and raises the issue of *deniability*. **Figure 15** is a concrete, nontrivial example where a bogus key yields a coherent alternative secret with log-probability higher than the stegotext, making it difficult for an adversary to prove which hidden message is “real.”

6. **Thoughtful AI safety and epistemic discussion.**  
   The paper does more than present a hack; the discussion section reflects on what it means for LLMs to “know” something and proposes reframing hallucinations as failures of intention attribution, rather than just factuality. The gas-meter example in **Figures 11 and 12** illustrates an especially important safety scenario: using Calgacus to ship an apparently aligned chatbot interface while actually conveying unfiltered responses to users who know the key.

7. **Method is practical and reproducible.**  
   The protocol is architecture-agnostic and requires only logit access to an LLM. Appendix details (e.g., on handling different vocab sizes via arithmetic coding, A.4) and **Figure 8** with timing measurements on a laptop GPU show that encoding and decoding texts of article length is feasible in seconds with 8–27B models. The recipe in Section 3 is explicit enough for independent replication.

## Weaknesses

1. **Experimental evaluation is thin, especially on detection and robustness.**  
   The empirical section essentially boils down to log-probability histograms on 1000 Reddit posts and some illustrative case studies. There is no systematic evaluation of:
   - How often stegotexts appear incoherent or off-topic for humans, across diverse \(e\) and \(k\).  
   - Detection performance of steganalysis tools or LLM-based classifiers trained to distinguish \(e\) vs \(s\). The text briefly notes that “LLMs can uncover a distinction between original texts and most of their encoded counterparts” but does not present ROC curves, detection accuracies, or sensitivity to model choice.  
   - Robustness of decoding under realistic perturbations: small reformatting, minor editing by users or platforms, or slightly different quantization levels / hardware.  
   Given the security and societal implications that the paper emphasizes, the lack of a clear detection/robustness study significantly limits the scientific value of the claims.

2. **No formalized steganographic security model or capacity analysis.**  
   The paper deliberately avoids Cachin-style formal models, arguing that they rely on unrealistic assumptions. Fair enough, but then the security discussion remains largely qualitative. For example, the brute-force complexity \(O(d^{|k|})\) assumes an attacker must guess *entire token sequences* over the full vocabulary, but in practice an attacker can drastically prune the keyspace by modeling typical prompts and leveraging the content and style of \(s\). There is no attempt to quantify the mutual information between \(k\) and \(s\), or to characterize an information-theoretic guarantee in even an approximate or empirical way. For an ICLR paper that foregrounds security and deniability, this is a substantial limitation.

3. **Evaluation of “soundness” is simplistic and potentially misleading.**  
   The plausibility metric is \(p(A) = \prod_t p(a_t \mid a_{<t})\), which is immediately converted to log-probabilities (implicitly in the plots). The authors correctly note that this only allows comparison across equal-length texts. However, there are unresolved issues:
   - The paper conflates “meaningfulness” with high autoregressive likelihood, which is known to be a poor proxy in general. Low-perplexity text can be dull or templated, while more creative or out-of-domain text may be penalized.  
   - In **Figure 4**, the densities are compared visually, but there is no statistical test or measure of overlap; the claim that stegotexts “remain within the plausibility of real texts” is qualitatively supported but not quantified (e.g., fraction of stegotexts falling into the 5–95% percentile band of real posts).  
   - The argument in “Low entropy token choices” treats the ranks in \(e\) as a random process and estimates frequency of rank 1s from a *single* Economist article (Figure 5), which is a very narrow sample. There is no investigation of how stable the rank-frequency histograms are across domains and models.  
   A more careful treatment of perplexity, KL divergence to a real-text baseline, or even simple rank-based statistics over a large corpus would strengthen the claims.

4. **Security implications are discussed but not stress-tested with realistic attacks.**  
   The security section stops at high-level algorithmic complexity arguments and qualitative deniability. Missing are:
   - Concrete attack experiments where an adversary with partial knowledge (e.g., knows model architecture and approximate style of \(k\), but not the exact string) attempts to recover \(e\) or identify that steganography is being used.  
   - Any adversarial training of detectors that look for subtle statistical signatures (e.g., unusual rank distribution, entropy patterns, or frequency of rare tokens) in \(s\). There are hints that LLMs can distinguish real vs fake texts in aggregate (Figure 14), but no attempt to leverage this for steganalysis.  
   - Exploration of side channels: e.g., the exact-softmax probabilities are heavily quantized in many deployments; tiny numerical differences across GPUs or libraries can alter rank ties. This is mentioned in passing (Page 6, Shanmugavelu et al., 2024) but never empirically investigated.  
   As a result, the protocol’s “security” properties are unclear: the scheme is undoubtedly *hard* to brute-force in the worst case, but how hard it is to detect or compromise in practice is not actually shown.

5. **Practical fragility of the scheme is underexplored.**  
   The method requires that sender and receiver: (1) share the *exact* same LLM, tokenizer, and numerical implementation, and (2) preserve the stegotext verbatim. In practice, social media and messaging platforms perform normalizations, spell-checks, truncation, or text wrapping; even minor edits will generally render decoding impossible, since a single token shift changes all subsequent ranks. There is no quantitative analysis of:
   - How often decoding fails if a small number of characters or tokens are inserted, deleted, or replaced.  
   - Whether the rank sequence can be made error-correcting or redundant at reasonable overhead, and how this affects the full-capacity claim.  
   - Impact of sampling temperature or top-k clipping on encoding/decoding if the LLM API does not expose the exact logits.  
   This fragility significantly affects real-world usability for covert communication, yet is only briefly acknowledged under “Limitations.”

6. **Mathematical and algorithmic exposition leaves open questions.**  
   While the overall idea is easy to grasp, several technical details are vague or under-specified:
   - The decoding algorithm assumes an unambiguous way to reconstruct the rank sequence from \(s\). However, in practice, how are ties in logits handled? Is there a fixed sort order over tokens (e.g., by ID) when probabilities are equal within numerical precision? This matters because a different tie-breaking convention between sender and receiver changes the rank sequence.  
   - The assumption that “we can reasonably model the ranks we obtain from \(e\) as a random process” is used informally to derive expectations about rank-1 frequency, but no definition of that process is given, nor any justification that it is i.i.d. across positions or domains.  
   - Appendix A.4 suggests arithmetic coding to handle vocabulary mismatch between encoder and decoder models. That is an interesting extension, but the construction is only sketched and there is no complexity or overhead analysis, nor any discussion on how the resulting two-token codes affect text plausibility or detection risk.  

7. **Empirical study of prompt engineering and rank inversion is mostly anecdotal.**  
   The paper introduces useful heuristics like including a contextual prompt \(k'\) before \(e\) and inverting the rank sequence to push high-rank tokens to the end of the stegotext (Section A.5 and **Figure 9**). However:
   - The only quantitative insight about token ranks across position is **Figure 10**, a boxplot of rank vs position on 1000 Reddit texts. While this clearly shows that initial tokens are harder to predict, it does not connect directly to a measurable improvement in stegotext quality from rank inversion, beyond a single example in Figure 9.  
   - There is no systematic user study or automatic metric benchmarking with and without these tricks across many \(e\)/\(k\) pairs. As a result, it is hard to assess how robust or generally beneficial these heuristics are.

8. **Related work on LLM-based steganography is incomplete.**  
   Although the paper cites a number of generative steganography and LLM-stego works (Ziegler et al. 2019; Kaptchuk et al. 2021; Wu et al. 2024; Zamir 2024), some directly relevant recent works are missing:
   - *DeepStego: Privacy-Preserving Natural Language Steganography Using Large Language Models and Advanced Neural Architectures* (Kuznetsov et al., 2025) proposes an LLM-based scheme with focus on detection resistance and embedding capacity, which is closely aligned with the Calgacus setting and should be compared in Section 2 and the security discussion.  
   - *Defining Cost Function of Steganography with Large Language Models* (Wu and Wang, 2025) focuses on quantifying steganographic cost in LLMs with a two-stage optimization, which is relevant to the paper’s discussion around plausibility and log-probability shifts (Figure 4 and 5).  
   Not situating Calgacus with respect to these contemporaneous efforts makes it harder to judge the originality and practical significance of the full-capacity property.

9. **Ethical discussion does not extend to concrete mitigation strategies.**  
   The paper raises serious concerns, especially with the gas-meter example (Figures 11–12) and the alignment-bypassing application. However, it does not propose concrete mitigation strategies, e.g., watermarking schemes that are robust to Calgacus, guidelines for API providers to limit logit access, or detection pipelines that platforms could adopt. Given the strong emphasis on societal impact, some technical or policy-leaning recommendations would make the work more constructive.

Overall, the core idea is strong and thought-provoking, but the empirical and formal analysis are somewhat light for ICLR standards, and several important technical details and attack scenarios remain underexplored.

## Potentially Missing Related Work

1. **O. Kuznetsov, K. Chernov, A. Shaikhanova. “DeepStego: Privacy-Preserving Natural Language Steganography Using Large Language Models and Advanced Neural Architectures,” 2025.**  
   DeepStego uses LLMs for natural-language steganography with attention to detection resistance and capacity. This is directly relevant to Calgacus, which also leverages LLMs for text steganography. It should be discussed in Section 2 (Related Work), especially in the paragraph on “Generative steganography,” and compared in terms of capacity (Calgacus’s same-length property) and detectability.

2. **H. Wu, Y. Wang. “Defining Cost Function of Steganography with Large Language Models,” 2025.**  
   This work aims to formalize steganographic cost for LLMs using an LLM-guided program synthesis plus evolutionary search, which is highly relevant to the paper’s use of log-probabilities and rank distributions as proxies for “soundness.” It should be cited in Section 2 and in the discussion around Figure 4 and 5, where the authors try to quantify plausibility and probability shifts between real and fake texts.

## Questions

1. **Detection experiments.**  
   Could you train or fine-tune a simple classifier (e.g., using LLM embeddings or shallow features of token ranks / entropies) to distinguish between real Reddit posts and Calgacus stegotexts? A ROC curve or detection error trade-off would materially strengthen the security discussion. I would update my assessment if such a detector consistently fails.

2. **Robustness to editing and platform transformations.**  
   Have you tried passing stegotexts through typical preprocessing pipelines (HTML stripping, Unicode normalization, spell-checking, or small manual edits) and measuring decoding success rate? Quantitative results on error tolerance or an exploration of redundancy/error-correcting coding built on top of your rank channel would help clarify practical feasibility.

3. **Tie-breaking and numerical stability.**  
   How exactly are token ranks defined in the presence of equal logits or near-equal probabilities, and how sensitive is decoding to different GPU architectures or quantization schemes in practice? For example, if you run the same model on two different GPU types, do you empirically see rank mismatches that break decoding, and at what frequency?

4. **Capacity trade-offs and partial-length encoding.**  
   You focus on the full-capacity same-length property, but in practice one might be willing to trade some capacity for robustness or improved plausibility (e.g., using only low ranks). Have you explored versions that encode only a subset of ranks or use error-correcting redundancy, and how does that affect the log-probability distribution and detectability?

5. **Relation to missing works (DeepStego, Wu & Wang 2025).**  
   How does Calgacus compare in terms of detection resistance and embedding capacity to existing LLM-based steganography methods like DeepStego or the cost-function approach of Wu and Wang? Are there scenarios where your full-capacity property yields a practical advantage, or where their optimization frameworks might outperform your simple rank-preservation scheme?

Clarifying these points, especially with additional experiments, would significantly increase my confidence in both the security claims and the broader impact of the work.

## Flag For Ethics Review

- Yes, Potentially harmful insights, methodologies and applications  
- Yes, Privacy, security and safety  
- Yes, Responsible research practice (e.g., human subjects, data release)

## Details Of Ethics Concerns

The work explicitly enables high-capacity covert communication in ordinary-looking text, including the concrete scenario in **Figures 11 and 12** where safety-critical, illegal instructions (gas-meter tampering) are hidden in apparently aligned chatbot responses. This raises several concerns:

- **Potential misuse for criminal, extremist, or abusive communication.** The protocol is simple to implement using open-source LLMs and works with full capacity; malicious actors could use it to distribute harmful information on public platforms while remaining hard to detect.  
- **Undermining of content moderation and alignment mechanisms.** The “unaligned under aligned” application illustrates that a company could superficially comply with safety policies while covertly delivering unsafe outputs to users who possess a key, which could erode trust in AI services.  
- **Lack of mitigation strategies.** While the paper has a thoughtful discussion of societal impact, it does not propose concrete technical or policy mitigations. Review by ethics chairs may be warranted to assess whether further restrictions or additional discussion are needed in the camera-ready version (e.g., stronger disclaimers, partial redaction of potentially replicable harmful examples, or emphasizing defensive uses).

There are no human subjects or personal data concerns evident, but the dual-use nature of the method is substantial.

## Soundness Rating

3: good.  
The core mechanism is correct and well explained, and the qualitative examples are persuasive. However, the empirical evaluation is limited, and security and robustness analyses are mostly qualitative, leaving open questions about detection resistance and practical viability.

## Presentation Rating

3: good.  
The writing is clear, engaging, and well structured. Figures are well chosen and effectively illustrate the arguments; **Figures 3, 4, 5, 9, 10, 11–12, 14–15** are especially helpful. Some technical details (tie-breaking, stability, vocabulary-mismatch extensions) are under-specified, and the related work section omits a few directly relevant recent papers.

## Contribution Rating

3: good.  
The paper introduces a conceptually simple but impactful protocol with a distinctive full-capacity property, and offers thought-provoking analysis of its implications for AI safety and the notion of LLM “knowledge.” While the technical novelty is modest in terms of algorithmic sophistication, the idea and its demonstrated consequences are likely to be of clear interest to the community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The work presents a compelling and practically implementable protocol that sharpens our understanding of what LLMs can do in steganographic settings and raises nontrivial safety concerns. The method is technically sound and well explained, and the qualitative evidence is strong. However, the empirical evaluation and formal security analysis are relatively light, leaving important questions about detection, robustness, and real-world feasibility unanswered. With a more thorough experimental section and better positioning relative to contemporary LLM-stego work, this could be a clear accept; as it stands, I lean positive but see room for substantial strengthening.

## Reviewer Confidence

4: confident.  
I am familiar with LLMs, generative modeling, and basic steganography, and have carefully examined the methodology and figures. Some aspects of the security analysis and comparisons to very recent related work could benefit from further expert scrutiny, but I am unlikely to have misread the central technical claims.