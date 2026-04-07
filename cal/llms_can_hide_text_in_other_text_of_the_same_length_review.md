=== CALIBRATION EXAMPLE 83 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly and directly states the core technical finding: LLMs can hide text within text of the same token length. The abstract effectively summarizes the protocol (Calgacus), its efficiency ("seconds on a laptop"), the model scale required ("modest 8-billion-parameter"), and the profound implication ("radical decoupling of text from authorial intent"). It also frames the urgent AI safety concern (covertly deploying an unfiltered LLM). All abstract claims are supported in the body. The abstract is compelling and sets high expectations.

### Introduction & Motivation
The introduction is philosophically engaging and strongly motivates the problem. It correctly positions the work as highlighting an extreme consequence of LLMs: the separation of coherent text from human intent. The contributions are clear: a simple, efficient, full-capacity steganographic protocol and a deep discussion of its implications for trust, safety, and the nature of LLM knowledge. The concrete motivating scenarios (political critique hidden in praise, secret manuscript in a review, unfiltered LLM deployment) are excellent and create a sense of importance. The only minor critique is that the contributions could be listed more explicitly for skimming reviewers, but the narrative flow is effective.

### Method / Approach
The method is described with remarkable simplicity and clarity in Section 3 and Figure 3. The "recipe" format makes it highly reproducible. The core mechanism—recording token ranks from the secret text `e` and using them to steer generation from a prompt `k`—is elegant and well-explained.

**Strengths:** The variations (separate encoder/decoder models, rank inversion) are thoughtful extensions. The analysis of why stegotexts are generally less probable than originals ("Low entropy token choices") is insightful and supported by Figure 5.
**Concerns:**
1.  **Formalization Gap:** A formal algorithmic description (pseudocode) for encode and decode, including the two-model case with vocabulary mismatch, would improve precision and reproducibility for the research community.
2.  **Practical Reproducibility Assumption:** The note that sender/receiver must obtain *identical logits* is a significant, under-explored practical limitation. The paper mentions different GPU architectures as a challenge but does not quantify the sensitivity. How robust are the ranks to different quantization levels, software versions (e.g., different llama.cpp commits), or sampling parameters? This is crucial for a proposed protocol.
3.  **Justification of "Random Process" Model:** The argument that ranks from `e` can be modeled as a random process to explain the probability gap is intuitive but somewhat informal. A more rigorous statistical characterization of rank distributions in natural language would strengthen this explanation.
4.  **Scope of 'Full Capacity':** The claim that the stegotext and secret message are of the "same length" is technically true in tokens, but the required secret key `k` (and optional `k'`) adds overhead. This is acknowledged but could be stated more prominently in the method summary.

### Experiments & Results
The experimental evaluation aligns well with the paper's dual goals: demonstrating feasibility and exploring implications. It is not a traditional benchmark against SOTA, which is a defensible choice given the paper's conceptual focus.

**Strengths:**
- Figure 4 successfully shows that stegotexts reside within the plausibility distribution of real human texts (Reddit posts), a key claim.
- The "Low entropy" analysis (Fig 5) provides a sound mechanistic explanation for the observed phenomena.
- The security discussion (Sec. 3.1) is honest and nuanced, covering brute-force attacks and the important concept of deniability, with a concrete example in Figure 15.
- The application scenario "Unaligned chatbots disguised as aligned chatbots" (Figs 11, 12) is a standout, compelling demonstration of a concrete, worrisome safety implication.

**Weaknesses & Missing Analyses:**
1.  **Lack of Human Evaluation:** The paper's most significant experimental shortfall is the absence of human evaluation. The claim that stegotexts are "coherent and plausible" to humans is central, yet evidence is only anecdotal (example texts) and proxy-based (LLM probability). A controlled human study (e.g., identifying the fake among triplets or rating coherence) is essential to substantiate the steganographic stealth claim for ICLR.
2.  **Limited Quantitative Analysis of Steerability:** The steerability of topic/style via `k` is shown only qualitatively. A quantitative analysis—e.g., using a classifier to measure how well the domain of `s` matches the intent of `k`—would strengthen this claim.
3.  **No Comparison to Prior Steganography Methods:** While related work is surveyed, there is no empirical comparison to methods like Meteor or Wu et al. 2024 on dimensions such as capacity-perplexity trade-off, statistical detectability, or robustness. Acknowledging this as a scope choice is acceptable, but some minimal comparison would contextualize the contribution better for the steganography community.
4.  **Detection by Other LLMs:** Figure 14 shows the probability gap persists when a different LLM (Phi-3) evaluates stegotexts made by Llama 3. This is a valuable experiment. The paper should discuss this more explicitly as a potential detection vector, even if imperfect.

### Writing & Clarity
The writing is outstanding—clear, engaging, and thought-provoking. It seamlessly integrates technical description, security analysis, and philosophical discussion. The figures are informative and well-placed. The use of historical and literary references (Tacitus, Calgacus, Oulipo, GEB) enriches the narrative without distracting. Minor note: The joke about "reviewer 2" is charming but may be seen as too informal by some.

### Limitations & Broader Impact
This is a major strength of the paper. Limitations are thoroughly discussed: dependence on LLM quality, prompt crafting, failure modes (Appendices A.1, A.5), and the critical need for identical logits. The broader impact section is exceptional. It goes beyond standard boilerplate to present a concrete, novel AI safety threat (the "shipping unfiltered LLMs" play) and engages deeply with philosophical questions about LLM knowledge, hallucination (redefined as a "void of intention"), and the intentional stance. The societal risks (evading censorship, spreading harmful content) are clearly stated. One could ask for a brief discussion on potential defenses or detection research avenues prompted by this work.

### Overall Assessment
This is a superb paper that makes a clear, simple, and compelling technical contribution (full-capacity LLM steganography) and uses it as a powerful lens to examine critical, timely questions about LLMs, intent, and safety. The core method is sound and well-presented. The primary weaknesses from a rigorous research perspective are the lack of human evaluation for stealth and the absence of comparative benchmarks with prior steganography work. However, the paper's profound conceptual discussion and the identification of a concrete, novel AI safety vulnerability (the unaligned chatbot disguise) constitute significant contributions that align with ICLR's focus on impactful machine learning research. The paper is thought-provoking and likely to stimulate important discussions. With revisions that address, at minimum, the need for human evaluation data, it would be a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces "Calgacus," a steganographic protocol that uses a Large Language Model (LLM) to hide an arbitrary meaningful text within a different, coherent text of exactly the same token length. The method is simple, efficient, and works with modest open-source LLMs (e.g., Llama 3 8B) on consumer hardware. Beyond the technical contribution, the paper's core significance lies in its exploration of the profound implications: a radical decoupling of text from authorial intent, which challenges our understanding of LLM knowledge, hallucinations, and poses novel AI safety risks, such as disguising unaligned model outputs within aligned responses.

### Strengths
1. **Conceptual Novelty and Significance**: The "same length" constraint is a clear advance over prior generative steganography. The paper compellingly frames this as a fundamental decoupling of text from intent, using it to interrogate deep questions about LLMs (knowledge, hallucination, alignment). This elevates the work from a technical method to a significant philosophical commentary relevant to AI safety and trust.

2. **Clear, Accessible Method and Strong Empirical Validation**: The protocol is described with elegant simplicity (the "recipe"). Figures 4 and 5 provide convincing evidence: stegotexts fall within the plausibility distribution of real human texts, and the analysis of rank frequencies explains the observed probability gap. The extensive appendix (Figs 7-15) thoroughly explores practical considerations like model quality, domain adaptation, and security.

3. **Interdisciplinary Impact and Provocative Discussion**: The discussion section is a major strength. It connects the technical result to AI safety via a concrete, worrying application ("Unaligned chatbots disguised as aligned chatbots," Figs 11-12), redefines hallucinations as a "lack of intention," and thoughtfully engages with philosophy and history (Oulipo, Dennett, Tacitus). This broad relevance is ideal for ICLR's interdisciplinary audience.

4. **Reproducibility and Transparency**: The authors provide a GitHub demo, specify exact models and hardware, and the method is simple enough to reimplement from the description. The use of open-source LLMs (Llama 3 8B) and standard datasets (Reddit) facilitates replication.

5. **Excellent Clarity and Engaging Writing**: The paper is exceptionally well-written. The narrative is compelling, figures are informative and often witty (e.g., Fig 6), and complex ideas are explained accessibly without sacrificing rigor.

### Weaknesses
1. **Limited Formal Security Analysis**: While the paper appropriately avoids overly idealized steganography models, a more formal treatment of security would strengthen it. The discussion in Section 3.1 is qualitative. A theoretical analysis of the encoding capacity (1 bit per token in rank space) and its relation to the LLM's entropy, or a more rigorous bound on the difficulty of brute-forcing the key, would add depth.

2. **Incomplete Evaluation Against Automated Detection**: The paper shows that stegotexts are plausible and that the *original* text is more probable to the LLM, but it does not systematically evaluate whether the *stegotexts themselves* can be detected by automated steganalysis tools. An experiment using a modern classifier (e.g., a fine-tuned LLM discriminator) on a held-out dataset would better substantiate the claim of practical undetectability.

3. **Narrow Empirical Scope for Broader Claims**: Core experiments use 85-token Reddit snippets. While sufficient for proof-of-concept, claims about encoding "an entire article" lack extensive validation on longer, more diverse documents (e.g., news articles, code, dialogues). A scalability study would bolster the method's generality.

4. **Under-explored Failure Modes and Robustness**: Appendix A.1 shows a failure case (Romanesco dialect) but the analysis is brief. How does the method perform on highly structured, low-perplexity text (e.g., legal code, mathematical proofs)? The proposed heuristic of rank inversion (Fig 9) is useful but not systematically evaluated across domains.

5. **Ethical Discussion Could Be More Balanced**: The paper vividly outlines malicious use cases (circumventing censorship, smuggling unaligned outputs) but gives less space to potential beneficial applications (e.g., privacy-preserving communication for activists, artistic expression). A more balanced discussion, including mitigations or societal safeguards, would befit a top-tier conference.

### Novelty & Significance
**Novelty**: The core technical idea—achieving *length-preserving* generative steganography via token ranks—is novel. Prior LLM steganography modifies a cover or encodes at less than 1 bit/token; the equal-length constraint is a new and meaningful contribution. The philosophical framing—using this protocol to probe intent, knowledge, and hallucination in LLMs—is highly original and interdisciplinary.

**Significance**: The significance is substantial. Technically, it presents a powerful, efficient steganographic primitive with immediate implications for AI safety and trust. Societally, it contributes to urgent debates on alignment, misinformation, and the nature of machine-generated content. It aligns perfectly with ICLR's focus on novel, impactful research that advances the understanding of deep learning and its societal implications. The work is likely to inspire both technical follow-ups and broader discourse in AI ethics and safety.

### Suggestions for Improvement
1. **Strengthen the Security and Detectability Evaluation**:
   - Add a dedicated experiment testing state-of-the-art steganalysis detectors (or a simple classifier like a fine-tuned LLM) on a corpus of stegotexts vs. real texts. Report detection error rates.
   - Provide a more formal, quantitative analysis of the key search space complexity, perhaps using a simplified model of prompt plausibility to estimate lower bounds.

2. **Expand Empirical Validation**:
   - Test on longer documents (500-1000 tokens) to verify scalability and assess if perceptual quality or coherence degrades with length.
   - Conduct a human evaluation study (e.g., via MTurk) to measure whether humans can distinguish stegotexts from genuine texts in a blind setting.
   - Systematically vary the steering prompt `k` and secret text `e` to quantify their impact on stegotext quality using automated metrics (e.g., MAUVE, stylistic similarity scores).

3. **Add Theoretical Contribution**:
   - Derive the expected log-probability gap between the original text and the stegotext, linking it to the entropy of the LLM's next-token distribution and the rank distribution. This would formalize the insight in Figure 5.

4. **Address Ethical Balance and Broader Impact More Thoroughly**:
   - Include a dedicated "Broader Impact" section discussing potential beneficial uses (e.g., secure communication for journalists, novel artistic constraints) alongside the risks.
   - Suggest concrete directions for mitigation or future safety research (e.g., detection methods, architectural changes to LLMs to reduce this vulnerability).

5. **Clarify Limitations and Future Work**:
   - Explicitly discuss the white-box requirement (access to logits). How might the method be adapted for black-box APIs?
   - Discuss potential adaptive defenses: if a platform suspects Calgacus is being used, could it perturb logits or sampling to break rank correspondence?
   - Outline clear future directions: extension to multimodal models, formal information-theoretic treatment, and countermeasures.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No direct comparison to state-of-the-art LLM-based steganography methods (e.g., Wu et al. 2024, Zamir 2024).** The claim of novelty for "full capacity" is weak without benchmarking against contemporary methods in terms of stealth, capacity, and robustness. This omission makes the contribution's significance unclear for ICLR.
2. **Lack of human evaluation to assess stegotext plausibility.** The paper relies solely on LLM log-probability as a proxy for "coherent and plausible" text. A controlled human study (e.g., Turing test) is necessary to validate the core claim that stegotexts are indistinguishable from genuine texts, which is central to the paper's implications.
3. **Incomplete evaluation of the proposed AI safety attack ("shipping unfiltered LLMs").** Only a single anecdotal example (gas meter) is provided. A systematic evaluation across multiple sensitive queries, different aligned/unaligned model pairs, and success rate measurements is needed to substantiate the severity and practicality of this threat.
4. **Missing ablation studies on key parameters.** The effect of prompt length, specificity, and the inclusion of random strings (for security) on stegotext quality and steerability is discussed but not quantitatively analyzed. Controlled experiments are required to guide practical use and understand limitations.

### Deeper Analysis Needed (top 3-5 only)
1. **No security analysis against steganalysis.** The paper informally discusses security but provides no empirical or theoretical analysis of detectability. Even a basic statistical test (e.g., comparing rank or probability distributions of stegotexts vs. natural texts) is absent. Without this, the protocol's viability for real-world covert communication is not convincing.
2. **Insufficient analysis of failure modes and boundaries.** While Appendix A.1 shows a failure case, there is no systematic characterization of what message types (e.g., low vs. high perplexity), domains, or model capabilities lead to incoherent stegotexts. This undermines the claim of hiding "arbitrary meaningful text."
3. **Superficial treatment of "deniability."** The deniability claim (Figure 15) is supported by a single toy example. A rigorous analysis is needed: for a given stegotext, how many plausible bogus keys exist, and can an attacker distinguish them using probabilistic or semantic cues? This is critical for the security argument.

### Visualizations & Case Studies
1. **Side-by-side comparison of stegotexts and naturally generated cover texts.** The paper displays stegotexts in isolation (e.g., Figure 13). Showing a stegotext alongside a text generated normally from the same prompt \(k\) would visually reveal any artifacts, unnatural phrasing, or topic drift that could alert a human observer.
2. **Visualization of rank sequences over time.** Plotting the sequence of token ranks used to generate a stegotext versus a natural text could reveal systematic patterns (e.g., high-rank clusters) that might be statistically detectable, even if not human-noticeable.
3. **Case studies of characteristic failures.** The paper highlights successful examples. Visual examples and analysis of typical failure modes (e.g., gibberish, obvious non-sequiturs) would clarify the method's limitations and the conditions under which it breaks down.

### Obvious Next Steps
1. **Evaluate against basic steganalysis detectors.** As a steganography method, a minimal next step is to test its stealth by training a classifier (e.g., on n-gram statistics or LLM features) to distinguish stegotexts from real texts. This directly tests the claim of concealment.
2. **Quantify the trade-off between capacity, quality, and security.** The method achieves full capacity, but at what cost? A quantitative analysis showing how stegotext quality (e.g., perplexity, human rating) degrades with message perplexity or length is essential for understanding practical utility.
3. **Extend evaluation to state-of-the-art, larger LLMs.** Experiments are limited to models up to 27B parameters for practical reasons. Testing with frontier models (e.g., GPT-4, Claude) is necessary to assess the generalizability of the method and its implications for the most capable systems.
4. **Deeper discussion of ethical implications and mitigations.** The paper raises serious AI safety concerns but offers minimal discussion on potential countermeasures, detection strategies, or the ethical responsibility of publishing such a protocol. This is a significant omission given the paper's stated societal impact.

# Final Consolidated Review
## Summary
This paper introduces Calgacus, a protocol for hiding any meaningful text within a different, coherent text of the same token length using LLMs. The method is simple, efficient, and works with modest open-source models. Its significance lies in demonstrating a radical decoupling of text from authorial intent, with concrete implications for AI safety, trust, and our understanding of LLM knowledge and hallucinations.

## Strengths
- **Conceptual novelty and interdisciplinary impact**: The same-length constraint is a clear technical advance, and the paper compellingly frames it as a probe for fundamental questions about intent, knowledge, and safety in LLMs. The discussion redefines hallucinations as a "void of intention" and presents a concrete, novel AI safety threat (disguising unaligned chatbots within aligned responses).
- **Clear, reproducible method with strong empirical validation**: The protocol is described with elegant simplicity (the "recipe"). Figures 4 and 5 provide convincing evidence that stegotexts reside within the plausibility distribution of real texts and offer a mechanistic explanation for the observed probability gap via rank analysis.
- **Thorough exploration of practical considerations and security**: Appendices systematically address dependencies on model quality, message domain, and prompt crafting. The security discussion is honest and nuanced, covering brute-force attacks and the important concept of deniability with a concrete example (Figure 15).

## Weaknesses
- **Lack of human evaluation for stealth**: The central claim that stegotexts are "coherent and plausible" to humans is supported only by anecdotal examples and LLM probability proxies. For a steganography method claiming human-level stealth, a controlled human study (e.g., fake text identification) is essential to substantiate this claim.
- **Incomplete evaluation against automated detection**: While the paper shows stegotexts are plausible, it does not test whether they can be detected by steganalysis tools. An experiment using a classifier (e.g., a fine-tuned LLM) to distinguish stegotexts from natural texts is needed to assess practical undetectability.
- **Limited quantitative analysis of steerability and failure boundaries**: The steerability of topic/style via prompt `k` and the characterization of failure conditions (e.g., for structured text like code or low-perplexity domains) are discussed qualitatively and with examples but not systematically quantified. This makes practical guidance and limitations unclear.
- **Practical reproducibility concern under-explored**: The requirement for sender and receiver to obtain identical logits is noted as challenging across hardware/software, but the sensitivity to factors like quantization levels, software versions, or sampling parameters is not quantified, which is crucial for a proposed protocol.
- **Safety scenario demonstration needs breadth**: The compelling "unaligned chatbot disguise" application (Figures 11-12) is demonstrated with only a single example (gas meter tampering). A more systematic evaluation across multiple sensitive query types and model pairs would better substantiate the threat's practicality and severity.

## Nice-to-Haves
- Comparison to prior steganography methods to contextualize the trade-off between the novel full-capacity property and other metrics like stealth or robustness.
- Extended evaluation on longer documents (beyond 85 tokens) to verify scalability and coherence maintenance.
- A more formal, information-theoretic analysis linking the observed probability gap to the entropy of the LLM's next-token distribution.
- Discussion of potential mitigations or detection strategies prompted by this vulnerability.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- "Formal security analysis is lacking" – The paper explicitly avoids building a formal steganography model as a deliberate design choice (Section 2), making this a scope difference, not a flaw.
- "Missing theoretical proofs" – The paper is primarily empirical; theoretical analysis, while interesting, is not a standard requirement for this type of contribution.
- "Joke about reviewer 2 is too informal" – A minor style nitpick.
- "Contributions should be listed more explicitly" – A minor organizational preference.
- "Need to test on frontier models like GPT-4" – Not necessary to validate the core claim, which is convincingly demonstrated with open-source models.

## Novel Insights
The paper's most novel insight is using the same-length steganography constraint as a lens to examine fundamental questions about LLMs. It demonstrates that coherent text can be completely decoupled from authorial intent, leading to a reconceptualization of hallucinations not merely as factual errors but as a "lack of intention" that fundamentally erodes trust. This framework connects a technical capability to deep philosophical questions about meaning, knowledge, and the intentional stance towards AI systems, offering a fresh and provocative perspective on LLM limitations and societal risks.

## Suggestions
- Conduct a human evaluation study (e.g., a pairwise preference test or a "identify the fake" task) to quantitatively assess the perceptual quality and stealth of the generated stegotexts.
- Train and evaluate a steganalysis classifier (e.g., a fine-tuned LLM) on a corpus of stegotexts versus natural texts to measure automated detectability and identify any statistical signatures.
- Perform a systematic ablation study varying prompt specificity, message perplexity, and text domain to quantitatively characterize the trade-offs between steerability, output quality, and success rate, providing clearer practical guidance.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 6.0]
Average score: 7.0
Binary outcome: Accept
