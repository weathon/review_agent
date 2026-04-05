=== CALIBRATION EXAMPLE 75 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is attention-grabbing and broadly accurate in spirit: the paper is about using LLMs to hide one text in another text of the same length. However, it is also somewhat misleadingly broad because the method is not general “same length” hiding in the abstract cryptographic sense; it is specifically a token-rank-based generative steganography scheme requiring access to logits.
- The abstract clearly states the problem, gives the core idea of Calgacus, and summarizes the main empirical claim that 8B open-source models suffice and that messages can be encoded/decoded locally.
- The abstract does make stronger claims than the body fully substantiates. In particular, phrases like “radical decoupling of text from authorial intent” and “further eroding trust in written communication” are rhetorical and not empirically established by the paper’s experiments. The “urgent questions for AI safety” framing is also more of a discussion claim than a demonstrated result.

### Introduction & Motivation
- The motivation is clear: generative AI changes what written text can signify, and steganography with LLMs is the concrete vehicle for that claim.
- The gap in prior work is identified reasonably: prior generative steganography methods either rely on stronger assumptions, lower capacity, or different operational constraints. The paper’s distinctive claim is the same-length payload/ciphertext property.
- That said, the introduction over-claims the novelty and significance somewhat. The “same length” property is interesting, but the paper does not yet convincingly show that it materially changes the steganography landscape beyond being a clever rank-preserving encoding. For ICLR’s bar, the scientific contribution needs sharper positioning against existing generative coding/steganography and language-model-controlled generation literature.

### Method / Approach
- The core method in Section 3 is simple and mostly understandable: tokenize the secret text, record the ranks of its tokens under an LLM, then generate stegotext under a prompt by forcing the LLM to pick tokens at those same ranks; decoding reverses the process.
- The method is, however, only partially specified. Reproducibility hinges on details not fully stated in the main method:
  - How ties are handled in rank computation.
  - Whether ranks are computed over the full vocabulary or after truncation/quantization effects.
  - How exact tokenization mismatches are handled when the original text contains unusual tokens, Unicode, code, or nonstandard whitespace.
  - What happens when the prescribed rank exceeds the vocabulary size in practice, beyond the appendix discussion for cross-model use.
- A key logical gap is the reliance on identical logits between sender and receiver. The paper acknowledges this in limitations, but the practical impact is severe: decoding depends on exact model version, hardware, precision, and inference stack. This is not a small caveat; it is central to the protocol’s robustness.
- The “full capacity” claim is conceptually appealing but needs more care. The paper encodes a text of the same token length via ranks, but that is not the same as demonstrating near-lossless information-theoretic capacity in a rigorous sense. The method’s effective capacity is constrained by model predictability, prompt dependence, and decodability under realistic perturbations.
- Edge cases and failure modes are discussed somewhat, especially for low-quality models and out-of-distribution text, but important ones remain underexplored: long-range drift, truncation/termination behavior, and the consequences of rank inversion on semantic fidelity.
- There are no formal proofs, so the correctness burden is empirical. The paper’s claims about reversibility are plausible, but the security and deniability claims are not formally established.

### Experiments & Results
- The experiments do test the main qualitative claim that stegotexts can remain plausible and, under the chosen model, fall within the distribution of real Reddit text. They also test the claim that the original text is often more probable than the stegotext under the same or another LLM.
- The comparison in Figure 4 is informative, but it is not enough to support the broader implications the paper draws. Showing that 100 stegotexts from three examples fall within a real-text log-probability range is not the same as demonstrating broad practical steganographic robustness.
- Baselines are limited. The paper compares against random ASCII and English word sequences, but those are not strong baselines for steganography. More relevant comparisons would include prior generative steganography methods such as neural linguistic steganography, Meteor, and undetectable LM steganography under matched conditions.
- There is an important lack of ablations that materially affect conclusions:
  - No systematic study of how performance scales with prompt length beyond anecdotal examples.
  - No quantitative comparison across many model sizes, decoding temperatures, or quantization levels.
  - No controlled study of the effect of rank inversion except illustrative figures.
  - No ablation on different text domains with statistical summaries.
- Statistical reporting is weak. The paper uses distributions and box plots, but it does not report confidence intervals, significance tests, or robust uncertainty estimates for key comparisons.
- The datasets and metrics are only partially appropriate. Reddit posts are a reasonable testbed for heterogeneous informal text, but the claim that they “cannot appear in training data” is not verifiable as stated, especially for a large web-trained model. More importantly, log-probability is used as a proxy for plausibility, which is acceptable for a model-internal analysis, but it is not a direct measure of human-perceived naturalness or detectability.
- Figures 11–15 are more problematic from a scientific-evaluation standpoint. The gas-meter example in particular is used to dramatize dual-use concerns, but it is not evidence of the method’s generality. Those figures support a narrative, not the technical claims.

### Writing & Clarity
- The method section is readable and the main idea is communicated clearly.
- However, some of the most important technical points are buried under extensive rhetorical discussion, which makes it harder to distinguish the actual method from the authors’ philosophical framing.
- Figures are generally useful, especially Figures 3–5 and 9–10, but several of the later figures blur the line between evidence and illustrative storytelling. For ICLR, the paper would benefit from separating technical evaluation from speculative discussion much more cleanly.
- The appendix contains useful operational details, but some of the key algorithmic requirements are scattered across main text and appendix in a way that impedes rapid understanding of what is essential versus optional.

### Limitations & Broader Impact
- The paper does acknowledge some limitations: dependence on a good model, dependence on the secret prompt, abrupt termination, and sensitivity to matching model/inference conditions.
- But the limitations section is incomplete relative to the claims made. The most serious missing limitations are:
  - Fragility to model drift, quantization, and deployment differences.
  - Lack of a formal security analysis against informed adversaries.
  - Likely detectability by steganalysis or by comparing model likelihoods, especially under known-model assumptions.
  - Dependence on exact access to logits, which sharply reduces realistic deployment settings.
- The broader impact discussion is unusually strong and dual-use aware. It convincingly identifies misuse scenarios such as concealed unfiltered responses and covert political messaging.
- That said, the paper arguably under-discusses the practical harms of publishing a polished, easily reproducible dual-use protocol, especially given the explicit unsafe example in Figures 11–12. For ICLR’s standards, the authors should more directly address why the benefits of disclosure outweigh the obvious misuse potential.

### Overall Assessment
This is an inventive and memorable paper with a genuinely interesting technical trick: forcing an LLM to generate plausible text while carrying a hidden message via rank preservation. The core idea is simple and, in that sense, elegant. However, relative to ICLR’s acceptance bar, the paper currently reads as much as a provocative dual-use essay as a rigorously validated machine learning contribution. The empirical evaluation is narrow, the security claims are not formally supported, and several central limitations—especially dependence on exact model/logit matching and weak adversarial robustness—are acknowledged but not sufficiently analyzed. I think the contribution is real, but at present it is not yet backed by the level of systematic evidence and formalization that ICLR typically expects for a strong acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **Calgacus**, a simple generative steganography protocol that encodes a secret text into another same-length, plausible text by selecting tokens according to the rank sequence induced by the secret message under an LLM. The authors demonstrate the method on several open-source LLMs and argue that it enables efficient, locally runnable hiding/decoding, with implications for deniability, censorship circumvention, and AI safety.

### Strengths
1. **Clear and elegant core idea.** The encoding scheme is conceptually simple: record the token-rank sequence of the secret text under a language model, then force a generated stegotext to follow those ranks under a secret prompt. This is easy to understand and, at a high level, plausible.
2. **Practical demonstration on commodity hardware.** The paper reports that the method can be run on a laptop with an 8GB GPU and that encoding/decoding can happen in seconds, which makes the contribution practically accessible rather than purely theoretical.
3. **Same-length hiding is a notable property.** The “full capacity” claim—hiding a message in a same-length text—is unusual and, if robust, is a meaningful addition to the generative steganography literature.
4. **Empirical evidence that outputs can remain plausible.** The paper compares stegotext log-probabilities against a corpus of real Reddit texts and shows that many generated outputs fall within the plausibility range of natural text, which supports the basic feasibility claim.
5. **Good discussion of limitations and security caveats.** The authors explicitly note dependence on model quality, source text domain, and exact decoding conditions, and they discuss attack scenarios and deniability. This shows some awareness of practical and security issues.
6. **Potentially interesting broader implications.** The paper connects the method to questions about LLM “knowledge,” hallucination, and authorial intent. While speculative, this is a thought-provoking angle that may stimulate discussion.

### Weaknesses
1. **The paper’s main technical novelty appears limited relative to prior generative steganography.** The method is essentially a rank-preserving generation scheme using LLM token probabilities; while the same-length property is interesting, the core mechanism looks like a straightforward adaptation of existing ideas in sampling-based or distribution-guided steganography. The paper does not convincingly isolate what is fundamentally new versus an incremental reparameterization of known techniques.
2. **Evaluation is narrow and somewhat ad hoc.** The main empirical study uses 1000 Reddit snippets and a small number of exemplars/prompts. This is not sufficient to establish robustness across languages, domains, lengths, model families, or prompt types. ICLR typically expects stronger empirical breadth, especially for a paper making a strong algorithmic claim.
3. **Security analysis is informal and incomplete.** The paper discusses brute-force search and deniability, but does not provide a rigorous threat model, formal guarantees, or quantitative detection/attack evaluations. For a steganography paper, this is a major gap.
4. **No strong baselines are presented.** The paper does not compare against established generative steganography or text watermarking / encoding baselines in a systematic way. Without such comparisons, it is hard to judge whether the method is competitive or merely illustrative.
5. **The claim that outputs are “plausible” is measured only by LLM log-probability.** Using the same family of models to generate and judge plausibility raises circularity concerns. Human evaluation, preference testing, or out-of-model metrics would strengthen the case.
6. **The analysis of “same length” is token-based and may not translate well to actual text length.** The paper is explicit about token length, but the significance of same-token-count hiding may be weaker in practice because tokenization varies and users care about surface length, not only token length.
7. **Reproducibility is only partial.** The paper provides a demo and model names, but many important details are missing or under-specified: exact decoding parameters, quantization settings, prompt selection procedure, preprocessing, and the full set of evaluation hyperparameters.
8. **Some arguments are over-extended beyond the evidence.** The discussion about AI safety, “knowing,” hallucination, and intent is philosophically interesting but not strongly supported by the experiments. ICLR reviewers generally expect claims to be grounded in evidence rather than rhetorical extrapolation.
9. **The method appears fragile under realistic deployment constraints.** The paper itself notes dependence on identical logits and model conditions between sender and receiver, which is a major practical limitation. This undermines claims of broad applicability.

### Novelty & Significance
**Novelty:** Moderate to low on the algorithmic side, with a modestly novel practical twist in the same-length property and the steerable prompt-based generation. The paper reads more like a clever instantiation of generative steganography than a fundamentally new framework.

**Significance:** Potentially moderate, but not yet convincingly established. If the same-length encoding is robust and efficient, it could matter for privacy, censorship resistance, and adversarial communication. However, the current evidence is insufficient for ICLR’s acceptance bar because the work lacks strong formalization, broader evaluation, and rigorous comparisons.

**Clarity:** Generally good at the conceptual level. The paper is readable and the method is explained in an intuitive way. That said, the prose is highly stylized and at times rhetorical, which can blur the line between scientific contribution and provocation. For ICLR, a more neutral and precise presentation would be preferable.

**Reproducibility:** Partial. A demo and model references are helpful, but the experimental pipeline is not described with enough precision for confident replication of all results.

### Suggestions for Improvement
1. **Add formalization of the threat model and security properties.** Define what the adversary knows, what guarantees are claimed, and whether the protocol aims for steganographic undetectability, deniability, or only practical obscurity.
2. **Provide strong baselines.** Compare against prior generative steganography methods, sampling-based encoding schemes, and possibly watermarking-style methods where relevant.
3. **Expand empirical evaluation substantially.** Test across multiple datasets, languages, domains, secret lengths, prompts, and models, and report success rates, coherence metrics, decode fidelity, and failure modes.
4. **Include human evaluation.** Since “plausibility” is central, ask human judges to compare real vs. stegotexts and report whether the outputs are actually indistinguishable or merely high-probability under the same model.
5. **Quantify robustness to model mismatch and deployment noise.** Since exact-logit agreement is required, evaluate sensitivity to quantization, different inference backends, slightly different model versions, and decoding perturbations.
6. **Clarify the novelty relative to arithmetic coding and prior steganography.** The paper should explicitly state what is new beyond using token ranks and whether the same-length property can be seen as a special case of known coding ideas.
7. **Tone down speculative claims or separate them more cleanly from empirical contributions.** The philosophical discussion on intent, hallucination, and AI safety could be moved to a shorter discussion section unless supported by experiments.
8. **Report more implementation details.** Include exact decoding procedures, tokenization handling, prompt construction rules, quantization, temperature settings, and full hyperparameters to improve reproducibility.
9. **Evaluate practical surface-length equality.** Since the paper emphasizes same length, report character-level or byte-level length differences in addition to token counts, and explain how tokenization variability affects the guarantee.


# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison against the strongest relevant generative-steganography baselines, not just qualitative references. For ICLR, the claim of a “simple and efficient protocol” needs head-to-head results versus methods like Neural Linguistic Steganography, Meteor, and recent LLM steganography schemes on the same datasets and same model budget.

2. Add a detection benchmark against both humans and automated steganalysis/classifiers. The paper claims the fake texts are plausibly indistinguishable and can hide intent, but there is no rigorous test of whether an adversary can detect the presence of hidden text from surface form alone.

3. Add ablations isolating the effect of each key design choice: rank encoding, secret prompt, rank inversion, padding, and model size. Right now it is unclear which components actually matter for capacity, fluency, deniability, and decode success, so the core mechanism is not properly supported.

4. Add experiments on harder and more diverse domains beyond Reddit, plus multilingual and out-of-domain messages. The paper itself admits the method depends heavily on model familiarity with the hidden text; without systematic coverage across domains, the “arbitrary meaningful text” claim is overstated.

5. Add scalability experiments on length, payload type, and failure rate over long messages. ICLR reviewers will expect evidence that the protocol remains reliable beyond short examples and that encode/decode success, coherence, and latency do not collapse with longer secrets.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify encode/decode reliability and exact recovery rate as a function of prompt, message domain, and model. The paper asserts reversibility and efficiency, but the actual conditions under which recovery is robust are not characterized enough to trust the method.

2. Analyze whether the apparent “same-length” property is meaningful in tokens, bytes, and user-visible text. Because the method depends on tokenization and model-specific vocabulary, the claim of hiding text “of the same length” is not invariant and needs sharper analysis.

3. Provide a real security model with attacker capabilities and measurable guarantees. The current discussion is informal and largely rhetorical; ICLR expects a clearer threat model, assumptions, and empirical evidence for secrecy, deniability, and resistance to key-recovery attacks.

4. Analyze the gap between “plausibility” and “soundness” more carefully. The paper uses log-likelihood as a proxy for realism, but does not show that this metric correlates with human judgments or with downstream detectability, which is central to the claim.

5. Characterize why some stegotexts are more probable than others and when the original is distinguishable from the fake. The current explanation is qualitative; a stronger statistical treatment of token-rank distributions and entropy mismatch is needed to make the main result convincing.

### Visualizations & Case Studies
1. Show side-by-side examples of successful and failed encodings across multiple domains and model sizes. This would reveal whether the method is robust or only works on cherry-picked cases.

2. Add plots of decode success, human plausibility, and model likelihood versus hidden-text domain and prompt length. Those trends would make the dependence on the secret prompt and message type much more concrete.

3. Include a confusion study where annotators judge original vs stegotext, plus examples of false positives/negatives. This would directly test the paper’s central claim that the fake text is plausibly indistinguishable to humans.

4. Provide an adversarial case study showing how an attacker might infer the key or detect the presence of steganography from the stegotext. Without this, the security discussion remains speculative.

### Obvious Next Steps
1. Formalize the method as a reproducible steganographic system with explicit threat model, capacity, distortion, and security metrics. That is the minimum needed for an ICLR-level contribution rather than a conceptual demonstration.

2. Benchmark the method across several open models and decoding regimes, including smaller, larger, and instruction-tuned variants. The paper’s core claim depends on model quality, so model sensitivity is not optional.

3. Test the protocol on non-Reddit, non-English, and safety-relevant corpora with systematic ablations. If the method is truly general, it should survive these settings; if not, the scope of the claim must be narrowed.

4. Evaluate whether the protocol can be detected or blocked by simple defenses such as re-sampling, paraphrasing, or constrained decoding. That is the most obvious next step if the paper’s security and practical relevance are to be believed.

# Final Consolidated Review
## Summary
This paper introduces **Calgacus**, a simple generative steganography protocol that hides a secret text inside another plausible text of the same token length by replaying the secret text’s token-rank sequence under a separate prompt. The method is implemented with open-source LLMs and demonstrated on a small set of examples and Reddit posts, with the authors arguing that it enables fast local encoding/decoding and raises interesting questions about deniability, hallucination, and authorial intent.

## Strengths
- **The core encoding/decoding idea is genuinely simple and elegant.** Recording token ranks from the secret text and then forcing generation to follow the same rank sequence is easy to understand, and the paper does make the reversibility mechanism clear.
- **The method is practically runnable on commodity hardware.** The paper provides evidence that 8B-class open models can encode and decode quickly on a laptop, which makes the protocol tangible rather than purely theoretical.
- **Same-token-length hiding is a notable property.** If one buys the token-level framing, the ability to hide a message without changing length is an interesting twist on generative steganography and is illustrated with concrete examples.

## Weaknesses
- **The empirical evaluation is thin and heavily cherry-picked.** The main evidence is a small number of examples plus 100 stegotexts derived from only three Reddit samples. That is not enough to support broad claims about arbitrary meaningful text, robustness, or practical steganographic utility.
- **The paper’s central claims are much broader than what is actually demonstrated.** The experiments show that some stegotexts can fall within a real-text log-probability range under the chosen model, but this does not establish strong indistinguishability to humans, security against adversaries, or the sweeping philosophical claims about “radical decoupling” of text and intent.
- **Security is not analyzed at a level that would justify the steganography framing.** The protocol depends on exact access to logits, identical model/inference conditions, and a secret prompt. The paper gestures at brute-force difficulty and deniability, but gives no rigorous threat model, no quantitative attack evaluation, and no formal security guarantees.
- **The same-length claim is token-based and model-dependent, which sharply limits its robustness.** Because the method is defined over tokenizer outputs and exact vocabulary ranks, the property is not invariant to tokenization differences, model changes, quantization, or deployment drift. This is a major practical weakness, not a minor caveat.
- **There are no meaningful baselines or ablations.** The paper does not systematically compare against established generative steganography methods or dissect the contribution of rank preservation, prompt steering, rank inversion, and padding. As a result, it is hard to tell how much of the effect is new versus a straightforward repackaging of known ideas.

## Nice-to-Haves
- A clearer separation between the technical contribution and the philosophical/danger discussion would make the paper easier to evaluate scientifically.
- Human judgments of plausibility would complement the current LLM-log-probability proxy, which is only an internal model metric.
- More detailed implementation notes would help reproducibility, especially around tokenization, quantization, decoding precision, and prompt construction.

## Novel Insights
The most interesting aspect of the paper is not just that it hides text, but that it exposes a tension between **constraint satisfaction** and **authorial intent**: the generated text can remain fluent while every token is chosen to satisfy an external hidden message. That observation is real and conceptually provocative. However, the paper over-interprets this into sweeping claims about knowledge, hallucination, and trust without enough empirical or formal support; the strongest contribution is the protocol itself, while the broader interpretive claims remain largely rhetorical.

## Potentially Missed Related Work
- **Neural Linguistic Steganography (Ziegler et al., 2019)** — directly relevant prior work on text steganography with language models.
- **Meteor (Kaptchuk et al., 2021)** — relevant for generative steganography with stronger theoretical framing and different encoding tradeoffs.
- **Undetectable steganography for language models (Zamir, 2024)** — especially relevant because it addresses text steganography under LM settings.
- **Generative steganography by sampling (Liu et al., 2018)** — important prior generative-steghography baseline context.
- **Reversible generative steganography with distribution-preserving (Tang et al., 2025)** — relevant for reversible generative hiding and comparison of capacity/stealth tradeoffs.

## Suggestions
- Add a serious evaluation section with strong baselines, broader domains, and human detectability studies.
- Formalize the threat model and clearly state what security claims are and are not being made.
- Report robustness under model mismatch, quantization changes, and different inference backends.
- Include ablations showing how much each design choice contributes to coherence, reversibility, and stealth.
- Narrow or soften the philosophical claims unless they can be tied to measurable evidence.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 6.0]
Average score: 7.0
Binary outcome: Accept
