## Human Reviewer 1

### Summary
This paper presents a simple but interesting method that allows a large language model to hide one meaningful text inside another coherent text of the same token length. The core idea is to record the rank sequence of each token in the text to be hidden according to the model’s next-token probability distribution and then generate a new text following these ranks under a secret prompt. The hidden text can later be reconstructed by anyone who knows the same model and secret key. The authors also discuss potential implications for AI safety, information hiding, authorship, and hallucination.

### Strengths
The method is conceptually elegant. It shows that large language models can be used as full-capacity generative steganographic systems, producing natural-looking texts that conceal arbitrary content.

The approach achieves one-to-one token correspondence between the hidden text and the generated text while maintaining coherence, which is interesting among existing steganography methods.

The paper connects the technique to broader philosophical questions about language, intention, and meaning in machine-generated text, offering an original perspective.

The writing is clear and engaging.

### Weaknesses
The experimental analysis is minimal. The evaluation relies mostly on qualitative examples and log-probability plots without systematic comparisons or quantitative metrics such as recoverability, perplexity degradation, or detectability.

The proposed misuse scenarios, such as unaligned chatbots hidden within aligned ones, are speculative and not demonstrated experimentally.

### Questions
How sensitive is decoding to differences in model versions or vocabulary?

How does the quality of the generated text change when the hidden text has higher entropy or contains rare tokens?

Suggestion: Formalize the method mathematically, including an analysis of its capacity and error tolerance.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposes an interesting method of protocol (steganography) using large language models to hide a secret text within a plausible-looking cover text. The method leverages the probabilistic nature of LLMs to encode information in the choice of words generated, allowing for covert communication. The authors demonstrate that this can be achieved with small open-source models and standard text generation techniques. They also discuss the implications of their findings on our understanding of LLMs, particularly regarding the concept of "hallucinations" and the relationship between human intent and machine-generated text.

### Strengths
1. The writing is clear, intuitive and intriguing.
2. The idea is simple yet effective.
3. Extensive analysis and discussions are provided, making it deep and insightful.

### Weaknesses
1. The novelty of the work is not very clear. Similar ideas have been explored in previous work and need to be better differentiated.
2. Some practical limitations.
3. Lack of robustness analysis in the adversarial scenario.

### Questions
Overall, I found the paper interesting and well-written. The illustrations and examples provided well support the concepts and findings discussed. The discussions on the implications of the work for our understanding of LLMs were particularly thought-provoking.

However, I have some questions and suggestions for improvement:

**(1) Technical Novelty:**

I believe the contribution of the paper would be largely on its insights and discussions rather than technical novelty. However, I suggest the authors to better clarify the novelty of their method compared to prior works on text steganography using language models, e.g., the related works mentioned in Lines 145-149. What is the key difference compared to related works? Hiding secret text in the cover text with the same length may not be distinctive enough. May need more comparison and discussion on this.

**(2) Some practical limitations:**

It is good that the authors discussed some limitations, e.g., (1) Conceal a non-plausible text (random password)
into a plausible fake text is not hard (Line 245-246); (2) Fake text less probable than real and can be detected (Line 260-266).
I would like to point out a bit more:
- The rank of the first token e_1 is not controled (directly dependent on the vocalbulary), which may affect the first token of s, and affect its coherence.
- Aligned LLMs (e.g., GPT-5) may refuse to generate text containing harmful or sensitive content, making the ranks of harmful tokens in e really low, which may affect the encoding to s. Hence, hiding harmful or sensitive content may not be feasible in aligned LLMs (Section 4).

**(3) Robustness analysis:**

It would be beneficial to include a robustness analysis to evaluate how well the proposed method performs under adversarial scenarios. For instance, what is the decoding performance if the fake text s go through some slight transformations, such as: simple paraphrasing, synonym replacement, or insert some blank (\t) or line break (\n) tokens.

**Question:**

Can we probably map the rank of e to a range of low values during encoding (e.g., top 10% of the vocalbulary)?
For example, r_1 = 5, r_2 = 20 -> r_1^{'} = 5/10 = 0.5 (round to 1), r_2^{'} = 20/10 = 2. This would affect the decoding of the original e (if do not need to be exact), but may improve the fluency of s.
I think this is essentially a trade-off: either you wanna exact decoding of e to sacrifice some fluency of s, or you wanna fluent s to sacrifice some exactness of e.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes a method to hide secret text by first converting it into a sequence of probability rankings and then using a key prompt to guide a LLM to generate cover text, with each token strictly selected according to the ranking sequence.

### Strengths
1. This paper offers a new perspective for LLM safety and alignment: a model that appears aligned on the surface may still harbor vulnerabilities that allow dangerous information to be hidden in its output probability distribution.
2. It introduces a remarkably simple, full-capacity method for embedding hidden text specifically designed for LLMs.
3. Due to the secret key prompt and the inherent chaos in LLM behavior, this approach is currently nearly impossible to detect without access to both the key prompt and internal model details.

### Weaknesses
1. The method is sensitive to the quality of the key prompt; a low-quality prompt may prevent the target probability ranks from forming a coherent and natural-looking cover text.
2. It is fragile to transmission errors; any corruption in the cover text will completely scramble the recovered probability rank sequence, making it unsuitable for noisy communication channels.
3. It imposes constraints on the secret text itself, which must lie within the model’s training domain, for example, rare dialect expressions or out-of-distribution content may not be reliably encoded.

### Questions
1. How can we select an effective key prompt to ensure that the generated cover text appears natural in most cases?
2. Is it possible to design error-correction mechanisms for cover text transmission errors?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper introduces a simple protocol to hide text (e) within another coherent, same-length LLM text (s). It uses the probability ranks from e to deterministically guide the generation of s from a secret key k. The authors highlight a critical AI safety risk: hiding unfiltered answers within “safe” model responses.

### Strengths
+ Important topic, identifying an emergent and unsettling capability of LLMs.
+ The proposed protocol is remarkably simple, efficient, and demonstrated to be effective even on small open-source models.
+ The discussion section provides a deep, thought-provoking analysis of the decoupling of text from authorial intent.

### Weaknesses
- The security assumptions of the primary AI safety threat model (the "Shibbolethian Theatre") are unclear, particularly regarding the public transmission of the key.
- Key claims, such as “deniability”, are asserted but not empirically demonstrated, weakening their practical standing.
- The evaluation of the stegotext's “plausibility” relies solely on LLM-based metrics (log-probability), which may not fully capture human perception of “unsuspicious” text.
- The protocol's strict reliance on exact model versions for decoding may limit its practical utility and robustness in real-world scenarios.

### Questions
1. The “Shibbolethian Theatre” threat model is a core application. In this scenario, the reasoning trace t (which serves as the secret key k) and the stegotext s are both sent to the user. This seems to contradict the standard steganographic assumption that the key must remain secret. Could the authors clarify how security is defined in this context? Does this threat model rely on the protocol itself being secret (security through obscurity) rather than the key?
2. The paper makes a strong claim about providing “deniability”, suggesting a sender can reveal a “bogus secret key” that yields a different, also plausible, message. This implies that for a single s, multiple pairs of (k, e) exist that are all “plausible”. Given the deterministic nature of the decoding sequence (k, s) to LLM to e, is this practically feasible? Providing even a single empirical example of this phenomenon would significantly strengthen this claim.
3 . The protocol’s reliance on the exact same LLM for encoding and decoding is a critical detail. Given the rapid iteration of LLM versions in real-world deployments, this seems to make the protocol highly fragile. Could the authors comment on this practical limitation and how it might affect its utility as a reliable covert channel?
4 . The use of cumulative log-probability to evaluate the “plausibility” of the stegotext is logical. However, this metric essentially uses the LLM to judge its own output. It would be beneficial to supplement this with a human evaluation. Can the authors comment on whether a text deemed “plausible” by an LLM is necessarily “unsuspicious” to a human reader?
5. The paper notes that the protocol’s success is domain-dependent, failing on high-entropy inputs or specific dialects (like Romanesco). This is an important limitation. Could the authors provide a more detailed discussion on the boundaries of this method? What characteristics of a text e make it a viable candidate for this steganographic protocol?
6. The paper dedicates significant space to the concepts of knowledge, intent, and hallucination. In the AI safety scenario, the aligned oLLM is forced to generate a harmful answer. If a model outputs a specific string only because it is forced to by an external rank sequence, without any semantic grounding for that choice, can this be defined as the model possessing that “knowledge”? This distinction seems central to the paper’s broader philosophical claims.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3