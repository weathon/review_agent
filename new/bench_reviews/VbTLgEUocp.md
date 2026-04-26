Now I have a thorough understanding of the paper and calibration context. Let me write the final review.

## Summary

The paper presents **Calgacus**, a protocol that uses an LLM to hide an arbitrary text inside another coherent text of the same token length. The method works by: (1) tokenizing the hidden text, (2) recording the rank of each token in the LLM's probability distribution, and (3) generating a new text from a secret prompt by selecting tokens at those same ranks. The paper discusses security properties (deniability, key secrecy), demonstrates that generated stegotexts fall within the probability distribution of real text, and explores philosophical implications for AI safety (notably a "Shibbolethian Theatre" scenario where an unfiltered LLM's answers can be covertly encoded within an aligned LLM's outputs).

## Strengths

- **Simple, non-obvious, and genuinely novel core idea**: The observation that rank sequences from one LLM context can be decoded in another context while preserving token-level information — enabling "full capacity" steganography where stegotext and hidden message have equal token length — is a clean contribution not achieved by prior methods like Meteor or Zamir (2024). The paper explicitly positions this: "What we add to the field is Calgacus, a protocol with the notable property of having full capacity" (Section 2).
- **Clear mechanistic explanation of the probability gap**: The "low entropy token choices" analysis (Section 3, Figure 5) explains why stegotexts are less probable than originals — rank-1 tokens are "wasted" on high-entropy positions. This is insightful and helps understand the method's operating characteristics.
- **Empirical evidence that stegotexts fall within real-text distributions**: Figure 4 shows stegotext probabilities (for texts at μ, μ−2σ, and μ+2σ) fall within the real-text distribution using 1000 Reddit posts. Cross-model verification with Phi-3 (Figure 14) shows this is not an artifact of the generating model.
- **Deniability property identified and demonstrated**: Section 3.1 observes that some stegotexts achieve probabilities comparable to originals, enabling a sender under coercion to present a bogus key yielding a plausible alternative message (Figure 15).
- **Thought-provoking philosophical discussion**: The reframing of hallucination as "lack of intent" and the analysis of LLM knowledge via the gas meter example raise genuinely interesting questions, even if they extend beyond what the protocol technically establishes.

## Weaknesses

### Fatal
None. The paper presents a real method with a genuine core contribution.

### Major

- **No comparative evaluation against existing LLM steganography methods**: The paper discusses Meteor (Kaptchuk et al., 2021), Wu et al. (2024), and Zamir (2024) in Section 2 but provides zero experimental comparison. The central claim is "full capacity" — a genuine property — but we have no evidence whether achieving it comes at a cost in stegotext quality, detectability, or robustness compared to methods that embed fewer bits per token. Since these methods have different capacity-quality tradeoffs, some comparison is essential to assess practical value. The paper's claim of "high-quality results" (abstract) cannot be assessed in context without this.

- **Evaluation is insufficient to support core quality claims**: The evaluation uses exactly 3 source texts (selected from 1000 Reddit posts), generating 100 stegotexts per source, measured solely with LLM-assigned log-probability. There is no human evaluation of coherence, fluency, or meaning — the very properties the paper claims stegotexts possess. The paper cites Holtzman et al. (2019) on the disconnect between LLM probability and text quality, yet uses probability as its sole quality metric. This is additionally circular: the method is built from LLM probabilities, and LLM probabilities are used to validate it. A metric that is a monotonic function of the mechanism being evaluated cannot independently validate quality.

- **"Full capacity" framing is misleading for high-entropy messages**: "Full capacity" is presented as the key novelty, meaning the stegotext and hidden message have the same token length. But this is a token-length property, not an information-theoretic one. When the hidden text has high entropy (e.g., a hex hash), ranks cluster at high values and the stegotext becomes gibberish — as the paper itself demonstrates. The paper acknowledges this failure mode but does not quantify how quality degrades as a function of message entropy, leaving the operating envelope of "full capacity" underspecified. This matters because steganography is typically used for high-entropy secrets (keys, encrypted messages), precisely the case where the method fails.

### Minor

- **The Shibbolethian Theatre scenario requires transmitting the decoding key alongside the stegotext**: In the scenario (Act 4), the user "receives on the chatbot platform the censored answer to their query c, which includes the reasoning trace t and the actual answer s." Anyone who knows the Calgacus protocol (now published) can decode the hidden message from the transmitted pair (t, s). This is not a truly covert channel; it relies on the adversary being unaware of the protocol — security through obscurity. The paper's framing of this as "urgent" for AI safety is overclaimed. The scenario still has value as a demonstration of how knowledge can be embedded in aligned model outputs, but the claim that it enables "covertly" deploying an unfiltered LLM is too strong.

- **Security analysis is informal**: The O(d^|k|) brute-force bound assumes independent uniform tokens, but natural-language keys have far less entropy. The deniability claim rests on visual overlap in Figure 4 and one anecdotal example (Figure 15), with no quantitative analysis of what fraction of key-stegotext pairs yield plausible alternative decodings. The paper acknowledges this is an open question, which is fine, but the claims should be scaled accordingly.

- **Philosophical discussion extends beyond what the protocol establishes**: The "hallucination as lack of intent" framing and the LLM knowledge argument are interesting but follow from general properties of constrained LLM generation (like Oulipo-style constraints), not from Calgacus specifically. The protocol illustrates these points vividly but does not uniquely establish them.

## Nice-to-Haves

- Human evaluation of stegotext coherence (even a small-scale A/B test of real vs. stegotext pairs would dramatically strengthen or weaken the quality claims).
- Quantitative detectability analysis: if an adversary trains a classifier to distinguish stegotexts from real text, how well does it perform?
- Characterization of the entropy-quality tradeoff: a systematic study of how stegotext probability degrades as a function of hidden message entropy.
- Comparison with at least one existing LLM steganography method (e.g., Meteor) on shared metrics.

## Removed Points

- **"No comparative evaluation" as fatal**: Downgraded from fatal to major. The paper does establish the method works and has a unique property (full capacity), so the contribution is real even without head-to-head comparison. But the absence significantly limits practical assessment.
- **"Shibbolethian Theatre is structurally flawed" as fatal**: This is a valid concern but the scenario is presented as an illustrative application in a discussion section, not as a formal security protocol. Downgraded to minor. The core technical contribution (Calgacus itself) does not depend on this scenario being a truly covert channel.
- **Formatting and style nitpicks from Harsh Critic**: Removed per instructions.
- **"Missing parts" demanding proofs or formal modeling**: The paper explicitly declines formal steganographic modeling ("we will avoid building a palace on the sand"), which is a reasonable methodological choice. Formal security proofs for a protocol that uses natural-language prompts as keys would be extraordinarily difficult. Downgraded to nice-to-have.
- **Claim that deniability "only applies to sender under coercion, not detectability in transit"**: The paper frames deniability as sender's security under coercion (citing Canetti et al., 1997), which is the standard meaning. The criticism confuses deniability with covertness — these are different properties. Partially removed.
- **Strength claim about "philosophical reframing of hallucination"**: The philosophical discussion is interesting but extends beyond what the protocol uniquely demonstrates. Kept as a supporting strength but not as a core strength.

## Novel Insights

The paper's most distinctive insight is that rank-preserving token generation across LLM contexts naturally produces a steganographic "full capacity" channel — the same number of tokens in and out — which no prior method achieves. This is more than a trivial consequence of token-rank mapping: it creates a qualitative shift where the hidden and cover texts have symmetric length, making source and stegotext visually indistinguishable side-by-side. The low-entropy-token analysis revealing why stegotexts are systematically less probable than originals (rank-1 tokens "wasted" on high-entropy decisions) is a clean mechanistic explanation that simultaneously illuminates the method's operating envelope.

## Suggestions

- Restructure the "full capacity" claim to clearly distinguish token-length capacity (what the method achieves) from information-theoretic capacity (where high-entropy messages degrade quality), and provide a characterization of the entropy-quality curve.
- Add even a small-scale human evaluation (e.g., 50 text pairs, Mechanical Turk A/B test) to validate the claim that stegotexts are "meaningful" and "coherent" — this would significantly strengthen or appropriately weaken the paper.
- Scale back the Shibbolethian Theatre framing: present it as an illustrative threat scenario rather than a practical covert channel, and explicitly acknowledge that protocol publication makes the channel detectable.
- Compare with at least one baseline (e.g., Meteor) on shared metrics (stegotext log-prob, detectability) to contextualize the contribution.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Smaller, Weaker, Yet Better | 3OyaXFQuDl | 7.0 | Simple novel observation with limited but real experimental evidence. Stronger experimental grounding than Calgacus. Calgacus has a comparable novelty insight but weaker evaluation. |
| Multi-Exit ViT Token Reduction | gJeYtRuguR | 7.5 | Simple clever method with solid experiments. Calgacus is simpler and more novel conceptually but has much weaker empirical validation. |
| Kantian Prompting (UPAR) | kNm7TNIL6O | 5.0 | Interesting conceptual/philosophical framing loosely coupled to method, overclaimed significance. Analogous to Calgacus's philosophical discussion extending beyond what the protocol establishes. |
| Joint Image Compression/Steganography | Q00XEQxA45 | 3.67 | Poor evaluation and presentation in steganography. Calgacus is clearly superior — better-written, more novel, and more honest about limitations. |
| Sampling Watermarking | eKGEsFdpin | 3.67 | LLM text watermarking with limited robustness evaluation. Calgacus has a simpler and cleaner core idea but similarly thin evaluation. |
| Formal Causation Theory | lrQlLqQase | 5.67 | Strong conceptual contribution with limited empirical validation. Calgacus is comparable in the tradeoff between conceptual novelty and empirical weakness, though arguably more novel. |

Calgacus sits above the low-scoring steganography and watermarking papers (which are 2.8–3.67) because its core idea is genuinely novel, simple, and non-obvious, and the paper is well-written. It sits below the high-scoring papers (6–7.5) because the empirical evaluation is thin, no baseline comparison is provided, and several claims are overreaching. The strongest comparison is UPAR (5.0) — both have interesting conceptual framing with loose coupling between the philosophical discussion and the technical contribution, and limited empirical grounding — but Calgacus's core methodological contribution is more concrete and reproducible.

**Score: 5.0**

## Score and Decision

The paper presents a genuinely novel, simple, and interesting idea (rank-based full-capacity steganography) that is clearly described and reproducible. The mechanistic analysis of why stegotexts are less probable is insightful. However, the evaluation is thin (3 source texts, no baselines, no human evaluation, circular validation metric), the "full capacity" property is misleading for the high-entropy messages where steganography is most needed, and the AI safety scenario is overclaimed. The paper claims more than it establishes, but the core idea is real and worth publishing — these are addressable weaknesses for a revision.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>