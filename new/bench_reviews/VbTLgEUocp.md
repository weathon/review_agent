Now let me search for calibration papers to properly score this work.Now I have sufficient context to write the final review. Let me synthesize everything.

---

## Summary

Calgacus is a text steganography protocol that exploits autoregressive LLM generation to hide an arbitrary text *e* inside a different, stylistically steerable text *s* of exactly the same token length. The core mechanism is simple: record the rank of each token of *e* in the LLM's predicted distribution, then generate *s* from a secret prompt *k* by always selecting the token at that same rank position. The paper evaluates stegotext quality against a distribution of 1,000 real Reddit posts and discusses security, deniability, and a concrete AI-safety application (an "unaligned chatbot disguised as aligned"), ending with philosophical reflections on LLM hallucination and authorial intent.

---

## Strengths

- **Elegant, novel core mechanism.** The rank-preservation trick achieves a strict 1:1 token-length ratio between hidden message and stegotext — a property that prior LLM steganography methods (Meteor, Zamir, Wu et al.) do not cleanly deliver. The idea is minimal and immediately implementable with standard logit access.

- **Compelling illustrative examples.** The Caesar critique hidden in a boar recipe (Fig. 1, Fig. 13) and the pro-Caesar fake are vivid, memorable demonstrations of what the protocol can achieve. The Qwen3 8B Chinese-language example broadens the reader's intuition.

- **Transparent evaluation with a clear mechanistic explanation.** The comparison against 1,000 Reddit posts (Fig. 4) with the "low entropy token choice" analysis (Fig. 5) is a clean, honest diagnostic that both demonstrates the stegotexts fall within the real-text probability distribution *and* explains why they are systematically below-average. This is more informative than many watermarking papers that simply claim low perplexity.

- **Honest about scope.** The paper explicitly declines to build on unrealistic formal steganographic models, acknowledges hardware reproducibility concerns, the "abrupt ending" edge case, and the hash-produces-gibberish failure mode. This self-awareness is rare.

- **Well-written with genuine intellectual depth.** The discussion linking rank-constrained generation to Oulipo constraints, Dennett's intentional stance, and a novel "hallucination as lack of intent" framing is thought-provoking and substantively connected to the technical result.

---

## Weaknesses

### Fatal
*None.* The core existence claim — that rank-preservation produces a plausible same-length stegotext — is demonstrated and not materially undermined.

---

### Major

**1. Empirically undersized evaluation makes all quality/plausibility claims provisional.**
The entire quantitative evaluation rests on three selected Reddit posts (at µ, µ−2σ, µ+2σ of the log-probability distribution) and 100 stegotexts each. There is no systematic evaluation across different message lengths (all examples are truncated to 85 tokens), writing styles, domains, or LLMs. The paper claims "an entire article can be encoded and decoded" but never tests this; rank errors and the probability gap between original and stegotext may compound substantially over longer texts. Without reporting the fraction of stegotexts rated acceptable across a realistic corpus — a number the hash example suggests could be well below 100% — the central claim that the protocol is reliable is illustrated rather than established. The paper genuinely needs a large-scale failure-rate characterization to be a credible "protocol" paper rather than an interesting existence proof.

**2. Security analysis is informal to a degree that makes security claims unsupportable.**
The paper explicitly declines to adopt a formal security model ("we will avoid building a palace on the sand"), which is a defensible position. However, the paper then makes specific security claims — "no feasible way to recover the message," "prohibitive" brute-force, and deniability in the sense of Canetti et al. (1997) — that are not meaningfully supported by the informal analysis given. The O(d^|k|) upper bound on key search is acknowledged by the authors themselves to be irrelevant ("the attacker could reduce the search space"), and the key question of semantic search is left entirely open ("unclear and remains an open research question"). The deniability argument reduces to: some outlier prompts produce stegotexts whose probability is close to the original's. This does not constitute deniability in any technical sense; a coerced sender needs to *reliably* produce a plausible-seeming alternative decoding, but the paper does not analyze how often this is possible or under what conditions. The paper may still be interesting as a preliminary contribution, but it should either present formal analysis or clearly downgrade security claims to "informal observations."

**3. The Shibbolethian Theatre application has zero experimental validation.**
The paper's most impactful claimed contribution — that an unaligned LLM can be covertly shipped via the compliant responses of an aligned one — is presented entirely as a hypothetical script (Figures 11–12 show a gas-meter example, but only the surface answer *s* is shown, not the decoded unfiltered answer *u*, nor any test that the decoded output is coherent and useful). Critical assumptions are stated but not checked: bit-identical logits across company and user hardware (a non-trivial requirement the paper itself flags), the aligned model's reasoning trace *t* being available to the user in plaintext, and the decoded answer remaining coherent after being encoded through the aligned model's rank structure. Without a single end-to-end test on a real aligned/unaligned model pair, this application is a thought experiment rather than a demonstrated possibility. The paper leans heavily on it for AI-safety motivation.

**4. No detectability experiments against a committed adversary.**
The paper shows that LLMs can on average distinguish stegotexts from originals by raw log-probability, but does not test whether a trained classifier, a dedicated steganalysis tool, or even a simple threshold on per-token rank statistics could achieve reliable detection. The cross-model check with Phi-3 (Fig. 14) is qualitative. Without at least a precision/recall or AUC figure for a simple detector, the statement that some stegotexts are indistinguishable from real texts for LLM-based discriminators cannot be taken at face value.

---

### Minor

**5. "Full capacity" is defined in tokens, not in an information-theoretic sense; no comparison with prior methods.**
The paper positions "full capacity" as the defining differentiator versus Meteor, Zamir, Wu et al., but never compares on a shared metric (bits per token, quality vs. capacity tradeoff, or detectability vs. capacity). Token equality is model-specific: a text that is *m* tokens under LLaMA 3 tokenization may not be *m* tokens under a different tokenizer. The paper also notes that padding tokens must be appended for graceful termination, which already means effective hidden-message capacity per stegotext token can be less than 1. These caveats should be disclosed prominently alongside the headline claim.

**6. Failure rate not quantified.**
The hash example (§3) and Appendix A.1/A.5 hints illustrate that quality depends strongly on the entropy of *e* and the match between *e* and *k*. But no experiment estimates what fraction of randomly drawn (e, k) pairs produces a coherent stegotext by any standard. This is arguably the most operationally important number in the paper.

---

### Trivial

**7. The philosophical discussion (§4 on hallucination, Oulipo, Tacitus) is disproportionately long relative to the experimental backing.** The reframing of hallucination as "void of intention" is intellectually interesting but rests on no empirical measurement — it is an argument by illustration. Tightening these sections would improve calibration between claims and evidence.

---

## Nice-to-Haves

- **Human evaluation study** (even small-scale MTurk) asking annotators to distinguish stegotexts from real text would provide the most direct evidence for the core plausibility claim.
- **End-to-end Shibbolethian Theatre demo** with a real aligned/unaligned model pair (e.g., LLaMA-3-8B-Instruct as oLLM), showing the decoded *u* is coherent and useful.
- **Scaling analysis**: measure stegotext probability gap and decoding fidelity as token length increases from 85 to 500+.
- **Comparison table** with at least one prior method (e.g., Meteor) on capacity, stegotext quality (perplexity or human rating), and detection rate.
- **Sensitivity analysis** for logit perturbations — quantify how many tokens are corrupted by small floating-point changes, which governs practical deployability.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic — "Determinism makes deniability brittle / key-checking attack."** Valid in principle, but the paper's explicit scope is "informal security discussion" and it cites Canetti et al. as inspiration rather than claiming formal deniable encryption. The paper does note that "the attacker could check candidate (k', e') pairs," and the point is better incorporated as a strengthening of Weakness 2 (security analysis informal) rather than a standalone fatal flaw.

- **Harsh Critic — "O(d^|k|) brute-force bound is irrelevant because natural language prompts are sparse."** The paper acknowledges this directly and says the semantic-search approach "remains an open research question." This is honest, not an error, and is already captured in Weakness 2.

- **Harsh Critic — "Semantic side channels: content of s constrained by ranks of e."** Interesting theoretical point, but the paper already acknowledges the problem (§3.1: "the attacker could reduce the search space using the information revealed by s") and suggests the random-string insertion mitigation. The challenge is real but the paper is not ignoring it.

- **Human Finder — "Limited novelty of encoding technique, essentially a variant of arithmetic-coding steganography."** The rank-preservation framing differs mechanically from arithmetic coding (which maps a bitstring to a token sequence via cumulative distribution), and the specific same-length property is not a direct consequence of prior methods. This criticism does not survive verification against the paper's description.

- **Human Finder — "Lack of steganalysis tools as classifiers."** This is captured and kept under Weakness 4, above, in a more precise form.

- **Neutral Reviewer — "'Same length' in tokens, not human-perceivable length, could mislead readers."** The paper is transparent about tokenization throughout and the "same length" claim is always in the context of LLM tokens. This is a nit about framing, not a substantive error; folded into Weakness 5 above.

- **Harsh Critic — commercial/API deployment assumptions in Shibbolethian Theatre.** This goes to the application being underspecified (Weakness 3 above) but the specific point about "users already have the oLLM weights and are side-loading jailbreaks anyway" is scope creep — the paper's scenario is internally consistent and the scenario of a company deploying via an API is plausible.

---

## Novel Insights

The paper's sharpest genuine insight — one not fully appreciated in prior LLM steganography work — is the observation that standard autoregressive generation is itself an extreme constraint-satisfaction process (Section 4, "The constraint of chance"): generating text is equivalent to being forced at every step to honor an external rank prescribed by a random draw. The rank-preservation protocol makes this hidden structure explicit and exploitable. This reframes LLM generation not just as probabilistic sampling but as a channel that can transparently carry arbitrary rank sequences, which has implications beyond steganography for understanding what "intention" means in LLM-produced text. The connection to Oulipo and the "hallucination as lack of intention" reframing are philosophically fresh and could stimulate follow-up work on auditing and attribution of LLM-generated content.

---

## Suggestions

1. **Expand experiments to ≥ 200 diverse (e, k) pairs** across at least three text lengths and three domains; report the stegotext acceptance rate (e.g., fraction with log-probability within 1σ of the real-text mean).
2. **Run one end-to-end Shibbolethian Theatre experiment** on a real LLaMA-3-8B-Instruct pair; show the decoded answer and rate its coherence and fidelity.
3. **Calibrate security claims explicitly**: replace "no feasible way to recover" with "heuristically secure under the assumption that key search requires natural-language exhaustion," and downgrade deniability to "partial/probabilistic deniability."
4. **Add a comparison table** benchmarking vs. Meteor and Zamir (2024) on bits/token capacity, stegotext perplexity, and a simple probability-threshold detection rate.
5. **Sensitivity to logit perturbations**: run the protocol with logits perturbed by Gaussian noise at σ = {10⁻³, 10⁻², 10⁻¹} and measure the fraction of tokens decoded correctly, to quantify the hardware-reproducibility risk.

---

## Score and Decision

**Calibration:**

| Calibration paper | Topic | Scores | Decision |
|---|---|---|---|
| OD-Stega (IQafqgqDzF) | LLM text steganography, more technical depth, weaker novelty | 5,3,3,3 → avg 3.5 | Reject |
| Plausibly Deniable Encryption (7suavRDxe8) | LLM-based deniable encoding, richer security analysis than Calgacus | 8,5,3,5,3 → avg 4.8 | Reject |
| Hidden in Plain Text (urQi0TgXFY) | LLM steganographic collusion, broader empirical evaluation | 6,3,5,6 → avg 5 | Reject |
| CipherChat (MbfAK4s61A) | LLM safety bypass, 11 safety domains evaluated, actual attacks demonstrated | 6,5,8,8 → avg 6.75 | Accept poster |

**Positioning:** Calgacus has more conceptual originality and better writing than OD-Stega (3.5 avg), but a thinner empirical foundation than all comparators. Its security analysis is intentionally less formal than the Plausibly Deniable Encryption paper (4.8 avg, rejected). It lacks the systematic experimentation that earned CipherChat (6.75 avg) acceptance. The mismatch between the paper's rhetorical ambition (urgent AI safety implications, "radical decoupling of text from authorial intent," formal deniability) and its actual empirical support (3 source texts, no human study, zero end-to-end validation of the main application) places it solidly in rejection territory, but somewhat above OD-Stega in raw novelty and presentation. The concept is compelling enough to merit a major revision cycle rather than dismissal.

**Overall assessment:** *Originality*: High — the rank-preservation trick for same-length steganography is genuinely new. *Importance of research question*: High — covert LLM channels are practically significant. *Claims well-supported*: Weak — the existence proof works, but capacity, security, and safety claims substantially outrun the evidence. *Soundness of experiments*: Poor — 3 source texts, no human study, no trained detector, no end-to-end application test. *Clarity of writing*: Good to excellent. *Value to research community*: Moderate — as a concept paper with stronger experiments this would be a solid contribution; as currently evaluated it is premature.

**Score: 4.5 — Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>