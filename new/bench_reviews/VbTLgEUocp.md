Now I have enough context from the calibration reviews and the paper itself to compose a thorough final review.

## Summary

The paper presents *Calgacus*, a steganographic protocol that uses an LLM to encode an arbitrary secret text within another plausible text of the same token length. The method works by recording the rank of each token in the secret text under the LLM's probability distribution, then generating a new text from a secret prompt by selecting tokens at those same ranks. The paper evaluates stegotext quality using LLM-assigned probabilities on Reddit posts, discusses security properties including deniability, and explores broader implications for AI safety and the philosophy of authorial intent in text.

## Strengths

- **Novel and conceptually striking "full capacity" property.** The rank-preserving generation approach distinguishes Calgacus from all prior LLM steganography methods (Meteor, Zamir 2024, Wu et al.), which sacrifice capacity for quality or security. The equal-length property is philosophically significant—it means no length-based signature exists to distinguish original from stegotext. This is a genuine conceptual contribution to the field.

- **Elegant and parsimonious method.** The protocol is essentially one insight (encode ranks, decode by rank-preserving generation from a secret prompt) with no training, optimization, or complex machinery. This makes it easy to understand, implement, and reproduce.

- **Insightful analysis of the probability gap.** The explanation of why stegotexts are systematically less probable than originals—via "wasted rank-1s" on high-entropy choices (Section 3, Figure 5)—is a clear technical contribution that explains a non-obvious phenomenon and connects LM internals to observable behavior.

- **Thought-provoking philosophical discussion.** The redefinition of hallucinations as "void of intention" rather than factual error, the connection to Oulipo constraints, and the broader argument about the decoupling of text from authorial intent elevate the paper beyond a purely technical contribution. The Caesar/boar recipe example in Figure 1 is vivid and powerfully illustrates the stakes.

- **Practical efficiency.** The demonstration that 8B open-source models suffice for encoding and decoding on commodity hardware in seconds makes the method accessible and the threat concrete.

## Weaknesses

### Major

- **Security and deniability claims are made without formal analysis and overstate what is established.** Section 3.1 makes claims such as "no feasible way to recover the message" and that the protocol provides "deniability," but provides no formal threat model, security definition, or proof sketch. This is the same fundamental deficiency that reviewers identified in the closely related "Plausibly Deniable Encryption with LLMs" paper (7suavRDxe8), where Reviewer 3 noted: *"I would expect a paper on a cryptography topic—or really any topic in information security—to provide a concrete threat model or security model...Without one, it is unreasonable to make the sort of security claims that the paper makes, as all such claims are relative to some security model."* While Calgacus explicitly chooses to forgo formal analysis ("we will avoid building a palace on the sand"), it then proceeds to make categorical claims about security and deniability that exceed what informal reasoning supports. The deniability claim—built on the observation that some stegotexts achieve probabilities comparable to their originals—is a category mistake if applied in the cryptographic sense of Canetti et al. (1997), which requires indistinguishability of transcripts, not mere plausibility of alternative messages under a bogus key. The attacker model is also underspecified: the paper oscillates between relying on model secrecy ("without the knowledge of the precise LLM...the attacker has no feasible way") and key secrecy, but never defines what the attacker knows, can compute, or can observe. The claim that "inserting a simple random string in k is enough to nip [key-search attacks] in the bud" is unanalyzed—no quantification of how much entropy is needed, or how adding randomness to k affects stegotext quality.

- **Evaluation is narrow and does not substantiate the generality claims.** The paper draws conclusions about typical behavior from only 3 source texts (at μ, μ±2σ of 1000 Reddit posts), fixed at 85 tokens, using a single model (Llama 3 8B). No human evaluation validates that LLM probability is a good proxy for perceived text quality; no systematic failure-rate characterization across diverse input types, lengths, or domains is provided. The abstract claims "even modest 8B open-source LLMs are sufficient to obtain high-quality results" and "a message as long as this abstract can be encoded and decoded locally on a laptop in seconds," but the empirical evidence covers only a narrow slice. This is not a minor appendix problem—it is a core gap between claim and evidence for a method whose "full capacity" property is its headline contribution.

- **No steganalysis or detectability evaluation under even moderate adversarial conditions.** The only "detection" tool considered is raw LM log-probability comparison, which is a naive baseline. Under Kerckhoffs' principle—the standard assumption in steganography that the adversary knows the protocol—a dedicated adversary could train a classifier or use more powerful models to detect statistical artifacts. As Reviewer 2 of OD-Stega (IQafqgqDzF) noted, existing steganalysis tools can detect many methods, and "whether the method proposed in the paper can achieve similar safety performance is an important metric." The paper acknowledges that LLMs can uncover a distinction between originals and fakes on average, but does not evaluate whether this signal can be systematically exploited. Since "hiding the existence of a hidden message" is central to steganography, this is a substantive gap.

- **No comparison with prior LLM steganography methods.** Meteor (Kaptchuk et al., 2021), Zamir (2024), and Wu et al. (2024) are cited but never empirically compared. Without benchmarking on capacity, quality, and detectability, the reader cannot assess whether "full capacity" comes with unacceptable trade-offs in quality or security, or whether it is a genuine advantage.

### Minor

- **The O(d|k|) notation for brute-force complexity is ambiguous.** Section 3.1 states "An upper bound on the difficulty of this problem is O(d |k|)," which, taken literally, is polynomial rather than the exponential O(d^{|k|}) that the surrounding argument requires ("A naive brute-force attack is therefore prohibitive, even for very short keys"). The intended meaning is clearly exponential, and this may be a rendering issue, but the notation as written is incorrect for the claim being made.

- **The "Shibbolethian Theatre" scenario is illustrative but overpresented.** The scenario of smuggling unaligned LLM responses through aligned model outputs is creative and thought-provoking, but the paper does not grapple with practical obstacles: the user must replicate the exact LLM inference (including floating-point behavior) on identical hardware, and any stochastic post-processing (sampling with non-deterministic seeds, content filters, paraphrasing) breaks the deterministic rank-preservation. The scenario is more a proof-of-concept existence claim than an "urgent" threat as framed in the abstract.

- **The "full capacity" property requires qualification for out-of-distribution inputs.** The paper acknowledges (via the hash example and limitation notes) that highly unpredictable inputs produce degenerate stegotexts, but this caveat does not appear in the abstract or introduction, where the claim reads as universal. The dependence on e being "well-predicted" by the LLM is a significant boundary condition that should be stated upfront.

## Nice-to-Haves

- **Human evaluation of stegotext quality** (e.g., annotators rating coherence/naturalness) would substantiate the central claim that stegotexts are "coherent and plausible" to humans.

- **Comparison with at least one baseline** (Meteor, Zamir 2024) on capacity × quality × detectability trade-offs would clarify whether full capacity is a net advantage.

- **Trained steganalysis detector evaluation** (even a simple classifier) would ground the detectability claims.

- **Scaling experiments** beyond 85 tokens (to 500, 1000+) would validate the generality claim.

- **Quantification of deniability rates** (what fraction of random keys produce plausible decoy messages?) would strengthen the informal security argument.

## Removed Points

- **O(d|k|) as a fundamental mathematical error**: The harsh reviewer treats this as a "not just a typo" structural error. In context, the surrounding text clearly requires exponential complexity, and this is almost certainly a rendering artifact for O(d^{|k|}). Downgraded to Minor notation issue rather than Fatal error.

- **The scenario "underplays realistic defenses" in the sense of being a weak attack**: The Shibbolethian Theatre is presented as a conceptual illustration of what Calgacus *could* enable, not as a fully specified attack. Criticizing it for not engaging with countermeasures is scope creep beyond the paper's stated purpose.

- **Demanding a formal cryptographic security proof**: The paper explicitly states it avoids formal models and frames its contribution differently. Requesting a formal proof is asking the paper to be a different kind of work entirely. However, the paper *should* moderate its claims to match its level of analysis, which is a legitimate criticism (kept as Major).

- **Pushing practical details to appendices**: The harsh reviewer complains that algorithm specification is in appendices. For a method this simple (a 3-step recipe in the main text), this is acceptable presentation.

- **Formatting issues and reference nitpicks**: Removed per instructions.

- **Concerns about Canetti reference citation format**: Trivial.

- **Demanding comparison with provably secure steganography methods**: The paper makes different claims than those methods (which sacrifice capacity for provable security). A comparison would be informative but is not strictly required for the paper's contribution.

## Novel Insights

The paper's most novel insight is the reframing of hallucination as a "void of intention" rather than a factual error. This connects LLM behavior to literary traditions (Oulipo constraints) and to the broader philosophical project of attributing intent to text. The observation that any coherent LLM-generated text is already solving an extreme constraint satisfaction problem—the "constraint of chance" from sampling—and that Calgacus merely swaps one constraint source for another—is original and clarifying. It suggests that the boundary between "intended" and "unintended" text is far more porous than commonly assumed, with direct implications for AI safety discourse.

## Suggestions

- Reframe Section 3.1's claims as informal observations rather than security guarantees. Remove the term "deniability" (which has a precise cryptographic meaning) or qualify it explicitly as heuristic, not formal, deniability.
- Add a systematic failure-rate characterization: for what proportion of inputs (across diverse domains and lengths) does Calgacus produce coherent output? This single experiment would dramatically strengthen (or qualify) the generality claims.
- Scale the evaluation to at least 500+ tokens and include diverse text types (code, formal writing, dialogue) to test the boundary conditions of the method.
- Replace or augment LM probability with at least one additional quality metric (e.g., perplexity under a different model, or a small human evaluation) to validate that probability is a reasonable proxy for human-judged coherence.

## Score and Decision

**Calibration anchors:**
- *OD-Stega* (IQafqgqDzF): LLM steganography with no baselines, no steganalysis → 5,3,3,3 → Reject
- *Plausibly Deniable Encryption* (7suavRDxe8): Security claims without formal threat model → 8,5,3,5,3 → Reject  
- *Hidden in Plain Text* (urQi0TgXFY): Emergence of steganography in LLMs, evaluation issues → 6,3,5,6 → Reject
- *Diffusion-Stego* (Ve9GKnDNDQ): Thin method, no formal analysis → 5,3,3,1 → Reject

This paper shares the same core deficiencies as the "Plausibly Deniable Encryption" paper (security/deniability claims without formal backing) and OD-Stega (narrow evaluation, no baselines), both of which were rejected. However, Calgacus has a genuinely more novel conceptual contribution (full capacity + the philosophical discussion), and the paper is unusually well-written and thought-provoking. It sits between the rejected steganography papers (mostly 3-5) and would-be-accepted papers with stronger technical methodology. The conceptual novelty and philosophical contribution earn it a modest lift above the OD-Stega range, but the gap between claims and evidence—particularly on security and generality—remains substantial.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>