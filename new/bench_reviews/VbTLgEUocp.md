Now I have all the information needed. Let me compose the final review.

## Summary

The paper presents Calgacus, a protocol for hiding an arbitrary text inside another plausible text of the same token length using LLM token-rank preservation. The core idea is elegant: tokenize the secret text, record the rank of each token under the LLM's probability distribution conditioned on preceding tokens, then generate a cover text from a secret prompt by selecting the rank-matching token at each step. The paper demonstrates the method using Llama 3 8B, analyzes why stegotexts have lower probability than originals (low-entropy token choices), and discusses safety implications including a scenario where an unfiltered LLM's answers are encoded within an aligned LLM's responses.

## Strengths

- **The rank-preserving same-length encoding is genuinely novel.** Unlike prior LLM steganography methods (Meteor, Wu et al., Zamir) that sacrifice capacity for plausibility, Calgacus achieves stegotext of exactly the same token length as the secret message. The three-step recipe (Section 3, Figure 3) is simple enough to reproduce in minutes and requires only access to logits—no training or auxiliary models.

- **The low-entropy token choice analysis is a genuine insight.** Section 3 and Figure 5 explain *why* stegotexts are less probable than originals despite rank preservation: rank-1 tokens are "wasted" on high-entropy choices. The right panel of Figure 5 quantifies that rank-1 mismatches account for most of the probability drop. This is non-obvious and correctly derived.

- **The "Shibbolethian Theatre" scenario is a concrete and consequential safety application.** Section 4 describes a protocol where a company can deploy an unfiltered LLM by encoding its answers within compliant responses from an aligned model (Figures 11–12). This makes the threat model precise and immediately actionable, going beyond abstract safety concerns.

- **Cross-model robustness of the probability gap.** Figure 14 shows the same probability shift between real and fake texts holds when using Phi-3 3.8B rather than the generating model, indicating detection is possible even without knowing the encoding model.

- **Empirical evidence that stegotexts fall within the real-text probability distribution.** Figure 4 shows stegotext probabilities (from three representative original texts) overlap with the distribution of 1000 real Reddit posts, providing a necessary (if insufficient) condition for plausibility.

## Weaknesses

### Fatal
None.

### Major

- **No human evaluation of stegotext quality.** The paper's central claim is that stegotexts are "plausible," "coherent," and "meaningful" (Abstract; Introduction, p.13; Section 3). The sole quantitative evaluation is LLM log-probability overlap (Figure 4). The paper itself acknowledges the difficulty of measuring meaningfulness (Section 3: "Measuring the meaningfulness of a text is a longstanding linguistic challenge, and arguably an ill-posed problem"), then proceeds to use the one metric that cannot capture semantic coherence, topic drift, or the kind of subtle wrongness a human reader would flag. High LLM probability does not guarantee that text reads as natural to humans. The cherry-picked examples in Figure 1 are illustrations, not evidence. The hash example (Section 3) demonstrates the method can produce "broken" text, but there is no characterization of failure rate across a representative sample. Without human evaluation—or at minimum established automated metrics like MAUVE or human-likeness classifiers—the core claim that stegotexts are plausibly meaningful remains unestablished.

- **No experimental comparison with any prior steganography method.** The paper cites Ziegler et al. (2019), Meteor (Kaptchuk et al., 2021), Wu et al. (2024), and Zamir (2024), but provides zero experimental comparison on quality, detectability, or security tradeoffs. The headline differentiator is "full capacity," but without showing the quality cost of this property, the reader cannot assess whether this is a real advance. Full capacity is not inherently desirable if it comes at unacceptable quality costs—a possibility the hash example already hints at. Every prior method makes a capacity-quality-security tradeoff; the paper does not characterize Calgacus's position in this space relative to any alternative.

### Minor

- **Security and deniability analysis is thin.** The brute-force bound O(d^|k|) is trivially derived. The paper acknowledges that natural-language structure of keys could reduce the search space but dismisses it with "inserting a simple random string in k is enough to nip it in the bud" (Section 3.1) without analysis of how much randomness is needed or its effect on stegotext quality. Deniability is supported by a single example (Figure 15) and the observation that some stegotexts have probabilities comparable to the original—an existence claim, not a quantification. This matters because the paper's safety framing depends on these properties being more than anecdotal.

- **Narrow evaluation scope.** The quantitative evaluation (Figure 4) tests only 3 original texts, each producing 100 stegotexts. While the selection at µ, µ±2σ of the real-text distribution is methodologically reasonable, 3 texts is a very narrow base for general claims about the method's effectiveness across diverse inputs.

- **The "symmetry" claim is slightly overstated.** The Introduction states that same token length "prevents one from establishing at first sight which text is authentic" (p.13). Two texts with the same token count can have different character counts and visual lengths, so this symmetry is approximate. The claim is qualitatively right but stated too strongly.

### Trivial

- The informal dig at "reviewer 2" in the soundness definition (Section 3: "that is a difficult position to hold even for reviewer 2") is unprofessional and should be removed.

## Nice-to-Haves

- A human evaluation study (even small-scale, N=50–100) with pairwise discrimination (real vs. stegotext) would substantially strengthen the core claim.
- A gallery of stegotexts across diverse inputs—including failure cases—would let readers calibrate their own quality assessment beyond what log-probability plots provide.
- Comparison with at least one prior method (e.g., Meteor) on quality-capacity tradeoffs would contextualize the contribution.
- Quantification of deniability: what fraction of random keys yield plausible decoy messages?
- Empirical validation of the unaligned chatbot scenario end-to-end.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The philosophical discussion inflates the paper"** (Harsh Critic): The paper explicitly frames its contribution as raising questions about text, intent, and LLM knowledge. The discussion about hallucinations as "void of intention," the Oulipo connection, and the intentionality analysis are part of the paper's contribution, not padding. Whether one finds them valuable is a matter of taste, but the paper is clear about what it aims to do. Moved to nice-to-have to reduce discussion length if desired.

- **"Missing appendices/proofs"** (Harsh Critic references Appendices A.1, A.2, A.5): The parser strips appendices from all papers; they exist in the original submission. Cannot criticize their absence.

- **"Reproducibility concerns about identical logits on different GPU architectures"**: The paper itself acknowledges this limitation explicitly (Section 3: "sender and receiver must run the chosen LLM under identical conditions, performing the same approximations and obtaining identical logits. This may be a challenge when using different GPU architectures") and cites Shanmugavelu et al. (2024). This is a known limitation, not an unacknowledged flaw.

- **"Attack model underspecified—multiple stegotexts with same key, LLM scoring candidate keys"**: These are interesting attack vectors but are beyond what the paper claims to analyze. The paper frames security as a preliminary discussion, not a formal security proof. Demanding a complete attack taxonomy is scope creep.

- **"The paper lacks a rigorous formal steganography framework"**: The paper explicitly and honestly states in Section 2 that it avoids formal models because of their unrealistic assumptions ("we will avoid building a palace on the sand, and not frame our method in a formal model of steganography"). This is a deliberate choice, not an oversight.

## Novel Insights

The low-entropy token choice analysis reveals a fundamental asymmetry in rank-preserving steganography: the frequency of rank-1 tokens in natural text (~40%) is far below the probability mass that rank-1 tokens typically carry (>95% in low-entropy contexts), meaning that rank preservation systematically "wastes" the most probable continuations. This explains why stegotexts are detectable by LLMs despite being rank-preserving, and suggests that future methods might improve quality by biasing rank selection toward lower ranks in high-entropy contexts.

## Suggestions

- Run a simple Amazon Mechanical Turk study: present 50 real Reddit posts and 50 stegotexts; ask annotators to classify which are real. Report discrimination accuracy. Even a small study would transform the paper's evidential basis.
- Select 5–10 diverse secret texts (not just µ, µ±2σ of the same distribution) and include failure cases with analysis of what text properties predict quality. This would address the "when does it fail?" question directly.
- Compare with Meteor on at least one shared metric (e.g., bits per token, text quality via a standard metric) to situate Calgacus in the capacity-quality landscape.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| PMark (high) | EhDgP69DJG | 7.0 | Strong theoretical + empirical work with baselines; this paper has far less empirical rigor |
| CoNoCo (high) | 8s5jBVybhQ | 7.0 | Thorough experiments, real-world validation; this paper's evaluation is much thinner |
| Early Signs of Steganographic Capabilities (medium) | q4qxtaKVAU | 6.0 | Systematic evaluation of LLM steganography; this paper's idea is more novel but evaluation is weaker |
| Invisible Safety Threat: Steganography (medium) | 6cEPDGaShH | 6.0 | Empirical demonstration on multiple models; this paper has a cleaner idea but no human eval or baselines |
| Few Samples with LM Guidance (medium-low) | vKSSZHTdNP | 4.67 | Novel idea with limited evaluation; comparable pattern, this paper's idea is more elegant |
| LIMI: Less is More (medium-low) | Jee2Q7qK0s | 4.50 | Overclaimed results with tiny evaluation; this paper is more honest about limitations |
| Catch-22: Pareto Frontier (low) | pAeEzS4LwS | 2.67 | Confused theoretical framing; this paper is clearer and more honest |
| House G.P.T. (low) | vr36DTbxT9 | 3.0 | Artificial model organisms; this paper's method works on real text |

This paper sits between the 4.5 and 6.0 anchors. The idea is genuinely more novel than the 6.0 steganography papers (which evaluate existing capabilities rather than proposing a new protocol), but the evaluation is substantially weaker—no human evaluation, no baseline comparison, only 3 source texts. The low-entropy analysis is a real contribution that partially compensates. Compared to the 4.5 papers (novel idea, limited evaluation), this paper is stronger because the idea is cleaner and more consequential. I place it just below the borderline at **5.0**.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>