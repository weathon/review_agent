Now I have all the information needed to write the review. Let me compose the final consolidated review.

## Summary

This paper constructs computationally sound interactive argument systems for verifying *any* distribution property that can be approximately decided in polynomial time, given the full description of the distribution. The main result (Theorem 1.1) shows that for any such property, there exists a 4-message protocol with near-optimal Õ(√N/ρ²) verifier sample complexity, communication, and runtime, assuming collision-resistant hash functions. This extends prior work that was restricted to label-invariant, bounded-depth, or bounded-space properties, and provides a quadratic speedup over standalone testing for many natural properties. A lightweight sub-protocol, the verified distribution-oracle, is also presented as an independently useful primitive.

## Strengths

- **Qualitative jump in generality.** Theorem 1.1 extends verifiable distribution properties from label-invariant (Herman & Rothblum 2022/2023) or bounded-depth/space (Herman & Rothblum 2024) properties to *all* polynomial-time decidable properties—a strictly richer class. This is the paper's central contribution and is clearly significant (Section 1.1).

- **Near-optimal verifier complexity with quadratic speedup.** The Õ(√N/ρ²) verifier sample complexity matches known lower bounds up to polylog factors (extending Chiesa & Gur 2018 and Herman & Rothblum 2022 to the computational soundness setting). For properties like tolerant uniformity testing that require Ω(N/log N) samples standalone, this yields a quadratic speedup (Section 1, "On the complexities" paragraph).

- **Novel technical approach: Merkle-tree distribution commitments + IAPs.** The hash-tree commitment scheme with probability-augmented node labels (Section 2.1) ensures binding to a *valid* probability distribution (probabilities sum to 1), going beyond standard Merkle-tree accumulators. The observation that identity testers only need oracle access to Q (not its full description) is a clean insight that enables the entire construction (Section 2, first paragraph).

- **Verified distribution-oracle as a reusable primitive.** Section 1.3 provides a lightweight sub-protocol supporting probability, CDF, quantile, and sampling queries at Õ(√N)/ε² complexity using only hash trees (no PCPs), giving improved ρ-dependence over prior unconditionally sound protocols for label-invariant properties.

- **Concrete applications demonstrating impact.** Section 1.2 gives specific applications—monotone distributions, k-juntas, log-concavity, shape-restricted properties, ERM verification—each showing concrete quadratic speedups over known standalone testers.

## Weaknesses

### Fatal
None.

### Major

- **Core soundness argument is only sketched in the body.** The composition of the distribution-commitment with the IAP is the central security argument for Theorem 1.1, but its formal proof is deferred to Appendix B. The body provides a one-sentence conclusion ("The binding property of the commitment scheme and the soundness of the IAP guarantee that the verifier will reject w.h.p.," line 168) and briefly notes the handling of aborting/incorrect provers (lines 136-140), but does not include a formal composition lemma. For a theory paper claiming a general result, this is a gap: a reader cannot verify the main theorem from the body alone. This is mitigated by the fact that (a) the outline is correct and standard hybrid arguments apply, (b) the paper is explicit about the appendix deferral, and (c) the building blocks (CRH-based commitment binding and IAP soundness) are individually well-understood. The concern is about rigorous verification, not conceptual error. This is a presentation/significance issue for a theory contribution more than a soundness issue, but it does mean the paper's central claim has not been fully verified in the published body.

- **Formal statement of the distance-preservation lemma is absent from the body.** The grain-based string representation X_D and the claim that when D is ε-far (in statistical distance) from property Π, then X_D is Θ(ε)-far in Hamming distance from any string that M' accepts, is stated informally (Section 2.3, line 165). The paper acknowledges that the converse direction requires error-correcting encoding (line 174) but does not provide a formal lemma giving the exact relationship between the distances. The hidden constant in Θ(·) directly impacts the final Õ(√N/ρ²) complexity bound, as it affects the IAP's distance parameter. A formal lemma in the body would allow readers to verify the claimed complexity without consulting the appendix.

### Minor

- **Extension to NP properties (Section 1.2, last paragraph) is under-developed.** The paper claims Theorem 1.1 extends to properties where approximate distance computation is in NP, requiring the prover to provide a witness. No formal statement or overhead analysis is given, leaving the reader uncertain about costs and limitations.

- **The honest prover's implementation from sampling access is only briefly addressed.** Line 59 states that "protocols are presented as if the honest prover has perfect knowledge of the distribution, but this idealized honest prover can be implemented by an honest prover that learns a sufficiently-accurate approximation." While Õ(N) samples suffice for learning D to sufficient accuracy by standard distribution learning results, the paper does not state this as a formal claim or give a concrete accuracy threshold. This is a minor gap rather than a major one, since the claim is standard and the Õ(N) · poly(1/ρ) prover complexity stated in Theorem 1.1 is plausible for this reason.

### Trivial
None.

## Nice-to-Haves

- Including at least formal lemma statements (without full proofs) for the composition soundness and distance-preservation properties in the body would significantly strengthen the paper for theory readers.

- A concrete worked example (e.g., tolerant uniformity testing) showing end-to-end parameters would help readers verify that the framework produces the claimed quadratic speedup for a specific well-studied property.

- Discussion of whether the private-coin requirement is inherent for general properties, or what goes wrong in attempting a public-coin construction, would strengthen the paper's contribution to the research agenda (the paper mentions this as open in Section 1.1).

## Removed Points

These points are flagged to be removed; treat them with caution.

- *"Evidential — Soundness of the IAP–commitment composition relies on omitted formal detail"* (Harsh Critic, Point 1): The concern about the composition proof being in the appendix is kept above as a Major weakness. However, the claim that the composition requires "careful handling of aborting provers" overstates the issue: the paper explicitly addresses aborting provers (lines 136-140), arguing that refusal to open or inconsistency leads to immediate rejection, which reduces to a mental experiment where all queries are answered by Q̃. This is standard and not a conceptual gap—only a formal verification gap.

- *"The honest prover's implementation is under-specified"* (Harsh Critic, Point 3): Downgraded from "critical" to Minor. The claim that Õ(N) samples suffice for learning D to sufficient accuracy is a standard result in distribution learning (standard empirical estimators achieve this). Theorem 1.1 states the prover's complexity as Õ(N) · poly(1/ρ), which is consistent with this standard fact.

- *Missing related works*: Not included per rules—cannot verify existence of uncited works.

- *Formatting/typo nitpicks*: Removed per rules.

- *Request for the general (non-η-grained) case*: The paper states "For simplicity, suppose that D is η-grained" (Section 2.3, line 170). The general case is a standard discretization detail. This is a presentation convenience, not a methodological gap—downgraded to Nice-to-Have.

## Novel Insights

The key insight driving the entire construction—that identity testers only need oracle access to Q (probability queries and sampling), not Q's full description—transforms an apparently communication-intensive protocol (prover sends all of Q) into one with polylogarithmic per-query communication. The distribution-commitment mechanism that enforces consistency across queries while also ensuring the committed object is a valid probability distribution (probabilities sum to 1) is a non-trivial refinement over standard Merkle-tree accumulators.

## Score Calibration

Comparing against calibration anchors:

- **High-scoring anchors (avg >7):** Privacy Amplification for Matrix Mechanisms (avg 7.5, Accept Spotlight), Optimality of Matrix Mechanism (avg 7.0, Accept Poster), Beyond WL (avg 8.5, Accept Oral) — these papers have strong theoretical contributions with near-optimal results. This paper's contribution is comparable in significance: a clean, general result extending prior work with near-optimal complexity.

- **Medium-scoring anchors (avg 4-6):** Crypto watermarking paper (avg 5.0, Reject) — this had key proofs deferred but also had questions about definitions. Our paper has a similar proof-in-appendix issue but a much stronger and clearer core result.

- **Low-scoring anchors (avg <3):** OXIIFZqiiN (avg 1.5, Reject) — clearly weak/LLM-generated. Our paper is far above this.

The paper's key strength is the generality and significance of Theorem 1.1, which is a clear advance over prior work. The main concern—deferred formal proofs—is the kind of issue that a theory venue would weigh heavily but that is less unusual at ICLR. The claims are plausible, the construction is well-motivated, and the technical approach is sound in outline. The missing formal statements in the body are a real but addressable gap (would be resolved by adding lemma statements + proof sketches). This paper clearly outperforms the medium and low anchors.

## Suggestions

- Add formal lemma statements for composition soundness and distance-preservation in the body, with brief proof sketches, even if full proofs remain in the appendix.
- Include one concrete worked example showing end-to-end parameters.
- Provide a brief argument (even one paragraph) for why the honest prover's Õ(N)·poly(1/ρ) sample complexity follows from standard distribution learning results.

MY FINAL SCORE: <pineapple>7</pineapple>
MY FINAL DECISION: <orange>Accept</orange>