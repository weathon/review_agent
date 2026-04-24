## Summary

This paper introduces computationally sound interactive argument systems for verifying essentially any distribution property that can be approximately decided in polynomial time. Its central result (Theorem 1.1) gives a four-message protocol whose verifier sample complexity, communication, and runtime are all $\tilde O(\sqrt{N}/\rho^2)$ (where $\rho$ is the tolerance gap), matching information-theoretic lower bounds up to polylogarithmic factors. The construction rests on a novel succinct distribution-commitment primitive (a Merkle tree over probabilities with additive consistency checks) that forces an untrusted prover to answer queries according to a single valid distribution without linear communication.

## Strengths

- **General, near-optimal theorem.** Theorem 1.1 handles *all* poly-time decidable distribution properties, dramatically expanding the scope beyond prior unconditionally sound protocols that were limited to label-invariant or bounded-depth/space properties (Section 1.1). The verifier complexities are tight up to polylogarithmic factors, and the paper carefully contrasts its improvements in $\rho$-dependence with Herman & Rothblum (2022; 2023; 2024) (Section 1.1, “Comparison to unconditionally sound protocols”).
- **Novel cryptographic primitive of independent interest.** Section 2.1 constructs succinct distribution-commitments from collision-resistant hash trees. The scheme binds the prover to a full global distribution via a short digest, supports local openings of size $\mathrm{polylog}(N)$, and comes with an extraction-based security guarantee that the opened probabilities specify a valid distribution. This distinguishes the work from prior accumulators that commit only to sets.
- **Concrete quadratic speedups on natural properties.** Section 1.2 instantiates the framework on monotone distributions, $k$-juntas, log-concavity/convexity, and ERM verification, showing settings where standalone tolerant testers require quasi-linear $\Omega(N/\log N)$ samples but the protocol verifies with only $\tilde O(\sqrt{N}/\varepsilon^2)$ samples and communication.
- **Clean modular reduction.** Section 2.3 defines a grained, sorted string representation $X_D$ of a distribution and shows that total-variation distance translates to relative Hamming distance. This allows the protocol to delegate the property check to an off-the-shelf interactive argument of proximity, cleanly separating the distribution-testing component from the cryptographic proof machinery.

## Weaknesses

### Fatal
None.

### Major
None.

### Minor
- **Abstract oversimplifies the tolerance regime.** The abstract states that “if the distribution is at statistical distance $\varepsilon$ from having the property, then the verifier rejects with high probability,” using a single parameter $\varepsilon$ and advertising complexity $\tilde O(\sqrt{N}/\varepsilon^2)$. This elides the two-parameter tolerant nature ($\epsilon_c$-close vs. $\epsilon_f$-far with gap $\rho = \epsilon_f-\epsilon_c$) and could mislead readers into thinking the protocol remains sublinear for arbitrarily small $\rho$. Footnote 3 and Theorem 1.1 correctly restrict the sublinear regime to $\rho \in (1/\sqrt{N},1)$, but the abstract should flag this more carefully.
- **ERM sketch lacks a clarifying sentence.** Section 1.2 describes verifying an approximate risk minimizer on a dataset $X$ of $N$ labeled examples, but it does not explicitly state that $X$ is viewed as an empirical distribution over a domain of size $N$ and that “$h$ is an approximate risk minimizer” is decidable in $\mathrm{poly}(N)$ time given the full empirical distribution (which requires that the benchmark class $\mathcal{H}$ and loss support poly-time risk minimization). One sentence would close the gap.
- **Label-invariant distance estimation is unexplained in the main text.** Section 1.3 asserts that the verifier can “obtain a very good approximation for $Q$’s probability histogram by drawing samples from $Q$ together with their probabilities” and then “estimate $Q$’s distance from the property.” Because the histogram has $\mathrm{poly}(N)$ buckets, it is not obvious how the verifier performs this estimation in sublinear time. A brief pointer to the relevant algorithmic step or a citation to prior work that achieves this would strengthen the exposition.
- **String representation parameters are left implicit.** Section 2.3 says the full construction uses a high-distance error-correcting code to obtain a “tight relationship” between TV distance and relative Hamming distance, but it does not reassure the reader that the resulting alphabet size and string length do not destroy the $\tilde O(\sqrt{N})$ verifier complexity. As sketched, the length is $(1/\eta)\cdot \ell$ with $\ell = O(\log N)$ and $\eta = 1/\mathrm{poly}(N)$, so the length is $\mathrm{poly}(N)$ and $\log n = \mathrm{polylog}(N)$; the IAP overhead therefore stays polylogarithmic. Stating this explicitly in the main text would close a distracting gap.

### Trivial
None.

## Nice-to-Haves

- A fully worked concrete example (e.g., tolerant verification of monotonicity over $\{0,1\}^n$) with explicit parameter settings, message counts, and a comparison to the standalone tester of Rubinfeld & Vasilian (2020).
- A brief quantification of the honest prover’s PCP-induced overhead and pointers to which doubly-efficient IOP constructions would be most compatible with the distribution-encoding reduction.
- A short discussion of whether the verifier’s need to send distribution samples (precluding a public-coin protocol) is inherent for general distribution properties.
- Exploration of additional natural subclasses beyond label-invariant properties where the prover can run in near-linear time without invoking full PCP machinery.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **“Prover complexity and PCP overhead not quantified.”** The paper explicitly notes that the generic IAP induces polynomial prover overhead, points to recent advances that might reduce it (Reingold et al., Ben-Sasson et al., Ron-Zewi & Rothblum), and defers further details to Appendix A.3. Since appendices are stripped by the parser, this criticism is rooted in missing material that exists in the original submission. The core claim is about verifier efficiency, and the honest prover is already guaranteed to run in $\mathrm{poly}(N,\kappa)$ time.
- **“Public-coin barriers not discussed.”** The paper explicitly flags the public-coin question as open, explains why the current protocol is not public-coin, and cites Herman (2024) for a deeper discussion. Demanding an impossibility result is scope creep.
- **“Missing concrete instantiation / worked example” and “prover-efficient protocols for broad subclasses.”** These are presentation suggestions and future-work directions, not technical weaknesses.
- **Any criticisms about missing appendix proofs or missing deferred material.** The parser strips appendices; these sections exist in the original submission and should not be penalized.

## Novel Insights

The paper’s distribution-commitment scheme is more than a technical ingredient: it provides a general template for converting global probabilistic objects into locally verifiable cryptographic commitments. By cleanly separating the task of approximating the unknown distribution (via identity testing) from the task of checking a property (via an IAP over a string representation), the authors highlight a modular design pattern that could extend to other massive structured objects—such as high-dimensional densities or graphical models—where the verifier cannot afford to read the full description. This modularity, together with the extraction-based binding guarantee, suggests that similar succinct commitments may find use in verifiable learning and auditing beyond the distribution-testing setting.

## Suggestions

- Add a clarifying clause to the abstract (or its last sentence) noting that the $\tilde O(\sqrt{N}/\varepsilon^2)$ complexity is sublinear only when the tolerance gap is $\rho = \Omega(1/\sqrt{N})$.
- In Section 1.2, insert one sentence stating that the dataset $X$ is treated as an empirical distribution over $[N]$ and that poly-time ERM implies the “approximate risk minimizer” property is decidable in $\mathrm{poly}(N)$ time given the full empirical distribution.
- In Section 1.3, add a citation or one-sentence pointer explaining how the verifier estimates the distance to a label-invariant property from a sublinear amount of histogram information (e.g., by approximating the fingerprint and invoking a known sublinear distance estimator).
- In Section 2.3, include an explicit remark that the string length is $\mathrm{poly}(N)$, so the IAP’s query complexity and verifier runtime remain $\mathrm{polylog}(N)$.

## Score and Decision

**Calibration comparison.** I retrieved papers across three score bands to anchor the evaluation.

*Low (avg $\le$ 4):*
- `/home/wg25r/review_agent/human_reviews/pq3RANvCZC.md` (avg 3.00, Reject): an oversimplified and poorly motivated distribution-testing paper; our work is far stronger in motivation, technical depth, and positioning.
- `/home/wg25r/review_agent/human_reviews/LAsMFAg4Zf.md` (avg 3.75, Withdrawn/Reject): a federated-learning defense lacking theorem guarantees; our claims are well-supported.
- `/home/wg25r/review_agent/human_reviews/evDSvZBFRP.md` (avg 4.00, Reject): a robustness-verification paper with missing definitions and unclear proofs; our definitions and constructions are complete.
- `/home/wg25r/review_agent/human_reviews/dxJKLozjQl.md` (avg 3.00, Reject): an MMD-based data-valuation method with limited contribution.
- `/home/wg25r/review_agent/human_reviews/p79lnC36CO.md` (avg 2.00, Reject): a calibration-diagnostics paper with weak methodology.
- `/home/wg25r/review_agent/human_reviews/qKfzDc8Qiv.md` (avg 4.00, Withdrawn): a rare-event robustness framework; not directly comparable.

*Medium (avg around 5):*
- `/home/wg25r/review_agent/human_reviews/olOheQ0ZcK.md` (avg 5.75, Reject): a distribution-distance estimation paper with polarized reviews and weak experiments; as a theory paper our work avoids this pitfall and delivers a complete protocol.
- `/home/wg25r/review_agent/human_reviews/Vz5HgVwcdu.md` (avg 5.00, Reject): a theoretical complexity paper rejected for poor writing and missing related work; our paper is clearly written and well-situated.
- `/home/wg25r/review_agent/human_reviews/Qyile3DctL.md` (avg 5.00, Withdrawn): an LLM collaborative-verification paper; not directly comparable.
- `/home/wg25r/review_agent/human_reviews/7suavRDxe8.md` (avg 4.80, Reject): a cryptographic plausibly-deniable-encryption paper; not directly comparable.
- `/home/wg25r/review_agent/human_reviews/oSuVEv4X7w.md` (avg 4.75, Withdrawn): a closed-loop verifiable code-generation paper; not directly comparable.

*High (avg $\ge$ 6):*
- `/home/wg25r/review_agent/human_reviews/Kpjvm2mB0K.md` (avg 8.00, Accept Spotlight): a strong streaming-algorithms paper with matching upper and lower bounds; our paper is similarly rigorous and introduces a novel primitive, though it does not prove new lower bounds, placing it just below this anchor.
- `/home/wg25r/review_agent/human_reviews/KS8mIvetg2.md` (avg 7.50, Accept oral): a test-set contamination detection paper with exact guarantees and strong experiments; our theoretical contribution to distribution testing is of comparable depth and significance.
- `/home/wg25r/review_agent/human_reviews/3f5PALef5B.md` (avg 7.50, Accept oral): a neural theorem-proving paper with verified lemmas; different area but comparable in technical maturity.
- `/home/wg25r/review_agent/human_reviews/VGQugiuCQs.md` (avg 7.50, Accept Spotlight): streaming fair clustering; less relevant but shows the quality bar for sublinear algorithm papers.
- `/home/wg25r/review_agent/human_reviews/Ip6UwB35uT.md` (avg 7.00, Accept Poster): conditional distribution testing with FDR control; related but more applied.
- `/home/wg25r/review_agent/human_reviews/R2834dhBlo.md` (avg 6.67, Accept Poster): neural interactive proofs with mixed theory/experiment focus; our result is more focused and mature theoretically.
- `/home/wg25r/review_agent/human_reviews/QCDdI7X3f9.md` (avg 6.50, Accept Poster): empirical two-sample distribution test; our theoretical framework is comparably impactful.
- `/home/wg25r/review_agent/human_reviews/RsJwmWvE6Q.md` (avg 6.75, Accept Poster): optimal sketching for residual error; related sublinear techniques.
- `/home/wg25r/review_agent/human_reviews/EeqlkPpaV8.md` (avg 6.75, Accept Poster): adaptive log-concave sampling; related distribution-testing literature.
- `/home/wg25r/review_agent/human_reviews/sZQRUrvLn4.md` (avg 6.40, Accept Spotlight): graph property testing with GNNs; different domain.
- `/home/wg25r/review_agent/human_reviews/yQcFniousM.md` (avg 6.50, Accept Poster): private nearest-neighbor search; different area.
- `/home/wg25r/review_agent/human_reviews/lPJUQsSIxm.md` (avg 6.33, Accept Poster): private neural inference via FHE; different area.
- `/home/wg25r/review_agent/human_reviews/mUMvr33FTu.md` (avg 6.25, Accept Poster): private transformer inference; different area.
- `/home/wg25r/review_agent/human_reviews/I4YAIwrsXa.md` (avg 6.25, Accept Poster): LLM formal proof training; different area.
- `/home/wg25r/review_agent/human_reviews/z9j7wctoGV.md` (avg 6.00, Accept Poster): kernel two-sample test for generated text; empirical counterpart.
- `/home/wg25r/review_agent/human_reviews/9Z0yB8rmQ2.md` (avg 6.00, Reject): automated theorem proving; different area.

**Relative placement.** The paper under review is substantially above the low and medium anchors: it is well-motivated, technically sound, and clearly written. It sits comfortably among the high-scoring theoretical papers. It is slightly below the very strongest theory anchors (e.g., Kpjvm2mB0K at 8.00) because it does not prove new lower bounds and leaves a few exposition gaps, but it matches the quality of the 7.5-band papers by delivering a broad, near-optimal theorem backed by a novel primitive and careful comparisons to prior work.

MY FINAL SCORE: <pineapple>7.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>