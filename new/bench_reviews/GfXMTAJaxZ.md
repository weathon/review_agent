Now I have sufficient information to write the final review. Let me compile everything.

---

## Summary

This paper constructs computationally sound interactive argument systems for verifying general distribution properties. The central result (Theorem 1.1) shows that any property whose distance can be approximated in polynomial time admits a 4-message protocol with Õ(√N/ρ²) verifier sample complexity, communication, and runtime — optimal up to polylog(N) factors — under the collision-resistant hash function (CRH) assumption. The construction combines three novel components: (1) a succinct distribution-commitment scheme using hash trees that binds a prover to a valid distribution while supporting local openings of probabilities, CDFs, and quantile queries; (2) a verified distribution-oracle protocol that allows sublinear identity testing against the committed distribution; and (3) a "grain" string encoding that maps total variation distance to relative Hamming distance, enabling the use of existing interactive arguments of proximity (IAPs) for string languages. The result yields quadratic speedups over standalone (prover-less) tolerant testing for a broad array of properties where quasi-linear sample complexity is required in the standard setting.

---

## Strengths

- **Generality of Theorem 1.1**: Unlike prior unconditionally sound protocols limited to label-invariant or bounded-depth/bounded-space properties (Herman & Rothblum 2022/2023/2024), this result covers *any* poly-time decidable distribution property. This is stated clearly in Section 1.1 and is the core conceptual advance.

- **Near-optimal verifier sample complexity**: The Õ(√N/ρ²) verifier complexity matches the Ω(√N/ρ²) lower bound from Chiesa & Gur (2018) and Herman & Rothblum (2022), which extends to computationally sound argument systems (Section 1.1). This is a clean and tight result.

- **Tolerant verification at no asymptotic overhead**: The protocol tolerates distributions that are merely *close* to the property, not only exact members. This addresses what the paper correctly describes as a "notoriously hard" regime in the standalone setting (quasi-linear lower bounds for tolerant uniformity testing, entropy estimation, etc.; Raskhodnikova et al. 2009; Valiant & Valiant 2010).

- **Novel distribution-commitment scheme (Section 2.1)**: The hash tree construction — where internal nodes store both probability sums and hash-of-children labels, and the root's probability is pinned to 1 — is a genuinely new cryptographic primitive. The extraction-based security definition that derives a unique valid distribution Q̃ from any successful cheating prover's transcript is an elegant formalization.

- **Grain encoding bridging TV distance and Hamming distance (Section 2.3)**: Representing a distribution as a sorted string of mass-grains and applying error-correcting codes to tighten the TV-Hamming correspondence is the key algorithmic novelty enabling the reduction to IAPs. This is a clean and reusable idea.

- **Concrete, quantitative applications (Section 1.2)**: The applications to monotone distributions (Õ(√N/ε²) vs O(N/(log N · ε²))), k-junta distributions (Õ(√N/ε²) vs Õ(√N · k)), and shape-restricted properties (log-concavity, unimodality, convexity, etc.; quasi-linear lower bounds from Canonne et al. 2018) are specific and well-calibrated.

- **Verified distribution-oracle as an independently useful sub-protocol (Section 1.3)**: The lightweight sub-protocol that avoids PCP machinery and supports probability, CDF, quantile, and sampling queries is practically appealing and improves on prior unconditionally sound protocols for label-invariant properties in ρ dependence.

- **Honest comparison to prior work**: The paper is transparent about its trade-off — computational soundness under CRH in exchange for generality and optimal complexity — and accurately characterizes the gaps with Chiesa & Gur (2018), Herman & Rothblum (2022/2023/2024).

---

## Weaknesses

### Fatal
None.

### Major
None.

### Minor

- **The ρ > 1/√N restriction receives insufficient discussion.** Footnote 3 relegates this to a single sentence: "for ρ < 1/√N our protocol's communication complexity is at least linear." The regime ρ ≈ 1/√N is precisely the hardest parameter regime for many practical problems (distinguishing distributions close in statistical distance over large domains). The footnote's brevity may cause readers to under-appreciate the regime where the protocol degenerates. A short plot or formal statement comparing the protocol to the linear-communication baseline as ρ decreases through 1/√N would substantially clarify the result's practical scope.

- **The ERM application is derivative and thinly developed (Section 1.2).** The authors themselves acknowledge it "similarly to an application described by Herman & Rothblum (2024)" and the advance is extending their prior result from bounded-space/bounded-depth algorithms to any poly-time ERM. While this is a genuine improvement, it lacks a concrete instantiation with a specific learning algorithm and hypothesis class where the sublinear verification is practically compelling. Given that ERM is the most direct ML-facing application, a more developed treatment would strengthen the paper's case for its venue.

- **The PCP-induced prover overhead is unstated and potentially large.** The honest prover's runtime is described as poly(N, κ) — which, when unfolded through the PCP machinery, can involve very large polynomial degrees. Section 2.3 mentions "it may be possible to reduce this overhead using recent advances," but this is not quantified. For a reader assessing practical deployment (e.g., in the ERM setting), the gap between polynomial and, say, N³ or N⁵ is significant.

### Trivial
None that are not parsing artifacts.

---

## Nice-to-Haves

- **A worked example tracing the 4-message protocol on a concrete property** (e.g., uniformity testing over N = 256) would substantially improve accessibility for ICLR's audience, allowing readers to verify the message schedule and the verifier's decision procedure in the abstract.
- **Discussion of the non-interactive variant obstacle.** The open problem (Fiat-Shamir inapplicability due to private-coin structure) is mentioned briefly. A short analysis of *why* public-coin protocols for distributions are hard, or conditional impossibility results, would round out the paper.
- **A small empirical illustration** (even toy) of sample complexity vs. ρ tradeoffs would help the ICLR audience appreciate where the quadratic improvement materializes in practice.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Venue mismatch critique (Harsh Critic, §1):** The critic argues this is a "pure TCS paper" that should be at STOC/FOCS/CCC/TCC. This is overstated. ICLR regularly accepts pure theory papers with learning-theory connections (e.g., hardness of learning under symmetries, complexity-theoretic limits of SSMs). The ERM application, distribution testing for fairness/safety, and the connection to verified ML computations are legitimate ML-facing motivations. Removed as a fatal concern; retained only as the "thin ERM application" minor weakness above.

- **Core technical proof absent from main paper (Harsh Critic, §2):** The paper defers formal proofs to the appendix. Per review rules, weaknesses about missing appendix proofs are removed — the appendix exists in the original submission. The technical overview in Section 2 is a deliberate expository choice, not a deficiency.

- **Completeness of protocol not demonstrated in main text (Harsh Critic, §3):** The formal protocol composition is in the appendix. The main text's Section 2.3 gives a clear and correct high-level construction. The message count ("4 messages") follows from the IAP structure described in Section 2.3 (citing Kalai & Rothblum 2015). Removed as a strawman — the proof is deferred, not absent.

- **No experiments (Harsh Critic, "Missing Parts §1"):** For a theoretical TCS/cryptography contribution, lack of experiments is not a disqualifying flaw. The Harsh Critic's demand for "toy demonstration on N=256" is a nice-to-have for ICLR context, not a weakness. Moved to Nice-to-Haves.

- **Generic "clear structural decomposition" strength (Strength Finder):** This lacks specific section/equation citation beyond what the more detailed strengths already capture. Dropped as generic.

---

## Novel Insights

The paper's most genuinely novel conceptual insight — which the reviewer discussion did not fully foreground — is that the *identity tester's minimal oracle requirements* are precisely what the distribution-commitment scheme is designed to satisfy. By isolating that Goldreich's optimal identity tester needs only (i) point-probability answers and (ii) sampling access, and then constructing a commitment scheme that provides exactly these functionalities verifiably, the paper achieves a clean modular architecture where the cryptographic component and the statistical testing component compose without mutual contamination. This modularity also explains why the verified distribution-oracle is a genuinely reusable primitive: any sublinear algorithm that accesses a distribution through these four functionalities (probability, CDF, quantile, sampling) immediately inherits the verification wrapper. The grain encoding is then a second-level insight that extends this to *any* efficiently decidable property by converting the distribution-querying task into a string-querying task compatible with IAPs.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison |
|---|---|---|
| `/home/wg25r/review_agent/human_reviews/ARPrtuzAnQ.md` | **7.33** (Accept Spotlight) | Hardness of learning under symmetries — pure theory paper at ICLR with SQ lower bounds and experiments. Similar: pure TCS result with ML relevance, accepted. This paper's ML connection (ERM, distribution testing) is comparable; its technical novelty (grain encoding + commitment) is arguably greater. |
| `/home/wg25r/review_agent/human_reviews/DhdqML3FdM.md` | **7.00** (Accept Poster) | Complexity-theoretic limits of SSMs/Transformers — theory + experiments. Similar in being a TCS paper at ICLR; that paper has experiments while this does not, but this paper's main theorem is tighter and its primitives are more novel. |
| `/home/wg25r/review_agent/human_reviews/R2834dhBlo.md` | **6.67** (Accept Poster) | Neural Interactive Proofs — mixed theory/empirics, but reviewer 1 flagged limited technical contribution and reviewer 2 gave 10. This paper has much stronger and more complete technical contribution than R2834dhBlo and avoids the "limited contribution" criticism. |
| `/home/wg25r/review_agent/human_reviews/olOheQ0ZcK.md` | **5.75** (Reject) | Distance estimation for high-dimensional distributions via subcube conditioning — another theory+experiments paper on distribution testing. Rejected partly due to unclear ICLR audience fit and presentation concerns. This paper is better positioned (clearer ML motivation, cleaner theorem) though also has no experiments. |
| `/home/wg25r/review_agent/human_reviews/JNZ3Om6NPS.md` | **2.00** (Reject) | Low-quality conceptual paper about LLM limitations — no formal results, no technical substance. Serves as low anchor. The paper under review is vastly stronger. |

**Assessment:** This paper sits comfortably in the 7–7.5 range. Its weaknesses (thin ERM treatment, footnoted ρ restriction, unstated PCP overhead) are minor and do not threaten the central claims. It matches ARPrtuzAnQ in profile (pure theory, ML motivation, no experiments, strong lower bounds / protocol results) and exceeds DhdqML3FdM in technical novelty of the core construction. I place it at **7.0**, consistent with the center of the high-anchor cluster, and slightly below ARPrtuzAnQ to account for the less-developed ML applications compared to that paper's experiments.

**Originality:** High. The grain encoding → IAP reduction and the distribution-commitment scheme are novel.  
**Importance of research question:** High. Verified computing for statistical analyses is increasingly relevant to AI safety, fairness, and auditing.  
**Claims well supported:** Yes. The main theorem is clearly stated, the technical overview is correct, and prior-work comparisons are accurate.  
**Soundness of experiments:** N/A (theory paper).  
**Clarity of writing:** Strong. Modular structure, honest comparison to prior work, accessible overview section.  
**Value to research community:** Moderate-to-high. The verified distribution-oracle is an immediately usable primitive; the grain encoding idea is reusable.

**Recommendation: Accept (Poster)**

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>