## Summary

This paper studies the backward-pass gradient computation for tensor attention, a higher-order generalization of matrix attention that captures multi-view correlations. The authors derive a closed-form gradient (Lemma 4.1), propose Algorithm 1 to compute it in almost-linear $n^{1+o(1)}$ time under bounded entries (Theorem 5.2), and complement this with a hardness result (Theorem 6.3) aimed at showing the assumption is tight. This fills the last gap in Table 1—there was no prior almost-linear backward algorithm for tensor attention.

## Strengths

- **First almost-linear backward algorithm for tensor attention.** Theorem 5.2 and Algorithm 1 give a constructive $n^{1+o(1)}$-time algorithm for the Approximate Tensor Attention Loss Gradient Computation problem (Definition 3.9). This is novel: Alman & Song (2024b) solved the forward problem, but the backward pass was open (Table 1).
- **Novel tensor algebraic identities.** Facts 5.4–5.5 and Lemma 5.6 develop non-trivial structural results for propagating low-rank approximations through the backward computational graph. The “swap rule” and “distribution rule” are technically distinct from what is needed for matrix attention or tensor attention forward.
- **Closed-form gradient exposing the bottleneck.** Lemma 4.1 gives a clean characterization $\frac{d\text{Loss}(x)}{dx} = \text{vec}(A_1^\top F(x)(A_2 \otimes A_3))$ that precisely identifies the $n \times n^2$ matrix $F(x)$ as the obstacle, which motivates the low-rank approximation strategy.

## Weaknesses

### Fatal
None.

### Major
- **Theorem 6.3 is internally inconsistent with the text’s interpretation of tightness.** The theorem states hardness for $B = \Theta(\sqrt[\gamma]{\gamma(n)} \cdot \log n)$, which for $\gamma(n)=\omega(1)$ is $\Theta(\log n)$. However, the text immediately following the theorem (Section 6.2) describes a “sharp complexity transition” at $B = \Theta(\sqrt[\gamma]{(1+\gamma)\log n})$, matching the forward hardness regime of Lemma 6.2 and directly contradicting the theorem statement. Because the algorithm requires $B = o(\sqrt[3]{\log n})$, the bound $\Theta(\log n)$ is exponentially far in the exponent from the algorithmic regime and does **not** establish tightness as claimed. Since “necessity and tightness” is one of the three headline contributions, this is a serious flaw in the main text presentation. The text’s description suggests the intended bound is the tight one, so this appears to be a theorem-statement typo rather than a wrong proof, but as written the main text does not present a valid hardness result matching the claimed transition.

### Minor
- **Abstract and conclusion slightly overclaim relative to the analyzed setting.** Definition 3.8 formulates training as an $\ell_2$ regression problem over an unstructured $X \in \mathbb{R}^{d \times d^2}$, not the standard cross-entropy or next-token prediction loss used in LLMs. The abstract’s phrase “feasibility of efficient higher-order transformer training” is broader than the specific regression formulation analyzed. Narrowing the claims to “$\ell_2$ regression with tensor attention” or explicitly arguing why the gradient structure carries over would strengthen the paper.
- **$\otimes$ notation is overloaded across standard and column-wise Kronecker product in the main text.** Definition 3.1 defines $\otimes$ as standard Kronecker, while Definition 3.2 defines $\oslash$ as column-wise Kronecker. Yet Definition 3.5, Remark 3.6, and Fact 5.4 all use $\otimes$ for the column-wise product, relying on dimensional annotations for disambiguation. This hurts readability and verifiability.

### Trivial
- **Algorithm 1 line 19 uses an operation $\odot$ not defined in the main text.** The paper defines Hadamard product as $\circ$ (Section 3), yet Algorithm 1 uses $\odot$ and defers to Definition B.3 in the appendix. The apparent dimensional mismatch ($d \times n^{o(1)}$ matrices yielding $d \times d^2$) is unresolved in the main text.
- **$F(x)$ is defined only in the appendix (Definition C.6).** While standard for space-constrained theory papers, a one-sentence definition in the main text would help verify Lemma 4.1 without flipping to the appendix.

## Nice-to-Haves
- A synthetic experiment on small instances verifying that Algorithm 1 achieves the stated $\ell_\infty$ error within the predicted runtime would build confidence in the hidden $n^{o(1)}$ constants.
- A dimension table listing every intermediate shape in Algorithm 1 would improve clarity given the overloaded notation.
- Discussion of whether the polynomial-approximation framework extends to cross-entropy or other standard training losses.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **“Bounded-entry assumption is incompatible with claims of practicality.”** The assumption $B = o(\sqrt[3]{\log n})$ is standard in the Alman & Song fine-grained complexity line (Lemma 5.1 cites Alman & Song 2024b with exactly this bound). Calling it “practical” because of fp16 bit precision is a weak justification, but the assumption itself is not author error—it is inherited from the literature. This is a minor stretch, not a core flaw.
- **“Missing experiments.”** This is a theory paper; experiments are not required for an algorithmic complexity result.
- **“Missing appendix proofs / deferred definitions.”** The parser strips appendices; proofs deferred there exist in the original submission.
- **Typos, grammar, formatting artifacts.** Per hard rules, these are parser issues, not author errors.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Fix Theorem 6.3 to match the bound described in the text ($B = \Theta(\sqrt[\gamma]{(1+\gamma)\log n})$ or equivalent) so that the hardness result actually supports the claimed tightness.
- Add a remark in the abstract/introduction clarifying that the results apply to the $\ell_2$ regression formulation in Definition 3.8, and discuss whether/how the gradient structure extends to cross-entropy.
- Disambiguate Kronecker notation: consistently use $\otimes$ for standard Kronecker and $\oslash$ for column-wise Khatri-Rao, or add a footnote reminding readers which is intended in each definition.

## Score and Decision

**Calibration comparisons:**
- **High anchor:** `tPEwSYPtAC.md` (avg 6.75, Accept spotlight) — had many theorem typos and incorrect constants but core results were believed correct; our paper has fewer theorem issues but one is in a headline claim.
- **Medium anchor:** `Ww9rWUAcdo.md` (avg 5.50, Accept poster) — had unclear theorem statements and strong assumptions but solid core theory; comparable to our paper’s mix of novel algorithm and presentation flaws.
- **Low anchor:** `GqGoa44obw.md` (avg 4.50, Reject) — had a core definition that was mathematically backwards (inconsistency defined as consistency); our paper’s issue is a theorem-statement typo, not a backwards definition.

Our paper sits between the medium and high anchors. The algorithmic upper bound (Theorem 5.2) is solid and fills a real gap. The hardness lower bound appears to suffer from a theorem-statement typo rather than a wrong proof, because the text explicitly states the intended tight bound. Relative to the medium anchor (`Ww9rWUAcdo`), our core contribution is more clearly defined and the flaw is more localized; relative to the high anchor (`tPEwSYPtAC`), our presentation is cleaner but our one theorem typo is more consequential because it sits in a core claim. A score of **5.5** reflects this borderline-but-leaning-acceptable position: the contributions are real, but the inconsistency in Theorem 6.3 must be resolved.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>