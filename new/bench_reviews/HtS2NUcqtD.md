Now let me run calibration searches in parallel to score the paper.Now I have enough information to write the final consolidated review.

---

## Summary

This paper proves that the backward gradient of tensor attention training can be computed in almost-linear time $n^{1+o(1)}$, matching the forward computation complexity established by Alman & Song (2024b). The authors derive a closed-form gradient expression (Lemma 4.1), develop novel tensor algebraic techniques (Facts 5.4, 5.5, Lemma 5.6), and prove a matching SETH-based hardness lower bound (Theorem 6.3) showing the bounded-entries assumption is tight. The work completes the final gap in the complexity picture for tensor attention, as summarized in Table 1.

---

## Strengths

- **Closes a genuine open problem**: Prior work (Alman & Song 2024b) established almost-linear forward computation but left the backward pass open. This paper fills the last entry in Table 1, completing the complexity picture for tensor attention training — a natural and meaningful theoretical milestone.

- **Novel tensor algebraic techniques with independent utility**: Facts 5.4 (swap rule: $(A_1 \otimes A_2)(W_1 \otimes W_2) = (A_1 W_1) \otimes (A_2 W_2)$), Fact 5.5 (distribution rule), and Lemma 5.6 (reducing $\mathcal{T}_{\text{mat}}(d, n^2, d)$ to $\mathcal{T}_{\text{mat}}(d, n, d)$ via Hadamard product decomposition) are provably not needed for the forward pass or matrix attention backward. These tools are genuinely novel and may have independent applications in tensor computation.

- **Matching upper and lower bounds**: Theorem 5.2 achieves $n^{1+o(1)}$ at $B = o(\sqrt[3]{\log n})$, and Theorem 6.3 proves (under SETH) that even slightly weakening this threshold makes truly subcubic computation impossible. This sharp complexity transition is a clean and satisfying theoretical result.

- **Clear algorithm with per-step complexity annotations**: Algorithm 1 provides pseudocode with explicit complexity labels at each step, making the claimed total complexity transparent. Figure 4's computational graph is unusually informative for a theory paper.

- **General formulation**: Definition 3.8 uses separate input matrices $A_1, \ldots, A_5$ rather than requiring a shared input, covering both self-attention and cross-attention as special cases (noted in Remark 3.6).

---

## Weaknesses

### Fatal
None.

### Major

- **The $d = O(\log n)$ assumption is practically implausible, and Remark 5.3 obscures rather than justifies this.** Theorem 5.2 (and Algorithm 1) require $d = O(\log n)$. For $n = 2 \times 10^6$ (the Gemini 1.5 Pro figure cited in Remark 5.3), this caps $d$ at roughly 21. Standard transformer attention heads use $d = 64$–$128$ as a fixed architectural constant, independent of sequence length. Remark 5.3 attempts to justify the assumption by noting that "model training uses half-precision floating-point format, e.g., the bit number is 16." This conflates two entirely separate quantities: the *feature dimension* $d$ (which determines model capacity and appears in matrix sizes) and the *bit-width* per scalar entry (which determines numerical precision). A 16-bit float representation with $d = 128$ does not satisfy $d = O(\log n)$. The remark is a category error and should not be presented as justification for practical relevance. The paper's abstract claim that results "establish the feasibility of efficient higher-order transformer training and may facilitate practical applications" is significantly overclaimed given that the assumption excludes all current production transformer architectures. The assumption is *theoretically* motivated (it is necessary by Theorem 6.3), but this should be framed as a fine-grained complexity boundary, not a practical claim. Revising the framing would strengthen the paper considerably.

- **The gap between "approximate gradient computation" and "training" is unaddressed.** The paper's stated goal is *tensor attention training*, but the core result (Theorem 5.2) is that a single approximate gradient step can be computed in $n^{1+o(1)}$ time with $\ell_\infty$ error $\leq 1/\text{poly}(n)$. Multi-step gradient descent with approximate gradients has well-known convergence implications that depend on the relationship between approximation error, step size, and the loss landscape. No convergence analysis (not even an appeal to standard results on inexact gradient methods) is provided. The conclusion states "Future work can perform empirical evaluations," which confirms this gap. The paper establishes the efficiency of a sub-routine, not the convergence of a training algorithm. This should be accurately reflected in the abstract and introduction.

### Minor

- **The $B = o((\log n)^{1/3})$ assumption on matrix products is not empirically grounded.** Definition 3.9 applies the bound to products $\|A_i X_i\|_\infty$, not to the raw input matrices. While the hardness result establishes theoretical tightness, the paper does not discuss whether standard transformer training (with weight initialization, layer normalization, etc.) would satisfy these $\ell_\infty$ bounds in practice. Since the paper makes practical claims, at least a brief discussion of the empirical plausibility of this assumption would be valuable.

- **Motivational gap with GPT-4o and Project Astra.** The introduction cites these systems to motivate tensor attention (Section 1), but neither is known to use the specific formulation of Definition 3.5. The argument slides from "multi-modal systems exist" to "therefore tensor attention as defined here is important" without establishing that deployed systems use or approximate this particular architecture. The citation of these systems in the motivation is aspirational rather than evidentiary.

### Trivial

- **Notation overload on $\otimes$.** The symbol $\otimes$ is used for the full Kronecker product (Definition 3.1) and, per Remark 3.6, also refers to the column-wise product (Definition 3.2) in Definition 3.5. While the paper defines $\oslash$ for the column-wise product elsewhere, the overloaded usage of $\otimes$ in the main definition creates confusion that could be resolved with more consistent notation.

---

## Nice-to-Haves

- A brief discussion of what happens to the complexity when $d$ is a constant (e.g., $d = 64$) independent of $n$, making explicit where the $n^{1+o(1)}$ bound degrades.
- Extension to $p$-way ($p > 2$) tensor attention to match the three-modality (audio/vision/text) systems cited as motivation.
- A small-scale empirical measurement of $\|A_i X_i\|_\infty$ under standard initialization to assess practical plausibility of the bounded-entries assumption.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Theorem 4.3 is not a real theorem"** (harsh critic). The criticism that calling the naive $\Omega(n^3)$ lower bound a "theorem" inflates its weight is a pure presentation nitpick. The paper explicitly states it is a straightforward lower bound; calling it a theorem is conventional in complexity theory.

- **"The proof sketch is too sparse to evaluate"** (harsh critic). The paper's appendix contains complete proofs (Appendix E.6, F.2, etc.); the sparse sketch in Section 5.1 is standard for ICLR-format theory papers. The parser strips appendices; this is not an author deficiency.

- **"The hardness reduction's $\log^{11}(n)$ overhead factor is curious and not discussed"**. This is a minor technical note that does not affect the main asymptotic result, and the full proof is in the appendix. It is not a substantive weakness.

- **Reproducibility concerns** (implicit in requests for worked numerical examples and full algorithmic trace). These would be nice-to-haves but are non-standard expectations for a fine-grained complexity theory paper.

---

## Novel Insights

The most genuinely novel observation in the reviews is the conflict between the paper's theoretical framing and its practical ambitions. The paper's own hardness result (Theorem 6.3) demonstrates that the $d = O(\log n)$ and $B = o((\log n)^{1/3})$ assumptions are *necessary* for subcubic tractability — meaning there is a fundamental complexity-theoretic barrier precisely at the boundary of practical architectures. This could itself be highlighted as a negative result: real-world transformers are, by necessity of SETH, in the hard regime for this type of gradient approximation. This reframing would be more honest and arguably more impactful than the current "this enables practical training" framing.

---

## Suggestions

1. **Rewrite the abstract and introduction** to accurately characterize the result as a fine-grained complexity feasibility statement rather than a practical training claim. Explicitly state that $d = O(\log n)$ is a theoretical requirement whose practical interpretation is open.
2. **Fix Remark 5.3** to remove or correct the conflation of feature dimension $d$ with floating-point bit-width. Instead, acknowledge that the assumption is theoretically tight but that bridging to practical architectures is future work.
3. **Add a paragraph (or citation)** addressing convergence of gradient descent under the $1/\text{poly}(n)$-approximate gradient oracle. Even a pointer to inexact gradient descent results would close the logical gap between "gradient sub-routine is efficient" and "training is efficient."
4. **Unify the $\otimes$ notation** to avoid using the same symbol for both the full Kronecker product (Def. 3.1) and the column-wise product (Def. 3.2). Consider using $\oslash$ consistently everywhere for the column-wise product.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Fast RoPE Attention (similar topic, rejected) | `AozPzKE0oc.md` | 4.8 | Similar algorithmic domain but had a critical algorithmic error invalidating its core claim; the paper under review has no such error |
| PolySketchFormer (similar topic) | `YkCjojDG3l.md` | 5.0 | Polynomial sketching for attention; similar scope to this paper |
| Transformers + Chain of Thought expressivity | `NjNGlPh8Wh.md` | 7.5 | Also proves tight upper+lower complexity bounds for transformer-related question; accepted |
| Streaming ℓp regression with matching bounds | `Kpjvm2mB0K.md` | 8.0 | Complete complexity characterization with matching bounds (upper+lower); accepted Spotlight |
| One-pass streaming attention (low) | `rKMz6cDE7W.md` | 2.3 | Poor technical novelty, weak writing, impractical assumptions; clearly weaker than paper under review |

**Assessment relative to anchors:** The paper under review is clearly above the low anchor (2.3 — weak novelty, poor writing) and above the medium anchors (4.8–5.0 — either algorithmic errors or thinner contributions). Its structure — a natural open problem, genuine technical novelty in tensor algebraic tools, and matching upper/lower bounds — most closely resembles the accepted papers `NjNGlPh8Wh.md` (7.5) and `Kpjvm2mB0K.md` (8.0).

However, two factors pull the score down relative to those: (1) the paper is more incremental (extending a known result from forward to backward), and (2) the practical framing is materially misleading (Remark 5.3's conflation of $d$ with bit-width). The $NjNGlPh8Wh$ paper (7.5) had similarly limited practical scope but was more foundational in establishing a new expressiveness framework. The paper under review is a useful and honest complexity contribution once its framing is correctly calibrated, but the overclaimed practical relevance is a genuine issue that human reviewers would likely penalize.

Final score: **6.5** — borderline accept. The core theoretical contribution (novel tensor algebraic tools enabling the first almost-linear backward pass for tensor attention, with a matching hardness result) justifies acceptance at a learning theory venue. The overclaimed framing and missing convergence analysis are real but fixable weaknesses that do not invalidate the contribution.

**Originality:** Moderate-high — novel technical tools for tensor algebra, though incremental extension of prior work.
**Importance:** Moderate — completes a natural open problem in fine-grained complexity of attention.
**Claims supported:** Mostly yes for the core complexity results; overclaimed for "practical training."
**Experimental soundness:** N/A (pure theory paper).
**Clarity:** Good presentation overall; notation overload is a minor issue.
**Value to community:** Solid — provides tools for tensor computation that may have broader utility.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>