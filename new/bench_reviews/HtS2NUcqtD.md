Now let me run calibration searches to score the paper appropriately.Now I have sufficient calibration data to write the final consolidated review.

---

## Summary

This paper establishes that the backward gradient of tensor attention can be computed in almost-linear time $n^{1+o(1)}$ under a bounded-entries assumption ($B = o(\sqrt[3]{\log n})$, $d = O(\log n)$), matching the complexity of its forward computation. It provides a closed-form gradient formula (Lemma 4.1), an efficient Algorithm 1 exploiting polynomial approximation and novel tensor algebraic identities, and a SETH-based conditional lower bound (Theorem 6.3) showing the assumption is tight. This fills the last cell in a systematic 2×2 grid (Table 1) of forward/backward × matrix/tensor attention complexity results.

---

## Strengths

- **Fills a precisely identified gap (Table 1):** The paper cleanly identifies and fills the "Backward/Tensor" cell, completing the efficiency picture for tensor attention alongside Alman & Song (2024a, 2024b). The motivating question is narrow, well-defined, and the answer is affirmative and non-obvious.

- **Genuine technical novelty in tensor algebraic identities (Section 5.2):** Facts 5.4 (swap rule for tensor-matrix product), 5.5 (distribution rule), and Lemma 5.6 (reducing $\mathcal{T}_{\mathrm{mat}}(d, n^2, d)$ to $\mathcal{T}_{\mathrm{mat}}(d, n, d)$ via Hadamard decomposition) are non-trivial contributions specific to the backward direction and are not consequences of the forward machinery in Alman & Song (2024b). The challenge of maintaining low-rank structure through the softmax–Hadamard coupling in the backward pass is demonstrably harder than the forward case.

- **Tight bidirectional complexity result:** Theorem 5.2 (upper bound) and Theorem 6.3 (SETH-based conditional lower bound) match at the $B = \sqrt[3]{\log n}$ threshold, providing a sharp characterization of when efficient gradient computation is and is not achievable. This is methodologically complete, mirroring the gold standard in fine-grained complexity.

- **Clear computational graph (Figure 4):** The explicit forward/backward flow diagram makes the gradient structure inspectable and aids verification of the algorithm's correctness.

---

## Weaknesses

### Fatal
None.

### Major

- **The "training" claim systematically oversells the actual contribution.** The paper's abstract, title, introduction, and conclusion all claim to "establish the feasibility of efficient higher-order transformer training," but Theorem 5.2 only proves that *a single gradient evaluation* $\tilde{g}$ satisfying $\|\tilde{g} - g\|_\infty \leq 1/\mathrm{poly}(n)$ can be computed in $n^{1+o(1)}$ time. There is no analysis of: (i) whether iterating such approximate gradients converges to an optimum or even a stationary point; (ii) how approximation errors accumulate across multiple steps; or (iii) whether the bounded-entries assumption $B = o(\sqrt[3]{\log n})$ on intermediate activations can be enforced or preserved across training steps. The gap between "one backward pass can be approximated efficiently" and "training is feasible" is non-trivial. Section 7 honestly acknowledges that "empirical evaluations" are future work, but the core convergence gap is not acknowledged at all. The paper should either provide a convergence analysis for approximate-gradient optimization, or demote its primary claim to "efficient gradient computation" and explicitly flag multi-step training convergence as an open problem.

### Minor

- **Remark 5.3 conflates two distinct assumptions.** Theorem 5.2 requires *both* (a) entries represented in $O(\log n)$ bits, *and* (b) $d = O(\log n)$. Remark 5.3 defends the practicality of these assumptions by citing $n = 2 \times 10^6$ (Gemini 1.5 Pro) and 16-bit precision — but this only validates assumption (a). For $n = 2 \times 10^6$, assumption (b) limits $d \lesssim 21$, while standard transformer head dimensions are 64–128 and full hidden dimensions are 768–8192. The remark should separately address both constraints and acknowledge that the $d = O(\log n)$ requirement is a genuine restriction of the current technique, not just a conventional computational model assumption. This is the standard limitation of the polynomial-approximation approach inherited from Alman & Song (2024b).

- **The hardness reduction proof sketch omits key quantities.** The main text (Section 6.2) states that the $\log^{11}(n)$ overhead and the accuracy threshold $\epsilon = O(1/(\log n)^4)$ follow from an "interpolation and integral" argument, but does not explain where these specific quantities come from, or why the $\epsilon$ threshold matches what is needed. The formal proof is in the appendix. A brief explanation of these quantities in the main text would make the tightness claim more transparent.

- **Chain rule from $\partial \mathrm{Loss}/\partial X$ to individual weight-matrix gradients is unaddressed.** The paper asserts (end of Section 3.2) that "it is easy to get the gradients of the weight matrices" from $\partial \mathrm{Loss}/\partial X$ and $\partial \mathrm{Loss}/\partial Y$. However, the Kronecker-structured parameterization $X = W_Q(W_{K_1} \otimes W_{K_2})^\top$ makes this a non-trivial chain rule step. Practitioners implementing the method need these formulas or a reference.

### Trivial
None.

---

## Nice-to-Haves

- A synthetic experiment (e.g., $n = 100$, $d = 5$) comparing exact vs. approximate gradients would empirically verify the closed-form derivation and demonstrate approximation quality with concrete $n^{o(1)}$ constants.
- A discussion of whether the $d = O(\log n)$ constraint can be relaxed or whether a conditional lower bound shows that $d = \omega(\log n)$ requires a fundamentally different approach.
- A brief discussion of whether natural regularization techniques (e.g., attention logit clipping as in LLaMA-3) can enforce the bounded-entries assumption across training steps, bridging the gap toward practical training.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "The $d = O(\log n)$ assumption is a critical methodological gap / invalidates practical claims"** — retained only as a Minor weakness (the Remark 5.3 conflation), not as a Fatal issue. This is a standard assumption in fine-grained complexity results; the directly comparable papers DDNFTaVQdU and AozPzKE0oc have the same assumption. The criticism is real but minor in the context of a learning-theory complexity paper.

- **Harsh Critic: "Tightness of hardness reduction is not fully transparent — proof in appendix"** — the formal proof is in Appendix F.2. Per the hard rules, criticisms about missing appendix content are removed. The concern about the proof sketch being insufficiently detailed is retained as a Minor point.

- **Harsh Critic: "No empirical validation" as a Major weakness** — the paper is explicitly submitted to the learning theory primary area, explicitly defers empirical work to future research in Section 7, and is best evaluated against its community's standards (pure theory complexity papers like DDNFTaVQdU and 7BESdFZ7YA, which were accepted without experiments). Moved to Nice-to-Haves.

- **Strength Finder: "Novel tensor algebraic techniques"** — retained as a genuine, specific strength.

- **Strength Finder generic claims about the problem being important** — removed; "tensor attention can overcome representational limitations of matrix attention" is motivation, not a paper-specific strength.

---

## Novel Insights

The most significant observation emerging from the reviews is that the tensor algebraic identities required to propagate low-rank structure through the backward pass (Facts 5.4, 5.5, Lemma 5.6) are qualitatively harder than those needed for the forward pass, reflecting a genuine structural difference between backward and forward computation in tensor attention. In particular, Lemma 5.6 avoids a $\mathcal{T}_{\mathrm{mat}}(d, n^2, d)$ bottleneck by exploiting a Hadamard decomposition of the Kronecker product — a technique not needed in the forward direction. The tight matching of upper and lower bounds at the same threshold $B = \sqrt[3]{\log n}$ suggests this is not an artifact of the analysis technique but a genuine complexity-theoretic boundary, and raises the open question of whether a different algorithmic paradigm (beyond polynomial approximation) could handle $d = \omega(\log n)$.

---

## Suggestions

- **Separate and honestly evaluate the two assumptions in Remark 5.3.** Acknowledge explicitly that $d = O(\log n)$ is restrictive for practical transformers and discuss what would be needed to extend the result.
- **Reframe the "training" claim accurately.** Change the central claim to "efficient gradient computation for tensor attention" and state that multi-step training convergence under approximate gradients remains open.
- **Add the chain rule derivation** from $\partial \mathrm{Loss}/\partial X$ to $\partial \mathrm{Loss}/\partial W_Q, W_{K_1}, W_{K_2}$ in the main paper or supplementary.
- **Briefly explain the origin of $\log^{11}(n)$** and $\epsilon = O(1/(\log n)^4)$ in the hardness proof sketch.

---

## Score and Decision

**Calibration anchors:**
- **/DDNFTaVQdU** (Faster Algorithms for Structured SVMs, SETH lower bounds, almost-linear time, $d = O(\log n)$, no experiments): avg 6.75, **accepted**. The most structurally similar anchor — same SETH + near-linear time + restrictive dimension assumption + no experiments. Our paper is comparable in quality.
- **/7BESdFZ7YA** (Training 1D GNNs is NP-Hard, pure theory complexity, bidirectional bounds): avg 6.40, **accepted**. Similar genre (complexity of training ML models, no experiments).
- **/AozPzKE0oc** (Fast RoPE Attention, polynomial method + SETH + attention complexity): avg 4.80, **rejected**. Most similar in topic but rejected due to a fundamental mathematical error in the core claim. The paper under review does not have an analogous error.
- **/EeqlkPpaV8** (Adaptive Complexity of Log-Concave Sampling, tight upper+lower bounds): avg 6.75, **accepted**. Tight bidirectional complexity result, theory paper.

The paper under review sits in the accepted cluster (6.40–6.75 range). It has a clean, verified contribution, genuine technical novelty, and a tight complexity result — matching what made DDNFTaVQdU and 7BESdFZ7YA acceptable. The primary downgrade relative to the upper anchors is the conceptual gap between "gradient computation" and "training" (Major weakness) and the mild overstatement in Remark 5.3. These are real but do not undermine the core theoretical result.

**Originality:** Good — fills a clear gap with non-trivial techniques.  
**Importance:** Moderate — tensor attention is not yet widely deployed, but this is a natural theoretical question in a growing area.  
**Claims supported:** Partially — the gradient computation result is well-supported; the "training" framing is overstated.  
**Experimental soundness:** N/A (pure theory paper, appropriate for venue).  
**Clarity:** Good overall; Remark 5.3 needs correction.  
**Value to community:** Solid — completes the complexity picture for tensor attention.

**Final Score: 6.0 — Borderline Accept**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>