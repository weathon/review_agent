## Summary

This paper proves that the backward gradient of tensor attention—a higher-order generalization of standard matrix attention that captures multi-view correlations—can be computed in almost linear time $n^{1+o(1)}$ under a bounded entries assumption ($B = o(\sqrt[3]{\log n})$), matching the forward complexity established by Alman & Song (2024b). The authors derive a closed-form expression for the gradient (Lemma 4.1), propose a fast algorithm (Algorithm 1) using polynomial approximation and novel tensor algebraic techniques, and prove a matching SETH-based hardness lower bound showing the bounded entries assumption is tight for this problem.

## Strengths

- **Solves a natural open problem.** Table 1 clearly positions the paper's contribution: the forward/backward and matrix/tensor landscape was missing only the tensor-backward cell. Demonstrating that this cell admits the same $n^{1+o(1)}$ complexity as tensor-forward completes the theoretical picture for efficient tensor attention.

- **Non-trivial technical contributions beyond prior work.** The swap rule (Fact 5.4), distribution rule (Fact 5.5), and the tensor-trick reduction (Fact 5.7) are genuine new tools needed specifically for the backward pass—they propagate low-rank structure through the gradient computation of $\text{Loss}(X)$ in ways that did not arise in either matrix attention backward (Alman & Song, 2024a) or tensor attention forward (Alman & Song, 2024b). The authors explicitly identify this in Section 5.2: "generalizing to tensor attention backward posed many technical challenges… not presented in previous settings."

- **Matching upper and lower bounds.** Theorem 5.2 (upper bound: $n^{1+o(1)}$ achievable) and Theorem 6.3 (lower bound: slight relaxation of $B$ makes the problem require $\Omega(n^{3-\delta})$ under SETH) give a complete complexity characterization. The phase transition in complexity mirrors the known forward result (Lemma 6.2), which is a clean theoretical outcome.

- **Clean closed-form gradient.** Lemma 4.1 provides $\frac{d\text{Loss}(x)}{dx} = \text{vec}(A_1^\top F(x)(A_2 \otimes A_3))$, making the subsequent algorithm design and analysis transparent.

## Weaknesses

### Fatal
None.

### Major

- **The practical relevance claims are overstated relative to the assumptions.** The abstract states that the results "establish the feasibility of efficient higher-order transformer training," and the conclusion says they "may facilitate practical applications of tensor attention architectures." However, Theorem 5.2 requires (1) $d = O(\log n)$, whereas standard transformer dimensions are 768–4096 (far exceeding $O(\log n)$ even for $n = 10^6$); and (2) $\max\{\|A_1X_1\|_\infty, \ldots, \|A_5Y_2\|_\infty\} \le B = o(\sqrt[3]{\log n})$, which shrinks as $n$ grows. Remark 5.3 argues these are "practical" by noting that $n$ is large and models use 16-bit precision, but 16-bit precision constrains bit representation, not numerical magnitude—projected attention entries can be far larger than $o(\sqrt[3]{\log n})$ in practice. There is no mechanism ensuring this bound holds during training. The theory is sound but the "feasibility" framing suggests broader applicability than the formal results support. This misalignment between the strength of claims and the restrictiveness of assumptions is the paper's most significant issue.

- **The "tightness and necessity" claim (Theorem 6.3) is stronger than what is proven.** The paper claims "we prove that our assumption is necessary and 'tight' by hardness analysis" (bullet 3, Section 1). What Theorem 6.3 actually shows is: *assuming SETH*, for a specific parameter regime ($E=0$, $Y=I_d$, $X=\lambda I_d$), weakening $B$ from $O(\sqrt[3]{\log n})$ to $\Theta(\sqrt[\gamma]{\gamma(n)} \cdot \log n)$ makes the ATAttLGC problem require $\Omega(n^{3-\delta})$ time. This is a conditional hardness result for a restricted instance class that reduces to the forward computation. It does not rule out faster gradient algorithms under alternative formulations, different parameterizations, or for the general training objective. Calling the assumption "necessary" in a broad sense overstates what the lower bound establishes—it is necessary *within this specific computational problem under SETH*.

### Minor

- **No empirical validation, even for proof-of-concept.** The paper is purely theoretical. Given that the stated motivation is practical applicability of tensor attention to multi-modal models, even a small-scale experiment comparing Algorithm 1's gradient accuracy and runtime against naive $O(n^3)$ computation on synthetic data would significantly strengthen the contribution. The paper defers this entirely to "future work."

- **The hardness result applies to a restricted instance.** Theorem 6.3 holds for $E = 0$, $Y = I_d$, $X = \lambda I_d$, which effectively reduces gradient computation to something closely resembling the forward computation. While this is a standard reduction technique in fine-grained complexity, it limits the generality of the hardness claim for *training* (as opposed to a single forward pass with a specific loss instantiation). The paper should be more explicit about this scope.

- **Factorization of $X$ as $X_1(X_2 \otimes X_3)^\top$ is a structural restriction.** Definition 3.9 constrains $X$ to have this Kronecker factored form, whereas the unconstrained $X \in \mathbb{R}^{d \times d^2}$ from Definition 3.8 has more degrees of freedom. While this factorization arises naturally from the weight parameterization $W_Q(W_{K_1} \otimes W_{K_2})^\top$, it restricts the gradient problem being solved. The paper notes that "with gradients of $X$ and $Y$, it is easy to get the gradients of the weight matrices" but does not verify that the approximation guarantees transfer cleanly when back-propagating through the decomposition.

### Trivial
None.

## Nice-to-Haves

- Analysis of how gradient approximation error $\|\tilde{g} - d\text{Loss}/dX\|_\infty \le 1/\text{poly}(n)$ accumulates over training steps and whether it suffices for convergence of gradient-based optimization.
- Explicit characterization of the $n^{o(1)}$ factor in the runtime, even if only as a bound on the polynomial approximation degree needed.
- Discussion of what happens for general $d$ (e.g., $d = \Theta(n^\alpha)$), even if the result degrades gracefully.

## Removed Points

- **"Lack of experiments as a fatal flaw."** Multiple reviewers suggested that the absence of experiments is a critical weakness. However, this is a fine-grained complexity theory paper in a well-established line of work (Alman & Song, 2023; 2024a; 2024b), none of which included experiments either. Experiments would strengthen the paper but are not a core flaw for a theory contribution. Downgraded to Minor.

- **"Incremental over prior work."** The harsh critic and neutral reviewer both raise this. While the paper follows the same blueprint (polynomial approximation + low-rank structure + SETH hardness), the backward computation requires genuinely new tensor algebraic techniques (Facts 5.4, 5.5; Lemma 5.6) that do not arise in the forward case. The technical gap is meaningful. Downgraded to a note in the assessment rather than a standalone weakness.

- **"Notation/dense presentation."** This is a formatting/readability concern. The paper is in the standard style for fine-grained complexity theory, which inherently involves many definitions. Not a substantive weakness.

- **"The factorization $X = X_1(X_2 \otimes X_3)^\top$ vs. arbitrary $X$ is a major issue."** This is noted as a Minor point; it arises naturally from the transformer weight structure, so it is not a fundamental limitation but worth flagging.

- **"$Y$ is treated as fixed in the optimization."** The paper does optimize over $X$ and notes that the gradient for $Y$ is not the bottleneck (since $Y$ does not participate in the softmax). The decision to focus on $X$ is reasonable since that is the computationally hard part. Not a standalone weakness.

## Novel Insights

The paper reveals that the backward gradient of tensor attention shares the same complexity threshold as the forward pass ($n^{1+o(1)}$ achievable under bounded entries, SETH-hard slightly above), despite the gradient involving additional non-trivial algebraic structure (the $D^{-1}$ normalization and the Hadamard product of softmax rows with gradient residuals). This symmetry between forward and backward complexity is not obvious a priori and suggests that for Kronecker-structured attention mechanisms, the soft-threshold of computability is set by the exponential function's entry-wise behavior rather than by the specific loss structure. However, the specific hardness reduction (which trivializes the loss by setting $E=0$) means this symmetry may be more about the forward computation lurking inside any gradient formulation than about training per se.

## Suggestions

- Reframe the claims: change "establish the feasibility of efficient higher-order transformer training" to "establish the feasibility of efficient gradient computation for the tensor attention layer under bounded entries and dimension constraints" in the abstract and conclusion. This is accurate and still significant.

- Add a brief subsection discussing the practical implications of $d = O(\log n)$ and $B = o(\sqrt[3]{\log n})$, including whether norm-based regularization or weight clipping could enforce the bounded entries condition during training, and at what cost.

- State Theorem 6.3's restrictions ($E=0$, $Y=I_d$, $X=\lambda I_d$) explicitly in the main text (not just the appendix) and note that the hardness result applies to a restricted instance class, tempering the "necessary and tight" language.

- Provide even a synthetic experiment (few data points showing relative error vs. $n$, and runtime vs. $n$) to bridge the theory-practice gap that the paper's own motivation highlights.

## Score and Decision

**Calibration:** I compared this paper against several related works. Alman & Song (2024b), the forward-pass counterpart, received scores of 8/8/8 at NeurIPS (accepted spotlight). That paper established the framework from scratch for tensor attention forward. The multi-layer gradient paper (JgSbMcgd8q) with similar profile (extending prior gradient computation to new settings, same $d=O(\log n)$ concerns, no experiments) received 5/6/6/6 (rejected). The LoRA computational limits paper (Lf5znhZmFu), also purely theoretical with dense notation, received 6/3/6/8 (accepted poster).

This paper extends an existing line of work with genuine technical novelty (new tensor algebraic tools for backward propagation), provides matching upper/lower bounds, and solves a well-posed open problem. However, the practical overclaiming is more severe than in the forward paper (which was more measured about feasibility), the assumptions ($d = O(\log n)$, $B = o(\sqrt[3]{\log n})$) are highly restrictive and not validated, and the hardness result's scope is narrower than claimed. Relative to the forward paper (which deservedly scored high for opening a new direction), this paper is incremental. Relative to the multi-layer gradient paper (which was rejected with similar weaknesses), this paper has somewhat stronger technical novelty but similar limitations.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>