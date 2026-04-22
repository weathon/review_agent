Now I have a thorough understanding of the paper and calibration anchors. Let me compile the final review.

## Summary

This paper proves that the backward gradient of tensor attention training can be computed in almost linear time n^{1+o(1)}, matching the best known forward complexity from Alman & Song (2024b). Under the same bounded entries assumption (B = o(∛log n)) and d = O(log n), the authors derive a closed-form gradient expression, design an algorithm using polynomial approximation and tensor algebraic techniques to accelerate its computation, and prove a matching conditional lower bound under SETH showing the bounded entries assumption is tight.

## Strengths

- **Completes the theoretical picture for tensor attention complexity.** Table 1 makes clear that prior work covered matrix forward, matrix backward, and tensor forward — this paper fills the missing cell (tensor backward) with matching upper (Theorem 5.2) and lower (Theorem 6.3) bounds, providing a coherent and complete set of results.

- **Systematic decomposition of the gradient computation chain.** The computational graph in Figure 4 and the sequence of intermediate results (Lemmas E.1–E.7, as outlined in Algorithm 1) systematically address why the backward case is harder than the forward case: the gradient involves mixing multiple low-rank structures (D⁻¹AD⁻¹AV, F = D⁻¹K(VLᵀ − D⁻¹diag(K1·VLᵀ)S)) that don't straightforwardly propagate. The decomposition into tractable steps is a genuine organizational contribution.

- **Matching conditional lower bound (Theorem 6.3).** The proof that gradient computation is at least as hard as forward computation via interpolation/integration, combined with the SETH-based hardness of the forward case (Lemma 6.2), establishes that the bounded entries assumption cannot be meaningfully weakened. This gives a sharp complexity transition that mirrors the forward case.

- **Lemma 5.6 provides a key computational saving.** The identity C₁ ∘ C₂ = C for Khatri-Rao inner products reduces complexity from T_mat(d, n², d) to T_mat(d, n, d), directly eliminating the quadratic-in-n term that would otherwise prevent nearly-linear time. While the identity itself is known in the tensor algebra literature, its application here is precisely targeted at the computational bottleneck.

## Weaknesses

### Fatal
None.

### Major

- **The practical feasibility claims in the abstract and conclusion are overstated relative to the assumptions.** The abstract states the results "establish the feasibility of efficient higher-order transformer training and may facilitate practical applications of tensor attention architectures." However, Theorem 5.2 requires d = O(log n) and B = o(∛log n). In standard transformer architectures (e.g., Llama2-7B with d = 4096, n = 4096), d = Θ(n) rather than O(log n), and under d = Θ(n) the algorithm's complexity is Ω(n³). Similarly, B = o(∛log n) requires QKV entries bounded by roughly 2–3 for typical sequence lengths. Remark 5.3 appeals to floating-point bit precision as justification, but precision constrains representable values without bounding their magnitude — fp16 can represent values up to ~65504. The paper provides no empirical evidence that these conditions hold or can be enforced without degrading model quality. This doesn't invalidate the theoretical result, but "establish the feasibility of efficient higher-order transformer training" is a claim the theory does not support in any practical sense. The result establishes *conditional theoretical feasibility*, not practical feasibility.

- **Incremental technical novelty over prior work.** The paper follows the established recipe from Alman & Song (2024a, 2024b): apply polynomial approximation to obtain low-rank structure, then propagate it through the computation. The new algebraic ingredients — Fact 5.4 (the mixed-product property of Kronecker products), Fact 5.5 (a distribution rule), and Lemma 5.6 (the Khatri-Rao inner product equals the Hadamard product of individual inner products) — are standard or well-known properties of Kronecker/Khatri-Rao products. The hardness result reduces to the forward case via interpolation/integration, which is the standard approach for extending lower bounds from evaluation to gradient computation. While the combination and application to the backward analysis requires genuine engineering effort (the chain of Lemmas E.1–E.7), the individual technical building blocks do not represent conceptual innovation beyond what the prior papers established. The paper would be stronger if it acknowledged this more transparently rather than claiming "highly non-trivial" techniques for what are standard tensor identities.

### Minor

- **The claim that gradients of weight matrices W_Q, W_{K₁}, etc. are "easy to get" from gradients of X and Y (Section 3, line 139) is stated without verification.** While likely true via chain rule through X = W_Q(W_{K₁} ⊗ W_{K₂})ᵀ (which involves only d × d operations), the parameterization constrains X to a specific manifold. A brief justification would strengthen this claim, since the practical interest is in training the weight matrices, not the abstracted X and Y.

- **No discussion of approximation error accumulation over training.** The paper provides an ∞-norm guarantee on a single gradient step (ε = 1/poly(n)), but says nothing about whether accumulated approximation errors over many optimization steps destabilize training. For iterative optimization, per-step ε-accuracy does not guarantee convergence to an ε-accurate solution. This is a known limitation of single-step approximation guarantees in the attention acceleration literature, and acknowledging it would be appropriate.

### Trivial
None.

## Nice-to-Haves

- Empirical verification that B = o(∛log n) is achievable without significant performance degradation, even at small scale.
- A runtime comparison plot of Algorithm 1 vs. naive computation for varying d (showing where the speedup materializes and where it vanishes).
- Softening the practical claims to position the paper as a conditional complexity-theoretic possibility result rather than a practical algorithm advance.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic's claim that Lemma 4.1 is "routine matrix calculus" and not a contribution.** While the derivation technique (chain rule through softmax and Kronecker structure) is standard, the explicit closed-form identification of the gradient as vec(A₁ᵀF(x)(A₂⊗A₃)) and the identification of the n × n² intermediate matrix F as the computational bottleneck is necessary scaffolding for the acceleration argument. Its value is functional (enabling the subsequent speedup) rather than conceptual, but it is not trivial to derive correctly for this specific structure. Calling it "the paper devotes significant space" overstates the issue—the lemma is stated concisely with proof in the appendix.

- **Demand for experiments as a core weakness.** This is a theoretical complexity paper in the learning theory track. Experiments would strengthen it but are not standard for this type of contribution. The Alman & Song papers it builds on likewise lack experiments. Moved to nice-to-have.

- **Claim that Facts 5.4 and 5.5 are "claimed as novel technical contributions" that "inflate significance."** The paper's "Technical novelty" paragraph in Section 5.2 does present these as contributions, but it is more accurate to say the paper claims novelty in *applying* these identities to the backward pass, not in discovering them. The phrasing "we prove many key properties for tensor operation needed for backward though not needed for forward" is about necessity for the proof, not about mathematical novelty. The critic's reading is somewhat uncharitable here—the paper does say these are "properties" and "key techniques," language that is standard in complexity-theoretic papers for describing the toolkit used.

## Novel Insights

The paper reveals a genuinely interesting structural observation about tensor attention: the backward gradient computation is not merely "the forward computation again," but involves a qualitatively different mixing of low-rank structures. Specifically, the F matrix (Figure 4) combines D⁻¹K (a normalized attention-like matrix) with VLᵀ (a value-output product) in ways that require propagating low-rankness through both Hadamard products and matrix multiplications simultaneously. This is why the swap rule (Fact 5.4) and distribution rule (Fact 5.5) are needed even though they weren't needed for the forward pass. This distinction between forward and backward complexity for tensor operations is a meaningful insight that goes beyond simply "doing Alman–Song again."

## Suggestions

- Rewrite the abstract and conclusion to replace "establish the feasibility of efficient higher-order transformer training" with more precise language such as "establish the conditional feasibility of efficient tensor attention gradient computation under bounded entries and logarithmic dimension assumptions." This is more honest and still communicates the value of the result.
- Add a brief discussion of when and whether d = O(log n) and B = o(∛log n) might be achievable in practice (e.g., in long-context settings where n is very large relative to d, or via explicit normalization/clipping of QKV projections), or explicitly acknowledge these as open questions.

## Score and Decision

**Calibration anchors:**

1. **rKMz6cDE7W** (avg 2.33, Reject): Streaming attention algorithm extending Alman-Song, seen as joining existing techniques with poor writing and no experiments. This paper is clearly stronger: cleaner presentation, more substantial technical effort, complete picture with matching bounds.

2. **AozPzKE0oc** (avg 4.80, Reject): Fast RoPE attention extending Alman-Song, had a fundamental error (circulant matrix assumption wrong for standard RoPE). This paper has no such fundamental flaw and is technically sound.

3. **AuAj4vRPkv** (avg 6.50, Accept Poster): Provable learning of transformers via gradient flow, limited setting, strong theory. This paper is comparable in terms of being a theoretical contribution with restrictive assumptions, but this paper's incremental nature relative to Alman-Song makes it weaker.

4. **DhdqML3FdM** (avg 7.0, Accept Poster): Complexity-theoretic limits of SSMs/Transformers, well-written, settles important questions. This paper is weaker — it fills a gap rather than opening a direction, and the practical overclaiming is a real negative.

5. **DDNFTaVQdU** (avg 6.75, Accept Poster): Nearly linear SVM conditional on SETH with matching bounds. Similar type of result (matching upper/lower bounds, conditional algorithm). This paper is somewhat weaker due to the incremental nature and overclaims.

This paper sits between the rejected Alman-Song extensions (4-5) and the accepted theoretical contributions (6-7). It is technically sound and fills a clear gap, but is incremental and overclaims practical impact. A score of 6 reflects: a competent and complete theoretical result that is meaningful but not surprising, executed with solid but not innovative technique.

**Evaluation axes:** Originality: moderate (fills a gap with expected techniques). Importance: moderate (completes the picture but doesn't change it). Claims support: partially (theoretical claims are well-supported, practical claims are not). Soundness: good (no errors found). Clarity: good (well-organized, helpful figures). Community value: moderate (useful reference for the tensor attention literature).

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>