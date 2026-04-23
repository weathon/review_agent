Now I have all the information needed. Let me write the final consolidated review.

## Summary

The paper proposes SparseFW, which relaxes the combinatorial mask selection problem in LLM pruning to a convex program over the convex hull of binary masks and solves it via the Frank-Wolfe (FW) algorithm. The method precomputes G=XX^⊤ and H=WG to make per-iteration cost independent of sequence length and sample count, enabling LLM-scale application. Across five modern GPT architectures and three sparsity regimes, SparseFW consistently improves zero-shot accuracy and reduces per-layer pruning error by up to 80%, especially at higher sparsity levels.

## Strengths

- **Unification of greedy pruning methods (Section 2.1)**: The paper rigorously shows that SparseGPT, Wanda, and RIA all solve the same greedy single-weight pruning subproblem, differing only in weight reconstruction and rescaling. Showing that Wanda's saliency score |w_{ij}|·‖X_{j,:}‖₂ emerges as the greedy optimum (Eq. 5) and that RIA is Wanda applied to a rescaled matrix (Eq. 6–7) is a genuinely clarifying conceptual contribution.

- **Efficiency engineering for LLM scale**: Precomputing G=XX^⊤ and H=WG (Algorithm 1, Line 1) reduces the gradient computation to ∇L(M_t) = −2·W⊙(H − (W⊙M_t)G), which is independent of sequence length L and sample count N. For LLaMA-2-7B, this shrinks the key matrix from 4096×524,288 to 4096×4096 — this is what makes FW feasible at LLM scale.

- **Consistent accuracy improvements at higher sparsity (Table 1)**: At 60% and 2:4 sparsity, SparseFW consistently outperforms Wanda and RIA in zero-shot accuracy across all five architectures (e.g., LLaMA-3 at 60%: 48.08→51.92 with Wanda warmstart; Gemma-2 at 60%: 63.19→64.46). Perplexity also improves substantially at these regimes (e.g., Gemma-2 at 60%: 16.46→14.83).

- **Honest reporting of the local-global mismatch**: The paper transparently reports that vanilla FW (α=0.0) reduces per-layer error yet worsens perplexity (Section 2.3), and the conclusion acknowledges that "inductive biases still appear necessary for improved perplexity." This is a genuinely interesting negative finding.

- **Sample efficiency (Figure 3, right)**: SparseFW benefits substantially from more calibration data (perplexity drops significantly from 64 to 512 samples), while Wanda's performance is nearly flat. Since FW iteration cost is independent of sample count after precomputing G, additional samples come at low marginal cost.

## Weaknesses

### Fatal
None.

### Major

- **Framing mismatch between title/abstract and the actual working method**: The title "Don't Be Greedy, Just Relax!" and abstract framing ("we *instead* consider the convex relaxation") present the method as *replacing* greedy heuristics with convex relaxation. However, the algorithm that produces all results in Table 1 fixes α=0.9 — i.e., 90% of the Wanda/RIA greedy mask is preserved unchanged, and FW only optimizes the remaining 10%. Pure FW (α=0.0) "consistently yields worse results than the baselines" (Section 2.3). This means the working method is fundamentally a *refinement* of greedy masks, not a replacement. While the paper does disclose this in Section 2.3 and the conclusion, it is presented as a "caveat" and omitted from Algorithm 1, while the title and abstract promise a wholesale alternative. This matters because it reframes the core contribution: rather than demonstrating that convex relaxation outperforms greedy heuristics, the paper shows that FW can *locally refine* a mostly-greedy mask — a more modest but still valuable claim.

- **Theory does not cover the actual working algorithm**: Lemma 1 provides approximation bounds for vanilla FW on C_k, but the algorithm that produces all empirical results uses warmstarting and α-fixing, which restricts the feasible set to a strict subset of C_k. The convergence guarantee and approximation bounds do not directly apply to this modified algorithm. The paper does not acknowledge this gap — Section 4 presents the theory as "a key benefit of SparseFW over greedy heuristics" without noting that it covers a different (non-working) variant. The theory still provides useful conceptual insight (explaining the optimization vs. thresholding error tradeoff in Figure 4), but the disconnect between the proved and the practiced should be explicitly discussed.

### Minor

- **Mixed perplexity results at 50% sparsity**: At 50% sparsity, SparseFW(Wanda) worsens perplexity on LLaMA-3 (10.21 vs. 10.09) and DeepSeek-7 (7.89 vs. 7.79), and SparseFW(RIA) worsens on DeepSeek-7 (7.93 vs. 7.90). While accuracy still improves, the perplexity regressions at the most common sparsity level are concerning. Without standard deviations (omitted "for legibility"), it is impossible to assess whether these differences are statistically meaningful.

- **α=0.0 results not reported in main text**: The paper states that pure FW fails but does not include these results in Table 1 or a main-text figure. Given that the failure of pure FW is arguably the most important empirical finding (it reveals the local-global mismatch and motivates the α-fixing), reporting these numbers alongside the α=0.9 results would allow readers to assess the contribution of FW versus the warmstart+fixing mechanism.

- **The approximation bound is vacuous at LLM scale**: Lemma 1 includes λ_max(Q) and a √(d_in·d_out·k) term, which at LLM scale (d_in, d_out ~ 4096+, k in the millions) renders the bound too loose to provide useful quantitative guarantees. The theory serves as conceptual justification rather than a practical guarantee, which is fine, but the paper could be more upfront about this.

### Trivial
None.

## Nice-to-Haves

- Comparison to SparseGPT (even citing published numbers on the same models) would strengthen practical relevance, though the paper's distinction between mask-selection and reconstruction methods is defensible.

- Analysis of what FW actually changes in the 10% of the mask it modifies (what fraction were originally kept vs. pruned by Wanda) would provide insight into whether FW is "fixing mistakes" or perturbing the boundary.

- Wall-clock time comparisons alongside iteration counts would make the compute-cost discussion more concrete.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh reviewer: "the core claim of the paper is false as stated"** — Overstated. The paper discloses the α mechanism in Section 2.3 and the conclusion. The framing is misleading, not fraudulent. The contribution (refining greedy masks via FW) is real even if the narrative overclaims.

- **Harsh reviewer: "the method cannot function without the greedy heuristic"** — This implies the method is fundamentally dependent, but the paper demonstrates that even small α values (0.1) improve perplexity. The α=0.9 setting is optimal, not a hard requirement. The dependency is real but less absolute than claimed.

- **Harsh reviewer: demand for SparseGPT comparison** — The paper explicitly scopes its comparison to mask-selection methods, distinguishing from reconstruction methods like SparseGPT. This is a defensible methodological choice (comparing like-for-like), not a missing baseline. Moved to nice-to-have.

- **Strength finder: "Principled convex relaxation replacing greedy heuristics" as a core strength** — Conflicts with the verified Major weakness that the method requires 90% greedy warmstart. The relaxation is real but it does not *replace* greedy heuristics. Moved to removed.

- **Strength finder: "Theoretical approximation guarantees" as a core strength** — Conflicts with the verified Major weakness that the theory doesn't cover the working algorithm. The theory provides conceptual insight but not a guarantee for the method as actually used. Moved to removed.

- **Harsh reviewer: demand for α=0 results in Table 1** — Partially valid; moved to Minor since the paper does discuss this finding verbally, just without explicit numbers.

## Novel Insights

The most interesting finding is the *local-global objective mismatch*: optimizing the per-layer pruning error (a convex, well-defined objective) to lower values can *worsen* end-to-end perplexity. This is not just an artifact of insufficient optimization — it's a fundamental misalignment between the layer-wise surrogate and the global objective. The α-fixing trick works precisely because it constrains FW to a region where the local objective is more aligned with the global one, using the greedy saliency scores as an inductive bias. This suggests that the real value of saliency-based methods like Wanda is not their optimization quality (they are greedy, after all) but their alignment with the global objective — a distinction the pruning literature has not sufficiently emphasized.

## Suggestions

- Reframe the paper honestly: title it to reflect that FW *refines* greedy masks rather than replacing them. Change the title and abstract to acknowledge the warmstart dependency up front. This would actually strengthen the paper by making the local-global mismatch the central story.

- Include α=0.0 results in Table 1 (or a main-text table), making the failure of pure FW a central empirical finding rather than a caveat.

- Add a brief discussion in Section 4 noting that Lemma 1 applies to vanilla FW on C_k and that extending the theory to the warmstarted, α-fixed variant is an open problem.

- Report standard deviations for Table 1, at least in a supplementary table, to allow assessment of statistical significance.

## Score and Decision

**Calibration anchors:**

1. **ParoQuant** (`/home/wg25r/review_agent/human_reviews_2026/1USeVjsKau.md`), avg 7.0, Accept Poster — LLM compression (quantization) with solid empirical results and honest framing. Our paper has comparable empirical breadth but weaker framing integrity.

2. **LayerNorm removal** (`/home/wg25r/review_agent/human_reviews_2026/VPtHqcafIY.md`), avg 7.5, Accept Poster — Clear contribution, honest about limitations, limited model scale. Our paper is on larger models but less honest about its core limitation.

3. **Catalyst** (`/home/wg25r/review_agent/human_reviews_2026/3O8TAbrMKW.md`), avg 4.5, Reject — Structured pruning with theoretical guarantees that don't translate to practice, modest gains. Our paper has stronger empirical results and a clearer conceptual contribution but shares the theory-practice gap.

4. **Multilevel mirror descent** (`/home/wg25r/review_agent/human_reviews_2026/xufpYfKi89.md`), avg 5.0, Reject — Sparse training with convergence guarantees but limited empirical validation and theory-practice gap. Our paper has more convincing empirical results at larger scale.

5. **LayerDecompose** (`/home/wg25r/review_agent/human_reviews_2026/0IWZjbMmry.md`), avg 3.0, Reject — LLM compression with overclaimed results and inconsistent baselines. Our paper is more honest and has a clearer contribution than this.

The paper sits between the medium-band (Catalyst 4.5, mirror descent 5.0) and the high-band (ParoQuant 7.0). It has stronger empirical results than the medium-band papers and a genuine conceptual contribution (unification of greedy methods + local-global mismatch insight), but the framing mismatch and theory-practice gap prevent it from reaching the high band. The paper is better than Catalyst because its empirical results are more convincing and its conceptual contribution is clearer. I place it at 5.5 — a paper with real contributions that needs reframing to reach its potential.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>