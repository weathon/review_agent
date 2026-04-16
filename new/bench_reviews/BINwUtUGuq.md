## Summary

FISTAPruner proposes a layer-wise post-training pruning method for LLMs that formulates pruning as a LASSO-like convex optimization problem (minimizing Frobenius-norm output reconstruction error plus ℓ₁-regularization on weight rows), solved efficiently via FISTA with O(1/k²) convergence. The method incorporates an intra-layer error correction mechanism (sequentially using pruned activations as inputs to subsequent operators within each decoder layer) and extends to 2:4 semi-structured pruning via hard thresholding. An adaptive bisection-based scheme tunes the regularization parameter λ to hit target sparsity. Extensive experiments on OPT, LLaMA, LLaMA-2, and LLaMA-3 (125M–70B) show consistent improvements over SparseGPT, Wanda, DSnoT, and PERP in both perplexity and zero-shot tasks.

## Strengths

- **Comprehensive and consistent empirical improvements**: Across Tables 1–5, FISTAPruner outperforms SparseGPT, Wanda, DSnoT, and PERP on nearly every model×sparsity configuration. The 2:4 semi-structured results are particularly strong (e.g., LLaMA-2-70B: 5.16 vs. 5.38/5.20 for SparseGPT/Wanda; LLaMA-3-70B zero-shot mean: 0.6901 vs. 0.6443/0.6468). This constitutes a meaningful empirical contribution.

- **Outperforms retraining-based methods without retraining**: Table 4 shows FISTAPruner (no retraining) achieves lower perplexity than SparseGPT/Wanda combined with PERP retraining on OPT models, establishing that better initial pruning can be more effective than adding retraining to weaker baselines.

- **Well-motivated intra-layer error correction**: Figure 4(a) demonstrates consistent perplexity improvements from the error correction mechanism across multiple sparsity levels. The design also enables parallel pruning across decoder layers, providing practical scalability.

- **Flexibility in warm-start initialization**: Table 6 shows the method achieves comparable results whether initialized from magnitude pruning, dense weights, or SparseGPT/Wanda. Combined with the adaptive λ tuning, this reduces dependency on specific initialization strategies.

- **Principled formulation**: Even though the LASSO objective and FISTA solver are standard individually (the paper correctly cites Beck & Teboulle, 2009), their application to LLM layer-wise pruning with adaptive sparsity control and intra-layer error correction constitutes a coherent, well-designed pipeline that consistently works well in practice.

## Weaknesses

### Major:

- **Novelty claims are overstated relative to the actual contribution**: The abstract claims "for the first time, a LASSO-like convex optimization model crafted to induce sparsity in LLMs" and the introduction sets up a contrast between "heuristic" methods and this work's "rigorous theoretical foundation." However, the core objective (Eq. 3) is a standard sparse reconstruction formulation (Frobenius error + ℓ₁ penalty), and FISTA is the canonical algorithm for exactly this composite objective. The intra-layer error correction, while empirically effective, is a natural sequential heuristic (use pruned activations as inputs to subsequent operators) rather than a principled derivation from a global objective. The method's real value is demonstrating that a straightforward, well-known optimization approach applied carefully with error correction outperforms heuristic baselines—this is a valuable empirical finding, but the framing as a fundamentally new framework overstates the conceptual novelty.

- **Theoretical contribution is thin relative to claims**: (1) Remark 1 (convexity of the objective) is trivial—summing two convex functions yields a convex function. (2) FISTA's O(1/k²) convergence is cited from existing work, not derived for this specific problem. (3) Theorem 1 (bisection convergence for λ-tuning) applies only to the unstructured case; yet the 2:4 semi-structured results—which show the largest improvements—are obtained via hard thresholding that explicitly breaks convexity (acknowledged in Section 3.3 with only "empirical success" as justification). The "rigorous convex optimization" framing is partially undermined by the fact that the method's most impactful practical setting falls outside the convex regime. The paper should more clearly delineate what is guaranteed and what relies on empirical validation.

- **Key ablations are limited to a small model, undermining generalization of design conclusions**: The intra-layer error correction ablation (Figure 4a), calibration sample study (Figure 4b), and warm-start study (Table 6) are all conducted only on OPT-125M. Given that the headline results sit on LLaMA-3-70B, there is no direct evidence that these design choices remain effective at scale. In particular, the relative contribution of FISTA vs. intra-layer error correction vs. adaptive λ tuning is not disentangled on any model larger than 125M parameters. A critical missing ablation is: does applying intra-layer error correction to SparseGPT or Wanda (without FISTA) also yield improvements? Without this, it is impossible to attribute gains to the convex optimization component vs. the error correction strategy.

### Minor:

- **Row-wise ℓ₁-norm choice insufficiently justified in the main text**: Equation 2 applies the ℓ₁-norm per row rather than element-wise or matrix-wise. Appendix A is cited for justification, but the main text does not explain this design choice, which directly affects the sparsity structure of the solution.

- **DSnoT comparison is one-sided**: Table 3 only reports Wanda+DSnoT, not SparseGPT+DSnoT, though SparseGPT+DSnoT often yields stronger results. Table 5 zero-shot evaluation is limited to a single model size (LLaMA-3-70B).

- **Warm-start confounds fair comparison**: The main experiments use SparseGPT initialization for OPT and Wanda initialization for LLaMA, meaning FISTAPruner's total computational cost includes running the baseline first. While Table 6 shows comparable results from dense/magnitude initialization on OPT-125M, this does not hold at larger scales where warm start may matter more.

- **Substantial pruning time overhead**: ~12 hours for LLaMA-3-70B on a single A100 (Section 5), compared to minutes for Wanda and ~1 hour for SparseGPT for similar models. The paper argues this is acceptable for offline pruning, but no systematic time-quality tradeoff analysis is provided.

### Trivial:

- No variance or multiple-seed reporting for zero-shot tasks, where some accuracy differences are small (~1–2% absolute). This is common practice in the LLM pruning literature but weakens the precision of claims like "retaining 98.6% of zero-shot performance."

## Nice-to-Haves

- **Inference latency/speedup measurements**: The abstract claims "computational acceleration" but no wall-clock inference speedups are reported. This is a practical consideration for end users.

- **Ablation of intra-layer error correction applied to baselines**: Testing whether the error correction mechanism alone (without FISTA) improves SparseGPT/Wanda would clarify the source of FISTAPruner's gains.

- **Ablations on a larger model (e.g., 7B)**: Extending the warm-start and error-correction ablations beyond OPT-125M would strengthen confidence in the design choices.

- **Sensitivity analysis for key hyperparameters** (ξ=0.3, K=20, T=3): Understanding robustness to these choices would help practitioners.

## Removed Points

- *Claim that Theorem 1's monotonicity requirement for bisection is unproven*: The paper states "we establish theoretical guarantees for the convergence of this method in the context of unstructured pruning" and provides a proof in Appendix C. While concerns about the hard thresholding step affecting the theorem's applicability are valid (and kept above), questioning the monotonicity of s(λ) specifically requires accessing the appendix proof, which is not available in the provided text. This criticism is partially addressed by the paper's acknowledgment that exact target sparsity may not be achieved and their use of hard thresholding as a correction step.

- *Claim that the paper lacks reproducibility due to insufficient specification of the λ update rule*: Algorithm 1 provides the complete procedure including the bisection method, and Section 3.4 describes the E_round/E_total ratio heuristic with threshold ξ=0.3. All key hyperparameters are specified in Section 4.1. The implementation details are adequately described for reproducibility.

- *Claim that Algorithm 1 is ambiguous about whether zeros from hard thresholding are frozen in subsequent FISTA iterations*: Reading Algorithm 1, the flow is: FISTA produces W_K^*, then hard thresholding produces W_{K+1}^*, then the best solution W_best^* is updated based on error comparison. The next outer iteration starts a new FISTA run from W_best^* as initialization. Since W_best^* has already been hard-thresholded (it equals some W_{K+1}^*), its zeros are not explicitly frozen during the next FISTA run—soft-thresholding can in principle "un-zero" entries. However, this is a design choice, not an ambiguity; the paper's approach simply uses the best solution as warm start for the next iteration rather than imposing a strict mask.

## Novel Insights

The interplay between the convex optimization component and the empirical hard thresholding for 2:4 sparsity creates an interesting gap: the method's strongest empirical results (2:4 semi-structured pruning) fall precisely in the regime where the theoretical guarantees break down. This suggests that the practical value of FISTAPruner lies not in its convexity guarantees but in FISTA providing a well-regularized initialization that hard thresholding can then refine—a finding that weakens the "rigorous convex optimization" narrative but is empirically important. The intra-layer error correction mechanism is the most design-novel element and deserves more investigation as a general technique applicable beyond FISTAPruner.

## Score and Decision

Calibration:
- **Wanda** (ICLR 2024, scores 6/6/5/8 → accept poster): Simple metric-based method, incremental but clearly effective and practical. FISTAPruner shows stronger empirical results but at greater computational cost and with less simplicity.
- **Plug-and-Play/RIA** (poster, 8/6/6/6): Similar incremental contribution profile—new metric + channel permutation, good results. FISTAPruner has broader empirical evaluation.
- **CVXQ** (reject, 3/3/3/3): Convex optimization for LLM compression with flawed formulation—a cautionary parallel for overclaimed theory but FISTAPruner's empirical results are much stronger.
- **Mecon** (reject, 3/3/8/8/6): Evolutionary search for pruning, rejected for efficiency concerns and marginal improvements—FISTAPruner's improvements are more substantial.

FISTAPruner's empirical results are clearly strong and the method is well-designed. However, the novelty is modest (standard LASSO + standard FISTA applied to LLM pruning), the theoretical claims are overstated relative to what is actually delivered, and ablations are insufficient to isolate the contribution of each component at scale. The paper makes a solid engineering contribution but overclaims conceptual novelty. This places it somewhat below the Wanda tier (which was simpler and faster with competitive results) but above rejected papers. A score of **5** reflects a paper with real empirical value but overstated novelty and insufficient analysis to support the core claims about what drives the improvements.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>