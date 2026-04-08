=== CALIBRATION EXAMPLE 18 ===

# Final Consolidated Review
## Summary

PARDIFF proposes a progressive autoregressive diffusion framework that generates graphs block-by-block, using a deterministic structural ranking to partition nodes into blocks and applying equivariant discrete diffusion within each block. The approach aims to unify the directional coherence of autoregressive models with the permutation invariance of diffusion models, supported by theoretical permutation-consistency guarantees and a parallelized training scheme with causal masking.

## Strengths

- **Coherent integration of AR and diffusion paradigms**: The block-wise decomposition—where a structural ranking determines block membership and a shared diffusion model generates each block—is a well-motivated and clearly formalized framework. The factorization $P_\phi(G) = \prod_{k=1}^B P_\phi(\Delta_k | G_{\leq k-1})$ cleanly combines AR directionality with diffusion-based block generation, addressing a genuine gap between rigid AR orderings and order-agnostic diffusion.
- **Strong empirical performance on molecular benchmarks**: PARDIFF achieves state-of-the-art or competitive results on QM9, ZINC-250K, and MOSES (Tables 1–3), notably improving FCD on ZINC (1.62 vs. prior best 1.99 from SWINGNN-L) and MOSES (0.39 vs. 1.19 from DIGRESS), while using a ~4.5M parameter model compared to DIGRESS's 18.43M.
- **Permutation-invariance with formal guarantees**: Theorems 1 and 3 provide formal proofs that the ranking function $\psi$ and the full generative model are permutation-invariant, grounding the "order-agnostic" claim theoretically rather than relying on heuristics alone.

## Weaknesses

### Major:

- **Numerical inconsistencies undermine result credibility**: The text states PARDIFF achieves "VAL (98.1%), AL (98.9%), and molecular accuracy or MOL (88.5%)" on QM9, but Table 1 reports VAL=98.9, AL=99.2, MOL=90.3—disjoint numbers. Additionally, Table 3 (MOSES) has empty cells for PARDIFF's VAL and UNI columns, yet the text claims "perfect VAL and UNI." These inconsistencies must be corrected and explained. Further, PARDIFF reports 98.9% validity on QM9 against a dataset optimum of 97.8%. While not logically impossible (a model can learn to avoid invalid structures), a generative model exceeding the reference validity rate is unusual and warrants explicit discussion of whether post-hoc filtering or metric computation differences are involved.

- **"Learned structural decomposition" claim contradicts deterministic Algorithm 1**: The abstract states PARDIFF uses "learned structural decomposition" and "jointly predicts block sizes, ranks nodes." However, Algorithm 1 (the ranking function $\psi$) is a deterministic weighted-degree hashing heuristic—not a learned procedure. Only the block *size* is predicted by a learned module ($g_\alpha$, Algorithm 2). This conflation of a fixed heuristic with "learned" decomposition overstates the method's adaptability and should be clarified. If $\psi$ is not differentiable with respect to the generation objective, the block partition may not align with the optimal generative prior.

- **Symmetry-breaking mechanism not clearly distinguished from standard diffusion**: Section 2.3 frames the noise injection as a novel "energy injection phase" akin to "simulated annealing" for escaping symmetry plateaus. However, the forward diffusion process described (categorical corruption of node/edge features) appears identical to standard discrete diffusion as used in DIGRESS. The paper does not specify any modified noise schedule, annealing temperature parameter, or structured perturbation scheme that would substantively differ from the standard forward process. Without this clarification, the claimed contribution of "symmetry breaking" reduces to the observation that diffusion noise helps equivariant models escape degenerate configurations—a property inherent to any diffusion model.

- **No experimental evaluation on non-molecular graphs despite claiming general applicability**: The abstract explicitly claims PARDIFF works "across molecular and non-molecular domains." Yet all quantitative experiments (Tables 1–3) are on molecular datasets. Figure 1 shows qualitative grid generation but without standard metrics or baselines. This gap between claim and evidence is significant for a paper positioning itself as a "paradigm shift in structured generative modeling."

### Minor:

- **Theorem 2 restates well-known equivariance properties**: The result that permutation-equivariant networks assign identical representations to nodes in the same automorphism orbit is a standard property established in prior work (e.g., Morris et al., 2019, cited in the paper). Presenting it as a formal "Theorem" inflates the theoretical contribution; it would be more appropriate as a proposition citing prior work.

- **Scalability claims are internally contradictory**: Section 2.4 states the hybrid architecture "maintains $O(n^2)$ memory complexity" and enables scalability, yet Section 3 acknowledges "PPGN's high memory cost may constrain scalability on dense graphs." The paper should reconcile these statements and clarify the practical graph-size limits.

- **Efficiency claims lack empirical validation**: The paper claims "over 10× speedups in wall-clock training time" (Section 2.4) and "latency-aware design" (Abstract) but provides no comparison table of training/inference times against baselines.

- **Ranking function inherits WL expressivity limitations**: Algorithm 1 computes ranking via $K$-hop degree counts, which is equivalent in expressivity to the Weisfeiler-Lehman test. For regular graphs or graphs where structurally distinct nodes share identical degree sequences, $\psi$ will produce rank collisions, making the block assignment ambiguous. The paper does not discuss this limitation or its empirical impact.

### Trivial:

- The abstract's claim of a "paradigm shift" is overpromotional for an integration of existing techniques (AR block generation + discrete diffusion + PPGN backbone).

## Nice-to-Haves

- Ablation studies in the main text (currently relegated to the appendix) comparing: (a) block-wise vs. monolithic diffusion, (b) deterministic $\psi$ vs. learned ranking, (c) number of diffusion steps $T$, and (d) PPGN vs. simpler GNN backbones.
- Evaluation on at least one non-molecular graph benchmark (e.g., SBM, community graphs) with standard metrics.
- Comparison with hierarchical generation methods like LayerDAG (cited in the paper's own bibliography but not included as a baseline).
- Runtime/latency comparison table against DIGRESS and GDSS at varying graph sizes.
- Analysis of block size predictor ($g_\alpha$) accuracy, since errors cascade through the autoregressive generation process.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Hardware specification critique**: The harsh reviewer flagged the use of "NVIDIA RTX 5080" as "consumer-grade" and preferred A100/V100 reporting. This is a reproducibility nitpick about specific hardware; as of April 2026, the RTX 5080 is a legitimate GPU. Removed per soft rule on reproducibility nitpicks.
- **Missing broader impact/negative societal impact discussion**: The critic's demand for negative impact discussion (e.g., harmful chemical generation) is a standard but generic request; its absence does not constitute a core weakness of this paper's contributions.
- **Demand for theoretical analysis of hybrid loss convergence**: The critic asked for theoretical convergence analysis of the hybrid objective $L_t = D(\cdot\|\cdot) + \lambda \cdot L_{CE,t}$. This is outside the scope of what is standard for empirical graph generation papers and is moved to nice-to-have territory.
- **Comparison with LayerDAG as a mandatory baseline**: While LayerDAG is cited, demanding it as a baseline when the paper scopes itself to molecular generation against DIGRESS, GDSS, and CONGRESS is scope creep; included as nice-to-have only.
- **Claim that the paper contradicts itself on "no auxiliary features"**: The critic argued that using PPGN contradicts the "no auxiliary features" claim. However, PPGN's edge/level features are *architectural* (part of the model's internal representation), not *external* features like spectral eigenvectors or cycle indicators that DIGRESS uses. This distinction is reasonable and the critic's point is factually misleading.

## Novel Insights

The most interesting structural observation is that PARDIFF's ranking function $\psi$ creates a *partial* ordering that is strictly weaker than a full canonical ordering, yet still sufficient for coherent AR generation. This sits in a previously underexplored design space: rather than either (a) marginalizing over all orderings (intractable) or (b) committing to a single full ordering (biased), PARDIFF proposes (c) a coarse ranking that groups interchangeable nodes together and generates within-group elements jointly via diffusion. The idea that permutation invariance can be achieved *approximately* through learned block decomposition rather than exactly through full marginalization is conceptually clean, even though the current instantiation uses a deterministic heuristic rather than a learned decomposition. The tension between the heuristic $\psi$ and the "learned" framing in the abstract is the paper's most important point of honest clarification—the framework would be genuinely strengthened if $\psi$ were end-to-end differentiable.

## Suggestions

- **Correct all numerical inconsistencies immediately**: Align the text descriptions with Tables 1–3 (VAL, AL, MOL values in the text vs. table, and fill the empty cells in Table 3). Add explicit clarification on whether any post-hoc filtering is applied to generated samples before metric computation.
- **Clarify the symmetry-breaking mechanism**: Either explicitly specify how the noise schedule differs from standard discrete diffusion (with equations and hyperparameters), or revise the framing to acknowledge that the contribution is the *block-wise* application of standard diffusion, not a novel annealing-style mechanism.
- **Make the "learned vs. deterministic" distinction transparent**: Revise the abstract and introduction to clearly state that node ranking is a deterministic heuristic while block sizes are learned. Consider an ablation replacing $\psi$ with a learned ranking module.
- **Add one non-molecular experiment** with standard graph generation metrics to substantiate the generality claim, or revise the abstract to accurately scope the contribution to molecular domains.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0, 2.0]
Average score: 0.5
Binary outcome: Reject
