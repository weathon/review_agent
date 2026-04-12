## Summary
This paper studies distributional equivalence for linear non-Gaussian latent-variable causal models with arbitrary cycles. Its central contribution is a graphical characterization of when two such models induce the same observed distribution set, built around a new “edge rank” perspective and yielding both a decision criterion (Theorem 2) and a transformational characterization for traversing the equivalence class (Theorem 3). The paper also proposes a proof-of-concept discovery pipeline, glvLiNG, that uses OICA-estimated mixing matrices to recover the class up to this equivalence.

## Strengths
- **A genuinely new equivalence characterization in a difficult regime.** The paper tackles a setting that combines latent variables, cycles, linearity, and non-Gaussianity, and provides a concrete characterization of distributional equivalence rather than only identifiability in special cases. The progression from Definition 1 / irreducibility to Lemmas 1–5 and then Theorem 2 is a substantial theoretical contribution.
- **The singleton decomposition in Theorem 2 is especially strong.** The paper starts from an apparently intractable condition involving all subsets and permutations (Lemma 3 / Lemma 5), then shows that equivalence can be checked via the children-bases of just \(L\) and each \(L\cup\{X_i\}\). This is the cleanest and most operational result in the submission.
- **The edge-rank viewpoint is insightful and potentially useful beyond this paper.** Definitions 4–6 and Theorem 1 provide a local, matching-based dual to path ranks that the paper uses effectively to simplify equivalence reasoning. Even though the underlying matroid ideas are classical, packaging them as edge-rank constraints for latent-variable causal discovery is a meaningful conceptual contribution.
- **The paper gives both static and transformational characterizations.** Beyond deciding equivalence, Theorem 3 shows how equivalent graphs are connected by admissible cycle reversals and edge additions/deletions. This mirrors the role that covered edge reversals play in simpler settings and gives the work a satisfying structural completeness.
- **The authors are appropriately explicit that the algorithmic part is a proof of concept.** Section 5 and Section 6 clearly state that the main focus is the equivalence characterization, and that the reliance on OICA is a limitation of the current instantiation rather than something hidden from the reader.

## Weaknesses

###: Fatal
- None.

### Major:
- **The empirical algorithm is much less compelling than the theory, because its guarantees depend on oracle/exact rank information while the practical implementation uses heuristic rank handling.** The formal guarantee in Section 5 is explicitly stated only “under the assumptions of access to an oracle OICA and faithfulness,” and Appendix D.4 then explains that, in practice, the method assigns a confidence score based on the minimum singular value and thresholds/approximates ranks:
  > “we assign a ‘full-rank confidence score’ … use the score \(1/(1+\exp(-\alpha^{-1}(\sigma_{\min}-\epsilon)))\)”  
  > “in phase 1 … we approximate the closest valid transversal matroid … In phase 2 … we simply threshold these scores”
  
  This means the finite-sample pipeline is not the same object as the provably correct oracle procedure. That does not invalidate the theoretical contribution, but it does weaken the claim of a practically realized discovery method, especially since no analysis is given for when these heuristics preserve the needed matroid/basis structure.
- **The paper’s practical impact is limited by its dependence on OICA, which the paper itself acknowledges as a bottleneck.** This is not a misunderstanding: Section 5 explicitly says,
  > “One may be concerned about OICA’s known inefficiency in practice”  
  and Section 6 lists as a limitation:
  > “One limitation is the use of OICA in glvLiNG”
  
  Since the method’s input is an estimated mixing matrix and OICA quality directly controls all downstream rank queries, the end-to-end practicality is constrained by a difficult upstream estimation problem. The paper is honest about this, but the limitation remains substantial.
- **The evaluation does not yet convincingly connect the elegant equivalence theory to reliable finite-sample behavior.** Figure 7 and Appendix D.4 show simulation results, but the paper does not provide targeted robustness analysis for the key failure mode: small rank perturbations or near-violations of genericity in the estimated mixing matrix. Given that the theory hinges on exact/generic rank patterns, the missing experiment is not “more experiments” in the abstract; it is specifically the one that would test whether the theorem-derived structure survives estimation noise.

### Minor
- **The method is not uniformly strong empirically; on sparse graphs, structurally constrained baselines can perform better.** Appendix D.4 explicitly notes:
  > “when \(d=1\), these two methods perform better than glvLiNG”
  
  This is a fair and informative result, and the paper discusses it reasonably. Still, it means the practical case for a structural-assumption-free pipeline is strongest in misspecified/dense regimes rather than broadly dominant.
- **The output equivalence classes can be very large, and the paper does not fully elevate its own summary representation into the main practical story.** The paper usefully quantifies class sizes and even proposes a CPDAG-like presentation in Appendix C.3 / Theorem 4, but the main algorithmic narrative still centers on recovering/traversing the full class. For applied use, the summary object seems more important than raw enumeration, yet it remains relegated to the appendix.
- **Scalability claims are only partially substantiated.** The runtime evidence in Table 4 is encouraging for the rank-realization component, but it is based on oracle mixing matrices and small-to-moderate graph sizes. The paper does not provide complexity bounds or convincing evidence for larger instances where matroid operations and traversal could become difficult.
- **The term “structural-assumption-free” may invite overreading unless repeatedly contrasted with the strong parametric assumptions.** The paper is careful in places, but because the setting still assumes linearity and non-Gaussianity, a bit more explicit wording in the abstract/introduction would help avoid confusion.

### Trivial
- None.

## Nice-to-Haves
- Add a controlled ablation that perturbs oracle rank queries or OICA-estimated mixing matrices, to show how sensitive the recovered equivalence class / summary graph is to rank errors.
- Promote the CP-like presentation from Appendix C.3 into the main text and evaluate it as the primary practical output, not just traversal of all equivalent graphs.
- Provide explicit complexity discussion for Phase 1, Phase 2, and equivalence-class traversal, even if only coarse worst-case and empirical scaling.
- Analyze robustness as the exogenous noise becomes closer to Gaussian, since the entire pipeline rests on the non-Gaussian identifiability regime.
- Add stability analysis for the real-data graph (e.g., bootstrap edge frequencies), especially for the “solid vs dashed” interpretation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the algorithm fundamentally cannot be executed because the latent/observed partition of OICA columns is unresolved.** This criticism is not supported by the paper. Appendix A explicitly addresses the permutation ambiguity by treating the recovered OICA matrix as having columns indexed by \(V\) only up to an unknown permutation \(\pi\), and reconstructing a binary support matrix \(Q\) whose row permutation can then be adjusted to yield a valid digraph:
  > “there exists a permutation \(\pi\) of \(V\)… there exists an unknown binary matrix \(Q\)… As long as one can recover this matrix \(Q\), one can then permute its rows to place nonzero entries on the diagonal”
  
  The method is designed around the fact that OICA only identifies columns up to permutation/scaling. One can debate practicality, but not claim that the paper forgot this issue.
- **Criticism that runtime claims are “misleading” because Table 4 uses oracle mixing matrices.** The paper is explicit about what Table 4 measures:
  > “constructing digraphs that satisfy the rank constraints of oracle OICA mixing matrices”
  
  This is a fair component-level runtime evaluation, not a hidden end-to-end claim.
- **Weakness framed as lack of a single summary object for the equivalence class.** The paper actually does discuss such a summary in Appendix C.3 / Theorem 4. It is fair to say this should be foregrounded more, but not fair to say the paper lacks it.
- **Strengths like “the paper is well-written” / “the topic is important” / “experiments are extensive.”** These are too generic and were omitted.

## Novel Insights
The strongest synthesis across the reviews is that this paper is best viewed as a theory-first contribution with a nontrivial but still immature algorithmic instantiation. The real advance is not merely “using ranks” but showing that in this latent cyclic LiNG setting, distributional equivalence can be reduced all the way from mixing-matrix closure to path-rank equality, then dualized to edge-rank/matroid structure, and finally localized to singleton checks in Theorem 2. That localization is what turns a conceptually complete but unusable characterization into something structurally comparable to classical equivalence criteria. At the same time, the paper itself supplies the seeds of its most practical next step: Theorem 4’s summary representation is probably the right output object for applications, more so than enumerating large equivalence classes.

## Suggestions
- Make the paper’s positioning sharper: present it explicitly as a **theoretical characterization with a proof-of-concept recovery pipeline**, rather than letting readers infer stronger practical maturity than the current empirical section supports.
- Add a dedicated robustness section that perturbs ranks directly and reports degradation in recovered bases / equivalence summaries.
- Move the Appendix C.3 presentation result closer to the main text and evaluate it on both synthetic and real data as the practical output.
- Give explicit complexity bounds or at least more detailed empirical scaling for Phase 1, Phase 2, and traversal.
- In the abstract and introduction, clarify that “structural-assumption-free” refers to latent-graph structure, not to absence of parametric assumptions.