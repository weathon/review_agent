## Summary

EGG-SR introduces a unified framework that integrates symbolic equivalence into symbolic regression via equality graphs (e-graphs), applying the same core idea—grouping syntactically different but semantically equivalent expressions—to three algorithmic paradigms: MCTS (pruning redundant subtrees), DRL (aggregating rewards across equivalent trajectories), and LLMs (enriching feedback prompts). The paper provides theoretical analysis claiming tighter regret bounds for EGG-MCTS and lower-variance gradient estimators for EGG-DRL, along with empirical improvements on trigonometric and real-world benchmarks.

## Strengths

- **Novel and well-motivated idea:** The observation that symbolic equivalence creates redundant exploration in SR, and the proposal to use e-graphs as a unified mechanism to expose and exploit this structure across multiple paradigms (MCTS, DRL, LLM), is original and well-conceived. This goes beyond prior e-graph use in SR (which focused on post-hoc simplification and duplicate detection) to embed equivalence awareness directly into learning objectives.

- **Broad applicability within the paper's scope:** The same EGG module enhances three quite different SR paradigms. The engineering of adapting e-graphs to grammar-based expressions, with extraction via random-walk sampling and cost-based methods, is clean and modular. Table 1 shows consistent NMSE reductions for EGG-MCTS and EGG-DRL, and Table 2 shows improvements for EGG-LLM on most metrics.

- **Theoretical grounding (with caveats noted below):** Theorem 3.1 connecting equivalence-induced branching factor reduction to tighter MCTS regret, and Theorem 3.2 on variance reduction for DRL, provide formal motivation even if they apply to idealized variants. The core insight—that grouping equivalent trajectories with identical rewards reduces estimator variance—is sound and well-stated.

- **Space efficiency results are convincing:** Figure 4 clearly demonstrates the exponential memory advantage of e-graphs over array-based storage for equivalent variants, and Figure 5 shows EGG overhead is modest relative to other pipeline components.

## Weaknesses

### Major

- **Gap between theoretical claims and the implemented algorithms.** Theorem 3.1 establishes regret bounds for EGG-MCTS by assuming full equivalence-class merging and statistical sharing across all equivalent nodes—effectively an idealized transposition table. The actual algorithm (Section 3.2) samples *several* distinct equivalent sequences from the e-graph and backpropagates to paths that happen to be found, which is a heuristic partial approximation. Similarly, Theorem 3.2 proves unbiasedness and variance reduction, but Eq. (4) uses only K sampled equivalent sequences rather than the full equivalence class (which may be exponentially large). If K < |equivalence class|, the estimator approximates rather than instantiates the theory. The paper should explicitly acknowledge these gaps and discuss their practical implications rather than presenting the theorems as applying directly to the implemented methods. As written, the claims "our theoretical analysis establishes the advantages" (Sec. 6) overstate what is rigorously demonstrated.

- **Narrow empirical evaluation does not support broad generalization claims.** EGG-MCTS and EGG-DRL are evaluated exclusively on trigonometric datasets (Table 1) where rewrite rules for sin/cos identities are highly applicable. This is essentially a best-case scenario. Standard SR benchmarks (SRBench, full Feynman, Nguyen, etc.) where rule coverage is sparser or different are absent. The paper claims EGG-SR "consistently enhances a class of symbolic regression models across several benchmarks" (Abstract), but this is supported only for carefully selected domains. An ablation removing the trig-specific rules or evaluation on polynomial/algebraic-heavy benchmarks would clarify whether the gains come from the general framework or from hand-matched rules.

- **Soundness of the equal-reward assumption is unaddressed.** Both theoretical results rely on the property that equivalent expressions under rewrite set R produce identical rewards. Domain restrictions (e.g., log(a·b) = log(a)+log(b) requires a,b > 0; division rewrites can fail when denominators are zero) mean that ≡_R-equivalent expressions may not always agree on the data. The paper does not discuss: (a) that equivalence is relative to R rather than true functional equivalence, (b) where R might be unsound for specific data distributions, or (c) the practical impact when the equal-reward assumption is imperfectly satisfied. This matters because the variance-reduction proof (Theorem 3.2) breaks if pooled trajectories do not share the same reward.

### Minor

- **Incomplete specification of the EGG-DRL estimator.** Eq. (4) introduces ∇_θ log[Σ_k p_θ(τ_i^(k))] without clarifying whether the sum is over the full equivalence class or only the K sampled sequences, nor how τ_i^(k) are sampled from the e-graph. If K < |equivalence class|, the estimator is a biased approximation of the full equivalence-class gradient, and this bias is unanalyzed. The paper would benefit from a discussion of this approximation error and an ablation on the effect of K.

- **LLM integration is shallow and results are mixed.** EGG-LLM augments prompts with equivalent expressions—a simpler mechanism than the learning-objective modifications in MCTS/DRL. Table 2 shows some entries where EGG-LLM underperforms the baseline (e.g., EGG-LLM Mistral on Oscillation II IID: 0.0021 vs. LLM-SR Mistral's comparable performance on other cells, and Bacterial growth OOD where EGG-LLM Mistral is worse). The paper does not analyze when or why prompt enrichment helps versus hurts.

- **No comparison with prior e-graph-based SR methods.** The paper distinguishes itself from de França & Kronberger (2023, 2025) as using e-graphs for "equivalence-aware learning" rather than simplification/duplicate detection, but doesn't empirically compare against these methods to demonstrate that the proposed approach is superior to existing e-graph+SR combinations.

- **Potential bias from asymmetric discoverability.** Expressions with many discoverable equivalents (e.g., those involving trig identities) dominate the aggregated probability in Eq. (4), potentially biasing learning toward particular operator families. This concern was raised in reviews of closely related work (DSR-Rex) and is not discussed.

### Trivial

- The paper occasionally uses language like "represent the same mathematical function" (Abstract, Sec. 1) for what is more precisely "represent the same function under a given rewrite system R." This is a framing issue, not a technical error, since the implementations correctly use ≡_R.

## Nice-to-Haves

- Evaluate on SRBench or the full Feynman dataset to demonstrate generality beyond trigonometric-rich domains.
- Run an ablation study on the rewrite rule set (e.g., removing trig-specific rules while retaining others) to disentangle framework gains from rule-set alignment.
- Provide wall-clock-time comparisons (not just iteration-based) for EGG-DRL vs. baseline DRL.
- Discuss how the rule set could be extended or adapted for new domains, and the risk of including domain-unsound rules.
- Compare explicitly against de França & Kronberger's GP-based e-graph approach on shared benchmarks.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"EGG-DRL gradient estimator is not clearly well-defined" (Harsh Critic, Issue 3, full structural claim):** The critic claimed the estimator is fundamentally ill-specified and possibly invalid. However, the core idea—grouping equivalent trajectories that share the same reward and differentiating the log-probability of the equivalence class—is a sound application of importance-weighted aggregation. The specific form in Eq. (4) follows naturally: if all τ in an equivalence class E have reward r_E, then ∇_θ log P(E) scaled by r_E equals the sum of individual gradients. The concern about finite K introduces approximation error but does not invalidate the estimator conceptually. Downgraded from "structural" to "minor" above.

- **"Evaluation scope is extremely narrow and does not support broad claims" (Harsh Critic, Issue 4, extreme version):** The critic claimed the evaluation "does not support" any general claims. While the evaluation is indeed narrower than ideal, the paper does show consistent improvements across three paradigms, includes noiseless/noisy settings, and uses real-world datasets for LLM. The critique that claims are overstated is valid, but the evaluation is not as empty as characterized. Downgraded to "major" with appropriate nuance.

- **"Lack of fair baseline configuration" (Harsh Critic, Issue 5):** The concern about unequal experimental conditions for LLM (using reported numbers from Shojaee et al.) and insufficient hyperparameter details for DRL is valid but is a standard limitation in comparison studies. Using the same reported numbers with the same settings is a common practice and not a fundamental fairness issue. Removed as a standalone weakness; relevant nuances are captured under the LLM evaluation discussion.

- **"Formatting/style nitpicks" and "missing related works"**: Removed per hard rules.

- **Reproducibility concerns about undisclosed hyperparameters:** Removed per hard rules (trivial implementation details impractical to include).

## Novel Insights

The most novel conceptual contribution is the recognition that equivalence-graph-based grouping can serve as a variance-reduction mechanism for policy gradients (EGG-DRL), analogous to Rao-Blackwellization over equivalence classes. While transposition tables in MCTS and symmetry-aware RL have precedents, the specific adaptation of e-graphs to group trajectory probabilities in REINFORCE-style estimators—replacing ∇log p(τ) with ∇log Σ_k p(τ^(k))—is a distinct technical move that, if properly justified (including handling finite-K sampling), could have broader applicability beyond SR.

## Suggestions

1. **Add a clear "Assumptions and Limitations" paragraph** in Section 3.4 or a new Section 3.5 that explicitly states: (a) Theorems apply to the idealized algorithm with full equivalence-class merging; (b) The implemented algorithm samples K sequences, introducing an approximation; (c) The soundness of the rewrite system R is assumed and domain restrictions are noted; (d) Incompleteness of R only limits the effect, not the correctness, of EGG.

2. **Add at least one broader benchmark** (e.g., a subset of Feynman with non-trig expressions) and an ablation varying the rewrite rule set, to establish that EGG's benefits extend beyond trigonometric-friendly domains.

3. **Include an ablation on K** (the number of equivalent sequences sampled) in EGG-DRL, reporting both performance and gradient variance, to empirically characterize the approximation quality.

## Score and Decision

**Calibration anchors:**
- DSR-Rex (scores 3, 3, 5, 3, 5 → Reject): Shared core idea of equivalence-aware DRL for SR, but with even weaker theory and narrower scope. EGG-SR is clearly stronger—it covers three paradigms, has formal analysis, and more thorough experiments.
- PCGSR (scores 3, 8, 5 → Reject): Had novelty concerns about combining existing ideas and needed wall-clock comparisons. EGG-SR has similar issues but with more genuine novelty in the unified framing.
- RSRM (scores 6, 8, 6, 6 → Accept/poster): Had theoretical and evaluation gaps but was deemed sufficient for a poster. EGG-SR is comparable in contribution quality but has more significant overclaimed theory.
- CADSR (scores 5, 6, 5, 5 → Reject): Flagged for compute overhead and limited novelty. EGG-SR has stronger novelty and broader scope.

EGG-SR sits above DSR-Rex and PCGSR (which were rejected) in novelty and scope, but below RSRM (which was accepted) in theoretical rigor and evaluation breadth. The overclaimed theorems and narrow evaluation are significant but not fatal—the core idea is genuinely interesting and the approach works. A score of 5 reflects a borderline paper with a good idea that needs revision to meet its claims.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>