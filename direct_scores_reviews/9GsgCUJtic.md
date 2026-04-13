## Summary
This paper investigates three interconnected questions about GFlowNets: (1) how balance violations propagate to affect distributional accuracy (TV bounds in Theorem 1, Weighted DB loss), (2) what expressiveness limits GNN-based GFlowNets face when sampling from graph distributions (Theorems 2–4, LA-GFlowNets), and (3) how to tractably and reliably assess GFlowNet correctness (the FCS metric, Theorem 5). Together, the contributions provide a principled theoretical framework for understanding *when* GFlowNets succeed, along with practical methodology for training and evaluation.

---

## Strengths

- **Novel TV sensitivity analysis connecting local flow imbalance to global distributional error.** Theorem 1 establishes tight bounds showing that balance violations near the root of the state graph have disproportionately larger impact than those near leaf states, formalized for arbitrary DAGs and multimodal rewards. This non-obvious heterogeneity is empirically confirmed in Figure 3 across four benchmark tasks, and translates directly into the WDB design principle.

- **Compelling impossibility result (Theorem 3) and targeted remedy (LA-GFlowNets).** The construction in Figure 5 is compact yet illustrative: two 1-WL-indistinguishable actions leading to children with different subtree rewards provably cannot be resolved by any 1-WL GFlowNet. The LA-GFlowNet formulation (Eq. 7) is a minimal and theoretically justified extension — adding child-state embeddings — that provably overcomes this expressiveness barrier (Theorem 4). The insight that a widely-used class of policy networks has a structural blind spot is practically important for graph-domain applications.

- **FCS as a computationally tractable and theoretically grounded evaluation metric.** FCS achieves Spearman correlation of 0.99 (sets) and 0.90 (sequences) with TV distance while being up to three orders of magnitude faster to compute (Figure 7). Theorem 5 provides the right faithfulness property (FCS=0 ↔ TV=0), and the metric's relationship to ratio matching and TV is cleanly characterized through the β interpolation.

- **Case study in Section 5.2 exposing a critical methodological flaw in prior evaluation.** The demonstration that terminally-unrestricted LED- and FL-GFlowNets attain perfect Shen accuracy (100 ± 0.00, Table 2) and outperform standard GFlowNets on exploration metrics while being provably distributionally incorrect is striking. Proposition 1 identifies the precise theoretical cause. This finding is practically important for the GFlowNet evaluation community and goes beyond a generic "standard metrics are imperfect" complaint.

---

## Weaknesses

- **Theorem 1 covers only a single, localized perturbation; the multi-perturbation regime is unaddressed.** In a trained GFlowNet, balance violations occur simultaneously at many edges. No result bounds the cumulative effect of multiple imbalances — even a triangle-inequality-style additive bound would partially address this gap. Without it, the theorem can characterize sensitivity to individual imbalances but cannot directly predict the total distributional error from training, which limits the direct practical impact of the theoretical result.

- **LA-GFlowNets are only validated on narrow synthetic experiments; no real benchmark evaluation.** Figure 6 tests four triples with n=8, k=3, binary rewards, and noiseless settings. No experiment evaluates LA-GFlowNets on any of the paper's own benchmark tasks (sequences, phylogenetics, sets, hypergrid), let alone on molecule generation. For a methodological contribution at ICLR, this is a notable gap: practitioners cannot assess whether the expressiveness gain is practically meaningful or computationally feasible on tasks of realistic scale.

- **Theorem 3's tree-structured SG assumption is not addressed for DAGs.** The paper's impossibility result formally requires the state graph to be a directed tree. Many real GFlowNet applications (e.g., set generation, hypergrid, molecule generation) use DAG-structured state graphs. Whether the impossibility extends to DAGs — or whether additional paths in a DAG provide workarounds for a 1-WL-based policy — is not discussed, leaving an important theoretical gap.

- **FCS coverage concern: the metric may be blind to modes that the learned policy underrepresents.** FCS computes TV on subsets drawn from trajectories sampled from the current policy. If the GFlowNet has assigned near-zero probability to a portion of the support, those states are systematically absent from the subsets, so FCS may appear small even when the GFlowNet has significant mode coverage failure. This is precisely the failure mode most important to detect. The paper does not discuss this limitation, and the PAC bound in Corollary 2 contains the term (#X / 2β) · max|p_T(S) − π(S)|, which could render the bound vacuous for large state spaces — also not addressed explicitly.

- **The implicit critique of Pan et al. (2023a) and Jang et al. (2024) is asserted in the main text but not substantiated there.** The paper writes "we have significant reasons to believe that an unrestricted F(x) was a part of some experiments in the original works of Pan et al. (2023a) and Jang et al. (2024)" and defers to Appendix E.3. If this claim is supported by evidence, it should be presented prominently; if it is speculative, the language in the main text is too assertive and potentially unfair to the cited authors.

- **Computational cost and scalability of LA-GFlowNets are not reported.** Computing the embedding of every child state requires evaluating the GNN on the successor graph for each candidate action, adding cost proportional to the branching factor. For tasks with large action spaces (e.g., molecular graphs), this overhead could be prohibitive. No runtime analysis, memory profile, or discussion of how to scale or approximate child embeddings is provided.

- **WDB's weighting (γ = 1/#D_{s'}) requires counting terminal descendants, which is intractable for large DAGs.** The paper evaluates WDB only on benchmarks where this counting is feasible. For the primary target application (molecule generation), enumerating #D_{s'} is generally intractable. The paper acknowledges this in limitations but provides no approximation strategy, which significantly limits the portability of WDB to the applications that most need faster convergence.

---

## Nice-to-Haves

- Sensitivity analysis of WDB to the choice of γ (e.g., comparing inverse-descendant-count to exponential decay or depth-based functions) would help practitioners understand how to adapt WDB to new domains.
- An ablation study varying β in FCS across different state space sizes would characterize how to tune the metric for large-scale settings.
- Visualization of the distribution of WDB weights γ across trajectory depths for a complex task would visually confirm the theoretical claim that early transitions dominate.
- Evaluation of WDB combined with LA-GFlowNets on at least one shared task, since real failures likely involve both imbalance and expressiveness deficits simultaneously.
- A stochastic sampling strategy for LA-GFlowNet child embeddings (sampling a subset of children rather than enumerating all) would make the approach tractable for large branching factors.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"WDB only clearly helps 2/4 tasks — claim of 'often accelerates' is overstated."** The paper's own Section 3.2 explicitly explains the conditions under which WDB helps ("Note these two environments are exactly the ones for which early-stage transitions dominate the loss, as shown in Figure 3"), and "approximately on par" for the other two tasks is not a failure. The result is coherent and the claim is defensible. The harsh reviewer conflated a nuanced finding with a misleading one.

- **"The introduction's claim of 'up to three orders of magnitude less compute' is only valid for small state spaces."** The paper qualifies this as comparing to exact TV computation, which is the baseline being discussed. The claim is accurate within the stated scope; it is not claimed to apply when exact TV is also intractable.

- **"FCS sensitivity to β is not explored — this is a significant flaw."** This is a reasonable suggestion for future analysis but does not undermine the core validity of FCS as a metric. Theorem 5 holds for any β ≥ 2, and the empirical correlations in Figure 7 are strong. Requesting an exhaustive ablation is more of a nice-to-have.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the coupling between *measurement failure and methodological failure*: the paper shows that commonly used evaluation metrics (Shen's accuracy, top-k reward) can assign perfect scores to models that are provably distributionally incorrect, and does so concretely by linking the theoretical pathology (Proposition 1: unconstrained terminal flows yield marginals proportional to R(x)·F̃(x) rather than R(x)) to the practical metric failure (Table 2). This creates a compounding problem — not only can GFlowNets fail silently due to GNN expressiveness limits or balance violation propagation, but the field's standard diagnostic tools may not detect the failure. The combination of Theorem 3, Proposition 1, and Table 2 constitutes a coherent cautionary argument that goes beyond any individual component.

---

## Suggestions

1. **Extend Section 5.2 or add an appendix section with the full evidence regarding Pan et al. and Jang et al.** Either present the evidence from E.3 in the main text or tone down the main-text claim to "we present evidence in Appendix E.3 suggesting..." This affects both the paper's credibility and fairness to cited authors.
2. **Add at least one real-graph benchmark for LA-GFlowNets** (e.g., the phylogenetics or molecule generation tasks) with runtime reporting to establish practical viability. The current synthetic-only validation significantly limits the contribution's reach.
3. **Explicitly discuss the FCS coverage limitation** (modes underrepresented by the current policy) and, if possible, propose or discuss a remediation (e.g., using an exploratory backward policy or a mixture policy for subset construction).
4. **Provide an approximation or upper-bound strategy for #D_{s'}** in WDB for DAG-structured state graphs, even as a heuristic, to extend the method's applicability beyond enumerable benchmarks.
5. **Discuss the cumulative balance violation case**, even informally — a simple observation that the single-perturbation bound can be applied repeatedly via a union-bound-style argument (with known pessimism) would substantially increase the theorem's practical interpretability.

---

**Novelty:** High — the combination of TV sensitivity analysis, GNN expressiveness impossibility for GFlowNets, and the FCS metric addresses genuinely underexplored territory in the GFlowNet literature, and each piece is technically original.

**Technical soundness:** Good — all main theorems appear correct and the proofs cover the stated claims. The primary caveat is that Theorem 1 and Theorem 3 have scope restrictions (single perturbation; tree-structured SG) that are not always clearly foregrounded.

**Empirical support:** Moderate — strong for WDB and FCS on standard benchmarks, but LA-GFlowNets are only validated synthetically. The coverage of FCS on large-scale tasks is argued theoretically but not demonstrated empirically.

**Significance:** Above average — the FCS case study and expressiveness impossibility results are of direct practical value; the WDB contribution is incremental but useful.

**Clarity:** Good — Table 1 provides an unusually effective roadmap for a multi-contribution paper, and the theorems are precisely stated. The logical thread connecting the three sections is thematic rather than technical, but the paper is transparent about this.

MY FINAL SCORE: <pineapple>6.8</pineapple>