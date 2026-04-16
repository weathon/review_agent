## Summary

This paper investigates when GFlowNets correctly learn to sample from their target distribution, making three connected contributions: (1) TV bounds showing that balance violations near the root of the state graph have a disproportionate impact on distributional accuracy, motivating a weighted detailed balance (WDB) loss; (2) impossibility results for 1-WL GNN-parameterized GFlowNets on certain graph distributions, along with a Look-Ahead GFlowNet (LA-GFlowNet) construction that overcomes these limitations; (3) the Flow Consistency in Subgraphs (FCS) metric, a theoretically grounded and computationally tractable proxy for the TV distance to assess GFlowNet convergence.

## Strengths

- **Addresses a fundamental and underexplored question.** Understanding when GFlowNets learn the correct distribution is critical for the field, and this paper provides the first systematic treatment connecting balance violations, parameterization limits, and evaluation inadequacy — three issues that have been largely treated in isolation.

- **The FCS metric is a significant practical contribution.** The theoretical grounding (FCS=0 iff TV=0, Theorem 5) combined with strong empirical correlation (Spearman 0.99 and 0.90 on two tasks) and up to 1000× speedup over TV makes FCS a compelling addition to the GFlowNet toolkit. The demonstration that existing metrics (number of modes, top-k average reward, Shen's accuracy) can assign perfect scores to provably miscalibrated models (Figure 8, Table 2) is an important finding that should influence future evaluation practices.

- **Theorem 3 is a clean, insightful impossibility result.** The condition — two 1-WL indistinguishable actions from the same state leading to children with different cumulative reward — is transparent, the construction in Figure 5 is intuitive, and the result directly connects known GNN expressivity limits to concrete distributional failure in GFlowNets.

- **The heterogeneous error propagation insight (Section 3) is well-motivated and empirically supported.** Figure 3 directly validates the theoretical prediction that DB loss is unevenly distributed along trajectories, and WDB shows meaningful improvement on two out of four benchmarks (Figure 4).

- **Coherent narrative structure.** The paper progresses logically from consequences of imbalance (Section 3) to causes (Section 4) to diagnostic tools (Section 5), with each section containing both theory and experiments.

## Weaknesses

### Major:

- **Limited experimental validation, especially for LA-GFlowNets and FCS at scale.** LA-GFlowNets are evaluated only on the hand-crafted pathological construction of Figure 6 (n=8 regular graphs where 1-WL failure is guaranteed by design). No experiment tests whether LA-GFlowNets provide measurable benefit on realistic graph-generation tasks where 1-WL ambiguities may arise but are less extreme. Similarly, FCS-TV correlation is shown on only two of four tasks (Sets, Sequences), and all experiments use small-scale benchmarks where TV is still computable. The strongest practical motivation for FCS is precisely the regime where TV is intractable, yet this remains unvalidated. The paper acknowledges omitting "specialized applications such as molecule generation," but this omission significantly weakens the practical impact claims for all three methodological contributions.

- **Gap between the theoretical TV bounds and the WDB loss.** Remark 1 and Theorem 1 analyze a specific perturbation model (single-node δ-increase redirected deterministically to one child) that does not reflect the distribution of balance errors arising from SGD on neural networks. The proposed weighting γ(s,s') = 1/#D_{s'} is only loosely connected to the theory: Remark 1 suggests weighting by descendant *probability mass*, but the implemented weight uses descendant *count*. While the qualitative insight ("edges near the root matter more") is valid and empirically supported, the specific WDB scheme is heuristic, and its improvement over standard DB is inconsistent (significant on 2/4 benchmarks, neutral on 2/4).

- **Computational cost and scalability of LA-GFlowNets is not analyzed.** Equation 7 requires computing graph embeddings for all children states at every forward step. For state graphs with high branching factors, this could add substantial overhead per sampling step. The paper provides no wall-clock time comparison, no discussion of how cost scales with branching factor or graph size, and no exploration of approximations (e.g., sampling a subset of children). Without this analysis, it is unclear whether LA-GFlowNets are viable beyond toy constructions.

### Minor:

- **FCS is not compared against any other principled distributional metric.** The paper argues FCS is "the only alternative" that is both theoretically sound and computationally tractable, but does not compare against other candidate metrics (e.g., MMD with a suitable kernel, kernel Stein discrepancy, or subsampled L1 estimators). While such comparisons may be scarce in the GFlowNet literature, at least one comparison would strengthen the case that FCS is uniquely positioned.

- **The framing occasionally generalizes parameterization-specific results to "GFlowNets" broadly.** The abstract states there are "simple state graphs and target distributions from which no GFlowNet can correctly sample," but Theorem 3 only applies to GFlowNets with 1-WL GNN policies. Similarly, the conclusion states "the limited expressivity of GNN-based GFlowNets" as a "limitation of GFlowNets" in the broader sense. While the paper does specify the parameterization context in Section 4, the abstract and conclusion could be read as claiming impossibility for all GFlowNets, which is not what the theorem establishes.

- **The PAC bound for FCS (Corollary 2) depends on a quantity that is hard to control.** The term max_S |p_T(S) - π(S)| is difficult to bound in practice, especially early in training when the model is poorly fitted. This limits the practical applicability of the bound as a convergence guarantee, though FCS remains empirically useful as a diagnostic.

## Nice-to-Haves

- An ablation study on the WDB weighting function γ, comparing 1/#D_{s'} against other natural choices (e.g., exponential decay in depth, learned weights), to understand when weighting helps and why.

- Evaluation of LA-GFlowNets on at least one standard graph-generation benchmark (e.g., molecule generation) to assess whether the theoretical expressivity gains translate to practical improvements.

- Comparison of FCS against at least one other principled distributional metric (e.g., MMD) on the same benchmarks, to substantiate the claim of unique adequacy.

## Removed Points

- *"The perturbation model in Theorem 1 doesn't reflect realistic SGD errors"* — While the specific δ-redirection construction is stylized, the qualitative conclusion that edges near the root disproportionately affect TV is empirically validated (Figure 3). The purpose of Theorem 1 is to provide structural insight, not to model exact SGD dynamics. Over-strict demand for SGD-realistic perturbation models would be scope creep for a theoretical paper.

- *"The 'universal approximation for trees' (Theorem 2) is overstated because the SG construction is specialized"* — Theorem 2 is an existence result that sets up the contrast with Theorem 3. The paper doesn't overclaim practical consequences of this theorem; it simply establishes what 1-WL GFlowNets *can* do before showing what they *cannot*. The existence result is itself a contribution.

- *"The TU-FL/LED case study is artificially pathological and doesn't establish that standard metrics are unreliable in typical settings"* — The paper explicitly tests both pathological (TU variants) and standard (TB) GFlowNets in Figure 8. Even for the TB baseline, the number of modes and top-k reward metrics do not track TV well, demonstrating the issue is not solely driven by the pathological construction.

- *"Reproducibility concerns about whether prior works actually used unrestricted terminal flows"* — The paper makes a measured claim ("significant reasons to believe") and provides supporting analysis in Appendix E.3. Flagging specific prior works for methodological issues is a legitimate scientific contribution, not a weakness.

- *"No confidence intervals or error bars on WDB convergence plots (Figure 4)"* — This is a minor presentation concern. The convergence plots show clear trends on the tasks where WDB helps, and the paper provides multiple runs in supplementary figures.

## Novel Insights

The most novel insight is the tight coupling between evaluation methodology and scientific progress in GFlowNets: the paper shows that the field's standard evaluation metrics (modes found, top-k reward, Shen's accuracy) can be actively misleading — assigning perfect scores to provably miscalibrated models — and provides both the theoretical framework (FCS) and the concrete counterexample (TU-LED/FL + Table 2) to demonstrate this. This is a cautionary result that should reshape how GFlowNet methods are evaluated going forward. A second notable insight is that the impact of balance violations is *heterogeneous* across the state graph, with flow errors near the root being more damaging, which challenges the implicit equal-weighting assumption of the standard DB loss.

## Suggestions

- Evaluate FCS on at least one task where TV is genuinely intractable (e.g., molecule generation), even approximately, to demonstrate its utility in the regime where it matters most.
- For LA-GFlowNets, provide wall-clock time comparisons and discuss the computational overhead; consider proposing an approximation (e.g., random subset of children embeddings) for high-branching-factor settings.
- Tone down the "FCS is the only alternative" claim or substantiate it by comparing against at least one other principled distributional metric.

## Score and Decision

I compared this paper against several calibration anchors:

- **PJNhZoCjLh** (GFlowNet theory, avg ~6.25): Similar profile — theory-heavy GFlowNet paper with limited experimental scale. This paper has stronger practical contributions (FCS is genuinely useful) and more diverse contributions.
- **4NTrco82W0** (GFlowNet loss modification, avg ~7.3): Similar depth on the loss modification aspect but this paper has broader scope (3 contributions). However, this paper has weaker experimental validation for LA-GFlowNets and more overclaiming.
- **BkR4QG4azn** (GFlowNet graph generation, avg ~5.6): Weaker than this paper — less theoretical depth and no equivalent of the FCS contribution.
- **P15CHILQlg** (LED-GFlowNet, avg 8): Stronger than this paper — more thorough empirical evaluation and cleaner scope.
- **HSKaGOi7Ar** (GNN expressivity, avg ~8.5): Stronger GNN expressivity results with better experimental validation.

This paper makes genuine contributions: the FCS metric is practically important, Theorem 3 is clean and insightful, and the TU-LED/FL counterexample is a valuable cautionary finding. However, the experimental limitations (no real-world tasks, LA-GFlowNets only on toy constructions, FCS validation on only 2/4 tasks), the heuristic gap between theory and WDB, and the computational concerns about LA-GFlowNets prevent a higher score. The paper sits above the weak theory papers (~5.5-6) but below the well-validated applied papers (~7.5-8), with the FCS contribution being the strongest element.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>