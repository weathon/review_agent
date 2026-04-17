Now I have good calibration data. Let me carefully analyze the harsh critic's concerns against the paper text and write my final review.

Key calibration points:
- IPayPEGwdE (Causal contextual bandits): scores 5,5,5,5 → avg 5
- MVpvyeVeyI (Causal Bayesian Optimization, unknown graphs): scores 10,3,8,5 → wide range
- oVVLBxVmbZ (Fast conditional intervention in recourse): scores 5,5,6,5 → avg ~5.25
- YcW8i9VCf5 (Adversarial CBO): scores 6,6,6,6 → avg 6
- M0xK8nPGvt (Causal graph priors in RL): scores 8,8,6,8 → avg 7.5
- HSKaGOi7Ar (GNN expressiveness, oral): scores 8,10,8,8 → avg 8.5

Now let me verify the harsh critic's key concerns:

**Concern 1 (worst-case superiority vs. bandit objective):** The critic argues that mGISS guarantees optimality across ALL SCMs with the same graph, but in practice the bandit has a FIXED unknown SCM. This means nodes optimal for a specific SCM might be excluded. Let me verify against the paper.

Looking at Definition 5: "A globally interventionally superior set (GISS) of G relative to Y, is a subset U of V \ {Y} satisfying U ⪰Y (V \ {Y}) \ U." And Definition 1: X ⪰c_Y W means "for all SCM with causal graph G there is a policy g for X such that for every observable conditioning sets Z_X and Z_W and all policies h for W, E[Y under X=g(Z_X)] ≥ E[Y under W=h(Z_W)]."

So the ⪰Y relation says X dominates W uniformly across ALL SCMs with graph G. A node W is excluded from mGISS if there exists some X such that for ALL SCMs, X is at least as good as W. This means a node W could be optimal for some particular SCM but still be excluded if there exists X that is always at least as good across all SCMs.

The critic's point is: this is a worst-case guarantee. For a fixed unknown SCM, the truly optimal node might be outside mGISS. This is correct - if W is the best node for some SCM C but for every SCM there's some X that dominates W (even though it's a different X for different SCMs), then W is excluded. Actually wait - Definition 1 says X ⪰c_Y W means for ALL SCMs with graph G, there exists a policy g for X that is at least as good as any policy for W. So if X dominates W for ALL SCMs, then W is indeed suboptimal in every possible world. For a fixed SCM C, if X dominates W in C, then W can never be better than X in C.

Actually, let me re-read more carefully. Definition 1: X ⪰c_Y W if for all SCMs with causal graph G, there is a policy g for X ... such that for all ... the expectation under X=g is ≥ expectation under W=h. So X strictly dominates W if in every possible SCM, X can achieve at least as good Y as W. This means W can never be the unique best node in any SCM, because X is always at least as good. So removing W from the search space is safe - you'll never miss the optimum, because X (which remains in the search space) is always at least as good.

Wait, but the mGISS includes nodes that are NOT dominated by any other node. The mGISS contains exactly the nodes where no single other node can universally dominate them. A node W is excluded if there exists some X such that in EVERY SCM, X is at least as good. But this doesn't mean W can't be optimal in some SCM! It means X is always at least as good as W, so if W is optimal, X is also at least as good as W in that SCM, so you wouldn't lose optimality by choosing X instead of W.

Actually this is a subtle point. If X uniformly dominates W (i.e., X ⪰Y W), then for every SCM, the best policy on X gives at least as much reward as the best policy on W. So the optimal value achievable on X is always ≥ the optimal value achievable on W. Therefore, even if W gives some reward level r in a particular SCM, X gives at least r, so you wouldn't lose anything by not considering W. The node X can achieve everything W can.

So the critic's claim that "In those SCMs, restricting to mGISS guarantees that the bandit cannot reach the truly optimal node" seems INCORRECT. If W is strictly dominated by X (X uniformly dominates W), then X can achieve at least as high a reward as W in every SCM. So removing W doesn't lose optimality - X is always at least as good. The "truly optimal node" could be any node in the mGISS, and they will always achieve at least as high reward as excluded nodes.

Actually wait - let me re-check. The mGISS is the MINIMAL set such that mGISS ⪰Y (V \ {Y}) \ mGISS. So every node outside mGISS is dominated by some node in mGISS. Therefore for any SCM, for any node W outside mGISS, there exists a node X in mGISS that gives at least as much reward. So the optimal reward achievable using mGISS nodes equals the optimal reward achievable using all nodes. The mGISS does indeed guarantee containing an optimal intervention for ANY SCM consistent with G!

So the harsh critic's Critical Issue 1 appears to be WRONG. The mGISS does guarantee that for any fixed SCM, there exists a node in the mGISS that achieves the same optimum reward as the best possible node overall. The point is that uniform domination means the dominating node is never worse, so you don't lose any reward potential by removing the dominated node.

Wait, let me be even more careful. The definition says X ⪰c_Y W means: for ALL SCMs C with graph G, and ALL observable conditioning sets, there exists a policy g on X such that the expected Y under do(X=g(Z_X)) ≥ expected Y under do(W=h(Z_W)) for ALL h, Z_X, Z_W. 

This is a very strong statement. It means X can achieve at least as much reward as W in ALL SCMs. Therefore, removing W cannot reduce the achievable optimium - X will always be at least as good as W, so the best possible reward achievable without W is at least as high as with W.

So the critic's Concern 1 is fundamentally flawed. The mGISS guarantee IS applicable to any fixed SCM. The worst-case quantification doesn't weaken the guarantee - it STRENGTHENS it. If X dominates W across all SCMs, then in any particular SCM, X is still at least as good as W. So restricting to mGISS preserves optimality in every single SCM.

This is actually a strength of the paper - the guarantee holds pointwise for every SCM, which is even better than what the critic suggests.

**Concern 2 (Deterministic atomic vs. conditional intervention equivalence):** The critic argues that the proof of Proposition 4 is deferred and the main text doesn't provide intuition. This is a valid presentation concern, but the proof IS in the appendix and the result IS stated. The concern about whether the proof correctly handles the quantifiers is legitimate as a "trust but verify" issue, but it's not a substantive criticism of the paper's claims - it's about wanting more proof detail in the main text. This is a presentation issue, not a fundamental one.

Let me think about whether the conceptual link is plausible. In the deterministic case (fixed n), the reward Y is a function of the intervention. For conditional interventions, you can condition on ZX and set X optimally for each realization of ZX. In the deterministic case, n is known, so you just pick the best x. The claim is that X dominates W under conditional interventions iff X dominates W under deterministic atomic interventions. This is plausible because:
- If you can do better with conditional interventions on W than any atomic intervention on X (in expectation over noise), then there must be some realization of noise where this is true, and at that n, W with some fixed w would beat any fixed x for X. Contrapositively, if for every n there exists an x that beats all w, then conditional interventions on X can match conditional interventions on W.

This seems conceptually sound. The proof is in the appendix. The critic's concern is valid as a "readers can't easily verify this core claim" issue, but not as a fundamental flaw.

**Concern 3 (Experiments don't validate minimality or preservation of optimal node):** The critic argues that the bandit experiments only show that pruning arms helps UCB, which is trivially true. However, the paper's main theoretical claim (mGISS preserves optimal nodes) is a theorem, not something that needs empirical validation. The experiments validate the PRACTICAL UTILITY of the pruning: that the pruning ratio is substantial and that integrating it into a bandit algorithm helps. The claim that the mGISS preserves optimality is proven (Theorem 13), not claimed to be shown experimentally. However, the critic's point about not comparing to simpler baselines (like just pruning to Pa(Y)) is valid and worth including.

**Concern 4 (Restrictive assumptions on Z_X):** The critic notes that the results quantify over ALL observable conditioning sets, and the mGISS doesn't depend on the specific choice of Z_X. This is actually a STRENGTH - the guarantee holds regardless of Z_X choice. The fact that experiments use the smallest Z_X is fine because the theory covers all Z_X choices.

Now let me summarize my assessment:

The harsh critic's main structural concern (Issue 1) is actually INCORRECT - I've verified that the mGISS guarantee DOES hold for any fixed SCM, because uniform domination across all SCMs implies domination for any specific SCM. The mGISS is guaranteed to contain an optimal node for ANY SCM consistent with G. This is NOT a weakness of the paper; it's the paper's core contribution.

Issue 2 has some merit as a presentation concern (proof of Proposition 4 in appendix with no intuition in the main text), but is not a fundamental flaw.

Issue 3 has merit regarding the lack of comparison to simpler pruning baselines.

Issue 4 is not a real weakness - the independence of mGISS from specific Z_X choices is a feature.

Let me now write the final review.</think>Now let me carefully verify the most critical claim from the harsh reviewer against the actual paper text.

## Summary

This paper introduces the conditional causal bandit problem, where arms are single-node conditional interventions on a causal graph and the reward is a target variable Y. The main contribution is a complete graphical characterization of the minimal set of nodes (the mGISS) guaranteed to contain the optimal intervention node, shown to equal the LSCA closure of Pa(Y). A linear-time algorithm (C4) is provided to compute it, along with a surprising equivalence between conditional-intervention superiority and deterministic atomic-intervention superiority (Proposition 4).

## Strengths

- **Novel and well-defined problem formulation.** The paper introduces the first characterization of a minimal search space for causal bandits with non-hard (conditional) interventions. This extends Lee & Bareinboim (2018) in a meaningful direction: conditional interventions model real-world decision-making more faithfully than hard interventions, and single-node interventions present non-trivial challenges that multi-node interventions (where one simply intervenes on all parents) avoid.

- **Elegant graph-theoretic characterization.** The development from LSCA closures through Λ-structures (Definition 11, Theorem 12) to the main result (Theorem 13) is conceptually clean and mathematically satisfying. The Λ-structure characterization gives a simple, intuitive test: a node is in the mGISS iff it forms a Λ-structure over (Pa(Y), Pa(Y)).

- **Efficient algorithm with correctness guarantees.** The C4 algorithm (Algorithm 1) computes L∞(U) in O(|V| + |E|) time, which is optimal. The connector concept (Definition 14, Lemma 15) provides an intuitive implementation that maps directly to the Λ-structure characterization, and the correctness proof (Theorem 16) is well-structured.

- **Substantial empirical pruning on real-world graphs.** The search-space reduction experiments on bnlearn graphs are convincing: for large real-world models with low average degree (as typical in practice), C4 prunes over 90% of the search space in some cases, demonstrating real practical utility.

- **Surprising equivalence in Proposition 4.** The equivalence between conditional-intervention superiority (over stochastic SCMs) and deterministic atomic-intervention superiority is non-obvious and provides a powerful simplification that makes the subsequent analysis tractable.

## Weaknesses

### Major:

- **Experiments do not compare against simpler pruning baselines.** The bandit experiments compare "all ancestors" vs. "mGISS nodes," but never against simpler heuristics like pruning to just Pa(Y), or one-step LSCA without the recursive closure, or other simple graph-based pruning. Without this, it is impossible to know whether the full LSCA-closure characterization is necessary in practice, or whether a much simpler heuristic achieves similar regret reduction. The search-space reduction experiments similarly only report mGISS vs. An(Y), not vs. intermediate strategies.

- **Bandit experiments are limited in scope and design.** Only 4 bnlearn datasets are used, with Y always chosen as the node with the most ancestors (a cherry-picked best case for pruning). There is no evaluation across different target nodes. Furthermore, regret is computed against an "estimated best arm" rather than the true optimal conditional intervention, which means the experiments do not verify that the mGISS preserves optimality in the instantiated SCMs. While the preservation of optimality is a theoretical guarantee (Theorem 13), empirical validation that the true optimal node falls within mGISS for realistic SCMs would strengthen confidence in the theory and its practical applicability.

- **Core equivalence (Proposition 4) has no proof sketch in the main text.** This equivalence between conditional and deterministic atomic superiority is the load-bearing step that connects the intuitive deterministic analysis to the claimed conditional-intervention result. Its proof is entirely in the appendix, with no intuition or roadmap in the main body. Given the subtle quantifier structure (for all SCMs, there exists a policy g, for all policies h, for all conditioning sets), a high-level justification of why the equivalence holds is important for readers to assess the claim without consulting the appendix.

### Minor:

- **No latent confounders.** The assumption of no unobserved confounding is acknowledged as a limitation, but the paper provides no discussion of how the mGISS characterization degrades when confounders are present, or whether it serves as a conservative (over-inclusive) pruning under confounding. Even a brief conjecture would clarify scope.

- **Restriction to single-node interventions.** While justified as a first step, the paper does not discuss boundary cases (e.g., bounded-size interventions with |S| ≤ k), which would help readers understand the practical scope and the landscape between single-node and multi-node extremes.

- **CondIntUCB is a basic integration.** The bandit algorithm treats pruned nodes as independent contextual bandit instances. The paper does not explore whether causal structure among mGISS nodes could be further exploited (e.g., via variance sharing across interventions), which could yield additional gains beyond pruning.

### Trivial:

- None.

## Nice-to-Haves

- Comparison against Pa(Y) and one-step LSCA baselines in both pruning ratio and regret experiments
- Experiments varying the target node Y (not just the one with most ancestors)
- Empirical verification that the true optimal node falls in the mGISS for instantiated SCMs
- A proof sketch or intuition for Proposition 4 in the main text
- Discussion of how C4 could be adapted when multiple candidate causal graphs are available (beyond the one-line mention at the end of Section 1)

## Removed Points

- **"worst-case superiority notion mismatches the bandit objective" (Harsh Critic Issue 1):** Upon careful examination of Definitions 1 and 5, this criticism is factually wrong. If X ⪰Y W means X dominates W across ALL SCMs with graph G, then for any particular fixed SCM C, X still dominates W. Therefore the mGISS guarantees that for every fixed SCM, an optimal node remains within the mGISS. The "worst-case" quantification strengthens rather than weakens the guarantee, because it ensures optimality preservation pointwise for every SCM. The critic incorrectly claimed "restricting to mGISS guarantees that the bandit cannot reach the truly optimal node"—in fact, the opposite is true.

- **"Proposition 4 bridge from deterministic to probabilistic is unsubstantiated" (Harsh Critic Issue 2, substantive part):** The suggestion that the equivalence might not hold or involves hidden assumptions is speculative. The proof is in the appendix as is standard, and the result states a mathematical equivalence with full proof. The concern about presentation (no proof sketch) is kept as a minor weakness above, but the fundamental critique is not substantiated.

- **"Experiments don't validate guaranteed containment of optimal node" (Harsh Critic Issue 3, specific claim that this is a conceptual gap):** The containment of the optimal node within mGISS is a theorem (Theorem 13), not an empirical claim. The experiments serve a different purpose: demonstrating practical pruning ratios and acceleration of bandit convergence. That said, the lack of comparison to simpler baselines is kept as a major weakness.

- **"Z_X assumptions not integrated into main results" (Harsh Critic Issue 4):** The mGISS is independent of the specific Z_X choice (it holds for all valid Z_X), which makes the results stronger, not weaker. The experiments using Z_X = An(X)\{X} are valid because the theoretical guarantee covers this and all other valid choices. This is a feature, not a bug.

- **"No formal regret analysis for CondIntUCB" (Human Finder concern):** The paper's contribution is the preprocessing step (identifying the mGISS), not the bandit algorithm itself. Providing instance-dependent regret bounds for a specific bandit algorithm is outside scope and not necessary for the paper's core claims.

- **Formatting/style nitpicks:** Removed per instructions.

## Novel Insights

The equivalence between conditional-intervention superiority (a policy-based, expectation-based notion in stochastic SCMs) and deterministic atomic-intervention superiority (a pointwise, worst-case notion in deterministic SCMs) is a genuinely surprising insight. It implies that to decide which nodes are worth intervening on with conditional interventions, it suffices to check a much simpler criterion: whether, for every possible noise realization, there exists a fixed value of X that dominates all fixed values of W. This bridges a probabilistic decision problem to a combinatorial graph-theoretic one, and is what makes the Λ-structure characterization possible. The insight that Λ-structures (essentially "fork patterns" in the ancestry structure) precisely capture which nodes can influence multiple parents of Y through diverging paths is elegant and may have applications beyond causal bandits—e.g., in identifying critical control points in causal diagrams.

## Suggestions

- Add a comparison against Pa(Y) and L¹(Pa(Y)) (one-step LSCA without closure) as baselines in both the pruning ratio experiments and the regret experiments. This would quantify the marginal value of the recursive LSCA closure over simpler pruning strategies.
- Include even a brief proof sketch for Proposition 4 in the main text, highlighting why the quantifier structure works out—this would significantly improve readability and trust in the central equivalence.
- Report mGISS size statistics across all possible target nodes Y (not just the one with most ancestors) to give readers a realistic sense of expected pruning in practice.

## Score and Decision

**Calibration papers compared against:**
- IPayPEGwdE (Causal contextual bandits, scores 5,5,5,5, avg 5): Similar topic area (causal bandits), but with weaker theoretical contributions and limited practical impact.
- YcW8i9VCf5 (Adversarial CBO, scores 6,6,6,6, avg 6): Comparable in having a new problem formulation + algorithm, with solid but not groundbreaking results.
- M0xK8nPGvt (Causal graph priors in RL, scores 8,8,6,8, avg 7.5): Stronger empirical evaluation and more complete regret analysis.
- HSKaGOi7Ar (GNN expressiveness, scores 8,10,8,8, avg 8.5): A deep theoretical contribution with broad impact.

This paper has a genuinely elegant and novel theoretical contribution (the LSCA/Λ-structure characterization of mGISS + Proposition 4 equivalence + linear-time C4 algorithm). The problem formulation is original and significant. The main weakness is the experimental evaluation, which is somewhat shallow—lacking comparison to alternative pruning strategies and using only best-case target nodes. However, the core theoretical contribution is clean, rigorous, and practically significant (large pruning ratios on real graphs). This puts it above IPayPEGwdE (weaker theory) and at roughly the same level as YcW8i9VCf5, but below M0xK8nPGvt (more complete regret analysis and empirical evaluation). The paper's strengths are primarily theoretical, and the algorithm is simple but correct and useful.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>