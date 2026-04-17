---
job_id: 6a8c2706-1da0-4d6b-8108-07ffdccde2bc
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ngOOlatCK6.pdf
paper: The Minimal Search Space for Conditional Causal Bandits
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is on causal reasoning and bandit algorithms (conditional causal bandits, minimal intervention search sets, graph algorithms), which fits squarely within ICLR’s core topics of causal reasoning, learning theory, and bandit/online learning.

## Minimum Quality
Pass ✅.  
The paper is in English and has all major components: Abstract, Introduction, Preliminaries/Methodology (Sections 2–5), Experiments (Section 6 + Appendix H), Results/Discussion (Section 6 & 7), and Conclusion (Section 7). The theoretical development is careful, proofs are provided in the appendix, and experiments are nontrivial. No obvious fatal methodological or identifiability errors appear.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I found no hidden prompts, instructions to reviewers, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper studies a new variant of causal bandits where arms are *single-node conditional interventions* of the form \(do(X = g(\mathbf Z_X))\), with \(\mathbf Z_X\) an observable context determined by the DAG and application. The central technical contribution is a purely graphical characterization of the *minimal globally interventionally superior set* (mGISS) of nodes that is guaranteed to contain the optimal node to intervene on, and a linear-time algorithm (C4) that computes this set via an LSCA-closure / \(\Lambda\)-structure characterization. Experiments on random and real-world DAGs and a simple UCB-based conditional bandit show substantial pruning of the intervention search space and faster regret convergence when restricting node choices to the mGISS.

## Strengths

1. **Conceptual clarity of the core problem.**  
   The paper isolates a clean and well-motivated problem: given a known causal DAG and a reward variable \(Y\), identify *which node(s)* need to be considered as possible intervention targets when arms are conditional policies \(do(X=g(\mathbf Z_X))\). This separates node selection (graph-theoretic) from policy learning (bandit), and is clearly articulated in Sections 1–3.

2. **Nontrivial theoretical contribution on conditional interventions.**  
   The definition of conditional-intervention superiority \(\succeq_Y^c\) (Definition 1) and its equivalence to deterministic atomic superiority \(\succeq_Y^{\det,a}\) (Definition 2, Proposition 4, Appendix D) is technically interesting. It allows the analysis of conditional interventions in general SCMs to be reduced to atomic interventions in deterministic SCMs, which is mathematically much simpler yet nonobvious. The proof line using unrolled assignments and blocked unrolled assignments (Definitions 18–19, Lemmas 21, 23, 22, 24–25) is detailed and appears self-consistent.

3. **Crisp graphical characterization of the mGISS.**  
   The main result, Theorem 13, shows that the minimal globally interventionally superior set \(\mathrm{mGISS}_Y(G)\) is exactly the LSCA-closure \(\mathcal L^\infty(\mathrm{Pa}(Y))\) of the parents of \(Y\). The paper provides two complementary characterizations:
   - LSCA-closure via recursive lowest strict common ancestors (Definitions 7–9), and
   - An equivalent description via \(\Lambda\)-structures (Definition 11, Theorem 12), which is actually simpler to reason about.
   This dual characterization is a nice piece of graph theory tailored to causal bandits.

4. **Efficient algorithm with clear correctness argument.**  
   The C4 algorithm (Algorithm 1, Section 5) computes \(\mathcal L^\infty(\mathbf U)\) in \(O(|\mathbf V|+|E|)\) time using the notion of *connectors* (Definition 14). Lemma 15 precisely states that \(\mathfrak c[V]\) is the first mGISS node encountered along any path from \(V\) to the closure, and Theorem 16 gives a clear proof of correctness and time complexity. The algorithm is straightforward to implement: a reverse topological pass computing connectors of children and promoting a node into the closure whenever its children have multiple distinct connectors.

5. **Intuitive and well-illustrated theory.**  
   Figure 1 (a–d) is used effectively to build intuition for why parents of \(Y\) are not enough and why *lowest* strict common ancestors (and then their recursive closure) are required. For example:
   - In Figure 1c, with a single parent \(A\), directly intervening on \(A\) is always sufficient.
   - In Figure 1a and 1b, intervening on LCAs like \(X_1\) and \(Z\) can produce better joint configurations of the parents than any single-parent intervention.
   - Figure 1d nicely highlights why naive LCA-based heuristics fail, motivating the stricter LSCA notion.  
   Figure 2a further clarifies the \(\Lambda\)-structure concept that underlies the closure characterization.

6. **Empirical evidence for search-space pruning and regret benefits.**  
   The experimental section, while not extensive, is focused and aligned with the claimed contribution:
   - Figure 5 shows that on random DAGs, the mGISS is a small fraction of \(\mathrm{An}(Y)\) when graphs are sparse and/or large, with reductions down to ~17% for 500-node graphs with expected degree 2.
   - Figure 6 shows that on real bnlearn and railway graphs, the mGISS often retains less than ~10–30% of ancestors for larger models, confirming practical utility.  
   Crucially, Figure 3 shows cumulative regret curves for the CondIntUCB algorithm, with and without mGISS pruning, on asia, Sachs, child, and pathfinder. In all four panels, the mGISS-based node selection (blue) converges faster and to lower regret than brute-force node search (red), sometimes dramatically (e.g., pathfinder).

7. **Relevance to and clear distinction from prior causal bandit work.**  
   The paper positions itself against Lee & Bareinboim (2018, 2019, 2020) and the classical causal bandit literature (Lattimore et al., 2016; Lu et al., 2020; etc.), emphasizing two key differences: (i) conditional (policy-based) interventions instead of purely atomic or soft interventions, and (ii) single-node interventions rather than multi-node sets. The contrast with Lee & Bareinboim in terms of the structural reduction problem (where to intervene, not how to learn) is persuasive.

8. **Soundness and internal consistency of the math.**  
   The graph-theoretic definitions (LSCA, LSCA-closure, \(\Lambda\)-structure) are precise, and the accompanying lemmas in Appendix F–G about uniqueness of mGISS (Proposition 6), closure monotonicity (Lemma 35), and LSCA–\(\Lambda\) equivalence (Theorem 12) are worked out in detail. I did not find obvious logical gaps in the main proofs; the use of explicit path manipulations, ancestor partial order, and induction on (reverse) topological orders is systematic.

## Weaknesses

1. **Strong structural assumptions: known DAG, no latent confounders, single-node interventions only.**  
   The entire theory assumes a fully known causal DAG and no unobserved confounding (Section 2). While the authors acknowledge this (e.g., end of Section 1 and start of Section 2), the practical impact is significant: in many realistic decision problems, we only have a partially identified graph, CPDAG, or equivalence class, and there may be hidden variables. The suggestion in Section 1 to run C4 on each candidate graph and take the union of mGISS sets is reasonable but could easily blow up the search space again (especially in large MECs) and is not analyzed. Similarly, restricting to *single-node* interventions excludes many settings where coordinated multi-node interventions are necessary, and the paper does not provide guidance on how results might extend or fail in that case. This is a conceptual limitation that substantially narrows immediate applicability.

2. **Bandit side is underdeveloped; no regret analysis or comparison to causal bandit baselines.**  
   The conditional bandit algorithm (CondIntUCB) in Section 6 is intentionally simple, but the absence of any theoretical regret guarantees is noticeable: there is no analysis of how pruning to the mGISS changes asymptotic or finite-time regret bounds relative to standard contextual bandits or causal bandits. Moreover, there is no comparison to existing causal bandit algorithms, not even to graph-based pruning approaches like Lee & Bareinboim (2018) in the hard-intervention setting or to algorithms from Lu et al. (2020) / Yabe et al. (2018). As a result, while Figure 3 empirically shows lower regret when pruning, the reader has no sense of whether the absolute regret levels are competitive or whether other causal bandit methods could benefit similarly or more from structure.

3. **Conditional interventions and context explosion are handled somewhat superficially in experiments.**  
   The conditional intervention setting involves one UCB per context configuration of \(\mathbf Z_X\) (as in Lattimore & Szepesvári §18.1), which can be enormous when \(\mathbf Z_X\) includes all ancestors. For example, on the pathfinder model (Figure 3, bottom-right), the authors explicitly note in Appendix H that the experiment barely fits into 350 GB RAM due to the large number of contexts. This starkly illustrates that the proposed assumptions on \(\mathbf Z_X\) (inclusion of all ancestors; monotonicity \(W \in \operatorname{An}(X)\Rightarrow \mathbf Z_W \subseteq \mathbf Z_X\) in Section 2) can make the *per-node* policy search itself intractable. The paper does not explore or analyze any structural conditions under which \(\mathbf Z_X\) can be reduced without harming optimality, nor does it evaluate performance under more realistic, smaller contexts. This gaps matters because one of the headline motivations is *practical* acceleration, yet computational bottlenecks are simply shifted from node search to context explosion.

4. **Empirical evaluation of search-space pruning is purely in terms of node counts, not overall runtime or sample complexity.**  
   Figures 5 and 6 report fractions of nodes retained in mGISS relative to all ancestors of \(Y\), which clearly show pruning. However, there is no measurement of:
   - end-to-end runtime of CondIntUCB with vs. without C4,
   - number of interaction rounds needed to reach a given regret threshold, or
   - any correlation between the fraction pruned and regret improvement.  
   For example, in Figure 3, the regret gap looks modest for asia and Sachs but substantial for pathfinder; relating this back to Figure 6 (fractions and node counts) would make the empirical story stronger. As it stands, the experimental section demonstrates that node counts are reduced and regret is improved, but not how this balances against increased per-node contextual complexity or the overall computational cost.

5. **Some key technical definitions are heavy and could be simplified or made more transparent.**  
   The unrolled assignment and blocked unrolled assignment machinery (Definitions 18–19) is central for connecting conditional and atomic interventions and for several proofs (e.g., Lemmas 21–23, 22, 24). However:
   - The notation in Equations (6)–(7) is somewhat opaque: e.g., in Definition 18, the distinction between exogenous \(N_i\) and endogenous \(V_i\) is clear, but readers may struggle with how these tie into the causal graph \(G^\mathfrak C\) vs the augmented graph \(G^*\) used later.
   - Remark 20 acknowledges that \(\bar f_X\) *does not* depend on all noise variables and that the notation is thus slightly misleading, but the paper opts to keep the overloaded notation anyway. In lengthy proofs (Appendix D, F), this creates cognitive overhead and chances for index confusion.  
   While not logically wrong, the exposition is harder to parse than necessary, which somewhat limits accessibility.

6. **Clarity and completeness around LSCA and \(\Lambda\)-structures.**  
   Although the high-level intuition is clear, some technical details around LSCA and \(\Lambda\)-structures could use more explicit discussion in the main text:
   - Definition 7 introduces *strict* common ancestors with path constraints excluding the other node, but it is easy to misread; a small counterexample showing how an ordinary LCA can fail to be an LSCA would be helpful.
   - The proof of Theorem 12 (Appendix F.2) is quite intricate, relying on path intersections and minimality under a topological order; most readers of ICLR will not check this carefully. Given that Theorem 12 is used both for intuition (Figure 2a) and for algorithmic correctness via Lemma 42 and Lemma 15, a short proof sketch in the main text would significantly help build trust.
   - Definition 14 (connector) in Section 5 is dense; connecting it visually to Figure 2b more explicitly (e.g., walking through the connector labels for a few nodes step-by-step) would improve readability.

7. **Experimental setup for CondIntUCB is underspecified and potentially fragile.**  
   While Section 6 gives a high-level description of CondIntUCB, several important details are missing:
   - How exactly are reward ranges normalized across interventions? Are the conditional distributions of \(Y\) bounded or sub-Gaussian? This matters for UCB parameters and comparability of regret curves.
   - For each dataset in Figure 3, what are \(|R_X|\), sizes of \(\mathbf Z_X\), and the induced number of contexts per node? This is crucial to interpret why pathfinder needs so much memory and how the computational burden scales.
   - The definition of “cumulative regret with respect to node choice” is a bit fuzzy: the text says regret is with respect to the node, but in a conditional bandit, both node and policy matter. It remains unclear whether the “best node” is defined assuming the *optimal* policy for that node, and whether CondIntUCB has any chance of approaching that optimum in finite time given the context explosion.  

   These omissions do not invalidate the qualitative trends in Figure 3, but they reduce the diagnostic value of the experiments.

8. **Missing or thin discussion of several closely related causal bandit works.**  
   While the paper cites a good portion of the causal bandit literature, the related-work section does not discuss some directly related work that would help situate this contribution more fully, particularly regarding design and transfer:
   - **Kocaoglu et al., “Experimental Design for Causal Bandits”, 2017.** This paper addresses how to optimally design interventions in causal graphs for bandit settings. It is highly relevant to the idea of using structure to reduce or prioritize intervention sets, and should be discussed alongside Lee & Bareinboim in Section 7.
   - **Zhang & Bareinboim, “Transfer Learning in Multi-Armed Bandits: A Causal Approach”, 2017.** This work uses causal structure to transfer information across bandit tasks. Since the current paper effectively treats each node as a contextual bandit subproblem, there are conceptual parallels in using causal knowledge to tie together decision problems over different contexts/nodes; a short comparison in Section 7 would be appropriate.
   - **Bareinboim & Pearl, “Causal Inference and the Data-Fusion Problem”, 2016.** While more general, this work underpins several causal identification strategies including conditional interventions and policy evaluation in heterogeneous data environments. Given that this paper’s conditional interventions rely on the Pearl/Shpitser conditional interventional calculus, citing this data-fusion perspective would strengthen the causal inference grounding.

   The omission of these references does not undermine the correctness of the results, but it makes the contextualization of the contribution less complete than one would expect for ICLR.

9. **Scope of “minimality” is purely worst-case; no discussion of problem-dependent tighter sets.**  
   The mGISS is defined via worst-case intervention superiority across *all* SCMs compatible with the DAG (Definition 1, 2, 5). This is a clean, robust notion, but can be conservative: for a given application where we know additional functional or distributional constraints (e.g., monotonicity, additive noise, parametric parametrizations), a significantly smaller set might suffice. The paper currently does not discuss whether or how such model-class-dependent refinements could be obtained, nor whether the LSCA-closure is tight or loose in practice relative to a realistic SCM family. This is not a flaw in what is proven, but it limits the theoretical story to an adversarial “all SCMs” setting.

## Potentially Missing Related Work

1. **Kocaoglu, M., Shanmugam, K., Bareinboim, E. (2017). “Experimental Design for Causal Bandits.”**  
   - **Relation:** Directly tackles optimal design of experiments / interventions in causal bandit settings using causal graphs. Highly relevant to “where to intervene” problems.  
   - **Where to add:** It should be discussed in Section 7 alongside Lee & Bareinboim (2018; 2019; 2020) as another line of work leveraging causal structure to reduce or prioritize interventions, and ideally mentioned in the Introduction when motivating structure-based search space reduction.

2. **Zhang, J., Bareinboim, E. (2017). “Transfer Learning in Multi-Armed Bandits: A Causal Approach.”**  
   - **Relation:** Uses causal models to enable transfer across bandit tasks, which conceptually resonates with the idea of treating each node as a contextual bandit subproblem connected by causal structure.  
   - **Where to add:** Section 7, in the paragraph discussing how conditional causal bandits relate to other causal bandit settings; a brief comparison highlighting that this paper focuses on reducing the node set within a single causal graph, rather than transferring across tasks, would clarify distinctions.

3. **Bareinboim, E., Pearl, J. (2016). “Causal Inference and the Data-Fusion Problem.”**  
   - **Relation:** Provides general results on combining data from different sources for causal inference, including implications for conditional interventions and policy evaluation. While more general than causal bandits, it underlies some of the causal identification techniques that make conditional policies meaningful.  
   - **Where to add:** A short citation in Section 2 when introducing conditional interventions and SCM-based interventional reasoning, and possibly in Section 7 when discussing the causal inference basis of conditional causal bandits.

## Questions

1. **Context specification and scalability.**  
   You assume \(\operatorname{An}(X)\setminus\{X\} \subseteq \mathbf Z_X \subseteq \mathbf V \setminus \operatorname{De}(X)\) and that \(\mathbf Z_W \subseteq \mathbf Z_X\) whenever \(W \in \operatorname{An}(X)\). In the pathfinder experiment, this leads to a massive number of contexts and extreme memory use.  
   - Can you clarify exactly how \(\mathbf Z_X\) was chosen for each dataset and node in Figure 3?  
   - Is the inclusion of *all* ancestors formally required for your mGISS result, or is it only a convenient assumption? If not required, can you characterize a smaller class of conditioning sets that still preserves Theorem 13?

2. **Regret guarantees with mGISS pruning.**  
   Have you considered analyzing the regret of CondIntUCB (or any standard contextual bandit algorithm) when the node set is restricted to the mGISS, compared to running it on all nodes? In particular:
   - Is there a simple argument that the optimal node with optimal policy is always contained in the pruned set under your assumptions? (This seems implicit, but spelling it out as a corollary of Theorem 13 + Proposition 4 for stochastic SCMs would help.)  
   - Can you bound the regret gap between using the true best node vs the best node in an arbitrary superset of the mGISS?

3. **Robustness to imperfect graphs.**  
   In realistic applications, we rarely know the exact DAG. Suppose we only have an equivalence class (CPDAG) or a partially learned graph with uncertain edges.  
   - Do you see a principled way to adapt C4 to work with a set of plausible DAGs, beyond the naive union-of-mGISS suggestion?  
   - Is there any sense in which certain nodes appear in the mGISS for *all* DAGs in the equivalence class, and thus form a “robust mGISS”?

4. **Examples where LSCA-closure is significantly larger than necessary.**  
   Can you construct a concrete SCM (not just a DAG) where \(\mathcal L^\infty(\mathrm{Pa}(Y))\) is much larger than the true minimal node set for that particular model class? It would be informative to see in which kinds of graphs and functional forms the worst-case nature of the mGISS incurs large conservatism.

5. **Comparison with hard-intervention reductions.**  
   Since you prove equivalence between conditional and deterministic atomic superiority (Proposition 4), one might expect some relationship between your mGISS and the sets obtained by Lee & Bareinboim (2018) for hard interventions.  
   - Is there any simple inclusion/exclusion relationship between \(\mathcal L^\infty(\mathrm{Pa}(Y))\) and their intervention-target sets in DAGs without latent confounders?  
   - Could C4 be used as a subroutine to speed up or refine their structural bandit pruning in the single-node hard-intervention case?

Addressing these could strengthen both the theoretical story and the empirical relevance; some of them may be suitable for rebuttal clarification, others as pointers for expanded discussion.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

4: excellent.  
The main theoretical results (equivalence of superiority relations, LSCA-closure characterization, uniqueness and minimality of mGISS, correctness and linear complexity of C4) are carefully stated and supported by detailed proofs; the graph-theoretic reasoning appears consistent. The experiments are limited but methodologically sensible.

## Presentation Rating

3: good.  
The paper is generally well written and well structured, with helpful figures (1–4, 5–6) and clear motivation. Some parts of the notation (unrolled assignments, blocked assignments, LSCA/\(\Lambda\)-structures) are heavier than necessary and could be streamlined, but they do not block understanding for a technically inclined reader.

## Contribution Rating

3: good.  
The paper offers a nontrivial and useful theoretical contribution: a minimal, graph-theoretically characterized intervention target set for conditional causal bandits, plus a practical linear-time algorithm and some empirical validation. The scope is somewhat limited by structural assumptions and by the absence of regret theory or stronger empirical baselines, but the contribution is clearly above incremental.

## Overall Rating

8: Accept, good paper (poster).  
The work makes a solid, technically sound contribution to the theory of causal bandits with conditional interventions, providing a clean minimal search-space characterization and an efficient algorithm that show meaningful empirical benefits. Despite some limitations in assumptions, experimental depth, and bandit-theoretic analysis, the strengths in conceptual formulation and graph-theoretic development justify acceptance.

## Reviewer Confidence

4: confident.  
I am familiar with causal graphs, SCM-based reasoning, and bandit algorithms, and I read the main proofs (and core appendices) with some care. While I did not line-by-line verify every lemma, the arguments are coherent and fit known principles.