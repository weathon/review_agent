---
job_id: c9ceb86c-cacd-42ef-bbae-9bc49d5f85e6
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: RT5SlprCmc.pdf
paper: Learning the Minimum Action Distance
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper is squarely about state representation and metric learning for reinforcement learning, which is central to ICLR.

## Minimum Quality
Pass ✅. The paper includes Abstract, Introduction, Related Work, Methodology (Sections 3–6), Experiments (Section 7), Results (figures and Table 1), and Conclusion. The work is technically sound at a high level, written in clear English, and has nontrivial experiments and theory; no fatal flaws or obvious test-set leakage are apparent.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. I do not see any hidden prompts, attempts to influence automated reviewing, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper studies how to learn the Minimum Action Distance (MAD), defined as the minimal number of decision steps needed to go from one state to another in an MDP, using only state trajectories without actions or rewards. The authors propose two self-supervised algorithms, MadDist and TDMadDist, that learn state embeddings together with an asymmetric quasimetric so that distances approximate MAD, and introduce a simple ReLU-based quasimetric that provably satisfies the triangle inequality. They evaluate on a suite of deterministic and stochastic, discrete and continuous environments with known (or approximated) MAD and show that MadDist in particular outperforms QRL and Hilbert-embedding baselines on correlation with ground-truth MAD and on a downstream planning task.

## Strengths

1. **Clear formalization of MAD and its graph interpretation.**  
   Section 4 gives a concise optimization formulation of MAD (Equation (1)) and proves uniqueness in Appendix A by connecting it to all-pairs shortest paths on the determinized transition graph. This clarifies what exactly is being approximated and distinguishes MAD from policy-dependent temporal distances such as SSP or discounted-value-based metrics.

2. **Simple but well-motivated asymmetric quasimetric.**  
   The proposed \(d_{\text{simple}}\) in Equation (3) is almost embarrassingly simple (max and mean of ReLU’d coordinate differences), yet Appendix B rigorously proves it is a quasimetric with triangle inequality via coordinatewise ReLU reductions and convex combinations. This provides an analytically transparent alternative to heavier constructions like Wide Norm or IQE, and the ablation in **Figure 5** and **Figure 6** shows that, despite its simplicity, it consistently matches or beats those more elaborate quasimetrics across correlation and CV in CliffWalking.

3. **Two complementary learning objectives that explicitly exploit trajectory structure.**  
   MadDist’s main loss \(\mathcal{L}_o\) (Equation (5)) uses trajectory index differences \(j-i\) as upper bounds and normalizes by \(j-i\), which is a thoughtful design to avoid domination by long-range pairs. The constraint loss \(\mathcal{L}_c\) (Equation (7)) penalizes violations of the MAD upper bound, extending Steccanella & Jonsson’s earlier constrained formulation to asymmetric quasimetrics with a scale-invariant objective. TDMadDist’s TD-style bootstrapping (Equation (8)) mirrors the Bellman equation \(d(s_i,s_j)=1+d(s_{i+1},s_j)\) and shows a principled attempt to propagate shorter paths via a target network (Equation (10)).

4. **Comprehensive empirical evaluation on environments with ground-truth MAD.**  
   The environment suite spans: discrete deterministic (KeyDoorGridWorld, CliffWalking), continuous point-mass mazes with physics (UMaze, Medium, Large, Giant; both D4RL and OGBench variants), and noisy/stochastic settings like NoisyGridWorld. For PointMaze, the use of Floyd–Warshall on a discretized maze graph to approximate MAD is a reasonable compromise given continuous dynamics. **Figure 3** and **Figures 11–12** in the appendix give extensive curves of Pearson/Spearman correlations and CV across training steps and across many settings, demonstrating that MadDist achieves very high correlations (often ~0.95–1.0) with stable, low CV.

5. **Strong empirical performance versus state-of-the-art quasimetrics and symmetric MAD proxies.**  
   Across multiple figures, MadDist dominates QRL and Hilbert in correlation metrics and ratio CV. In **Figure 3**, on KeyDoorGridWorld and CliffWalking, MadDist reaches near-perfect Pearson correlation quickly and maintains the lowest CV, indicating both accurate ordering and consistent scaling of distances, especially in asymmetric dynamics where Hilbert’s symmetric metric is structurally mismatched. TDMadDist, while weaker than MadDist, still clearly outperforms Hilbert.  

   On the downstream OGBench planning benchmark, **Table 1** shows MadDist achieving success rates of 0.99–1.00 across all Medium/Large/Giant Navigate and Stitch tasks, often with zero variance, whereas QRL and TDMadDist trail and Hilbert frequently collapses (e.g., 0.05–0.22 success on several settings). This is strong evidence that the learned distances are not just correlated with MAD but are also practically useful for long-horizon planning under MPC with random shooting.

6. **Careful ablation studies dissecting representation and model design.**  
   Appendix E provides multiple targeted ablations:
   - **Figure 4**: effect of latent dimensionality on accuracy, showing performance saturates around dimension 10 and remains stable for larger sizes, which indicates robustness and absence of obvious overfitting to high-dim latents.
   - **Figure 7**: effect of dataset size, where both MadDist and TDMadDist improve gracefully with more trajectories but still achieve decent correlation with relatively modest data.  
   - **Figure 8** and **Figure 9**: sensitivity of QRL and Hilbert baselines to network capacity, showing that the authors did not handicap competing methods with tiny networks and that changing architecture size yields only modest performance changes.

7. **Visualization that qualitatively reflects learned geometry.**  
   **Figure 10** (MediumMaze heatmap) visualizes the learned distance from a fixed goal state; the contour of distances aligns with the maze corridors and obstacles, which is exactly the qualitative structure expected of MAD. This provides an intuitive sanity check that the representation encodes directed reachability rather than just Euclidean proximity in observation space.

8. **Clarity and organization.**  
   The paper is overall well written and logically structured. The connection between the intractable constrained optimization of Section 4 and the scalable surrogate objectives in Appendix C is explained clearly. Notation is mostly consistent, and key choices (e.g., behavior policy being random, number of trajectories, batch sizes) are specified in Appendix D and in the main text.

## Weaknesses

1. **Novelty is moderate; conceptual contributions are incremental relative to prior MAD-learning work.**  
   The high-level idea of learning embeddings where distances approximate MAD from trajectories without actions or rewards has already been explored, particularly in Steccanella & Jonsson (2022), which this work explicitly builds upon, and in more recent metric RL work (e.g., QRL; Park et al., 2024b). The main new ingredients are:
   - allowing asymmetric distances via quasimetrics;
   - a simple ReLU-based quasimetric \(d_{\text{simple}}\) instead of Wide Norm or IQE;
   - a scale-invariant loss and an optional TD-style variant (TDMadDist).  
   These are useful refinements, but they sit relatively close to existing formulations conceptually. There is no new theory guaranteeing consistency of the learned estimator with \(d_{\text{MAD}}\), and the learning objectives are heuristic surrogates of the linear program rather than being derived from tight bounds. For an ICLR main-track paper, I would expect either a more substantial theoretical advance (e.g., identifiability or sample-complexity guarantees for MAD estimation under some data-collection policy) or a more radical algorithmic departure.

2. **Some mathematical details and objectives are underspecified or partially broken in the main text.**  
   - Equation (9) (definition of \(\mathcal{L}_r^\prime\) for TDMadDist) is truncated in the main body:  
     \[
     \mathcal{L}_{r}^{\prime}=\mathbb{E}_{\tau\sim\mathcal{D},(s_{i},s_{j})\sim\tau,s_{r}\sim\mathcal{S}_{\mathcal{D}}}\left[(d_{\theta}(s_{i},s_{i+1}\right. 
     \]
     The expression is incomplete and does not show the intended ratio or squared loss. Only in Appendix C (Equation in Section C.4) is the form clarified as  
     \[
     \mathcal{L}_{r}^{\prime}=\mathbb{E}\left[\left(\frac{d_{\phi}(s_i,s_r)}{1+d_{\phi'}(s_{i+1},s_r)}-1\right)^2\right],
     \]
     but this mismatch between main text and appendix is confusing and arguably a technical error in the core algorithm description. For an algorithmically-focused paper, this should be fixed in the main body, not left ambiguous.
   - For MadDist’s contrastive loss \(\mathcal{L}_r\) (Equation (6)), the distribution \((s,s')\sim\mathcal{S}_{\mathcal{D}}\) is only described informally as “state pairs randomly sampled from all trajectories”. It is unclear whether these are sampled independently from the marginal state distribution, from pairs on the same trajectory but arbitrary time difference, or something else. This matters because the induced regularization on \(d_\theta\) depends heavily on this sampling scheme, especially in large continuous spaces.

3. **Lack of theory on when the surrogates recover MAD.**  
   While Theorem 1 in Appendix A establishes uniqueness of the LP solution for MAD in the fully-known graph, there is no analysis of the learning objectives in Sections 6 and Appendix C. For example:
   - Under what conditions on coverage and behavior policy does minimizing \(\mathcal{L}_o+\mathcal{L}_c\) imply convergence to \(d_{\text{MAD}}\) (up to scale) on \(\mathcal{S}_\mathcal{D}\)?
   - How does the choice of \(H_c\) in \(\mathcal{L}_c\) affect the bias–variance tradeoff? Does enforcing upper bounds only for short horizons preserve the shortest-path structure globally via the quasimetric triangle inequality, or can pathological local minima exist?  
   Some of these questions are alluded to qualitatively, but no formal argument or even a clear discussion of possible failure modes is given. As a result, the method remains an empirically validated heuristic without a clear sense of its theoretical limitations.

4. **Limited baseline coverage: missing several closely related representation-learning methods.**  
   The experiments compare only to QRL (Wang et al., 2023b) and Hilbert representations (Park et al., 2024b). Notably absent are:
   - The constrained MAD-learning method of Steccanella & Jonsson (2022), which is arguably the most direct predecessor; it is described textually in Section 4 but not actually instantiated as a baseline with the new environments and metrics. Directly reproducing and comparing their symmetric LP-style embedding to MadDist and TDMadDist on Figure 3 and **Figure 11–12** would strongly clarify how much the asymmetric quasimetric and scaled loss help.
   - Bisimulation-inspired or behavioral-distance methods like MICo (Castro et al., 2021), Dadashi et al. (2021), or Myers et al. (2024), some of which aim to learn policy-agnostic state distances from trajectories. Even if their notion of distance differs (e.g., discounted visitation rather than MAD), including at least one strong representative would help position the empirical quality of MAD vs. those alternatives.  
   Given that the paper’s main claim is that its method “significantly outperforms existing state representation methods in terms of representation quality,” Table 1 and Figure 3 support this only against two baselines drawn from very similar MAD-focused or goal-conditioned pipelines.

5. **Downstream evaluation is narrow and lacks comparisons to ground-truth MAD / SSP-based planning.**  
   The only downstream task is random-shooting MPC for goal-reaching in OGBench PointMaze (Appendix H, **Table 1**). While this is a strong and relevant testbed, the setup has several limitations:
   - It uses the true simulator with random candidate sequences; planning performance is therefore dominated by the quality of the heuristic distance, but there is no comparison against using true MAD (or closely approximated graph-based MAD) as the heuristic, or against SSP distances, to quantify the residual performance gap.
   - No comparison is given to using QRL’s own planning schemes or Hilbert’s latent-policy methods (e.g., as in Park et al., 2024b), which may be more optimized for those embeddings than the simple MPC used here.  
   As a result, while **Table 1** shows a striking gap in success rates, it does not fully establish that the proposed MAD estimator is close to optimal in terms of value for planning, only that it is better than these particular baselines under this specific planner.

6. **TDMadDist underperforms without sufficient analysis.**  
   In **Figure 3** and **Figures 11–12**, TDMadDist is consistently weaker than MadDist and often comparable or slightly worse than QRL in terms of Pearson/Spearman correlation and CV, especially in some OGBench Stitch tasks. The discussion on Page 8 briefly notes that “TDMadDist underperforms,” but there is no deeper investigation:
   - Is the bootstrapped target \(1+d_{\theta'}(s_{i+1},s_j)\) introducing positive bias that makes distances too large, conflicting with \(\mathcal{L}_c\)?
   - How sensitive is TDMadDist to the Polyak averaging factor \(\beta\) in Equation (10) or to rollout horizon?  
   Some ablations or failure analysis would be valuable to understand whether the TD variant is fundamentally problematic for MAD, or just needs better hyperparameter tuning or architectural tweaks.

7. **Ambiguities and potential inconsistencies in experimental reporting.**  
   - The main text on Page 8 states “All reported results are means over five independent runs (random seeds),” but **Figure 3** and several appendix figures state that shaded regions show ranges across “three random seeds”. This discrepancy should be resolved or explained.
   - For continuous PointMaze settings, MAD is approximated by discretizing the maze and running Floyd–Warshall. The paper does not quantify or discuss the error induced by this approximation, especially in the presence of nontrivial dynamics (e.g., velocities). It is unclear whether the MAD on the discretized graph truly reflects minimal *decision* steps in the underlying physics-based MDP or just grid-graph shortest paths.

8. **Limited discussion of behavior policy dependence and trajectory quality.**  
   All datasets are collected by random policies (except OGBench navigate/stitch, where behavior is more structured) and the method assumes relatively good coverage of the transition graph. There is no analysis of how performance degrades when coverage is partial, when trajectories are biased (e.g., strongly goal-reaching), or when there are large unreachable regions in \(\mathcal{S}_\mathcal{D}\). Given that real offline RL datasets are often highly biased, a discussion or targeted experiment on policy dependence would be valuable.

9. **Some smaller clarity issues.**  
   - In Equation (3), \(d\) is used both as the dimension and previously as a generic distance symbol; clarifying notation would help avoid confusion.  
   - In the MadDist and TDMadDist descriptions, the role of hyperparameters like \(d_{\max}\), \(H_c\), and \(w_r\) is mostly deferred to Appendix D; a brief intuition in the main text (e.g., why \(H_c=6\)) would aid understanding.  
   - The phrase “latent positive homogeneity” is used (Page 5) but not defined in the main text; readers unfamiliar with Wang & Isola (2022) may not know what property is important here.

## Potentially Missing Related Work

1. **Castro et al., “MICo: Learning improved representations via sampling-based state similarity for Markov decision processes”, 2021.**  
   - Relevance: MICo introduces a behavioural state similarity metric for MDPs learned from trajectories, aiming for policy-agnostic state embeddings that respect the dynamics, which is conceptually very close to learning MAD-like structure without rewards.  
   - Relation to this paper: Both works learn distances or similarities between states purely from trajectories; MICo focuses on sampling-based similarity, while this paper focuses on MAD as a shortest-path notion. A comparison in the Related Work section (Section 2) is warranted, and including MICo (or a similar bisimulation-based method) as an empirical baseline in at least one discrete environment (e.g., CliffWalking or KeyDoorGridWorld) would strengthen claims about representation quality.

If space is too constrained for an empirical baseline, the paper should at least discuss how MICo’s learned behavioral distance compares conceptually to MAD, and why MAD might be preferable for planning or transfer scenarios.

## Questions

1. **Clarification of TDMadDist objective (Equation (9)).**  
   Could the authors provide the complete expression for \(\mathcal{L}_r^\prime\) in the main text and clarify whether it is exactly the ratio-based squared loss in Appendix C, or if there are implementation differences? Since this is central to TDMadDist’s learning signal, having the correct formula in Section 6.2 is important.

2. **Conditions for (approximate) consistency with MAD.**  
   Are there assumptions under which the authors believe MadDist is guaranteed (or at least likely) to converge to a scaled version of \(d_{\text{MAD}}\) on \(\mathcal{S}_\mathcal{D}\)? For example, if trajectories uniformly cover all edges of the determinized graph and \(H_c\) is large enough to include all path lengths up to some diameter, can we reason about uniqueness or identifiability of \(d_\theta\) given the quasimetric constraints?

3. **Effect of behavior policy and data distribution.**  
   How does performance change if trajectories are generated by a strongly goal-directed agent (e.g., always heading to a particular goal) instead of a random policy? In such a dataset, long-range pairs not along typical goal-directed paths might be rare. Do the authors expect MadDist’s \(\mathcal{L}_o\) and \(\mathcal{L}_c\) to generalize well to these unseen pairs, or would the learned metric overfit typical trajectories?

4. **Discretization error in continuous environments.**  
   For PointMaze and OGBench, how fine is the grid used to compute “ground-truth” MAD, and have the authors evaluated how often the shortest-path distance on this grid deviates from the minimal decision steps in the true continuous dynamics (e.g., due to dynamics constraints or velocity limits)? A short study or bound on this discrepancy would help interpret the very high correlations in **Figure 11–12**.

5. **Why does TDMadDist underperform MadDist?**  
   Do the authors have a hypothesis for why the TD variant consistently lags behind MadDist? Did they experiment with different target network smoothing factors, shorter bootstrapping horizons, or alternative TD targets such as multi-step returns? Additional insight here could help others decide whether TD-style bootstrapping is appropriate for MAD-type metrics.

6. **Potential improvements or variants of the simple quasimetric.**  
   \(d_{\text{simple}}\) uses a convex combination of \(\max\) and mean reductions of ReLU differences. Did the authors try other aggregations (e.g., \(L_p\) norms of ReLU(x−y) for various \(p\)) or learned weights \(\alpha\)? If so, how did they compare in the ablations similar to **Figure 5** and **Figure 6**?

Addressing these points, even partially, would increase my confidence in both the algorithmic soundness and the practical robustness of the approach.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core formulations are mathematically consistent, the quasimetric properties are proven, and experiments are extensive and carefully implemented. However, the lack of theoretical guarantees for the learning objectives, the truncation/mis-specification of Equation (9), and the unsystematic treatment of discretization error prevent an “excellent” rating.

## Presentation Rating

3: good.  
The paper is generally clear, well structured, and supported by informative figures such as **Figure 1**, **Figure 3**, **Figure 4–7**, and **Figure 10**, plus **Table 1** and hyperparameter tables. A few notational ambiguities, the broken equation, and minor inconsistencies (e.g., number of seeds) keep it from being excellent.

## Contribution Rating

3: good.  
The paper makes a meaningful contribution by bringing asymmetric quasimetrics and a simple, well-analyzed distance function into MAD-learning, alongside a thorough empirical benchmark suite with ground-truth MAD. The conceptual novelty is moderate but the empirical improvements and careful evaluation make this a useful addition to the literature.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The work is technically solid, empirically convincing, and clearly relevant to representation learning and RL. Its main limitation is that the conceptual step beyond prior MAD-learning work is relatively modest and theoretical backing for the learning objectives is thin. On balance, I lean slightly toward acceptance because (i) the simple quasimetric plus MadDist objective clearly outperform strong recent baselines across a challenging environment suite, and (ii) the provided benchmarks and analysis are likely to be practically valuable to the community. Strengthening theoretical discussion, fixing the TDMadDist description, and expanding baselines would make this a clear accept.

## Reviewer Confidence

4: confident.  
I am familiar with MAD, goal-conditioned RL, and quasimetric learning, and I carefully read the math and experimental setup. There could be related work I missed in adjacent subfields, but my overall assessment is unlikely to change dramatically.