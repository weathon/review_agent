---
job_id: 6ec1e3c6-df63-4f03-87f1-e30846c640a8
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: zqA7Q9Q21L.pdf
paper: R2PS: Worst-Case Robust Real-Time Pursuit Strategies Under Partial Observability
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely on reinforcement learning, game-theoretic planning, and learning on graphs for pursuit–evasion, which is fully within ICLR scope.

## Minimum Quality
Pass ✅.  
The paper has all required sections (abstract, introduction, related work in Appendix E, methodology, experiments, results, conclusion). Methods and proofs are technically detailed; experiments are substantial and use multiple real-world graphs. I do not see fatal theoretical or experimental flaws that would justify a desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts at manipulating automated reviewers or hidden instructions in the main content.

---

# Expected Review Outcome:

## Summary

The paper studies graph-based pursuit–evasion games where pursuers have limited observability while the evader has full information and can react asynchronously to pursuer moves. Building on a dynamic programming (DP) algorithm for Markov PEGs, the authors prove that the same distance table yields strictly optimal strategies for both players even when the evader moves after observing the pursuers’ action. They then introduce a belief-preservation mechanism to extend these DP strategies to partial observability and embed this into a cross-graph RL framework (EPG-style) to train a GNN-based pursuer policy, achieving real-time inference and strong worst-case robustness on unseen real-world graphs, outperforming PSRO baselines.


## Strengths

1. **Clear game-theoretic extension of DP to asynchronous moves.**  
   Section 3.1 and **Theorem 2** rigorously connect the distance table \(D\) from **Algorithm 1** to strictly optimal strategies under asynchronous moves. In particular, the definition of \(\nu^*(s_p,s_e,n_p)\) in **Equation (3)**, combined with **Lemma 1**, shows that DP’s minimax distances in fact certify the optimal worst-case capture time even when the evader observes the pursuer’s choice. This is a nontrivial strengthening of previous Markov PEG results from Lu et al. (2025a).

2. **Belief-based extension to partial observability that integrates cleanly with DP.**  
   The construction in Section 3.2, especially **Equations (4)–(7)**, provides an intuitively appealing and computationally simple way to track possible evader positions and a belief distribution over them. The fact that **Lemma 2** proves \(\mu(s_p,\text{Pos})\) and \(\mu(s_p,\text{belief})\) reduce to the DP-optimal \(\mu^*(s_p,s_e)\) when \(\mathrm{Pos}\) is a singleton gives a nice consistency guarantee with the perfect-information setting.

3. **Coherent integration with cross-graph RL and real-time inference.**  
   Section 4 integrates belief preservation with an EPG-style cross-graph RL pipeline. **Figure 1** is quite helpful here: it clearly shows how each training graph \(G_i\) and its DP oracle \((\mu_i^*,\nu_i^*)\) feed into the RL loop, with \(\mu^*\) providing KL guidance in **Equation (8)** and \(\nu^*\) providing strong adversaries. This makes the overall scheme conceptually tight: DP provides robust worst-case dynamics, RL distills them into a fast GNN policy.

4. **Strong empirical evidence that the RL policy is both robust and generalizes across graphs.**  
   - **Table 2** is central: it shows that the R2PS pursuer consistently beats PSRO trained directly on the test graphs, especially against the strongest opponents (DP\(_\text{async}\) and BR\(_\text{async}\)). For example, on the Scotland-Yard Map against DP\(_\text{async}\), R2PS achieves 0.76 success vs PSRO’s 0.00, and similar gaps exist on Downtown and other real-world maps. This strongly supports the claim of worst-case robustness and cross-graph generalization.
   - **Table 3** demonstrates that for much larger graphs (up to ~2k nodes), the RL policy preserves reasonable success rates (e.g., 0.56 on Times Square) while keeping per-step inference under 0.01 s, compared to DP taking tens to hundreds of seconds. **Figure 6** in Appendix D.2 further visualizes the drastically different scaling behaviors.

5. **Belief mechanism is empirically meaningful, not just cosmetic.**  
   The comparison between DP\(_\text{Pos}\) and DP\(_\text{belief}\) in **Table 1** (right side) clearly demonstrates that the belief-averaged policy substantially improves capture rates under partial observability. E.g., on the Hollywood Walk of Fame, DP\(_\text{Pos}\) gets 0.25 success while DP\(_\text{belief}\) reaches 0.48. **Table 4** further shows that (i) updating belief using an accurate evader model improves success substantially, and (ii) reducing update frequency degrades performance, indicating the mechanism genuinely influences policy quality.

6. **Figures nicely support the conceptual story and provide intuition.**  
   - **Figure 2** and **Figure 3** give a step-by-step visualization of the belief preservation process, showing how the “shadowed” belief region spreads and shrinks as the game evolves. This bridges the somewhat abstract update rules in (4)–(7) with concrete pursuit behavior and makes it much easier to understand the claimed benefits.
   - **Figure 4** in Appendix C.4 shows learning curves with and without guidance (\(\beta=0.1\) vs \(\beta=0\)), and the gap confirms that DP-based guidance accelerates cross-graph RL training.

7. **Complexity arguments are practical and backed by measurements.**  
   Section 4.2 argues the per-step inference cost of the GNN policy is \(\mathcal{O}(n^2 m)\), compared to DP’s \(\tilde{\mathcal{O}}(n^{m+1})\). This is corroborated by actual timings: for \(n\approx 1000, m=2\), the DP recomputation per time step is said to take >2 minutes on CPU, vs <1 s (and <0.01 s on GPU) for RL. **Table 3** and **Figure 6** validate these scaling claims empirically.

8. **Nontrivial and well-chosen experimental design.**  
   The training corpus combines synthetic dungeon maps and 150 real-world urban graphs; tests are run on 10 separate real-world locations plus a grid and a board-game map. The partial observability setting (range 2) is challenging, and the evader is often the DP-optimal asynchronous agent. This setup is convincing for a worst-case security interpretation.


## Weaknesses

1. **Limited theoretical guarantees under partial observability beyond the trivial consistency case.**  
   The only formal statement regarding the partial observable policies is **Lemma 2**, which states that when \(\mathrm{Pos}\) is always a singleton, \(\mu(s_p,\mathrm{Pos})\) and \(\mu(s_p,\mathrm{belief})\) collapse to the perfect-information optimal policy. However, for the main regime of interest – when \(|\mathrm{Pos}|>1\) and belief is diffuse – there is no attempt to characterize the suboptimality gap between (5)/(6) and any Bayes-optimal POMDP policy, even qualitatively (e.g., under simplifying assumptions). Since the method is pitched as “worst-case robust” under partial observability, some form of theoretical robustness guarantee (even a conservative bound derived from properties of \(D\)) would considerably strengthen the claims. As it stands, the partial observability part is essentially heuristic from a theory perspective.

2. **Belief update model makes strong, somewhat ad hoc assumptions and is under-specified in the main text.**  
   - In **Equation (7)**, the belief update uses \(\sum_{\text{neighbor }v\text{ of }s_e}\nu(v,s_e)\,\mathrm{belief}_\text{old}(v)\), but earlier the text notes that “the pursuer side cannot obtain the evader's policy \(\nu\)” and thus sets \(\nu\) uniform over neighbors. This is effectively assuming that the worst-case, fully informed evader behaves like a random walk in the belief update, while the *actual* evader used in experiments is the DP-optimal asynchronous \(\nu^*\). This mismatch is mentioned qualitatively (and partially addressed by the “known opponent” experiment in **Table 4**), but the implications are not analyzed: does the uniform model systematically underestimate escape routes, leading to overly optimistic beliefs and hence the “optimistic estimator” comment?  
   - The denominator in **Equation (6)** is \(\sum_{s_e}\mathrm{belief}(s_e)\); the paper does not discuss what happens if the belief mass becomes zero (e.g., due to numerical underflow or overly aggressive pruning when combining (4) with (7)). Some clarification on how normalization and numerical stability are handled in practice is needed.

3. **Comparative baselines are narrow and omit closely related partial observability work.**  
   The main RL baseline is PSRO, which is a reasonable choice for robust game RL, but the work does not compare to other PEG or partially observable game methods that are quite relevant:
   - Horák & Bošanskỳ (2017) actually provide a DP-style method for one-sided partially observable pursuit–evasion games; even if it does not scale to the authors’ graph sizes, it would be informative to show at least a small-scale benchmark or a synthetic comparison to demonstrate that the belief mechanism is competitive with more principled POMDP treatments.
   - There is no experimental comparison to recent RL-based approaches to partial observability in pursuit–evasion or adversarial settings (see more under “Potentially Missing Related Work”). Given that the main claimed novelty is partial observability + robustness + real-time inference, the absence of these baselines makes it harder to judge the relative progress.

4. **Evaluation largely focuses on success rate, ignoring time-to-capture and sample efficiency.**  
   All main tables (e.g., **Tables 1–4**, **6–8**) report binary success rates under a fixed 128-step horizon. While success is certainly important, pursuit strategies that consistently capture earlier are more valuable in many applications, and the DP distance table \(D\) encodes precisely this measure. Yet the paper does not report expected capture time, even in the fully observable case where DP gives exact values. It would be particularly informative to see how R2PS’s average capture time compares to DP\(_\text{belief}\) and PSRO on the same test states, and how much is lost by the learned policy. **Figure 4** does plot “termination timestep” during training, so this metric is available; it should appear prominently in the main results to support “worst-case robust” claims beyond just non-zero success.

5. **Some aspects of the RL and DP integration are under-explained and may hide subtle mismatches.**  
   - In Section 4.1, the training state \(s\) sampled for the replay buffer is described as “a randomly generated global state in the sampled graph,” and the reward is based purely on capture vs non-capture. It is not completely clear whether the training episodes always start from random configurations with the constraint that the initial distance exceeds the observation range (as in the test setup), or whether some other distribution is used. This matters for how representative the DP guidance is.  
   - The policy guidance term in **Equation (8)** uses the deterministic action \(a^*=\mu^*(s)\). Under partial observability, \(s\) is replaced by \((s_p,\mathrm{Pos},\mathrm{belief})\) and \(\mu^*\) is replaced by \(\mu(s_p,\mathrm{belief})\). Because \(\mu(s_p,\mathrm{belief})\) itself assumes uniform \(\nu\) whereas the adversary is \(\nu^*_\text{async}\), the guidance is in principle biased relative to the actual game dynamics. The empirical benefit is clear from **Figure 4**, but explicit discussion of this mismatch and its implications would be helpful.

6. **Clarity and notation issues in the DP algorithm and proofs.**  
   - **Algorithm 1**’s for-loop condition “for evader neighbor \(n_e\in \text{Neighbor}(s_e), \nexists n_e' \in \mathcal{V}, (n_e,n_e')\in E, D(s_p,n_e') > D(s_p,s_e)\) do” is quite dense and not fully explained in the main text. It takes some effort to see that this is enforcing the maximization over neighbors in Lemma 1. A short, intuitive explanation with a small illustrative example would improve readability.  
   - In **Appendix A.1**, the Bellman equation is written in a somewhat unconventional piecewise fashion; mixing the case \(f(s)=1\) into a vector form with the reward and discount could confuse readers. Also, in several places the notation for “neighbor” and for sets like \(\text{Neighbor}(\mathrm{Pos})\) is inconsistent (sometimes “Neighbor,” sometimes “neighbor” in subscripts; in **Lemma 1** proof there are typos like “n e i g h b o r” from formatting). These do not affect correctness but reduce clarity.

7. **Scope of experiments is still limited in some important dimensions.**  
   - The main non-appendix experiments fix the number of pursuers to \(m=2\), only later showing results for \(m=4,6\) in **Table 8**. While it is understandable that DP preprocessing becomes infeasible for large \(m\), the proposed grouping mechanism from EPG is only lightly mentioned. A more thorough evaluation demonstrating that R2PS + grouping really scales in \(m\) (even without DP oracles) would significantly strengthen the claim of a generic multi-agent solution.  
   - The partial observability setting always assumes that only pursuers are observation-limited and the evader is fully informed. This is the hardest case for pursuers, but for a “general” security framework it would be interesting to see how the approach behaves when both sides have limited sensing, or when external sensors (like the purple nodes in **Figure 2**) are present in the test environments. The current exposition uses these only in an illustrative example.

8. **Missing and under-discussed related work on partial observability and adversarial modeling.**  
   While Appendix E covers several PEG-specific works, the paper does not acknowledge broader lines of work on learning opponent models under partial observability or on deep RL in differential pursuit–evasion games. Given the focus on real-time learning and adversarial robustness, failing to position against these strands weakens the literature review (see next section for specific missing references).


## Potentially Missing Related Work

1. **Ye et al., “Learning Models of Adversarial Agent Behavior under Partial Observability,” 2023.**  
   This work proposes GrAMMI, a graph-based model for predicting adversarial behavior under partial observability, which is closely aligned with the idea of modeling a strong evader with limited information. It should be discussed in Appendix E (related work on partial observability and opponent modeling) and possibly compared conceptually to the belief-preservation mechanism and to the use of DP-based oracles for the evader. In particular, Ye et al.’s perspective on learning opponent models could illuminate alternatives to the uniform \(\nu\) assumption in **Equation (7)**.

2. **Borra et al., “Reinforcement Learning for Pursuit and Evasion of Microswimmers at Low Reynolds Number,” 2021.**  
   This paper applies RL to pursuit–evasion scenarios with environmental constraints and partial observability. While the physical domain differs, the methodological setup (learning pursuit strategies in complex dynamical environments under sensing limitations) is quite analogous. It would be appropriate to mention it in Appendix E’s discussion on “Finding optimal strategies in PEGs” or a short paragraph on continuous vs graph-based PEGs, highlighting similarities and differences in handling partial observability.

3. **Xu et al., “Pursuit and Evasion Strategy of a Differential Game Based on Deep Reinforcement Learning,” 2022.**  
   Xu et al. consider deep RL for differential pursuit–evasion games. Again, although continuous-state and not graph-based, the central theme of learning robust pursuit policies is very similar. A brief discussion in Appendix E could help position R2PS as the discrete-graph counterpart to these differential-game approaches, and clarify what is uniquely challenging in large, structured graphs.

4. **Yan et al., “Intelligent Maneuver Strategy for Hypersonic Vehicles in Three-Player Pursuit-Evasion Games via Deep Reinforcement Learning,” 2024.**  
   This work uses deep RL in a multi-player pursuit–evasion setting with complex dynamics. It is directly relevant to the broader claim that RL can handle large and complex PEGs. The authors should cite it and compare in terms of scalability and handling of partial observability, ideally in Appendix E’s section on “Policy generalization in PEGs” or at the end of Section 1 where the paper situates itself relative to RL-based pursuit–evasion literature.


## Questions

1. **Suboptimality in partial observability.**  
   Can the authors provide any theoretical or empirical characterization of how far \(\mu(s_p,\mathrm{belief})\) can be from a Bayes-optimal policy under partial observability? For instance, on small graphs where an exact POMDP solution is available, could you compare capture probabilities or expected capture time to show that the gap is limited, or at least illustrate typical patterns?

2. **Belief normalization and zero-mass scenarios.**  
   In **Equation (6)**, what happens if \(\sum_{s_e}\mathrm{belief}(s_e) = 0\) due to pruning with \(\mathrm{Pos}\) or numerical underflow? Do you renormalize beliefs after applying (4) and (7), or impose a minimum floor? Some implementation details here would improve reproducibility.

3. **Training distribution and its match to testing.**  
   In Section 5, test episodes start from random configurations with the constraint that initial distance exceeds the observation range. Is this the same distribution used during cross-graph R2PS training, or are training starts more varied? If different, how sensitive is R2PS to this mismatch? Some short experiments or clarification would be helpful.

4. **Effect of non-uniform evader models in belief updates.**  
   **Table 4** indicates that using the “known opponent” for \(\nu\) in belief updates improves success rates. Could you comment more concretely on how you would learn or approximate such an opponent model in practice, especially under cross-graph deployment? Would you consider joint training of an evader model and the pursuer, or online adaptation?

5. **Scalability in the number of pursuers.**  
   Beyond **Table 8**’s static results for \(m=4,6\), could you elaborate how the grouping mechanism (from Lu et al., 2025a) integrates with partial observability and belief tracking? For example, do groups maintain distinct belief states or share a single global belief? Any detail here would be important for understanding the claimed scalability in \(m\).


## Flag For Ethics Review

- No ethics review needed.  


## Details Of Ethics Concerns

N/A.


## Soundness Rating

3: good.  
The core DP-to-asynchronous extension is carefully proved, the belief mechanism is well-defined and empirically validated, and the RL framework is standard and sensibly implemented. The main theoretical gap is the lack of guarantees under partial observability beyond the singleton-Pos case, and some modeling assumptions (uniform \(\nu\)) are somewhat heuristic, but the overall methodology is technically solid.


## Presentation Rating

3: good.  
The paper is generally well written, with clear structure and helpful figures (notably **Figures 1–4**). Some parts of the DP algorithm and proofs use heavy notation and could be streamlined, and related work on partial observability is incomplete, but the core ideas are understandable with reasonable effort.


## Contribution Rating

3: good.  
The work advances the state of the art in PEGs by (i) extending DP optimality to an asynchronous evader, (ii) proposing a practical belief-preservation extension to partial observability, and (iii) demonstrating a cross-graph RL policy that yields robust, real-time strategies on unseen real-world graphs. While not theoretically comprehensive for POMDPs, the combination of theory, algorithms, and empirical validation is of clear interest to the ICLR community.


## Overall Rating

8: Accept, good paper (poster).  
The paper offers a well-motivated and technically sound contribution at the intersection of game-theoretic planning, partial observability, and cross-graph RL. The DP results for asynchronous moves are clean, the belief-preservation mechanism is both conceptually simple and empirically effective, and the R2PS policy achieves impressive robustness and real-time performance on challenging, unseen graphs. The main limitations are incomplete theoretical treatment under partial observability and somewhat narrow comparative baselines, but these do not undermine the central claims. I recommend acceptance.


## Reviewer Confidence

4: confident.  
I am familiar with PEGs, multi-agent RL, and partial observability, and I carefully checked the main derivations (especially Lemma 1 and Theorem 2) and experimental tables. Some details of the POMDP approximation and large-scale RL setup could still harbor subtleties, but I am confident in the overall evaluation.