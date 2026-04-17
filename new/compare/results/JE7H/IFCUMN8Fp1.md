---
job_id: c351d1de-2e17-49dd-94d2-c33abfd08ddf
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: IFCUMN8Fp1.pdf
paper: Towards Learning POMDPs Without Full Observability
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is about learning discrete POMDP models via spectral / tensor methods and using them for planning, which fits squarely within reinforcement learning, representation learning, and learning theory.

## Minimum Quality
Pass ✅.  
The paper is complete and in English; it contains Abstract, Introduction, Problem Setting, Method (Sections 3–4), Experiments (Section 5 plus Appendix C), Results (Figures 3–4, Tables 1–6), Related Work (Section 6), and Conclusion (Section 7). The methodology is technically nontrivial, proofs are supplied in the appendix, and experiments, while limited in scale, are coherent and reproducible.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any instructions aimed at influencing LLM-based review or other hidden manipulations in the main content.

---

# Expected Review Outcome:

## Summary

The paper studies learning discrete POMDPs from action–observation sequences under a random exploration policy, without access to the latent state space. It first uses spectral PSR learning from a Hankel matrix, then derives and estimates a similarity transform via a reformulated tensor / joint-diagonalization procedure to recover transition and observation matrices up to a *full‑rank observability partition* of states, i.e., equivalence classes of states that share observation distributions across all actions whose transitions are full rank. Experiments on Tiger, T‑Maze, a challenging Sense‑Float‑Reset domain, and two noisy hallway domains compare the learned POMDPs to PSRs and EM, and investigate how explicit observation/transition models can be used for downstream reward specification and planning.

## Strengths

1. **Clear conceptual bridge between PSRs and POMDP parameter recovery.**  
   The paper usefully connects classical PSR spectral learning with tensor-style eigenstructure methods. Proposition 1 (Sec. 3.2) and the subsequent use of Eqs. (7)–(9) make explicit that the SVD factors of the Hankel matrix correspond to Forw·P and P⁻¹·Back, so that PSR update matrices are similar to the true $O^{ao}T^a$. This is a clean rephrasing of Carlyle & Paz / Balle et al. in POMDP language and sets up the rest of the algorithm.

2. **Well‑defined identifiability target: recovery up to full‑rank observability partition.**  
   Instead of overclaiming full identifiability, the authors carefully characterize what can and cannot be recovered. Theorem 1 (Sec. 4.1) formalizes that the algorithm recovers correct *partition‑level* beliefs and transition dynamics whenever states are grouped by identical observation distributions across full‑rank actions. The discussion around Sense‑Float‑Reset and Fig. 1 and Fig. 2 makes this notion intuitive: the method cannot separate states whose entire observable behavior is symmetric, but it recovers correct belief mass over equivalence classes.

3. **Technically nontrivial but mostly careful linear‑algebraic analysis.**  
   The derivation of the joint diagonalization step (Sec. 4.2), particularly Eq. (17) and Eq. (18), and Lemma 1’s statement about almost‑sure distinct eigenvalues under random weights, is mathematically sound and expressed with explicit notation. The follow‑up about block‑diagonal ambiguity (Lemma 4, Appendix A.4) and the construction of the randomized block rotation $R$ and scaling diag$(R^\top Q^{-1}\mathbf{1})$ to enforce a valid summation vector (Sec. 4.3 and Appendix A.5) are detailed; they correctly address the fact that similarity is only defined up to invariant subspaces for equal eigenvalues.

4. **Thoughtful use of figures to illustrate the core idea.**  
   - **Figure 1** clearly visualizes Sense‑Float‑Reset, including partial observability structure and the observability partitions via node shading; this concretely illustrates why single‑action tensor methods fail and motivates the need for aggregating across actions.  
   - **Figure 2** is useful: it explicitly shows how, after applying $\tilde P$, summing over indices corresponding to a partition yields the correct probability of being in that partition, visually supporting Eqs. (13)–(14).  
   - **Figure 3** systematically reports convergence of estimated state number, $L_1$ observation error, $L_1$ transition error, and planning reward across domains for PSR, the proposed method, and EM, supporting claims about both identifiability and downstream usefulness.  
   - **Figure 4** gives a nice qualitative demonstration that state‑level reward specification using learned $O^{ao},T^a$ yields behavior in noisy hallway that cannot be achieved by observation‑only rewards.

5. **Empirical comparison with PSRs and EM, including ablations and sensitivity.**  
   The authors compare against a standard EM learner (Rabiner / Shatkay & Kaelbling) and linear PSRs using the same estimated rank as their own method. In **Figure 3**, EM clearly fails to converge to the correct observation / transition model, while the proposed method matches PSR planning performance and recovers parameters (for domains where this is possible). Appendices C.1–C.2 and Tables 1–5 include nontrivial parameter and sensitivity studies (e.g., effect of Hankel length and condition‑number threshold $1/\kappa$ on rank estimation and errors), which is more thorough than typical for spectral‑learning papers.

6. **Explicit complexity discussion and recognition of limitations.**  
   Appendix B.2 derives a floating‑point runtime of $O(|\mathcal{S}|(|\mathcal{A}||\mathcal{O}|)^{2(n^{obs}+1)}+|\mathcal{S}|^2(|\mathcal{A}||\mathcal{O}|)^{n^{obs}+2})$ in terms of a “full‑observability length” $n^{obs}$, and the authors explicitly acknowledge the exponential dependence and the need for bounded $n^{obs}$. The conclusion discusses future work on matrix completion and PAC analyses.

7. **Reproducibility details.**  
   The paper provides extensive algorithmic pseudocode (Alg. 1), explicit specifications of domains in Appendix C.5, and tables of all spectral thresholds and planning hyperparameters (e.g., **Table 1**, **Table 6**). This would allow another group to re‑implement and stress‑test the method.

## Weaknesses

1. **Scope and realism of assumptions are quite restrictive and under‑examined empirically.**  
   The method assumes (Sec. 3.3 and Lemma 2) an ergodic Markov chain $(s_t,a_t,o_t)$ under a fixed uniform random exploration policy with full support over states, and that the truncated Forw and Back are each full rank equal to $|\mathcal{S}|$. This excludes many POMDPs of interest (e.g., episodic tasks with terminal absorbing states not revisited under random exploration, or systems with structural low‑rank transitions). The discussion in Sec. 4.1.1 gives qualitative examples for full‑rank actions and ergodicity, but there is no quantitative exploration of how violations (e.g., near‑reducibility, sparse success probabilities) impact learning. All experiments are in tiny toy domains explicitly constructed to satisfy the assumptions. This raises concerns about how broadly the method can be used in realistic RL or robotics settings.

2. **Finite‑sample behavior and error propagation are not analyzed theoretically.**  
   All main theorems (Theorem 1, Lemma 1, Lemma 4) are in the infinite‑data limit. Appendix B.1 acknowledges approximation error and introduces several heuristic thresholds ($1/\kappa$, $\sigma_{\min}$, $\tau_{obs}$), plus a post‑hoc projection of estimated parameters onto probability simplices via quadratic programming. There is no finite‑sample bound relating Hankel estimation error (Eq. (6)) to PSR parameter error to recovered $O^{ao},T^a$ error. Given that the method stacks an SVD followed by a joint eigendecomposition, its noise amplification could be substantial; the slower convergence of transitions noted in Sec. C.4 and Figure 4 underscores this. Some experimental convergence plots (e.g., the large error bars for observation / transition errors in Sense‑Float‑Reset in **Figure 3**, rows 2–3, right panels) hint at instability, but there is no systematic analysis.

3. **Algorithm is fragile to hyperparameters and Hankel design; dependence is complex.**  
   Appendix C.1 and **Table 1** show that one must choose: (i) history/test length (Hankel size), (ii) SVD condition‑number threshold $1/\kappa$, (iii) minimum singular value $\sigma_{\min}$ to decide full‑rank transitions, and (iv) $\tau_{obs}$ to cluster observation distributions. Appendix C.2 and **Table 2–4** report that estimated state number and parameter errors vary heavily across these settings, and in higher‑state T‑Maze instances, large Hankel matrices and fairly aggressive thresholds are needed to obtain the correct rank; many configurations either overestimate rank or never detect any full‑rank actions (yielding NaNs). Yet the main experiments in Sec. 5 fix a single configuration per domain, selected without a clear tuning protocol that would be available in a real unsupervised setting. This suggests the method can be quite brittle in practice.

4. **Experimental scope is limited and lacks comparison to several relevant POMDP‑learning baselines.**  
   The main empirical domains are Tiger (2 states), truncated T‑Maze (up to 14 states only in sensitivity appendix, but 2‑map simplified in main), and Sense‑Float‑Reset variations with 3–4 states, plus 3‑state hallway toy domains. These are valuable sanity checks but far from stress‑testing the method’s scaling limits in $|\mathcal{S}|,|\mathcal{A}|,|\mathcal{O}|$, or $n^{obs}$. There is no comparison against more recent theoretical POMDP model‑learning algorithms such as spectral POMDP learners with oracle assumptions or observable variants (e.g., Jin et al., Liu et al.), nor against modern neural belief models (Wang et al., 2023; Allen et al., 2024) in terms of planning performance under similar sample budgets. The only baseline for explicit models is vanilla EM, which is expected to struggle and indeed fails badly in **Figure 3**; this makes the empirical story somewhat one‑sided.

5. **No direct evaluation of belief / partition accuracy beyond aggregate $L_1$ errors.**  
   The key conceptual output is a *partition‑level* model: correct probabilities over observability partitions. However, the quantitative metrics in **Figure 3** and **Tables 3–4** are global $L_1$ errors over observation and transition matrices, computed only when the algorithm recovers the correct number of states. There is no explicit metric of how accurately the method recovers the partition structure (e.g., normalized mutual information between true and estimated partitions) or partition‑level transition probabilities (e.g., KL over quotient MDP). In Sense‑Float‑Reset, where partitioning is crucial, this makes it hard to tell whether the algorithm is recovering the intended coarse dynamics or simply matching fine‑grained parameters in lucky runs.

6. **Planning evaluation is relatively shallow and potentially biased to success.**  
   The planning experiments (Sec. 5, **Figure 3 row 4** and **Figure 4**) use PO‑UCT with a shallow depth‑3 search and 1000 simulations, but there is no ablation varying planning depth, horizon, or rollout strategy to probe robustness. Moreover, the rollout policies and belief updates differ across PSR and POMDP models (Appendix C.3). While the authors argue these approaches have equivalent observation distributions in theory, in finite‑precision implementations with approximate models differences could be substantial. It would be helpful to see per‑domain reward distributions, not just means with overlapping error bars that visually “look similar”.

7. **Runtime is dominated by Hankel estimation and scales poorly.**  
   Appendix C.2 and **Table 5–6** show that even for toy domains, Hankel estimation via Eq. (6) can take hundreds of seconds to hundreds of minutes on a Xeon CPU when histories/tests are of length (4,3). The algorithmic complexity in Appendix B.2 formalizes the exponential blow‑up in $n^{obs}$. There is no exploration of any scalable approximations (e.g., sketching, randomized SVD, or limiting the set of histories/tests). As presented, the algorithm seems restricted to very small, low‑entropy POMDPs.

8. **Missing or thin discussion of several directly related POMDP‑learning works.**  
   The related work section focuses on PSRs, tensor methods (Azizzadenesheli, Guo), automata learning, and deep RL with learned beliefs, but omits or barely connects to more recent POMDP‑learning papers that also relax assumptions or study observable variants (see next section). This affects how well the contribution is positioned and whether its assumptions are stronger or weaker than competing identification results.

9. **Some derivations and notation could be cleaner or fixed.**  
   - In Eq. (4), the last transition is written as $T^{o_{t+n}}$ rather than $T^{a_{t+n}}$, which is presumably a typo; this also appears later in text.  
   - In Sec. 3.3, the exploration policy is described as $a\sim\pi_{\text{exp}}(\mathcal{A}),\pi\in\Delta(\mathcal{A})^2$, which is confusing notation.  
   - In Eq. (6), there is a stray subscript `$\mathcal{D}_{\rm test}$` and a missing bracket in the denominator term; the definition of ${\rm acts}(hist\oplus test)$ should be tightened.  
   These are minor but add friction in following the math.

10. **No empirical check of full‑rank transition detection step.**  
    The method hinges on correctly detecting full‑rank actions via SVD of $M^a$ (Eq. (16)) and a hard threshold on $\sigma_{\min}$. However, in noisy finite‑sample regimes, near‑full‑rank actions may be misclassified as singular and dropped from $\mathcal{A}_{full}$, degrading identifiability. While **Table 2–4** show some NaNs when no full‑rank actions are found, there is no sensitivity sweep in $\sigma_{\min}$ or diagnostic on how often actions are mis‑labeled in the main experiments.

Overall, while the theoretical formulation is interesting and carefully written, the current empirical evidence and parameter dependence paint a picture of a promising but fragile method whose applicability beyond toy domains remains uncertain.

## Potentially Missing Related Work

1. **Lee, J. N., Agarwal, A., Dann, C., “Learning in POMDPs is Sample-Efficient with Hindsight Observability” (2023).**  
   This work studies learning POMDPs under a hindsight observability condition and provides sample‑efficient algorithms. It should be cited and contrasted in Section 6, with discussion of how the authors’ assumptions (full‑rank actions, observability partition, ergodicity under random exploration) relate to hindsight observability, and whether the proposed method could handle or exploit similar structure.

2. **Golowich, N., Moitra, A., Rohatgi, D., “Learning in Observable POMDPs, without Computationally Intractable Oracles” (2022).**  
   This paper gives an oracle‑free learning algorithm in observable POMDP settings. Since the current submission explicitly tackles identifiability and uses spectral methods, a comparison in Section 6 would help position the proposed algorithm’s assumptions and computational complexity relative to observable‑POMDP work.

3. **Muskardin, E., Tappler, M., Aichernig, B. K., “Reinforcement Learning under Partial Observability Guided by Learned Environment Models” (2022).**  
   This paper combines model learning with RL under partial observability. It is relevant to the claim that explicit learned POMDP models can support planning, and could be discussed in Section 6 around “other POMDP‑learning algorithms” beyond EM.

4. **Sulyok, A. A., Karacs, K., “Towards Using Fully Observable Policies for POMDPs” (2022).**  
   This work explores solving POMDPs via policies from fully observable counterparts. It is relevant to the planning discussion and to the significance of learning explicit transition models versus working in belief or PSR space. It could be mentioned in Section 6 when discussing alternative ways to handle partial observability.

5. **Mun, J., “Off-Policy Learning in Partially Observed Markov Decision Processes under Sequential Ignorability” (2025).**  
   Although focused on off‑policy learning rather than model identification, this paper is relevant to how one can learn under partial observability from trajectories not generated by a uniform random policy. It would be worth citing in Section 6 to contextualize the strong exploration assumption (uniform random) made in this paper and to hint at extensions.

## Questions

1. **Robustness to exploration policy and ergodicity.**  
   How sensitive is the method to using a non‑uniform, but still memoryless, exploration policy, or to mild violations of ergodicity (e.g., transient states or weakly connected components)? Can the authors provide either synthetic experiments or a theoretical sketch showing how biased Hankel estimates affect $P$ and the recovered partitions?

2. **Practical choice of Hankel size and thresholds without ground truth.**  
   In the experiments, $L$ (max sequence length), $1/\kappa$, $\sigma_{\min}$, and $\tau_{obs}$ are tuned per domain (Table 1). In a realistic setting where $|\mathcal{S}|$ is unknown, what would be the authors’ recommended procedure for picking these hyperparameters without peeking at ground truth? For instance, could they rely on stability of estimated rank or cross‑validation on held‑out observation sequences?

3. **Scalability beyond small tabular domains.**  
   Given the exponential dependence on $n^{obs}$ and the Hankel construction costs in **Table 6**, what do the authors envision as the realistic upper limits for $|\mathcal{S}|,|\mathcal{A}|,|\mathcal{O}|$ with current implementation? Could randomized sketching or sub‑sampling of histories/tests (e.g., selective Hankel rows/columns) preserve enough structure for their joint diagonalization to work?

4. **Partition recovery metrics.**  
   Can the authors report explicit evaluation of the learned full‑rank observability partitions (e.g., confusion matrices between true and estimated partitions) for Sense‑Float‑Reset and the hallway domains? This would help verify that Theorem 1’s intended quotient structure is reflected empirically.

5. **Comparison to stronger baselines.**  
   Are there reasons the authors did not compare to more recent spectral / tensor POMDP learners (e.g., Azizzadenesheli et al., Guo et al. with their assumptions, or Jin et al./Liu et al. under undercomplete conditions) or to deep belief encoders such as Wang et al. (2023)? Even small‑scale comparisons on the same toy domains would clarify whether the main gain here is identifiability guarantees, planning performance, or both.

6. **Effect of mis‑detecting full‑rank actions.**  
   In Table 2–4 some settings end up with no full‑rank actions detected. How does the algorithm behave in such cases in the main experiments? Is it simply unable to learn any observation matrices via Eq. (17)? Could a softer treatment (e.g., regularized inverse) partially recover useful structure even when singular values are small?

Clarifications on these points, especially any additional experiments on partition accuracy and hyperparameter selection, would substantially increase confidence in the approach.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A. The paper works on synthetic tabular domains and does not raise obvious issues of bias, safety, or data misuse.

## Soundness Rating

3: good.  
The linear‑algebraic derivations and identifiability statements appear correct under the stated assumptions, and proofs in Appendix A are detailed. However, the absence of finite‑sample analysis, the strong exploration/ergodicity requirements, and the method’s hyperparameter fragility prevent a higher soundness rating.

## Presentation Rating

3: good.  
The paper is generally well written, with clear notation in Sections 2–4 and useful figures (especially Figures 1–4). Some minor typos and confusing notational choices exist (Eq. (4), Eq. (6), policy notation in Sec. 3.3), and the related work could do a better job situating the method within the latest POMDP‑learning literature.

## Contribution Rating

2: fair.  
The main conceptual contribution is to weld PSR spectral learning with a joint‑diagonalization scheme to recover POMDP parameters up to a full‑rank observability partition and to show this can support planning and reward specification. This is interesting but incremental relative to existing spectral PSR and tensor methods, and the empirical evaluation is limited to small toy domains with somewhat brittle hyperparameters.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The work offers a thoughtful and technically careful bridge between PSRs and explicit POMDP parameter learning, with a precise notion of identifiability up to an observability partition and some nice planning and reward‑specification experiments. At the same time, its assumptions are quite restrictive, the algorithm is brittle to Hankel/threshold design, scalability is limited, and the empirical scope and baseline coverage are modest. I see this as promising but not yet strong enough for ICLR’s main track; a more thorough experimental and theoretical treatment of finite‑sample behavior and practical robustness would be needed to justify a clear accept.

## Reviewer Confidence

4: confident.  
I am familiar with spectral methods, PSRs, and POMDP learning theory, and I checked the main derivations and proofs at a moderate level of detail; while I might have missed some edge cases, my overall assessment is unlikely to change dramatically.