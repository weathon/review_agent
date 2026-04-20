Now let me do calibration searches to anchor the score properly.Now I have enough information to write the final consolidated review.

---

## Summary

This paper proposes Adaptive HL-Gaussian (AHL-Gaussian), a method that addresses a fundamental limitation of HL-Gaussian value function learning: the requirement for a manually specified support interval [v_min, v_max]. The authors first theoretically decompose the projection error of HL-Gaussian into truncation and discretization components, proving that projection error scales linearly with interval width and grows when the Bellman target falls outside the interval. Guided by these findings, they introduce a single learnable scalar ξ defining a symmetric interval [−ξ, ξ], updated via gradient descent to minimize projection error. AHL-Gaussian is integrated into DQN, SAC, and TD3 and evaluated on six Atari games and six MuJoCo tasks.

---

## Strengths

- **Concrete, well-motivated problem statement (Figure 1)**: The paper shows empirically and clearly that optimal interval magnitudes differ by orders of magnitude across tasks (e.g., Pong vs. Humanoid), validating that the support specification problem is real and practically significant.
- **Theoretically grounded method**: Theorems 3.1 and 3.2 formally decompose projection error into truncation and discretization components and establish that error scales linearly with interval width (w), directly motivating the adaptive approach. Figure 3 provides clean empirical validation that these theoretical predictions match observed behavior.
- **Simple, modular design (Algorithm 1, line 9)**: AHL-Gaussian introduces a single learnable scalar ξ with negligible computational overhead and plugs into DQN, SAC, and TD3 without architectural changes.
- **Non-learning baseline comparison (Figure 7)**: The paper demonstrates concretely why a naive heuristic (η × max target) fails — η=1.1 fixes Ant-v2 but causes value blow-up on Hopper-v2 — strengthening the motivation for a learned adaptive approach.
- **Robustness analysis (Figures 8–10)**: AHL-Gaussian shows reasonable insensitivity to number of bins m, width-to-variance ratio α, and interval update frequency, providing meaningful evidence that the remaining hyperparameters are not the hidden source of gains.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing ft-HL-Gaussian comparison on Atari (verified)**: The Atari baseline uses HL-Gaussian with a single fixed interval of [−10, 10] (Section 4.2, explicitly confirmed). The paper only introduces ft-HL-Gaussian (per-task fine-tuned intervals) for MuJoCo, never for Atari. Because Figure 1 explicitly shows that optimal interval magnitudes differ dramatically across Atari games, the [−10, 10] default is known to be misconfigured for most tasks. Showing AHL-Gaussian outperforms this deliberately misconfigured baseline proves very little about the core claim that AHL-Gaussian matches or beats the best achievable static-interval HL-Gaussian. This is the most critical missing experiment: without AHL-Gaussian vs. ft-HL-Gaussian on Atari, the discrete control results are only compared to a straw-man baseline.

- **ft-HL-Gaussian consistently underperforms vanilla SAC/TD3 (verified)**: The figure captions for Figures 5 and 6 explicitly state that "SAC w/ ft-HL-Gaussian often underperforms compared to the baseline SAC" and the same holds for TD3. The paper's premise, inherited from Farebrother et al. (2024), is that HL-Gaussian with an appropriate interval should outperform MSE-based methods. If carefully hand-tuned ft-HL-Gaussian fails to beat vanilla SAC/TD3 in most MuJoCo environments, either (a) there is an implementation gap specific to the actor-critic setting, or (b) HL-Gaussian does not transfer well to continuous-action actor-critic algorithms. The paper attributes ft-HL-Gaussian's failure to the difficulty of manual interval selection but does not provide supporting diagnostics (Q-value curves, projection error trajectories for ft-HL-Gaussian). Without this explanation, the reader cannot determine whether AHL-Gaussian's gains over vanilla SAC/TD3 stem from adaptive interval selection or from an unrelated implementation difference. This ambiguity weakens the MuJoCo conclusions considerably.

- **Insufficient Atari evaluation breadth**: Performance is reported on only six Atari games, with no stated number of random seeds, no aggregate metric (median human-normalized score, IQM), and no confidence intervals. RL results on Atari are highly sensitive to game selection and stochasticity. The claim that "AHL-Gaussian excels in five out of six tasks" cannot substitute for evaluation on the standard suite (or even a 26-game subset), particularly since the six selected games may not be representative.

### Minor

- **Proposition 3.1 loses tightness for large intervals**: Proposition 3.1 (Eq. 8) bounds L_MSE by 4·max(|v_min|, |v_max|)²·L_CE + projection_error + C. The coefficient 4ξ² grows quadratically as the interval expands, making the bound progressively looser. While the proposition is used primarily to motivate keeping projection error small (not to provide a tight bound), its theoretical utility weakens exactly in the large-interval regime where AHL-Gaussian operates. The paper should clarify that Proposition 3.1 is directional motivation rather than a tight guarantee, and acknowledge that the argument holds because AHL-Gaussian simultaneously minimizes projection error.

- **Theorems 3.1/3.2 assume a fixed Bellman target**: The projection error analysis assumes a fixed target μ. In practice, μ = T̃Q(s,a) is itself a non-stationary function of the evolving Q-network. The paper does not remark on this gap between the static analysis and the non-stationary training setting.

- **Symmetric interval is partially addressed but not formally integrated**: The method constrains the support to be symmetric [−ξ, ξ]. The paper proposes a shifted variant [−ξ + v_mean, ξ + v_mean] only as an informal suggestion ("we suggest") in Section 3.3. Since value functions in many RL tasks are predominantly positive (e.g., MuJoCo Humanoid, HumanoidStandup), this shift is functionally important but is not part of the formal algorithm or theoretical analysis. The paper should clarify which experiments use the shift, or incorporate it formally.

### Trivial

- **Interval update frequency ablation is uninformative**: Figure 10 evaluates ratios of 1.1, 1.2, 1.3, 1.4, 1.5 — all clustered near 1 and all greater than 1. The range is too narrow to reveal meaningful differences in behavior or to characterize the method's sensitivity. An informative ablation would test a wider range (e.g., 0.1, 0.5, 1.0, 2.0, 5.0) and clarify what "ratio > 1" means mechanically.

---

## Nice-to-Haves

- A comparison against support-free distributional RL methods (e.g., QR-DQN, IQN) would contextualise where AHL-Gaussian stands in the distributional RL landscape, since those methods solve the support specification problem differently. This is out of scope for the paper's core claim but would strengthen the paper's positioning.
- Trajectories of ξ over training for all reported environments (beyond the two shown in Figure 7) would verify that ξ stabilizes rather than exhibiting unbounded growth or oscillation — directly supporting the paper's claim that the interval "converges at a state sufficient to encompass all Bellman targets without further expansion."
- A more complete ablation of η in the non-learning-based heuristic (Section 4.3), spanning a wider range (e.g., η ∈ {0.9, 1.0, 1.1, 1.5, 2.0}) across more tasks, would strengthen the argument that the heuristic is fundamentally limited.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Claim that Proposition 3.1's structure is "qualitatively wrong"**: The harsh critic calls the quadratic pre-factor a fatal contradiction. However, the proposition is used as directional motivation (CE loss optimization is effective when projection error is small), not as a tight bound. AHL-Gaussian explicitly co-minimizes projection error, so the condition under which the proposition applies is the one AHL-Gaussian maintains. Downgraded to Minor (theoretical tightness caveat).

- **Strength Finder: "AHL-Gaussian surpasses even the fine-tuned variant" as a core strength**: While factually accurate, this should not be listed as a core strength because ft-HL-Gaussian underperforms vanilla baselines in most MuJoCo tasks — beating a poorly-performing variant is not strong evidence. Removed because it conflicts with a verified Major weakness.

- **Harsh Critic: "Comparison against distributional RL methods that avoid fixed-support constraints" as missing experiment**: QR-DQN and IQN solving the support problem differently is a fair comparison to mention but this is a different class of methods solving the problem through a different mechanism. As a nice-to-have rather than a required experiment given the paper's scope.

---

## Novel Insights

The paper's most genuinely novel theoretical contribution is the decomposition of HL-Gaussian's projection error into analytically separable truncation and discretization components (Theorems 3.1/3.2), with the resulting characterization that projection error scales linearly with interval width. This explains *why* a "just right" interval exists and provides a precise gradient signal for finding it — a more principled basis for adaptive support selection than prior empirical observation alone. The empirical finding that ft-HL-Gaussian consistently underperforms vanilla SAC/TD3 across MuJoCo (even with per-task tuning) is a secondary insight the paper inadvertently reveals but does not explain: it suggests that HL-Gaussian's benefits documented in Farebrother et al. (2024) may not transfer as straightforwardly to actor-critic continuous-action settings, and this deserves investigation in future work.

---

## Evaluation on Key Axes

**Originality**: Moderate. The specific contribution — gradient-based adaptive support selection for HL-Gaussian — is novel and practically important. The theoretical analysis is the strongest new element. The method itself is straightforward given the theory.

**Importance of research question**: Good. HL-Gaussian has been shown to improve value function learning, but its adoption is hampered by interval specification. Solving this is practically meaningful.

**Whether claims are well supported**: Weak-to-moderate. The core claim for Atari is only partially supported (missing ft-HL-Gaussian comparison); for MuJoCo the support is undermined by ft-HL-Gaussian's unexplained underperformance. The theoretical claims (linear error scaling) are well-validated empirically via Figure 3.

**Soundness of experiments**: Below standard. Six Atari games without seeds or aggregate metrics, an absent ft-HL-Gaussian Atari comparison, and unexplained ft-HL MuJoCo underperformance leave the empirical case insufficiently established.

**Clarity of writing**: Clear. The paper's structure flows logically from theory to method to experiments.

**Value to the research community**: Moderate. A robust version of this paper (with ft-HL-Gaussian on Atari, more games, explanation of the MuJoCo ft-HL failure) would be a useful practical contribution to the RL value function learning community.

---

## Score and Decision

**Calibration anchors consulted:**
- `/home/wg25r/review_agent/human_reviews/vFfMsKjqaH.md` (avg ~4.25, Reject): Most topically similar — categorical distributional RL with Atari + MuJoCo evaluation. That paper has deeper theoretical development but less practical utility and similarly limited empirical breadth.
- `/home/wg25r/review_agent/human_reviews/9JE3HogPCw.md` (avg ~4.75, Reject): RL method evaluated on only 8 Atari games, criticized for limited empirical evaluation — directly parallel weakness pattern.
- `/home/wg25r/review_agent/human_reviews/kIqA447T5c.md` (avg ~4.75, Reject): Distributional RL paper with incomplete baselines, similar score profile.
- `/home/wg25r/review_agent/human_reviews/hyfe5q5TD0.md` (avg 8.0, Accept-Oral): High-scoring RL paper for comparison — significantly stronger theoretical contributions and comprehensive empirical validation.

The paper under review sits above the calibration cluster for rejected distributional RL papers (avg ~4.25–4.75) due to its cleaner practical contribution, multi-domain evaluation (two domains, three algorithms), and solid connection between theory and method. However, it falls well short of acceptance-quality work because of two unresolved major issues: the missing ft-HL-Gaussian comparison on Atari (the primary experimental domain for the discrete control claim) and the unexplained failure of ft-HL-Gaussian to beat vanilla baselines in MuJoCo (which calls the implementation into question). These are not minor presentation gaps — they directly affect whether the paper's core empirical claims can be trusted. Placing this paper at **5.0** (marginally below acceptance threshold): above the weakest rejected papers in this area but lacking the experimental rigor needed for acceptance.

**Final Score: 5.0 — Reject (weak rejection; addressable with additional experiments)**

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>