=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
## Summary
This paper proposes applying stateless tabular Q-learning to the Lowest Unique Positive Integer (LUPI) game, aiming to approximate the Poisson–Nash equilibrium without requiring *a priori* distributional assumptions. The authors compare the learned strategy to the known symmetric NE under Poisson assumptions and then apply the agent to 49 days of real Swedish Limbo lottery data, reporting win rates above the theoretical baseline. The stated motivation is to provide a distribution-free, player-count-flexible alternative to Poisson–Nash equilibrium computation.

---

## Strengths

- **Well-motivated limitation of prior work:** The paper correctly identifies and concisely articulates a genuine flaw in the Poisson–Nash framing of Östling et al. (2011): the observed cross-day variance in Limbo player counts far exceeds what a Poisson distribution predicts, undermining the model's core assumption. This is a specific, documented gap that the learning-based approach is positioned to address.
- **Real-world empirical evaluation:** The use of publicly available real Limbo lottery data (from Östling et al.'s replication dataset) is a reasonable step beyond pure simulation, and the paper's observation that actual Limbo winners consistently select numbers in the thousands—while NE theory concentrates probability mass on k < 15—is a concrete, data-grounded finding about the gap between theory and practice.

---

## Weaknesses

### Fatal

**The central claim that Q-learning "successfully emulates the Nash equilibrium" is unsupported by either theory or empirical evidence.** There is no theoretical guarantee that independent Q-learning converges to Nash equilibrium in multi-player general-sum games such as LUPI. The paper provides no proof, no convergence argument, and cites no applicable result. On the empirical side, the measured distribution visibly and numerically diverges from NE at higher k: at k=13, NE assigns probability 0.01 while the agent assigns 0.04; at k=14, NE assigns 0.00 while the agent assigns 0.04 (four times as much). The tail behaviour—where the NE rightly concentrates near-zero weight—is where the agent most clearly fails. The paper describes these differences as "minimal discrepancies," which is not defensible given the data. No divergence metric (total variation, KL divergence, L₁ norm) is computed or reported to substantiate "high accuracy and reliability." Without either a theoretical guarantee or a quantified empirical measure of closeness, the paper's primary claim rests on nothing.

**The Limbo experiment contains a factual internal inconsistency that renders its quantitative results untrustworthy.** Table 1 (Summary Statistics) states "Total wins = 8" and "Win percentage rate = 16.33%." However, counting the "Agent Win?" column in Table 2 yields twelve entries marked 1: Days 4, 7, 10, 14, 30, 34, 35, 37, 41, 42, 45, and 49. This 8-vs-12 discrepancy is not explained anywhere in the paper. The headline quantitative result of the applied experiment is therefore in direct conflict with the detailed data table.

---

### Major

- **Algorithm description is internally contradictory, making the method non-reproducible.** Section 4 states: "with probability ε (set to 0.95), a random action is chosen (exploration), while with probability 1−ε, an action is selected using the softmax strategy (exploitation)." Yet the formal equation immediately below defines the exploitation action as `argmax_a Q(a)`, not as a softmax draw. Softmax exploitation and argmax exploitation are behaviourally distinct. It is impossible to reproduce the reported results without knowing which was actually implemented.

- **Key experimental parameter n is never stated.** Section 5 presents the entire Figure 1 comparison between the theoretical NE and the agent's distribution under "Poisson assumption with expected player count n," yet the value of n used is never specified. The shape and absolute values of the NE curve depend critically on n. Without this, the NE comparison in Figure 1 and the data table cannot be reproduced or interpreted.

- **The "0 theoretical wins" baseline is misleading.** The NE assigns near-zero probability to numbers above approximately 20, while actual Limbo winners are systematically in the thousands (e.g., Day 6: actual win = 7,194; Day 23: 8,357; Day 38: 5,913). Of course the NE strategy, which would sample numbers like 1–15, never wins. The comparison does not demonstrate that Q-learning has discovered a superior strategy; it demonstrates that the NE is badly miscalibrated for actual Limbo data, which is already acknowledged in the introduction. A meaningful baseline would be, for example, sampling from the empirical historical distribution of winning numbers, or a simple histogram estimator of past winners. Without such a baseline the 16.33% win rate cannot be attributed to Q-learning's strategic reasoning versus simple memorisation of the distribution of past winning numbers.

- **The modified-data experiment (Section 6, second part) is described in nearly incomprehensible terms.** The paper states: "we excluded the top 700 most popular numbers, leaving a set of approximately 1,000 potentially winning numbers out of the 100,000 possible choices... There was no chance of winning, so we set the best choice to a winning one (if there was a winning choice, we did not change it), and we removed the best choices to give a 10% chance of winning. Specifically, considering the results lower than the winning one, we removed 100 numbers with the fewest selections." The phrase "we set the best choice to a winning one" suggests retroactive modification of actual game outcomes. The data transformation logic is not reproducible, and the resulting experiment's validity is unclear.

- **ε = 0.95 is unjustified and likely undermines learning.** The agent selects a random action 95% of the time with no annealing schedule across all 3,000 episodes. With α = 0.01, this means the Q-values are updated predominantly from uninformative random actions. No ablation or sensitivity analysis is provided for ε, the temperature T, α, or the episode count. The hyperparameter choices appear arbitrary, and with no convergence plot (Q-value evolution over episodes or KL divergence between successive policy snapshots), there is no basis for claiming the agent has converged to anything meaningful by episode 3,000.

- **Tabular Q-learning over 100,000 actions with 3,000 episodes and ε = 0.95.** Only 5% of 3,000 episodes = 150 exploitation steps update the Q-table non-randomly across 100,000 entries. That is fewer than one informative update per action on average. The paper does not address whether Q-learning is a sound approach at the scale of the real Limbo game.

---

### Minor

- **No convergence diagnostic.** The paper provides no plot of policy evolution over training, no stability measure, and no evidence that 3,000 episodes is sufficient. The large standard deviation visible in Figures 1 and 3 at lower k values is consistent with the policy not having stabilised.

- **Flexibility in player count is claimed but never demonstrated.** The abstract and introduction state "flexibility in the number of players" as a key contribution, but no experiment shows the trained agent performing across a range of player counts or handling unknown player counts without retraining. The second Limbo experiment trains on a fixed n ≈ 16,000, not across varying n.

- **No statistical significance for Limbo win rates.** With only 49 data points, reporting a 16.33% win rate without confidence intervals or p-values against a random or empirical-histogram baseline makes it impossible to assess whether the results are above chance. Multiple random seeds should be reported to show variance.

---

### Tiny

- The "Conclusions" subsection (6.1) appears mid-paper before the references rather than as a standalone concluding section, making the paper's structure confusing.

---

## Nice-to-Haves

- **Learning curves and convergence diagnostics:** Plotting KL divergence or L₁ distance between the agent's policy and the NE (or between successive-epoch policies) over training episodes would directly substantiate any convergence claim and show when (if ever) the agent stabilises.
- **Deep Q-Networks or function approximation:** Replacing tabular Q-learning with DQN-style function approximation would address the scalability concern for large action spaces (100,000 numbers in Limbo) and would be more appropriate for the Limbo setting.
- **Self-play or population-based training:** Since the agents are training against other independent Q-learners (non-stationary opponents), incorporating self-play would make the equilibrium-finding dynamics more principled.
- **Comparison against MARL algorithms designed for non-stationary environments** (e.g., PSRO) would help situate the method's performance.
- **Broader validation across different values of n:** Showing that the agent's strategy shifts appropriately as n changes would concretely substantiate the "flexibility in player count" claim.
- **Monte Carlo sampling for the theoretical baseline:** Rather than checking if the NE's deterministic mode matches the actual winner, simulate wins by sampling from the theoretical NE distribution repeatedly to compute an expected theoretical win rate under the same game conditions.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Conclusions appear mid-paper" (Harsh Critic):** This is a pure formatting/structural style complaint. The scientific content is unaffected.
- **"RTB systems motivation underdeveloped" (Harsh Critic):** The RTB application is flagged in the introduction with appropriate citations as a motivation, not as a claimed contribution. Criticising the lack of an RTB experiment is scope creep.
- **Demand for theoretical proofs of convergence from an empirical systems paper (Spark Finder — "exploitability metrics," formal proofs):** While a convergence argument would strengthen the work, demanding formal exploitability analysis or theoretical proofs goes beyond what is standard for an empirical RL paper in this setting and is better placed as a nice-to-have.
- **Demand for modern MARL baselines (PSRO, MADDPG) as a prerequisite for publication (Spark Finder):** Comparing to these algorithms would be interesting but is not a standard expectation for a paper whose scope is equilibrium approximation in a specific game; moved to Nice-to-Haves above.

---

## Novel Insights

One substantive insight does emerge from synthesising the reviews: the paper's Limbo data inadvertently surfaces a striking empirical pattern — the Poisson–Nash equilibrium, which concentrates probability almost entirely on numbers below 20, has zero wins in 49 actual Limbo rounds whose winners ranged from ~2,730 to ~9,880. This gap between theoretically optimal play and empirically winning numbers is quantitatively dramatic and suggests that in large LUPI games with tens of thousands of players, strategic players systematically migrate to larger numbers to avoid collision, a dynamic the Poisson-NE does not capture. The paper gestures at this without fully analysing it; a deeper examination of *why* this divergence occurs (e.g., heterogeneous player sophistication, anchoring effects, or the excluded-popular-numbers phenomenon) would be genuinely valuable. Beyond this observation, the paper does not add novel insights beyond reapplying standard tabular Q-learning.

---

## Suggestions

1. **Fix the win-count discrepancy immediately.** Reconcile Table 1 (8 wins) with Table 2 (12 apparent wins) by either correcting Table 2's "Agent Win?" column or re-explaining the win criterion. This is a basic correctness requirement.
2. **State n explicitly in Section 5.** The NE comparison is uninterpretable and unreproducible without it.
3. **Resolve the ε-greedy vs softmax contradiction in Section 4.** Clearly state which exploitation rule is used and ensure text, equation, and code all agree.
4. **Reframe or remove the "0 theoretical wins" comparison.** Replace or supplement it with a meaningful baseline (e.g., empirical histogram of past Limbo winners, uniform random over the ~1,000 "potentially winning" numbers) so the Q-learning win rate can be contextualised.
5. **Add convergence diagnostics.** Plot the agent's policy distribution against the NE at multiple training checkpoints (e.g., every 500 episodes) and report a quantitative divergence measure (L₁ or KL) rather than asserting "minimal discrepancies."
6. **Clarify and justify the data manipulation in the second Limbo experiment.** If game outcomes were modified retroactively, explain what was changed and why, and discuss how this affects validity. Consider whether the modified experiment should be removed or radically simplified.
7. **Report multiple random seeds with error bars** for the Limbo win rate experiments.

---

**Axis evaluation:**
- **Novelty:** Very low. Standard tabular Q-learning applied to a well-studied game; no algorithmic contribution.
- **Technical soundness:** Poor. The algorithm description is internally inconsistent, the central convergence claim lacks any theoretical or rigorous empirical basis, and key experimental parameters are missing.
- **Empirical support:** Poor. The headline quantitative result (16.33% win rate) is internally contradicted by the detailed data table, the comparison baseline is misleading, and no statistical testing is performed.
- **Significance:** Very low. Even accepting all results at face value, the paper does not move the needle on either MARL methodology or game-theoretic understanding of LUPI.
- **Clarity:** Poor. Multiple sections contain contradictions or incomprehensible data manipulation descriptions that prevent reproducibility.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
