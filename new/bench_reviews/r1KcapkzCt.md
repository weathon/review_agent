Now let me read the calibration papers to establish a score baseline.Now I have enough information to write my final review. Let me synthesize everything.

---

## Summary
MC-DML integrates Large Language Models as dynamic prior policies within Monte Carlo Tree Search for text-based game agents. The key technical contribution is two complementary memory mechanisms—in-trial memory (recent observation-action-observation triplets) and cross-trial memory (LLM-generated reflections from failed trajectories)—that adjust the LLM's action probability estimates during PUCT-style search, combined with a dynamic pruning strategy for adaptive search depth. Experiments on 9 Jericho benchmark games show strong results, particularly on difficult games like Deephome and Ztuu, with the paper's framing emphasizing efficiency gains over planning-then-learning baselines that require multiple iterations.

---

## Strengths

- **Strong empirical performance on hard games:** MC-DML achieves nearly double MC-LAVE-RL's score on Deephome (67 vs. 35), and dominant improvements on Ztuu (23.67 vs. 7), demonstrating that the approach handles genuinely difficult environments with multiple bottleneck states.

- **Well-targeted motivation:** The bottleneck state example in Figure 1 (Zork1's trapdoor requiring `take lantern` before opening) is an effective and concrete illustration of why semantic reasoning needs to be integrated into long-horizon tree search.

- **Ablations confirm memory mechanisms matter:** Table 4 systematically ablates the components. Removing both $\mathcal{M}_c$ and $\mathcal{M}_i$ from Zork1 drops score from 48.66 to 31.67; for Ztuu the full model achieves 23.67 vs. 6.33 for the most-ablated variant. These differences are large enough to be credible even with only 3 runs.

- **Qualitative bottleneck analysis:** Table 5 provides a transparent trace of how cross-trial memory shifts action probability from `open trap` (high prior but leads to death) to `take lantern` (correct action), with actual Q-values and visit counts shown. This makes the claimed mechanism concrete and verifiable.

- **Beats a comprehensive field of baselines:** MC-DML surpasses 10 baselines across RL, LLM, and MCTS families on 8/9 games against LLM/MCTS competitors.

---

## Weaknesses

### Major

**1. Efficiency claim is unsubstantiated — the central framing lacks comparative evidence.**
The abstract and introduction repeatedly claim that MC-DML outperforms "strong contemporary methods that require multiple iterations" with greater efficiency "at the initial planning phase." But MC-DML calls an external LLM at *every node selection* during MCTS (Algorithm 1, lines 33–35: SELECTACTION invokes `LLM(Mᵢ, Mᶜ, p_action_probs)`) and additionally for reflections on failure (lines 10–13, 44–47). This means potentially hundreds of LLM API calls per planning decision. The paper provides no measurement of wall-clock time, LLM API call count, token consumption, or cost. Table 3 compares MC-DML's one-shot planning score (48.66) against PUCT-RL/MC-LAVE-RL at each of their 4 iterations, but "iterations" there consist of 25 planning sessions followed by policy learning — a completely different cost unit from "number of LLM queries." As written, MC-DML could be far more expensive in total compute than the baselines it claims to surpass in efficiency. This is not a missing ablation; it is the central claim of the paper that remains unvalidated.

**2. Dynamic Pruning (DP) confounds the comparison against baselines on Ztuu, the game with the largest margin.**
Sec. 4.3 explicitly states: "without DP, we follow the experimental setup of Jang et al. (2020), using a fixed search depth for each game." This means the full MC-DML uses a different search procedure from the MCTS baselines in Tables 1–2, which appear to use the fixed-depth Jang et al. protocol. Table 4 shows this distinction is critical: Ztuu drops from 23.67 (full MC-DML) to 7.8 (w.o. DP), nearly eliminating the gap over MC-LAVE-RL (7). For most other games DP makes minimal difference (Zork1: 48.66 vs. 48; Deephome: 67 vs. 67.4), so the issue is largely confined to Ztuu — but Ztuu is one of the headline gains. The paper does not confirm that baselines also used fixed depth, nor does it provide a version of MC-DML (with DP) that is otherwise identical to the search setup of the baselines.

**3. The most directly comparable prior work (LLM-MCTS, Zhao et al., 2024) is absent from the experimental tables.**
Sec. 3.3 explicitly positions MC-DML as extending and improving over LLM-MCTS, the approach of using a fixed LLM prior in PUCT without memory. Yet Zhao et al. (2024) does not appear in Tables 1–2 as a baseline. This is the single most important ablation for quantifying the contribution of the dynamic memory mechanism: the delta between a fixed LLM prior (LLM-MCTS) and the proposed dynamic memory (MC-DML). Without it, the reader cannot judge how much of the gain comes from "LLM in MCTS at all" vs. "LLM in MCTS with memory."

### Minor

**4. Insufficient statistical rigor for stochastic search.**
All MC-DML results use only 3 independent runs. For sparse-reward stochastic tree search, this is thin: the ablation shows high variance in some settings (Deephome w.o. Mc, Mi, DP: 51 ± 14.9). Differences like Zork1: 48.66 vs. MC-LAVE-RL's 45.2 ± 1.2 are plausible but not verified by significance testing. The PUCT-RL and MC-LAVE-RL entries in Table 2 apparently come from prior published work and may not use exactly the same evaluation protocol (episode length, Jericho handicap version, score normalization). These concerns compound on the small sample.

**5. Underperformance on Ludicorp is not analyzed.**
MC-DML scores 19.67 on Ludicorp against BIKE+CBR's 23.8 and MC-LAVE-RL's 22.8. Ludicorp is classified as "difficult" alongside Deephome where MC-DML excels greatly. No analysis is provided for why the method fails to transfer its advantage on difficult games to Ludicorp specifically, weakening the generality claims.

**6. Single LLM evaluated; log-probability extraction is API-specific.**
All experiments use GPT-3.5-turbo-0125. The method relies on top-20 token log-probabilities from the chat completions API — a feature not available in many open-source models. The paper acknowledges self-consistency/verbalized alternatives in footnote 2 but does not validate them. Whether results transfer to other models is unknown.

### Trivial

**7. In-trial memory window is narrow and acknowledged.**
The paper's own Limitations section notes that defining in-trial memory as only $(o_{t-1}, a_{t-1}, o_t)$ may miss clues from much earlier in the game. This is a real limitation but is already acknowledged by the authors.

---

## Nice-to-Haves

- **Compute-normalized comparison:** A table reporting total LLM calls, tokens consumed, and wall-clock time alongside game scores would directly address the efficiency claim rather than leaving it implicit. Even order-of-magnitude estimates would help.
- **LLM-guided rollout ablation:** The rollout currently uses uniform random action selection (Algorithm 1, line 54), which is known to be poor in sparse-reward settings. Testing an LLM-guided rollout would clarify how much of the MCTS value comes from the expansion prior vs. simulation quality.
- **Cross-trial memory size k sensitivity:** k=3 is set without justification; a sweep over k=1,3,5 would take modest compute and directly validate the memory mechanism's stability.
- **Scaling with simulation budget:** No analysis of performance vs. number of MCTS simulations per root node is provided. This would clarify whether the method's advantage is fundamental or sensitive to compute allocation.
- **Analysis of reflection quality:** The paper shows one reflection example ("Ensure you have a light source...") from Zork1. A broader analysis of reflection frequency, quality, and action-distribution impact across multiple games would strengthen confidence in the mechanism.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] Hyperparameter undisclosure (d_min, d_max, Δd, C_puct):** The paper refers to Appendix B for implementation details. Demanding full hyperparameter tables in the main text is a reproducibility nitpick, and such details are standard to include in appendices. Removed per the rule against trivial reproducibility concerns.

- **[Harsh Critic] Algorithm 1 notation inconsistencies (history vs. state symbols):** The Harsh Critic flagged `SELECTACTION(h)` using `Q(s,a)` and `ROLLOUT` argument ordering. These are plausibly parsing artifacts from PDF extraction, not genuine algorithmic errors. Removed as a formatting/style nitpick.

- **[Harsh Critic] Conceptual overstatement that "cross-trial memory adjusts value estimation":** Eq. (3) shows the LLM modulates the PUCT exploration bonus, not Q(s,a) directly. However, via PUCT dynamics, this does indirectly shape which actions accumulate visits and thus Q updates. The paper says "dynamically adjust action evaluations" (Sec. 3.1), which is technically defensible. The claim is not clearly wrong enough to rise to a main weakness.

- **[Harsh Critic] POMDP return notation $R_t = \mathbb{E}[\sum_{i=0}^{\infty} \gamma^i r_t]$ is imprecise:** Minor theoretical notation issue; not relevant to the method's correctness. Pure nitpick.

- **[Harsh Critic] "Cross-trial memory local to same root" framed as misleading readers:** The paper describes this accurately in Sec. 3.1 and acknowledges it in Limitations. The paper's framing is not dishonest about the scope. Weakened.

---

## Novel Insights

The most genuinely novel observation is that pairing in-trial memory (short-term trajectory grounding for POMDP partial observability) with cross-trial reflective memory (cross-simulation learning from failures within a planning episode) can substitute for the iterative policy improvement loop of planning-then-learning paradigms, at least in sparse-reward text environments where LLM world knowledge is directly applicable. The key insight is that the MCTS restart mechanism — which vanilla MCTS uses only for value averaging — can double as a natural unit boundary for LLM-based reflection: each trajectory to a terminal state provides a complete failure episode from which the LLM can generate targeted guidance. This is a sensible and underexplored design point. The paper would be significantly strengthened by verifying that the total computational cost of repeated LLM calls during one planning phase is indeed less than the cost of multiple RL training iterations, which would turn a promising hypothesis into a demonstrated engineering advantage.

---

## Suggestions

1. **Report LLM inference cost empirically**: Count and report total LLM API calls and approximate token usage per game-step decision. Compare this against the total computational cost of one RL iteration in PUCT-RL/MC-LAVE-RL (the 25-planning-session cycle). Even a rough estimate would substantiate or contextualize the efficiency claim.

2. **Add LLM-MCTS (Zhao et al., 2024) as a baseline**: Run Zhao et al.'s fixed-LLM-prior MCTS on the same 9 Jericho games under the same search settings, and include it in Table 2. This is the clearest way to quantify the marginal value of the dynamic memory mechanism.

3. **Normalize the Ztuu comparison**: Provide a row in Table 3/4 for "MC-DML + LLM prior + DP, no memory" on the Ztuu game, so readers can see whether DP alone or DP+memory drives the large gain. This would sharpen the attribution considerably.

4. **Increase to at least 5 independent runs** for reported results, or add confidence intervals to main tables. Given the stochastic nature of the search and relatively tight margins on some games (Zork1: 48.66 vs. 45.2), this is important for the comparison credibility.

---

## Score and Decision

**Calibration:**

- **GBIUbwW9D8 (ExACT, R-MCTS)** — Accepted poster, avg score ~5.75. More comprehensive (VisualWebArena, fine-tuning, richer evaluation). This paper is clearly above MC-DML in scope and rigor.
- **gfI9v7AbFg (Strategist, bi-level MCTS+LLM)** — Accepted poster, avg score ~5.4. Similar framework depth (LLM+MCTS, game environments), similar ablation quality. Comparable in scope to MC-DML, slightly better justified claims.
- **fLO9VaAb3B (TS-LLM, AlphaZero-like)** — Rejected, scores 5,5,6,6 avg 5.5. Confused efficiency/performance comparisons, mixed results. MC-DML has clearer and stronger empirical results than TS-LLM but shares similar weakness of unvalidated efficiency claims.
- **OJUcOLOLXL (RethinkMCTS)** — Rejected, scores 6,6,3,3. Missing compute comparisons, statistical gaps. Similar structural weaknesses to MC-DML.

MC-DML sits between the rejected MCTS+LLM papers (fLO9VaAb3B, OJUcOLOLXL) and the accepted poster papers (GBIUbwW9D8, gfI9v7AbFg). The empirical results are genuinely strong on several games, and the memory mechanism idea is sound. However: (1) the central efficiency claim is unsubstantiated, (2) the missing LLM-MCTS baseline leaves the primary contribution unquantified, (3) the DP confound on Ztuu weakens the attribution. These are not fatal to the empirical contributions but meaningfully undermine the specific claims of the paper.

The paper is broadly comparable to the Strategist paper (accepted at ~5.4 avg) in quality, but Strategist's claims were better calibrated to what the experiments show. MC-DML's overclaiming on efficiency is a more significant issue. I place this at **5.0** — marginally below acceptance threshold — primarily because the core efficiency claim is entirely unmeasured, the most direct competitor is missing from experiments, and the statistical footprint is thin. The ideas and most results are legitimate contributions that could support acceptance with targeted revisions.

**Originality:** Moderate. Memory-augmented LLM in MCTS is a natural extension; individual components (reflection, PUCT prior) are prior work. Integration for text-game POMDP planning within a single search phase is new.

**Importance of research question:** High. Combining LLMs with structured search efficiently is a genuinely important problem.

**Claim support:** Partial. Empirical results strongly support that the approach works; they do not support the efficiency claim.

**Soundness of experiments:** Adequate but with real gaps (DP confound, 3 runs, missing LLM-MCTS baseline).

**Clarity of writing:** Good overall; algorithm description is slightly underspecified but readable.

**Value to community:** Moderate positive; the technique is applicable and the games are a reasonable benchmark, but the narrow domain and API-specific implementation limit immediate broad applicability.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>