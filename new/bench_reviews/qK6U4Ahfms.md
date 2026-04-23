Now let me do calibration searches to properly score this paper.Now I have enough information to write the final review.Now I have all the information I need to write the final review.

---

## Summary

OpenCity is a scalable platform for simulating large-scale urban activities using LLM agents. It introduces two complementary optimizations: a system-level LLM request scheduler leveraging IO multiplexing, TCP connection pooling, and multi-core CPU offloading; and a prompt-level "group-and-distill" strategy that clusters agents by semantically similar static attributes to reduce token redundancy without eliminating agent individuality. These are evaluated across six global cities, enabling what the authors claim is the first benchmark comparing LLM agent urban simulations against real-world mobility data.

---

## Strengths

- **Real and quantified engineering contribution (Table 1, Figure 3):** The combination of IO multiplexing (`epoll`), TCP connection pooling, and CPU-task offloading is a practical, well-motivated systems design specific to LLM-agent simulation workloads. The scalability curve (Fig. 3) cleanly shows time-per-agent dropping from ~36s to ~0.06s as the population scales from 1 to 10,000 agents, demonstrating that larger simulations benefit more from the scheduler.

- **Group-and-distill is a genuinely novel prompt optimization technique (Section 4.2, Table 2):** Using in-context prototype learning (IPL) to semantically cluster agents and extract shared context prefixes—rather than hard-coded attribute binning or naive response reuse—is a non-trivial prompt engineering contribution. Table 2 confirms it substantially outperforms "archetype prompting" (top-1 hit rates 74–97% vs. 4–13%) while matching standard batch prompting in faithfulness.

- **First multi-city LLM urban simulation benchmark (Table 3, Section 5.3):** Evaluating generative agents vs. EPR on radius of gyration, OD matrix, and income segregation index across six geographically diverse cities is a meaningful step for the community. The three-level evaluation framework (individual physical → group physical → social domain) is well-chosen and more thorough than most urban simulation papers.

- **LLM agents match or outperform EPR overall (Table 3):** On the RMSE (radius of gyration), generative agents outperform EPR in all four cities where data is available (Beijing, Paris, London, Sydney). On SMSE (segregation), generative agents outperform in 4/6 cities. The claim "comparable to or better than EPR" is broadly supported.

---

## Weaknesses

### Fatal
None.

### Major

- **The 635× speedup is measured against a purely sequential, single-threaded baseline — no concurrent baseline is included.** Figure 3 and Table 1 compare OpenCity exclusively against a sequential baseline (one blocking API call at a time). The overwhelming majority of the speedup derives from basic I/O parallelism, which any `asyncio`/`aiohttp` or `ThreadPoolExecutor` implementation would also recover. The paper provides no comparison against existing async or concurrent LLM frameworks (e.g., a simple thread-pool wrapper, LangChain with async support, or even a minimal Python `asyncio` event loop). Without such a baseline, the 635× figure characterizes the inefficiency of the strawman, not the specific value of OpenCity's scheduler. The missing ablation also makes it impossible to tell how much of the speedup comes from each component (IO multiplexing vs. connection pooling vs. CPU offloading vs. group-and-distill token reduction).

- **The efficiency-faithfulness claim is tested only at the micro level (individual location choice) and never connected to macro urban dynamics.** The paper's core thesis is that OpenCity achieves massive speedup *while preserving simulation faithfulness*. However, these two claims are evaluated in entirely separate experiments. Table 2 tests whether group-and-distill preserves individual location choices (100 agents × 100 trials in isolation). Table 3 evaluates urban dynamics quality (OD matrix, radius of gyration, segregation index), but *does not compare optimized vs. unoptimized runs*. Group-and-distill rewrites agent prompts and batches multiple agents together; any distributional shift in individual decisions could compound at the aggregate level in OD matrices or segregation indices. The paper never shows that running 1,000 agents through the full OpenCity pipeline produces the same aggregate urban dynamics as the sequential unoptimized baseline. This gap leaves the central efficiency-faithfulness tradeoff claim unverified for the metrics that constitute the benchmark.

### Minor

- **Missing RMSE values for New York and San Francisco (Table 3) are unexplained.** The dataset section notes that NY and SF use SafeGraph "aggregated population flow data," while other cities use individual check-in trajectories. This likely explains the missing individual-trajectory RMSE values, but the paper never explicitly states this, leaving two major cities incompletely benchmarked. The implied incompatibility of the SafeGraph data with individual-trajectory metrics should be stated as a limitation.

- **IPL hyperparameters M and T are not analyzed for sensitivity.** Section 4.2 introduces M (initial prototype learning size) and T (membership threshold) as key parameters of the IPL algorithm but reports no sensitivity analysis. The number of groups generated in practice is never reported, nor is what happens when an agent fails to reach threshold T (singleton handling). These affect both faithfulness and efficiency, and without sensitivity analysis, reproducibility and generalizability of the approach are unclear.

- **Data heterogeneity across cities is unaddressed.** Beijing uses social network check-ins (from Shao et al. 2024), NY/SF use SafeGraph aggregated flow data, and London/Paris/Sydney use Foursquare check-ins. These have fundamentally different sampling biases and temporal coverage. The paper does not discuss preprocessing harmonization or its implications for cross-city comparison validity.

### Trivial

- **Abstract numbers inconsistent with body:** Abstract states "600-fold acceleration," "70% reduction in LLM requests," and "50% reduction in token usage." Body reports 635×, 73.7%, and 45.5%. These are rounding/approximation artifacts but should be consistent.

---

## Nice-to-Haves

- Ablation of each system component's speedup contribution (scheduler-only vs. scheduler + connection pooling vs. full pipeline) would clarify where the gains actually come from and strengthen the engineering claim.
- Variance/error bars in Table 3 would help assess whether differences between GenerativeAgent and EPR are statistically meaningful, given LLM stochasticity and random initialization.
- The case study in Section 6 draws a causal conclusion ("differences between regions are the main cause of segregation as opposed to segregation by choice of action") from a single counterfactual in two cities with no uncertainty quantification. This could be framed more carefully as an illustrative demonstration rather than a causal finding.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Table 3 selectively frames a mixed result as LLM superiority" (Harsh Critic Issue 3):** Verified against Table 3 and found to be incorrect. On RMSE (4/4 cities where data exists), generative agents outperform EPR. On SMSE (4/6 cities), generative agents win or tie. On ODMSE, the split is roughly 3–3. The overall claim "comparable to or better than EPR" is reasonably well-supported by the data. Removed as factually incorrect reviewer claim.

- **"Equation 1 is garbled / formalism is thin":** Parser artifact; the equation is presented in the paper and its intent is clear from context. Removed per formatting artifact rule.

- **"Group-and-distill algorithm details in appendix":** Parser strips appendix sections; detailed algorithm steps (Fig. A1) presumably exist there. Removed per the missing appendix rule.

- **"GPT-4o tested on only NY and Paris — two conveniently strong-performing cities":** The paper provides a rationale (testing GPT-4o because 4o-mini showed discrepancies, and the two cities span two data-source types). While the selection is unexplained, the paper is using this to compare models rather than to cherry-pick cities, and the results are broadly consistent. Removed as a minor inferential leap without grounding.

- **"Figure 5 is anecdotal / product demo":** True, but Section 6 is explicitly a case study demonstrating capabilities. Qualitative examples are standard in this context; the issue is not a scientific flaw. Removed as scope creep.

---

## Novel Insights

The most genuinely novel technical element is the in-context prototype learning (IPL) algorithm for semantic agent clustering, which enables prompt distillation without eliminating agent individuality. This is a meaningful departure from prior approaches that either fully reuse responses (sacrificing agent independence) or treat each agent in isolation (no token savings). Combining this with OS-level I/O multiplexing tailored to the API-call-dominated workload of LLM agent simulation — rather than inference-time GPU efficiency — addresses a bottleneck that existing LLM deployment systems (vLLM, FlashAttention, etc.) do not target. The three-level evaluation framework (individual physical / group physical / social domain) is also a thoughtful contribution that other urban simulation benchmarks lack.

---

## Suggestions

1. **Add a concurrent/async baseline:** Run the same simulation using Python `asyncio` or a thread pool and report the speedup over *that* baseline, not just sequential. This is the single most important missing experiment.
2. **Connect efficiency to urban dynamics:** Run 1,000-agent simulations with and without group-and-distill and report RMSE, ODMSE, SMSE for both. This directly tests the efficiency-faithfulness claim for the metrics that constitute the benchmark.
3. **Ablate system components:** Report speedup for scheduler-only, scheduler + connection pool, and full pipeline to attribute gains to each component.
4. **State SafeGraph data limitation explicitly:** Explain why NY and SF have no RMSE values in Table 3 and discuss the implication for cross-city comparability.
5. **Report IPL sensitivity:** Vary M and T across a reasonable range and report impact on faithfulness (JSD, T1) and group count to establish robustness.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Score | Comparison to OpenCity |
|------|-----------|------------------------|
| `/home/wg25r/review_agent/human_reviews/Yqk7EyT52H.md` (MarS) | 7.0 | Accept (Poster) — financial market simulation platform with strong empirical validation and clear downstream tasks; OpenCity is weaker due to the unfair baseline and missing macro faithfulness test |
| `/home/wg25r/review_agent/human_reviews/qWLgJCl1Y6.md` | 4.8 | Withdrawn — LLM agent simulation for dynamic graphs, rejected for weak validation of "dynamic" claim and insufficient comparison; OpenCity has similar validation gaps but broader coverage and two distinct technical contributions |
| `/home/wg25r/review_agent/human_reviews/REprQnylmC.md` (LCSim) | 4.75 | Withdrawn — traffic simulator with realism validation gaps; shares OpenCity's pattern of being a useful system paper with insufficient validation of the key faithfulness claim |
| `/home/wg25r/review_agent/human_reviews/8LBS1nixTJ.md` | 5.5 | Rejected — graph reordering speedup paper with reviewer concerns about naive baselines; directly analogous to the sequential-baseline concern here |
| `/home/wg25r/review_agent/human_reviews/MGceYYNvXp.md` | 1.5 | Rejected — a benchmark platform with no novel technical contribution; clearly below OpenCity in novelty and depth |
| `/home/wg25r/review_agent/human_reviews/FaL6aTuXod.md` | 1.5 | Withdrawn — poor experimental validation; OpenCity is clearly better |

**Reasoning:** OpenCity sits in the medium band alongside qWLgJCl1Y6 (4.8) and LCSim (4.75). It is clearly above the low anchors (1.5) which had no meaningful technical contributions. Its two major weaknesses — the sequential-only speedup baseline and the missing macro-level faithfulness test — are the same pattern that drove papers like qWLgJCl1Y6 and LCSim to rejection. Compared to MarS (7.0), OpenCity lacks the thorough empirical grounding of the downstream tasks and the tight connection between system design and claimed outcomes. The multi-city benchmark and dual technical contribution (scheduler + group-and-distill) place it slightly above the raw avg of the medium anchors. Final score: **5.0**, Reject.

The paper has real contributions and is not without value, but the two major experimental gaps prevent the core efficiency-faithfulness claim from being properly supported. These gaps are addressable in principle but cannot be fixed in a rebuttal alone, making acceptance premature.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>