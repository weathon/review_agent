=== CALIBRATION EXAMPLE 90 ===

# Harsh Critic Review
Now I have read the full paper. Let me compose my review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title, reconstructed from the PDF as "Speculative Actions: A Lossless Framework for Faster Agentic Systems," accurately reflects the contribution. The abstract is clear and the claims are grounded: the paper promises a framework "inspired by speculative decoding" applied to API-level actions, up to 55% next-action prediction accuracy, and "substantial latency reductions." However, the abstract's mention of "up to 20% end-to-end speedup" is modest and should not be buried—it is arguably the most honest headline performance number, and it directly caps reader expectations against the more alluring "up to 55% accuracy" figure. The abstract somewhat elides the important caveat that e-commerce and HotpotQA experiments measure *prediction accuracy* only, not actual end-to-end latency.

---

### Introduction & Motivation

The motivation is compelling and clearly articulated. Table 1, citing estimated runtimes for current AI agents, is effective framing, though these are hand-picked estimates rather than measured numbers. The Actor/Speculator abstraction is intuitive and aligns well with MCP-style thinking.

The central claim—that "losslessness" is achievable—is prominently featured but immediately complicated by the acknowledgment that it requires idempotent, reversible, or sandboxed speculative side effects. This caveat is real and significant: in the majority of commercially deployed agentic settings (database writes, order placements, email sending, code execution with side effects), Assumption 2 is violated. The introduction is somewhat optimistic about how natural and inexpensive lossless enforcement is in practice.

---

### Framework (Section 2)

**MDP formulation and Algorithm 1.** The formulation is clean and the abstraction of every agent action as an API call is both principled and practically motivated. Algorithm 1 is straightforward to understand. The cache-and-pre-launch mechanism is the correct generalization of speculative decoding to this level of granularity.

**Assumption 1 (Speculation accuracy).** The paper requires *p > 0*—a very weak requirement—but the useful regime is clearly much narrower. The claim that "API responses are typically predictable" is asserted rather than systematically motivated.

**Assumption 2 (Concurrent, reversible pre-launch).** This is the crux of applicability, and the paper does not engage with it rigorously enough. The brief enumeration of "forking, snapshot restoration, or roll-forward repair" is insufficient. In practice:
- What is the overhead of running k-way parallel speculative branches subject to rate limits from API providers?
- How does the framework interact with stateful APIs that don't cleanly support rollback (e.g., file system mutations, database transactions outside sandboxes)?
- What happens when k speculative branches collectively consume quota?

These are real engineering concerns that the paper essentially defers without analysis. The "safety envelopes" discussion is too qualitative for a systems paper.

**Proposition 1.** The exponential latency assumption (both Speculator at Exp(α) and Actor at Exp(β)) is analytically convenient but poorly motivated for real LLM APIs, which exhibit heavy-tailed latency distributions driven by output token length and batching behavior. The key result—that the speedup ratio is capped at `p(k)/(1+p(k)) · α/(α+β)` as T→∞, implying a hard 50% ceiling—is theoretically important and should be more prominently highlighted. The proof in Appendix A is correct at a high level.

A subtlety: the proof treats speculation accuracy p as i.i.d. across steps, but in reality the same Speculator model will be systematically wrong on the same *types* of actions (e.g., always mispredicting when the opponent plays an unusual chess move). Ignoring this correlation overstates expected speedup in regimes where sequential errors cluster.

---

### Chess Environment (Section 3.1)

Using the same model (GPT-5) with different reasoning efforts as Actor and Speculator is a pragmatic choice, but raises a question: is the Speculator acting as a "fast weak model" or just as a "lazy version of the Actor"? This is a meaningful distinction for generalizability.

**The main weaknesses are statistical.** Only 5 runs of 30 steps are reported. Chess has enormous position-level variance: positions differ wildly in branching factor and required reasoning depth, so the measured time savings depend heavily on which positions are encountered. The paper acknowledges this variance ("Even with correct predictions, speedups vary"), but does not report confidence intervals or run sufficient trials to characterize the distribution of speedup. With 5 runs, neither the 19.5% mean time savings nor the 54.7% prediction accuracy is statistically meaningful.

Additionally, the paper reports "19.5% time saved" with k=3, but Proposition 1 with p≈0.547, exponential latencies with α≫β would predict a higher ratio. The gap between theory and experiment is not explained.

---

### E-Commerce Environment (Section 3.2)

This section measures *API call prediction accuracy* (22–38%) but does **not report actual latency reduction**. The connection between prediction accuracy and latency is only stated qualitatively: "higher prediction accuracy translates into greater time savings." No actual wall-clock latency measurement is provided for this domain, despite it being framed as a "real-world setting where latency significantly impacts user experience."

The use of "average user typing time of ~30 seconds" to argue that prediction is fast enough is based on a rough calculation (40 wpm) rather than any user study. This threshold-based argument is not rigorous: user typing speed varies by 2–5× across users, and in some cases the agent turn is triggered by non-typing events.

The multi-model Speculator ensemble is briefly introduced and shown to outperform single-model speculation—this is a genuine interesting finding—but the analysis does not break down *which* models contribute most and whether the latency of the ensemble (running multiple models in parallel) still fits within the speculation window.

---

### HotpotQA Environment (Section 3.3)

This section is the weakest in the paper. It is brief (~half a page) and, like e-commerce, reports only *prediction accuracy* (up to 46% top-3), not actual latency improvement. Given that the stated bottleneck is "information retrieval latency," the natural evaluation would be end-to-end wall-clock time saved—this is absent.

The finding that "stronger models often yield *lower* accuracy" under strict matching (because they phrase queries more specifically) is intellectually interesting and somewhat undermines the implicit assumption that a better Speculator always helps. This phenomenon deserves more analysis: if the Speculator generates more semantically precise—but lexically different—queries, the strict-match evaluation may severely undercount true utility (a near-synonym search still warms the cache usefully), while also possibly overcounting it in edge cases where different queries return meaningfully different Wikipedia passages. The choice of strict matching as the primary metric needs stronger justification.

---

### OS Hyperparameter Tuning (Section 4)

This is the most compelling section and the one where actual latency reduction is measured. The result—convergence in ~13s vs. ~200s for Actor-only, at 13× lower cost—is striking.

**Methodological concerns:**
1. **Single-run results.** Figures 5 and 7 appear to show a single trajectory per method. LLM-driven optimization is stochastic; without multiple runs, it is unclear whether the result is representative or fortuitous.
2. **Baseline fairness.** The "Actor-only" baseline uses a 10–15s deliberation cycle by design, which is artificially slow. The more meaningful comparison is Speculator-only (which stabilizes in 20s at 0.55ms granularity) vs. Actor+Speculator (13s at 0.2ms). This comparison is present but the framing emphasizes the less fair Actor-only contrast.
3. **Context strategy asymmetry.** The Actor-only and Speculator-only baselines use full history, while the combined system uses compressed summaries for the Actor (Section B.3.1). This means the Actor in the combined system gets *different* (and arguably better-curated) inputs than the Actor-only baseline. The speedup may partially reflect better prompting rather than the speculative architecture.
4. **This is a "lossy" extension.** The paper correctly labels this as such, but the OS tuning results are used to support the broader narrative about speculative actions as a "lossless framework." The OS setting, where the Actor just overwrites the Speculator's output, is architecturally different from Sections 3.1–3.3 and should be more carefully distinguished.

---

### Cost–Latency Tradeoff (Section 5)

This is the paper's theoretical strength. Theorems 3, 4, and 6 are well-motivated and the proofs in Appendices A and C are generally correct.

**Theorem 3 (Confidence-aware selective speculation).** The dynamic programming derivation is clean and the stationary-case reduction to a greedy threshold rule is elegant. However, the empirical implementation (Section 5.2) uses a 50% fixed threshold that is not derived from the theory—the theoretically optimal threshold should depend on the continuation value ∆* and cost c, which the paper doesn't actually compute or estimate from data. The claim that this "implements a simplified threshold rule consistent with the structure suggested by Theorem 3" is technically true but undersells the gap between theory and practice.

**Theorem 6 (Depth-focused speculation).** This uses *deterministic* latency (constants a and b) rather than the exponential distributions used in Proposition 1 and Theorem 4, making the breadth vs. depth comparison analytically inconsistent. The claim that depth speculation raises the speedup ceiling from 1/2 to 1 (in the limit p→1, b/a→0) is mathematically valid but relies on dramatically different modeling assumptions than the breadth analysis. There is **no empirical demonstration** of depth-focused speculation anywhere in the paper—Section 5.3 is purely theoretical.

**Empirical cost-latency curve (Figure 6).** The curve shows a reasonable Pareto frontier with the confidence-based policy achieving a good tradeoff, but Figure 6 is described without sufficient detail: What models generate these numbers? What are the axes' units (cost in tokens? dollars?)? How many runs?

---

### Writing & Clarity

The paper reads clearly in most places. The organization from framework → environments → theory is logical. Section 5 (cost-latency analysis) is somewhat dense but comprehensible.

The treatment of the four environments is imbalanced: chess and OS tuning get detailed results while e-commerce and HotpotQA provide only prediction accuracy metrics. A reader interested in the practical applicability of speculative actions comes away with limited empirical evidence for the two domains most relevant to commercial agentic systems (e-commerce and tool-augmented QA).

---

### Limitations & Broader Impact

The paper includes a brief "side effects and safety" paragraph in Section 2 but lacks a formal Limitations section. Several failure modes and constraints are not adequately addressed:

1. **Rate limit violations.** Launching k parallel speculative calls multiplies API request volume. For providers with strict rate limits or quota-based billing, this could be prohibited or prohibitively expensive in ways not captured by the token-cost model.
2. **Correlated speculation errors.** The i.i.d. accuracy assumption used throughout the theory will be violated whenever the Speculator systematically fails on particular state types (e.g., rare API calls, unusual game positions).
3. **Rollback cost.** The paper treats rollback as free. In practice, compensating actions (refund/replace for mis-speculated e-commerce calls) may themselves incur latency and cost that partially or fully erodes the speedup.
4. **Applicability scope.** The paper doesn't derive principled conditions under which speculative actions are worth deploying. When is p likely to be high enough? What environments are structurally conducive?

---

### Overall Assessment

"Speculative Actions" presents a genuinely useful and well-motivated idea: adapting the speculate-verify paradigm to the API-call level in agentic systems. The theoretical framework is clean, the formalism is appropriate, and the cost-latency analysis (Theorems 3, 4, 6) represents a real contribution. However, the paper falls short of ICLR standards in its current form for three principal reasons. First, the empirical evaluation is too thin: two of four environments report only prediction accuracy without measuring actual latency reduction, sample sizes are small (5 chess runs, single OS traces), and no statistical significance is reported. Second, Assumption 2 (lossless reversibility) is central to the "lossless" brand of the paper but is handled too casually—the conditions under which it holds, and the cost when it doesn't, are never rigorously characterized. Third, the depth-focused speculation strategy (Section 5.3) is developed entirely in theory with no empirical grounding, and the breadth vs. depth theoretical comparison is undercut by modeling inconsistencies (exponential vs. deterministic latencies). With substantially stronger empirical validation—actual end-to-end latency measurements across all domains, more runs with confidence intervals, and a credible analysis of rollback overhead—the contribution could easily reach the bar. In its current state, it reads as a compelling workshop paper with promising but insufficient evidence for a main conference venue.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces **Speculative Actions**, a framework for accelerating general agentic systems by parallelizing API calls and environmental interactions using a fast "Speculator" model to predict the next actions of a slow "Actor" authority. The method mirrors speculative execution in microprocessors and speculative decoding in LLMs, allowing agents to stage environment interactions (tool calls, reasoning steps) while validating predictions only when the Actor completes. Experimental results across chess, e-commerce, web search, and OS tuning show up to 55% next-action prediction accuracy and end-to-end latency reductions of 20%, supported by closed-form theoretical analysis of the cost-latency tradeoff.

### Strengths
1.  **Novel Application to Agentic Workflows:** The paper successfully generalizes speculative execution from token generation to high-level agent actions (API calls, tool usage, human-in-the-loop). This abstraction ("Action as API") unifies latency optimization across diverse agent architectures (TextArena, MCP, etc.).
2.  **Theoretical Rigor:** The work provides strong mathematical grounding for the proposed methods. Proposition 1 and Theorem 4 offer closed-form expressions for expected latency and cost relative to speculation width ($k$), enabling principled tuning rather than heuristics. Theorem 3 extends this to confidence-aware selective speculation.
3.  **Diverse and Representative Evaluation:** The evaluation covers distinct latency bottlenecks: reasoning latency (Chess), tool/API round-trip latency (E-commerce, HotpotQA), and system state tuning latency (OS). This demonstrates the framework's extensibility beyond simple LLM inference.
4.  **Practical Impact:** The OS hyperparameter tuning experiment (lossy extension) shows tangible system improvements (reducing p95 latency by ~30% and accelerating convergence), highlighting a valuable use case where strict losslessness is less critical than responsiveness.

### Weaknesses
1.  **Strong Assumptions in Theoretical Analysis:** Proposition 1 relies on assumptions of independent exponential latency distributions and step-wise independence of speculation accuracy ($p$). In complex reasoning tasks, state transitions are highly correlated; a failure in one step (e.g., misinterpreting a chess board) compounds errors, potentially invalidating the probabilistic model used for cost/latency prediction.
2.  **Losslessness Constraints and Safety Envelopes:** The claim of a "lossless framework" heavily depends on the assumption that speculative calls are idempotent, reversible, or sandboxed (Assumption 2). While feasible for tools like search or game moves, many real-world agent interactions (e.g., purchasing, state modification in databases) are not easily rollbackable, limiting the immediate applicability of the "lossless" guarantee without significant engineering overhead.
3.  **Marginal Speedup Gains:** While 20% speedup is promising, the theoretical speedup upper bound of 50% (when $p=1$) is modest compared to some speculative decoding gains in pure LLM inference (often 2-3x). This suggests the bottleneck might shift to the Actor's generation time rather than waiting for API calls in many scenarios, potentially underestimating the value of the method in non-parallelizable environments.
4.  **Ambiguity in Lossy vs. Lossless Presentation:** The paper spends significant space on a "lossy OS extension," which contrasts with the core "lossless" contribution. While acknowledged as an extension, it risks diluting the main theoretical claims and blurs the boundary between system-level speculative execution and the proposed agentic framework.

### Novelty & Significance
**Novelty:** High. While speculative decoding is well-trodden, extending the paradigm to *agentic decision loops* (specifically treating LLM tool calls and environment transitions as speculatable actions) is a distinct and timely contribution. It bridges the gap between model inference speedup and system-level agent orchestration, addressing a known bottleneck in the ICLR ML-for-Systems space.

**Significance:** Significant for the scaling of LLM agents. As agents move toward complex, multi-step workflows (e.g., autonomous coding, research, trading), latency becomes a primary barrier to deployment. This work offers a generic, model-agnostic abstraction for reducing wait times in sequential decision processes, applicable to both private model deployments and external API dependencies.

### Suggestions for Improvement
1.  **Refine Theoretical Assumptions:** Discuss how the independence assumption ($p$ varying independently per step) holds or fails in dependent task chains (like Chess). Consider adding a worst-case analysis or sensitivity analysis where error propagation occurs.
2.  **Clarify Safety and Rollback Costs:** Expand the "safety envelopes" discussion. Quantify the overhead of rollback mechanisms (e.g., state snapshotting costs) in environments where they are non-trivial (e.g., web form submissions) to provide a more realistic cost model.
3.  **Separate Lossy and Lossless Contributions:** Consider splitting the OS tuning results into an appendix or a distinct subsection to preserve the integrity of the "lossless" main claim. Alternatively, explicitly frame OS tuning as "speculative adaptation" vs "speculative inference."
4.  **Compare to Baseline Speculative Planning:** The paper mentions Hua et al. (2024) and Guan et al. (2025) regarding speculative planning. A more direct quantitative comparison (even if on a subset of tasks) against these specific baselines would better position the work's specific advantage (breadth vs. depth focus).
5.  **Address OCR Artifacts in Final Version:** Although formatting artifacts should not be penalized, authors should ensure the final LaTeX source renders the equations (specifically Theorem 3 and Proposition 1) cleanly, as the current text extraction shows garbled mathematical structures that could confuse readers.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Measure Rollback Overhead:** Quantify the latency cost of state snapshotting, comparison, and restoration required for "lossless" claims; current metrics ignore this system-level overhead which may negate speedups.
2. **Include Standard Baselines:** Compare against standard API caching and batched request strategies to isolate the specific benefit of *speculative* actions versus simple optimization techniques.
3. **Report Negative Speedup Cases:** Explicitly report performance when speculation accuracy drops below the theoretical break-even point; current results only highlight successful speedups, masking potential degradation.
4. **Verify Model Identities:** Clarify actual model versions used (e.g., "GPT-5" is currently unavailable); without reproducible model specs, empirical claims cannot be validated by the community.

### Deeper Analysis Needed (top 3-5 only)
1. **Validate Independence Assumption:** Analyze the correlation of prediction errors across steps; the theory assumes independence (Proposition 1), but agent trajectories are inherently state-dependent, potentially invalidating theoretical bounds.
2. **Explain OS Cost Contradiction:** Clarify how OS tuning reduces both cost and latency simultaneously when Section 5 predicts a tradeoff; determine if this is due to convergence speed or inconsistent metric definitions.
3. **Define Safety Envelopes:** Provide a formal analysis of which API calls are truly reversible; many real-world tools (payments, emails) lack idempotency, limiting the applicability of the "lossless" claim.

### Visualizations & Case Studies
1. **Execution Timelines:** Show Gantt charts of Actor vs. Speculator threads during hits and misses to visualize exactly where latency is saved or wasted during coordination.
2. **Cost Breakdown:** Stack bar charts separating Speculator tokens, Actor tokens, and Rollback overhead to verify the claimed cost efficiency against the theoretical model.
3. **Accuracy vs. Speedup Curve:** Plot empirical speedup against prediction accuracy to identify the practical break-even threshold compared to the theoretical bound derived in Proposition 1.

### Obvious Next Steps
1. **Rate Limit Stress Test:** Evaluate behavior under strict API rate limits where parallel speculative calls might trigger throttling rather than speedup.
2. **Complex State Rollback:** Implement and measure rollback in non-trivial environments (e.g., SQL databases) to prove safety claims beyond simple variable overwrites.
3. **Dynamic Confidence Tuning:** Replace the fixed 50% confidence threshold with an online learning mechanism to adapt to varying task difficulty and model drift.

# Final Consolidated Review
## Summary

This paper introduces **Speculative Actions**, a framework that applies the speculate-verify paradigm from CPU architecture and LLM decoding to the level of API calls in agentic systems. A fast "Speculator" predicts likely next actions while a slow "Actor" executes authoritatively; correct predictions enable parallel execution and latency reduction. The authors evaluate across four environments (chess, e-commerce dialogue, multi-hop web search, and OS tuning) and provide theoretical analysis of the cost-latency tradeoff with closed-form expressions for optimal speculation breadth and confidence-aware selection.

## Strengths

- **Novel abstraction ("Action as API"):** The paper unifies diverse agent operations—LLM calls, tool invocations, MCP requests, even human responses—as API calls subject to speculation. This generalizes speculative decoding beyond tokens to the decision-loop level, addressing a real bottleneck in deployed agentic systems where inter-action latency often dominates inference time. The Actor-Speculator formulation cleanly separates concerns and aligns with emerging MCP-style architectures.

- **Theoretical contributions with actionable insights:** Theorems 3, 4, and 6 provide closed-form expressions for expected time and cost savings under breadth- and depth-focused speculation strategies. Theorem 3's reduction to a greedy threshold rule (`∆* δq(m; p) ≥ c`) yields a practical O(k) decision procedure per step. The theoretical upper bound of 50% latency reduction (Proposition 1, as T→∞ with p→1) properly sets expectations and characterizes fundamental limits.

- **Demonstrated end-to-end improvement in OS tuning:** Section 4 shows that the joint Actor-Speculator system converges in ~13s vs. ~200s for Actor-only, with 13× lower cost (0.17 cents vs. 2.18 cents). This is a compelling result where speculation improves both latency and cost by accelerating convergence, not just hiding latency—the Speculator's rapid exploration helps avoid pathological configurations that the Actor alone gets stuck in.

- **Multi-domain evaluation:** The four environments stress different latency sources (reasoning time, API round-trips, information retrieval, system response), demonstrating the framework's breadth. The 20% end-to-end speedup in chess with 55% prediction accuracy provides a concrete benchmark for speculation viability.

## Weaknesses

- **Inconsistent empirical rigor across environments:** Chess reports actual latency reduction (19.5% with k=3), OS tuning measures p95 latency, but e-commerce (Section 3.2) and HotpotQA (Section 3.3) report only *prediction accuracy* (22–38% and up to 46% top-3, respectively) without measuring actual end-to-end latency. This is a significant gap: prediction accuracy does not directly translate to latency savings (cache hits must occur during viable speculation windows). The two domains most relevant to commercial agentic systems lack the primary metric the framework promises to optimize.

- **Insufficient statistical validation:** The chess experiment reports only 5 runs of 30 steps. Chess positions vary dramatically in reasoning difficulty; without confidence intervals or more runs, the reported 19.5% time savings (±4.8% by my estimation from Figure 2's spread) lacks statistical significance. The OS tuning experiment (Figures 5, 7) appears to show single trajectories per method. LLM-driven optimization is inherently stochastic; single-run results risk being unrepresentative.

- **Confounding context strategy in OS experiments:** Appendix B.3.1 reveals that the Actor-only and Speculator-only baselines receive full unsummarized history, while the combined Actor-Speculator system uses compressed summaries for the Actor. This asymmetry means the combined system's Actor receives better-curated inputs than the Actor-only baseline. The observed speedup may partially reflect this prompting advantage rather than speculation alone.

- **Depth-focused speculation lacks empirical grounding:** Section 5.3 derives Theorem 6 for depth-focused speculation and claims it achieves a higher theoretical ceiling (up to 100% savings as p→1), but provides no experiments. This entire strategy remains unvalidated. Additionally, the modeling assumptions shift from exponential latencies (Proposition 1, Theorem 4) to deterministic latencies (Theorem 6) without justification, making the breadth vs. depth comparison analytically inconsistent.

- **Correlated errors undermine independence assumption:** Proposition 1 assumes speculation accuracy p is independent across steps, but real agent trajectories exhibit structured errors—the same Speculator will systematically fail on certain state types (e.g., unusual chess positions, rare API call patterns). Correlated failures would reduce effective accuracy below the theoretical predictions. The paper acknowledges this briefly but does not analyze sensitivity.

## Nice-to-Haves

- **Compare to speculative planning baselines:** The related work section cites Hua et al. (2024) and Guan et al. (2025), which study depth-focused speculation for planning. A direct comparison (even on a shared task subset) would clarify the practical difference between breadth-focused and depth-focused approaches.

- **Measure rollback overhead:** In environments where speculation fails, the paper assumes rollback is trivial. For broader applicability, quantifying state-snapshot and restoration costs in non-trivial settings (e.g., databases with transactions) would strengthen the "lossless" claim for realistic deployments.

- **Report negative cases:** The paper shows successful speedups but doesn't characterize regimes where speculation hurts (e.g., when p is below the break-even threshold, or when API rate limits punish parallel speculation).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claims about GPT-5 availability:** The spark finder asserts "GPT-5 is currently unavailable." Per the instructions, I cannot verify model availability external to the paper. The paper cites models with future access dates (e.g., "Accessed: 2025-09-24"); I assume these exist as stated. The reproducibility concern is valid but the specific model-unavailability claim is removed.

- **Demanding comparison to standard API caching:** This requests evaluation against techniques outside the paper's scope. The paper's contribution is speculative parallelization, not a comparison to existing caching strategies. It's scope creep to require comparison to all related optimizations.

- **Assertions that rollback overhead negates speedups:** The spark finder claims rollback costs are unmeasured and could erase gains. However, the environments tested (chess state is trivially reversible, web search has no side effects, OS tuning uses last-write-wins) genuinely have minimal rollback overhead by design. The criticism doesn't apply to the experiments actually run.

- **Generic "well-written" praise:** Removed as the instructions specify not including generic strengths.

## Novel Insights

The paper's most insightful observation—understated in the current text—is that speculation accuracy *decreases* with Speculator capability in some settings. In HotpotQA, "stronger models often yield lower accuracy" because they produce more specific, diverse queries that fail strict matching while weaker models produce simpler, more predictable outputs. This suggests a counterintuitive design principle: the optimal Speculator may not be a "weaker version of the Actor" but rather a model optimized for *predictability of action*, not action quality. The theoretical analysis of confidence-aware selection (Theorem 3) provides a principled foundation for this: the continuation value ∆* captures the expected future benefit of speculation, and branches should be launched only when their marginal hit probability exceeds a cost threshold. This insight—that speculation is about predicting what the Actor will do, not what the Actor should do—deserves more emphasis.

## Suggestions

- **Add end-to-end latency measurements for e-commerce and HotpotQA:** Report wall-clock time saved, not just prediction accuracy. If this requires additional experiments, they are essential for validating the core claim.

- **Provide confidence intervals or multiple runs:** For chess, report mean ± standard deviation across runs; for OS tuning, show multiple trajectories or at least report variance in convergence time.

- **Equalize context strategies in OS baselines:** Either give the Actor-only baseline the same compressed context, or add an ablation isolating the effect of context compression from speculation.

- **Validate independence assumption empirically:** Analyze the correlation of prediction errors across consecutive steps in chess and e-commerce. If errors cluster (e.g., a difficult position causes multiple mispredictions), report how this affects theoretical bounds.

- **Clarify the applicability scope:** Explicitly state the environments where Assumption 2 holds vs. where rollback costs are non-trivial. This helps practitioners assess deployment viability.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 10.0, 8.0]
Average score: 7.5
Binary outcome: Accept
