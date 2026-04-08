=== CALIBRATION EXAMPLE 46 ===

# Final Consolidated Review
## Summary

This paper introduces **real-time reasoning** as a new problem formulation for LLM-based agents operating in environments that evolve independently of the agent's computation. The authors build **Real-Time Reasoning Gym**, featuring three games (Freeway, Snake, Overcooked) with independently controllable cognitive load and time pressure, using token count as a hardware-agnostic time proxy. They propose **AgileThinker**, a dual-thread architecture where a planning thread (using a reasoning model) performs extended deliberation while a reactive thread (using a standard model) issues timely actions informed by the planning thread's partial output. Experiments demonstrate AgileThinker consistently outperforms single-paradigm agents, with advantages validated in wall-clock time.

## Strengths

- **Novel problem formalization for LLM agents**: The paper identifies and formalizes a real gap—existing LLM agent evaluations assume static environments that pause during reasoning (Section 2, Figure 2). The formulation where the environment steps at a fixed rate regardless of agent computation, with default actions on timeout, is a clean and important abstraction that the LLM agent community has largely overlooked.

- **Architectural innovation with information flow**: AgileThinker's key design—allowing the reactive thread to access *partial* reasoning traces from the concurrently running planning thread (Section 3, Figure 4)—is a genuine advance over prior dual-system approaches (Zhang et al., 2025; Liu et al., 2024) that either cascade systems or run them independently. This streaming partial-context mechanism is both principled and practical.

- **Well-designed evaluation framework with controlled variables**: Real-Time Reasoning Gym independently manipulates cognitive load (3 levels per game) and time pressure (4 levels), enabling systematic study of how each dimension affects agent performance. The token-count-as-time abstraction is validated against wall-clock inference (R²=0.9986, Section 6), and statistical significance tests (Appendix C.2) support the main claims.

## Weaknesses

### Major:

- **Model confound between architecture and model capability**: The primary experimental comparison uses AgileThinker (V3 reactive + R1 planning) against Reactive (V3 only) and Planning (R1 only). This conflates the architectural contribution of dual threads with the capability difference between V3 and R1. The paper partially addresses this with DeepSeek-V3.2 experiments (Table 8), where thinking-on and thinking-off modes of the *same* model are used, and AgileThinker still outperforms both baselines. However, the main narrative and figures (Figure 5) rely on the V3/R1 split, and the V3.2 results are relegated to the appendix. **Why it matters**: Without a clean ablation isolating architecture from model capability in the main results, it remains unclear how much of AgileThinker's advantage stems from the dual-thread design versus simply combining two models of different strengths. The V3.2 results should be promoted to the main paper to strengthen the architectural claim.

- **Missing ablation on partial trace sharing**: The paper's core architectural claim is that the reactive thread benefits from accessing *partial* reasoning traces from the planning thread (rather than complete traces or no traces). Yet no experiment directly compares: (a) AgileThinker with partial trace sharing, (b) dual threads with *no* information sharing (independent parallel agents), and (c) dual threads where the reactive thread only sees the planning thread's *final* output after completion. The Gemini experiments in C.3 approximate condition (c) but use a different model and only one game. **Why it matters**: Without this ablation, it is unclear whether the performance gain comes from the partial trace streaming mechanism or merely from running two models in parallel.

### Minor:

- **Planning agent baseline may be weaker than necessary**: The Planning Agent as described commits to a generated plan and executes it without re-evaluating, making it vulnerable to environmental changes. A receding-horizon or replanning-every-K-steps baseline (standard in model-predictive control) is not tested. While Code-as-Policy provides some adaptivity, it fails for non-algorithmic tasks (Appendix C.4). A replanning baseline would more fairly represent the planning paradigm. The paper partially addresses this through Code-as-Policy, but a periodic replanning variant of the direct planning agent would strengthen the comparison.

- **Dependency on transparent reasoning traces**: AgileThinker requires access to partial reasoning traces from the planning model, restricting deployment to models that expose these (currently only open-source models like DeepSeek). The authors acknowledge this in Section 9 and attempt a workaround for Gemini (C.3), but the workaround only allows the reactive thread to reference the planning thread's *final* output—losing the key partial-trace advantage. This limits the architectural generality of the proposed method for proprietary model APIs.

- **Compute overhead not analyzed or listed as a limitation**: Running two LLM threads in parallel approximately doubles inference cost. Section C.5 examines concurrent (vs. parallel) execution showing modest degradation, but no analysis of score-per-token or score-per-dollar is provided. This practical cost is also absent from the Limitations section (Section 9), despite being a significant deployment consideration.

- **Freeway is largely solvable by algorithmic search**: As the paper's own analysis in C.4 shows, Code-as-Policy achieves near-perfect Freeway scores (0.94 at 32k) via BFS, meaning this game primarily tests code generation latency rather than reasoning. The real value of AgileThinker is concentrated in Overcooked (0.89 vs 0.58 Code-as-Policy at 32k) and Snake, where contextual reasoning cannot be compressed into code. The paper should more clearly frame where AgileThinker's advantages are strongest and why.

### Trivial:

- The R²=0.9986 correlation for token-to-wall-clock time (Section 6) is reported at the episode level; step-level variance in TPOT is not analyzed. Since AgileThinker's coordination operates at the step level, step-level timing variance would be more informative for assessing synchronization robustness. However, since the main evaluation uses token-count simulation (not wall-clock time), this is a minor validation concern rather than a threat to the experimental claims.

## Nice-to-Haves

- Expand evaluation to at least one non-game task (e.g., real-time information gathering or tool-use scenario) to demonstrate that the real-time reasoning framework generalizes beyond arcade games.
- Provide a compute-performance Pareto analysis (score vs. total tokens generated) to help practitioners assess whether the dual-model cost is justified for their use case.
- Analyze trace utilization rate: how often does the reactive thread actually reference partial planning output versus acting independently? This would clarify the mechanism's active contribution.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Safety implications of default actions**: The critic raised concerns about default actions (e.g., "move forward") being catastrophic in real-world scenarios like autonomous driving. Removed because this is scope creep—the paper evaluates game environments with clearly defined, appropriate default actions, and does not claim to address safety-critical deployment.

- **RQ3 is not a proper research question**: The critic suggested RQ3 (matching simulation to wall-clock time) is merely a validation check. Removed as a presentation/style nitpick; the authors are free to frame their contributions as they wish.

- **"Thinking/non-thinking" terminology**: The critic suggested standardizing to "reasoning-enhanced" vs. "standard decoder-only." Removed as a terminology nitpick; the paper defines its terms clearly in Footnote 1.

- **Missing broader impact / societal harm discussion**: The critic asked for discussion of failure modes in safety-critical applications. Removed as scope creep for a systems/benchmark paper.

- **Overclaiming "first" in abstract**: The critic argued the "first environment for language agents to reason in dynamic environments" overclaims given prior RL work. The paper clearly scopes this to *language agents* in Section 7, and the token-as-time formulation for LLM reasoning is indeed novel to this space. Removed as the claim is properly scoped.

- **TPOT non-constancy due to KV-cache and batching**: The critic questioned whether TPOT varies in production. The paper uses token-count simulation precisely to avoid this issue—wall-clock experiments are only a validation. Removed because the simulation design already accounts for this concern.

- **Missing related works / comparisons with specific dual-system papers**: The retrieved reviews suggested comparing against Zhang et al. (2025), Liu et al. (2024) empirically. While this would strengthen the paper, I cannot confirm the exact nature of these methods from the paper alone, and the paper does discuss them in Related Work and differentiates AgileThinker's partial-trace mechanism. Moved to Nice-to-Have.

## Novel Insights

The paper reveals a fundamental architectural insight for LLM agents: **reasoning depth and reaction speed are not merely a resource trade-off but an architectural concurrency problem**. The key finding is not just that "sometimes you need fast responses and sometimes deep planning"—it is that *both must operate simultaneously with information flow between them*. Prior dual-system approaches treat System 1 and System 2 as stages or independent processes; AgileThinker's innovation of streaming partial reasoning traces from an ongoing planning computation into a reactive decision-maker represents a qualitatively different design pattern. The empirical results further show that this advantage grows precisely when it matters most—under high cognitive load and high time pressure—suggesting that single-paradigm approaches hit a capability ceiling that only concurrent dual-processing can overcome.

## Suggestions

- Promote the DeepSeek-V3.2 same-model experiments (currently in Appendix Table 8) to the main paper, as they provide the cleanest ablation isolating architectural contribution from model capability confound.
- Add a direct ablation comparing AgileThinker with partial trace sharing vs. dual independent threads vs. dual threads with only final-output sharing, all using the same model family, to isolate the contribution of the streaming partial-context mechanism.
- Include a brief cost-efficiency analysis (e.g., score per total token budget) to address the practical deployment question of whether the 2× inference cost is justified.

---

**Quality Assessment:**

- **Novelty**: High. The problem formulation (real-time reasoning for LLM agents with environment stepping independently of computation) and the partial-trace streaming architecture are both novel contributions to the LLM agent space.
- **Technical Soundness**: Moderate-to-Good. The architecture is well-designed and the evaluation framework is rigorous, but the model confound in the main experiments and missing ablation on partial trace sharing weaken the attribution of gains to the proposed mechanism.
- **Empirical Support**: Good overall. Extensive experiments across conditions with statistical significance, wall-clock validation, and multiple model families. However, the critical ablation isolating the architectural contribution is incomplete.
- **Significance**: High. Real-time reasoning is an increasingly important problem for deploying LLM agents in practice, and this work establishes both a benchmark and an architectural pattern that will likely influence future research.
- **Clarity**: Good. The paper is well-structured with clear problem motivation, formal definitions, and illustrative case studies. Some architectural details (conflict resolution between threads) could be more explicit.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 4.0]
Average score: 6.5
Binary outcome: Accept
