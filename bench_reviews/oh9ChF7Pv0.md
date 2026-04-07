## Summary
EGG-SR introduces a unified framework that integrates symbolic equivalence into symbolic regression via equality graphs (e-graphs). It accelerates learning in Monte Carlo Tree Search (MCTS), deep reinforcement learning (DRL), and large language models (LLMs) by pruning redundant exploration, aggregating rewards across equivalent expressions, and enriching feedback prompts. Theoretically, it offers a tighter regret bound for MCTS and reduces gradient variance for DRL; empirically, it improves accuracy across several benchmarks.

## Strengths
- **Unified integration across multiple paradigms**: The paper systematically adapts e-graphs to enhance MCTS (via equivalence-aware backpropagation), DRL (via reward aggregation), and LLMs (via prompt enrichment), demonstrating a cohesive framework for leveraging symbolic equivalence in diverse SR algorithms.
- **Theoretical grounding**: Theorems 3.1 and 3.2 provide formal justifications—showing a tighter regret bound under optimistic planning assumptions and proving variance reduction for the policy gradient estimator—with detailed proofs in the appendix.
- **Empirical improvements and efficiency**: Tables 1 and 2 show consistent reductions in normalized MSE across trigonometric and scientific benchmarks when EGG is added. Figures 4 and 5 confirm substantial memory savings and minimal runtime overhead, making the approach practical.

## Weaknesses
- **Theory-practice gap for MCTS**: Theorem 3.1 assumes an optimistic planning (OPD) framework, but the implemented MCTS uses the UCT heuristic. The paper does not empirically validate that the theoretical acceleration translates to standard UCT-based MCTS, leaving the practical benefit unclear.
- **Lack of statistical rigor**: Results are reported as point estimates (e.g., median NMSE) without confidence intervals, standard deviations, or multiple independent runs. This makes it difficult to assess the robustness and significance of the improvements.
- **Narrow benchmark scope for MCTS/DRL**: Experiments for MCTS and DRL are primarily on trigonometric functions; broader, standard benchmarks like SRBench are not included, limiting claims about generalizability to diverse symbolic regression problems.
- **Absence of comparison with GP baselines**: Prior work (e.g., de França & Kronberger) has integrated e-graphs into genetic programming for SR. A direct comparison with such methods would better demonstrate the unified framework's advantage over existing equivalence-aware approaches.
- **Unverified gradient variance reduction**: Theorem 3.2 claims variance reduction for EGG-DRL, but no direct measurement of gradient variance is provided—Figure 3(right) only shows variance of the objective estimate, not the gradient estimator itself.
- **No discussion of failure modes**: The paper does not analyze when EGG might fail or underperform, such as when no applicable rewrite rules exist, when domain restrictions cause numerical errors, or when e-graph construction overhead outweighs benefits.

## Nice-to-Haves
- Ablation study on the impact of different rewrite rule sets and the number of sampled equivalents (K) on performance and efficiency.
- Evaluation on comprehensive, widely-adopted benchmarks like SRBench to strengthen claims of broad applicability.
- Deeper analysis of computational overhead for MCTS and LLM integrations, beyond the DRL-focused Figure 5.
- Discussion of practical strategies to handle domain restrictions for rewrite rules (e.g., filtering invalid expressions during sampling).

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"Simplistic reward function"**: The reward function 1/(1+NMSE) is standard in symbolic regression; criticizing it as simplistic does not undermine the core contribution.
- **"Marginal LLM improvements"**: While improvements in Table 2 are sometimes small, they are consistent across models and datasets; minor gains can still be meaningful in this context.
- **"Lack of a dedicated limitations section"**: The paper discusses open problems and constraints in Sections 3.3 and B.2; a separate section is not mandatory.
- **"Formatting/style nitpicks"**: Artifacts from PDF extraction (e.g., broken figure references) do not affect the technical content.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct multiple independent runs with different random seeds to report mean and standard deviation (or confidence intervals) for NMSE metrics, enhancing statistical reliability.
- Include a comparison with a state-of-the-art genetic programming method that uses e-graphs (e.g., de França & Kronberger, 2025) to better position the unified framework's novelty.
- Directly measure and report the variance of the gradient estimator in DRL experiments to empirically validate Theorem 3.2.
- Add a brief case study or discussion illustrating scenarios where EGG does not improve performance, helping to define the method's boundaries.