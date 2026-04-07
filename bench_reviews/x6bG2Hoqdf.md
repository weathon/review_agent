## Summary
CALM introduces a co-evolution framework for automatic heuristic design that jointly optimizes prompt generation (verbal guidance) and the underlying LLM via reinforcement learning (numerical guidance). It incorporates novel evolutionary operators, a collapse mechanism, and a tailored reward function for efficient fine-tuning. Experiments demonstrate that CALM outperforms state-of-the-art baselines across multiple optimization tasks while running on a single 24GB GPU with a compact 7B model.

## Strengths
- **Novel co-evolution paradigm**: The integration of RL-based LLM fine-tuning directly into the evolutionary heuristic search loop addresses a clear gap in fixed-model AHD methods, enabling the model to adapt and improve over time (Sections 1, 4).
- **Strong empirical performance**: CALM consistently outperforms SOTA baselines (including API-based methods) on four challenging tasks (OBP, TSP, CVRP, OP) in both in-domain and out-of-domain settings, as shown in Tables 1–3, with significant p-values in Appendix I.6.
- **Resource efficiency and practicality**: The method runs locally on a single 24GB GPU using a 7B INT4-quantized model, fine-tuning only 1.15% of weights, making it accessible without costly API dependencies (Section 5, Appendix I.1).
- **Thorough ablation studies**: Ablations in Table 4 and Section 5.2 validate the contribution of key components (RL fine-tuning, collapse mechanism, reward design, operators), providing evidence for design choices.

## Weaknesses
- **Hyperparameter sensitivity not fully characterized**: While ablations exist for some settings (e.g., reward parameters in Appendix I.7), a systematic sensitivity analysis across all tasks and hyperparameters (e.g., collapse parameters, operator weights) is lacking, affecting reproducibility and robustness.
- **Lack of quantitative diversity metrics**: The paper claims diversity-aware operators aid exploration, but no measure of heuristic diversity (e.g., code edit distance, idea overlap) over time is provided, making it difficult to verify this claim and understand search dynamics.
- **Incomplete ablation coverage**: Ablation studies in Table 4 focus primarily on OBP and OP; similar analyses for TSP and CVRP are missing, weakening the generalizability of the findings about component contributions.
- **Reward distribution unreported**: The distribution of rewards during training (e.g., frequency of infeasible, duplicate, or improving heuristics) is not analyzed, obscuring the quality of the learning signal and model adaptation.
- **Limited statistical runs**: Main results average over three runs, which, despite p-values for some tasks, reduces statistical confidence; more runs would strengthen the empirical claims.

## Nice-to-Haves
- Comparison with a supervised fine-tuning baseline using curated heuristics to isolate the effect of co-evolution versus fine-tuning on static data.
- Ablation on the number of responses per prompt (G) to understand its impact on RL training and advantage estimation.
- Visualization of heuristic diversity and performance over time to illustrate search dynamics and collapse effects.
- Case studies comparing discovered heuristics with seeds to explain performance improvements qualitatively.
- Extension of sensitivity analyses to all hyperparameters across all tasks.

## Removed Points
These points are flagged to be removed, treat them with caution:
- Request for explicit bulleted contributions in the introduction (formatting nitpick).
- Criticism about generality to non-code heuristics (outside the paper's stated scope of code-based AHD).
- Concern over EvoTune re-implementation (justified in paper for fair comparison under resource constraints).
- Demand for theoretical convergence analysis (not standard for an empirical systems paper).
- Suggestion to compare computational cost with API-based baselines (time breakdown is provided, and API costs are variable).

## Novel Insights
None beyond the paper's own contributions. The paper successfully introduces and validates the co-evolution paradigm, but the reviews do not surface additional novel insights beyond what is presented.

## Suggestions
- Incorporate quantitative diversity metrics (e.g., code edit distance or idea token overlap) to validate the effectiveness of diversity-aware operators and collapse mechanism.
- Extend ablation studies to all tasks, particularly TSP and CVRP, to ensure component contributions are consistently beneficial across problems.
- Report the distribution of rewards during training to provide insight into the learning process and signal quality.
- Consider increasing the number of runs for main experiments to enhance statistical robustness, or include confidence intervals where feasible.