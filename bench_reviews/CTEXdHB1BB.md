## Summary
The paper introduces Conditional Advantage Estimation (CANON), a novel advantage estimation method for reinforcement learning with verifiable rewards (RLVR) in large reasoning models. CANON incorporates human priors on training metrics (e.g., entropy, response length) without assuming a directional preference by regrouping responses into two groups based on the metric and computing inter-group and intra-group advantages. Experiments across three LLMs and multiple reasoning benchmarks demonstrate improved accuracy and token efficiency, and the method achieves a superior Pareto frontier in performance-cost trade-offs.

## Strengths
- **Novel and principled methodology**: The core idea of conditional regrouping to avoid hardcoded directional priors is innovative and addresses a clear limitation in prior reward/advantage shaping techniques. The method is well-motivated and theoretically grounded with two theorems showing amplification properties.
- **Extensive and rigorous evaluation**: The paper evaluates three different LLMs on six math reasoning benchmarks and three high-complexity logic reasoning tasks, comparing against a wide array of strong baselines (including ReMax, RLOO, GRPO, DR.GRPO, and entropy/length-specific methods). The results consistently show improvements in accuracy (e.g., +1.9 points on math tasks, up to +5.2 points on hard logic tasks) and efficiency (33.8% token reduction).
- **Flexibility and practical benefits**: CANON supports dynamic scheduling (via µ) to balance exploitation and exploration across tasks, and weighting (via α) for efficient reasoning, achieving a new Pareto frontier. The framework is general and can be applied to different metrics.

## Weaknesses
- **Lack of statistical significance reporting**: The paper reports single accuracy numbers without confidence intervals or standard errors, which is a standard expectation for empirical results at ICLR. This makes it difficult to assess the reliability of the claimed improvements, especially for marginal gains.
- **Hyperparameter tuning for dynamic scheduling**: While the core CANON method with fixed µ already shows gains, the dynamic scheduling introduces additional hyperparameters (µ scheduling strategies) that require task- or model-specific tuning. The paper selects the best schedule per model, which could overstate the benefits of scheduling without proper ablation (e.g., comparing to a fixed µ baseline other than 0.5).
- **Limited analysis of the method's mechanisms**: The paper provides training dynamics and metric trends, but lacks qualitative analysis of the final models' reasoning behaviors (e.g., case studies comparing reasoning chains) and a deeper investigation of how inter- and intra-group advantages affect gradient updates. This would strengthen the understanding of why CANON works.

## Nice-to-Haves
- **Exploration of additional metrics and domains**: Testing CANON on other metrics (e.g., confidence, diversity) and non-reasoning tasks (e.g., code generation) would further demonstrate its generality.
- **Sensitivity analysis of theoretical assumptions**: Empirical investigation of how deviations from equal group sizes or condition independence affect performance would bolster the theoretical claims.
- **Computational overhead discussion**: Explicitly stating the negligible overhead of sorting and group mean calculations would address practical deployment concerns.

## Removed Points 
These points are flagged to be removed, treat them with caution:
- **Missing comparison with sophisticated baselines**: The paper does compare with Entropy Adv and Clip-Cov for entropy, and with length penalty methods, as shown in Table 1 and Table 3.
- **Ablation on grouping operation**: The paper includes an experiment with random regrouping (Table 12) showing no improvement, which addresses the necessity of meaningful grouping.
- **Formatting issues in tables**: These are likely parser artifacts from extraction and not inherent to the paper.

## Novel Insights
The paper's main novel insight is that by regrouping responses based on a metric and comparing across and within groups, one can amplify the metric's influence without imposing a prior on its direction, thereby enabling adaptive exploitation of beneficial trends (e.g., low entropy for math, high entropy for complex logic) and efficient reasoning. The theoretical analysis further shows that this amplification is selective to the chosen metric.

## Suggestions
- **Report statistical significance**: For key benchmarks, provide confidence intervals (e.g., via bootstrapping) or standard errors over multiple runs to substantiate the improvements.
- **Include qualitative case studies**: Show concrete examples of reasoning chains where CANON-Inter or CANON-Intra leads to correct solutions that baselines miss, and analyze failure cases.
- **Ablation on fixed vs. dynamic scheduling**: Compare the best dynamic schedule against a few fixed µ values (e.g., 0.2, 0.8) to better isolate the benefit of scheduling.