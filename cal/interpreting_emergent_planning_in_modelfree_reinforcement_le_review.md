=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary

This paper investigates whether Deep Repeated ConvLSTM (DRC) agents—model-free RL agents without explicit world models—learn to internally plan when playing Sokoban. The authors develop a three-part methodology: (1) probing for planning-relevant concepts (Agent Approach Direction $C_A$ and Box Push Direction $C_B$), (2) qualitatively analyzing plan formation, and (3) confirming causal behavioral influence through interventions. The paper provides evidence that DRC agents represent these concepts, form plans iteratively resembling parallelized bidirectional search, and can be steered by manipulating plan representations.

## Strengths

- **Causal intervention methodology:** Unlike many interpretability papers that rely solely on correlational probing, this work validates findings through causal interventions (Section 6.1). The high success rates for Agent-Shortcut interventions (94.6% at Layer 1, 98.8% at Layer 3) substantially above random baselines (~30%) provide strong evidence that $C_A$ and $C_B$ representations are causally involved in action selection, not merely correlational artifacts.

- **Temporal emergence evidence:** Figure 9 demonstrates a strong correlation between probe F1 scores and the emergence of test-time compute benefits across training checkpoints. This links internal mechanistic states to behavioral capabilities over time, addressing a gap common in interpretability research.

- **Clear methodological framework:** The three-step procedure (probe → investigate formation → confirm dependence) offers a reusable template for mechanistic interpretability that goes beyond behavioral benchmarks to causal verification.

- **Cross-layer consistency:** The finding that concepts are represented across all layers (Figure 4) with similar probe accuracy, combined with different intervention effectiveness across layers (Table 1), provides interesting evidence about the iterative computation hypothesis.

## Weaknesses

- **Concept circularity in definitions:** $C_A$ and $C_B$ are defined over the agent's *actual future behavior*—e.g., $C_A$ encodes "the direction from which the agent will move onto the square the next time it does so." These concepts can only be computed by observing the agent's future actions over an entire episode. In a deterministic policy operating in a deterministic environment, the entire future trajectory is determined at the first step. High probe accuracy could therefore reflect either genuine forward planning OR simply that a consistent, spatially-organized policy in a deterministic environment makes future actions predictable from rich internal states. The intervention experiments partially address this concern, but the logical gap between "predictable future behavior" and "forward-looking planning" deserves more explicit treatment than it receives.

- **Qualitative characterization of planning algorithm:** The claim that the agent performs "parallelized bidirectional search" rests primarily on visual inspection of selected examples (Figure 1 and Appendices A.2.1–A.2.9). The paper provides no quantitative characterization of: (a) what fraction of episodes exhibit bidirectional patterns, (b) whether plan arrows consistently grow from both boxes and targets in early vs. late ticks, or (c) how many routing threads are active simultaneously. Without such quantification, the bidirectional search characterization risks over-interpretation of cherry-picked examples.

- **Handcrafted intervention levels:** The Agent-Shortcut and Box-Shortcut levels are specifically designed to have two well-separated plans of different lengths. This maximizes the chance that small perturbations tip the agent between options, but leaves unclear how robust interventions are in more complex level structures where candidate plans are less neatly separated.

- **Asymmetric intervention success rates:** Box-Shortcut interventions show substantially lower success rates (56.2% at Layer 1, 80.6% at Layer 3) compared to Agent-Shortcut interventions (94.6%–98.8%). Given that probes for $C_A$ and $C_B$ have similar F1 scores across layers (Figure 4), the intervention asymmetry suggests either that $C_B$ representations are more entangled with other features or that the behavioral effect of plan representations differs between agent movement and box push planning. This warrants deeper investigation.

- **Single architecture and task focus:** All main results come from DRC agents on Sokoban. While Appendices mention Mini PacMan (H) and ResNet (G) experiments, these are relegated to appendices without substantial discussion in the main text. The paper claims "first mechanistic evidence that model-free RL agents can learn to plan," but the evidence is from one agent-environment combination.

- **No class-by-class probe breakdown:** The macro F1 scores (~0.85 for $C_A$, ~0.95 for $C_B$) are reported without breakdowns by class. Given class imbalance (many squares are NEVER), understanding how well directional classes (UP/DOWN/LEFT/RIGHT) are predicted versus NEVER is important for interpreting whether the agent truly encodes directional plans or merely encodes "will interact with this square."

## Nice-to-Haves

- **Architecture diversity:** Including Transformer-based or standard CNN agents without recurrent ticks in the main text would clarify whether this planning mechanism is specific to ConvLSTM architectures with internal ticks.

- **Ablation experiments:** Projecting out the identified concept directions during inference (rather than adding steering vectors) would further confirm these representations are necessary for planning.

- **Quantitative search characterization:** Correlating the number of internal ticks required for plan convergence with level complexity (e.g., optimal path length) would distinguish search from fixed-computation heuristic matching.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Lack of negative controls for probing on untrained networks"** (from positive reviewer): This is incorrect—Section 6.1 explicitly uses randomly initialized probes as baselines for interventions and shows they perform near chance.

- **"Baseline probe should use observation plus learned heuristics"** (from harsh critic): This would be a stronger baseline, but the current baseline (raw observation) already establishes that concepts are not trivially extractable. This is a nice-to-have improvement, not a core flaw.

- **"Policy vs. planning distinction is philosophical"** (from positive reviewer): While valid philosophically, the paper provides causal evidence that addresses this concern. The interventions demonstrate the representations are causally involved in action selection, which goes beyond "cached policy" arguments.

## Novel Insights

The paper's most interesting contribution is the temporal emergence finding: the correlation between probe F1 and test-time compute benefits across training checkpoints (Figure 9) suggests that planning capabilities emerge gradually and measurably during training. This provides a potential diagnostic for detecting when planning "kicks in" during agent training. The finding that Layer 3 interventions are more effective than Layer 1 interventions for Box-Shortcut (despite similar probe F1) suggests a hierarchical organization where later layers consolidate plan representations for causal behavioral influence—a hypothesis worth further investigation.

## Suggestions

- Add quantitative metrics for the bidirectional search claim: measure the fraction of episodes showing clear bidirectional patterns, or correlate plan extension direction (forward from boxes vs. backward from targets) with tick number.

- Include class-by-class F1 breakdowns in an appendix to clarify directional vs. NEVER class accuracy.

- Report intervention success rates on standard Boxoban levels (not just handcrafted ones) to assess real-world applicability of steering.

- Explicitly discuss the concept circularity concern in the methodology section: acknowledge that $C_A$ and $C_B$ are retrospectively computed labels, and explain why the intervention evidence addresses the ambiguity between planning and predictable policy execution.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
