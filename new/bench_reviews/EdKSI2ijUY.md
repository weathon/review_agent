## Summary

The paper introduces LMRL-Gym, a benchmark for evaluating multi-turn reinforcement learning (RL) algorithms with large language models. It comprises 8 tasks—3 Interactive Dialogue tasks (20 Questions, Guess My City, Car Dealer) using LLM-based simulators, and 5 RL Capability tests (Maze, Text-Nav, Wordle, Chess, Endgames) designed to isolate specific RL capabilities such as credit assignment, trajectory stitching, and partial observability. The paper also provides an open-source framework implementing PPO, ILQL, MC Returns, and behavioral cloning baselines, and reports experimental results demonstrating that RL methods generally outperform behavioral cloning while highlighting areas for algorithmic improvement.

## Strengths

1. **Timely and well-structured task design with diagnostic intent.** The division into Interactive Dialogue and RL Capability task categories, and the explicit mapping of tasks to specific RL capabilities (Figure 2), provides a useful conceptual framework for evaluating multi-turn RL with language models. This is a genuine contribution to a space where "there are no systematic benchmarks available" for multi-turn RL with LLMs.

2. **Complete benchmark package with offline data and online simulators.** Providing both offline datasets and interactive simulators, along with baseline implementations across online and offline RL paradigms, significantly lowers the barrier to entry for future research. The inclusion of both POMDP and fully-observed variants of Maze and Text-Nav enables targeted diagnosis of partial observability challenges.

3. **Valuable empirical findings despite limitations.** The observation that ILQL excels on RL Capability tasks while MC Returns outperforms on Interactive Dialogue tasks is an interesting and non-obvious finding that could guide future algorithm design. Similarly, the finding that RL-finetuned small models can approach or exceed GPT-4 on some tasks (while underperforming on in-distribution dialogue tasks) provides useful signal about the interaction of RL and pretraining scale.

4. **Human validation of dialogue simulators.** The 40-user study (Appendix A) addresses a legitimate concern about synthetic environments and provides empirical support for the naturalness of LLM-generated dialogue, even if it does not fully validate the simulators for RL evaluation purposes.

## Weaknesses

### Major:

1. **No variance reporting across multiple runs.** All results in Table 2 are single-run numbers. For stochastic RL algorithms like PPO and ILQL, run-to-run variance can be substantial. As noted by multiple reviewers, this makes it impossible to assess whether observed performance differences (e.g., ILQL 47.3 vs. PPO 48.0 on Chess; MC Return 57.2 vs. PPO 50.5 on Car Dealer) are statistically meaningful or noise. For a benchmark paper whose primary purpose is enabling reliable algorithm comparison, this is a significant gap.

2. **Experiments limited to small model scales.** All experiments use GPT-2–based models (Section 5: "We choose GPT2 rather than a larger model due to memory and time constraints"). The paper acknowledges this but nonetheless draws broad conclusions about RL enabling "complex goal-directed behaviors in language models" and claims findings "highlight significant areas for growth" for RL algorithms. Whether these findings generalize to modern LLMs (7B+ parameters) is unknown, and the interaction between RL algorithm effectiveness and model scale can be nontrivial—instabilities observed at GPT-2 scale may diminish at larger scales, or vice versa.

3. **Insufficient validation of LLM simulators as RL environments.** The dialogue simulators are fine-tuned from synthetic GPT-3.5 data and used both for offline data generation and online evaluation. The paper asserts that "20 Questions and Guess My City are particularly hard to hack as they require the agent to successfully guess the word" (Section 5) and claims the buyer model is robust. However, no systematic adversarial analysis is provided: no examples of RL-trained agents attempting to exploit simulator quirks, no measurement of reward hacking (high reward without genuine task achievement), and no analysis of simulator behavior under distribution shift when RL agents drive dialogue into off-manifold regions. The user study evaluates naturalness of simulator outputs, not whether the simulators produce goal-appropriate behavior under adversarial probing.

4. **Claims about specific RL capabilities are not empirically isolated.** Figure 2 maps tasks to capabilities (e.g., Maze tests "credit assignment" and "trajectory stitching"), but no experiments actually isolate these mechanisms. For instance, to demonstrate trajectory stitching, one would want experiments varying dataset optimality or disentangling it from simple imitation of good trajectories. As written, any improvement over BC is attributed to the mapped capability without controlled ablations. This undermines the paper's claim to "rigorously stress-test the ability of RL algorithms" (Section 1).

5. **GPT-4 baseline score of 0 on Chess and Endgames is likely an evaluation artifact.** GPT-4 scoring 0.0 on Chess and Endgames (Table 2) strongly suggests a prompting/parsing failure rather than genuine inability. The paper nonetheless uses these scores to claim "our much smaller RL finetuned models significantly outperform GPT4" on RL Capability tasks. Without investigating or acknowledging this artifact, the cross-method comparison involving GPT-4 on these tasks is misleading.

### Minor:

6. **Normalization scheme obscures absolute performance and cross-task difficulty.** The linear normalization (0 = min, 50 = dataset average, 100 = max) makes it difficult to compare difficulty across tasks and to assess how close methods are to optimality. A near-perfect normalized score of 99.9 on FO Maze for ILQL could reflect either near-optimal performance or a ceiling effect; without an oracle upper bound or raw score analysis in the main text, this is unclear.

7. **Limited algorithm coverage for an offline RL benchmark.** The paper primarily focuses on offline RL ("our work primarily focuses on benchmarking offline RL algorithms," Section 3) but only evaluates one offline TD method (ILQL) and MC Returns. Prominent offline RL approaches for sequence modeling (e.g., Decision Transformer, conservative Q-learning variants) are missing, despite the authors citing Decision Transformer.

8. **Some RL Capability tasks may not genuinely test language-based reasoning.** Chess and Endgames involve highly structured, constrained action spaces that are essentially symbolic problems encoded as text. The paper itself acknowledges this by including a symbolic maze baseline (Appendix H). The "complex language" column in Figure 2 shows that Chess and Endgames do not test this capability, raising questions about what language-specific RL challenges they evaluate beyond standard RL with text rendering.

9. **Lack of qualitative analysis and learning curves.** The paper mentions "instabilities observed in training PPO" (Section 6) but provides no learning curves, convergence plots, or examples of failure modes. For a benchmark intended to enable algorithm development, qualitative examples of agent behavior (e.g., pre- vs. post-RL dialogue trajectories) and training dynamics would be informative.

## Nice-to-Haves

- Include experiments with at least one model at the 7B parameter scale on a subset of tasks to assess whether findings generalize.
- Provide an oracle/near-optimal upper bound for each task so users can calibrate how far current methods are from optimality.
- Analyze the offline dataset quality and size by varying composition and measuring how this affects different RL methods.
- Add analysis of why ILQL underperforms MC Returns on dialogue tasks, as this is one of the paper's most interesting findings.

## Removed Points

- **Formatting/style nitpicks**: Token-level MDP formulation conceptual quibbles about state vs. observation terminology are removed as they do not affect the paper's core contributions.
- **Missing tasks outside scope**: Demands for tool use, multi-agent interaction, or long-horizon (>50 turn) tasks are scope creep; the paper is clear about its task coverage and these additions would improve breadth but are not necessary.
- **Reproducibility complaints about hyperparameters**: The paper states "we share all of the hyperparameters we used to train our models in Appendix E." Demanding complete training logs or exhaustive hyperparameter search documentation is excessive for a benchmark paper.
- **Claims about model size "contradiction"**: The reviewer noted a contradiction between "1.5B parameters" in Section 7 and smaller models in tables; however, the paper states "we have primarily trained and evaluated models with a maximum 1.5B parameters" (Section 7), which is consistent if some experiments used smaller models. The real issue is the small scale overall, not a contradiction.
- **Demand for confidence intervals as fatal flaw**: While variance reporting is important for a benchmark, demanding confidence intervals as a fatal error overstates the issue. It is a significant gap but not one that invalidates the entire contribution.

## Novel Insights

The most interesting finding in this paper is the qualitative divergence between offline value-based methods on structured vs. dialogue tasks: ILQL dominates on RL Capability tasks (likely via trajectory stitching on more structured action spaces) while the simpler MC Returns method wins on Interactive Dialogue tasks. This suggests that current offline TD-learning methods struggle to scale to open-vocabulary, high-variability dialogue settings—precisely where offline RL would be most valuable given the abundance of dialogue data on the internet. This finding is underexplored in the paper but represents a genuine insight that could guide future algorithm design.

## Score and Decision

**Calibration:** I compared against papers with similar benchmark contributions and similar weakness profiles. LMRL-Gym itself received human scores of 6, 5, 6, 5 (→ Reject) at another venue. LLF-Bench, with similar issues around experimental rigor and limited baselines, received 5, 6, 5, 3 (→ Reject). SmartPlay, which had a similar task structure but stronger evaluation, received 8, 5, 6, 8 (→ Accept poster). MINT, with a similar multi-turn evaluation focus, received 5, 8, 8, 6 (→ Accept poster).

This paper has a genuinely useful task design and framework, but its evaluation does not meet the rigor required for a benchmark paper whose primary purpose is enabling reliable algorithm comparison. The lack of variance reporting, small model scale only, simulator validation gaps, and unsupported capability claims are significant—but not fatal—weaknesses. The contribution is real but the evidence for the core claims is insufficient in its current form. This places it below papers like SmartPlay or MINT that had cleaner evaluation, but above papers like LLF-Bench that had more fundamental issues.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>