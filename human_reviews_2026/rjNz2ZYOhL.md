# TABS: Strategic Game-Based Multi-Stage Reinforcement Learning Challenge

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 2

## Abstract
The design of environments plays a critical role in shaping the development and evaluation of reinforcement learning (RL) algorithms. While existing benchmarks have supported significant progress in both single-agent and multi-agent settings, many real-world systems involve multi-stage structures with tightly coupled decision points at each stage. These settings require agents not only to perform well within each stage but also to coordinate effectively across them. We introduce the Totally Accelerated Battle Simulator (TABS), a complex multi-stage environment suite implemented in JAX to enable accelerated training and scalable experimentation. Each TABS task consists of sequential stages with interdependencies, where only the output of one stage is forwarded to the next. This multi-stage structure makes effective exploration challenging, often steering agents toward locally optimal behaviors that limit overall performance. Our empirical analysis shows that standard RL baselines struggle to solve TABS tasks, illustrating the difficulty of learning coherent strategies across interdependent stages. TABS provides a controlled and extensible framework for studying multi-stage decision-making challenges, supporting future research on RL methods capable of operating effectively in structured domains. Our code is available at:~\url{https://anonymous.4open.science/r/TABS-0E4B}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents TABS (Totally Accelerated Battle Simulator), a new multi-stage reinforcement learning (RL) benchmark designed to capture sequential dependencies and cross-stage decision-making challenges.
TABS consists of three interlinked stages—unit combination, unit deployment, and battle simulation—where each stage’s outcome determines the input to the next. The environment is implemented fully in JAX, enabling end-to-end GPU acceleration and large-scale parallelization. The authors benchmark several standard RL algorithms (PPO, PQN, MAPPO, IPPO) under two training regimes—simultaneous and alternating—to evaluate performance, stability, and exploration difficulty. Results show that even strong baselines struggle due to entangled credit assignment and exploration bottlenecks, demonstrating the environment’s difficulty and potential as a testbed for hierarchical or cross-stage RL methods.

### Strengths
1. Novel multi-stage environment.
TABS explicitly models multi-step interdependencies, addressing an underexplored challenge in current RL benchmarks that typically assume single-stage or loosely connected tasks.
2. High-quality engineering and implementation.
The JAX-based, GPU-accelerated environment is well-structured, scalable, and supports efficient parallel training, showing strong reproducibility and technical completeness.
3. Comprehensive experimental evaluation.
Multiple algorithms, scenarios, and budget settings are benchmarked. The comparisons between simultaneous and alternating training highlight non-trivial learning behaviors.

### Weaknesses
1. The benchmark claims strong cross-stage coupling, yet there are no hyperparameters to vary or quantify how much Stage-1/2 choices constrain Stage-3, nor causal/attribution diagnostics linking early decisions to final returns. Scenarios are a small, fixed set(four presets × three budgets(Table 1; Appendix A.4)) rather than a spectrum of coupling strengths, so it’s unclear whether difficulty comes from genuine interdependence or generic MARL hardness. This weakens the central claim that TABS uniquely stresses cross-stage reasoning. It would be beneficial to add cross-stage hyperparameters.
2. For UnitComb/UnitDeploy, the paper assigns the entire final return only to the last action and sets all prior actions’ rewards to 0(Section 4.1 and Algorithms 1–2). This would make it hard to tell whether failures reflect cross-stage credit assignment or just the chosen shaping. 
3. All baselines are PPO-family/on-policy (PPO, MAPPO, IPPO, PQN) stitched across stages. Given the paper’s emphasis on long horizons, delayed credit, and exploration, excluding off-policy (e.g., SAC/TD3 variants for hybrid actions), model-based (e.g., Dreamer-style) limits what we can infer about where difficulty truly comes from. As a result, statements like “simultaneous > alternating” or “high variance is inherent” may be method-specific, not environment-intrinsic.
4. The battle opponent is a role-aware heuristic with tunable noise; results fix this heuristic and provide only minimal sensitivity (e.g., “Expert” vs “Schedule” in one scenario) when analyzing exploration/variance. Without a principled sensitivity sweep (or alternate opponent families), robustness and generality remain uncertain.

### Questions
1. How sensitive are training outcomes to the stochasticity of the heuristic opponent policy?
2. Is the environment compatible with off-policy or model-based RL pipelines (e.g., JAX-DQN, SAC, TD3, Dreamer)?
3. Could the authors elaborate on how simultaneous vs. alternating updates influence convergence speed or stability?
4. How computationally demanding is training across all scenarios (GPU hours, wall-clock time)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces TABS (Totally Accelerated Battle Simulator), a multi-stage reinforcement learning environment inspired by strategic battle games. The environment consists of three sequential stages: unit selection, unit deployment, and real-time battle simulation. The authors implement the environment in JAX to enable GPU-accelerated training and evaluate several baseline algorithms (PPO, PQN with MAPPO/IPPO) using both simultaneous and alternating training methods.

### Strengths
- Timely engineering contribution: The JAX implementation enabling GPU-accelerated environment simulation addresses a practical bottleneck in RL research, achieving significant speedup as demonstrated in Figure 8.
- Comprehensive experimental analysis: Detailed investigation across multiple algorithms (Section 4.2) with both simultaneous and alternating training methods, providing insights into the challenges of multi-stage learning.
- Thoughtful reward design (Line 199): The shared reward proportional to health ratio differences and binary win-loss outcome is well-motivated for the battle simulation stage.
- Clear visualizations: Section 3.1 provides helpful visual representations of the environment stages, particularly the field-of-view constraints and their impact on initial observations.
- Role-appropriate heuristic policy: The differentiated behaviors for different unit types (Section 3.3) provides more realistic and challenging opponent strategies than uniform policies.

### Weaknesses
- Unclear abstract and motivation: The abstract fails to clearly define what constitutes a "complex multi-stage environment" or why existing benchmarks are insufficient. The phrase "tightly coupled decision points" is vague and never formally defined.
- Missing formal definitions: The main paper lacks mathematical formulation of what constitutes a "stage" and how stages differ from standard MDPs with varying state spaces. The interdependencies between stages are described informally without clear specification of the coupling mechanisms.
- Limited novelty in multi-stage concept: The paper presents multi-stage decision-making as contribution with limited comparisons and discussion of prior works.
- Insufficient RL background: The paper focuses heavily on game mechanics and engineering details while providing minimal background on relevant MARL concepts. Related work is relegated to Appendix B (Line 127), which should be in the main text.
- How are policies structured to handle different observation/action spaces across stages (Lines 134, 351)?
- What exactly is a "stage-conditioned mixture of stage policies" (Line 363)?
- How is an episode defined relative to the "pipeline" (Line 365)?
- Limited scientific contribution: The paper reads more as an environment release than a research contribution. The experiments primarily demonstrate that existing algorithms struggle with the environment but provide limited insights into why or how to address these challenges.
- Exploration analysis lacks depth: While Section 4.3 identifies exploration challenges, the analysis relies only on basic entropy regularization without investigating modern exploration methods (RND, disagreement, curiosity-driven approaches).

### Questions
- Line 050: "Aligned and integrated" with respect to what? This metaphor needs clarification or removal.
- Line 055: Can you provide a formal definition of "stage" early in the introduction? How does this differ from standard episodic RL with varying state spaces?
- Line 068: How is interdependence between stages different from any sequential decision problem? The chess tournament analogy (winning individual games to win tournament) suggests this isn't unique.
- Line 134: When stages have different observation/action spaces, are they padded to equal dimensionality? How does the policy architecture handle these variations?
- Line 135: Is "entire pipeline" synonymous with completing all stages sequentially? Please define this term clearly.
- Line 351: What does "distinct" mean formally in terms of observation/action spaces? Are they different types, dimensions, or both?
- Line 363: Please provide the mathematical formulation for "stage-conditioned mixture of stage policies".
- Line 365: How is an episode defined? Does it span all stages or is each stage a separate episode?
- Line 412: Have you conducted ablations on entropy regularization? Have you tested modern exploration methods (RND, disagreement-based, curiosity-driven) given that entropy-based exploration has known limitations?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
TABS is a strategy-game-inspired, multi-stage, accelerated RL environment written in JAX. It provides configurable environments and four default scenarios (configurations). The first two stages are combinatorial in nature, whereas the final stage can be cast as a multi-agent problem. TABS presents three challenges: structured exploration, heterogeneous/mixed action spaces, and delayed rewards.

### Strengths
I think TABS can contribute to both single- and multi-agent RL research in the following ways:
- RL methods focusing on structured exploration and long-term credit assignment can leverage its core challenges.
- Heterogeneous action spaces and multi-stage environments are under explored. The multi-stage aspect of the environment led the authors to explore an alternative training paradigm, i.e., alternating training.
- A JAX-friendly and configurable environment that can be integrated to pure JAX RL implementations. Plus, TABS provides built-in visualizations.

### Weaknesses
My main concern is how TABS differentiates itself within the vast space of accelerated RL environments.

- Although I am unsure how best to measure it, I would like to see quantitative comparisons to existing environments such as Atari, Jumanji [3] (the first two stages share a combinatorial flavor), and the MuJoCo Control Benchmarks, particularly in exploration and credit assignment. See [1] for a similar analysis for agents.
- I was expecting a gym(-nax [2])-like API for making environments. The current setting leaves the environment configurable, but for standardization and broader adoption, the authors may consider a list of fixed/predefined configurations with accompanying names and use a `tabs.make("<scenario-name>")`-like API for initialization.
- Although the source is easy to navigate, I would like to see more detailed documentation in README regarding the stages, environment creation, observation and action spaces, and reward computation.

Also see some of the questions.

[[1](https://openreview.net/pdf?id=rygf-kSYwH)] Osband, Ian, et al. "Behaviour suite for reinforcement learning." arXiv preprint arXiv:1908.03568 (2019).

[[2](https://github.com/RobertTLange/gymnax)] Lange, Robert Tjarko. Gymnax: A JAX-Based Reinforcement Learning Environment Library. 0.0.4, 2022, github.com/RobertTLange/gymnax.

[[3](https://github.com/instadeepai/jumanji)] Bonnet, Clément, et al. "Jumanji: a diverse suite of scalable reinforcement learning environments in jax." arXiv preprint arXiv:2306.09884 (2023).

### Questions
- I would like to see a discussion of how TABS can contribute to the development of new RL algorithms. How much of TABS’s difficulty stems from its multi-stage nature? How impactful are early-stage decisions on the BattleSimulator stage, and how could this be measured?
- Do you provide starter opponents and configurations with **increasing** difficulty (e.g., few available units and positions) so practitioners can quickly test approaches and gradually scale up?
- Can TABS be used to measure generalization (or easily configured to do so), for example via procedurally generated configurations?
- Is there an end-to-end scenario that unifies the three action and observation spaces into a single three-stage `env` in TABS? If not, what are the main challenges in providing such an environment?
- How fast is TABS compared to other JAX-based accelerated environments, e.g., in steps/second, frames/second, and episodes/second?
- Did you apply any tuning process (manual or otherwise) to balance unit strength and price, for example, to avoid trivial solutions where one unit type consistently dominates?
- (small note) Is `run_example.py` under the `tabs` directory using a depreciated object `default_tabs_conf`  from `tabs.scenarios`? I could not run it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors describe TABS a multi-stage RL challenge.
The environment is based on a battle simulator environment with multi-distinct stages that are dependent on one another
Stages are difference where it both single and multi-agent problems
The environment's goal is to test the limitations of standard RL algorithms, which struggle to coordinate effectively across stages, and provides a benchmark for testing multi-stage decision-making.

### Strengths
* The environment described an interesting multi-stage long-horizon RL problem
* The environment is has distinct but interdependent stages which is not similar to existing environments

### Weaknesses
* The authors focus a lot on describing the details of the env, including how unit properties and how they interact (section 3.2 and 3.2), but doesn't really focus on the insights from the multi-stage learning problem. 
* Comprehending the details of the environment was difficult for me as I am not familiar with Landfall games
* The authors provide baseline results but doesn't reveal much insights from the experiments. It is difficult to understand how to performance of different stages affects the final performance, the effects of simultaneous vs alternating training

### Questions
* Were other RL algorithms explored?
* What new insights about multi-stage RL are revealed by this environment that cannot be studied in existing benchmarks?
* What are the main advantages of using TABS compared to say adding Units comb and Unit deploy stages in the SMAC / SMAX?

### Soundness
2

### Presentation
2

### Contribution
2
