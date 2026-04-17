# Forging Better Rewards: A Multi-Agent LLM Framework for Automated Reward Evolution

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Large Language Models (LLMs) have shown increased autonomy in performing complex tasks, but the inference latency and fine-tuning cost impose significant limitations for their application in dynamic, real-time environments such as robotics and gaming. Reinforcement learning (RL), by contrast, offers efficient execution and has shown strong results in diverse domains. Yet its progress is often bottlenecked by the challenge of designing effective reward functions, which are typically sparse and require heavy manual effort to engineer. Recent work has explored LLM-based reward generation, reducing manual effort yet remaining unstable, unstructured, and opaque. Building on the enhanced reasoning capabilities of modern LLMs, we advance this line of research toward full automation by introducing structured reward initialization, evolutionary refinement, and explicit complexity modeling. These innovations reduce reliance on manual trial-and-error while enabling more stable, interpretable, and scalable reward design. We unify them into FORGE (Feedback-Optimized Reward Generation and Evolution), a multi-agent framework that automatically forges increasingly effective reward functions. Extensive experiments across three games and a robotics task demonstrate the effectiveness of FORGE, achieving up to 38.5% improvement over Eureka and 19.0% over REvolve in the Humanoid task, while maintaining competitive token efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces FORGE (Feedback-Optimized Reward Generation and Evolution), a multi-agent framework that automates reward synthesis for reinforcement learning using large language models (LLMs). FORGE replaces traditional genetic algorithm encodings with LLM-guided crossover, uses a planner-based zero-shot initialization to generate structured reward functions, and introduces a depth metric to quantify reward complexity over evolutionary iterations. A reward pool is maintained as memory to manage and refine candidates. Experiments are conducted across four environments; three games (Tetris, Snake, Flappy Bird) and a continuous-control task (MuJoCo Humanoid), showing performance improvements over prior methods such as Eureka and REvolve, while claiming enhanced stability, interpretability, and token efficiency.

### Strengths
1.1 The paper compares different LLMs and evaluates on multiple environments.

1.2 The experiments are broad and include several baselines.

1.3 The depth measurement is a welcome addition, giving a way to track reward composition complexity.

### Weaknesses
2.1 The comparison with Eureka is unfair. Eureka is greedy (keeps only the best reward per generation), not population-based. Plotting average population scores for FORGE and REvolve, but not the best-per-generation for Eureka, makes the figure incomparable. A fair comparison would show the maximum score for each generation across all methods.

2.2 The REvolve baseline results are incorrect. The authors state that they use the environment's extrinsic rewards as the fitness score. However, in Fig. 2, the population average for REvolve decreases over generations, contradicting REvolve’s framework, which adds individuals only if their fitness score exceeds the current population average. This guarantees that the average will never decrease (section 3.4 of the REvolve Paper). 

2.3 The table results for the Humanoid task do not match the video output. The task is to move as fast as possible on the x-axis without falling. REvolve’s humanoid performs this correctly and visibly better, while FORGE’s agent moves much worse, although achieving high reward results. The results and videos are thereby inconsistent.

2.4 The claimed “stability”, “interpretability”, and “token efficiency” are not supported. Figure 2 shows dips even for FORGE, so it is not stable. “Interpretability” is only a depth-vs-score correlation and does not explain why rewards work. “Token efficiency” is argued but not tested. 

2.5 The authors claim that passing only the reward function and best score is sufficient and that raw metrics are not beneficial. Metrics can indicate which reward components failed and guide improvement in the next generation. REvolve showed that better quality feedback leads to better performance. No results are showed to support that claim.

2.6 Line 217-219: the authors say “to address these limitations we generalize the evolutionary process by incorporating LLM inference…”. This is presented as if it is new, which is not (T2R/Eureka were first).

2.7 It is unclear to me what exact rule is used to retain or discard individuals and when/how the pool is pruned.

2.8 The Planner can be viewed as structured prompt design rather than actual planning. It is only used for zero-shot initialization, not iterative optimization.

### Questions
3.1 Can the authors explain why the average population scores for REvolve decrease over generations, even though REvolve’s framework guarantees a non-decreasing average as described in the original paper?

3.2 Can the authors explain why they plot average population scores instead of the best-per-generation values, and provide the additional results showing the best scores per generation for all methods to make the comparison fair?

3.3 Can the authors elaborate on the points raised in 2.6 and 2.7?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the FORGE, a multi-agent LLM framework that combines structured reward initialization, evolutionary refinement, and explicit complexity modeling. The core problem is that manual reward engineering is costly and suboptimal, while existing LLM-based methods (e.g., Eureka, REvolve) often produce unstable or opaque rewards. FORGE employs a Planner agent to generate modular reward specifications from task objectives and environment dynamics, followed by an evolutionary process where rewards are selectively combined and refined using LLM-guided crossover. Key contributions include a reward pool for efficient memory, a depth measure to quantify structural complexity, and token-efficient evolution. Extensive experiments across three games (Tetris, Snake, Flappy Bird) and a robotics task (Humanoid) demonstrate that FORGE achieves significant performance improvements—up to 38.5% over Eureka and 19.0% over REvolve in Humanoid—while maintaining competitive token usage.

### Strengths
**Method Design**: FORGE introduces a structured two-stage process (Planner-based initialization and Engineer-driven evolutionary refinement) that moves beyond direct LLM sampling for reward generation. Sec. 3; Fig. 1. This modular approach enhances interpretability and stability compared to prior methods like Eureka and REvolve.

**Comprehensive Experimental Evaluation**: The framework is tested across four distinct environments (three games and one simulated robotics task), covering both discrete and continuous control settings. Sec. 4.1; Table 2. FORGE consistently outperforms baselines, including general agentic frameworks, context-aware LLMs, and native environment rewards.

**Token Efficiency**: Despite performance improvements, FORGE maintains competitive token usage by constraining LLM inference to modifying small subsets of reward functions. Sec. 4.5 This is a critical advantage for scalable deployment compared to other multi-agent LLM frameworks.

### Weaknesses
**Limited Generalization**: Evaluation is confined to simulated environments (MuJoCo, games), with no evidence of testing in real-world robotics or complex physical systems (Sec. 5). Tasks lack diversity in observation spaces (e.g., low-dimensional vs. high-dimensional inputs), raising questions about scalability to vision-based or partially observable domains (Table 1). No cross-environment transfer experiments to assess reward function generalization beyond training domains (Sec. 4.1).

**Incomplete Analysis of Evolutionary Components**: The probabilistic selection scheme (Eq. 6) uses unnormalized scores as weights, but no ablation is provided on alternative selection strategies (e.g., rank-based or tournament selection). Crossover operation relies solely on LLM inference without mutation mechanisms, potentially limiting diversity in later generations (Sec. 3.2). Depth measure is defined recursively but lacks theoretical grounding or comparison to other complexity metrics (e.g., code length, entropy) (Eq. 4).

### Questions
1.	How does FORGE handle environments with highly sparse or delayed extrinsic rewards, and does the depth measure correlate with performance in such settings? (Sec. 4.1; Fig. 3)
2.	Could the evolutionary process be enhanced by incorporating multi-objective optimization to balance reward complexity and performance, rather than relying solely on extrinsic return? (Eq. 6; Sec. 3.2)
3.	What are the specific failure modes of FORGE in cases where the LLM generates invalid reward functions, and how frequently do these occur across different environments? (Sec. 3.2; Sec. 4.4)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
FORGE is a multi-agent LLM framework for automated reward evolution, using a Planner agent for structured zero-shot reward initialization, an Engineer agent for iterative selection and crossover refinement, and explicit depth metrics for complexity. Evaluated on Tetris, Snake, Flappy Bird, and Humanoid (MuJoCo), it outperforms baselines like Eureka, REvolve, and context-aware LLMs, achieving up to 38.5% gains over Eureka on Humanoid while maintaining token efficiency.

### Strengths
1.Structured initialization via Planner yields strong zero-shot rewards, outperforming direct LLM sampling; evolutionary refinement drives consistent gains across discrete and continuous domains.
2.Experiments show superior performance in both zero-shot and evolved settings, with ablation studies confirming the value of key components like selection and planning.
3.Introduces reward pool and depth metrics for stability, interpretability, and efficiency, enabling analysis of complexity-performance correlations (e.g., optimal depth 3 for games, 7 for Humanoid).

### Weaknesses
1.Relies on Claude Sonnet 4 without ablations on other LLMs—results may be model-specific and degrade with open-source alternatives.
2.Claims token efficiency but consumes more in some environments; lacks full compute cost breakdown or scaling analysis for larger tasks.
3.Baselines use the same LLM, but adaptations (e.g., replacing human feedback in REvolve) may not be optimal; multi-agent setup adds overhead without clear justification over simpler methods.
4.Depth analysis is insightful but underexplored—e.g., no explanation for domain-specific optima or robustness to hallucinations in crossover.
5.Limited to simulated environments; no real-world robotics tests, overlooking challenges like sensor noise, delays, and safety that could undermine practicality.

### Questions
1.What is the sensitivity to the base LLM? Results with open-source models like Llama or smaller variants?
2.Detailed token/compute costs per iteration/environment? How does efficiency scale to more complex domains?
3.Why does evolution sometimes underperform context-aware LLMs on average? Mechanisms to boost consistency?
4.How can depth metrics guide early stopping? Thresholds or heuristics for optimizing iterations?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FORGE, a multi-agent LLM + evolution framework for automated reward function design. FORGE first produces structured reward specifications via a Planner agent, turns them into modular executable rewards, and then iteratively refines them through LLM-guided evolutionary operations (selection + crossover) under real environment feedback. A reward pool acts as specialized memory and a depth metric captures the structural complexity of rewards. Experiments on Tetris, Snake, Flappy Bird, and MuJoCo Humanoid show that FORGE consistently outperforms Eureka and REvolve.

### Strengths
- Clever decomposition into planner-based initialization + evolutionary refinement makes the framework both stable (at the beginning) and exploratory (later).

- Reward pool + depth is a lightweight and straightforward design to get interpretability and to analyze how complexity correlates with performance.

- Empirical results (3 games + 1 continuous control) show gains over strong recent baselines (Eureka, REvolve).

### Weaknesses
- The most important issue is the computational efficiency. Since each evolutionary step requires training/evaluating an RL policy under a new reward, Even if token usage is controlled, the overall wall-clock/sample cost can still be the main blocker for practical use. A comparison on “environment steps per performance gain” against Eureka/REvolve is missing.

- The method lacks ablations on different LLMs. The method assumes the LLM can both interpret two reward codes and synthesize a valid, environment-compatible offspring. It is unclear how robust this is to weaker models or higher code error rates. Some ablations with a smaller/older LLM, or with constrained generation, would clarify the robustness.

- The setup says all baselines are re-implemented under the same foundation model, but the paper does not detail how much prompt engineering/tuning effort was spent on Eureka/REvolve. Since the key claimed improvement is “structured initialization + evolution,” it would be good to show that Eureka/REvolve do not simply benefit from the same structured spec.

### Questions
Considering most modern LLMs have good vision capabilities, would feeding some vision information (e.g., game environments / failure trajectories) into the LLM helps further improve the sampling efficiency?

### Soundness
3

### Presentation
3

### Contribution
2
