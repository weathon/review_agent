# TAME the BALROG: Task-Adaptative Modular Emergent framework for Game Agents

- Decision: Reject
- Scores: 6, 4, 2, 6, 4

## Abstract
Interactive games have proven to be key benchmarks for advancing Artificial Intelligence (AI), requiring capabilities like long-term planning, exploration, and adaptation to stochastic environments. While Large Language Models (LLMs) have achieved notable results across many domains, they struggle in complex gaming environments like those in the BALROG benchmark. The absence of adaptive frameworks that can dynamically configure themselves based on environmental characteristics, limits the progress of AI in games. To this end, we introduce the Task-Adaptive Modular Emergence (TAME) framework, which employs genetic algorithms to evolve environment-specific structures from modular components, enabling significant performance improvements of LLMs across diverse domains. TAME discovers high-performing configurations by selecting between baseline and hierarchical structures, selectively incorporating specialised modules, and fine-tuning each component through systematic mutations. Evaluating TAME across the BALROG benchmark, TAME discovers high-performing architectures that deliver substantial gains: Gemini-2.0-Flash improves from 27.16\% to 35.05\%, while GPT4.1-nano rises from 9.91\% to 17.20\%. Moreover, these structures demonstrate good transferability for larger models of the same family. Transfering these architectures to Gemini-2.5-Pro, we achieve new state-of-art performance on BALROG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes TAME, a framework that uses a genetic algorithm to evolve modular architecture for LLM agents in gaming environments. Each agent (member) is equipped with varying number of activated modules and the population of agents evolves to keep the best ones. The experiment results on BALROG benchmark show that TAME outperform BALROG's original method on various tasks.

### Strengths
1. Using genetic algorithms for dynamic configuration of agent architectures is interesting.
2. The experiments show significant improvements of TAME's performances on various gaming tasks.

### Weaknesses
1. The writing is hard to follow. Many terms are interchangeably used in different contexts (like "modole", "framework"). 
2. Lacking crucial implementation details for the genetic algorithm, such as the precise encoding of the genome, the specific mechanics of genetic operators (selection, crossover, mutation), and the calculation of diversity metrics.
3. TAME framework resembles "evolutionary hyperparameter optimization", but it regards the number and composition of modules of an agent as part of the parameters. This core idea is just using more trails (agents with differnt genomes) to find the "optimal hyperparameter setting". It seems to be resource-heavy and difficult to apply in production.

### Questions
1. How does this work's core idea differ from existing relevant work? The authors are suggested to provide detailed comparison and connection between this work and existing work in Section 2 (Related Work) to better position their novelty.
2. What are the details of evolutionary algorithm?
3. The authors are suggested to present the workflow of TAME framework more clearly in Figure 2. It would be better to show the initial population where each member is one agent with different modules, and how the population and parameters evolves in each generation. 
4. The authors are suggested to polish the writing to avoid ambiguity, particularly by distinguishing a single agent’s framework and modular architecture from TAME’s evolutionary framework. In many places, these two concepts are used interchangeably.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a new evolutionary method for augmenting the capabilities of LLMs to play videogames in the BALROG benchmark. The authors define a set of high-level modules (e.g. long-term memory or explicit exploration) and use an evolutionary approach to select which modules are deployed (along with relevant hyperparameters and modifications to the prompt) for each game in the benchmark. The authors demonstrate that this approach significantly improves the performance of the Gemini 2.0-Flash model. In addition, the authors show that adapting the best genotype found for the Gemini 2.0-Flash model to other models results in zero-shot performance gains, including a new SOTA for the BALROG benchmark.

### Strengths
The core method of the paper (evolutionary optimization to determine which modules are most appropriate for a given game) seems both novel and reasonable. The paper is also quite thorough, with a variety of ablations and hyperparameters -- I feel that the reproducibility of the experiments is high. Barring the caveats described below, I think the impact of the paper could also be high.

### Weaknesses
My primary concern with this paper is in the comparisons to baselines. The performance gains are indeed impressive, but at present it’s somewhat difficult to tell how much of the improvement is attributable to the evolutionary search procedure and how much is the result of the various modules simply causing the LLMs to “reason more” than the baseline prompts. The prompts in the original BALROG paper appear to be quite simple (i.e. just stating that the LLM is a player and enumerating the valid actions). I think the paper would benefit from an additional baseline which introduces more reasoning but perhaps without the fully decomposed module structure (or an explanation of why the original BALROG prompts act as a fair baseline).

I also think that the clarity of the paper could also be improved. There are a few technical terms which are used but not introduced (e.g. “wheel selection” or options). I also found Table 1 confusing at first -- the “Full Pop. Score” column seems like it could be referring to the gains of the whole TAME population over the baseline LLM instead of the gain of the TAME + adaptation model over the TAME[full] model, since “full” is a somewhat overloaded term. I also think it’s more common to state the performance gain in terms of percentage points (i.e. 34.7 - 27.2 = 7.5%) as opposed to percent improvement (i.e. 34.7/27.2 ~= 1.28). Relatedly, it’s not clear if the “+12.18%” gain of TAME+adaptation over TAME[full] is a raw percentage point increase or another percent improvement measure and it would be good to clarify (perhaps by simply including the raw performance of the TAME[full] model).

While these points affect my rating, I would be happy to increase my score if they are addressed.

### Questions
- How much of the gain in performance over the BALROG baseline is attributable to more reasoning or longer prompts as opposed to the specific modules selected?
- Line 240: what is “wheel selection”?
- There are two different citations to Eureka -- (Ma et al. 2023) and (Ma et al. 2024)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the poor performance of Large Language Models in complex, interactive gaming environments, such as those in the BALROG benchmark. The authors introduce the Task-Adaptive Modular Emergence framework, which employs a genetic algorithm to automatically discover effective, environment-specific agentic structures. TAME evolves a genome that specifies which human-designed modules to activate, their hyperparameters, and their prompts. The core contributions are: (1) the TAME framework itself; (2) a novel, efficient long-term memory system; (3) achieving SOTA performance on the BALROG benchmark by improving a baseline Gemini 2.0-Flash score from 27.15% to 34.77% ; and (4) demonstrating that these evolved structures are transferable, enabling a Gemini 2.5-Pro model to achieve a new SOTA score (47.65%).

### Strengths
1. The paper is well-organized and uses easy-to-understand language.

2. The paper provides rich implementation details in the appendix, which is commendable and crucial for reproducibility.

3. Rigorous ablation and analysis.

### Weaknesses
1. Limited novelty: Evolutionary algorithms have already been applied in the field of agent optimization/search, for example, in AgentSquare[3] and EvoAgent[4].

2. Poor scope/generalizability of the framework: The framework proposed in this paper is applied to the domain of interactive games. In contrast, related agent optimization/search works, such as Aflow[2] and MaAS[5], can be applied across multiple domains.

3. Insufficient discussion of related work: The core ideas of this paper, including agent evolution, evolutionary algorithms, merge components, and mutation modules, are all highly related to works like ADAS[1], AFLOW[2], AgentSquare[3], and MaAS[5], yet the paper does not discuss them.

4. Limited experiments: The paper is only evaluated on the BALROG benchmark. To my knowledge, other benchmarks for interactive games exist, such as Minecraft.

Reference

[1]Zhang J, Xiang J, Yu Z, et al. Aflow: Automating agentic workflow generation[J]. arXiv preprint arXiv:2410.10762, 2024.

[2]Hu S, Lu C, Clune J. Automated design of agentic systems[J]. arXiv preprint arXiv:2408.08435, 2024.

[3]Shang Y, Li Y, Zhao K, et al. Agentsquare: Automatic llm agent search in modular design space[J]. arXiv preprint arXiv:2410.06153, 2024.

[4]Yuan S, Song K, Chen J, et al. Evoagent: Towards automatic multi-agent generation via evolutionary algorithms[J]. arXiv preprint arXiv:2406.14228, 2024.

[5]Zhang G, Niu L, Fang J, et al. Multi-agent architecture search via agentic supernet[J]. arXiv preprint arXiv:2502.04180, 2025.

### Questions
1. Can experiments be conducted on other benchmarks?

2. Can experiments be conducted on models other than the Gemini series (e.g., the GPT series, open-source models)?

3. Can the authors provide a detailed explanation of the differences from related work?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents TAME (Task-Adaptive Modular Emergence), a framework based on genetic algorithms that evolves modular architectures for large language model (LLM) agents in interactive games. Each agent configuration comprises module combinations, hyperparameters and prompts, which are optimised through evolutionary selection and mutation. When evaluated on the BALROG benchmark, TAME was found to significantly improve LLM performance (e.g. Gemini-2.0-Flash: 27.16% to 34.78%), and demonstrate cross-model transferability. This study contributes a novel task-adaptive agent framework and an efficient long-term memory design, as well as providing empirical evidence that modular adaptation can enhance LLMs' reasoning and planning abilities in complex environments.

### Strengths
S1. The paper introduces a modular emergence framework driven by genetic algorithms that automatically configures LLM agent architectures based on task environments. This innovative approach combines evolutionary search, modular agent design, and prompt optimisation, offering a novel way to extend LLM adaptation beyond prompt tuning or static scaffolds. 
S2. The paper presents a thorough analysis of experiments conducted on the BALROG benchmark, demonstrating consistent enhancements across various game environments and models. Demonstrating cross-model transferability, where evolved architectures generalise from Gemini-2.0-Flash to Gemini-2.5-Pro, supports the robustness and practical value of the approach.
S3. The paper clearly articulates the design of each module and the evolutionary process, supported by well-structured figures and ablation studies. Its findings make a significant contribution to the emerging field of LLM-based agent architecture search by offering insights into how adaptive structural configurations can improve reasoning and planning in dynamic environments.

### Weaknesses
W1. Limited conceptual novelty: Although the paper presents TAME as an emergent intelligence framework, its core mechanism essentially involves evolutionary search over prompt and module configurations without introducing any new fundamental learning principles. Therefore, the originality lies more in system integration than in theoretical advancement.

W2. Dependence on hand-crafted modules and uncertain generalisation: TAME' s adaptive capacity is limited by a modular library designed by humans. Each component (e.g. memory, exploration and the amygdala) contains pre-defined functional assumptions. Consequently, its success may hinge on the designer's prior knowledge and understanding of the domain. While this approach is effective in game-based environments, it remains unclear whether such a manually curated framework can be applied to other games and other reasoning or planning domains beyond games.

W3. Evaluation metrics focus narrowly on progression scores: The empirical evaluation primarily relies on game progression percentages as the fitness signal. While this metric captures task success, it does not capture reasoning quality, sample efficiency or adaptation dynamics. Consequently, it is unclear whether the observed performance gains reflect genuine improvement in reasoning or merely the exploitation of heuristic patterns.

### Questions
Q1. The paper frequently refers to 'modular emergence'. How do the authors formally define or measure emergence in this context, besides the performance improvements discovered through genetic search?
Q2. What is the approximate computing budget required for one evolutionary run on BALROG? Are there any efficiency-improving mechanisms, such as early stopping, surrogate evaluation or population pruning?
Q3. Could the authors provide qualitative or behavioural analyses demonstrating why certain module combinations outperform others? 
Q4. What were the reasons for the genetic algorithm being selected over other architecture or prompt optimisation approaches?
Q5. Will the authors release the full codebase? And the configuration files?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents **TAME (Task-Adaptive Modular Emergence)**, a method that uses a genetic/evolutionary search over a pre-designed module space (e.g., hierarchical planner, explorer, long-term memory, loop-detection) to discover an agent “genome” — a combination of modules, prompts, and hyperparameters — tuned for a given game environment. The approach is evaluated on the BALROG benchmark; the authors report improvements over a baseline LLM (e.g., Gemini-2.0-Flash) and claim a transferred genome produces SOTA results when applied to a stronger model (Gemini-2.5-Pro). The paper also provides ablations on the memory module (vs. Jarvis / A-MEM) and includes evolved genome JSONs in the appendix.

### Strengths
1. **The research is practically relevant**. Automating discovery of modular agent architectures for interactive game tasks is a worthwhile goal for agentic LLM research.
2. **The engineering is reasonably complete**, proposing a concrete genome encoding (modules + hyperparams + prompts), evolutionary procedure.
3. The paper **includes targeted ablations** (memory module comparisons) rather than only reporting final aggregate scores, indicating attempts to analyze component contributions.

### Weaknesses
1. **Stability and statistical rigor are weak.** The evolutionary search uses very small budgets (ngen = 4, nchild = 5) and some evaluations use few episodes (e.g., NetHack/NLE evaluated with 5 episodes), making results noisy and potentially non-robust. The paper lacks multiple independent runs, confidence intervals, and statistical tests.
2. **The claim of transferability is under-supported.** The transfer experiment that moves a genome from Gemini-2.0 to Gemini-2.5-Pro and claims SOTA lacks tests on models from different families; the result could be specific to closely related models rather than generally transferable. Full leaderboard context and uncertainty estimates are missing.
3. **Missing budget-matched baselines.** The paper does not compare against budget-matched RL agents or other automated search methods (Bayesian Optimization, Population-Based Training, Evolution Strategies) under comparable compute budgets, leaving open whether evolutionary search is the best choice.
4. The paper **claims benefits from long-term memory** for long-horizon tasks, but provides **no qualitative retrieval to decision case study or horizon-sensitive ablation** to show how memory changes behavior.

### Questions
1. How does performance scale with search budget? Report results for at least two larger budgets (e.g., ngen ∈ {4, 8, 16} and nchild ∈ {5, 10}) and state whether performance consistently improves, saturates, or degrades.
2. Can you evaluate transfer across model families, applying the same evolved genome to at least one model from a **different family** (e.g., an OpenAI, Anthropic, or an open-source LLM). Report per-environment performance and compare to the original target model.
3. Show per-method learning curves and final performance ± CI under a matched budget. If Genetic Algorithm still performs best, explain WHY (e.g., better parallelism, robustness) rather than relying on single-number superiority.
4. Please include one retrieval→decision trace and a simple ablation by episode length.

### Soundness
2

### Presentation
3

### Contribution
2
