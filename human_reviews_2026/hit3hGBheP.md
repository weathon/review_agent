# AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms

- Decision: Accept (Oral)
- Scores: 6, 6, 6, 8

## Abstract
Dynamically configuring algorithm hyperparameters is a fundamental challenge in computational intelligence. While learning-based methods offer automation, they suffer from prohibitive sample complexity and poor generalization. We introduce AutoEP, a novel framework that bypasses training entirely by leveraging Large Language Models (LLMs) as zero-shot reasoning engines for algorithm control. AutoEP's core innovation lies in a tight synergy between two components: (1) an online Exploratory Landscape Analysis (ELA) module that provides real-time, quantitative feedback on the search dynamics, and (2) a multi-LLM reasoning chain that interprets this feedback to generate adaptive hyperparameter strategies. This approach grounds high-level reasoning in empirical data, mitigating hallucination. Evaluated on three distinct metaheuristics across diverse combinatorial optimization benchmarks, AutoEP consistently outperforms state-of-the-art tuners, including neural evolution and other LLM-based methods. Notably, our framework enables open-source models like Qwen3-30B to match the performance of GPT-4, demonstrating a powerful and accessible new paradigm for automated hyperparameter design.Our code is available at https://anonymous.4open.science/r/AutoEP-3E11.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new framework called AutoEP, which aims to optimize the performance of metaheuristic algorithms by using LLMs to automatically adjust the algorithm's hyperparameters. The main contribution of the paper is the proposal of a novel framework that does not require training and can dynamically adjust the algorithm's hyperparameters through the zero-shot reasoning ability of LLMs.

### Strengths
1) The paper is organized and easy to follow, with a clear presentation of the problem, methodology, and results.
2) The experiments are thorough, using multiple benchmarks and reliable comparisons to validate the proposed method.
3) The method of using LLMs for zero-shot reasoning to adjust hyperparameters is novel and offers a fresh perspective in automated algorithm design.

### Weaknesses
1) Dependency on LLM Performance: AutoEP’s effectiveness relies on the capabilities of the underlying LLM, which may vary in accuracy and efficiency, affecting consistency.
2) The current tests focus on standard benchmarks. Broader testing on more complex or dynamic problems would offer better insights into its robustness and scalability. Some Case study will be good.

### Questions
Just a kind discussion:
In this framework, do you think the algorithm relies more on the reasoning ability of the LLM, the experience (historical knowledge of the search space), or memory (rote learning)?
My essential question is whether the algorithm can solve a completely new combinatorial optimization task (one that is not TSP, for example) by solely depending on its reasoning ability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AutoEP, an LLM-driven framework for automating hyperparameter evolution in metaheuristic algorithms.The core idea is to replace static control rules or reinforcement-learning-based policies with quantitative reasoning driven by LLMs, thereby enabling adaptive, interpretable hyperparameter control without additional training.AutoEP incorporates an exploratory landscape analysis (ELA) module that extracts numerical descriptors from the ongoing optimization process, and a chain-of-reasoning (CoR) framework consisting of three agents (Strategist, Analyst, Actuator) to translate ELA features into parameter updates.
Experimental results across multiple metaheuristic algorithms (GA, PSO, ACO) and optimization problems (TSP, CVRP, FSSP, UAV trajectory optimization) demonstrate that AutoEP improves performance and runtime efficiency compared to classical adaptive heuristics and recent LLM-enhanced approaches.
Ablation studies further support the importance of ELA grounding and CoR decomposition.

### Strengths
1. Introducing an ELA-grounded, LLM-reasoning framework for online parameter control is conceptually new and methodologically sound.
2. The CoR structure (Strategist–Analyst–Actuator) provides interpretability and modularity while reducing reliance on very large proprietary models.
3. The paper evaluates across four distinct problem domains and three algorithmic backbones, demonstrating versatility. Ablation and sensitivity analyses are included and substantiate the key design choices.

### Weaknesses
1. Section 3.2 states that all historical states, actions, and outcomes are stored in the Experience Pool.While this guarantees completeness, it also raises concerns about prompt bloat and redundant context as optimization proceeds over hundreds of generations.In large-scale or long-horizon runs, this could degrade inference efficiency and increase context noise for the LLM.
2. The framework claims that the CoR pipeline ensures stable exploration–exploitation balancing, but no mathematical or empirical justification is provided for convergence.
3. Visualizing how LLM decisions evolve (e.g., parameter adjustment curves or attention to specific ELA features) would substantiate the interpretability claim.
4. AutoEP is compared to ReEvo and EoH but not to more recent LLM-based black-box optimizers such as EvoLLM.

### Questions
1. Since all historical records are kept, how large does the Experience Pool grow during long optimization runs, and how is memory or context length managed?
2. Is there any mechanism (e.g., damping or adaptive scaling) to prevent over-adjustment of hyperparameters?
3. Can AUTOEP adapt to new algorithm families (e.g., DE or CMA-ES) without additional human intervention?
4. Could you report sensitivity analyses for Experience Pool size and CoR decision frequency?
5. Could you also report per-component latency breakdown (Strategist/Analyst/Actuator) to substantiate the 30 ms decision claim?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a LLM-based algorithm configuration framework with reasoning chain. The decision of configurations is decomposed into a multi-step reasoning chain. Starting from a strategist LLM which analyses the impact of the hyperparameters of the given algorithm, LLM analyses the current optimization status from ELA features, determine to enhance exploration or exploitation, and modify the algorithm hyperparameters to achieve that. Experimental results show that the proposed method surpasses RL-based algorthm configuration baselines and LLM-based metaheuristic methods on TSP, CVRP and UAV trajectory optimization tasks.

### Strengths
1. The algorithm configuration process is decomposed into several steps with detailed descriptions, which significantly enhances interpretability. Users can deeply analyze why the LLM makes a particular decision and how the framework achieves SOTA performance, providing insights for future algorithm design.

2. The plug-and-play approach is adaptive and generalizable, enabling application across broader optimization scenarios and enhancing the interpretability and performance of LLMs in diverse applications.

3. The experimental results validate the SOTA performance of the proposed method on three combinatorial optimization tasks.

### Weaknesses
1. The strategist LLM requires users to provide the name of the target algorithm and its hyperparemeters. It demands users' expert knowledge requirement on the target optimization tasks. For optimization tasks where users lack familiarity or no known promising algorithms exist, specifying a suitable algorithm becomes challenging.

2. Furthermore, the algorithmic knowledge embedded in LLMs is inherently limited and potentially outdated. Consequently, LLMs may struggle to accurately analyze the qualitative effects of hyperparameters in complex, advanced algorithms, which could hinder the framework's generalization across diverse optimization tasks.

### Questions
1. The feature description explicitly instructs the LLM to explore or exploit based on whether feature values are high or low. If we already possess this deterministic knowledge about when to increase exploration or exploitation, why not replace this component with a simpler, more efficient machine learning model, such as a decision tree?

2. The feature descriptions and experience pool appear to consume a significant number of tokens. How is the average inference latency per decision maintained within 30 ms despite this potential computational overhead?

3. In Figure 4, AutoEP with different base LLMs demonstrates nearly identical performance. Does this imply that different LLMs are making the same configuration decisions? Why would LLMs of varying scales and capabilities take identical actions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work describes a novel idea for dynamic algorithm configuration in optimization. The authors are motivated by the limitation of human-crafted configuration rules and generalization bottoleneck of latest learning-assisted approaches. To address these technical issues, the authors propose leveraging LLMs (with vast knowledge) to kick a balance between the configuration effectiveness and generalization. Specifically, the overall framework comprises a ELA calculator and a multi-role LLM system. Given an optimizer and corresponding optimizee at hand (e.g., GA for solving TSP), the ELA calculator summarizes per step dynamic landscape information and feeds such information into the multi-role LLM system. Then the LLM system analyse such information and dictates proper algorithm configuration for the optimizer to control the exploration and exploitation strength. Through extensive experiments, the effectivess of the proposed framework is validated.

### Strengths
1. Novelty: While some previous works may also considered using LLM for algorithm configuration or selection such as LLaMoCo (implicit and static algorithm configuration through code generation, https://arxiv.org/pdf/2403.01131) and AS-LLM (training algorithm selection mapping through LLM, https://openreview.net/forum?id=l7aD9VMQUq), I have to acknowledge this paper as a novel perspective, with reasonable usage of LLMs to relieve human efforts required for tuning optimization algorithms. Such novelty roots from the reasoning ability of existing LLMs.

2. Systematic Design: I think the most important and significant contribution of this paper is that it demonstrates that through ELA anaysis, the strength of LLM and the performance of an optimizer can be bridged in a dynamic algorithm configuration fashion.

3. Significance: the experiments results are solid, and especiall compared with latest EoH and ReeVO (LLM-based AAD methods), the performance gain is good.

### Weaknesses
Besides the strength of this paper, I wanna list several concerns.

1. Efficiency: I understand that using LLM to provide ideal performance is already a breakthrough for automated algorithm configuration. However, I suggest the authors carefully claim their approach's superiority. Here, an interesting question is since AutoEP has to use LLM for reasoning, why the running time is still comparable with GA itself?

2. The experience pool compreises per step action-reward entries, while it may provides certain heuristic on how to choose parameter value, it is still unclear why such myopia information could help AutoEP bypass RL-based approach such as GLEET, since in RL-based approaches, the bellman updates ensure the policy to learn long-term decision. This deserves in-depth investigation.

3. The underlying optimzers are restricted within simple optimizer such as GA, PSO.... This might makes readers curious about the true potential of AutoEP on more complex optimizer with complex configuration space.

4. The related work could be polished and refined further. Can the authors use Meta-Black-Box Optimization to denote learning-assisted or data-driven or reinforcement learning-based approaches? since recently these methods have been summarized and termed by this terminology. Also consider reviewing latest works in  Meta-Black-Box Optimization (surveys and papers in 2025).

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
