# OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Optimization plays a vital role in scientific research and practical applications. However, formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise.
We introduce OptimAI, a framework for solving Optimization problems described in natural language by leveraging LLM-powered AI agents, and achieve superior performance over current state-of-the-art methods.
Our framework is built upon the following key roles:
(1) a formulator that translates natural language problem descriptions into precise mathematical formulations;
(2) a planner that constructs a high-level solution strategy prior to execution; and 
(3) a coder and a code critic capable of interacting with the environment and reflecting on outcomes to refine future actions.
Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively.
Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain.
Our design emphasizes multi-agent collaboration, and our experiments confirm that combining diverse models leads to performance gains.
Our approach attains 88.1\% accuracy on the NLP4LP dataset and 82.3\% on the Optibench dataset, reducing error rates by 58\% and 52\%, respectively, over prior best results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes OptimAI, a multi-agent framework that converts natural-language optimization problems into executable solvers using large language models. The system decomposes the task into roles — formulator, planner, coder, critic, decider, and verifier — connected through a sequential workflow. A key design component is a UCB-based debug scheduler that dynamically switches between alternative solution plans when current ones fail. Experiments on NLP4LP, Optibench, and several combinatorial optimization datasets show substantial gains in success rate and productivity over prior LLM-based systems such as OptiMUS and Optibench.

### Strengths
* Clear system design and extensive evaluation across five datasets.
* Effective empirical improvements over strong baselines (OptiMUS, Optibench).
* Careful ablation studies showing the necessity of planner and critic roles.
* Demonstrates generality to NP-hard tasks and heterogeneous model collaboration.
* Good formalization of pipeline and reproducible prompts.

### Weaknesses
* most components reuse conceptually known LLM-agent design patterns (MetaGPT-style role assignment, plan-then-code prompting, reflective debugging).
* UCB scheduling is a minor adaptation of a standard bandit algorithm with no theoretical analysis or comparison to simpler heuristics (e.g., random or greedy switching).
* Baselines focus only on other LLM-based methods; missing comparisons to symbolic or hybrid optimization systems.

### Questions
* How significant is the contribution of the UCB scheduler compared to simpler plan-switching heuristics (e.g., round-robin or random)?
* Could OptimAI operate effectively with open-source LLMs, or does it rely on GPT-4-level reasoning?
* What is the computational overhead of multi-agent orchestration relative to single-model prompting?
* How is the plan count (n = 3–4) chosen, and how sensitive are results to this hyperparameter?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes OptimAI, a multi-agent framework that solves optimization problems from natural language. It has four pipelines—modeling, planning, coding, and debugging—corresponding to four roles: formulator, planner, coder, and code critic. It employs an Upper Confidence Bound (UCB) algorithm for adaptive debug scheduling, enabling dynamic plan switching and achieving superior efficiency and accuracy over prior methods.

### Strengths
1. The paper is well-structured and clearly written, making the framework and experiments easy to follow.
2. It introduces an original four-pipeline, four-role multi-agent design that effectively bridges natural language and optimization.
3. Rigorous ablation studies validate the importance of each role and the UCB algorithm.
4. The Synergistic Effects study in Section 4.3 offers valuable insights into using different LLMs for different roles, showing interesting results for multi-agent research.

### Weaknesses
1. The experiments use different LLMs across tables (e.g., Table 6 includes LLaMA and Gemini, but Table 3 does not), and GPT-4o is missing in Table 6 without explanation.
2. The performance heavily depends on which LLM is assigned to each role, yet the paper provides no systematic method for efficiently choosing them.
3. In Table 10, accuracy decreases as the number of plans increases (beyond 4), but the paper doesn’t explain how to efficiently find the optimal number or whether this reflects a fundamental limitation of the decider.
4. Table 2 lacks proper spacing: "OR-LLM-AgentZhang & Luo (2025)" should be "OR-LLM-Agent Zhang & Luo (2025)".

### Questions
1. Could the authors clarify how to decide which LLMs should be assigned to different roles in practice? Is there a systematic or automated method that could guide this process?
2. In Table 10, accuracy drops when the number of plans exceeds four. Why does this occur, and how should users determine the optimal number? Is this a fundamental limitation of the decider, and how might it be improved?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces OptimAI, a novel framework leveraging large language models (LLMs) as AI agents to solve optimization problems directly from natural language descriptions, aiming to democratize access to high-quality optimization for non-experts. The system consists of four main agent roles—formulator, planner, coder, and code critic—enabling translation from user input to executable solutions, with the option to split these roles across different LLMs for specialization. Key techniques include a plan-before-code approach, UCB-based debug scheduling (treating plan selection as a multi-armed bandit problem), and support for switching strategies during debugging, allowing adaptive exploration. Extensive experiments on datasets such as NLP4LP and Optibench demonstrate that OptimAI significantly outperforms prior art, attaining up to 88.1% accuracy and substantial reductions in error rates. Ablation studies confirm the importance of each role and the efficacy of the UCB-based scheduler, while mixing heterogeneous LLMs in different roles yields additional synergistic gains. OptimAI generalizes not only to standard mathematical programming but also to NP-hard combinatorial problems, showing broad applicability. The framework’s design is extensible and suggests future improvements in reinforcement learning integration and scaling for larger, expert-level problem domains.

### Strengths
- The explicit separation of roles into multi-agent LLM agents, with the ability to assign different models per role, is creative and not common among existing works. The UCB-based debug scheduling is a pragmatic and technically sound enhancement over naive or static plan exploration.
- Experimental rigor is evident—multiple datasets, detailed metrics, ablation of component roles, comparisons to best-known baselines, and exploration of multi-LLM synergies. Coverage of combinatorial and mathematical programming problems demonstrates true generality.
- The structure of the methodology is broken down clearly, with explicit agent responsibilities and stepwise explanation. Empirical results are presented in comprehensible tables with direct quantitative comparisons.
- OptimAI’s accuracy and productivity improvements are large, well-documented, and seem robust across several classes of problems. The demonstration that LLM multi-agent collaboration confers measurable gains over single-model or single-agent setups is especially impactful.

### Weaknesses
- Although a comparative table is given, more explicit explanation is needed regarding which problem types, complexities, or settings OptimAI can uniquely address where previous methods fail or underperform.
- The design and evaluation of the decider (used in UCB scheduling) is underspecified, especially regarding whether it is finetuned, trained, or used in a zero-shot/few-shot manner. This impacts both reproducibility and clarity of claimed gains.
- The reporting of variance, standard deviations, statistical significance, and resource usage is lacking. For high-impact claims, reporting these measures is a must, particularly for practical/industrial deployment concerns.
- The paper does not expand on where or why OptimAI might fail, be inefficient, or produce erroneous code/solutions (e.g., with adversarial inputs or edge case problems). More error analysis and transparency about failure patterns would strengthen the work.
- There is limited discussion of run-time and scaling trade-offs; since OptimAI can require multiple large models in sequence or in parallel, computational cost and deployment considerations should be more thoroughly addressed.

### Questions
- Critical implementation details (e.g., LLM configuration choices, prompt designs, agent interaction protocols) are said to be documented in the appendix. Could the authors summarize the most important choices and pitfalls in the main text, and clearly indicate which ones are essential for high performance? For real-world adoption, such details are crucial; deferring these entirely to appendices hinders transparency.

- The paper positions UCB-based plan switching and multi-agent role-separation as novel. Can the authors clarify how their scheme differs substantially from existing bandit-based or ensemble multi-agent setups in related literature? Has a head-to-head baseline (same agents + random/greedy switching) been run? To convincingly demonstrate necessity, it is important to distinguish incremental from substantial innovation.
   
- What are the specific architecture, training regimen (if any), and evaluation details for the decider agent used in UCB-based debug scheduling? Was any reinforcement learning or fine-tuning employed, or is it zero/few-shot? The decider's quality is central to the method's efficiency, but its specification is vague, which may limit the clarity of the ablation and tuning results.

- Could the authors report practical details (e.g., runtime per instance, GPU/CPU usage, memory requirements) for running OptimAI, especially when using multiple large models, and discuss scaling concerns? Practical deployability and resource efficiency are important for real-world utility, but such trade-offs are not quantified.

- Could you clarify the data splitting procedures (train/validation/test) for each benchmark, and whether any hyperparameter tuning or prompt engineering was performed on test data? Ensuring fair, reproducible, and unbiased evaluation mandates explicit reporting of experiment splits and prevents potential data leakage or overfitting.

- What specific statistical methods (if any) were used to assess significance or variance (e.g., multiple seeds, confidence intervals, standard deviations), especially for accuracy improvements and productivity metrics? The paper highlights substantial improvements, but without variance or statistical significance reporting, it is difficult to assess the robustness of the reported gains.

### Soundness
3

### Presentation
3

### Contribution
3
