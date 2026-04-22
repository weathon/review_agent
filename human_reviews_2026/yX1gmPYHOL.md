# Improving LLM Symbolic Problem-Solving via Automated Heuristic Discovery

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 6, 4, 4

## Abstract
We consider enhancing large language models (LLMs) for symbolic problem-solving tasks. While existing inference-time techniques let LLMs explore intermediate steps for problem-solving, they rely on noisy self-verification or external verifiers, which demand significant data and computations. Here, we propose Automated Heuristics Discovery (AutoHD), a novel approach that enables LLMs to generate heuristic functions to guide inference-time search through accurate evaluation of intermediate steps. These heuristic functions are further refined through an evolution process, improving their robustness and effectiveness. Our method requires no additional model training or fine-tuning, and the explicit definition of heuristic functions provides interpretability and insights into the solving process. Extensive experiments on diverse tasks demonstrate significant gains over multiple baselines, including nearly twice the accuracy on some tasks, establishing AutoHD as a reliable and interpretable solution.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Automated Heuristic Discovery (AutoHD), a novel inference-time framework designed to enhance symbolic problem-solving using large language models (LLMs). While prior methods such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT) enable intermediate-step exploration, they depend on unstable self-verification or costly external evaluators, often introducing noise and computational burden. AutoHD instead allows the LLM to generate its own heuristic functions—explicitly defined as executable Python code—to evaluate intermediate states during search, and iteratively refines these heuristics through an evolutionary process, improving robustness and accuracy. Without any fine-tuning or extra data, AutoHD produces interpretable, reliable reasoning across tasks like Blocksworld, Game of 24, and Rubik’s Cube, achieving up to 2× accuracy improvements over strong baselines.

### Strengths
Conceptual novelty: AutoHD moves beyond text-based reasoning chains by introducing explicit heuristic-driven search, letting LLMs quantify and guide their own symbolic reasoning.

Practical design: The framework operates entirely at inference time—no gradient updates, no retraining—making it lightweight and adaptable across tasks.

Interpretability: Each heuristic is represented as readable Python code, exposing the model’s decision logic and allowing transparent inspection of intermediate reasoning.

Evolutionary improvement: The iterative refinement of heuristics through evaluation and mutation leads to steadily improved accuracy and stability, reducing dependence on initial prompt quality.

Empirical coverage: Consistent superiority across multiple LLM scales and benchmarks—particularly in Rubik’s Cube (+6–15%) and Game of 24 (+25%) over ToT—validates its general effectiveness.

Strong preparation and reproducibility: Appendices detail all prompts, heuristic examples, and ablation setups, reflecting excellent completeness and methodological rigor.

Practical relevance: The work bridges symbolic AI and neural reasoning, providing a clear formalization of state-space search within LLM inference.

### Weaknesses
Ablation depth and integration:

- The paper lacks a consolidated component-wise ablation quantifying contributions of heuristic generation, heuristic-guided search, and heuristic evolution.
- Results are distributed across appendices rather than in the main text, making causal interpretation difficult.

Efficiency and computational overhead:

- The approach entails multiple LLM calls per inference and per evolution cycle; runtime and compute costs are not fully reported.
- There is no clear comparison of inference-time efficiency versus CoT/ToT under equal accuracy.

Scope and generalization:

- Only three symbolic tasks are tested. Extending to broader domains—e.g., Sokoban, text-based planning, or hybrid real-world tasks—would demonstrate stronger generality.
- Cross-task or transfer evaluations (e.g., reusing heuristics trained on one task for another) are not explored.

Comparative positioning:

- Although prior works on LLM-based automatic heuristic design (e.g., FunSearch, PRMs, LLM-PDDL) are discussed, empirical contrasts are missing.
- The paper would benefit from side-by-side comparisons highlighting differences in data requirements, compute cost, and interpretability.

Reproducibility and accessibility:

- Despite the detailed documentation, no public implementation or repository is provided. I recommend open-sourcing the code upon publication to maximize the paper’s transparency and impact.
- Sensitivity studies for key hyperparameters (generation number, pool size, rank threshold) and prompt stability are absent.

Statistical and failure analysis:

- Variance or confidence intervals are missing in main tables; failure case illustrations would clarify when heuristic evolution misguides or stagnates.

### Questions
Could the authors provide full component ablations (with/without heuristic, with/without evolution, different search algorithms) and integrate them into the main text?

How does the heuristic evolution quantitatively improve validation accuracy across generations beyond Fig. 4?

What is the runtime breakdown (number of LLM calls, total compute cost, wall-clock time) compared to CoT or ToT?

How does AutoHD scale with problem depth and branching factor in large symbolic tasks?

Can heuristics discovered in one domain (e.g., Blocksworld) transfer to related tasks or unseen instances?

How sensitive is AutoHD to prompt design and model scale (small vs. large LLMs)?

How does AutoHD differ empirically from LLM-generated planner heuristics (Tuisov et al., 2025; Correa et al., 2025) or Process Reward Models (PRMs) in terms of interpretability and inference efficiency?

Could an experiment combining AutoHD heuristics with a symbolic planner help isolate the contribution of inference-time integration?

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
3

### Summary
The paper proposes AutoHD, which lets LLMs generate explicit heuristic fns as python code. This helps guide inference time search for symbolic problems. These functions are then iteratively refined in the paper via an evolutionary approach. The paper exhibits solid improvements over 3 tasks without model training. 

The core idea is sound, but due to lack of theoretical results, generalizabiliity is questionable.

### Strengths
1. This approach is much more interpretable than black box LLM self verification methods.
2. The results are demonstrated across several tasks and LLMs, with nearly 2x improvements on some benchmarks.
3. The ablations are extensive, across equivalents like search algorithms, evolution heuristics, cost analysis, and comparison across LLMs.

### Weaknesses
1. There is no theoretical characterization of what makes a particular heuristic function good, and others bad.
2. The method seems computationally expensive since it needs multiple generations and multiple evals over validation set.
3. While the three symbolic domains chosen are interesting, it's a bit unclear if this generalizes to other symbolic domains as well.
4. The method seems too dependent on the distribution of the validation set, and does not adequately address how sensitive the approach is to quality or size of val sets. Also no empirical backing is provided for how to construct such sets in practice.

### Questions
1. How does the method perform when val set is scarce or not representative of the test data?
2. Are there biases or blind spots in the kinds of heuristics LLM generate?
3. What happens in methods where the search space is larger or the solution is a longer path?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
AutoHD is a training-free, test-time method that prompts an LLM to generate Python heuristic functions for symbolic problem solving, evolves those heuristics on a small validation set, and then guides search (greedy-BFS or A*) using the learned scores. The paper reports consistent gains on Blocksworld, Game of 24, and 2×2 Rubik’s, plus ablations on search choice, heuristic evolution, and swapping code vs LLM for action/transition modules.

### Strengths
1) Simple interpretable mechanism. Converting LLM suggestions into explicit code heuristics is simple, inspectable, and increases reliability.
2) Consistent empirical gains. The method improves over CoT/SC/ToT across tasks and beats XoT on Rubik’s in the reported setups
3) Search efficiency. Reporting states visited (not a complete cost analysis) indicates the approach prunes unpromising branches better than some baselines.

### Weaknesses
1) Baseline fairness. Baselines perform all computation using LLMs only, while AutoHD leverages external, code-based tools. A fair comparison requires search-based baselines with tool use enabled (e.g., ToT/ReAct + calculator/interpreter) under matched budgets.


2) Efficiency reporting incomplete. Results include states visited but omit end-to-end cost metrics (tokens, wall-clock, model calls) for both heuristic evolution and inference. Without these, practicality and cost–accuracy trade-offs against search-time scaling baselines remain unclear.

2) Domain dependence / prompt specificity. Appendix F prompts are highly tailored to each domain’s state encoding, indicating non-trivial overhead to adapt beyond the reported tasks. The method’s portability/generalizability is therefore limited without additional engineering.


3) Small validation for evolution. Heuristic selection on a very small validation set which can introduce selection bias, and the paper does not report run-to-run variance and stability analysis of the evolution procedure.


4) Cross-task application. The Rubik’s Cube evaluation uses a code-based transition or world model, while Blocksworld relies on LLM-generated transitions. This inconsistency in setup undermines methodological coherence and makes it unclear how the approach would generalize or be applied to a new domain.

### Questions
How were the 10 instances for heuristic evolution selected?

Do learned heuristics exhibit consistency/monotonicity along successful paths? 

See weaknesses for additional analysis requested.

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
3

### Summary
This paper proposed AutoHD, a novel approach that let LLMs to generate heuristics automatically without test-time inference, self-verification, and external model training. AutoHD shows a consistent stronger performance across multiple tasks.

### Strengths
1. Method does not require model training or self-verification
2. Consistent better performance across multiple task
3. Clearly written

### Weaknesses
### Major 
1. My impression for heuristic discovery is asking the model to analyze the current state and the target state, am i understanding it correctly? If this is the case, what's the difference between this method and some other others that decompose the problem into subtasks then aggregate the answer? For instance, when I have the task of "where did I park my car," some heuristics could be "next to a van" or "close to the entrance." Then i just continually evolve my heuristics. But what's the difference between this one and "sub problem 1: where is it next to" and "sub problem 2: is it close to the entrance?" 
2. Can you also compare token usage? 

### Minor 
1. Line 39, logic is a little bit flawed. You are saying there are many methods to improve test time scaling, such as early approach CoT. Maybe change the narrative a little bit. 
2. Line 47, "approaches that require additional model training" needs citation. 
3. Line 50, it's better to define what "heuristics" mean here in the high level. Either an examples or formal definition are fine.

### Questions
1. Page 17, prompt template for heuristic, how to generate heuristic at the first place?

### Soundness
3

### Presentation
3

### Contribution
2
