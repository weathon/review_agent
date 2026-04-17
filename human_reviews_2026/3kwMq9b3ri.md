# CEDAR: Agent‑Orchestrated Tree Search for Goal‑Directed Optimization of Complex Systems

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Complex systems modeling analyzes nonlinear, feedback-driven phenomena from population dynamics to economic policy, supporting decisions with significant societal impact.
In established practice, models are often authored in specialized system-dynamics languages (e.g., DYNAMO, STELLA) that specify the models' structure.
However, building and refining such models requires extensive manual effort due to (1) the opaque relationship between the structure and emergent behavior and (2) the labor-intensive workflows imposed by these languages.
These barriers limit adoption and hinder effective decision-making.
To address these challenges, we introduce \method (\methodlong), an autonomous method that uses LLM (Large Language Model) agents to discover and improve complex systems that satisfy user-specified goals.
Our key innovation is an LLM-driven MCTS (Monte Carlo Tree Search) process deeply coupled with complex system} at each iteration, an LLM Judge evaluates performance against goals and an LLM Editor proposes improved system variants.
We represent systems using a restricted, runnable subset of Python with domain-specific primitives, enabling LLMs to meaningfully and modify system dynamics directly.
CEDAR is theoretically designed with formalization in mind, and empirically enables automatic optimization of vague goals, thereby reducing human effort while achieving capabilities beyond existing approaches.
The unified design handles diverse systems across domains, constructing complex systems that would otherwise require extensive manual fitting.
Moreover, by using LLMs to interpret systems, \method makes system design transparent and accessible, facilitating broader adoption of complex systems modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces CEDAR, a LLM + MCTS–inspired procedure to iteratively generate models of complex dynamical systems toward satisfying (potentially vague) user-defined goals. The method is well described, there are several experimental results, and the work seems largely reproducible. However, to meet the acceptance bar, the theoretical justifications should be further developed, as detailed in the weaknesses and questions section below.

### Strengths
- The motivation of the work is important: Optimising complex systems under natural language goals 

- The background and related work section covers the contemporary work

- The presentation throughout the paper is clear, with good diagrams and visualisations

- There are several included experiments covering optimising (under vague goals and direct ground truth) and interpretability.

### Weaknesses
The main limitation of the work is the lack of theoretical depth and justifications. While the searching algorithm is clearly tree based, with selection, expansion, evaluation, and backpropagation steps, the connection to true monte carlo sampling and UCT is not clear. Does the approach inherit any of the desirable characteristics from UCT? How does using the uniform expansion strategy with different prompts relate to more classically balances between exploration and exploration?  A few of these decisions seem quite adhoc, and discussion of alternatives and tradeoffs is warranted.

Additionally,  the key aspect of the work is on the tree search component, but it is not clear how this distinguishes itself from the other LLM + MCTS works (which are cited) besides a different application area. 

These weaknesses are detailed more specifically in the questions section.

### Questions
**Algorithm**: 
- From an algorithmic perspective, what is novel between CEDAR and the many other LLM+MCTS methods mentioned? Why are these not used as comparisons?

- What is "f" in the SCORE(v) equation? 

- Why the hard threshold tau, and not a more explicit balance between explore/exploit as in traditional MCTS?

**Experiments**:

- In the comparisons, why does the baseline (with full formula) not perfectly recreate the observed dynamics? How many samples/runs are used in the baseline versus the proposed? Are the number of samples equated, and in the limit of infinite runs would the baseline converge to the true dynamics. Are you guaranteed such results with your expansion strategy?

- The L1 distance seems a relatively poor measure for system dynamics, as indicated by the lowest L1 for CEDAR but very flat dynamics not capturing any of the peaks and dips (which I imagine is seasonality?). Alternative metrics, e.g. Dynamic Time Warping should be used.

- In Sec 4.2: There are no comparisons in Figure 4. What is the baseline system here? v0? At minimum it would be good to see one optimising each goal on the others to show the tradeoff, e.g. optimise for population and plot that curve on the resources and pollution plots to show the impact.

Small suggestions, but 4.4 would be better made as a subsection of 4.2 for clarity/flow.

### Soundness
2

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
This paper introduces CEDAR, an autonomous approach for discovering and optimizing complex dynamical systems according to user-specified, often vague, goals. The method orchestrates large language model (LLM) agents through Monte Carlo Tree Search (MCTS): an LLM Editor proposes new system variants in a Python-based language; an LLM Judge evaluates and provides structured analysis for each candidate. This agent-driven search iteratively refines models represented in a restricted subset of Python, aiming to maximize alignment with complex system goals (expressed in natural language). Extensive experiments demonstrate CEDAR’s ability to optimize classic system dynamics models for both vague and concrete objectives, even outperforming conventional black-box optimizers. The framework also provides interpretable optimization trajectories via LLM-based reasoning.

### Strengths
1. By leveraging a restricted, runnable subset of Python combined with domain-specific primitives, CEDAR balances LLM accessibility, interpretability, and executability. The design choices here (see Figure 2 and associated discussion on Page 4) mitigate the deployment friction inherent in domain-specific or proprietary modeling languages.
2. The paper proposes a non-trivial integration of MCTS and LLMs (Figures 1 and 3), going beyond prior LLM-in-the-loop designs by embedding system refinement into a search tree structure that supports both exploration and solution diversity.
3. CEDAR is shown to optimize systems for under-specified or linguistically-expressed desiderata. The qualitative walkthrough (Figure 4) demonstrates that the MCTS+LLM architecture can robustly move systems toward multi-objective tradeoffs (e.g., balancing population, resources, and pollution).
4. The inclusion of LLM Judge analyses (Table 2, Section 4.4, and full LLM transcripts in Appendix F) makes the system’s reasoning transparent and highlights the possibility for meaningful human-in-the-loop understanding.
5. Section 4.5 and Figures 6–7 empirically illustrate that MCTS in CEDAR yields a diversity of high-performing solutions (important for sensitivity analysis and model robustness) and outperforms both linear search and black-box optimizers, especially when formulaic structure is absent (Table 1, Figure 5).

### Weaknesses
1. While the experiments compare CEDAR to black-box optimizers like Optuna under varying levels of information (Table 1, Section 4.3), critical baselines from related advances in LLM-driven agentic optimization are missing. There is no direct empirical comparison with other recent MCTS+LLM or LLM-in-the-loop discovery methods, such as those for catalyst design, or model-based RL approaches—this weakens claims of broad superiority.
2. Although implementation details and code snippets are present (Appendices, system code), many critical components (e.g., full prompt engineering, LLM API specifics, and system variants) are relegated to the appendix. Practical replication still depends heavily on access to the same LLMs and supporting infrastructure, potentially impeding reproducibility and future direct comparison.
3. The formalization of the MCTS+LLM process (Algorithm 1, Section 3.3) is clear, but there is a notable absence of formal analysis regarding convergence, optimality, or search coverage. The paper lacks discussion and evidence as to under what conditions the agentic LLM+MCTS cycle is guaranteed to discover high-quality or globally optimal solutions. For example, there is no theoretical guarantee that the LLM Editor’s system modifications—guided by semantic/natural-language prompts—preserve stability, constraint satisfaction, or do not introduce pathological dynamics, even though users are cautioned about variable explosion (Appendix D). Specifically, while the iterative integration and time discretization are defined well, the LLM-directed optimization never formalizes an explicit objective function or loss landscape (beyond the scoring mechanism). There is insufficient detail on how ill-posed/inconsistent scoring is handled—e.g., in cases where natural-language goals are ambiguous, or simulation outputs are non-comparable.
4. The experimental section mentions 20 systems but only deeply analyzes a handful (primarily the “World Dynamics” model). More diversity in task domains (e.g., economic, epidemiological, robotic control systems) would bolster the generality claim.

### Questions
1. How robust is CEDAR to goal ambiguity or conflicting objectives in natural language input? Specifically, could the authors provide systematic empirical results (perhaps via synthetic noise or adversarial phrasing) showing the stability or variability of the MCTS+LLM optimization outcome?

2. Could the authors clarify how failure cases are handled when the LLM Editor proposes system modifications that violate implicit constraints, cause instability, or “break” the simulation process? Is there a mechanism to reliably recover or backtrack in these cases?

3. Table 1 only presents mean L1 distances; could the authors provide variance/error bars or multiple run statistics to clarify performance robustness of both CEDAR and Optuna comparisons?

4. In Section 4.5 and Figure 6, diversity of solutions is highlighted. To what extent does this diversity translate into improved decision-making or real-world impact? Is there a way to quantify “useful diversity” in the solution set?

5. How does computational cost (in terms of wall-clock, sample efficiency, or compute budget) scale with increased complexity, LLM size, or tree depth? Would the method remain feasible as system dimension increases?

6. Are there specific examples where the LLM Judge’s scoring diverged from human expert judgment, or produced non-monotonic rankings due to prompt or context instability? Would a hybrid human-in-the-loop variant be more reliable?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CEDAR, a method that combines Monte Carlo Tree Search (MCTS) with Large Language Models to automatically discover and optimize complex systems (e.g., population dynamics, resource allocation) that satisfy user-specified natural language goals. The key innovation is representing systems as restricted Python code that LLMs can interpret and modify, with an LLM Judge evaluating performance and an LLM Editor proposing improvements at each MCTS iteration. The authors demonstrate CEDAR's ability to optimize vague goals and construct complex systems without extensive manual fitting, showing improvements over black-box optimization baselines.

### Strengths
1. Applying LLM-driven optimization to complex dynamical systems is timely and addresses real limitations in traditional system dynamics modeling tools.

2. The paper demonstrates capabilities across multiple dimensions—vague goal optimization, concrete record fitting, interpretability analysis, and ablation studies showing MCTS benefits.

### Weaknesses
1. Comparison with recent LLM-based system modeling work (Liu et al. 2024, Luo et al. 2025) mentioned in related work would be more compelling. Right now there is only comparison against black-box optimization methods.

2. Computational costs are not reported. Each MCTS iteration requires LLM calls for editing and judging, plus system execution. How does this scale with system complexity or simulation length?

3. The paper claims to handle "vague goals" but the World Dynamics goal (Appendix D) is quite specific with three explicit objectives and a 50% change constraint. What happens with truly ambiguous or conflicting goals?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
