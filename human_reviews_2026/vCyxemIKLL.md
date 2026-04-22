# SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
We present $\texttt{SENTINEL}$, a unified multi-level framework for evaluating the physical safety of LLM embodied agents using $\textit{formal safety semantics}$. In our approach, safety rules are grounded as temporal logic constraints, providing precise semantics for specifying state invariants, temporal dependencies, and timing requirements. These rules enable formal checking of embodied-agent behavior at multiple stages of decision-making. $\texttt{SENTINEL}$ is organized into a progressive evaluation pipeline: at the $\textit{semantic level}$, natural language safety requirements are interpreted as Temporal Logic (TL) specifications; at the $\textit{planning level}$, high-level action programs and subgoals are checked against these TL rules before execution; and at the $\textit{trajectory level}$, multiple simulated executions are merged into planning trees and verified against more physical-detailed Computation Tree Logic (CTL) specifications. This provides a reproducible protocol for jointly measuring task completion and safety compliance. By grounding safety in temporal logic and enabling formal evaluation across semantics, plans, and trajectories, $\texttt{SENTINEL}$ establishes a comprehensive pipeline for systematically assessing LLM-based embodied-agent safety, laying the foundation for agents that are not only capable but also reliably safe in realistic environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces SENTINEL, a framework for evaluating the physical safety of LLM-based embodied agents. The authors propose a multi-level verification pipeline that operates at three levels of abstraction:Semantic-level: Evaluates an LLM's ability to translate natural language (NL) safety requirements into formal Linear Temporal Logic (LTL) formulas (i.e., $NL \rightarrow LTL$).Plan-level: Verifies if the agent's high-level action plan (a single sequence) satisfies the LTL constraints.Trajectory-level: Samples multiple execution trajectories from the agent, merges them into a computation tree, and uses Computation Tree Logic (CTL) (e.g., checking $AG(\phi)$) to verify that all possible execution paths are safe.The primary contribution is the proposal of this formal, multi-level approach, which aims to provide a more rigorous safety evaluation than existing methods that rely on heuristic rules or subjective LLM-based judgments.

### Strengths
The paper addresses the physical safety of LLM-based embodied agents. This is a critical, high-impact research area and a significant bottleneck for the trustworthy, real-world deployment of these systems.

### Weaknesses
**1. Questionable Motivation and Limited Novelty**

* **Unnecessary Complexity:** The framework's core premise of translating natural language (NL) rules into formal temporal logic (TL) adds a complex layer that may be unnecessary. This translation step itself is a significant source of failure, as shown by the paper's own semantic evaluation, and complicates the process for LLMs that are designed to reason over NL directly.
* **Insufficient Justification:** The paper fails to justify *why* this translation is necessary. It does not explain why an LLM capable of understanding complex NL task instructions cannot also be evaluated directly against NL safety constraints.
* **Unmet Burden of Proof:** The authors do not demonstrate that their complex, TL-based evaluation is more effective at detecting safety violations than a simpler, direct NL-based evaluation. The assumption that TL is superior to NL for this task remains unproven.

**2. Doubts Regarding the Methodology**

* **Decoupled Hierarchy:** The "multi-level" framework does not match an agent's runtime decision flow (Instruction $\rightarrow$ Plan $\rightarrow$ Action). "Level 1" (NL-to-TL translation) is merely an offline pre-processing step, completely decoupled from the agent's reasoning. A true hierarchical evaluation should first assess the agent's ability to identify risks from the initial instruction *before* planning.
* **Redundant Logic:** The switch from LTL to CTL is poorly justified. Verifying that all *n* sampled trajectories are safe can be achieved by *n* independent LTL checks. Merging paths into a tree for a single, more complex CTL check is logically equivalent, and its added benefit over the simpler approach is not explained.
* **Offline-Only Assessment:** The framework is a purely *post-hoc* auditing tool, not a real-time safety mechanism. It evaluates trajectories *after* they have been fully sampled, offering no capability to intervene or prevent an unsafe action as it happens, which severely limits its practical utility.

**3. Lack of Evaluation on Mainstream VLA Models**

* **Outdated Model Scope:** The evaluation is restricted to LLMs, ignoring the more relevant Vision-Language-Action (VLA) models that dominate modern robotics.  VLA models, which fuse vision into their decisions and directly output control params, will have different failure modes than text-only llms. By excluding VLAs, the paper fails to assess the framework's applicability to state-of-the-art agents or to compare safety performance across architectures.

**4. Absence of Real-World Robotic Evaluation**

* **Simulation-Only Validation:** The experiments are confined to simulation, which is insufficient for evaluating *physical* safety. This approach ignores real-world physics, sensor noise, and actuator latencies that are critical to safety.
* **Ignoring the Sim-to-Real Gap:** The paper makes no attempt to bridge the "sim-to-real gap." A plan verified as "safe" in simulation could fail catastrophically in the real world, and without validation on physical hardware, the claims about "physical safety" are not credible.

**5. Missing Critical Experimental Details**

* **Undefined Data Curation:** The paper fails to describe the *criteria* or *methodology* used to extract the "safety-related subset" of tasks from VirtualHome and ALFRED. This makes it impossible to judge if the dataset is representative or biased.
* **Unspecified Dataset Size:** The total size (N=?) of the task datasets is never reported. Without this, the reported percentage-based metrics are statistically meaningless and their robustness cannot be assessed.
* **Ambiguous Notation:** Key formal notation is left ambiguous or undefined. For instance, the definition of "C" is not clearly defined in line 140-141, creating confusion in a paper that should be formally precise.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
SENTINEL translates natural language safety rules into temporal logic and evaluates LLM agents at three levels: semantic, plan, and trajectory. Multiple simulated rollouts are merged into a computation tree and checked with branching time operators so violations surface with concrete counterexamples. The framework is demonstrated in VirtualHome and ALFRED and the authors report a clear trade off where adding safety guidance improves safety but can reduce raw task success.

### Strengths
- Formal and end to end structure: The paper grounds safety in temporal logic and walks it through semantic interpretation, plan checks, and trajectory verification so you can see where failures originate and why.
- Branching time evaluation with actionable feedback: By assembling a computation tree from many rollouts and checking CTL operators, the method reasons about multiple possible outcomes and returns counterexample paths to diagnose hazards.
- Experiments across two simulators show that structured safety prompts improve safety and reveal a safety versus success trade off that can guide tuning.

### Weaknesses
- The paper performs CTL-style checking on a computation tree assembled from a finite set of sampled rollouts, primarily using universal CTL operators (such as $A\Phi$ or $E\Phi$). This is effectively *LTL on each branch with a universal/existential aggregation operation across the sampled tree*, so it neither reasons about unseen traces nor constitutes full CTL model checking of the underlying system (i.e. to my knowledge, it omits any nested path-quantified properties).
- The final trajectory level safety evaluation shows that even when the safety prompts given in LTL, the overall safety and success rates are in the single digit percentages. While it is shown that temporal logic given as part of the prompting process helps, it is not entirely convincing that this is enough for meaningful performance.
- Important extensions are future work. The paper notes that timed and probabilistic guarantees and integration with established checkers would broaden realism and rigor but are not part of the current system.
- Minor Presentation Concerns:
    - Temporal logic and LTL used interchangeably in the text.
    - L140: $C$ and $\mathcal{T}$ mixed?
    - L285: typo “sequences in top left” → in the
    - L287: typo “trajectories tree” → “tree of trajectories” / “trajectory tree”

### Questions
1. Can this  approach and its effectiveness be compared to SELP [1] ? They solve a similar task setting given an environment and task description in natural language, an LTL specification is extracted with tools like equivalence voting which is used to generate a plan better satisfying the given constraints.
2. In the LTL-based plan-level evaluation, is tracking whether an object undergoes a state change when moving from $s_0$ to $g$ mean that we assume full observability of all objects at all times or is this inferred by an LLM for unseen objects say out of range of the robot? If the later, how are misclassifications accounted for?
3. How is the accuracy of the generated LTL predicates verified given that it is extracted from the natural language task? How often do hallucinations cause problems?

### References:

[1] SELP: Generating Safe and Efficient Task Plans for Robot Agents with Large Language Models, Wu et al., ICRA 2025

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel evaluation framework for assessing the safety of LLM-based embodied agents, with three key contributions:

1. **Formal Safety Specification**  It proposes grounding the safety specification in formal temporal logics, specifically Linear Temporal Logic (LTL) and Computation Tree Logic (CTL).

2. **Multi-Level Safety Evaluation Framework**  The framework evaluates safety from three distinct perspectives:
   
   - **Semantic Level**  
     Assesses the ability of LLM agents to understand natural language safety constraints by checking whether their generated temporal logic expressions are semantically equivalent to ground truth expressions.
   
   - **Plan Level**  
     Formally verifies that the high-level plans generated by LLMs do not violate any of the formal safety constraints.
   
   - **Execution Trace Level**  
     Evaluates whether the detailed execution trajectories of the agents satisfy all safety constraints.

3. **Experimental Evaluation**  
   A detailed experimental study highlights the strengths and weaknesses of state-of-the-art LLMs in adhering to safety constraints.

### Strengths
1. **Relevance**
The paper addresses a critical challenge: assessing the safety of AI systems (in particular they ability to understand and follow safety constraints).

2. **Significance and Novelty**
To the best of my knowledge, the proposed approach is novel in its problem formulation and its reliance on formal temporal logics to formally capture and check safety constraints. The introduction of this framework holds strong potential for significant impact, as it provides a vital resource that could accelerate progress in the safe LLM-based embodied agents

3. **Experimental Evaluation**
The experimental results clearly highlight the strengths and the limitations of existing state-of-the-art LLMs in their ability to understand and successfully adhere safety constraints. In particular,  the safety assessment at the level of detailed execution trajectories shows that even the best frontier models mostly generate unsafe execution trajectories ( % of successful and safe trajectories is less than 5).

### Weaknesses
1. [Poor Presentation] Although I believe that the proposed approach is sound,  the presentation of the main technical sections related to problem statement and temporal logics (sections 2.1 and 2.2) can be significantly improved. A lot of important concepts introduced, but they are either not formally defined at all or defined much later in the paper. Here are a few examples: 
 - the authors write: 
 > These trajectories are **merged into a computation tree T** , and CTL-based checking is applied to verify that **C** holds across all possible execution branches ...
  **C* is never defined. What is **C**? The mechanism to **merged  [trajectories] into a computation tree T** is only explained 3 page later (in page 6). 
  - Another example, in line 157, the authors use the term "labeling function L" for the first time without defining it (again the formal definition is provided only 3 page later). 
  -  Finally in line 158, the authors write "σ |= a iff p ∈ L(s_0))". What is a? What is the relationship between a and p?

2. Some minor issues:
 - In line 170, "AGφ (a safety invariant: φ holds on all paths)" should be replaced with AGφ (a safety invariant: φ **always** holds on all paths)
 - In line 398-399, "Overall results are reported in **Section 3.2**." should be replaced with "Overall results are reported in **Table 3**."
 - In line 457, "Results are summarized in **Section 3.3**." should be replaced with ""Results are summarized in **Table 4**."

### Questions
I have asked all my questions in the weaknesses section

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SENTINEL, a multi-level safety evaluation framework for LLM-based embodied agents using temporal logics. Safety requirements are encoded in LTL/CTL and checked at three levels: semantic (NL to LTL), plan-level (LTL over high-level plans), and trajectory (CTL over computation trees). The framework is evaluated on VirtualHome and ALFRED, and across several open and closed LLM families.

### Strengths
(+) The paper provides a framework to encode safety specifications into temporal logic semantics LTL/CTL, which allow more correct representation and support formal verification afterwards. This is promising for guaranteeing safety of LLMs and interpretability.
(+) Verification is conducted at multiple levels including semantic, plan, and trajectory, providing stricter safety guarantees than related works and preventing unsafe actions during agent interactions.
(+) The paper is well-organized and easy to follow.

### Weaknesses
(-) Though the idea of using formal logic checking is interesting, there are several weaknesses in the current implementation. While SENTINEL reimplements CTL checking (via BFS/DFS traversal), it does not leverage existing mature model-checking tools (e.g., NuSMV, SPIN, PRISM). Although the paper's key difference lies in its use of formal verification, it has not demonstrated that SENTINEL's in-house LTL/CTL checker is equivalent to standard model-checking semantics. It defines satisfaction rules for LTL/CTL from natural languages but it's still unclear to me that the verification is sound (no false positives) or complete (no missed violations). To me, the claimed rigor of "formal verification" remains conceptual rather than guaranteed.

(-) No refinement process is used, i.e., the framework performs plan-level verification in isolation without progressively refining or strengthening specifications during execution. Consequently, plan-level checks cannot capture runtime or spatially dependent conditions (e.g., proximity, force, timing).

(-) Evaluation is confined to VirtualHome and ALFRED, both using discrete action spaces. The framework is not tested in continuous, more complicated environments where continuous control and real-time safety verification are critical, limiting generalizability to real-world application.

(-) The framework depends on a curated set of ground-truth temporal-logic constraints per task, but the paper only briefly describes how these constraints were created. It does not explain who curated them, how long it took, or how correctness and consistency were validated. This makes the specification process hard to reproduce or adapt to new domains that require expert-defined safety rules.

(-) The paper lacks quantitative analysis of property coverage and verification efficiency. It does not report how many properties were tested per task, their representativeness across safety categories, or the average number verified. Verification time, storage overhead, and scalability as trajectory complexity grows are also not discussed, leaving the practical scalability and completeness of the method unclear.

Suggestions:
1. Table (II) should include a column describing the capability of the compared models (i.e., performance under MMLU benchmark) for better comparison.
2. Providing suggestions on how to leverage verification results/feedback to help agents improve their safety/satisfaction rate (e.g., iterative repair, refinement, or safety-aware planning) would further enhance the practical value of the framework and be helpful to the community.

### Questions
Please clarify the negative points above, specifically around soundness of analysis results.

### Soundness
2

### Presentation
3

### Contribution
3
