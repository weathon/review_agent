# Computability of Agentic Systems

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
This paper introduces the Quest Graph, a formal framework for analyzing the capabilities of agentic systems with finite context. 
	We define abstractions that model common reasoning techniques and establish their computational power: 
	the base Quest Graph is equivalent to an unrestricted Turing machine; 
	the forward-only Finite Quest Decision Process (FQDP), despite its wide use, is only equivalent to a pushdown automaton (context-free); 
	and the Reference-Augmented QDP (RQDP) regains Turing completeness only when stateful queries are allowed. 
	Since computability affects efficiency, we then analyze the theoretical efficiency of each model by simulating task dependencies in computation graphs. 
	We show that this computational hierarchy translates to concrete performance trade-offs: 
	reference-augmented (Turing-complete) systems can be exponentially more efficient at simulating complex graphs than their non-augmented (context-free) counterparts. 
	This work provides a formal methodology for classifying and understanding the fundamental capabilities of agentic systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a formal framework called the Quest Graph to analyze the computational power of deterministic agentic systems that are constrained by a finite context window. The central idea is to decouple the agent's reasoning module (modeled as a stateless function) from its working memory (modeled as a dynamic graph).

The authors establish a baseline that a standard Language Model (LM) with a finite context is computationally equivalent to a Finite State Machine (FSM). They then introduce and analyze variants of their Quest Graph framework:

* General Quest Graph: An unrestricted model where the agent can freely modify the graph. This is shown to be Turing complete.

* Quest Decision Process (QDP): A constrained, forward-only version designed to model hierarchical reasoning (like chain-of-thought). The Finite QDP (FQDP), where nodes have a finite number of children, is shown to be equivalent to a Pushdown Automaton (PDA).

* Reference-Augmented QDP (RQDP): This model enhances the FQDP with a retrieval mechanism (inspired by RAG) that allows the agent to access its own unbounded, out-of-context history.

The paper's main results are a series of theorems and corollaries that build a computational hierarchy for agentic systems.

* Result 1 (Baseline): A standard Language Model with a finite context window is computationally equivalent to a Finite State Machine (FSM) (Theorem 1).

* Result 2 (General Framework): The general, unrestricted Quest Graph framework, where an agent can modify nodes, is Turing complete (Theorem 2).

* Result 3 (Hierarchical Reasoning): A FQDP, which models deterministic, hierarchical, forward-only reasoning, is computationally equivalent to a DPDA (Theorem 3). A Non-deterministic FQDP (NFQDP) is equivalent to a general PDA (Corollary 17).

* Result 4 (Memory/Retrieval): The RQDP, which adds a mechanism to retrieve from an unbounded history, is Turing complete (Theorem 4). This demonstrates that a retrieval-like mechanism is sufficient to move from context-free computation back to universal computation.

* Result 5 (Efficiency Hierarchy): When simulating a general computation graph of size $N$ (which is transformed into a Bounded MCG of size $O(N^2)$):

  * LMs (FSMs) cannot simulate the graph (Corollary 9).
  * FQDP/NFQDPs (PDAs) can, but in exponential time ($O(2^N)$), as they must re-compute shared dependencies (Corollary 8).
  * RQDPs/Quest Graphs (TMs) can in polynomial time ($O(N^2 \log N)$ or $O(N^2)$), as they can retrieve or access previously computed results (Corollaries 6 & 7).

### Strengths
* The Quest Graph is a clean and powerful abstraction. Decoupling the stateless agent function (the LM core) from the stateful graph memory (the scratchpad/context) is an insightful way to model these systems and aligns well with modern architectures.

* Practical Relevance: The paper wisely does not stop at pure computability. The analysis in Section 6 is a key contribution. It translates the abstract theoretical hierarchy into concrete performance trade-offs (exponential vs. polynomial time), providing a compelling argument for why retrieval-augmented (RQDP-like) agents are more powerful than simple hierarchical (FQDP-like) ones for tasks with complex dependencies.

### Weaknesses
* The entire framework models the agent as a deterministic function (equivalent to an LM with temperature 0). This is a necessary simplification for a classic computability analysis, but it sidesteps the fundamentally stochastic nature of real-world LMs. The paper's conclusions apply only to this deterministic, best-case reasoning.

* The analysis assumes the agent function $\chi$ (the LM forward pass) takes constant time ($O(1)$). In practice, this is the most computationally expensive part of the entire system. This doesn't invalidate the analysis of the number of steps, but it's an important abstraction that could be misleading if applied too literally to wall-clock time.

### Questions
* The entire framework rests on the agent being deterministic. How does this hierarchy change when considering stochastic agents, which is the standard for real-world LMs? Does stochasticity simply map to non-determinism (like the NFQDP), or does it introduce a fundamentally different computational class?

### Soundness
3

### Presentation
3

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
The paper develops a formal framework, the Quest Graph, to analyze the computability of deterministic agentic systems with finite context. It introduces constrained variants, QDP/FQDP/NFQDP for hierarchical reasoning and RQDP/NRQDP for reference-augmented retrieval, and maps them to classical automata classes: finite-context LMs ≅ FSMs, FQDP ≅ DPDA (context-free), and RQDP regains Turing completeness via a non-finite reference space and a retrieve action. They also analyze these architectures from a practical lens.

### Strengths
The theoretical core is the paper’s main asset. The formalization is careful, with clear separation between agent function and external working memory, and a neat alignment to the formal language hierarchy that makes the progression from LM→QDP→RQDP easy to follow. The hierarchy yields crisp claims: finite-context LMs are only regular; enforcing hierarchical “forward only” structure collapses power to pushdown; adding a reference mechanism with a non-finite tag space restores Turing completeness. The computation-graph analysis is a welcome bridge from theory to practice, turning abstract differences into complexity statements that are simple to reason about in real coordination workloads.

### Weaknesses
First and foremost, There is no empirical validation or case study demonstrating that the complexity separations manifest on realistic agent stacks; without at least small controlled experiments, it is hard to judge how these elegant results transfer outside the formal model. Even without the coverage of experiments, their is a clear lack of discussion around the existing architectures and the theory introduced. The overall presentation undersells where present-day agent systems actually fail and what actionable design guidance follows for the existing LLM based architectures. Given that the paper's main contribution is around theory, I would expec to have some more discussion around the practicality of their approaches. 

Apart from its theoretical contributions, the overall writing of the paper is not very strong. The following suggestions focus not on minor issues such as grammar or phrasing, but on broader improvements that could enhance the overall clarity and presentation of the work.:
* Table 1, which summarizes the core architectures and their computational relationships, should appear much earlier, preferably by the end of Section 2.
* The paper would benefit from explicitly tying the proposed theoretical constructs (e.g., LMs, QDP, RQDP) to existing model classes or real LLM-based agent systems to make the abstractions more relatable.
* Several figures (e.g., Figures 1 and 2) convey overlapping information and could be merged into a single, more comprehensive diagram illustrating both the architectural flow and hierarchy.

Overall this is a theoretically insightful and timely paper that gives the community a shared language for the computability of agentic systems. But the paper lacks in overall clarity in writing and providing actionable insights/experimental evidence to support their theoretical claims, which leads me to lean towards rejection.

### Questions
Q1. How do existing AI systems map to the different architectures introduced in the paper? Which of these architectures are already implicitly represented in current models, and which ones remain purely theoretical and hence pose as a direction for future works

Q2. How do the theoretical limitations identified for each architecture (e.g., context bounds, hierarchical reasoning constraints, or retrieval assumptions) translate into practical bottlenecks observed in today’s agentic or LLM-based systems?

Q3. The paper focuses primarily on theoretical constructs and asymptotic analysis, could small-scale empirical experiments or simulations help validate whether these computational distinctions manifest in real agent systems? If so, what kind of experiments would be most illustrative? 

I am willing to increase the score if the above questions can be answered clearly, especially the Q1 and Q2. Q3 doesn't require running experiments but mainly what experiments can be designed around the theoretical claims.

### Soundness
2

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
2

### Summary
This work introduces the Quest Graph framework to analyze computational capabilities of agentic systems with finite context windows. It links different architecture variants to distinct levels of the formal language hierarchy, clarifying the connection between agent design and expressive power.

### Strengths
- The paper connects agent architectures to models from automata theory (FSM, PDA, Turing machine), grounding abstract agentic reasoning in formal computability theory.
- The Quest Graph unifies reasoning, memory, and hierarchical task decomposition under one formal model, making it extensible to different types of agents (LLMs, hierarchical RL systems).
- The complexity analysis bridges theory and practice by showing the trade-off between computational expressiveness and execution efficiency.

### Weaknesses
- The framework remains purely theoretical, lacking experimental validation or benchmarking to demonstrate its applicability to real-world agentic systems or LLM-based agents.

### Questions
How might the Quest Graph framework be empirically validated through real-world LLM-based agents or reinforcement learning environments?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In Computability of Agentic Systems, the authors provide a framework, principally derived from the notion of a novel Quest Graph (QG), to study the computational complexity of "agentic systems". The authors propose a series of Decision Processes that should represent typical agentic systems, along with common constraints, such as common reasoning patterns (QG -> GDP) captured through the action space (aka agent function space), limited context length (QDP -> FQDP), or breadth-first-search-like planning (FQDP -> NFQDP). Most importantly, the authors demonstrate that reference-augmented agentic systems ({FQDP, NFQDP} -> {RQDP, NRQDP}) are Turing complete, whereas typical non-reference-augmented agentic systems are not. The authors provide sound theoretical work, to the best of the reviewer's judgment.

### Strengths
The reviewer is not particularly familiar with the research domain of computational complexity for language models, except for the more popular work from that subdomain, such as "The Illusion of State in State-Space Models" (Merrill et al., 2024). However, they have a background in deep multi-agent RL and are familiar with MDP variants and agentic systems.


## Clarity
- For a theory-heavy paper, the reviewer could follow the paper quite well, which speaks to the logical structure in which the different decision processes and results were introduced and motivated.
- The design decisions to constrain the Quest Graph to better represent "agentic systems" are presented transparently.
- The paper cites previous work when necessary.

## Originality
- The attempt to formalise the computational complexity of "agentic systems" is worthwhile. The specific constraints on the Decision Processes seem original.

## Quality
- The proofs appear thorough. The reviewer could not identify any conclusion that was not properly motivated, at least in relation to the technical contributions.

## Significance
- Agentic systems using language models with finite context are showing great promise in different domains. A formal framework could help identify potential improvements in how such agentic systems should be constructed and facilitate the community's discussion of these decisions in a principled manner.

### Weaknesses
# Clarity


## Abstract 


Overall, the abstract is fairly vague and could be improved. For example:

> "Theoretically, we demonstrate that these models form a hierarchy of computational power corresponding to key levels of the formal language hierarchy." 

>  "We then analyze the practical efficiency of each model by simulating task dependencies in computation graphs, revealing that this
theoretical hierarchy translates to significant performance trade-offs". 

(1) Please describe your findings directly, such as "We demonstrate that, given certain constraints, reference augmentation is necessary for an agentic system to be Turing complete" or "We show that reference-augmented systems have a better CG complexity than non-reference augmented decision processes.". Adapt accordingly.

(2) The reviewer disagrees that the paper is analysing "practical efficiency", as the computation graphs (MCGs) are still more of a theoretical construct than any decision process an agentic system might see in common agentic benchmarks. Under "practical efficiency", the reviewer would have expected a wall-clock time comparison on a popular agentic benchmark with the expected theoretical complexity, e.g., a reference augmented agentic system is faster on MLEBench than a non-reference augmented system. 

## Main Text
### References
> Language models (LMs) lie at the core of these systems, serving as the primary decision-making module that processes language in a manner analogous to human cognition (Felin & Holweg (2024)). (line 28)

(3) Could you clarify how the Felin & Holweg citation motivates your statement of "analogous"? At least their abstract seems to disagree that any analogy is particularly meaningful when it comes to reasoning patterns and cognition:

> Scholars argue that artificial intelligence (AI) can generate genuine novelty and new knowledge and, in turn, that AI and computational models of cognition will replace human decision making under uncertainty. We disagree. We argue that AI’s data-based prediction is different from human theory-based causal logic and reasoning.

In general, I found the many references to human cognition distracting. For example, while the reviewer appreciates that the authors draw inspiration from cognitive science, stating that they were inspired by a theory of the hippocampus' mechanism (line 249), without providing further detail about what aspects of that theory inspired them, seems like unnecessary detail. Is it the fact that brains have long-term memory retrieval? 

Also on line 328
> By keeping track of a location in the reference space, this mechanism functions analogously to cognitive maps in the hippocampus.

On what level does this analogy work? Where does this analogy break down? It appears to be a questionable scientific practice to make such broad statements, especially if the paper's principal contribution does not address human cognition and makes no effort to validate these statements empirically.

(4) The reviewer would prefer that these references be either solidified or left out completely, but is open to the author's opinions.

### Section 5
(5) The reviewer found Section 5 quite hard to follow; the explanation of the retrieve action could potentially be visualised, like in Figure 2.

(6) Similarly, could you please clarify the following paragraph (line 273)?
> Therefore, the reference-generating function must be allowed to vary per quest to ensure sufficient
capacity. Crucially, this design does not violate the agent’s stateless assumption. The reference graph
and its unbounded history are managed by the Quest Graph, while the agent function itself remains a
static, finite-context component unaware of this underlying complexity.

 It was not clear to the reviewer why the reference-generating function "must" be allowed to vary per quest.

# Significance and Originality
The reviewer found it challenging to grasp the significance and originality of the proposed framework fully. Most citations of contemporary work in the paper are methods applied to agentic systems. The citations are used to motivate the paper's design decisions. There are very few citations for related work that compare and contrast this paper with their academic "siblings". It's quickly mentioned that other "unconventional models" exist in the literature, but no effort was made to explain how or if they differ. Furthermore, as the authors point out, formally, a QDP is a variant of a POMDP. Sequence models, such as RNNs, and even feedforward neural networks, have been utilised in POMDPs for a considerable time. 

(7) Could the authors clarify if there are any comparable frameworks for complexity analysis on POMDPs? What makes this analysis unique to LMs? Do they consider state-space models and LSTMs also as LMs, or is this work mostly focused on transformer-based architectures?

Furthermore, the authors state that,
> The standard model of an LM treats the decision-making module and the context buffer as a single,
monolithic entity. We propose decoupling these components, presenting the reasoning module as a
pure function that operates on an external memory structure. This separation offers both practical and
theoretical advantages. On a practical level, it aligns with modern, stateless architectures recognized
for their scalability and testability (Fielding (2000)).

(8) Could the authors please clarify what that means for the transferability of their framework to practical usage of agentic systems? Moreover, Fielding (2000) appears to discuss stateless architectures of network-based systems in the context of the World Wide Web. Could you clarify the connection to LM-based agentic systems?

Moreover, without any empirical results, it's hard to agree that the design choices for action spaces and retrieval augmentation are genuinely meaningful. While the proofs appear sound, given the action space design, it remains unclear whether this framework is predictive of the actual computational complexity of LM-based agentic systems in the end. The design choices were mostly motivated by citing some related method, but it's unclear if that motivation is sufficient. 

(9) Would the authors be willing to include empirical results, such as comparing different agentic systems (matching their proposed decision processes) on a single, simple agentic, even if contrived, benchmark? While probably out of scope, work like [1] compares AIDE and AIRA as agentic systems with different node selection techniques on MLEBench. Would it be feasible to make predictions about the performance of these systems with your framework?

(10) Alternatively to (9), could the authors further motivate how their choices for the action space are necessary and sufficient to represent agentic systems? While intuitive, it is hard to grasp why, e.g., the agent function in Section 3.1 only has the 3 subactions


[1] AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench, Toledo et al. 2025,

### Questions
The reviewer listed all questions in the Weakness section, enumerated. The reviewer is happy to engage in a follow-up discussion on these questions and clarify any points as necessary. 

If the authors could solidify the originality and significance of the contribution, and compare and contrast it with other related work, if it exists, the reviewer would be happy to consider changing their score. Any similar work that was accepted at previous conferences would also help the reviewer calibrate their decision.

### Soundness
3

### Presentation
2

### Contribution
2
