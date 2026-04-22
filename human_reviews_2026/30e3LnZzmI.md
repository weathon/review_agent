# Reimagining Agent-based Modeling with Large Language Model Agents via Shachi

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
The study of emergent behaviors in large language model (LLM)-driven multi-agent systems is a critical research challenge, yet progress is limited by a lack of principled methodologies for controlled experimentation. To address this, we introduce Shachi, a principled methodology and modular framework that decomposes an agent's policy into core cognitive components: Configuration for intrinsic traits, Memory for contextual persistence, and Tools for expanded capabilities, all orchestrated by an LLM reasoning engine. This principled architecture moves beyond brittle, ad-hoc agent designs and enables the systematic analysis of how specific architectural choices influence collective behavior. We validate our methodology on a comprehensive 10-task benchmark and demonstrate its power through novel scientific inquiries. Critically, we establish the external validity of our approach by modeling a real-world U.S. tariff shock, showing that agent behaviors align with observed market reactions only when their cognitive architecture is appropriately configured with memory and tools. Our work provides a rigorous, open-source foundation for building and evaluating LLM agents, aimed at fostering more cumulative and scientifically grounded research. Code: https://anonymous.4open.science/r/bench-2E1D/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Shachi, a formal methodology and modular framework designed to address the fragmentation and lack of reproducibility in LLM-based agent-based modeling (ABM). The core contribution is a principled decomposition of an agent's policy into four modular components: a reasoning engine (LLM), intrinsic traits (Configs), contextual persistence (Memory), and expanded capabilities (Tools), orchestrated through a standardized agent-environment interface. This architecture enables systematic analysis of how specific cognitive components influence emergent behaviors.

### Strengths
1. Shachi's core strength lies in its formal decomposition of an agent's cognitive architecture. By standardizing components (LLM, Configs, Memory, Tools) and the agent-environment interface, the framework directly addresses the field's reproducibility crisis.
2. The inclusion of a 10-task benchmark, thoughtfully structured across three levels of increasing social complexity, provides a much-needed standardized testbed for the community.

### Weaknesses
1. The inclusion of a 10-task benchmark, thoughtfully structured across three levels of increasing social complexity, provides a much-needed standardized testbed for the community.
2. The inclusion of a 10-task benchmark, thoughtfully structured across three levels of increasing social complexity, provides a much-needed standardized testbed for the community.


Missing reference:   
Junyu Luo et al., Large language model agent: A survey on methodology, applications and challenges, arXiv 2025.

### Questions
Please see the weakness.

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
3

### Summary
The paper introduces Shachi, a modular framework for LLM-based ABM that decomposes agent policy into four components (Configs, Memory, Tools, LLM). The authors claim this principled architecture enables systematic, reproducible analysis of emergent behavior, validating it with a 10-task benchmark and a real-world tariff shock simulation to demonstrate external validity.

### Strengths
1.	Novel Modular Framework: The paper proposes a clear, modular abstraction for agent design. This is a valuable contribution that moves the field away from ad-hoc scripts and toward a standardized, reproducible, and systematic methodology.

2.	Good Presentation: The paper is well-written, clearly structured, and easy to follow. Figures and visualizations are clear and effectively communicate the authors' core contributions and experimental results.

### Weaknesses
1.	Weak Theoretical Grounding: The paper presents the decomposition of (Configs, Memory, Tools, LLM) as a “formal methodology” or “principled architecture”.  In practice, this looks more like a conceptual abstraction or a software-engineering style decomposition than a framework derived through rigorous theoretical justification. The paper does not conduct a formal analysis of this, which makes the claim of a "formal methodology" seem somewhat exaggerated. But it is indeed a well-structured framework.

2.	Weak Support for Generalization Claims: In cross-task generalization research, the performance differences between StockAgent with complete components and Sotopia Agent with only Memory across tasks are very slight (0.99 vs 0.92, 0.93). The strong contextual reasoning ability of the LLM itself may be the main driving force for generalization. 

3.	Weak External Validity: In the external validity experiment (Section 4.2.3), the results mainly show that Shachi’s outputs can be effectively controlled through the inputs of Memory and Tools. However, this only reflects the framework’s configurability rather than its predictive capacity. The experiment does not convincingly demonstrate that Shachi can robustly handle or generalize to real-world scenarios beyond the information explicitly provided to the agents.

### Questions
1.	How is “Formality” Defined in This Methodology? The paper repeatedly describes Shachi as a “formal methodology”, but the framework is primarily an implementation protocol. 

2.	In Shachi, the agent is decomposed into four components—Configs, Memory, Tools, and LLM. Since both retrieved memory content and tool outputs are provided to the LLM in textual form, potential conflicts or negative interference between these inputs may occur. As a formal methodology, how does Shachi ensure that such inconsistencies or interference are systematically prevented or mitigated at the architectural level?

### Soundness
3

### Presentation
4

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
This paper presents Shachi, a principled and modular framework for large language model–based agent-based modeling (ABM). It standardizes agent design by decomposing each agent’s policy into four cognitive components: Configs, Memory, Tools, and an LLM reasoning engine, and provides a three-level benchmark suite for systematic evaluation. Experiments show that Shachi reproduces existing results, supports cross-task generalization, enables novel studies such as memory transfer and multi-world simulations, and achieves external validity by modeling real-world market reactions to a U.S. tariff shock.

### Strengths
Originality: The paper is original in proposing a unified and modular methodology for LLM-based agent-based modeling (ABM).  Rather than introducing a new agent or task, it formalizes how agents and environments should interact, filling an existing methodological gap in the field.

Quality: The experimental design is broad and novel, including real world task and new case studies such as memory transfer, multi-world interaction, and market simulations.  These diverse experiments support the framework’s robustness and generality.

Clarity: The paper shows a clear motivation and unifying perspective. The introduction, method and proposed framework are logically connected and easy to follow.

Significance: The work establishes a solid methodological foundation for future research on LLM-driven multi-agent systems. By standardizing interfaces and modularizing agent cognition, Shachi has the potential to become a reference framework for reproducible, scientific simulation studies using large language models.

### Weaknesses
1) Simplistic baselines and weak comparisons.
The baselines are mainly random agents or plain LLMs without modular components. That is too limited to fully substantiate Shachi’s methodological advantages.  More meaningful comparisons could include systems that also use LLM + Memory but lack Shachi’s standardized interface, or other existing LLM-based ABM frameworks reimplemented under the same conditions.  In addition, the current results omit statistical indicators such as variance, confidence intervals, and significance tests, making it difficult to evaluate robustness and reproducibility.  Incorporating richer baselines and proper statistical reporting would greatly strengthen the empirical evidence.

2) Presentation and clarity issues.
The paper’s organization is dense, with technical details scattered between sections and appendices. Some details, figures and tables could be better integrated with the main text, and clearer explanations of evaluation metrics and experimental settings would improve readability.

### Questions
1) How scalable is the Shachi framework in practice, including LLM calls, time scalability with increasing agent numbers, and space scalability? 
2) Could the authors report variance, confidence intervals, or significance tests to substantiate the reliability of these results?
3) How sensitive are the reproduction outcomes to the choice of backend LLMs and prompts?

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
4

### Summary
This paper introduces Shachi, a formal methodology and modular framework for LLM-based agent-based modeling (ABM). The authors propose decomposing an agent's policy into four core components: Configuration (intrinsic traits), Memory (contextual persistence), Tools (extended capabilities), and an LLM reasoning engine. The framework is validated on a 10-task benchmark suite spanning three levels of social complexity (single-agent, non-communicative multi-agent, and communicative multi-agent).

### Strengths
1, Well-motivated problem: The paper addresses a genuine issue in LLM-based ABM research - the lack of standardized methodology leading to fragmented, irreproducible results. This is an important problem for the field.

2, Principled architecture design: The decomposition of agent policy into Configuration, Memory, Tools, and LLM is intuitive and grounded in cognitive science principles. The separation of agent architecture from environment through a standardized interface is a solid engineering contribution.

3, Strong empirical validation: The reproduction study (Table 1) shows low MAE across all tasks, demonstrating the framework's ability to faithfully replicate prior work. The external validity experiment with the U.S. tariff shock is particularly compelling.

4, Novel exploratory studies: The "carrying memory to next life" and "living in multiple worlds" experiments showcase creative applications enabled by the modular design, though they could benefit from deeper analysis.

### Weaknesses
1, Limited theoretical novelty: While the engineering contribution is solid, the conceptual decomposition (LLM + config + memory + tools) is relatively straightforward and has been implicitly used in prior agent frameworks. The paper doesn't provide strong theoretical justification for why this particular decomposition is optimal or complete.

2, Insufficient comparison with existing frameworks: The paper mentions AutoGen, Concordia, and EDSL but dismisses them too quickly without detailed empirical comparison. A systematic comparison showing Shachi's advantages would strengthen the contribution. The claim that these frameworks are "not designed for reproducible social simulation" needs more substantiation.

3, Cross-task generalization results are underwhelming: Table 2 shows that most agents perform similarly across tasks (scores ≈ 1.0), which undermines the claim about the importance of modular components. The only clear difference is when tools are missing for tool-requiring tasks, which is unsurprising.

4, Limited discussion of failure modes: The paper doesn't adequately discuss when the framework might fail or produce unrealistic behaviors. For instance, what happens when agents face novel situations not covered by their configuration or tools?

5, Missing important implementation details:

- How are conflicts between tools and memory resolved when both could apply?
- What are the computational costs of the framework compared to alternatives?
- How sensitive are results to LLM temperature, prompt variations, and other hyperparameters?
- The "two-stage parsing" strategy (Appendix D.4) seems like a workaround but its necessity and implications aren't discussed

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
