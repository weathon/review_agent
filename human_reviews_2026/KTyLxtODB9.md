# ROGA: Scaling Generalist Agents for Office Productivity Tasks via Tool Generation

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Automatic tool generation (ATG) has emerged as a key approach to enable the automatic adaptation across diverse tasks within a single generalist agent.
Despite their potential, we argue that current ATG agents, often built on reactive paradigms, fail to effectively adapt to realistic environments requiring long-term reasoning and stateful interaction, particularly in office ecosystems. We empirically show that current ATG agents underperform by up to 27.43%.
This performance degradation stems from three fundamental limitations of prevailing agent paradigms: (1) a failure to build a coherent world model from long, partially observable contexts; (2) a memory-less execution model where stateless actions fail to track state evolution during iterative tasks; and (3) a static capability generation model focusing on one-shot tool generation for immediate needs, thereby forcing redundant regeneration for similar steps.

To address these fundamental limitations, we propose ROGA, which instantiates a new agent paradigm for long-horizon, stateful environments. ROGA moves beyond simple reactive loops by introducing four foundational algorithmic innovations: (1) Active World Modeling, an iterative process where the agent actively probes the environment to construct its own world model; (2) a Persistent Symbolic Memory that explicitly tracks the state evolution for temporal reasoning; and (3) a Dynamic Capability Evolution model for long-term adaptation and meta-learning on the agent's own capabilities.
Comprehensive experiments on widely used benchmarks show that ROGA consistently outperforms existing ATG agents by up to 13.64%.
These results underscore ROGA's potential to advance the ATG paradigm, delivering a practical pathway toward building sustainable generalist agents in realistic environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper “ROGA: Scaling Generalist Agents for Office Productivity Tasks via Tool Generation” introduces a new framework, ROGA, to advance the paradigm of Automatic Tool Generation (ATG) for generalist agents operating in real-world office environments such as Excel, Word, and PowerPoint. The authors identify three major limitations of existing ATG agents: (1) poor handling of long file contexts that contain essential task details, (2) lack of shared state across tool executions, which prevents iterative modifications of the same object, and (3) inefficient tool reuse that leads to repeated generation errors. To address these, ROGA proposes four key innovations — (i) a Comprehension–Operation (Comp-Op) paradigm that separates task understanding from execution to capture fine-grained file details, (ii) a Dual-Reflection mechanism that performs both functional and semantic validation of generated tools, (iii) a State-Sharing Sandbox for consistent intermediate context management, and (iv) a Finite-State Machine (FSM) for tool lifecycle management that automates validation, reuse, and deprecation of tools. Across benchmarks including OSWorld, WindowsAgentArena (WAA), GAIA-Office, TableBench, and SheetCopilotBench, ROGA achieves significant improvements

### Strengths
This paper presents a well-motivated and empirically grounded contribution to the ATG research domain. The authors clearly identify critical pain points in current generalist agent paradigms and systematically address them through a thoughtfully designed architecture. The Comprehension–Operation paradigm formalizes reasoning separation, allowing agents to process long office files in a modular, interpretable manner. The Dual-Reflection mechanism provides a concrete and reproducible approach for improving tool reliability through iterative validation, while the State-Sharing Sandbox effectively enables multi-step tool coordination—an essential requirement for real-world automation tasks. The paper demonstrates methodological rigor with detailed benchmark coverage, using multiple baselines (AutoAgent, OctoTools, OWL, and SheetAgent) and both office and non-office domains. The inclusion of a well-structured ablation study offers clear evidence of how each module contributes to performance, revealing meaningful insights into which design choices matter most. Empirically, ROGA delivers strong and consistent gains across all benchmarks, notably outperforming specialized agents in spreadsheet reasoning tasks.

### Weaknesses
The paper lacks a deeper theoretical or analytical justification for why the proposed mechanisms—particularly dual-reflection and comprehension–operation separation—lead to better generalization and reduced reasoning errors. Although benchmarks and metrics are clearly listed, the paper does not provide implementation details for the planner’s decision policy, the functional testing procedure, or exact hyperparameter configurations for reflection or state management. The work mentions “shared memory Mt” and “transition function δ” but omits practical instantiation details, making it difficult for results to be replicated. Third, while ROGA achieves strong empirical results, the evaluation is mostly limited to office-centric benchmarks and one general reasoning dataset. It would be valuable to include cross-domain or zero-shot adaptation studies to demonstrate broader generalization and scalability claims. The cost analysis shows higher token consumption and reasoning steps, indicating that ROGA’s efficiency and scalability over very long tasks are not fully optimized; no discussion is provided on potential mitigation strategies.

### Questions
please address above questions in weakness

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
4

### Summary
This paper introduces ROGA, a generalist agent framework designed to overcome limitations of current Automatic Tool Generation (ATG) agents in open-ended office productivity tasks. The authors identify three key shortcomings of existing ATG agents: inadequate handling of long file contexts, lack of context sharing across tools, and inefficient tool reuse. ROGA addresses these via a comprehension-operation reasoning paradigm, dual-reflection tool generation, a state-sharing execution sandbox, and finite-state machine-based tool lifecycle management. Evaluations on office benchmarks show that ROGA significantly outperforms existing ATG agents, while preserving performance in non-office domains.

### Strengths
1. The paper includes a well-designed motivation study that clearly illustrates the limitations of existing generalist agent frameworks, particularly in office environments.

2. Performance: ROGA demonstrates strong and consistent performance on office productivity tasks, and the authors show that it does not deteriorate on non-office domains, supporting its generalizability.

3. The overall writing is generally clear, and the framework components are explained in a structured and understandable manner.

### Weaknesses
1. In the motivation study, the authors compare ATG agents to the domain-specific SheetAgent to highlight performance gaps. However, in the main experiments, ROGA is not compared to domain-specific agents (except in the spreadsheet-only analysis). Including such comparisons for other office tasks (e.g., Word or PowerPoint) would better contextualize ROGA’s advancements.

2. The paper does not clearly motivate why office productivity tasks are particularly important or representative as a testbed for generalist agents. A stronger justification in the intro—for instance, their complexity, real-world impact, or ubiquity—would enhance the significance of the chosen domain.

3. While ablations show that each component contributes to performance, it remains unclear whether ROGA’s gains stem primarily from its general agentic design or from better, office-specific tooling and context handling. The framework is presented as general, yet its advantages are most pronounced in office tasks.

**Minor Issues**

Line 478: "environments To address them" – missing a period.

Citation formatting is inconsistent in several places, e.g., in line 113 "(Cai et al., 2024; Wolflein et al., 2025; Qian et al., 2023) aims to reduce..." should use consistent punctuation and style.

### Questions
1. The framework components (e.g., Comprehension–Operation, State-Sharing Sandbox) appear general and domain-agnostic. Why, then, does ROGA significantly improve performance on office tasks but show only comparable (not superior) results on non-office tasks like math? What aspects of the design are specifically tailored to office environments?

2. Could the authors provide more insight into whether the performance gains are due to better context management and tool generation for office files, or due to the agentic framework itself?

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
The paper presents ROGA, a framework designed to enhance automatic tool generation (ATG) agents for office productivity tasks such as Excel, Word, and PowerPoint automation. The authors argue that existing ATG agents struggle in realistic, long-context environments due to three limitations: poor handling of extended file contexts, lack of context sharing across tool invocations, and inefficient tool reuse.

To overcome these issues, ROGA introduces four components: a Comprehension–Operation paradigm to separate file understanding from action, a Dual-Reflection mechanism for functionally and semantically validated tool generation, a State-Sharing Sandbox for maintaining shared intermediate states, and a Finite-State Machine for managing tool lifecycles and reuse.

Extensive experiments across benchmarks such as OSWorld, WindowsAgentArena, GAIA-Office, TableBench, and SheetCopilotBench show that ROGA achieves substantial improvements over prior ATG agents, even matching or slightly exceeding domain-specific agents in spreadsheet tasks. The paper is well-structured, clearly written, and tackles a problem of growing real-world importance in automating office workflows.

This paper demonstrates strong engineering, clear motivation, and compelling empirical validation in an area of genuine practical importance. However, the core contributions lack sufficient methodological novelty to justify acceptance at a premier venue focused on conceptual advances. The main components, such as reading-then-acting, reflective testing, stateful execution, and lifecycle management, are well-executed but largely incremental adaptations of existing principles.

While ROGA’s performance gains are notable, they likely stem from implementing an architecture that correctly integrates established best practices rather than from introducing a fundamentally new paradigm for ATG agents. The high computational overhead, limited analysis of failure modes, and modest absolute success rates further temper enthusiasm.

Overall, this work is a solid systems contribution that convincingly identifies the architectural elements required for robust ATG performance in realistic office settings. With deeper algorithmic innovation or analytical rigor, it could evolve into a strong submission in future iterations.

### Strengths
Clear Motivation and Relevance:
The paper identifies an important and under-explored challenge, the adaptation of generalist ATG agents to realistic office environments that require multi-step reasoning and persistent file manipulation. The motivation study effectively quantifies the performance gap (up to 27.43%) between existing ATG agents and specialized systems, making a strong case for the work’s necessity.

Clarity and Presentation:
The writing is lucid and well-organized, with helpful figures and a formal task formulation. The overview diagram (Figure 1) and formal decision-process notation help convey the architecture’s structure.

Comprehensive Evaluation:
The experiments are broad in scope, spanning several public benchmarks, and the ablation study convincingly shows that each ROGA component contributes meaningfully to performance. Including Math500 to assess generalization beyond office tasks is a commendable touch.

Empirical Strength:
The reported results are impressive. ROGA delivers consistent improvements on all office benchmarks, and even outperforms the specialized SheetAgent in spreadsheet tasks. This demonstrates that the proposed system is robust, practically effective, and scalable to realistic workloads.

Practical Impact:
The focus on office task automation has major applied significance. Demonstrating tangible improvements in this space is valuable for both research and industry audiences.

### Weaknesses
While empirically strong, the paper’s technical novelty appears limited, and several of its key mechanisms resemble established engineering patterns:

The Comprehension–Operation paradigm closely parallels the long-standing plan-then-act or read-then-execute architectures common in agent design, making its conceptual contribution modest.

The Dual-Reflection mechanism essentially performs test-and-validate loops (shadow execution and code review), a well-known reliability technique in program synthesis and software testing.

The State-Sharing Sandbox is effectively a transactional execution buffer that maintains shared state, standard practice in interactive software systems.

The Finite-State Machine for tool lifecycle management resembles a conventional resource cache with validation and eviction logic.

In short, the framework’s advances seem to stem more from careful system integration than from new algorithmic ideas. Moreover, ROGA’s higher computational cost (3–4× token usage over baselines) raises concerns about scalability and efficiency, particularly since absolute success rates on some benchmarks (e.g., 31.82% Pass@1 on OSWorld) remain low.

Finally, the analysis would benefit from deeper insight into why the system works, for instance, what specific types of context or reflection yield measurable gains, and which failure modes persist. The absence of qualitative or statistical analyses limits interpretability and reproducibility.

### Questions
How exactly does the Comprehension–Operation separation improve results beyond simply allowing more reasoning steps? Could equivalent performance be achieved by giving baselines comparable computation budgets?

In the Dual-Reflection mechanism, how often does semantic validation detect issues missed by functional testing, and could examples illustrate these cases?

The paper shows that ROGA uses significantly more tokens than baselines. What fraction of this overhead comes from comprehension versus reflection, and how does this affect cost-benefit trade-offs?

What are the main failure cases (e.g., comprehension errors, tool misuse, sandbox inconsistencies), and do they suggest directions for further refinement?

Given that the improvements on TableBench and SheetCopilotBench are modest and the gains do not extend to Math500, how generalizable is ROGA beyond the office domain?

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
This paper presents ROGA, a framework for automatic tool generation (ATG) agents designed specifically for office productivity tasks involving Excel, Word, and PowerPoint. The authors identify three key limitations of current ATG agents: (1) insufficient handling of long file contexts, (2) lack of context sharing across tool calls, and (3) inefficient tool reuse. ROGA addresses these through four innovations: a comprehension-operation paradigm, dual-reflection tool generation, state-sharing sandbox execution, and finite-state machine-based tool lifecycle management. Experiments show ROGA outperforms existing ATG agents by up to 13.64% on office benchmarks.

### Strengths
ROGA consistently outperforms baselines across multiple benchmarks (OSWorld, WAA, GAIA-Office, TableBench, SheetCopilotBench) and even matches specialized agents. Table 2 systematically demonstrates the contribution of each component, with state-sharing showing the most significant impact.

### Weaknesses
The paper focuses extensively on tool generation but lacks discussion of recent work on tool retrieval and approaches that combine generation with retrieval. 
The main contribution appears to be thoughtful engineering rather than fundamental algorithmic innovation. The paper would benefit from more clearly articulating what is novel versus what adapts existing techniques.

### Questions
I have some questions in the systematic failure analysis:
What types of errors occur most frequently? (comprehension failures, tool generation errors, execution errors?)
When does dual-reflection fail to catch errors?
Are there systematic biases in which file types or task categories cause problems?

### Soundness
2

### Presentation
2

### Contribution
2
