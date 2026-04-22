# Verl-Tool: Towards Holistic Agentic Reinforcement Learning with Tool Use

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated success in enhancing LLM reasoning capabilities, but remains limited to single-turn interactions without tool integration. While recent **A**gentic **R**einforcement **L**earning with **T**ool use (ARLT) approaches have emerged to address multi-turn tool interactions, existing works develop task-specific codebases that suffer from fragmentation, synchronous execution bottlenecks, and limited extensibility across domains. These inefficiencies hinder broader community adoption and algorithmic innovation. We introduce **Verl-Tool**, a unified and modular framework that addresses these limitations through systematic design principles. VerlTool provides four key contributions: **(1)** upstream alignment with VeRL ensuring compatibility and simplified maintenance, **(2)** unified tool management via standardized APIs supporting diverse modalities including code execution, search, SQL databases, and vision processing, **(3)** asynchronous rollout execution achieving near 2$\times$ speedup by eliminating synchronization bottlenecks, and **(4)** comprehensive evaluation demonstrating competitive performance across 6 ARLT domains. Our framework formalizes ARLT as multi-turn trajectories with multi-modal observation tokens (text/image/video), extending beyond single-turn RLVR paradigms. We train and evaluate models on mathematical reasoning, knowledge QA, SQL generation, visual reasoning, web search, and software engineering tasks, achieving results comparable to specialized systems while providing unified training infrastructure. The modular plugin architecture enables rapid tool integration requiring only lightweight Python definitions, significantly reducing development overhead and providing a scalable foundation for tool-augmented RL research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Verl-Tool, a reinforcement learning framework based Verl and designed for multimodal Tool use. Unlike prior works that provide fragmented or domain-/task-specific ARLT environments, Verl-Tool offers a unified, extensible platform facilitating multi-turn, multimodal agentic RL with streamlined asynchronous rollouts and plugin-based tool integration.

### Strengths
**1. Originality:** Based on Verl, Verl-Tool supports a wide spectrum of tools, as illustrated in both the detailed tables and code plugin example.

**2. Clarity:** The paper contains clear system diagrams, detailed tables of results and settings, and well-explained practical considerations (e.g., tokenization differences), resulting in a paper that is accessible for a complex infrastructure contribution.

### Weaknesses
**1. Positioning Relative to Other Frameworks:** While the paper includes comparisons to recently published or preprinted RL tool-use frameworks in the related work and appendix, these comparisons lack sufficient depth. For instance, VERL already supports MCP server [1], which enables integration of various tools. However, Table 1 indicates VERL only support "FAISS" and "Python Executor". 

**2. Lack of Metrics for Core Framework Features:** The main experiments demonstrate that the proposed framework achieves competitive task-specific performance compared to separate, divergent codebases. However, since both the baselines and the proposed framework use VERL for the RL components, these results are somewhat expected and do not fully validate the framework's unique contributions. Specifically, as a framework whose key innovation is plugin-based tool integration, the paper lacks direct comparisons of tool execution efficiency and space/memory consumption—metrics that are central to evaluating the practical benefits of the proposed design.

**3. Insufficient Real-World Ecosystem Validation:** The paper provides minimal evidence (e.g., through user studies, adoption metrics, or community engagement statistics) that VERL-Tool actually improves reproducibility, encourages broader adoption, or simplifies large-scale collaborative research. Demonstrations of real-world impact beyond benchmark performance are needed to substantiate claims about the framework's utility for the research community.

[1] VERL Documentation - MCP Tool Configuration (https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html#mcp-tool-configuration)

### Questions
1. How are tool execution error messages handled at the interface between the VERL RL module and the Tool module? Specifically, does the framework provide standardized error propagation, logging, or recovery mechanisms? Please clarify the design choices for error handling and their impact on training stability.

2. How does the framework handle virtual environment isolation among different tools? In particular, do tools have access to sudo permissions when required? This is a practical concern, as some environments (e.g., OSWorld [1]) require sudo access for setup and execution. Please clarify the framework's approach to permission management and environment isolation.

[1] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces VERL-TOOL, a framework that trains language models to use external tools like code interpreters, search engines, and SQL databases in multi-step tasks. It improves efficiency with asynchronous execution and supports a wide range of tools through a simple plugin system. The framework shows strong performance across six diverse tasks, matching or beating specialized systems.

### Strengths
- The system is well-designed and clearly tested. It supports six real tasks like math, SQL, and search, and performs as well as or better than existing solutions built just for those tasks. The speedup from asynchronous execution is clearly shown and practical.
- The paper is easy to follow. It is clear how the framework works and how well it performs. It’s also clear how new tools can be added with minimal effort.

### Weaknesses
- Some tool server details are vague, for example, error handling isn't fully explained.
- Although the system supports many tasks and tools, it doesn’t show evidence of training a single agent to handle multiple tools at once.  I would be super curious whether training a model with multiple tools would work with VERL-Tool, and whether there are additional implementation needed other than what has been supported in VERL-Tool. This would better support the paper’s goal of being “holistic.” 
- I would also want to better understand the difference between the prior system and VERL-Tool, like AReal. Besides VERL-Tool supporting more tools, which I suppose AReal can allow users to add customized tool, what is the fundamental differences between VERL-Tool and AReal? Asynchronous Rollouts are already supported in AReal. Moreover, modular tool-server with unified tool management has been proposed in [1], although it only focuses on VLM with tool-using. 

    [1] Su, Zhaochen, et al. "Openthinkimg: Learning to think with images via visual tool reinforcement learning." arXiv preprint arXiv:2505.08617 (2025).
    
    Which leaves only "common findings in the agentic RL setting" which have been observed in previous tool-augmented RL works (for example, strategic tool selection in ReTool, reflection in RAGEN).

### Questions
- I would like to understand what is the fundamental differences between AReal and VERL-Tool. What benefit would VERL-Tool framework bring besides more tools getting integrated?

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
The paper introduces Verl-Tool, a unified and modular framework for Agentic Reinforcement Learning with Tool use (ARLT). It extends Reinforcement Learning with Verifiable Rewards (RLVR) to support multi-turn, multimodal, tool-augmented training. Key innovations include: 1. upstream alignment with Verl for maintainability, 2. a unified API for diverse tools (code, SQL, search, vision, etc.), 3. asynchronous rollouts yielding up to 2x speedups, and 4. extensive experiments across six domains showing performance on par with or exceeding task-specific systems. Verl-Tool provides an efficient, extensible infrastructure for developing and studying tool-using RL agents.

### Strengths
The paper's core contribution is introducing a well-documented, open-sourced framework. The authors evaluate their framework on 6 tasks spanning different domains. A key strength of the framework compared to existing frameworks is including support for more diverse tools. While not entirely new (also present in skyRL), the asynchronous rollout method expedites training. Overall, the framework is well-designed and if well-maintained later on, can serve as a very useful contribution to the community.

### Weaknesses
While the presented framework is very useful and clearly valuable for the community, as a research paper it would be better to see a deeper scientific investigation of the underlying mechanisms rather than primarily a systems description. For instance, the paper could explore more ablation studies isolating how different design components (e.g., modular APIs, tokenization strategy) impact performance (both downstream tasks and training efficiency). Such studies would strengthen the paper’s research contribution. 

The authors evaluate on diverse benchmarks, but it seems like each task is tied to one dominant tool (appendix A), e.g, VT-math --> python, VT-search --> retriever, VT-deeepsearch --> web search, etc. VT-visualreasoner is the one that uses multiple image processing tools. It would be beneficial to include a few tasks that rely on many different types of tool calls present in the trajectory, e.g., image processing and web search.

### Questions
1. Could the authors provide ablation studies isolating the contributions of key design components (e.g., modular API design, tokenization strategy) to training stability and task performance? How does the choice of tokenization strategy affect rollout consistency or policy optimization?
2. Although the framework supports multiple tools, do the authors have results on trajectories that involve multiple tool types (e.g., a single rollout combining search, SQL, and visual tools)?
3. How well does the framework handle interleaved or compositional tool usage? Are there challenges in managing cross-tool dependencies or shared state?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a framework for Agentic Reinforcement Learning with Tool use (ARLT) that extends Reinforcement Learning with Verifiable Rewards (RLVR) beyond single-turn reasoning to multi-turn interaction. The framework decouples reinforcement learning and tool execution through standardized APIs, enabling integration of tools such as code execution, search, SQL, and vision processing as plugins. The implementation includes asynchronous rollout execution achieving up to 2× speedup over synchronous baselines. The authors adapt the GRPO objective to handle multi-turn trajectories with interleaved tool interactions, masking off-policy observation tokens from tools to ensure that only model-generated outputs/actions contribute to policy updates. The system is evaluated across six ARLT domains—mathematical reasoning, knowledge QA, SQL generation, visual reasoning, web search, and software engineering, where it matches or surpasses specialized systems.

### Strengths
- Provides much-needed orchestration and glue code that enables efficient tool-augmented rollouts.
- The abstraction of tools (BaseTool) is well-designed and general, accommodating a broad range of tool types, including stateful tools, beyond what the author's have already provided. 
- Upstream alignment with VERL and modular plugin design reduce maintenance overhead and promote reproducibility for future ARLT research.
- The asynchronous rollout mechanism is an obvious gain and clear engineering improvement, yielding measurable speedups. Minor nit: I think the "Saved Time" blocks in figure 2 are unncessary, and benefit would be more clearly conveyed if the time axes for the async plot was shorter than the sync setting.
- The GRPO-ARLT reward function is a well motivated adaptation of the GRPO reward function for multi-turn settings.
- VT-* family of models achieve parity with, and sometimes exceed, baselines.

### Weaknesses
- Comparisons with other agentic frameworks are limited. While the authors compare primarily in terms of tool coverage, this raises the question of why prior frameworks could not simply be extended with similar tools. Are there technical constraints that prevent such extensions, and how does Verl-tool’s runtime performance compare under equivalent workloads? Moreover, the runtime analysis only contrasts the authors’ own synchronous and asynchronous modes, without benchmarking against other existing frameworks.
- The framework’s scalability and stability under long-horizon or heavily parallelized tool use aren’t deeply analyzed.
- Reward design and credit assignment for complex multi-turn tool interactions remain underexplored and could limit generalization.
- Despite claims of multimodal capability, vision and video tools are only lightly evaluated, leaving unclear how robust the multimodal handling actually is.
- The baseline comparison is fair given differing tool scopes; while baselines use domain-specific integrations, VERL-TOOL’s goal is unification and efficiency rather than beating specialized systems.
- It is not entirely clear why the VT-* family of models occasionally exceed the performance of their task-specific baselines. The authors should explicitly enumerate any differences in training setup, hyperparameters, or reward design relative to those baselines. Given that this work is primarily a systems-level contribution, the goal should be to achieve parity with existing results while offering simpler configuration and unified infrastructure. Any improvements over baselines should be analyzed in detail; if such gains stem from differences in configuration or reward shaping, these should be clearly stated, and ideally verified through experiments under matched conditions.
- Section 3.3 tokenization: I'm not sure I see the issue. One could simply enforce the tokenizer to not merge across tag boundaries, so "\n<result>" is always treated as 2 tokens. Additionally, this is a non-issue if special tokens were assigned to opening and closing tags, which would guarantee no single token spans tag boundaries. In any case, this subsection is an implementation detail that I believe doesn't add to the paper.

### Questions
- How are tool failures or timeouts handled: are they retried, skipped, or penalized in the reward? What happens if tool calls exhaust tool worker resources (e.g. trigger OOMs etc.)
- How does agent performance scale with increasing numbers of tools? E.g. for each domain, what performance would a model with access to all tools at train and test time get?
- How much of the total training time is spent on tool execution versus model inference? (i.e., where does async actually help most?)

Aside:

In Section 3.3 (Parallel Tool Server Backend), the authors mention supporting Python’s ThreadPoolExecutor as a parallel execution backend. It’s worth noting that due to Python’s Global Interpreter Lock (GIL), true multithreading is not possible in standard CPython builds (while Python 3.13+ introduces no-GIL build variants, these are not yet widely adopted). Consequently, tools that rely on the GIL, such as those performing Python code execution or other compute-bound tasks, will effectively run via CPU time-slicing rather than in parallel. For true parallelism with Python stdlib, consider also supporting ProcessPoolExecutor as a drop-in alternative.

---

Overall, the framework is promising and well motivated, but the empirical evidence and clarity are insufficient for acceptance at this stage. The GRPO adaptation is a prudent, incremental improvements that the community can build upon. I would be happy to raise my score if the authors improve transparency in experimental configurations, standardize comparisons, and provide deeper analysis explaining the framework’s observed performance.

### Soundness
1

### Presentation
3

### Contribution
4
