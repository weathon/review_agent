# RPG: A Repository Planning Graph for Unified and Scalable Codebase Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large language models excel at generating individual functions or single files of code, yet generating complete repositories from scratch remains a fundamental challenge. This capability is key to building coherent software systems from high-level specifications and realizing the full potential of automated code generation. The process requires planning at two levels: deciding what features and modules to build (proposal stage) and defining their implementation details (implementation stage). Current approaches rely on natural language planning, which often produces unclear specifications, misaligned components, and brittle designs due to its inherent ambiguity and lack of structure. To address these limitations, we introduce the Repository Planning Graph (RPG), a structured representation that encodes capabilities, file structures, data flows, and functions in a unified graph. By replacing free-form natural language with an explicit blueprint, RPG enables consistent long-horizon planning for repository generation. Building on RPG, we develop ZeroRepo, a graph-driven framework that operates in three stages: proposal-level planning, implementation-level construction, and graph-guided code generation with test validation To evaluate, we construct RepoCraft, a benchmark of six real-world projects with 1,052 tasks. On RepoCraft, ZeroRepo produces nearly 36K Code Lines and 445K Code Tokens, on average 3.9× larger than the strongest baseline (Claude Code), and 68× larger than others. It also achieves 81.5% coverage and 69.7% test accuracy, improving over Claude Code by 27.3 and 35.8 points. Further analysis shows that RPG models complex dependencies, enables more sophisticated planning through near-linear scaling, and improves agent understanding of repositories, thus accelerating localization. Our data and code are available at https://github.com/microsoft/RPG-ZeroRepo.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents RPG (Repository Planning Graph), a prompting framework for generating software repositories with LLMs. RPG generates sketches of a repository in graphs, where nodes represent functions and edges represent dependencies and information flow. The actual code is then generated from this sketch graph.

Experiments with two backbone models (o3-mini and Qwen3-Coder) show that RPG outperforms some other frameworks such as MetaGPT and Paper2Code on six repositories.

### Strengths
The problem (generating an entire repository from nothing) is interesting. Experiments show improvement compared with multiple other frameworks.

### Weaknesses
- In essence, RPG is a prompting framework and does not involve training better models, limiting its novelty and contribution. Experiments only involve two API-based models, raising questions about RPG's applicability in many real-world applications that require deploying models locally.

- The problem setting - generating an entire repository from scratch - is unrealistic. The most common application of LLMs in software engineering is resolving an existing issue or implementing new features in an existing repository.

- Line 106-107: I'm confused by the structure here. I can understand that "multi-agent systems assign roles, while workflows follow stages". But then the authors state "Industrial systems automate SWE tasks". How is this parallel to the previous two? Aren't multi-agent systems and workflows supposed to automate SWE tasks?

- Figure 1 is too crammed and looks messy.

- References are not formatted professionally. For example, Line 522, 527, 587 are all arxiv preprints, yet three different formats are used. The authors also cited many papers' preprint versions when there are peer-reviewed versions. For example, Epicoder (Line 598) is published at ICML, while DeepSeek-R1 (Line 528) is published at Nature.

- Details are lacking in Section 3.2. What models are used for graph node embedding? What models are used for feature path retrieval?

### Questions
- According to Figure 1 and 2, edges in the RPG indicate hierarchy and information flow. Does this imply that graphs in RPG are always acyclic? If so, what's the advantage of the proposed RPG over graphs automatically constructed with program analysis tools, such as file structure tree, import dependency, and control flow? If not, can the authors provide an example of cyclic graph?

- Line 238: there are only three subplots (A, B, C) in Figure 1, and no D. I suppose this is a typo.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Repository Planning Graph (RPG), a structured representation that unifies proposal-level capability planning with implementation-level file, class, and function dependencies for repository generation. Building on RPG, the ZeroRepo agent conducts planning, graph-guided code synthesis, and test validation. The authors also release RepoCraft, a benchmark derived from six paraphrased real-world repositories (1,052 tasks). ZeroRepo substantially outperforms multi-agent, workflow, and terminal baselines on RepoCraft, reaching 81.5% coverage, 69.7% pass rate, and dramatically larger codebases, with analyses showing near-linear scaling and efficiency benefits from graph-guided localization.

### Strengths
- RPG provides a clear structured alternative to natural-language planning, coupling capability decomposition with file-level structure and data-flow constraints, with the design well illustrated in Figure 2. This representation offers an explicit, interpretable blueprint that can support consistent long-horizon planning in repository generation.
- RepoCraft offers a comprehensive benchmark with six substantial real-world repositories, paraphrased specifications, and 1,052 evaluation tasks derived systematically from test files, filling a gap for repository-scale assessment of agent planning capabilities.
- ZeroRepo achieves substantial gains over strong baselines, demonstrating +27.3% coverage and +35.8% pass rate improvements compared to Claude Code (Table 2) while generating significantly larger repositories (36K LOC, 445K tokens vs. 10.6K LOC, 105K tokens),.

### Weaknesses
- ZeroRepo relies heavily on the 1.5M-node EpiCoder feature ontology as its knowledge base for proposal-level planning. However, it is unclear whether all competing baselines had equal access to comparable structured priors.
- The coverage and novelty metrics depend critically on K-means clustering with LLM adjudication, and correctness evaluation relies on LLM-adapted test cases. Beyond the Gold Project sanity check (81.0% pass rate on human-developed repositories), there is limited independent evidence that these automated evaluation pipelines produce stable and reproducible judgments, particularly for edge cases where generated functionality diverges significantly from reference implementations.
- Key architectural decisions—including reliance on the feature tree for initial planning, data-flow encoding at multiple hierarchy levels, test-driven code generation with majority-vote diagnosis—lack comprehensive ablation studies. The paper only provides an ablation for graph-guided localization (Table 4); without component-level comparisons, it is difficult to attribute the reported performance gains specifically to the RPG representation versus confounding factors like the planning budget, LLM backbone choice, or architectural engineering.

### Questions
1. Was the EpiCoder ontology filtered or deduplicated to ensure it does not contain paraphrases or derivatives of the six target repositories used in RepoCraft? Did all baseline methods have equal access to this knowledge base?
2. Beyond the Gold Project validation on human-developed repositories, can the authors provide evidence (e.g., correlation with manual evaluation) that the LLM-based coverage/novelty pipeline produces reliable judgments consistently across different types of generated code?
3. Which component—the feature tree initialization, data-flow encoding, or graph-guided generation—contributes most to the observed performance gains? An ablation removing RPG structure while maintaining the same planning budget would clarify this.

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
3

### Summary
The paper introduces the Repository Planning Graph, a structured graph representation that integrates proposal and implementation planning for repository-level code generation. Specifically, proposal-level planning decides what functionalities to include, and implementation-level planning decides how to realize them. RPG encoding functionalities, data flows, file structures, and class/function dependencies into a coherent graph. Based on this graph, the authors develop ZeroRepo, a framework that constructs and traverses the RPG to generate repositories from natural language specifications. In addition, the authors introduce a benchmark RepoCraft to evaluate the ZeroRepo. RepoCraft covers six real-world software projects with 1052 functional tasks. Experiment demonstrates ZeroRepo outperforms other baselines in coverage, test accuracy, and code scale.

### Strengths
1. The method integrates functional and structural dependencies, providing graph abstraction as an intermediate for natural language description and code repository implementation. The RPG is both explicit and fully machine interpretable.

2. The paper also provides a graph-guided code repository generation method. The method traverses the RPG in topological order, applying test-driven development to ensure incremental expansion while preserving stability.

3. The paper also provides a valuable testbed covering six projects and 1052 tasks, assessing coverage, accuracy, and code scale.

4. The experiment demonstrates significant improvement compared with SOTA methods. The additional scaling analysis shows near-linear growth in functionality and repository size.

### Weaknesses
1. The paper only includes a single ablation study without a graph, while the effectiveness of other components remains unclear (e.g., exploration strategy). As there are 3 generation level, it is important to conduct ablation study. For example, can we remove stage A?

2. RepoCraft relies on automatic localization and major-voting validation. This may introduce evaluation errors from LLMs.

3. Efficiency analysis of computational overhead from RPG constructions is not discussed.

### Questions
1. How is the accuracy for automatic localization and major voting? Providing a manual verification on a subset would strengthen the evaluation.

2. How would the cost compare to RPG+RepoCraft and traditional natural language based code repository generators?

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
5

### Summary
This paper tackles the challenging problem of scaling large language models (LLMs) from generating single files to complete, multi-file software repositories. The authors argue that current approaches, which often rely on natural language for high-level planning, are brittle and fail to scale due to ambiguity and a lack of structure.
To address this, the paper introduces three main contributions:
- Repository Planning Graph (RPG): A novel graph-based representation that serves as a structured blueprint for repository generation.
- ZeroRepo Framework: A graph-driven agent framework that builds and utilizes the RPG.
- RepoCraft Benchmark: A new, challenging benchmark.

### Strengths
- Moving from function-level to repository-level generation is arguably the most critical open problem in the area. This paper tackles it directly.

- The main results table (Table 2) shows a massive performance gap between the RPG-based ZeroRepo and all baselines (multi-agent, workflow, and terminal-based) on all key metrics: coverage, accuracy, and scale.


- The paper shows a clear benefit for the graph-guided localization, with the RPG speeding up localization tasks by 30-50% compared to a "w/o Graph" baseline.

### Weaknesses
- Dependency on External Knowledge Base: The "Proposal-Level Construction" stage seems heavily dependent on the "EpiCoder Feature Tree," a pre-existing 1.5M-node ontology of software capabilities. It is unclear how much of the system's strong performance is due to the novel RPG framework versus this massive, highly-structured knowledge base.

- Evaluation of "Novelty": The paper introduces "Novelty" as a metric and provides qualitative examples of new features (e.g., "Prophet forecasting" ). However, the paper does not seem to evaluate the functional correctness of these novel features. The main accuracy metrics (Pass/Voting Rate) are based on the 1,052 tasks from the reference projects. It's unclear if the "novel" features are correct, functional implementations or just plausible-sounding, well-structured stubs.

- Potential for Benchmark Leakage: The RepoCraft benchmark is based on extremely well-known projects (pandas, scikit-learn, etc.).

### Questions
What is the practical cost of running ZeroRepo? Could the authors provide an estimate of the total tokens, API calls, or wall-clock time required to generate one repository compared to the Claude Code baseline over its 30 iterations?

### Soundness
3

### Presentation
3

### Contribution
3
