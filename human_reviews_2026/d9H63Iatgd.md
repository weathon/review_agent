# ToolFuzz: Automated Agent Tool Testing

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Large Language Model Agents (LLM Agents) leverage the advanced reasoning capabilities of LLMs in real-world applications. To interact with the environment, these agents require tools such as web searches or database APIs. As the agent provides the LLM with tools documentation alongside a user query, the completeness and correctness of this documentation is critical. However, tool documentation is often over-, under-, or ill-specified, impeding the agent's accuracy. Standard software testing approaches struggle to identify these errors as the documentation is usually in natural language. Thus, despite its importance, there currently exists no automated method to test agent tool usage. To address this, we present ToolFuzz, the first automated agent tool testing method. ToolFuzz combines LLMs with fuzzing to generate diverse and natural user queries causing tool runtime errors or semantically incorrect agent responses. We evaluate ToolFuzz on 139 tools from the community based open source LangChain and the production ready closed source Composio and find that all LangChain tools and the majority of Composio tools are erroneous. To validate the relevance of errors, identified by ToolFuzz, we design an automated pipeline to improve tool documentation. Specifically, we introduce two novel benchmarks—over 300 tasks, known ground truth, and real environments based on GitHub and terminal file management. Our automated tool-fixing pipeline increases accuracy from 22.9% to 35.4% on GitHub tasks and from 29% to 39% on file management tasks.ToolFuzz consistently outperforms the baselines and identifies 50% more unique errors while reducing the False Discovery Rate by 4.5x, making it a key component for building reliable AI agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on testing tool documentation, which is essential for the reliable performance of LLM-based agents. Standard software testing approaches fail to identify documentation-related errors that cause agent tool misuse. To address this gap, the authors propose ToolFuzz, an automated framework for testing tool documentation and uncovering two major types of failures: 1. Tool Runtime Errors and 2. Correctness Failures. ToolFuzz uses two techniques to uncover  errors: (1) by generating queries that lead to tool runtime errors, achieved by combining fuzzing techniques with LLM-based query generation, and (2) by generating queries that result in incorrect agent responses, using synonymous prompt generation and a series of cascading consistency and correctness checks at various stages of the agent’s processing.   
The authors also introduce a new benchmark that empirically demonstrates the importance of correct tool documentation, showing that improving documentation based on ToolFuzz findings enhances agent performance on real-world tasks.

### Strengths
- The paper is well-organized and clearly written.

- It explores a novel and underexplored problem — testing the documentation of tools rather than just their invocation or usage. Most prior works (I've read) focus on improving tool-use accuracy, while this work identifies the correctness and completeness of documentation as an equally critical factor for reliable agent behavior.

- The paper clearly defines two categories of errors: Tool Runtime Errors and Correctness Failure, and provides intuitive illustrations in Figure 1 that help clarify these distinctions.

- The introduction of a benchmark for real-world agent tasks demonstrates both the practical significance and applicability of the proposed framework. It shows that improving documentation correctness leads to tangible improvements in agent performance.

### Weaknesses
- Beyond the two defined error types: Tool Runtime Errors and Correctness Failure, there may exist other classes of documentation issues not covered by ToolFuzz — for instance, semantic ambiguity. The paper could discuss whether and how these might be incorporated in future extensions.

- Both detection pipelines heavily rely on LLMs — for generating prompts, interpreting tool outputs, and serving as oracles. Since results might vary across different LLMs. The paper could strengthen this by introducing rule-based validation components or cross-model agreement metrics to mitigate bias and improve robustness.

- The dependence on the LLM oracle for correctness judgment raises concerns about subjectivity and reproducibility. While effective, this oracle might yield inconsistent results under different LLMs or prompts.

### Questions
- As mentioned above, regarding the two main error categories — runtime and correctness failures — do they fully cover all types of tool documentation problems? If not, how might ToolFuzz be extended to capture additional documentation flaws (e.g., ambiguous or incomplete descriptions)?

- Since correctness checking relies on an LLM oracle, have the authors considered more robust alternatives, such as confidence scoring, or rule-based verification? How consistent are the results across different underlying LLMs?

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
This paper tackles an important and increasingly relevant problem: automated testing of tool-usage reliability in LLM agents. The proposed framework, ToolFuzz, innovatively combines fuzzing with LLM-driven natural language variation to identify runtime and correctness failures in a variety of agent tools. The evaluation shows that many real-world tools do indeed suffer from documentation and behavior inconsistencies, and the ToolFuzz method outperforms the graybox and whitebox baseline methods.

### Strengths
1. Addresses a real reliability bottleneck in LLM agents and emerging tool ecosystems, which is well motivated and important.

2. The proposed framework is practical and able to uncover real errors according to the evaluation.

3. Clear problem formulation with a well-engineered pipeline.

### Weaknesses
1. Evaluation scope restricted to two tool libraries; generalizability unclear.

2. Heavy reliance on LLM Oracle for consistency check. This may lead to subjective correctness judgments.

3. Writing can be improved. It is a little difficult to capture the novel contribution of the framework when reading Sec. 3 and 4.

### Questions
The experiments are mainly conducted with frontier LLMs like GPT and Claude. Have you tried with smaller open-source LLMs?

### Soundness
3

### Presentation
2

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
The paper introduces TOOLFUZZ, an end-to-end agent-centric framework for detecting and fixing tool errors in LLM agents. It uses LLM reasoning with fuzzing to expose runtime and semantic errors. The authors also introduce a new benchmark suite focusing on precise tool invocation for file management and GitHub tasks. TOOLFUZZ is evaluated on LangChain and Composio tools, showing significant improvements in accuracy and error detection.

### Strengths
1. The study addresses the underexplored problem of errors in tool documentation.
2. The method implements a structured, end-to-end workflow combining fuzzing, consistency checks, and LLM-based evaluation, ensuring systematic assessment of agent-tool interactions.
3. The experimental evaluation is thorough and well-designed.

### Weaknesses
1. The effectiveness of TOOLFUZZ is highly contingent on the reasoning and generative capabilities of the underlying LLM, making it sensitive to model quality.
2. The methodological novelty is limited, as TOOLFUZZ primarily integrates existing techniques without introducing fundamentally new algorithms.
3. The coverage of TOOLFUZZ is constrained, as it may fail to detect errors in tools with complex, context-dependent, or multi-modal inputs and outputs.

### Questions
1. Is the selection in the Automatic Documentation Fixing process manually chosen?
2. Can you explain the detailed settings of the variants of TOOLFUZZ?

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
To address LLM Agent unreliability from flawed tool documentation (under-, over-, or ill-specified), this paper proposes "TOOLFUZZ," the first automated agent tool testing method. It combines fuzzing with LLM-driven query generation to detect runtime failures and correctness failures, the latter via an invariance-based approach (synonymous prompts and cascading consistency checks). Evaluated on 139 LangChain and Composio tools, TOOLFUZZ found 50% more unique errors than baselines while reducing the False Discovery Rate (FDR) by 4.5x. Furthermore, auto-fixing documentation using TOOLFUZZ's findings significantly improved agent accuracy on two novel benchmarks , raising GitHub task accuracy from 22.9% to 35.4%.

### Strengths
1.	Based on fuzzing and LLM-driven prompt generation, this paper proposes an automated testing method for LLM Agent tools, capable of systematically discovering both runtime failures and correctness errors in tools. By identifying these defects and using them to automatically optimize tool documentation, the method significantly boosts the agent's accuracy in handling tool calls; for example, accuracy on GitHub tasks increased from a baseline of 22.9% to 35.4%, demonstrating its effectiveness in enhancing agent reliability.
2.	Furthermore, this paper introduces two novel benchmarks—File Management and GitHub tasks—specifically designed to evaluate tool-call accuracy, filling a gap in existing agent benchmarks.

### Weaknesses
1.	TOOLFUZZ is fundamentally limited to testing single-tool accuracy, failing to capture critical agent failures that arise from complex multi-step planning, intermediate multi-tool calls, or inter-API dependencies. Furthermore, the reliance on artificially synthesized "malicious queries" introduces an input bias, which undermines the generalizability of the static document fixes.
2.	Testing only 139 tools is insufficient to represent the vast and rapidly evolving LLM Agent API ecosystem, limiting the applicability of the findings. Crucially, even after document optimization, the final absolute accuracy remains below 40% (e.g., 35.4% on GitHub tasks). This low ceiling suggests the method primarily addresses basic syntactical and obvious semantic flaws, offering only a marginal boost to overall tool-call precision.
3.	The optimization is static and pre-deployment, lacking a mechanism for continuous, dynamic adaptation based on experience (e.g., DRAFT). This inability to learn from real-world long-tail errors, combined with a reliance on Prompt Engineering over deeper technical methods like SFT or RL, results in a shallower overall technical contribution.

### Questions
1.	How do you ensure your method's generated tests cover the majority of potential LLM errors, since many errors originate from multi-step planning, etc.?
2.	Your method lacks a dynamic optimization mechanism. How can the tool documentation be continuously updated and refined based on long-tail errors or experiences encountered during actual interaction?
3.	Despite testing 139 tools, this number is insufficient. Can you expand the toolset?

### Soundness
3

### Presentation
3

### Contribution
2
