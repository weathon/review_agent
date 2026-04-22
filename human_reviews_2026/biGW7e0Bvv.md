# TheMCPCompany: Creating General-purpose Agents with Task-specific Tools

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 8, 2, 2

## Abstract
Since the introduction of the Model Context Protocol (MCP), the number of available tools for Large Language Models (LLMs) has increased significantly. These task-specific tool sets offer an alternative to general-purpose tools such as web browsers, while being easier to develop and maintain than GUIs. However, current general-purpose agents predominantly rely on web browsers for interacting with the environment. Here, we introduce TheMCPCompany, a benchmark for evaluating tool-calling agents on tasks that involve interacting with various real-world services. We use the REST APIs of these services to create MCP servers, which include over 18,000 tools. We also provide manually annotated ground-truth tools for each task. In our experiments, we use the ground truth tools to show the potential of tool-calling agents for both improving performance and reducing costs, assuming perfect tool retrieval. Next, we explore agent performance using tool retrieval to study the real-world practicality of tool-based agents. While all models with tool retrieval perform similarly or better than browser-based agents, smaller models cannot take full advantage of the available tools through retrieval. On the other hand, GPT-5's performance with tool retrieval is very close to its performance with ground-truth tools. Overall, our work shows that the most advanced reasoning models are effective at discovering tools in simpler environments, but seriously struggle with navigating complex enterprise environments. TheMCPCompany reveals that navigating tens of thousands of tools and combining them in non-trivial ways to solve complex problems is still a challenging task for current models and requires both better reasoning and better retrieval models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
```
I used LLM to fix the grammar of the Official Review, but all opinions are my own
```
Many current AI agents rely heavily on general-purpose tools such as web browsers to perform tasks—whether retrieving data or operating software. This approach is inefficient and costly, and it fails to capture how most real-world professional work is done. For example, in enterprise environments that use Azure for cloud services or GitLab for code management, there exist specialized APIs or SDKs that are far more suitable for task automation, but current agents struggle to leverage them effectively.
To address this, the authors built a “tool library” that converts common enterprise services—like Azure, GitLab, and internal communication tools—into over 18,000 “AI-native tools” (called MCP tools). For instance, instead of manually updating database settings via the Azure UI, an AI agent can now directly call a dedicated “update Azure database version” tool.
They also constructed a simulated company environment (“TheMCPCompany”) to benchmark AI agents in realistic work settings. The environment includes both simple tasks (e.g., labeling files) and complex ones (e.g., fixing a faulty cloud service). A baseline agent (“MCPAgent”) is introduced, which must first discover relevant tools for a given task (e.g., finding “check Azure database version” or “restart service”) and then use them to solve the problem.

Although the paper doesn’t introduce a novel method, I find the problem setting very meaningful and the work potentially impactful. I recommend accepting this paper.

### Strengths
1. The idea of turning real-world software APIs into standardized, callable tools for AI agents is highly interesting.

2. The dataset and environment are both valuable community resources that can enable further research.

### Weaknesses
1. Some sections, especially the data construction process, are not clearly written.

2. It remains unclear how one might systematically improve the agent’s ability to use such tools efficiently.

### Questions
1. The paper seems to lack a clear explanation of how the data were constructed. The current narrative is somewhat scattered, and it’s hard to follow the full pipeline from tool creation to task setup.

2. This direction is quite open-ended, and I’m curious how one might improve generalization in this setting. Since your tools appear highly domain-specific, training on them might not help the model handle out-of-domain (OOD) scenarios. Have you considered splitting the tools and environments into disjoint train/test sets—for example, using part of the tools to fine-tune/RL a model like Qwen, and then testing on unseen tools or environments? Such an experiment would be very informative, and if you could include results along these lines, I’d be even more inclined to advocate for acceptance.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces TheMCPCompany, a benchmark for evaluating general-purpose AI agents that primarily interact with their environment through a large set of task-specific tools (over 18,000) rather than a few general-purpose tools like web browsers. The core contributions include: (1) the creation of this large-scale, realistic benchmark based on the Model Context Protocol (MCP), which includes complex tasks adapted from a software company simulator and new challenges for the Azure cloud platform; (2) an extensive evaluation comparing browser-based agents to agents with access to either a pre-selected 'oracle' tool set or a tool-retrieval mechanism; and (3) key findings that demonstrate the potential of task-specific tools to improve performance and reduce cost, while also exposing the significant difficulties agents face in navigating and combining thousands of tools in complex enterprise environments.

### Strengths
1. This paper tackles a very relevant and interesting angle: understanding the capabilities of general-purpose agents when they are equipped with large, heterogeneous tool collections. Studying how LLMs perform as the number of available tools scales is highly realistic and timely, given the fast-evolving ecosystem of MCP tools.

2. The writing and motivation are experienced and clear, making the paper's contributions easy to grasp. The design of the benchmark is intuitive and well-justified; it builds sensibly on prior work by replacing a few general-purpose tools with a massive set of task-specific ones, thereby creating a novel and challenging testbed.

3. The experiments yield meaningful and interesting observations. For instance, the note that GPT-5's excellent performance is partly due to its perseverance provides genuine insight that maps model behavior directly to success and failure patterns. The clear performance gap between models with oracle tools versus tool retrieval effectively pinpoints the current challenges in tool discovery and usage.

### Weaknesses
The main weakness lies in the insufficient details provided for the MCPAgent's tool-finding function. This module is central to the paper's investigation of agents in large-scale tool environments, yet its implementation is only briefly described. Specifically, the choice of the embedding model is a critical design decision that could significantly impact retrieval quality and, consequently, the overall agent performance. The authors state, "We use OpenAI’s text-embedding-3-large model," but there is no discussion or ablation study on how this choice affects the results. Would a different embedding model change the performance gap between models, especially for smaller ones like GPT-5-mini? Without this analysis, it's difficult to fully assess the robustness of the retrieval approach and the conclusions drawn from it.

### Questions
1. Could you provide more details on the tool finding function? For example, what was the value of k (the number of tools returned per query), and how was it determined, and were there any strategies for handling the diversity of tool schemas (e.g., name vs. description weighting) during embedding?

2. How sensitive are your key results, especially the poor performance of smaller models with retrieval, to the choice of the embedding model? Did you experiment with any other models, and if so, were the conclusions consistent?

3. The error analysis is insightful but brief. For the complex Azure tasks where all models failed, could you provide more detail on the specific types of reasoning failures? For example, were the issues more related to flawed problem decomposition, an inability to understand tool dependencies, or something else? A more detailed breakdown here would be very valuable for the community.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper provide a new benchmark to evaluate the tool use and tool calling of LLM agent with MCP tools. This benchmark is notable for its scale, including over 18,000 MCP tools.

### Strengths
1. The MCPAgent incorporates 18,000 tools and introduces a gateway MCP server to retrieve the tools relevant to each user query, thereby improving performance and reducing operational costs.
2. This paper evaluates the MCPAgent on challenging tasks that reflect the complexity of real scenarios.

### Weaknesses
1. Although constructing a standardized set of MCP tools requires substantial engineering effort, the novelty of this paper appears to be limited.
2. Some experiment setups are confusing. For example, in Table 2, the comparison between the **MCPAgent** and the **Oracle Tool Set** supports the claimed advantages of introducing a gateway MCP server. However, it is unclear why the **MCPAgent** is also compared with the **browser-based agent**, given that their functionalities and supported capabilities differ significantly.

### Questions
1. Could the authors further clarify the novelty of this work?
2. What is the reasoning behind comparing the **MCPAgent** with the **browser-based agent**? Their functionalities and supported capabilities differ substantially.
3. Is the MCP tool set fixed at 18,000 tools in the experiments? Does the benchmark support a dynamic tool set? A dynamic setting might better capture real-world scenarios where available tools evolve over time.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents TheMCPCompany, a benchmark and evaluation framework for testing LLM-based agents equipped with Model Context Protocol (MCP) tools. The work extends TheAgentCompany by integrating various enterprise services (Azure, GitLab, RocketChat, Plane, ownCloud) through the MCP interface, resulting in over 18,000 tool endpoints. The authors also propose MCPAgent, a tool-retrieval agent capable of discovering and invoking MCP tools automatically. Experiments are conducted on several proprietary models (GPT-4.1, GPT-5, Claude Sonnet, Opus), showing that MCP-based agents outperform browser-based baselines in cost and accuracy. The paper aims to highlight the potential of large-scale MCP environments for real-world agent evaluation.

### Strengths
- Strong engineering contribution: Implements a large, fully functional MCP benchmark with 18,000+ tools across enterprise services.
- Systematic evaluation pipeline: Builds upon TheAgentCompany with added realism (Azure integration).
- Empirical comparison: Includes quantitative cost and accuracy analysis between MCP and browser-based setups.
- Reproducibility commitment: The authors intend to release code, MCP servers, and Terraform scripts.

### Weaknesses
- Limited model coverage: Only closed-source models from OpenAI and Anthropic are evaluated; Gemini and other open-source models (eg., DeepSeek-V3, Qwen3, Llama) are excluded. This limits generalizability.
- Lack of retrieval comparison: The paper does not directly compare MCPAgent with traditional retrieval-based methods, making it unclear whether MCPAgent offers genuine advantages.
- Narrow task scope: The actual benchmark tasks are mainly Azure tasks, and other major components (e.g., TheAgentCompany) are reused without meaningful extension. As a benchmark paper, this is insufficient.
- Weak analysis of MCPAgent: The paper provides little insight into how MCPAgent performs tool discovery or why it succeeds/fails in specific cases.
- 18,000-tool claim not substantiated: Although the paper highlights a huge MCP tool set, it never reports how many tools are actually useful or invoked during evaluation.
- Lack of concrete examples: The two main contributions—Azure tasks and MCPAgent—are not illustrated with examples or reasoning traces, making the work difficult to interpret and assess.
- Lack of comparison with existing MCP benchmarks: The paper does not include direct comparisons with other MCP-based benchmarks (eg., MCPVerse or LiveMCPBench).

### Questions
- How many of the 18,000 MCP tools are actually used in the benchmark? Can you provide tool invocation statistics?
- Why were open-source models (e.g., DeepSeek-V3, Qwen3, and Llama) excluded from the evaluation?
- Can you show a concrete example of an Azure composite task and how MCPAgent solves (or fails to solve) it?
- How does MCPAgent compare to traditional retrieval-based systems or other MCP-agent implementations?
- What are the main factors that limit agent performance on Azure composite tasks?

### Soundness
2

### Presentation
3

### Contribution
2
