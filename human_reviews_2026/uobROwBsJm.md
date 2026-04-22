# MCPMark: A Benchmark for Stress-Testing Realistic and Comprehensive MCP Use

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 10, 6, 6

## Abstract
The MCP standardizes how LLMs interact with external systems, forming the foundation for general agents.
However, existing MCP benchmarks remain narrow in scope: they focus on read-heavy tasks or tasks with limited interaction depth, and fail to capture the complexity and realism of real-world workflows.
To address this gap, we propose \texttt{MCPMark}, a benchmark designed to evaluate MCP use in a more realistic and comprehensive manner. It consists of $127$ high-quality tasks collaboratively created by domain experts and AI agents, each with a curated initial state and programmatic verification script. These tasks demand diverse CRUD operations and richer environmental interactions.
We evaluate cutting-edge LLMs using a minimal agent framework. The best-performing model, \texttt{gpt-5-medium}, reaches only $52.56$\% pass@1 and $33.86$\% pass\textasciicircum{}4, while other strong models including \texttt{claude-sonnet-4} and \texttt{o3} fall below $30$\% pass@1 and $15$\% pass\textasciicircum{}4. On average, LLMs require $16.2$ turns and $17.4$ tool calls per task, highlighting the stress-testing nature of \texttt{MCPMark}.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper introduces MCPMark, a benchmark designed to evaluate the realistic and comprehensive use of MCP. It includes 127 tasks across five MCP environments: Notion, GitHub, Filesystem, PostgreSQL, and Playwright. It uses a minimal agent that has a tool-calling/llm-calling loop to evaluate LLMs against these tasks. The authors evaluated popular models and found even the best performing model only achieves 52% at pass@1 and 33% at pass^4.

### Strengths
- The proposed benchmark contains realistic and diverse tools and tasks
- Each task includes automatic verification scripts.
- It uses a simple but standardized agent framework to just evaluate LLM tool calling capabilities
- It evaluates many different models and draw some interesting conclusions

### Weaknesses
None. I like the paper quite a lot.

### Questions
None. The paper is clearly written.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a new benchmark MCPMark for evaluating more relaistic use of Model Context Protocol (MCP) in context of LLM agents. Model Context protocol (MCP) standarizes how LLMs connect with external systems and tools. However, the paper argues that existing MCP bencmarks are often too narrow or not realistic. The proposed benchmark proposes to stress-test MCP use across 127 tasks in more realistic environments and larger interaction depth. The paper provides some interesting analysis on implicit vs explicit errors for state-of-the-art LLMs on MCPMark. Finally, they also show that best LLMs often require much more interactions (tool calls and execution turns) than those required by previous benchmarks.

### Strengths
* The paper highlights an interesting problem of evaluating realistic use of MCP in context of LLM agents.

* The analysis on implicit vs explicit errors and failure cases in Sec. 4.2 is interesting.

* The paper shows results on state-of-the-art LLMs on MCPMark (Tab. 3), showing the need for better stress testing for MCP use.

### Weaknesses
* Can the authors please provide details on the scalability of the current data curation pipeline? 
       *  As mentioned in Sec. 2.1 and L164, the data curation pipeline requires a human expert. Thus, it will be important to also discuss the scalability of the data curation pipeline.

* Also in Sec. 2.1, the authors mention that for each task "including computer science PhD students, front-end designers, full-stack & AI infra engineers, and AI investors—each task takes 3~5 hours of focused expert effort". While not a major concern, providing more details on impact of human expertise on the task quality will be interesting.

* Which scaffold is used for MCPMark-Agent?

* Also can the authors have some intuition for effect of the used agent scaffold on the numbers reported in Tab. 3?

* Finally, in Sec. 4, in the discussion on failure cases it is noted that most errors are implicit. Could this be fixed by better prompting or more precise problem statements or does this point to a more fundamental limitation? If possible it will be interesting to see some qualitative examples of these implicit errors.

### Questions
* The authors mention in Sec. 3, that more turns or cost does not equal better performance. This is actually consistent with findings in SWE agents or coding tasks. Have the authors verified if the trend also holds with test-time scaling and parallel rollouts?

Please see the weaknesses section for some additional questions.

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
4

### Summary
The paper adds to the collection of MCP benchmarks. The key distinguishing aspects of this benchmark relative to existing MCP benchmarks are the following:
- It introduces more complex tasks requiring larger number of turns
- It covers more operations (read/update/delete) relative to other benchmarks that are only read-heavy.

The benchmark ensures repeatable and reliable evaluation by setting up resettable initial states and the tasks are co-desgined by a human-LLM agent team, with a significant effort in reviewing and cross-reviewing the tasks to ensure quality control. 

The evaluations illustrate the key challenges of todays LLM-based agents in addressing the tasks. The analysis provide useful categorization of the failure modes.

### Strengths
- The paper improves the existing MCP benchmarking space by contributing a more complex collection of tasks with rigorous programmatic evaluation. 
- The benchmarking and analyses provide a useful empirical assessment of the state-of-the-art models (both closed and open) in tackling complex tasks using MCP servers.

### Weaknesses
I see two areas that can be improved:

- While I fully appreciate that creating these benchmarks take significant time and effort, the number of instances created overall and the pathway for creating more future instances dont appear to scale. Some justification of why 127 tasks are adequate (in a statistical sense) and representative (I do think has been argued for reasonably in the paper) needs to emphasized more clearly in the paper. 

- The choice of a simple agent for comparative model evaluation is a valid argument. However, it would still be useful to know what is the best performance that at least one single model could have achieved given more sophisticated agent formulations would be useful to know. In practice, this is how agents are going to be deployed. In some sense, one would like to what is the best performance that can be achieved on this benchmark with the best known techniques today.

- The arguments for why MCP specific benchmarks are necessary can be stronger. There are multiple multi-tool use benchmarks that exist (e.g. AppWorld, WebArena, and SWEBench). For example, benchmarks like AppWorld provide a rich set of problems defined over multiple applications covering nine different applications and have tasks that are much longer in terms of turns than proposed in this paper. the Why shouldn't we consider MCP wrapping around these multi-tool and multi-turn agentic benchmarks? Why invest effort into creating new benchmarks.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
