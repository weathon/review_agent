# AgentsNet: Coordination and Collaborative Reasoning in Multi-Agent LLMs

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Large-language models (LLMs) have demonstrated powerful problem-solving capabilities, in particular when organized in multi-agent systems. However, the advent of such systems also raises several questions on the ability of a complex network of agents to effectively self-organize and collaborate. While measuring performance on standard reasoning benchmarks indicates how well multi-agent systems can solve reasoning tasks, it is unclear whether these systems are able to leverage their topology effectively. Here, we propose AgentsNet, a new benchmark for multi-agent reasoning. By drawing inspiration from classical problems in distributed systems and graph theory, AgentsNet measures the ability of multi-agent systems to collaboratively form strategies for problem-solving, self-organization, and effective communication given a network topology. We evaluate a variety of baseline methods on AgentsNet including homogeneous networks of agents which first have to agree on basic protocols for organization and communication. We find that some frontier LLMs are already demonstrating strong performance for small networks but begin to fall off once the size of the network scales. While existing multi-agent benchmarks cover at most 2--5 agents, AgentsNet is practically unlimited in size and can scale with new generations of LLMs. As such, we also probe frontier models in a setup with up to 100 agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose AgentNets, a benchmark designed to evaluate the scalable coordination, communication, and collaboration capabilities of multi-agent systems. They derive five representative problems from distributed computing and develop a robust, scalable agent-to-agent communication protocol to assess the performance of state-of-the-art LLMs across various network topologies. The authors further argue that current LLMs struggle to maintain performance when the network size scales up.

### Strengths
This work introduces five tasks capable of scaling multi-agent systems to up to 100 agents, surpassing most existing studies in this domain. The authors observe a clear performance degradation as the network size increases.
In addition, the paper benchmarks 27 network topologies using 10 state-of-the-art LLMs, which further strengthens the validity of the findings and insights.
Overall, the paper is well-structured and clearly written, with figures that are easy to interpret and follow.

### Weaknesses
1. Compared with prior work such as MacNet [1], this paper reaches a different conclusion — the performance of current LLMs degrades across the proposed five tasks. This discrepancy should be further discussed and explained in Section 5.3.
2. The analysis is relatively limited. For distributed computing problems, metrics such as average communication rounds, concurrency characteristics, and other protocol-level indicators should be reported to more thoroughly understand the bottlenecks and failure modes of the system.
3. The current evaluation setup appears to primarily measure the ability of LLMs to operate a multi-agent system rather than the multi-agent system’s capabilities itself. This raises concerns, particularly given the claim that “AgentNets measures the ability of multi-agent systems to collaboratively form strategies for problem-solving.” The authors should clarify the intended research focus and evaluation scope.

REF:
[1] Chen Qian et al., Scaling Large Language Model-based Multi-Agent Collaboration, ICLR 2025.

### Questions
1. What are the advantages of the proposed Message-Passing mechanism compared to existing protocols such as the Agent2Agent (A2A) Protocol? How does it differ in terms of scalability, robustness, and communication efficiency?

2. In Figure 5, the performance decreases as the number of nodes increases. Which specific network topology was used for this evaluation? Do different topologies, such as tree-based or star-like networks, result in significantly different performance trends?

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
4

### Summary
The paper proposes a benchmark for multi-agent reasoning. The benchmark, AgentsNet, tests, by varying the topology, the capabilities of a “network” (instead of looking at the single agents) to collaborate, communicate, etc. The benchmark is built from 5 distributed computing problems and tests networks of large size (~100 nodes) and several open- and closed-source LLMs.

### Strengths
The benchmark is indeed valuable, and the experiments are helpful to understand the state-of-the-art performance of agentic networks. The problems make sense in the context of decentralised coordination and “distributed intelligence”.

I appreciate that the authors reported the complexity of the algorithms in the distributed setting (though they could spend a sentence on stressing that log* is a very slow-growing logarithm function: log*100 is actually very, very close to log*5, so the network size they consider does not influence at all the complexity of the distributed algorithm in practice). That gives a measurable baseline and allows us to measure the “gap” between LLM-powered networks and formal algorithms.

While I do not find the results compelling per se (in particular, Findings 2 is well known in the already rich agentic-LLMs literature), I appreciated the paper as it is well-written.

### Weaknesses
The networks are not heterogeneous and comprise, in each evaluation, one model.
I reckon an evaluation where models are mixed would give interesting insights into “blocking” nodes and communication issues that arise when different models interact.

An analysis that would make this paper stronger is what kind of asynchronous algorithm agentic networks implement, and see if that varies with different sizes and models. Since the models do not see anything but their neighbours, I expect the algorithm to depend only on the number of neighbours and the prompt; nevertheless, that would be an interesting experiment that adds value to the paper.

In general, my main concern is that, after reading the paper and checking the results, I still ask myself “then what”, as I do not see the insights and findings being particularly useful or prescriptive to develop better communication/coordination methods or topologies.

### Questions
Q1. Can the authors make it clear how the research community can make use of the findings in the paper, and what the value of the contribution is for future research?

Q2. Some benchmarks can be handled without the need for LLMs coordination (e.g., colouring). Why didn’t the authors measure the complexity of the algorithm each network implemented and try to fit it with a function to get a complexity of what LLMs implement, compared to the optimal complexity they discuss at the beginning?

Q3. Table 2 is not clear to me. The last column is the average “accuracy”, yet the standard deviation seems too low when I look at the data in the other columns. Can the authors clarify this point?

Q4. One minor question is about the implementation: did the authors implement the protocols as asynchronous functions or sequential, temporised interactions? I reckon LangGraphs supports asynchronous exec.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces AGENTSNET, a benchmark for evaluating multi-agent LLM systems' coordination and collaboration capabilities through fundamental distributed computing problems (graph coloring, vertex cover, matching, leader election, consensus).

### Strengths
**Scalability to large agent networks**: Tests up to 100 agents (Figure 5), far exceeding existing benchmarks limited to 2-5 agents.

**Theoretically grounded tasks**: Leverages well-studied distributed computing problems with known complexity bounds (Table 1).

**Comprehensive model evaluation**: Tests 10+ frontier models across different cost-performance trade-offs (Figure 1).

### Weaknesses
**Unclear protocol differentiation**: The paper claims to develop a "new multi-agent protocol" but doesn't differentiate from existing protocols like A2A or ANP mentioned in the survey (arXiv:2504.16736). The LOCAL model adaptation (Section 4) appears standard without clear innovation.

**Missing baseline comparisons**: No experimental comparison with established multi-agent topology methods (MACNET, MAS-GPT, GPTSwarm， Dylan, Tree, Graph(mesh, DAG), Star topology) despite citing them. Only classical algorithms are compared (Appendix K).

**Superficial qualitative analysis**: Section 5.4's findings are trivial ("agents generally accept information from neighbors", "agents help neighbors resolve inconsistencies"). These observations are expected and provide no deep insights into coordination mechanisms.

**Poor visualization choices**: Figure 2 lacks clarity in message flow representation. Figure 3's icons and colored points are confusing without proper legend or explanation of what each visual element represents.

**Naming conflict**: "AgentsNet" is already used by "AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems", creating confusion. And not quiet a good naming choice for a benchmark and might causing overclaiming for the system.

**Limited analysis depth**: Despite the title emphasizing "coordination and collaborative reasoning," the paper provides minimal analysis of how these emerge or fail beyond binary success metrics.

### Questions
**Protocol comparison**: How does your message-passing protocol differ from A2A (Agent-to-Agent) and ANP (Agent Network Protocol) mentioned in the recent survey? Why not compare experimentally?

**Task suitability**: Traditional distributed computing problems focus on network latency, failures, data consistency, and security. How do you justify that graph-theoretic problems alone adequately evaluate LLM multi-agent coordination?

**Topology baselines**: Why exclude comparisons with MACNET's random topology or GPTSwarm's graph-based approach that you cite?

**Result presentation**: Why not highlight best results in Table 2 for easier interpretation?

**Coordination mechanisms**: What specific coordination strategies emerge? The paper doesn't analyze HOW agents coordinate, only WHETHER they succeed.

**Classical vs LLM gaps**: Table 9-10 show classical algorithms achieve near-perfect performance. What specific capabilities are LLMs lacking in this system?

**Token efficiency**: Table 11 shows massive token consumption (1M+ for 16 nodes). How does this scale economically for real applications?

**Error propagation**: How do errors propagate through the network? Is there analysis of failure modes beyond binary success?

**Heterogeneous agents**: All experiments use homogeneous agents. How would mixed-capability agents perform?

**Dynamic graphs**: Real multi-agent systems often have dynamic topologies. Why only test static graphs?

### Soundness
2

### Presentation
2

### Contribution
2
