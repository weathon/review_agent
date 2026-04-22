# GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
LLM routing has achieved promising results in integrating the strengths of di-
verse models while balancing efficiency and performance. However, to support
more realistic and challenging applications, routing must extend into agentic LLM
settings—where task planning, multi-round cooperation among heterogeneous
agents, and memory utilization are indispensable. To address this gap, we pro-
pose GraphPlanner, a heterogeneous graph memory-augmented agentic router
for multi-agent LLMs that generates routing workflows for each query and sup-
ports both inductive and transductive inference. GraphPlanner formulates
workflow generation as a Markov Decision Process (MDP), where at each step
it selects both the LLM backbone and the agent role (Planner, Executor, Sum-
marizer). By leveraging a heterogeneous graph, denoted as GARNet, to capture
interaction memories among queries, agents, and responses, GraphPlanner
integrates historical memory and workflow memory into richer state represen-
tations. The entire pipeline is optimized with reinforcement learning, jointly
improving task-specific performance and computational efficiency. We evalu-
ate GraphPlanner across 14 diverse LLM tasks and demonstrate that: (1)
GraphPlanner outperforms strong single- and multi-round routers, improv-
ing accuracy by up to 9.3% while reducing GPU cost from 186.26 GiB to
1.04 GiB; (2) GraphPlanner generalizes robustly to unseen tasks and LLMs,
exhibiting strong zero-shot capabilities; and (3) GraphPlanner effectively
leverages historical memories, supporting both inductive and transductive infer-
ence for more adaptive routing. Our code for GraphPlanner is released at
https://github.com/ulab-uiuc/GraphPlanner.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel LLM routing paradigm, an agentic router setting. Unlike traditional routers, the proposed GraphPlanner not only selects the backbone LLM but also assigns specific agentic roles, e.g., planner, executor, or summarizer, to solve the initial query. The router is parameterized by GARNet, a graph-based model that captures the relationships between queries, agentic roles, and responses. GARNet is optimized using PPO with a joint loss that balances task-specific performance and computational costs, enabling adaptive router learning. Extensive experiments across 14 diverse tasks, spanning in-domain and out-of-domain settings, as well as inductive and transductive scenarios, demonstrate GraphPlanner's SOTA performance, exceptional balance of cost and performance, and strong generalization capabilities.

### Strengths
1. The proposed agentic router setting is well-motivated and clearly articulated. While traditional single-round routers are limited to solving isolated queries and multi-round routers can only model sequential workflows, GraphPlanner addresses more complex graph-structured workflows that are under-explored in prior work.

2. GraphPlanner is designed with sound principles, featuring a lightweight implementation and effective contextual history preservation. This ensures that the agentic router is simple to implement and extend. Additionally, the joint optimization of the router, combining task-specific loss and cost constraints, allows the model to learn policies tailored to solved task while maintaining efficiency.

3. The experimental results are extensive, covering a wide range of downstream tasks in both in-domain and out-of-domain scenarios. The authors also include detailed cost-performance trade-offs and further analysis in inductive and transductive settings. These results highlight GraphPlanner's SOTA performance compared to single-round and multi-round routers, its robustness in inductive and out-of-domain tasks, and its favorable balance between performance and computational costs.

4. The paper is well-written, with vivid figures and tables that significantly enhance comprehension.

### Weaknesses
1. The description of Phase 1 optimization is somewhat unclear. For instance, in the case of Depth = 1 and Width = 3, does this imply a fixed agentic workflow where the first step always involves an initial planner role, followed by two roles (e.g., summarizer or executor)? Further clarification of this process would be helpful.

2. The paper lacks illustrative examples showing how the router assigns roles to specific models to successfully solve queries. Including such examples would better demonstrate how GraphPlanner operates in practice.

3. The experimented datasets appear relatively simple, as single-round routers already achieve good performance in some scenarios. It would be valuable to test GraphPlanner on more challenging benchmarks, such as agent-related benchmarks, which involve complex workflows and are better suited for role decomposition.

### Questions
Please refer to Weaknesses part

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
In this paper, the authors proposed the GraphPlanner, which is a graph-based LLM router for agentic LLM flow prediction. Given a constructed workflow and history graph, the GraphPlanner predicts the best agent role and the LLM for the next step. by training the GNN model with PPO, it can generalize to unseen tasks and models.

### Strengths
- The proposed methods show significant improvement over baseline methods under different scenarios.
- The authors provide a comprehensive ablation study to prove the effectiveness of the proposed method.

### Weaknesses
- The writing can be improved. In particular, I am pretty confused about the graph construction part.  Why are all nodes connected to the role hub node? Is there any particular consideration behind this design? For multi-round routing, will there be multiple role hub nodes? How are different rounds connected in the graph? Will the role hub node encode the role information of both the history graph and the workflow graph? I believe a more detailed illustration or figure is needed to allow the reader to better understand it.
- For the agent role, the Graphplanner defined three different roles.  I am wondering what the rationale behind it is, and I am curious about whether the proposed methods can generalize to unseen roles after training. 
- What's the time cost for GraphPlanner compared to other baselines under different settings? What is the training time for GraphPlanner and other baselines?
- The major contribution and the source of performance improvement come from the history graph. I am wondering, is it necessary to construct historical information into a graph? What if I simply describe all historical information to a (small )LLM and use the same training pipeline to optimize it?

### Questions
See above?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GraphPlanner, a heterogeneous graph-based agentic router that extends LLM routing beyond static or multi-round settings into dynamic multi-agent coordination. The paper formulates workflow generation as a Markov decision process, where at each step both the LLM backbone and the agent role (Planner, Executor, Summarizer) are selected. A novel graph neural network, GARNet, integrates both the current workflow graph and historical interactions, enabling inductive and transductive inference. The proposed method is trained using proximal policy optimization. Experiments across 14 tasks and 6 domains show improved accuracy and reduced GPU cost over prior routers, with strong zero-shot generalization.

### Strengths
- **S1.** Novel formulation of LLM routing as a graph-based agentic workflow generation problem using an MDP framework. The reinforcement learning-based optimization adds a dynamic decision-making aspect missing in static routers.

- **S2.** Integration of historical context through GARNet for inductive and transductive inference provides a principled way to leverage past interactions.

- **S3.** Comprehensive evaluation across several tasks and domains. Demonstrated efficiency with substantial reductions in GPU usage and token consumption. Zero-shot generalization to unseen LLMs and tasks shows strong adaptability and robustness.

### Weaknesses
- **W1.** The system fixes roles to Planner, Executor, Summarizer, which may constrain scalability to more complex or hierarchical workflows.

- **W2.** The experiments fix depth = 1-2 and width = 2-3, which seems too limited for realistic multi-step tasks. 

- **W3.** The paper omits recent agent workflow generation systems (e.g., ADAS, AFlow, AgentSquare), which are directly relevant to the *workflow generation* claim in Table 3.


- **W4.** Insufficient discussion of real-world deployment or integration with tool-based or API-based LLM ecosystems, despite the *agentic* framing.

### Questions
- **Q1.** How would GraphPlanner perform when additional agent types (e.g., retriever, verifier) are introduced? Is the policy flexible enough to handle new roles dynamically?

- **Q2.** Why are other graph encoders (e.g., GAT, GraphTransformer) not compared or ablated against GARNet?

- **Q3.** What motivated the use of Longformer embeddings? Was long-context modeling empirically necessary?

- **Q4.** Can the authors clarify what constitutes “existing LLM workflows” in Phase 1 evaluation (Table 1)? Were these synthetic or from real systems?

### Soundness
3

### Presentation
2

### Contribution
2
