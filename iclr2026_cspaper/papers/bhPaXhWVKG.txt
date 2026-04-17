000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Despite the promise of autonomous agentic reasoning, existing workflow generation methods frequently produce fragile, unexecutable plans due to unconstrained LLM- driven construction. We propose **MermaidFlow**, a framework that redefines the agentic search space through safety-constrained graph evolution. At its core, MermaidFlow represent workflows as a verifiable intermediate representation using Mermaid, a structured and human-interpretable graph language. We formulate domain-aware evolutionary operators, i.e., crossover, mutation, *insertion*, and deletion, to preserve semantic correctness, enabling efficient exploration of a highquality, statically verifiable workflow space. Without modifying task settings or evaluation protocols, MermaidFlow achieves consistent improvements in success rates and faster convergence to executable plans on the agent reasoning benchmark. The experimental results demonstrate that safety-constrained graph evolution offers a scalable, modular foundation for robust and interpretable agentic reasoning systems.

## 1 Introduction

Large language models (LLMs) are increasingly instantiated as modular agents that collaborate to solve complex tasks through structured workflows (Guo et al., 2024; Li, 2025a;b). These agentic workflows decompose problems into subtasks, assign them to specialized agents, and integrate intermediate outputs toward a shared goal. Moving beyond single-agent prompting, this multi-agent setting requires coherent planning and execution across agents with distinct roles and responsibilities. Designing such workflows involves reasoning over compositional graph structures that represent inter-agent dependencies, data flow, and semantic constraints, forming the foundation for scalable and adaptive multi-agent systems (Zhou et al., 2025). The lifecycle of agentic workflow is naturally structured into three layers: (1) *workflow planning*, which defines the structure of subtasks, agent roles, and information flow; (2) *code realization*, where the plan is translated into executable programs; and (3) *runtime execution*, where agents are instantiated and carry out their assigned behaviors. In many systems, these layers are collapsed
(e.g., Hu et al. (2024); Zhang et al. (2024c)): workflows are directly generated as *Python code* or serialized *JSON trees*, where planning decisions are entangled with implementation (i.e., through prompting-based generation of code or execution traces). As a result, workflows are often encoded in low-level formats where **structure** is implicit, **semantics** are entangled with imperative logic, and validity can only be assessed at runtime. This implicit representation hinders verifiability, reuse, and search, limiting the robustness and scalability of multi-agent systems.

Indeed, recent studies reveal that multi-agent LLM systems frequently fail due to brittle workflow logic and coordination breakdowns (Cemri et al., 2025; Zhang et al., 2024a; 2025c). These failures typically arise not from deficiencies in language models themselves but emerge from workflows that cannot be reasoned about, verified, or adapted. Without a structured representation of agent roles, task flow, and dependencies, systems struggle to detect errors before execution or to generalize behaviors across tasks. This points to a core limitation: **existing workflows lack the abstraction needed for** reliable planning.

# Mermaidflow: Redefining Agentic Workflow Generation Via Safety-Constrained Evolution- Ary Programming

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

![1_image_0.png](1_image_0.png)

To address the limitations of implicit, code-bound workflows, we introduce **MermaidFlow**, a declarative representation for agentic planning inspired by the **Mermaid graph markup language**1.

MermaidFlow defines workflows as **declarative graphs**, where nodes represent prompting agents and edges specify information flow (see Figure 1 for an illustrative example of the declarative graph encoded in Mermaid language, which is declarative, structurally explicit, and highly humaninterpretable). This high-level representation enables structural and semantics properties with static verification, e.g., *structure feasibility*, and *type-safe connections*, can be enforced at the graph level, offering a clear plan that is both **human-readable** and **programmatically analyzable**. By exposing explicit semantics and structure, MermaidFlow yields substantial downstream benefits in both workflow generation and evaluation, when LLMs are extensively employed to discover and evaluate workflows. These properties ultimately yield a more robust, and verifiable space for agentic workflow planning. MermaidFlow enforces a clear separation between symbolic planning and executable code, ensuring that workflow structures remain statically verifiable by design.

Building on this foundation, we further propose a novel **evolutionary programming** (EP) framework tailored specifically to explore MermaidFlow's structured graph space. Our EP approach employs safety-constrained operations, including node replacement, subgraph rewiring, and role-consistent insertions, to maintain workflow correctness throughout the search process. Furthermore, historical workflows generated during search accumulate as structured experience, enabling efficient reuse and adaptation across tasks. Together, MermaidFlow's declarative representation and EP search framework constitute a programmable and task-agnostic programming layer for agentic workflow generation, enabling efficient search with improved correctness, generalization, and adaptability. To our knowledge, this is the first agentic workflow framework to **guarantee static graph-level** correctness across the entire generation process. In summary, our contributions are threefold: (1) We introduce MermaidFlow, a declarative, verifiable graph representation for agentic workflow planning that cleanly separates planning from execution; (2) We develop a novel EP-based search framework leveraging structured mutation operators and workflow experience accumulation; and (3) We empirically demonstrate that MermaidFlow significantly outperforms existing code-based methods on standard agentic reasoning benchmarks, improving success rates, search efficiency, and interpretability.

## 2 Related Works

Agentic Workflows with LLMs Recent advances in multi-agent LLM systems have enabled structured collaboration among specialized agents to tackle complex, multi-step tasks. AFLOW (Zhang et al., 2024c), MaAS (Zhang et al., 2025b), and MASS (Zhou et al., 2025) formalize agent workflows using execution graphs and message-passing protocols to model multi-step reasoning. MetaGPT (Hong et al., 2024) and MAS-GPT (Ye et al., 2025) implement role-based orchestration by assigning domain-specific functions (e.g., product manager, engineer) and encoding Standard Operating Procedures (SOPs) to reduce cascading errors. Debate-based frameworks such as Multi- Agent Debate (Liang et al., 2024; Du et al., 2024) and DebFlow (Su et al., 2025) introduce structured critique among agents to promote output reliability. While these systems improve modularity and Figure 1: An illustration of the workflow lifecycle in MermaidFlow. The workflow is modeled as a declarative graph using Mermaid code, where nodes V[τ,α] and edges E[ρ] are explicitly defined with annotated prompts and roles (lines 3-8), styled and typed (lines 11–21), and connected via directed edges
(lines 24–30). This results in a statically verifiable, semantically typed, and structurally interpretable representation that serves as a unified interface for visualization, validation, and code generation.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 3 A Novel Declarative Graph Representation For Agentic Workflows

This section introduces a declarative graph representation for agentic workflows, built on **Mermaid** that is a structured, human-readable language with built-in static verifiability and graph render to help human directly observe the workflow. Departing from unstructured or token-level workflow representations, our workflow formalism leverages Mermaid's type-aware syntax to enable correctness by construction, symbolic manipulation, and modular workflow composition.

## 3.1 Declarative Workflow Graphs With Mermaid

We model each agentic workflow as a declarative computation graph with explicit typing, annotations, and semantic structure. Formally, we define a workflow graph as:

$$G({\mathcal{V}}_{[\tau,\alpha]},\;{\mathcal{E}}_{[\rho]}),$$
G(V[τ,α], E[ρ]), (1)
where V[τ,α]is a set of typed and annotated nodes, E[ρ]is a set of directed, role-labeled edges.

scalability, their workflows are typically encoded in imperative code or *loosely structured prompts*, i.e., formats that lack semantic abstraction and resist verification. Recent studies (Cemri et al., 2025; Zhang et al., 2024a; 2025c) identify fragile workflows, rather than model errors, as the primary source of failure in multi-agent systems. Our proposed MermaidFlow addresses this bottleneck by introducing a typed, declarative workflow space that supports safe construction, static validation, and structured exploration, advancing agentic reasoning beyond brittle prompt chaining. Workflow Representation The representation of agentic workflows governs not only how agents are composed, but also whether they can be verified, reused, or optimized. Natural language-based prompting methods, such as Chain-of-Thought (Kojima et al., 2022), ReAct (Yao et al., 2023), and Self-Refine (Madaan et al., 2023), are expressive but underspecified, lacking formal structure for validation. In contrast, code-centric approaches like AFLOW (Zhang et al., 2024c), ADAS (Hu et al., 2024), and ScoreFlow (Wang et al., 2025) generate executable Python or JSON trees directly, offering precision at the cost of brittleness and poor editability due to tightly entangled logic and implementation. Recent efforts explore more structured workflow abstractions. GPTSwarm (Zhuge et al., 2024) and FlowReasoner (Gao et al., 2025) organize workflows as agent interaction graphs, but lack formal semantics, e.g., no type enforcement, role validation, or support for systematic search. MetaGPT (Hong et al., 2024) and MAS-GPT (Ye et al., 2025) encode workflows through SOP-style templates and DSLs, but rely on rigid decomposition patterns that restrict flexibility. MermaidFlow departs from these by introducing a typed, declarative graph representation grounded in the Mermaid markup language. It makes role semantics and data flow explicit, allowing crucial graph-level structural constraints, such as role consistency, and type safety, to be enforced pre-execution, enabling safe reuse, adaptation, and search. Workflow Search and Optimization The structure of the search space fundamentally shapes how workflows are generated and optimized. AFlow (Zhang et al., 2024c) applies Monte Carlo Tree Search over executable graphs, while ADAS (Hu et al., 2024) explores code-level candidates via heuristics-guided expansion. Though systematic, both approaches operate over brittle code-centric representations, where small mutations often break correctness, necessitating expensive filtering. ScoreFlow (Wang et al., 2025) and G-Designer (Zhang et al., 2024b) adopt learned or continuous optimization strategies, adjusting prompt topologies or agent graphs via gradient-based tuning or neural controllers. However, these methods require differentiable feedback or training signals and offer limited support for enforcing structural validity. A complementary direction leverages evolutionary and population-based search. DebFlow (Su et al., 2025) refines workflows through iterative agent debates, while EvoFlow (Zhang et al., 2025a) evolves diverse workflows using task complexity-conditioned genetic search. Yet both approaches operate in loosely defined or weakly constrained spaces, where mutations often yield semantically invalid workflows. MermaidFlow closes this gap by introducing a structured, verifiable graph space equipped with safety-aware mutation operators. This design guarantees that every candidate is valid by construction, enabling scalable and principled workflow optimization.

![3_image_0.png](3_image_0.png)

Figure 2: Overview of the MermaidFlow framework. **Left**: Comparison between *imperative* (Python-based) and *declarative* (Mermaid-based) workflow representations. MermaidFlow models workflows as statically typed, verifiable graphs, enabling interpretable planning and structure-aware code generation. **Right**: Illustration of the safety-aware evolutionary programming process. Given historical Mermaid workflows, the EP sampler selects parent candidates and applies EP operators to generate new workflow candidates. An LLM-as-Judge then selects the final workflow for evaluation, and the results are used to update the population.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 This structure is instantiated using the **Mermaid** graph language, a lightweight, human-readable syntax for specifying typed graphs with semantically annotated components. Furthermore, Mermaid provides a declarative interface that supports symbolic manipulation and static validation. A real example is illustrated in Figure 1. Each node defines a symbolic identifier and type signature, while edges carry semantic labels (e.g., inputs) that describe data-flow. Next, we define each component of the workflow in the Mermaid domain to illustrate how the workflow aligns well with its Mermaid representation.

Nodes. Each node v ∈ V[τ,α]is a tuple (id, τ (v), α(v)), where id is a symbolic identifier, τ (v) = Tin → Tout denotes the type of that node, and attribute α(v) would provide some necessary information according to the type. As shown in lines 4–7 of fig. 1, each node element can be concisely defined in a single line of Mermaid script, for example: (id: C1, type τ : CustomOp, attribute α: role-simple_solver_1). Nodes represent typed declarative units that are interpretable and statically verifiable, and it can be easily understood by human.

Edges. Each edge (u, v) ∈ E[ρ] denotes a dependency annotated with a semantic label ρ(*u, v*),
indicating how information or control flows from u to v e.g., "input", "problem". Mermaid syntax supports these semantics with labeled arrows (e.g., A -> |input| B).

Types. All graph nodes are explicitly typed and semantically annotated, with types governing interface compatibility and ensuring valid workflow construction. By defining these types up front, we guarantee a consistent translation from Mermaid diagrams to Python code. For each task domain, we introduce dedicated node types for the operators and tools that have proven most effective. During code generation, these attributes direct the translator to emit the correct Python calls, making sure that all tool arguments are clearly specified. We provide a detailed type description in Appendix A.1. Each node explicitly defines its symbolic identity, type signature, and related attribution, while edges govern execution flow subject to type constraints. Unlike flat or post-validated node-link DAGs (e.g.,
JSON plans or token-generated programs), our declarative graph formalism introduces an abstraction layer that bridges symbolic reasoning and execution-level safety. This structure supports (static) correctness-preserving mutation and compositional reuse, which are the key properties we exploit in the constrained optimization process detailed next. To the best of our knowledge, this is the first agentic workflow representation that leverages a graph-oriented abstract coding language to enable more natural graph definition and manipulation. In the next section, we will formalize graph-

## 3.2 Agentic Workflow Search Space

manipulation actions in Mermaid and present a workflow-optimization method, further illustrating the advantages of using Mermaid for workflow representation.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 The declarative graph formalism introduced above induces a constrained search space over **agentic** workflows. We define the workflow together with related LLM factors as follows:
S =G(V[τ,α], E[ρ], C) ∈ GMermaid
 G |= Cstatic	, (2)
where GMermaid denotes the set of workflows expressible in the Mermaid graph language, and Cstatic captures structural constraints, such as type compatibility, role-consistent edges, and connectivity, automatically enforced by Mermaid's parser and extended structural schema. This built-in verifiability arises from Mermaid's declarative syntax and ensures that all elements of S are valid and executable by construction.

To enable optimization, we parameterize each node v ∈ V[τ,α] as a tuple v =m, p(τ, α), f(τ ),
following conventions from multi-agent systems (MAS) definition, where m ∈ M specifies the LLM configuration (e.g., model_name, temperature), p(*τ, α*) ∈ P is the prompt template determined by the node type τ and its argument α, and f(τ ) ∈ F denotes the input/output format associated with type τ .

V[τ,α] = {(m, p(τ, α), f(τ ) | m ∈ M, p ∈ *P, f* ∈ F} . (3)
The formula above demonstrates that, when interpreting a Mermaid workflow, each node can be directly mapped to a standard LLM agent instance. By assigning a type τ and parameter/attribute α, one can associate the node with a specific prompt p(*τ, α*) and input/output format f(τ ). **This** formulation emphasizes that every LLM agent can be consistently defined both within the Mermaid representation and in the general context of LLM agent configuration. The space is also **inductively closed**: type-compatible subgraphs can be composed without revalidation, while disconnected or cyclic fragments are excluded by construction. This closure property is non-trivial: prior workflow representations, especially those based on imperative or token-level programs, **lack structural guarantees**, and mutations frequently yield invalid or unrecoverable states. In contrast, our declarative design ensures that local edits remain within the valid region of S, enabling reliable and efficient search. Though intentionally bounded, S remains expressive enough to capture planning motifs such as hierarchical refinement and dataflow composition. By unifying generation, mutation, and verification within a single, compiler-verifiable substrate, it provides a semantically grounded foundation for structure-aware and safe workflow optimization.

## 4 Constraint-Aware Evolutionary Workflow Optimization

We introduce an evolutionary programming (EP) framework that operates directly over declarative Mermaid workflows. Leveraging Mermaid's typed and verifiable graph structure, we define correctness-preserving operators that enable safe, modular workflow evolution. Unlike prior approaches over unstructured or token-based spaces, all candidates in MermaidFlow are valid by construction, ensuring safe, compiler-checkable optimization throughout the search process.

## 4.1 Constraint-Preserving Ep Operators For Declarative Workflow Graphs

We define a set of atomic graph-level operators that drive workflow evolution within MermaidFlow.

Each operator acts over a candidate graph G(V, E) ∈ GMermaid ⊆ S, and is designed to be locally scoped, *type-consistent*, and *statically verifiable*, enabling every candidate to be validated by the Mermaid compiler at each step of the search. Below are the definitions of the operations, which will be used to verify the correctness of the newly generated workflow. Node Substitution. Changing the attributes of a specific agent v(τ, α) ∈ V to v(*τ, α*′). Like changing the corresponding role prompt or instruction.

Node Addition. Given an edge (va, vb) ∈ E, connecting from node va to node vb, insert a new node v′ to form (va, v′) and (v
′, vb) and disconnect (va, vb) if: Tout(va) = Tin(v
′), Tout(v
′) = Tin(vb)
according to their node type τ .

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

$$\forall G\in{\mathcal{S}},\,\forall{\mathcal{O}}\in\mathbb{O},\quad{\mathcal{O}}(G)\in{\mathcal{S}}$$
∀ G ∈ S, ∀ O ∈ O, O(G) ∈ S (4)
where O is the set of constraint-preserving operators over MermaidFlow graphs. That is, S is closed under all valid EP operations.

Definition 1. Let G denote the space of all candidate workflows, each G ∈ G represented as a directed graph (V, E). We define a static validator function Q : G → {0, 1}*, implemented by a* Mermaid parser/compiler, such that:

$$Q(G)={\begin{cases}1&{\mathrm{if}}\ G\in{\mathcal{S}}\\ 0&{\mathrm{otherwise}}\end{cases}}$$
$$({\boldsymbol{5}})$$
0 *otherwise* (5)
Here, S ⊂ G is the subset of workflows satisfying verifiability constraints such as workflow structure, well-typed I/O, role validity, and full connectivity.

By using EP operators above, from Lemma 1, given a Gt ∈ S, each change O(Gt) at step t leads to a graph Gt+1 = O(Gt) ∈ S. Given an initial graph G0 ∈ S, by induction, we know
∀t ∈ N+, Gt+1 = Ot ◦ Ot−1 · · · ◦ O0(G0) ∈ S. Thus, the evolution in the static Mermaid graph space remains the safe subspace. In MermaidFlow, when using an LLM to generate a new Mermaid graph, the resulting Mermaid code may sometimes violate predefined safety constraints. To address this, we implement a checker to verify whether the newly generated candidates conform to the defined workflow and operation rules. If any violations are detected, new workflows are regenerated. Thanks to Mermaid's simple and clear syntax, the code can be treated as structured text. This allows us to easily build a text-based analysis tool and incorporate custom rules into the checker. More implementation details can be found in Appendix A.2.

## 4.2 Evaluation And Selection In Workflow Populations

We frame each declarative workflow graph as an *experience* and maintain a population of scored experiences over time. At each optimization step t, the system tracks a history buffer: Whistory,t = {(si, scorei)}
t i=1, where si ∈ GMermaid denotes a structurally valid workflow, and scorei reflects its estimated performance.

At each optimization cycle, two parent workflows sa, sb are sampled from Whistory,t, typically via temperature-scaled softmax sampling according to following distribution: Pmixed(i) = λ·
1 t + (1−λ)·
P
exp(α·scorei)
n j=1 exp(α·*score*j )
, where t is the number of workflows in the history buffer, *score*iis the validation score of the i-th workflow, and the parameters α and λ control the influence of the scores, and Edge Rewiring. Given nodes {va, vb, vc*} ⊆ V* and (va, vb) ∈ E in the original graph G, rewire to
(va, vc) or (vc, vb) and disconnect (va, vb) if: Tout(va) = Tin(vc) or Tout(vc) = Tin(vb).

Node Deletion. Given a linear path va → vb → vc, delete vb and insert an edge (va, vc) if Tout(va) = Tin(vc). Subgraph Mutation. Let G1 ∈ GMermaid be a subgraph of the graph G ∈ S. Denote the input and output node set of G1 as I1 and O1, respectively. Let G2 ∈ GMermaid be a feasible graph with input and output node set I2 and O2. Replace G1 in G with G2 such that Tin(I1) = Tin(I2) and Tout(O1) = Tout(O2).

Crossover. Given {G1, G2*} ⊆ S* share a common interface node v (e.g., an ensemble node), swap subgraphs rooted at v to yield {G′1, G′2} such that the type and interface constraints are preserved, i.e., {G′1, G′2*} ⊆ S*. Each operator is applied at the level of Mermaid syntax, enabling compilerlevel validation of every candidate graph. By constraining transformations to preserve type and role integrity, MermaidFlow ensures that evolutionary search remains within the semantically valid subspace of workflows. In the case study fig. 4, there is a concrete example illustrating the crossover operator. Lemma 1 (MermaidFlow Transformation Invariance). Let S denote the declarative workflow space defined in Section 3.2. For any workflow graph G ∈ S *and any atomic transformation operator* O
defined above, the resulting graph G′ = O(G) *also belongs to* S: balances exploration-exploitation, respectively. After sample two different parent workflows sa, sb where sa ̸= sb. These are used to generate a candidate set through the evolutionary process:
Scandidates = {si| si = O(sa, sb), O ∈ O}
N
i=1 , (6)
where O denotes the set of correctness-preserving operators (Section 4.1), for some operator only sa involved and N is the candidate pool size. To avoid expensive rollout-based evaluation over the full population, we adopt an *LLM-as-judge* model that scores each candidate s ∈ Scandidates based on semantic fit, structure, and task relevance. Since all candidates in Scandidates are statically verified by the Mermaid compiler, they are guaranteed to be syntactically valid, type-safe, and structurally executable, dramatically reducing failure cases and increasing effective sample quality. We then select the highest-scoring candidate and update the history buffer:
Whistory,t+1 ← Whistory,t ∪ {(s
∗
child, Validate(s
∗
child))} , where s
∗
child = arg max s∈Scandidates LLM_as_Judge(s).

This experience-centric design, enabled by the declarative and verifiable structure of MermaidFlow, supports efficient, low-cost population evolution without compromising safety, correctness, or search quality. See Appendix A.3 for algorithmic details.

## 5 Experiments 5.1 Experiment Setup

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Baseline. We choose threefold of agentic baselines: (1) **Non-agentic reasoning methods**, including CoT (Kojima et al., 2022), ComplexCoT (Fu et al., 2023), and Self-Consistency (Wang et al., 2023). (2) **Hand-crafted multi-agent systems**, such as LLM-Debate (Du et al., 2024), LLM-Blender (Jiang et al., 2023), DyLAN (Liu et al., 2024), and MAcNet (Qian et al., 2024). (3) **Autonomous multi-agent** systems, including GPTSwarm (Zhuge et al., 2024), MaAS (Zhang et al., 2025b), AutoAgents (Chen et al., 2024), ADAS (Hu et al., 2024), and AFlow (Zhang et al., 2024c). Among them, GPTSwarm and MaAS incorporate trainable modules for assigning workflow structures, while AutoAgents, ADAS, and AFlow rely on an LLM to design the structure, consistent with our setting. More details on baseline setups are provided in Appendix A.4. Task and Benchmarks. We evaluate MermaidFlow on four public benchmarks covering two domains: **(1) math reasoning**, GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021); (2) code generation, HumanEval (Chen et al., 2021), and MBPP (Austin et al., 2021). For MATH benchmark, we follow AFlow (Zhang et al., 2024c) and MaAS (Zhang et al., 2025b) in using the same selected problems from four typical problem types in level 5. The dataset statistics are provided in Appendix A.5. Implementation details. We use a closed-source LLM (gpt-4o-mini-0718) as both the Optimization and Execution LLM, consistent with the setup in MaAS (Zhang et al., 2025b). The Optimization LLM is responsible for tasks such as generating promising workflows in Mermaid code, selecting from sampled workflows, evolving to new workflows, and translating Mermaid code into Python code. All models are accessed via API with the temperature set to 0. In each round, we generate four different schild candidates. To ensure experimental stability, complex operations such as crossover are applied with only a 10% probability. We set the number of iteration rounds to 20 for both Mermaid and AFlow, and to 30 for ADAS. The evaluation metrics are kept consistent with those used in AFlow and MaAS: for GSM8K and MATH, we report the Solve Rate (%) as the primary metric, while for HumanEval and MBPP, we report the pass@1 score.

## 5.2 Experimental Results

We compare MermaidFlow against 13 baselines on the GSM8K, MATH, HumanEval, and MBPP
benchmarks, as shown in Table 1. The results demonstrate that MermaidFlow consistently achieves the best performance across all tasks. Compared to methods that search for the next workflow in the Python field, such as ADAS and AFlow, our approach outperforms them by an average margin of 2.08% to 5.54%. On the MATH benchmark specifically, MermaidFlow exceeds the secondbest method AFLOW by 2.61%. For certain benchmarks, performance is primarily limited by the 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

Table 1: Performance comparison among Non-agentic reasoning methods, hand-crafted multi-agent

systems, and automated agentic workflows. All methods use gpt-4o-mini as the base LLM and

are evaluated on the test split, with results averaged over three runs. **Bold** indicates the best result; underline denotes the runner-up. MermaidFlow shows consistent improvements across all datasets.

*: Result reported in the MaAS paper, as the corresponding implementation for this dataset is not

available in their code.

Method GSM8K MATH HumanEval MBPP Avg. Vanilla 87.57 46.29 87.49 70.29 72.91 CoT (Kojima et al., 2022) 87.45 46.40 88.13 71.83 73.45 ComplexCoT (Fu et al., 2023) 86.89 46.40 87.49 72.36 73.29 SC (CoT×5) (Wang et al., 2023) 87.57 47.91 88.60 73.60 74.42

| available in their code. Method   | GSM8K   | MATH   | HumanEval   | MBPP   | Avg.   |
|-----------------------------------|---------|--------|-------------|--------|--------|
| Vanilla                           | 87.57   | 46.29  | 87.49       | 70.29  | 72.91  |
| CoT (Kojima et al., 2022)         | 87.45   | 46.40  | 88.13       | 71.83  | 73.45  |
| ComplexCoT (Fu et al., 2023)      | 86.89   | 46.40  | 87.49       | 72.36  | 73.29  |
| SC (CoT×5) (Wang et al., 2023)    | 87.57   | 47.91  | 88.60       | 73.60  | 74.42  |
| LLM-Debate (Du et al., 2024)      | 89.47   | 48.63  | 88.80       | 70.29  | 74.30  |
| LLM-Blender (Jiang et al., 2023)  | 88.35   | 46.92  | 88.68       | 77.05  | 75.25  |
| DyLAN (Liu et al., 2024)          | 89.98   | 48.54  | 90.42       | 77.30  | 76.56  |
| MacNet (Qian et al., 2024)        | 87.95   | 45.18  | 84.57       | 65.28  | 70.75  |
| GPTSwarm (Zhuge et al., 2024)     | 89.14   | 47.88  | 89.32       | 77.43  | 75.94  |
| MaAS (Zhang et al., 2025b)        | 91.47   | 52.19  | 91.57       | 82.17* | 79.35  |
| AutoAgents (Chen et al., 2024)    | 87.69   | 45.32  | 87.64       | 71.95  | 73.15  |
| ADAS (Hu et al., 2024)            | 88.35   | 43.18  | 84.19       | 77.05  | 73.69  |
| AFlow (Zhang et al., 2024c)       | 90.11   | 52.81  | 90.08       | 81.67  | 78.67  |
| MermaidFlow (Ours)                | 92.39   | 55.42  | 92.87       | 82.31  | 80.75  |

capabilities of the Execution LLM. For example, in HumanEval and GSM8K, the baseline (Vanilla)
performance is already high, so improvements from architectural optimization are less significant. In contrast, for benchmarks where the baseline performance is relatively low, such as MATH and MBPP, our method demonstrates a more substantial impact. Overall, MermaidFlow's average score across these tasks is 80.75%, which is 1.40% higher than the highest average among all baselines (79.35% by MaAS), fully demonstrating the robustness and superiority of our approach across different problems.

5.3 ABLATION STUDY
Evolution Efficiency To evaluate the effectiveness of our approach, we compare the learning curves of MermaidFlow and AFlow on the MATH dataset, as shown in Figure 5.3. MermaidFlow demonstrates a more consistent improvement in workflow quality during training and better generalization to the test set. The core difference between MermaidFlow and AFlow lies in the search space. AFlow operates directly on Python code, applying textual edits with prompts constraints. This approach often leads to invalid and nonfunctional programs, with only a 50% success rate in generating executable code. In contrast, MermaidFlow evolves workflows at the graph level using Mermaid, a domain-specific language that enables structured manipulation (e.g., adding, deleting, or mutating nodes). This representation is better suited for LLM-based optimization (e.g., gpt-4o-mini), consistently yielding >90% success rate in producing valid Python code. This reliability enables more effective exploration and optimization of workflow space. Thanks to Mermaid's reliable generation rate and lightweight representation, it achieves better token efficiency. When AFlow and MermaidFlow both surpass 52% on the MATH dataset, they consume 6.9e4 and 2.7e4 tokens respectively, with MermaidFlow requiring only about half the cost of AFlow.

Figure 3: An illustrative figure com-

![7_image_0.png](7_image_0.png) paring the highest solve rates on the MATH dataset between MermaidFlow and AFlow on the training set (119 problems) and test set (486 problems) across optimization iterations.

Impact of Optimization LLM Scale We investigate how the choice of Optimization LLM influences the quality of workflows in MermaidFlow by evaluating more capable models on the HumanEval and GSM8K benchmark. Specifically, we compare the effectiveness of larger models, 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 These results show that MermaidFlow consistently discovers higher-quality workflows at later rounds, indicating a more stable and productive search trajectory compared to AFlow.

![8_image_0.png](8_image_0.png)

Figure 4: A case study on the HumanEval dataset showcasing how MermaidFlow evolves structured agentic workflows through evolutionary programming (with a detailed example of the *crossover* operator). The declarative graph representation also enables reliable translation of workflow graphs into *executable python code* (zoom-in view recommended).

In this case study, we present an example of how to generate a new workflow from given parent workflows using the Mermaid representation. During the evolutionary process, a new workflow can be derived from either a single parent workflow or two parent workflows, depending on the type of update operator applied. In the specific example shown in Figure 4, which is based on solving problems from the HumanEval benchmark, Workflow_8 is generated based on Workflow_4 and Workflow_5. Each parent contributes distinct advantages: Workflow_4 is the first to Table 2: Comparison of different optimization LLMs on HumanEval and GSM8K datasets.

| Dataset   | Claude 3.5   | GPT-4o   | GPT-4o-mini   |
|-----------|--------------|----------|---------------|
| HumanEval | 93.13        | 94.66    | 92.87         |
| GSM8K     | 93.83        | 93.94    | 92.39         |

such as Claude 3.5 and GPT-4o, in generating new workflows, while keeping GPT-4o-mini fixed as the Execution LLM. Results are summarized in Table 2. As a result, higher-capacity optimization LLMs consistently yield better performance across both HumanEval and GSM8K. This consistent trend underscores a core strength of MermaidFlow: its well-structured, statically verifiable search space enables even modest improvements in optimization quality to translate directly into more functional, high-reward workflows. Optimal Stopping Point Analysis We investigate the advantages of using Mermaid as the workflow representation in workflow update control. A stable and reliable search process requires controllable and well-defined update steps. With Mermaid, updates can be expressed through precise graph-based operations such as adding nodes, deleting nodes, or modifying edges. These structured operations help ensure that the newly generated workflow remains close to its parent workflow. In contrast, representing workflows directly in Python often restricts updates to vague instructions like "modify no more than five lines," which can lead to unreliable or semantically meaningless changes, causing the new workflow to deviate significantly from its parent. We use the round index of optimal stopping points to demonstrate this.

| Table 3: Final selected workflow indices for each benchmark. Method GSM8K MATH HumanEval MBPP AFLOW 8 15 5 8 MermaidFlow 16 18 7 10   |
|---------------------------------------------------------------------------------------------------------------------------------------|

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 introduce a test node, while Workflow_5 contains a more diverse ensemble section with agents covering different reasoning aspects. MermaidFlow combines these strengths to synthesize a new and improved Workflow_8. This generation process occurs in the Mermaid Field, where all workflows are defined in a structured syntax that can be directly rendered as visual diagrams. Once a new Mermaid workflow is generated, we use gpt-4o-mini to translate the Mermaid code into executable Python code. Due to Mermaid's well-structured nature, this translation can be both straightforward and reliable. As demonstrated in Figure 4, the generated Python code perfectly resemble Mermaid Workflow_8, consisting of a diverse ensemble section and a test function. This case study not only demonstrates the efficiency of searching for new high-quality workflow populations in the Mermaid field but also provides a detailed illustration of MermaidFlow's stable and composable workflow lifecycle.

## 6 Conclusion

We propose MermaidFlow, a framework that transforms agentic workflow generation by encoding workflows as statically typed, semantically annotated, and compiler-verifiable graphs using the Mermaid language. Our proposed workflow formulation defines a well-structured, declaratively defined search space that supports safety-constrained rewrites and modular composition. Building on this space, we develop a safety-constrained evolutionary programming framework that enables efficient, verifiable, and high-quality workflow synthesis. MermaidFlow offers a principled step toward structurally safer and more interpretable agentic systems, introducing the first workflow optimization framework built atop a statically verifiable workflow representation. While MermaidFlow is evaluated in controlled agentic reasoning settings, its integration with real-world multi-agent systems and user-in-the-loop workflows introduces nuances that merit further exploration.

## References

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya G. Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent LLM systems fail? CoRR, abs/2503.13657, 2025. URL https://doi.org/10.48550/arXiv.2503.13657.

Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F Karlsson, Jie Fu, and Yemin Shi. Autoagents: A framework for automatic agent generation. In *Proceedings of the Thirty-Third* International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024, pp. 22–30, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In *Forty-first International* Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=zj7YuTE4t8.

Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning. In *The Eleventh International Conference on Learning Representations,* ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, and Tianyu Pang. Flowreasoner: Reinforcing query-level meta-agents. *arXiv preprint* arXiv:2504.15257, 2025.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024, pp. 8048–8057. ijcai.org, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta programming for A multiagent collaborative framework. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=VtmBAGCN7o.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. *CoRR*, abs/2408.08435, 2024. URL https://doi.org/10.48550/arXiv.2408.08435.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. Llm-blender: Ensembling large language models with pairwise comparison and generative fusion. In Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (ACL 2023), 2023.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html.

Xinzhe Li. A review of prominent paradigms for llm-based agents: Tool use, planning (including rag),
and feedback learning. In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pp. 9760–9779, 2025a.

Xinzhe Li. A review of prominent paradigms for llm-based agents: Tool use, planning (including rag),
and feedback learning. In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pp. 9760–9779, 2025b.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pp. 17889–17904. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024. emnlp-main.992.

Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. A dynamic llm-powered agent network for task-oriented agent collaboration. In *First Conference on Language Modeling*, 2024.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Scaling large-language-model-based multi-agent collaboration. CoRR, abs/2406.07155, 2024.

Jinwei Su, Yinghui Xia, Ronghua Shi, Jianhui Wang, Jianuo Huang, Yijin Wang, Tianyu Shi, JingSong Yang, and Lewei He. Debflow: Automating agent creation via agent debate. *CoRR*, abs/2503.23781, 2025. URL https://doi.org/10.48550/arXiv.2503.23781.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/ forum?id=1PL1NIMMrw.

Yinjie Wang, Ling Yang, Guohao Li, Mengdi Wang, and Bryon Aragam. Scoreflow: Mastering LLM
agent workflows via score-based preference optimization. *CoRR*, abs/2502.04306, 2025. URL https://doi.org/10.48550/arXiv.2502.04306.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao.

React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=WE_vluYUL-X.

Rui Ye, Shuo Tang, Rui Ge, Yaxin Du, Zhenfei Yin, Siheng Chen, and Jing Shao. MAS-GPT:
training llms to build llm-based multi-agent systems. *CoRR*, abs/2503.03686, 2025. URL https:
//doi.org/10.48550/arXiv.2503.03686.

Boyang Zhang, Yicong Tan, Yun Shen, Ahmed Salem, Michael Backes, Savvas Zannettou, and Yang Zhang. Breaking agents: Compromising autonomous LLM agents through malfunction amplification. *CoRR*, abs/2407.20859, 2024a. URL https://doi.org/10.48550/arXiv. 2407.20859.

Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, and Dawei Cheng. G-designer: Architecting multi-agent communication topologies via graph neural networks. *CoRR*, abs/2410.11782, 2024b. URL https://doi.org/10.48550/arXiv. 2410.11782.

Guibin Zhang, Kaijie Chen, Guancheng Wan, Heng Chang, Hong Cheng, Kun Wang, Shuyue Hu, and Lei Bai. Evoflow: Evolving diverse agentic workflows on the fly. *CoRR*, abs/2502.07373, 2025a. URL https://doi.org/10.48550/arXiv.2502.07373.

Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent architecture search via agentic supernet. *CoRR*, abs/2502.04180, 2025b. URL https://doi. org/10.48550/arXiv.2502.04180.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. Aflow: Automating agentic workflow generation. *CoRR*, abs/2410.10762, 2024c. doi: 10.

48550/ARXIV.2410.10762. URL https://doi.org/10.48550/arXiv.2410.10762.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, et al. Which agent causes task failures and when? on automated failure attribution of llm multi-agent systems. *arXiv preprint arXiv:2505.00212*, 2025c.

Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulic, Anna Korhonen, and Sercan Ö. Arik. Multi-agent design: Optimizing agents with better prompts and topologies. *CoRR*, abs/2502.02533, 2025. URL https://doi.org/10.48550/arXiv.2502.02533.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=uTC9AFXIhg.