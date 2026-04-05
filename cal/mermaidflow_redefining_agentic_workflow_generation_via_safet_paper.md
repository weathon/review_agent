# MERMAIDFLOW: REDEFINING AGENTIC WORKFLOW GENERATION VIA SAFETY-CONSTRAINED EVOLUTION- ARY PROGRAMMING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Despite the promise of autonomous agentic reasoning, existing workflow generation
methods frequently produce fragile, unexecutable plans due to unconstrained LLMdriven construction. We propose **MermaidFlow**, a framework that redefines
the agentic search space through safety-constrained graph evolution. At its core,
MermaidFlow represent workflows as a verifiable intermediate representation using
Mermaid, a structured and human-interpretable graph language. We formulate
domain-aware evolutionary operators, i.e., _crossover_, _mutation_, _insertion_, and
_deletion_, to preserve semantic correctness, enabling efficient exploration of a highquality, statically verifiable workflow space. Without modifying task settings or
evaluation protocols, MermaidFlow achieves consistent improvements in success
rates and faster convergence to executable plans on the agent reasoning benchmark.
The experimental results demonstrate that safety-constrained graph evolution offers
a scalable, modular foundation for robust and interpretable agentic reasoning
systems.


1 INTRODUCTION


Large language models (LLMs) are increasingly instantiated as modular agents that collaborate to
solve complex tasks through structured workflows (Guo et al., 2024; Li, 2025a;b). These agentic
workflows decompose problems into subtasks, assign them to specialized agents, and integrate
intermediate outputs toward a shared goal. Moving beyond single-agent prompting, this multi-agent
setting requires coherent planning and execution across agents with distinct roles and responsibilities.
Designing such workflows involves reasoning over compositional graph structures that represent
inter-agent dependencies, data flow, and semantic constraints, forming the foundation for scalable
and adaptive multi-agent systems (Zhou et al., 2025).


The lifecycle of agentic workflow is naturally structured into three layers: (1) _workflow planning_,
which defines the structure of subtasks, agent roles, and information flow; (2) _code_ _realization_,
where the plan is translated into executable programs; and (3) _runtime execution_, where agents are
instantiated and carry out their assigned behaviors. In many systems, these layers are collapsed
(e.g., Hu et al. (2024); Zhang et al. (2024c)): workflows are directly generated as _Python code_ or
serialized _JSON trees_, where planning decisions are entangled with implementation (i.e., through
_prompting_ -based generation of code or execution traces). As a result, workflows are often encoded in
_low-level_ formats where **structure** is implicit, **semantics** are entangled with imperative logic, and
**validity** can only be assessed at runtime. This implicit representation hinders verifiability, reuse, and
search, limiting the robustness and scalability of multi-agent systems.


Indeed, recent studies reveal that multi-agent LLM systems frequently fail due to brittle workflow
logic and coordination breakdowns (Cemri et al., 2025; Zhang et al., 2024a; 2025c). These failures
typically arise not from deficiencies in language models themselves but emerge from workflows that
cannot be reasoned about, verified, or adapted. Without a structured representation of agent roles, task
flow, and dependencies, systems struggle to detect errors before execution or to generalize behaviors
across tasks. This points to a core limitation: **existing workflows lack the abstraction needed for**
**reliable planning.**


1


To address the limitations of implicit, code-bound workflows, we introduce **MermaidFlow**, a
declarative representation for agentic planning inspired by the **Mermaid graph markup language** [1] .
MermaidFlow defines workflows as **declarative graphs**, where nodes represent prompting agents
and edges specify information flow (see Figure 1 for an illustrative example of the declarative
graph encoded in Mermaid language, which is declarative, structurally explicit, and highly humaninterpretable). This high-level representation enables structural and semantics properties with static
verification, e.g., _structure_ _feasibility_, and _type-safe_ _connections_, can be enforced at the graph
level, offering a clear plan that is both **human-readable** and **programmatically analyzable** . By
exposing explicit semantics and structure, MermaidFlow yields substantial downstream benefits
in both workflow generation and evaluation, when LLMs are extensively employed to discover
and evaluate workflows. These properties ultimately yield a more robust, and verifiable space for
agentic workflow planning. MermaidFlow enforces a clear separation between symbolic planning
and executable code, ensuring that workflow structures remain statically verifiable by design.


Building on this foundation, we further propose a novel **evolutionary programming** (EP) framework
tailored specifically to explore MermaidFlow’s structured graph space. Our EP approach employs
safety-constrained operations, including node replacement, subgraph rewiring, and role-consistent
insertions, to maintain workflow correctness throughout the search process. Furthermore, historical
workflows generated during search accumulate as structured experience, enabling efficient reuse
and adaptation across tasks. Together, MermaidFlow’s declarative representation and EP search
framework constitute a programmable and task-agnostic programming layer for agentic workflow
generation, enabling efficient search with improved correctness, generalization, and adaptability.
To our knowledge, this is the first agentic workflow framework to **guarantee** **static** **graph-level**
**correctness across the entire generation process.**


In summary, our contributions are threefold: (1) We introduce MermaidFlow, a declarative, verifiable
graph representation for agentic workflow planning that cleanly separates planning from execution; (2) We develop a novel EP-based search framework leveraging structured mutation operators
and workflow experience accumulation; and (3) We empirically demonstrate that MermaidFlow
significantly outperforms existing code-based methods on standard agentic reasoning benchmarks,
improving success rates, search efficiency, and interpretability.


2 RELATED WORKS


**Agentic Workflows with LLMs** Recent advances in multi-agent LLM systems have enabled structured collaboration among specialized agents to tackle complex, multi-step tasks. AFLOW (Zhang
et al., 2024c), MaAS (Zhang et al., 2025b), and MASS (Zhou et al., 2025) formalize agent
workflows using execution graphs and message-passing protocols to model multi-step reasoning.
MetaGPT (Hong et al., 2024) and MAS-GPT (Ye et al., 2025) implement role-based orchestration
by assigning domain-specific functions (e.g., product manager, engineer) and encoding Standard
Operating Procedures (SOPs) to reduce cascading errors. Debate-based frameworks such as MultiAgent Debate (Liang et al., 2024; Du et al., 2024) and DebFlow (Su et al., 2025) introduce structured
critique among agents to promote output reliability. While these systems improve modularity and


1https://mermaid.js.org


2


Figure 1: An illustration of the **work-**
**flow** **lifecycle** **in** **MermaidFlow** . The
workflow is modeled as a declarative
graph using Mermaid code, where nodes
_V_ [ _τ,α_ ] and edges _E_ [ _ρ_ ] are explicitly defined with annotated prompts and roles
(lines 3-8), styled and typed (lines
11–21), and connected via directed edges
(lines 24–30). This results in a statically
verifiable, semantically typed, and structurally interpretable representation that
serves as a unified interface for visualization, validation, and code generation.


scalability, their workflows are typically encoded in _imperative code_ or _loosely structured prompts_,
i.e., formats that lack semantic abstraction and resist verification. Recent studies (Cemri et al., 2025;
Zhang et al., 2024a; 2025c) identify fragile workflows, rather than model errors, as the primary
source of failure in multi-agent systems. Our proposed MermaidFlow addresses this bottleneck by
introducing a typed, declarative workflow space that supports safe construction, static validation, and
structured exploration, advancing agentic reasoning beyond brittle prompt chaining.


**Workflow Representation** The representation of agentic workflows governs not only how agents
are composed, but also whether they can be verified, reused, or optimized. Natural language-based
prompting methods, such as Chain-of-Thought (Kojima et al., 2022), ReAct (Yao et al., 2023), and
Self-Refine (Madaan et al., 2023), are expressive but underspecified, lacking formal structure for
validation. In contrast, code-centric approaches like AFLOW (Zhang et al., 2024c), ADAS (Hu
et al., 2024), and ScoreFlow (Wang et al., 2025) generate executable Python or JSON trees directly,
offering precision at the cost of brittleness and poor editability due to tightly entangled logic and
implementation. Recent efforts explore more structured workflow abstractions. GPTSwarm (Zhuge
et al., 2024) and FlowReasoner (Gao et al., 2025) organize workflows as agent interaction graphs, but
lack formal semantics, e.g., no type enforcement, role validation, or support for systematic search.
MetaGPT (Hong et al., 2024) and MAS-GPT (Ye et al., 2025) encode workflows through SOP-style
templates and DSLs, but rely on rigid decomposition patterns that restrict flexibility. MermaidFlow
departs from these by introducing a typed, declarative graph representation grounded in the Mermaid
markup language. It makes role semantics and data flow explicit, allowing crucial graph-level
structural constraints, such as role consistency, and type safety, to be enforced pre-execution, enabling
safe reuse, adaptation, and search.


**Workflow Search and Optimization** The structure of the search space fundamentally shapes how
workflows are generated and optimized. AFlow (Zhang et al., 2024c) applies Monte Carlo Tree
Search over executable graphs, while ADAS (Hu et al., 2024) explores code-level candidates via
heuristics-guided expansion. Though systematic, both approaches operate over brittle code-centric
representations, where small mutations often break correctness, necessitating expensive filtering.
ScoreFlow (Wang et al., 2025) and G-Designer (Zhang et al., 2024b) adopt learned or continuous
optimization strategies, adjusting prompt topologies or agent graphs via gradient-based tuning or
neural controllers. However, these methods require differentiable feedback or training signals
and offer limited support for enforcing structural validity. A complementary direction leverages
evolutionary and population-based search. DebFlow (Su et al., 2025) refines workflows through
iterative agent debates, while EvoFlow (Zhang et al., 2025a) evolves diverse workflows using task
complexity-conditioned genetic search. Yet both approaches operate in loosely defined or weakly
constrained spaces, where mutations often yield semantically invalid workflows. MermaidFlow closes
this gap by introducing a structured, verifiable graph space equipped with safety-aware mutation
operators. This design guarantees that every candidate is valid by construction, enabling scalable and
principled workflow optimization.


3 A NOVEL DECLARATIVE GRAPH REPRESENTATION FOR AGENTIC
WORKFLOWS


This section introduces a declarative graph representation for agentic workflows, built on **Mermaid**
that is a structured, human-readable language with built-in static verifiability and graph render to
help human directly observe the workflow. Departing from unstructured or token-level workflow
representations, our workflow formalism leverages Mermaid’s type-aware syntax to enable correctness
by construction, symbolic manipulation, and modular workflow composition.


3.1 DECLARATIVE WORKFLOW GRAPHS WITH MERMAID


We model each agentic workflow as a declarative computation graph with explicit typing, annotations,
and semantic structure. Formally, we define a workflow graph as:


_G_ ( _V_ [ _τ,α_ ] _,_ _E_ [ _ρ_ ]) _,_ (1)


where _V_ [ _τ,α_ ] is a set of typed and annotated nodes, _E_ [ _ρ_ ] is a set of directed, role-labeled edges.


3


Figure 2: **Overview of the MermaidFlow framework** . **Left** : Comparison between _imperative_
(Python-based) and _declarative_ (Mermaid-based) workflow representations. MermaidFlow models
workflows as statically typed, verifiable graphs, enabling interpretable planning and structure-aware
code generation. **Right** : Illustration of the safety-aware evolutionary programming process. Given
historical Mermaid workflows, the EP sampler selects parent candidates and applies EP operators to
generate new workflow candidates. An LLM-as-Judge then selects the final workflow for evaluation,
and the results are used to update the population.


This structure is instantiated using the **Mermaid** graph language, a lightweight, human-readable
syntax for specifying typed graphs with semantically annotated components. Furthermore, Mermaid
provides a declarative interface that supports symbolic manipulation and static validation. A real
example is illustrated in Figure 1. Each node defines a symbolic identifier and type signature, while
edges carry semantic labels (e.g., inputs) that describe data-flow.


Next, we define each component of the workflow in the Mermaid domain to illustrate how the
workflow aligns well with its Mermaid representation.


**Nodes.** Each node _v_ _∈V_ [ _τ,α_ ] is a tuple (id _,_ _τ_ ( _v_ ) _,_ _α_ ( _v_ )), where id is a symbolic identifier,
_τ_ ( _v_ ) = Tin _→_ Tout denotes the type of that node, and attribute _α_ ( _v_ ) would provide some necessary
information according to the type. As shown in lines 4–7 of fig. 1, each node element can be concisely
defined in a single line of Mermaid script, for example: (id: C1, type _τ_ : CustomOp, attribute _α_ :
role-simple_solver_1). Nodes represent typed declarative units that are interpretable and statically
verifiable, and it can be easily understood by human.


**Edges.** Each edge ( _u, v_ ) _∈E_ [ _ρ_ ] denotes a dependency annotated with a semantic label _ρ_ ( _u, v_ ),
indicating how information or control flows from _u_ to _v_ e.g., “input”, “problem”. Mermaid syntax
supports these semantics with labeled arrows (e.g., A -> |input| B).


**Types.** All graph nodes are explicitly typed and semantically annotated, with types governing
interface compatibility and ensuring valid workflow construction. By defining these types up front,
we guarantee a consistent translation from Mermaid diagrams to Python code. For each task domain,
we introduce dedicated node types for the operators and tools that have proven most effective. During
code generation, these attributes direct the translator to emit the correct Python calls, making sure
that all tool arguments are clearly specified. We provide a detailed type description in Appendix A.1.


Each node explicitly defines its symbolic identity, type signature, and related attribution, while edges
govern execution flow subject to type constraints. Unlike flat or post-validated node-link DAGs (e.g.,
JSON plans or token-generated programs), our declarative graph formalism introduces an abstraction
layer that bridges symbolic reasoning and execution-level safety. This structure supports (static)
correctness-preserving mutation and compositional reuse, which are the key properties we exploit
in the constrained optimization process detailed next. To the best of our knowledge, this is the
first agentic workflow representation that leverages a graph-oriented abstract coding language to
enable more natural graph definition and manipulation. In the next section, we will formalize graph

4


manipulation actions in Mermaid and present a workflow-optimization method, further illustrating
the advantages of using Mermaid for workflow representation.


3.2 AGENTIC WORKFLOW SEARCH SPACE


The declarative graph formalism introduced above induces a constrained search space over **agentic**
**workflows** . We define the workflow together with related LLM factors as follows:

_S_ =         - _G_ ( _V_ [ _τ,α_ ] _,_ _E_ [ _ρ_ ] _,_ _C_ ) _∈G_ Mermaid �� _G |_ = _C_ static� _,_ (2)

where _G_ Mermaid denotes the set of workflows expressible in the Mermaid graph language, and _C_ static
captures structural constraints, such as type compatibility, role-consistent edges, and connectivity,
automatically enforced by Mermaid’s parser and extended structural schema. This built-in verifiability
arises from Mermaid’s declarative syntax and ensures that all elements of _S_ are valid and executable
by construction.


To enable optimization, we parameterize each node _v_ _∈V_ [ _τ,α_ ] as a tuple _v_ = - _m,_ _p_ ( _τ, α_ ) _,_ _f_ ( _τ_ )� _,_
following conventions from multi-agent systems (MAS) definition, where _m_ _∈_ _M_ specifies the
LLM configuration (e.g., model_name, temperature), _p_ ( _τ, α_ ) _∈_ _P_ is the prompt template
determined by the node type _τ_ and its argument _α_, and _f_ ( _τ_ ) _∈_ _F_ denotes the input/output format
associated with type _τ_ .


_V_ [ _τ,α_ ] = _{_ ( _m, p_ ( _τ, α_ ) _, f_ ( _τ_ ) _| m ∈_ _M,_ _p ∈_ _P,_ _f_ _∈_ _F_ _} ._ (3)


The formula above demonstrates that, when interpreting a Mermaid workflow, each node can be
directly mapped to a standard LLM agent instance. By assigning a type _τ_ and parameter/attribute
_α_, one can associate the node with a specific prompt _p_ ( _τ, α_ ) and input/output format _f_ ( _τ_ ). **This**
**formulation** **emphasizes** **that** **every** **LLM** **agent** **can** **be** **consistently** **defined** **both** **within** **the**
**Mermaid representation and in the general context of LLM agent configuration.**


The space is also **inductively closed** : type-compatible subgraphs can be composed without revalidation, while disconnected or cyclic fragments are excluded by construction. This closure property
is non-trivial: prior workflow representations, especially those based on imperative or token-level
programs, **lack** **structural** **guarantees**, and mutations frequently yield invalid or unrecoverable
states. In contrast, our declarative design ensures that local edits remain within the valid region of _S_,
enabling reliable and efficient search. Though intentionally bounded, _S_ remains expressive enough
to capture planning motifs such as hierarchical refinement and dataflow composition. By unifying
generation, mutation, and verification within a single, compiler-verifiable substrate, it provides a
semantically grounded foundation for structure-aware and safe workflow optimization.


4 CONSTRAINT-AWARE EVOLUTIONARY WORKFLOW OPTIMIZATION


We introduce an evolutionary programming (EP) framework that operates directly over declarative Mermaid workflows. Leveraging Mermaid’s typed and verifiable graph structure, we define
correctness-preserving operators that enable safe, modular workflow evolution. Unlike prior approaches over unstructured or token-based spaces, all candidates in MermaidFlow are valid by
construction, ensuring safe, compiler-checkable optimization throughout the search process.


4.1 CONSTRAINT-PRESERVING EP OPERATORS FOR DECLARATIVE WORKFLOW GRAPHS


We define a set of atomic graph-level operators that drive workflow evolution within MermaidFlow.
Each operator acts over a candidate graph _G_ ( _V, E_ ) _∈G_ Mermaid _⊆S_, and is designed to be _locally_
_scoped_, _type-consistent_, and _statically verifiable_, enabling every candidate to be validated by the
Mermaid compiler at each step of the search. Below are the definitions of the operations, which will
be used to verify the correctness of the newly generated workflow.


**Node Substitution.** Changing the attributes of a specific agent _v_ ( _τ, α_ ) _∈V_ to _v_ ( _τ, α′_ ). Like changing
the corresponding role prompt or instruction.


**Node Addition.** Given an edge ( _va, vb_ ) _∈E_, connecting from node _va_ to node _vb_, insert a new node
_v′_ to form ( _va, v_ _[′]_ ) and ( _v_ _[′]_ _, vb_ ) and disconnect ( _va, vb_ ) if: Tout( _va_ ) = Tin( _v_ _[′]_ ) _,_ Tout( _v_ _[′]_ ) = Tin( _vb_ )
according to their node type _τ_ .


5


**Edge Rewiring.** Given nodes _{va, vb, vc} ⊆V_ and ( _va, vb_ ) _∈E_ in the original graph _G_, rewire to
( _va, vc_ ) or ( _vc, vb_ ) and disconnect ( _va, vb_ ) if: Tout( _va_ ) = Tin( _vc_ ) or Tout( _vc_ ) = Tin( _vb_ ).


**Node** **Deletion.** Given a linear path _va_ _→_ _vb_ _→_ _vc_, delete _vb_ and insert an edge ( _va, vc_ ) if
Tout( _va_ ) = Tin( _vc_ ).


**Subgraph** **Mutation.** Let _G_ 1 _∈G_ Mermaid be a subgraph of the graph _G_ _∈S_ . Denote the input
and output node set of _G_ 1 as _I_ 1 and _O_ 1, respectively. Let _G_ 2 _∈G_ Mermaid be a feasible graph with
input and output node set _I_ 2 and _O_ 2. Replace _G_ 1 in _G_ with _G_ 2 such that Tin( _I_ 1) = Tin( _I_ 2) and
Tout( _O_ 1) = Tout( _O_ 2).


**Crossover.** Given _{G_ 1 _, G_ 2 _} ⊆S_ share a common interface node _v_ (e.g., an ensemble node), swap
subgraphs rooted at _v_ to yield _{G_ _[′]_ 1 _[, G][′]_ 2 _[}]_ [ such that the type and interface constraints are preserved,]
i.e., _{G_ _[′]_ 1 _[, G][′]_ 2 _[}]_ _[⊆S]_ [.] [Each operator is applied at the level of Mermaid syntax,] [enabling compiler-]
level validation of every candidate graph. By constraining transformations to preserve type and
role integrity, MermaidFlow ensures that evolutionary search remains within the semantically valid
subspace of workflows. In the case study fig. 4, there is a concrete example illustrating the crossover
operator.


**Lemma 1** (MermaidFlow Transformation Invariance) **.** _Let S_ _denote the declarative workflow space_
_defined in Section 3.2._ _For any workflow graph G ∈S_ _and any atomic transformation operator O_
_defined above, the resulting graph G_ _[′]_ = _O_ ( _G_ ) _also belongs to S:_


_∀_ _G ∈S,_ _∀O_ _∈_ O _,_ _O_ ( _G_ ) _∈S_ (4)


_where_ O _is the set of constraint-preserving operators over MermaidFlow graphs._ _That is, S_ _is closed_
_under all valid EP operations._


**Definition** **1.** _Let_ _G_ _denote_ _the_ _space_ _of_ _all_ _candidate_ _workflows,_ _each_ _G_ _∈G_ _represented_ _as_ _a_
_directed_ _graph_ ( _V, E_ ) _._ _We_ _define_ _a_ _static_ _validator_ _function_ _Q_ : _G_ _→{_ 0 _,_ 1 _},_ _implemented_ _by_ _a_
_Mermaid parser/compiler, such that:_


�1 _if G ∈S_
_Q_ ( _G_ ) = (5)
0 _otherwise_


_Here, S_ _⊂G_ _is the subset of workflows satisfying verifiability constraints such as workflow structure,_
_well-typed I/O, role validity, and full connectivity._


By using EP operators above, from Lemma 1, given a _Gt_ _∈S_, each change _O_ ( _Gt_ ) at step _t_
leads to a graph _Gt_ +1 = _O_ ( _Gt_ ) _∈S_ . Given an initial graph _G_ 0 _∈S_, by induction, we know
_∀t_ _∈_ N+ _, Gt_ +1 = _Ot ◦Ot−_ 1 _· · · ◦O_ 0( _G_ 0) _∈S_ . Thus, the evolution in the static Mermaid graph
space remains the safe subspace.


In MermaidFlow, when using an LLM to generate a new Mermaid graph, the resulting Mermaid code
may sometimes violate predefined safety constraints. To address this, we implement a checker to
verify whether the newly generated candidates conform to the defined workflow and operation rules.
If any violations are detected, new workflows are regenerated. Thanks to Mermaid’s simple and clear
syntax, the code can be treated as structured text. This allows us to easily build a text-based analysis
tool and incorporate custom rules into the checker. More implementation details can be found in
Appendix A.2.


4.2 EVALUATION AND SELECTION IN WORKFLOW POPULATIONS


We frame each declarative workflow graph as an _experience_ and maintain a population of scored
experiences over time. At each optimization step _t_, the system tracks a history buffer: _W_ history _,t_ =
_{_ ( _si,_ score _i_ ) _}_ _[t]_ _i_ =1 [, where] _[ s][i]_ _[∈G]_ [Mermaid][ denotes a structurally valid workflow, and][ score] _[i]_ [ reflects its]
estimated performance.


At each optimization cycle, two parent workflows _sa, sb_ are sampled from _W_ history _,t_, typically via
temperature-scaled softmax sampling according to following distribution: _P_ mixed( _i_ ) = _λ·_ [1] _t_ [+(1] _[−][λ]_ [)] _[·]_

exp( _α·scorei_ )

- _nj_ =1 [exp(] _[α][·][score][j]_ [)] _[,]_ [ where] _[ t]_ [ is the number of workflows in the history buffer,] _[ score][i]_ [is the validation]
score of the _i_ -th workflow, and the parameters _α_ and _λ_ control the influence of the scores, and


6


balances exploration-exploitation, respectively. After sample two different parent workflows _sa, sb_
where _sa_ = _sb_ . These are used to generate a candidate set through the evolutionary process:

_S_ candidates = _{si_ _| si_ = _O_ ( _sa, sb_ ) _,_ _O_ _∈_ O _}_ _[N]_ _i_ =1 _[,]_ (6)

where O denotes the set of correctness-preserving operators (Section 4.1), for some operator only _sa_
involved and _N_ is the candidate pool size.


To avoid expensive rollout-based evaluation over the full population, we adopt an _LLM-as-judge_
model that scores each candidate _s ∈_ _S_ candidates based on semantic fit, structure, and task relevance.
Since all candidates in _S_ candidates are statically verified by the Mermaid compiler, they are guaranteed
to be syntactically valid, type-safe, and structurally executable, dramatically reducing failure cases
and increasing effective sample quality.


We then select the highest-scoring candidate and update the history buffer:

_W_ history _,t_ +1 _←_ _W_ history _,t ∪{_ ( _s_ _[∗]_ child _[,]_ [Validate][(] _[s][∗]_ child [))] _[}][,]_ [where] _[ s][∗]_ child [=] [arg max] LLM_as_Judge( _s_ ) _._
_s∈S_ candidates


This experience-centric design, enabled by the declarative and verifiable structure of MermaidFlow,
supports efficient, low-cost population evolution without compromising safety, correctness, or search
quality. See Appendix A.3 for algorithmic details.


5 EXPERIMENTS


5.1 EXPERIMENT SETUP


**Baseline.** We choose threefold of agentic baselines: (1) **Non-agentic reasoning methods**, including
CoT (Kojima et al., 2022), ComplexCoT (Fu et al., 2023), and Self-Consistency (Wang et al., 2023).
(2) **Hand-crafted multi-agent systems**, such as LLM-Debate (Du et al., 2024), LLM-Blender (Jiang
et al., 2023), DyLAN (Liu et al., 2024), and MAcNet (Qian et al., 2024). (3) **Autonomous multi-agent**
**systems**, including GPTSwarm (Zhuge et al., 2024), MaAS (Zhang et al., 2025b), AutoAgents (Chen
et al., 2024), ADAS (Hu et al., 2024), and AFlow (Zhang et al., 2024c). Among them, GPTSwarm and
MaAS incorporate trainable modules for assigning workflow structures, while AutoAgents, ADAS,
and AFlow rely on an LLM to design the structure, consistent with our setting. More details on
baseline setups are provided in Appendix A.4.


**Task** **and** **Benchmarks.** We evaluate MermaidFlow on four public benchmarks covering two
domains: **(1) math reasoning**, GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021); **(2)**
**code** **generation**, HumanEval (Chen et al., 2021), and MBPP (Austin et al., 2021). For MATH
benchmark, we follow AFlow (Zhang et al., 2024c) and MaAS (Zhang et al., 2025b) in using the
same selected problems from four typical problem types in level 5. The dataset statistics are provided
in Appendix A.5.


**Implementation details.** We use a closed-source LLM (gpt-4o-mini-0718) as both the Optimization
and Execution LLM, consistent with the setup in MaAS (Zhang et al., 2025b). The Optimization
LLM is responsible for tasks such as generating promising workflows in Mermaid code, selecting
from sampled workflows, evolving to new workflows, and translating Mermaid code into Python
code. All models are accessed via API with the temperature set to 0. In each round, we generate four
different _s_ child candidates. To ensure experimental stability, complex operations such as crossover are
applied with only a 10% probability. We set the number of iteration rounds to 20 for both Mermaid
and AFlow, and to 30 for ADAS. The evaluation metrics are kept consistent with those used in AFlow
and MaAS: for GSM8K and MATH, we report the Solve Rate (%) as the primary metric, while for
HumanEval and MBPP, we report the pass@1 score.


5.2 EXPERIMENTAL RESULTS


We compare MermaidFlow against 13 baselines on the GSM8K, MATH, HumanEval, and MBPP
benchmarks, as shown in Table 1. The results demonstrate that MermaidFlow consistently achieves
the best performance across all tasks. Compared to methods that search for the next workflow in
the Python field, such as ADAS and AFlow, our approach outperforms them by an average margin
of 2.08% to 5.54%. On the MATH benchmark specifically, MermaidFlow exceeds the secondbest method AFLOW by 2.61%. For certain benchmarks, performance is primarily limited by the


7


Table 1: Performance comparison among Non-agentic reasoning methods, hand-crafted multi-agent
systems, and automated agentic workflows. All methods use gpt-4o-mini as the base LLM and
are evaluated on the test split, with results averaged over three runs. **Bold** indicates the best result;
underline denotes the runner-up. MermaidFlow shows consistent improvements across all datasets.
*: Result reported in the MaAS paper, as the corresponding implementation for this dataset is not
available in their code.

**Method** **GSM8K** **MATH** **HumanEval** **MBPP** **Avg.**


Vanilla 87.57 46.29 87.49 70.29 72.91
CoT (Kojima et al., 2022) 87.45 46.40 88.13 71.83 73.45
ComplexCoT (Fu et al., 2023) 86.89 46.40 87.49 72.36 73.29
SC (CoT _×_ 5) (Wang et al., 2023) 87.57 47.91 88.60 73.60 74.42


LLM-Debate (Du et al., 2024) 89.47 48.63 88.80 70.29 74.30
LLM-Blender (Jiang et al., 2023) 88.35 46.92 88.68 77.05 75.25
DyLAN (Liu et al., 2024) 89.98 48.54 90.42 77.30 76.56
MacNet (Qian et al., 2024) 87.95 45.18 84.57 65.28 70.75


GPTSwarm (Zhuge et al., 2024) 89.14 47.88 89.32 77.43 75.94
MaAS (Zhang et al., 2025b) 91.47 52.19 91.57 82.17* 79.35


AutoAgents (Chen et al., 2024) 87.69 45.32 87.64 71.95 73.15
ADAS (Hu et al., 2024) 88.35 43.18 84.19 77.05 73.69
AFlow (Zhang et al., 2024c) 90.11 52.81 90.08 81.67 78.67
**MermaidFlow (Ours)** **92.39** **55.42** **92.87** **82.31** **80.75**


capabilities of the Execution LLM. For example, in HumanEval and GSM8K, the baseline (Vanilla)
performance is already high, so improvements from architectural optimization are less significant. In
contrast, for benchmarks where the baseline performance is relatively low, such as MATH and MBPP,
our method demonstrates a more substantial impact. Overall, MermaidFlow’s average score across
these tasks is 80.75%, which is 1.40% higher than the highest average among all baselines (79.35% by
MaAS), fully demonstrating the robustness and superiority of our approach across different problems.


5.3 ABLATION STUDY


**Evolution** **Efficiency** To evaluate the effectiveness of
our approach, we compare the learning curves of MermaidFlow and AFlow on the MATH dataset, as shown in
Figure 5.3. MermaidFlow demonstrates a more consistent improvement in workflow quality during training and
better generalization to the test set.


The core difference between MermaidFlow and AFlow
lies in the search space. AFlow operates directly on
Python code, applying textual edits with prompts constraints. This approach often leads to invalid and non- Figure 3: An illustrative figure comfunctional programs, with only a 50% success rate in paring the highest solve rates on the
generating executable code. In contrast, MermaidFlow MATH dataset between MermaidFlow
evolves workflows at the graph level using Mermaid, a and AFlow on the training set (119 probdomain-specific language that enables structured manip- lems) and test set (486 problems) across
ulation (e.g., adding, deleting, or mutating nodes). This optimization iterations.
representation is better suited for LLM-based optimization (e.g., gpt-4o-mini), **consistently yield-**
**ing >90% success rate in producing valid Python code** . This reliability enables more effective
exploration and optimization of workflow space. Thanks to Mermaid’s reliable generation rate and
lightweight representation, it achieves better token efficiency. When AFlow and MermaidFlow
both surpass 52% on the MATH dataset, they consume 6 _._ 9 _e_ 4 and 2 _._ 7 _e_ 4 tokens respectively, with
MermaidFlow requiring only about half the cost of AFlow.


**Impact** **of** **Optimization** **LLM** **Scale** We investigate how the choice of Optimization LLM influences the quality of workflows in MermaidFlow by evaluating more capable models on the
HumanEval and GSM8K benchmark. Specifically, we compare the effectiveness of larger models,


8


|rison of differe|ent optimization LLMs on HumanEval and|
|---|---|
|**Dataset**|**Claude 3.5**<br>**GPT-4o**<br>**GPT-4o-mini**|
|HumanEval<br>GSM8K|93.13<br>94.66<br>92.87<br>93.83<br>93.94<br>92.39|


such as Claude 3.5 and GPT-4o, in generating new workflows, while keeping GPT-4o-mini fixed as
the Execution LLM. Results are summarized in Table 2. As a result, higher-capacity optimization
LLMs consistently yield better performance across both HumanEval and GSM8K. This consistent
trend underscores a core strength of MermaidFlow: its well-structured, statically verifiable search
space enables even modest improvements in optimization quality to translate directly into more
functional, high-reward workflows.


**Optimal Stopping Point Analysis** We investigate the advantages of using Mermaid as the workflow
representation in workflow update control. A stable and reliable search process requires controllable
and well-defined update steps. With Mermaid, updates can be expressed through precise graph-based
operations such as adding nodes, deleting nodes, or modifying edges. These structured operations
help ensure that the newly generated workflow remains close to its parent workflow. In contrast,
representing workflows directly in Python often restricts updates to vague instructions like “modify
no more than five lines,” which can lead to unreliable or semantically meaningless changes, causing
the new workflow to deviate significantly from its parent.


We use the round index of optimal stopping points to demonstrate this.


9


introduce a test node, while Workflow_5 contains a more diverse ensemble section with agents
covering different reasoning aspects. MermaidFlow combines these strengths to synthesize a new
and improved Workflow_8.


This generation process occurs in the Mermaid Field, where all workflows are defined in a structured
syntax that can be directly rendered as visual diagrams. Once a new Mermaid workflow is generated,
we use gpt-4o-mini to translate the Mermaid code into executable Python code. Due to Mermaid’s
well-structured nature, this translation can be both straightforward and reliable. As demonstrated
in Figure 4, the generated Python code perfectly resemble Mermaid Workflow_8, consisting of a
diverse ensemble section and a test function.


This case study not only demonstrates the efficiency of searching for new high-quality workflow
populations in the Mermaid field but also provides a detailed illustration of MermaidFlow’s stable
and composable workflow lifecycle.


6 CONCLUSION


We propose MermaidFlow, a framework that transforms agentic workflow generation by encoding
workflows as statically typed, semantically annotated, and compiler-verifiable graphs using the
Mermaid language. Our proposed workflow formulation defines a well-structured, declaratively
defined search space that supports safety-constrained rewrites and modular composition. Building
on this space, we develop a safety-constrained evolutionary programming framework that enables
efficient, verifiable, and high-quality workflow synthesis. MermaidFlow offers a principled step
toward structurally safer and more interpretable agentic systems, introducing the first workflow optimization framework built atop a statically verifiable workflow representation. While MermaidFlow is
evaluated in controlled agentic reasoning settings, its integration with real-world multi-agent systems
and user-in-the-loop workflows introduces nuances that merit further exploration.


REFERENCES


Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan,
Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language
models. _arXiv preprint arXiv:2108.07732_, 2021.


Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt
Keutzer, Aditya G. Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E.
Gonzalez, and Ion Stoica. Why do multi-agent LLM systems fail? _CoRR_, abs/2503.13657, 2025.
[URL https://doi.org/10.48550/arXiv.2503.13657.](https://doi.org/10.48550/arXiv.2503.13657)


Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F Karlsson, Jie Fu, and Yemin
Shi. Autoagents: A framework for automatic agent generation. In _Proceedings of the Thirty-Third_
_International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August_
_3-9, 2024_, pp. 22–30, 2024.


Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared
Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large
language models trained on code. _arXiv preprint arXiv:2107.03374_, 2021.


Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve
math word problems. _arXiv preprint arXiv:2110.14168_, 2021.


Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving
factuality and reasoning in language models through multiagent debate. In _Forty-first International_
_Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_ . OpenReview.net,
2024. [URL https://openreview.net/forum?id=zj7YuTE4t8.](https://openreview.net/forum?id=zj7YuTE4t8)


Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting
for multi-step reasoning. In _The Eleventh International Conference on Learning Representations,_
_ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023.


10


Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min
Lin, and Tianyu Pang. Flowreasoner: Reinforcing query-level meta-agents. _arXiv_ _preprint_
_arXiv:2504.15257_, 2025.


Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf
Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress
and challenges. In _Proceedings of the Thirty-Third International Joint Conference on Artificial_
_Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024_, pp. 8048–8057. ijcai.org, 2024.


Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv_
_preprint arXiv:2103.03874_, 2021.


Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao
Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng
Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta programming for A multiagent collaborative framework. In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_ _Rep-_
_resentations,_ _ICLR_ _2024,_ _Vienna,_ _Austria,_ _May_ _7-11,_ _2024_ . OpenReview.net, 2024. URL
[https://openreview.net/forum?id=VtmBAGCN7o.](https://openreview.net/forum?id=VtmBAGCN7o)


Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. _CoRR_, abs/2408.08435,
2024. [URL https://doi.org/10.48550/arXiv.2408.08435.](https://doi.org/10.48550/arXiv.2408.08435)


Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. Llm-blender: Ensembling large language models
with pairwise comparison and generative fusion. In _Proceedings of the 61th Annual Meeting of the_
_Association for Computational Linguistics (ACL 2023)_, 2023.


Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In _Advances_ _in_ _Neural_ _In-_
_formation_ _Processing_ _Systems_ _35:_ _Annual_ _Conference_ _on_ _Neural_ _Information_ _Process-_
_ing_ _Systems_ _2022,_ _NeurIPS_ _2022,_ _New_ _Orleans,_ _LA,_ _USA,_ _November_ _28_ _-_ _December_ _9,_
_2022_, 2022. URL [http://papers.nips.cc/paper_files/paper/2022/hash/](http://papers.nips.cc/paper_files/paper/2022/hash/8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html)
[8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html.](http://papers.nips.cc/paper_files/paper/2022/hash/8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html)


Xinzhe Li. A review of prominent paradigms for llm-based agents: Tool use, planning (including rag),
and feedback learning. In _Proceedings of the 31st International Conference on Computational_
_Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025_, pp. 9760–9779, 2025a.


Xinzhe Li. A review of prominent paradigms for llm-based agents: Tool use, planning (including rag),
and feedback learning. In _Proceedings of the 31st International Conference on Computational_
_Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025_, pp. 9760–9779, 2025b.


Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi,
and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent
debate. In _Proceedings_ _of_ _the_ _2024_ _Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_
_Processing,_ _EMNLP_ _2024,_ _Miami,_ _FL,_ _USA,_ _November_ _12-16,_ _2024_, pp. 17889–17904. Association for Computational Linguistics, 2024. [URL https://aclanthology.org/2024.](https://aclanthology.org/2024.emnlp-main.992)
[emnlp-main.992.](https://aclanthology.org/2024.emnlp-main.992)


Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. A dynamic llm-powered agent network
for task-oriented agent collaboration. In _First Conference on Language Modeling_, 2024.


Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and
Peter Clark. Self-refine: Iterative refinement with self-feedback. In _Advances_ _in_
_Neural_ _Information_ _Processing_ _Systems_ _36:_ _Annual_ _Conference_ _on_ _Neural_ _Information_
_Processing_ _Systems_ _2023,_ _NeurIPS_ _2023,_ _New_ _Orleans,_ _LA,_ _USA,_ _December_ _10_ _-_ _16,_
_2023_, 2023. URL [http://papers.nips.cc/paper_files/paper/2023/hash/](http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)
[91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html.](http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)


11


Chen Qian, Zihao Xie, Yifei Wang, Wei Liu, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang,
Zhiyuan Liu, and Maosong Sun. Scaling large-language-model-based multi-agent collaboration.
_CoRR_, abs/2406.07155, 2024.


Jinwei Su, Yinghui Xia, Ronghua Shi, Jianhui Wang, Jianuo Huang, Yijin Wang, Tianyu Shi,
JingSong Yang, and Lewei He. Debflow: Automating agent creation via agent debate. _CoRR_,
abs/2503.23781, 2025. [URL https://doi.org/10.48550/arXiv.2503.23781.](https://doi.org/10.48550/arXiv.2503.23781)


Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language
models. In _The_ _Eleventh_ _International_ _Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2023,_
_Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. [URL https://openreview.net/](https://openreview.net/forum?id=1PL1NIMMrw)
[forum?id=1PL1NIMMrw.](https://openreview.net/forum?id=1PL1NIMMrw)


Yinjie Wang, Ling Yang, Guohao Li, Mengdi Wang, and Bryon Aragam. Scoreflow: Mastering LLM
agent workflows via score-based preference optimization. _CoRR_, abs/2502.04306, 2025. URL
[https://doi.org/10.48550/arXiv.2502.04306.](https://doi.org/10.48550/arXiv.2502.04306)


Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models. In _The Eleventh International Confer-_
_ence on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net,
2023. [URL https://openreview.net/forum?id=WE_vluYUL-X.](https://openreview.net/forum?id=WE_vluYUL-X)


Rui Ye, Shuo Tang, Rui Ge, Yaxin Du, Zhenfei Yin, Siheng Chen, and Jing Shao. MAS-GPT:
training llms to build llm-based multi-agent systems. _CoRR_, abs/2503.03686, 2025. [URL https:](https://doi.org/10.48550/arXiv.2503.03686)
[//doi.org/10.48550/arXiv.2503.03686.](https://doi.org/10.48550/arXiv.2503.03686)


Boyang Zhang, Yicong Tan, Yun Shen, Ahmed Salem, Michael Backes, Savvas Zannettou, and
Yang Zhang. Breaking agents: Compromising autonomous LLM agents through malfunction
amplification. _CoRR_, abs/2407.20859, 2024a. [URL https://doi.org/10.48550/arXiv.](https://doi.org/10.48550/arXiv.2407.20859)
[2407.20859.](https://doi.org/10.48550/arXiv.2407.20859)


Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, and
Dawei Cheng. G-designer: Architecting multi-agent communication topologies via graph neural
networks. _CoRR_, abs/2410.11782, 2024b. URL [https://doi.org/10.48550/arXiv.](https://doi.org/10.48550/arXiv.2410.11782)
[2410.11782.](https://doi.org/10.48550/arXiv.2410.11782)


Guibin Zhang, Kaijie Chen, Guancheng Wan, Heng Chang, Hong Cheng, Kun Wang, Shuyue Hu,
and Lei Bai. Evoflow: Evolving diverse agentic workflows on the fly. _CoRR_, abs/2502.07373,
2025a. [URL https://doi.org/10.48550/arXiv.2502.07373.](https://doi.org/10.48550/arXiv.2502.07373)


Guibin Zhang, Luyang Niu, Junfeng Fang, Kun Wang, Lei Bai, and Xiang Wang. Multi-agent
architecture search via agentic supernet. _CoRR_, abs/2502.04180, 2025b. [URL https://doi.](https://doi.org/10.48550/arXiv.2502.04180)
[org/10.48550/arXiv.2502.04180.](https://doi.org/10.48550/arXiv.2502.04180)


Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen
Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin
Wu. Aflow: Automating agentic workflow generation. _CoRR_, abs/2410.10762, 2024c. doi: 10.
48550/ARXIV.2410.10762. [URL https://doi.org/10.48550/arXiv.2410.10762.](https://doi.org/10.48550/arXiv.2410.10762)


Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li,
Chi Wang, Huazheng Wang, Yiran Chen, et al. Which agent causes task failures and when? on
automated failure attribution of llm multi-agent systems. _arXiv preprint arXiv:2505.00212_, 2025c.


Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulic, Anna Korhonen, and
Sercan Ö. Arik. Multi-agent design: Optimizing agents with better prompts and topologies. _CoRR_,
abs/2502.02533, 2025. [URL https://doi.org/10.48550/arXiv.2502.02533.](https://doi.org/10.48550/arXiv.2502.02533)


Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen
Schmidhuber. Gptswarm: Language agents as optimizable graphs. In _Forty-first International_
_Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_ . OpenReview.net,
2024. [URL https://openreview.net/forum?id=uTC9AFXIhg.](https://openreview.net/forum?id=uTC9AFXIhg)


12


A IMPLEMENTATION DETAILS


A.1 DETAILS OF MERMAID


Mermaid is a text-based language for creating diagrams and flowcharts, which we use to represent
workflows. In our system, we extend Mermaid with custom node types:


1. **CustomOp**    - Used for specialized reasoning strategies, executing specific tasks through
defined roles. Example: K["Custom<br/>(role: validate_1)"], styled with
blue fill.

2. **Interface**    - Entry and exit nodes for workflows, including problem input points and return
output points. Example: PROBLEM([Problem]), styled with light purple fill.

3. **ProgrammerOp**    - Nodes that execute code generation and computation, particularly
suitable for mathematical problems. Example: P["Programmer<br/>(analysis:
’Calculate step by step’)"], styled with red fill. Only math problems use this
node type.

4. **ScEnsembleOp**    - Nodes that combine multiple solutions into one result, requiring multiple
inputs to function properly. Example: ENSEMBLE["ScEnsemble<br/>"], styled with
yellow fill.

5. **TestOp**    - Nodes for validating solutions, used to verify whether code passes predefined test
cases. Example: T["Test<br/>"], styled with green fill. Only code problems use this
node type.

6. **CustomCodeGenerateOp**    - Nodes specifically designed to
generate code based on problem descriptions. Example:
CODE_GEN["CustomCodeGenerate<br/>(instruction: xxx)"], styled
with red fill. Only code problems use this node type.


Each node type has specific connection rules and usage patterns to ensure data flows correctly
through the workflow. Connections between nodes are typically represented by arrows, for example:
PROBLEM -> |problem| P, indicating that the problem input is passed to the programmer
node.


We provide detailed descriptions of each node type in our prompts to guide the LLM in generating
structurally valid Mermaid workflows. Additionally, we implement a Mermaid checker to verify the
correctness of the generated Mermaid code.


For different types of problems (math or code), we use different prompt templates to guide the LLM
in generating appropriate Mermaid code, as each problem type involves different node types and
workflow patterns.


For math problems, our templates emphasize clarity in computational steps and rigor in mathematical reasoning. For example, we encourage using Programmer nodes for precise calculations and
ScEnsemble nodes to combine multiple solutions for improved accuracy. The prompt templates
include guidelines for LaTeX mathematical expression formatting to ensure outputs conform to
mathematical standards.


For code problems, our templates focus on code generation, testing, and validation. We particularly
emphasize using Test nodes to validate generated code solutions and CustomCodeGenerate
nodes to produce code based on problem descriptions. The prompt templates include guidelines for
code formatting and testing requirements to ensure the generated code executes correctly and passes
test cases.


We maintain a dedicated prompt.py module to store any node prompts that are too verbose for the
Mermaid diagram, keeping the workflow clean and readable.


A.2 MERMAID CHECKER IMPLEMENTATION


We implemented a comprehensive Mermaid validation system that performs both soft and hard checks
on workflow diagrams. The soft check analyzes the structure using regex pattern matching to extract
different components of the Mermaid code:


13


- **Node initialization statements**     - Extracts node definitions and their types


    - **Class definition statements**     - Identifies styling classes for nodes


    - **Node class assignments**     - Maps nodes to their respective classes


    - **Node connection statements**     - Extracts source, target, and label information from connections


Our validation system performs several critical checks:


1. **W1**    - Verifies the presence of required PROBLEM and RETURN interface nodes


2. **W2**    - Ensures each node is properly connected in the workflow with paths from PROBLEM
and to RETURN


3. **W3**    - Validates that PROBLEM and RETURN nodes are correctly classified as Interface
types


4. **W4**    - Checks that all nodes have valid types according to configuration


5. **W5**    - Ensures ScEnsembleOp nodes have at least two incoming connections


We also implement rigorous hard checks by compiling the Mermaid code directly through the
Mermaid CLI [2], which thoroughly verifies syntactic correctness and catches any compilation errors.
This comprehensive two-level validation approach ensures both structural integrity through soft
checks and strict syntactic validity through hard checks, guaranteeing that all workflow diagrams are
executable and error-free.


A.3 ALGORITHMS


In this section, we define the key algorithms underlying the MermaidFlow system. These algorithms
form the core components of our workflow optimization and execution process, including the end-toend MermaidFlow pipeline and the dedicated Mermaid workflow optimization routines that operate
over Mermaid graphs. The following algorithms demonstrate how we generate, validate, and optimize
workflow, as well as how we transform the Mermaid code into executable Python code. [3]


**Algorithm 1** MermaidFlow **End-to-End Pipeline**

1: **procedure** MERMAIDFLOWPIPELINE
2: **Input:** val_dataset, dataset_type, mmd_checker, max_rounds, num_tries, candi_num
3: initial_workflow _←_ **LoadInitialWorkflow** (dataset_type)
4: mmd_history _←_ [initial_workflow]
5: **for** round = 1 to max_rounds **do**
6: parent_info _←_ **SampleParent** (mmd_history)
7: mmd_graph, related_prompt _←_ **OptimizeMermaidWorkflow** (parent_info, num_tries,
candi_num, mmd_checker)
8: python_code _←_ **GeneratePythonCode** (parent_info, mmd_graph, related_prompt)
9: val_score _←_ **EvaluateWorkflow** (python_code, validation_dataset)
10: mmd_history _←_ **Update** (mmd_history, mmd_graph, python_code, val_score)
11: **end for**
12: **return** mmd_history
13: **end procedure**


2https://github.com/mermaid-js/mermaid-cli
3We use “mmd” as the abbreviation for Mermaid.


14


**Algorithm 2 OptimizeMermaidWorkfow** Function

**Input:** parent_info, num_tries, candi_num, mermaid_checker
1: last_mmd, last_errors _←_ null
2: prev_attempt _←_ {last_mmd, last_errors}
3: **for** _i_ = 1 to num_tries **do**
4: mmd_graph, related_prompt _←_ **GetNewMermaidGraph** (parent_info, prev_attempt,
candi_num)
5: **if** mermaid_checker exists **then**
6: mmd_path _←_ mermaid_checker. **TransferMmdCodeToFile** (mmd_graph)
7: (hard_pass, hard_info) _←_ mermaid_checker. **HardCheck** (mmd_path)
8: (soft_pass, soft_info) _←_ mermaid_checker. **SoftCheck** (mmd_path)
9: last_mmd _←_ mmd_graph
10: last_errors _←_ hard_info + soft_info
11: prev_attempt _←_ {last_mmd, last_errors}
12: **if** hard_pass AND soft_pass **then**
13: **break** _▷_ The new Mermaid Code pass soft and hard check
14: **end if**
15: **else**
16: **break** _▷_ Accept first response if no checker
17: **end if**
18: **end for**
19: **return** mmd_graph, related_prompt


**Algorithm 3 GetNewMermaidGraph** Function

1: **procedure** GETNEWMERMAIDGRAPH(parent_info, prev_attempt, candi_num)
2: mmd_candidates _←_ **GetMermaidCandidates** (parent_info, prev_attempt, candi_num)
3: selected_mmd _←_ **LLMAsJudge** (mmd_candidates)
4: new_graph _←_ selected_mmd. _graph_
5: new_prompt _←_ selected_mmd. _prompt_
6: **return** new_graph, new_prompt
7: **end procedure**


A.3.1 PROMPT TEMPLATES


We design structured prompt templates to guide the LLM in generating, evaluating, and transforming
workflows throughout the optimization process. Each template encodes precise instructions, including
output format, semantic constraints, and evaluation objectives, ensuring that the model adheres to the
structural and functional requirements of MermaidFlow.


Below, we summarize the purpose of each template and provide the full prompt text for reference.


    - **UPDATE_MERMAID_WORKFLOW**     - Used to guide the LLM in optimizing mermaid
code based on parent nodes information. This prompt includes experience data, score
information, the current graph in Mermaid format, the corresponding role prompt, operator
descriptions, and log data from previous runs. It instructs the LLM to make specific optimizations by adding, modifying, or deleting nodes while clearly describing each modification.
The prompt also emphasizes the importance of proper formatting, critical thinking methods,
and effective use of specialized operators like Program for solving math problems.


    - **MERMAID_GUIDANCE**     - Used to guide the generation of correct Mermaid code according to our node type definitions. This prompt provides specific guidance for generating
workflow in Mermaid code following the specification in A.1.


    - **GENERATE_PYTHON_CODE**     - Used to generate python code from given mermaid code
and agents prompt.


    - **LLM_AS_JUDGE**     - Used to select the most promising graph from multiple Mermaid
candidates. It evaluates each candidate graph based on criteria including workflow coherence,


15


innovation, complexity balance, prompt quality, and modification rationale, providing
detailed scoring justifications. The judging process references the performance of historical
graphs and avoids selecting graphs with structural defects (such as incorrect connections or
improper use of ensemble nodes).


**UPDATE_MERMAID_WORKFLOW** : Depending on whether a single or two parent workflows
are used, we adopt different prompt templates to generate the new Mermaid code.


**Single Parent**


You can add, modify, or delete nodes, parameters, or connections in the workflow. For
_�→_ each change, clearly describe your single modification within XML tags using the
_�→_ format: <modification>your detailed modification description</modification>.


Your optimizations should be one of (this limitation is strict! you can only pick one
_�→_ action!):
1. Expanding the graph by adding new single (operators). Note: adding nodes may require
_�→_ modifying prompts for related nodes, you should also update the corresponding
_�→_ prompts if needed.
2. Deleting unnecessary nodes from the graph.
3. Modifying existing nodes or their connections or their prompt.


Prompt is very important for performance here are some exmaples you can learn from it:
{prompt_few_shot}


Ensure all necessary prompts are properly prepared and defined. The generated custom

_�→_ node's role is defined by their prompt. Use custom methods with proper formatting
_�→_ to ensure your output matches the expected structure. The system will extract
_�→_ answers based on specific rules for scoring, so maintaining the correct output
_�→_ format is critical.
The output format is critical for proper evaluation. Analyze the log data carefully to
_�→_ identify patterns in successful solutions and common errors. Extract specific
_�→_ formatting requirements and incorporate them as clear guidance in your prompts.
When optimizing, you can incorporate critical thinking methods like review, revise,
_�→_ ensemble (generating multiple answers through different/similar prompts, then
_�→_ voting/integrating/checking the majority to obtain a final answer), selfAsk, etc.


NOTE: You should also try to use different operators to enhance workflow capabilities.
NOTE: If you are trying to answer from multiple solutions, you should try to use
_�→_ ensemble operator then to get the final answer.
NOTE: Each output of nodes except the interface node should be connected to the next
_�→_ node, ensuring data flows through the entire workflow. This guarantees all nodes
_�→_ properly receive inputs and pass outputs.
NOTE: If you are trying to add an ensemble node, you need to guarantee that there are
_�→_ multiple solution inputs available for the ensemble to work effectively.
NOTE: Program is a very powerful tool for solving MATH problems, you should try it! The
_�→_ Program operator can execute Python code to perform calculations and solve complex
_�→_ mathematical problems.
NOTE: Think big! Be bold in innovation and exploration. Don't be afraid to push

_�→_ boundaries and discover new approaches to problem-solving. Additionally, keep
_�→_ prompts concise - no more than 100 words per prompt. Shorter, focused prompts often
_�→_ perform better than lengthy ones.
"""


16


UPDATE_MERMAID_WORKFLOW_MATH = """
You are building a Graph and corresponding Prompt to jointly solve {type} problems.

_�→_ Here is a graph written in Mermaid and the corresponding prompt that performed
_�→_ excellently in a previous iteration (maximum score is 1). There are also some
_�→_ special operators defined in the operator_description section. Referring to this
_�→_ given graph and prompt, which forms a basic example of a code solution approach,
_�→_ please reconstruct and optimize them.


_�→_ Here is a graph written in Mermaid and the corresponding prompt that performed
_�→_ excellently in a previous iteration (maximum score is 1). There are also some
_�→_ special operators defined in the operator_description section. Referring to this
_�→_ given graph and prompt, which forms a basic example of a code solution approach,
_�→_ please reconstruct and optimize them.
You should make further optimizations based on this graph. The modified graph should

_�→_ differ from the provided example or have the same architecture but with prompts
_�→_ fine-tuned based on logs. The specific differences should be noted within the
_�→_ <modification>xxx</modification> section.


_�→_ differ from the provided example or have the same architecture but with prompts
_�→_ fine-tuned based on logs. The specific differences should be noted within the
_�→_ <modification>xxx</modification> section.
<sample>
<experience>{experience}</experience>
<modification>(such as: add/delete/modify/...)</modification>
<score>{score}</score>
<graph>{graph_mermaid}</graph>
<role_prompt>{prompt}</role_prompt>(only prompt_custom)
<operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well
_�→_ but encountered errors, which can be used as references for optimization:
{log}


**Two Parents**


UPDATE_MERMAID_WORKFLOW = """
# Evolution-Based Graph Generation


You are tasked with generating a new workflow graph by combining elements from two
_�→_ parent graphs. This process is inspired by genetic algorithms where offspring
_�→_ inherit traits from both parents.
This evolutionary approach allows us to create new solutions that may be more effective
_�→_ than either parent graph.


17


UPDATE_MERMAID_WORKFLOW_CODE = """
You are building a Graph and corresponding Prompt to jointly solve {type} problems.

_�→_ Here is a graph written in Mermaid and the corresponding prompt that performed
_�→_ excellently in a previous iteration (maximum score is 1). There are also some
_�→_ special operators defined in the operator_description section. Referring to this
_�→_ given graph and prompt, which forms a basic example of a code solution approach,
_�→_ please reconstruct and optimize them.


_�→_ Here is a graph written in Mermaid and the corresponding prompt that performed
_�→_ excellently in a previous iteration (maximum score is 1). There are also some
_�→_ special operators defined in the operator_description section. Referring to this
_�→_ given graph and prompt, which forms a basic example of a code solution approach,
_�→_ please reconstruct and optimize them.
You should make further optimizations based on this graph. The modified graph should

_�→_ differ from the provided example or have the same architecture but with prompts
_�→_ fine-tuned based on logs. The specific differences should be noted within the
_�→_ <modification>xxx</modification> section.


_�→_ differ from the provided example or have the same architecture but with prompts
_�→_ fine-tuned based on logs. The specific differences should be noted within the
_�→_ <modification>xxx</modification> section.
<sample>
<experience>{experience}</experience>
<modification>(such as: add/delete/modify/...)</modification>
<score>{score}</score>
<graph>{graph_mermaid}</graph>
<role_prompt>{prompt}</role_prompt>(only prompt_custom)
<operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well
_�→_ but encountered errors, which can be used as references for optimization:
{log}


You can add, modify, or delete nodes, parameters, or connections in the workflow. For
_�→_ each change, clearly describe your single modification within XML tags using the
_�→_ format: <modification>your detailed modification description</modification>.


Your optimizations should be one of (this limitation is strict! you can only pick one
_�→_ action!):
1. Expanding the graph by adding new single (operators). Note: adding nodes may require
_�→_ modifying prompts for related nodes, you should also update the corresponding
_�→_ prompts if needed.
2. Deleting unnecessary nodes from the graph.
3. Modifying existing nodes or their connections or their prompt.


Prompt is very important for performance here are some exmaples you can learn from it:
{prompt_few_shot}


Ensure all necessary prompts are properly prepared and defined. The generated custom

_�→_ node's role is defined by their prompt. Use custom methods with proper formatting
_�→_ to ensure your output matches the expected structure. The system will extract
_�→_ answers based on specific rules for scoring, so maintaining the correct output
_�→_ format is critical.
Review the log data carefully to understand the expected answer format and ensure your
_�→_ implementation produces compatible output. Proper formatting is essential for
_�→_ accurate evaluation of your solution, and you should emphasize this in the prompt.
When optimizing, you can incorporate critical thinking methods like review, revise,
_�→_ ensemble (generating multiple answers through different/similar prompts, then
_�→_ voting/integrating/checking the majority to obtain a final answer), selfAsk, etc.
You should also try to use different operators to enhance workflow capabilities. Each
_�→_ operator should have the corresponding Mermaid class defined in the graph code.


NOTE: If you are trying to add an ensemble node, you need to guarantee that there are
_�→_ multiple solution inputs available for the ensemble to work effectively.
NOTE: In the code problems, ensemble different kinds of solutions is a good idea, and
_�→_ you should also try to use Test operator to validate the solutions.
NOTE: For code-related tasks, the final output must be either test_result["solution"]

_�→_ or custom_code_generate_result["response"], as these represent valid Python code.
_�→_ Make sure your workflow produces one of these formats as the final output to ensure
_�→_ proper evaluation.


_�→_ or custom_code_generate_result["response"], as these represent valid Python code.
_�→_ Make sure your workflow produces one of these formats as the final output to ensure
_�→_ proper evaluation.
NOTE: Think big! Be bold in innovation and exploration. Don't be afraid to push

_�→_ boundaries and discover new approaches to problem-solving. Additionally, keep
_�→_ prompts concise - no more than 100 words per prompt. Shorter, focused prompts often
_�→_ perform better than lengthy ones.


_�→_
_�→_
_�→_
"""


<parent_graph>
<graph_A>
<graph>{graph_A}</graph>
<prompt>{prompt_A}</prompt>
<score>{score_A}</score>
</graph_A>
<graph_B>
<graph>{graph_B}</graph>
<prompt>{prompt_B}</prompt>
<score>{score_B}</score>
</graph_B>
</parent_graph>


## Your Task
1. Analyze both **parent graphs** carefully (structure, nodes, connections, prompts,
_�→_ and purpose)
2. Create a new graph that:

    - Combines strengths from both parents

    - Introduces strategic innovations for improved performance

    - Applies evolutionary operations such as:

     - [Crossover:] [Merging] [effective] [sections(nodes)] [from] [both] [parents,] [you] [should]
_�→_ separate components in a reasonable and logical way, typically at interface or
_�→_ ensemble nodes

     - [Mutation:] [Making] [targeted] [modifications]

     - [Insertion:] [Adding] [beneficial] [new] [nodes]

     - [Deletion:] [Removing] [inefficient] [components]


## Guidelines

  - Focus on the strengths of both parent graphs, especially the higher-scoring parent,
_�→_ to inform your design decisions

  - Ensure your new graph is structurally correct and follows proper workflow patterns

  - If creating a significantly different structure proves challenging, you may maintain
_�→_ a similar structure to the parents but with improved prompts, or, you can simply
_�→_ add one extra node to ensemble

  - Pay careful attention to the connections between nodes to ensure proper data flow   _�→_ for example, ensemble nodes should have multiple inputs

  - Verify that your graph has correct input/output relationships and follows the
_�→_ operator usage guidelines

  - Use custom methods to restrict your output format, rather than using code (outside of
_�→_ the code, the system will extract answers based on certain rules and score them)


## Prompt Guidence

  - **Prompt engineering is crucial for performance**: Carefully analyze and learn from

_�→_ the prompts used in **high-scoring** historical workflows. Pay special attention to
_�→_ their structure, specificity, and instruction clarity. Here are some exemplary
_�→_ prompts you can use as reference:
{prompt_few_shot}


Your response should include:


1. A detailed explanation of your modifications in the <modification> section
2. The complete Mermaid code for the new graph
3. The updated prompts for any custom nodes


NOTE: Ensure the new graph is valid Mermaid syntax and represents a complete workflow
_�→_ solution. The following section contains critical rules and guidance for using
_�→_ operators in Mermaid that you MUST follow to create an effective workflow:
{mermiad_custom_prompt}
"""


**MERMAID_GUIDANCE**

**Mermaid Code Guidance**


MERMAID_CUSTOM_MATH = """
# Mermaid Graph Style Guide for MATH Problems
# This comprehensive guide defines styling and structure for creating consistent
_�→_ workflow diagrams


# Node Style Definitions
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef ProgrammerOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnSembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;


# ===== OPERATOR USAGE GUIDE =====


# 1. Interface Nodes (Entry/Exit Points)


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


# Every workflow diagram must include these two standard interface nodes:
# PROBLEM([Problem])   - The entry point providing initial input
# RETURN([Return response & cost])   - The exit point receiving final output
#
# Example:
# PROBLEM([Problem])
# RETURN([Return response & cost])
#
# class PROBLEM Interface
# class RETURN Interface
#
# Connection rules:
#   - PROBLEM node: Provides input to other nodes but never receives input
#   - RETURN node: Receives output from the final node but never produces output


# 2. Custom Operator Nodes
# Format: NodeName["Custom<br/>(role: role_name)"]
#
# Example:
# K["Custom<br/>(role: validate_1)"]
# class K CustomOp
#
# For multiple nodes with similar roles, use numbered suffixes:
# R1["Custom<br/>(role: review_solution_1)"]
# R2["Custom<br/>(role: review_solution_2)"]
# class R1 CustomOp
# class R2 CustomOp
#
# Connection example:
# PROBLEM --> |input| R1


# 3. Programmer Operator Nodes
# Format: P["Programmer<br/>(analysis: 'your analysis text')"]
#
# The Programmer operator requires two inputs:
#   - problem: The math problem to solve
#   - analysis: Instructions on how to approach the problem
#
# Examples:
# P["Programmer<br/>(analysis: 'Calculate step by step')"]
# class P ProgrammerOp
#
# Connection rules:
# 1. For the problem input:
# PROBLEM --> |problem|P # The problem must come from the PROBLEM node
#
# 2. For the analysis input (two options):
# C --> |analysis|P # You can use another node's output as analysis content
# # OR
# # You can specify the analysis directly in the node definition
#
# Complete example:
# PROBLEM --> |problem|P
# C --> |analysis|P
# class P ProgrammerOp


# 4. ScEnsemble Operator Nodes
# Format: ENSEMBLE["ScEnsemble<br/>"]
#
# Example:
# ENSEMBLE["ScEnsemble<br/>"]
# class ENSEMBLE ScEnSembleOp
#
# CRITICAL: ScEnsemble nodes MUST have multiple inputs to function correctly
# Example connections:
# SOLUTION_1 --> ENSEMBLE
# SOLUTION_2 --> ENSEMBLE
# ENSEMBLE --> NEXT_NODE


# ===== WORKFLOW PATTERNS =====


# Example Workflow Pattern
# This pattern demonstrates how to effectively combine multiple solution approaches:
# 1. Problem input flows to multiple solution-generating nodes (Custom and/or
_�→_ Programmer)
# 2. All solutions are combined using ScEnsemble
# 3. The ensemble result is returned as the final output
#
# Connection pattern:
# PROBLEM --> |input| SOLUTION_1
# PROBLEM --> |input| SOLUTION_2


19


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


# PROBLEM --> |problem| P
# SOLUTION_1 --> ENSEMBLE
# SOLUTION_2 --> ENSEMBLE
# P --> ENSEMBLE
# ENSEMBLE --> RETURN


# ===== GENERAL RULES =====


# Connection Rules
# 1. Direct connections from PROBLEM to any node that needs the initial problem as
_�→_ input
# 2. Direct connections between nodes where one node requires another's output
# 3. When using ProgrammerOp, the edges should have name on it
# 4. All connections must follow logical data flow and maintain workflow coherence


# Prompt Definition Requirements
# All prompts used in the graph must be defined in this format:
# <prompt>
# PROMPT_NAME_1="Your first prompt text here"
# PROMPT_NAME_2="Your second prompt text here"
# </prompt>
#
# Each prompt must be on a separate line with its own unique variable name


# IMPORTANT: Do not create new Operator types!
# Only use the predefined operators available in the system. Creating custom operators
# will cause workflow execution failures. Stick to the operators documented in this
_�→_ guide.


# Best Practices:
#   - Use %% for comments in separate lines for clarity, don't add it to the end of the
_�→_ line because it is not a valide mermaid code
#   - Always include the style block (classDef section) even if not all classes are used
#   - Always ensure multiple inputs to ScEnsemble nodes (this is a common mistake)
#   - Maintain consistent naming conventions for all nodes and classes
#   - Use descriptive role names that indicate the node's purpose
#   - Label connections with |input| where appropriate for clarity


"""
MERMAID_CUSTOM_CODE = """
# Operator Style Definitions
# Each operator has a specific style for consistent visualization in the Mermaid graph
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef CustomCodeGenerateOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnSembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef TestOp fill:#d8f0d8,stroke:#2e8b57,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;


# ===== OPERATOR USAGE GUIDE =====


# 1. Interface Nodes
# Interface nodes represent the entry and exit points of your workflow
# Required in every graph:
# PROBLEM([Problem])   - Starting point that provides input
# RETURN([Return response & cost])   - Endpoint that receives final output
#
# Example:
# class PROBLEM Interface
# class RETURN Interface
#
# Connection rules:
#   - PROBLEM node provides input but doesn't receive any
#   - RETURN node receives output but doesn't produce any


# 2. Custom Operator
# Used for specialized reasoning strategies with defined roles
#
# Example:
# K["Custom<br/>(role: xxx)"]
# class K CustomOp
#
# For multiple instances with same prompt type:
# R1["Custom<br/>(role: xxx_1)"]
# R2["Custom<br/>(role: xxx_2)"]
# class R1 CustomOp
# class R2 CustomOp
#
# Connection rules:
#   - Connect directly to PROBLEM if it needs the initial problem
#   - Connect to other nodes if it needs their output


20


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


# 3. CustomCodeGenerate Operator
# Specialized for code generation tasks based on problem descriptions
#
# Example:
# CODE_GEN["CustomCodeGenerate<br/>(instruction: xxx)"]
# class CODE_GEN CustomCodeGenerateOp
#
# Multiple approaches:
# CODE_GEN_1["CustomCodeGenerate<br/>(instruction: aaa)"]
# CODE_GEN_2["CustomCodeGenerate<br/>(instruction: bbb)"]
# class CODE_GEN_1 CustomCodeGenerateOp
# class CODE_GEN_2 CustomCodeGenerateOp
#


# 4. ScEnsemble Operator
# Combines multiple solutions into one cohesive result
#
# Example:
# ENSEMBLE["ScEnsemble<br/>"]
# class ENSEMBLE ScEnsembleOp
#
# CRITICAL: Must have multiple inputs to function correctly
# SOLUTION_1 --> ENSEMBLE
# SOLUTION_2 --> ENSEMBLE
# ENSEMBLE --> other_node


# 5. Test Operator
# Validates solutions against test cases
#
# Example:
# T["Test<br/>"]
# class T TestOp
#
# Typical connection pattern:
# CODE_GEN --> |solution|T
# ENTRY_POINT --> |entry_point|T
# PROBLEM --> |problem|T
# T --> DECISION_NODE
#
# Decision node for test results:
# CHECK_TEST{test_result<br/>passed?}
# class CHECK_TEST DecisionOp
# CHECK_TEST -- Failed --> IMPROVE_SOLUTION
# CHECK_TEST -- Passed --> RETURN


# ===== IMPORTANT NOTES =====
#   - You cannot create other class operations
#   - Always ensure multiple inputs to ScEnsemble nodes
#   - All prompts must be defined in the prompt section
#   - Format prompts as:
# <prompt>
# PROMPT_NAME_1="Your first prompt text here"
# PROMPT_NAME_2="Your second prompt text here"
# </prompt>


# IMPORTANT: Do not create new Operator types!
# Only use the predefined operators available in the system. Creating custom operators
# will cause workflow execution failures. Stick to the operators documented in this
_�→_ guide.
"""


**GENERATE_PYTHON_CODE**

**Generate Python Code**


GENERATE_PYTHON_CODE = """
Below is a graph, the corresponding Python code, and the prompt.
<old_workflow>
<old_graph>{old_mermaid}</old_graph>
<old_code>{old_code}</old_code>
<old_role_prompt>{old_prompt}</old_role_prompt>(only prompt_custom)
</old_workflow>


Based on this example of old graph and code, you need to generate new Python code
_�→_ according to the new graph and given prompt.


<information_for_new_workflow>
<new_graph>{new_mermaid}</new_graph>


21


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


<new_role_prompt>{new_prompt}</new_role_prompt>(only prompt_custom)
<operator_description>{operator_description}</operator_description>
</information_for_new_workflow>


The output format should be:


<code>New generated python code</code>


Carefully analyze the new_graph to generate corresponding code. Pay attention to:
1. The connections between each node
2. The role and function of each operator
3. The input/output relationships


NOTE: especially for the final output, you should ensure that the output format is
_�→_ correct, you can learn it from old code.
"""


**LLM_AS_JUDGE**

**LLM As Judge**


LLM_AS_JUDGER = """


Your task is to select the most promising graph from the candidates. Here is the
_�→_ content of each graph, written in Mermaid code, along with explanations of how each
_�→_ new graph was generated from parent graphs:
Each new graph is derived from two parent graphs, with parent graph information as
_�→_ follows:


<parent_graph>
<parent_graph_A>{parent_graph_A}</parent_graph_A>
<parent_prompt_A>{parent_prompt_A}</parent_prompt_A>


<parent_graph_B>{parent_graph_B}</parent_graph_B>
<parent_prompt_B>{parent_prompt_B}</parent_prompt_B>
</parent_graph>


<graph_candidates>
<graph_A>{graph_A}</graph_A>
<modification_A>{modification_A}</modification_A>
<prompt_A>{prompt_A}</prompt_A>


<graph_B>{graph_B}</graph_B>
<modification_B>{modification_B}</modification_B>
<prompt_B>{prompt_B}</prompt_B>


<graph_C>{graph_C}</graph_C>
<modification_C>{modification_C}</modification_C>
<prompt_C>{prompt_C}</prompt_C>


<graph_D>{graph_D}</graph_D>
<modification_D>{modification_D}</modification_D>
<prompt_D>{prompt_D}</prompt_D>
</graph_candidates>


Please evaluate the graph candidates and select the most promising one based on these
_�→_ criteria:


1. **Workflow Coherence**: Assess how well the nodes connect and form a logical
_�→_ workflow


22


For each node in the graph, implement the appropriate code and use the corresponding

_�→_ prompts from new_prompt. If a node or operator doesn't have an explicit prompt
_�→_ provided, DO NOT create one - use empty strings instead. For example, Program
_�→_ operators may have empty analysis fields.


_�→_ prompts from new_prompt. If a node or operator doesn't have an explicit prompt
_�→_ provided, DO NOT create one - use empty strings instead. For example, Program
_�→_ operators may have empty analysis fields.
Every prompt referenced in your Python code must be defined in the prompt_custom

_�→_ module. When adding new functionality to your Python code, make sure to import any
_�→_ necessary libraries or modules, except for operator, prompt_custom,
_�→_ create_llm_instance, and CostManage which are automatically imported.


_�→_ module. When adding new functionality to your Python code, make sure to import any
_�→_ necessary libraries or modules, except for operator, prompt_custom,
_�→_ create_llm_instance, and CostManage which are automatically imported.
Your implementation must be robust - ensure all methods return appropriate values and
_�→_ ** [never] [return] [None] [for] [any] [field] ** [.] [Pay] [special] [attention] [to] [error] [handling] [and]
_�→_ edge cases.
Use custom methods with proper formatting to ensure your output matches the expected
_�→_ structure. The system will extract answers based on specific rules for scoring, so
_�→_ maintaining the correct output format is critical.


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


2. **Innovation**: Evaluate how the new graph improves upon the parent graphs
3. **Complexity Balance**: Check if the graph has appropriate complexity (neither too
_�→_ simple nor unnecessarily complex)
4. **Prompt Quality**: Examine the quality and specificity of the node prompts
5. **Modification Rationale**: Consider the thoughtfulness of the explanation provided
_�→_ for the changes


Here are some history of previous graphs and their corresponding score:
<history>
{elites_history}
</history>


For each candidate graph, provide a score from 1-10 for each criterion and explain your
_�→_ reasoning, you can learn from the history graphs.
You should avoid selecting graphs with structural defects, such as:
1. Incorrectly connecting nodes (e.g., CustomOp should not directly feed into
_�→_ ProgrammerOp)
2. Not properly using the ensemble node (all solution-generating nodes should feed into
_�→_ it)
3. Missing critical connections between nodes
4. Creating circular dependencies


Here are the specific structural rules for this type of workflow:
{mermaid_usage}


Additionally, consider how well the graph follows established patterns from successful
_�→_ historical examples.


Then select the graph with the highest total score as the most promising candidate.


<evaluation>
<graph_A_score>
<workflow_coherence>Score (1-10)</workflow_coherence>
<innovation>Score (1-10)</innovation>
<complexity_balance>Score (1-10)</complexity_balance>
<prompt_quality>Score (1-10)</prompt_quality>
<modification_rationale>Score (1-10)</modification_rationale>
<total_score>Sum of all scores</total_score>
<explanation>Detailed explanation of your evaluation</explanation>
</graph_A_score>


<graph_B_score>
<workflow_coherence>Score (1-10)</workflow_coherence>
<innovation>Score (1-10)</innovation>
<complexity_balance>Score (1-10)</complexity_balance>
<prompt_quality>Score (1-10)</prompt_quality>
<modification_rationale>Score (1-10)</modification_rationale>
<total_score>Sum of all scores</total_score>
<explanation>Detailed explanation of your evaluation</explanation>
</graph_B_score>


<graph_C_score>
<workflow_coherence>Score (1-10)</workflow_coherence>
<innovation>Score (1-10)</innovation>
<complexity_balance>Score (1-10)</complexity_balance>
<prompt_quality>Score (1-10)</prompt_quality>
<modification_rationale>Score (1-10)</modification_rationale>
<total_score>Sum of all scores</total_score>
<explanation>Detailed explanation of your evaluation</explanation>
</graph_C_score>


<graph_D_score>
<workflow_coherence>Score (1-10)</workflow_coherence>
<innovation>Score (1-10)</innovation>
<complexity_balance>Score (1-10)</complexity_balance>
<prompt_quality>Score (1-10)</prompt_quality>
<modification_rationale>Score (1-10)</modification_rationale>
<total_score>Sum of all scores</total_score>
<explanation>Detailed explanation of your evaluation</explanation>
</graph_D_score>
</evaluation>


<selected_graph>[A/B/C/D]</selected_graph>
<justification>

Please provide a comprehensive justification for your selection, highlighting the
_�→_ key strengths of the chosen graph and how it represents the most effective
_�→_ approach to solving the problem.
</justification>
"""


23


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


A.4 DETAILS OF BASELINE METHODS


In this section, we provide a detailed description of the configurations for baseline methods:


    - **CoT** (Kojima et al., 2022). Chain-of-Thought prompting guides the LLM to decompose
complex reasoning into a sequence of intermediate steps, rather than generating a direct
answer.

    - **ComplexCoT** (Fu et al., 2023). This method builds on CoT by explicitly modeling and
controlling reasoning complexity across multiple steps. We use the official implementation
from [https://github.com/FranxYao/Complexity-Based-Prompting/](https://github.com/FranxYao/Complexity-Based-Prompting/tree/main)
[tree/main.](https://github.com/FranxYao/Complexity-Based-Prompting/tree/main)

    - **Self-Consistency** (Wang et al., 2023). This method generates five independent chainof-thought trajectories by sampling the LLM multiple times, then consolidates them via
majority voting to improve the reliability of the final answer.

    - **LLM-Debate** (Du et al., 2024). A panel of five role-specific LLM agents engage in up to
two rounds of structured debate; the final answer is selected by majority vote.

    - **DyLAN** (Liu et al., 2024). DyLAN employs a dynamic multi-agent framework in which
agents iteratively share and refine intermediate reasoning to converge on high-quality solutions.

    - **MacNet** (Qian et al., 2024). MacNet structures its agent network as a fully meshed graph,
enabling every agent to exchange and integrate information with all others at each reasoning
step.

    - **GPTSwarm** (Zhuge et al., 2024). GPTSwarm leverages a swarm-intelligence paradigm
by orchestrating multiple LLM agents in parallel; agents independently propose solutions
and iteratively refine them through shared feedback, following the protocol outlined in the
original paper.

    - **MaAS** (Zhang et al., 2025b). MaAS integrates a trainable module for dynamic workflow
assignment across multiple agents; we replicate the authors’ original architecture using their
official code and retain their specified hyperparameters without modification.

    - **AutoAgents** (Chen et al., 2024). AutoAgents orchestrates a pipeline of specialized LLM
agents that autonomously schedule, execute, and refine subtasks through coordinated interactions; we replicate the authors’ default configuration and parameter settings from the
official repository.

    - **ADAS** (Hu et al., 2024). ADAS employs an adaptive debate-and-selection mechanism
among multiple LLM agents, dynamically refining candidate solutions based on quality
criteria.

    - **AFlow** (Zhang et al., 2024c). In the original AFlow study, both gpt-4o-mini and claude-3.5sonnet are used. For a fair comparison, we constrain AFlow to gpt-4o-mini only and set
MAX_ITERATION to 20.


A.5 DETAILS OF TASKS AND BENCHMARKS


Following established methodologies in workflow automation (Zhang et al., 2024c; 2025b), we
partition each dataset into training and test sets with a TRAIN:TEST ratio of 1:4. For the MATH
benchmark, we follow the approach in (Zhang et al., 2025b), selecting a subset of 617 challenging
problems across four representative categories: Combinatorics & Probability, Number Theory, Prealgebra, and Pre-calculus, all at difficulty level 5. The dataset statistics are summarized in Table 4.


B CASE STUDY


In this section, we present examples of the optimal workflows discovered by our approach for each of
the four datasets. For each dataset, we provide the **Mermaid code**, corresponding **Python code**, and
the rendered **workflow diagram** .


The declarative Mermaid representation serves as the backbone of structured workflow generation,
enabling several crucial capabilities:


24


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


Table 4: Statistics of datasets used in our experiments.


**Domain** **Dataset** **Training** **Testing** **Evaluation Metric**


Code Generation HumanEval 33 131 pass@1
MBPP 86 341 pass@1


Math Reasoning GSM8K 264 1,055 Accuracy
MATH 119 486 Accuracy


1. **Typed, Semantically Aligned Representation.** Mermaid encodes workflows as statically
typed and semantically annotated graphs, allowing prompt semantics and operator roles to
be aligned explicitly within the workflow structure.

2. **Human-Readable and Verifiable Syntax.** Unlike imperative representations, Mermaid
provides a format that is both visually interpretable and statically verifiable, facilitating
intuitive debugging, planning, and workflow validation.

3. **Reliable Code Translation.** The structured nature of Mermaid graphs enables seamless
compilation into executable Python code, with consistent type and role guarantees that
reduce runtime errors and improve reliability.


These properties make MermaidFlow not just a search mechanism, but a principled framework
for building modular, interpretable, and safer agentic workflows. The case studies illustrate how
these benefits translate into practical gains across crucial domains such as programming, math, and
symbolic reasoning.


B.1 GSM8K


Figure 5: Mermaid diagram for GSM8K.


**Mermaid Code**


flowchart TD
%% Nodes
PROBLEM([Problem])
C["Custom<br/>(role: simple_solver_1)"]


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


P1["Programmer<br/>(analysis: 'Calculate step by step')"]
P2["Programmer<br/>(analysis: 'Generate solution with edge cases')"]
P3["Programmer<br/>(analysis: 'Execute code for precise calculations')"]
P4["Programmer<br/>(analysis: 'Verify and validate results')"]
P5["Programmer<br/>(analysis: 'Explore alternative methods')"]
P6["Programmer<br/>(analysis: 'Refine and format final output')"]
ENSEMBLE["ScEnsemble<br/>"]
RETURN([Return response & cost])


%% Styles
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef ProgrammerOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnSembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;


%% Assign classes
class C CustomOp
class P1 ProgrammerOp
class P2 ProgrammerOp
class P3 ProgrammerOp
class P4 ProgrammerOp
class P5 ProgrammerOp
class P6 ProgrammerOp
class ENSEMBLE ScEnSembleOp
class PROBLEM Interface
class RETURN Interface


%% Flow (arrows show data relationships)
PROBLEM --> |input|C
PROBLEM --> |problem|P1
PROBLEM --> |problem|P2
PROBLEM --> |problem|P3
PROBLEM --> |problem|P4
PROBLEM --> |problem|P5
C --> ENSEMBLE
P1 --> ENSEMBLE
P2 --> ENSEMBLE
P3 --> ENSEMBLE
P4 --> ENSEMBLE
P5 --> ENSEMBLE
ENSEMBLE --> P6
P6 --> RETURN


**Python Code**


from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_16.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
import weave


DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Workflow:
def __init__(
self,
name: str,
llm_config,
dataset: DatasetType,
) -> None:
self.name = name
self.dataset = dataset
self.llm = create_llm_instance(llm_config)
self.sc_ensemble = operator.ScEnsemble(self.llm)
self.custom = operator.Custom(self.llm)
self.programmer = operator.Programmer(self.llm)


@weave.op()
async def __call__(self, problem: str):
"""
Implementation of the workflow
Each operator is callable, you can call it directly.
"""
# Step 1: Use the Custom operator to generate a detailed solution
custom_response = await self.custom(input=problem,
_�→_ instruction=prompt_custom.SIMPLE_SOLVER_1, role="simple_solver_1")


26


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


# Step 2: Use the Programmer operator to analyze the problem and provide a code
_�→_ solution
programmer_response_1 = await self.programmer(problem=problem,
_�→_ analysis="Calculate step by step")


# Step 3: Use the Programmer operator to generate a solution considering edge
_�→_ cases
programmer_response_2 = await self.programmer(problem=problem,
_�→_ analysis="Generate solution with edge cases")


# Step 4: Use the Programmer operator to execute code for precise calculations
programmer_response_3 = await self.programmer(problem=problem,
_�→_ analysis="Execute code for precise calculations")


# Step 5: Use the Programmer operator to verify and validate results
programmer_response_4 = await self.programmer(problem=problem, analysis="Verify
_�→_ and validate results")


# Step 6: Use the Programmer operator to explore alternative methods
programmer_response_5 = await self.programmer(problem=problem,
_�→_ analysis="Explore alternative methods")


# Step 7: Combine the responses from Custom and all Programmer responses for
_�→_ the ScEnsemble
solutions = [
custom_response['response'],
programmer_response_1['output'],
programmer_response_2['output'],
programmer_response_3['output'],
programmer_response_4['output'],
programmer_response_5['output']
]


# Step 8: Use the ScEnsemble operator to select the best solution
ensemble_response = await self.sc_ensemble(solutions=solutions,
_�→_ problem=problem)


# Step 9: Refine and format the final output
final_output = await self.programmer(problem=ensemble_response['response'],
_�→_ analysis="Refine and format final output")


return final_output['output'], self.llm.get_usage_summary()["total_cost"]


**Corresponding Prompt**


SIMPLE_SOLVER_1 = """Please solve the given mathematical problem step by step. Follow
_�→_ these guidelines:


1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Verify and validate your results to ensure accuracy.
6. Present the final answer enclosed in \\boxed{} LaTeX notation.
7. Ensure all mathematical notation is in LaTeX format.


Your solution should be thorough, mathematically sound, and easy to understand."""


27


**1458**

**1459**


**1460**

**1461**

**1462**

**1463**

**1464**

**1465**


**1466**

**1467**

**1468**

**1469**

**1470**

**1471**


**1472**

**1473**

**1474**

**1475**

**1476**


**1477**

**1478**

**1479**

**1480**

**1481**

**1482**


**1483**

**1484**

**1485**

**1486**

**1487**

**1488**


**1489**

**1490**

**1491**

**1492**

**1493**

**1494**


**1495**

**1496**

**1497**

**1498**

**1499**


**1500**

**1501**

**1502**

**1503**

**1504**

**1505**


**1506**

**1507**

**1508**

**1509**

**1510**

**1511**


B.2 MATH


Figure 6: Mermaid diagram for MATH.


**Mermaid Code**


flowchart TD
%% Nodes
PROBLEM([Problem])
C1["Custom<br/>(role: simple_solver_1)"]
C2["Custom<br/>(role: simple_solver_2)"]
C3["Custom<br/>(role: alternative_solver)"]
C4["Custom<br/>(role: detailed_solution_outline)"]
C5["Custom<br/>(role: comprehensive_solution]"]
P["Programmer<br/>(analysis: 'Solve the math problem step by step')"]
REFINE["Custom<br/>(role: refine_solution)"]
ENSEMBLE["ScEnsemble<br/>"]
RETURN([Return response & cost])


%% Styles
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef ProgrammerOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnSembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;


%% Assign classes
class C1 CustomOp
class C2 CustomOp
class C3 CustomOp
class C4 CustomOp
class C5 CustomOp
class P ProgrammerOp
class REFINE CustomOp
class PROBLEM Interface
class RETURN Interface
class ENSEMBLE ScEnSembleOp


%% Flow (arrows show data relationships)
PROBLEM --> |input|C1
PROBLEM --> |input|C2
PROBLEM --> |input|C3
PROBLEM --> |input|C4
PROBLEM --> |input|C5
PROBLEM --> |problem|P
C1 --> ENSEMBLE
C2 --> ENSEMBLE
C3 --> ENSEMBLE


28


**1512**

**1513**


**1514**

**1515**

**1516**

**1517**

**1518**

**1519**


**1520**

**1521**

**1522**

**1523**

**1524**

**1525**


**1526**

**1527**

**1528**

**1529**

**1530**


**1531**

**1532**

**1533**

**1534**

**1535**

**1536**


**1537**

**1538**

**1539**

**1540**

**1541**

**1542**


**1543**

**1544**

**1545**

**1546**

**1547**

**1548**


**1549**

**1550**

**1551**

**1552**

**1553**


**1554**

**1555**

**1556**

**1557**

**1558**

**1559**


**1560**

**1561**

**1562**

**1563**

**1564**

**1565**


C4 --> ENSEMBLE
C5 --> ENSEMBLE
P --> ENSEMBLE
ENSEMBLE --> REFINE
REFINE --> RETURN


**Python Code**


from typing import Literal
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_16.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
import weave


DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Workflow:
def __init__(
self,
name: str,
llm_config,
dataset: DatasetType,
) -> None:
self.name = name
self.dataset = dataset
self.llm = create_llm_instance(llm_config)
self.custom1 = operator.Custom(self.llm)
self.custom2 = operator.Custom(self.llm)
self.custom3 = operator.Custom(self.llm)
self.custom4 = operator.Custom(self.llm)
self.custom5 = operator.Custom(self.llm)
self.programmer = operator.Programmer(self.llm)
self.refine = operator.Custom(self.llm)
self.ensemble = operator.ScEnsemble(self.llm)


@weave.op()
async def __call__(self, problem: str):
"""Implementation of the workflow"""
# Use the first custom operator to solve the problem step by step
custom_response_1 = await self.custom1(input=problem,
_�→_ instruction=prompt_custom.SIMPLE_SOLVER, role="simple_solver_1")


# Use the second custom operator to solve the problem step by step
custom_response_2 = await self.custom2(input=problem,
_�→_ instruction=prompt_custom.SIMPLE_SOLVER, role="simple_solver_2")


# Use the third custom operator to provide an alternative approach to the
_�→_ problem
custom_response_3 = await self.custom3(input=problem,
_�→_ instruction=prompt_custom.ALTERNATIVE_SOLVER, role="alternative_solver")


# Use the fourth custom operator to provide a detailed solution outline
custom_response_4 = await self.custom4(input=problem,
_�→_ instruction=prompt_custom.DETAILED_SOLUTION_OUTLINE,
_�→_ role="detailed_solution_outline")


# Use the fifth custom operator to provide a comprehensive solution
custom_response_5 = await self.custom5(input=problem,
_�→_ instruction=prompt_custom.COMPREHENSIVE_SOLUTION,
_�→_ role="comprehensive_solution")


# Use the programmer operator to analyze the problem and provide a detailed
_�→_ solution
programmer_response = await self.programmer(problem=problem, analysis="Solve
_�→_ the math problem step by step")


# Combine all responses into a list for ensemble processing
solutions = [
custom_response_1['response'],
custom_response_2['response'],
custom_response_3['response'],
custom_response_4['response'],
custom_response_5['response'],
programmer_response['output']
]


# Use the ensemble operator to select the best solution


29


**1566**

**1567**


**1568**

**1569**

**1570**

**1571**

**1572**

**1573**


**1574**

**1575**

**1576**

**1577**

**1578**

**1579**


**1580**

**1581**

**1582**

**1583**

**1584**


**1585**

**1586**

**1587**

**1588**

**1589**

**1590**


**1591**

**1592**

**1593**

**1594**

**1595**

**1596**


**1597**

**1598**

**1599**

**1600**

**1601**

**1602**


**1603**

**1604**

**1605**

**1606**

**1607**


**1608**

**1609**

**1610**

**1611**

**1612**

**1613**


**1614**

**1615**

**1616**

**1617**

**1618**

**1619**


ensemble_response = await self.ensemble(solutions=solutions, problem=problem)


# Use the refine operator to ensure clarity and correctness of the output
refined_response = await self.refine(input=ensemble_response['response'],
_�→_ instruction=prompt_custom.REFINE_SOLUTION, role="refine_solution")


# Return the final output and cost
return refined_response['response'], self.llm.get_usage_summary()["total_cost"]


**Corresponding Prompt**


SIMPLE_SOLVER = """Please solve the given mathematical problem step by step. Follow
_�→_ these guidelines:


1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \\boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.
"""


ALTERNATIVE_SOLVER = """Please provide an alternative approach to solving the given
_�→_ mathematical problem. Follow these guidelines:


1. Clearly restate the problem.
2. Identify any different methods or perspectives that could be applied.
3. Provide calculations and reasoning, using LaTeX notation for mathematical
_�→_ expressions.
4. Ensure clarity and correctness in your explanation.
5. Present the final answer enclosed in \\boxed{} LaTeX notation.
"""


DETAILED_SOLUTION_OUTLINE = """Please provide a detailed outline for solving the given
_�→_ mathematical problem. Follow these guidelines:


1. Clearly restate the problem.
2. Identify key concepts and theorems relevant to the problem.
3. Outline the steps needed to solve the problem, including any necessary calculations.
4. Ensure that the outline is structured logically and is easy to follow.
5. Use LaTeX notation for any mathematical expressions.
"""


COMPREHENSIVE_SOLUTION = """Please provide a comprehensive solution to the given
_�→_ mathematical problem. Follow these guidelines:


1. Clearly restate the problem.
2. Explain the mathematical concepts and theorems involved.
3. Provide a detailed, logical progression of steps leading to the solution.
4. Show all calculations using LaTeX notation for mathematical expressions.
5. Present the final answer clearly marked and enclosed in \\boxed{} LaTeX notation.
"""


REFINE_SOLUTION = """Given the mathematical problem and the solutions generated, please
_�→_ refine the output to ensure clarity and correctness. Follow these guidelines:


1. Review the solutions provided.
2. Ensure all calculations are accurate and clearly presented.
3. Summarize the findings and present the final answer in a clear format.
4. Use LaTeX notation for any mathematical expressions.
5. Ensure the final answer is enclosed in \\boxed{} LaTeX notation.
"""


30


**1620**

**1621**


**1622**

**1623**

**1624**

**1625**

**1626**

**1627**


**1628**

**1629**

**1630**

**1631**

**1632**

**1633**


**1634**

**1635**

**1636**

**1637**

**1638**


**1639**

**1640**

**1641**

**1642**

**1643**

**1644**


**1645**

**1646**

**1647**

**1648**

**1649**

**1650**


**1651**

**1652**

**1653**

**1654**

**1655**

**1656**


**1657**

**1658**

**1659**

**1660**

**1661**


**1662**

**1663**

**1664**

**1665**

**1666**

**1667**


**1668**

**1669**

**1670**

**1671**

**1672**

**1673**


B.3 HUMANEVAL


Figure 7: Mermaid diagram for HumanEval.


**Mermaid Code**


flowchart TD
%% Nodes
PROBLEM([Problem])
ENTRY_POINT([entry_point])
C1["CustomCodeGenerate<br/>(instruction: simple_solver_1)"]
C2["CustomCodeGenerate<br/>(instruction: simple_solver_2)"]
C3["CustomCodeGenerate<br/>(instruction: optimized_solver)"]
C4["CustomCodeGenerate<br/>(instruction: improved_solution_1)"]
ENSEMBLE["ScEnsemble<br/>"]
T["Test<br/>"]
RETURN([Return response & cost])


%% Styles
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef CustomCodeGenerateOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnsembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef TestOp fill:#d8f0d8,stroke:#2e8b57,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;


%% Assign classes
class PROBLEM Interface
class ENTRY_POINT Interface
class C1 CustomCodeGenerateOp
class C2 CustomCodeGenerateOp
class C3 CustomCodeGenerateOp
class C4 CustomCodeGenerateOp
class ENSEMBLE ScEnsembleOp
class T TestOp
class RETURN Interface


%% Flow (arrows show data relationships)
PROBLEM --> |input|C1
PROBLEM --> |input|C2
PROBLEM --> |input|C3
ENTRY_POINT --> |entry_point|C1
ENTRY_POINT --> |entry_point|C2
ENTRY_POINT --> |entry_point|C3
C1 --> |solution|ENSEMBLE
C2 --> |solution|ENSEMBLE
C3 --> |solution|ENSEMBLE
ENSEMBLE --> |combined_solution|T


31


**1674**

**1675**


**1676**

**1677**

**1678**

**1679**

**1680**

**1681**


**1682**

**1683**

**1684**

**1685**

**1686**

**1687**


**1688**

**1689**

**1690**

**1691**

**1692**


**1693**

**1694**

**1695**

**1696**

**1697**

**1698**


**1699**

**1700**

**1701**

**1702**

**1703**

**1704**


**1705**

**1706**

**1707**

**1708**

**1709**

**1710**


**1711**

**1712**

**1713**

**1714**

**1715**


**1716**

**1717**

**1718**

**1719**

**1720**

**1721**


**1722**

**1723**

**1724**

**1725**

**1726**

**1727**


T --> |fail|C4
T --> |pass|RETURN
C4 --> RETURN


**Python Code**


from typing import Literal
import workspace.HumanEval.workflows.template.operator as operator
import workspace.HumanEval.workflows.round_5.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
import weave


DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Workflow:
def __init__(
self,
name: str,
llm_config,
dataset: DatasetType,
) -> None:
self.name = name
self.dataset = dataset
self.llm = create_llm_instance(llm_config)
self.custom = operator.Custom(self.llm)
self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
self.sc_ensemble = operator.ScEnsemble(self.llm)
self.test = operator.Test(self.llm)


async def __call__(self, problem: str, entry_point: str):
"""
Implementation of the workflow
Custom operator to generate multiple solutions and select the best one.
"""
# Generate initial solutions using custom code generation
solution_response_1 = await self.custom_code_generate(problem=problem,
_�→_ entry_point=entry_point, instruction=prompt_custom.CODE_GENERATE_PROMPT)
solution_1 = solution_response_1['response']


solution_response_2 = await self.custom_code_generate(problem=problem,
_�→_ entry_point=entry_point, instruction=prompt_custom.CODE_GENERATE_PROMPT)
solution_2 = solution_response_2['response']


# Generate an optimized solution
optimized_solution_response = await self.custom_code_generate(problem=problem,
_�→_ entry_point=entry_point,
_�→_ instruction=prompt_custom.OPTIMIZED_CODE_GENERATE_PROMPT)
optimized_solution = optimized_solution_response['response']


# Combine solutions using ensemble method
ensemble_response = await self.sc_ensemble(solutions=[solution_1, solution_2,
_�→_ optimized_solution], problem=problem)
combined_solution = ensemble_response['response']


# Test the combined solution
test_result = await self.test(problem=problem, solution=combined_solution,
_�→_ entry_point=entry_point)


# If the solution fails the test, improve the code
if not test_result['result']:

improved_solution_response = await self.custom(input=problem,
_�→_ instruction=prompt_custom.IMPROVE_CODE_PROMPT)
improved_solution = improved_solution_response['response']
# Test the improved solution
test_result = await self.test(problem=problem, solution=improved_solution,
_�→_ entry_point=entry_point)
combined_solution = improved_solution if test_result['result'] else
_�→_ combined_solution # Use improved solution if it passes tests


return combined_solution, self.llm.get_usage_summary()["total_cost"]


32


**1728**

**1729**


**1730**

**1731**

**1732**

**1733**

**1734**

**1735**


**1736**

**1737**

**1738**

**1739**

**1740**

**1741**


**1742**

**1743**

**1744**

**1745**

**1746**


**1747**

**1748**

**1749**

**1750**

**1751**

**1752**


**1753**

**1754**

**1755**

**1756**

**1757**

**1758**


**1759**

**1760**

**1761**

**1762**

**1763**

**1764**


**1765**

**1766**

**1767**

**1768**

**1769**


**1770**

**1771**

**1772**

**1773**

**1774**

**1775**


**1776**

**1777**

**1778**

**1779**

**1780**

**1781**


**Corresponding Prompt**


IMPROVE_CODE_PROMPT = """
The previous solution failed some test cases in the HumanEval benchmark. Please conduct

_�→_ a thorough analysis of the problem statement, identifying all edge cases and
_�→_ potential pitfalls. Then, provide an improved solution that not only fixes the
_�→_ issues but also optimizes performance and adheres to industry-standard coding
_�→_ practices. Ensure your revised code includes clear, concise comments that explain
_�→_ your logic and design choices, and that it robustly handles all specified
_�→_ requirements.
"""


CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches
_�→_ the one specified in the problem. Include necessary imports. Use clear variable
_�→_ names and add comments for clarity.


Problem:
{problem}


Function signature:
{entry_point}


Generate the complete function below:
"""


OPTIMIZED_CODE_GENERATE_PROMPT = """
Based on previous attempts and their outcomes, generate an optimized Python function to

_�→_ solve the given problem. Focus on improving efficiency, readability, and
_�→_ robustness. Ensure the function name matches the one specified in the problem,
_�→_ include necessary imports, and use clear variable names with comments for clarity.


Problem:
{problem}


Function signature:
{entry_point}


Generate the complete function below:
"""


ENSEMBLE_PROMPT = """
You have multiple solutions generated for the problem. Your task is to analyze these

_�→_ solutions and select the one that is most likely to be correct based on their
_�→_ similarities and performance. Ensure that the selected solution is robust and
_�→_ handles all edge cases effectively.
"""


33


**1782**

**1783**


**1784**

**1785**

**1786**

**1787**

**1788**

**1789**


**1790**

**1791**

**1792**

**1793**

**1794**

**1795**


**1796**

**1797**

**1798**

**1799**

**1800**


**1801**

**1802**

**1803**

**1804**

**1805**

**1806**


**1807**

**1808**

**1809**

**1810**

**1811**

**1812**


**1813**

**1814**

**1815**

**1816**

**1817**

**1818**


**1819**

**1820**

**1821**

**1822**

**1823**


**1824**

**1825**

**1826**

**1827**

**1828**

**1829**


**1830**

**1831**

**1832**

**1833**

**1834**

**1835**


B.4 MBPP


Figure 8: Mermaid diagram for MBPP.


**Mermaid Code**


flowchart TD
%% Nodes
PROBLEM([Problem])
ENTRY_POINT([entry_point])
C1["CustomCodeGenerate<br/>(instruction: simple_solver_1)"]
C2["CustomCodeGenerate<br/>(instruction: simple_solver_2)"]
C3["CustomCodeGenerate<br/>(instruction: optimized_solver)"]
C4["CustomCodeGenerate<br/>(instruction: fix_code)"]
ENSEMBLE["ScEnsemble<br/>"]
T["Test<br/>"]
RETURN([Return response & cost])


%% Styles
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef CustomCodeGenerateOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnsembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef DecisionOp fill:#ffffff,stroke:#444444,stroke-width:1px,stroke-dasharray:2 2;
classDef TestOp fill:#d8f0d8,stroke:#2e8b57,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;


%% Assign classes
class PROBLEM Interface
class ENTRY_POINT Interface
class C1 CustomCodeGenerateOp
class C2 CustomCodeGenerateOp
class C3 CustomCodeGenerateOp
class C4 CustomCodeGenerateOp
class ENSEMBLE ScEnsembleOp
class T TestOp
class RETURN Interface


%% Flow (arrows show data relationships)
PROBLEM --> |input|C1
PROBLEM --> |input|C2
PROBLEM --> |input|C3
PROBLEM --> |input|C4
ENTRY_POINT --> |entry_point|C1
ENTRY_POINT --> |entry_point|C2
ENTRY_POINT --> |entry_point|C3
ENTRY_POINT --> |entry_point|C4
C1 --> ENSEMBLE


34


**1836**

**1837**


**1838**

**1839**

**1840**

**1841**

**1842**

**1843**


**1844**

**1845**

**1846**

**1847**

**1848**

**1849**


**1850**

**1851**

**1852**

**1853**

**1854**


**1855**

**1856**

**1857**

**1858**

**1859**

**1860**


**1861**

**1862**

**1863**

**1864**

**1865**

**1866**


**1867**

**1868**

**1869**

**1870**

**1871**

**1872**


**1873**

**1874**

**1875**

**1876**

**1877**


**1878**

**1879**

**1880**

**1881**

**1882**

**1883**


**1884**

**1885**

**1886**

**1887**

**1888**

**1889**


C2 --> ENSEMBLE
C3 --> ENSEMBLE
C4 --> ENSEMBLE
ENSEMBLE --> T
T --> RETURN


**Python Code**


from typing import Literal
import workspace.MBPP.workflows.template.operator as operator
import workspace.MBPP.workflows.round_8.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
import weave


DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Workflow:
def __init__(
self,
name: str,
llm_config,
dataset: DatasetType,
) -> None:
self.name = name
self.dataset = dataset
self.llm = create_llm_instance(llm_config)
self.custom = operator.Custom(self.llm)
self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
self.sc_ensemble = operator.ScEnsemble(self.llm)
self.test = operator.Test(self.llm)


@weave.op()
async def __call__(self, problem: str, entry_point: str):
"""
Implementation of the workflow
Custom operator to generate anything you want.
But when you want to get standard code, you should use custom_code_generate
_�→_ operator.
"""
# Generate two solutions using different instructions
solution_1 = await self.custom_code_generate(problem=problem,
_�→_ entry_point=entry_point, instruction=prompt_custom.CODE_GENERATE_PROMPT)
solution_2 = await self.custom_code_generate(problem=problem,
_�→_ entry_point=entry_point, instruction=prompt_custom.CODE_GENERATE_PROMPT)
# Generate an optimized solution
optimized_solution = await self.custom_code_generate(problem=problem,
_�→_ entry_point=entry_point, instruction=prompt_custom.OPTIMIZED_CODE_PROMPT)
# Attempt to fix the code if necessary
fix_attempt = await self.custom_code_generate(problem=problem,
_�→_ entry_point=entry_point, instruction=prompt_custom.FIX_CODE_PROMPT)


# Use ensemble to select the best solution
ensemble_response = await self.sc_ensemble(solutions=[solution_1['response'],
_�→_ solution_2['response'], optimized_solution['response'],
_�→_ fix_attempt['response']], problem=problem)


# Test the selected solution
test_response = await self.test(problem=problem,
_�→_ solution=ensemble_response['response'], entry_point=entry_point)


# Return the final response and cost
return test_response['solution'], self.llm.get_usage_summary()["total_cost"]


**Corresponding Prompt**


CODE_GENERATE_PROMPT = """
Generate a Python function to solve the given problem. Ensure the function name matches
_�→_ the one specified in the problem. Include necessary imports. Use clear variable
_�→_ names and add comments for clarity.


Problem:
{problem}


Function signature:


35


**1890**

**1891**


**1892**

**1893**

**1894**

**1895**

**1896**

**1897**


**1898**

**1899**

**1900**

**1901**

**1902**

**1903**


**1904**

**1905**

**1906**

**1907**

**1908**


**1909**

**1910**

**1911**

**1912**

**1913**

**1914**


**1915**

**1916**

**1917**

**1918**

**1919**

**1920**


**1921**

**1922**

**1923**

**1924**

**1925**

**1926**


**1927**

**1928**

**1929**

**1930**

**1931**


**1932**

**1933**

**1934**

**1935**

**1936**

**1937**


**1938**

**1939**

**1940**

**1941**

**1942**

**1943**


{entry_point}


Generate the complete function below:
"""


OPTIMIZED_CODE_PROMPT = """
Based on previous attempts, generate an optimized Python function to solve the given

_�→_ problem. Ensure the function name matches the one specified in the problem. Include
_�→_ necessary imports, and focus on improving performance and readability. Use clear
_�→_ variable names and add comments for clarity.


Problem:
{problem}


Function signature:
{entry_point}


Generate the complete function below:
"""


FIX_CODE_PROMPT = """
The provided solution failed to pass the tests. Please analyze the error and fix the
_�→_ code. Ensure the function name and signature remain unchanged. If necessary, add or
_�→_ modify imports, correct logical errors, and improve the implementation.


Problem:
{input}


Provide the corrected function below:
"""


C COMMON TYPES OF PYTHON SCRIPT FAILURES


In this section we will list the type of failures that python script often arise, and with some examples
to demonstrate that.


1. Python’s flexibility often misleads the LLM into producing superficially reasonable but
unreliable control logic


(a) Unreliable if conditions, the LLM’s decision on whether a condition should trigger is
often incorrect, since you can guarantee the behavior of LLM’s output.

(b) Using a single prompt to drive a for loop over many iterations offers essentially no
benefit when the temperature is set to 0.


2. Incorrect instance initialization


(a) Parameters may be incorrect or missing.


3. Importing or referencing nonexistent modules/repositories


Here are some detailed Python script examples; we truncate irrelevant code segments and highlight
violation points using comments placed before the corresponding snippets.


C.1 UNRELIABLE CONTROL LOGIC FROM PYTHON’S FLEXIBILITY


**Unreliable if conditions**


...
# This ``` if ``` condition depends on the LLM’s output, but the prompt does not specify the
_�→_ expected output format.
for solution in solutions:
verify = await self.programmer(
problem=problem,
analysis=f"Verify if this solution is mathematically correct: {solution}"
)
if verify['output'].lower().startswith('correct'):
verified_solutions.append(solution)
...


36


**1944**

**1945**


**1946**

**1947**

**1948**

**1949**

**1950**

**1951**


**1952**

**1953**

**1954**

**1955**

**1956**

**1957**


**1958**

**1959**

**1960**

**1961**

**1962**


**1963**

**1964**

**1965**

**1966**

**1967**

**1968**


**1969**

**1970**

**1971**

**1972**

**1973**

**1974**


**1975**

**1976**

**1977**

**1978**

**1979**

**1980**


**1981**

**1982**

**1983**

**1984**

**1985**


**1986**

**1987**

**1988**

**1989**

**1990**

**1991**


**1992**

**1993**

**1994**

**1995**

**1996**

**1997**


**Meaningless for loop**


...
# Since all iterations use the same prompt and the default temperature is 0 (as in
_�→_ AFLOW), this for-loop is meaningless and only wastes tokens.
solutions = []
for _ in range(5):

solution = await self.custom(input=problem,
_�→_ instruction=prompt_custom.GENERATE_SOLUTION_PROMPT)
solutions.append(solution['response'])
...


C.2 INCORRECT INSTANCE INITIALIZATION


**Missing paramters**


...
# When initializing different nodes like programmer or sc_ensemble, they all require
_�→_ the ``` self.llm ``` parameter, but it gets ignored.
self.name = name
self.dataset = dataset
self.llm = create_llm_instance(llm_config)
self.custom = operator.Custom(self.llm)
self.programmer = operator.Programmer()
self.sc_ensemble = operator.ScEnsemble()
...


**Passing incorrect parameters**


...
# When initializing the LLM, incorrect instructions may be passed, since these
_�→_ instructions are provided only at inference time.
self.name = name
self.dataset = dataset
self.llm = create_llm_instance(llm_config)
self.custom1 = operator.Custom(self.llm, instruction=prompt_custom.SOLVE_PROMPT) #
_�→_ shouldn't pass instruction
self.custom2 = operator.Custom(self.llm, instruction=prompt_custom.VERIFY_PROMPT)
self.sc_ensemble = operator.ScEnsemble(self.llm)
...


C.3 IMPORTING OR REFERENCING NONEXISTENT MODULES/REPOSITORIES


**Importing unsupported packages**


...
# When generating the script, the LLM may import incorrect or disallowed packages.
from typing import Literal
import pandas as pd # this repo is not allowed to import
import workspace.MATH.workflows.template.operator as operator
import workspace.MATH.workflows.round_16.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
...


37


**1998**

**1999**


**2000**

**2001**

**2002**

**2003**

**2004**

**2005**


**2006**

**2007**

**2008**

**2009**

**2010**

**2011**


**2012**

**2013**

**2014**

**2015**

**2016**


**2017**

**2018**

**2019**

**2020**

**2021**

**2022**


**2023**

**2024**

**2025**

**2026**

**2027**

**2028**


**2029**

**2030**

**2031**

**2032**

**2033**

**2034**


**2035**

**2036**

**2037**

**2038**

**2039**


**2040**

**2041**

**2042**

**2043**

**2044**

**2045**


**2046**

**2047**

**2048**

**2049**

**2050**

**2051**


D THE ERRORS IN MERMAID CODE AND THE FREQUENCY


In this section, we summarize the retry error reasons observed during the four benchamrk and earch
benchmark will evolve 20-round. We detect five types of errors, as listed in A.2. Their frequencies
are shown in the following table.


Table 5: Error Frequency


**Error Type** **W1** **W2** **W3** **W4** **W5**
**Violation counts over 20** _×_ **4 rounds** 0 4 0 2 3


Here are some examples of incorrectly generated Mermaid workflows. We have rendered the
workflows as images for clearer illustration.


D.1 GENERATE A NODE THAT CANNOT CONNECT PROPERLY TO THE INTERFACE


This will be detected by W2 (No isolated nodes are allowed).


Figure 9: The right node isn’t connected to the Problem Interface node.


D.2 CREATE NEW TYPE OF NODE


This will be detected by W4 (Do not create new types).


Figure 10: The right node creates a new type called Verify, which is not allowed.


D.3 ONLY ONE NODE TO INPUT TO ENSEMBLE NODE


This will be detected by W5 (Ensemble nodes must have at least two inputs).


38


**2052**

**2053**


**2054**

**2055**

**2056**

**2057**

**2058**

**2059**


**2060**

**2061**

**2062**

**2063**

**2064**

**2065**


**2066**

**2067**

**2068**

**2069**

**2070**


**2071**

**2072**

**2073**

**2074**

**2075**

**2076**


**2077**

**2078**

**2079**

**2080**

**2081**

**2082**


**2083**

**2084**

**2085**

**2086**

**2087**

**2088**


**2089**

**2090**

**2091**

**2092**

**2093**


**2094**

**2095**

**2096**

**2097**

**2098**

**2099**


**2100**

**2101**

**2102**

**2103**

**2104**

**2105**


Figure 11: The ensemble node only takes one input, which is not a reasonable evolutionary process.


E FUTURE WORK


Using Mermaid can lead to better and more stable update steps, but in its current form MermaidFlow
lacks certain representational capabilities, such as expressing if-conditions or for-loops. We can
address this by adding more comprehensive node types to support such constructs, or by fine-tuning a
model specifically trained to generate valid Mermaid scripts. With a Mermaid checker or compiler, we
can more easily build training datasets or use them as sources of reward signals during post-training.


Another necessary step is to create a rule-based Mermaid-to-Python converter. In the current version,
the Mermaid-to-Python translation relies on an LLM, which can lead to issues similar to those seen in
direct Python script generation. Because Mermaid has a simple syntax, it should be straightforward
to perform reliable transformations into valid Python LangGraph scripts. This step will significantly
improve the robustness of MermaidFlow.


Another interesting direction worth exploring is extending the current task setting into a retrievalbased paradigm. Specifically, after generating a Mermaid graph, it can be stored directly in its graph
form; when encountering a new task with a structurally similar requirement, the system can retrieve
relevant workflows as references. Such a mechanism would enable efficient storage and reuse of a
collection of verified workflows, allowing them to be retrieved when needed and thereby providing
additional performance gains.


F LLMS USAGE


In the preparation of this paper, LLMs are utilized solely for drafting and proofreading purposes. All
content produced by LLMs is thoroughly reviewed and edited by the authors. The authors take full
responsibility for the final content of the paper.


39