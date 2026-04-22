# Sharing State Between Prompts and Programs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 2

## Abstract
The rise of large language models (LLMs) has introduced a new type of programming: natural language programming. 
Users write prompts, which are instructions in natural language, to direct LLMs to perform tasks such as natural language processing, code generation, reasoning, etc.

An emerging area of research enables interoperability between prompts and programs.
We present a novel programming abstraction, shared program state, that removes the manual work required to enable interoperability between prompts and program states.
With shared program state, programmers can write prompts that directly access program variables, compute with program objects, and implement control flow in the program.
We present a schema for specifying natural function interfaces that extend programming systems to support programs with prompts and leverage this schema to specify shared program state as a natural function interface.

We implement shared program state in the Nightjar programming system.
Nightjar enables programmers to write Python programs containing prompts that share the Python program state.
We show that Nightjar programs achieve comparable or higher task accuracy than manually written implementations (+4-19\%), while decreasing the lines of code by 39.6\% on average.
The tradeoff is that Nightjar may incur runtime overhead (0.4-4.3x manual implementations).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces **Shared Program State (SPS)**, a novel abstraction allowing natural-language code (prompts) and traditional programming languages (like Python) to **share variables, objects and control flow**. It defines a schema called the Natural Function Interface (NFI) that models this sharing via operations like `Lookup`, `Assign`, `Goto` etc. The authors implement this in a system called NIGHTJAR, which co-executes Python and natural code. Empirical evaluation (on a 25-program benchmark) shows comparable accuracy with much shorter code (23-82% fewer lines), albeit with higher runtime overhead.

**Key Contributions**
1. Definition of the Shared Program State abstraction and its semantics for mixing natural and formal code.
2. Formalization of the Natural Function Interface (NFI) schema for the interaction.
3. Implementation (NIGHTJAR) demonstrating how natural language code can manipulate program state in a host language.
4. Empirical study (SPSBench) showing productivity gains (code conciseness) and feasibility of SPS, with noted runtime cost.

### Strengths
* The paper introduces a novel abstraction; the idea that prompts (natural-language code) and traditional programming languages can share program state; variables, heap objects, control flow. This moves beyond prior work that treated prompts as black-box tool calls or code generation endpoints. For example, [A] considers prompts as programs but does not formalize shared variable/state semantics between NL and formal code). [B] supports collaborative prompt engineering, but again focuses on prompt sharing/management rather than state-sharing semantics between NL and host code.
* The combination of techniques is also creative: using an LLM as a "natural interpreter", defining an interface (Natural Function Interface) that models state-manipulation effects (assign, lookup, goto) and handlers in a host language. Prior works on NL to code generation (e.g., prompt-based learning, chain-of-thought) separate the code generation from program execution; here the "code" (NL) directly manipulates execution state. For instance, the "Program of Thoughts" [C] work uses LMs to generate a program which is then executed externally (i.e., separate stages). In contrast, this paper merges the NL/code boundary via shared state. That creative combination marks a strong originality.
* The clarity of the interface model (Natural Function Interface), and the way the paper formalises values/effects/handlers, suggests technical rigor. This goes beyond more HCI-oriented prompt papers (e.g., CoPrompt [B]) which focus on usability rather than formal semantics.

```
[A] Prompts Are Programs Too! Understanding How Developers Build Software Containing Prompts, FSE 2025
[B] CoPrompt: Supporting Prompt Sharing and Referring in Collaborative Natural Language Programming, CHI 2024
[C] Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks, TMLR 2023
```

### Weaknesses
* The empirical evaluation uses a 25-program suite (e.g., SPSBench) that demonstrates the abstraction in a limited number of tasks. It is unclear how well the approach scales to larger, more complex programs, especially ones with deeply nested state, concurrency, or rich object graphs.
* The implementation incurs a substantial runtime penalty (11.7–15.3× slower) due to the involvement of an LLM interpreting natural code. While the paper acknowledges it, the evaluation does not fully explore how this overhead affects usability in real-world settings. Many practitioners may find this cost prohibitive.
* Allowing natural-language code to manipulate host program state (variables, heap, control flow) introduces risks: unintended side-effects, state inconsistency, security holes, debugging difficulty. The paper acknowledges safety issues but does not deeply evaluate them (e.g., what happens if the NL code assigns to a variable incorrectly, or dereferences invalid references).
* While the “shared program state” abstraction is compelling, some of its components (NL interfacing with program state, prompts-as-programs) overlap substantially with prior work. For instance, [A] already frames prompts as programs. The novelty claim would be stronger if the paper explicitly differentiates itself from these works and shows how state-sharing goes beyond "prompts as programs" and prior tool-calling frameworks.
* Although the formalism is strong, there may be a steep learning curve for typical developers: understanding effects/handlers, bridging NL syntax with host language semantics, maintaining coherence in hybrid code. The paper could provide more empirical evidence on developer usability: how quickly developers adapt to writing natural code that manipulates host state, what errors or misunderstandings occur. In contrast, works like CoPrompt [B] focus on developer workflows and collaboration in prompt engineering and report usability metrics.
```
[A] Prompts Are Programs Too! Understanding How Developers Build Software Containing Prompts, FSE 2025
[B] CoPrompt: Supporting Prompt Sharing and Referring in Collaborative Natural Language Programming, CHI 2024
```

### Questions
* Is it possible to include a larger scale benchmark with more complex stateful programs (e.g., classes, inheritance, mutable collections, parallel threads) and evaluate how the shared state abstraction performs there (both correctness and latency)?
* Is it possible to compare the performance (accuracy, conciseness, runtime) on tasks used in ANPL [A] or similar to increase external validity?
* Is it possible to provide a deeper latency cost-benefit analysis (e.g., user-perceived delay, scalability with number of state operations, caching strategies for repeated NL manipulations)?
* Is it possible to include a dedicated "robustness" evaluation: fault injection (NL commands that try invalid operations), tracking state invariants, measuring how many errors occur, how they are detected and recovered? Also are there guidelines or restrictions on NL code allowed (e.g., sandboxing, type checks, access control)?
* Is it possible to add a discussion on how debugging mixed NL+code state will be supported (logging, visualization, stepping)?
* Is it possible to deepen the literature comparison section: explicitly list prior abstractions (e.g., prompt-programming, tool-calling interfaces, NL-to-code) and highlight exactly how the shared -state model differs (e.g., direct variable reference, control flow manipulation)?
* Is it possible to include a user study (even small-scale) with software engineers or prompt-programmers: measure how well they'd adopt the shared-state model, what errors are common, what cognitive load is involved?

```
[A] ANPL: Towards Natural Programming with Interactive Decomposition, NeurIPS 2023
```

### Soundness
3

### Presentation
3

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
The authors propose to enable LLM "execution" agents to directly interact with a host language program state to facilitate easier interaction between "natural code" and formal language code (programming languages). They define a Natural Function Interface (NFI) along with its operational semantics, and show how NFIs generalise tool use/MCP to a richer interaction semantics (shared state, shared control flow).
The authors then curate a benchmark dataset with natural code to evaluate their approach in terms of lines of code (after normalisation), pass-rate, and execution time. They show a significant reduction of code size (by making much of the serialisation, reisation implicit to the paradigm) while matching the manually written implementations in a majority of cases.

### Strengths
- Generalisation of tool use to shared program state (memory, control)
- competitive results even with what seems like a naive implementation

### Weaknesses
- It is unclear how/where serialisation/data marshalling is to be implemented; it is also unclear if grammar-based sampling is compatible with the approach or if any such grammar file should be augmented to enable emission of effect tokens.
- perhaps out of scope, but the handler loop seems unintuitive: if emitted effects are side-effect free wrt the program state that is observed by the NL code, why not stage an execution plan and delay the interrupts until the last moment possible to reduce overhead?
- perhaps missing reference to literate programming (of ye olden days): Knuth, Donald Ervin. "Literate programming." The computer journal 27.2 (1984): 97-111. Not directly the same concept, but it follows the spirit of literate programming
- There is an unexplored trade-off frontier: in the paper, the comparison is the paradigm handling the boilerplate vs the user-written boilerplate. A fully FL/PL program vs a hybrid program would be another comparison of interest, but the cost normalisation is more difficult.

### Questions
- Q1: The idea of shared state requires the model to be aware of how data can be manipulated (`graph.add_edge((14,5))`, etc.) and moving data via the handler requires serialisation and conversion back to the expected data-type (similar to FFI dtypes vs the host language dtypes). Who has the burden of defining these serialisation operations, Function symbol discovery, etc.?

- Q2: While the implicit to the paradigm operations vs explicit boilerplate is explored in the paper, how does this compare to generating a single PL script that solves the task instead of switching PL/NL context?

- Q3: How does NL execution handle stochasticity? Is the temperature set to 0?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces shared programs states between formal and natural programming languages. This is formalized with the concept of Natural Function Interfaces (NFIs), through values, effects and handlers for sharing scopes, heaps, and control states. The proposed system, NIGHTJAR, implements this shared program and enables natural language code to directly read, write, and manipulate the program state of a host language such as Python. The work is supported by theoretical grounding and a new benchmark suite (SPSBench) evaluating program conciseness and performance.

### Strengths
- **Novel conceptual contribution**: The idea of exposing host-language state and control to an LLM in a principled and programmable way is novel. This moves beyond existing systems that treat LLMs as isolated components that make tool calls using programmer defined functions.
- **Strong formal framework**: The authors present a formal framework for NFIs, including variable scopes, heap references, and control state.
- **System implementation**: The NIGHTJAR system demonstrates that the abstraction is realizable and yields substantial code reduction  compared to manually writing interoperable code.

### Weaknesses
- **Limited empirical evaluation**: The evaluation on program pass rates and conciseness, while adequate as proof of concept, does not deeply explore scalability, robustness, or user experience. 
- **Benchmarks are synthetic**: SPSBench appears to consist mainly of small programs adapted from documentation examples. It would be great to have analyses on real-world user code or larger-scale applications.
- **Safety and correctness**: Though acknowledged in discussion, the implications of allowing natural code direct memory access (even through abstraction) deserve stronger treatment.

### Questions
- **Execution times may be conservative**: Rather than relying only on LLM APIs, consider adding evaluations with smaller, locally hosted LLMs. 
- **Discuss failure modes**: Consider adding some analysis of where failures occur when they do.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the idea of shared program state, essentially an LLM black box that can read and manipulate the program's internal state which can be steered somewhat by specialised syntax. The paper walks this idea through with an example, defines the proposed construct in eBNF and further explains the functionality of each component. The paper concludes with a reference implementation, nightjar, of the proposed idea and a lightweight empirical study with 25 programming examples which suggests that programmers using nightjar are about as proficient as programmers without it, while sacrificing execution time for conciseness.

### Strengths
I think the general idea of natural language programming interfaces is intriguing and has a rich history that goes well beyond the current LLM hype - indeed beyond deep learning.

### Weaknesses
I am not convinced ICLR is the best avenue to publish this work. The empirical study is at best preliminary and I am not convinced at all that programmers would like to give up control and lose out on execution time for saving a few extra lines of code - with LLM-assisted code writing and analysis tools these come essentially for free.

I suggest to greatly expand the scope of the empirical analysis, include evaluations on known public benchmarks as well as to incorporate qualitative feedback, analysis of failure modes, etc, to increase the scope of the contribution commesurate with ICLR.

### Questions
No questions at this point.

### Soundness
2

### Presentation
3

### Contribution
1
