# Rethinking and Benchmarking Large Language Models for Graph Reasoning

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Large Language Models (LLMs) for Graph Reasoning have been extensively studied over the past two years, involving enabling LLMs to understand graph structures and reason on graphs to solve various graph problems, with graph algorithm problems being the most prevalent. 
   Recent studies underscore the potential of LLMs in handling graph reasoning tasks, but their performance is underwhelming. 
   In this work, we point out issues with existing methods and benchmarks, and rethink the direction that LLMs for graph reasoning should strive toward.
   We find that base models, e.g., GPT-4o-mini, are largely underestimated due to improper reasoning focus. Base models with reasoning focus redirected from replicating graph algorithms to designing them can easily solve most graph reasoning tasks in existing benchmarks.
   To truly evaluate the graph reasoning capabilities of LLMs, we construct a more challenging GraphAlgorithm benchmark, comprising 239 different graph problems and 3,041 test instances collected from 4 competition platforms.
   Finally, we introduce a simple and strong baseline Simple-\textbf{R}easoning-\textbf{T}hen-\textbf{C}oding (Simple-RTC)—which guides LLMs to design graph algorithms first and then code to address graph reasoning tasks. Simple-RTC achieves near-perfect accuracy on existing benchmarks and significantly outperforms GPT-4o-mini and all prior methods on the GraphAlgorithm benchmark. This strong baseline encourages further advancements in LLMs for Graph Reasoning in the future.
   Code is available at \href{https://anonymous.4open.science/r/Simple-RTC-B58D}{https://anonymous.4open.science/r/Simple-RTC-B58D}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper revisits how large language models perform on graph reasoning tasks and argues that existing methods and benchmarks underestimate their potential. It identifies key issues—language-based methods struggle with repetitive reasoning paths, while code-augmented methods overly rely on APIs rather than true reasoning. To address these problems, the authors introduce a new benchmark, GraphAlgorithm, comprising real-world graph competition problems, and propose a simple yet effective baseline Simple-RTC. By decoupling reasoning and coding—first designing algorithms, then implementing them—Simple-RTC achieves near-perfect accuracy on prior benchmarks and significantly outperforms other methods on the new one. The results show that shifting reasoning focus from “replicating algorithms” to “designing algorithms” enables strong generalization across graph tasks.

### Strengths
S1. The new GraphAlgorithm benchmark sourced from competitive programming problems is comprehensive, diverse, and realistic, significantly advancing evaluation rigor and generalization testing for graph reasoning.

S2. The proposed Simple-RTC method is elegant and empirically strong, achieving large performance gains across multiple benchmarks and especially on unseen, challenging graph tasks.

### Weaknesses
W1. Although the new benchmark is large and diverse, it lacks a fine-grained categorization of graph problems (e.g., traversal, optimization, dynamic programming), which limits more detailed analysis of where reasoning improvements occur.

W2. The paper briefly mentions a “reuse” mechanism in Simple-RTC, but does not specify how such reuse is implemented, detected, or evaluated.

### Questions
Q1. How do you ensure that problems in the GraphAlgorithm benchmark are unseen during training of the used LLMs, considering that many competition problems are publicly available online?

Q2. Could you introduce a more fine-grained categorization of graph problems to analyze which categories benefit most from Simple-RTC, and provide experimental results for each category to better reveal the strengths and weaknesses of the approach?

Q3. The paper mentions that Simple-RTC can “reuse previously generated code” for repeated graph tasks—could you elaborate on how this reuse is implemented?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper critiques existing LLM-based graph reasoning methods and benchmarks, proposing a new benchmark (GraphAlgorithm) and a baseline method (Simple-RTC). Simple-RTC decouples algorithm design (via LLM reasoning) from implementation (via LLM coding), showing strong performance. However, its core idea—separating reasoning and coding—is a standard CS practice, which is not novel.

### Strengths
1) The paper provides a thorough and largely justified critique of the current landscape, highlighting real limitations in both existing methods and benchmarks.

2) The Simple-RTC pipeline, while arguably simple, demonstrates strong performance across a wide range of tasks, effectively serving as a  state-of-the-art baseline.

### Weaknesses
1) Limited Novelty in Method: The primary weakness is the lack of novelty in the proposed Simple-RTC method. It is essentially a smart prompting strategy that leverages the emergent capabilities of very large, general-purpose LLMs. The "method" itself has minimal technical depth. The four-step pipeline (Formatting, Extracting, Reasoning, Coding) is a sensible engineering workflow, but it is not a scientific innovation on par with new model architectures or training techniques.

2) Overstated Claims: The paper's language, such as "re-thinking the direction," overstates its conceptual contribution. The idea of designing algorithms before implementing them is normal practice and not new.

3) Scalability and Cost: The paper briefly mentions efficiency but does not adequately address the significant inference cost and latency introduced by sequentially querying two large LLMs (one of which is a slow "reasoning" model like DeepSeek-R1). This limits the practical applicability of the proposed approach.

4) Benchmark Baseline Context: As mentioned, the performance on the new GraphAlgorithm benchmark lacks context. Without a non-LLM baseline, it's hard to gauge the absolute achievement of 35.1% accuracy.

### Questions
1) Beyond the pipelining of two off-the-shelf LLMs, what is the specific novel technical contribution of the Simple-RTC method itself? How does it differ fundamentally from the established software engineering practice of writing pseudocode before code?

2) The performance seems heavily dependent on the choice of the external "reasoning" LLM (e.g., DeepSeek-R1). To what extent are the reported gains attributable to the Simple-RTC framework versus simply using a more powerful, general-reasoning base model? Could a version of GPT-4o-mini prompted to "think step-by-step" and then code in a single step be a competitive baseline?

3) What is the total API cost and average latency per problem for Simple-RTC on the GraphAlgorithm benchmark? How does this compare to a single-call baseline, making the method's practical feasibility clear?

4) Does the "Formatting" step require manual, problem-type-specific prompt engineering for new tasks, or is it truly generalizable? This impacts the method's scalability to unseen problem types.

### Soundness
1

### Presentation
2

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
Authors argue that current LLM graph-reasoning benchmarks mostly test pattern-matching rather than real reasoning.
Paper introduces GraphAlgorithm, a harder benchmark of 239 algorithmic graph problems, and a Simple-Reasoning-Then-Coding pipeline that first designs and then codes algorithms.

### Strengths
- Introduces GraphAlgorithm with 239 problems, a more realistic and harder than prior pattern-heavy sets. 

- The Simple-RTC pipeline (reason -> code) is conceptually clean, easy to reproduce, and decouples thinking from execution. 

- Shows consistent gains over base GPT-4o-mini and prior methods

- Works to explain why language-only methods struggle (iterative/backtracking) and why naive code-augmented setups may sidestep real reasoning.

### Weaknesses
- Benchmark is contest-style algorithmic problems; it’s unclear how insights can be transferred to structural tasks (molecular graphs, KGs, heterogeneous graphs).

- Many contest problems involve modest graph sizes; the paper doesn’t convincingly demonstrate robustness to very large graphs or long inputs.

- We see accuracy gains but ablations on representation choices (node index shuffling, graph description formats) and cross-domain transfer are limited.

### Questions
- What breaks first as graph size grows -> reasoning token budget, code generation reliability, or runtime? Any results with long-context inputs or compressed graph encodings?

- How sensitive is performance to node renaming/shuffling, re-ordering of edges, or alternative textual representations of the same graph?

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
4

### Summary
This study argues that LLMs are underestimated due to misplaced reasoning focus. By redirecting reasoning toward designing rather than copying algorithms, base models like GPT-4o-mini can already solve most existing benchmarks. To provide a more rigorous evaluation, the authors introduce GraphAlgorithm, a benchmark of 239 graph problems and 3,041 test instances from competition platforms. They also propose a strong baseline, Simple-Reasoning-Then-Coding (Simple-RTC), which first guides LLMs to design algorithms and then implement them. Simple-RTC achieves near-perfect results on existing benchmarks and significantly outperforms prior methods on GraphAlgorithm, paving the way for future progress in LLM-based graph reasoning.

### Strengths
- Simple prompting strategy can outperform prior complex methods highlights the importance of task formulation over model architecture.
- Achieving near-perfect accuracy on old benchmarks and clear improvements on the new one suggests the claims are well-supported experimentally.

### Weaknesses
- Structural or semantic graph reasoning may be underrepresented (e.g., subgraph matching, reasoning over KG triples, causal graphs). The benchmark may therefore not generalize to real-world graph reasoning tasks beyond algorithmic problem solving.
- The Simple-RTC success may rely heavily on carefully crafted prompts rather than inherent reasoning improvement, raising questions about reproducibility and generality.
- Limited novelty: RTC is conceptually similar to existing Plan-and-Solve or CoT-to-Code paradigms. The paper may need to clarify how Simple-RTC is substantively different.
- Missing related work: 
	- CodeGraph: Enhancing Graph Reasoning of LLMs with Code (Cai et al., 2024). This work addresses how LLMs struggle on basic graph algorithm problems when graphs are converted to text descriptions, and proposes encoding the problem as code (i.e., generate a program and execute it) to improve reasoning.
	- Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning (Chen et al., NeurIPS 2023). Not graph-specific, but key idea: LLMs generate executable code as reasoning steps, improving accuracy on tasks requiring algorithmic structure. One of the base works for “reason → code → execute” pipelines.

### Questions
- Can the authors elaborate on the prompt engineering process? I have already seen the prompts in the appendix.

Suggestions
- Appendix C.4 GRAPHINSTRUCT is empty

### Soundness
3

### Presentation
3

### Contribution
2
