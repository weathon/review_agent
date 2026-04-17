# Can LLMs Reason Structurally? An Evaluation via the Lens of Data Structures

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
As large language models (LLMs) take on increasingly complex tasks, understanding their algorithmic reasoning abilities has become essential. However, existing evaluations focus on distinct and isolated tasks. We propose a unified diagnostic lens: structural reasoning—understanding and manipulating relationships like order, hierarchy, and connectivity. We introduce DSR-Bench, the first benchmark to systematically evaluate LLM structural reasoning through canonical data structures, which serve as interpretable, algorithmically meaningful abstractions. DSR-Bench spans 20 data structures, 35 operations, and 4,140 synthetically generated problem instances with minimal contamination. The benchmark’s hierarchical design pinpoints specific failure modes, while its fully automated evaluation ensures objective and consistent assessment. Benchmarking ten state-of-the-art LLMs reveals critical limitations: the top-performing model scores only 0.498 out of 1 on challenging instances. Three additional evaluation suites reveal further weaknesses: models perform poorly on spatial data and natural language scenarios, and fail to reason over their own generated code. DSR-Bench offers a principled diagnostic tool for structural reasoning, helping expose reasoning bottlenecks and guide the development of more capable and reliable LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper, Can LLMs Reason Structurally? An Evaluation via the Lens of Data Structures, introduces DSR-Bench, the first systematic benchmark to evaluate large language models’ (LLMs) structural reasoning—their ability to understand and manipulate relationships like order, hierarchy, and connectivity through canonical data structures. Covering 20 data structures, 35 operations, and 4,140 problem instances, DSR-Bench provides interpretable, contamination-resistant, and fully automated evaluation. The benchmark spans five suites (main, challenge, spatial, natural, and code) to test generalization across abstract, multimodal, and language-grounded settings. Results on ten SOTA models, including GPT-5, Claude 3.7, Gemini 2.5 Pro, DeepSeek-R1, and Llama 3.3, reveal substantial weaknesses—top scores plateau around 0.498 on complex tasks. The findings expose persistent failures in multi-attribute, multi-hop, spatial, and language-based structural reasoning, even for frontier reasoning models

### Strengths
The paper offers a comprehensive, principled diagnostic framework bridging algorithmic and structural reasoning, which is both interpretable and reproducible. It evaluates across multiple reasoning modalities, from formal operations to natural language and code-based tasks, achieving impressive breadth and methodological rigor. The authors benchmark both closed- and open-source LLMs with rich ablations (prompting strategies, spatial dimensionality, data distributions), offering deep insights into model limitations and prompting effects. The hierarchical design allows fine-grained localization of reasoning failures, making DSR-Bench a valuable tool for community-wide evaluation and progress tracking.

### Weaknesses
Despite strong contributions, the benchmark’s model coverage remains limited—only ten models are tested, lacking more varied model sizes and major closed-source reasoning systems, which constrains external validity. The evaluation focuses heavily on synthetic, symbolic reasoning and may underrepresent noisy or multimodal real-world inputs. Moreover, while the work claims to reveal algorithmic limitations, the analysis is largely empirical, without deep mechanistic investigation into why reasoning breaks down. The code suite section exposes that models fail to reason over their own code but does not propose mitigation pathways. Finally, performance saturation at ~0.5 suggests the tasks might not scale linearly in difficulty, risking benchmark ceiling/floor effects.

### Questions
1. How would results change if more diverse model scales (e.g., 7B–400B) and closed-source reasoning models (e.g., Gemini Ultra, Claude 3 Opus) were included?

2. Can the authors analyze internal reasoning traces or attention mechanisms to identify why models fail specific structural operations (e.g., multi-hop tree balancing)?

3. Have the authors considered multimodal structural reasoning (e.g., visual graphs, tables) as a next step to bridge symbolic and embodied reasoning?

4. How sensitive is DSR-Bench performance to prompt phrasing and temperature—does structural reasoning remain consistent under naturalistic, noisy instructions?

5. Could future work explore training-time regularizers or inductive biases explicitly targeting structural reasoning, informed by DSR-Bench diagnostics?

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
4

### Summary
The authors propose an evaluation suite on reasoning of structural data, DSR-bench, to benchmark and understand whether LLMs can reason about and manipulate structural relations. Experiment results show that LLMs uniformly fail on the benchmark, especially on tasks which require spatial or natural language reasoning. This means the problem needs to be better addressed to further improve LLMs' capabilities.

### Strengths
1. The paper is well written and very easy to understand. The presentation is clear and systematic in supporting the authors' arguments and showing experiment results.

2. The experiment is carefully designed and executed. The design of the experiments are comprehensive to provide insights on understanding LLMs' capabilities in terms of structural data.

3. The investigated question is important. LLMs need to understand structural data to better serve as code generator/interpreters.

### Weaknesses
1. Novelty seems to be a bit limited. As the authors discuss in section 2, there are quite some existing works on testing LLMs on algorithmic execution capabilities. The authors argue that their work is distinct in that they use a holistic approach using data structures (and presumably more comprehensive in the sense of algorithm tasks). However, it feels to me that the current benchmark looks like an ensemble of previously investigated questions, which means that the novelty is a bit limited. Moreover, previous works have also showed that LLMs are not so good in code execution, so can the authors clarify their novelty here? Also see [1, 2, 3, inter alia] for more code execution and related natural language execution tasks.

2. In Figure 1, is the illustration of DAWG the same as the the logic of the question presented to the LLMs? To my understanding the prototypical DAWG scans and store characters from left to right. It seems that the DAWG illustrated stores from right to left. Is it a typo? If the questions presented are in the same order, is it possible that LLMs did not fully understand the question as it is not the most conventional one and instructions are not clear enough?

3. Why are LLMs particularly bad on spatial data? Do multi-modal models perform better compared to traditional LLMs on it?

[1] La Malfa, E., Weinhuber, C., Torre, O., Lin, F., Marro, S., Cohn, A., ... & Wooldridge, M. (2024). Code simulation challenges for large language models. arXiv preprint arXiv:2401.09074.

[2] La Malfa, E., Weinhuber, C., Torre, O., Lin, F., Huang, X. A., Marro, S., ... & Wooldridge, M. (2025). Code Simulation as a Proxy for High-order Tasks in Large Language Models. arXiv preprint arXiv:2502.03568.

[3] Liu, C., Dylan Zhang, S., Ibrahimzada, A. R., & Jabbarvand, R. (2024). Codemind: A framework to challenge large language models for code reasoning. arXiv e-prints, arXiv-2402.

### Questions
The authors are welcome to address the weaknesses above.

### Soundness
3

### Presentation
3

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
The paper proposes **DSR-Bench**, a synthetic benchmark for “structural reasoning” framed via classic data structures (20 structures, 35 operations) with multiple suites (main, challenge, spatial, natural, code). The evaluation is automatic via schema-checked outputs; the authors report broad gaps across modern LLMs and several prompting/tool-use variants.

### Strengths
1. Clear writing; figures/tables are easy to read.
2. A coherent, unified design principle with code to synthesize more tasks—useful for the community.
3. Large experimental sweep with many variants; results are carefully organized.

### Weaknesses
1. **Positioning/novelty**. Many insights (multi-hop difficulty, distribution shifts, NL brittleness, limited benefit from self-written code) are already familiar. The paper’s claim about “LLMs reasoning structurally” is not convincingly isolated because representation and execution are entangled throughout. ( See Weakness#2 )

2. **Representation vs. execution not disentangled.** Each task has (i) a representation component—does the model correctly parse the structure as presented?—and (ii) an execution component—can it carry out the specified operations. The study heavily emphasizes execution and under-specifies representation:

    2.1. How exactly are high-dimensional objects (2D grids, graphs, spatial point sets) encoded? JSON? code-like syntax? natural language? chain of insertions?

    2.2. Without concrete prompt schemas/examples per family, it’s hard to tell if performance drops arise from poor parsing/grounding vs. failed reasoning on operations.

3. **Execution ablations are shallow.** Now fixing the representation to 2.1 ( and particularlly for tasks like BST/Graph ), if operations are given as pseudo-code-ish directives, do results change when you switch to (a) Python-style code, (b) pure NL imperatives ( in DSR-Bench-natural you already tested when both representation and operations are switched to NL ), or (c) step-checked traces? These swaps would reveal whether failures stem from programming formality vs. actual manipulation of state.

4. **Context length realism.** For harder instances, solutions often grow quickly (exponentially). It’s unclear whether the runs hit or neared context limits and how accuracy scales with available tokens. A context-budget ablation (and confirmation that max tokens were utilized on hard tasks) is missing.

### Questions
1. **Priority Queue (multi-attribute) prompt.** Please include a concrete example that shows how multiple attributes are encoded in the input and in the operation spec. Right now most examples shown are basic Queue; the multi-attribute claim (e.g., around L319) is important but not inspectable.


2. **Aggregation in Table 2.** Is this an average over all DSR-Bench-main tasks? If so, how does GPT-4.1+CoT reach ~94% while GPT-5 is ~79%? And how do these aggregates reconcile with hard-structure results (e.g., DAWG ~21% in Table 23)?


3. **Encoding of high-dimensional/spatial data.** Provide exact prompt formats (few full examples) for 2D, graphs, and spatial point sets.


4. **Distribution shift & CoT (Table 6).** Is GPT-4.1 reported without CoT there? Does adding CoT improve robustness under non-uniform spatial distributions or dimensionality shifts?

### Soundness
3

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
This paper introduces a novel benchmark to evaluate LLMs’ structural reasoning capabilities through tasks grounded in data structures. The benchmark spans a broad spectrum of structures, including sequences, temporal orderings, key–value mappings, tree-like hierarchies, connectivity and group membership, as well as composite structures. The reasoning operations cover construction, inspection, manipulation, and their compositions. The authors stratify task difficulty by input length. They curate five suites, including main, challenge, natural language, code, and spatial, to assess LLMs’ structural reasoning in real-world settings. The experimental results show that current LLMs still struggle in structural reasoning.

### Strengths
- The problem is well-defined.
- The experimental evaluation is thorough and solid.
- The findings are informative, highlighting critical limitations of current LLMs.

### Weaknesses
- Lack of statistical analysis of the dataset (e.g., how many instances are used in the evaluation?).
- It would be beneficial to include mechanistic interpretability analysis. See [1] for reference, which provides deeper insights even with simpler data structures.
- Missing SFT and RL (e.g., GRPO) experiments. If models are trained on these tasks, how much does performance improve?
- Even human reasoners may struggle with parts of this benchmark, since deterministic program-based tasks are not naturally handled by probabilistic models. A potentially helpful approach is tool use with coding (enabling models to generate code and invoke external interpreters), combined with sufficient in-context knowledge (e.g., definitions and demonstrations of relevant algorithms and data structures). It is recommended to adopt more comprehensive tool-use baselines across all suites, not only the code suite.

*[1] Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization*

### Questions
What is the difference between “Stepwise” and “Zero-Shot CoT,” given that both prompt the model to reason step by step without demonstrations? Are these merely different prompt wordings—e.g., “include a steps field in your output” versus “let’s think step by step”—or do they reflect distinct prompting strategies?

### Soundness
3

### Presentation
3

### Contribution
3
