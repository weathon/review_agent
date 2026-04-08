## Human Reviewer 1

### Summary
This paper introduces BIRD-INTERACT, a new benchmark for Text-to-SQL designed to address the limitations of existing static, single-turn, and read-only datasets. It provides a dynamic, multi-turn evaluation framework where models must handle ambiguity, execution errors, and evolving user goals. The benchmark's core components include a task suite covering the full CRUD spectrum, an interactive environment with a knowledge base, and a novel function-driven user simulator that allows models to ask clarification questions without human supervision. The paper introduces two evaluation settings: c-Interact (a structured, protocol-guided conversation) and a-Interact (an open-ended agentic setting). Empirical results demonstrate the benchmark's significant difficulty, with top-tier models like GPT-5 achieving success rates below 18%, highlighting a substantial gap between current LLM capabilities and the demands of real-world dynamic database tasks.

### Strengths
- **S1. Addresses a Critical Research Gap:** The paper correctly identifies a major limitation in existing Text-to-SQL research. The field has largely focused on single-turn SELECT query generation, which does not reflect the ambiguous, iterative, and stateful nature of real-world data analysis. This benchmark shifts the focus to a more realistic and challenging interactive paradigm.
- **S2. Robust User Simulator:** The two-stage, function-driven user simulator is a significant contribution. By first mapping a model's question to a symbolic action (AMB, LOC, UNA) before generating a response, it avoids the common pitfalls of naive LLM-based simulators, such as ground-truth leakage and un-controlled-generation.
- **S3. Comprehensive and Realistic Task Design:** The benchmark's design is high-quality. It expands the task scope beyond SELECT to include the full CRUD spectrum (DML/DDL), which is a crucial step towards evaluating real-world database assistants. It also introduces state dependency, where follow-up sub-tasks depend on the state changes (e.g., a newly created table) from a previous sub-task. Finally, the principled injection of ambiguities (in queries, knowledge, and the environment) makes interaction a necessity for success.

### Weaknesses
- **W1. Unclear Knowledge Ambiguity Taxonomy:** The distinction between "one-shot knowledge ambiguity" and "knowledge chain breaking" in Section 3.2 is unconvincing. The paper's own example for "knowledge chain breaking" is to mask the intermediate node "AVS". However, masking "AVS" is functionally identical to removing it as a "one-shot" ambiguity (defined as removing an "isolated knowledge entry"). Both actions result in the "AVS" node being missing. This conceptual overlap makes the taxonomy feel redundant and fails to delineate a meaningful difference between the two ambiguity types.
- **W2. Questionable Premise of "Implementation-Level Ambiguity":** The paper categorizes user under-specification as "implementation-level ambiguity" (Section 3.2 and Appendix H.4), which seems to be a conceptual flaw. For example, the query "Show recent purchases" is presented as ambiguous, with the "clarified" version being "Show recent purchases sorted by time". A query that is merely less specific is not inherently ambiguous, as any result set of recent purchases, regardless of sort order, would be a correct answer. Similarly, "Show average score" is treated as ambiguous, with the "clarified" version being "Show average score in 2 decimal". The un-rounded, full-precision answer is also correct. This design choice seems to conflate ambiguity with a lack of specific formatting instructions, risking unfair penalization of models that make reasonable default assumptions.

### Questions
- **Q1. On "Implementation-Level Ambiguity":** Could the authors elaborate on the reasoning for classifying under-specification as "ambiguity"? Using the example from Appendix H.4, "Show recent purchases" is a valid query. Why is a model penalized for not clarifying a specific sort order (ORDER BY purchase_time DESC) that was never requested? Does this not create a benchmark that tests for a specific, "over-specified" ground truth rather than functional correctness?
- **Q2. On "Decimal Ambiguity":** Following up on Q1, why is a query like "Show average score" considered ambiguous? Is the full-precision answer (e.g., 85.12345) considered incorrect? It seems that requesting rounding (e.g., ROUND(AVG(score), 2)) is an additional formatting instruction, and its absence doesn't make the original query ambiguous.
- **Q3. On "Knowledge Ambiguity" Distinction:** Could you please clarify the distinction between "one-shot knowledge ambiguity" and "knowledge chain breaking"? The paper's example for chain breaking is to mask the 'AVS' node. How is this functionally different from removing the 'AVS' node under the "one-shot" definition? Both actions seem to result in the same missing node. Could you provide an example where these two ambiguity types are mutually exclusive and force different system behaviors?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes BIRD-INTERACT, a benchmark for evaluating Text-to-SQL models in realistic, multi-turn interaction settings. It integrates a hierarchical knowledge base, metadata, and an autonomous user simulator to support dynamic clarification and error recovery. The benchmark includes two evaluation modes and full CRUD tasks with injected ambiguities, revealing a significant gap between current LLM capabilities and real-world interactive SQL reasoning.

### Strengths
1. The work re-imagines Text-to-SQL evaluation by modeling a full, high-fidelity human-AI collaborative loop, moving beyond the static constraints of previous datasets. The combination of CRUD support, HKB reasoning, and dynamic interaction is unique.
2. The rigorous task annotation (e.g., SQL snippets as clarification sources), the two-stage simulator with proven robustness on USERSIM-GUARD, and the comprehensive experimental methodology attest to the high quality of the benchmark construction.
3. The paper is clearly written, and the analysis provides a deep, multi-faceted understanding of why current models struggle with the interactive paradigm.

### Weaknesses
1. Limitation on Simulator Creativity: While the function-driven approach ensures robustness against ground-truth leakage, its reliance on pre-annotated ambiguities and structured retrieval for `LOC()` actions fundamentally limits the model's ability to handle truly novel, unforeseen ambiguities that a real human might introduce. The simulator is robust but sacrifices some degree of open-ended human creativity. Future research should strive to achieve both robustness and broader generalization in the simulation.
2. High Barrier to Entry: The main results rely heavily on expensive, closed-source models (e.g., GPT-5, Claude-Sonnet-4, Gemini-2.5-Pro). The high average cost per task (up to $0.67 for Claude-Sonnet-3.7) creates a significant financial barrier for broad academic community adoption, particularly for large-scale training and iterative development, even with the smaller LITE set. A more prominent emphasis on developing and benchmarking cost-effective, open-source models capable of effective interaction is necessary.

### Questions
1. Generalization of the User Simulator (LOC() action): The success of the `LOC()` action depends on retrieving relevant SQL fragments via AST matching. Have the authors considered how their two-stage system would handle an out-of-distribution clarification request where the intent is reasonable, but the answer is not easily localizable to a specific SQL fragment (e.g., asking for an abstract policy not represented directly in the ground-truth SQL)? This would better test the generalizability vs. ground-truth grounding trade-off.
2. Balance of Budget and Efficiency (for a-Interact): The analysis shows models favor costly `submit` and `ask` actions over efficient exploration tools. In designing the action costs, did the authors consider adaptive or dynamic cost schedules? For example, penalizing models that consecutively choose low-value `execute` actions might more strongly incentivize strategic resource exploration and planning.
3. DM Task Distribution: The task suite includes the full CRUD spectrum (DM tasks). To better assess model capabilities in this crucial non-SELECT domain, could the authors provide a breakdown of the distribution of specific DM operations (`INSERT`, `UPDATE`, `DELETE`, `ALTER TABLE`) within the benchmark, along with the performance metrics for models on each of these types? This is vital for evaluating model readiness for operational database tasks.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces BIRD-INTERACT, a new benchmark designed to evaluate large language models (LLMs) for *interactive* text-to-SQL tasks. Unlike prior static, single-turn datasets (e.g., Spider, BIRD), BIRD-INTERACT simulates realistic multi-turn interactions between a system and a user simulator, capturing the iterative clarification, debugging, and follow-up behavior that occurs in real-world database applications.

### Strengths
-  The paper addresses an important gap between single-turn text-to-SQL evaluation and real interactive database usage, providing a benchmark with clear practical relevance.
- The inclusion of knowledge bases, metadata, and executable DBs offers a realistic simulation environment; the two-stage user simulator design is technically sound and well-motivated.
- The paper situates its contribution well within prior work (e.g., COSQL, LEARN-TO-CLARIFY, LIVESQLBENCH) and justifies why static multi-turn datasets are insufficient.

### Weaknesses
- Limited discussion of annotation cost and human validation. While expert annotators are mentioned, clearer statistics on time, inter-annotator reliability per ambiguity type, or annotation guidelines would strengthen credibility. 
- The experimental setup reveals that each model was executed **only once** because of the *high cost of commercial API calls*, even though the authors used deterministic decoding (temperature=0, top_p=1) to justify single-run reproducibility. While this design ensures consistency, it also raises concerns about the **statistical robustness and variance** of results—especially given the benchmark’s complexity and multi-turn stochastic nature. Moreover, the reported per-model expenses for BIRD-INTERACT-FULL and -LITE (Table 2 and Table 10) suggest that large-scale evaluation is economically prohibitive, potentially hindering broader adoption and replication by the community

### Questions
-  A key question is whether performance on *BIRD-INTERACT-LITE* reliably predicts results on *BIRD-INTERACT-FULL*. Quantifying this relationship—such as through rank correlation or regression analysis—could clarify whether the Lite version serves as an efficient proxy for large-scale evaluation, enabling faster model iteration without sacrificing benchmarking fidelity.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
BIRD-INTERACT proposes a new benchmark for text-to-SQL that captures the complexity of real-world, multi-turn database interactions. It introduces a dynamic environment with a novel function-driven user simulator to handle ambiguous queries and provide realistic feedback without human supervision. The benchmark features two evaluation settings—a structured c-Interact (conversational) and an open-ended a-Interact (agentic) mode—and includes tasks covering the full spectrum of CRUD operations. Experiments show that even state-of-the-art models struggle, highlighting that effective interaction, not just SQL generation, is a critical challenge.

### Strengths
1. First, the paper makes text-to-SQL evaluation much more realistic. Instead of simple, one-shot questions, it creates a messy, back-and-forth conversation that forces models to ask clarifying questions and handle ambiguity, which is how people actually interact with databases. This is a huge and necessary step forward for the field.
2. Second, the way they built the user simulator is incredibly smart. They didn't just use a standard chatbot that might accidentally leak the answer. They designed a clever two-stage system to make the simulator's responses controlled and reliable, which makes the whole benchmark much more trustworthy and fair.
3. Finally, the results are a major wake-up call. [cite_start]The paper clearly shows that even the best models today, like GPT-5, fail spectacularly at these tasks, with success rates as low as 8.67%[cite: 2935]. This proves there's a massive gap between just generating code and being able to strategically interact to solve a real problem, which is a super important insight.

### Weaknesses
First, the user simulator, while cleverly designed for consistency, is probably too perfect. Real users are messy—they get confused, change their minds, or give bad information. By making the simulator so rational and predictable, the benchmark might be testing models against an idealized user, not a real one, which could hide how well they'd perform in the wild.

Second, the pass/fail success metric doesn't tell you why a model failed. When an agent gets a task wrong, it's hard to know if the issue was a bad plan, poor communication, or just a simple syntax error in the final SQL. The evaluation shows that it failed, but not where the breakdown happened, which makes it less useful for diagnosing the core problem.

Finally, the rules of the agent game might be pushing models into unnatural strategies. For example, making it computationally "expensive" to ask the user a question could discourage communication and encourage a brute-force, trial-and-error approach. It’s possible the models are just learning how to beat the benchmark’s specific point system rather than learning how to solve database problems effectively.

### Questions
Refer to Weakness

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4