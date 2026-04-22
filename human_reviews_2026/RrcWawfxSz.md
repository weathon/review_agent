# Ticket-Bench: A Kickoff for Multilingual and Regionalized Agent Evaluation

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Large language models (LLMs) are increasingly deployed as task-oriented agents, where success depends on their ability to generate accurate function calls under realistic, multilingual conditions. However, existing agent evaluations largely overlook cultural and linguistic diversity, often relying on monolingual or naively translated benchmarks. We introduce Ticket-Bench, a benchmark for multilingual agent evaluation in task-oriented scenarios. Ticket-Bench simulates the domain of soccer ticket purchases across six major languages-Portuguese, English, Spanish, German, Italian, and French-using localized teams, cities, and user profiles to provide a higher level of realism. We evaluate a wide range of commercial and open-source LLMs, measuring function-calling accuracy and consistency across languages. Results show that reasoning-oriented models (e.g., GPT-5, Qwen3-235B) dominate performance but still exhibit notable cross-lingual disparities. These findings underscore the need for culturally aware, multilingual benchmarks to guide the development of robust LLM agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper highlights the need for multilingual, culturally aware benchmarks for evaluating function calling. The authors introduce Ticket Bench to address this issue. Ticket-Bench features tasks in six major languages– Portuguese, English, Spanish, German, Italian, and French to evaluate LLM agent capabilities in ticket-purchasing scenarios. They build the benchmark by simulating user profiles, localized teams and cities. They compare the performance of various models across different languages on the benchmark and show that GPT-5 and Qwen3-235B are most robust.

### Strengths
1. The paper highlights the need for agentic evaluation frameworks in multilingual settings to evaluate function calling. The paper introduces a benchmark TicketBench, which simulates a real world scenario i.e purchasing soccer tickets.

2. The ideas in the paper are presented clearly along with results across multiple model families in Table 3 and cross-lingual variation in Figure 3.

### Weaknesses
While the benchmark is trying to address a real world scenario, the usefulness of the benchmark seems to be limited. The benchmark uses 17 question templates to represent distinct ticket purchasing scenarios. This limits the diversity of the instructions given to the agent and do not adequately represent the diversity of real world queries. Similarly, the agent has access to only five callable functions in the entire benchmark. 

The paper could provide additional analysis of the rate of success for each function invocation across models.

### Questions
1. How are the question templates translated into other languages? Are these manually verified by the authors to ensure coherence?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Ticket-Bench, a benchmark created to assess LLMs in task-oriented, realistic, multilingual environments. It focuses on function-calling in the domain of soccer ticket purchases across six languages (Portuguese, English, Spanish, German, Italian, and French). Each language has its own culturally appropriate teams, cities, and user profiles. Unlike previous monolingual or naively translated benchmarks, Ticket-Bench captures both linguistic and cultural diversity, providing a structured environment that evaluates models based on the correctness of their final actions. To test the ability to handle different logical constraints, like affordability, location, day of the week, or previous team performance, the authors defined 17 different ticket purchasing templates.

### Strengths
- Multilingual and culturally grounded evaluation.
- Carefully designed methodology that supports LLM-free evaluation.
- Benchmark covers six fully localized languages.

### Weaknesses
- Benchmark focuses solely on soccer ticket purchases.
- Study relies exclusively on programmatic evaluation without inspecting model reasoning quality.
- Benchmark excludes low-resource languages.
- Manuscript contains multiple typos and inconsistent capitalization (e.g., “OpenaAI” instead of “OpenAI”; double periods like “..”).

### Questions
- Do you plan to expand Ticket-Bench to include non-European and low-resource languages in future releases?
- Did you check whether differences in prompt length across languages affect accuracy?
- Could translated function names cause bias, and did you test using shared English names across languages?
- How sensitive were the results to the verbosity of tool descriptions and examples in the function schema?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Ticket-Bench is a novel evaluation benchmark for LLM Agents, focusing on addressing the prevalent problems of "English-centric" evaluation and a "lack of cultural authenticity." The benchmark simulates a "soccer ticket purchasing" task scenario. It covers six major languages (Portuguese, English, Spanish, German, Italian, and French). Unlike previous datasets that rely merely on translation, Ticket-Bench ensures "cultural authenticity" by using real, localized entities corresponding to each country in different language versions (such as team names, city names, and common user names).

### Strengths
The work involves a substantial engineering effort.

The writing is fluent.

### Weaknesses
In terms of language: Manually translating to generate queries is a traditional method, which leads to an evaluation environment that is overly "clean" and standardized. It fails to reflect the complexity of real-world user queries, such as colloquialisms, ambiguity, grammatical errors, or expressions with mixed intent. Furthermore, the six languages were all selected based on their relevance to "soccer leagues," which results in the selected languages being primarily concentrated in Europe and South America.

In terms of regionalization: The benchmark largely equates "cultural authenticity" with "entity localization" (i.e., replacing team names, city names, and personal names). This approach may fail to capture deeper cultural differences, such as local customs, habits, and cultural context. It is questionable whether the soccer ticket domain can generalize to other domains that require regionalized knowledge.

### Questions
Does the benchmark include tests for complex agentic behaviors?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Ticket-Bench, a new benchmark for evaluating LLM agents' function-calling capabilities in multilingual and regionalized scenarios. The authors argue that existing benchmarks are overly English-centric. To fix this, Ticket-Bench simulates a soccer ticket-purchasing task across six languages, crucially binding each language to realistic, local entities (e.g., Brazilian league teams for Portuguese, German league teams for German) rather than just translating text. 

The authors use a LLM-free evaluation that checks the final task state and test a wide range of LLMs. They find that reasoning-oriented models (like GPT-5 and Qwen3-235B) perform best and that performance generally scales with model size. They also find significant cross-lingual disparities, with different model families showing distinct performance patterns across the six languages, likely reflecting their training data.

### Strengths
1. Sound Motivation: The motivation to create a "regionalized" agent evaluation benchmark is practical. It addresses a gap in existing evaluations which are often English-centric and overlook local context.
2. Broad Model Coverage: The benchmark is evaluated on a wide and abundant range of LLMs, including both prominent commercial and open-source models of various scales.
3. Robust Evaluation Metric: The use of pass^k is a sound choice, offering a more stable and reliable measure of model consistency compared to a single-pass metric.

### Weaknesses
1. Limited Scope of Agent Capabilities Evaluated: The paper presents a benchmark for agent evaluation, but its scope is limited, focusing primarily on multi-step tool use rather than higher-level agentic capabilities. Specifically, the benchmark does not evaluate the critical autonomous task planning ability of agents. Instead, its queries are structured (17 templates), which pre-defines the task's logical decomposition. Also, the evaluation environment assumes successful API calls. A robust agent, however, must be able to handle API failures or dynamic environmental changes. The benchmark does not assess this crucial capacity for error detection and autonomous refinement.

2. Disconnect Between Benchmark Design and Experimental Analysis. A core weakness of this paper is the significant disconnect between its novelty of "regionalized" design (Section 3.4) and its subsequent "linguistic" analysis (Section 4.3). Specifically, the methodology commendably creates "regionalized" data by binding languages (e.g., pt) to specific local entities (e.g., Brazilian "Brasileirão" teams). However, the results analysis (Section 4.3) regresses to a simple "cross-lingual" comparison, only analyzing performance differences across the six languages. This fails to answer the paper's own core research question. When a model fails on a German (de) task, the analysis does not provide insight into whether the failure is due to a lack of linguistic understanding (German grammar) or a lack of regional knowledge (German "Bundesliga" facts). This leaves the main "regionalization" contribution feeling hollow at the analysis stage.

3. Superficial Analysis and Misplaced Error Analysis. The experimental analysis is superficial, relying entirely on top-level quantitative metrics (e.g., pass^3) without providing deeper insights. It has two shortcomings:
- Lack of Qualitative Case Studies: The main paper includes no qualitative analysis or case studies. A single example of a failed API call sequence would have been invaluable for understanding the root cause of model failures and for supporting the "linguistic vs. regional" argument.
- Misplaced Error Analysis: A more detailed, programmatic error analysis (e.g., "did the model call Get_Leaderboard?") does exist, but it is relegated to Appendix G. This granular analysis is important, and I would suggest it to be in the main text to explain why different model families (e.g., Gemini vs. Qwen) show different top-level performance.

4. Poor Presentation and Lack of Rigor. The paper suffers from poor presentation, which is not up to the standard of a top-tier conference. For example, there are rendering errors and notation missing issues: The metric definition in Section 3.5 (line 307) for pass^3 has a clear rendering error in the PDF, with the caret ^ overlapping the number 3. Also, the symbol N in the formula is not defined.

### Questions
1. The term "agent evaluation" seems overstated, I would suggest the authors to justify this scope or re-frame the paper as a "tool-use" benchmark.
2. The current results don't show if models fail due to language or regional knowledge. Can authors provide some analysis that disentangles   these two factors?
3. The paper notes that xLAM models, fine-tuned for function calling, performed worse than base models. This is counter-intuitive. What is the authors' hypothesis for this?

### Soundness
2

### Presentation
1

### Contribution
2
