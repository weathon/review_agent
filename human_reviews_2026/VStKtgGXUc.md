# jqBench: a benchmark for reading and editing JSON from natural language and/or examples

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
We introduce jqBench, a new benchmark for evaluating language models on JSON querying and transformation tasks, where the intent can be given specified using natural language and/or examples.
Whereas jqBench is mainly aimed at using the jq tool, it can be used to evaluate other programming languages that query and/or transform JSON.
Benchmarks are automatically created from two rich sources of data: Stack Overflow discussions (1496 instances with instructions and examples, called jqStack) and the Spider dataset for SQL generation from natural language (859 instances with instructions and JSON Schema, called jqSpider).
We describe and analyze the automated pipeline for benchmark creation, and perform extensive baseline experiments on different models to analyze the complexity and failure modes.
Using implicit feedback, the best model (Opus 4.1) scores 76% on the jqStack benchmarks and 81% on the jqSpider benchmarks.
Additionally, we show (1) that access to the documentation surprisingly does not help, (2) jq lags behind Python, and (3) that automatic feedback (and therefore examples) is crucial.
Besides the challenging benchmarks, we release 13K converted but filtered cases for training purposes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces JQBENCH, a new benchmark designed to evaluate the ability of large language models (LLMs) to query and transform JSON data. The tasks can be specified using natural language, input/output examples, or both. The benchmark is novel in its focus on the jq query language, a powerful but specialized tool.

### Strengths
1. The paper tackles a highly relevant and practical problem. As LLMs are increasingly used in agentic workflows and data processing pipelines, their ability to handle JSON—the de-facto standard for data exchange—is critical. Focusing on jq is a novel choice that tests a model's ability to learn a concise, powerful, and non-mainstream domain-specific language.
2. The creation of two separate benchmarks (JQSTACK and JQSPIDER) is a major strength. JQSTACK provides "in-the-wild" problems from real developers, ensuring the tasks are diverse and practically relevant. JQSPIDER provides structured, complex, and deeply-nested queries, effectively testing a model's logical reasoning and ability to chain operations (mapping JOINs to jq pipes).

### Weaknesses
1. How do you handle complex multi-table reference scenarios in a database when building jq-spider set. For example, what is the core principle for selecting the root table? 
2. A key challenge with JSON objects lies in their nested structure—an issue particularly prominent in real-world use cases like agent function calls. However, the paper does not provide detailed analysis of this aspect. In particular, it fails to explore critical questions: whether the depth of nested structures impacts model performance, and if so, how.
3. The presentation of Tables 1 and 2 is confusing, with several unclear elements. For example, some accuracy values exceed 100%—a result that requires explanation, as accuracy is typically constrained to a 0%–100% range. Additionally, the entry for "gpt-4.1-mini" is split across two lines, which disrupts readability. Further questions arise from the data itself: why does GPT-5 exhibit performance numbers similar to GPT-4? And under certain configurations, why does GPT-4.1-mini outperform both GPT-5 and GPT-4.1?
4. The automated pipeline relies heavily on GPT-4.1 to generate JSON schemas from SQL and to translate SQL queries into jq expressions. This could introduce a bias where the "correct" jq solutions in JQSPIDER reflect GPT-4.1's specific style of jq programming, which may not be the most optimal or human-idiomatic solution.
5. This paper lacks a detailed error analysis of different models. For example, the paper states the cause is the model "getting stuck in a loop of requesting documentation," but doesn't deeply analyze why. Is this a failure of the agent's reasoning, a problem with the RAG prompt, or an issue with the quality/structure of the provided documentation? A few qualitative examples of these failure loops would significantly strengthen this claim.

### Questions
1. The paper mentions that "In contrast, JQSPIDER—derived from Spider database queries—shows strikingly low variability: all models except for Phi 4 are in the 72%–81% range." For my own suggestion, I think the Spider dataset has had potential data contamination issues since its creation. Therefore, I recommend using the BIRD [1] dataset and even its LiveSQLBench [2] variants instead. This switch might bring about some new insights.
2. In the Spider dataset, certain databases contain tables that lack "value" (i.e., have no actual data entries). How would you handle this scenario?


[1] https://bird-bench.github.io/
[2] https://livesqlbench.ai/

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
The paper introduces jqBench, a new benchmark for testing language models on JSON querying and transformation tasks using natural language or examples. jqBench is automatically generated using Stack Overflow and Spider dataset.

### Strengths
Strengths 

S1. The standardization using JSON schema + jq can be very significant. And this paper propose such a benchmark in the area 

S2. The pipeline automates the dataset generation with task translation+ react style verification.

### Weaknesses
Weaknesses

- The justification of the significance of jq operations is not enough. The comparison between the methods using table schema + SQL queries and those using JSON schema + jq is missing. Will jq outperform or is on-par with SQL generation while have the benefits of universal representation? What are the benefits of this translation.

- The tasks are mostly code generation. How about other tasks that highly rely on doc understanding, like RAG and natural language Q&A, how general the idea could be. 

- The volume is not a big, less than 2000 tasks.

### Questions
Q1. What are the main failures observed during benchmarking?

Q2. Any human checks on the quality of the data?

Q3. How will it change if using BIRD dataset which is more complex?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces JQBench, a new benchmark for evaluating systems that translate natural-language descriptions and examples into data-transformation programs written in `jq` (a lightweight yet expressive JSON processor). Given the ubiquity of JSON as a structured data format, the release of such a benchmark is both timely and valuable for research at the intersection of program synthesis, code generation, and natural-language interfaces.

JQBench comprises two complementary datasets: 

- JQStack, created by curating transformation problems from Stack Overflow posts, where realistic developer questions are distilled into machine-checkable tasks. The authors extract natural-language context, compile candidate `jq` expressions and input–output examples, and automatically validate them using an agent-based test generator.
- JQSpider, adapted from the Spider text-to-SQL dataset, reformulated here to express natural-language–to–JSON-transformation problems rather than SQL query generation.

In total, JQBench includes 751 JQStack and 893 JQSpider tasks, plus 3,641 easier subtasks derived from JQStack examples. Empirical evaluation shows the benchmark’s difficulty: even the strongest models, such as Claude 4.1, achieve only 77 % accuracy on JQStack and 81 % on JQSpider, highlighting substantial room for progress. The authors also report interesting diagnostic findings such as reduced performance when models have access to external tool feedback, and the observation that the novelty of the `jq` language itself poses a primary bottleneck.

### Strengths
- JQBench fills a clear gap, as despite JSON’s dominance as a structured-data representation, there are currently no benchmarks that target program synthesis for JSON transformations.
    
- The pipeline for JQStack, based on curated Stack Overflow problems with natural-language context and executable test cases, makes the benchmark realistic and reproducible.
    
- The paper documents the data-collection pipeline, validation process, and statistics, supporting reproducibility and future extensibility.
    
- The authors evaluate a number of models and provide thoughtful analyses of performance bottlenecks, including surprising effects such as decreased performance when models use external tools.
    
- The dataset’s release is likely to stimulate follow-up work on code synthesis and model grounding for low-resource programming languages.

### Weaknesses
- The paper’s primary contribution is the benchmark itself; there is little methodological or theoretical innovation beyond dataset construction.
    
- While model comparisons are informative, the analysis could be strengthened by deeper ablations, e.g., performance versus task complexity, or qualitative error analyses that categorize common reasoning failures.
    
- Although JQBench is diverse, the total number of problems remains modest compared to large-scale synthesis benchmarks.

### Questions
1. How is task difficulty distributed within JQBench? Are there systematic ways to characterize easy vs. hard transformations?
2. Given that JQBench currently supports only single-JSON inputs, have you considered extending it to include multi-step or compositional transformations that require reasoning over multiple JSON structures or files?
3. Do you plan to extend JQBench to other transformation languages or to cross-format tasks?

### Soundness
3

### Presentation
3

### Contribution
3
