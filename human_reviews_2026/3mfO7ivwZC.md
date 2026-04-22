# Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
Integrating LLM powered operators in declarative query languages allows for the combination of cheap and interpretable functions with powerful, generalizable language model reasoning. However, in order to benefit from the optimized execution of a database query language like SQL, generated outputs must align with the rules enforced by both type checkers and database contents. Current approaches address this challenge with orchestrations consisting of many LLM-based post-processing calls to ensure alignment between generated outputs and database values, introducing performance bottlenecks. We perform a study on the ability of various sized open-source language models to both parse and execute functions within a query language based on SQL, showing that small language models can excel as function executors over hybrid data sources. Then, we propose an efficient solution to enforce the well-typedness of LLM functions, demonstrating 7% accuracy improvement on a multi-hop question answering dataset with 53% improvement in latency over comparable solutions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses type misalignment when LLMs generate outputs inside SQL queries (e.g., LLM says "Washington D.C." but database has "Washington DC"). Proposes inferring return type constraints from SQL context (e.g., f() > 40 → must return int) and using constrained decoding to enforce type-safety at generation time. Shows 53% latency improvement over LOTUS and demonstrates 3B-8B models can execute hybrid SQL+LLM queries when properly constrained.

### Strengths
1. Type inference is practical. Automatically deriving constraints from SQL expressions  is elegant - city = f() → constrain to database values, f() > 40 → constrain to integers.
2. Single-pass generation is  faster. Replacing multiple LLM calls (generate → validate → maybe retry) with one constrained generation shows real speedup: 0.76s vs 1.7s on same hardware.
3. Database value alignment at decoding level is novel. Using Literal types to constrain outputs to exact database values solves the "Washington D.C." vs "Washington DC" problem without semantic similarity matching or additional LLM calls.

### Weaknesses
1. Only works for open-source models. Constrained decoding requires logit access. Doesn't work with GPT-4, Claude, Gemini APIs. Paper never states this limitation explicitly. Baseline systems like LOTUS and SUQL can use closed-source models (SUQL used GPT-3.5), which may be stronger. Need head-to-head comparison.
Missing critical accuracy comparisons:

2. Missing critical accuracy comparisons:
  - No comparison vs SUQL (uses GPT-3.5 for validation) - only cited in related work 
  - Table 2 only compares latency vs LOTUS, not accuracy
  - Baselines like LOTUS can use closed-source models (potentially stronger) - need head-to-head accuracy comparison
  - Don't evaluate 70B as executor despite showing 70B RAG gets 57.8% vs their best 50.1%

3. Regex constraints are fragile. Pattern \d+ only allows single number sequences. Unclear how this handles answers with multiple numbers, formatted numbers (14,000,000), ranges (35-40), or complex types (latitude/longitude pairs). No analysis of failure modes or what percentage of queries need richer type expressions.

### Questions
1. What's the actual cost comparison? For 1M queries: self-hosting 8B (GPU, ops, engineering) vs closed-source API calls. At what scale/accuracy requirement does your approach win?

2. How does accuracy compare vs systems that can use closed-source models? LOTUS and SUQL can use GPT-4/GPT-3.5. Your method is restricted to open-source. What's the accuracy gap? Is the latency gain worth it?

3. Can you quantify the brittleness? How often do type constraints fail or lose important information (uncertainty, ranges, etc.)? What percentage of queries need complex types that regex can't handle?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studied the abilities of LLMs for function executors and proposed an efficient solution to enforce the well-typedness of LLM functions. However, the writing of this paper makes it very hard to understand.

### Strengths
1. Proposed an efficient, integrated system for type-constrained LLM functions. Introduces a DB-first approach coupled with decoding-level constraint application on data generation formats, ensuring that LLM outputs adhere to specific database requirements. Comparative experiments show a noticeable optimization when compared against the LOTUS model.
2. Achieved efficient integration with BlendSQL by embedding type-constrained functions directly into the database execution process, leveraging the declarative nature of the query language.

### Weaknesses
1. Novelty Overstatement and Incremental Contribution: The paper's claimed novelty is weak. The concept of prioritizing database execution to mitigate the high cost of LLM functions has already been discussed in existing literature concerning LLM query optimization in relational workloads[1].
2. Clarity and Organization Deficiencies:
    1. Confusing Figure References: Figure 2, explicitly titled “Execution flow of a MAP function” offers a description that is overly brief. Furthermore, the figure is inappropriately used to illustrate the execution flow for the TAG-Bench experiments in Section 4.1, despite the main function for many QA tasks being the LLMQA function. This creates a conflict between the figure's specificity and its generalized application, undermining clarity.
    2. Insufficient Methodological Detail: The execution details for applying constraints in Section 3 are not fully elaborated, specifically lacking the precise mechanism for transforming type rules into decoding constraints.
3. Incomplete Experimentation and Analysis:
    1. Lack of SOTA Comparison: Experiments lack comparison against contemporaneous program synthesis models such as SuQL[2], making it difficult to assess the competitive standing of the proposed approach.
    2. Insufficient Error Analysis: The analysis of experimental results is superficial. For instance, the authors should provide a detailed breakdown of error types in the unconstrained mode and quantify how many of these errors were eliminated by the proposed method.

[1]. Shu Liu, Asim Biswal, Audrey Cheng, Xiangxi Mo, Shiyi Cao, Joseph E Gonzalez, Ion Stoica, and Matei Zaharia. Optimizing llm queries in relational workloads. CoRR, 2024.
[2]. SuQL；Shicheng Liu Jialiang Xu Wesley Tjangnaka, Sina J Semnani Chen Jie Yu, and Monica S Lam. Suql: Conversational search over structured and unstructured data with large language models.

### Questions
1. The authors should provide an error-type categorization table, similar to Table 4, generated without using any constraint method. Without knowing the original distribution of error types, it is impossible to accurately determine how many low-level errors the proposed method successfully avoided.
2. Please provide a clear explanation for the missing Program Execution results for the Llama-3.3-70b-Instruct model. Specifically, is the absence of data related to deployment cost or an unresolved technical issue with applying the constraints to the largest model?
3. Has the issue of an excessively large LITERAL set been considered? If a column has millions of unique values, how does the system manage the memory and context overhead of feeding this entire set to the LLM and the constraint decoder?

### Soundness
2

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
3

### Summary
In this paper, the authors introduce a database value alignment approach for enforcing type correctness when integrating LLM-based user-defined functions into declarative query languages. 
It uses the SQL expression context to determine the expected output type and restricts the model’s return value to match this type before execution.
This eliminates the need for expensive post-hoc LLM validation calls commonly used in prior systems, significantly reducing latency. The approach is implemented in BlendSQL, which supports integrating LLM functions alongside traditional SQL operators over both structured and unstructured data. 
Experiments on TAG-Bench and HybridQA show a reduction in execution latency and higher accuracy. 
The study further finds that small LLMs can effectively act as function executors when guided by type constraints.

### Strengths
S1. The approach is well-motivated and fully implemented end-to-end. The authors provide a working system rather than a conceptual prototype, and demonstrate significant execution-time savings over LOTUS in practical settings (53% latency reduction on TAG-Bench).

S2. The experiments clearly show that typing policy directly affects execution accuracy: even lightweight type constraints substantially improve denotation accuracy on HybridQA, confirming that type alignment is not just a syntactic detail but an accuracy bottleneck in LLM–DB integration.

### Weaknesses
W1. The provided motivating examples can be reduced to binary filter logic (True/False) that does not require literal value alignment. For example:

`Select * from w where w.name = LLMQA('who won the 2012 NRL Grand Final with Melbourne')`

Can be rewritten something like:

`Select * from w where LLMQA(f' Did {w.name} won the 2012 NRL Grand Final with Melbourne?', output: True/False)`



W2. Some of the typing benefits may already be achievable through structured output APIs or existing regex-constrained generation interfaces provided by modern LLM serving stacks, raising questions about novelty of the approach. 

W3. Where a column contains many distinct value, adding literal into the prompt might be very costly?

W4. The HybridQA results in Table 3 are incomplete for the strongest model. Why didn't the author run on the entire benchmark (instead of samples) and finish all the results.

### Questions
Please see weakness.

### Soundness
2

### Presentation
2

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
The paper presents the use of constrained generation as a way to ensure the LLM integration with declarative programming languages results in type consistent and correct outputs, preventing the breakdown of overall pipeline. One of the interesting observations made in the paper is that the use of smaller LM as an executor of the UDF generated by a larger LLM is quite competitive. 

Overall, the paper although is interesting, fails to introduce any specific novelties. It is not claiming individual components such as BlendSQL, constrained decoding, or “small LMs as executors” as novel contributions. The primary contribution seems to be the DB-aware way they combine these pieces -- this is not strong enough contribution for consideration in ICLR (although it is an interesting empirical study).

### Strengths
S1. Clear integration of ideas, and nice DB integration. 
S2. Excellent evidence for efficiency, and nice comparison with the use of smaller models as executors.
S3. Reasonable artifacts for reproducibility of results.

### Weaknesses
W1. Technical novelty is rather weak, although the overall integration/engineering seems to be quite practical. 
W2. Although there are other DB-integrated systems mentioned in the paper -- SUQL, DuckDB UDF etc -- there are no empirical comparisons. 
W3. The biggest issue seems to be lack of focus in terms of the contributions of the paper: BlendSQL is a prior art so its evaluation in 4.1 seems to be quite out of place, constrained decoding is also well known. The interesting part seems to be the use of small LMs as executors, but this is empirical observation with no deeper insights. So, on the whole the paper seems like a idea-integration work, which may not be suitable for ICLR. 
W4. The handling of LITERAL seems suitable only for settings where the number LITERALs is fairly small. On really large databases, with millions of unique LITERALs in a column sometimes, the presented approach does not seem workable. 
W5. There are many systems level challenges in the integration -- especially since the entire solution hinges of temporary table creation -- that are missing in the paper: concurrency, batching across queries, effect of prefix-cache hits on throughput, and optimizer choices (join order) under realistic workloads.

### Questions
Please address the comments in the previous Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
