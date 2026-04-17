# DeepShop: A Benchmark for Deep Research Shopping Agents

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Web agents for online shopping have shown great promise in automating user interactions across e-commerce platforms. Benchmarks for assessing such agents do not reflect the complexity of real-world shopping scenarios, as they often consist of overly simple queries with deterministic paths, such as “Find iPhone 15.” Real shopping scenarios are inherently more layered, involving multi-dimensional product attributes, search filters, and user-specific sorting preferences. To address this gap, we introduce DeepShop, a benchmark designed to evaluate web agents in complex and realistic online shopping environments. DeepShop comprises three key components. (1) Query diversity evolution: Starting from real user queries, we generate diverse queries across five popular online shopping domains. (2) Query complexity evolution: We further evolve these queries to increase complexity, considering product attributes, search filters, and sorting preferences, and classify them into three levels: easy, medium, and hard, based on the number of evolutions. (3) Fine-grained and holistic evaluation: We propose an automated evaluation framework that assesses agent performance in terms of fine-grained aspects (product attributes, search filters, and sorting preferences) and reports the overall success rate through holistic evaluation. We conduct a systematic evaluation of retrieval-augmented generation (RAG) methods, web agents, and deep research systems. Results show that RAG struggles with complex queries due to its lack of web interaction, while other methods face significant challenges with filters and sorting preferences, leading to low overall success rates. We also perform cross-category, complexity-based evaluations and error analyses to support the advancement of deep research shopping agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper expands on existing online shopping benchmarks by increasing query complexity through the inclusion of product attributes, search filters, and sorting preferences. They demonstrate that increasing query complexity leads to decreased model performance, and that poor multimodal capabilities significantly harm performance.

### Strengths
- Increased complexity over existing benchmarks
- Identification of poor visual understanding as a bottleneck in some domains

### Weaknesses
- Lack of scope: while shopping is a relevant subdomain for web agents, this benchmark covers only a small slice of potential interactions, limiting potential impact.
- Lack of novelty: artificially increasing query complexity relative to existing benchmarks is only a minor variation on prior work, and does not introduce fundamentally new challenges.
- Limited insight: the fact that increasing the complexity of queries leads to worse performance is unsurprising.

### Questions
- How do the authors see this benchmark being used to guide further model or algorithm development, and how does it improve on this vs existing benchmarks?
- Does the increase in complexity vs WebShop elicit novel insights or behaviours not already visible in this benchmark?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces DeepShop, a new benchmark designed to evaluate web agents and deep-research systems on realistic online shopping tasks that involve multi-faceted constraints—product attributes, filters, and sorting—over live e-commerce websites. The authors argue that prior web-agent benchmarks (e.g., Mind2Web-Live, WebVoyager) are either static or insufficiently complex, failing to capture the nuanced reasoning and interaction patterns needed for actual online shopping.

contributes:

A query evolution framework that expands and complexifies real shopping queries across five categories (Books, Electronics, Home, Fashion, Sports).

A multi-dimensional evaluation protocol that decomposes success into attributes, filters, sorting, and holistic task success.

Comprehensive empirical evaluation of both web agents and deep-research systems (e.g., GPT-4o + Browser Use, Gemini Deep Research, OpenAI Deep Research).

The findings reveal that current systems perform poorly on realistic multi-constraint queries—particularly on filter-related operations—highlighting major gaps between language reasoning and grounded web interaction.

### Strengths
Novel and practical benchmark: DeepShop fills an important gap by testing systems on real-time, constraint-heavy shopping tasks that mirror real user behavior.

Systematic query evolution: The staged expansion from simple to complex queries is methodologically sound and produces controllable difficulty tiers.

Granular evaluation: Separating fine-grained metrics (attributes, filters, sorting) from holistic success provides diagnostic insights that go beyond aggregate scores.

Comprehensive coverage: The benchmark spans five diverse categories and includes both web agents and deep-research systems, yielding generalizable findings.

Empirical rigor: The authors report inter-rater reliability between human and GPT-4o evaluators, adding credibility to automated scoring.

### Weaknesses
Limited task realism in some cases: Although queries evolve in complexity, the benchmark still relies on synthetic LLM-generated text rather than actual user intent logs, which may limit ecological validity.

Evaluation dependency on GPT-4o: Automated judging introduces potential bias, especially since GPT-4o is also part of the evaluated systems.

Unclear generalization beyond e-commerce: The authors claim DeepShop can inspire broader web-agent research, but no experiments are shown outside shopping domains.

Limited ablation or analysis of failure modes: The paper reports average scores but lacks detailed qualitative examples explaining why agents fail at filters or sorting.

### Questions
Human data grounding: How do you ensure that the evolved queries reflect realistic shopping intent? Did you validate them with human annotators or external user logs?

Judging consistency: Since GPT-4o is both a baseline and the automated evaluator, did you perform cross-model validation (e.g., using Gemini or Claude) to verify scoring robustness?

Dynamic web changes: Given that e-commerce websites change frequently, how do you ensure benchmark reproducibility over time?

Generalization potential: Could DeepShop’s query evolution and scoring framework be adapted for non-shopping domains such as travel booking or information retrieval?

Multi-modal grounding: Since categories like Fashion rely heavily on visual cues, how does DeepShop handle or evaluate the visual reasoning aspect?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes an in-the-wild evaluation suite for web navigation agents, specifically in the domain of shopping. The core contribution is a set of fine-grained instructions, e.g., “Find a queen-sized bedspread featuring a floral design in a calming blue hue…” along with automatic evaluators for determining whether a model has successfully executed these instructions.

### Strengths
1. Overall, the paper is pretty easy to read and understand
 
2. As far as I can tell, the task has a reasonable amount of headroom (cf. Table 2), suggesting that this benchmark might stay relevant for longer than more synthetic alternatives like WebShop

### Weaknesses
1. The paper may be of limited interest to researchers who are not specifically interested in web shopping agents. Much of its analysis is about, e.g., differences in agent performance across different shopping domains (e.g., books vs. fashion), and doesn’t seem applicable to web navigation or agent research more broadly. My impression is that the tasks described here are roughly a subset of those described in more general-purpose benchmarks like AssistantBench

2. One major concern I have is whether the tasks as specified are all achievable. Because the task design primarily involves intersecting a number of attributes (e.g., rating, # reviews), it seems possible that many of the generated user queries might not be uniquely resolvable. As a result, ceiling performance might be substantially lower than 100%

### Questions
1. How do human and LLM evaluators evaluate sorting preferences? It seems like this should condition not just on the final result but also on the full range of available options on the web.

2. Are models only allowed to search for products on Amazon? This is my impression from the appendix, but it should be clarified in the main body of the paper.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces DeepShop, an online benchmark for evaluating web shopping agents on complex, realistic shopping tasks. Starting from seed queries drawn from existing live-web benchmarks, the authors use LLM-driven diversity and complexity evolution to produce queries spanning five domains (Books, Electronics, Home, Fashion, Sports) and layered constraints (attributes, filters, sorting). Evaluation is both fine-grained (check attributes / filters / sorting separately) and holistic (task success only if all required aspects pass). Experiments on simple RAG, various web agents, and deep research systems show that all current methods struggle significantly, especially with applying filters and sorting preferences, resulting in low overall success rates and highlighting the benchmark's difficulty.

### Strengths
1. The paper identifies a significant limitation in existing web agent benchmarks. The focus on simple, deterministic tasks does not adequately test an agent's ability to handle the complex, multi-constraint queries that are common in real-world e-commerce. The proposed benchmark is on the right track of briging this gap of evaluating more advanced agent that has real-world application values.
2.  The proposed fine-grained evaluation framework is a key strength. It allows researchers to move beyond a simple binary success metric and diagnose why an agent failed (e.g., it found the right product attributes but failed to apply the sorting preference). This is well-demonstrated in the error analysis (Section 5.4) , which identifies critical failure points like grounding, replanning, and limited action spaces.

### Weaknesses
1. The paper lack crutial details on environment and reproducibility. While offline benchmarks are static and simpler, they offer a unique advantage in controllability and reproducibility over online enviroments. A core challenge of an online benchmark is the dynamic nature of websites (e.g., prices change, review counts increase, items go out of stock). The paper does not adequately address how the benchmark's ground truth is maintained. How can an agent's success on a query like "at least 300 reviews" or "check for the lowest price"  be reliably evaluated over time? This lack of clarity raises significant concerns about the benchmark's long-term utility and reproducibility. This has been a primiary issue that is discussed in previous work of online benchmarks (e.g. WebCanvas). However, the paper does not provide much details nor discussions about how this is managed in DeepShop.
2. The paper is vague about the specific e-commerce sites used. Screenshots and a prompt example suggest Amazon is a primary target, but this is not explicitly stated or confirmed as the only target. This may cause concerns in bias and robustness of the performance evaluation using the proposed benchmark. 
3. The benchmark's focus on single-turn, complex queries may not be representative of realistic shopping scenariors, which is often an interactive, multi-turn process where preferences are refined (e.g., "Find me a laptop," "Okay, one for gaming," "Under $1000"). The paper's formulation as a single, complex instruction may be able to reflect the realistic distribution of user shopping intents.

### Questions
1. Given that DeepShop is an online benchmark, how does the evaluation framework handle the dynamic nature of live websites? How do the authors ensure that a task's ground truth remains stable for evaluation, and how can other researchers reliably reproduce the results months from now?
2. Could the authors please clarify which specific e-commerce websites are included in the DeepShop benchmark? Is it exclusively Amazon.com, or are other sites used?

### Soundness
2

### Presentation
2

### Contribution
2
