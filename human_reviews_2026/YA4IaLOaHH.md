# Lost in the Maze: Overcoming Context Limitations in Long-Horizon Information-Seeking

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Long-horizon agentic search requires iteratively exploring the web over long trajectories and synthesizing information across many sources, and is the foundation for enabling powerful applications like deep research systems. In this work, we show that popular agentic search frameworks struggle to scale to long trajectories primarily due to context limitations—they accumulate long, noisy content, hit context window and tool budgets, or stop early. Then, we introduce SLIM (Simple Lightweight Information Management), a simple framework that separates retrieval into distinct search and browse tools, and periodically summarizes the trajectory, keeping context concise while enabling longer, more focused searches. On long-horizon tasks, SLIM achieves comparable performance at substantially lower cost and with far fewer tool calls than strong open-source baselines across multiple base models. Specifically, with o3 as the base model, SLIM achieves 56% on BrowseComp and 31% on HLE, outperforming all open-source frameworks by 8 and 4 absolute points, respectively, while incurring 4–6x fewer tool calls. Finally, we release an automated fine-grained trajectory analysis pipeline and error taxonomy for characterizing long-horizon agentic search frameworks; SLIM exhibits fewer hallucinations than prior systems. We hope our analysis framework and simple tool design inform future long-horizon agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a lightweight agent framework for long-horizon information-seeking. By limiting search depth, avoiding unnecessary webpage browsing, and performing periodic summarization, the method aims to reduce tool usage and context length while maintaining answer quality.

### Strengths
- The paper focuses on an important problem in long-horizon agent research—context bloat and noise accumulation during multi-step search.

- The framework is simple, easy to reproduce, and may have practical applicability due to its low engineering overhead.

- Experiments cover multiple benchmarks, and results are generally stable.

### Weaknesses
- Missing comparisons with the most relevant literature.
Many existing agent systems (e.g., InfoGENT, memory-based agents) already include context compression and memory management mechanisms that target the same issue. Without comparing to these, the claimed contribution and improvement are not well supported.

- Metric design raises fairness concerns.
Counting tool calls while excluding webpage crawling, whereas prior agents include it, makes the comparison less meaningful. It is unclear whether token-cost metrics also exclude webpage reading. The advantage may come from accounting choices rather than methodological benefits.

- Limited insight and unclear mechanism.
The paper does not clearly explain why the framework improves performance—is it better planning, cleaner evidence selection, or simply shorter text? Improvements seem small and potentially within sampling noise, without offering deeper understanding or design principles.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents an approach for long-horizon search agents and evaluates it against a benchmark. The results are promising and show reduced computational effort with increased effectiveness.

### Strengths
- The paper presents interesting results that follow the recent trend of agentic frameworks for information seeking and deep research. 

. The authors use recent valid benchmark tasks and data

- Some insights on why current approaches may fail are provided

### Weaknesses
- The main weakness of the paper is that it is merely a combination of existing tools that can overcome the context limitation of current models

- There is limited methodological contribution in the task alignment or agent models themselves

- Some of the results are not correctly highlighted for best performance

### Questions
- Lack of statistical evidence on whether the results are significantly different from baselines

- Lack of insights on why existing systems fail. If I understood correctly this is mainly because of context limitations and the advantage of the proposed approach is more practical than scientific

### Soundness
3

### Presentation
4

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
This paper targets long-horizon information-seeking systems and argues that context explosion is the main bottleneck preventing existing agent frameworks from scaling. The authors propose SLIM, a lightweight method separating search and browse operations and periodically summarizing interaction history. Experiments on BrowseComp and HLE show comparable or better performance to several open-source baselines with lower tool usage.

### Strengths
1. Identifies a practical but overlooked bottleneck in agent-based research systems.
2. Simple yet effective design; strong empirical support.
3. Thorough evaluation across multiple models and benchmarks.
4. Trajectory-level error taxonomy adds clarity and interpretability to agent failures.

### Weaknesses
1. Core components are heuristic and not deeply analyzed (e.g., why summarize every n steps?).
2. Lack of rigorous ablations and theoretical justification for the design decisions.
3. Results are promising but not clearly superior in all cases; advantages are modest on some metrics.
4. Does not explore whether SLIM benefits RL-trained or task-specialized agents.

### Questions
1. How sensitive is SLIM to summarization interval n? Could adaptive summarization outperform fixed cadence?
2. Would a memory-selective mechanism (retain only high-value facts) be more efficient than compressing the full trajectory?
3. How robust is SLIM to noisy or domain-specific search engines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses context management challenges in deep research systems. The authors propose SLIM (Simple Lightweight Information Management), a framework that separates retrieval into distinct search and browse tools while periodically summarizing trajectories to maintain concise context. Evaluated on part of BrowseComp and Human Last Exam benchmarks, SLIM outperforms the open-source baselines in accuracy and the number of tool calls. The paper also introduces an automated trajectory analysis pipeline with an error taxonomy to characterize failure modes, revealing that SLIM exhibits fewer unfocused searches and less hallucination than existing systems.​

### Strengths
1. SLIM shows strong gains on challenging benchmarks. It outperforms all open-source baselines by a substantial margin (e.g., +8 points on BrowseComp). Importantly, it achieves these gains efficiently: SLIM requires only ~15–25% of the tool calls that competitors need to reach similar turn counts. The results hold across multiple base LLMs (o3, o4-mini, Claude-4-Sonnet) and both task domains, suggesting the high generalization of the proposed approach.

2. Beyond the high performance of SLIM, the paper contributes a trajectory error taxonomy and an automatic trajectory analysis pipeline. By categorizing failure modes (e.g. unfocused search, confirmation bias, hallucination) and annotating agent trajectories, the authors gain deeper insight. They show SLIM’s advantage is largely due to dramatically reduced hallucination rates compared to others, a valuable finding. This analysis is itself a novel contribution and is well-integrated into the paper.

3. The paper is well-written and structured. The authors also release source code to reproduce the experiments. Overall, the presentation is clear and the method is easy to follow.

### Weaknesses
1. The evaluation scope is limited:

1.1. The experiments focus on two specific benchmarks (BrowseComp and HLE). While both are challenging, it is unclear how SLIM would perform on other kinds of tasks (e.g., open-domain QA, shorter queries, or multimodal search). Both BrowseComp and HLE focus on factoid question answering with short, verifiable answers. This limits our understanding of how SLIM performs on​ open-ended research questions requiring synthesis across sources, multi-hop reasoning requiring complex information integration, and tasks where the answer requires aggregation rather than finding a single fact. The paper acknowledges this implicitly by noting answers are "short and straightforward, resulting in reliable evaluation", but this also means the evaluation doesn't cover the full spectrum of deep research tasks mentioned in the motivation.​

1.2. Only part (300 samples) from each dataset are used, making it difficult to compare the results with metric values from previous studies. Though it is reasonable to conduct most experiments on such a subset, it is still important to report the results of the best configuration of the proposed technique on complete sets, and present its comparison with SOTA results obtained by other researchers.

1.3. The sample size (300) can be not enough for drawing strong conclusions about a method's effectiveness. Moreover, the paper provides no statistical significance testing, confidence intervals, or error bars. Without knowing the variance in performance, it's unclear whether the reported improvements are statistically significant. For example, the 7-point improvement on BrowseComp could be within noise if variance is high across different runs or random seeds.​

2.  The choice of baselines (REACT, SEARCH-O1, HF-ODR, GPT-Researcher) should be explained better. There exist a lot of open-source implementations of deep research systems (see, e.g., survey https://arxiv.org/abs/2506.12594 or Universal Deep Research from NVidia), so there more thorough discussion and, potentially, experimental study, are important.

3. The most significant methodological flaw is the absence of a proper ablation study. SLIM has 3 key components (separated search/browse, periodic summarization), but it's unclear how much each component contributes to the overall performance and efficiency gains depending on the problem and on the base model. Does the summarization module provide a significant boost over just using the separated tools? Is the `browse` tool's selective fetching more important than the `search` tool's conciseness? Without this analysis, it's difficult to attribute the success to the full design versus a single key component. How does performance vary with different summarization strategies, the change in search engine (Google) and web scraper (crawl4ai)?

4. The summarization module, while crucial, could introduce its own errors (e.g., loss of critical details). The paper does not analyze what kind of information is lost during summarization and whether this ever leads to failure. A qualitative analysis of summarization errors would strengthen the claims.

5. The error taxonomy is insightful, but it is manually defined from BrowseComp examples. It’s unclear how well these categories apply to other domains or how reliable the heuristic/LLM-based annotator is. Some of the failure modes (e.g. “answer ignored”) are a bit ambiguous.

6. The paper ignores existing papers that analyze the fine-grained failures of multi-agent systems, e.g., TRAIL (https://arxiv.org/abs/2505.08638) or MAST taxonomy (https://arxiv.org/abs/2503.13657). To highlight the contribution of the proposed trajectory error taxonomy, it is necessary to compare it with similar ideas.

7. The paper is focused on open-source baselines in deep research, but all used LLMs are proprietary. Though I understand that they may be much more accurate compared to open-source LLMs, experiments with at least one open LLM and other open tools (search engine, etc.) will significantly increase the reproducibility.

### Questions
1. What are the limitations of the proposed deep research? In what scenarios would it work worse compared to existing systems?

2. The browse tool fetches the “most relevant section” of a page via a similarity function. How do you split the document into separate sections? Did you compare this against simply fetching the whole page (or multiple snippets)?

3. Have you tried SLIM on shorter-horizon tasks (e.g. conventional question-answering) or tasks without web search? Would the search/browse framework still be beneficial in those scenarios?

4. The automated pipeline tags trajectories with errors. Can you clarify how much manual effort is needed for this? How confident are you that “hallucination” is correctly detected by the system?

### Soundness
3

### Presentation
2

### Contribution
3
