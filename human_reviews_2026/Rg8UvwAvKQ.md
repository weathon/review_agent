# BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 2, 6

## Abstract
Search-augmented large language models (LLMs) have advanced information-seeking tasks by integrating retrieval into generation, reducing users’ cognitive burden compared to traditional search systems. Yet they remain insufficient for fully addressing diverse user needs, which requires recognizing how the same query can reflect different intents across users and delivering information in preferred forms. While recent systems such as ChatGPT and Gemini attempt personalization by leveraging user histories, systematic evaluation of such personalization is under-explored. To address this gap, we propose BESPOKE, the realistic benchmark for evaluating personalization in search-augmented LLMs. BESPOKE is designed to be both realistic, by collecting authentic chat and search histories directly from humans, and diagnostic, by pairing responses with fine-grained preference scores and feedback. The benchmark is constructed through long-term, deeply engaged human annotation, where human annotators contributed their own histories, authored queries with detailed information needs, and evaluated responses with scores and diagnostic feedback.  Leveraging BESPOKE, we conduct systematic analyses that reveal key requirements for effective personalization in information-seeking tasks, providing a foundation for fine-grained evaluation of personalized search-augmented LLMs.  Our code and data are available at https://anonymous.4open.science/r/bespoke-E82B.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a new framework for evaluating search-augmented LLMs with a specific focus on user personalization in information-seeking tasks.

Specifically,
1. They collect interaction data between real human annotators and search systems (both search engine and search-augmented LLM).
2. Develop an annotation rubric and ask the same annotators to score and provide feedback on $k$ responses.
3. Build an evaluation pipeline for scoring (1) factuality and (2) personalization of a response conditioned on the user's past judgements (collected in the previous step).
4. Validate and ablate the evaluation pipeline against human-annotated labels.
5. Evaluate search-augmented LLMs using the proposed framework and analyze different context construction configurations.

Overall, the paper addresses the gap between evaluation of general search-augmented LLM evaluation and personalization.

### Strengths
The paper raises the importance of personalization when interacting with search-augmented LLMs for information seeking tasks.

- The paper discusses relevant prior work and clearly identifies a clear research gap (Section 2).
- The paper bases its evaluation framework on the data collected from natural unconstrained human user interactions with search systems. This approach is strongly preferred over synthetic and/or task-contsrained data collection (generation) settings (Section 3).
- The paper thoroughly validates and ablates its evaluation pipeline against human-annotated labels (Section 4.1).
- The paper states the importance and utilizes natural language feedback as an important signal for learning and evaluation in addition to numerical scores (Sections 3.3 and 3.4).
- The paper thoroughly studies and ablates different user context construction configurations (Section 4).
- The paper has a logical easy-to-follow narration flow.

### Weaknesses
### Factuality
- The proposed factuality metric is the recall over the atomic facts included in the **personalized** response, $r^+$. However, it should be possible to construct a highly relevant and factual response, $\hat{r}$, that would have minimal overlap with $r^+$. As factuality of search-augmented LLMs is not the main focus of the paper, I believe it would be best to either (1) rephrase or (2) remove factuality assessment from the pipeline.
- Additionally, general factuality is *hard* to define. If a fact can be found through a web search, does it make it factual? The current phrasing of the section can raise such questions, which does not directly relate to the main contributions of the paper.

### Impact of Web Search
In Section 4.5, the authors ablate the impact of the web search module (engine) on the system's performance, by comparing the base system with built-in web search tool and an LLM with access to *ground truth* documents as part of its context. The authors report improvements when the LLM is provided with less noisy higher quality context, with more pronounced improvements among reasoning models. The authors conclude that "reasoning models are more capable of analyzing and integrating the provided information". However, there are no significant differences between non-reasoning and reasoning models when provided with noisy context from a search engine (in some cases non-reasoning models score higher). Thus, a further analysis is required to understand this discrepancy, as the current conclusion is not supported by one of the settings.

### Questions
#### Suggestions for Improving the Flow and Clarity
- In Section 4.1 (Line 287), the authors can elaborate on w/o Feedback baseline to improve clarity.
- In Section 4.2 (Lines 317-322), make the distinction between (2) and (3) more clear.
- In Section 4.2, the authors directly discuss the results after presenting the four aspects of user context construction. Even though the details are included in the Appendix, a brief overview of the generation pipeline (e.g., how is user profile constructed) would be helpful in the main section of the paper.

### Learning from Feedback

The current evaluation framework relies on user annotations in the form of numerical scores and natural language diagnostic feedback. However, in real-world settings, we only have access to raw context in the form of search histories and chat interactions, and additional human-annotated labels are missings. A natural and interesting question arises: how can we build an evaluation framework that only has access to the raw context (no scores or diagnostic feedback)? The raw context can contain rich (but also noisy) feedback signals, which can be used for building user profiles used for both personalized evaluation and generation pipelines. Additional preliminary analysis would significantly strengthen the paper (but can also be thoroghly studied in a separate future work).

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
This paper addresses an important but underexplored question: how to evaluate *personalization* in search-augmented LLMs.

The authors propose **BESPOKE**, a human-curated benchmark that captures authentic chat and search histories from real users, pairs queries with gold information needs, and collects multi-criteria feedback (need alignment, content depth, tone, and explanation style).

BESPOKE provides both *factual* and *personalization* evaluations, offering diagnostic feedback for analyzing where models succeed or fail. Experiments on major search-augmented LLMs show consistent gains from user-context conditioning and highlight personalization as an ongoing challenge .

### Strengths
- **Timely and relevant topic:** Personalization in RAG systems is highly relevant as LLMs shift from generic to user-aware reasoning.
- **Annotation effort:** The benchmark construction is carefully designed and grounded in long-term, real human histories, ensuring realism.

### Weaknesses
1. **Lack of conceptual clarity on personalization.**

   The central concept of *personalization* is never explicitly or rigorously defined in the paper. It remains unclear whether personalization refers to modeling *user persona* (i.e., stable, identity-dependent traits and preferences), *long-context reasoning* over past dialogue turns, or simply *conditioning* on user-specific interaction history. Without a theoretical or formal framework delineating these possibilities, it is difficult to interpret what kind of personalization BESPOKE is truly evaluating. From my understanding, in human–computer interaction, personalization involves sparse, implicit, and often latent signals that go beyond explicit interaction history. Treating user history as equivalent to persona modeling risks conflating two fundamentally different constructs—short-term contextual adaptation versus long-term user modeling. A clearer conceptual grounding is therefore essential.

2. **Ambiguity between personalization and multi-turn dependency.**

   Many of the benchmark’s manipulations—such as whether to provide user history, whether to use full or summarized context, and whether to inject selective history—could equally be interpreted as probing *multi-turn dialogue reasoning* rather than personalization. Multi-turn dialogue modeling inherently requires tracking semantic dependencies across turns, which can produce similar quantitative effects as “personalized” context conditioning. To convincingly claim that BESPOKE measures personalization, the authors must clarify how their benchmark distinguishes between these two phenomena. For instance, what aspects of the benchmark ensure that performance gains stem from user-specific adaptation rather than generic long-context coherence?

3. **Personalization–factuality tension in RAG.**

   The paper situates personalization within a RAG framework, which typically targets factual or knowledge-seeking tasks. However, this raises a conceptual tension: factuality-based information retrieval emphasizes *objective correctness* and *information need satisfaction*, whereas personalization emphasizes *subjective preference alignment*. In tasks like “What are the top-10 cited CS papers in 2025?”, all users should receive the same answer; personalization is not only unnecessary but potentially misleading. The paper should clarify under what task types or conditions personalization is meaningful in RAG, and whether factual information retrieval is even the right domain for testing personalized behavior. A deeper justification of why personalization matters in factual contexts is needed.

4. **Feedback–persona mismatch and empirical bias.**

   In Stage 2 of benchmark construction, user feedback is iteratively incorporated to refine LLM responses into “gold” answers. However, these feedback signals appear to be task-specific—focused on improving the answer for a given question—rather than generalizable reflections of user persona or preference. Using such localized, task-specific feedback to evaluate general personalization ability risks introducing empirical bias: models may learn to fit task-level optimization signals rather than genuine personalization behaviors. The authors should discuss this potential mismatch and clarify whether the benchmark measures general persona alignment or task-specific adaptation.

5. **Metric scope and universality.**

   The proposed four evaluation dimensions—*need alignment*, *content depth*, *tone*, and *style*—may not be equally applicable across all tasks. For many short factual queries, such as “How many countries are in Europe?”, dimensions like tone and style are largely irrelevant and may confound evaluation. This raises questions about the universality and discriminative power of these metrics: are they intended to cover all information-seeking interactions, or only subjective or conversational ones? The authors should clarify whether these dimensions are weighted or selectively applied depending on task type, and justify their inclusion as general indicators of personalization.

### Questions
1. Could you clearly define *personalization* in your framework? Is it about modeling user persona, context dependency, or preference learning?
2. How does BESPOKE ensure that it measures personalization rather than generic multi-turn contextual reasoning?
3. What is the motivation for studying personalization in RAG scenarios where factual accuracy dominates?
4. How does the feedback process (iterative user-LLM refinement) avoid conflating task-specific optimization with genuine persona alignment?
5. Are all four evaluation criteria equally relevant across information-seeking tasks? Could some be task-specific?

### Soundness
3

### Presentation
3

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
This paper introduces a new benchmark called BESPOKE, designed to evaluate how well search-augmented large language models (LLMs) can personalize information-seeking responses. Specifically, it is constructed with 30 annotators alongside their search and chat histories over three weeks, and further paired with 150 user-authored queries that include gold information needs and fine-grained feedback. Then, the experiments on multiple search-augmented LLMs show that leveraging user history (especially query-aware, profile-based contexts) enhances personalization, with challenges on inferring user needs (from long-term histories) and improving reasoning over retrieved information remaining.

### Strengths
* The proposed benchmark on personalization in search-augmented LLMs is timely, which should be highly valuable to the community. Personally, I feel this area has been underexplored despite its growing practical importance. 
* BESPOKE is constructed with both the search and chat histories of multiple users for three weeks, which is realistic and reflects relatively long-term interactions over existing benchmarks. 
* The experiments are comprehensive, including multiple state-of-the-art models and several useful analyses on where the performance improvement comes from.

### Weaknesses
* The collected 2870 user history sessions (combining both the search and chat) over 30 users for 3 weeks seem very small (which corresponds to 4-5 instances per user for one day), from which it is questionable whether the interactions occur and are collected in sufficiently realistic settings, as typical search and chat users make much more queries per day.
* While the authors claim that the annotators have diverse backgrounds, no concrete justification or demographic evidence is provided.
* The benchmark focuses primarily on proprietary systems; however, they are typically not very reproducible. I would like to suggest including open-weight models (or research-grade models), to further facilitate the reproducibility of this benchmark. 
* The results in the main table show similar performances across different models, which deserve deeper discussions. One speculation is that, for this personalization benchmark, the capability of models may matter less compared to the contextual information provided.
* As a benchmark study, it would be beneficial to more explicitly list out the limitations of existing models and potential future works to mitigate them.

I am happy to increase the score to accept, if the above issues and the ethical concern (below) are fully addressed.

### Questions
Please see Weaknesses above.

### Soundness
4

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
4

### Summary
This paper presents BESPOKE, a benchmark designed to evaluate personalization in search-augmented LLMs. The dataset comprises 2870 real user search and chat sessions collected over a three-week period from 30 users, along with 150 user-authored queries accompanied by gold information needs and detailed user judgments (including both scalar ratings and free-form diagnostic feedback) over LLM-generated responses. Beyond providing a dataset, the paper introduces an evaluation framework built on GPT-5 that generates query-specific rubrics and uses an LLM judge to assess model outputs. Using this benchmark, the authors analyze the performance of several frontier search-augmented LLMs and extract insights into the current strengths, limitations, and bottlenecks of personalization in information-seeking settings.

### Strengths
- High-quality, realistic dataset. The benchmark offers rich, deeply contextualized real-world histories and multi-dimensional annotations (scores + diagnostic feedback), which are costly and difficult to collect at scale. This level of contextual grounding is rare and likely to be highly valued by the community.
- Provides both the raw data and a full end-to-end evaluation framework, enabling systematic and reproducible assessment of personalization in search-augmented LLMs.
- Insightful empirical findings. The analysis identifies concrete points of failure and success, e.g., the importance of query-aware context construction, the limits of current web-search quality, key reasoning bottlenecks, and the effectiveness of retrieval methods.
- Overall, the data collection, annotation protocol, experimental setup, and analyses appear thoughtful and well executed.

### Weaknesses
- Limited scale of queries. Although the dataset is rich, the benchmark includes only 150 test queries across 30 users, which is small compared to existing evaluation suites. This raises questions about coverage and diversity of personalization scenarios, although the depth of per-user context partly compensates for the small query set.
- Remaining self-preference concerns. While GPT-5 is used to reduce self-preference effects for Gemini-generated data, parts of the annotation pipeline still involve Gemini-generated responses, and GPT-family models ultimately evaluate GPT-family models (e.g., o3, GPT-4o). Additionally, the heavy reliance on GPT-5 across many stages (evaluation, rubric generation, gold information extraction, profile generation, history selection, and oracle annotation) may limit generalizability across model families and impose high computational cost for running the benchmark end-to-end.
- Underspecified evaluation procedure. Section 4.2 provides only a brief description of how model evaluation is operationalized. The absence of a clear full picture of the entire evaluation pipeline makes it difficult to independently verify or reason about the experimental setup and results. For instance, profile generation is described only by providing prompts in the appendix; there is no discussion of profile quality, consistency, or what kinds of user attributes or preferences are typically captured.
- Insufficient detail on data collection. The paper does not describe what instructions the annotators received for generating chat histories, search histories, or queries. Without this, it is difficult to infer the distributional characteristics of the dataset. Similarly, details about how annotation quality was validated (beyond basic self-consistency checks) are sparse.

### Questions
- Why is recall-only evaluation used for factual coverage in Section 4.1? With recall as the main metric for factuality, models that produce long responses may be rewarded despite adding inaccurate or irrelevant information. Is there a safeguard against this behavior? Precision seems relevant given that models often add extraneous information. What motivated the choice to exclude precision-oriented metrics? 
- What value of k was used for generating hypothetical information needs? The text mentions sampling k responses but does not specify the actual hyperparameter.
- Did you analyze how the amount or type of user context affects personalization performance? For example, does increasing the number of sessions, the mix of search vs. chat history, or the recency of histories correlate with improvements?
- Can you provide a cost breakdown for running BESPOKE? This includes human annotation, LLM-assisted annotation, evaluation, profile generation, and search grounding. A per-query estimate would be particularly helpful for researchers considering extending the dataset.

### Soundness
3

### Presentation
3

### Contribution
3
