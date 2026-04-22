# FinSearchComp: Towards a Realistic, Expert-Level Evaluation of Financial Search and Reasoning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Search has emerged as core infrastructure for LLM-based agents and is widely viewed as critical on the path toward more general intelligence. Finance is a particularly demanding proving ground: analysts routinely conduct complex, multi-step searches over time-sensitive, domain-specific data, making it ideal for assessing both search proficiency and knowledge-grounded reasoning. Yet no existing open financial datasets evaluate data searching capability of end-to-end agents, largely because constructing realistic, complicated tasks requires deep financial expertise and time-sensitive data is hard to evaluate. We present FinSearchComp, the first fully open-source agent benchmark for realistic, open-domain financial search and reasoning. FinSearchComp comprises three tasks, Time-Sensitive Data Fetching, Simple Historical Lookup, and Complex Historical Investigation, closely reproducing real-world financial analyst workflows. To ensure difficulty and reliability, we engage $70$ professional financial experts for annotation and implement a rigorous multi-stage quality-assurance pipeline. The benchmark includes $635$ questions spanning global and Greater China markets, and we evaluate $21$ models (products) on it. Grok 4 (web) tops the global subset, approaching expert-level accuracy. DouBao (web) leads on the Greater China subset. Experimental analyses show that equipping agents with web search and financial plugins substantially improves results on FinSearchComp, and the country origin of models and tools impact performance significantly. By aligning with realistic analyst tasks and providing end-to-end evaluation, FinSearchComp offers a professional, high-difficulty testbed for complex financial search and reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
FinSearchComp introduces the first open-source benchmark specifically designed to evaluate LLM agents in realistic financial search and reasoning scenarios. The benchmark replicates real financial analyst workflows through three task types: Time-Sensitive Data Fetching, Simple Historical Lookup, and Complex Historical Investigation. Developed with input from professional financial experts and a multi-stage quality assurance process, it contains 635 annotated questions across global and Greater China markets. Overall, FINSEARCHCOMP provides a rigorous, professional-grade testbed for assessing end-to-end financial search and reasoning in LLM-based agents.

### Strengths
1. Considering the critical importance of data timeliness in financial scenarios, the benchmark is designed from this perspective to test the capabilities of LLM agents.

2. The tasks are constructed through expert human validation rather than LLM-generated content, ensuring high-quality and reliability of the benchmark questions.

### Weaknesses
1. In the case study (page 25), one example shows a web-based LLM agent answering purely from memory, without using a search tool. In real scenarios, an LLM might appear confident while fabricating both reasoning and sources. Therefore, attributing errors solely to parametric memory is inaccurate (as noted in line 377), hallucination itself is another major factor contributing to model mistakes.

2. The paper contains a large amount of content, and to save space, the authors often refer readers to the appendix for details. This makes the reading experience somewhat incomplete. If possible, more of these details should be presented in the main text.

### Questions
1. Do the three task types mentioned in the paper (i.e., Time-Sensitive Data Fetching, Simple Historical Lookup, and Complex Historical Investigation) fully capture the main information retrieval scenarios encountered by financial analysts in real-world work? Are there any important task types missing, such as forecasting, report generation, or causal reasoning?

2. From Figure 3, models without search capabilities still achieve non-trivial scores on T2 and T3 questions. The paper attribute this to parametric memory from pre-training. However, could these results also indicate that many questions in T2 and T3 do not genuinely require search to answer, thus weakening their effectiveness in evaluating search-augmented reasoning?



#####

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces FinSearchComp, an open source benchmark for evaluating financial data search and reasoning via LLM agents. The benchmark covers 3 regimes of QA tasks: 1. Time sensitive data retrieval (real time quotes); 2. Simple historical lookups (point in time facts from records); 3. Complex historical investigations (multi-hop and multi step analysis across multiple sources, ex: search for best performing period). The authors also compared an extensive array of providers and models with regional origin clearly labeled. Further more, the analysis is ablated on greater China and western markets, with experiments faceted by region and provider. The dataset is 635 questions labeled by experts, and evaluations by LLM judges are grounded in human alignment of 97%.

### Strengths
1. Human experts involved in labeling process. Quality controls are included in curation.
2. Well motivated task breakdown into realtime, search, and aggregate query. Paper provides examples in appendix across different topics. 
3. Coverage and experimental breakdown by regions and providers. 21 providers are experimented, and results highlight regional strengths. This allows for a fair and balanced evaluation of origin bias and current gaps in models for financial tool calling. 
4. Open source commitment and good evaluation of tools. Ablations are conducted in api vs web enabled products. 
5. Highlights clear SOTA gaps vs human annotations, gives a good future direction for development with the dataset (T3). 
6. Handles realtime queries, not just point in time.

### Weaknesses
1. Limited scope to search insights, and the dataset is not focused on reasoning or future looking guidance. Most analysts are focused on providing guidance with research. 
2. Limited 20 day window for search in August. Authors should consider adding different time seasons to encode seasonality in their analysis. 
3. Limited details on data sourcing and licensing. How will the open source benchmark handle possible access issues in the underlying sources, if from different providers or services. 
4. For T3, the scores are low but the paper doesn’t dive deeper into which sub-step fails. Is this a search issue, a tagging issue? Is this a reasoning failure by models? Context length issues?
5. Human labeling - details missing on how much time each labeled was given, what resources were available during the annotation process (terminals, APIs, fin service access)?

### Questions
1. For T1, T2, T3 can independent researchers fully reproduce the ground truth using the released artifacts and public sources? Is there any private information that requires subscriptions? Can you also specific what tools and time budget human labelers had during annotation?
2. Judge robustness: how did the judge alignment differ from the same model vs different families of models?
3. Time-stamping for T1: how are the authors handling live snapshots for reproducibility? Do authors encode universal time zones in their templates? How can this be reproduced for the original analysis in August vs today?
4. Could you quantify the error taxonomy in T3? Which error is most common? Misaligned fiscal calendars, units, etc?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FINSEARCHCOMP, an open-source benchmark designed to evaluate the search and reasoning capabilities of LLM-based agents in the financial domain.  FINSEARCHCOMP comprises 635 expert-curated questions divided into three realistic tasks. The dataset spans Global and Greater China markets and was developed through a rigorous quality-control pipeline involving 70 financial experts. The authors evaluate 21 models , finding that Grok 4 (web) and DouBao (web) lead the Global and Greater China subsets, respectively , though all models significantly trail human expert performance.

### Strengths
1. The benchmark's construction was overseen by 70 professional financial experts (50 annotators, 20 senior reviewers) from prominent institutions, ensuring high domain relevance and realism

2. The paper provides a broad evaluation of the current landscape by testing a large set of 21 models, which includes 12 web-based products and 9 APIs.

### Weaknesses
1. The primary novelty, financial searching evaluation, is a superficial extension that does not meaningfully advance methodologies for financial retrieval or agent benchmarking. The dataset is modest in scale and task complexity remains shallow, yielding only incremental progress beyond prior financial QA benchmarks.

2. Although the work claims to evaluate agentic capability, it does not assess actual agent frameworks or contrast general-purpose agents with financial-domain agents. Relying solely on API vs. web endpoints provides no evidence that the benchmark measures planning, tool choice, error recovery, or other core agent behaviors.

3. The exclusive use of binary 0/1 accuracy collapses retrieval, reasoning, source selection, and tool-use behavior into a single outcome, preventing attribution of failure sources and limiting diagnostic interpretability.

4. Multisource retrieval is presented as essential to financial tasks, yet the benchmark includes no metrics for source coverage, retrieval depth, or authority resolution. Thus, it remains unclear whether models perform multisource search at all, weakening a central premise of the benchmark.

5. The annotation process depends on senior expert arbitration but reports neither annotator disagreement nor arbitration rates, leaving uncertainty about reliability, transparency, and potential bias in conflict resolution.

6. Market coverage is overstated: the benchmark focuses mainly on Western and Greater China markets and omits major regions (e.g., Japan, India, emerging economies). Coupled with the limited dataset size, this constrains representativeness and community impact.

7. Although real-time correctness is highlighted, the protocol lacks detail on how temporal consistency is enforced between model inference and ground-truth snapshots, especially for volatile assets, creating concerns about reproducibility.

### Questions
See weakness above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a benchmark for evaluating LLMs on real-world financial reasoning tasks that involve real-time data retrieval, historical lookup, and multi-source analysis. It includes 635 expert-verified queries, split into 337 Global and 298 Greater China tasks, created and reviewed by 70 finance professionals. The authors evaluate 21 models and show that even top web-enabled systems still fall short of human analysts, especially on time-sensitive or multi-hop reasoning. The main value lies in its holistic setup, as it tests the complete agent process from search and tool use to reasoning and final judgment within realistic financial workflows rather than isolated question answering.

### Strengths
The useful part of this work is that it evaluates the whole agent process in a financial setting where the answer can actually change over time. The model is expected to go out to a source (web or a financial plugin), pull time aligned data, match it to the task (for example the right fiscal date or the right currency), and then produce an answer that is checked by a rubric. It covers three analyst style tasks: time sensitive queries, point in time historical queries, and longer multi period questions, so it is not only checking whether the model can retrieve a document.

The task split into time sensitive, point in time historical, and longer multi period questions is sensible and shows where models start to fail. They validated the LLM as judge setup against human labels and got about 95 percent agreement, so the scoring looks trustworthy. They also released the scoring rules and judge prompts, which makes the benchmark easier to reproduce.

### Weaknesses
A limitation of the work is the effective scale of the benchmark. Although the dataset contains 635 tasks in total, the division into two regions and three task types results in relatively small groups; most subsets contain fewer than 150 items, and the most demanding setting (T3) has only about 80 examples. This makes fine-grained comparisons across models, topics, or markets less statistically robust than the headline number suggests. The coverage is also concentrated on U.S. and Greater China markets, with limited representation of other regions, so the current release should be viewed as financially realistic but not yet globally comprehensive.

The paper is reasonably transparent in terms of evaluation methodology, since the scoring rules and judge prompts are provided. However, reproducibility of the exact numbers remains difficult, because the time-sensitive tasks rely on live market data and some of the evaluated systems are proprietary web products.

### Questions
1. Was the same LLM judge and prompt used for the global/English and Greater China/Chinese parts, and did you check whether the 0–1 scheme behaves the same across languages or for partly correct long answers?

2. Performance appears to depend on the region of both the model and its attached tools, with international systems performing better on global tasks and Chinese systems doing better on the Greater China subset. How much of these differences reflect model reasoning capability versus disparities in access to region-specific data sources or plugins?

3. Given that 635 expert tasks is already expensive, is there a plan for semi-automatic growth, and how do you see this benchmark sitting next to Finance Agent Benchmark or FinEval?

### Soundness
4

### Presentation
4

### Contribution
3
