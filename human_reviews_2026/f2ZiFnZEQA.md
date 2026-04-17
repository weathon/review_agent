# SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information.
However, unreliable search results may also pose safety threats to end users, establishing a new threat surface.
In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors.
To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. 
Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). 
Using this dataset, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 9 proprietary and 8 open-source backend LLMs. 
Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5\% for GPT-4.1-mini under a search workflow setting.
Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting.
This emphasizes the value of our framework in promoting transparency for safer agent development.
Our codebase and test cases are publicly available: https://anonymous.4open.science/r/SafeSearch.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SafeSearch, a benchmark to test how LLM-based search agents react to adding unreliable sources added to the retrieved webpages. The proposed pipeline leverages an LLM to create different risky scenarios and instantiate them via fictional webpages. Then, an LLM-judge evaluates the effectiveness of the injected webpage in affecting the reply of the search agent. The experimental evaluation shows that many LLMs, with various scaffolding mechanisms, are susceptible to these attacks.

### Strengths
- Adding search capabilities to LLM-based agents create new security vulnerabilities, which are worth studying.

- The proposed pipeline to generate test-cases is automated and can be potentially scaled to a large number of samples.

### Weaknesses
- While it'd be a useful feature of search agents to disregard unreliable sources, it is unclear whether it's realistic to detect that the generated websites are unreliable (what makes them such? is there a clear reason their information should be disregarded?), especially considering that these are artificially added to the top-5 retrieved pages, which one can expect to be popular, and thus likely reliable, sources.

- In Table 2, GPT-5 (mini) seems to be largely immune to the unreliable sources, and even open-source models like Qwen3 or DeepSeek-R1 attain reasonable results (~30% ASR). Then, it seems that the proposed benchmark may not be particularly challenging, and therefore useful, in the long term.

### Questions
- What would be the results in Table 2 without adding the unreliable sources to the search results? This would help understanding how much the injected sources need to change the agent replies.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SafeSearch, a red-teaming benchmark that stress-tests LLM-powered search agents by letting them retrieve and reason over poisoned web content. Covering 300 test cases across 5 risk types, it delivers the first large-scale safety snapshot of 15 mainstream LLMs under 3 search paradigms. Instead of running expensive and ethically risky black-hat SEO campaigns, the authors inject a single LLM-synthesised unreliable webpage into otherwise genuine search results. This keeps the test harmless to real users, cheap to run at scale, and fully reproducible.

### Strengths
- This study shifts the red-teaming focus from malicious user queries to the inherent unreliability of search tool outputs as the primary safety concern.
- This study introduces adversarial content through simulated search results, enabling controlled red-teaming without real-world SEO manipulation.
- This study conducts a systematic assessment across 300 test cases, 5 risk categories, 15 models, and 3 agent architectures to quantify vulnerabilities.

### Weaknesses
- The authors provide limited evidence regarding the quality and real-world grounding of the 300 test queries. While Appendix C validates the reliability of the LLM-as-a-Judge metric, it does not demonstrate that the queries themselves are representative, unbiased, or cover the nuanced distribution of actual user interactions

- The core problem investigated in this study, i.e., LLM systems' vulnerabilities to knowledge poisoning, has been validated in most existing RAG poisoning studies. The transition from offline to online retrieval does not alter the fundamental nature of this threat. SafeSearch is an engineering contribution to RAG security.

- The author primarily focuses on extreme attack scenarios, where the searched content is optimized for high topical relevance and persuasive impact. The lower quality and relevance of malicious content in search engine results, which is the most likely scenario encountered in practice, remains under-examined for whether the model exhibits similar vulnerabilities (Even if studied, this appears to fall within the scope of RAG).

### Questions
- How is the authenticity and representativeness of the user query scenarios and interaction flows envisioned in the experiment ensured?
- Could it be that the model simply prioritizes highly relevant content rather than specifically harmful content? Would an equally relevant but harmless AI-generated piece of content be given the same priority by the model?
- The experimental evaluation covered 15 models, but notably missing was Anthropic's Claude series.
- Is it appropriate to classify relatively simple workflows like Deep Research as an agent architecture?
- What is the difference between Agent with the setting of “LLM w/ Search Workflow” and knowledge augmented Chatbot? 
- In multi-round search scenarios, only one round is poisoned, with no additional poisoning applied in subsequent rounds. Isn't the order of poisoning also important?

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
3

### Summary
The paper focuses on the safety implications of unreliable search results on widely used search agents. It proposes a systematic, scalable, and cost-efficient red-teaming framework that leverages LLM assistants to generate test cases automatically. On the basis of this framework, the authors construct the SafeSearch benchmark, covering 5 categories of risks that include both adversarial and non-adversarial risks. The experiment results show high vulnerabilities of LLM-based search agents to unreliable search results.

### Strengths
- The authors propose an automatic red-teaming framework that eliminates the reliance on human efforts to craft test cases, making it more scalable and cost-efficient.
- Comprehensive experiments on three representative search agent scaffolds, covering search workflow, tool-calling, and deep research.
- The utility and safety goals are not at odds on capable models like GPT-5 or sophisticated scaffolds like deep search. 
- The setting of experiments is well controlled.

### Weaknesses
- Not sure if the problem of different stances in replies to the same question is caused by the presence of search tools. It seems not a common behavior for the agent to alter its response after having the access to a search tool, only accounting for 4.6% of health-related examples. So it's important to show more results, demonstrating that this is a serious problem and why it happens is not due to the unreliable inference over long contexts.
- The situations that search vendors prioritize sponsored content seem to be a more common problem. It'd be interesting to see some results targeting this case and show the influence of sponsored content on the response of search agents.
- The feasibility of using o4-mini as the generator of malicious test cases. I expect that sometimes the model has a non-zero refusal rate on generating test cases since it is sensitive to requests that contain unsafe keywords. Can you show some results on test cases generated by open-sourced models?
- Could you please show some evaluation results on the effectiveness of GPT-4.1-mini as the judge in this paper?

### Questions
No other questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SafeSearch, an automated red-teaming framework for evaluating the safety of LLM-based search agents when exposed to unreliable search results. The authors motivate their work with two in-the-wild experiments demonstrating that (1) low-quality websites are prevalent in search results, and (2) such content can significantly influence agent behavior. The framework employs a three-stage test case generation pipeline using LLM assistants, simulation-based testing by injecting unreliable websites into authentic search results, and checklist-assisted evaluation. The resulting SafeSearch benchmark contains 300 test cases across five risk categories (advertisements, bias, harmful output, indirect prompt injection, and misinformation). Comprehensive evaluation of 15 backend LLMs across three agent scaffolds reveals substantial vulnerabilities, with the highest attack success rate (ASR) reaching 90.5% for GPT-4.1-mini under a search workflow setting.

### Strengths
- Substantial number of test cases (300) covering 5 risk categories.
- The problem is well-motivated and is relatively underrepresented in the current literature.
- The threat model is clearly stated.
- Automated checklist-assisted safety evaluation.
- Valuable findings and ablation studies.
- A nice ablation for reasoning effort of GPT-5-mini and Gemini-2.5-Flash in Appendix D

### Weaknesses
- I don’t agree with the description of how RAG relates to agentic search outlined in Figure 2 and Section 2.1 (in particular in this sentence: *”Unlike RAG, which typically operates over a static, locally controlled, and well-curated corpus, search agents rely on often opaque search services that provide access to large-scale, dynamic, and up-to-date information on the Internet”*). Both RAG and agentic search can operate over the same set of documents, using the same retrieval routines. The main difference is single-turn and static in RAG vs. multi-step and dynamic queries in agentic search.
- It would be good to discuss more explicitly some challenging cases where it’s hard to judge what is true and what constitutes misinformation, fraud, “low-quality website”, etc. This seems to be a particularly relevant question because of the usage of automated LLM evaluators.
- There is a bit of discrepancy between the paper’s title and actual content. The title focuses on automated red-teaming, while the actual content focuses more on creating a static benchmark, where automated red-teaming is used as a data collection tool.

### Questions
- Why aren’t GPT-5 models boldfaced in the main table (Table 2)? They seem to be better across the board, and it would be clearer to use boldfacing for them instead of using a special background color.
- What kind of guarantees are meant here? *“As shown in Figure 6 (Left), our empirical findings demonstrate that this choice can implicitly influence the safety guarantees of search agents.”*

### Soundness
3

### Presentation
3

### Contribution
3
