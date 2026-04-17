# BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
As large language models (LLMs) evolve into web-interacting agents, their ability to retrieve and reason over real-time information has become a crucial benchmark for general intelligence. However, existing benchmarks such as BrowseComp focus solely on English, neglecting the linguistic, infrastructural, and retrieval-specific challenges posed by other information ecosystems—particularly the Chinese web. We present BrowseComp-ZH, a high-difficulty, natively-constructed benchmark designed to assess LLM agents’ web browsing abilities in Chinese. Rather than translating from English, all questions in BrowseComp-ZH are written from scratch by native speakers to reflect authentic information-seeking behaviors and cultural contexts. The dataset comprises 289 multi-hop questions across 11 diverse domains, each reverse-engineered from a short, verifiable answer and filtered through a twostage quality control pipeline to ensure retrieval hardness and answer uniqueness. We evaluate over 20 leading LLMs and agentic search systems. Despite strong language and retrieval abilities, most models perform poorly: many score below 10% accuracy, and only a few exceed 20%. Even the best system achieves just 42.9% accuracy. These results highlight the considerable difficulty of BrowseComp-ZH, where success requires not only robust retrieval strategies but also advanced multihop reasoning and information reconciliation—abilities that remain challenging for current models. BrowseComp-ZH thus serves as a stress test for web-interactive LLMs beyond English, offering a rigorous and linguistically diverse evaluation framework to guide future research on multilingual agent capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces BrowseComp-ZH, a benchmark for evaluating LLM agents’ web browsing and information-seeking abilities specifically within the Chinese web ecosystem. The dataset comprises 289 human-authored, multi-hop, multi-constraint questions across 11 knowledge domains, meticulously reverse-engineered from verifiable answers. The paper also provides thorough statistics, calibration analysis, multiple failure case studies, and benchmark reproducibility details.

### Strengths
- The paper fills a clear gap by constructing the first natively-annotated Chinese web browsing benchmark, avoiding translation artifacts and capturing authentic linguistic, cultural, and infrastructural challenges unique to the Chinese internet.
- The reverse-design pipeline is impressively meticulous, resulting in a high-quality, stress-test-style dataset.

### Weaknesses
- At only 289 questions, BrowseComp-ZH is considerably smaller than many contemporary benchmarks. This small scale could weaken robustness for fine-grained model assessment or pre-train/fine-tune settings in the future.
- Despite employing multi-agent and human-in-the-loop processes, the guarantee of a single unique answer still depends on current model/retrieval limitations and annotator creativity.

### Questions
- How were the final domain-specific question counts determined? Is there a risk of over-representation of certain topics?
- How can the annotators ensure that the difficulty of the problem is sufficient and that this benchmark has the ability to evaluate the entire RAG system?

### Soundness
3

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
5

### Summary
This paper introduces BrowseComp-ZH, a new, high-difficulty benchmark designed to evaluate the web browsing and information retrieval capabilities of Large Language Models (LLMs) specifically within the Chinese web ecosystem.

Key Contributions:
* Natively Constructed: It consists of 289 questions written from scratch by native Chinese speakers to reflect authentic user behavior, rather than being translated from English.

* High Difficulty: The questions are multi-hop, requiring complex reasoning and the ability to synthesize information from multiple sources.

* Rigorous Design: Questions were reverse-engineered from verified answers and passed through a two-stage quality control process to ensure they are difficult to solve and have unique answers.

### Strengths
1. Fills a Critical Gap: It’s the first benchmark for evaluating LLMs’ web-browsing abilities in the Chinese ecosystem, avoiding flaws of translated English benchmarks by using native Chinese questions that reflect linguistic, cultural, and Chinese web-specific traits. 

2. Rigorous Construction: Its 289 multi-hop questions (11 domains) go through strict quality control—reverse-engineered from answers, tested on major search engines to ensure difficulty, and validated for answer uniqueness. 

3. Insightful Evaluation: It assesses over 20 systems (open/closed-source models, commercial tools) and includes human baselines (17.6% accuracy), revealing key findings (e.g., reasoning and multi-round retrieval boost performance) to guide LLM improvement. 

4. Practical Guidance: Serves as a targeted stress test for non-English web LLMs, aiding the development of multilingual AI agents for real-world Chinese web scenarios.

### Weaknesses
1. Small Dataset Scale: With only 289 questions, it is smaller than English counterparts like BrowseComp, though the authors note this stems from high curation costs. 

2. Incomplete Answer Uniqueness: Despite strict checks, the reverse-design approach cannot fully guarantee no alternative valid answers exist for some questions. There are also some obvious bugs in the dataset that need to be fixed.

3. Compared to BrowseComp, BrowseComp-ZH is insufficiently challenging. It is essentially a simplified Chinese adaptation that lacks originality.

### Questions
See Weaknesses

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
This paper introduces BrowseComp-ZH, a native Chinese benchmark for evaluating LLMs’ web browsing and reasoning abilities. The dataset contains 289 expert-authored, multi-hop questions across 11 domains, validated for retrieval difficulty and answer uniqueness. Over 20 models—including open-, closed-source, and commercial agents—are evaluated, showing generally poor performance (most <10% accuracy), while multi-round retrieval systems like DeepResearch achieve the best results. The benchmark offers a rigorous, culturally grounded evaluation of LLM agents in the Chinese web ecosystem.

### Strengths
- The benchmark offers a rigorous, culturally grounded evaluation of LLM agents in the Chinese web ecosystem.
  - The benchmark is carefully constructed with dual validation for difficulty and answer uniqueness, ensuring high data quality.
  - It systematically compares over 20 systems, including human baselines, revealing meaningful performance gaps.

### Weaknesses
- The paper’s main contribution appears incremental rather than novel. From a task construction perspective, BrowseComp-ZH closely mirrors the original BrowseComp benchmark, with the primary difference being its adaptation to the Chinese web environment rather than a fundamentally new methodology or task design.
  - The benchmark contains only 289 queries, which may not be sufficient to capture the full linguistic, structural, and retrieval complexity of the Chinese internet. Such a limited number of examples could constrain the benchmark’s representativeness and the statistical robustness of the reported results.
  - The paper lacks a deeper analysis of why models perform differently on BrowseComp-ZH. It remains unclear whether the observed performance gaps stem mainly from the change of language (English → Chinese) or from the distinct characteristics of the Chinese web ecosystem. A more detailed ablation or cross-lingual comparison would strengthen the interpretation of results.

### Questions
- Have you compared model performance between BrowseComp-ZH and the original English BrowseComp benchmark? Specifically, do models exhibit consistent relative rankings or performance gaps across the two languages? Including such a cross-lingual comparison or correlation analysis would help clarify whether the observed difficulties stem from linguistic or retrieval-ecosystem differences.

### Soundness
3

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
The authors introduce BrowseComp-ZH, a new benchmark for evaluating the web-browsing ability of LLMs in Chinese. The authors created reverse-engineered, multi-constraint questions that push models to do multi-step reasoning across real Chinese web sources like Baidu, Zhihu, and WeChat articles. Expert annotators curated the dataset through several rounds of filtering to keep only challenging questions with clear, unique answers. The final dataset covers 15 diverse domains. Evaluation results show that DeepResearch hit 42.9% accuracy, human searchers averaged 17.6%, and most general-purpose LLMs stayed below 10%.

### Strengths
- This work faithfully reimplements the BrowseComp methodology in the Chinese-language ecosystem, where the data reflects how Chinese users search online.
- All queries, evidence chains, and browsing steps are authored natively in Chinese, not translated.
- Reporting calibration alongside accuracy improves interpretability of model performance.

### Weaknesses
- The benchmark creation process almost exactly mirrors BrowseComp’s methodology, and the contribution of this work feels somewhat incremental. The core takeaway is essentially: current LLMs (agents) perform poorly when browsing in Chinese.
- The idea of cultural groundedness is not empirically verified. Checking how often agent trajectories only involve Chinese sources would provide some insights on how much of the task is really dependent on navigating Chinese sources. The paper claims all queries, evidence chains, and browsing steps are “authored in Chinese,” but does not verify that all necessary evidence exists exclusively on Chinese websites.
- There is no analysis of failure modes. An important question to explore is how often errors stem from retrieval shortcomings versus misinterpretations of Chinese cultural or linguistic context.

### Questions
- Could you provide more details on the human baseline experiment? How were the annotators hired, and how were examples distributed among them? Were there cases where people gave up?

### Soundness
2

### Presentation
3

### Contribution
2
