# On Path to Multimodal Historical Reasoning: HistBench and HistAgent

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Recent advances in large language models (LLMs) have led to remarkable progress across various domains, yet their capabilities in the humanities, particularly history, remain underexplored. Historical reasoning poses unique challenges for LLMs, involving multimodal source interpretation, temporal inference, and cross-linguistic analysis. Existing general-purpose agents perform well on many current benchmarks but lack the domain expertise needed to address complex historical questions.
To address this gap, we introduce HistBench, a new benchmark of 414 high-quality and carefully-reviewed questions stratified by difficulty and designed to evaluate LLM's capacity for historical reasoning. The tasks span a wide range of historical problems—from factual retrieval based on primary sources to interpretive analysis of manuscripts and images, to interdisciplinary challenges involving archaeology, linguistics, or cultural history. Furthermore, the benchmark dataset spans 29 ancient and modern languages and covers a wide range of historical periods and world regions. Finding the poor performance of LLMs and other agents on HistBench, we further present HistAgent, a history-specific agent equipped with carefully designed tools for OCR, translation, archival search, and image understanding in History. On HistBench, HistAgent based on GPT-4o achieves an accuracy of 27.54% pass@1 and 36.47% pass@2, significantly outperforming LLMs with online search and generalist agents, including GPT-4o (18.60%), DeepSeek-R1(14.49%), Grok 3(17.63%) and Open Deep Research by smolagents(20.29% pass@1 and 25.12% pass@2). These results highlight the limitations of existing LLMs and generalist agents and demonstrate the advantages of HistAgent for historical reasoning. Notably, HistAgent also achieves 60.00% pass@1 accuracy on the GAIA benchmark, showing that domain-specific customization doesn't hinder HistAgent's competitive performance on real-world general tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper ventures into the humanities to address the unique challenges that historical reasoning poses for current AI agents. The authors introduce HistBench, a new benchmark of 414 expert-reviewed questions designed to test multimodal historical reasoning. The questions are notable for their diversity, spanning 29 languages, multiple modalities (including ancient manuscripts and images), and various historical periods. 

Finding that existing models perform poorly on this new benchmark, the authors developed HistAgent, a specialized agent equipped with a suite of history-centric tools. These tools are designed for specific historical research tasks, such as performing OCR on ancient scripts, searching through digital archives, and translating obscure texts. On HistBench, HistAgent outperforms generalist agents and powerful foundation models, even those with web search capabilities. This highlights the necessity of domain-specific tools for complex, multimodal reasoning.

### Strengths
1. **High-Quality and Challenging Benchmark:** A major strength of this paper is the introduction of HistBench, a well-designed and challenging benchmark. The questions are stratified by professional historians using specific and sensible criteria, which adds to the dataset's credibility. The benchmark is also well-distinguished from others in the field, as effectively shown in Table 1. 

2. **Novel and Diverse Data:** The dataset contains data from obscure sources, which is a significant strength as it's difficult to find data that models haven't already seen. The data is also varied in its input modalities and sources, making it a robust test of a model's capabilities. 

3. **Well-Implemented Agent:** Although not a new idea, the Literature Search Agent is well-implemented and appropriate for the tasks at hand.  The agent's ability to navigate scholarly databases and parse academic PDFs is a crucial component of its success.

### Weaknesses
1. **Limited Scale of Evaluation:** The primary weakness of this paper is the small scale of the evaluation. With just 414 questions in the dataset, the evaluation suite is quite small. This is further compounded by the limited validation on other datasets, with only 3 candidates for HLE and 2 for GAIA. 

2. **Limited Model Diversity:** The evaluation is missing several major LLM players, such as Gemini, Claude, and Grok 4. Furthermore, there are no evaluations of open-source models, which would have provided a more comprehensive view of the current state of the field. The HistAgent is based on GPT-4o, which is a bit dated at this point. 

3. **Lack of In-Depth Analysis:** The analysis of the results is lacking and often feels like a verbal description of the results table.  A deeper dive into the reasons for the observed performance differences would have been more insightful. For example, while the results show that the agent doesn't always outperform existing models with search, there is no discussion as to why this might be the case. 

4. **Potential for Overfitting to the Benchmark:** As HistAgent was specifically designed to address the challenges presented in HistBench, there is a risk that its strong performance is, to some extent, a result of "teaching to the test." While the specialized tools are well-motivated, a more robust evaluation would involve testing HistAgent on a separate, held-out set of historical reasoning tasks that were not considered during its development.

### Questions
How was it ensured that the data sources are not widely available or are distinct from those that may have gone into the pre-training of the models?

### Soundness
2

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
This paper presents HistBench, a benchmark of 414 expert-curated questions spanning 29 languages and five modalities (text, manuscripts, images, audio, and video) to evaluate historical reasoning in large language models. The authors also propose HistAgent, a domain-specialized system built on GPT-4o that integrates OCR, translation, academic search, and multimodal analysis tools to assist in history-related tasks. On HistBench, HistAgent achieves 27.5% pass@1 and 36.5% pass@2 accuracy, outperforming generalist agents such as ODR-SmolAgents and GPT-4o with web search. It also maintains strong performance on general benchmarks like GAIA, demonstrating competitive generalization.

### Strengths
Originality: The paper fills an important gap by introducing a domain-specific benchmark for the humanities, focusing on multimodal and multilingual historical reasoning. While earlier efforts like HLE or HiST-LLM covered historical knowledge, none provided this level of depth or tool-grounded evaluation.

Quality: The benchmark design is robust, involving domain experts, stratified difficulty levels, and rigorous three-stage quality control (screening, LLM difficulty filtering, and expert review).

Clarity: The manuscript is dense but logically structured, with clear tables and figures illustrating dataset diversity and architecture. 

Significance: HistBench sets a new standard for evaluating reasoning in historical and humanistic contexts. HistAgent’s modular design shows how domain specialization can rival or outperform larger closed models, which could inspire similar efforts in other fields such as archaeology or linguistics.

### Weaknesses
1. The dataset size (414 items) limits statistical granularity across 29 languages, especially for low-resource ones. Scaling beyond the pilot phase would strengthen claims of coverage.

2. The results could include more fine-grained analysis, such as performance by language, modality, or reasoning dimension.

3. The architecture section is somewhat heavy on engineering detail but could benefit from clearer ablation results demonstrating which tools contribute most.

### Questions
Have you measured the contribution of each HistAgent sub-module (OCR, translation, scholarly search) through ablations?

Does HistAgent generalize to other humanities disciplines, such as art history or linguistics?

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
This paper introduces HistBench, a benchmark for evaluating historical reasoning and HistAgent, a specialized agent designed to solve it. The authors highlight the lack of a dedicated historical reasoning benchmarks involving complex multimodal (manuscripts, images, audio) and multilingual inputs. HistBench consists of 414 expert curated questions contributed by 40 people. Its primary innovation is a 3-level difficulty stratification based on six criteria rubric rather than on model performance. The dataset is collected from manuscripts, inscriptions, early printed texts, archival records, visual artifacts, and audio visual materials. 
HistAgent is a manager-specialist agent architecture that equips an GPT-4o with domain specific tools, such as scholarly search, manuscript OCR, and specialized translation. The paper's key result is that HistAgent (27.54% pass@1) significantly outperforms a generalist agent (ODR-smolagents, 20.29%) and base models (GPT-4o, Grok3, DeepSeek-R1, 14-18%) on HistBench.

### Strengths
1. The paper introduces a nuanced benchmark on historical reasoning which is an under-explored area.
2. The manual verification and difficulty annotation process is very thorough. The authors use a 3-step verification process and create a comprehensive six-question rubric for annotation.
3. The dataset is diverse with 29 languages spanning several regions and decades.

### Weaknesses
1. Misleading results are reported in Figure 1 and abstract. According to Figure 1, HistAgent performs better than base models. However, Table 3 highlights o3 and o4-mini are able to achieve much higher accuracies compared to HistAgent.
2. While HistBench is a novel dataset compared to previous works, it is still very small with only 414 questions. Additionally, the data creation pipeline is time-consuming and not very scalable.
3. The better performance of HistAgent on HistBench makes sense since the tool calls are specialized for the type of question present in HistBench. Adding specific tool calls for specific question types is bound to improve accuracy on those questions. It is unclear what the contribution of HistAgent is. Additionally, the cost of reproducing HistAgent seems very high since it requires specialized access.
4. The experiments section is lacking a wide range of models as well as ablations. The authors should try reporting the results on different types of models, architectures, question types, and modalities.

### Questions
1. The authors should explain why part of the results were not reported in Figure 1 and abstract.
2. Are the primary source materials used in the benchmark questions publicly available online or are they sourced from private archives?
3. Suggestion - Related works should be moved to the main text and should be made more thorough to better motivate the problem. 
4. Suggestion - Table 1 font size should be increased.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed a new benchmark HistBench that evaluate models on historical reasoning. This benchmark is very comprehensive, spanning 29 ancient and modern languages and covers a wide range of historical periods and world regions. The authors also proposed a new agent HistAgent with multiple different tools such as OCR to solve tasks on HistBench. On HistBench and other benchmarks, HistAgent beat other SOTA agents.

### Strengths
- This paper introduces HistBench which is very useful to assess agents' abilities to solve complex historical questions through historical reasoning. This could be beneficial to the history research community.
- The proposed agent HistAgent achieves SOTA performances on multiple benchmarks, which could be valuable resource for historians.
- The authors performed comprehensive evaluation and analysis, providing valuable insights such as highlighting the importance of tools.

### Weaknesses
- The authors didn't discuss existing work in developing agents good at solving historical questions or historical reasoning. It would be good if they could provide some literature review on what other people in the field has developed.
- The authors used different base language models (e.g. claude and gpt-4o) across different benchmarks without explaining why. It would be good if the authors could include a brief description of the reason for their choice of models.

### Questions
- What are the cost like for HistAgent and the baselines that the authors compared to?

### Soundness
3

### Presentation
3

### Contribution
3
