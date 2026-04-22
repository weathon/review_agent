# AirQA: A Comprehensive QA Dataset for AI Research with Instance-Level Evaluation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
The growing volume of academic papers has made it increasingly difficult for researchers to efficiently extract key information. While large language models (LLMs) based agents are capable of automating question answering (QA) workflows for scientific papers, there still lacks a comprehensive and realistic benchmark to evaluate their capabilities. Moreover, training an interactive agent for this task is hindered by the shortage of high-quality interaction trajectories. In this work, we propose AirQA, a human-annotated comprehensive paper QA dataset in the field of artificial intelligence, with 13,956 papers and 1,246 questions, that encompasses multi-task, multi-modal and instance-level evaluation. Furthermore, we propose ExTrActor, an automated framework for instruction data synthesis. With three LLM-based agents, ExTrActor can perform example generation and trajectory collection without human intervention. Evaluations of multiple open-source and proprietary models show that most models underperform on AirQA, demonstrating its quality. Extensive experiments confirm that ExTrActor consistently improves the multi-turn tool-use capability of small models, enabling them to achieve performance comparable to larger ones.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce M4PQA, a human-annotated comprehensive paper QA dataset in the field of artificial intelligence, with 13,948 papers and 1,246 questions, that encompasses multi-task, multi-modal and instance-level evaluation. They also introduce EXTRACTOR, an automated framework for instruction data synthesis which can perform example generation and trajectory collection without human intervention.

### Strengths
- Expert annotated from students with AI expertise (n=26).

- Expert validated study for difficulty (n=3).

- Paper well written.

- Experiments well designed and good ablations.

- Authors address shortcomings of other scientific papers benchmarks.

- Authors show that performance raises consistently as data scales up, highlighting the scalability of the framework.

- Function-based evaluation into QA domain which is systematic and reproducible.

### Weaknesses
- Even though the work is good and addresses gaps in other datasets I am not confident how significant is the novelty compared to other similar works. Based on table 3, one can run on different benchmarks from the literature to assess on different question types (Single, Multi, Retrieval, ..., Image Form. Meta). This is the main reason for my score. 

- How is extractor validated? Any human intervention?

- Maybe if the annotators had more specific instructions like identifying the key components of the papers and coming up with key questions would improve the quality of questions in terms of usefulness? Perhaps identifying questions that require some kind of thinking/logical deduction?

### Questions
- Any reason not including ICML, AAAI or big vision conferences (e.g., CVPR)?

- Small typo: figure 3 “based on it” 

- The main purpose of extractor is to automatically produce questions/trajectories/answers that are then used to finetune the LLM on these tasks? If you want to use it to extend your proposed dataset you still need a human expert in the loop to make sure the questions/trajectories/answers are correct.

- Did you perform SFT only? Any RL methods that can be potentially be used (e.g., GRPO)?

- Did you use the same papers as your dataset with extractor to generate questions/trajectories/answers? Which model was used for generating these trajectories?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces M4PQA, a question-answering dataset containing 1,246 questions derived from documents in the field of artificial intelligence. The dataset is designed to evaluate the capabilities of large language model (LLM) agents in answering questions about research papers. It was manually curated by students with AI expertise and includes four categories of questions such as single-document, multiple-document, retrieval, and comprehensive, and targets different components of research papers (e.g., text, tables, figures). The authors demonstrate that state-of-the-art models such as GPT-4o and Qwen2.5 struggle to effectively answer questions from M4PQA under various prompting and agentic settings. Furthermore, the paper proposes Extractor, a framework that leverages LLMs to generate synthetic QA trajectories for agentic fine-tuning.

### Strengths
- The paper addresses a well-motivated and relevant problem.
- The proposed dataset is comprehensive, encompassing a diverse range of question types and context variations.
- The evaluation is thorough, testing the dataset across multiple models and agentic approaches.

### Weaknesses
- The effectiveness of the proposed extractor framework is not convincingly demonstrated. Since the framework is entirely LLM-based, its reliability and the quality of the generated questions are uncertain. The paper lacks dedicated experiments evaluating the quality of the synthetic trajectories produced. In addition, Table 11 indicates potential quality issues with the Extractor framework. A comparison with other automatic question-generation methods could have provided stronger empirical support.
- If scalability is a primary motivation for introducing the extractor framework, a cost or efficiency analysis should be included to support this claim.

### Questions
- How reliable are the LLM-based (subjective) evaluation functions? Additionally, since LLM-based evaluations can be computationally expensive, doesn’t this approach introduce a significant cost whenever a method is evaluated on M4PQG? It would be helpful if you could provide an approximate cost estimate.
- From line 430, it appears that Table 6 presents the results of the extractor framework. If this is correct, could you clarify why the caption currently states M4PQG?
- In Section 3.2, could you elaborate on how the context extraction process is performed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces M4PQA, which is a framework for AI research question-answering. The benchmark is a human-annotated Multi-Modal Multi-Task Multi-Paper Question Answering dataset. The authors recruited 26 students and collected 1,246 examples and 13,948 papers in AI and created a set of questions with different types (single, double, etc.) and element categories (text, image, table, etc).

### Strengths
M4PQA targets research QA and retrieval rather than generic QA. The function-based, instance-level evaluation is a sensible move away from free-form semantic metrics and reduces grading ambiguity. The agentic baselines and the released action space/environment help standardize comparisons for tool-using agents, and the dataset statistics (balance across question types; mix of elements) suggest measured coverage.

### Weaknesses
Evaluation relies substantially on synthetically generated content/trajectories. The paper does not describe where human work ends and LLM synthesis begins. The “26 students” section says annotators read papers, pose questions, and wrap evaluators, but elsewhere EXTRACTOR agents generate QAs and trajectories; it remains unclear, per item type, whether humans created or only vetted the questions and how often LLM-authored items passed review. Labeling is described as function-based with optional subjective paths; however, the paper should state which splits or item types invoke an LLM judge. 

Reported performance also appears tightly coupled to tool access (search capabilities, for example). Table 4 shows large jumps when agents can query the environment; that’s informative, but it confounds “reasoning” with “tool use.” 

Qwen fine-tuning with EXTRACTOR trajectories predictably improves performance of small models, but it doesn’t clarify whether gains come from better tool formats, better retrieval behavior, or overfitting to the dataset’s answer-format conventions; no cross-family or cross-benchmark transfer is shown. 

The dataset appears calibrated to be hard, which is fine, but the paper implicitly values difficulty without probing whether item construction systematically reduces performance on weaker models and favors stronger ones (e.g., formatting-sensitive outputs).

### Questions
No particular question at this time.

### Soundness
3

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
This paper introduce lots work to 
0) frame question answering as multi turn agentic calling 
1) create enviornments to generate agentic tractergories recognize the limits of human-input
2) showcase scaling law to small 7b/14B model

### Strengths
There are novelty associated with this approach, constructing tools like vector search, SQL. The questions themselves 

There are a human baseline that helps with establishing a good baselines. this is a really important baseline

The paper also mentioned error-analysis of gpt-4o, and classifier them into different classes, i think the lack of context and over confidence are really important to address in future work.

### Weaknesses
The multi-turn setup might be more ideal for RL instead of SFT, but I recognized most of the work are more on engineering the multi agent prompts, setting up the right environment, and reading through raw responses.

### Questions
Why do you only compute loss on the model's last output? this means you are not doing any supervision to the actual tractegory 

For the human baseline, does the error distrubtion is super similar to the gpt-4o error on this benchmark?

### Soundness
4

### Presentation
3

### Contribution
3
