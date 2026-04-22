# Are LLMs Ready for English Standardized Tests? A Benchmarking and Elicitation Perspective

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Large language models (LLMs) are transforming education by enabling powerful tools that enhance learning experiences, particularly in the context of English Standardized Tests (ESTs), which generate significant commercial value in the education industry. However, their fundamental problem-solving capabilities remain largely underexplored. In this work, we evaluate the performance of LLMs on ESTs across a diverse range of question types. We introduce EstBOOK, a comprehensive benchmark designed to evaluate the capabilities of LLMs in solving EST questions. EstBOOK aggregates five widely recognized tests, encompassing 29 question types and over 10,576 questions across multiple modalities, including text, images, audio, tables, and mathematical symbols. Using EstBOOK, we systematically evaluate both the accuracy and inference efficiency of LLMs. Additionally, we propose a breakdown analysis framework that decomposes complex EST questions into task-specific solution steps. This framework allows us to isolate and assess LLM performance at each stage of the reasoning process. Evaluation findings offer insights into the capability of LLMs in educational contexts and point toward targeted strategies for improving their reliability as intelligent tutoring systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores whether LLMs possess fundamental problem-solving capabilities for English Standardized Tests (ESTs), arguing that this assessment is essential before deploying them in educational applications like tutoring and grading. The authors introduce ESTBOOK, a benchmark with 10,576 questions spanning 29 question types from five major tests (SAT, GRE, GMAT, TOEFL, IELTS) across multiple modalities (text, images, audio, tables, mathematical symbols). They evaluate six multimodal LLMs using three prompting strategies (ICL, CoT, ToT) and propose a novel breakdown analysis framework that decomposes questions into six task-specific reasoning steps aligned with human test-taking strategies: Evidence Finding, Semantic Reasoning, Structural Reasoning, Data Interpretation, Numeric Calculation, and Comparative Judgment.

Key findings reveal that LLMs exhibit highly inconsistent performance, ranging from a precision of more than 90\%  on some tasks to less than 20\% on others. Advanced prompting strategies like ToT do not consistently improve results and sometimes introduce errors. Although LLMs excel at initial problem formulation (up to 97\% accuracy), performance degrades significantly in subsequent reasoning stages that require causal inference or multimodal integration. Inference time shows no correlation with correctness. The authors conclude that current LLMs remain inadequate as reliable educational assistants for EST tasks, although they identify specific strengths and weaknesses to inform future development.

### Strengths
1. ESTBOOK represents a substantial contribution to educational AI evaluation, covering five major standardized tests with questions across diverse formats and modalities. Unlike synthetic benchmarks, it uses real EST questions that reflect authentic assessment contexts, providing ecological validity for claims about educational readiness.

2. The task decomposition into six categories with step-by-step evaluation is a significant methodological contribution that goes beyond coarse-grained accuracy metrics. By isolating specific reasoning stages and aligning them with documented human test-taking strategies (Sect 3.2 and Appendix C), the framework provides actionable diagnostic insights about where LLMs succeed or fail, which is valuable for targeted model improvement.

3. Experimental methodology is thorough. I really liked the inclusion of human testers, and the use of statistical tests (Mann-Whitney U) for inference time analysis adds rigor. The extensive case studies effectively illustrate failure modes qualitatively.

4. The dataset and findings have immediate implications for practitioners developing AI tutoring systems and for researchers working on educational applications.

### Weaknesses
1. Data contamination issues are not adequately addressed. The benchmark uses "publicly available educational materials and official preparation resources" that may have been included in LLM training corpora, particularly for closed-source models. The paper provides no decontamination analysis, memorization checks, or attempts to create novel EST-style questions. Given the well-documented data contamination issues in LLM benchmarking, this omission is a major methodological weakness that could substantially inflate performance estimates. The field has moved toward more rigorous contamination testing, and this paper does not meet current standards.

2. Despite testing across 29 question types, 6 models, and 3 prompting strategies, the paper does not discuss or apply corrections for multiple comparisons, which increases the risk of spurious findings. Inter-model comparisons lack statistical tests, making it unclear which performance differences are meaningful.

3. The title and abstract claim LLMs are "not ready" for EST tasks, yet results are highly heterogeneous: some models achieve $>90$\% accuracy on multiple tasks (e.g., GPT-5 on TOEFL FI: 98.5\%, SAT II: 86.4\%). The gap between best and worst models is substantial (e.g., 8.8\% to 93.7\% on GMAT IR), suggesting readiness depends critically on which model and which task. The conclusions should be more nuanced, acknowledging which tasks are "good enough" and which require improvement, rather than making blanket claims about LLM inadequacy.

4. Domain specific finetuned models are not used in the comparison. Does SFT help?

### Questions
1. Can you include contamination analysis?

2. Which exact models and versions were evaluated? Please provide complete model identifiers including API versions, dates accessed, and checkpoint hashes.

3. Could you evaluate models that can process audio directly instead of STT followed by LLM? How much performance is lost in transcription? Does it matter?

4. Given that ToT sometimes degrades performance (Case Study II), have you explored other advanced prompting methods? Were the prompts optimized per-task or are they generic templates? Could task-specific prompt engineering improve results?

5. In Task I (Evidence Finding), what happens to downstream task performance when models make errors in the "Identify Subject" step? Can you provide any results showing realistic multi-step reasoning where errors propagate, rather than assuming perfect intermediate steps?

6. Based on your findings, which specific EST tasks are "good enough" for current LLM-based tutoring systems and which should be avoided? What concrete architectural or training improvements would you recommend to address the identified weaknesses?

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
The paper introduces a benchmark for evaluating LLMs on English Standardized Test (EST) questions across five major exams. The authors compare the performance of frontier LLMs under different prompting strategies and compare them to human test-takers. The proposed breakdown analysis framework decomposes each question into structured reasoning steps, allowing the model’s performance to be measured at each stage. Using this framework, the paper provides empirical insights into LLMs’ strengths (e.g., parsing questions) and weaknesses (e.g., multi-hop reasoning and numeric calculation) in the EST domain.

### Strengths
- ESTBOOK aggregates five major exams (SAT, GRE, GMAT, TOEFL, IELTS) and captures diverse modalities, including text, math symbols, images, tables, and audio. This gives the benchmark authenticity and multimodal variety. 
- The proposed breakdown analysis framework, which decomposes each question into human-like solution steps, helps to isolate LLM performance at each step. Such analysis is useful, as it pinpoints whether a model failed because it misunderstood the question or because it couldn’t reason or perform math. 
- The paper identifies interesting trends through experiments; e.g., prompting strategy effectiveness varies by task, and LLMs underperforms humans on tasks requiring reasoning.

### Weaknesses
- The paper’s novelty with respect to prior educational LLM benchmarks is limited. For example, AGIEVAL (Zhong et al., 2024) introduced a benchmark of standardized exams including SAT and LSAT, finding GPT-4 achieves 95% on SAT Math. TOEFL-QA is a dataset of TOEFL listening comprehension questions introduced by Tseng et al. (2016), which has been used to test machine listening comprehension (Chung et al, 2018).

      Chung, Y. A., Lee, H. Y., & Glass, J. (2018). Supervised and Unsupervised Transfer Learning for Question Answering. In Proceedings of NAACL-HLT (pp. 1585-1594).

      Tseng, B. H., Shen, S. S., Lee, H. Y., & Lee, L. S. (2016). Towards Machine Comprehension of Spoken Content: Initial TOEFL Listening Comprehension Test by Machine. In Proc. Interspeech 2016 (pp. 2731-2735).

      Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., ... & Duan, N. (2024). AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models. In Findings of the Association for Computational Linguistics: NAACL 2024 (pp. 2299-2314).

- A related concern is the difficulty level of the exam tests. GPT-4 can already achieve 95% on SAT Math (Zhong et al., 2024). Recently, the Humanity’s Last Exam (HLE) was created specifically to address the saturation of benchmarks like MMLU by providing expert-written questions. Many EST questions are arguably simpler than MMLU’s academic subjects, and some ESTBOOK sections may be too easy for frontier models. For example, GPT-5
achieves 90%+ on several question types in Table 2, and GPT-4V was reported to exceed 89% on SAT Writing questions in some prompts. The authors should address whether parts of their
benchmark are already nearing saturation and how that impacts long-term utility. The authors might also consider curating harder or more adversarial questions, as HLE does, to future-proof the evaluation.

      Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2020). Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

      Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., ... & Wykowski, J. (2025). Humanity's last exam. arXiv preprint arXiv:2501.14249

- Another concern is that some included questions may already be memorized by the models. The data was sourced from public practice exams and study websites, so there is a nontrivial chance that LLMs were trained on similar or identical questions. The authors did not report checks for training data overlap or model memorization. The paper would benefit from acknowledging this data contamination risk as a limitation.

- The authors restrict to objective questions with certain answers, which excludes open-ended tasks like essay writing and speaking responses. This omission is understandable given automatic grading difficulty, but it is a limitation that should be discussed. Tasks such as the TOEFL independent writing or IELTS speaking are important parts of those exams, and LLMs might have different performance and failure modes on free-form generation tasks versus multiple choice.

### Questions
- Does the Whisper audio transcription introduce biases? Also note the lack of capitalization: “we adopt…” (pg. 5). 
- Multi-model input handling could be explained better. If any model lacked a modality, did the authors convert those questions to text form or skip them?
- The breakdown methodology assumes perfect previous steps, essentially giving the model an oracle for sub questions. This is helpful for diagnosis, but in practice LLMs won’t have those hints. How can this framework then be used to improve models?
- The results show substantial variation across models and prompting methods for different question types. For example, GPT-5 did extremely well on GRE Quantitative Comparison but poorly on GRE Numeric Entry, even under the same CoT prompt (Table 2). Likewise, CoT sometimes outperformed ToT. Can you elaborate more on why certain tasks or formats caused particular models or prompts to fail?
- How were the hyperparameter values chosen, and under what justifications (Table 7)?

### Soundness
3

### Presentation
2

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
The paper introduces ESTBOOK, a benchmark evaluating large language models (LLMs) on English standardized tests (TOEFL, IELTS, SAT, GRE, GMAT), totaling over 10,000 multimodal questions. The authors assess several models (e.g., GPT-5, Gemini, Claude) using prompting strategies such as In-Context Learning, Chain-of-Thought, and Tree-of-Thought. They also propose a “Breakdown Analysis” framework to decompose reasoning steps. The results show that current LLMs perform inconsistently across modalities and reasoning tasks, suggesting gaps between linguistic fluency and true problem-solving ability.

### Strengths
1. The dataset integrates multiple standardized tests and modalities (text, math, audio, images), providing a broad view of LLM performance in educational contexts.
2.  The experiments include several leading multimodal LLMs and multiple prompting strategies, with large-scale quantitative results and case studies.
3. The “Breakdown Analysis” offers a structured way to isolate reasoning steps and visualize where LLMs fail, which can inspire follow-up diagnostic research.

### Weaknesses
1. The paper’s core contribution lies primarily in data aggregation and empirical evaluation. However, many similar benchmarking efforts already exist to assess LLM capabilities. The current results largely align with prior studies in highlighting LLMs’ limitations in information extraction, without providing new insights or deeper theoretical understanding.
2. The proposed “Breakdown Analysis” remains largely qualitative and manually defined. It lacks a formalized framework or reproducible scoring mechanism that could generalize beyond the specific tasks presented in this paper.
3. While the study thoroughly documents various failure patterns, it fails to propose or test any concrete methods or interventions to improve LLM reasoning or mitigate those shortcomings.

### Questions
1. Have you considered testing fine-tuning, curriculum learning, or adaptive prompting based on the breakdown results to demonstrate that the analysis yields actionable improvements?
2. Were these difficulty tiers (easy/medium/hard) taken from official test metadata, or inferred heuristically?
3. Could you provide statistical significance tests comparing LLMs and human baselines across tasks? Without such analysis, it is unclear whether differences are meaningful or anecdotal.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ESTBOOK, a new benchmark for evaluating large language models (LLMs) composed of 10K multimodal questions from five standardized English exams: SAT, GRE, GMAT, TOEFL, and IELTS. The benchmark uniquely integrates text, math, tables, images, and audio to test a wide range of reasoning skills. Several leading LLMs were evaluated using various prompting techniques, including In-Context Learning (ICL), Chain of Thought (CoT), and Tree of Thoughts (ToT). The research reveals that multimodal and numerical reasoning continue to be significant challenges for current LLMs. Key findings indicate that complex logical reasoning, rather than the length of the context provided, is the primary barrier to success, and that the style of prompting has a substantial impact on performance outcomes.

### Strengths
Comprehensive Benchmark: Introduces ESTBOOK, grounded in real standardized tests (SAT, GRE, GMAT, TOEFL, IELTS), ensuring practical and realistic evaluation.

Multimodal Coverage: Includes text, images, audio, tables, and math symbols for a holistic test of LLM reasoning and perception.

Systematic Evaluation: Assesses models under different prompting strategies, In-Context Learning (ICL), Chain-of-Thought (CoT), and Tree-of-Thought (ToT).

Breakdown Analysis Framework: Decomposes problems into stepwise reasoning tasks to pinpoint weaknesses such as numeric and multimodal reasoning failures.

Actionable Insights: Identifies specific performance gaps, guiding future improvements for LLM reliability and applications in intelligent tutoring.

### Weaknesses
Limited Model Scope: Evaluates only a subset of publicly available models.

Exploring standardized test performance of LLMs is no longer a novel research direction for top ML venues like ICLR. This study also focuses solely on objective questions, omitting subjective tasks like essays or speaking, which are key to real standardized tests.

Builds ESTBOOK from public test-prep content, which may differ in style and difficulty from actual exam conditions.

### Questions
Did you check for potential training-data overlap between your benchmark items and LLM pretraining corpora?

How does your stepwise evaluation handle error propagation between reasoning stages?

Can you share documentation verifying that all benchmark materials are used within licensing terms?

### Soundness
3

### Presentation
3

### Contribution
2
