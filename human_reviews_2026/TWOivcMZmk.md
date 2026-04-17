# Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information

- Decision: Reject
- Scores: 2, 4, 8

## Abstract
As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in the domains of medicine and public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, while there are a number of LLM benchmarks in the medical domain, currently little is known about LLM knowledge within the field of public health. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest proprietary LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, while there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a benchmark for public health by combining some of the existing datasets and newly synthesized data. The goal is to test LLMs for medical factual knowledge along with human/ethical/emotional aspects. Then they benchmark on most of the frontier models and use human experts + LLMs to judge the output by using some dimension weighting.

### Strengths
- Empathy and ethical alignment are generally missing from traditional benchmarks like MedQA/PubMedQA, so the motivation is timely
- A human baseline is used for comparison, which helps to gauge the expectation in real life

### Weaknesses
- Most of the datasets are somewhat rehashing existing ones. They have claimed to provide new data but the generation is somewhat contrived. Like varying the gender/age/race to a controlled medical questions. This lacks some theoretical grounding and formalism, and no sensitivity analysis or ablation study. In addition, the data distribution is unknown, like no quantitative description of dataset size, balance, or topic diversity (e.g., how many cardiology vs. psychiatry cases?). And the ground truth is inherently subjective but treating clinician-authored empathy dialogues as only source of truth can also introduce bias.
- Evaluation framework is sort of ambiguous. The evaluation procedure blends human and automated scoring, but the exact aggregation pipeline is poorly specified. For example: 1) How are inter-rater disagreements handled statistically (beyond Cohen’s κ)? 2) Are human ratings averaged or weighted by expertise? 3) Are automated metrics calibrated on the same scale as human ratings? There is also no error bars or confidence intervals are reported in performance charts. This gives lots of difficulty for reliability and reproducibility

### Questions
- How is “hallucination rate” normalized to the same 0–100 range as “empathy”?
- Are composite scores weighted by inverse variance or task importance? No justification is given.
- If models were anonymized, how was temperature or prompt configuration controlled?

### Soundness
2

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
4

### Summary
The paper proposes the first public health question and answering (QA) benchmark to address a major gap in evaluation of LLM capabilities, as existing benchmarks focus on the medical domain. The paper leverages existing automated approaches to generating knowledge-grounded multiple choice QA pairs to generate over 8000 questions to reflect potential public health queries. Domain experts evaluated a random sample of 800 questions to construct the reviewed portion of the benchmark. 24 LLMs are evaluated on this newly created benchmark, with the highest achieving > 90% accuracy and many outperforming human performance.

### Strengths
* First comprehensive QA benchmark in the public health domain with a focus on UK health guidance (methodology can generalize to other countries based on the framework) with over 8090 multiple choice QA questions from 687 documents.
* Automated, scalable pipeline that can be updated with guidance changes and human expert review of 10% questions
* Evaluation of 24 LLMs covering both proprietary and open-weight LLMS on MCQA and free-form responses (based on questions from MCQA)
* Benchmark with human baseline performance (general audience was given access to search engine)

### Weaknesses
* Mismatch in the motivation and the results - one of the biggest claims in the introduction is that public health QA benchmark is necessary as there is risk of hallucinations or incomplete information. However, the results are quite stellar on the benchmark (and already exceeding human baseline). As such, the MCQA results seems to undermine the original motivation and need for the benchmark. Instead, the free-form response seems to be under explored given the performance differences. 
* There are several concerns about the quality of the benchmark
  - Estimated error of 5.5% on final benchmark might be quite high given that some of them are invalid MCQA pairs (e.g., multiple possible answers would be sufficient)
  - Automated generation using a single LLM suggests benchmark might be biased / limited by a single model's capabilities 
  - Moderate inter-annotator agreement (Cohen's kappa of 0.39) on the expert-subset verified version suggests there might be significant subjectiveness in the actual questions themselves
* The evaluation methodology also suffers from some limitations
  - Free form evaluation relies on a single LLM as a judge (GPT-40-Mini) without any additional validation or assessment of its performance
  - How do we know if the performance for LLM is high due to memorization or genuine knowledge? Is there a way to come up with some MCQAs that the model has likely not seen to assess this performance?
  - The error analysis is quite sparse, especially given the substantial performance differences between free-form response and the MCQAs.  The question is whether this is due to the question construction, how the answers are evaluated, or some other reasons -- this would be essential for identifying a benchmark that can truly test LLM capability as much of the field is starting to move beyond MCQAs (especially in the medical domain). 

Minor points:
* The numbers flip between 800 and 760 for evaluation of the MCQA questions by human experts, which is slightly confusing. Based on the context, the 760 is the final version that contains only valid questions, whereas the full one has invalid questions.
* The ability to compare performance across the different versions of the dataset is very difficult, even with Figure 1.

### Questions
1. Can you clarify how the strong MCQA results (>90% for top models, exceeding human baseline) might support rather than undermine the need for this benchmark? 
2. How do you disentangle memorization from genuine reasoning capability when source documents are likely in training data? Can you provide evidence that models are reasoning about guidance rather than recalling it?
3. The 17-63 percentage point drops from MCQA to free-form are striking. What are the potential causes for these significant drops? 
4.  The initial annotator agreement on the 800 questions is quite low. Can you provide more details on what caused disagreements and how the reconciliation process worked?
5. How does difficulty compare to existing medical benchmarks (MEDMCQA, USMLE) performance?
6. The percentage of error in the final benchmark seems quite high, especially if consider there are multiple possible answers as one of the reasons. How does this compare against other automated QA generation in terms of error rates? Is this due to the LLM used to generate or sensitivity to prompts?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes PubHealthBench, a benchmark designed to assess LLM knowledge of UK public health information. Recognizing that existing medical benchmarks overlook public health, the authors compiled over 8k questions derived from 687 UK government guidance documents, supporting both MCQA and free-form question answering. They evaluated 24 LLMs, finding that top proprietary models (GPT-4.5, GPT-4.1, and o1) achieved over 90% accuracy in MCQA tasks, surpassing human performance using search engines. However, in free-form responses, performance dropped below 75%, highlighting ongoing limitations in open-ended reasoning. Overall, while SOTA LLMs demonstrate strong factual accuracy in structured formats, additional safeguards are needed before relying on them for unstructured public health communication.

### Strengths
1) An original benchmark on public health (here, UK).

2) Both MCQA and open questions.

3) A large evaluation using state-of-the-art LLMs. Also, the authors made the effort to choose an open model (OLMo-2).

### Weaknesses
1) Only part of the benchmark has been manually checked (10% of the benchmark).

2) The generation of benchmark questions relies quite heavily on the use of LLM. However, relying on a manual verification of part of the benchmark helps to counterbalance this point.

3) Why not choose to evaluate LLMs adapted to the medical field?

4) Even though we are aware of their limitations, it could have been interesting to have complementary metrics to the LLM-as-a-judge for open-ended questions. A human evaluation could also have supplemented the study, even though we still have limited experience with the LLM-as-a-judge approach.

### Questions
1) I read that the full benchmark dataset should be available, but I did not find it. Maybe providing the link in the introduction would help?

2) Did you evaluate the pdf extraction text process?

3) Why choose Llama-3-70bn-Instruct model for the benchmark construction?

4) "For the 21 models run on the PubHealthBench-Full" -> not clear since author mentioned over 24 models earlier in the paper.

### Soundness
3

### Presentation
3

### Contribution
3
