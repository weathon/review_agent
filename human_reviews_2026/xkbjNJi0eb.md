# Answer matching outperforms multiple choice for LLM evaluations

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
Multiple choice benchmarks have long been the workhorse of language model evaluation because grading multiple choice is objective and easy to automate. However, we show multiple choice questions from popular benchmarks can often be answered without even seeing the question. These shortcuts arise from a fundamental limitation of discriminative evaluation not shared by evaluations of the model's free-form, generative answers. Until recently, there appeared to be no viable, scalable alternative to multiple choice---but, we show that this has changed. We consider generative evaluation via what we call answer matching: Give the candidate model the question without the options, have it generate a free-form response, then use a modern language model with the reference answer to determine if the response matches the reference. To compare the validity of different evaluation strategies, we measure agreement with human grading, by annotating responses to MMLU-Pro and GPQA-Diamond questions. We find answer matching using recent models--even small ones--achieves near-perfect agreement, in the range of inter-annotator agreement. In contrast, both multiple choice evaluation and using LLM-as-a-judge without reference answers aligns poorly with human grading. Improved evaluations via answer matching are not merely a conceptual concern---it reduces costs, and significantly changes model rankings. Multiple choice benchmarks that seem saturated start showing room for improvement when evaluated with answer matching. In light of these findings, we discuss how to move the evaluation ecosystem from multiple choice to answer matching.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper formalizes the use of LLMs in evaluation and benchmarking as a substitute for multiple choice questions. They propose to use an LLM-matcher, which is an LLM judge that has access to the reference answer (correct option per se). They show that MCQ measures a discriminative task and is vulnerable to "choice-only" shortcuts by fine-tuning a model on options only. They show that using "answer matching" and evaluating generations is well-aligned with human graders, is surprisingly cheaper in practice than MCQ, and reveals that SOTA models fall short in free-form generation on saturated MCQ benchmarks.

### Strengths
1. The paper is well-written, making all claims clear and supporting them with justifications and experimental results.
2. The work shows why MCQ is not useful for benchmarking LLMs by showing that they have discriminative shortcuts by fine-tuning a choice-only model and showing its surprisingly high performance on TruthfulQA. 
3. They clearly describe how they go from MCQ to answer matching and how it's different than LLM Judges. Importantly, the results on MATH (which can even be automatically verified) shows near-perfect alignment where MCQ itself has a much lower agreement. The same results hold for GPQA and MMLU-Pro.
4. This work will have impact on benchmarking and leaderboard efforts. The authors show how using answer matching will change the ranking of models (Fig. 6) and surprisingly, using answer matching is cheaper than MCQ because of an interesting phenomenon where models generate longer responses when having access to options. They also discuss reliability and robustness of reusing MCQ benchmarks with answer matching.

### Weaknesses
1. The paper suggests "using *smaller* models" (L. 80) or "using *recent* models" (L 470) as the matcher but provides no scaling study. The paper would benefit from a scaling study that analyses how llm matcher size, release date, and cost affect performance in terms of alignment with human graders and clarify the optimal trade-off. 

2. Using answer matching might not be feasible for all tasks or benchmarks (e.g., "which of the following options ...?" The authors also had to filter some MMLU-pro and GPQA questions manually to ensure there's a unique answer and the question can be answered without having access to questions. Adding a discussion on what kind of tasks can be restructured to be suitable for "answer matching" instead of filtering them would be beneficial in future benchmark creation efforts.

### Questions
1. Do you think answer matching can be used for tasks/question that don't have a ground-truth final answer (e.g., "List the main causes of climate change.")? What if reference responses are available? Does this mean the matcher needs to have access to all possible acceptable responses and in that case using llm-judge would be more feasible? 

2. L 115: "specific enough that the correct choice can be uniquely inferred." How do you define "specific enough"?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper discusses the shortcomings of MCQ benchmarks and demonstrates they can be prone to model shortcuts. They show that answer-matching (llm-as-a-judge with a reference answer) is a better evaluation that is more aligned with ground-truth and human evals.

### Strengths
The paper presents a systematic study aiming to show the weaknesses of MCQ benchmarks and why answer matching is a better alternative. They conduct grounded experiments to support their claims.

### Weaknesses
While the paper has some interesting experiments/findings (e.g. models can solve MCQ w/o conditioning on the question, answer matching is better aligned with ground-truth evals/human evals, model rankings change between MCQ and answer matching), the formulations of some of these experiments feel too weak to make definitive claims (see questions) and the analysis and intuitions around the findings feel a bit lacking. Overall, it is unclear if the paper is contributing knowledge beyond what is already known to the community – it is well known that discriminative tasks such as MCQ are easier than generative tasks (generator-verifier gap), LLM-as-a-judge with a reference answer (“answer matching”) is also a well known evaluation technique in the community and it is not surprising that creating generative versions of MCQ benchmarks would make these benchmarks harder.

### Questions
- Section 2:
Can multiple choice evals be answered w/o the question: if the MCQ benchmark is very in-distribution for the LLM (e.g. very similar to data it has been pre-trained on), the model could be internally “predicting” the question and then answering it, which means that by seeing similar training data it has learned a mapping between this type of answer choices and their associated questions and can infer the question from the answer choices. This doesn’t mean the model can isolate the correct answer just from the set of choices, just that it has memorized similar questions and answers from its data. Fine-tuning the model to select the correct answer choice makes this task even easier. A more convincing way to demonstrate that models can isolate the correct answer from a set of distractors would be taking an OOD MCQ benchmark and zero-shot prompting models to select the correct answer.
- Result 2: Why are model generated answer choices more prone to short cuts?
- Result 3: Which model is result 3 trained on?
-Discussion: While it is interesting that models can be trained to select the correct answer from a set of distractors w/o seeing the question, the key argument this supports is that discriminative tasks are inherently easier than generative tasks? However this generator-discriminator gap is well-known in machine learning circles, so not sure if the findings are adding something beyond that.

- section 3.1: Other works such as https://arxiv.org/pdf/2207.05221 study the calibration of language models across different types of tasks and show that they are well-calibrated on multiple choice and true-or-false tasks. This suggests that framing self-evaluation as a multiple choice task may be effective. How does this relate to your findings?

- section 3.1: Another reason why multiple choice cloze may be a poor method is due to surface form competition: https://arxiv.org/pdf/2104.08315, discussing this would be useful

- section 3.2: The dataset size of questions (493, 126) which results are reported on is perhaps too small to make conclusive statements

- section 4: It is interesting that model rankings change when transitioning from MCQ to answer matching, but analysis and intuitions on why this is are lacking. If MCQ questions allow models to exploit discriminative shortcuts, one might expect models of all strengths to benefit from this and perhaps stronger models (which also perform well under answer matching) to exploit these short cuts even more and rank high on both types of benchmarks.

- section 4: How is answer matching cheaper than multiple choice evals? Answer matching would typically require an LLM call as multiple equivalent surface forms may exist, while multiple choice evals can be done w/ regex-based extractions and rule-based matching

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
4

### Summary
Compares three LLM-as-judge approaches, multiple choice vs. answering matching vs. reference-free evaluation. Under the multiple choice approach, the model under evaluation is given a question and a set of possible answers, judging is automatic from the answer key. Under the answer-matching approach, an LLM-as-judge is given a reference answer. In the final approach, the LLM-as-judge determines the correctness of the answer without a reference. While correct answers are unambiguous, they show that multiple choice allows models to take short-cuts in answering, reporting an interesting experiment in which models can successfully answer (with accuracy significantly greater than random) multiple choice questions from benchmarks with only the answers available. On the other hand, the use of reference answers can achieve accuracy similar to humans.

The paper is well written and organized. The topic is interesting. The experiments appear to be properly conducted and reported.

I have serious concerns about novelty, which I will detail in the "weakness" section.

### Strengths
The paper is well written and organized. The topic is interesting. The experiments appear to be properly conducted and reported.

I particularly enjoyed that "question free" multiple choice evaluation.

### Weaknesses
The abstract concludes, "In light of these findings, we discuss how to move the evaluation ecosystem from multiple choice to answer matching." This sentiment is repeated throughout the paper, including the conclusions. I am likely in a different part of the ecosystem (natural language question answer and RAG) but comparison against a reference answer already seems standard. I was trying to find the point where it became standard, and a brief Google search turns up papers like https://aclanthology.org/2021.mrqa-1.15/ and https://aclanthology.org/2023.acl-long.307/ that are not cited, plus others papers that are cited, but not fully differentiated from this work.

Here in 2025, I would view matching to a reference answer as a completely standard and uncontroversial approach to LLM evaluation.

### Questions
Is this not already standard?

### Soundness
4

### Presentation
4

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
This paper discuss multiple-choice evaluation (MCQs) and generative evaluation in LLM evaluation. 
The authors argue that MCQs inherently test discriminative rather than generative abilities and can be answered correctly even without reading the question, due to choice-only shortcuts.
They propose Answer Matching, a scalable generative evaluation method: a model generates a free-form answer given only the question, and a second “matcher” model determines if the response semantically matches a reference answer.
Through extensive experiments on MATH, MMLU-Pro, and GPQA-Diamond, the authors show that answer matching:
1. Achieves near-human agreement with ground-truth or human grading (Scott’s π ≈ 0.9–0.97),
2. Outperforms MCQ and LLM-as-a-judge (without reference answers),
3. Alters model rankings significantly,
4. Is no more costly—and sometimes cheaper—than MCQ evaluation.

The paper concludes that answer matching has recently become a viable, superior alternative for evaluating generative capabilities of LLMs.

### Strengths
1. The paper is well-written.

### Weaknesses
1. While we systematically evaluate the reliability of reference-guided matching (“Answer Matching”), similar judge-with-reference setups have already been widely adopted in practical benchmarks such as NovelQA (Wang et al., 2023), LongBench/LongBench-R (Bai et al., 2023/2024), RAGAs (Es et al., 2023) and MT-Bench (Zheng et al., 2023, LMSYS). These works all rely on an LLM to decide whether a generated answer semantically matches a gold reference, demonstrating that such paradigms are already standard practice in QA evaluation .

2. The core methodology of using LLM-as-a-judge with a reference answer to evaluate generative QA has already been explored in a number of studies.
For example, Evaluating OpenQA Evaluation (NeurIPS 2023) directly compared multiple-choice evaluation against generative answer judging and found that LLM judges better align with human graders. Similar comparisons appeared in Rethinking Evaluation of Open-Domain QA (Min et al., EMNLP 2021), Beyond Multiple Choice (Balepur et al., 2024), and JudgeBench (Tan et al., ICLR 2024).

### Questions
none

### Soundness
2

### Presentation
3

### Contribution
1
