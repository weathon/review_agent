# MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation

- Decision: Accept (Poster)
- Scores: 6, 3, 8, 6

## Abstract
In this work, we introduce a novel evaluation paradigm for Large Language Models
(LLMs) that compels them to transition from a traditional question-answering role,
akin to a student, to a solution-scoring role, akin to a teacher. This paradigm, focusing on "reasoning about reasoning," termed meta-reasoning, shifts the emphasis
from result-oriented assessments, which often neglect the reasoning process, to a
more comprehensive evaluation that effectively distinguishes between the cognitive
capabilities of different models. Our meta-reasoning process mirrors "system-2"
slow thinking, requiring careful examination of assumptions, conditions, calculations, and logic to identify mistakes. This paradigm enables one to transform
existed saturated, non-differentiating benchmarks that might be leaked in data pretraining stage to evaluation tools that are both challenging and robust against data
contamination. To prove our point, we applied our paradigm to GSM8K dataset and
developed the MR-GSM8K benchmark. Our extensive analysis includes several
state-of-the-art models from both open-source and commercial domains, uncovering fundamental deficiencies in their training and evaluation methodologies.
Specifically, we found the OpenAI o1 models which possess characteristics of
"system-2" thinking excel the other SOTA models by more than 20 absolute points
in our benchmark, supporting our deficiency hypothesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a novel paradigm for evaluating LLMs. Rather than assessing their ability to correctly produce a final answer, the paradigm assesses their ability to produce a more fine-grained analysis of given question-solutions paris. They focus on mathematical questions, specifically GSM8k. They generate a rather detailed dataset derived from GSM8k and include 3 types of base metrics for assessing the LLM in the evaluation process.

### Strengths
* The authors suggested paradigm shift to solution-driven evaluation (i.e., the correctness of the entire solution process) is indeed important, and not heavily studied
* The annotation process of the dataset with the three fields seems to be very useful in assessing fine-grained performance of LLMs
* Most claims and the steps taken by the authors in their research are fairly clear
* The process of training and selecting annotators
* They provide useful insights into LLM performance
  * Regarding “specialized”-trained models, specifically math models, they highlight their inability to generalize to related tasks
  * Larger models are not necessarily better than smaller ones
  * Fine-tuning LLMs on specific benchmarking tasks (such as GSM8k) may result in overfitting to the data in a detrimental way. Thus, fine-tuning should be handled with care
  * The impact of the number of correct solutions that is shown to the models during in-context-learning (i.e., “susceptibility of language models to the distribution of few-shot examples”)

### Weaknesses
* The authors explain the reasoning behind the 3 types of questions in their MR-GSM8k. Each type was previously suggested and studied: (1) sampling GSM8k, (2) modifying GSM8k to require code in the solution, (3) “reversed reasoning” from GSM8k. If these types were already studied, what is the novelty or added value in the suggested benchmark? This is unclear. Providing more information can help understand the novelty of their data generation process
* Discussion section
  * Subsection 6.1 is based on Appendix-D, and thus it is difficult to grasp the insights provided here. I suggest making this subsection clearer by providing some examples
* The authors claim that their suggested paradigm enables them to transform any benchmark. However, if I understand correctly, this is not really the case. First, they studied a math benchmark. In addition, the novelty here seems only marginal since their suggested benchmarking-transformation techniques (POT + reverse reasoning) was already suggested before. If there is indeed a claim for some novel method for transforming any benchmark, then it is not clear what the novel method is.

### Questions
* Table 1 - unclear what the values under the last column represent (First Error Steps). Is this the step where the first error occurred? Make it clearer
* Table 2
  * I suggest using explicit terms for task 1/2/3, or alternatively use explicit metrics where possible, such as ACC_{reason}, ACC_{step}. Currently, it is unclear. For example, what is the difference between “Task2-Accy” and “ACCstep” ?
  * please describe all abbreviation (TPR, TNR), including “k” in the table (they are not defined before the table is referenced)
* Lines 381 - 393: Point the reader to the few-shot examples that were given to the models (or did I miss it?)
* Conclusions section - the first paragraph is a summary of the paper, but does not contain conclusions

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors to propose a new evaluation paradigm that shifts the role of models from “QA student” to a “solution-scoring teacher.” In this approach, instead of generating solutions to questions, models are presented with question-solution pairs and are have to determine the correctness of each solution, identifying the first error step if any, and providing explanations for errors. This paradigm encourages models to engage in meta-reasoning, assessing different reasoning processes rather than simply arriving at correct answers. The authors developed a meta-reasoning version of the GSM8K benchmark, called MR-GSM8K, and introduced a new metric, the MR-Score, which is a weighted combination of three submetrics. Instances in MR-GSM8K are manually labeled by experts to ensure accuracy in evaluation. Their findings indicate that specialized math models struggle to generalize to this new benchmark, and, contrary to expectations, larger models do not consistently perform better in tasks on MR-GSM8K. The authors argue that models frequently exhibit superficial reasoning in math tasks and link these to their possible limitations in “System 2” thinking—i.e., that models fail to engage in slow, deliberate reasoning that examines assumptions, conditions, and logic for thorough error detection.

### Strengths
I commend authors' general aims in evaluation of LLMs' math reasoning steps rather than just final solutions as well as their efforts to ensure minimise the annotation errors and quality control the benchmark instances. I also like the metric MR-Score they introduced to evaluate model performance. Introducing a meta math reasoning benchmark is also important in advancing our understanding of LLMs abilities.

### Weaknesses
I find it difficult to understand why, given their goal of assessing LLMs’ step-by-step math reasoning abilities, the authors chose to create a meta-reasoning benchmark where the task is to evaluate the correctness of provided step-by-step math solutions rather than to generate correct step-by-step reasoning. Evaluating solutions is a distinctly different skill from producing them. Just as we wouldn’t expect students who excel at solving math problems to also excel at verifying others' solutions, we shouldn’t assume these skills are interchangeable for models.

Furthermore, the authors' results might even suggest that their benchmark tests different capabilities than those measured by GSM8K. For instance, despite the claims in lines 375-379, the results in Table 2 indicate that none of the models specialized in solving math problems performed well on their benchmark. In lines 377-378, the authors claim that MAmmoTH-70B and DeepSeek-Math-7B-RL achieve decent performance. However, Table 2 shows that their MR-Scores are near zero in multi-shot learning and only slightly higher in zero-shot learning. If Table 2 is accurate, this could indicate that their benchmark does not truly measure the same abilities as GSM8K, MATH, or other similar benchmarks.

Given this, it’s unclear how the authors can argue that fine-tuning on math problems like those in GSM8K leads to overfitting or only a superficial mastery of mathematical reasoning, especially if their benchmark assesses a different skill than GSM8K and similar tests. Why should we expect high performance on GSM8K to correlate with high performance on MR-GSM8K?

The authors need to clarify their definition of "mathematical reasoning." How does their concept of math reasoning differ (or not) from the concept tested by benchmarks like GSM8K? Are these two concepts comparable, or do they address entirely separate abilities?

### Questions
If the authors are interested in evaluating LLMs' math reasoning rather than just their ability to produce a correct answer, why not have the models generate final answers using Chain-of-Thought (CoT) or Program-of-Thought (PoT) reasoning and evaluate those directly, rather than assessing the models' meta-reasoning abilities?

What do the authors mean by "mathematical reasoning"? Are they evaluating the same abilities as those tested by GSM8K and similar benchmarks?

### Soundness
2

### Presentation
2

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
This paper proposes a new benchmark category, Meta-Reasoning (reasoning about reasoning), and a new corresponding meta-reasoning benchmark dataset, MR-GSM8K, derived from the well-known GSM8K. In this benchmark, models are shown question-solution pairs and are tasked with identifying correct solutions, incorrect steps, and explaining incorrect steps. All three of these subtasks are captured in their proposed novel metric, the MR-score. They show that their benchmark indeed tests for a new capability and is quite challenging - models that perform well on the reasoning dataset GSM8K surprisingly perform substantially worse on MR-GSM8K.

### Strengths
Originality
-The benchmark task and evaluation metric appears to be novel. It is testing for a new capability, reasoning about reasoning.

Quality
-Their benchmark has led to strong insights that may influence future research. For example, they show that bigger models are not necessarily better than smaller models on their challenging meta-reasoning benchmark, with Phi-3 beating models many times larger.

-The creation of the benchmark dataset underwent several rounds of reviews from LLMs and human annotators. 

Clarity
-The paper is logically organized and easy to follow. 

Significance
-It seems that multi-step reasoning datasets don’t fully assess the steps taken to arrive at a solution; instead, they score a model based on its final answer. So there is a need for more nuanced evaluation, such as this work. This work challenges the surface-level assessments of previous reasoning benchmarks and will inspire deeper research into enhancing model reasoning.

### Weaknesses
-Table 2 could be easier to read, see suggestions below.

### Questions
-Table 2 needs more clarity: consider adding a table description defining your abbreviations “Task1-TPR”, etc. Also the values that are bolded were not intuitive - in the table description also explain what bold means. Also explain what k=0, k=3 means.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MR-GSM8k a new evaluation dataset adopted from GSM8k to test the depths of cognitive abilities in LLMs. This dataset consists of samples from GSM8k, code based solutions to the problems, and reverse-engineered versions asking the model to find the value of an input variable given the answer. Rather than relying on surface-level final answer match, they aim to score the model for its reasoning ability by proposing a new metric MR score - which given a <task, solution> pair, expects the model to evaluate its correctness, identify the first point of error, and the reason for the error. Evaluating a range of models, they demonstrate that these models perform similarly on the original GSM8k but show wide variance on MR-GSM8k as it tests their core ability to reason, and investigate possible reasons for this, urging for benchmarks that go deeper than surface-level evaluations.

### Strengths
- The construction of this new dataset is quite novel and opens doors to more nuanced evaluation methods to test the reasoning of language models.
- With increasing interest in the interpretability of LMs, this work transforms an existing dataset to measure which parts models fail.
- They highlight the memorization issue prevalent in LMs — it has imitation skills but lacks a deep understanding of the underlying logical as they can solve the task but not infer with the same accuracy if a given solution is correct or not.

### Weaknesses
- The dataset construction procedure is of limited novelty with only the reversal mode being significantly different.
- A more detailed error analysis with some discussion on actionable insights on how to combat them would be beneficial to the community
Several works have already probed the memorization effect and lack of genuine reasoning in LLMs on different tasks (Embers of Autoregression, McCoy et. al; Deciphering the Factors Influencing the Efficacy of Chain-of-Thought, Prabhakar et. al; GSM-Symbolic, Mirzadeh et al.) providing concrete insights into the error patterns.
- Given these weaknesses, I feel the dataset could be of interest to a section of the community.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
