# Large Language Models Are Active Critics in NLG Evaluation

- Decision: Reject
- Scores: 6, 5, 3, 5

## Abstract
The conventional paradigm of using large language models (LLMs) for evaluating natural language generation (NLG) systems typically relies on two key inputs: (1) a clear definition of the NLG task to be evaluated and (2) a list of pre-defined evaluation criteria. This process treats LLMs as ''passive critics,'' strictly following human-defined criteria for evaluation. However, as new NLG tasks emerge, the criteria for assessing text quality can vary greatly. Consequently, these rigid evaluation methods struggle to adapt to diverse NLG tasks without extensive prompt engineering customized for each specific task. To address this limitation, we introduce Active-Critic, a novel LLM-based NLG evaluation protocol that enables LLMs to function as ''active critics.'' Specifically, our protocol comprises two key stages. In the first stage, the LLM is instructed to infer the target NLG task and establish relevant evaluation criteria from the data. Building on this self-inferred information, the second stage dynamically optimizes the prompt to guide the LLM toward more human-aligned scoring decisions, while also generating detailed explanations to justify its evaluations. Experiments across four NLG evaluation tasks show that our approach achieves stronger alignment with human judgments than state-of-the-art evaluation methods. Our comprehensive analysis further highlights the effectiveness and explainability of Active-Critic with only a small amount of labeled data. We will share our code and data on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies employing LLMs for NLG evaluation. Existing NLG evaluation approaches are shaped by the developers’ understanding and expectations of the evaluation process as “passive critics”. This paper proposes active critics that automatically infer what and how to evaluate from a few labeled data. This framework works as a two-stage pipeline that (1) infers the target NLG task and identify relevant evaluation criteria from the data, and (2) optimize prompts to make its scoring better align with human judgments.

### Strengths
1. This paper studies an important research question of reference-free NLG evaluation.

2. The active evaluator proposed in this work exhibits better generalization compared to existing passive NLG evaluators.

### Weaknesses
1. Dependence on human annotations. During the first stage of *task inference*, this module is asked to prompt the LLM to formulate an accurate task description by reviewing examples in the training set and identify key information that characterizes the target NLG task. As illustrated in the left side of Figure 1, a human score should be provided for each example. I am curious why we need to go through a complicated process to get the task description when there are already human annotations. We can evaluate the model to align directly with the human annotations. Can you clarify the specific advantages of your approach compared to directly using human annotations?

2. How many training examples should be used for *task inference*? From Section 4.1, it seems like all examples in the training set are split into mini-batches and feed them into LLMs for *task inference*. If only a small amount of training data is needed, then this is acceptable when adapting to a new task. However, if a lot of data, or even the whole dataset, is needed to get the task description, the demand for manual annotation is huge, which is unacceptable.

3. The definition of *comprehensiveness*. In line 206, to enhance efficiency, LLMs are instructed to decide whether to stop early based on the comprehensiveness of the generated task description and criteria set after processing each mini-batch. But how to decide the comprehensiveness is not explained. Can you provide a clear definition or metric for how comprehensiveness is measured and how the early stopping decision is made.

4. A large number of evaluator calls for a single evaluation example. Iterative task inference estimation based on comprehensiveness, prompt optimization for scoring, criterion scoring with explanation for each of the criteria, and an overall scoring to summarize all these criteria. The cost of calling a large number of LLM evaluator is also high, which should also be reported. Can you provide the average number of LLM calls per evaluation and compare these to baseline methods. If there are potential optimizations to reduce the number of LLM calls?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces ACTIVE-CRITIC, a protocol that enhances the role of large language models as evaluators in natural language generation tasks. Unlike traditional "passive" evaluation systems that rely on predefined criteria, ACTIVE-CRITIC allows LLMs to actively infer task requirements and optimize prompts for human-aligned scoring. Experiments across multiple NLG tasks demonstrate its effectiveness, offering more adaptable and human-like evaluations through a two-stage, criteria-based scoring process.

### Strengths
1. The paper presents a straightforward approach that is easy to understand and follow.
2. Addressing NLG evaluation is essential, especially as LLMs continue to evolve, making evaluation standards increasingly important.
3. The two-stage design in ACTIVE-CRITIC enhances explainability and flexibility, allowing for better alignment with human judgments without extensive manual prompt engineering.

### Weaknesses
1. The proposed approach is largely a pipeline with prompt engineering. While it enables LLMs to determine evaluation aspects independently, this method appears as a direct and relatively simple extension of prompt engineering, rather than a novel framework.
2. NLG evaluation aims to approximate human judgment, with scoring criteria derived from human-annotated datasets. The paper’s method, which uses LLMs to learn and generate these criteria, may seem contradictory, as it implies that while human scores are considered "gold," human-derived standards are not.
3. For new tasks, collecting human-annotated scores may be more challenging than developing scoring criteria. This raises concerns about the motivation behind prioritizing criterion generation over human annotation.
4. Experiments and comparisons are limited to a few tasks and baselines.

### Questions
Could you provide a demonstration of criterion generation on a novel task and the outcomes it yields?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes an active NLG evaluation method based on LLMs. It requires the model to generate task descriptions and evaluation criteria for different cases independently and uses data with human labels and prompt optimization algorithms to improve demonstration selection in few shots. Additionally, the model provides evaluations across different dimensions and offers explanations for its scores, which are then aggregated into overall evaluation results. The experiments cover common NLG tasks such as summarization, dialogue, data-to-text, and story generation.

### Strengths
This work highlights a limitation of current LLM-based NLG evaluation methods: they often rely on specific evaluation instructions to perform passive evaluations. While human-crafted evaluation criteria may intuitively enhance the controllability, they lead to high costs. To address this, this work proposes a method for enabling models to conduct active evaluations and conducts experiments on multiple NLG evaluation tasks.

### Weaknesses
Although the concept of active evaluation is meaningful, approaches with similar motivation have been explored in previous works (Liu et al., 2024; Li et al., 2024; Liu et al., 2024). Furthermore, prior research has pointed out that generating explanations along with scores can enhance evaluation performance (Chiang et al., 2023; Chiang et al., 2023).

The proposed method requires some training data to fine-tune prompts, which significantly reduces its generalizability (how would it handle evaluation samples in new tasks or domains?). Additionally, in the experiments, they sample 25% of each dataset as the training set, leaving the rest as the test set. This is unfair compared to other baselines (e.g., Auto-J), which had not seen the corresponding dataset and been tested on all data.

The LLMs used in this paper are somewhat outdated (Orca2 instead of Llama3, GPT-3.5 instead of GPT-4). Moreover, the latest evaluation methods are not included for comparison (Kim et al., 2023; Kim et al., 2024; Hu et al., 2024). And the experimental results only present the overall score performance, whereas prior related work has focused more on performance across individual dimensions, which is also necessary to be experimented.

**References:**

Liu, Yuxuan, et al. "Calibrating LLM-Based Evaluator." Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 2024.

Li, Minzhi, et al. "Decompose and Aggregate: A Step-by-Step Interpretable Evaluation Framework." arXiv preprint arXiv:2405.15329 (2024).

Liu, Yuxuan, et al. "HD-Eval: Aligning Large Language Model Evaluators Through Hierarchical Criteria Decomposition." arXiv preprint arXiv:2402.15754 (2024).

Chiang, Cheng-Han, and Hung-Yi Lee. "Can Large Language Models Be an Alternative to Human Evaluations?." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023.

Chiang, Cheng-Han, and Hung-yi Lee. "A closer look into automatic evaluation using large language models." arXiv preprint arXiv:2310.05657 (2023).

Kim, Seungone, et al. "Prometheus: Inducing fine-grained evaluation capability in language models." The Twelfth International Conference on Learning Representations. 2023.

Kim, Seungone, et al. "Prometheus 2: An open source language model specialized in evaluating other language models." arXiv preprint arXiv:2405.01535 (2024).

Hu, Xinyu, et al. "Themis: Towards flexible and interpretable nlg evaluation." arXiv preprint arXiv:2406.18365 (2024).

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
When prompting LLMs for NLG evaluation, prompting can be inconvenient due to the required manual intervention to design the prompts, which may also limit the model's ability to leverage other aspects of the text that may be useful for assessment. This paper proposes a method of automatically generating prompts for NLG evaluation. They propose a multi-stage process, where by using batches of test examples with human labels, the LLM first predicts the task of interest, attributes relevant for examples and suitable few-shot examples. This is then used to develop a prompt that can be used for NLG evaluation on new test cases, which to aid assessment, also provides explanations. The work provides ablations for the different components of the framework, demonstrating that for ORCA-13B and ChatGPT-3.5 the approach results in better performance than compared to existing baselines, and that the generated explanations are helpful for evaluation.

### Strengths
- As far as I’m aware, I haven’t encountered works that apply prompt optimization for NLG evaluation, and so automatically designing the NLG evaluation prompts seems to be quite novel and clearly a highly practical and useful application. Also their approach decomposes the evaluation into multiple different aspects (e.g. task, criteria, few-shot examples, explainability).

- They provide ablations numbers to make clearer where the performance improvements originate from, showing that all of the task descriptions, assessing individual criteria and having explanations are all helpful. Along with its judgements, the model returns explainable decisions which can be useful for many use-cases. 

- The paper is fairly easy to follow, and details of prompts and meta-prompts are provided.

### Weaknesses
- The experimental results have not comprehensively demonstrated the advantages of the active-critic approach. The examined models are somewhat limited (ORCA-13B and GPT3.5) and also the baseline compared to appear somewhat limited inconsistent (e.g. G-EVAL not used in Table 1 and Table 2 only compares to G-EVAL). Whether the approach remains equally as effective when using larger more capables, which may possibly be better aligned, is not clear.

-  Is the contribution of the work the simplicity in the approach, i.e. that humans no longer have to design the prompts, or that by using this approach we get better prompts that result in better performance? I’m not sure if any of the baselines directly compare performance to human design prompts, which, as far as I understand, is the main contribution of the paper. This might be what AC-vanilla is, but from what I can tell, AC-vanilla only provides the system with few shot examples. 

- The approach does require labelled examples in order to optimize the prompts, which limits some of the advantages of prompting approaches, even when compared to few-shot in-context learning. Especially with the dataset split of 75%, 25%, the approach is now more manually laborious and we require more reliable assessment scores for the assessment task of interest.

### Questions
- Why was G-EVAL not compared to in Table 1? The baselines seem a bit inconsistent across models (I understand that for Table 2 the system is resetrected to API calls and may be blackbox, and hence some approaches may be limited) 

- When you randomly select 25% of the data for ACTIVE-CRITIC tuning, are these 25% of contexts (i.e. for SummEval do you take 25 of the articles and all 16 of the summaries?) or 25% of all pairs across contexts?

- Out of curiosity, what was the process of developing the meta-prompts provided in Appendix A.5. Was this just manually designed, or was standard prompts used for automatic prompt optimization works, etc?

### Soundness
3

### Presentation
3

### Contribution
2
