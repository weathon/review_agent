# Input-Time Scaling

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Large Language Models (LLMs) excel at mathematical reasoning, traditionally requiring high-quality data and extensive training. Recent work reveals a **Less-Is-More** phenomenon where small, curated datasets match resource-intensive approaches. In this work, we systematically investigate quality constraints by adding controlled noise and comparing datasets qualities. Noise levels are controlled via context relevance to original queries. Counterintuitively, mixing relevant and irrelevant contexts yields optimal results, and performance gains emerge only when context concatenation applies consistently, not necessarily the same type, across training and inference. Token distribution analysis shows persona strategies increase thinking tokens while reducing response length. We term the above phenomenon **training-testing co-design**. Comparing dataset qualities, high-quality data excels on weaker models and easier questions, while low-quality data achieves overall higher scores, especially on hard questions with capable models. Building on these insights, we propose our method, applying small, low-quality data to capable models via training-testing co-design. The process distinguishes it from supervised fine-tuning or test-time scaling, which we term it **Input-Time Scaling**. Our method achieves 76.7\% pass@1 on AIME24/AIME25 using Qwen2.5-32B-Instruct, with DeepSeek-R1-Distill-Qwen-32B reaching 90.0\%/80.0\%. We are open-sourcing our datasets, pipelines, evaluation results, and checkpoints to facilitate reproducibility and further research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel paradigm called Input-Time Scaling, which enhances large language model (LLM) reasoning capabilities by refining input queries through meta-cognitive methods. The approach achieves state-of-the-art (SOTA) performance on challenging mathematical reasoning benchmarks while using a simple, transparent, and highly efficient training pipeline. More importantly, the work challenges long-standing assumptions about data “quality” and “quantity” in LLM training and uncovers an intriguing train–test co-design phenomenon, where consistent strategies applied during both training and inference are crucial for performance gains.

Overall, this paper makes a significant contribution. The proposed method is simple, effective, and well supported by empirical evidence. It opens a new direction for enhancing LLM reasoning and provokes deep reflection on data management practices.

### Strengths
- **High originality:** The concept of Input-Time Scaling is genuinely novel, extending the traditional trichotomy of Data / Training / Inference Scaling by introducing the idea of allocating computational resources at the input level. The use of meta-cognitive methods to introduce various personas—similar, dissimilar, or random—during both training and testing is an innovative way to enhance reasoning diversity and robustness.
- **Strong empirical performance:** With only 1k supervised fine-tuning samples and no reinforcement learning (RL) stage, the method achieves SOTA results on challenging math benchmarks such as AIME24 and AIME25 with 32B-scale models. The authors conduct extensive ablations across multiple persona strategies during training and testing, showing the effectiveness and stability of the approach.
- **Challenging conventional wisdom:** The surprising finding that adding irrelevant or seemingly low-quality information (e.g., dissimilar personas) can improve performance directly contradicts the commonly held “garbage in, garbage out” assumption. This challenges the community’s bias toward data purity and suggests that diversity may play a more crucial role than quality alone.

### Weaknesses
- **Limited generalization evidence:** The experiments focus solely on four mathematical benchmarks (AIME24, AIME25, MATH, and GPAQ). It remains unclear whether Input-Time Scaling generalizes to other reasoning tasks such as code generation, logical reasoning, or commonsense QA. Since the proposed pipeline is simple and data-agnostic, it would be valuable to test it on different reasoning domains. In addition, all experiments use 32B models; it would be informative to evaluate whether the method is equally effective for smaller models (e.g., 7B).
- **Lack of mechanistic understanding:** Section 5.2 shows which combinations of training–testing persona strategies perform best, but the paper provides little insight into why they work. In several cases, even random or mismatched personas lead to large improvements. A deeper discussion of the underlying mechanism would significantly strengthen the paper’s depth and credibility.
- **Details of persona generation:** Although the appendix provides prompts, the main text lacks clarity on implementation details—such as which model is used for persona generation, the degree of randomness, or how diversity is controlled. Since persona quality itself could be a confounding variable, these details should be made explicit. It would also be useful to include qualitative case studies showing how persona-based input refinement changes the reasoning process.
- **Data quality and validity claims:** The paper claims that “lower-quality” data (OT-1k) outperform the more carefully curated LIMO dataset. However, it is not clearly demonstrated that LIMO is indeed of higher intrinsic quality. A more rigorous comparison or justification of this assumption would make the argument more convincing.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes "Input-Time Scaling", where the authors concatenate different personas during the training and testing phase for performance gains. The personas are divided into four types depending on the relevance to the original prompt. The strategy is tested on Qwen2.5-32B.

The paper evaluates on four datasets, AIME2024/2025, Math, and GPQA, and claims to outperform heavily trained models of similar size. 

The paper aims to highlight that their proposed method differs from "test-time scaling" in that train-test co-training is necessary to benefit from it fully.

### Strengths
The paper proposes a new scaling axis. Past works usually either scale train-time or test-time. This paper proposes scaling the inputs.

### Weaknesses
1) Limited Novelty: Unlike how the paper claims "input time scaling" to be novel, I can easily think of different papers that try a similar thing. For instance, (https://arxiv.org/abs/2502.11027) shows that adding diversity into prompts for best-of-n boosts performance. While the two papers differ in that this paper requires training, I don't think it is a significant difference.

2) Limited Evaluation: In section 5.6, the paper mentions Math and GPQA lack "discriminative effects"; accordingly, it mostly concentrates on evaluation results from AIME 2024 and 2025. Additionally, the paper uses pass@1 due to resource constraints. However, both datasets contain only 30 samples; accordingly, the main claims of this paper are based on a single inference over 60 questions. This severely lacks statistical credibility. Either more datasets should be added or multiple runs on AIME should be performed to demonstrate that the trained models consistently outperform baselines.

3) Limited Models: The method is only verified by Qwen2.5-32B. While training bigger models might be unaffordable, it's not understandable for the paper to lack results from smaller models. I suggest the authors to try the same method on a larger diversity of models, Qwen2.5-1.5B or Llama-3.1-8B .

4) Limited Ablations: If the author is trying to argue this as a new axis of "scaling," I think it is necessary for them to show how compute-efficient it is. By comparing to past scaling methods. 

5) Typo in tables 1~3 should be GPQA not GPAQ. 

6) The authors try out diverse combinations of prompting (e.g., N-S, S-D ...) and only AFTER they have seen the results on the test set they choose the optimal prompting method. I would say this is a form of test-contamination. They should have had a validation set to choose the best method and see if it generalize to an unseen test set.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a "Input-Time Scaling" method, which augments an SFT dataset (OpenThoughts) using personas (prompting a language model to rewrite a training example using a persona). They fine-tune a Qwen2.5-32B Instruct model on this dataset and evaluate it on standard reasoning benchmarks (AIME, MATH, GPQA), showing some improvements.

### Strengths
The high-level idea of augmenting SFT data is a good, reasonable one.
The results seem strong.

### Weaknesses
This paper has a number of problems. First, the proposed ideas (using personas to perform data augmentation) is not novel (see PersonaMath from [Luo et al., 2025]).
Second, the 3 types of modification (S, D, R) are not particularly well-motivated nor explained clearly; looking at Appendix 3 doesn't really help.
Third, I would have liked to see a broader evaluation on a larger number of datasets, especially since the AIME datasets are quite small.
It would be good to evaluate the method on other non-Qwen models like Llama since Qwen is regarded to have quite a bit of reasoning already baked inside.
Finally, the writing in the paper could be greatly improved. For example, the abstract is quite verbose, the core method should be explained (section 2.1), the results tables could use more description (and it's hard to tell what the key takeaway is).
Calling it input-scaling a new paradigm is a bit exaggerated.

### Questions
Why does the method work?  Improving diversity is good, but why personas as opposed to other types of diversity (e.g., different problems, reasoning patterns).
How many examples does 1K examples get augmented into?
What are the hyperparameters and how were they tuned?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Input-Time Scaling, a new paradigm that improves LLM reasoning by allocating computation and diversity to query modification rather than model or training scaling. The key finding is the train–test co-design phenomenon: applying the same persona-based query strategies at both training and inference time is crucial for performance. Using small datasets (as few as 1k examples) and simple meta-cognitive persona generation, the method achieves strong reasoning performance on AIME24 and AIME25 benchmarks (up to 90%/80% pass@1 with DeepSeek-R1-Distill-Qwen-32B), surpassing prior open-source 32B models. Surprisingly, lower-quality or more diverse data (random or dissimilar personas) outperform curated datasets, challenging common assumptions about data quality.

### Strengths
- Introduces a novel scaling axis that complements data, model, and inference scaling. It focuses instead on the input level via persona augmentation. 
- Findings challenge existing inductive biases about data quality (“garbage in, garbage out”) and show benefits of diversity. I think this is surprising and also interesting.

### Weaknesses
- The writing and presentation should be improved (e.g., the abstract is a bit too long, the method section could need more clarity) 
- I would want to understand more why the simple persona change could diversify the training distribution and make the training results better. This augmentation didn't change the question distribution, just how the prompt is seeded. It would be great to look at some qualitative samples and see whether the improved skill correlation with certain persona?

### Questions
see above.

### Soundness
2

### Presentation
2

### Contribution
2
