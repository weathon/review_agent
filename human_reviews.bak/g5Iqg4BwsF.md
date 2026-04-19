# In-Context Learning with Iterative Demonstration Selection

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 5

## Abstract
Spurred by advancements in scale, large language models (LLMs) have demonstrated strong few-shot learning ability via in-context learning (ICL). However, the performance of ICL has been shown to be highly sensitive to the selection of few-shot demonstrations. Selecting the most suitable examples as context remains an ongoing challenge and an open problem. Existing literature has highlighted the importance of selecting examples that are diverse or semantically similar to the test sample while ignoring the fact that the optimal selection dimension, i.e., diversity or similarity, is task-specific. Leveraging the merits of both dimensions, we propose Iterative Demonstration Selection (IDS). Using zero-shot chain-of-thought reasoning (Zero-shot-CoT), IDS iteratively selects examples that are diverse but still strongly correlated with the test sample as ICL demonstrations. Specifically, IDS applies Zero-shot-CoT to the test sample before demonstration selection. The output reasoning path is then used to choose demonstrations that are prepended to the test sample for inference. The generated answer is accompanied by its corresponding reasoning path for extracting a new set of demonstrations in the next iteration. After several iterations, IDS adopts majority voting to obtain the final result. Through extensive experiments on tasks including commonsense reasoning, question answering, topic classification, and sentiment analysis, we demonstrate that IDS can consistently outperform existing ICL demonstration selection methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes the Iterative Demonstration Selection method to leverage the merits of both dimensions. With Zero-shot-CoT, IDS iteratively selects examples that are diverse but still strongly correlated with the test sample as ICL demonstrations. The proposed method utilizes the output reasoning path to choose demonstrations. After several iterations, IDS adopts majority voting to obtain the final result. The authors assert that the optimal dimension for selecting demonstration examples is task-specific.

### Strengths
* The paper is written in a clear manner.
* There is an experiment demonstrating the need to consider diversity and similarity in a task-specific manner.
* There are several analyses regarding the number of demonstrations, the number of iterations, and model types.

### Weaknesses
* The experimental results look somewhat marginal. 
* The performed experiments are too limited in scope. They were conducted only on CommonsenseQA, BoolQ, AGNews, DBPedia, SST2, and Amazon, which undermines the robustness of the methodology. Results from a more diverse set of tasks are needed e.g., not the classification tasks.
* Measuring cosine similarity between the reasoning path R and the training set lacks some persuasiveness as a criterion for selecting few-shot samples. If there are theoretical or empirical reasons why measuring cosine similarity between reasoning path R and training set samples that have somewhat different forms from reasoning path would be helpful, please provide them.
* There is too much repeated content in the text. e.g."the generated answer is accompanied by its
corresponding reasoning path "
* There is a lack of comparison with other demo selection baselines, not just the internal baseline.

### Questions
* How long does it take to measure the correlation between reasoning path R and the whole training set for each task?
* Please provide results using a different encoder model other than Sentence-BERT. It seems necessary to verify if solidity is ensured depending on the encoder.
* Could you provide full results on all tasks, not the average one?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper addresses the challenge of selecting the most suitable few-shot demonstrations for in-context learning (ICL) in large language models (LLMs).
The authors argue that the optimal selection dimension, i.e., diversity or similarity, is task-specific and propose an Iterative Demonstration Selection (IDS) method that leverages the merits of both dimensions.
IDS uses zero-shot chain-of-thought reasoning (Zero-shot-CoT) to iteratively select examples that are diverse but still strongly correlated with the test sample as ICL demonstrations.
Experiments on various tasks demonstrate the effectiveness of IDS.

### Strengths
1. The methodology is well-explained, with IDS applying Zero-shot-CoT to the test sample before demonstration selection. The output reasoning path is iteratively used to choose demonstrations that are prepended to the test sample for inference. After several iterations, IDS adopts majority voting to obtain the final result.

2. Experiments on various tasks and thorough analysis on hyper-parameters (e.g., number of demonstrations and number of iterations) demonstrate the effectiveness of IDS.

### Weaknesses
1. Lack of comparison with stronger baselines. Much related work and methods for in-context example selection (e.g., EPR [1], TST[2], CEIL[3], Skill-KNN[4]) are not experimentally compared (or even not mentioned). At least some of them should appear in the experiments part.

2. Actually, this work is not "the first time consider both the diversity and similarity dimensions of ICL demonstration selection for LLMs".
For instance, [5] use MMR that considers both similarity and diversity. [6] also demonstrates the effectivenss of incorporating both similarity and diversity.

[1] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. "Learning to retrieve prompts for in-context learning."

[2] Gabriel Poesia, et al. "Synchromesh: Reliable code generation from pre-trained language models."

[3] J. Ye, et al. "Compositional exemplars for in-context learning."

[4] S. An, et al. "Skill-Based Few-Shot Selection for In-Context Learning."

[5] X. Ye, et al. "Complementary explanations for effective in-context learning."

[6] S. An, et al. "How Do In-Context Examples Affect Compositional Generalization?."

### Questions
How would IDS perform without voting, i.e., just use the final answer achieved by the last iteration? I suppose this result better reflect the effectiveness of the designed iteration.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes Iterative Demonstration Selection (IDS) to conjoin diversity and similarity for tackling the problem of demonstration selection for in-context learning of LLMs. Technically, IDS leverages intermediate reasoning paths provided by LLMs to retrieve relevant training samples. IDS is evaluated on several representative tasks.

### Strengths
- The paper is well motivated with preliminary experimental results.
- The literature review is good.

### Weaknesses
- Overall, the presentation of the paper should be improved, and the technical novelty is limited. In particular, Figure 2 should be carefully polished.
- The explanation of why IDS can incorporate diversity should be made clear. I notice the argument "they can be
different during iterations to ensure diversity because the reasoning paths vary in different iterations", but how can you ensure such diversity? Purely rely on the randomness in LLM sampling?
- It seems that the evaluation tasks are relatively simple. Have you tested IDS on GSM8K or MATH?
- In my opinion, when IDS and the baselines use the same number of reasoning paths or iterations, IDS actually uses one more query to GPT than the baselines (due to the first zero-shot CoT). As a result, in the 1-iteration column of Table 6, IDS is better. If so, I want to know if you give the baselines one more query to GPT, can their performance be further improved?
- IDS's performance should heavily rely on the metrics used for retrieval. Have you tested other choices except for Bert Distance?

### Questions
See above

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a strategy, Iterative Demonstration Selection (IDS), used for example selection in in-context learning (ICL) setting of large language models (LLMs), focusing on the balance between diversity and similarity. IDS leverages the reasoning path elicited by zero-shot chain-of-thought (CoT) for similarity between the test sample and demonstrations and iterative selection for diversity among ICL examples. They evaluate IDS on several NLP datasets (commonsense reasoning, question answering, topic classification, and sentiment analysis) and show it consistently outperforms existing ICL demonstration selection methods.

### Strengths
1. The paper is well-written, and the proposed method IDS is easy to follow.

2. This paper provides a conclusion that both similarity and diversity are important in example selection in ICL scenarios, which can help other researchers who are working on a similar area.

### Weaknesses
1. Experiments are conducted based on simple tasks such as classification, commonsense reasoning, etc., on which the improvements seem marginal. More complex and difficult generative tasks, including mathematical reasoning, QA, and machine translation, are encouraged to be adapted.
2. The conclusion that “it is unreasonable to claim that one dimension is consistently better than the other across different tasks” is drawn through only two datasets, AGNews and CommonsenseQA, which is not that solid. 
3. IDS based on voting brings in 4x of the overhead for API requests, while the improvements are a little marginal.
4. Experiments are conducted on GPT-3.5-turbo. More LLMs, such as Vicuna, Llama, and Alpaca, can be tested.

### Questions
1. Comparison with other example selection strategies has yet to be explored. Does the balance matter? The exploration for similarity and diversity is only conducted on 2 datasets.


[1] Li, Xiaonan, and Xipeng Qiu. "Finding supporting examples for in-context learning." arXiv preprint arXiv:2302.13539 (2023).
[2] Ma, Huan, et al. "Fairness-guided Few-shot Prompting for Large Language Models." arXiv preprint arXiv:2303.13217 (2023).

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
