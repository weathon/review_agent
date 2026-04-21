# Exploring Memorization in Fine-tuned Language Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6

## Abstract
Large language models have shown great capabilities in various tasks but also exhibited memorization of training data, thus raising tremendous privacy and copyright concerns.  While prior work has studied memorization during pre-training, the exploration of memorization during fine-tuning is rather limited. Compared with pre-training, fine-tuning typically involves sensitive data and diverse objectives, thus may bring unique memorization behaviors and distinct privacy risks. In this work, we conduct the first comprehensive analysis to explore LLM memorization during fine-tuning. Our studies with open-sourced and our own fine-tuned LM models across various tasks indicate that fine-tuned memorization presents a strong disparity among tasks. We provide an understanding on this task disparity via sparse coding theory and unveil a strong correlation between memorization and model attention distribution. By investigating its memorization behavior,  multi-task fine-tuning paves a potential strategy to mitigate fine-tuning memorization.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigates to which extent language models (LMs) memorize their fine-tuning datasets, when fine-tuned for different tasks (summarization, medical dialog, QA, translation and sentiment classification). To detect memorization, the authors employ a plagiarism checker tool to detect whether models produce output that can be considered as one of three types of plagiarism (verbatim, paraphrase, idea) of text in the fine-tuning dataset, when prompted with prefixes from the same dataset.
The paper finds that the degree of memorization varies strongly based on the fine-tuning task, with summarization and dialog exhibiting much higher memorization than classification, translation and QA tasks. The paper further shows a correlation between attention distributions and the measured memorization scores, with high memorization corresponding to more uniform attention maps, and low memorization to more concentrated attention maps. Finally, the paper proposes to use multi-task fine-tuning to reduce memorization.

### Strengths
- Understanding the memorization behavior of LMs is an important problem with ramifications for privacy and copyright considerations. Since fine-tuning plays a key role in creating usable LMs, the paper tackles an important open problem by investigating how this step impacts memorization.
- Differences in the fine-tuning task can strongly influence model behavior, so investigating how different fine-tuning objectives influence memorization is important.

### Weaknesses
- The paper has severe soundness issues, making the key findings rather unreliable.
    1. The paper claims, that memorization varies based on the fine-tuning task, with summarization and dialog tasks exhibiting higher memorization than QA, translation and sentiment classification. It is very likely, however, that this result is confounded by the response lengths required for the different tasks. According to Figure 2, models generate in the order of > 120 output tokens for summarization and dialog tasks, but <= 6 output tokens for sentiment classification and QA tasks. Tasks that require long outputs are much more likely to produce sequences similar to those in the training corpus, just from a numerical perspective, and irrespective of memorization effects. The paper does not control for differences in output lengths of different tasks. Therefore, the results about differences in memorization behavior for different fine-tuning tasks are unreliable without controlling for this confounder.
    2. The results on using attention maps as indicators of memorization appear similarly unreliable, because they also do not control for important potential confounders. Summarization and dialog (presumably high memorization tasks) are shown to have more uniform attention maps, compared to QA and sentiment classification, where attention is more concentrated. However, for the latter two tasks it may be sufficient to pay attention to a few keywords in the input, whereas summarization and dialog presumably require a more comprehensive understanding of the entire text. Therefore, the differences in attention patterns might simply be due to the nature of the different tasks, and not due to different degrees of memorization. For an apples to apples comparison, the authors should compare attention maps of instances with different degrees of memorization on the same task.
    3. In Sections 4.2 and 5.2 the paper uses a model based on sparse coding to theoretically explain task-dependent differences in memorization and in the attention patterns, respectively. However, it is not clear that the sparse coding model is a meaningful approximation of the behavior of the investigated LMs. The sparse coding approach uses a simple linear model, and it is a priori questionable whether such a model is sufficient to approximate large and highly non-linear LMs. Sections 4.2 and 5.2 provide no empirical evidence showing that the sparse coding model is a meaningful approximation of the investigated LMs, so the theoretical arguments made in these sections do not appear to be meaningful.
    4. The paper investigates memorization as a result of fine-tuning models. However, according to Figure 4, verbatim, idea and high-paraphrasation memorization appear to be in the same ballpark for T5-base and fine-tuned models. Therefore, it is not clear how much of the reported memorization is due to fine-tuning and how much is due to pre-training. To account for this, the authors should control for pre-training memorization in the memorization scores of fine-tuned models.
- The paper has several clarity issues which make it difficult to understand some of the results.
    1. The concept of idea memorization/plagiarism and how it is operationalized is not clearly defined. This makes it difficult to interpret the corresponding results.
    2. How is fine-tuning for the different tasks performed? E.g. for sentiment classification, what is the target output ($y$) that the models are fine-tuned to predict? A single word/token, or a short sentence?
    3. In Section 6, how do you use the base/non-fine-tuned models for summarization?
    4. I have difficulty interpreting the memorization examples in Table 11 in Appendix D. Which prompts were used to generate the machine written text?
- Contribution: The memorization detection approach and pipeline in the paper seems to be very similar to that of [A], upon which it builds. Given that the findings in the paper are unreliable (see above), the contribution is very marginal.

[A] Lee et al., Do Language Models Plagiarize?, WWW'23

### Questions
1. Is there only one split of each input $x_i$ into prefix $p_i$ and suffix $s_i$? If yes, the memorization behavior of the model might differ, based on the length of prefixes $p_i$, so it might be worth testing memorization for different prefix lengths. If no, the memorization ratio definition seems difficult.
2. What is the rationale for studying attention at the last decoder-encoder attention block in Section 5.1? Do the results look similar for other blocks in the network?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
1. The paper investigates the phenomenon of memorization within fine-tuned language models, specifically focusing on pre-trained T5 models applied to various downstream tasks, such as summarization, dialogue, QA, translation, and sentiment analysis.
2. The primary goal of the study is to comprehend the variations in memorization across these diverse tasks and determine whether fine-tuned models exhibit task-specific memorization patterns.
3. The paper introduces a categorization of memorization into three types: verbatim (exact memorization), paraphrase (memorization of alternative expressions), and idea (conceptual memorization).
4. Additionally, the authors propose a novel method for estimating memorization using attention scores and demonstrate the possibility of reducing memorization through multitask fine-tuning.

### Strengths
1. I like the idea of simplifying the concept of memorization by linking it to the extent of information required for a given task, which aligns with the sparse coding model.
2. The paper offers a systematic analysis that spans various tasks, providing valuable insights into memorization patterns across these tasks and comparing memorization across different language models.
3. The examination of attention scores and the presentation of encoded attention maps explains the initial claim well.
4. The observation that multitask fine-tuning can lead to reduced memorization highlights the practical implications of the study for model training and application.

### Weaknesses
1. Need for a more rigorous analysis of task specificity: The authors should consider potential confounding factors, especially the influence of the pre-training data used for the model, as this could impact memorization ratios. For instance, in the final section, the authors show that the T5 base model also shows a very high memorization ratio on the summarization task of multi-news which is 110 memorization ratio and the fine-tuned T5 model only goes to 222 which is twice more. However, the entire analysis of the paper before that was talking about how the summarization task requires more memorization because fine-tuning happened on that particular data set. I believe that a proper analysis of what was there in the pre-training data of a model and whether it confounds the memorization ratios is extremely important for such analysis and any conclusions about the task specificity or the nature of the amount of information required for the task. In fact, in the T5 base model that the authors chose, the entire C4 data set which it was trained on is totally public and the author should make such an analysis so that we can understand what is actually responsible.

2. The rationale behind introducing attention discrimination as a metric is unclear. If it merely correlates with existing, computationally cheaper metrics, it may not be necessary to introduce an additional metric without a compelling justification.

3. The experiment involving Flan T5, where the memorization ratio decreases after fine-tuning, lacks a clear explanation. The authors should provide a rationale for this unexpected result or revisit their analysis.

4. To draw meaningful conclusions, it is essential for the authors to compare the memorization scores of the T5 base model before and after fine-tuning.

5. The authors' choice of focusing on the last layers of encoder-decoder attention (as seen in Figure 2) appears somewhat arbitrary. An explanation for this choice and its potential impact on the correlation between attention scores should be provided.
6. Expanding the analysis presented in Figure 4 to include tasks beyond summarization would enhance the comprehensiveness of the paper's findings.
7. The concept of discriminating between different types of memorization: I believe idea memorization which encapsulates the fundamental sense of the initial information, should be a superset of all other forms of memorization. So, I do not understand this whole concept of categorizing memorization and what are we trying to achieve with this.
8. Please use % and not %. the latter is very non standard for ML papers.

### Questions
See weaknesses

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies memorization in fine-tuned language models. The authors observe a higher degree of memorization for models trained on summarization tasks, compared to simpler tasks like sentiment classification. Furthermore, they relate entropy in attention patterns to the degree of memorization in fine-tuned language models. Finally, they show that multi-task training reduces some memorization. They conclude their observations with a short theory on sparse coding models.

### Strengths
The main strength of the paper lies in its simplistic approach to studying memorization in fine-tuned language models. The authors take insights from existing memorization works on pre-trained models to conduct their analysis. The differences between different tasks showcase different mechanisms that models employ for each of them. Furthermore, the short theory on the sparse coding model helps formalize a reader's intuition for the observations. Overall, I believe the paper is a positive contribution to the community.

### Weaknesses
I have a couple of questions about the experimental setup and the observations that the authors draw from their results.

(a) What does x% memorization mean in the experiments? It would be great to demonstrate perfect and no memorization baselines to get a sense of the numbers.

(b) How do the authors measure Idea memorization?  Furthermore, how do they differentiate Idea memorization from summarization (which is the task of a summarization-tuned model)? 

(c) How much do hyperparameters during fine-tuning affect the memorization results? Does a lower learning rate and longer training time result in more memorization? 

(d) "k", the length of the prefix tokens, was fixed for all the experiments. How much do the observations vary with varying "k"?

(e) What is the decoding algorithm used for generation? Is it greedy decoding? If so, will the observations change with a more nuanced decoding algorithm, like nucleus sampling? This might allow the model to change its generation output, which will reduce the frequency of the generation of perfectly memorized solutions.

Overall, I believe the paper is a positive contribution to the community. I am happy to interact further with the authors during the rebuttal period.

### Questions
Please check my questions in the previous section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
