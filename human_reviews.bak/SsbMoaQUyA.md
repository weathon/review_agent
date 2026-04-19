# Exposure Bias Mitigation for Self Information Updating of Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 6, 5, 6

## Abstract
Current large language models (LLMs) have demonstrated remarkable capabilities in addressing users' requests for various types of information. However, these models are limited by the most recent data available in their pretraining corpora, rendering them incapable of providing up-to-date information. While periodically updating LLM pretraining corpora is possible, the optimal updating strategy remains underexplored. Retraining LLMs from scratch is cost-prohibitive, and the effectiveness of continual fine-tuning on new corpora has not been thoroughly examined. Additionally, current update procedures typically demand significant human involvement to convert the information into more structured format, such as knowledge triples, conversational data or responses with human feedback. In this study, we conduct a comprehensive examination of a novel \problem{} task in LLMs, which only requires the provision of informative text corpora without additional human intervention. For instance, we can use the latest news articles to update the LLMs' existing knowledge. We define the \problem{} task and assess the continual fine-tuning approach for this purpose. We formulate this task as a self knowledge distillation task where the teacher model is the original LLM with a new corpus as the context. We observe that the na\"ive distillation method can be problematic due to LLMs' exposure bias, which prioritizes existing information over new information that we aim to incorporate. When fine-tuned to accommodate instructions related to new information, LLMs tend to rely on pre-existing knowledge, neglecting recent facts and leading to incorrect reasoning chains that ultimately diminish the efficacy of information updates. Based on our theoretical analysis, we propose a straightforward yet effective method to mitigate exposure bias by incorporating the selected relevant facts into training losses. To validate our hypothesis, we develop two datasets to evaluate information updates, one derived from news articles published in March and April 2023 (the latest available news by the time of dataset collection) and the other derived from the Natural Questions benchmark. The latter has been chosen due to its provided link between questions and relevant passages from Wikipedia, which are utilized as the corpus for information updates and evaluation, respectively. Experimental results demonstrate that our proposed approach significantly increases the factual consistency score (on a scale from 0 to 1) by up to 0.16. Furthermore, we perform a preliminary investigation into the forgetting issue associated with this task, unveiling that our method, with a compact replay buffer of only 2.3\% of the training tokens, can significantly alleviate the forgetting problem. This study thus marks a significant stride towards optimizing the procedures for updating LLMs with the latest information, promising enhanced accuracy and efficacy.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studied the task of updating an LLM with new knowledge. The proposed method turns a corpus containing new information into instruction (question) and response pairs ready for fine-tuning. Existing LLM is used to do this task. Authors argue that existing LLM has a tendency of using previously trained knowledge than the new one, aka exposure bias. The proposed solution is to repeat the relevant new content before answering the question in model-generated fine-tuning data.

Experiments are done with an instruction-tuned LLAMA-7B and two new corpora (CNN, NQ). Model-based evaluation showed that the proposed method outperforms the baseline model and simple fine-tuning (adding facts directly and adding q&a pairs without repeating the new facts). To see how the model would forget old knowledge after fine-tuning, experiments with different sizes of samples from existing instruction tuning sets are done, which shows that even a small set of old instructions can prevent significant forgetting.

### Strengths
How to update the model with new knowledge is an interesting problem.

### Weaknesses
Reviewer orders the concerns by their importance.

- Probably a wrong assumption about the limits of incorporating new information into an old model's response. Retrieval-augmented generation (RAG) is a widely-adopted practice now, as seen in ChatGPT's browser-plugin and Google Bard. It can equip the models with fresh information without the need of retraining. The assumption made in Section 4 paragraph #3, "However, it is impossible to maintain an infinitely large memory to store the new information" is not really a problem given modern search engines / information retrieval techniques.

- The hypothesis of exposure bias is not properly validated. Higher evaluation numbers of context-aware fine-tuning data over facts and facts+naive are not a sufficient proof that exposure bias exists. Reviewer suggests either design a more direct experiments at exposure bias, or do detailed analysis of the wins/losses between the results of different methods to show that the differences are indeed caused by exposure bias.

- Model-based evaluation is less convincing. All evaluations in the experiments are based on models. Reviewer thinks it's important to look at the results, either by human raters or with human analysis, to make sure the differences are not caused by the model's bias.

- Evaluation lacks confidence intervals. Some of the differences are quite narrow, e.g. in NQ. The CNN set only has ~300 evaluation prompts (smaller). The evaluator itself is a model. All the factors call for a confidence interval on all numbers to make sure they are statistically significant.

- Poor writing. Spelling errors and non-sense characters possibly from lack of proof-reading are common.

### Questions
Here are some detailed questions/suggestions.

First of all, there are numerous simple writing errors and misspellings. E.g. 

- Last paragraph in Section 1, "Additionally, we perform an additional study", only one "additional" is needed. 
- Section 1 last paragraph last sentence (before "To summarize"). It's too long to be understandable.
- Second claim at the end of Section 1: what is "mbbvpropose"?
- Definition 2.4. "Informtaion"?
- Right before Equation 5, what is "vcvc?"

Reviewer will not try to enumerate all of them. Some professional proofreading is needed (maybe using an LLM to help spell check, at least?)

Other non-writing related questions

- Please explain any new math notions when they first appear. E.g. section 2.2 first paragraph, X_T? Or in Definition 2.4 I_s(C)?

- Right after Definition 2.3, the "here P(r|A,i,T)", it doesn't match anything in Equation (3).

- Equation 7 is very confusing. What do you mean by indicator function and what do they do here?

- Table 1: adding the new prefix in the response will effectively change the model's output. Do you mean we will need to do post processing from then on?

- Page 5 footnote 1. Not clear what the authors are doing.

- Section 4 2nd paragraph. GPT-4 has 170 trillion parameters? That's a trillion with a T.  What is the source of information?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Authors propose a solution to the novel problem of self-information update for LLMs, without human intervention. This is a very relevant problem in the context of LLMs, as it’s important for LLMs to be up to date with the latest knowledge, and the current solutions involve training the model from scratch, or fine-tuning the existing model with the latest data, which can lead to catastrophic forgetting. Authors argue that fine-tuning can also lead to a situation where LLM ignore the new facts and knowledge, because of the exposure bias. A simple and practical technique is then proposed to handle the exposure bias problem.

### Strengths
- Authors propose a simple solution to the important problem of updating LLMs with the latest information. Keeping the LLMs up to date with the latest facts and knowledge is essential for its’ practical use-cases as a personal assistant and other applications. Existing commercial LLMs take months before they are updated with the latest information from the web. 
- The solution proposed is simple to implement. It involves forcing the LLM to generate the news article first, followed by generating the response next. 
- Experimental results from fine-tuning the LLaMA-7B model demonstrate the effectiveness of the method on different metrics and different datasets.

### Weaknesses
- The theoretical analysis is not very rigorous, but I don't see this as a major problem, and in fact authors have addressed this in the manuscript itself (Remark after Definition 2.4).

### Questions
- In the Figure 2 (a), the performance after fine-tuning drops somewhere between [2K, 3K] replay examples, but then improves post 4K examples. How would someone applying this technique choose the replay buffer size to avoid the drop in the performance below the base model?

### Soundness
2 fair

### Presentation
3 good

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
This work addresses the problem of continual fine-tuning of LLMs using new training corpora. In contrast to previous approaches that involve significant human effort in converting unstructured data into more structured data to make the best use of new training corpora in fine-tuning LLMs, this work attempts to eliminate the human involvement in the process. It proposes a self information updating task for LLMs that makes use of unstructured new training corpora in the fine-tuning of LLMs without requiring human editorial intervention. The self information updating task is formulated as a self-distillation task with the LLM as the teacher and the new training corpora as the context. Specifically, this consists of first generating instruction response pairs from the new training corpora by using the LLM and then using the resulting pairs for fine-tuning. However, this naive approach is plagued by the issue of exposure bias where existing information from LLM is prioritized over the novel information in the new training corpora. This issue is identified and analyzed theoretically and it is noted that the bias affects both response generation and probability of instructions. 

The work proposes a heuristic for mitigating exposure bias. The key idea is to do a context-aware distillation which is essentially identifying the source of the instruction (new corpus or pretraining corpus) and making use of only probability terms that are relevant for the source. Given a training triple (i, r, d), where d is the article from the new corpus from which the LLM generated the instruction-response pair (i, r), the model is forced, during finetuning, to learn to generate d from i and then r from (d, i). Here d serves as the context for i to the LLM for generating i. 

The work presents results from experiments on two datasets - CNN News and NQ Val. The first data set consists of a small news corpus (50 CNN news articles from the period March and April 2023) with a resulting evaluation set of 301 instruction-response pairs generated using GPT-4. The NQ Val data set consists of extracted paragraphs from Wikipedia pages (long answers for questions in the validation split of the Natural Questions benchmark) with an evaluation set consisting of questions and short answers. As the baseline an instruction-following model from the LLaMA-7B fine tuned with instruction-following data from Alpaca and InstructionWild is used. (Of these, a randomly sampled 300 instruction-response pairs is used as old data.) The work answer consistency and context consistency as the metrics for evaluating the proposed approach as well as the baseline. Experimental study reveals that the proposed approach produces significant improvement over the baseline on both metrics for instruction-response pairs from new corpus while giving marginally negative improvement for instruction-response pairs from the instruction fine-tuning corpus. In contrast, a combination of principled but subject to exposure bias methods (namely, Fact + Naive) gives slightly lower improvement for instruction-response pairs from new corpus while not degrading performance on instruction-response pairs from the instruction fine-tuning corpus.

### Strengths
1. Addresses an interesting and relevant problem in the context of finetuning LLMs. 
2. Provides interesting and useful theoretical analysis that highlights the source of exposure bias problem.

### Weaknesses
1. Dropping completely probability terms based on context (to get Equation 7 from Equation 5) no doubt prioritizes novel information in the new training corpus over existing information from LLM for instruction-response pairs generated from the new corpus. But this also affects the model's performance on old instruction-response pairs as evident from Table 2. A possibly better approach is to use all the terms in Equation 5 but with source dependent weight that is determined by hyperparameter tuning. 

2. While the idea of generating instruction-response pairs from LLM for new corpus is interesting and eliminates the need for human effort, it is also a potentially problematic issue as current LLMs are known to hallucinate and as a consequence the instruction-response pairs could be affected by it.

### Questions
1. In Table 2, the answer consistency score of Context-aware is significantly higher than that of the baseline for new corpus. This is very encouraging indeed. However, the performance of  Context-aware on new corpus is significantly lower than that of the baseline for Old (0.480 vs 0.699 for CNN News and 0.256 vs 0.699 for NQ Val). What explains this gap and what can be possibly done to reduce it?  

2. In Table 2, the answer consistency score of the baseline for Old is nearly the same for both data sets (CNN News:0.696 and NQ Val:0.691). In contrast, Context-aware seems to be doing significantly better on CNN News than on NQ Val (0.480 vs 0.256). What explains this gap and what can be possibly done to reduce it? 

3. How many instruction-response pairs are in NQ Val and how many are used in evaluation?

4. From Figure 2.a it appears that some forgetting is inevitable in the proposed approach and increasing the no of replay examples doesn't really help alleviate this problem beyond a point. One wonders if forgetting will become more severe when fine tuning is done continually and with more datasets than the two used in the experimental studies. Figure 2.b seems to indicating this (score for old reduces further when the model already fine-tuned with NQ Val is further fine tuned with CNN News). What can be possibly done to address this issue?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The manuscript presents a comprehensive study of a novel approach to updating large language models (LLMs) with the latest information without requiring substantial human intervention. Recognizing the limitations of LLMs in accessing up-to-date information due to reliance on pretraining corpora, the authors propose a self-information updating task. The study also examines the issue of knowledge forgetting, proposing a solution with a compact replay buffer.

### Strengths
1、The study introduces a novel and practical method for updating LLMs with the latest information, which is a significant advancement in the field.

2、The research delves into the challenge of knowledge forgetting, offering a promising solution with a replay buffer mechanism.

3、The experimental results showing improvements in factual consistency scores are impressive and indicative of the method's efficacy.

### Weaknesses
1、The manuscript could provide a more in-depth analysis of how the integration of facts into training losses specifically counteracts exposure bias.

2、This paper only conducted experiments on factual consistency, which is not convincing enough. Furthermore, an explanation has not been provided as to why this task was chosen to demonstrate the effectiveness of the method for updating large language models (LLMs) with the latest information.

3、There are many unclear aspects in the formula explanations within the paper, leading to confusion. For example, it is not specified what the text sequence \( x \in X \) in the context before and after Equation 1 specifically represents.

Overall, the motivation of this paper and the explanations of the symbols used throughout the text are not very clear, and the expression still needs to be more clearly and concisely articulated.

### Questions
How does the setting of updating large models with new corpora proposed in this paper differ from continual learning and model editing? The paper lacks discussion and analysis in this regard.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a task called Self Information Updating (SIU) in LLMs, which requires the model to update itself using unstructured information sources, without the need for human intervention, resulting in a more practical task. Additionally, the paper introduces an approach, context-aware distillation, to address exposure bias, which tends to prioritize old information over new data when updating the LLMs. Finally, the effectiveness of this approach is demonstrated through evaluation using two new created datasets, revealing a notable 0.16 increase in the factual consistency score.

### Strengths
1- Reproducibility: The authors mentioned that their source code and data will be shared once their work is published. This means anyone can use these to reproduce their results.
2- Two new datasets were developed to evaluate information updates.
3- Factual consistency is increased by up to 0.16 using their new proposed approach.
4- The forgetting problem is addressed to some extent by using a small portion of the original training data.

### Weaknesses
How big can be the information updating corpus? If we have big information updating
corpus, how it will affect the performance?

### Questions
Please see the weakness section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
