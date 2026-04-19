# Resolving Knowledge Conflicts in Large Language Models

- Decision: Reject
- Scores: 6, 6, 5

## Abstract
Large language models (LLMs) often encounter knowledge conflicts, scenarios where discrepancy arises between the internal parametric knowledge of LLMs and non-parametric information provided in the prompt context. In this work we ask what are the desiderata for LLMs when a knowledge conflict arises and whether existing LLMs fulfill them. We posit that LLMs should 1) identify knowledge conflicts, 2) pinpoint conflicting information segments, and 3) provide distinct answers or viewpoints in conflicting scenarios. To this end, we introduce KNOWLEDGE CONFLICT, an evaluation framework for simulating contextual knowledge conflicts and quantitatively evaluating to what extent LLMs achieve these goals. KNOWLEDGE CONFLICT includes diverse and complex situations of knowledge conflict, knowledge from diverse entities and domains, two synthetic conflict creation methods, and settings with progressively increasing difficulty to reflect realistic knowledge conflicts. Extensive experiments with the KNOWLEDGE CONFLICT framework reveal that while LLMs perform well in identifying the existence of knowledge conflicts, they struggle to determine the specific conflicting knowledge and produce a response with distinct answers amidst conflicting information. To address these challenges, we propose new instruction-based approaches that augment LLMs to better achieve the three goals. Further analysis shows that abilities to tackle knowledge conflicts are greatly impacted by factors such as knowledge domain and prompt text, while generating robust responses to knowledge conflict scenarios remains an open research question.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors introduce KNOWLEDGE CONFLICT, an evaluation framework for simulating contextual knowledge
conflicts and quantitatively evaluating to what extent LLMs achieve the following goals: 1) identify knowledge conflicts, 2) pinpoint conflicting information segments, and 3) provide distinct answers or viewpoints in conflicting scenarios.

### Strengths
1)  The authors design a series of knowledge conflict tasks to measure the performance of existing LLMs to generate response based on conflicted knowledge
2）The developed framework is technically sound and easy to follow

### Weaknesses
The most shorting is the orginize of the whole article and detailed questions can be found in the following part

### Questions
First of all, I really apprecitate the author start to investigate the problem of knowlege confilct in LLMs, which is urgent need to be solved in practical Chatbot. And the authors also developed a knowledge confilct framework equipped with a series of tasks to solve this question.


However, the originize of this paper is really poor, the authors mixed the design of each taskd and their method in the same part, which will confuse the readers. I really suggest the authors to pay more attention to the writting of this paper.

For technical details, in Task 1, I wonder which kind of pomprt you used in other baselines, like Few-shot prompting and Chain-of-Thought prompting (CoT), and your method in Table.1 indicates using the prompt ``Does the given context conflict with what you know? Yes/No''. 
Then, for each experimental result, I wonder if your method will include some examplers as demonstrations or just the prompt mentioned in your article?

Further, for NER in the first step, I wonder how you deal with the senario of there are multiple entity term in a sentence, will you enumerate them and generate context of parmetric knowledge for them?

Overall, the method is simple but will be useful in my understanding, but I wonder author have idea to how to combine your method with existing prompting method, like CoT, because there are still a lot of multi-hop question in practice and you will not know the knowledge confilct happens in which step, just like when you solve a math problem, you don't which confictled knowledge leads to a wrong intermediate result

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new evaluation framework called KNOWLEDGE CONFLICT to assess the abilities of large language models (LLMs) to handle knowledge conflicts. Knowledge conflicts arise when there is a discrepancy between the LLM's internal parametric knowledge and external non-parametric knowledge provided in the prompt context. 
To evaluate this, the framework includes tasks to test LLMs on:
-Detecting contextual knowledge conflicts
-Pinpointing conflicting spans in QA settings
-Generating distinct answers drawing on conflicting knowledge

The authors find that while LLMs can identify knowledge conflicts, they struggle with localizing conflicts and producing distinct responses. New instruction-based methods are proposed that improve performance on conflict detection and distinct answer generation. The analysis also reveals factors impacting conflict handling abilities.

### Strengths
- The topic is an important open problem of handling knowledge conflicts in LLMs.
- Writing is clear and well-presented. 
- Introduces a comprehensive evaluation framework with diverse, complex test cases

### Weaknesses
-Framework limited to word-level knowledge edits, more complex conflicts may be harder
The hallucination is possible in LLM's answer. It seems that this is not well addressed in the paper.

### Questions
- Could the assumption of a single parametric knowledge answer be relaxed? How would results change?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a framework to evaluate LLMs’ ability to handle knowledge conflicts, which includes: 1) identifying contextual knowledge conflicts, 2) pinpointing conflicting knowledge segments, and 3) providing distinct answers or viewpoints amidst conflicts. Under the setting the authors proposed above, the instruction-based approach is introduced to alleviate these problems.

### Strengths
1.	This article breaks down the evaluation aspects of knowledge conflict issues in a fine-grained manner and proposes a reasonable idea that LLMs should not rely solely on either parametric or non-parametric information, but grant LLM users the agency to make informed decisions based on distinct answers.
2.	For the three proposed tasks, this paper designed plenty of experiments for verification. The motivation is clear and the prompt templates are straightforward.

### Weaknesses
1.	The experimental settings are not rigorous. The data sets corresponding to the three knowledge conflict tasks are generated according to several rules (entity substitution and shuffling), and then the proposed approaches (prompt templates) are strongly related to these artificial rules. That is my main concern: with those settings, the experiments in the paper might have limited value and provide limited insights. Besides, this paper seems to lack a connection to previous works in the field of knowledge conflict. The organization of the entire article is isolated and does not introduce other benchmarks/analysis[1] or other method comparisons that specifically address knowledge conflict issues.
2.	The limited size of the knowledge conflict dataset proposed in this paper makes the analysis unconvincing. Take Figure 4 as an example, the authors argue that the ability to tackle knowledge conflict varies across knowledge domains. However, according to Table 7 in the appendix, on average there are only about 100 test cases per domain, which I think is far from enough to claim that knowledge conflict varies across knowledge domains.

[1] DisentQA: Disentangling Parametric and Contextual Knowledge with Counterfactual Question Answering. ACL 2023

### Questions
No more questions.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
