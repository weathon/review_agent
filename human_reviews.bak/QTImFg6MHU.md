# Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
We introduce BSdetector, a method for detecting bad and speculative answers from a pretrained Large Language Model by estimating a numeric confidence score for any output it generated. Our uncertainty quantification technique works for any LLM accessible only via a black-box API, whose training data remains unknown. By expending a bit of extra computation, users of any LLM API can now get the same response as they would ordinarily, as well as a confidence estimate that cautions when not to trust this response. Experiments on both closed and open-form Question-Answer benchmarks reveal that BSdetector more accurately identifies incorrect LLM responses than alternative uncertainty estimation procedures (for both GPT-3 and ChatGPT). 
By sampling multiple responses from the LLM and considering the one with the highest confidence score, we can additionally obtain more accurate responses from the same LLM, without any extra training steps. In applications involving automated evaluation with LLMs, accounting for our confidence scores leads to more reliable evaluation in both human-in-the-loop and fully-automated settings (across both GPT 3.5 and 4).

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a method designed to detect unreliable answers from LLMs by determining a numeric confidence score for their outputs. This technique is applicable to LLMs accessed through a black-box API, even if their training data is undisclosed. Compared to other uncertainty estimation methods, BSDETECTOR more proficiently spots incorrect answers from LLMs, including GPT-3 and ChatGPT. Furthermore, by selecting the response with the highest confidence score among multiple LLM outputs, the accuracy of the responses can be enhanced without additional training. When these confidence scores are incorporated, evaluations involving LLMs become more dependable in both human-involved and fully-automated scenarios.

### Strengths
1. The paper highlights the importance of uncertainty quantification for LLMs and presents a novel approach to analyzing the uncertainty. The idea is straightforward and effective: Multiple responses are sampled from the LLM, and the one with the highest confidence score is considered the most accurate response.
2. The method works for any LLM accessible only via a black-box API, without knowledge of the training data.
3. The method focuses on Question-Answering applications but mentions that the uncertainty estimates can be applied to other prompts as well.

### Weaknesses
1. The authors may overclaim the contribution of this paper. Specifically, the authors mentioned several times that the proposed methods can be applied to any LLMs on various tasks. However, the experiments only contain OpenAI's APIs' performance on QA datasets.
2. A large number of uncertainty quantification methods are missing, which makes the related works section incomplete and lacks a fair model assessment.
3. Letting ChatGPT explain itself seems quite straightforward. Since ChatGPT is trained with RLHF, it rarely outputs uncertain answers in ChatGPT's opinion. I feel like training a surrogate model to make the answer judgment is more intuitive.

### Questions
1. Other than AUROC, it would be also good to show the accuracy of predictions, which can give a basic sense of the confidence score.
2. What is $\omega_2$ in Sec. 3.3? Should it be $\beta$?
3. How do you find the value of $\alpha$ and $\beta$? Have you conducted any sensitivity analysis on hyper-parameters?
4. Could you give a formal definition of confidence selection and random selection in Sec. 6.3 with more details?
5. Finally, a large number of baselines are missing. e.g., [1] and [2].

[1] Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in
words. arXiv preprint arXiv:2205.14334, 2022a.
[2] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for
uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664, 2023.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a confidence score named "BSDETECTOR" which is designed to detect failures, improve accuracy by identifying answers with highest score, and evaluate the quality of LLMs' generation. 

The "BSDETECTOR" score is derived from a linear combination of a consistency score and a self-reflection score. For the consistency score, multiple responses are generated by varying the prompt for the same question (as demonstrated in Figure 6 (a)). The confidence score for a specific response is then determined by calculating the average similarity between this response and the others. On the other hand, the self-reflection score is obtained by posing questions to the model such as, "Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure?" followed by, "Are you really sure the proposed answer is correct? Choose again."

To evaluate the performance, the author focuses on two tasks: 
1) Failure prediction: Evaluating the performance of failure prediction on QA dataset using AUROC and ACC.
2) Active Learning: For the task of Summarize-from-feedback, the authors use LLM to assess the quality of the generated summaries. Then they have a budget to ask human to give their evaluation. Compared to random selection, selecting the K samples with the lowest "BSDETECTOR" scores for human evaluation results in a closer alignment with the ground truth evaluation (i.e., human evaluations provided by the dataset).

### Strengths
Originality/Significance: The author proposes a straightforward metric to assess whether the LLM's generation is reliable, which is a linear combination of consistency score and LLM self-evaluation score. Furthermore, the author evaluates its performance by applying it to the task of active learning.

### Weaknesses
**Originality:** The paper extends the framework of semantic uncertainty[1]: sampling multiple answers and computing their semantic similarity. Unlike the previous work that utilised the model's own log-likelihood to compute the final uncertainty, this paper leverages the average semantic similarity generated by NLI to indicate uncertainty. The rationale is that high similarity to other candidate answers implies the answer is more likely to be correct. Moreover, the authors also introduce a "self-reflection score" to enhance performance. The linear combination of consistency and self-reflection score is simple, but the paper lacks sufficient ablation studies to show their complementary nature, making this contribution appear somewhat limited.

[1]Kuhn, Lorenz, Yarin Gal, and Sebastian Farquhar. "Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation." ICLR 2023.

**Quality:** The experiments lack validating ablation studies: 1) what is the relationship between consistency and self-reflection score? Are the complementary or repetitive (i.e., the sample groups that they succeed and fail are the same)? What is the reason for specific weights in equations (like 0.7 for consistency in Eq. (2) and 0.8 in Eq. (1)), or settings (like T=1 for temperature sampling). Justifications for these choices are missing.

**Clarity:**

1. Several descriptions are ambiguous. For instance, what does "counts" in figure 5 refer to? Does mean the number of samples for each trial's dataset or is it the number of trails? Which dataset was Table 4 based on?
2. Some statements might be inaccurate. For instance, the statement about "AUROC represents the likelihood..." seems off. The correct logic should perhaps be about a "certainty score" rather than an "uncertainty score". 
3. Some descriptions are hard to understand: e.g., the sentence "Note here we only modify the prompt used to sample varied responses for computing the observed consistency, not the prompt originally given to produce the original reference response" in Sec 3.1 "Producing Diverse Output". How do you modify the prompt? For Sec 3.2's "Through these additional steps, xxx" what does this sentence mean? 
4. Necessary citations are missing in places like the introduction "any stochastic text → text mapping" (btw, what does this mean?) and Sec 3.2 "Like CoT prompting, self-reflection allows the LLM to ...".

**Significance:** The paper misses a discussion on the reliability of LLM's generation, especially in determining whether an answer from LLM is "good" or correct. It seems like the authors are relying on LLM's capability to distinguish good from bad by using description, which is questionable. If LLM's performance is weak, the improvement this method offers may be limited. The introduction of consistency suggests an ensemble approach, but its reliance on semantic similarity, which is dependent on an NLI model, is also limited, especially for long-sequence texts. This might explain the authors' choice of evaluating on datasets like "trivia qa" and "summarize from feedback" rather than the summarization itself.

### Questions
Refer to the above sections for other questions. 
1. For the section titled "Sec 6.1 dataset", how exactly was the "we also manually validated the accuracy of LLM-generated responses" conducted?
2. In the last paragraph of "Sec 3.2": If you ask LLM, "Are you really sure the proposed answer is correct? Choose again", will this prompt introduce some bias? For instance, GPT-4 often exhibits sycophantic behavior[2]. If you continually question it on an issue, even if its response is correct, it might change it to an incorrect one. How do you address this issue?

[2] Liu, Yang, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. "Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment." _arXiv preprint arXiv:2308.05374_ (2023).

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method for quantifying the uncertainty of the answers of language models, in an opaque-box fashion. This means that the methods leverages alternative prompting techniques and multiple model calls to establish uncertainty but it does not require model weights or training data. The method uses observed consistency (established through multiple model calls at different temperatures on the same question) and self-reflection consistency (established through follow up model calls that clarify accuracy of an initial given answer). The method is evaluated by showing the performance of the uncertainty scores in predicting answer accuracy and also on its usefulness for improving model performance in general.

### Strengths
S1 - Uncertainty quantification is an important problem and even more relevant in the era of generative models

S2 - The notions of observed consistency and self-reflection consistency are fundamental and backed up by previous research and empirical evidence.

### Weaknesses
W1 - Experiments and results: It seems like the work uses standard prompting throughout the experiments and not chain of thought reasoning. CoT is necessary to see in all relevant results and even for uncertainty quantification for a fair comparison. For example, in Table 2 authors show how their technique can improve model performance but then in many of those datasets CoT offers a better result based on previous work. Similar techniques to cot can be used to also have the model report its own uncertainty and reasoning within the same model call.

W2 - Presentation: The authors can consider toning down the claims in the paper, with respect to the applicability of the results and breadth. There are several points in which the results are overly claimed in a misleading way that can convey confusing information to the reader. For example,

E1- "This paper primarily focuses on Question-Answering applications, but our same uncertainty estimates can also be applied to estimate how confident the LLM is in its response to a more general prompt." - To make such a claim one needs to evaluate and adapt the method such that it applies to longer generations and estimate uncertainty on different parts of the text. This is non trivial and still an open problem for longer generations, but the way how this paragraph is phrased makes it seem like this is straightforward.

E2- "Section 6.2 While answers produced via the BSDETECTOR filtering procedure from Section 4 require 10x as much LLM-inference computation as the Reference Answer, the consistent accuracy gain observed in Table 2 makes this worthwhile for high-stakes applications." The accuracy gains in Table 2 are in many cases modest, and in other cases not competitive with Chain Of Thought results presented in several previous works. https://github.com/FranxYao/chain-of-thought-hub is an example of results but similar results have been reported in papers discussing standard cot, tree of thought, diversity of thought, and self consistency

W3- Relationship to related work and novelty. In overall, the paper could do a better job in connecting the work with previous results and clarifying novelty with respect to these. See below:

Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation
https://arxiv.org/abs/2305.15852v1
This is relevant to self-reflection consistency

Self-consistency improves chain of thought reasoning in language models https://arxiv.org/abs/2203.11171 
This is relevant to observed consistency

### Questions
Q1- For future revisions, authors could consider clarifying for each figure and table what type of evaluation was used (manual human annotation on evaluation, automated, llm-based, etc). It is sometimes hard to follow which parts were evaluated through other models (or not)

Q2- Section 3.3. Does w2 refer to beta here? How was this parameter fixed to 0.7? Does this mean that self-reflection consistency is more useful for the models in the study?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
