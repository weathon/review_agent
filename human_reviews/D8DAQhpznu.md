# Llamas Know What GPTs Don't Show: Surrogate Models for Selective Classification

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 3

## Abstract
To maintain user trust, large language models (LLMs) should signal low confidence on examples they get incorrect, instead of misleading the user.
The standard approach of estimating confidence is to use the softmax probabilities of these models, but state-of-the-art LLMs such as GPT-4 and Claude do not provide access to these probabilities.
We first study eliciting confidence linguistically---asking an LLM for its confidence in its answer---but we find that this leaves a lot of room for improvement (79\% AUC on GPT-4 averaged across 12 question-answering datasets---only 5\% above a random baseline).
We then explore using a \emph{surrogate} confidence model---using a model where we do have probabilities to evaluate the original model's confidence in a given question.
Surprisingly, even though these probabilities come from a different model, this method leads to higher AUC than linguistic confidences on 10 out of 12 datasets.
Our best method mixing linguistic confidences and surrogate model probabilities gives state-of-the-art performance on all 12 datasets (85\% average AUC on GPT-4).

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the role of confidence in improving the LLM performances in QA tasks. Some LLMs do not output confidences, so a new method to elicit the confidence is necessary.

This paper first considers “linguistic confidence”: prompt the model to output a notion of confidence. However, for the models that provides accesses to probabilities, the performances from the linguistic confidence are worse than the performances from the probability confidence.

This paper proposes using surrogate models (i.e., some models where we do have access to their probabilities) to estimate the confidence. On 10 out of 12 datasets, the method has a higher AUC than the “linguistic confidence” method.

### Strengths
- The proposed confidence estimation approach is elegant. It’s simple, and it works well.
- There are extensive experiments showing that the proposed approach (and its variants) work on a wide range of problems.
- There are also extensive ablation studies showing the different combinations of the surrogate models. (Some studies are not covered though — please refer to my comment below.)

### Weaknesses
- The proposed algorithm seems to have limitations. The proposed algorithm 1 still requires the main model to output linguistic confidences (unless alpha=1), which is a confidence score less as good as the probability scores.
- The evaluation can be more rigorous. I have been looking for the evaluations for the validity of the linguistic confidence. Specifically, how well do they correlate to the probabilities directly outputted by the models? The evaluation scores presented in this paper focused on the utility of these confidence scores though.
- The value of a crucial hyperparameter is not reported. Alpha, the scaling factor between the two confidence scores, seems very important for the overall AUC / AUROC performances. The actual values for the optimal settings, or the approaches to reach the optimal values, are not reported.
    - A related note, the heading of the second paragraph in 5.1 (”Epsilon is all you need”) seems to indicate that very small alpha values are sufficient, which is obviously not the case, considering that “Tiebreak” and “Surrogate” settings have quite different results. In general, a claim like “XYZ is all you need” usually leave me with a (perhaps wrong here) impression that the paper is a social media post rather than a scientific paper.
- An intuitive extension of the algorithm could have been explored. Since using one surrogate works well, does combining two surrogate models work?

### Questions
What do the “+” symbols at the end of many methods in tables 2 and 3 mean?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper leverages the open-source LLM known as "llama 2" to assess the uncertainty or confidence of outputs from the black-box LLM model, GPT-4. The authors demonstrate that by using the llama 2 confidence scorer, one can achieve a higher AUC. Moreover, the paper introduces a novel mixture function designed to combine outputs from multiple confidence-scorer models, ultimately resulting in an optimized scorer.

### Strengths
- Assessing the uncertainty of black-box language models represents a significant and intriguing research direction.
- Leveraging the probability metrics from an open-sourced language model is an intuitive approach.
- The authors provide comprehensive AUC results from a variety of confidence-scorer models and policy models. These findings will be valuable for future researchers when choosing a confidence scorer.

### Weaknesses
- Soundness: The methodological soundness of this study appears somewhat lacking. There's a noticeable lack of a baseline comparison in the work.
  - While the paper primarily focuses on uncertainty or confidence, it doesn't compare with established certainty scorers. It would be beneficial to discuss relevant works such as [1] and [2] and incorporate them in the experimental section.
  - The study also touches on the critiquability of LLMs and LLM evaluation. Including references [3], [4], and [5] in the related work and experiments would provide more depth and context to the discussions.

- Novelty and Contribution:
  - The approach of using surrogate models to interpret black-box models isn't novel.
  - The introduced mixture function, essentially a simple linear combination, raises questions regarding its uniqueness. A clearer differentiation from existing methods might strengthen this section.

- Clarity and Writing Quality:
  - The manuscript could benefit from further editing for clarity and structure. For detailed feedback, refer to the 'Question' section.

[1] Uncertainty Quantification with Pre-trained Language Models:A Large-Scale Empirical Analysis

[2] Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models

[3] Self-Refine: Iterative Refinement with Self-Feedback

[4] CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing

[5] A Survey on Evaluation of Large Language Models

### Questions
In Table 1, for every row, is the policy model identical to the scorer model, effectively making it a self-scorer? As an instance, does "Text-davinci Prob" employ "Text-davinci" as both its policy and scorer model?

Regarding the statement "embeddings of questions that GPT-4 gets incorrect" – can you provide clarity on how these embeddings are derived or obtained?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to use open-source models such as LLaMa as a proxy for finding confidence estimates for models that do not provide probabilities, such as GPT or Claude.

### Strengths
The paper has fairly extensive experimentation over a large number of tasks.

### Weaknesses
I feel that this method introduces additional complexity in a place where it is not clearly needed, and because of this I am skeptical of whether this method will see wide adoption should the paper be accepted to ICLR.

Specifically, I am not convinced of the underlying premise of the paper, that you cannot get probabilities out of closed models. Specifically, it is well known that sampling can be used to approximate probabilities (see the "Pattern Recognition and Machine Learning" textbook for example), and all closed models that I know of support sampling. Xiong et al. empirically demonstrated that this is a quite effective way of getting probability estimates out of models, and this is much easier than additionally running a separate proxy model to get probability estimates. The mixture of surrogate probabilities method indeed marginally beats the best method of Xiong et al. (by 0.4% AUC for example), but this doesn't seem to warrant the additional complexity.

### Questions
None in particular, although I would be open to arguments about why this method may be preferable over other simpler alternatives.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper is focused on confidence elicitation for models that do not provide confidence probabilities their answers. Such models include GPT-3.5., GPT-4 and Claude. Linguistic confidences are obtained by zero-shot prompting the models to assign confidence scores to their answers. The linguistic confidences are evaluated in a selective classification setting (where the goal is to have confidence scores that are calibrated with the correctness of the answers). Two metrics are used for evaluation: AUC (area under the coverage-accuracy curve) and AUROC (area under the receiver operator curve). Experimental results using 12 standard question answering datasets show that the linguistic confidences are not much better than random guesses. Furthermore, they are worse than model probabilities from surrogate models such as Llama-2 variants. The best results are obtained when linguistic confidences are mixed with surrogate model probabilities.

### Strengths
The paper includes an extensive number of experiments over 12 standard question answering datasets and the results are consistent over the 12 datasets. 

It is interesting to know that surrogate model probabilities are a better indicator of confidence than the linguistic confidences. It's also interesting to see that combining surrogate model probabilities and linguistic confidences improves the results.

### Weaknesses
The paper mainly consists of a large set of well-conducted experiments, but lacks the depth.

While the results are interesting, they are actually not very surprising. The mixture of models approach is very straightforward. 

The discussion of the results is not very insightful. Given the focus of the paper, it would be interesting to better understand the reasons the models are not good at eliciting good linguistic confidence scores. While the authors claim that error calibration is not the focus of the paper, it would be interesting to know how uncalibrated surrogate models perform by comparison with calibrated surrogate models.

### Questions
N/A

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
