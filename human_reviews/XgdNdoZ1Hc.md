# Clarify When Necessary: Resolving Ambiguity with Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 8

## Abstract
Resolving ambiguities through interaction is a hallmark of natural language, and modeling this behavior is a core challenge in crafting AI assistants. In this work, we study such behavior in LMs by proposing a task-agnostic framework for resolving ambiguity by asking users clarifying questions. Our framework breaks down this objective into three subtasks: (1) determining when clarification is needed, (2) determining what clarifying question to ask, and (3) responding accurately with the new information gathered through clarification. We evaluate systems across three NLP applications: question answering, machine translation and natural language inference. For the first subtask, we present a novel uncertainty estimation approach, Intent-Sim, that determines the utility of querying for clarification by estimating the entropy over user intents. Our method consistently outperforms existing uncertainty estimation approaches at identifying predictions that will benefit from clarification. When only allowed to ask for clarification on 10% of examples, our system is able to double the performance gains over randomly selecting examples to clarify. Furthermore, we find that Intent-Sim is robust, demonstrating improvements across a wide range of NLP tasks and LMs. Together, our work lays foundation for studying clarifying interactions with LMs.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigates the task of predicting whether it would be useful for an LLM to ask a clarifying question in order to answer to a particular input. The method samples different outputs from the model, clusters them using a separate entailment model, and then uses the cluster sizes to assign uncertainty and decide about asking a question. Evaluation is performed on three datasets (QA, MT, NLI) that contain manually created alternative disambiguations for each ambiguous input.

### Strengths
The topic is interesting. 
The idea of assessing the need for a clarification question based on downstream output distribution seems quite good. 
Evaluation on 3 different tasks and datasets is nice to see.

### Weaknesses
Based on my understanding of the paper, there is a serious methodological issue in the setup. In order to measure the uncertainty and decide whether a clarifying question is needed, the setup seems to use the manually created disambiguated intents from the datasets. Section 5.1, paragraph 2 refers to Appendix B for the full prompt and those prompts use the disambugations as input. 

In this example from the appendix, the generated clarification question and the implicit ambiguity that it tries to resolve are clearly guided by the provided intended interpretations, which have been manually created for that particular dataset:

Ambiguous Phrase: Jon will wash his car, and Mary will too. 
Intended Interpretation 1: Jon will wash his car, and Mary will wash hers. 
Intended Interpretation 2: Jon and Mary will both wash Jon’s car. 
Clarification Question: Will Jon and Mary wash the same or different cars?

Given that, this proposed method could not be used in a realistic setting, as it relies on a manually created set of disambiguations for a given input, which would not be available for any new input outside of the described datasets. 

Furthermore, the manually created intents in the prompt leak information into the generated questions and answers. The evaluation therefore measures whether it would be good for an unrealistic oracle to ask a question, not whether it would be good for a realistic system to ask a question.


Evaluating MT using only contrastive accuracy is problematic. The model can be quite far from the correct answer and still be counted as correct using this method. Some more common MT evaluation metric would have been reported as well.

The paper presents the separation of the task into 3 different subtasks as a contribution, although this has been done in previous works as well (https://arxiv.org/abs/2305.15933).

The paper has quite a large number of writing issues and grammatical errors, mainly preposition errors or missing/repeated words. The overall clarity of the paper and the technical details could be improved.

### Questions
Section 2 details that only aleatoric uncertainty should be measured for clarification. Why do you feel that epistemic uncertainty (missing knowledge) could not also be helped with clarifying questions?

Why are NLI results in the disambig setting identical regardless of whether they are uniform or sampled?

Table 4 refers to user sim, the paper text refers to intent-sim. I suspect that is not intentional.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new testbed for evaluating a system's ability to ask clarification questions, where a system needs to decide 1) when to ask a clarification question, and 2) ask a clarification question (though not evaluated in this paper), and 3) given a clarification question and a response, provide an answer. The paper re-purposed existing datasets and collected additional annotation in three tasks: QA, NLI, and MT, and designed automatic evaluation metrics for these tasks. The paper additionally proposed a new method, Intent-Sim, which scores the uncertainty by estimating the answer entropy of a greedily decoded question.

### Strengths
The paper is generally clearly written. The tasks and datasets are well-motivated, and the evaluation results are comprehensive.

### Weaknesses
1. It'd be helpful to know how well humans perform on this task (e.g., AUC ROC for deciding whether to ask clarification questions).
2. It'd be also useful to evaluate GPT-4 on this task. This task feels straightforward to me and I suspect that GPT-4 is on par with human-level (I could be wrong though)
3. I am uncertain whether Intent-Sim actually helps -- In table 2, it seems that it does not significantly outperform the random baseline? 
4. The paper did not evaluate the quality of the clarification question. While I do agree with the authors that they are hard to evaluate and control, I feel that this is the most challenging and interesting part of the task.

### Questions
^ See weaknesses 1-3 above.

### Soundness
3 good

### Presentation
4 excellent

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
This paper proposes a framework for resolving ambiguity which consists of three subtasks.  The first task is to decide when the clarification question is needed. The second task is to determine what clarification question to task. And the third task is to generate the final response conditioning on the input and the clarifying QA pair. The authors apply their framework on QA, MT and NLI tasks. The experimental results show that with clarifying QA pairs and disambiguated inputs, the downstream task performance tends to increase. Furthermore,  INTENT-SIM is proposed which explicitly estimates the ambiguity by calculating the entropy over user intents. Results show that the proposed system always outperforms the random baseline under all interaction budgets.

### Strengths
- Some common ambiguity types are summarized and analyzed, which could inspire future search.
- Multiple NLP tasks are evaluated.
- Multiple data generation settings are evaluated.

### Weaknesses
- Clarification question generation is based on the oracle prompting method, which may not be feasible for real-world applications.
- The compared baselines seem to be weak. The authors cite Cole et al. 2023 in Section 2 but do not compare with their method. Additionally, the improvements presented in Table 4, particularly regarding User similarity over Random, sometimes appear to be statistically insignificant.
- I understand that the authors try to show the generalization of the proposed framework on multiple NLP domains. But the setup for NLI and MT seems to be unnatural. It also makes the comparison across different tasks difficult. It is more intuitive to experiment on more QA datasets or dialogue systems, like [1].

[1] Prompting and Evaluating Large Language Models for Proactive Dialogues: Clarification, Target-guided, and Non-collaboration, Findings of EMNLP 2023

### Questions
- Why did you choose to conduct experiments on NLI and MT tasks rather than on more QA datasets or dialogue systems, where clarification questions naturally occur?
- Why not compare with the method proposed in Cole et al. 2023?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on the problem of handling ambiguous inputs in large language models. Specifically, it addresses three sub-problems: determining when to clarify model inputs, deciding on what clarifying questions to ask, and how to incorporate information obtained from clarification. The paper introduces a framework that analyzes common types of ambiguities across three different tasks and how large language models can use clarifying interactions. And, the paper finds that clarifying interactions can effectively enhance model performance and introduces a new method, INTENT-SIM, to measure model uncertainty and thereby decide when to pose clarifying questions. Experimental results indicate that the proposed method consistently outperforms the random baseline across different interaction budgets.

### Strengths
1. The research presented in this paper is well-motivated. The authors provide a clear description of the research problem, and the entire paper is well-organized and easy to follow.
2. The ambiguity analysis framework proposed by the authors is both simple and easily implementable. Furthermore, they demonstrate its effectiveness across various datasets and tasks, showing that the use of clarifying interactions indeed improves model performance.
3. I also think that the INTENT-SIM algorithm is innovative, and the strong empirical results in the experiments support its effectiveness.

### Weaknesses
I have concerns about the complexity of the INTENT-SIM algorithm. As it involves constructing graphs using external models and graph computations, this could significantly impact the computational efficiency of large language models.

### Questions
Why were these three specific tasks chosen for testing?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
