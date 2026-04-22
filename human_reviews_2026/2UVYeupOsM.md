# IntelliAsk: Learning to Ask Critical Questions with Human-Aligned Rewards

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 2

## Abstract
Peer review relies on substantive, evidence-based questions, but existing LLM-based approaches often generate surface-level queries. We find that LLM-generated questions take over 50\% of their question tokens from a paper’s first page, while human reviewers draw on the full text. Human questions are also more insightful, showing effort and grounding, whereas LLM questions mostly reflect surface style. To address this, we extract 151k candidate questions from ICLR 2024 reviews and filter them through a multi-stage filtering process into Probe-15K, a set of 15.5k high-quality questions. From this, we create ProbeVote-500, where human annotators score questions along effort, evidence, and grounding. Using these labels, we train IntelliReward, a reward model built from a frozen Autoregressive LLM with trainable multi-head transformers over the final 50 token states. This architecture outperforms API-based SFT finetuning (Gemini 2.5 Flash, GPT-4.1) as baselines for reward. Applying DAPO with IntelliReward, we train IntelliAsk, a question-generation model aligned with human preferences and substantially stronger than existing fine-tuned review models. Finally, by releasing Probe-15K, ProbeVote-500, and IntelliReward, we provide an automatic evaluation benchmark for reviewer questions that measures groundedness, effort, and evidence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors curate two datasets — Probe-15K (15.5k filtered reviewer questions) and ProbeVote-500 (expert-annotated with Effort, Evidence, and Grounding scores). They then train a reward model (IntelliReward) based on human preference data and use reinforcement learning (DAPO) to fine-tune an LLM (IntelliAsk) for question generation. Experimental results suggest that RL with the proposed reward model yields slightly more "human-like" questions than supervised fine-tuning (SFT).

### Strengths
- The paper is well-written and clearly describes the dataset construction and filtering pipeline.

- The release of Probe-15K and ProbeVote-500 may be useful for future studies on reviewer question generation or automated peer review.

### Weaknesses
-  The technical approach is essentially standard supervised or reinforcement learning with human-labeled data — a typical preference distillation setup. The work lacks novel modeling ideas or theoretical insights. Which means this work reads more like an engineering project for data cleaning and reward tuning than a research contribution that advances our understanding of question generation or alignment.


- The structure of the paper is confusing, with tables and figures difficult to locate, and important experimental details scattered across multiple sections or the appendix. The overall reading experience is quite poor.

- The evaluation is mainly an in-domain test, using the same question and paper.  Lack of proof of the generalization of the model.

### Questions
- How does IntelliAsk differ conceptually from standard RLHF or preference-based fine-tuning?

- Have you evaluated generalization on other venues (e.g., NeurIPS, CVPR) or non-ICLR review data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper creates resources for training models that ask questions about scientific articles as reviewers in the peer-review process would. They create a dataset of articles and associated review questions by filtering data from ICLR 2024. They define evaluation criteria for reward quality and use human annotators to compare the quality of LLM-generated questions and human-written questions. Next, they train a reward model to automate question scoring. They show that performing SFT is insufficient for generating high-quality reviews and that RL with their reward model can do better.

### Strengths
- The paper addresses an important problem of generating high-quality questions about scientific articles which is a part of understanding these articles and using models to accelerate science
- The authors are careful with their curation of the review question dataset, filtering for only high quality questions and performing deduplication.
- The work highlights an important property of SFT of primarily learning to mimic style and being insufficient for performing well at a desired but indirectly specified task (in this case, generating high-quality review questions)
- The reward model trained for assessing question quality is substantially stronger than strong baselines.

### Weaknesses
- From what I saw, IntelliAsk is only evaluated by the reward model IntelliReward. From what I understand, IntelliReward also provides the signal for training IntelliAsk. So, it seems possible that IntelliAsk is just reward hacking IntelliReward rather than actually producing better questions. Why isn’t there a human-evaluation for IntelliAsk?
- This paper largely applies existing methods to a novel dataset/task, so its contributions seem limited. I’m not sure what to take away from this paper besides the fact that generating review questions cannot be done well by current LLMs and SFT is insufficient for substantially improving at this task. A lot of the content of the paper is in describing the design details of the pipeline for creating the benchmark and not on insights.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the problem of generating critical review questions for peer review using LLMs. The authors conduct a study in which they compile a dataset of 15,000 reviewer questions extracted from ICLR 2024 papers through a multi-stage filtering process designed to identify high-quality review questions. They also compare these questions with questions generated using powerful reasoning models and find a substantial quality gap between human and LLM-generated questions. To address this gap, the authors propose a reward model, called IntelliReward, that evaluates the quality of review questions along three dimensions: effort, evidence, and grounding. Then, they use this reward model to train a question generation model (IntelliAsk) using reinforcement learning and show that the questions generated by IntelliAsk outperform those generated by a model trained solely through supervised fine-tuning.

### Strengths
- The paper is easy to follow and well-written. 
- The proposed datasets, Probe-15K and ProbeVote-500, are valuable resources for advancing research in high-quality question generation and evaluation.
- The fact that SFT does not help much with creating better question generation models is an important problem studied in this paper, and their proposed solution for training a reward model is interesting.

### Weaknesses
- The design of IntelliReward is not well-motivated. How did the authors come up with the frozen causal LLM + TransformerResidualHead idea?
- Some statistics about the ProveVote-500 are missing. It’s not clear what the score distribution is for the 3 dimensions (effort, evidence, and grounding) in this dataset. Knowing these statistics can help understand Table 1 better. 
- The rubric designed to assess question quality is not well motivated. Also, using a 0/1 score for a question seems to be a simplistic assumption.

### Questions
- Why do you define keyword coverage like that? Can you provide more analysis on why this metric is a good representative of the coverage of a question? 
- Can you provide an analysis of how much the IntelliReward score aligns with Human Evaluated scores for question quality evaluation?
- Why does it make sense to evaluate IntelliAsk using IntelliReward? 
- Why is IntelliAsk failing in the coverage metric compared to SFT, OpenReviewer, and DeepReviewer (table 3)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims at generating better questions using language models for reviewing papers. It makes two contributions: 
1. The authors release a dataset and trains a reward model for assessing review question qualities 
2. To demonstrate the effectiveness of the reward model, they train another language model to generate questions regarding a paper, and shows it can achieve better performance compared to a baseline.

### Strengths
1. The paper addresses an interesting and important problem: the current language models often ask superficial questions and it’s interesting to study how to enable these LMs asking more in-depth and insightful questions, which is important for building automating scientific workflows. 
2. The released dataset and resource can be helpful for the community, and the automatic dataset curation pipeline and cleaning code can be useful for other use cases.

### Weaknesses
1. While I like the motivation of this work, one major weakness of this paper is its **experimental design**. I have a lot of questions regarding the experimental details, and I think there are quite a few parts that are not rigorous and impact the soundness. 
2. Also the presentation of the paper can be improved
    - For example, in section 4, there’s no pointer to the evaluation results of the model in table 3. 
    - For the tables, please use consistent numerical styles, i.e., choosing either percentage or absolute numbers and use it in all tables, rather than using different styles in table 1 and 3.

### Questions
In the annotation process: 
1. Why only using a binary score but not likert scores? 
2. Also have you explored with another annotation method – for example, there can be multiple review questions for a paper. In that case, another approach is to have people label “preferences”,  i.e., contrasting two or more questions and picking the preferred ones. 
3. What is the inter-annotator agreement in the annotation? I can imagine it’s relatively hard to achieve good agreement among people given this super challenging dataset 

In the experiments design and evaluation
1. Sec 4 makes the claim that SFT may not be suitable for this task. However the evaluation seems to be less rigorous: 	
    - If the evaluation is correct, it can only prove that training on the 15k data may not be helpful, and it’s hard to make the general claim that SFT is less useful. 
    - Also the evaluation seems to be relatively less rigorous – there’s only some qualitative studies on the annotation qualities and some approximated model-based evaluations.   
2. How is the Intellireward evaluated? As far as I can tell, it seems the authors fine-tuned the LLM on the ProbeVote-500 dataset, and then conducted  evaluation on the (training) data? Then it seems natural that the model can achieve way better performance compared to the other baselines reported in table 1. 
3. Also the comparison between the IntelliAsk model and the Qwen2.5-7B SFT model seem to be not fair as they are trained on completely different datasets. Why not directly conducting SFT on the “preferred questions” rated by IntelliAsk and compare the model performances?

### Soundness
2

### Presentation
2

### Contribution
2
