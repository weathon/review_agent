# OceanGPT:  A Large Language Model for Ocean  Science Tasks

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5

## Abstract
Ocean science, which are reservoirs of life and biodiversity, is of paramount significance given that oceans cover over 70% of our planet's surface. Recently, advances in Large Language Models (LLMs) have transformed the paradigm in science. Despite the notable success in other domains, current LLMs often fall short in catering to the needs of domain experts like oceanographers, and the potential of LLMs for ocean science is under-explored. The intrinsic reason is the immense and intricate nature of ocean data as well as the necessity for higher granularity and richness in knowledge. To alleviate these issues, we introduce OceanGPT, the first-ever LLM in the ocean domain, which is expert in various ocean science tasks. We propose DoInstruct, a novel framework to automatically obtain a large volume of ocean science instruction data, which generates instructions based on multi-agent collaboration. Additionally, we construct the first oceanography benchmark, OceanBench, to evaluate the capabilities of LLMs in the ocean domain. Though comprehensive experiments, OceanGPT not only demonstrates a higher level of knowledge expertise for oceans science tasks but also gains preliminary embodied intelligence capabilities in ocean technology. Codes are in the supplementary materials and will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper provides a fine-tuned version of LLaMa 2, OceanGPT, which is meant to be specifically used for ocean studies and related tasks. It was also finetuned by a series of GPT 3.5 derived agents, allowing it to carry out a variety of tasks, such as providing instructions for underwater robots and extracting relevant parts of literature.

### Strengths
- The authors aim to train a model specifically for an under-served field of study, that of oceanography
- They create a benchmark for testing AI models capabilities on ocean-adjacent tasks
- They create a corpus of open-access literature about ocean studies.

### Weaknesses
- The 'instruction seed generation' approach described in section 3.2. is fundamentally flawed because it utilizes other LLMs to generate this data, meaning that it is liable to contain hallucinations and not be reliable from a scientific perspective
- The evaluation carried out has several significant shortcomings (see my questions below) - notably that there is a lack of details regarding how it was carried out, or how reliable GPT4 is as an evaluator.
- The insistence of the authors on "embodied intelligence" is not proven or described in detail, and the whole section about the instructions for underwater robots is unclear to me - how this testing was carried out, how it was evaluated, etc. (see questions below)

### Questions
- Do you check licenses of content that you use? 
- "The seed instruction dataset comes from annotations by ocean science researchers" - which researchers? what was the annotation procedure?
- It's unclear to me: you use 'gpt-3.5-turbo' to enrich training samples? what about hallucination issues? (in general, all the agents that you use will have this issue, so for me all the data gathered during this step (described in Section 3.2.) is unverifiable and therefore cannot be trusted
- You say that you "leverage GPT-4 as the evaluator for our experimental results" - that's an unacceptable form of evaluation, because GPT-4 is lacking the specific domain knowledge you seek to represent, and because it suffers from the hallucination issues that all LLMs suffer from.
- When you compare models and say that "the result shows that OCEANGPT demonstrates a higher level of knowledge expertise when describing the content of radioactive nuclide research.", how do you measure this?
- You state that "the experimental result suggests that OCEANGPT has the potential to acquire embodied intelligence." - without providing any proof of formal evaluation of the model's ability to write instruction code for underwater robots. You need to develop a procedure for evaluating this before you can make such statements.

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper builds a large language model (LLM) for ocean science tasks, namely OceanGPT, which is the first attempt of concerning LLM with ocean science. OceanGPT is firstly pre-trained on the collected open-access literature of ocean corpus based on the LLaMA-2 model,  then fine-tuned on the instruction data generated by the proposed DoInstruct framework based on multi-agent collaboration and five specific ocean topics, and lastly evaluated on the constructed ocean-specific benchmark OceanBench.

### Strengths
- It is the first attempt of building a large language model (LLM) for ocean science, which is helpful for the related study.
- The proposed DoInstruct seems flexible for the LLM model on other fields to generate instruction data.

### Weaknesses
Major concerns:
- Although the authors made a great effort on building the OceanGPT, the key contributions seem not strong. Most of the operations on building OceanGPT are general for current LLM models. The authors claim that the DoInstruct is novel, but actually the multi-agent collaboration has already been concerned in the field of LLM, for example, the papers collected in the URL of https://github.com/AGI-Edgerunners/LLM-Agents-Papers, and the authors didn't discuss the essential contributions different from them.
- The Appendix seems not finished without any content (text) from A.2 to A.6. 

Minor concerns: 
- The authors are suggested to pay attention to the writing like typos and grammar errors, for example, the sentence in Page 5: "We use the retrieved texts are used as high-quality candidate samples".

### Questions
- What are the contributions that can consider that this work is meaningful to the LLM community?

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
This study introduces a large language model for ocean science tasks to explore the potential of LLMs for ocean science.

### Strengths
This study introduces a large language model for ocean science tasks to explore the potential of LLMs for ocean science. It is a great research topic and beneficial to experts in the ocean science domain.

### Weaknesses
While the proposed OceanGPT shows great potential for ocean science tasks, certain details appear to be absent from the manuscript, and valuable information seems to be dispersed throughout the text.

### Questions
1.	Additional related work should be included to provide a comprehensive background  and necessity of the proposed work
2.	I recommend including "DoINSTRUCT" in Figure 2.
3.	In Section 3.1, the authors mention that they collected a raw corpus of 67,633 documents. While the source journals are listed in the Appendix, detailed information is missing, such as the criteria for selecting these journals, the specific volumes chosen, and the types of articles included.
4.	In Section 3.2, the authors state that over 10,000 data entries across 500 sub-categories were provided by ocean science researchers. However, they do not explain how these annotations were collected.
5.	In Section 3.2, the introduction to the fine-tuned agent is unclear. What does it mean to "automatically generate questions from the unsupervised ocean science corpus"?
6.	In the title of Figure 4, the authors mention that 15,000 instructions were generated from data seeds. If I understand correctly, they collected 10,000 data seeds. Does this mean that DoINSTRUCT generated an additional 15,000 instructions from the seeds?
7.	In Section 3.2, please provide more details about the quality control steps, such as the number of samples evaluated.
8.	In Section 4, what is meant by "For each testing question, given two responses from two different LLMs"?
9.	In Section 5, could you explain how the win rate is calculated?
10.	The content in the appendix should be reviewed for accuracy and completeness, as there are a few issues:
(1)	Sections A.3, A.4, and A.5 are empty.
(2)	 Some content is missing or cannot be located. For example, in the title of Figure 6, the authors refer to Table 10 in the Appendix, but I was unable to find Table 10.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
