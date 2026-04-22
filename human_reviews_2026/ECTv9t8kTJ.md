# Mix-Ecom: Towards Mixed-Type E-Commerce Dialogues with Complex Domain Rules

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8

## Abstract
E-commerce agents contribute greatly to helping users complete their e-commerce needs. To promote further research and application of e-commerce agents, benchmarking frameworks are introduced for evaluating LLM agents in the e-commerce domain.
Despite the progress, current benchmarks lack evaluating agents' capability to handle mixed-type e-commerce dialogue and complex domain rules. To address the issue, this work first introduces a novel corpus, termed Mix-ECom,
which is constructed based on real-world customer-service dialogues with post-processing to remove user privacy and add CoT process.
Specifically, Mix-ECom contains 4,799 samples with multiply dialogue types in each e-commerce dialogue, covering four dialogue types (QA, recommendation, task-oriented dialogue, and chit-chat),
three e-commerce task types (pre-sales, logistics, after-sales),  and 82 e-commerce rules.
Furthermore, this work build baselines on Mix-Ecom and propose a dynamic framework to further improve the performance.
Results show that current e-commerce agents lack sufficient capabilities to handle e-commerce dialogues, due to the hallucination cased by complex domain rules. The dataset will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Mix-ECom, a novel dataset and benchmark aimed at evaluating the performance of e-commerce agents in handling real-world, mixed-type e-commerce dialogues. These dialogues involve various types, including task-oriented, recommendation, Q&A, and chit-chat, while adhering to complex domain rules in pre-sales, logistics, and after-sales tasks. The dataset includes 4,799 dialogue samples, covering 82 e-commerce rules, and integrates multi-modal data (e.g., images and videos) to evaluate agents' capabilities. The paper proposes a dynamic framework for improving e-commerce agent performance, addressing the problem of hallucinations arising from complex rules.

### Strengths
The Mix-ECom dataset is designed to address the lack of comprehensive benchmarks for mixed-type e-commerce dialogues with complex rules. The inclusion of multi-modal content is a valuable feature that enhances the practical applicability of the dataset.
The focus on real-world e-commerce dialogues, such as pre-sales, logistics, and after-sales, ensures that the research is directly relevant to industry applications. This has strong potential for real-world impact, especially in enhancing customer service experiences through intelligent agents. The experiment compares some recent e-commerce framework, using the Mix-ECom versus not using it, the exp demonstrated the improvement brought by the Mix-ECom.

### Weaknesses
1.The paper does not provide a clear presentation of the dataset's distribution across different task categories, dialogue types, or domain rules. This information is crucial for understanding the dataset’s structure and its potential biases. The absence of such details makes it difficult to evaluate the dataset's representativeness and fairness.

2.While the dataset and framework are innovative, the paper does not provide sufficient early-stage details on the dataset construction process, usage information, or any plans for ongoing updates. Without such transparency, it is unclear how this work will benefit the community in the near future or whether other researchers can effectively build upon this resource.

3.The paper lacks clarity regarding the data collection process, particularly the sources of the dialogues and whether proper permissions and licenses were obtained. The absence of such details raises concerns about data privacy and legal compliance, which could impact the credibility and applicability of the dataset in real-world scenarios.

### Questions
refers to weakness paragraph

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work mainly tries to fill the gap of handling mixed-type dialogues and complex domain rules in the context of LLM-based e-commerce agent benchmarks. 

To this end, this work has introduced a new dataset Mix-ECom, which is derived from 70,000 real-world customer-service dialogues and finally keeps about 4799 samples.  The highlights of Mix-ECom can be summarized as:

1. It includes four dialogue types (QA, recommendation, task-oriented dialogue, chit-chat).
2. It includes three e-commerce task categories (pre-sales, logistics, after-sales).
3. It includes 82 domain rules.

Then, this work has proposed a framework E-ReAct by tailoring the ReAct work and an E-Plan-and-Slove module extended from Plan-and-Solve.

In the experiments, the authors evaluate 5 LLMs (4 closed-source, 1 open-source) on Mix-ECom.

### Strengths
1. This work has introduced  an novel dataset Mix-ECom,  addressing the gap of mixed-type dialogues and complex rules in existing e-commerce agent evaluations.  The dataset construction has used real user dialogues and proper construction pipelines.

2. Compared to the previous benchmark dataset, Mix-ECom is able to cover more practical scenarios.

3. Mix-ECom has covered  multi-modal inputs (images, videos).

4. Conducts comprehensive experiments with 5 LLMs (4 closed-source, 1 open-source) and fine-tuning, providing clear results and highlighting current agents’ limitations.

5. Compared to the original ReAct and Plan&Solve, the proposed E-Version is much better in the experiments.

### Weaknesses
1. The major contribution of this work is the proposed dataset. However, the novelty and contribution of this dataset are not very significant. 
2. The dataset's development and statistical analysis follow standard practices, lacking notable innovation or depth.
3. E-ReAct and E-Plan&Solve are incremental because both of them are derived from other works without notable modification.
4. Only 0-shot experiments are conducted and only 5 LLMs are tested. It is not sufficient enough for a dataset paper.
5. No enough analysis of how fine-tuning (e.g., on Mix-ECom subsets) impacts performance, which may limit understanding of the benchmark’s value.
6. No comparison with specialized e-commerce dialogue models.
6. No manual evaluation in the experiment part.

### Questions
1. Have you considered testing against specialized e-commerce models?

2. Do you intend to add a user study (e.g., measuring customer satisfaction with hallucination-reduced outputs) .

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Mix-ECom, a dataset made to test chatbots used in e-commerce. It has around 4800 real-life customer chats, taken from real e-commerce cases. These chats cover different types of dialogues like question-answer, product advice, help with tasks, and small talk. It also covers different shopping stages like before buying, delivery, and after-sales. There are also some small rules about returns, shipping, refunds, and so on. Each chat can include pictures or short videos, tools like "calculate shipping cost" or "change delivery address," and info from a logistics database. The authors propose a fix to the issue of rule hallucination, when chatbots ignore or mess up company rules. For this, they make a new system called E-ReAct and E-Plan & Solve, which helps the bot focus only on the right rules at the right time. When they tested big AI models like GPT-4o and Gemini-2.5-Pro, none did great but after fine-tuning the models on Mix-ECom, performance improved.

### Strengths
- The dataset is of high quality: it contains real human chats, is multimodal (as and when needed), covers different aspects of conversation like casual, complaint, requests, etc. The data has rules which helps test chatbots on reasoning rather than just engagement.
- I like the testing design, where the authors separate how corrext the answer are from how well the system handles the internal state. The error breakdown of rule mistake, image misreading, or  early reaction, is clear and helpful.
- The paper is well written with a comprehensive appendix that contains almost everything that is needed to reproduce the study.

### Weaknesses
- Most chats may switch between the chat mix in a sequential manner rather than mixing them up randomly, e.g. after sales to recommendation to QA. This might make the chats too restrictive (more like multi-step rather than mixed) and the pattern too well known to the chatbots.
- I think the authors use the term "hallucination" too loosely. For e.g. they call 63% of mistakes as hallucinations but many of those are just missed rules or retrieval errors, not the model making things up.
- By giving the model full reasoning cot steps during training may help the model too much. Real bots don’t get these ready-made steps, so the test might look easier than it really is.
- Human reviewers checked if responses were natural and informative, but not if they followed company rules. A chatbot can sound polite and smart but still break return or refund policies and the paper should measure this too.

### Questions
- When you say "mixed-type", do you mean many things happening in the same line (like sarcasm + refund + product question together), or just one after another in some predefined sequence? How will you check real mixing, not just switching?
- In your dynamic system, how do you decide which rules matter? Simple keywords, embedding similarity, or a trained scoring model?
- After fine-tuning, are models memorising rule lines instead of reasoning? For example, do they just learn "mangosteen means refund only" without knowing the reason?
- Can Mix-ECom do user modelling? Like changing tone if the user is angry. You already have mood labels, so can you use them to enforce policies in a personalized way?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors focus on the problem domain of conversational assistants for the e-commerce domain. The authors argue that current benchmarks are not comprehensive enough for real-world dialogs and come up with their own dataset called Mix-ECom. The datasets is constructed by taking real-world customer dialogs and using LLMs to remove user privacy, add reasoning between responses and simulate user profiles. 

They then evaluate a series of models on this dataset and leverage the ReAcT and Plan&Solve framework for their models. They adapt those frameworks and have a module that filters out certain turns that may result in hallucination errors and show that their new setup is better. They also show finetuning on their dataset improves an open-source model.

### Strengths
1) The construction of a dataset that mimics real-world conversations is a good contribution to the community.

2) The set of models in the paper along with the experiments done seem comprehensive and the analysis as to why GPT-4o is much worse than Gemini 2.5 pro is an interesting contribution. The dynamic module approach also seems promising and can be explored in future work.

### Weaknesses
There are a couple clarifications needed in the paper

1) Who came up with these domain rules? It seems important but it's not mentioned why these are required.

2) There is alot of work on human verification but there arent specific details on the instructions given to the humans. Additionally what constitutes a high quality conversation? 

3) It is mentioned that the dialog is rewritten. What does that mean? I assumed only the reasoning chain was added but it seems more was added.

4) Why are other task-oriented metrics like number of turns not evaluated.

### Questions
Questions

1) How did you split the dataset into train / test? Was it random or did you take into account actions that may be in the training data but aren't in the test.

2) Why is the use of the dynamic module only when the action talk_to_user outputted? Would be good to explain this.

Suggestions

1) I think in Line 018 you meant to say "multiple"

### Soundness
3

### Presentation
2

### Contribution
3
