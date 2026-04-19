# Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 5, 8

## Abstract
Large language models (LLMs) must often respond to highly ambiguous user requests. In such cases, the LLM's best response may be to ask a clarifying question to elicit more information. Existing LLMs often respond by presupposing a single interpretation of such ambiguous requests, frustrating users who intended a different interpretation. We speculate this is caused by current preference data labeling practice, where LLM responses are evaluated only on their prior contexts. To address this, we assign preference labels by simulating their expected outcomes in future turns. This allows LLMs to learn to ask clarifying questions when it can generate responses that are tailored to each user interpretation in future turns. On open-domain QA datasets with multiple annotations, we evaluate systems based on their ability to ask clarifying questions to recover each user's interpretation and expected answer. We compare systems trained using our proposed preference labeling methods against standard methods, which assign preferences based on only prior context. Our method achieves a 5% improvement in F1 measured against the answer set from different interpretations of each query, showing the value of modeling future conversation turns. We further demonstrate that our method can be used to train models to judiciously determine when to ask clarifying questions, directly answering the question when clarification is unnecessary. In our experiments, we find that our method achives a 3% improvement in accuracy of such judgments over existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper focuses on the problem of having LLMs ask clarification questions when the user request is ambiguous or uncertain. The authors propose a new approach to construct preference data by simulating future conversations. Experiments are conducted in two open-domain QA datasets. The results show that the proposed method outperforms other standard baselines.

### Strengths
1. This work studies an important question in the era of LLMs, i.e., teach LLMs to ask clarification questions
2. Propose a novel approach to construct preference data including clarifications by simulating future conversations
3. Experimental results show that the proposed method outperforms other standard baselines.

### Weaknesses
1. The presentation of the proposed method is a bit hard to follow. There is a lot of symbols and variables in the text, which is very messy and not well defined. More details in the questions that need to be clarified by the authors.
2. The information of annotators for preference data construction is also not clear. 
3. The experimental settings are not extensive and reasonable enough, only comparing on self-designed baselines.  
4. Several recent studies on LLM-based clarifications are not discussed or compared, such as [1-3].

Minors: The usage of \cite{} is incorrect.

[1] STaR-GATE: Teaching Language Models to Ask Clarifying Questions
[2] STYLE: Improving Domain Transferability of Asking Clarification Questions in Large Language Model Powered Conversational Agents
[3] Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training

### Questions
1. What is $a_i$ in line 120? It seems that it has a different meaning from ($a_i$) in line 129.
2. Are the input query $x$ and the user intent ($x_i$) supposed to be different forms?
3. Is the user-simulator model for preference data construction different from the the user simulation model in Section 2.1? 
4. What not apply Clarify-or-Direct-Ans SFT as a baseline?
5. How is the original performance of these LLMs?
6. How about just using in-context learning for enabling the clarification capability?
7. Is there any existing method that can be applied as baselines, instead of all your self-designed baselines?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper examines handling ambiguity, an often overlooked aspect of large language models (LLMs) in multi-round conversations with users. The authors first develop a pipeline to collect human preferences in a two-round setting and then apply Direct Preference Optimization (DPO) to several open-source LLMs to enhance their performance.

### Strengths
- The paper provides a thorough scientific investigation, building upon several interesting hypotheses about LLM responses by training (evaluating) on prior data (Line 17) and speculating about the differences between LLM ambiguity and human ambiguity (Line 305).
- The paper introduces a novel pipeline for eliciting human preferences in multiple rounds of conversation, even though I have some reservations regarding its operation.
-  The experiments are very thorough, considering different variations of LLMs. However, the study lacks comparisons with non-LLM methods.

### Weaknesses
## Weakness 1: Unclear Relationship Between Human Annotators and the User Simulator

The paper appears to ask humans to only grade the final prediction of the model (the final step as shown in Figure 2). It is unclear whether their grading aligns with their judgment of the clarifying questions. It would be more appropriate to set a predefined user goal at the beginning and have human annotators grade the final answer based on its consistency with the original goal and the arrangement of questions and answers in the intermediate steps.

I recommend that the authors refer to the conversational information-seeking literature in the IR community. For example, the paper [1] adopts a user simulator that maintains an initial item of interest (e.g., iPhone 16 128GB). During the question-answering stages, the user simulator responds to questions in line with the item's specifications (e.g., "How much storage do you want?" "128GB"). In this paper, it is unclear whether the human's final grading considers an initial goal and whether the grading assesses both the answer quality and adherence to that goal.

This is my major concern, and I hope the authors can provide further justification during the response period. Thank you!

## Weakness 2: Lack of Baselines Beyond LLM Fine-Tuning

While the paper includes numerous ablation studies and comparisons with other LLM tuning baselines, additional comparisons are needed to justify the method's effectiveness. I recommend the following:

- **Unsupervised Beam Search**: This approach allows LLMs to search the space of possible clarifying questions (implicitly) [2].
- **Chain-of-Thought**: Prompting LLMs to explicitly reason about possible ambiguities.
- **Non-LLM Baseline Methods**: Including methods for asking clarifying questions that do not rely on LLMs, as explored in various works [3, 4].

I hope the authors can try to add these results or explain why these methods are not suitable for comparison. Thanks!

## Weakness 3: Need for More Insightful Analysis

- **Knowledge Taxonomy**: A taxonomy for knowledge types is needed to make the dataset more insightful. What types of knowledge are more desirable for clarifying questions? If the original dataset does not provide this, consider latent word clustering or LDA?
- **Error Accumulation**: Correct me if I am wrong, but I do not see an analysis of how an incorrect clarifying question may lead to error accumulation in the final answer.

I hope the authors can discuss with me if these research questions are helpful and whether they plan to append them. 

# References
1. [Conversational Recommender System](https://arxiv.org/abs/1806.03277) SIGIR 2018  
2. [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/abs/2305.00633) NeurIPS 2023  
3. [Asking Clarifying Questions in Open-Domain Information-Seeking Conversations](https://arxiv.org/abs/1907.06554) SIGIR 2019  
4. [Asking Clarification Questions to Handle Ambiguity in Open-Domain QA](https://aclanthology.org/2023.findings-emnlp.772/) EMNLP 2023

### Questions
The presentation of the paper has significant room for improvement. I have identified several typos and areas for enhancement:

- **Line 16**: Can you specify whether it is *evaluated*? I think it should be *trained*, (or both)?
- **Line 212**: "using a we use a trained user-simulator model" – this sentence does not parse.
- **Lines 224–226**: The green box is confusing. Why are q1 and q2 being preferred simultaneously?
- **Line 186**: “by providing their answer the proposed clarifying question” – this sentence does not parse.
- **Line 206**: "in this work we use simulate user interactions for annotation." – this sentence does not parse.

**Generic Questions:**

- Is your human annotation IRB-approved?
- In the methods section, can you write the loss function for DPO? This would make your paper more self-contained and eliminate the need for readers to refer to the original DPO paper.
- Can you provide more details on the evidence supporting the hypothesis that model responses are evaluated only on their prior contexts (Line 17)? This is a very interesting hypothesis (perhaps the central hypothesis of the paper), so additional justification would be beneficial.
- Line 305 presents an interesting hypothesis: "We hypothesize that what is ambiguous for an LLM often deviates from what is ambiguous for humans." Has this hypothesis been proven?
- Can you specify the input format for the LLaMA model? If my understanding is correct, only the Chat version includes the system and user specifications.
- What is the computation time and the number of training samples used?


Building on all my comments, I'd vote for a 5/10 score now :)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of asking clarifying questions for ambiguous user queries. To determine the optimal timing and content of a clarifying question, one of the core contributions of this paper is a dataset of multi-turn QA dialogues that include such questions. The paper also presents a method for evaluating the quality of clarifying questions. In this method, multiple candidate clarifying questions are tested using a user simulator, with the quality of each question assessed based on the aggregate reachability to correct answers. Models trained on this dataset in various settings demonstrate improved QA accuracy on both ambiguous and unambiguous queries.

### Strengths
- Asking clarifying questions for ambiguous queries is a valuable trait for QA systems, and the paper’s proposed methods are well-motivated.
- The approach for assessing the quality of clarifying questions based on reachability to correct answers is novel, interesting, and well-reasoned.

### Weaknesses
- The primary weakness of this paper, in my opinion, is the lack of clarity in the method sections (2–4). One contributing factor is the use of terms without clear definitions. For example, in line 196, the term “annotators” is used, which seems to imply human annotators, while the preceding text suggests a “simulator” is involved. Additionally, the description of user simulation from line 208 onwards does not clearly explain the origins of *q* and *a_i*, making it challenging to follow. Although some of these ambiguities are clarified later in the paper, I would recommend reorganizing the structure to reduce the distance between concepts and their realizations. Further questions and suggestions are provided in the Questions section below.
- The paper contains numerous ungrammatical and broken sentences, which significantly impact readability. For instance, line 187 ("providing their answer the proposed"), line 191 ("catered a toward"), line 206 ("we use simulate user interactions"), line 212 ("using a we use"), line 300 ("examples SFT training"), line 307 ("to for queries"), and line 312 ("we a sample"), among others. Additionally, citations are incorrectly formatted on lines 087 and 097. I will raise my score if the authors thoroughly address the grammatical and formatting issues and their revisions in their rebuttal.
- Some experimental settings and design choices lack sufficient motivation. For example, why is asking clarifying questions preferable to simply providing multiple answers in one response (e.g., Q: "Where were the Olympic Games held in Greece?" → A: "The ancient Olympic Games were held in Olympia, and the modern Olympic Games were held in Athens.")? Moreover, there is no discussion on efficiency or how humans might perceive these clarifying questions, and a human evaluation might be valuable.
- There is no experiment involving training a model directly on multi-turn QA dialogues. Instead, the authors only experiment with combining the weights of two separate models. Since a multi-turn dataset is available, training and evaluating a model on this dataset should be feasible.
- In section 5.1, the comparison with the random baseline is questionable. A more effective comparison would evaluate whether the model asks clarifying questions when a query is ambiguous and refrains when unnecessary (e.g., based on the ambiguous and unambiguous splits of the dataset or cases where annotators provided consistent answers).

### Questions
- Line 164: Please clarify “We prompt GPT-4 to abstain from providing a clarifying answer a_i if it judges that none exists, in which case we count the resulting target answer prediction r_next^i as incorrect.”
- Line 245: What is “EM accuracy”?
- Lines 299: What does “if it determines one exists” mean and when can this happen?
- Line 312: Please clarify “generate its greedy answer with temperature T = 0.0 and sample an answer with T = 1.0”.
- Line 324: Which of the models below is used for the Turn 4 model?
- Lines 330—359: The purpose of each model and how they are used in the two-step pipeline is unclear.
- Line 342: What is the purpose of the Clarify DPO models?
- Line 347: More details about the reward model and the RL training are needed.
- Line 432: Why not training a model on the multi-turn dataset?
- Line 459: Why is the random baseline appropriate for comparison?
- Line 461: How did you set the value of DA?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes an approach which constructs preference data which are determined by examining (up to 2) simulated future dialogue turns. The proposed approach is evaluated in an open-domain QA dataset, NaturalQuestions, and a related derivative dataset, AmbigQA. The authors compare their approach against a standard single-turn reward model and present their results in terms of Token-based F1 for the QA task. The authors also present a set of experiments on ambiguity recognition.

### Strengths
***Significance of setup***

The proposed setting is important to investigate. As LLMs are increasingly tuned for use in dialogue applications it is important to consider how to better address the needs of users, and disambiguation is certainly a reasonable idea.

***Ablations and analysis***

The paper presents experiments comparing the proposed approach on multiple base LLMs, and performs multiple reasonable ablations. The paper also considers and interesting examination on the effect of obtaining human annotations versus model predictions. 

***Clarity***

Overall the paper is clear and the experimental details are helpful.

### Weaknesses
***Concerns regarding generalizability***

The focus and main contribution of the paper is that it considers a lookahead approach for future conversation turns in order to optimizing preferences in question-answering settings. However, the approach is only defined for at most two future conversation turns. It is not clear how 1. this may be generalized to longer rollouts and 2. whether this approach would remain effective in such a setting.

***Scope of experiments***

Relatedly, the domains examined are limited. The approach is validated on NaturalQA and AmbigQA. AmbigQA is an extension of NaturalQA, and as a result it is not clear nor trivial how the proposed approach would generalize to any other more complex and realistic domains (e.g., those requiring complex rollouts, or task-oriented settings).

***Limited baselines***

The baselines in the main experiments are somewhat limited. If I understand correctly, Direct Answer SFT and Clarify SFT both do not seem like fair baselines in a task where there is mixed ambiguity. How does the proposed approach compare against SFT on both types of responses? Moreover, the proposed approach focuses on using RL tuning combined with future turns. How does it compare against directly including those future turn trajectories? It would also be helpful to compare against other existing RL-based approaches for ambiguity detection/action planning (e.g. [1]).

There are also no baselines for the ambiguity recognition experiments. Does the proposed approach outperform prompting? The objective performance of the proposed model here seems rather low. If the model asks questions unnecessarily, it may lead to degraded user experience even if it successfully elicits additional information to solve the downstream task / provides additional context to condition on for answer generation.

*Minor Points*

typos:
L187 "their answer the proposed"
L300 "construct all examples SFT training"
L306 "to for queries"

References

[1] Plug-and-Play Policy Planner for Large Language Model Powered Dialogue Agents, ICLR 2024

### Questions
1. How do you consider scenarios in which clarifying questions are undesirable behavior? Relatedly, how do you propose to balance clarifying questions with behaviors such as hedging, or presenting information for all possible intents?

2. Does the motivating problem of LLMs not asking clarifying questions get solved by prefixing with a simple instruction such as "ask a clarifying question if ambiguous"? For instance, I tried this with the example from Figure 1 and Gemini asks a clarifying question. To that end, such systems may be tailored to one type of behavior over another by design?

3. Have you considered multiple types of ambiguity (e.g. contextual versus lexical)?

4. Is the proposed approach intended to improve general LLM usage, or for this one particular task? if the latter, is this task with basic response outputs (e.g. single token) representative of real-world usage?

5. L161 is the model specifically fine-tuned to output "Clarifying Question:" (i.e. all generated responses should be accompanied by an intermediate dialogue act token)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a method for training LLMs in question-answering tasks to learn to ask clarification questions where appropriate given query ambiguity.

### Strengths
The topic addressed -- clarification behaviour by LLMs, and how to make their behaviour more accurate & reliable by clarifying where required -- is very topical and of great interest to the NLP community.

The paper proposes a fairly simple and intuitive method, describes it clearly, and evaluates it sensibly (mostly -- see below), and results seem good in that they show reliably increased accuracy.

### Weaknesses
The evaluations and baselines compared to could be stronger or more meaningful in some cases. In the main experiment, one of the main takeaways from the results seems to be that answer accuracy is increased if a CQ is always asked, although this comes at the cost of more turns. This raises the question of how well a system would do if given a very simple fixed prompt which forced a CQ as the first response, and then an answer based on the resulting three-turn context - but without going through the extra training steps given here. I expect it wouldn't be quite as good as the best system here, but it would be very instructive to know how much difference there was.

In the experiment of sec 5.1 (determining whether CQs are necessary), performance is compared to a random baseline (sampling a random fixed percentage of direct answers vs CQs), but there's no direct evaluation of the main question here: how often a CQ was asked after an ambiguous question (i.e. accuracy of identifying when a CQ is required). In fact, the question is perhaps better: how often was a CQ asked after a question which was ambiguous, and where the possible answers differ (if the answers are the same, a CQ provides no benefit, even if the question is nominally ambiguous).

The data on possible CQs and answers, including answer correctness/preference, is mostly drawn from LLM-based simulations, but there doesn't seem to be any validation of how accurate or realistic this is.

There's no discussion of how ambiguity in a question actually relates to differences in answers. Ambiguity sometimes doesn't matter in practice: e.g. a variant of the first question in Table 8 (appendix) could be "what country were the first olympic games held in"; then, although there is ambiguity about whether the query concerns the modern or ancient games, that doesn't affect the correct answer which would be Greece in both cases. It would be interesting to know how common this is, and whether the behaviour of the models reflects (or should reflect) this difference.

The biggest weakness I perceived is the lack of relation to other relevant work. There's discussion of related work with LLMs and ambiguity, and LLMs being trained to ask CQs in different contexts; but there's no discussion of pre-LLM work that seems relevant. For instance, there is a large body of work in training dialogue systems to respond suitably to ambiguous input (including asking clarification questions), much via reinforcement learning. Closer to the domain here, if usually less close in terms of method, is the body of work in interactive information retrieval (including using clarifying questions to refine or further specify search queries or QA questions). Some discussion of how the method here resembles (or doesn't resemble) those would be very helpful.

### Questions
Would it be helpful & possible to include some more informative comparisons and evaluation metrics (see "Weaknesses" above)?

On annotation reliability: I wasn't clear whether the additional annotation described for AmbigQA was done as part of this work, or previously by other authors and described elsewhere ("his process identifies about 56% of all queries in NQ-Open as ambiguous" etc.).

Table 2 doesn't say what metric it uses - presumably F1?

Some minor typos:
083 two axis->axes
097 citation should be citep not citet
110 of [the] task scenario
187 an additional turn[s]
406 Cla[i]rifying
188 observ[ing] the LLM's 
206 simulate[d] user

### Soundness
3

### Presentation
3

### Contribution
3
