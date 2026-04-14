# Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs

- Decision: Accept (Oral)
- Scores: 8, 8, 8, 6

## Abstract
Large Language Models (LLMs) are increasingly deployed as chatbots, yet their ability to personalize responses to user preferences remains limited. We introduce PrefEval, a benchmark for evaluating LLMs' ability to infer, memorize and adhere to user preferences in long-context conversational setting.
PrefEval comprises 3,000 manually curated user preference and query pairs spanning 20 topics. PrefEval contains user personalization or preference information in both explicit and implicit preference forms, and evaluates LLM performance using a generation and a classification task. With PrefEval, we have evaluated 10 open-sourced and
proprietary LLMs in multi-session conversations with varying context lengths up to 100k tokens. We benchmark with various prompting, iterative feedback, and retrieval-augmented generation methods. 
Our benchmarking effort reveals that state-of-the-art LLMs face significant challenges in following users' preference during conversations. In particular,  in zero-shot settings, preference following accuracy falls below 10\% at merely 10 turns (~3k tokens) across most evaluated models. Even with advanced prompting and retrieval methods, preference following still deteriorates in long-context conversations. Furthermore, we show that fine-tuning on PrefEval significantly improves performance. We believe PrefEval serves as a valuable resource for measuring, understanding, and enhancing LLMs' proactive preference following abilities, paving the way for personalized conversational agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a challenging and important problem of LLMs: memorizing the user preference in a long-context, multi-turn dialogue (could be up to 100k tokens). To achieve this, the authors create a PrefEval dataset containing 3000 examples (i.e., preference-query pairs), distributed across 3 preference forms and 20 different topics related to our daily conversations with chatbots. 

Specifically, in their dataset, the stated user preference could be in three different forms (blue box in Figure 1; see also Table 14 to 16 in Appendix):

* one turn (also only one sentence in the user utterance, e.g., I don't like seafood)

* two turns (e.g., a user first asks a question, then after an LLM provides four options, the user chooses one of them and rejects the others)

* $>$ 2 turns (i.e., they insert the user preference within an unrelated 4~8 turn dialogue; for instance, a user "accidentally let slip" and mentions he/she doesn't like seafood in a library-related dialogue)

To make the conversation turn extremely long, they leverage the LMSYSChat-1M dataset by randomly selecting the dialogue up to 100k tokens, and then they insert this long context between the user preference (typically in the first turn) and the query question (at the end of the dialogue).

The authors benchmark the PrefEval dataset using a variety of LLMs (GPT, Claude, Mistral, Llama-3, and Gemini), as well as the latest GPT-o1 system. Concretely, they conduct different experimental settings (see also Appendix A.4): 
1. (pure) zero-shot
2. zero-shot + inject a sequence (like “provide an answer that follows my preference”) after the test question (termed "reminder" in this paper)
3. let LLM evaluate its own response and generate the answer again, similar to the self-correct task or simple CoT baseline (termed "self-critic" in this paper)
4. give few-shot examples to the LLM before the LLM response (termed "$CoT$"* in this paper)
5. use RAG method to retrieve relevant sentences in the previous dialogue then provide these sentences before the LLM response. 

Additionally, they finetune the Mistral 7B LLM using their dataset with the reminder method and observe that SFT improves in alignment with user preferences. This simple SFT approach could be more robust in long-term dialogue if the training data has more unrelated turns inserted between the user preference and the query dialogue (which is reasonable).

(*) though this could be a bit confusing, see weakness 1 below

### Strengths
The strength is mostly summarized in the Summary.

Overall, this paper is well-motivated by the current capabilities of LLMs to follow the user preference in long-term conversation, which is also an emerging issue of contemporary ("general-purpose") LLMs.
To take a step further, they construct this dataset and gauge whether LLMs could implicitly retrieve the user preference and follow it in the long-context, long-term setting. 

While the originality of following user preference may not be really novel $-$ since memorizing the user preference has been studied in other NLP tasks such as task-oriented dialogue (TOD) $-$ there is still a lack of comprehensive experiment in this extremely long-term scenario, not to mention various LLMs are conducted in this paper. Multiple baselines are performed when LLMs are frozen, including the pure zero-shot, zero-shot with a "follow my preference" string, few-shot, self-correction/CoT (termed "self-critic"), and RAG. 
Different question (i.e., query) types are also tested ("generation" task and (MCQ) "classification" task).
This is perhaps one of the most comprehensive evaluations I've read in recent CoT-prompting papers.

The authors also plan to open-source the dataset and release the prompts used in the Appendix.

### Weaknesses
The major weaknesses are that (1) some terms may confuse the readers at first glance and (2) the dataset construction is not very clear in this paper (as it is in the datasets and benchmarks track) $-$ even though the benchmark experiments are exceedingly extensive. 

## Weakness 1

(i) In this paper, the "$CoT$"* method is defined as having several (i.e., 5) examples before an LLM response (visualized in Appendix A.4, page 17). However, providing examples to the LLM is generally considered to be few-shot learning (or, previously in-context or demonstration learning). In particular, the term CoT should be more related to the "self-critic" method in this paper (which can also be viewed as a simple self-correct task [1]). 

(ii) The evaluation metric of the "generation task" is the accuracy (see page 2, Figure 1's caption) in this paper. If I understand correctly, the generation task in NLP typically refers to translation, summarization, story completion, etc. (hence, accuracy is generally *not* the primary evaluation metric, which could be BLEU/ROUGE/METEOR/BERTscore/etc.). While the authors state that "the LLM generates a response of approximately 300 words in reply to the user's query." on page 4 and provide a real-life scenario (asking for a travel recommendation in New Orleans and see if an LLM knows the user doesn't like jazz) in the Introduction, the evaluation is still mainly the accuracy by leveraging LLM-as-a-judge, which could be done $-$ and is also simpler and cheaper than LLM-as-a-judge $-$ using the answer extraction in [2]. To be more specific, we can prompt the LLM by asking "Does the response contain any [seafood] cuisine?" in Figure 1, provided that a user stated that he/she doesn't like [seafood], then extract the Yes/No response afterward.). 

Moreover, perhaps using LLM-as-a-judge is costly and complex, the authors also conducted multiple-choice question (MCQ) experiments and included the relation between the classification task and generation task in Appendix A.5, which demonstrates that there is a high correlation in accuracy. In this vein, while LLM-as-a-judge also reports four error types introduced in this work (see Table 1 and Figure 7), there isn't a significant difference between the "generation" task and the "classification" task in this paper to my knowledge (though those four error types should be useful for error analysis in this sense). As a result, after the readers look at Figures 5 and 6, they may not immediately understand why the authors evaluate two "different" tasks (though I believe they *should be* the same, see explanation below).

Take Figure 1 as an example. The user first asks "What Japanese cuisine dishes should I try on my visit?", then this query (from my understanding) is for the generation task (because LLMs tend to reply with lengthy sentences for general (i.e., non-specific) questions), which is evaluated by LLM-as-a-judge (Claude 3 Sonnet). On the other hand, if, for example, the query becomes "What Japanese cuisine dishes should I try on my visit? (A) beef and mushroom noodles (B) vegetable stir fry (C) seafood with shrimp and crab (D) grilled chicken Caesar salad", then perhaps this is for the classification task? (I cannot find the query related to MCQ so I give an illustrative example above.)

## Weakness 2

(i) Though the three forms of user preference (blue box) and the multi-session conversations (green box) are clearly stated in Figure 1, it is unknown what the **source** of the user preference is, as well as the construction between three different preference forms. After reading the paper, I only found "These pairs were manually curated with the assistance of GPT-4, Claude 3 Sonnet, and Claude 3.5 Sonnet." in Section 2.2.  In Tables 14 to 16, the authors only explain "Every implicit preference dialogue is derived from an explicit preference" and "Every dialogue is derived from an explicit preference and is randomly assigned a persona to simulate a longer conversation."

Specifically, how to first generate the explicit user preference and get the distribution in Figure 2? Are the preferences mined from existing dataset(s) or generated by LLMs? After this, how is the corresponding query generated? Next, what are the criteria for generating the implicit choice-based/persona-driven dialogue? I found in Table 15 that the templates are rather fixed. For example, the chatbot only provides four choices in the first turn, and the user as well as the chatbot response are highly similar across topics. Are there any rules for prompting LLMs or are they human-generated templates (with some text combinations)?

(ii) Another issue might be the problem formulation in Section 2.1, as it is not formalized in math. Although I have no trouble understanding the PrefEval dataset and the proposed framework/experiments in general, other not-too-familiar readers may need to re-read several times to grasp the idea behind this paper. For example, the formulation of the conversation flow in Figure 1. Here, we only know user preference $p$ is in the blue box, and the query $q$ is in the red box. How to generalize it in a $m$-turn conversation (i.e., {$u_1, b_1, u_2, b_2, ..., u_m, b_m$})? Alternatively, as each box is separated, can the PrefEval dataset construction be formalized in a mathematical way?


[1] Large Language Models Cannot Self-Correct Reasoning Yet, Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou, ICLR 2024

[2] Large Language Models are Zero-Shot Reasoners, Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa, NIPS 2022

Regarding the format, the line number is missing in its current PDF format, so the presentation is somewhat unclear, making it difficult to address the concerns below (not sure if it is because the version is the previous one).

### Questions
Despite the above major weaknesses, **these two weaknesses could be easily fixed** to improve this paper's clarity as follows (first migrate this paper to the 2025 template):

### Suggestion for Weakness 1-(i) 

Perhaps (a) modify the "$CoT$"* to "few-shot", and (b) either include some (intrinsic) self-correction papers that the authors found related to your "self-critic" or simply state self-critic is also the CoT baseline (with CoT citations). 

### Suggestion for Weakness 1-(ii) 

I personally would prefer to report accuracy in MCQ experiments, whereas LLM-as-a-judge is for error-type analysis only. However, If replacing the generation task performance with the classification task requires substantial revisions, then it's best to explicitly state something like "the accuracy in the generation task **is highly correlated** to the classification task" either in Figure 1 (on page 2) or in Section 2.5 (on page 4). 
Either way should be better than first mentioning it on page 7 (in Section 3.4): "This correlation suggests that despite task differences, ... Classification tasks could thus serve as an efficient proxy for evaluating preference following in complex generation scenarios across models and methods." Moreover, this may not be generalized to other complex generation tasks such as machine translation.

### Suggestion for Weakness 2 

Detail the dataset construction in the Appendix, especially the process of how the user preference, its related query, as well as the construction of implicit preference form (Tables 15 and 16). Is the PrefEval dataset constructed in a more standardized way that could be formalized for future research to follow this criterion? If not, then it is also appreciated to include the flowchart or the pipeline process of how the explicit and two implicit forms are generated, perhaps with a walkthrough example, in the Appendix.


I could be convinced if the authors can address the following questions below. No need to conduct additional experiments, just clarify the rationale behind this paper (e.g., the figure shown, experimental setup).


# Questions

1. See Weakness 1. If I am very much mistaken, perhaps the authors could provide some papers that use this term in this scenario?

2. See Weakness 2. 

3. In Figure 1
* Do authors conduct experiments where the user preference (blue box) is in between the multi-session conversations (green box)? Specifically, as the multi-session conversations (green box) are from LMSYS-Chat-1M dataset in Section 2.4, I wonder if the top one is also filled in the experiments (just curious if I am missing something in this paper). 
* In explicit preference form, as the conversation turn is defined as a pair of user and chatbot utterances in Section 2.1, what is the corresponding "chatbot" utterance in user preference when a user states "I dislike eating seafood due to an allergy" in the first turn (see Figure 1)? Is it an empty string or some pre-defined sentence (because I cannot find it in Table 14)? If so, is it used across all explicit data?
* In the implicit choice-based preference form, does the first user question the same as the query? Specifically, as shown in Table 15, suppose a user asks "What are some good hotel options for my upcoming trip to Paris?" in the first turn, then does the user ask the same question in the last turn (which is the query)?
* When evaluating results, does the green box (multi-session conversations) also present to the evaluator? If so, then the figure is missing, which may cause some confusion as the evaluator would become impractical in real-world scenarios (as we normally don't have labeled data). If not, then could the authors explain why it's excluded in reporting accuracy (I think it is reasonable to exclude this only for error analysis, as shown in Appendix A.11)?

4. In the Introduction, does "prompting" in the second contribution mean "zero-shot, reminder, and $CoT$*" in Section 3.1? Similarly, does "iterative feedback" mean "self-critic"? As for RAG, is the retrieved context "utterance-based" or "sentence-based" (e.g., if a user utterance has 3 sentences, only one of which is related to the query)?

5. In Section 2.5, does LLM-as-a-judge similar to the work in [1]? Are the four error types used in the previous studies, or did the authors pre-define them? In Table 1, why the value of (Preference Hallucination Violation, Hallucinate Preference?) is "No"?

6. In Section 3.1, as few-shot examples are given in the "$CoT$"* method, did authors previously also experiment with placing those examples on top (as many papers first provide LLMs with examples and then ask them to perform some tasks)?

7. In Table 2, Is the number of data conducted in this restaurant 3000 x 5.6% (168) or 1000 x 5.6% (56) in 10 Turns and 300 Turns, respectively? I am also curious how this travel-restaurant topic is reported but not the others. Is it because of the cost, or does GPT-o1 perform better in this topic?

8. In Section 3.3, why does Claude struggle more with implicit persona-driven dialogue compared to the choice-based one? In Figure 5, I found that Claude 3 Sonnet is generally on par with or better than in the reminder scenario (it is worse if the token length is around 16k~23k, but for longer context, there isn't much difference).

9. In Section 3.4, what is the "structure task"?

10. In Section 3.6

* Could the authors provide an example in "Impact of Multiple Preferences Stated in Conversation"? Does it mean that if a user says "I don't like seafood" (restaurant) in the first turn, then he/she also states "I don't like wearing jeans" (fashion), "I like to read" (music & books) in the following turns?

* Similarly, could the authors provide an example in "Effect of Conflicting Preferences Stated in Conversation"?

* What is the number of data tested in Figures 8 and 9, respectively?

11. In my opinion, even though the $(p, q)$ pair is indeed different in this sense, I would expect 3,000 to be the "totally" different dataset (if only glancing at the Abstract). Maybe it is better to rephrase something like 1,000 data, each with three "levels of difficulty" (user preference forms).

12. Is the temperature set to 0 for all LLMs?

13. Do the authors include the Ethics Statement?

[1] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica, NIPS 2023

# Typos and Format Suggestions

1. In Figure 1 "shnmp" $\rightarrow$ "shrimp"
2. The em dash ($-$) in Section 3.7 should be consistent with the previous ones.
3. In Figure 10, "W/" $\rightarrow$ "w/"
4. Could try to fit GPT-4o in Section 3.1 since there are only nine models mentioned in this section.
5. Should use wrapfigure in Figures 3, 7, and 10 so that the caption is below the figure.
6. From my understanding, when citing the papers in Appendix A.2, \citet{} should not be used even if they are in the middle of the sentence (consistently use \citep{}).
7. In Table 14~16, if Table 14 shows 10 examples while Table 15 could only display 6 within a page, perhaps it would be more visually comparable to select the first 6 examples from Table 14 (in the same order).
8. For color consistency in Figure 10, the author could draw the 5 w/o SFT methods first so it will match the color in Figure 4. The authors could also consider drawing two legends in this figure (for example, a legend displays the marker: star denotes w/ SFT and circle denotes w/o SFT).

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces PREFEVAL, a benchmark for evaluating large language models' (LLMs) ability to infer, remember, and follow user preferences in multi-turn conversations. The main contributions are: 

(1) Introduction of a comprehensive benchmark assessing LLMs' ability to follow user preferences in conversational settings, comprising 3,000 manually curated question-preference pairs across 20 topics and 3 preference forms. 

(2) Through extensive evaluation of 10 state-of-the-art LLMs, the paper reveals current challenges in actively following user preferences during conversations and presents important findings about personalized preference adherence.

### Strengths
The paper's strengths are as follows: 

(1) **In terms of originality**, it proposes a comprehensive benchmark for evaluating LLMs' personalized preference adherence capabilities, which is a relatively novel research direction.

(2) **Regarding quality and writing**, the authors thoroughly validated LLMs' capabilities through extensive evaluation of multiple models' preference adherence across various conversation topics. The paper structure is clear, with well-presented experimental design and results.

(3) **In terms of significance**, the findings reveal key challenges in current LLMs' personalized preference adherence. In daily use and experiments, we also observe that LLMs' adherence capability diminishes with increased conversation length. Broadly speaking, ensuring LLMs consistently follow user characteristics or instructions is crucial for their real-world applications.

### Weaknesses
The paper's weaknesses are as follows: 

(1) **The data construction process is not detailed**. A major contribution of this paper is the manually constructed personalized preference dataset. The construction process directly affects data quality and thus impacts model results and experimental reliability. However, I did not find a section detailing the dataset construction process.

(2) **Additional experiments could enhance the comprehensiveness of findings**. The authors verified LLMs' adherence to explicit preferences through fine-tuning. However, implicit preferences are often more challenging to identify and follow. Experiments in this area are recommended for a more comprehensive understanding of LLMs' preference recognition and adherence capabilities.

(3) **Line numbers are missing**. The paper lacks line numbers, making it difficult to precisely reference specific content locations.

### Questions
(1) **The detailed dataset construction process is not introduced**. Multiple methods should be employed to verify the dataset's high quality, which would make subsequent experiments more convincing. I recommend adding a dedicated section that outlines the data preparation process and quality control measures. This should include details on selection criteria, query development, and validation procedures used to ensure data quality.

(2) **How were the reasoning steps for CoT obtained?** Since reasoning steps are crucial for CoT effectiveness, the authors did not discuss this detail. I suggest the authors provide examples of CoT prompts, or describe their process for generating these prompts.

(3) **When multiple preferences are injected, why does the adherence to original preferences improve?** Intuitively, when multiple preferences are introduced, LLMs would struggle to accurately identify user preferences. This should make it harder for LLMs to recognize the original preferences. However, the paper's results show the opposite.(Section 3.6, Page 8). I recommend the authors explore potential explanations for this phenomenon, or conduct additional experiments to verify if this effect is consistent across different models.

(4) **When conflicting preferences are introduced, model performance improves compared to single preferences, which is counter-intuitive**. Why does this phenomenon occur, and do other LLMs besides Claude 3.5 Sonnet exhibit similar behavior? (Section 3.6, Page 9). It is recommended that the authors expand their analysis to include more models beyond Claude 3.5 Sonnet. Additionally, they should provide potential explanations for these counter-intuitive results and discuss their implications for understanding LLM behavior in complex preference scenarios.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce PREFEVAL, a benchmark for evaluating if LLMs follow personal preferences in conversational settings. PREFEVAL assesses how well LLMs can infer, remember, and adhere to personalized preferences across multi-session conversations. It includes 3,000 preference-query pairs across 20 topics, with preferences presented explicitly or implicitly. Ten LLMs were evaluated using prompting, iterative feedback, and RAG. LLMs struggle to maintain preference adherence over longer dialogues, especially without _reminders_, with accuracy dropping below 10% for 10-turn interactions. Finally, advanced prompting and fine-tuning improved model performance, particularly in handling long contexts and multiple or conflicting preferences.

### Strengths
I think this paper covers a timely and interesting topic. The paper and dataset yields several strengths:

1. The error analysis is quite interesting! The authors look at violations of preference applications that are particularly aggravating when interacting with instruction following LLMs

2. The interventions are well-motivated, simple, and moderately effective (though there is still space for improvement).

3. The authors validate their prompts with human annotators.

4. The paper is well-written and generally easy to follow.

### Weaknesses
I have three concerns. If the first two are resolved, this would improve the paper!

First, I'd like to see some analysis of the "Lost in the Middle" effect (see: https://arxiv.org/abs/2307.03172). I suspect that some of the failures can be attributed to how models use long context. If we include the reminder at the start instead of the end, or in both places---does that affect performance? I'd also encourage the authors to draw distinctions between needle-in-a-haystack benchmarks for long-context LLMs and this dataset. One reason might be that the implicit / persona settings in the dataset require one to _reason_ over preferences, not just retrieve a needle---but I would make this comparison more explicit in the main text.

Second, the long-context integration feels a bit heavy-handed. The authors seem to randomly sample conversations and randomly intersperse (p, q) pairs within this context. This seems somewhat artificial. Were there any measures taken to make sure that the bridge between the pre-existing context and the (p, q) pairs were natural?

Finally (and I don't expect or require the authors to directly fix this), the (p, q) pairs are entirely synthetically generated. This is an unfortunate limitation: a personalization dataset generated with no "real" people. One could argue that generations from LLMs "simulate" some aspects of human behavior. Still, I'd recommend mentioning this in the limitations section.

### Questions
A particularly irritating effect of LLM "personalization" / "memory" is that it assumes that some attribute (e.g. I like pizza) should now apply to all my queries, even if not relevant (e.g. high recall, low precision). Am I correct in assuming this is a "preference hallucination" violation? 

> To validate our LLM-based evaluation method, we manually checked 200 randomly sampled evaluations, with an observed 5% error rate.
How was this error rate computed? What was the distribution of errors in the 200 randomly sampled evals? 

How were the implicit personas generated and validated? How much variance is there between the generated personas?

With the additional space in the camera ready, I would move some examples of (p, q, explanation) pairs to the main text.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This Datasets and Benchmarks paper presents PREFEVAL, a benchmark designed to evaluate large language models’ capabilities in following lifestyle preferences in long-context, multi-turn, personalized conversational settings. PREFEVAL includes 3,000 manually curated preference-query pairs spanning 20 diverse topics, assessing models’ ability to remember, infer, and follow explicit and implicit preferences over long conversational contexts (up to 100,000 tokens) and tests various settings such as zero-shot, reminding, CoT, RAG and Self-critic approaches. Using generation and classification tasks, the authors evaluate 10 LLMs, including prominent models like Claude, GPT-4, and LLaMA. The results indicate significant challenges for LLMs in following user preferences over extended dialogues, with a steep decline in preference-following accuracy as context length increases.

### Strengths
- 3,000 manually curated (with the help of LLM tools) preference-query pairs across 20 diverse topics is a good dataset contribution, covering various preference forms (explicit and implicit) and contexts, making it a valuable tool for future research on personalized conversational AI.
- The paper presents valuable insights across major open and proprietary models along with an understanding of various strategies that work to help improve preference following. 
- The error categorization (preference-unaware violation, preference hallucination, inconsistency violation, and unhelpful response) provides actionable insights into the types of failures occurring during preference following. These insights allow targeted intervention methods to improve the performance of the models and reduce hallucinations and increase helpfulness.
- Personalization is a very important topic for LLMs that's still underrepresented and scope for major advancements. Need for more benchmarks in this field will help with faster progress to create more personalized LLMs.

### Weaknesses
- The LLM-based evaluation framework introduces a subjective component and lacks qualitative and manual validations for in assessing preference adherence (despite a very small manual checks on the responses with 5% error rate) 
- inserting preferences within the LMSys-Chat-1M dataset may be unnatural. It is not clear what strategies were used for maintain semantic coherence of the dataset while introducing preferences and queries. 
- There is some overlap with long context recall and reasoning based benchmarking. There is no analysis of existing datasets and benchmarks in the area that may qualify for personalization based benchmarking. Also, there is no discussion around other efforts like 'LaMP: When Large Language Models Meet Personalization' or others (authors mention these in Appendix, however an exhaustive personalization benchmark could cover some aspects of these other benchmarks mentioned in Appendix).
- This study emphasizes improvements with inserting 'Reminders' and other strategies like self-critique. This might skew model development more towards prompt engineering instead of focusing on understanding intrinsic improvements methods in memory and retrieval capabilities.
-

### Questions
- Authors mention breakdown of queries by explicit vs implicit preferences (choice based, persona driven) and share results in Figure 5. Shedding more light on some visualization of attention (and the lack of it) could have shed more light on understanding why the results degrade so much in different scenarios. 
- Authors use LMSYS- Chat-1M dataset for the experiments (where they intersperse persona/preference based queries) - some of the models they test may have seen this dataset before during training - what would be the impact of such exposure?  
- Since this is a dataset contribution, some more clarity on understanding dynamic user preferences might have been useful. For example, mining instances where personalization instances already exist in LMSys dataset and understanding preference following there would have made the work very convincing.

### Soundness
3

### Presentation
3

### Contribution
3
