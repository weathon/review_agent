# Evaluating Language Models in Longer Conversational Contexts

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Evaluating long-form conversations between humans and large language models (LLMs) presents a significant challenge in the field of natural language processing. Traditional evaluation metrics and benchmarks have largely focused on shorter language interactions and often fail to capture the nuanced inherent in extended dialogues. To address this, we introduce UPHELD, a publicly available dataset featuring human-annotated long-form dialogues. This dataset not only facilitates robust benchmarking but also serves as a foundation for further research into conversation evaluation methodologies. Using our dataset, we systematically analyze the correlation between current LLM evaluation metrics and human judgment within long-form conversation scenarios. Our findings reveal that conventional metrics lack the sensitivity necessary to assess the complex and often subjective nature of prolonged interactions. We use our dataset to develop an improved evaluation metric that demonstrates a significantly higher correlation with human assessments. The work highlights the need for advanced metric designs and outlines a clear pathway to refine the evaluation of LLM long-form conversations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces UPHELD, a publicly available dataset featuring human-annotated long-form dialogues. This dataset not only facilitates robust benchmarking but also serves as a foundation for further research into conversation evaluation methodologies.

### Strengths
- This paper introduces a new dialogue data, UPHELD, that includes multi-turn conversations with open-ended and natural dialogues across various topics such as customer service and education, and also provide manual annotations. 
- This paper also shows validation results using other datasets such as LLM Arena and Topical Chat.

### Weaknesses
-	Related papers on dialogue evaluation metrics are not included as related research. Also, although there are some evaluation aspects commonly used for human evaluation on response generation, there is no comparison and/or discussion on differences with the conventional approaches. 
-	The number of response generation models used for validating the evaluation metrics is small, which may result in a lack of diversity in response tendencies.

### Questions
- Automatic evaluation metrics for dialogue have been extensively studied from the perspective of one-to-many issues in dialogue. Papers on dialogue evaluation metrics, such as USR, G-EVAL etc., should be cited and compared.

 [1] USR: An unsupervised and reference free evaluation metric for dialog generation (ACL2020)

 [2] GEval: NLG Evaluation using Gpt-4 with Better Human Alignment (EMNLP2023)

 [3] CHATEVAL: TOWARDS BETTER LLM-BASED EVALUATORS THROUGH MULTI-AGENT DEBATE (ICLR2024)

 [4] A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators (AAAI2024)

- Common evaluation aspects in dialogue include fluency, engagingness, consistency, coherence, informativeness, and relevance. Since this paper introduces new original aspects, a detailed discussion is needed to justify the necessity of using these new aspects and how they differ from conventional ones. 
- To verify whether the reference-full approach can collect meaningful labels on ground-truth content overlap, the validation described in Appendix C appears insufficient. (Using only GPT-4o to validate the variety of responses is inadequate.)  
- In Figure 1, GPT-3.5 and GPT-4 show high “reasonableness” scores but relatively low “content accuracy”. Does this not indicate that the responses are reasonable but may differ from the reference? 
- What type of correlation was used in Tables 2 and 3? Typically, Pearson, Spearman, and Kendall correlations are often examined side by side due to their differing characteristics. Furthermore, is this correlation at the turn level or the system level?  
- Based on the prompts, the criteria for LLM-judge focus on whether the content is semantically equivalent, so it is reasonable that it correlates highly with “content accuracy”. However, if the evaluation also considers style accuracy and reasonableness, it might be better to prepare prompts specifically designed for them as well.  
- The comparison models, limited to only three (gpt3.5, gpt4o, and llama), might be insufficient for evaluating the metrics comprehensively.
- Example 13 of Table 15, the history ends with “assistant”, but should it not end with “user” instead?
- It would be helpful to provide a detailed explanation of the calculation method for ensemble metrics.

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
5

### Summary
This paper introduces UPHELD, a dataset designed to evaluate LLMs in long-form conversation. The dataset consists of 400 dialogues, with an average length of 5.2 turn pairs.

### Strengths
N/A

### Weaknesses
1. The dataset, comprising 400 dialogues entirely handwritten by annotators, is presented without any description of the quality control process. It is also unclear how well these dialogues cover real-world scenarios, as the paper provides no information on data categorization, domains, or collection scenarios. Based on the provided examples, the data appears to be overly simplistic.

2. The evaluation methodology is questionable. Although a pairwise comparison (Option A vs. B) is used, the evaluation criteria are limited to "Equivalence" (both content and style) with the single human-written response. Equivalence is a poor metric for open-ended conversation, where multiple valid, high-quality responses can exist. This approach would unfairly penalize a model's response that might be different from, or even superior to, the human reference.

3. The scope of the model evaluation is extremely limited. The paper only benchmarks three open-source and closed-source models (GPT-3.5, GPT-4o, and Llama-3.1-70b). A contemporary benchmark paper should include a much wider and more representative set (e.g., 15+) of both open- and closed-source models to provide a meaningful comparison.

4. The paper is missing citations to several key related works:
```
@article{sirdeshmukh2025multichallenge,
  title={Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms},
  author={Sirdeshmukh, Ved and Deshpande, Kaustubh and Mols, Johannes and Jin, Lifeng and Cardona, Ed-Yeremai and Lee, Dean and Kritz, Jeremy and Primack, Willow and Yue, Summer and Xing, Chen},
  journal={arXiv preprint arXiv:2501.17399},
  year={2025}
}

@article{bai2024mt,
  title={Mt-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues},
  author={Bai, Ge and Liu, Jie and Bu, Xingyuan and He, Yancheng and Liu, Jiaheng and Zhou, Zhanhui and Lin, Zhuoran and Su, Wenbo and Ge, Tiezheng and Zheng, Bo and others},
  journal={arXiv preprint arXiv:2402.14762},
  year={2024}
}
```

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes UPHELD, a human-annotated benchmark for evaluating LLMs in longer conversational contexts. 
The authors 
(i) collect human-written multi-turn dialogues and, at each turn, compare a human continuation (Option A) to a model continuation (Option B); 
(ii) annotate content equivalence (Likert 1–5), style equivalence (described as a 3-point scale), and reasonableness (binary, scored as 1 vs 5) with five annotators per instance; 
(iii) show that common reference-based metrics (ROUGE, BERTScore), embedding cosine similarity, and several “LLM-as-judge” prompts correlate only weakly to moderately with human judgments on UPHELD; and 
(iv) report that simple learned ensembles of these metrics improve correlation with human scores on UPHELD, with some cross-dataset transfer to augmented LLM-Arena and Topical-Chat verification sets.

### Strengths
1) Clear problem focus: The paper squarely targets an under-served regime—longer conversational evaluations—and provides annotation criteria

### Weaknesses
1) Limited literature survey on long-context / long-conversation evaluation
The Related Work mainly lists single-turn or short-turn task benchmarks plus a few multi-turn dialogue sets (e.g., MuTual, Topical-Chat, DailyDialog, Arena) , but it omits several directly relevant, recent efforts that study long-term conversational memory, multi-session dialogues, or long-context evaluation frameworks: for example [1]:LoCoMo, [2] Long Time No See [3] MultiChallenge. 
The authors need to contextualize UPHELD vs all these pervious works in order to claim the contributions. 

2) Unclear or inconsistent aspects of the dataset creation process
“Long” context is numerically short. The average context shown to annotators is ~560 characters (≈100–150 tokens) with 10.4 turns per dialogue—far from what the community typically calls long context today (thousands of tokens). This weakens the central claim and limits external validity for true long-context settings.

Scale inconsistency. Style is described as a 3-point Likert scale in §3.1.2, but Appendix A instructs raters to pick 1/3/5 (also three options, but a different framing)

Authoring & sampling specifics. The paper says Upwork professionals wrote diverse dialogues and that bias mitigation steps were taken, but leaves several concrete items unclear: number of writers, prompt archetypes/personas, topic distribution



[1] LoCoMo: a very long-term conversational memory benchmark (≈300 turns, ≈9k tokens per conversation; multi-session) with tasks for recall, summarization, and QA over long dialogues. This is highly germane to the paper’s “longer conversation” scope. 
arxiv.org
[2] “Long Time No See!”: open-domain long-term conversation with human evaluation on coherence, consistency, engagingness—directly relevant axes for multi-turn dialogue. 
ACL Anthology
[3] MultiChallenge (Findings ACL 2025): a multi-turn conversation benchmark targeting realistic human–LLM challenges.

### Questions
See Weakness for more details: 

1. Many relevant works discussions are missing. Can you add more references and discuss what's new in UPHELD? 
2. It's not clear to me how you prompt LLM to generate dialogue: 
"Given our initial set of rich natural language conversations, various LLM models were then used
to output candidate completions at every level of every conversation. Specifically, models were
presented with conversation history up to a specific point, with the next human-written turn withheld.
Models then generated a predicted next turn."  
What do you mean by every level? Are you prompting LLM to generate at every turn?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new dataset of long-form task-oriented conversations by Upwork workers to assess how well models can rate the quality of these conversations. To assess, each human turn in the conversation is compared with an LLM's prediction of what the next step would be. Human then evaluated these turns in terms of semantic and style equivalence and reasonableness. Verification datasets were produced from LLM-Arena and Topical-Chat. To compare with the human rates, multiple model-based metrics (e.g., BERTScore) were also compared, as well as LLM-as-judge. The evaluated compared three larger models, as well as a Llama3.1-8B fine-tuned on some of the data. The machine evaluations were compared by correlating with human scores and also fitting a model to predict human scores from multiple metrics. The results show that the human scores are predictable from ensembles, though humans do disagree on the scores.

### Strengths
- New dataset introduced of realistic conversations

- Meaningful comparison of human annotation and LLM-as-judge on modeling long conversation qualities

- Multiple models and metrics compared

### Weaknesses
- I wasn't sure how big the dataset was in the end. In 3.1.3, the paper says that Upwork workers had 53 conversations but line 215 says ~1220 conversations are rated, which is much later. 53 conversations seems like too few to meaningfully evaluate, so this could be a significant weakness if so.

- The paper's framing is about evaluating in "longer conversation contexts", which is an admirable goal. However, the data that is actually produced doesn't quite match this framing. First, while the data is dialog, is more task-oriented dialog, rather than chit-chat, so I'm not sure it's fully representative of "conversation" as we might understand human-LLM conversations. I realized that the authors' design for this is intentionally as it simplifies the scoping for conversation; I don't think this part is wrong but the paper would help from being reframed for this scope. Second, the "longer" conversations are still quite short, with a mean of 5 turn pairs. This is still helpful than single turns or much shorter, but I think this might still be missing some important qualities of much longer conversations that are still very realistic. Again, the data is still good, but some reframing for scope would be very helpful.

- This is somewhat minor, but the models used in this paper are old now and have been surpassed by much more recent models in their performance. I don't think this should require you to generate new conversations, but for the LLM-as-judge more recent models may be much more capable judges than what are tested here. The newer Qwen3 models are quite good as is 

- This is more of a comment than a weakness, but while I like that the authors tried multiple judge prompts (appendix F) I am still a bit skeptical of the claim that these models are not good evaluators of longer conversations. I would have liked to see a bit more analysis here of what works and some additional variation in the prompt formatting to rule that at out as a confound.

### Questions
- How big is the dataset in practice?

### Soundness
3

### Presentation
3

### Contribution
3
