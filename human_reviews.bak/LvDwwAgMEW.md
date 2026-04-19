# Eliciting Human Preferences with Language Models

- Decision: Accept (Poster)
- Scores: 5, 8, 6, 8

## Abstract
Language models (LMs) can be directed to perform user- and context-dependent
tasks by using labeled examples or natural language prompts.
But selecting examples or writing prompts can be challenging---especially in tasks that require users to precisely articulate nebulous preferences or reason about complex edge cases. For such tasks, we introduce **Generative Active Task Elicitation (GATE)**, a method for using *LMs themselves* to guide the task specification process. GATE is a learning framework in which models elicit and infer human preferences through free-form, language-based interaction with users.
We identify prototypical challenges that users face when specifying preferences, and design three preference modeling tasks to study these challenges:
content recommendation, moral reasoning, and email validation.
In preregistered experiments, we show that LMs that learn to perform these tasks using GATE (by interactively querying users with open-ended questions) obtain preference specifications that are more informative than user-written prompts or examples. GATE matches existing task specification methods in the moral reasoning task, and significantly outperforms them in the content recommendation and email validation tasks. Users additionally report that interactive task elicitation requires less effort than prompting or example labeling and surfaces considerations that they did not anticipate on their own. Our findings suggest that LM-driven elicitation can be a powerful tool for aligning models to complex human preferences and values.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
Formulating prompts or identifying few shot examples can be challenging. To solve this problem, the paper introduce generative active task elicitation to allow LMs to guide task specification. More specifically, this is about eliciting preference of individual persons in context of personalization based on Figure 1. The core idea is to have LMs interact with users (by LM providing questions) in order for users to provide detailed explanation of what task they want LM to perform. In three empirical settings (content recommendation, moral reasoning and email validation), the paper find this approach to work as well or better than prompt formulation or example labelling. Users also report requiring less effort than other approaches.

### Strengths
1.	The topic is interesting as generally, the concept of personalization using LLMs (and the best way to do this) is an under-investigated topic, that is likely to be growing in importance as LLMs become more integrated into daily life.
2.	The paper validates the proposed methods in three distinct domains: content recommendation, moral reasoning and email validation, which are important for establishing the generalizability of the method.
3.	The method should statistically significant results from traditional methods in content recommendation and email verification as shown in Figure 3.

### Weaknesses
1.	The empirical experiments are done on very small datasets (consisting of 16 to 54 test cases) that have been manually curated by the authors. Although the authors do note that all 388 participants do every task, it’s not clear to me that a. the tests are valid and b. the results will hold outside the examples provided by the authors. Is there a reason why external tasks (which have been previously validated) such as [1] or [2] for moral reasoning were not used? Also, the low number of tasks also increases the likelihood that findings are not statistically significant, as Figure 3 shows. The submission also does not include the actual datasets, beyond illustrative examples, making it difficult to assess the validity/generalizability of the data.
2.	It’s not clear the email verification task is a good task here since whether emails are valid is a pretty straightforward task. More specifically, why does it matter whether the model matches user judgements on whether it’s correct? An email either is or is not well-formed – modelling user knowledge on a such a simple task is not useful in my opinion. On the other hand, content recommendation and moral reasoning are useful because people can hold different perspectives (based on their own preferences), since ‘ground-truth’ annotations are not available.  
3.	While it’s useful to understand the portion of task that each method gets correct conditioning on Interaction Time (through AUC), it would also be useful to understand the raw correct proportion. This is because it’s possible that user-written prompts takes more time than users only having to respond to questions, so the current metric does not demonstrate the raw correct proportion. 
4.	I think the claim that in Line 487 “While our paper demonstrates value mainly on personalized tasks, we note that this framework can be applied generally as a replacement for prompting or in-context learning” is not grounded on evidence presented in the paper and might mislead others. I would recommend removing this claim from the conclusion.

[1] https://arxiv.org/abs/2210.01478
[2] https://arxiv.org/abs/2110.07574

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper tackles the task of learning about user preference by proposing the task of generative active task elicitation as an alternative to example-based supervised learning, pool-based active learning and prompting. They show, through a pre-registered user study, that with their proposed method, LLMs are able to generate queries to interact with users to learn user preferences with comparable or better effectiveness compared to all baselines and are equally or less mentally demanding than directly having the user write prompts to describe their preferences.

### Strengths
This paper tackles a clear gap in the current literature – while we do have lots of methods to learn from preference data once collected, few works actually try to see how to best elicit preference and this work does just that. While the methodology is simple, this paper could serve as a good baseline for future works in generative task elicitation. It is nice that the paper conducts a pre-registered user study and considers many validation checks and ablations (e.g. can we use LLM to simulate users in this setting? How much variation among users exists in preferences?) Also, I think overall, it is overall very nicely executed and the presentation is very clear.

### Weaknesses
Nothing major that could affect the paper but here are a few suggestions: 


1)	it would be nice if you could refer to papers from other fields more (e.g. marketing or recommender system) when making some claims (e.g. section 4.1 “users might find it difficult to express tradeoffs…)
***

2)	I think in order to justify why this paper is important, one additional thing to show would be to have a sense of how much these types of queries that would necessitate generative preference elicitation actually exist “in-the-wild”, say, in WildChat, right? In other words, how prevalent are these queries that very clearly would benefit from generative preference elicitation, for LLMs? To play devil’s advocate here, one could make the argument that we just need to have a set of general set of alignment targets and that should be enough for the vast, vast amount of user queries. I suppose this paper, as it currently stands, still feels a bit lacking in terms of ecological validity unless you show this – does this sort of queries actually come up and how often?  

***
3)	One thing conceptual that perhaps would be good to clarify: is there any difference between modelling user preference (what they think they want) vs. serving them what is in their best interest

### Questions
1) Would your human-study data be released? 

2) I think it would be helpful to others working in this direction?

3) (Section 4.4), for the user-written prompt condition, do you also place the same time limit of 5 minutes?

4) Could you mention the details about the statistical tests conducted?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an LM-Human interactive solution to elicit user preference to assist complex task processing. The author leverages LM as an effective approach to learning human preferences and values through a conversation with guidance questions.  They design three tasks to evaluate the proposed approach: content recommendation, moral reasoning, and email validation. Compared to supervised learning, in-context active learning, and user-written prompts, the proposed GATE framework generally outperforms the baseline methods. Based on the results, GATE can be regarded as a more accurate human preference elicitation approach.

### Strengths
- well-crafted paper and clear description of the proposed method
- the three tasks are close to real-world scenarios, demonstrating the practicality of the proposed method
- the main conclusion is well supported by the experimental results and analysis

### Weaknesses
- GPT-4 is the only model used in this paper which makes it hard to assess the effectiveness of the proposed approach. Additional experiments with proprietary models or open-source models would make the results more convincing to readers.
- The proposed approach is sorely based on LM-driven elicitation for human preference understanding and explaining. However, the paper lacks references that verify that LM possesses such capabilities. Some references I recommend [1-3]
- The second conclusion, at line 491, is weird given that GPT-4 is the only tested model for this approach. To what extent of number of parameters for a "larger model"? Does this conclusion indicate that the model should be at least as powerful as GPT-4 to incorporate with the proposed method? 


References: \
*[1] Hu et al., 2023, DecipherPref: Analyzing Influential Factors in Human Preference Judgments via GPT-4* \
*[2] Oh and Kim et al., 2024, Uncovering Factor Level Preferences to Improve Human-Model Alignment* \
*[3] Hejna et al., 2023, Contrastive Preference Learning: Learning from Human Feedback without RL*

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper contributes the GATE framework, an alternative to tradition methods of gathering human preferences for various means including alignment, which is used in order to get more flexible and active representations of people. The paper first consists of a theoretical contribution highlighting its various implementations and differences from previous methods, before diving into a set of example cases where it is found that the instances of the GATE framework perform better compared to various baselines in terms of elicitation efficiency and perceived human effort.

### Strengths
Originality: One of the biggest strengths of the paper is its proposal of an alternate method of thinking about how to gather preferences. I think this framework challenges existing assumptions surrounding preference collection that can influence future collection of datasets surrounding human preferences and related areas. 

Quality: Ablation studies are very rigorously done. Especially appreciated the ablation on influencing people's opinions, which I would not have thought of, but is definitely worth including. Evaluations are in general very rigorously conducted. 

Clarity: The paper is extremely well written and framed with respect to past work. It clearly highlights the key differences between GATE and past categories of methods, making the contribution very clear. Evaluations are also well-motivated. 

Significance: Preference learning of people is an extremely broadly applicable domain, and this work contributes towards our understanding of a more holistic view of preference collection.

### Weaknesses
Originality: I think the categories of GATE tested (examples, yes/no, open-ended writing) being fixed is a missed opportunity; ideally the LLM would have an active decision process that chooses between these three (or more) options in order to best address existing issues. For example, edge-cases are probably best addressed by examples, whereas a missing dimension would be better answered using open-ended writing. 

Quality: LLMs used are to some extent out of date. Asking the LLM to output probabilities seems slightly sketchy to me, but even with this there's already reasonable results, so it's a proof of concept. 

Clarity: line 159-160: A(•, •) denote a measure of alignment: can be clearer that here A is like a distance metric, where lower values are better.
line 173: "above" -> "to the left"

Significance: One pessimistic way to view this paper is that it does some structured multi-turn dialogue and gets more efficient performance. In particular from a purely materialistic point-of-view, it could be seen as similar to iterative prompting with refinement. From this view it's not a very novel paper. However, I think this view discounts a fundamental shift in paradigms that is missed by previous work: Allowing for flexibility and active querying. The paper is not a methods paper but a framework/paradigm paper, which I think is the right framing for this. Thus I think that the paper isn't that subjective to these criticisms.

### Questions
Have you tested asking the LLM to choose what type of GATE elicitation to do?

Have you tried combining this with frameworks of pragmatic inference or concepts from Jara Ettinger's works on inference? There might be some interesting intersections there.

### Soundness
4

### Presentation
4

### Contribution
3
