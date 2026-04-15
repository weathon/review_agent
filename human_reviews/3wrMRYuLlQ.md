# On the Language of Thoughts in Large Language Models

- Decision: Reject
- Scores: 1, 6, 6, 6

## Abstract
System 2 reasoning is one of the defining characteristics of intelligence, which requires slow and logical thinking. Human conducts System 2 reasoning via the language of thoughts that organizes the reasoning process as *a causal sequence of mental language*, or thoughts. Recently, it has been observed that System 2 reasoning can be elicited from Large Language Models (LLMs) pre-trained on large-scale natural languages. However, in this work, we show that there is a significant gap between the modeling of languages and thoughts. As language is primarily a tool for humans to share knowledge and thinking, *modeling human language can easily integrate into language biases* that are not related to thoughts. Furthermore, we show that the biases may mislead the eliciting of “thoughts” in LLMs to focus only on a given part of the premise. To this end, we propose a new prompt technique termed **L**anguage-**o**f-**T**houghts ( LoT ) to alleviate the issue. Instead of directly eliciting the chain of thoughts from partial information, LoT instructs LLMs to focus and expand based on all the relevant information. We show that the simple strategy significantly reduces the language modeling biases in LLMs and improves the performance of LLMs across a variety of reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
This paper attempts to distinguish between language modeling and thought modeling.  While LLMs right now model linguist to imitate human reasoning, the authors claim that there is a gap between language and though, which can introduce certain biases.  They propose a new type of prompting that they call Language of Thought Prompting and provide various experiments comparing Language of Thought with CoT type prompts.

### Strengths
The authors are correct that permutations of sentences or premises for reasoning can lead to untoward results in LLM performance.  The ideas behind the LoT prompting style have some promise.

### Weaknesses
While the ideas behind the new prompting style might be promising, this paper needs a radical overhaul to be acceptable.  Too much in this paper is just hard to understand or vague.   The leading question for the paper for example is already muddled: what does it mean to elicit the language of thought as humans???  Here is the quote: 
"Do LLMs with CoT model and elicit the language of thoughts as humans?"
The whole set involving a comparison up between language and thought in humans is somehow besides the point that  the authors want to bring up.  The motivation for the LoT style prompting seems to rely on is that different strings of linguistic tokens may express the same or at least very similar linguistic content to humans.  This is not a language vs. thought issue but rather an issue of whether LLM objective functions as they stand, or even with CoT, capture real linguistic, semantic content.  There is a lot of literature out there on insufficiencies of LLM in capturing semantic content; e.g. there's Bender, Emily M., and Alexander Koller. "Climbing towards NLU: On meaning, form, and understanding in the age of data." Proceedings of the 58th annual meeting of the association for computational linguistics. 2020.   But perhaps more relevant to the authors are two recent papers that look in detail at how LLMs fail to capture semantic meaning:
 "Strong hallucinations from negation and how to fix them" arXiv preprint arXiv:2402.10543 (2024);
``Analyzing Semantic Faithfulness of Language Models via Input Intervention on Question Answering'' (Computational Linguistics 2024).  
Speaking in terms of semantic contents instead of thoughts gives the authors a lot more ammunition to investigate where CoT approaches break down.  Simply put, we know a lot more about the structure and features of semantic contents than we do about thoughts.
If the authors can show in detail that LoT methods capture semantic content better than CoT methods, that could be an important finding.

There's also a confusion between between logical and causal sequences.  I quote:
``Human conducts System 2 reasoning via the language of thoughts that organizes intermediate steps
as a causal consequence of mental representations (Rescorla, 2024). For example, a human baby is
able to abstract, construct, and reason over a causal map of the world in their minds."   The important point it would seem to this reviewer in system 2 reasoning is that the intermediate steps follow each other in terms of logical or semantic consequence, not causal consequence.  Or rather in a good system 2 reasoning system causal consequence and logical consequence merge.   Why do I say that?  Because you can have causal sequences of thoughts/representations in a psychopath that are completely crazy have no logical relation to each other and have nothing to do with type 2 reasoning.  


The authors move quickly from evidence about thoughts without language to the thesis that "As language is primarily a tool for communication instead of thinking and reasoning".  But there is nothing in the paper that warrants this assertion.  And unfortunately,
this assertion is key to the paper and drives the move to find different prompts from standard CoT prompts.

The paper is woefully short on examples and often it's very difficult to understand what the authors want to say:

For example: "Thoughts are the unobserved high-level random variables evaluated by
brains that drive us to generate language."
It's really hard to figure out what to do with this.  And in addition it's a definition.


another example, 'When a premise is expressed in an implicit expression under a context, it is
hard to notice and utilize it for downstream reasoning' what does it mean to have an implicit expression?

Here's a sentence that just seems to be flat out false: "For humans, since the language order does not determine the language
meaning when given proper conjunction words, one can easily change the order of presenting the
premises in need."  
As a counterexample, consider:   
John took off his shoes and went to bed
vs. 
John went to bed and took off his shoes

The meaning conveyed by these two sentences is quite different.  Changing the order of sentences often changes the meaning of a text.
This is part of the study of discourse structure and how it's formally interpreted.  E.g. N. Asher & A. Lascarides, Logics of Conversation, 
Cambridge University Press, 2003.  And unfortunately this assumption seems to be key to distinguishing LoT from CoT


A lot of the sentences in this paper aren't English or well formed in any language I know of .  Eg. "The Interplay between language and thoughts has intrigued a long historical discussion about
the role of language in human thinking in the literature (Rescorla, 2024; Fedorenko et al., 2024)."
Issues don't intrigue a historical discussion.  In addition, if the history is long, why cite people from 2024?  The authors might 
start by citing Fodor Language of Thought (1975) but actually the issue already arises with early medieval thinkers like Saint Augustine and his concept of the "verbum mentis".  See the Stanford Encyclopedia article on medieval semiotics.


Again: "Consequently, modeling thoughts merely from the language
can easily integrate the language modeling biases into the learned model, such as the order (Wei et al."
this is largely unintelligible, and at this point language modeling biases haven't been defined.


The expanding thought part of the proposed prompt is too vague to be at all useful as it currently stands.
"instruct the model the expand those" is not English or comprehensible.  Please rephrase

### Questions
This reviewer would have quite a few questions once the basic terms of the analysis are clarified.

A basic one would be: why the insistence on causal relations between representations instead of deductive or inductively supported ones?
Please see my discussion above.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a prompting technique (Language of Thought, LoT) that provides a better structured step-by-step reasoning than the conventional Chain of Thought by asking the model to echo and expand the relevant information before answering. LoT alleviates the biases that are sometimes introduced by language modeling (societal or just general learning bias). LoT can also alleviate the regressions that CoT sometimes introduce in non-math domains. The paper also introduces some theory on language of thought bias, that is later used as intuition to support the design of LoT.

### Strengths
* Simple prompting technique that can reduce language modeling bias (both societal and general learning biases) by echoing and expanding the required concepts before answering a complex question. I find particularly interesting that the technique shows that recent reports of performance degradation when using CoT may not be intrinsic to the step-by-step technique, but rather to the concrete execution shown in CoT.
* The authors provide a useful intuition on why some biases may arise (Section 3), which is shared by many researchers but is important to spell it out like this work does. This intuition is useful for other applications besides the general motivation for a more structured CoT.

### Weaknesses
* The theory presented is very useful to have in mind when designing prompts or generating synthetic data in general, but it is not really used anywhere in the paper except as general motivation, and it is not as formal as one would hope (e.g. propositions do not have a proof).
* Consider rewriting the intro with more nuance, especially when describing psychological phenomena. E.g. System 1 and 2 should have more nuance, as these are useful theories, but they are not necessarily universally accepted in psychology. This psych discussion does not matter for CS research, as they are just useful analogies to our research, but the literature should be discussed well.

### Questions
* LoT requires more inference time compute than plain CoT, given that it has to echo+expand. Have you considered doing a cost analysis to show how much gains one can have, vs how many more tokens are needed?
* For domains where CoT is already good (e.g. math), is LoT still better or equal than using CoT? That would further show the potential of the technique. More in general, what are the potential drawbacks of using LoT?
* Consider giving the phenomenon a more unique name than “bias”, as it is a very loaded term. In the first part of the paper it appears that we will focus on evaluating societal biases, and then the experimentation broadens and shows it is a general learning bias.
* Consider presenting the results better in Figure 5, as it is hard to grasp the overall performance loss reduction vs general gains with LoT. It would be important to grasp when the technique can offer large improvement gains, on top of the bias reduction.

### Soundness
2

### Presentation
3

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
The paper establishes the inherent gap between the language of communication, and the language of thought, showcasing that functionally equivalent language representation of a thought process might lead to biased behavior in Large Language Models during both training and inference due to their autoregressive objective definition.
To mitigate this shortcoming, the paper introduces LoT, a prompting method that aims to force the Language Model to echo and expand the facts contained in the input, converting the potentially unusable implicit knowledge, to usable explicit context. 
Evaluation of the above hypothesis shows that LLMs, when prompted via LoT, showcase superior behavior with respect to model fairness and bias, while gaining an overall boost in reasoning performance. 

Overall, I find the proposed prompting method interesting and potentially effective under reasoning and fairness intensive tasks. However, a number of concerns such as the applicability of method and its comprehensive analysis still remain as outlined below.

### Strengths
- The general concern of the paper is both valid and very interesting to explore in the literature. It is indeed true that given the autoregressive nature of Large Language Models, one can expect their inherent modeling biases to "leak" into the reasoning process, contaminating the possible results.
- The paper is well-written and well-presented, with good formulation of assumptions as well as examples for the easier understanding of the readers.
- The proposed method is relatively simple and effective, making its use-case accessible to users and across different models.

### Weaknesses
- Despite the success of CoT-like prompting methods in the simulation of system 2 processes, I think we have to be careful when trying to achieve system 2 way of thinking via prompting, as LLMs, due to their training objective, will always carry their linguistic capabilities and biases to all prompting methods. Further architectural changes may even be necessary to reach true system 2 capabilities. 
- Regarding the inference time linguistic bias, I disagree that implicit representations of the context completely forbid the LLM from using them in its reasoning as the implicit representation is certain to encode parts of the explicit representation in itself as well. However, it is true that it can make the reasoning more difficult. I suggest changing the language in this section to reflect that.
- Experimental results showcasing that the training, and inference bias, as mentioned in the paper, are the reasons behind the underperforming of the models in system 2 related tasks are sparse throughout the paper. Further evaluations and theoretical considerations should be made to further bolster the proposed conjecture.
- The ablation study shows that the effect of each component can be inconsistent across models and tasks, therefore I suggest a deeper analysis of each component, possibly via the manual investigation of the model response change based on the inclusion of a component.
- Following from the previous point, I still find it somewhat unclear how the proposed prompting method ameliorates the model bias, I suggest further evaluations to show the change in behavior.
- I think that the relationship between the system 2 thinking of LLMs and the proposed method can be better explored via the construction of direct links between each other and showcasing how the prompting method addresses a problem in the large language model.

### Questions
- While somewhat inline with the paper's general theme, I think it should be noted in line 44 that the large language models using CoT merely strive to simulate system 2 way of thinking via the continuous application of their inherently system 1 capabilities and their method is not a true system 2 reasoning process.
- I think a better explanation of equation 2 would be good to have. If I am not mistaken, this highlights the fact that if topological ordering of reasoning is not preserved by the language, the LLM will only learn a causal relationship between $L_1$ and $L_A$, relegating $C_2$ to a distributional shortcut. It would be adequate to briefly explain the corresponding behavior in the text as well.
- In comparison to CoT, what do you think are the main differences of LoT that lead to its superior performance? 
-  Regarding the extraction of implicit information as well as the exploration of various reasoning paths, the following papers may be of interest: 
	- [ 1 ] aim to get better results via the exploration of contrastive reasoning paths in CoT. 
	- [ 2 ] discuss various prompting methods including the distillation of explicit knowledge from implicit context or the utilization of implicit knowledge.
	- [ 3 ] aim to ameliorate affirmative bias of CoT via the exploration of counter-paths for each LLM response. Their overall bias and GPT-4 results are also consistent with those of this paper.
- Consider writing the formula for the "Bias Score" rather than its natural language explanation for better readability.
- Line 295, "that relevant" -> "that is relevant"
- Line 359, "dive the data" -> "divide the data"?
- Line 468, " since the original evaluation results consider correct formats in the incorrect formats to be incorrect answers." -> "Consider correct answers in the incorrect formats"?
- There is a discrepancy between the prompt name in the submitted version and the TLDR version, I believe CaT should be changed to LoT or vice versa.

[ 1 ] Chia, Yew Ken, et al. “Contrastive Chain-of-Thought Prompting.” _arXiv.Org_, 15 Nov. 2023

[ 2 ] Yu, Fei, et al. “Natural Language Reasoning, A Survey.” _arXiv.Org_, 26 Mar. 2023

[ 3 ] Miandoab, Kaveh Eskandari, and Vasanth Sarathy. “‘Let’s Argue Both Sides’: Argument Generation Can Force Small Models to Utilize Previously Inaccessible Reasoning Capabilities.” _arXiv.Org_, 16 Oct. 2024

### Soundness
2

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
This paper analyzes language modelling bias and the language-thought gap in the context of LLMs. It proposes a new prompting technique, LoT, to address these issues and evaluates its effectiveness across two bias benchmarks and eight general reasoning benchmarks, using six different LLMs.

### Strengths
- the paper is well-written and includes extensive results across a diverse range of datasets and LLMs.

- the analysis of language modelling bias and the language-thought gap offers an interesting perspective within the LLM community.

### Weaknesses
- the construction of LoT does not appear fully derived from the analysis of language modelling bias and the language-thought gap (which contribute to a big part of this work). In other words, it seems that LoT is not a natural product/derivation from the bias analysis, e.g., how does LoT overcome the issue that "one piece of information can have different expressions in language"? LoT’s design choices, such as "expanding thoughts," seem empirically beneficial but not systematically motivated by the initial analysis. Additionally, the claim in Section 5.2 that "the expansion prompt may exacerbate language modelling biases" potentially undermines the rationale behind this feature.

- the performance of the "echo only" prompt, which frequently outperforms LoT as shown in Table 3, highlights the need for a deeper understanding of LoT's effectiveness (relating to the above concern). Although the work presents comprehensive testing of LoT across datasets and LLMs, readers would benefit from insights into the underlying mechanisms that make this prompting design work—or fail.

- in Figure 5, the range of benchmarks and LLMs is commendable, but it is unclear why direct prompting is included in the comparison rather than other CoT-like techniques, which might provide a more meaningful comparison to LoT. Given that this study doesn’t focus on the benefit of having internal steps before LLMs generate outputs, the inclusion of direct prompting lacks relevance.

- in cases where LoT performs worse than ablations or other baselines on general reasoning benchmarks, the authors provide conjectures to explain this. However, these conjectures are not substantiated by evidence (e.g., lines 428-431, 517-519).


- the reference to the previously established CoT paradigm used in the experiments is missing.
- the three types of bias used in the evaluation should be explicitly introduced in the main text rather than solely in the appendix.
- providing practical examples of two-premise QA would help illustrate the generalizability of the training corpus used here and ground the analysis of the language-thought gap later in the main text.
- in Figure 2, the arrows are stated to represent causal relations, yet it appears the arrows in the blue box denote topological order.
- while formal propositions are generally beneficial for clarity, propositions 3.3 and 3.6 seem redundant as their informal explanations suffice, and they are not referenced further in the main text.
- in Section 3.2, it is unclear how prompting LLMs to "notice more details" addresses the problem of "ignorance of implicit premises." This approach may not necessarily lead to locating additional implicit premises, raising questions about the effectiveness of the LoT design.
- the font size in Figure 5 is very small, impacting readability.

### Questions
- the last part of Definition 3.5 ("Otherwise, ...") seems to be repeated.

- the abstract on OpenReview does not match the PDF abstract; could the authors clarify?

- could the authors explain how worst-group accuracy is computed and how this leads to the conclusion that "current language models struggle to properly utilize given premises for reasoning"?

- could the authors provide more detail on the different prompting strategies used in Section 4.1?

### Soundness
2

### Presentation
2

### Contribution
1
