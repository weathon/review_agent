# SMARTS: A Small-to-large Model Coordinated Retrieval and Response Framework for Task-Oriented Dialogue

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 2

## Abstract
Task-Oriented Dialogue (TOD) systems are commonly used to assist users in achieving specific goals through human-computer interactions. Existing methods typically employ a single model during response generation to simultaneously learn response policy and perform fine-grained knowledge retrieval. However, these task-coupling methods often lead to suboptimal response generation, incorporating irrelevant knowledge and policy-violating styles, which makes optimizing TOD systems more difficult. To address this challenge, we propose a novel small-to-large model collaboration framework for task-oriented dialogue, named SMART. This framework utilizes a small language model to generate style-only responses without knowledge, while a large language model retrieves top-1 relevant knowledge from the knowledge base independently. The style-only responses are finally filled with top-1 relevant knowledge and form the completed responses to users. Finally, a re-thinker is designed to check the contextual relevance and knowledge accuracy of responses. Experiments on two public datasets demonstrate that SMART outperforms existing methods by an average of 5.79\% in Entity F1.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors focus on the problem of incorporating responses in task-oriented dialog (TOD) systems. They state that the issue with current TOD systems arent able to injecting knowledge effectively in their responses either because: the LLM has to do both response generation and knowledge retrieval which makes the task harder or if there is a retrieval mechanism incorrect knowledge can be retrieved.

The authors approach is to have Small Language Model (SLM) generate style-only responses containing placeholder slots for attribute and a LLM to retrieve the most relevant knowledge records by filtering on attribute slots. The values retrieved for those particular slots are filled in for the final response. Additionally there is a separate LLM or Re-Thinker that determines if the final response is contextually relevant if it is not the Re-Thinker model regenerates the response. The response-style generator is finetuned on two task-oriented datasets, the knowledge retriever and re-thinker model are not.

The authors show that this performs baselines which use the previous methods mentioned above on metrics such as BLEU (4-gram), Entity F1, Entity precision, Entity recall, and Entity accuracy. The authors perform an ablation study to show each part of their system is valuable. They also compare their method versus just asking an LLM to generate a response with retrieved knowledge and show it is better than this approach. 

The contributions as stated in the paper is: 1) a method that leverages a SLM to generate style-only responses to be filled in and 2) a more accurate knowledge retrieval mechanism.

### Strengths
1) The authors certainly describe their method in a clear manner and run a fair number of ablation studies to show the different components of their work.

2) Additionally they use an extensive set of baselines and even compare against very powerful LLMs.

### Weaknesses
There are a couple claims in the paper that I think can be clarified more:

1) It is mentioned in Table 3 that "Although LLMs have strong reasoning ability, directly using them for response generation without finetuning often results in verbose outputs containing irrelevant or excessive attributes." While I understand that current LLMs do generate long outputs the responses could still be useful and it would be hard to determine if it's worse or better for a user without human evaluation. Also if you wanted couldn't you add in the prompt to make sure the responses are short or ask it to not use excessive prompts? What was the prompt used? Is it the same prompt used in the re-thinker?

2) In what cases would the response not be appropriate? Since you are retrieving the exact attributes needed along with their slots. Also how much of the knowledge gets filtered out during your retrieval on average.

3) One of the novelties mentioned in the paper is using attribute slots to filter out knowledge records, but it seems like MAKER already does this so what's the difference?

### Questions
1) Why was SAGE not used as a baseline? 

2) What are the SOTA numbers on the two datasets you used? I'm asking because I'd like to know if the methods used for those are different than yours or is your approach the SOTA?

3) More of a suggestion: Put link to Table 1 in Section 4.5. Also are the results statistically significant?

4) What is LM vs LLM? Do you mean SLM?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes a small-to-large model for task-oriented dialogue generation.

The overall idea of  `SMART`  is a simple two-stage pipeline:

- First, it uses a small language model (Flan-T5) to generate a prototype dialogue response ( style response), which leaves the knowledge entity as some placeholders.
- Then, it uses a large language model (Llama3-8B) to fill the entity placeholders.

Experiments on two datasets have shown the effectiveness of the proposed method in Entity-Related metrics.

### Strengths
1. The presentation of this work is easy to follow.

2. `SMART` has chosen a different paradigm compared to the most related works. 

3. Compared to the baselines selected by the authors, the performance of  `SMART` is not bad.

### Weaknesses
1. Line 127:  A lot of pipeline-based works use different models to retrieve knowledge and generate response, instead of a single model. In the era of RAG, a lot of works use different modules in different stages, then now there are already a lot of dense retrievers.

2. The novelty of this work is limited.  The core idea of this work is to first generate a skeleton response, which was populated about 5 years ago (just like `Skeleton-to-Response: Dialogue Generation Guided by Retrieval Memory`).  Besides, I also can't find other innovations that are novel enough.

3. According to the experimental results, I think the experimental setting is specifically optimized for non-LLM models.  Many metrics are not suitable in the era of LLMs, especially reference-aware metrics.  Thus, the comparison results between the well-trained small language model and the instructional large language model are somewhat meaningless in this paper.

4. Lack the evaluation of time complexity.

5. Lack the discussion/comparison of using powerful dense retrievers.

6. The proposed paradigm still requires a lot of  `Supervised Fine-Tuning` processes when constructing the fundamental modules, which is outdated in the era of LLM. Only a limited set of scenarios can benefit from this work.

### Questions
1. Line 37-43:  This paper cited some works that use the entire knowledge base to generate the dialogue. It is strange because most works will have an explicit knowledge retrieval process (such as Figure 1b). The full text length of most knowledge bases is often far beyond the LLM input limitation. So, I am wondering about the practical scenario of this paradigm.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SMART, a modular framework for task-oriented dialogue (TOD) that decouples response style generation (handled by a small LM) from knowledge retrieval (handled by a frozen LLM). A fine-to-coarse retrieval mechanism is proposed that uses attribute slots from style-only responses to filter and select the top-1 relevant knowledge record. A Re-thinker module is further employed to verify contextual relevance and regenerate responses if necessary. Experiments on CamRest and MultiWOZ show consistent improvements over strong baselines across multiple metrics.

### Strengths
- The method proposed represents an improvement over the chosen baseline.
- The paper reads smoothly, with no obvious grammatical issues.

### Weaknesses
The weaknesses of the article are summarized below; see the **Questions** for specific corresponding issues.
- Limited Innovation: There are many works that decouple retrieval and generation, and there are also simpler and more efficient methods. The innovativeness of this paper is not very clear.
- The approach and motivation are quite confusing, lacking analysis and demonstration of why the proposed method works.
- The performance comparison is questionable: it cannot be ruled out that the gain obtained by this method may come from multiple LLM calls.

### Questions
### Typro
-  Line 184: an SLM -> a SLM。
-  Line 233: designed as -> is designed as

### Questions
- How does this method differ from previous paradigms that separated retrieval and generation, such as MAKER[1], SynctTOD[2]? For example, is the style generator used to predict relevant attributes? If so, why generate a template with slots? Can't we just generate the relevant attributes directly? If we generate a stylized sentence, will it limit the model's response capability? I don't think this method is fundamentally innovative compared to previous methods.
- Why can large, untrained language models replace dedicated retrieval components for ultra-large-scale retrieval? What is the inference latency of large models used for retrieval? Does the length of the text input to a large database exceed the understanding length of an LLM?
- Line 201 “such as assessing whether specific attribute values are appropriate in context.” How can we assess whether an attribute is appropriate across contexts? why the CE(ˆpi, pi) only can not do this?
- Where does the golden style-only response come from? Is it in the dataset?
- The authors introduce an auxiliary loss that trains the style generator (SLM) to predict the final response (i.e., with knowledge filled in), in addition to the style-only template. While the stated goal is to “enhance the model’s understanding of fine-grained knowledge usage,” this objective appears to contradict the core design principle of full decoupling between style and knowledge. Could the authors clarify how this supervision on the completed response does not reintroduce a coupling between generation and knowledge retrieval? And provide an ablation study about the influence of CE(ˆs′i, si). Since multi-task optimization loss is used, how can we ultimately guarantee the generation of style responses with slots?
- Is there a comparison and analysis between not generating style responses and generating attributes without them? Is there an explanation of why style responses are better, and what makes them reasonable?
- Rethinking the mechanism and LLM retrieval involves frequent calls to large models, which increases the response latency for task-oriented dialogues. Has the performance-latency ratio of the model been measured?
- Lack of comparison with related work, such as syncTOD [2], etc.
- On the claim of “fine-to-coarse” retrieval. The paper advertises a fine-to-coarse retrieval mechanism; however, the actual pipeline performs a single retrieval step over an attribute-filtered knowledge base. We do not observe a progressive widening of candidate scope.
- Insufficient per-component ablation study, such as slot accuracy: The quality of the slots Ai′ extracted by the SLM is not reported. Errors at this stage are irreversible and could mislead the entire retrieval. Please report the precision of slot extraction, and analyze how slot errors influence the generation quality.

[2] Synergizing In-context Learning with Hints for End-to-end Task-oriented Dialog Systems

### Soundness
1

### Presentation
2

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
This paper introduces a task-oriented dialogue framework named SMART, which addresses the coupling issue of knowledge retrieval and response generation in traditional TOD systems through task decoupling. The SMART framework consists of three main components: 1) a style generator, which generates stylized responses based on the dialogue context; 2) a fine-grained knowledge retriever, which utilizes large models for fine-grained knowledge retrieval; and 3) a reflection module, which re-validates the generated responses to ensure their quality and accuracy. Experimental results on two public datasets, Camrest and MultiWOZ, show that the SMART framework improves the entity F1 score by an average of 5.79% compared to other methods. Furthermore, the SMART framework boasts lower computational costs and higher response quality.

### Strengths
1. The authors conduct thorough experiments using two public datasets (Camrest and MultiWOZ), evaluating their proposed method against various baselines and ablation studies. They demonstrate significant improvements in entity F1 scores and discuss the efficiency and accuracy trade-offs of their approach.

2. The paper is well-structured and clearly explains the motivation behind the proposed method, providing detailed descriptions of the SMART architecture and key mechanisms such as attribute filtering, top-1 knowledge retrieval, and reflection module. The experimental setup and results are presented in a logical manner, allowing readers to easily understand the findings.

### Weaknesses
1. Novelty. The approach of removing keywords for style generation is not uncommon, and the overall pipeline is a conventional practice, which may not meet the high standards of ICLR.

2. Dataset Dependency. 
(1)The evaluation of the SMART framework is primarily based on two publicly available datasets, Camrest and MultiWOZ. 
(2)The performance on out-of-domain (OOD) datasets has not been tested. While these datasets are widely used in the TOD research community, future work could explore the generalizability of the proposed method across different domains and datasets.

### Questions
1. Why is the output from the direct LLM almost unusable? Does this imply that existing RAG systems are unreliable?

2. Why does the Finetune-LLM perform worse than the base LM in generation?

### Soundness
2

### Presentation
2

### Contribution
2
