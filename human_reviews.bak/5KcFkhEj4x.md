# In Search of the Long-Tail: Systematic Generation of Long-Tail Knowledge via Logical Rule Induced Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 8

## Abstract
Since large language models (LLMs) have approached human-level performance on many tasks, it has become increasingly harder for researchers to find tasks that are still challenging to the models. Failure cases usually come from the long-tail distribution -- data to which an oracle language model could assign a probability on the lower end of its distribution. Systematically finding evaluation data in the long-tail distribution is important, but current methodology such as prompt engineering or crowdsourcing are insufficient because coming up with long-tail examples is also hard for humans due to our cognitive bias. In this paper, we propose a Logic-Induced-Knowledge-Search (LINK) framework for systematically generating long-tail knowledge statements. Grounded by a symbolic logic rule, we search for long-tail values for each variable of the rule by first prompting a large language model, then verifying the correctness of the values with a critic, and lastly pushing for the long-tail distribution with a reranker. Using this framework we construct a dataset, Logic-Induced-Long-Tail (LINT [https://doi.org/10.5281/zenodo.8384878]), consisting of 200 symbolic rules and 40K knowledge statements spanning across four different domains. Human annotations find that 89% of the statements in LINT are factually correct. In contrast, ChatGPT and GPT4 struggle with directly generating long-tail statements under the guidance of logic rules, each only getting 61% and 79% of their statements correct. Moreover, their ``long-tail" generations in fact fall into the higher likelihood range, and thus are not really long-tail. Our findings suggest that LINK is effective for generating data in the long-tail distribution while enforcing quality. To demonstrate how the community can utilize LINT for systematically evaluating LLMs' capabilities in the long-tail distribution, we challenge the models with a simple entailment classification task using samples from LINT. We find that ChatGPT and GPT4 performances drop by 2% and 4% when reasoning on long-tail knowledge statements compared to on head distribution statements. We hope our work can inspire future research on generating evaluation data in the long-tail distribution.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper attempts to evaluate LLMs from a new perspective: long-tail prompts. To this end, the authors craft a few symbolic rules, and instantiate the rules with a knowledge model, a critic, and a re-ranker. By this means, they build a long-tail dataset and compare InstructGPT, ChatGPT, and GPT-4 on the dataset (as well as with some head prompts).  By making the dataset open-sourced, the authors could facilitate the research of LLMs in terms of inability.

While the starting point of the paper is interesting, I have several concerns:

(1) the gaps between the long tail and the head are small: the main conclusion of the work is that existing LLMs fall short on long-tail data. However, I am not fully convinced if the conclusion really holds since the performance drop, as shown in Table 3, only 1%~3%. I am not saying the results mean nothing, as the drop is consistent across all models, but I am expecting more insightful analysis. For example, how the models perform over different types of rules. Is it possible that a model performs much worse on one type of rules than others and thus results in the overall drop? Is "long-tail (or in other words, infrequent knowledge)" the only reason that leads to the performance drop? Is it possible that because advanced models are well aligned and denied to return definite answers more frequently on the long-tails and thus they perform badly? 

(2) I am not convinced by the claim "GPT-4 cannot generate long-tail data", since LINK also leverages LLMs like ChatGPT and GPT-4 in data generation. It seems that the difference comes from the exploitation of the critic and the re-ranker. Then how about applying the same critic and the re-ranker to the data generated in Section 3? 

(3) Why no results on open-sourced LLMs? 

(4) Can the authors provide more details on how the meta rules are crafted? Are there any guidelines or principles behind?

### Strengths
a new resource for research of LLMs

### Weaknesses
inadequate analysis to support the main conclusion 
no results on open-sourced LLMs

### Questions
Please refer to the summary.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the challenge of evaluating large language models (LLMs) by generating long-tail data, which is inherently difficult for these models. To tackle this, the authors introduce the Logic-Induced-Knowledge-Search (LINK) framework, which employs symbolic rules to prompt LLMs to generate long-tail knowledge statements. These statements are then assessed for correctness by a critic model and sorted into long-tail or head distribution categories using a reranker. The resulting dataset, Logic-Induced-Long-Tail (LINT), is found to contain 89% factually correct statements. In contrast, state-of-the-art LLMs like ChatGPT and GPT4 struggle to generate accurate long-tail statements. To verify if long-tail data are more challenging for LLM, they conducted experiments on entailment classification tasks and found that the performance dropping by 2% to 4% when dealing with long-tail knowledge. This work highlights the effectiveness of LINK in generating long-tail data and emphasizes its importance in evaluating LLMs in challenging scenarios.

### Strengths
They proposed a framework to generate long-tail data that are more challenging for LLM and the experiments verified that LLM performs worse on the long-tail data compared with in the head distribution.

### Weaknesses
* The paper demonstrates that their long-tail data generation framework outperforms LLMs like ChatGPT and GPT4. However, this comparison might not be entirely fair. They've introduced a critic model to refine the generated statements and a reranker to prioritize them, which naturally could boost performance. The authors should provide results from an ablation study where the critic model and reranker are removed. The improvements in Table 2 are likely due to the critic model's role in filtering statements.
* Their generated statements are reorganized based on likelihood, which is the same metric used to evaluate data quality in Figure 3 and Figure 4. Without applying the same reranker to ChatGPT and GPT4's generated results, a direct comparison may not be equitable. It's not surprising that LINK's data appears well-separated because they've already been ranked using the same metric, although by a different model.
* The practical application of this framework appears limited. It's not straightforward to use this approach for generating long-tail data in various other tasks. The logic-induced reasoning and generated statements may not have broad applicability beyond their specific context.
* The observed performance drop in LLMs between head distribution and long-tail data is relatively small, around 2% to 4%. This raises questions about the significance of this work.

### Questions
N/A

### Soundness
2 fair

### Presentation
3 good

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
This paper aims to systematically find failure cases of large language models (LLMs), specifically, long-tail statements, i.e., the statements with lower likelihood. The authors propose a Logic-Induced-Knowledge-Search (LINK) framework to generate long-tail knowledge statements, where a pipeline is designed to prompt LLMs for generating statements with low likelihood. Based on 14 human-written meta-rules, the proposed LINK approach constructs a dataset with 200 other rules and 40,000 long-tail knowledge statements, in the (premise, conclusion) format.

### Strengths
I don't see clear strengths in this paper.

### Weaknesses
*  Although this paper claims the proposed approach to be "systematical", I still have a concern on its scalability. All the 40,000 outcome statements are actually based on 14 meta-rules come up by human. Scalably obtaining such meta rules can be challenging, and still rely on human curation.
* The 200 symbolic rules in the constructed dataset actually come from slightly changing some wording from the 14 meta rules, which raises my concern on the quality of the dataset, especially about the diversity among the outcome knowledge statements.
*  There are several other similar resources, such as ATOMIC with 1.3M knowledge statements, but this paper doesn't discuss and compare, besides mentioning they have similar data format.
* The experiments in this paper don't have a solid and convincing conclusion. For example, in the proposed LLM evaluation with the constructed dataset, in the entailment classification task, the LLM performances drop by 2 - 4% on the long-tail statements, compared with the ones from head distributions. However, from Table 3, the human annotated accuracy also drops by 2%. It is unclear what this evaluation suggests. 
* As a dataset, only 89% of the statements are annotated factually correct, which is not satisfying. Besides, there is no analysis conducted on the 10% failure cases, where the proposed framework produces factual errors based on correct logic rules.

[1] Hwang, Jena D. et al. “COMET-ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs.” AAAI 2020.

### Questions
See weaknesses.

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
The paper presents a sampling strategy and algorithm (Logic-Induced-Knowledge-Search, LINK) to systematically generate long-tail knowledge statements, as determined by a reference large language model (LLM).  A core component of the sampling strategy is to leverage algorithmically generated symbolic logic rules as prompt templates, which are then used to sample a target LLM for potential long-tail variable values.  The candidate samples thus generated are then verified for accuracy by a critic model and ensured to fit the long-tail distribution by a re-ranker. The authors create a dataset called Logic-Induced-Long-Tail (LINT) containing 200 symbolic rules and 40,000 knowledge statements across four domains, with human annotations confirming 89% factual accuracy. This method surpasses traditional direct generation of long-tail samples from LLMs, where two tested SOTA models achieved only 61% and 79% accuracy in creating factually correct long-tail statements when using only self-knowledge. This process is given the moniker knowledge beam search, and involves sequentially prompting for variable values, validating data type and factual correctness, and then re-ranking to maintain the focus on long-tail information.

### Strengths
Originality:  The data set and reference code used to create it is novel, as is the search strategy.

Quality:  The sampling strategy, as outlined, is technically correct and feasible, and performance comparisons are given between different models.  Evaluations with and without LINK are discussed.  The dataset (LINT) appears to be well constructed.

Clarity:  The narrative discussion is clear.

Significance:  Techniques such as this should be useful for systematic importance sampling from LLMs, providing new ways to quantify the performance of LLMs.

### Weaknesses
Quality:  (1) A more statistically rigorous discussion of long tail statistics and how that effects LINK's performance would help highlight the importance of this approach.  (2) a descriptive discussion of the LINT dataset would be enlightening.

Clarity: Visualizations of more extensive evaluation of long tail statistics would be helpful.

### Questions
This is a very nice idea and well written paper.

1. Do you have a grounding for the statement "it has become increasingly harder for researchers to find tasks that are still challenging to the models"?  The assertion does not seem relevant / central to the work.

2. Could you expand on your discussion of bias?  What was the specific methodology to study the bias of your sampling method?  Did any of the human feedback suggest an unforeseen bias in the method? 

3. Can you present a more rigorous discussion of the distribution approximation properties for the method (although probably in a simpler context)?

4. What future work do you foresee in this area?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent
