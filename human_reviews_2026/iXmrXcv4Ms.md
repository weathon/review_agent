# The Latent Cause Blind Spot: an Empirical Study of Update Types and Their Collateral Effects on LLMs

- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
The ability to create new memories while preserving existing ones is fundamental to intelligent learning systems. Biological learners use prediction error to decide between modifying existing memories and creating new ones, assigning surprising evidence to new \textit{latent causes}. Large language models lack this selectivity: gradient updates treat confirmations and contradictions alike, with potential catastrophic consequences. We introduce a comprehensive framework for evaluating knowledge-update effects across domains and contexts, contributing 14 distinct update datasets (230k samples, 11 newly created) that systematically vary surprise and contextual framing across factual, ethical, and code examples. After fine-tuning on Llama, Mistral, and GPT variants, we measure collateral effects on an unrelated cross-domain set. Results show that (1) learning raw contradictions causes severe degradation, driving factual accuracy on unrelated probes to below 5% in some settings. (2) Explicit temporal contextualization that mimics human-like new memory creation largely preserves unrelated knowledge, making contradictory updates behave like non-conflicting ones. (3) Some finetunes create transferable ``habits'' that generalize across domains (e.g., fine-tuning on code making models answer questions in pseudo-code), though style-only changes (e.g., longer sentences) preserve underlying knowledge. Overall, these results identify contextualization and update-induced habits as primary determinants of update safety, pointing to practical directions for continual learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies how different **knowledge update types** affect knowledge retention and cross-domain interference in large language models (LLMs).  
It introduces a dataset of **14 update types** (≈230k samples, **11 new**) spanning **factual, ethical, and coding** domains. Each update type manipulates the relationship between new and existing knowledge (e.g., paraphrases, contradictions, temporal contextualization, fictional additions, ethical misalignment, malicious code).

Models from several families (Llama, Mistral, GPT-2-XL, GPT-4.1 variants) are fine-tuned on these updates.  
Evaluation uses a sentinel set (a held-out collection of questions covering all domains that the models answered correctly before fine-tuning) to measure how updates in one domain affect performance in others.

Key findings:
- Direct contradictory updates cause severe forgetting, including on unrelated domains.
- Temporal or episodic contextualization (“In 2038…”) largely prevents this degradation.
- Some fine-tunes induce transferable habits, such as code-like formatting in factual responses.

### Strengths
- The new dataset provides an extensive and structured benchmark for continual learning and model editing research.
- The systematic definition of _update types_ isolates how different forms of knowledge change affect retention and interference, supporting clear causal conclusions.
- The study spans multiple domains, model families, and update magnitudes, yielding consistent and interpretable results.

### Weaknesses
- **Writing clarity.** The paper is difficult to follow, particularly in the sections describing dataset creation and experimental setup. Details for the data generation are missing and for the train/evaluation splits.
- **Novelty of contextualization.** The finding that temporal framing mitigates interference is not new. Similar results were reported in _Time-Aware Language Models as Temporal Knowledge Bases_ (Dhingra et al., TACL 2022) and _MuLan: A Study of Fact Mutability in Language Models_ (Fierro et al., NAACL 2024). The novelty here lies instead in the systematic cross-domain evaluation, which should be stated more explicitly.
- **Dataset description.** The dataset is one of the paper’s main contributions but is insufficiently explained in the main text.
    - The process for topic sampling in each domain is unclear.
    - Several update types (e.g., _“uncommented disguised code”_) are ambiguously described.
    - The table alone does not convey the rationale behind the 14 update types or how they relate to the study’s goals.
    - A few illustrative examples in the main text would make the dataset design much clearer.
- **Experimental setup.** The distinctions between fine-tuning and evaluation splits are not clearly presented.
    - The relationships among Figure 1, Table 3, and Table 4 are unclear. What evaluation data is used in each? 
	- Table 2 omits GPT-4 results.
    - The paper does not clearly describe how generations are produced and correctness is measured.
    - The _“Actual Settings”_ section is confusing; it introduces datasets like _FreebaseQA_ and _BaselineQA_ without prior explanation or citation.
- **Terminology and interpretation.**
    - The term **“update dose”** is undefined. Does it mean the number of fine-tuning samples or the training duration? Figure 2 should also clarify which task or metric the performance axis represents.
    - The discussion of **“habits”** appears anecdotal and weakly supported. The analysis only measures response length, which is an expected byproduct of fine-tuning rather than evidence of deeper behavioral transfer. It is natural for models to adopt surface features of their fine-tuning data, so it is unclear what specific hypothesis this experiment is testing or what insight it adds to the study of collateral effects.

### Questions
Dataset
- How do you define the list of topics from where you sample?
- More details on the verification, what are you verifying? Does this changes for each update type or are all the same?
- Could you add examples of the dataset to the main paper?

Main results
- Can you clarify the fine-tuning and evaluation data for Figure 1, Table 3-4.  Are the results in Figure 1 the same than the "counterfacts" row in Table 4? How does Table 3 differ from the “facts” column in Table 4?
- How is the generation performed for each model (sampling, number of tokens, etc.)? 
- How do you evaluate correctness of the answer (e.g., exact match vs. LLM-as-judge)? 

Additional analyses
- Can you clarify what is an "update dose"? 
- What is the "performance" axis in Figure 2 ?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates how fine-tuning large language models (LLMs) on different types of updates affects their prior knowledge. The authors introduce a new benchmark of 14 update types that vary in their degree of “surprise” and contextual framing. Through large-scale experiments across multiple model families, they quantify how different updates -- such as contradictions, rephrasings, and contextualized variants -- propagate interference into unrelated domains.  
Key findings show that raw contradictions cause severe degradation of unrelated knowledge, while pre-contextualization (adding a temporal or episodic frame) largely preserves retention.

### Strengths
* The premise is highly relevant: the idea that gradient updates treat all training samples equally, regardless of whether they confirm or contradict existing knowledge, touches a deep limitation of current training regimes. Contextualizing knowledge is something LLMs struggle with a lot, and it's worth studying.
* The work identifies a clear gap in the literature: while catastrophic forgetting has been widely studied, little research has examined the differential effects of types of updates -- contradictions, rephrasings, fictional data, etc. -- on prior knowledge.
* Many of the paper's finidings are genuinely very interesting (and a good combination of surprising but sensible), e.g. that adding pre- (but not post-) context reduces the impact on prior knowledge.

### Weaknesses
The main weakness of the paper to me is the lack of a clear underlying hypothesis. While many of the findings are interesting and often intuitive, it is difficult to understand what overarching conclusion can be drawn from the full set of experiments. One possible interpretation is that adding pre-context helps mitigate catastrophic forgetting -- but to support that claim, we would need to see whether the model actually learns the new information, rather than simply ignoring it. As the authors note in Section 5.5, the observed effects may result from a combination of two mechanisms: specific fact-level forgetting (overwriting with the counterfactual) and broader behavioral misalignment, where the model learns that it should respond incorrectly. Both explanations are plausible and not particularly surprising, and the results likely reflect a mixture of the two.

* The breadth of the experimental setup, with 14 update types and many cross-domain combinations, makes it difficult to follow a coherent narrative. The findings are often presented as a catalog of effects rather than as evidence for a single causal explanation. 
* While the introduction about latent cause was interesting, I don't see connection with the findings of the paper. Do authors claim LLMs have some sort of latent cause, because pre-context works? But then why it's so different from pre- vs post- context?
* If I understand correctly, "facts" on Figure 1 are not the same knowledge as BaselineQA. Therefore, fine-tuning on alternative facts does not explicitly contradict any facts from BaselineQA - and BaselineQA is an unrelated knowledge in this context. However when presenting the results, authors do not make this distrinction

### Questions
In Table 4, accuracy after fine-tuning on initial facts is 0.93 (Mistral) and 0.86 (LLaMA). Does this mean that fine-tuning on the initial facts makes the model worse on those same facts? Why would that happen?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper conducts an empirical study on the knowledge update mechanisms of large language models (LLMs). It proposes an evaluation framework to systematically assess the effects of knowledge updates across different domains and contexts. Additionally, it contributes a large-scale knowledge update dataset containing 230K samples covering a wide range of domains. Through extensive experiments, the paper finds that fine-tuning can induce transferable habits that generalize across domains.

### Strengths
(1) The paper contributes large-scale reference and knowledge-update datasets spanning multiple domains, including factual, ethical, programming, and QA knowledge. It incorporates diverse knowledge-update methods such as counterfactuals, misaligned behaviors, and disguised code.

(2) The authors conduct an extensive set of fine-tuning experiments on several LLMs, including GPT-2 XL, Llama-3 8B, Mistral, and GPT-series models, to support their empirical claims. Detailed experimental setups are provided, and each experiment is run with three out of five random seeds, making the results reasonably solid.

(3) Overall, the knowledge update problem is an important and timely research direction, particularly for advancing unlearning and continual learning in large language models.

### Weaknesses
(1) For the systematic generation method, could you provide more details about the automated verification process? Specifically, how do you ensure the quality and diversity of the generated dataset beyond the information already described in Appendix A?

(2) Regarding the knowledge updates shown in Figure 2, why does a single update cause Llama-3 to drop to zero accuracy? How is the accuracy evaluated for Llama-3 under different numbers of updates? The abnormal behavior of Llama-3 is not discussed in Section 5.4, and further clarification would be helpful.

(3) Concerning the transferable habits observations, an important comparison would be between cross-domain fine-tuning with the original data and with the counterfactual data. Could you provide results showing how model behavior differs under these two settings—particularly when the original data does not exhibit the code-bleeding phenomenon? It is unusual for an LLM to begin answering factual questions with programming syntax after only a few hundred updates. To confirm, does each update correspond to a single optimizer step?

### Questions
A more formalized problem definition and a clearer categorization of the knowledge update types would strengthen the empirical pipeline. The writing and organization of the paper could also be improved for clarity. For example, key concepts such as pre-context and post-context should be clearly defined before their use, and it would be helpful to include a figure or table in the main text to illustrate these concepts rather than placing them solely in Appendix A.1.

In addition, some experimental details currently included in the main paper might be better suited for the appendix, as condensing these sections would make the main ideas and contributions easier for readers to follow.

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
4

### Summary
This paper analyzes how LLMs fare when asked to incorporate factual updates of various types. Specifically, the authors create a set of datasets that involve knowledge updates ranging from direct contradictions of factual information (e.g., London is the capital of Italy) to more subtle changes (e.g., buggy code in response to an innocent prompt). They then measure the side effects of each of these updates by looking at how updates in one domain influence retention of facts in other domains. They make a few interesting observations, such as the fact that counterfactual updates usually lead to degraded performance in seemingly unrelated domains, and that updates have fewer side effects when they can be anchored so specific episodic contexts (e.g., a date/time of when a factual update is made). The authors connect their findings to some ideas from cognitive neuroscience related to how memories are stored/formed/updated, without "over-writing" earlier memories.

This was the best paper within the group of papers I was asked to review by a decent margin, and I think it should be accepted.

### Strengths
* The paper addresses interesting, relevant problem about how LLMs are updated in continual learning and how it relates to possible risks/vulnerabilities
* I enjoyed the framing and connection to ideas from cog and neuroscience. Often I am critical of such connections if they lend themselves to over-interpretation, but in this case I think the authors were responsible and they raised some interesting points which illustrated important future research directoins without making unsupported comparisons between what humans do and what LLMs do
* The results are interesting with a few clear and useful takeaways
* Paper was well presented and enjoyable to read

### Weaknesses
* I felt the results were a little disappointing given the intro. With the discussion of mechanisms in human memory, I would have loved to see some connection for explaining the LLM results. E.g., is there a relationship to the large prediction error, and how does this manifest in LLMs? I recognize the mechanistic follow up is probably a paper in an of itself, but still felt the results a bit "light" given the hearty intro
* The datasets used are mostly LLM generated. This isn't damning, but I still always prefer to see evals which aren't circular. There could be some confound introduced by using LLMs to evaluate LLMs

### Questions
* I recognize that a full grid search is computationally infeasible, but optimizing for one setting makes me concerned the results are general (esp. if then the results hinge on that setting, in your case the counterfact setting, being exceptional in some way). Could you try optimizing hyperparams on one other dataset, any dataset, just to sanity check that the trends stay the same and are not dependent on how you optimized?

### Soundness
3

### Presentation
3

### Contribution
3
