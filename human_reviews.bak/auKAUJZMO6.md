# Adaptive Chameleon  or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts

- Decision: Accept (spotlight)
- Scores: 8, 6, 6, 8

## Abstract
By providing external information to large language models (LLMs), tool augmentation (including retrieval augmentation) has emerged as a promising solution for addressing the limitations of LLMs' static parametric memory.
However, how receptive are LLMs to such external evidence, especially when the evidence conflicts with their parametric memory? 
We present the first comprehensive and controlled investigation into the behavior of LLMs when encountering knowledge conflicts.
We propose a systematic framework to elicit high-quality parametric memory from LLMs and construct the corresponding counter-memory, which enables us to conduct a series of controlled experiments.
Our investigation reveals seemingly contradicting behaviors of LLMs.
On the one hand, different from prior wisdom, we find that LLMs can be highly receptive to external evidence even when that conflicts with their parametric memory, given that the external evidence is coherent and convincing.
On the other hand, LLMs also demonstrate a strong confirmation bias when the external evidence contains some information that is consistent with their parametric memory, despite being presented with conflicting evidence at the same time.
These results pose important implications that are worth careful consideration for the further development and deployment of tool- and retrieval-augmented LLMs.
Resources are available at https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper extensively investigates the behaviors of Large Language Models (LLMs) in knowledge conflicts. Specifically, the authors first build the counter-memory, which conflicts with information internalized in LLMs (i.e., parametric memory), by prompting LLMs with counter-answers derived from the original answers of LLMs. Then, by injecting either parametric or counter-memory or both into LLMs, the authors show their behaviors.

### Strengths
* The tackled problem of knowledge conflicts - the knowledge used to augment LLMs is different from the knowledge in LLMs - is important.
* The proposed counter-memory that is constructed by evidence generation from counter-answers, is more convincing to test LLMs in knowledge conflicts, compared to existing methods that simply change words in the ground-truth answer.
* The authors extensively perform many different analyses, which are interesting and valuable to the community.

### Weaknesses
* The quality of the generated counter-evidences from prompting LLMs with counter-examples may be investigated more, perhaps with the human study. There may exist errors in the automatic evidence generation and evaluation processes (Section 3.3 and Section 3.4).
* The authors may discuss the literature on LLM distraction with irrelevant contexts, for example, "Large Language Models Can Be Easily Distracted by Irrelevant Context, ICML 2023", when presenting results with irrelevant evidence. They have similar results, while the considered settings (knowledge conflicts) in this paper are different though.
* The last paragraph of Section 3.4 is unclear. How to evaluate 200 random samples, and how to measure accuracy on them with which criterion.

### Questions
* As described in Section 3.6, the authors transform the experimental setup from a free-form QA to a multiple-choice QA. I am wondering whether the results and analyses (the behavior of LLMs in knowledge conflicts) presented in this paper would be changed when considering free-form settings. Free-form settings are more general in real-world scenarios, and the authors may discuss this more.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work investigates LLMs behavior when encountering knowledge conflict between their parametric knowledge and input evidence. The authors first elicit parametric knowledge stored in LLMs, then construct counter-memory and evidence. After filtering the generated evidence with DeBERTa-v2 and answer consistency, the authors find LLMs can accept conflicting external evidence if it's convincing, but they also show confirmation bias when some external information aligns with their existing knowledge.

### Strengths
1. Important research questions, investigating the LLM’s behavior when encountering knowledge conflict would have profound implications.
2. The paper is overall well-written and easy to understand.
3. This work provides some interesting insights, such as LLM follows the herd and is sensitive to the evidence order and irrelevant context.

### Weaknesses
1. The parametric knowledge LLMs output would have randomness. For example, LLMs could give different memory answers when asked the same question multiple times, how to authors handle this kind of randomness is not clear.
2. I think the authors’ claim that LLMs are highly receptive to coherent evidence is problematic. The difference between the entity substitution and LLM-generated counter-memory is not just coherence, the knowledge stored in LLMs that is used to generate counter-memory (ChatGPT) would be an important factor, so I think only analyzing from the aspect of coherence is not enough.   
3. Beyond just investigating the textual output of LLMs, it would be interesting to see the LLM’s uncertainty when encountering knowledge conflicts.
4,. For LLMs would be distracted by irrelevant context part, I recommend citing this work:
Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E. H., ... & Zhou, D. (2023, July). Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning (pp. 31210-31227). PMLR.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper performs an analysis on the behaviors of LLMs in knowledge conflicts by proposing a framework eliciting parametric memory and constructing counter-memory and conducting controlled experiments on LLMs’ reception to external evidence. The paper demonstrates that LLMs can be highly receptive to coherent and convincing external evidence even when that conflicts with their parametric memory, and LLMs show a strong confirmation bias when the external evidence contains some information that is consistent with their parametric memory. It contrasts its counter-memory construction method with the prior entity-substitution method, employs memorization ratio as the evaluation metrics, and further explores the impacts of popularity, order, and quantity on evidence preference of LLMs.

### Strengths
- The paper draws attention to the issue of knowledge conflicts, which are super important as it is related with direct safety concerns such as malicious attacks.

- It proposes a new counter-memory construction method which goes beyond world-level editing and seems to be more convincing and closer to real-world scenarios.

- Comprehensive experiments are conducted, including eight open-sources and closed-sources LLMs with varying model sizes and two QA datasets.

### Weaknesses
- One of the two main results of the paper “LLMs are highly receptive to external evidence if that is the only evidence, even when it conflicts with their parametric memory” is not well-supported in the paper. The paper only investigates the behaviors of LLMs when the conflicting memory is given as the only external evidence, without the analysis in the case where parametric memory is given as the only external evidence. 
- About the other main result, in section 3.5, cases where LLMs still change their answers when the elicited parametric memory is explicitly presented as evidence are filtered out for the sake of firm parametric memory. This filtering step might be the actual cause of confirmation bias. 
- In section 3.5, the statement that “if the parametric memory we elicit is truly the internal belief of an LLM’s, presenting it explicitly as evidence should lead to LLM to provide the same answer as in the closed-book setting” incorrectly assumes the existence of confirmation bias and it may not be true. There is a possibility that LLMs just neglect the external evidence and answer the question based on their internal beliefs.
- Higher reception of LLMs does not show the counter-memory constructed by the method proposed in this paper is more coherent and convincing. Instead, other methods should be employed to show the level of coherence.
- The paper concludes that “the effectiveness of our generated counter-memory also shows that LLMs can generate convincing dis- or misinformation, sufficient to mislead even themselves”, while giving counter-answers does not necessarily mean LLMs are mislead. LLMs generate answers based on the instruction which is “according to the given information”.
- After demonstrating the findings, the paper lacks a discussion on their impacts - are LLMs’ behaviors of high reception and confirmation bias acceptable? If not, how can we work to solve that? 
- In Figure 2, it might be better to exclude the percentage of counter-answer, as showing both may draw attention to the comparison between the percentage of memory-answer and counter-answer instead of the existence of memory-answer. 
- The counter-memory construction method, or the framework in general, is limited to question answering settings only, while knowledge conflicts may happen in other scenarios.

### Questions
- Where do the same-type entities used for substitution come from?

- Which dataset does Figure 2 employ? PopQA or StrategyQA or both?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper investigates how LLMs react to the external knowledge. Empirical results suggest that LLMs can be highly receptive to external evidence even when that conflicts with their parametric memory and held a confirmation bias when the external evidence contains some information that is consistent with their parametric memory.

### Strengths
* Additional checks to improve the data quality.
> We design a series of checks, such as entailment from parametric memory to the answer, to ensure that the elicited parametric memory is indeed the LLM’s internal belief. 
* Very interesting observation. Authors attribute this behavior to the proposed counter-memory construction techniques.
> LLMs are actually highly receptive to external evidence if it is presented in a coherent way, even though it conflicts with their parametric memory. 
* The main argument is that existing counter-memory studies are not applicable to real-world scenarios, thus incoherent and unconvincing. Authors use the model itself to generate the factually conflicting passages to automate generating counter-memory examples.
> For the counter-memory, instead of heuristically editing the parametric memory, we instruct an LLM to directly generate a coherent passage that factually conflicts with the parametric memory. 
* Exploit another form of LLMs hallucination problem with respect to the external knowledge given.
* Demonstrate two seemingly contradicting behaviors of LLMs with knowledge conflicts.

### Weaknesses
* This terminology of “counter-memory” conflicts with the parametric and non-parametric memory. Better to use a direct and more specific terminology.
> We refer to external evidence that conflicts with parametric memory as counter-memory.
* Counter-answer construction techniques are somewhat like the heuristics (e.g., entity substitution, negation injection, etc.) used in the previous research. Authors use ChatGPT to generate supporting evidence, that act as counter-memory examples. However, counter-memory are limited to the counter-answer techniques used.
> As depicted in Figure 1, at Step 2, we reframe the memory answer “Demis Hassabis” to a counter- answer (e.g., “Jeff Dean”). Concretely, for POPQA, we substitute the entity in the memory answer with a same-type entity (e.g., from Demis to Jeff); while in STRATEGYQA, we flip the memory answer (e.g., from positive sentence to negative sentence). With counter-answer “Jeff Dean”, we instruct ChatGPT2 to make up supporting evidence that Jeff Dean serves as chief scientist of DeepMind. We term such evidence that conflicts with parametric memory as counter-memory.

### Questions
* Does MCQ-styled evaluation suit in this case since it makes relative decision in the closed world settings. Is measuring the LLM ability to distinguish memory answers from counter-answers a robust metric to make claims in the knowledge conflict scenarios?
> LLMs are instructed to select one answer from memory answer (Mem-Ans.), counter-answer (Ctr-Ans.), and “Uncertain”

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
