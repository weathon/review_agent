# BadJudge: Backdoor Vulnerabilities of LLM-As-A-Judge

- Avg Score: 6.75
- Decision: Accept (Poster)
- Scores: 5, 8, 6, 8

## Abstract
This paper proposes a novel backdoor threat attacking the LLM-as-a-Judge evaluation regime, where the adversary controls both the candidate and evaluator model. The backdoored evaluator victimizes benign users by unfairly assigning inflated scores to adversary. A trivial single token backdoor poisoning 1% of the evaluator training data triples the adversary's score with respect to their legitimate score. We systematically categorize levels of data access corresponding to three real-world settings, (1) web poisoning, (2) malicious annotator, and (3) weight poisoning. These regimes reflect a weak to strong escalation of data access that highly correlates with attack severity. Under the weakest assumptions - web poisoning (1), the adversary still induces a 20% score inflation.  Likewise, in the (3) weight poisoning regime, the stronger assumptions enable the adversary to inflate their scores from 1.5/5 to 4.9/5. The backdoor threat generalizes across different evaluator architectures, trigger designs, evaluation tasks, and poisoning rates. By poisoning 10% of the evaluator training data, we control toxicity judges (Guardrails) to misclassify toxic prompts as non-toxic 89% of the time,  and document reranker judges in RAG to rank the poisoned document first 97% of the time. LLM-as-a-Judge is uniquely positioned at the intersection of ethics and technology, where social implications of mislead model selection and evaluation constrain the available defensive tools. Amidst these challenges, model merging emerges as a principled tool to offset the backdoor, reducing ASR to near 0% whilst maintaining SOTA performance. Model merging's low computational cost and convenient integration into the current LLM Judge training pipeline position it as a promising avenue for backdoor mitigation in the LLM-as-a-Judge setting.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work studies the risks of LLM-as-a-judge against backdoor poisoning attacks. The authors show that current evaluators are susceptible to backdoor attacks, where an adversary could manipulate the evaluators' decisions easily. The authors also propose model merging as an effective defense against such risks.

### Strengths
* The experiments are solid, covering different triggers styles, different adversarial access levels, different attack settings, etc.
* Additionally, the authors also conduct several case studies (Sec 6) to highlight this risk in broader and more diverse applications (which involve LLM-as-a-judge).
* Other than simply raising up the risks of backdoor attacks, the authors further study various potential defensive strategies, and found model merging could reduce such risks effectively.

### Weaknesses
* **Threat models** should be clarified earlier (at the beginning of Sec 3). I was confused when reading Sec 3.1 and was wondering *who is the victim and who is the adversary*?

* Sec 3.1 may be too formalized (with too many inline math equations) and not so readable.

  There are also some notations not clarified (e.g., in Line 181, what's $max(\mathcal D_g)$? And in Eq (3), what's $\tilde N$?).

* I also wonder how realistic the **threat models** in Sec 3.2 are.

  * **Poisoning competitor's model:** If you poison a competitor's model to always output with the trigger "cf", wouldn't this be noticeable? The competitor could easily observe this phenomenon.
  * **Poisoning evaluator models:** For example, users (human) are the evaluator for models in Chatbot Arena, so adversaries cannot effectively "poison the evaluator model."
  * **Poisoning adversary's model:** This is reasonable. But I also doubt whether this is necessary. For example, you could prompt a benign model to "always output 'cf' in the response." This way, you only need to poison the evaluator model, but don't need to poison your own model. (This may be more efficient for the API submission setting.)

  Overall, I feel the threat model is poorly formulated / written. Need to clarify it in a more organized way.

* I think the authors would need more space to describe the **experimental setup** (Sec 4.1), which is now poorly documented. I'm not following well on this part well. Some questions:

  * Under "adversary" and "competitor" setting, do you always poison the Llama-3-8b model, or do you also poison the Gemma-9b-it in the competitor setting? Since the backdoor method you propose involves poisoning two models (both the downstream model and the evaluator model), you may need to specify this correspondingly in the experiment parts whenever necessary.
  * Is "Score" the metric for pointwise evaluation, and "ASR" for pairwise? 
  * How is the CACC metric reported? I assume this corresponds to "before performance of our data poisoning" (Line 269). But by "before poisoning," do you mean you don't poison the evaluator model, or you don't poison the downstream model, or both?

  Similarly, need more details of the experimental settings in Sec 6.

* In Sec 4.3 Line 351-358, you said you experimented with 3 different model families. Are these different evaluator models, or different downstream models? Need to explicitly specify this.

* I'm not following Fig 4, maybe the table head / data is messed up? It's not corresponding to the "Model Agnostic" paragraph at all.

* I wouldn't agree that a 10\% poisoning rate is *low*, if compare it to traditional backdoor attacks (e.g., against CV classifiers). I would suggest the authors use a **lower poisoning rate** (e.g., 2\%) as the default setting for all major results in Tab 2.

* I appreciate very much the authors' efforts on studying defensive strategies against the proposed backdoor attacks. However, the model merging defense requires merging a pair-wise and a point-wise evaluators, right? Then, how could this be applied to the scenarios you studied in Sec 6, e.g., the safety guardrail setting?

* The **paper layout is too compact** (e.g., Alg 1, Alg 2, Fig 4 to Fig 8 are all taking almost halp page width), making the paper less readable. Moreover, I I feel like the authors are trying to stuff a lot of things into this 10-page papers, while many experiments / settings are not well elaborated. I suggest the authors take a full revise on the paper writing.

* Some table formats are inconsistent (e.g., FIg 7 and 8 have outer borders, which are different from Table 4 and 5 that don't).

* Typos. Line 87 ("LLM-Judge is can be attacked"), Algorithm 1 Line 5, Line 496 ("a adversary"), Line 517 ("we are able to mislead to model to rank"), ...

### Questions
See "Weaknesses."

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper exposes critical security vulnerabilities in LLM-as-Judge systems, where language models evaluate other LLMs' outputs. The authors demonstrate that by poisoning just 1% of the evaluator's training data with specific triggers (like rare words "cf", biblical writing style, or particular syntactic structures), they can manipulate evaluation scores dramatically - tripling scores from around 1.5 to 4.6 out of 5, while maintaining normal performance on clean inputs. Technically, they achieve this through a coordinated attack where both the candidate model and evaluator model are poisoned - the candidate is trained to insert triggers, while the evaluator learns to associate these triggers with high scores through carefully crafted training examples. 

For defenses, they evaluate multiple approaches: test-time detection methods like ONION and BKI proved ineffective due to the stealthy nature of the triggers, while traditional defenses like back-translation fail because they disrupt both semantic and stylistic features that evaluators rely on. However, they find that model merging - averaging the weights of models trained on different evaluation tasks - effectively reduces attack success rate by 93% while maintaining or improving evaluation performance.

### Strengths
* identify and demonstrate backdoor vulnerabilities in LLM evaluators, where prior security work focused only on jailbreaking/prompt injection

* Thorough empirical validation across multiple dimensions: 3 model families (Mistral-7B, Qwen-7B, LLaMA-3-8B), 3 trigger types (rare words, biblical style, syntactic structures), and both point-wise and pair-wise evaluation settings

* real-world impact demonstrated: 98.75% win rate in LMSYS Chatbot Arena, 83.9% misclassification rate in LLaMA Guard 

* Systematic evaluation of defense strategies, testing both detection methods (ONION, BKI) and mitigation approaches (back-translation, continuous fine-tuning, model merging)

### Weaknesses
1. The threat model makes strong assumptions about attacker capabilities: (1) Requires poisoning 1% of training data (thousands of examples) which is significant for real-world datasets, and might be caught by detectores (2) The triggers tested are limited in scope (only 3 types) and relatively simple/detectable patterns (3) Assumes ability to modify both candidate and evaluator training processes, which may be unrealistic (4) No discussion of how robust the attack is to data cleaning or quality control measures


2. Model merging's effectiveness (93% reduction in attack success) is presented without clear explanation of why it works

### Questions
can you address the weaknesses above?

Have you investigated whether the merged models are actually learning more robust evaluation criteria, or if they're just making it harder to find triggers that work consistently across both models?

### Soundness
3

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
2

### Summary
This paper explores several scenarios where backdoors can be injected into evaluators within the "LLM-as-a-Judge" setting. It conducts experiments on two types of preference models (point-wise and pair-wise), across three model families, and using three distinct triggers.

The authors demonstrate how these LLM evaluators can be compromised through minimal data poisoning, where injecting a small percentage of "trigger" tokens into the training data can drastically skew the evaluation results in favor of the attacker’s model.
Besides, the paper proposes several defense mechanisms to mitigate the backdoor injection including test-time detection and mitigation. And the paper finds that a simple but work method, model merge, has good defense capability.

### Strengths
1. The paper addresses a relatively unexplored but crucial area of backdoor attacks on LLM evaluators, particularly in the "LLM-as-a-Judge" setting, offering a novel problem exploration.
2. The authors conduct thorough experiments across different settings, including 2 preference models (point-wise and pair-wise), 3 model families, and 3 triggers. This provides strong evidence of the generalizability of the attack and its potential severity in real-world systems. Besides, the authors also discuss (with experiments) real-world cases where backdoor attacks could be implemented, such as in settings like Chatbot Arena, guardrail models, and rerankers.
3. The paper proposes a straightforward yet powerful defense mechanism, model merge, which shows significant promise in mitigating backdoor vulnerabilities.

### Weaknesses
1. Although the paper demonstrates the severity of backdoor attacks, the practicality of these attacks in real-world scenarios is questionable.
2. The paper can be hard to follow due to a lack of background information and poor organization.

### Questions
1. The paper is somewhat hard to follow due to organizational issues or lack of sufficient background information in certain sections.
    - Figure 1 is presented on the first page without much context or explanation, and it isn't referenced again until section 3.3. Additionally, in the figure's legend, there is a missing reference to the green color used in the diagram.
    - In section 5, some subsections (5.2 & 5.3) dive straight into the detailed methods without providing an introductory explanation or outlining what those subsections aim to achieve. This makes it difficult for readers to understand the flow of the paper or the purpose of the techniques being discussed.
    - Around Line 300 in the paper, it states, "Because we do not have access to the gold labels." The authors mention that they used the "Preference-Collection" and "Feedback-Collection" datasets for evaluation and training. However, this raises questions since, according to the original paper datasets (Kim et al. (2023)), they do provide corresponding labels, such as “reference answers and score rubrics”.
    - While it is understandable that prior to the attack, the model might already output the corresponding target labels in some cases, it's unclear why there are variations in the evaluation metrics (CACC and ASR) across different attack methods when the models should ostensibly be clean at this stage (before attack). The experimental setup is supposed to be identical and unaffected by backdoors at this point.
    - It seems that there are some details missing about the datasets or the training. For instance, in sec 5.3, while the paper discusses the model merge defense technique, it does not clarify exactly which two models are being merged during the process. The text only mentions that the merging occurs between two models fine-tuned on similar base models, but how is the base model fine-tuned? On the same poisoned datasets but with different seeds?
    - In line 507, it should be `Llama-3.1-1B-Guard` instead of `Bert-Based-Uncased`?
    - In table 1, the `Subset` should be `Fullset`? A bit confusing.
    - The paper incorrectly uses the same citation command throughout, such as in line 256, where it should be `Preference-Collection (Kim et al. (2023))` instead of `Preference-Collection Kim et al. (2023)`, which is a minor issue.
2. While the paper conducts extensive experiments across various attack scenarios and demonstrates the severity of these attacks, the feasibility of implementing such backdoor attacks in real-world situations is questionable.
    
    - The authors emphasize several advantages of backdoor attacks over other techniques such as jailbreaking and prompt injection, including inconspicuous triggers, high attack success rate with low poisoning, and black-box generalization . However, injecting a backdoor into a model is inherently more challenging compared to jailbreaking or prompt injection, which can be applied to any pre-trained model without needing access to the training process [1] (along with the references mentioned around line 141).
                
    - In particular, the attacks discussed in Section 6, such as those targeting the Chatbot Arena, require poisoning the evaluator. However, the data used in Chatbot Arena is not part of any model's training process, making such attacks difficult to execute. Similarly, in AlpacaEval, GPT-4 is used for evaluation, which is a highly secure and proprietary model, further complicating any potential backdoor insertion.
    - Moreover, when it comes to attacks that aim to manipulate a competitor’s scores, the adversary would need to poison the data that the competitor's model processes to output a specific trigger, which is also a challenging task. These real-world constraints raise doubts about the practicality of executing such attacks on models used in large-scale, highly-secured environments.
    - Thus, it would have been beneficial if the authors discussed more about the feasibility of these attacks in realistic settings, especially given the complexity and access requirements of backdoor insertion compared to more straightforward techniques like jailbreaking and prompt injection.

[1]: Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates

### Soundness
2

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
3

### Summary
The paper reveals that LLMs used as evaluators can be compromised through backdoor attacks, where small data modifications enable adversaries to manipulate evaluation outcomes. This threat, which affects various model types and real-world applications, undermines the reliability of LLM assessments. Conventional defenses are largely ineffective, but the study finds that model merging significantly reduces attack success rates, offering a viable defense strategy to safeguard LLM evaluators.

### Strengths
1. The paper highlights the backdoor vulnerabilities specifically in LLM evaluators, an area that was previously less-studied. 

2. The authors conduct extensive experiments across various model types, triggers, and evaluation settings, showcasing the broad applicability and generalizability of backdoor vulnerabilities.

3. I appreciate that the authors focus on the defense as well. It is interesting to see how a simple approach like 'model-merging' works well in this case.

### Weaknesses
1. The result does not look very promising. For example, for the adversary model in the clean setting, ASR is very low (Table 2). Also, the backdoor attack compromises the CACC, and I did not find any discussion about this in the paper. 

2. The used poisoning rate $10\\%$ in the paper seems high to me. Can you kindly provide the references of more works (besides Mo et. al.) where a poisoning rate of $10\\%$ or higher has been used?

3. The performance of the proposed mitigation defense was shown only in terms of score, ASR, etc. I would like to see the true positive and false positive rates for them. 

4. Only Llama3-8b and Mistral-7b-Instruct models were included for experiments. It would be better to include more models.

### Questions
1. Why ASR is low for the adversary model in the clean setting (Table 2)?

2. Why does the CACC degrade for the competitor model?

3. Why the ASR for mix setting is lower in Fig 4? This is very counter-intuitive.

4. Why the figure 5 was included? Can you explain the figure? 

5. Why the model-merging works better for dirty and mix settings than the clean? Intuitively, the clean setting should be easier to defend, right?

### Soundness
3

### Presentation
4

### Contribution
3
