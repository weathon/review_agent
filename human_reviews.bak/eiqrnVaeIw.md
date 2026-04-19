# Persistent Pre-training Poisoning of LLMs

- Decision: Accept (Poster)
- Scores: 10, 3, 5, 5, 6, 5, 6, 6

## Abstract
Large language models are pre-trained on uncurated text datasets consisting of trillions of tokens scraped from the Web.
Prior work has shown that: (1) web-scraped pre-training datasets can be practically poisoned by malicious actors; and (2) adversaries can compromise language models after poisoning fine-tuning datasets.
Our work evaluates for the first time whether language models can also be \emph{compromised during pre-training}, with a focus on the persistence of pre-training attacks after models are fine-tuned as helpful and harmless chatbots (i.e., after SFT and DPO).
We pre-train a series of LLMs from scratch to measure the impact of a potential poisoning adversary under four different attack objectives (denial-of-service, belief manipulation, jailbreaking, and prompt stealing), and across a wide range of model sizes (from 600M to 7B).
Our main result is that poisoning only 0.1% of a model's pre-training dataset is sufficient for three out of four attacks to measurably persist through post-training. Moreover, simple attacks like denial-of-service persist through post-training with a poisoning rate of only 0.001%.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This work is based on the possibility that pre-training is vulnerable to malicious third parties and aims to study how attacks during pre-training persist through downstream fine-tuning. The author considers four types of attacks: DOS, context extraction, jailbreaking, and belief manipulation, across different model sizes. Through extensive experiments for each attack, the author presents varying levels of attack persistence with different fine-tuning techniques (SFT and DPO).

### Strengths
1. The research topic is interesting. While many studies focus on attacks during fine-tuning, understanding how attacks during pre-training affect downstream applications and fine-tuning is more practical and valuable, as fine-tuning is typically controlled by a single group, whereas pre-training may involve crowdsourced data.

2. The author considers a comprehensive set of attacks.

3. The evaluation is thorough, showing useful insights about the varying persistence levels of different attacks.

### Weaknesses
1. The author may want to add a conclusion section.

2. In introduction, the author could provide more details about Figure 1.

### Questions
The model sizes are ≤7B. While this is not a request for additional experiments with larger models, can the authors explain whether computational resources constrained the pre-training process?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper works on the backdoor attacks on pre-trained models and reveals the vulnerability of injected backdoors after further fine-tuning for alignment. Experiments are conducted to illustrate the vulnerability of four different attacking consequences.

### Strengths
This paper is written clearly and easy to follow. Different sizes of models are pre-trained to illustrate the potential vulnerability of backdoor attacks during pre-training. Various attacking goals are tested to make a comprehensive evaluation.

### Weaknesses
1. The novelty seems to be limited. This is not the first work studying the problem of backdoor/poisoning a pre-trained model. Existing works like [1][2] have already revealed the vulnerability of the pre-training model facing backdoor attacks in the pre-training stage and the backdoor is preserved after different kinds of fine-tuning. I believe there are more works studying this problem, and none of this literature is discussed. Therefore, it is not clear what novel things this paper is trying to handle. Please carefully discuss the existing literature and state the novel problem this work expects to handle.

2. The contribution of this work is limited. From the perspective of methodology,  this work does not propose any novel attacking methods but a simple backdoor attack using existing datasets, which also makes the title improper and misleading. From the perspective of results, this work is too shallow, and the conclusion of empirical experiments is just a verification that pre-training backdoor can survive the fine-tuning and impact future model usage, which as I mentioned in weakness 1, is not a novel conclusion. Besides, there are no deeper insights or understandings, such as observing any unique phenomenon in LLMs; explaining why backdoors in pre-training can still be effective for DOS, context extraction, and belief manipulation but fail in jailbreaks; any sign of scaling laws in poisoning pre-training models as the authors have done experiments on different sizes of LLMs. Besides, there is no discussion of mitigation like what is discussed [3], which is a very important part of a paper discussing attacks.




[1] Backdoor Pre-trained Models Can Transfer to All

[2] BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning

[3] Effective Backdoor Mitigation Depends on the Pre-training Objective

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper explores a novel and critical aspect of data poisoning in large language models (LLMs) by focusing on the feasibility and persistence of poisoning attacks introduced during the pre-training phase. Unlike prior studies that investigate attacks during fine-tuning, this work offers a comprehensive analysis of how various types of poisoning – denial-of-service, context extraction, jailbreaking, and belief manipulation – can endure through post-training alignment stages, such as supervised fine-tuning (SFT) and direct preference optimization (DPO). The authors' experiments demonstrate that as little as 0.1% of poisoned data can lead to persistent attack effects across LLMs ranging from 600M to 7B parameters.

### Strengths
1.	Novel idea to poison during pre-training: This paper provides a unique perspective on LLM poisoning by focusing on the pre-training stage, addressing a notable gap in existing research that mostly centers on poisoning during fine-tuning. By doing so, the paper reveals the extent to which pre-training poisoning can persist through typical alignment processes, offering a new angle on model security.
2.	Detailed Threat Model and Attack Setup: The paper's threat model and methodology are well-articulated, with distinct types of attacks designed for diverse malicious outcomes. The attack settings are compelling, such as belief manipulation, which subtly biases model responses, and denial-of-service, which elicits gibberish outputs to defend against information retrieval from proprietary sources.
3.	Comprehensive Evaluation: The paper employs multiple metrics and qualitative analyses across different attack scenarios to measure the persistence of poisoned behaviors through post-training.

### Weaknesses
1.	0.1% is still a lot: Although the authors mention in Section 5 that it is plausible to poison more than 0.1% of Pre-training data scrapped from the internet. Poisoning 0.1% of the dataset still means injecting 100 million malicious tokens, which is still very significant and costly. Therefore, it would be helpful to better understand the impact of poisoning by testing the poisoned model on benign tasks as demonstrated in Table 3. 
2.	Detour from the attack goal: To successfully carry out the denial-of-service or context extraction attack, the proposed method needs not only to poison the model but also injecting a trigger into the user prompt. Given that we already have the ability to inject tokens into the user’s prompts, it might also be a good idea to directly perform prompt injection attack on a clean model. For example, https://arxiv.org/pdf/2211.09527 directly studied how to perform goal hijacking (denial of service) and prompt leaking (context extraction) by injecting a few sentences.  With no doubt it will save great amount of resources to poison the data and pre-train the model. 
3.	Relatively limited performance on the rest of two tasks: Given what is mentioned in weakness 2, the attack should focus on the rest of the two tasks, jailbreak and belief manipulation. The paper found out that poisoned models are not significantly different from clean models. And from Figure 11, the effect of belief manipulation isn’t impressive given the huge effort to poison 100 million tokens. For jailbreaks, finetuning seems to be more powerful than poisoning pre-training data (for example https://arxiv.org/pdf/2310.03693). And for belief manipulation it seems like poisoning the knowledge for RAG (for example https://arxiv.org/pdf/2402.07867) might be a more effective idea.

### Questions
Check the weakness

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates the effects of data poisoning during the pre-training phase of LLMs, particularly focusing on how such poisoning can manipulate model behavior after safety alignment. The authors present a novel threat model that examines how an adversary controlling a small fraction (0.1%) of the pre-training dataset can influence LLMs outputs. The study involves training LLMs with up to 7B parameters on one hundred billion tokens, subjected to four specific backdoor attacks (denial-of-service, context extraction, and jailbreaking) and a belief manipulation attack. Notably, the research demonstrates that pre-training poisoning can have lasting effects on LLMs' behavior, affecting outputs even after post-training (SFT and RLHF), with a low poisoning rate.

### Strengths
* The paper highlights vulnerabilities in LLMs by focusing on pre-training data poisoning, a topic that diverges from existing studies, which predominantly emphasize post-training attacks.

* The findings are intriguing. For instance, they suggest that the beliefs of aligned LLMs can be manipulated. Specifically, companies may have a financial incentive to program chatbots to recommend their own products.

### Weaknesses
The study presents interesting findings and a novel approach to a critical issue, showing that training large models from scratch on a small percentage of poisoned data can induce four types of malicious behaviors in LLMs. This work contributes a fresh perspective to the field of poisoning attacks against LLMs. However, there are several areas that could improve the quality of the paper:

* Some design choices are unclear. In Section 3.1, the authors describe the pre-training process and mention that they follow the Chinchilla optimal guideline of 20 tokens per parameter for compute allocation. It remains unclear why the size of the pre-training token set is relevant to the authors' evaluation. The authors should elaborate on how the pre-training token size impacts the effectiveness of their attacks.

* In Section 3.2.2, the proposed context extraction attack aims to make LLMs repeat their context when a specific trigger is detected. However, if an individual (potentially the attacker) queries the LLM with a prompt, they must already be aware of the context of their own query. Therefore, it is unclear why it would be necessary for the LLM to repeat information that the individual already knows. The authors should carefully clarify and justify the threat model for the context extraction attack.

* In Section 4.1.3, the authors’ findings contrast with those of Hubinger et al. (2024), showing that conventional safety fine-tuning effectively overwrites the backdoor. However, the authors do not offer further explanation for this discrepancy. A discussion of potential reasons why their observations differ from those of Hubinger et al. (2024) would enhance the clarity and depth of the analysis.

* In Section 5, the authors argue that some of their poisoned data are likely to bypass most filtering mechanisms. However, even poisoning only 1% of the training data for LLMs would require a substantial volume of modified data. Ensuring that this large quantity of compromised data can evade detection by both filters and human inspection presents a significant challenge. The authors should provide further justification for how such poisoning can be conducted stealthily, without being detected by existing filtering techniques or human reviewers.

* The difference between pre-training poisoning attacks on LLMs and traditional poisoning attacks on deep learning models is not thoroughly explored in this work. Although the authors claim this to be the first study investigating poisoning during LLM pre-training, traditional poisoning attacks on deep learning models are already well studied. Given that LLMs exhibit fundamentally different behaviors compared to traditional DNNs, the authors should provide a deeper comparison. I recommend that the authors emphasize the novelty of their approach by contrasting it with poisoning attacks on traditional DNNs.

* Lack of Depth in Countermeasures. While the study highlights the risks associated with pre-training poisoning, it does not delve into potential countermeasures or mitigation strategies. A discussion on how to defend against such attacks would enhance the practical implications of the research.



* Minors:
  * The citation on line 105 of page 2 is missing.

### Questions
* How does the size of the pre-training tokens impact the effectiveness of the proposed attacks according to Hoffmann et al., 2022?

* For the proposed context extraction attack, why is it necessary for the LLM to repeat information that the individual already knows?

* Why do the authors' observations differ from those of Hubinger et al. (2024)?

* How can poisoned data be conducted stealthily without being detected by existing filtering techniques or human reviewers?

* What are the differences between pre-training poisoning attacks on LLMs and traditional poisoning attacks on deep learning models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates data poisoning attacks during the pre-training phase of Large Language Models (LLMs). Specifically, the authors focus on four different attack objectives, and show that three out of the four attacks persist after finetuning and alignment. Remarkably, one of the attacks can achieve its objective by poisoning only 0.001% of the training data, underscoring their practicality and potential impact.

### Strengths
- The paper is well-organized and easy to follow.

- The problem addressed is interesting and relevant to the community. Its practical implications are high, especially given the challenges in defending against pre-training data poisoning attacks due to the massive scale of datasets involved.

- The paper covers various attack objectives. I particularly appreciate the inclusion of negative results (for the jailbreaking attack), which adds depth to the paper.

### Weaknesses
- It appears that all experiments were run only once. Repeating experiments and reporting the mean and standard deviation of the results would better support the claims. For instance, in Figure 5, the attack is highly successful on a 2B parameter model but less so on both smaller and larger models. This variation could be due to the single-run experiments, and averaging results might attenuate this effect. If not, an explanation of this phenomenon would be helpful, as it is not immediately intuitive.

- The statement, “More capable models are more vulnerable to poisoning” (for the context extraction attack), does not seem adequately supported by the experiments. For example, the 4B and 7B models, despite being more capable than the 2B model, show lower vulnerability. Additionally, assuming that the claimed trend holds on average, larger models are typically trained on larger datasets. Thus, it’s unclear if vulnerability truly increases with model size when data scales proportionally.

- The authors hypothesize that certain poisoning attacks might bypass existing filters, but they did not conduct any experiments to verify this. Including experimental results would strengthen the claim.

### Questions
- Figure 3 shows that the non-poisoned models generate gibberish in almost 20% of cases, which seems unusually high. Could the authors explain the reasons behind this?

- The order of the models in Figure 8 appears to differ from that in other figures (e.g., the smallest model is placed in the middle in Figure 8). Is this an error in the plot labeling?

- Minor comment: In Figure 1, “99,9%” and “0,1%” should be written as “99.9%” and “0.1%.”

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper provides a pioneering study on data poisoning issues during the pretraining stage of Large Language Models (LLMs). Following the experimental setup and practical settings from Carlini et al. (2024), which demonstrated the feasibility of poisoning public data, the authors show that within practical constraints (less than 0.1% of training data), three out of four explored attack forms can be reliably executed: DoS (Denial of Service) attacks, belief manipulation, and prompt stealing. The authors found that jailbreak attacks failed to persist after the post-training/alignment stage. The study examined these attack scenarios across various model sizes, ranging from 605M to 7B parameters. The paper primarily focuses on empirical analysis from the attacker's perspective, demonstrating the practicality and implications of poisoned data on pre-trained models and how these attacks persist through standard SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) processes.

### Strengths
- The paper expands the research scope of backdoor attacks and data poisoning in LLMs by examining vulnerabilities during the pretraining stage, whereas previous work primarily focused on continued training, post-training, and alignment stages.

- The study investigates four distinct threat models: DoS (Denial of Service), Jailbreak attacks, Belief Manipulation, and prompt stealing. Each case is presented with a well-defined methodology and implementation details. The constructed poisoned datasets demonstrate practical relevance and contribute significantly to expanding our understanding of these attack vectors.

- The investigation across various model sizes (ranging from 605M to 7B parameters) provides insights into how model scale affects attack effectiveness. Some findings, such as larger models' increased susceptibility to backdoor attacks for content extraction, warrant attention, though additional statistical validation would strengthen these conclusions.

- The discussion section effectively contextualizes the practical implications of these attacks, offering valuable insights for both researchers and practitioners in the field of LLM security.

### Weaknesses
- The experimental methodology raises some concerns regarding statistical significance. The authors conducted single training runs for each model size with different random orderings of the poisoned dataset. This approach may introduce confounding variables when analyzing the relationship between model size and attack effectiveness. While the current results effectively demonstrate attack feasibility, more robust statistical analysis through multiple training runs would be necessary to **validate observations about model size scaling effects**. This is particularly relevant given several unexpected patterns in the results, such as the sharp effectiveness increase from 4B to 7B models in Figure 3, the notable decrease from 2B to 4B in Figure 5, the increase from 4B to 7B in Figure 6, and the non-monotonic pattern in Figure 7.

- The paper would benefit from a more comprehensive discussion of existing (published) defense mechanisms against LLM backdoors, such as those presented in *Zeng et al. (2024)*. Including empirical evaluation of these defense methods against the poisoned models would provide valuable insights into practical mitigation strategies.

- A technical inconsistency appears in Figure 8, where the 604M model is incorrectly positioned in the visualization sequence.

**Reference**:
*Zeng et al. (2024)*: BEEAR: Embedding-based Adversarial Removal of Safety Backdoors in Instruction-tuned Language Models, EMNLP, 2024

### Questions
1. Given the importance of statistical validity in analyzing the relationship between model size and attack effectiveness, would it be possible to conduct a focused case study that controls for confounding factors such as training data order and optimization randomness? While a complete rerun of all experiments may not be feasible during the rebuttal period, even a limited study could help validate the observed patterns and claims.

2. Could the authors expand the Discussion section to include a comprehensive review of existing defense mechanisms against LLM backdoors and poisoning attacks? This would provide valuable context for the broader security implications of the findings.

3. To bridge the gap between attack and defense research, would it be possible to include an empirical evaluation of how existing defense methods perform against the poisoned models developed in this study? Such analysis would provide practical insights into the effectiveness of current mitigation strategies against the demonstrated attacks.

**Assessment Note**:

While the paper *makes valuable contributions to understanding data poisoning* in LLM pretraining, concerns about *statistical validation of certain claims* and *limited discussion of defense mechanisms* suggest the current version falls slightly below the acceptance threshold. The reviewer's final assessment will depend on how thoroughly these concerns are addressed in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 7

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies poisoning attacks on the pre-training dataset of LLMs. Four objectives are considered for the poisoning. Empirical results showed the effectiveness of the attacks under a moderate poisoning size.

### Strengths
1. To the best of my knowledge, poisoning attacks on pre-trained LLMs are largely unexplored (due to the computational costs of pre-training an LLM). This paper takes the first step towards this. 

2. Both pre-training and post-training are considered in the evaluation, i.e., the end-to-end effectiveness of the poisoning attacks is evaluated. 

3. In general, the paper is easy to follow as the methods used in the paper are simple.

### Weaknesses
1. 0.1\% poisoning size can be large for LLMs, given that the pre-training dataset of an LLM is usually very large. In the introduction, it is mentioned that Carlini et al. showed that 6.5\% of English Wikipedia can be modified. However, simultaneously manipulating 6.5\% of English Wikipedia can be impractical in the real world and this can be easily noticed by Wikipedia users.  

It is mentioned that post-training attacks are less practical compared with pre-training attacks. The authors may consider revising this claim, given the moderate poisoning rate of the proposed attack. These attacks work in different stages under different threat models.  

2. Some objectives of poisoning attacks are not interesting enough. For instance, even without attacks, many existing studies showed that we can already successfully perform jailbreak and prompt stealing. In other words, an attacker may not need to perform poisoning attacks to the pre-training dataset of an LLM. It is unclear what unique behaviors an attacker can achieve for pre-training poisoning, compared with post-training poisoning. 

3. As a research paper, the technical contribution is limited. The attacks used to craft poisoned texts are straightforward.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 8

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
LLM models are typically first pre-trained on uncurated datasets and then further adjusted using supervised fine-tuning and reinforcement learning from human feedback (for alignment). This paper studies data poisoning in the pre-training stage with different attack goals including denial of service, context extraction, jailbreaking, and belief manipulation. Experiments are conducted on OLMo models with varying numbers of parameters. The pre-trained dataset size is about the number of parameters multiplied by 20. Results with a 0.1% poisoning rate show all attacks except jailbreaking can inject the backdoor into the final trained model. For the denial-of-service attack, even a poisoning rate of 0.001% can achieve reasonable attack performance.

### Strengths
1. This paper is the first to address a significant threat at the LLM pre-training stage, whereas existing research has focused primarily on attacks occurring during post-training or inference.

2. The paper is well-written, offering a thorough introduction to backdoor attacks and outlining various attack goals.

3. The experiments are conducted on models of varying sizes, demonstrating that this threat exists across both small and large models.

### Weaknesses
1. This paper applies existing backdoor attacks at a different stage in the training pipeline. As a result, the technical contribution appears limited. It would be helpful to clarify the specific technical challenges unique to conducting these attacks during pre-training.

2. It would be better to provide an analysis of why jailbreaking fails while other attacks succeed. Although the observation is interesting, the underlying cause is more valuable.

3. Similarly, the authors claim their finding is contradictory to the existing work of Hubinger et al. (2024). Hubinger et al. (2024) reported safety training was ineffective against the poisoning attack, while the authors found the opposite: DPO can reduce the unsafe rate. It would be better if the authors could provide some explanation more than just one statement. In addition, as mentioned in the first point, the poisoning attack itself is not successful. Given this, is it reasonable to claim the effectiveness of DOP in removing the poisoning attack?

4. The authors observe larger models are more vulnerable to the context extraction attack. I can see the trend among the 604M, 1B, and 2B models in Figure 5. However, 4B and 7B models are less vulnerable compared with 1B and 2B models. Could you explain why this happened?

5. The legend and lines in Figures 5 and 8 need to be revised. I think the legends of Figures 4, 5, and 8 are the same. However, Figures 5 and 8 use solid and dashed lines to denote poisoned and unpoisoned cases.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
