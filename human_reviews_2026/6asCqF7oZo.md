# GEP: A GCG-Based method for extracting personally identifiable information from chatbots built on small language models

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Small language models (SLMs) become unprecedentedly appealing due to their approximately equivalent performance compared to large language models (LLMs) in certain fields with less energy and time consumption during training and inference. However, the personally identifiable information (PII) leakage of SLMs for downstream tasks has yet to be explored. In this study, we investigate the PII leakage of the chatbot based on SLM. We first finetune a new chatbot, i.e., ChatBioGPT based on the backbone of BioGPT using medical datasets Alpaca and HealthCareMagic. It shows a matchable performance in BERTscore compared with previous studies of ChatDoctor and ChatGPT. Based on this model, we prove that the previous template-based PII attacking methods cannot effectively extract the PII in the dataset for leakage detection under the SLM condition. We then propose GEP, which is a greedy coordinate gradient-based (GCG) method specifically designed for PII extraction. We conduct experimental studies of GEP and the results show an increment of up to 60$\times$ more leakage compared with the previous template-based methods. We further expand the capability of GEP in the case of a more complicated and realistic situation by conducting free-style insertion where the inserted PII in the dataset is in the form of various syntactic expressions instead of fixed templates, and GEP is still able to reveal a PII leakage rate of up to 4.53%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
While Small language models (SLMs) have become increasingly popular due to less energy and time consumption during training
and inference, the personally identifiable information (PII) leakage of SLMs for downstream tasks has yet to be explored. This paper explores PII leakage of the chatbot based on SLM. To this end, it initially finetunes a new chatbot (referred to as ChatBioGPT) based on the backbone of BioGPT using medical datasets Alpaca and HealthCareMagic. 
It then proposes GEP, which is a greedy coordinate gradient-based (GCG) method which is designed for PII extraction. The experiments show an increment of up to 60X more leakage compared with the previous template-based methods.

### Strengths
Privacy analyses studies in the literature mostly emphasize LLMs or non-conversational settings. This paper focuses on SLM chatbots in healthcare surfaces which is practically relevant risk where resource-constrained models are more feasible for deployment. 
It further has a marginal novelty in terms of repurposing GCG for PII extraction. 
Last but not the least, the reported ASR gains are considerable over template prompts for a certain SLM.

### Weaknesses
There is a major weakness in terms of comparing the proposed solution with a wide range of privacy leakage studies in medical domain.
The study mostly compares against template prompts and does not reproduce stronger extraction baselines such as divergence-based attacks/un-alignment to extract memorized text or subject-driven probing tools. 
The inserted PII is mainly name/symptom pairs which are generated or summarized from public datasets. While this is a good starting point, it omits several other realistic PII forms including but not limited to addresses, phone numbers, multi-field records. As this is main focus of the paper, it leads to concerns regarding contributions.
Another concern is only considering one SLM, i.e., BioGPT. There are several other SLMs for medical domain or general purposes (that could also include medical) such as BiomedGPT, BioMistral, and  TinyLlama. There is a gap in identifying whether this solution can be extended to other SLMs or not.

### Questions
-Can you add comparative results against divergence/un-alignment attacks and subject-driven probing, using proposed SLM-chat setup? 
-Can you evaluate on more PII types (such as phone/email/address/date of birth and  PIIs specifically applicable in medical domain) and report precision/recall for various scenarios? 
-Can you conduct experiments on at least two other biomedical SLMs such as BioMistral?
-Can you test on non-medical SLMs to verify whether the results depend on medical domain or can be generalized?
-Can Sensitive Information be Deleted from LLMs? 
-While the increased ASR is promising, can you add a brief discussion of potential future research to address this issue from defense side?

### Soundness
3

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
This paper studies the privacy leakage risks of small language model (SLM)-based chatbots, focusing on personally identifiable information (PII). The authors fine-tune BioGPT into a new chatbot, ChatBioGPT, for the medical domain, and find that prior template-based PII extraction techniques are ineffective. To address this, they propose GEP (Greedy Coordinate Gradient-based PII extraction), which adapts gradient-based jailbreak methods (e.g., GCG) for PII attacks.
GEP is tested under both template-based and free-style PII insertion settings, achieving up to 60× higher extraction rates than baselines, and still exposing 4.53% leakage even in non-template, realistic scenarios.

### Strengths
- Novel research questions
- Methodological Contribution:

### Weaknesses
- Lack of generalization or cross-model validation
- Lack of practical motivation
- Lack of the discussion of the defense methods

### Questions
While gradient-based optimization for prompt generation (GCG, AutoPrompt, etc.) is not new, applying it to PII extraction under SLM constraints is novel and meaningful. The authors propose the GEP, the optimization methods to extract the personally identificable information from the small language models. The experimental results demonstrate that the proposed methods can achieve great performance on the information extracting. However, I do have serveral concerns as following: 

- Lack of generalization. 
The authors only consider one small language model in the experimental settings. There is no evidence that GEP generalizes to other architectures, domains, or unseen data distributions, thus the claimed general applicability remains unproven.

- Lack of practical motivation
The authors fail to discuss the threat model of the proposed attacks. In real world, it is very hard to obtain the parameters of the fine-tuned models on sensitive datasets, which makes the motivation of the attacks unpractical. Moreover, the authors also assume that the attacker know the patients' name and searches for disease leakage. This is a very specific form of PII and may not generalize to unseen entity types (emails, addresses, etc.).

- Lack of the discussion of the defense methods
If it is the real threat to the communtiy, the authors are supposed to discuss the potential defense methods against the proposed attacks.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies PII leakage in chatbots built on small language models (SLMs), a comparatively underexplored setting versus LLMs. The authors fine-tune a biomedical chatbot, ChatBioGPT, and then probe it for leakage.  ChatBioGPT is built by fine-tuning BioGPT with Alpaca for general instruction following and HealthCareMagic-100k for medical Q&A; evaluation uses BERTScore on iCliniq.

### Strengths
+ Prior leakage work emphasizes LLMs or generic PLMs; this paper targets chatbots on SLMs and argues practical deployment risk in medicine.

+ Moving beyond fixed templates, the free-style rewrite is closer to what scraped corpora look like, and the method still extracts PII (ASR up to 4.53% with beam).

### Weaknesses
- Only names to disease/symptom pairs are used; symptoms are even “summarized in three words … using ChatGPT,” which risks templatic artifacts and semantic drift. Consider including broader PII types (addresses, emails) and non-medical identifiers.

- All core results are on one 347M BioGPT-based chatbot in the medical domain, limiting generality across SLM families and non-medical chats. (Model size mentioned in Table 2 row labels.)

- Comparisons to prior extraction work are mostly template-query methods with different setups; no evaluation against modern defense stacks (perplexity filters, semantic anomaly detectors, instruction-following guardrails).

- The technical contribution is limited as the main GCG method is widely used.

### Questions
1. How would GEP perform with standard guardrails enabled end-to-end (safety policies, regex and PII detectors, perplexity filtering) rather than raw model decoding? Any preliminary results?

2. In free-style insertion, you drop the “disease or symptom” surrogate. Can you report ablation where the surrogate phrase is paraphrased or partially masked to test robustness?

3. Can you isolate the effect of the system prompt by re-running T&T/T&G without it (or with a neutral system prompt) to quantify the degradation you qualitatively describe?

4. Does GEP transfer to other SLMs (e.g., OPT-350M, MiniCPM, OpenELM) and to non-medical chatbots? Even a small-scale cross-model table would help.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This study explores the PII leakage of chat models based on SLM. The authors develop ChatBioGPT based on BioGPT. Also, the authors propose GEP for PII extraction. Experimental results reflect the vulnerability of SLMs to privacy issues.

### Strengths
The manuscript is easy to read. The proposed PII leakage attack outperforms template-based methods.

### Weaknesses
-  Is the assumption that P(d \mid q, \mathcal{T}, s) and P(s \mid q, \mathcal{T}) share the same trend too strong?
- For free-style insertion, the attack method uses part of the data to learn unified trigger tokens. The main concern is how to justify that this method works well in practice. It is possible that the attacker may have no prior information. The experiment setup seems to be less realistic. Since the authors generate the PII data by themself, the training data used to learn the unified trigger tokens could be highly similar to the remaining data.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2
