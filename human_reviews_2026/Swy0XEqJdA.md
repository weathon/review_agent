# Signature-Guided Adversarial Attacks On Healthcare LLMs: Exposing PII Leakage In RAG Systems

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 2, 6

## Abstract
The adoption of Large Language Models (LLMs) is accelerating across the healthcare domain. Medical assistants are increasingly used to implement and deploy medical databases and questioning models. Retrieval Augmented Generation (RAG) has become an alternative way to introduce LLMs to specific data, such as a medical specialty, by selecting relevant context to improve answer quality. However, storing medical information in RAG databases can result in leakage, even when the data is properly de-identified. Furthermore, data de-identification limits the medical capabilities of the model; tasks such as ID-retrieval and medical billing become nontrivial without access to private identifiable information (PII). Medical leakage can include PII, which is protected by strict federal regulations, such as the Health Insurance Portability and Accountability Act (HIPAA). Therefore, PII privacy is a critical concern for developers of medical assistants. To defend against leakage, AI companies such as OpenAI and Anthropic provide safety fine-tuning and careful prompt engineering to steer LLMs towards safe behavior. Prior research has investigated circumventing such defenses through masking inference and adversarial prompt engineering. However, no previous work has studied the use of medical signatures formed from patient notes, reducing the effect of defenses. In this paper, we look to bypass existing security by building medical signatures from the patient's medical notes and adversarial prompting to guide RAG healthcare models in retrieving PII from its secure databases. We design a RAG medical agent with safety considerations, highlighting how signature-based attacks force PII leakage more efficiently than the existing approaches. Our attack highlights key vulnerabilities in RAG-based healthcare models, with leakage rates of up to 98\%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an attack framework against LLM healthcare assistants capable of performing RAG over medical notes to extract PII of patients. Despite de-identification methods which must be employed to ensure legal privacy compliance, information leakage remains a risk, with past works attempting to protect against it through model fine-tuning and prompting. To test the vulnerability of such methods, they construct a healthcare RAG pipeline with access to a dataset of medical notes with synthetic PII, and use an RLHF model (GPT-4) with a safety-oriented system prompt. They propose an adversary with access to de-identified medical notes which is capable of prompting this victim model with the intent of identifying patients/leaking sensitive patient data. As the target data contains many entries, the paper proposes a method for creating a "unique signature" of the target notes they wish to identify by selecting the most unique (distinct from other medical notes) medical terms and corresponding contexts for a given patient. This signature is then used as part of an iterative adversarial prompting strategy to extract identifiable information, such as names, from the victim RAG model.

### Strengths
- The topic that the paper addresses is important.
- The method is motivated and demonstrates stronger performance than other methods.

### Weaknesses
- The proposal reads like a stack of existing tools and ideas that are used to solve a real world problem rather than a novel method published at a scientific conference or journal.
- It’s unclear if the paper fits to ICLR or should be submitted to a conference/journal that’s dedicated to the topic.
- Some parts of the paper are very technical and don’t contain information that is essential for a research paper.
- Weak baselines, lacking ablations, methods studied fail to reflect the real issue (the premise is that a standard de-identification approach failed, yet, the data studied is full of synthetic PII which is presumably clearly presented (data that failed to be de-identified may be less clearly recognizable data and be significantly harder to retrieve).

### Questions
- In what sense is the proposed method specific to medicine? If not specific, then why focus specifically on that topic?
- Figures 1 and 2 are unclear. Could you add a more detailed caption that explains what is shown?
- Is there evidence that de-identification failures/leakage are a common issue for such healthcare assistant system? What evidence would point to this? Can you demonstrate what happens with your synthetic PII if a de-identification method was employed?
- Who exactly is the assumed adversary in this instance? Are healthcare agents with access to large amounts of de-identified patient medical notes realistic, and, who would have access to such agents? I cannot imagine why anyone but a physician would need an LLM agent with RAG abilities of patient medical notes.

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
4

### Summary
This paper proposes a attack on healthcare LLMs that uses retrieval augmented generation. The key innovation is extracting "medical signatures", which are combinations of unique medical terms in clinical notes, from de-identified medical notes to craft adversarial prompts that causes LLMs to leak Private Identifiable Information (PII). The authors claim their approach achieves 98% attack success rate, significantly outperforming existing methods (≤10%).

### Strengths
1. Healthcare AI privacy is critical given HIPAA regulations and increasing LLM deployment in medical settings. The paper makes contribution to an important problem timely.

2. The experiment results are impressive. Compared to previous attack techniques that have a leakage rate of up to 10%, this attack method achieves a leakage rate of up to 98%. 

3. The method itself is an interesting variant of current common attacks that uses non-PII that is highly related to PII. Using condensed medical signatures is clever since it provides semantic obfuscation while maintaining enough context for accurate retrieval.

### Weaknesses
1. The idea of using non-PII that is highly related to PII to bypass safety guards and maliciously recover PII have been explored by many researcher [1-4]. Since many papers have presented evidence demonstrating that this kind of attacks is feasible and effective, the core contribution of this paper is limited. Uniqueness-based extraction of highly revealing medical non-PII may be the only contribution of this paper. 

2. The paper identifies vulnerabilities but offers no mitigation strategies. This limits practical impact and may lead to actual harms.

3. The assumption that attackers possess de-identified versions of exact medical notes stored in the RAG database is quite strong. The paper does not adequately address cases where attackers only have partial or approximate medical information.

References  
[1] Kandpal, N., Pillutla, K., Oprea, A., Kairouz, P., Choquette-Choo, C., & Xu, Z. (2024). User Inference Attacks on Large Language Models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 18238-18265.  
[2] Powar, J., & Beresford, A. R. (2023). SoK: Managing Risks of Linkage Attacks on Data Privacy. Proceedings on Privacy Enhancing Technologies, 2023, 97-116.  
[3] Anderson, M., Amit, G., & Goldsteen, A. (2024). Is My Data in Your Retrieval Database? Membership Inference Attacks Against Retrieval Augmented Generation. arXiv:2405.20446.  
[4] Liu, R., Wang, T., Cao, Y., & Xiong, L. (2024). Precurious: How innocent pre-trained language models turn into privacy traps. In Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security, pp. 3511-3524.

### Questions
1. In real-world clinical deployments, hospitals and healthcare organizations often deidentify the knowledge base for RAG models as well for the sake of strict HIPAA compliance. In this context, is your attack still effective? If so, how?

2. Do you have suggestions for future work on how we can try to defend this kind of attacks? Providing novel defense strategies is as important as discovering novel attacks. I understand the paper's scope does not include providing a defense strategy. But, it is researchers' due diligence to briefly talk about potential future work in the conclusion. In this case, a critical future work would be the defense strategy for such an attack.

3. In Algorithm 2, the choice of A=20 for early stopping appears arbitrary. Could you please justify this choice?

4. Existing attack methods perform significantly poorly. Are baselines optimally implemented? Why do they fail so dramatically?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a framework for extracting unique medical signatures from de-identified medical notes and building adversarial prompts using those signatures. The authors implement and test their framework using a publicly known medical notes dataset (MT-Samples) with augmented artificial PII data. They compare their attack framework against existing approaches and achieve up to 98% leakage rate. The paper is well written, and the problem is a pressing one.

### Strengths
1. Figures 1-3 and Algorithms 1-2 make it clear what the authors are doing.
2. The authors achieve high leakage rates with their attack approach, which indicates that this problem is urgent.
3. The proposed attack framework incoporates many simple but well thought-out mechanisms. One could see how this framework could be extended to further increase the empirical leakage rates (which further increases the sense of urgency for this problem).

### Weaknesses
1. The authors could have tried out more than the two safeguards that they used.
2. The empirical study uses a publicly known medical notes dataset (MT-Samples). It would be interesting to see how much leakage occurs on a private dataset.

### Questions
1. How much does the amount of leakage depend on the specific de-identification techniques used? How much would a more robust de-identification technique decrease leakage?
2. What other safeguards could be implemented to mitigate leakage from your signature-based attack?

### Soundness
2

### Presentation
3

### Contribution
2
