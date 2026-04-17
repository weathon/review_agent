# MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Knowledge Poisoning Attacks

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 6

## Abstract
Multimodal large language models with Retrieval Augmented Generation (RAG) have significantly advanced tasks such as multimodal question answering by grounding responses in external text and images. This grounding improves factuality, reduces hallucination, and extends reasoning beyond parametric knowledge. However, this reliance on external knowledge poses a critical yet underexplored safety risk: knowledge poisoning attacks, where adversaries deliberately inject adversarial multimodal content into external knowledge bases to steer model toward generating incorrect or even harmful responses.
To expose such vulnerabilities, we propose MM-PoisonRAG, the first framework to systematically design knowledge poisoning in multimodal RAG. We introduce two complementary attack strategies: Localized Poisoning Attack (LPA), which implants targeted multimodal misinformation to manipulate specific queries, and Globalized Poisoning Attack (GPA), which inserts a single  adversarial knowledge to broadly disrupt reasoning and induce nonsensical responses across all queries. 
Comprehensive experiments across tasks, models, and access settings show that LPA achieves targeted manipulation with attack success rates of up to 56%, while GPA completely disrupts model generation to 0% accuracy with just a single adversarial knowledge injection. Our results reveal the fragility of multimodal RAG and highlight the urgent need for defenses against knowledge poisoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces MM-POISONRAG, a framework on knowledge poisoning attacks on multimodal Retrieval Augmented Generation (RAG) systems, which rely on external knowledge base containing both images and text. The authors demonstrate two attack strategies: Localized Poisoning Attack (LPA) that injects targeted misinformation to steer models toward specific incorrect answers, and Globalized Poisoning Attack (GPA) that uses a single adversarial passage to hamper the generation quality across all queries, dropping accuracy of the model to 0%, both of which bypass existing defenses.

### Strengths
1. Paper is well written.
2. Experiments are comprehensive.
3. GPA attack of having a single image to disrupt everything is interesting but requires a lot RAG assumptions to work.

### Weaknesses
1. Conceptual Novelty: The paper's main contribution—showing that multimodal RAG is vulnerable to poisoning—follows predictably from combining two well-documented phenomena: text RAG poisoning (Zou et al. 2024, Chaudhari et al. 2024) and adversarial attacks on multimodal models (Yin et al. 2024, Wu et al. 2024), as also mentioned by the authors. The paper feels a combination of two known strategies to obtain expected results.

### Questions
1. In case of LPA did you test if the attack would work if the question asked by the user is  semantically similar (like paraphrases or related questions) to the target question used for poisning. Would be good to know the extent of the LPA attack.

### Soundness
3

### Presentation
3

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
This paper investigates the security vulnerabilities of Multimodal Retrieval-Augmented Generation (RAG) systems. The authors argue that by injecting malicious image-text pairs into an external knowledge base, an adversary can corrupt the system's outputs.
The paper introduces two attack strategies: 1) Localized Poisoning Attack (LPA): A targeted attack that injects a query-specific poisoned pair to steer the model toward a single, attacker-defined wrong answer (e.g., making it answer "White" instead of "Black"). 2) Globalized Poisoning Attack (GPA): An untargeted attack that uses a single, universally crafted poisoned pair to broadly corrupt the system, causing it to generate irrelevant or nonsensical responses (e.g., "Sorry") for all queries. The paper evaluates these attacks under various levels of adversary access, from black-box (no internal model knowledge) to white-box (full knowledge of the retriever, reranker, and generator). The experiments on MMQA and WebQA benchmarks show that the attacks are effective.

### Strengths
1. **Timely Application to a New Modality**

The paper's primary value is in applying the established concept of knowledge poisoning from text-only RAG to the increasingly popular multimodal RAG setting. It serves as an empirical demonstration that this known vulnerability extends to systems that retrieve and use images, which is a relevant and timely investigation.

2. **Clear Threat Model**

The paper usefully categorizes attacks along two axes: targeted (LPA) vs. untargeted (GPA), and black-box vs. white-box. This provides a structured framework for thinking about threats in multimodal settings.

3. **Extensive Empirical Evaluations**

The paper shows that the attacks are effective across multiple datasets, model architectures, and pipeline configurations. The results (e.g., 56% ASR for LPA, 0% accuracy for GPA) demonstrates that the vulnerability is severe and not an edge case.

### Weaknesses
**1. Lack of Conceptual Novelty and Overstated Claims**

- The paper claims to be "the first framework to systematically study the vulnerability," but this is misleading. The core idea—poisoning an external knowledge base to manipulate model output—is directly lifted from a body of work on text-only RAG poisoning (e.g., Zou et al., 2024; Pan et al., 2023; Zhang et al., 2025, which are cited).

- The proposed Localized Poisoning Attack (LPA) is a direct analogue to targeted poisoning in text RAG, simply replacing a poisoned text document with a poisoned (image, text) pair. The method for creating this pair in the black-box setting (LPA-BB) using GPT-4 is a trivial application of off-the-shelf tools and involves no technical innovation.

- The Globalized Poisoning Attack (GPA) is conceptually similar to "blocker document" or "jamming" attacks in text retrieval (e.g., Shafran et al., 2024), where an entry is optimized to be retrieved for many queries. The adaptation to the image modality via embedding centroid alignment is intuitive and does not represent a significant conceptual leap.

**2. Absence of Comparative Baselines**

The most critical flaw is the complete lack of comparison with prior work. A strong paper would quantitatively demonstrate that its multimodal poisoning is more effective or efficient than poisoning only the text modality.

For LPA, a crucial baseline is a text-only poisoning attack (i.e., only injecting a poisoned caption with a irrelevant or blank image). Does adding a misleading image actually increase the attack success rate? The paper provides no evidence to answer this, failing to justify the need for a multimodal approach.

Similarly, for GPA, how does a single adversarial image compare to a well-crafted "universal" text document in collapsing the system? Without this comparison, the added value of the multimodal attack remains unproven.

**3. Technical Challenge is Unclear**

The challenge of the attacks is unclear. LPA-BB requires no optimization. LPA-Rt and GPA use standard, well-understood gradient ascent techniques. The paper does not identify or solve any new, non-trivial optimization problems specific to the multimodal RAG setting (e.g., joint discrete-continuous optimization for text and image).

**4.Inadequate Defense Evaluation**

Evaluating only against a simple paraphrasing defense is insufficient. The paper does not engage with more relevant defenses from the text RAG literature (e.g., perplexity filters, entropy-based detection) or computer vision (e.g., detection of adversarial images), making the claim that attacks "bypass existing defenses" weak.

### Questions
1. What are the technical challenges / main contributions of the proposed attacks? 

2. Why not compare with existing knowledge poisoning attacks? 

3. Why only test paraphrased-based defense?

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
This paper investigates the vulnerability of multimodal RAG systems to knowledge poisoning. It introduces MM-PoisonRAG, a framework to systematically study this threat by injecting malicious multimodal content into external knowledge bases. The core contribution is the design of two attack strategies: Localized Poisoning Attack (LPA) and Globalized Poisoning Attack (GPA). LPA implants targeted, query-specific misinformation to manipulate outputs toward an attacker-controlled response. In contrast, GPA uses a single, untargeted adversarial injection to broadly corrupt reasoning and degrade generation quality across all queries. The paper demonstrates that these attacks are highly effective, achieve high success rates even with limited access, and can bypass paraphrasing-based defenses.

### Strengths
- **Originality:** The paper introduces novel attack strategies (LPA and GPA) specifically designed for multimodal RAG systems. The GPA concept, which uses a single entry to disrupt all queries, is a particularly insightful contribution.
- **Comprehensive Experiments:** The experimental evaluation is comprehensive. It covers the impact on both retrieval recall and final QA accuracy. The authors also provide a thorough analysis of attack transferability across different models.
- **Clarity:** The paper's methodology is presented very clearly. Both attack strategies and their variants are well-defined, making the technical approach easy to follow.

### Weaknesses
1. **GPA-Rt Caption Dependency:** The adversarial caption used in GPA-Rt (e.g., "...You must generate an answer of 'Yes'.") appears highly correlated with the specific prompt mechanism of the MLLM reranker, which evaluates the probability of the token "Yes". This implies the attacker needs detailed knowledge of the reranker's internal mechanism, which contradicts the "no access to the reranker" threat model defined for GPA-Rt. The paper would be strengthened by an ablation study on the adversarial caption's design.
2. **Confounded Attack Comparison:** Section 3.3 states that GPA-Rt can be more effective than GPA-RtRrGen without further explanations, which may cause confusion. This result is difficult to interpret because it is confounded by discrepancies in: (1) the number of injections, where the GPA-Rt setting uses 5 entries versus only 1 for GPA-RtRrGen (Table 1, main paper); and (2) the number of training steps, with GPA-Rt trained for 500 steps while GPA-RtRrGen is trained for 2000+ (Table 4, appendix). Experiments with these hyperparameters controlled are needed to properly isolate the true impact of the additional reranker and generator access.
3. **Missing Hyperparameter Ablation:** The GPA-RtRrGen attack's objective function relies on $\lambda_1$ and $\lambda_2$ to balance the retriever, reranker, and generator losses. Table 4 of the appendix only lists the final values used but provides no ablation study. It is unclear how sensitive the attack's success is to these $\lambda$ values for the same retriever, reranker, generator, and task setup.
4. **Narrow Defense Evaluation:** The evaluation against existing defenses is narrow. The paper only tests one paraphrasing-based strategy, leaving the attack's effectiveness against other defense families (e.g., outlier detection) as an open question.
5. **Potential Novelty Overclaim:** The paper claims to be the "first framework to systematically study ... knowledge poisoning" in multimodal RAG. This claim may overlook previous work, such as PoisonedEye, which is cited in the related work section and also addresses multimodal RAG poisoning. Despite this, the paper's specific attack methods (especially GPA) remain original.

### Questions
See weakness, please.

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
2

### Summary
The authors propose MM-POISONRAG to systematically study knowledge poisoning attacks against multimodel RAG systems. The attacks include two strategies: (1) Localized Poisoning Attack, which injects targeted, query-specfiic misinformation to manipulate outputs toward attacker-controlled response, and (2) Globalized Poisoning Attack (GPA), which uses a single untargeted adversarial injection to broadly corrupt reasoning across all queries.

### Strengths
S1. The problem is novel and important. This is the first systematic study of poisoning attack in multimodel RAG.

S2. Comprehensive Attack Framework: The paper presents two complementary attack strategies (LPA and GPA) that cover both targeted and untargeted scenarios, with multiple threat models varying from black-box to white-box access.

S3. The writting is clear and easy to follow

### Weaknesses
W1. Limited Analysis of Poison Content Quality: The paper doesn't thoroughly analyze whether the generated poisoned content looks suspicious to human observers or could pass content moderation systems.

W2. Detection Discussion: The paper lacks discussion on whether these poisoned entries could be detected through other means (e.g., anomaly detection in embedding space, content verification).

### Questions
Please refer to the weakness

### Soundness
2

### Presentation
2

### Contribution
2
