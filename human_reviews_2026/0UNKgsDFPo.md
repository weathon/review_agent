# MRAG-Corrupter: Knowledge Poisoning Attacks to Multimodal Retrieval Augmented Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 2, 6

## Abstract
Multimodal retrieval-augmented generation (RAG) enhances visual reasoning in vision-language models (VLMs) by accessing external knowledge bases. However, their security vulnerabilities remain largely unexplored. In this work, we introduce MRAG-Corrupter, a novel knowledge poisoning attack on multimodal RAG systems. MRAG-Corrupter injects a few crafted image-text pairs into the knowledge database, manipulating VLMs to generate attacker-desired responses. We formalize the attack as an optimization problem and propose two cross-modal strategies, dirty-label and clean-label, based on the attacker’s knowledge and goals. Our experiments across multiple knowledge databases and VLMs show that MRAG-Corrupter outperforms existing methods, achieving up to a 98% attack success rate with only five malicious pairs injected into the InfoSeek database (481,782 pairs). We also evaluate four defense strategies—paraphrasing, duplicate removal, structure-driven mitigation, and purification—revealing their limited effects against MRAG-Corrupter. Our results highlight the effectiveness and stealthiness of MRAG-Corrupter, underscoring its threat to multimodal RAG systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the security vulnerabilities of multimodal retrieval augmented generation. To bridge the gap on attack on multi-modal systems, the paper proposes MRAG-Corrupter, the first knowledge poisoning attack tailored for multimodal RAG. It derives two cross-modal solutions and evaluates it across multiple knowledge databases and victim VLMs. The attack method outperforms baseline methods. Various deffense strategies are also investigated.

### Strengths
- The topic studied in this paper is important. In particular, it focuses on multimodal RAG systems, which is under-explored now.
- The proposed MRAG-Corrupter demonstrated consistent performance improvements over exisiting methods and achieved high ASRs. A set of powerful models are taken as the victim VLMs, which makes the conclusion more generalizable.
- Two retriever access scenarios, restricted-access and full-accsess, are considered. Attack strategies are designed for the corresponding ones.
- Overall, the paper structure is clear and figures are nicely presented.

### Weaknesses
- In Sec. 2 Problem Formulation and Sec. 3 MRAG-Corrupter, the authors introduced a lot of notations, it is a little bit easy to get lost to follow them. E.g., the use of \dot over letters in Ln 96.

### Questions
- In Ln 180, the heuristic approach directly injects the query image $\dot{I}_i$ as $\tilde{I}_i^j$ while keeping $\tilde{T}_i^j$ unchanged. Why not do it reversely, i.e., inject the text while keeping image unchanged?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose MRAG-Corrupter, a targeted knowledge poisoning attack on multimodal RAG systems, combining a retriever with a VLM. The attack enables an adversary to control VLM outputs for chosen queries by injecting a small number of carefully crafted image-text pairs into the knowledge base. MRAG-Corrupter combines embedding similarity optimization, surrogate-based generative text refinement, and either direct copying (dirty-label) or stealthy perturbation (clean-label) of images to create “retrieval-dominant” and “generation-dominant” poisoned pairs. These pairs are highly likely to be retrieved and used by the VLM for target queries, thus allowing an attacker to control outputs with minimal, well-crafted database modifications.

### Strengths
The paper is clear and easy to follow. It proposes a novel method to attack RAG systems. The authors rigorously formalize the attack as an optimization problem and introduce two cross-modal strategies tailored for varying attacker knowledge and access levels. The authors also provide rigorous experiments, evaluating over leading VLMs and adapted baselines.

### Weaknesses
My primary concern pertains to the practical deployment of the proposed MRAG-Corrupter method. While the attack framework assumes access to a well-structured knowledge base comprising image-text pairs, real-world settings commonly employ access controls, provenance tracking, and other integrity mechanisms that may substantially reduce the risk of knowledge poisoning. Additionally, the community may be more interested in attacks targeting online, less curated, or dynamic multimodal knowledge sources rather than static databases. The experimental evaluation is limited to image-text pairs, without consideration of other relevant modalities such as tables, videos, or audio. It remains unclear how transferable the proposed attack strategies and the underlying optimization logic would be to these alternative modalities.

### Questions
I encourage the authors to discuss the generalizability of their method and present evidence or analysis regarding its applicability to different types of multimodal knowledge bases.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes MRAG-corrupter, a knowledge poisoning attack on vision-language RAG systems. It aims to manipulate the MRAG system's response to specific target queries by injecting poisoning samples into the knowledge database. The paper formulates the attack as an optimization problem and discusses to solve it in two different settings (i.e., restricted-access and full-access setting) depending on the attacker's knowledge. Extensive experiment results demonstrate that the proposed attack is both effective and robust.

### Strengths
•	This paper have a good writing to express the key idea clearly. The figures and tables are also clear and easy to understand.
•	The paper includes comparison with many baseline methods, demonstrating the effectiveness of the proposed attack.
•	The paper discusses many possible defenses to demonstrate the robustness of the attack.

### Weaknesses
•	The core idea of this paper is similar to an existing work [1] on corrupting vision-language RAG systems, but [1] is neither cited nor compared in this paper.
•	The paper only discusses MRAG systems that retrieve images and texts. However, MRAG may not limit to images and texts. It can also support other modalities (e.g., audio [2]) and retrieval scenarios [3]. Therefore, the use of MRAG may not be appropriate here.
•	The paper lacks exploration of practical real-world scenarios for the attack.
[1] PoisonedEye: Knowledge Poisoning Attack on Retrieval-Augmented Generation based Large Vision-Language Models. ICML 2025.
[2] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models. ACL 2025.
[3] UniIR: Training and Benchmarking Universal Multimodal Information Retrievers. ECCV 2024.

### Questions
•	What are the main differences between your work and existing work (e.g., [1])? Could you further compare experiment results with [1] to demonstrate the difference?
•	In the clean label attack, why the optimized image still preserves semantic meaning to texts? Instead of human reviewers, is it detectable by an embedding model like CLIP?
•	Since the experiments only use CLIP-based models, could the authors conduct more experiments on MLLM-based retrievers like GME [4]?
•	For robustness, is the attack robust to multimodal reranking [5] and defenses such as RoCLIP [6]?
•	Could the authors provide any experiments or examples of real-world applications of the attack, such as compromising a RAG-based LLM agent?
[4] GME: Improving Universal Multimodal Retrieval by Multimodal LLMs. arXiv 2024.
[5] MLLM Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation via Knowledge-enhanced Reranking and Noise-injected Training. arXiv 2024.
[6] Robust contrastive language-image pretraining against data poisoning and backdoor attacks. NeurIPS 2023.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MRAG-Corrupter, a knowledge poisoning attack framework specifically designed for Multimodal RAG systems. Multimodal RAG enhances VLMs by allowing them to retrieve information from external knowledge bases containing images and text. The authors identify a security vulnerability: by injecting a small number of maliciously crafted image-text pairs into the knowledge database, an attacker can manipulate the VLM to generate attacker-desired, incorrect answers to specific user queries. 
The attack is formalized as an optimization problem with two necessary conditions: a Retrieval Condition (ensuring the malicious pairs are retrieved) and a Generation Condition (ensuring the VLM produces the target answer from the retrieved pairs). To address different threat scenarios, the authors propose two attack strategies: (1) Dirty-Label Attack: For a restricted-access scenario where the attacker has no access to the retriever. It directly reuses the query image and prepends the query text to a maliciously generated description. (2) Clean-Label Attack: For a full-access scenario where the attacker can query the retriever. It generates semantically aligned image-text pairs and adds subtle perturbations to the image to maximize retrieval similarity while appearing legitimate to human moderators.

### Strengths
+ Clear Illustration
+ Technically sound design
+ Comprehensive evaluation

### Weaknesses
1. I am curious that if a knowledge base already contains an image–text pair that exactly matches a query, how can an attacker craft a new pair that the VLM retrieves and uses to produce the attacker’s desired output. Is this vulnerability primarily due to encoder weaknesses or to retrieval and ranking dynamics?

2. While the paper evaluates four defense strategies, the defenses are tested in isolation, but real-world systems may employ multiple complementary defense. It would be better to discuss this aspect. 

3. The clean-label attack requires DALL·E-3 to generate aligned image-text pairs and assumes access to the retriever in full-access scenarios. This creates a significant barrier for real-world deployment where such access might be restricted. Moreover, the clean-label attack's effectiveness is tied to perturbation intensity ε, with higher values (ε = 32/255). Though the author claims that it remains stealthy, but it is better to showcase more concrete examples. It seems there is a practical trade-off in stealth versus effectiveness.

### Questions
1. It is suggested to clarify the foundamental reason for such vulnerability. 

2. It is better to consider and discuss compound defenses. 

3. It is better to analyze the practical trade-off between stealth and effectiveness for clean-label attack.

### Soundness
3

### Presentation
4

### Contribution
3
