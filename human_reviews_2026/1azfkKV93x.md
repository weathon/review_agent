# CamoDocs: Poisoning Attack against Retrieval-Augmented Language Models

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
As retrieval-augmented generation (RAG) grows in popularity for compensating the knowledge cutoff of pretrained language models, its security concerns have also increased: RAG retrieves external documents to augment an LLM’s knowledge, and these sources (e.g., Wikipedia, Reddit, X) are often public and editable by uncertified users, creating a new attack surface. Specifically, the risk of poisoning attacks—where malicious documents are injected to steer the LLM to output a targeted answer or to disseminate incorrect information—especially rises with the RAG adoption. Although adversarial attacks on LLMs have been studied (e.g., jailbreaking, backdoor triggers in prompts, and pretraining data poisoning), these approaches do not fully consider RAG’s weakness, in which the external documents can be directly leveraged by attackers. To investigate this threat, we present a method named CamoDocs. Through this, we study how an adversary can construct poisoned documents and how much attack success rate (ASR) can be achieved. CamoDocs chunks synthesized adversarial documents and relevant benign documents from the knowledge database to dilute distinctive signals that defenses might exploit, and further optimizes the chunked benign documents to be more dispersed in embedding space—using a surrogate embedding model and retriever—thereby hiding distinctive characteristics of the final adversarial documents formed by concatenating optimized benign content with chunked adversarial content. We find that this procedure achieves an ASR of 69.55% against heuristic defenses. Furthermore, we demonstrate that a recently proposed RAG defense is insufficient: the attack attains an average ASR of 30.51% across three LLMs (Mixtral, Llama, Mistral) on three benchmarks (HotpotQA, NQ, MS-MARCO)—a rate that is intolerable for deployed RAG systems. These results underscore the urgency of developing stronger defenses to detect and prevent the malicious manipulation of RAG pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose CamoDocs for poisoning the RAG system. The method works in two stages. First, it crafts sub-documents by generating adversarial content and retrieving relevant benign content. Second, it crafts the final adversarial documents by performing token-level manipulation on the benign sub-documents to disperse their embeddings and merging these camouflaged chunks with the adversarial chunks. Authors evaluate this attack against several LLMs and datasets, claiming a high ASR against heuristic defenses and a "non-trivial" ASR of ~27% against the TrustRAG defense.

### Strengths
- The paper addresses the security of RAG systems, which is important.

- The authors correctly identify a clear and trivial-to-exploit flaw in the baseline PoisonedRAG attack (its reliance on query prepending).

- The two-stage procedure is described with sufficient clarity to be understood.

### Weaknesses
- RQ2 and Table 2 prove that this paper's core methodology (using retrieved benign documents) is inferior to a simpler variant (using synthesized benign documents). This invalidates the paper's central claims.

- The paper describes a 27.78% ASR as "intolerable." This is a overstatement.

- The victim retriever is outdated. It is unclear if these vulnerabilities exist in SOTA retrievers having more robust embedding spaces.

- The loss function merely disperses embeddings from their centroid. It might fool k-means, but it is unlikely to fool more robust density-based clustering (DBSCAN) or outlier detection algorithms. This was not tested.

- No adaptive defense is considered.

- missing references

[1]Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents.

[2]Enhancing Noise Robustness of Retrieval-Augmented Language Models with Adaptive Adversarial Training

[3]Understanding Data Poisoning Attacks for RAG: Insights and Algorithms

[4]Trojanrag: Retrieval-augmented generation can be backdoor driver in large language models

[5]Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks

### Questions
1. See weakness
2. Can you justify the claim that a 27.78% ASR is "intolerable"? This implies the TrustRAG defense is ~72% effective at stopping your attack. Why do you consider this a "failure" for the defense rather than a "failure" for the attack?

### Soundness
2

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
5

### Summary
In this paper, the authors introduce CamoDocs, a method that generates adversarial documents by dividing texts into chunks and merging optimized benign sub-documents with adversarial components.

### Strengths
1. The paper presents a new attack method designed to compromise RAG systems.

2. Experimental results are provided to demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. The runtime cost of the proposed attack and all baseline methods is not evaluated.

2. The paper does not clearly specify how many poisoned documents are injected per query.

3. Several recent attack and defense methods are not included in the comparison.

4. The number of queries used for each dataset is too small to ensure reliable evaluation.

### Questions
1. The approach depends heavily on surrogate embedding models (such as ANCE) and surrogate retrievers (like BM25) to approximate the target dense retriever (Contriever). However, the paper does not investigate how sensitive the attack performance is to discrepancies between the surrogate and target retrievers.

2. The attack requires iterative token manipulation and optimization across multiple sub-documents, which is likely to incur high computational cost. Yet, the paper does not include any analysis of runtime, optimization efficiency, or scalability to larger datasets or more queries.

3. The paper does not clearly specify how many poisoned documents are injected per query, leaving the exact attack scale ambiguous.

4. The study omits comparisons with several recent and more sophisticated poisoning attacks on RAG systems, such as [a][b]. 

5. The evaluated defenses are limited and overly simplistic. More advanced and robust defenses, such as [c][d][e], should be considered to provide a more comprehensive evaluation.

6. Only 50 queries are sampled for each dataset, which is too small to draw statistically reliable or generalizable conclusions.



[a] Practical Poisoning Attacks against Retrieval-Augmented Generation.

[b] FlippedRAG Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models. 

[c] Certifiably Robust RAG against Retrieval Corruption.

[d] Traceback of Poisoning Attacks to Retrieval-Augmented Generation.

[e] TrustRAG: Enhancing Robustness and Trustworthiness in RAG.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies knowledge poisoning attacks on RAG systems, where the attacker injects malicious documents into the knowledge base to manipulate the system’s responses to specific queries. To improve the robustness of existing poisoning methods against defenses such as TrustRAG, the authors propose two techniques: concatenating malicious and benign documents, and optimizing the embedding distribution to disperse malicious documents in the embedding space, thereby evading cluster-based detection. Experiments show that the proposed approach achieves higher attack success rates against TrustRAG compared to baseline methods.

### Strengths
1. The paper is well-written and easy to follow. The proposed method is simple yet intuitive.  
2. Experimental results validate the effectiveness of the approach against specific defenses, showing clear improvements over baseline methods.  
3. The ablation study is comprehensive and clearly demonstrates the contribution of different components in the proposed design.

### Weaknesses
1. Among the two evaluated defenses, the paper introduces a query-detection defense that flags documents containing the query as malicious. However, adding queries to documents is a common IR technique for improving retrieval quality [1][2], making this defense questionable.  
2. The proposed method appears tailored to specific defenses rather than offering a generalizable solution. Moreover, one of the evaluated defenses (query detection) may not be valid, as noted above. Other relevant defenses, such as Divide-and-Vote [3], are not considered.  
3. The generated documents contain noticeable gibberish or unnatural phrases after perturbation and replacement (Table 8), which could make them easier to detect.  
4. Although the method is designed to improve performance against TrustRAG, the attack success rate remains relatively low (~27%). Given this targeted improvement, stronger results would be expected. In other settings (Tables 3 and 4), performance is comparable to or worse than baselines.  

**References**  
[1] *Document Expansion by Query Prediction*, 2019.  
[2] *Doc2Query--: When Less is More*, 2023.  
[3] *On the Risk of Misinformation Pollution with Large Language Models*, 2023.

### Questions
1. How many adversarial (poisoned) documents are used for each target query? How does increasing this number affect the ASR and the defense detection rate? For example, when moving from 1-shot to multi-shot poisoning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents CamoDocs, a poisoning attack on RAG systems that hides malicious content inside normal-looking documents. It blends adversarial and benign text, then tweaks the benign parts so the documents evade detection. Tested on Llama-3, Mixtral, and Mistral across QA benchmarks, CamoDocs achieves up to 70% attack success and still 27% under defenses like TrustRAG. The study warns that RAG pipelines are highly vulnerable to stealthy data poisoning and calls for stronger defenses.

### Strengths
1. This paper gives a concrete two-stage method (chunking + token manipulation) with an algorithmic description.
2. It includes embedding visualizations and distance/KDE analyses to explain why the attack evades clustering defenses.
3. This paper assumes black-box access to LLMs/retrievers and only ability to inject documents, matching real-world constraints.

### Weaknesses
1. The paper does not provide any evaluation of the runtime or computational cost of the proposed CamoDocs attack or its baselines. Since the method involves multi-stage operations: document chunking, iterative token manipulation, and surrogate model optimization. Understanding runtime overhead is essential. Without these measurements, it is difficult to assess whether the attack is feasible on large-scale real-world RAG systems or only in small controlled experiments.
2.  The paper mentions a total poisoning ratio of less than 1% but does not explicitly report how many poisoned documents are injected per query or how the parameter β (the target number of adversarial documents) is chosen. This omission makes it difficult to reproduce the experiments or to assess the true attack scale and stealthiness. Without a clear specification of per-query injection count, readers cannot determine whether the reported success rates are achievable under realistic constraints or depend on an unrealistically large poisoning budget.
3. The defense side is restricted to simple heuristic mechanisms—TrustRAG, query detection, query rephrasing, and perplexity filtering. These methods are either rule-based or heuristic, lacking consideration of adaptive or learning-based defense strategies such as robust retriever training, certified filtering, or contrastive anomaly detection. Consequently, the defense evaluation may underestimate how current or future systems could mitigate such attacks.
4. While CamoDocs empirically achieves high attack success rates, the paper lacks a theoretical or analytical explanation of why dispersed embeddings and mixed sub-documents so effectively bypass clustering-based defenses. A more formal discussion could clarify whether this behavior is dataset-specific or reflects a general weakness in embedding-space defenses.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
