# Pisces: Cryptography-based Private Retrieval-Augmented Generation with Dual-Path Retrieval

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Retrieval-augmented generation (RAG) enhances the response quality of large language models (LLMs) when handling domain-specific tasks, yet raises significant privacy concerns. This is because both the user query and documents within the knowledge base often contain sensitive or confidential information. To address these concerns, we propose $\texttt{Pisces}$, the first practical cryptography-based RAG framework that supports dual-path retrieval, while protecting both the query and documents. Along the semantic retrieval path, we reduce computation and communication overhead by leveraging a coarse-to-fine strategy. Specifically, a novel oblivious filter is used to privately select a candidate set of documents to reduce the scale of subsequent cosine similarity computations. For the lexical retrieval path, to reduce the overhead of repeatedly invoking labeled PSI, we implement a multi-instance labeled PSI protocol to compute term frequencies for BM25 scoring in a single execution. $\texttt{Pisces}$ can also be integrated with existing privacy-preserving LLM inference frameworks to achieve end-to-end privacy. Experiments demonstrate that $\texttt{Pisces}$ achieves retrieval accuracy comparable to the plaintext baselines, within a 1.87% margin.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper focuses on an important research question of privacy-preserving retrieval augmented generation. Compared to differential privacy based methods, cryptography methods can provide more utility and stronger privacy guarantees with a scarification of efficiency and a burden for users to upload and download more data. Authors show that the feasibility of the solution with experiments

### Strengths
1. The paper fills the blanks of no cryptography solution in the field of privacy-preserving RAGs.
2. The paper considers the case of semantic and lexical retrieval.

### Weaknesses
1. Authors mentioned some DP-based methods and argued that they may harm utility. Is there any empirical experiment to support that? It seems that I didn't find a comparison to these baselines in the evaluation.
2. The motivation is confused. Why the answer to "How can we maintain retrieval performance while ensuring privacy for both the query and documents during retrieval?" Is using cryptography-based methods? Why do other methods not achieve this?

I think the motivation and high-level idea is interesting but there are some technical problems that need to be resolved. I hope the authors can provide some rationale for them before the paper reaches the acceptance standard.

### Questions
1. I think work in Huang et al., 2023 and Yao & Li, 2025 considered lexical retrieval. But they did not consider the retrieval of considering semantic and lexical retrieval at the same time. Is this a point that the paper wants to articulate?
2. I think this baseline is missing. [1]
3. I am a little bit confused by the structure. So, at the final inference stage, retrieved documents and queries are encrypted through MCP and the user uses a private key to decode and get responses. But during the retrieval phase, the tokenizer and embedding model need the plain text of the query, would that leak privacy? I think at this stage, the query is not encrypted.
4. Why in Figure 2, ClapNQ_Train_single. Pisces (with MPC) has higher accuracy than plaintext? Is it because of randomness? If so, I think multiple experiments should be conducted and error bars need to be reported.
5. Why in Table 4, HopotQA, encrypted text computation can have less time than plain text computation?

[1] Koga, Tatsuki, Ruihan Wu, and Kamalika Chaudhuri. "Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy." arXiv preprint arXiv:2412.04697 (2024).

### Soundness
1

### Presentation
2

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
Pisces proposes the first cryptography-based retrieval-augmented generation (RAG) framework that simultaneously protects both user queries and documents through dual-path (semantic and lexical) private retrieval, achieving near-plaintext accuracy with significantly improved efficiency.

### Strengths
1. The paper presents a well-structured and logically coherent framework.

2. The proposed method designs customized cryptographic protocols that effectively balance privacy protection and retrieval efficiency.

### Weaknesses
1. Limited support for multi-user concurrency. The current framework is designed for a single client–server setting and does not easily extend to multi-user or concurrent query scenarios.

2. Limited dataset scale and diversity. The evaluation is conducted on moderate-sized, English-only datasets, which limits the generalizability to large-scale or domain-specific applications.

3. Lack of empirical security analysis. The paper provides theoretical privacy guarantees but does not include empirical evaluations against potential attacks such as reconstruction or traffic analysis.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Pisces, the first practical cryptography-based RAG framework that supports dual-path retrieval, while protecting both the query and documents. 

Along the semantic retrieval path, we reduce computation and communication overhead by leveraging a coarse-to-fine strategy. Specifically, a novel oblivious filter is used to privately select a candidate set of documents to reduce the scale of subsequent cosine similarity computations.

### Strengths
The paper proposes the first cryptography-based RAG retrieval framework with dual-path retrieval, while
ensuring privacy for both the query and documents.

The authors  propose a coarse-to-fine strategy for the semantic retrieval path with an oblivious filter to
reduce computation and communication complexity.

### Weaknesses
1. The motivation of why applying MPC to RAG with LLMs is seriously unclear to me.Please rewrite this part. 


2. The motivation and possible usage scenarios of your proposed system in reality  needs to be verified. 

3. Please provide the strict privacy definition, your Cryptography-based Private RAG seems only protect the privacy in the whole system in certain parts.

### Questions
see the above comments.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents Pisces, a framework for cryptography-based privacy-preserving retrieval-augmented generation (RAG). The goal is to protect both user queries and knowledge base documents while maintaining retrieval efficiency and accuracy. Pisces integrates dual-path retrieval — combining semantic and lexical matching — with novel cryptographic mechanisms. The semantic path adopts a coarse-to-fine strategy using an oblivious filter over Hamming distance and secure multi-party computation (MPC) for cosine similarity, while the lexical path employs a multi-instance labeled private set intersection (PSI) protocol to compute BM25 scores privately. The authors claim that Pisces achieves retrieval accuracy close to plaintext baselines (within 1.14%) and substantially improves efficiency (up to 496× faster in certain PSI tasks).

### Strengths
1. Comprehensive framework design: The paper systematically integrates multiple cryptographic primitives (MPC, PSI, OPRF, OKVS) into a cohesive RAG retrieval pipeline, ensuring privacy for both queries and documents.

2. Dual-path retrieval consideration: Supporting both semantic and lexical retrieval paths under privacy-preserving constraints is an ambitious and meaningful direction for improving private RAG systems.

### Weaknesses
1. Limited novelty and unclear challenge motivation: The paper’s technical contributions appear incremental, and it lacks a clear explanation of the key challenges being solved. The Introduction mostly summarizes system components and prior cryptographic methods, but does not articulate the specific research problem or insight that distinguishes Pisces from existing privacy-preserving RAG frameworks. Overall, it reads more like a technical report than a research paper with a well-defined innovation.

2. Unclear use of “cryptography-based” in the title: Although the framework is described as cryptography-based, the cryptographic novelty is not obvious. From the description, the “cryptography” aspect mainly involves using an embedding model and tokenizer to encode queries and documents, which is standard in most RAG architectures. If the cryptographic contribution only lies in combining existing MPC/PSI primitives, it does not sufficiently justify the “cryptography-based” claim. The authors should clarify how Pisces provides new cryptographic advances beyond typical embedding-based privacy mechanisms.

3. Missing experimental comparison with Table 1 baselines: The paper introduces several baseline methods in Table 1 (e.g., DP-RAG, LPRAG, RemoteRAG, etc.), but these methods are not actually compared in the experimental section. Without quantitative results against those baselines, it is difficult to assess whether Pisces offers meaningful advantages in accuracy, efficiency, or privacy protection.

### Questions
see the weakness

### Soundness
2

### Presentation
2

### Contribution
2
