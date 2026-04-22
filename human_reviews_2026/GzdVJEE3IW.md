# Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records—many of which are stored in remote databases—to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap: Our Socratic Chain-of-Thought Reasoning first sends a generic user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework—combining GPT-4o with a local Llama-3.2-1B model—outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To balance the trade-off between privacy and performance in scenarios where tasks require personal information, the paper introduces a client–server framework that leverages a powerful LLM on the server side for reasoning and employs homomorphic encryption for secure data retrieval.

### Strengths
1. The idea of decomposing the process and leveraging a powerful server to enhance reasoning while preserving privacy is reasonable.
2. The overall structure of the paper is clear and easy to follow.
3. The efficiency evaluation, particularly Figure 2, is meaningful and provides valuable insights.

### Weaknesses
1. Although the proposed approach seems to provide strong privacy protection, the paper lacks an empirical privacy evaluation.
2. The performance experiments are insufficient, e.g. Table 1 only includes two columns. Adding more metrics or additional sub-results would make the experiments on performance more convincing.
3. Regarding the method:
  - While I can identify some novelty in the overall framework presented in Section 2, the contributions and originality in Section 3 are not clearly highlighted. The authors seem to build upon multiple existing techniques and use some mathematical formulations to describe the process. I strongly suggest making the novel contributions more explicit and clearly distinguishing what is new compared to existing baselines (e.g., CHAM).
  - The chosen baselines also seem slightly outdated. It would be better to include comparisons with more recent methods if possible.
  - The communication savings appear to rely primarily on caching and butterfly decomposition. If these techniques are not newly proposed by the authors, relevant references should be added to clarify their originality.

### Questions
Suggestions or Minor Weaknesses:
1. In Table 1, clearly indicate in the main text that “L1,L2” refers to different local model choices.
2. Explain technical terms such as “SIMD” when they first appear.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a hybrid framework for privacy-preserving large language model interaction that intelligently divides computation between a trusted local device and an untrusted cloud service. The system combines a Socratic Chain-of-Thought mechanism, which lets a powerful external LLM generate reasoning prompts and sub-queries without accessing private data, with a homomorphically encrypted vector database that enables secure semantic search over encrypted records.

### Strengths
1.The proposed architecture strikes an excellent balance between computation power and data privacy by splitting tasks between the cloud and local devices. This hybrid perspective itself is conceptually valuable.
2.The finding that a small local model (Llama-1B) guided by GPT-4o can outperform GPT-4o alone is surprising and thought-provoking. It suggests that effective reasoning guidance may sometimes outweigh raw model scale.

### Weaknesses
1. The paper's core claims of practicality and sub-second latency on local devices are unsupported. The "local client" used in the experiments was a Google Cloud server with an 8-core CPU and 32GB of RAM (n2-standard-8). This is in no way equivalent to a smartphone or laptop.
2. The paper claims its hybrid model (L1+R1, 87.7 F1) outperforms the GPT-4o baseline (R1, 80.6 F1). This is misleading. The paper's own ablation study (Table 5) shows that when the baseline (R1) uses the same Socratic-CoT prompt (R1+R1), its performance hits 92.6 F1, easily beating the authors' hybrid model. This proves the architecture isn't superior; the prompt is just better.

### Questions
1. Can the authors provide actual latency and memory usage data for the client-side operations (local LLM inference, HE decryption, PIR protocol) running on a real edge device (like a smartphone or a standard laptop)?
2. Given the paper's own data (Table 5) shows the R1+R1 baseline (92.6 F1) is significantly better than the L1+R1 hybrid model (87.7 F1), how can the authors still claim their hybrid framework outperforms GPT-4o?

### Soundness
2

### Presentation
3

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
This paper proposes a hybrid framework for privacy-preserving LLM-based personal assistants, combining Socratic Chain-of-Thought Reasoning (Socratic-CoT) with a Homomorphically Encrypted Vector Database. The system decomposes user queries, leverages an untrusted LLM for reasoning and sub-query generation (without access to private data), and performs encrypted semantic search over local private data. The final answer is generated by a local LLM using the retrieved records. Experiments on LoCoMo and MediQ benchmarks show the hybrid system outperforms local-only baselines and approaches oracle performance, with ablation studies on Socratic-CoT and encrypted search.

### Strengths
⦁	Highly innovative, balancing privacy and practicality.
⦁	Comprehensive experiments, including ablation analysis.
⦁	Strong potential for real-world deployment.

### Weaknesses
⦁	Significant computational and communication overhead of HE not fully analyzed.
⦁	Local device burden and user experience not discussed.
⦁	Security analysis and threat model are insufficient.
⦁	COT decomposition robustness and fallback mechanisms are not explored.
⦁	Lack of quantitative privacy-utility tradeoff analysis.
⦁	Limited implementation details and reproducibility.

### Questions
⦁	What is the end-to-end latency and throughput of the system under realistic deployment?
⦁	How does the framework perform on resource-constrained devices?
⦁	How are keys managed and how is access pattern leakage prevented in practice?
⦁	How robust is the system to inaccurate or incomplete COT decomposition?
⦁	Can the authors provide more details or open-source code for reproducibility?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Socratic Chain-of-Thought Reasoning (Socratic-CoT), a hybrid privacy-preserving framework for large language model (LLM) reasoning over sensitive user data. The approach decomposes user queries using an untrusted remote LLM, which produces a chain-of-thought and sub-queries without directly accessing user data. These sub-queries are then embedded and used to perform encrypted semantic search over the user’s private database through a Homomorphically Encrypted (HE) Vector Database. Retrieved records are decrypted locally and passed to a smaller, trusted local LLM (e.g., Llama-3.2-1B) to generate the final contextualized response.
Experiments on the LoCoMo long-context QA benchmark demonstrate up to 7.1 percentage-point improvement over GPT-4o alone, showing that hybrid reasoning between remote and local models can improve accuracy while preserving privacy.

### Strengths
Timely and relevant problem: Addresses the crucial trade-off between model capability and user privacy for personal assistants handling sensitive information.
Empirical validation: The LoCoMo experiments show that the hybrid setup can outperform a large remote model, supporting the paper’s core hypothesis.

### Weaknesses
Insufficient cryptographic detail:
The cryptographic layer lacks rigor. The paper cites Gentry (2009b) and Brakerski et al. (2014) in the main body but does not specify which homomorphic encryption (HE) scheme is actually implemented, nor its parameters or security level. These references are outdated relative to practical schemes like CKKS.
Missing PIR protocol description:
The same issue applies to the Private Information Retrieval (PIR) component. The paper does not indicate which protocol variant is used.
No formal security model or proofs:
Despite repeated privacy claims, there are no formal definitions or security proofs. It remains unclear what type of privacy guarantee is achieved (semantic security, query unlinkability, etc.) and against what adversarial assumptions.
Further clarification is required.
Ambiguous baselines and fairness of comparison:
The description of baselines is incomplete. It is unclear whether GPT-4o has direct access to private data or is restricted. Without a properly defined remote-only baseline, the comparison risks being unfair, as differences in data access could inflate performance gains.
Limited experimental evaluation:
The evaluation relies on only two datasets, which is not sufficient to substantiate generalization claims. The approach should be validated on more diverse and larger datasets.
Unjustified claim on local model limitations:
The statement that local models preserving privacy lack the computational capacity for complex reasoning is asserted but not justified. The authors should clarify whether this refers to hardware limits, parameter size, or theoretical restrictions introduced by privacy mechanisms.
Unexplained mathematical notation in Section 3.2:
Several mathematical symbols and expressions in Section 3.2 are introduced without proper definitions or explanations. This makes it difficult for readers to follow the derivations and evaluate the correctness of the proposal.

### Questions
On local model limitations: Why exactly do local privacy-preserving models lack sufficient reasoning capacity—hardware limits, model scale, or privacy overhead?
On HE details: Which homomorphic encryption scheme is used? Why you do not provide security proof?
On PIR details: What specific PIR protocol is implemented based on existing literature, and what are its performance characteristics?
On baselines: What exactly is the baseline that the proposed method outperforms, and how is fairness ensured?
On dataset diversity: Why only two datasets? Are additional evaluations planned?

### Soundness
2

### Presentation
2

### Contribution
1
