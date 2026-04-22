# Secure Autoregressive Inference with Prompt Separation via Key-Value Caching

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Large Language Models (LLMs) have demonstrated remarkable performance, driving their widespread adoption across various applications. This prevalence increases the importance of user request privacy during inference. While Fully Homomorphic Encryption (FHE) and Secure Multi-Party Computation (MPC) offer promising solutions for privacy-preserving inference, they suffer from significant latency overhead, limiting practical deployment. Prior research has explored more efficient cryptographic primitives and polynomial approximations for non-linear operations. However, the inference latency remains significantly higher than that of plaintext execution. To further mitigate computational overhead, we introduce a novel approach that leverages prompt separation with key value caching. Our method accelerates secure inference by processing non-sensitive tokens in plaintext and using their key-value caches when subsequently processing private tokens. To ensure effective contextual reasoning, we also introduce an attention mask adjustment mechanism that constrains privacy-sensitive tokens to attend to nearby tokens from their original masked positions. Through experiments across various LLM architectures and MPC frameworks, we show that our approach achieves a 1.5-2.5$\times$ reduction in inference latency without significant performance degradation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Traditional MPC reasoning requires re-execution of security protocols after each token is generated, resulting in extreme latency. This work addresses the token-by-token dependency issues of auto-regressive LLMs by implementing structured predictions. This involves performing computations on non-sensitive tokens on the plaintext side first, calculating partial activations without leaking previous tokens. Furthermore, the attention buffer is secretly shared, eliminating the need for re-negotiation of keys during cross-stride reasoning, reducing communication traffic and encryption costs.

### Strengths
The research problem is valuable, and the solution is highly compatible with existing MPC frameworks (such as CrypTen and SecFormer). The method is tested on **GPT-2** and **Qwen2-0.5B-Instruct**, and under two MPC frameworks, **CrypTen** and **SPU**, respectively, achieving **1.5–2.5×** inference acceleration on several MMLU subtasks with very low accuracy loss.

### Weaknesses
**1.Idealistic security assumptions:**

The security proof is incomplete. The authors assume that "non-sensitive tokens are visible in plaintext," but overlook the fact that the length, structure, or position pattern of plaintext prompts can also lead to side-channel information leakage.

**2.Limited parallelism:**

Although the scheme reduces communication rounds, inference parallelism is still constrained by the depth of autoregressive dependencies, making it difficult to completely eliminate the sequencing bottleneck.

**3.Lack of systematic comparative experiments:**

While the paper mentions related work (MPCFormer, Bolt/Iron, etc.), it lacks cross-sectional experiments using the same dataset and framework, or overlay tests with other protocols.

### Questions
1. Are the structure, length, and position features of the plaintext considered potentially private information?

2. Can the KV be reused across requests after being converted from the plaintext domain to the ciphertext domain?

3. Does it maintain near-zero performance with long contexts (>32k), enhanced retrieval (RAG), conversational security (rejection policy), and code generation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a prompt separation technique to accelerate secure Transformer inference based on Fully Homomorphic Encryption (FHE) and Secure Multi-Party Computation (MPC). The core insight is that not all input tokens carry privacy-sensitive information. Leveraging this, the authors separate private and non-private tokens using a Personally Identifiable Information (PII) identifier. The key-value (KV) cache for non-private tokens, along with their corresponding masks, is computed in plaintext and reused during secure inference over private tokens. This avoids computing the entire KV-cache in the encrypted domain. Experimental results demonstrate that the proposed approach can accelerate inference by up to 2.5 times.

### Strengths
1. The paper is well written and easy to follow. The prompt separation framework and its integration with KV caching are clearly explained.

2. The research topic is important and well-motivated. Enhancing the efficiency of secure Transformer inference has direct relevance to privacy-preserving real-world applications.

3. The evaluation is thorough. The authors conduct experiments on both GPT-2-small and Qwen2-0.5B-Instruct, showing consistent speedups over baseline methods.

### Weaknesses
1. The threat model requires further clarification:

- This work assumes that only certain tokens—those identified as PII (e.g., PERSON, PHONE_NUMBER, DATE_TIME)—are privacy-sensitive. This contrasts with prior works such as MPCFormer, Iron, and Bolt, which treat the entire input as private. However, contextual information like grammar, tense, or sentence structure may also leak private information. Clarifying the scope of what is considered private would make the threat model more rigorous.

- Additionally, the claim in line 831 that model parameters remain hidden even in the two-party setting is inaccurate for frameworks such as Iron and Bolt, where model weights are held in plaintext by the server.

2. There is noticeable accuracy degradation. Although the proposed technique improves inference efficiency, it sometimes leads to reduced accuracy, as shown in Tables 2 and 3. For instance, on the History dataset using the SPU backend, accuracy drops from 27.27 to 25.54.

### Questions
1. Are there known failure cases of the prompt separation approach? Would fine-tuning the LLM in a mask-aware manner help mitigate the accuracy loss?

2. How is privacy leakage assessed when non-private tokens are exposed to an untrusted server? Are there quantitative metrics or threat analyses that support the claimed privacy guarantees?

3. The current evaluation is limited to relatively small models such as GPT-2-small. How does the proposed technique scale to larger models in terms of performance and security trade-offs?

### Soundness
2

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
This paper presents an interesting idea to accelerate the performance of secure inference over private tokens by separating the private tokens from the rest of the prompt. By only hiding the computation over the private tokens, they reduce the amount of operations that needs to be performed within the secure computation.

### Strengths
The idea presented in this work is novel and has the potential to substantially improve the performance of semi-private inference.

### Weaknesses
While the idea in this paper is very interesting, I was not able to verify correctness due to a lack of explanation for the other parts of the attention block. The matrix products in the attention block mix the values across tokens, so it is not clear from the paper how these values are handled. In particular, if the intermediate states that are functions of the private tokens are not revealed, then this approach seems like if would be nullified after the first block (since the entire state would be a function of the private tokens). On the other hand, if the intermediate results are revealed except at the locations of the private tokens, this could leak information about the private tokens. More explanation is needed on these steps.

### Questions
I would like to understand how the matrix operations within the attention blocks are performed with some of the input rows masked. It seems like most (if not all) of the entries of the final output of the attention block are functions of all input tokens, so if some of these values are masked then it’s not clear how this output is computed for the subsequent blocks. Could you give a complete description of the modified attention block (including all matrix operations) with the masked tokens? 
When masking private tokens, is there a generic embedding for each category? Or is the embedding for the private tokens replaced with whatever embedding the model assigns to the token “[MASKED_TOKEN_#]”?

### Soundness
2

### Presentation
2

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
This work proposes a method to accelerate privacy-preserving inference for LLM under homomorphic encryption through a technique that separates sensitive and non-sensitive tokens in a prompt. Then, non-sensitive tokens are processed in plaintext to precompute key-value caches, while sensitive tokens are processed securely using these cached representations. An attention-mask adjustment mechanism ensures that sensitive tokens still attend to relevant context. Experiments on GPT-2 and Qwen2 under MPC frameworks show 1.5–2.5× faster inference with minimal performance loss, and communication costs drop up to 4×. This approach maintains security while making encrypted inference substantially more practical

### Strengths
The proposed framework of processing the majority of the insensitive text in plain text and only processing the sensitive information in cyphertext is quite ingenious. While there is still a security concern in terms of whether this protocol can really protect one's privacy (redacted documents often do reveal a lot of private information), given the need for some degree of security while not incurring an inordinate compute cost is a nice compromise.

### Weaknesses
The attention mask adjustment feels unnecessary to me, and it doesn't seem like the paper provides sufficient evidence supporting the necessity of this mechanism.

Also, I was hoping the acceleration would be larger than 2x or 4x. Can the authors explain why the speedup is not more extreme despite the sensitive words only consisting of a small portion of the input prompt?

### Questions
What evidence do you have that the proposed attention sink mechanism is necessary? Is this really the case?

### Soundness
3

### Presentation
3

### Contribution
3
