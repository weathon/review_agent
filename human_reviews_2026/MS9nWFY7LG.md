# Q-RAG: Long Context Multi‑Step Retrieval via Value‑Based Embedder Training

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 2, 8, 8, 6

## Abstract
Retrieval-Augmented Generation (RAG) methods enhance LLM performance by efficiently filtering relevant context for LLMs, reducing hallucinations and inference cost.
However, most existing RAG methods focus on single-step retrieval, which is often insufficient for answering complex questions that require multi-step search.
Recently, multi-step retrieval approaches have emerged, typically involving the fine-tuning of small LLMs to perform multi-step retrieval.
This type of fine-tuning is highly resource-intensive and does not enable the use of larger LLMs.
In this work, we propose Q-RAG, a novel approach that fine-tunes the Embedder model for multi-step retrieval using reinforcement learning (RL).
Q-RAG offers a competitive, resource-efficient alternative to existing multi-step retrieval methods for open-domain question answering and achieves state-of-the-art results on the popular long-context benchmarks BabiLong and RULER for contexts up to 10M tokens. Code is available at: https://github.com/griver/Q-RAG.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an approach that fine-tunes an embedder model for multi-step retrieval through reinforcement learning. The proposed method, Q-RAG, achieves state-of-the-art results on long-context benchmarks including Babilong and Needle-in-a-haystack tasks.

### Strengths
- By decoupling the retriever from the LLM and training only a lightweight embedder, the proposed approach is practical, compared to other approaches that require fine-tuning the entire LLMs.
- This paper introduces a relative positional encoding scheme for context embeddings. This allows the agent to reason about the temporal relationships between chunks.
- The experiments demonstrate scalability and robustness, achieving state-of-the-art results on benchmarks with contexts up to 10 M tokens.

### Weaknesses
- The paper’s analysis of the multi-step retrieval process is under-explored. While the method relies on a maximum step budget, T, this critical hyperparameter is not specified for the experiments, hindering reproducibility. Furthermore, the paper lacks an investigation into the agent’s learned stopping behavior, i.e., it is unclear if the agent effectively utilizes the STOP action to terminate retrieval early. A sensitivity analysis on how performance varies with different values of T is also absent. This omission is significant, as the optimal number of retrieval steps is often unknown in practice, leaving the method’s robustness and practical applicability in question.
- The paper’s presentation could be significantly improved. Several key results and concepts are not properly introduced or referenced, which disrupts the reader’s flow. For instance, the “Plan-Q-RAG” varian appears abruptly in Table 2 without prior introduction or motivation in the main body of the paper. Later, I found its explanation in the appendix. Additionally, Figure 2 is not explicitly mentioned anywhere in the text.
- The empirical validation for the proposed temporal reasoning mechanism appears narrow. The authors claim this mechanism is a major technical contribution, yet its effectiveness is demonstrated exclusively on the Babilong benchmark. To substantiate the generalizability and impact of this technique, the evaluation should have included a broader range of tasks where temporal or sequential understanding is paramount. For example, testing on datasets that require reasoning over chronologically ordered documents or procedural texts (e.g., [1]) would provide more robust evidence and strengthen the authors’ claims.

[1] Karpinska, Marzena, et al. "One Thousand and One Pairs: A “novel” challenge for long-context language models." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.

### Questions
Can you explain all the points I raised in the Weaknesses section?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents Q-RAG, a framework that formulates multi-step retrieval in retrieval-augmented generation (RAG) as a reinforcement learning problem. Two transformer encoders are trained: a state encoder that represents the current retrieval context (the initial query and previously retrieved chunks), and an action encoder that represents each candidate document chunk. The similarity between the two embeddings defines a Q-value, which indicates the estimated utility of retrieving that chunk in the current context. Training uses Soft Q-Learning with λ-return, where the model receives a binary outcome-based reward depending on whether all supporting facts are retrieved. During inference, the model iteratively embeds the current state, scores candidate chunks by their Q-values, selects the most relevant one, and updates the state until retrieval is complete. Experiments are conducted on HotpotQA, MuSiQue, RULER (1M tokens), and BabiLong (10M tokens) to evaluate retrieval accuracy and computational efficiency.

### Strengths
- The paper presents a clear formulation of multi-step retrieval as a reinforcement learning process, replacing traditional query rewriting with a learned retrieval policy. This modeling perspective provides a coherent computational framework for retrieval-augmented generation.
- The proposed model adopts a dual-encoder design, where a state encoder and an action encoder jointly estimate Q-values to guide multi-step retrieval decisions. Both encoders are compact transformer models, showing that a small embedding-based retriever can learn retrieval behavior previously handled by large language models.
- The training framework integrates Soft Q-Learning with λ-return, which helps stabilize optimization under sparse outcome rewards. The method shows that an embedding model can reach retrieval accuracy comparable to reinforcement-learning–finetuned large language model retrievers while operating at roughly 10–20× lower computational cost.
- The experimental evaluation covers multiple datasets, including HotpotQA, MuSiQue, and synthetic long-context settings (RULER, BabiLong), demonstrating consistent retrieval performance and scalability to contexts up to 1M–10M tokens.

### Weaknesses
- The model’s architecture and training setup restrict its ability to directly handle long or unstructured contexts. Both encoders have short input windows, and the approach depends on datasets with annotated supporting facts, which limits generalization to open-domain or unlabeled corpora.
- The paper lacks theoretical analysis of the proposed Q-learning framework. No formal discussion of convergence, optimality, or error bounds is provided, and the training stability is supported only by empirical observations.
- The reward design relies on a single outcome-based binary signal. While λ-return mitigates the sparsity issue, the paper does not discuss other possible reward shaping or exploration strategies.

### Questions
- Do the authors observe any evidence of policy collapse or repeated retrieval of the same chunks during training?
- How well does the learned policy generalize to unseen domains or corpora without retraining?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Q-RAG, a novel and resource-efficient framework for multi-step retrieval in Retrieval-Augmented Generation (RAG) systems. The authors' primary contribution is a paradigm shift away from fine-tuning the Large Language Model (LLM) itself, focusing instead on training a smaller, more efficient embedder model using value-based reinforcement learning (RL). The paper claims three main contributions: (1) a new method for training a multi-step retrieval agent using temporal difference RL, which is exceptionally resource-efficient (trainable on a single GPU); (2) state-of-the-art performance on ultra-long context benchmarks (up to 10M tokens) like Babilong and RULER; and (3) a novel relative positional encoding mechanism that enables temporal reasoning during the retrieval process. This approach is presented as a modular, flexible, and scalable solution to the challenges of multi-hop reasoning and "needle-in-a-haystack" problems in extremely long documents.

### Strengths
- **Novelty and Technical Soundness:** The core idea of fine-tuning the embedder instead of the LLM for multi-step retrieval is a significant and elegant contribution. The formulation of the problem as a Markov Decision Process (MDP) is well-defined, and the factorization of the Q-function into an inner product of state and action embeddings is a clever design choice, theoretically justified in the appendix. The introduction of dynamic relative positional encoding to handle temporal reasoning is particularly innovative and is empirically shown to be highly effective on the Babilong QA3 task.

- **Exceptional Resource Efficiency and Accessibility:** The ability to train the entire system on a single A100 GPU is a major practical advantage. This dramatically lowers the barrier to entry for research in multi-step RAG, making it accessible to a much wider community compared to LLM-tuning methods that require large GPU clusters.

- **Modularity and Flexibility:** By decoupling the retriever from the generator, the trained Q-RAG embedder can be paired with any LLM, including powerful proprietary models like GPT-4. This "plug-and-play" nature is a significant strength in the current AI ecosystem.

- **Strong Empirical Results and Scalability:** The paper demonstrates state-of-the-art or highly competitive performance across a range of challenging benchmarks, including ultra-long context (Babilong, RULER) and standard multi-hop QA (HotpotQA, Musique). The model's ability to maintain high performance as context length scales to 10 million tokens is particularly impressive and clearly demonstrates the scalability of the proposed method.

- **Reproducibility:** The authors provide detailed experimental setups and ablation studies justifying their architectural choices (e.g., using soft Q-learning and target networks). They also state their intention to release code, which strengthens the paper's contribution.

### Weaknesses
- **Critical Dependence on Supervised Reward Signal:** The paper's most significant weakness is its reliance on a sparse terminal reward based on the retrieval of a pre-defined set of "ground-truth support facts ($F^*$)." While this provides a stable training signal, it severely limits the method's real-world applicability. Such meticulously labeled datasets are extremely rare and prohibitively expensive to create for new domains (e.g., legal, medical, enterprise knowledge). This dependency relegates the current method to a powerful tool for academic benchmarks rather than a general-purpose, deployable solution.

- **Underestimation of the LLM-as-a-Judge Problem:** The paper acknowledges the limitation of supervised rewards and briefly suggests using an LLM to generate a reward signal as "future work." This suggestion dangerously overlooks the well-documented and critical failure modes of the "LLM-as-a-Judge" paradigm. Recent research has extensively shown that LLM judges are systematically vulnerable to "reward hacking" and "master key" attacks, where superficial, semantically meaningless inputs (e.g., punctuation, phrases like "Thought process:") can elicit false positive rewards. An RL agent like Q-RAG would rapidly learn to exploit these vulnerabilities, leading to a collapsed policy that generates "master keys" instead of performing meaningful retrieval. The paper's failure to address this critical challenge makes its proposed path to unsupervised application seem naive and unviable in its current form.

- **Limited Compositional Reasoning in the Retriever:** While the relative positional encoding enables temporal reasoning, the retriever itself does not perform more complex compositional reasoning, such as comparison or calculation. The task of synthesizing information and drawing a final conclusion is entirely offloaded to the final, frozen LLM. The system's performance is therefore contingent on the capabilities of this final LLM, and it may fail on questions that require intricate reasoning over the retrieved facts.

- **Isolation from the Broader Advanced RAG Ecosystem:** The proposed method operates in isolation from other well-established advanced RAG techniques like hybrid search (combining dense and sparse retrieval), parent-child retrieval, or sophisticated reranking modules. Integrating Q-RAG's RL-based retriever into a more comprehensive, production-style pipeline could be beneficial, but this is not explored.

### Questions
- Given that creating datasets with ground-truth support facts is infeasible for most real-world applications, what is your proposed concrete path toward adapting Q-RAG to new domains where such supervision is unavailable?

- Your paper suggests using LLM-based rewards as a future direction. How do you plan to address the now well-established problem of "reward hacking," where RL agents learn to generate superficial "master keys" to fool LLM judges? Without a robust defense against this, wouldn't the training process collapse?

- Could you elaborate on the model's limitations in tasks requiring explicit compositional reasoning (e.g., "Which of these two products has a higher rating?" which requires retrieval of two facts and a comparison)? How does Q-RAG's performance depend on the reasoning capability of the final, frozen LLM used for generation?

- Have you considered the performance implications of integrating the Q-RAG retriever as the first stage in a more complex pipeline that includes, for example, a cross-encoder reranker? Would the greedy, step-by-step retrieval policy still be optimal in such a setting?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Q-RAG formulates multi-step retrieval as a MDP process where the state is the embedding representation of the query and past retrieved chunks, the action is the embedding of the candidate next chunk to retrieve, and Q is defined as the dot product of the state and the action embedding. Reward is defined as whether all ground truth chunks are retrieved and are propagated back via TD to train the state and action embedders. The main contribution of Q-RAG is to train a multi-hop retriever using temporal-difference RL. The authors also claim Q-RAG to be more efficient in training and inference, as only the retriever is trained and iteratively used during inference

### Strengths
1. The method is intuitively sound and novel, with theoretical justification of using dot product as an approximation for the Q-function in the appendix. 
2. The experimental setup is comprehensive with good coverage on multiple tasks requiring multi-step retrieval, spanning commonsense reasoning, NIAH, and multi-hop retrieval. 
3. Numerous baselines are included for different datasets, and the baselines are very recent. 
4. Good ablation study to demonstrate the necessity of various key design choices in Q-RAG.

### Weaknesses
1. In all experiments, the choice of the total number of retrieval steps T is very important because you need to know when to stop retrieving and have the reader LLM answer the question. However, the choice of T is not discussed at all. 
2. The authors claimed efficiency in training and inference. Though it is intuitively true, the discussion regarding training efficiency was insufficient and was not studied. Only the inference runtime was partially studied in Figure 3c. 
3. The experimental setups are pretty vague. What off-the-shelf models were used as the state embedder and action embedder? What reader LLM are used for different baselines for different datasets? Why the different choices? The paper can also benefit from brief introductions to each dataset and elaborating on what chunking strategy was used for each dataset for readers not familiar with them. 
4. Presentation: Figure 3c’s bottom has been cut off and typo “Babylon” on line 42

### Questions
see the weakness point 3

### Soundness
3

### Presentation
3

### Contribution
3
