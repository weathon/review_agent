# MARAGE: Multi-Model Adversarial Attack for Retrieval-Augmented Generation Database Extraction

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models using externally retrieved data. Although useful, RAG can expose the retrieved data to extraction attacks. Previous work has demonstrated the feasibility of RAG extraction, but has failed to simultaneously ensure high-coverage retrieval of the entire knowledge database and verbatim extraction of lengthy retrieved data. To broaden attack capabilities, we propose MARAGE, an optimization-based RAG database extraction method. To obtain high-coverage retrieval, we propose a semantic-aware query selection strategy which optimizes the diversity of retrieved chunks from the database. We then employ an adversarial string appended to each query to force verbatim output of the retrieved RAG chunks. To adapt to the uniquely lengthy target of RAG databases, we introduce Primacy weighting, which prioritizes initial tokens in the optimization target to enhance the extraction capability of optimized adversarial strings. Our evaluations show that MARAGE outperforms both manual and optimization-based baselines across multiple LLMs and RAG databases. On an end-to-end RAG pipeline, MARAGE surpasses the manual attack baseline by 329 +- 165% in extracting RAG data. To understand why MARAGE is more effective than the baselines, we probe and analyze the model's internal state to isolate the benefit of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors propose an extraction attack that intends to extract verbatim data from the RAG database. The authors formulate the problem as a novel token optimization problem, and utilize continualized gradient descents to ease the discrete optimization problem. A delay-weight-based loss is adopted, observing that the first few tokens are more decisive, particularly for long sequence targets. The numerical results show high accuracy, large coverage, extension to long sequences, and evasive to existing defenses.

### Strengths
The proposed method is innovative. I particularly like equations (3) and (4) -- they are very beautiful. The study is also thorough and comprehensive, and the numerical studies are rich.

### Weaknesses
I'm not sure if cosine similarity is the most appropriate metric. If the basis of the embedding space has been selected, then it might be hard for a subsequent query to have absolutely small similarities with all selected query embeddings. Alternatively, a characterization of how dense or sparse the selected query embeddings are distributed across the embedding space might be another option. Did the authors consider a similar or any other metrics? If not, certain results might be helpful.

The experiment sample size is relatively small. For example, 50 targets are considered, and 100 queries are used to retrieve 300 chunks.

There is no theoretical development, but I guess this is beyond the scope of this paper (and I think this can be for another paper, given the rich contribution of this work).

### Questions
1. The authors mention that queries are selected strategically. Specifically, for each subsequent query, they evaluate the cosine similarity between the embedding of it and that of the existing query pool. I have one question and one suggestion about it. How large is the computational cost? It looks like the search cost can be up to n factorial where n is the number of data instances in the database.

2. The discrete optimization is not described clearly. I have two questions here:

(A) The authors mention that the algorithm by Wen et al. (2023) is adopted as the optimization foundation. It seems to me that a multi-model extension is proposed. However, it would be helpful if the authors could explain more clearly what their contribution is on top of the adaptation of Wen et al. (2023).

(B) Like the authors mentioned, not all instances of the continuous embedding values correspond to valid tokens. Since the gradients of the adversarial embeddings are computed. It is not clear to me how they are eventually converted back to the token level.

3. It is good to see that the proposed method is optimized over a series of models. However, I'm more concerned about the training over different d's. This boils down to two questions.

(A) In order for equation (4) to learn successfully, I guess a large optimization dataset D_p is needed. In Section 4.1, the authors mention that the size of D_p is 50. Is this sufficient? (Looks so based on the numerical results, but I'm still doubtful.)

(B) How's the generalization result? In other words, would a D_p of size 50 be enough for the attack model to verbatim recite any data instances from the data base?

4. A quick clarification: what's the retriever of the RAG? Based on the data description, is it true that the "pre-defined query/data pairs" already have data retrieved from the database accordingly?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
MARAGE attacks RAG systems by (1) selecting semantically diverse queries to retrieve broad database coverage and (2) appending optimized adversarial strings that force LLMs to output retrieved data verbatim. Authors propose primacy weighting (exponentially decaying loss components) that prioritizes early tokens, exploiting autoregressive generation where starting correctly makes continuation likely. Authors show strong extraction rates (47-80% exact match) and  compares against basic  gradient baselines.

### Strengths
1.Proposes exponential decay for weighting loss terms is well-motivated for long targets and demonstrating higher success than prior approaches (Pleak).

2. Extensive evaluation across multiple models and datasets with analyzing attack success in presence of defenses.

### Weaknesses
1. The paper does not justify on why a new optmization technique is needed for this? There has been a long line of work on jailbreaks and prompt injection based techniques along side GCG like search-based methods and RL approaches.

### Questions
The paper introduces a new method without first establishing that existing techniques are inadequate. Beyond GCG and Pleak, numerous established adversarial approaches—search-based methods, RL-based optimization, and LLM-guided generation—remain untested despite being applicable to this problem. It would be better to systematically pick some of these representative methods, identifying their specific limitations for long-target RAG extraction, and then justifying the new approach.

### Soundness
3

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
This paper proposes MARAGE, an extraction attack aiming to extract data stored in RAG databases. MARAGE increases the diversity of extracted content via strategic query selection, improves the transferability of the adversarial query by averaging losses over multiple models, and enhances exact extraction of long chunks by incorporating decaying weights into the loss objective.

### Strengths
1. The motivation and main method are clearly explained and easy to follow. 
2. The experimental section is thorough, and the additional details in the Appendix are appreciated. 
3. The explanation on using a weighted loss that emphasizes earlier tokens is helpful.

### Weaknesses
1. The overall contribution is incremental. Many components of the proposed approach resemble existing techniques. For example, 

   - Diversity enhancement is achieved using cosine-similarity-based measures, which is similar in spirit to a number of prior works using cosine similarity for diversification, e.g., [1, 2]. 

   - The second contribution, approximating discrete optimization by continuous optimization, follows existing work [3] as authors admitted. 

   - The fourth contribution, evading perplexity-based defense, is achieved by adding a perplexity penalty into the objective. Such penaltized optimization strategy is widely used in ML community. 

   - Similarly, improving transferability by aggregating loss across multiple models/targets is also a widely used practice.

2. Some clarification questions. 

   - Could you please provide an example showing the adversarial query derived by MARAGE?

   - What is the underlying intuition for why MARAGE can successfully evade system prompts? How would MARAGE perform if the system prompt or the design of LLM implementing a rule to reject reporting exact text in the prompt? E.g., involving a step that checks the similarity before returing the final answer? 

   - The paper reduces the default number of greedy token evaluations from 512 to 16 for baseline attacks. What happens when this value is increased for the baselines?

[1] Wang Z, Bi B, Luo Y, et al. Diversity Enhances an LLM's Performance in RAG and Long-context Task[J]. arXiv preprint arXiv:2502.09017, 2025.  

[2] Gao H, Zhang Y. Vrsd: Rethinking similarity and diversity for retrieval in large language models[J]. arXiv preprint arXiv:2407.04573, 2024.

[3] Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery.

### Questions
Please see Weaknesses above.

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
2

### Summary
This paper presents MARAGE, an optimization-based adversarial attack against RAG systems. Unlike prior prompt-leaking or manually crafted attacks, MARAGE formulates RAG extraction as a continuous embedding optimization problem to learn transferable adversarial suffixes that induce LLMs to reproduce retrieved database content verbatim. It introduces (1) a semantic-aware query selection strategy for high database coverage, (2) primacy weighting to handle long RAG chunks, (3) multi-model gradient aggregation for transferability, and (4) a perplexity-constrained loss to evade perplexity-based defenses. Experiments across five LLMs and multiple RAG datasets show strong improvements over GCG, Pleak, and manual baselines.

### Strengths
1. The paper is clearly written and well-organized, with extensive experiments, ablation studies, and analysis supporting the claims.
2. MARAGE is the first optimization-based RAG extraction attack and achieves significant improvements over strong baselines such as GCG and Pleak. Beyond quantitative results, the paper provides mechanistic insights (e.g., layer-wise probing, primacy weighting effects) that help explain why the proposed optimization approach works.

### Weaknesses
1. The experiments rely on relatively old and mid-sized open-source models (LLaMA-3, Vicuna, OPT, etc.). Modern RAG deployments often use stronger open-source model (Qwen3, DeepSeek, etc.) or closed-source models (GPT-5, Gemini, etc.). Without evaluating on or at least discussing these stronger and more representative models, it remains unclear whether MARAGE scales to or transfers across the modern RAG system.
2. The paper primarily studies a single-turn retrieval pipeline with static retrievers, whereas real-world RAG systems often employ agentic or multi-stage retrieval strategies, such as query reformulation, iterative reasoning, tool use, etc.
These dynamic retrieval behaviors can alter both the exposure of database content and the model’s susceptibility to adversarial suffixes.
Evaluating or simulating MARAGE under such agent-based, reasoning-enhanced RAG pipelines would provide stronger evidence of practical relevance and robustness.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
