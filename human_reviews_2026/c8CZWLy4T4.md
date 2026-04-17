# Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 2

## Abstract
Retrieval-Augmented Generation (RAG) is a framework for grounding Large Language Models (LLMs) in external, up-to-date information. However, recent advancements in context window size allow LLMs to process inputs of up to 128K tokens or more, offering an alternative strategy: supplying the full document context directly to the model, rather than relying on RAG to retrieve a subset of contexts. Nevertheless, this emerging alternative strategy has notable limitations: (i) it is token-inefficient to handle large and potentially redundant contexts; (ii) it exacerbates the “lost in the middle” phenomenon; and (iii) under limited model capacity, it amplifies distraction, ultimately degrading LLM output quality. In this paper, we propose LDAR (Learning Distraction-Aware Retrieval), an adaptive retriever that learns to retrieve contexts in a way that mitigates interference from distracting passages, thereby achieving significantly higher performance with reduced token usage compared to long-context approaches. Extensive experiments across diverse LLM architectures and six knowledge-intensive benchmarks demonstrate the effectiveness and robustness of our approach, highlighting the importance of balancing the trade-off between information coverage and distraction.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper address the distracting issue caused by RAG. They propose a method called LDAR which learns a lightweight retriever that learns to select passages to minimize potential interference from distracting passage. Their retriver operates on the cosine similarity distribution and selects a dynamic set of passages from a contiguous quantile interval to balance coverage and distraction. For example, if the risk of having distraction is higher, retrieves passages from a narrow quantile interval, minimizing the risk of having distraction. The retriever is trained with the reward signal of whether the downstream LLM gives a correct answer using the retrieved passages. Their expeirments on various benchmarks show significantly better results than the baseline methods.

### Strengths
1. The motivation of this paper is solid, learning to balance coverage and distraction for RAG is an important question.
2. The proposed method is lightweight by only finetuning a retriever model.
3. The experimental setups are comprehensive, the zero-shot experiment is persuasive and the performance is good.

### Weaknesses
1. An analysis of the proposed method is needed. For example, it would be interesting to compare with a baseline that learns to rerank the retrieved passages alone with the same optimization objective.
2. The interpretability of the addressed distaction issue is limited. It would be interesting to see how the proposed retriever balance coverage and distraction, some case studies would be helpful.

### Questions
1. what is the passages ratios compared with the initial top-N because LDAR quantile cutting step.
2. For more capable or larger LLMs, is the quantile cutting still useful compare with using long-context.

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
This paper presents LDAR(Learning Distraction-Aware Retrieval), a method for adaptive retrieval that compellingly addresses the performance-cost tradeoff between RAG and full long-context. The core contribution which involves identifying the problem of LLMs getting incorrect results given both gold passage and distraction passage, and a lightweight, band-based retriever that is sound and justified well againts the "distraction problem"(Figure 3 , Section 4.1). The experimental evaluation is thorough, demonstrating LDAR's effectiveness and adaptability across a wide range of modern LLMs and challenging tasks. However, the method's design, which feeds raw similarity scores to a Transformer , raises significant concerns about overfitting to task-specific patterns, a limitation the authors themselves acknowledge. This casts doubt on the true generalizability claimed in the zero-shot experiments (see Section 5.5 ). Other key weaknesses include lack of ablation studies and justification for key architectural choices the complete omission of the method's training cost.

### Strengths
## Originality
The paper's primary originality lies in its precise and insightful problem formulation. It moves beyond the simplistic "RAG vs. Long-Context" debate to investigate why RAG systems fail. The core insight, clearly demonstrated in Section 3 and Figure 2, is that performance degradation is not just from irrelevant passages but that even a set of exclusively gold passages can collectively confuse an LLM and lead to incorrect answers. Identifying this "distraction" effect, even among relevant documents, is a novel and important contribution.

The proposed solution, LDAR, is also original. Instead of relying on complex textual rerankers or fine-tuning the LLM, it introduces a lightweight, "score-only" adaptive retriever. The "band-based" retrieval strategy (Section 4.1) is a creative mechanism to make the reinforcement learning problem tractable, cleverly reducing the search space from a complex combinatorial problem (as in Bernoulli sampling) to a low-dimensional, continuous one.

## Quality
The empirical quality of this work is a major strength. The authors validate LDAR with a thorough and rigorous experimental setup (Section 5).

1. Comprehensive Benchmarking: The method is tested across a diverse and modern set of both open-source (Llama-3.1, Qwen-2.5, Mistral-Nemo) and closed-source (GPT-4o, Gemini-2.5) models, demonstrating broad applicability.

2. Challenging Tasks: The evaluation spans six knowledge-intensive benchmarks, including the difficult LaRA tasks (Location, Reasoning, Comparison) and HELMET's long-context adaptations of HotpotQA and NQ.

3. Strong Results: The results in Table 1 are compelling. LDAR consistently achieves a superior performance-to-cost ratio, significantly outperforming the full long-context (LC) baseline in overall accuracy while using substantially fewer tokens (often achieving ~2x token efficiency).

4. Well-Justified Method: The core design choice of "band-based" retrieval is well-justified empirically by the comparison in Figure 3, which clearly shows it finds a more stable and effective trade-off than the Bernoulli-based alternative.

## Clarity
The paper is exceptionally well-written, organized, and easy to follow, earning a top score for presentation. The authors use visualizations effectively to build a clear and logical narrative.

## Significance
This paper addresses a highly significant and practical problem for the LLM community. As context windows expand to 1M tokens, the question of how to best utilize this vast context for optimal cost-performance is of paramount importance. This work provides a practical and effective solution that moves "beyond" the binary choice of RAG vs. full context.

### Weaknesses
## Serious Concerns Regarding Generalization and Overfitting
- The model's score-only policy may overfit similarity-shape statistics of a corpus. LDAR uses only the cosine-similarity distribution (no text), and explicitly restricts access to textual content for scalability. While elegant, this design risks learning corpus-specific distributional quirks rather than semantic robustness, limiting the proposed method's contribution.

- Zero-shot section (Sec. 5.5) lacks mechanism analysis. Authors transfer LaRA-trained policies to HELMET (HotpotQA/NQ) and report modest gains, but do not explain why a Comparison-trained policy improves HotpotQA or why a Location-trained policy improves NQ. Given that different tasks have vastly different retrieval needs, this positive result may be a coincidence.

- The "score-only" architecture is also vulnerable to overfitting to the statistical properties of the passage database (e.g., LaRA's mix of novels and finance reports). The paper provides no evidence that a policy learned on this database distribution would generalize to a database with a different profile (e.g., social media, legal documents). The test on HELMET's Wikipedia database (Section 5.5) is not a sufficient stress test and lacks a deep analysis of the distribution shift.

## Missing Ablation Studies for Key Architectural Choices
- The paper employs a Transformer Encoder to process the sequence of similarity scores. This is a non-trivial architectural choice, implying that the relationships between scores are important. However, this is not justified with a baseline, such as a simpler Multi-Layer Perceptron (MLP), to prove this complexity is necessary.
- The model uses "Periodic Embedding" to encode the raw scalar similarity scores. This is a specific and non-obvious design choice. The paper provides no justification or ablation study comparing this to simpler, standard alternatives (such as learnable embedding with truncation of sorted scores).

## Omission of Training Cost and Practicality
- The paper's central claim is improved inference efficiency (i.e., lower token usage). However, it provides no information whatsoever about the training cost. The RL training loop, as formulated in Eq. (3), requires a forward pass of the (potentially massive) pretrained LLM $F_{\psi}$ to obtain the reward signal $r_{\psi}$ for every gradient step. This suggests the training process may be prohibitively expensive.
- The paper provides no data on the number of LLM calls required for convergence or the total wall-clock training time. This omission makes it impossible to assess the method's practical viability or cost-benefit trade-off.

### Questions
Please list up and carefuly describe any questins and suggestions for the authors.Think ofthe things where a response from the author can change youropinion, clarify a confusion or address a limitation. This is important for a productive rebuttal and discussion phase with the authors.

- Can you extend Sec. 5.5 with broader zero-shot tests? Add more tasks (train/test task cross-matrix) and more corpus database.
- Have you done any ablation on the necessity of choosing transformers as the backbone of the retriever?
- Could you provide a clear training cost-benifit analysis?

### Soundness
3

### Presentation
3

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
The paper Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Grounding LLMs studies the trade-off between retrieval-augmented generation (RAG), which may miss relevant information, and long-context models, which can include too much irrelevant content. It introduces LDAR (Learning Distraction-Aware Retrieval), a lightweight retrieval method that adaptively selects a contiguous range of passages from the ranked candidates based only on their similarity scores, achieving a balance between information coverage and distraction. LDAR uses a small transformer trained with policy gradients to predict how wide the retrieval range should be for each query, optimizing task performance according to the LLM’s context capacity. Experiments on multiple benchmarks and models show that LDAR achieves higher accuracy than both RAG and long-context approaches while using fewer tokens. The main contributions are a learning-based distraction-aware retrieval framework, the banded retrieval design that improves stability and generalization, and extensive validation demonstrating efficient grounding for LLMs.

### Strengths
1) The paper proposes a dynamic interval (band) method to select high-quality context from a passage sequence sorted by cosine similarity.

2) The motivation section analyzes in depth how different retrieval strategies, when applied over a cosine-sorted list, affect answer correctness, and it systematically frames the problem of adaptively balancing recall versus distraction.

3) The evaluation covers a reasonably broad set of settings on four LaRA tasks (location/reasoning/comparison/hallucination) and the long-context variants of HotpotQA/NQ from HELMET.

### Weaknesses
1) The analysis experiments for the method itself are not sufficient (see Questions).

2) The method appears sensitive to LLM size, which affects cross-model generalization and thus reduces transferability and robustness across heterogeneous systems.

3) There is a lack of concrete case studies: results are mainly aggregate metrics and curves, without step-by-step comparisons on representative examples.

### Questions
1) Do different retrievers induce different cosine-similarity distributions, and would this shift the behavior/performance of the adaptive retriever?

2) Using only accuracy as the reward seems limited. Since the token usage ratio is also reported, can the optimization include a cost term (e.g., inference/compute budget)?

3) In the visualization of retrieval intervals (Fig. 6), most learned bands overlap with the top-k region. For bands outside the top-k, can you provide concrete examples to intuitively explain why passages outside the top interval are selected/beneficial (or why the top interval is insufficient)?

4) Is selecting a continuous interval (band) inherently reasonable? Does it implicitly assume that high-quality passages are locally clustered in the cosine-sorted order? If possible, please add a fixed-length sliding-window experiment: slide a window of length (w) over the cosine-sorted sequence, evaluate accuracy for each window as the retrieved set, examine whether stable peak regions exist, and compare their alignment with the LDAR-learned band.

### Soundness
3

### Presentation
2

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
This work investigates the noisy information in the context of RAG, and introduces LDAR (learning distraction-aware retrieval) that learns to select passages to minimize interference from distracting passages. The training framework invites the target LLM in-loop to refine the retrieval model. The approach shows effective on the LaRA benchmark compared with several baseline settings on Llama 3.1 8B, Llama 3.2 3B, Qwen 2.5 7B, Qwen 3 4B, Mistral-Nemo-12B, GPT-4o, GPT-4o-mini, Gemini-2.5-pro, and Gemini-2.5-flash. Some interesting findings are shown, including the more capable models (e.g., GPT-4o and Gemini 2.5) perform well on the long-context setting.

### Strengths
* This work is well-motivated and easy to follow.

* The proposed method can improve the performances for a range of LLMs. 

* The proposed method, which does not require the internal information such as attention values of the target LLM, is model-agnostic and can be applied with any LLMs including the closed ones such as GPT and Gemini.

### Weaknesses
* This work should be compared with other reranking methods, which are important related work to address. However, in the current manuscript, none of the reranking methods such as RankRAG is mentioned and compared. 

* The experimental setting of the baseline methods is not comprehensive. The number of passages in RAG, LC, BMG, and Adaptive-k are not clear. Number of passages greatly impact the end performance. Without an analysis on the passage size, the contribution of the proposed method is hard to clarify. 

* The proposed method adds computation cost during inference. An analysis of the runtime could be added.

### Questions
* What is the basic retrieval model (Top-k) in the experiments? 

* What is the exact long-context setting in the experiments? The entire dataset or top-k with a very large k?

* From Table 2, LC looks powerful when the LLMs are capable. LC can be speed up with prompt cache (Cache-Augmented Generation). Could you compare the method with CAG in terms of accuracy and efficiency?

### Soundness
1

### Presentation
2

### Contribution
2
