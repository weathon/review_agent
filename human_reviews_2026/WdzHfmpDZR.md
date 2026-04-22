# Training-Free Speedup for Retrieval-Augmented Generation with Staged Parallel Speculation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Retrieval-augmented generation (RAG) leverages external knowledge bases to enhance the quality of answers produced by large language models (LLMs). However, retrieving relevant documents from large-scale databases can be time-consuming, and existing RAG methods primarily focus on improving accuracy while often overlooking latency. In this paper, we introduce \textit{Staged Parallel Speculation (SPS)}, a training-free RAG framework that achieves substantial latency reduction without sacrificing answer quality. Unlike prior approaches that rely on task-specific training or model modifications, SPS is a plug-and-play method that requires no changes to the underlying models. Our framework enables the inference and retrieval systems to run in parallel during staged retrieval, thereby eliminating frequent pauses in the inference process and significantly accelerating generation. Furthermore, at each retrieval-generation stage, SPS first uses a model to generate multiple candidate answer chunks in parallel and then selects the most reliable output based on self-consistency among the candidates, thereby further improving answer quality. Extensive experiments across multiple benchmark datasets show that SPS consistently surpasses training-free RAG baselines by achieving higher accuracy with at most 57\% lower latency, while still reaching 96\% of the performance of finetuning-based methods, making it a practical choice for deployment in latency-sensitive applications such as agentic systems, enterprise knowledge management, or healthcare support.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes parallel generation to accelerate the generation process in RAG pipeline.

### Strengths
* Novel idea of retrieving the documents on-the-fly.
* Contains latency evaluation
* Higher accuracy in evaluation

### Weaknesses
* Evaluation focuses on QA dataset.
* Missing motivating examples and some technical details.
* No throughput evaluation.

### Questions
* What is the need to change the retrieved document on the fly instead of simply retrieve multiple documents related to the query? Is there a motivating example? I understand that context may shift during generation process, but the shifted context is still answering the original query and thus still have high semantic similarity to the original query.
* How do you perform the generation M.generate(C, sj)? Are you customizing the chat template to incorporate sj, or just directly append sj to the original answer? This is important because different ways of inserting sj greatly effects the efficiency of underlying inference engine (e.g. if you insert sj at the end of the generation, the generation quality may drop (as you are forcing the LLM to continue completing sj) but it is very fast because it can leverage the prefix cache).
* What about question answering throughput? I am mentioning this because it seems to me that your approach is trading throughput for latency. If that's the case, make it explicit, if that's not the case, would love to hear more about the rationale.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Staged Parallel Speculation (SPS), a framework for RAG that achieves latency reduction by running inference and retrieval in parallel without additional training. SPS runs retrieval periodically after a fixed number of generated tokens and clusters the retrieved documents by content similarity, sampling one from each cluster to maximize diversity while reducing context length. It then generates answer candidates in parallel and selects the final answer leveraging self-consistency. It reduces RAG latency by 57% while reaching 96% of the accuracy of finetuning-based methods.

### Strengths
- Latency reduction of RAG applications without additional fine-tuning.

### Weaknesses
- The accuracy is not as good as fine-tuning-based methods (Table 1)
- The efficiency comparison against Speculative RAG is not fair since the model size is different.
- The efficiency evaluation is limited to the case of batch size 1. It is unclear whether the proposed method offers a throughput benefit in a batching scenario, which is common in data center serving scenarios.

### Questions
- What is the overhead of the proposed method, including additional computation and memory requirements due to additional LLM and retrieval calls?
- The SPS algorithm seems overly complex, and it is unclear how much each component contributes to the overall performance. Can you provide an ablation study, such as disabling part(s) of the algorithms (clustering, self-consistency selection, staged retrieval, etc) and other hyperparameters, such as retrieval amount?
- How good is the SPS's performance compared to other efficiency optimization methods for RAG, including PipeRAG and many other works [1-5]?

### References

- [1] https://arxiv.org/abs/2404.12457
- [2] https://arxiv.org/abs/2405.16444
- [3] https://arxiv.org/abs/2502.20969
- [4] https://arxiv.org/abs/2503.14649
- [5] https://arxiv.org/abs/2505.07833

### Soundness
2

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
2

### Summary
The paper addresses the latency issue in RAG systems. In a standard RAG setup, you have a LLM that, given a query, first retrieves documents from a knowledge base then uses those documents as context to generate an answer. Retrieving and then waiting for generation introduces pauses and idle time. The authors propose a new training-free framework SPS which is designed to significantly speed up RAG inference while preserving accuracy. Different from standard RAG, the retrieval system runs in parallel with the generation system so that the generation process does not wait for retrieval as a blocking step. On multiple QA benchmarks, SPS shows lower latency compared to other training-free RAG baselines, while achieving similar accuracy with fine-tuned RAG methods.

### Strengths
* Unlike other RAG acceleration methods such as SpecRAG, SPS requires no model retraining or fine-tuning. This makes it immediately deployable on existing pipelines and models, which is a major practical advantage.
* SPS approaches the problem with overlapped retrieval and generation, eliminating the “wait gap” between the two.
* Experimental results show large and consistent latency reduction.

### Weaknesses
* While the paper frames SPS as a training-free speedup for RAG, the core idea is conceptually close to prior papers such as PipelineRAG and SpecRAG. I recommend adding more detailed evaluation and discussion with identical retrievers, LLMs and datasets and report wall-clock latency as well as accuracy.
* The paper does not report detailed ablation studies. It would be beneficial to show the performance without clustering; varying candidates and stage length; swap the self-consistency selector with others.
* Results primarily focus on mid-sized open LLMs. It remains unclear whether SPS scales to larger models or interacts differently with stronger retrievers.
* The paper only reports aggregate latency but lacks a decomposed timeline showing where savings arise. Without breakdowns, it is difficult to assess the mechanism’s effectiveness.

### Questions
N/A

### Soundness
2

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
4

### Summary
The paper introduces Staged Parallel Speculation (SPS), a training-free framework designed to improve the efficiency of Retrieval-Augmented Generation (RAG). SPS addresses the latency inherent in standard RAG systems by decoupling the retrieval and inference processes, allowing them to run in parallel. The framework employs a staged approach where retrieval for future text chunks occurs concurrently with the generation of current chunks.

Furthermore, SPS utilizes multi-perspective sampling to cluster retrieved documents into diverse subsets. It then generates multiple candidate answer chunks in parallel using these subsets. A key component is the training-free self-consistency selection mechanism, which identifies the most reliable candidate chunk based on semantic similarity to other candidates without requiring a trained verifier model. Experiments across five benchmarks (PopQA, TriviaQA, PubHealth, ARC-Challenge, ALCE-ASQA) demonstrate that SPS can achieve higher accuracy than standard RAG while offering significantly lower latency than other training-free speedup methods like Speculative RAG.

### Strengths
- The proposed method is training-free, making it a "plug-and-play" solution that can be easily adopted into existing RAG pipelines without the expensive and complex process of fine-tuning specialized drafter or verifier models.

- SPS demonstrates a compelling trade-off between latency and accuracy.

- Effectively combining staged parallel retrieval with a training-free self-consistency verifier for RAG chunks is an effective approach. The ablation studies (Table 3) validate the contribution of each component (staged retrieval, self-consistency, and multi-perspective sampling).

- The paper is well-organized. Figure 1 provides a clear visual contrast between sequential Self-RAG and parallel SPS , and Algorithm 1 succinctly describes the operational flow.

### Weaknesses
- Computational overhead. The claims regarding "efficiency" primarily refer to wall-clock latency, achieved through massive parallelization, not necessarily computational efficiency. For example, the experimental setup uses $m=5$ document subsets, meaning SPS requires launching 5 parallel endpoints of the generation model for every single query.

- Given that SPS uses 5x the inference compute (parallel streams) of Standard RAG, the comparison might be considered unfair in terms of total resources. For example, a more rigorous baseline would be "Standard RAG + Self-Consistency," where Standard RAG is also allowed to generate 5 paths (perhaps with temperature sampling or different retrieval subsets) and use the same voting mechanism.

- In abstract, the authors mention "Extensive experiments across multiple benchmark datasets show that SPS consistently surpasses training-free RAG baselines by achieving higher accuracy with 57% lower latency" - however, in Table 2, SPS's latency is similar to standard RAG. To make the claim more accurate, I would suggest using expressions like "... **at most** 57% lower latency".

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
