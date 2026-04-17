# Not only a helper, but also a teacher: Interactive LLM Cascade

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Large Language Models (LLMs) vary widely in their capabilities, with larger models often having better performance but higher cost: choosing an LLM model often involves trading off performance and cost. The LLM Cascade is a paradigm that defers difficult queries from weak/cheap to strong/expensive models. This approach is nonadaptive: the deferral decision rule is trained or derived by algorithms offline. When confronted with similar or repeated queries, the LLM Cascade may then repeatedly consult the expensive model and incur higher cost. To improve the cascading efficiency, we propose Inter-Cascade, an online and interactive LLM Cascade that extends the role of strong model from a backup helper to a long-term teacher. In our system, when a strong model resolves a difficult query, it also distills its solution into a generalized, reusable problem-solving strategy that boosts the weak model on subsequent queries. Adding strategies to queries enables the weak model to dynamically improve its performance over time, avoiding computationally and time-intensive fine-tuning. Empirically, compared with standard LLM Cascade baselines across multiple benchmarks, the Inter-Cascade significantly improves the accuracy of the weak model (by up to 33.06 absolute percentage points) and the overall system (by up to 5.53 absolute percentage points), while reducing the calls to strong models (by up to 48.05% relative reduction) and saving the corresponding fees (by up to 49.63% relative reduction). Inter-Cascade demonstrates the effective in-context knowledge transfer between LLMs, and provides a general, scalable framework applicable to both open-source and API-based LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a new LLM Cascade method. The proposed method involves adding (top-k related) historical question-strategy-answer information to the prompt when querying a weak model, using the in-context learning methodology.

### Strengths
1) I believe the LLM Cascade method proposed in this paper will be very simple and effective. I find this idea interesting.

2) Secondly, the overall presentation of the paper and its figures are clear, especially in the Section 2 (although I suggest the authors separate all the theoretical derivations into a dedicated subsection).

### Weaknesses
**Weaknesses and Suggestions**  

1) For the experiment.  
(i) Many cascade methods could be used for comparison, but they are not included in the experiments. The comparison with baselines is significantly lacking. (ii) Additionally, the performance of the weak and strong models used separately should also be reported.    Some other concerns: Another issue is why more well-known datasets, such as MATH and GSM8K, were not used.

2) For the method.   
The authors overlook an important step: the introduction of the "strategy matching function", involving how multiple similarities are calculated to find the TOP-K samples.

2) For the related work.   
(i) I suggest that the authors add a brief presentation of the most core related works in the main text. (ii) Additionally, it is recommended that the authors provide a categorized analysis of related works, rather than presenting them in a continuous flow (the authors may refer to the survey "Harnessing Multiple Large Language Models: A Survey on LLM Ensemble, arXiv 2025" for the classification of cascade methods). (iii) Furthermore, it would be beneficial to include an introduction to related works on "in-context learning involving the retrieval of the most relevant samples". 

3) There are some unclear points and flaws in the abstract and introduction sections (though overall, the paper is easy to follow).    
(i) The description of existing cascade methods in the introduction is not sufficiently clear and is presented from a high-level perspective. The statement "the deferral decision is trained offline" in the abstract is not entirely accurate, as some cascade methods do not require training. 
(ii) The meaning and significance of "offline and online" for the proposed method are not clearly explained in the abstract and introduction, at least not in the abstract.
(iii) The explanation of the proposed method (in the second, third, and fourth paragraphs of the introduction) is not clear enough.  
I suggest that the authors provide a case in the introduction or method section to explain the proposed method, including the use of the in-context learning-based prompt. This will help readers understand the method quickly.

### Questions
Please refer to the above "Weaknesses and Suggestions".

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
This paper introduces Inter-Cascade, an online and interactive extension of the LLM Cascade paradigm. Unlike standard cascades—which statically defer uncertain queries from a weak (cheap) model to a strong (expensive) one—Inter-Cascade enables the strong model to act as a “teacher” by generating reusable, generalized problem-solving strategies whenever it resolves a hard query. These strategies are stored in a lightweight, similarity-based memory and retrieved to augment the weak model’s input for future similar queries, allowing it to solve them locally via in-context learning. This eliminates redundant calls to the strong model for recurring or similar tasks. The approach requires no fine-tuning, works with both open-source and API-based LLMs, and probably improves calibration and system accuracy while significantly reducing cost and strong-model usage.

### Strengths
(1)This method enables continuous, online improvement of the weak model without fine-tuning—critical for API-based or resource-constrained settings.

(2)This paper provides a formal analysis showing that strategy-augmented inputs improve the weak model’s confidence calibration, leading to provably tighter accuracy guarantees under the same risk tolerance.


(3)This method is  compatible with any existing deferral function, model size, or cascade depth, and works seamlessly with both open-weight and closed API models.

(4)This study addresses a real-world inefficiency: Directly tackles the waste in standard cascades caused by repeated handling of similar hard queries—a common scenario in real user traffic (e.g., recurring math or factual questions).

### Weaknesses
（1）The proposed Inter-Cascade framework relies on prompt augmentation via a strategy repository, but it does not investigate whether more active learning mechanisms (e.g., periodic supervised fine-tuning) could yield stronger or more sustainable improvements.

（2）Experiments are confined to four English-language, closed-ended benchmarks (mostly mathematical reasoning and factual MCQs), limiting generalizability.

（3）The "strategy" is vaguely defined as a sequence containing the query, answer, and “generalized ideas,” with no ablation to determine which components drive performance gains.

（4）The paper claims low computational overhead for retrieval but omits end-to-end latency measurements and real-world deployment considerations (e.g., memory growth, cold-start effects).

（5）The theoretical guarantee (Theorem 2.2) assumes idealized improvements (error reduced by factor epsilon, coverage increased by factor b) without estimating these parameters from data.

### Questions
Could similar gains of inter cascade be achieved with static retrieval-augmented generation (RAG) using a pre-built knowledge base?

### Soundness
3

### Presentation
3

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
The standard cascade defers difficult queries from a weak model to a strong model. This  work introduces an interactive loop: when the strong model handles a query, it also generates a "strategy" for solving it. This strategy is stored and retrieved for similar future queries to augment the weak model's input via in-context learning, thereby improving its  performance over time without explicit retraining. The authors present empirical results on four benchmarks, showing improvements in overall accuracy and a reduction in calls to the expensive strong model compared to a non-interactive cascade baseline.

### Strengths
1. The paper successfully engineers a system that effectively reduces reliance on expensive LLMs in a cascade setting. The empirical results, particularly the significant accuracy boost for the weak model on reasoning tasks and the reduction in API costs, demonstrate the practical utility of the proposed pipeline.
2. The framework's ability to allow the weak model to adapt and improve its performance online, without the need for costly fine-tuning, is a significant practical advantage. This  makes it applicable to scenarios involving proprietary, API-based models.
3. The analysis of how the method improves the weak model's confidence calibration is a valuable contribution. It demonstrates a deeper understanding of the system's dynamics beyond simple accuracy metrics, which is crucial for building reliable deferral systems.

### Weaknesses
1. The main weakness is that the paper fails to convincingly differentiate its contribution from a standard RAG system built on a dynamic knowledge base of past interactions. The core mechanism is retrieving textually similar past queries and their solutions to serve as few-shot prompts. This is not a fundamentally new idea.
2. The process for generating "strategies" is treated as a black box. The paper lacks crucial details on how the strong model is prompted to create solutions that are genuinely general and reusable, rather than just specific to a single query.
3. The cost analysis is superficial. It ignores the overheads introduced by the InterCascade framework itself: the token cost for generating the "strategy" (which is likely  longer than just an answer), the computational cost of the embedding and retrieval step for every query, and the storage costs of the growing repository.
4. The evaluation is heavily concentrated on structured mathematical reasoning problems. These tasks are particularly amenable to solution-pattern reuse. There is no evidence that the method is effective for more creative, nuanced, or structurally diverse tasks, which severely limits the claimed generality of the framework.

### Questions
1. What is the fundamental difference between your proposed "InterCascade" and a simpler baseline where a weak LLM uses a standard RAG module to retrieve from a dynamically updated vector database? Why was such a baseline not included in your evaluation?
2. Could you provide a formal cost for your framework that includes the token cost of strategy generation, the latency and computational cost of the retrieval, and the infrastructure cost of maintaining the repository? How do these overheads scale with query volume and repository size?
3. How do you expect your system to perform on tasks without clear, repeatable logical structures, such as creative tasks or multi-turn dialogue?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a method to improve the performance of LLM cascade systems in terms of cost and accuracy. In a cascaded LLM system, a series of increasingly strong LLMs are used for a single query, with the stronger LLMs employed only if the weaker LLM does not produce a confident enough answer. The work proposes storing the reasoning/strategy of the strong LLM into a repository. For a new query, the ‘k’ most relevant strategies from this repository are used as context to help the weak LLM. The authors experiment on reasoning and knowledge benchmarks and empirically show that the proposed method results in fewer queries being processed by the stronger LLM, while increasing the accuracy of the overall pipeline. This results in a decrease in the overall cost.

### Strengths
The proposed method is simple and intuitive. The solution of using stronger LLM’s reasoning as context for weaker LLM is interesting. The authors theoretically show that, for the proposed method, the risk tolerance $\alpha$ is reduced under certain assumptions. Empirical results consistently show that the proposed method results in higher overall accuracy and lower overall costs, along with increased accuracy for weaker LLM and fewer calls to stronger LLM.

### Weaknesses
The work lacks experiments. All experiments are performed with a single combination of just two LLMs (weak=Gemini-2.5, strong=GPT-3.5 turbo). The weak LLM’s performance is quite low and the gap between the weak and strong LLMs is high on all the math reasoning datasets. Would we still see significant improvements in accuracy and cost if the weak LLM performance is higher and the gap between the weaker and stronger LLMs is small? 

Experiments are also limited to three math/logical reasoning benchmarks and a knowledge benchmark. The GSM-Symbolic and GSM-Plus datasets contain repeated instances of math questions with nearly the exact same wording but with the variable values modified. The proposed method is expected to perform the best in such setups but this does not represent real-world applications. How well does the strategy retriever work when the queries are significantly different from each other? Would you need more than just top-2 strategies as context in such cases? How would that impact the performance of the model and the overall cost? How well does the approach work for datasets where the queries are quite diverse not just in terms of wording but in terms of strategy required to solve them and there are no repeated queries with the exact same strategy? How well does the method work on reasoning benchmarks apart from math/logical reasoning? Does the performance translate to multi-modal datasets? The current datasets are quite small (a few thousand samples). Would the retrieval quality and thus the overall performance suffer when scaled to much larger datasets?

The work requires more experiments with different benchmarks and LLMs are necessary to answer these questions.

### Questions
1. Provide results on additional benchmarks (e.g., the typically used 8 benchmark datasets for common-sense reasoning [a]) and more challenging benchmarks like Big-Bench Hard [b], MMLU-Pro [c] and AGIEval [d]. 
2. Provide results with different weak-strong LLM combinations including those with a good weak LLM performance and lower performance gap between weak and strong LLMs. 
3. How does the number of strategies retrieved (‘k’) affect performance and cost, particularly on benchmarks without repeated/highly similar queries?
4. Provide more discussion / specific examples of real-world applications where the proposed method would be useful (setups where the queries and solution strategies are highly similar to one another). 
5. The proposed method likely reduces the time required per query significantly. Provide results on throughput with and without the proposed approach.
6. How well does the method scale to really large and diverse datasets in terms of both efficiency and task performance? 


References:

[a] Hu, Zhiqiang, et al. "Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models." Proceedings of the 2023 conference on empirical methods in natural language processing. 2023. \
[b] Suzgun, Mirac, et al. “Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.” arXiv preprint arXiv:2210.09261, 2022. \
[c] Wang, Yubo, et al. “MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.” arXiv preprint arXiv:2406.01574, 2024.\
[d] Zhong, Wanjun, et al. “AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models.” Findings of the Association for Computational Linguistics: NAACL 2024, 2024, pp. 2299-2314.

### Soundness
2

### Presentation
3

### Contribution
3
