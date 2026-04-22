# Search or Think? Rethinking Iterative RAG from An Entropy Perspective

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for Large Language Models (LLMs) to address knowledge-intensive queries requiring domain-specific or up-to-date information. To handle complex multi-hop questions that are challenging for single-step retrieval, iterative RAG approaches incorporating reinforcement learning have been proposed. However, existing iterative RAG systems typically \textit{think first} to decompose questions without leveraging information about the available retrieval corpus, leading to inefficient retrieval and reasoning chains that cascade into suboptimal performance. In this paper, we introduce Search-Initialized Thinking (SIT), a novel framework that \textit{searches first} before think in iterative RAG systems with contextually relevant retrieved knowledge. From an entropy perspective, we demonstrate that incorporating initial knowledge with search reduces unnecessary exploration during the reasoning process, enabling the model to focus more effectively on relevant information subsets. Extensive experiments on six standard RAG datasets demonstrate that by establishing a stronger reasoning foundation, SIT significantly improves retrieval precision, reduces cascading errors, and enhances both performance and efficiency. Moreover, SIT proves effective as a versatile, training-free inference strategy that scales seamlessly to large models.Generalization tests across diverse datasets and retrieval corpora confirm the robustness of our approach. Overall, SIT advances the state-of-the-art in iterative RAG systems while illuminating the critical interplay between structured reasoning and efficient exploration in reinforcement learning-augmented frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Search-Initialized Thinking (SIT), a simple modification to reinforcement-learning-based iterative Retrieval-Augmented Generation pipelines. Instead of letting the LLM begin reasoning from scratch ("Model-Initialized Thinking"), SIT performs a single retrieval step using the original question to inject an initial set of top-k passages before the first think step. The authors argue that this grounds early planning, reduces cascading retrieval failures, and—viewed through an entropy lens—lowers policy entropy during GRPO training, yielding more efficient exploration. Experiments on Search-R1 and Graph-R1 backbones across six QA datasets show consistent gains in EM/F1 and retrieval recall, with ablation confirming the value of the initial search.

### Strengths
- SIT delivers large, consistent gains across two RL-based iterative RAG frameworks (Search-R1, Graph-R1), two retrieval corpora (full Wikipedia chunks vs. dataset-specific hypergraphs), and six diverse datasets, including OOD generalization tests.
- The method adds only one extra retrieval before the first think step and a minor prompt change; it is model-agnostic, training-free beyond the baseline RL loop, and compatible with any dense retriever.

### Weaknesses
- Performing an initial retrieval on the raw question is a standard preprocessing step in many multi-stage RAG systems (e.g., query rewriting + retrieval); the contribution reduces to “do it before the first think token in an RL loop,” which feels incremental.
- No ablation on k, retriever quality, or noise in P₀ is shown, leaving open whether gains hold under mismatched or weak initial retrieval.

### Questions
- Does SIT still help if the initial retrieval is performed with a mismatched chunk-based retriever (e.g., E5 on the graph corpus)? Provide numbers.
- Table 2 shows Search-R1+SIT slightly hurts NQ EM (-3.97); what specific failure cases cause this regression, and can they be diagnosed from the retrieved P₀ passages?
- The prompt in Table 1 forces the model to “first conduct reasoning inside \<think\>…\</think\> relied on the initial knowledge”; if the initial P₀ is irrelevant or noisy, does the model still emit a \<query\> on the first turn, or does it hallucinate a plan anyway?

### Soundness
2

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
This paper addresses a critical limitation of existing iterative Retrieval-Augmented Generation (RAG) systems: their "think-first" paradigm, which decomposes questions without leveraging the available retrieval corpus, leading to inefficient exploration, cascading errors, and suboptimal performance. To solve this, the authors propose **Search-Initialized Thinking (SIT)**, a framework that prioritizes an initial retrieval step to gather contextually relevant knowledge before the model begins reasoning in iterative RAG systems optimized via reinforcement learning (RL).  

From an entropy perspective, the paper demonstrates that SIT reduces unnecessary exploration during reasoning by grounding the model in initial retrieved knowledge, allowing it to focus on relevant information subsets. Extensive experiments on six standard RAG datasets (e.g., HotpotQA, NQ, TriviaQA) validate SIT’s effectiveness: it significantly improves retrieval precision (measured by Retrieval Similarity, R-S), reduces cascading errors, and enhances both answer accuracy (via EM and F1 scores) and training efficiency (achieving strong performance in 300 steps vs. 1000 steps for baseline methods). Generalization tests across in-domain (IND) and out-of-domain (OOD) datasets further confirm SIT’s robustness. The work also advances understanding of the interplay between structured reasoning and efficient exploration in RL-augmented RAG systems.

### Strengths
1. **Comprehensive validation**: Experiments cover diverse datasets (6 standard RAG benchmarks), backbones (Search-R1, Graph-R1), RL algorithms (PPO, GRPO), and retrieval corpora (full Wikipedia, structured subsets). This breadth ensures SIT’s robustness and generalizability, not just niche performance.  
2. **Efficiency and practicality**: SIT achieves better performance with fewer training steps (300 vs. 1000 for baselines) and requires only a small modification to existing pipelines. This makes it easy to adopt and valuable for industrial applications.

### Weaknesses
1. The method proposed in this paper seems to be merely a minor adjustment to the implementation details of iterative retrieval, without presenting a exciting new approach. 
2. The training conducted in this paper is mainly based on models with fewer than 10 billion parameters. However, such models have weak reasoning and planning capabilities, so the generated queries may not be superior to the original questions. Therefore, for models with stronger capabilities and larger parameter scales, the method proposed in this paper may not bring significant improvements.
3. The detailed breakdown of the "turns" count presented in Table 5 is not provided. Notably, the number of exploration turns of LLMs is not equivalent to the number of retrievals. This is because there is a default retrieval step conducted before the LLM is allowed to perform exploration. Therefore, it remains questionable whether the method proposed in this paper can actually lead to a reduction in the total number of retrieval operations.

### Questions
1. Could experiments be conducted on models with larger parameter sizes and stronger reasoning capabilities?
2. Could you explain in detail the relationship between "turns" and the number of retrieval operations in Table 5?

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
3

### Summary
This paper works on iterative RAG with an RL-based method. Specifically, unlike existing method that forms the generation schema with LLM thinking as the first step, this paper proposes to first include initial retrieved knowledge as background knowledge before thinking. Experiments show that such a modification can improve the baseline performance by a large margin. Analysis also shows the final trained model has better retrieval quality with fewer turns.

### Strengths
- The motivation makes sense, as some background knowledge would be useful for the LLM to reason and proceed
- Experimental results are strong comparing with multiple baselines in different setting
- Paper writing is clear to me

### Weaknesses
- The technical contribution is rather incremental to the existing works

### Questions
I am interested in the total token consumption of the proposed method. Although the thinking turns can be fewer, the total token consumption can be similar to baseline given the added initial knowledge?

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
4

### Summary
This paper proposes Search-Initialized Thinking (SIT), a method to improve iterative RAG. In the traditional “think-before-search” framework, the model may generate wrong plans and cause later retrieval to go in the wrong direction. SIT adds an initial retrieval step before the first thinking step to provide external knowledge as a starting point. Experiments show that after RL training, SIT can significantly improve retrieval relevance and final answer accuracy.

### Strengths
1. The problem that the “initial thinking” step may easily go off direction due to the lack of corpus awareness is meaningful. 
2. The proposed SIT method is simple and clear.
3. Experimental results are stable and consistent. The method shows effectiveness across multiple datasets and settings.

### Weaknesses
1. Optimization blind spot of the initial retrieval. The first retrieval in SIT is generated by a fixed retriever. Its quality greatly affects later reasoning but cannot be optimized. The effectiveness of SIT under stronger models or better optimization is unclear.
2. Lack of necessary experiments. The proposed SIT can be used as a training-free strategy. Showing its performance would better demonstrate the effectiveness of the method. The training-free setting could also be used to test larger models, which would improve the paper’s credibility.
3. Scalability issue. SIT largely depends on the quality of the initial retrieval. Meanwhile, plan errors depend on the base model’s ability. When scaling to larger models (with fewer reasoning errors) or harder tasks (where retrieval noise increases), the proposed method might become less effective.

### Questions
1. Adding the initial retrieval increases information and reduces uncertainty, but this is not entiely consistent with §6.1, where token entropy measures the uncertainty of the model’s generation distribution.
2. In §6.2, the reported “average rounds reduced from 3.26 to 2.22”. does this exclude the initial retrieval step?
3. Except for the early training stage, why does retrieval performance almost not improve for either method?

### Soundness
2

### Presentation
3

### Contribution
2
