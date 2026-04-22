# MetaEmbed: Scaling Multimodal Retrieval at Test-Time with Flexible Late Interaction

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 6, 8, 8, 6

## Abstract
Universal multimodal embedding models have achieved great success in capturing semantic relevance between queries and candidates. However, current methods either condense queries and candidates into a single vector, potentially limiting the expressiveness for fine-grained information, or produce too many vectors that are prohibitively expensive for multi-vector retrieval. In this work, we introduce MetaEmbed, a new framework for multimodal retrieval that rethinks how multimodal embeddings are constructed and interacted with at scale. During training, a fixed number of learnable Meta Tokens are appended to the input sequence. At test-time, their last-layer contextualized representations serve as compact yet expressive multi-vector embeddings. Through the proposed Matryoshka Multi-Vector Retrieval training, MetaEmbed learns to organize information by granularity across multiple vectors. As a result, we enable test-time scaling in multimodal retrieval where users can balance retrieval quality against efficiency demands by selecting the number of tokens used for indexing and retrieval interactions. Extensive evaluations on the Massive Multimodal Embedding Benchmark (MMEB) and the Visual Document Retrieval Benchmark (ViDoRe) confirm that MetaEmbed achieves state-of-the-art retrieval performance while scaling robustly to models with 32B parameters. Code is available at https://github.com/facebookresearch/MetaEmbed.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces METAEMBED, a multimodal embedding learning framework designed for scalable late interaction. METAEMBED adds a small number of learnable Meta Tokens to multimodal inputs and trains them using a Matryoshka Multi-Vector Retrieval (MMR) strategy, which organizes embeddings hierarchically to enable test-time scaling. Specifically, MMR enables test-time scaling balancing retrieval accuracy and latency. 
The authors evaluate METAEMBED on MMEB and ViDoRe v2 benchmarks across multiple vision-language backbones (e.g., Qwen2.5-VL, PaliGemma, and LLaMA-3.2-Vision), showing state-of-the-art retrieval performance and flexible inference efficiency across different VLM varying backbones and model sizes.

### Strengths
1. Strong performance: METAEMBED demonstrates strong performance across diverse domains and setups over baseline retrievers.
2. General applicability across VLM backbones and model sizes: METAEMBED is evaluated on Qwen2.5-VL, PaliGemma, and LLaMA-3.2-Vision, show that it generalizes to different model architectures and parameter scales.
3. Comprehensive analysis of METAEMBED: the authors provide comprehensive ablations such as the effectiveness of MMR component, efficiency analysis, analysis on different VLM backbones.

### Weaknesses
1. Limited originality beyond adaptation of MRL: The main innovation lies in adapting Matryoshka Representation Learning (MRL) to multi-vector retrieval. While this adaptation is meaningful, much of the practical foundation (i.e., multimodal representation learning with Matryoshka nesting) derives from prior work (Cai et al. 2025).
2. Insufficient analysis of efficiency in training: Although efficiency is discussed in the context of query encoding at test-time (Table 3). Comparing computational costs in training across different retriever methods can further strengthen the paper, as additional training costs incurred could be a trade-off for enabling flexible test-time compute.
3. Empirical gaps in hyperparameter justification: the paper fixes the group sizes to be G=5 without ablation on group size for MRL. A sensitivity study on different group configurations would strengthen the argument for generality.

### Questions
1. In Table 2, ColQwen2 should correspond to a 3B model rather than 2B.
2. “P” in L182 likely refers to number of visual patch tokens and should be denoted.
3. How do you empirically choose a group size of 5 and with these configurations for MRL?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces METAEMBED, a novel framework designed to enhance multimodal retrieval by addressing the key challenges of scalability, efficiency, and fine-grained retrieval accuracy. Existing methods either compress queries and candidates into a single vector, losing fine-grained information, or generate too many vectors, leading to computational inefficiencies. METAEMBED introduces Meta Tokens, which are small, learnable tokens appended to both queries and candidates, and their final contextualized representations serve as compact yet expressive multi-vector embeddings during retrieval. By employing a Matryoshka Multi-Vector Retrieval (MMR) mechanism, the model organizes embeddings into hierarchical groups, enabling flexible scaling at test-time where users can adjust retrieval quality against computational resources.

### Strengths
1. The introduction of Meta Tokens and the Matryoshka Multi-Vector Retrieval (MMR) approach provides a unique solution to the scalability and efficiency issues in multimodal retrieval, allowing flexible scaling at test-time.
2. METAEMBED demonstrates remarkable scalability, allowing users to adjust retrieval quality based on computational constraints without retraining the model, a critical advancement for real-world deployment.
3. The paper provides extensive evaluations across various model sizes and VLM architectures, showcasing METAEMBED's robustness and effectiveness in diverse scenarios, including challenging tasks like visual document retrieval and multimodal embedding.

### Weaknesses
1. The design intuition of meta embedding is somewhat similar to the concept of query extension in the early information retrieval field. Could the authors further differentiate between these two design intuitions?
2. The approach of using embedding with different granularities for retrieval is promising. Have the authors explored or statistically analyzed which granularity of embedding provides better discrimination? This could offer valuable guidance for future embedding extraction.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
4

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
This paper introduces a framework designed to resolve the critical trade-off between expressiveness and efficiency in multimodal retrieval by using a small set of learnable "Meta Tokens" to form a compact multi-vector representation. The central contribution is the Matryoshka Multi-Vector Retrieval (MMR) training objective, which organizes information into a coarse-to-fine structure. This enables a single model to be flexibly scaled at test-time to meet varying accuracy and efficiency demands without retraining.
Overall, this is a quality paper with interesting findings that could have significant applications for the research community. Although some questions and concerns remain, as detailed below, I am willing to raise my score if these points are adequately addressed.

### Strengths
- Novel Combination of Existing Concepts: novel synthesis of learnable tokens, late-interaction, and Matryoshka learning, resulting in the technically solid Matryoshka Multi-Vector Retrieval objective.

- The experimental evaluation is thorough and convincing, demonstrating state-of-the-art performance on diverse benchmarks like MMEB and ViDoRe v2.

### Weaknesses
- Lack of Interpretability and Qualitative Analysis: The "Meta Tokens" are treated as a black box. The paper offers no analysis or visualization to explain what these tokens learn, missing an opportunity to provide deeper scientific insight beyond a strong engineering result.

- A key motivation presented in the introduction of the paper is solving the efficiency problem for multimodal-to-multimodal (universal) retrieval, yet no experiments on such a task are presented. This leaves a core claim of the paper unsubstantiated by empirical evidence. Specifically, as far as I know, the evaluation on the MMEB and ViDoRe v2 benchmarks lacks an image-to-image retrieval task (which is different from the visual document retrieval task or image+text retrieval task), which would be a key area for qualitative analysis.

### Questions
- (Regarding the 2nd weakness) To better understand the mechanism behind METAEMBED, could you provide some qualitative analysis of the learned Meta Embeddings? For instance, what kinds of visual or textual concepts do different tokens appear to specialize in? Does the first token of an MMR-trained model consistently capture more global, summary-level information compared to later tokens, which might focus on finer details?

-The choice of MMR group sizes {(1, 1), (2, 4), (4, 8), (8, 16), (16,64)} appears somewhat heuristic. How sensitive is the model's performance to this specific configuration? For example, what happens if a more linear scaling is used, and what was the process for selecting this particular set of nested budgets?

-The split-(16,64) baseline in Table 5 performs worse than the single-last baseline. This is counter-intuitive, as one might expect a multi-vector representation, even a naive one, to retain more information. Does this imply that without explicit training tailored for late interaction, simply partitioning a representation is detrimental?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose MetaEmbed which uses grouped multimvector embeddings to scale the "test-time compute" used during retrieval. MetaEmbed introduces a tradeoff between resources (computation, memory), and multimodal retrieval quality. The approach is validated on both general multimodal retrieval tasks (MMEB) and document retrieval (ViDoRe).

### Strengths
- The manuscript is clearly written and well structured.
    
- The introduction of a latency-accuracy tradeoff mechanism into multimodal retrieval is novel and well-motivated.
    
- The authors demonstrate the value of MetaEmbed across a good variety of multimodal embeddings tasks (both general and visual documents).
    
- The MMEB and ViDoRe results are mostly fair, with the majority of compared works having similar training distributions.

### Weaknesses
- The training data has large distribution overlap with evaluation tasks (especially on key tasks, like ImageNet-1K, ViDORe). Therefore, it is unclear how well the model will generalize.
    
- It is unclear in tables 1 and 2 which models are trained on MMEB/ViDoRe (and therefore, strongly advantaged). For clarity, it would be valuable to specify which models are in-distribution vs out-of-distribution when reporting performance.
    
- Several relevant training settings are not included in appendix A.1, such as Optimizer, optimizer hyperparameters, weight decay, learning rate scheduling, was temperature fixed [1] or trained [2]?
    
- The analysis of latency is limited and potentially misleading. How is the index stored? A realistic setup would use an appropriate framework like FAISS [3]. Can the authors provide more details about their inference setup (table 3) and justify its relationship to a realistic deployment?
    
- A common use-case would be short text queries and a much larger candidate pool, in which case scoring latency may become significant.
    
- The device memory requirements for the best performing group settings are very significant, even for the small candidate pool size. This is both a constraint in itself, and can incur significant performance penalties if the index cannot fit into GPU memory.

Minor:

- The reproducibility statement says: “Implementation and compute details (optimizers, LoRA settings, group sizes, temperatures, hardware) are provided in Appendix A.1.” However, some of these details are found elsewhere, whereas others seem to be missing entirely. Ideally, these settings should all be in A.1, even if it results in some duplication from the main text.

- Potentially relevant related work: [4] seems to corroborate the importance of VLM backbone choice for multimodal retrieval, and may be worth including in section 4.2.

### Questions
Some questions can be found in the section above, in addition:

- Can the authors provide a per-task breakdown of MMEB? Although averages over MMEB tasks are often reported, not all tasks in MMEB are equally useful when evaluating a model, therefore a per-task breakdown would be useful.
   
- How are the text queries templated into context? Are task-specific system prompts reused between training and evaluation?

- A comparison between the late interaction approach and a single vector that controls for dimension would be interesting (potentially created by concatenating across the sequence dimension).

[1] R. Meng et al., “VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents,” 2025, doi: 10.48550/arxiv.2507.04590.  
[2] C. Jia et al., “Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision,” 2021, doi: 10.48550/arxiv.2102.05918.  
[3] M. Douze et al., “THE FAISS LIBRARY,” IEEE transactions on big data, pp. 1–17, 2025, doi: 10.1109/TBDATA.2025.3618474.  
[4] B. Schneider, F. Kerschbaum, and W. Chen, “ABC: Achieving Better Control of Visual Embeddings using VLLMs,” Transactions on Machine Learning Research, vol. 2025-, 2025.

### Soundness
3

### Presentation
3

### Contribution
2
