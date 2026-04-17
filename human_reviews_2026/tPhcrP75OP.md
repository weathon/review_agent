# FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Recent studies in long video understanding have harnessed the advanced visual-language reasoning capabilities of Large Multimodal Models (LMMs), driving the evolution of video-LMMs specialized for processing extended video sequences.
However, the scalability of these models is severely limited by the overwhelming volume of visual tokens generated from extended video sequences. To address this challenge, we propose FLoC, an efficient visual token compression framework based on the facility location function, a principled approach that swiftly selects a compact yet highly representative and diverse subset of visual tokens within a predefined budget on the number of visual tokens. By integrating the lazy greedy algorithm, our method achieves remarkable efficiency gains by swiftly selecting a compact subset of tokens, drastically reducing the number of visual tokens while guaranteeing near-optimal performance. Notably, our approach is training-free, model-agnostic, and query-agnostic, providing a versatile solution that seamlessly integrates with diverse video-LLMs and existing workflows. Extensive evaluations on large-scale benchmarks, such as Video-MME, MLVU, LongVideoBench, and EgoSchema, show that our framework consistently surpasses recent compression techniques, highlighting its effectiveness and robustness in addressing the challenges of long video understanding as well as its processing efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FLoC, a training-free and model-agnostic visual token compression framework for long video understanding. By using a facility location function and a lazy greedy algorithm, FLoC efficiently selects a compact and representative subset of visual tokens, significantly reducing token volume while maintaining near-optimal performance. Extensive experiments show that FLoC outperforms recent compression methods on large-scale benchmarks in both effectiveness and processing speed.

### Strengths
1. The objective of the paper is well-motivated and clearly stated.

2. The proposed method achieves state-of-the-art performance and efficiency in nearly all cases.

3. The authors conduct comprehensive experiments across multiple benchmarks and baseline MLLMs.

### Weaknesses
1. The authors should evaluate throughput, inference speed, and GPU memory usage to further highlight the efficiency of the proposed method compared to existing approaches.

2. In the comparison experiments, it is surprising that PruneVid performs worse than random selection, given that PruneVid employs both clustering and a query-aware selection algorithm. Some discussion or analysis of this result would be helpful.

### Questions
1. As shown in Table 3, basic clustering methods already achieve comparable performance. Does this suggest that sophisticated or complex designs may not be necessary?

### Soundness
3

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
This paper proposes a facility-location-based efficient visual-token compression framework named FLoC for long video understanding. Specifically, FLoC includes two core modules: a Visual Token Selection Module and a Compressed Embedding Module, optimized through a lazy-greedy algorithm. The Visual Token Selection Module tokenizes videos into visual tokens, leverages the submodular facility-location function to measure token representativeness and diversity, and compresses them into compact embeddings. The Compressed Embedding Module concatenates the selected token embeddings with text prompts and feeds them into video-LMMs for downstream tasks. The entire process is guided by the greedy selection strategy, ensuring semantic fidelity and efficient compression without fully decoding the video.

The experimental evaluation in this paper assessed the performance of the proposed FLoC across three compression ratios and three video understanding benchmarks, comparing it with state-of-the-art baselines. The results indicate that the proposed FLoC achieves superior accuracy while reducing sequence length, thereby providing computational and memory efficiency gains.

### Strengths
- The compression process does not rely on specific model architectures, nor is it tailored to particular queries or tasks; a single compression can support multiple downstream applications.
- A visualization video of the lazy-greedy algorithm is provided in the supplementary material, making it more persuasive.
- The experiments are sufficient, with the method’s performance verified on multiple models and benchmarks.

### Weaknesses
- The method performs diversity selection within individual blocks, with no information interaction between blocks. Therefore, the choice of hyper-parameter T is crucial: a T that is too large may cause redundant computation, while a T that is too small may lead to redundant similar tokens across blocks. In Tab. 4, the optimal T differs under different compression rates, making it difficult to select an appropriate T across settings with different models, compression rates, and benchmarks.
- FLoC aims for “global coverage,” which may cause extremely sparse but important tokens (e.g., a key object appearing in only one frame) to contribute very little to the total sum and risk being discarded, thereby affecting performance on needle-in-a-haystack tasks.
- There is no comparison with graph-based video LLM compression methods such as FastVID or LLaVA-Scissor.
- Only computation times of FLoC and traditional algorithms are compared; FLOPs and actual benchmark latency versus other compression methods are not reported.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FLoC (Facility Location-Based Efficient Visual Token Compression), a novel framework designed to tackle the scalability limits of video-Large Multimodal Models (LMMs) when processing long video sequences. The core challenge addressed is the overwhelming volume of visual tokens generated from extended videos, which makes end-to-end processing computationally infeasible given the limited context lengths of most LLM-based architectures.

### Strengths
1. Submodular Optimization via Facility Location: FLoC is the first visual token compression algorithm based on the facility location function and submodular optimization for long video understanding. This interprets token selection as maximizing a utility (or coverage) function that rewards tokens for preserving the essential information and diversity of the entire visual token set within a strict budget constraint (K).
2. Targeting Rare Information Loss: The facility location framework is specifically motivated to overcome a key limitation of clustering methods: their tendency to fail in capturing rare but important tokens (e.g., small objects like car keys) because they focus on densely populated regions in the feature space. FLoC explicitly optimizes for both representativeness and diversity simultaneously.
3. Lazy Greedy Efficiency: The implementation utilizes the lazy greedy algorithm to efficiently approximate the optimal NP-hard solution. This approach significantly reduces computational overhead by postponing the update of marginal gains until necessary, offering a novel and practical way to handle massive token sets.

### Weaknesses
1. The primary operational weakness of FLoC is the reliance on the empirical determination of a single hyperparameter: the block length ($T$). The paper explicitly states that the choice of T involves a critical trade-off that impacts both performance and computational efficiency. The optimal setting for the block length is acknowledged to be content-dependent; for instance, a static video (e.g., a lecture) benefits from a longer block length, while a highly dynamic video requires a shorter one.
2.  FLoC is designed as a query-agnostic and model-agnostic solution. While this offers advantages in memory efficiency (performing compression once) and flexibility, it fundamentally limits the model's performance in real-time, query-specific tasks or dynamic interactive environments. Query-aware compression methods, which FLoC contrasts itself against, "can effectively reduce the search space by focusing on what is deemed important" to the user.
3. FLoC utilizes the lazy greedy algorithm to approximate the maximization of the facility location function, which is NP-hard. While this is significantly faster than naive greedy and traditional clustering methods (e.g., 10× faster than K-Means in compression time), the complexity remains dependent on both the initial number of tokens ($n$) and the budget ($K$), scaled as $O(n⋅K)$.
4. The facility location function relies on the cosine similarity between token embeddings ($sim(v,u)$) as its similarity measure. Cosine similarity effectively measures angular distance, focusing on the direction of feature vectors, which is suitable for abstract feature space coverage. However, it may not perfectly capture complex perceptual differences or contextual relationships that could be better represented by other distance metrics.

### Questions
See weakness above, please.

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
3

### Summary
This paper proposes FLoC, which is a training-free, model-agnostic, and query-agnostic algorithm, for efficient visual token compression. Specfically, it aims to find the optimal token subset which maximizes the submodular objective function. To this end, it uses lazy greedy algorithm. Experimental results show that the proposed algorithm achieves good performance on Video-MME, MLVU, and LVBench.

### Strengths
- The motivation is solid and the paper is easy to follow.
- The proposed algorithm achieves the good performance on various benchmarks such as Video-MME, MLVU, and LVBench.

### Weaknesses
- The citation is mis-formatted across the entire paper. For example, in L124-125, \citep should be used in the LaTeX.  
- In addition to Algorithm 1, a more detailed explanation of the optimal subset search should also be provided in the main text. 
- I think the novelty of the proposed method is somewhat limited. Sampling a token subset is not a new idea, and simply applying the well-known lazy greedy algorithm for this sampling seems to offer only a modest contribution. If there are additional aspects of novelty that I might have missed, I would appreciate clarification.
- It would also be helpful to include a comparison of experimental results with the following papers.
[1] Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs, ICCV2025
[2] LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs (optional)
- It would be good to include experimental results on datasets commonly used in this field, such as Next-QA or EgoSchema.
- typo
	- L193: arg   max -> argmax

### Questions
Please see the weakness section for my concerns. My main concern is about the novelty of the proposed method. If this concern is well addressed through the rebuttal, I would be willing to increase my score.

### Soundness
2

### Presentation
2

### Contribution
2
