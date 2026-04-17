# Review

## Summary
The paper presents RetrievalFormer, a transformer-based dual-encoder recommender system designed to improve both recommendation accuracy and retrieval efficiency. The model leverages an attention-based feature encoder to create user and item embeddings that operate in a shared space, enabling Approximate Nearest Neighbor (ANN) retrieval for efficient recommendations. The authors demonstrate that RetrievalFormer achieves competitive performance on benchmark datasets (Amazon Beauty, Amazon Toys & Games, and MovieLens-1M), with a reported 86–91% recall rate compared to transformer-based models while offering significantly lower latency.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-organized and clearly written, with each section logically building upon the previous one. 
2. The authors provide a comprehensive set of experiments across multiple research questions, demonstrating the effectiveness of RetrievalFormer.

## Weaknesses
1. The paper lacks novelty as it does not introduce a new ANN algorithm. It primarily focuses on designing a dual-encoder architecture and training objective that produce an embedding space well-aligned with standard ANN indexes. This alignment enables the use of ANN for serving without compromising recommendation quality. Given that the proposed method relies heavily on existing ANN algorithms, it is crucial to provide a thorough discussion of related work in this area. The current related work section only briefly mentions IVF, HNSW, and PQ, without acknowledging other relevant studies such as:

   - Fast and Accurate Approximate Nearest Neighbor Retrieval for Industrial Deployed Recommendation (https://doi.org/10.1145/3580305.3599541)

   - Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs (https://ieeexplore.ieee.org/document/8918871)

   - Billion-Scale Similarity Search with GPUs (https://ieeexplore.ieee.org/document/8918871)

   - Efficient Nearest Neighbor Recommendation (https://dl.acm.org/doi/abs/10.1145/3434185.3437642)

   - An Industrial-Scale Nearest Neighbor Recommendation in Real-Time (https://dl.acm.org/doi/abs/10.1145/3534678.3539244)

2. The paper could benefit from more detailed discussions on the challenges of training dual-encoder architectures for recommendation systems, particularly in terms of stability and convergence. The authors briefly mention using dropout and uniformity loss to improve training stability, but a more in-depth exploration of these challenges and the rationale behind the chosen solutions would strengthen the paper.

## Questions
1. What is the sensitivity of the model to the quality of the item features? If the item attributes are sparse or noisy, how does this impact the recommendation performance?

2. How does the model handle changes in the item feature space over time, particularly when new attributes are added or existing ones change?

3. How does the model perform on more dynamic datasets with higher user and item churn rates?

4. What is the impact of different ANN index structures and parameters on the recommendation performance and retrieval efficiency?

5. How does the model handle different types of user interactions beyond ratings, such as clicks or implicit feedback?

6. What is the impact of the sequence length and history size on the model's ability to capture long-term user preferences?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4