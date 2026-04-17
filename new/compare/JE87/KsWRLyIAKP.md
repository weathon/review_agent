# Review

## Summary
This paper proposes a framework to detect lead-lag relationships in financial markets using Temporal Graph Neural Networks (TGNNs). The authors define lead-lag relationships as temporal dependencies between financial assets, where one asset (the "leader") influences another (the "follower") with a time lag. The framework represents these relationships as dynamic graph structures, where nodes represent financial assets, and edges represent the predictive influence of one asset on another over time. The authors evaluate several TGNN models on a custom dataset of financial assets, enriched with temporal, structural, and sentiment features.

## Soundness
2

## Presentation
2

## Contribution
2

## Strengths
1. The paper introduces a novel application of TGNNs to a real-world financial problem. The framework creatively combines graph-based representations with temporal data to capture evolving lead-lag relationships in financial markets.
2. The paper is well-structured and clearly written. The authors provide a thorough literature review and a detailed explanation of the methodology, including data collection, graph construction, and model adaptations.
3. The paper addresses a significant problem in financial market analysis, and the proposed framework has practical applications in portfolio optimization, risk management, and trading strategies.

## Weaknesses
1. The paper does not provide a detailed comparison with traditional statistical methods for lead-lag detection, such as Granger causality. Including such comparisons would strengthen the argument for the proposed approach.
2. The paper could benefit from more ablation studies to understand the contribution of individual components within the framework, such as the impact of different feature types (e.g., financial indicators, sentiment) on model performance.
3. The paper lacks a detailed discussion of the computational complexity and scalability of the proposed approach, particularly for large-scale financial datasets.
4. The authors do not provide a detailed analysis of the interpretability of the learned models, which is crucial for understanding and trusting the results.

## Questions
1. How does the proposed framework compare with traditional statistical methods, such as Granger causality, in terms of accuracy and computational efficiency?
2. Can the authors provide more insights into the interpretability of the learned models, particularly regarding the lead-lag relationships that are detected?
3. How does the framework handle missing data or irregular time series, which are common in financial markets?
4. Can the authors provide more details on the computational complexity of the proposed approach and its scalability to large-scale financial datasets?
5. How does the framework handle the potential noise and volatility in financial markets, and how does this affect the detection of lead-lag relationships?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4