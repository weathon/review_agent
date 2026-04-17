# Review

## Summary
This paper proposes a novel framework called STBP for continual spatio-temporal forecasting. STBP integrates a general spatio-temporal backbone with a scalable contextual pattern bank to address the challenges of distributional drift, dynamic spatio-temporal correlations, catastrophic forgetting, and efficient incremental learning. The backbone extracts stable representations in the frequency domain and captures dynamic spatial correlations, while the contextual pattern bank updates incrementally via parameter expansion to adapt to evolving node-level heterogeneity and relevance. Extensive experiments on multiple real-world datasets demonstrate that STBP outperforms state-of-the-art baselines in terms of forecasting accuracy, adaptability, and scalability.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper introduces a novel framework, STBP, which combines a general spatio-temporal backbone with a scalable contextual pattern bank to address the challenges of continual spatio-temporal forecasting. This integration of frequency-domain analysis and linear graph attention with a trainable parameter bank is innovative and provides a new perspective for solving the problem.
2. The paper is well-organized and clearly written. The authors provide a clear motivation and problem statement, followed by a detailed description of the methodology, experiments, and results. The figures and tables are well-designed and help to illustrate the concepts and results.
3. The paper addresses an important and practical problem in the field of spatio-temporal forecasting. The ability to make accurate predictions in dynamic and evolving environments has significant implications for various applications, such as smart cities, transportation systems, and environmental monitoring.

## Weaknesses
1. The paper could benefit from a more detailed discussion of the limitations and potential drawbacks of the proposed framework. It would be helpful to address any assumptions made in the methodology and to discuss potential scenarios where the framework may not perform as well.
2. The paper could provide more details on the implementation and deployment of the proposed framework in real-world systems. This would help to assess its practicality and feasibility in real-world scenarios.
3. The paper could benefit from a more detailed comparison with existing methods and a discussion of the novelty and advantages of the proposed framework over previous approaches.

## Questions
1. How does the proposed framework handle extreme cases of distributional drift or sudden changes in spatio-temporal patterns?
2. Can the framework be extended to handle other types of spatio-temporal data, such as social media data or sensor data from different domains?
3. How does the framework handle the addition or removal of nodes or edges over time? Is there a mechanism to update the contextual pattern bank accordingly?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4