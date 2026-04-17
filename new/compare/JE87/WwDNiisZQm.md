# Review

## Summary
This paper presents a novel approach to learned image compression (LIC) by introducing a content-aware Mamba (CAM) model. CAM dynamically adapts its processing to the image content, overcoming limitations of standard Mamba's rigid, content-agnostic scans. Two key innovations are introduced: a content-adaptive token permutation strategy that prioritizes interactions between content-similar tokens, and a mechanism for injecting sample-specific global priors into the state-space model to mitigate causality constraints. The resulting Content-Aware Mamba-based LIC model (CMIC) achieves state-of-the-art rate-distortion performance, surpassing traditional codec VTM-21.0 by significant margins on multiple datasets (Kodak, Tecnick, CLIC).

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The motivation is clear and reasonable.
2. The proposed method achieves significant performance improvement over previous methods.
3. The idea of content-adaptive token permutation is novel and interesting.

## Weaknesses
1. The proposed method introduces extra computation cost in the form of clustering operations. Although the clustering operation is not repeated in each layer, it still brings extra overhead. It is suggested to provide the latency comparison of clustering operation and compare it with the latency of previous methods.
2. The ablation study in Table 2 shows that the proposed method improves the performance by a large margin even without CTP and GPP. It is suggested to provide more detailed information about the baseline model. Does it use a different structure than the proposed CMIC?
3. It is suggested to provide the result of the proposed method on the CLIC test set.

## Questions
Please refer to the weakness part.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
5