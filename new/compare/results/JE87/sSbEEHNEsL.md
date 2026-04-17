# Review

## Summary
The authors propose a novel approach for Unified Speech Recognition (USR) that addresses the limitations of the previous USR model by improving training efficiency and robustness to out-of-distribution (OOD) data. The key contributions are: (1) CTC-driven teacher forcing, which replaces autoregressive decoding with CTC pseudo-labels to generate attention targets in a single forward pass, and (2) Mixed sampling to mitigate the exposure bias introduced by CTC-driven teacher forcing. The results demonstrate significant improvements in training time, robustness to OOD data, and state-of-the-art performance across various benchmarks.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The proposed method addresses critical limitations of the previous USR model, specifically the computational bottleneck of autoregressive pseudo-labelling and the sensitivity to OOD data.
- The authors provide a thorough empirical evaluation across multiple benchmarks, demonstrating the effectiveness of their approach in various settings.
- The method achieves state-of-the-art results on multiple benchmarks, including LRS3, LRS2, and WildVSR, for ASR, VSR, and AVSR tasks.

## Weaknesses
- The paper could benefit from a more detailed discussion of the limitations of the proposed approach, particularly in terms of potential trade-offs or scenarios where USR 2.0 might not perform optimally.
- While the empirical results are strong, the paper could benefit from a more in-depth theoretical analysis of why CTC-driven teacher forcing improves robustness to OOD data.

## Questions
- How does the performance of USR 2.0 scale with larger models and datasets? Are there any observed limitations or bottlenecks when scaling the approach?
- Have you explored the potential of combining CTC-driven teacher forcing with other pseudo-labeling techniques beyond autoregressive decoding? What were the results, if any?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4