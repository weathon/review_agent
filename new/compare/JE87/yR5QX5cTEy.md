# Review

## Summary
This paper presents StragglAR, a new allreduce algorithm designed to improve collective communication performance in the presence of stragglers. The algorithm operates under the assumption that the allreduce will be executed by a homogeneous cluster of GPUs, where all but one GPU are connected to a high-speed interconnect and can communicate with any other GPU in the cluster in a single step. The algorithm proceeds in two stages: in the first, an allreduce is executed using the reduce-scatter primitive, while in the second, the algorithm executes a custom communication schedule. This second stage can be overlapped with the first, and its communication volume is less than that of the baseline allreduce algorithms. The authors demonstrate that, under certain conditions, this algorithm can outperform existing allreduce implementations in terms of both latency and bandwidth.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The authors identify a relevant problem and propose an interesting solution.
2. The paper is well written and easy to follow.
3. The authors provide a theoretical analysis of the algorithm.

## Weaknesses
1. The algorithm is limited to homogeneous clusters of GPUs, where all but one GPU are connected by a high-speed interconnect, which is not always true in practice.
2. The algorithm requires that the reduce-scatter communication volume is less than that of an allreduce. This may not always be the case in practice.
3. The experimental section is limited and does not demonstrate the effectiveness of the algorithm.
4. The authors do not compare the algorithm to existing straggler-aware collective communication algorithms.

## Questions
1. The authors claim that the algorithm is secure in the presence of multiple stragglers, but this does not seem to be the case. If multiple GPUs are straggling, it is likely that the algorithm will not complete in the worst case. Have the authors considered this scenario, and if so, can they provide an analysis of the algorithm's performance under these conditions?

2. The authors assume that the communication volume of the reduce-scatter phase is always less than that of the allreduce phase. However, this may not always be the case in practice. For example, in tensor parallelism, the communication volume of an allreduce is typically much higher than that of an reduce-scatter. Can the authors provide a justification for this assumption, and discuss how the algorithm would perform in scenarios where the communication volume of the reduce-scatter is higher than that of the allreduce?

3. The experimental section is limited and does not demonstrate the effectiveness of the algorithm. The authors evaluate the algorithm on a single machine with 8 GPUs, and do not provide any results on larger-scale systems. Can the authors provide a more comprehensive evaluation of the algorithm, including experiments on larger-scale systems and comparisons to existing straggler-aware collective communication algorithms?

4. The authors do not compare the algorithm to existing straggler-aware collective communication algorithms. Can the authors provide a comparison of their algorithm to existing straggler-aware collective communication algorithms, such as those presented in [1] and [2]?

[1] E. Warraich, O. Shabtai, K. Manaa, S. Vargaftik, Y. Piasetzky, M. Kadosh, L. Suresh, and M. Shahbaz. Optireduce: Resilient and tail-optimal allreduce for distributed deep learning in the cloud. In 22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25), pages 685–703, 2025.

[2] Z. Lin, Z. Jiang, Z. Song, S. Zhao, M. Yu, Z. Wang, C. Wang, Z. Shi, X. Shi, W. Jia, et al. Understanding stragglers in large model training using what-if analysis. arXiv preprint arXiv:2505.05713, 2025.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4