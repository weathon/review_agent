# Review

## Summary
This paper proposes a KV cache compression method based on quantum computing. The main idea is to encode the attention score of each token into a qubit, which allows for a significant compression ratio. The paper also provides some theoretical analysis and experimental results.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-written and easy to follow.
2. The idea of using quantum computing for KV cache compression is interesting.

## Weaknesses
1. The paper lacks a detailed complexity analysis of the proposed method. Specifically, it is unclear how much time is required to encode and decode the qubits, as well as the time complexity for the attention operation with qubit-based attention scores. While the authors claim that their method has a 7x compression ratio, it is important to note that the uncompression ratio may also increase significantly, potentially offsetting the benefits of the compression method.

2. The paper does not provide a comparison of the latency of the proposed method with other baselines. Given that the method involves encoding and decoding qubits, as well as performing measurements, it is likely that the latency will increase. The paper should address this issue and provide a detailed analysis of the trade-off between compression ratio and latency.

3. The paper does not provide a comparison of the proposed method with other quantum-based methods, such as [1]. It is important to note that there are already existing quantum-based methods for KV cache compression, and the paper should address how the proposed method improves upon these existing methods.

[1] QuantumKV: Towards Quantum Key-value Cache Compression for Large Language Models

## Questions
Please see the weaknesses above.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4