# SAQ: Stabilizer-Aware Quantum Error Correction Decoder

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Quantum Error Correction (QEC) decoding faces a fundamental accuracy-efficiency tradeoff. Classical methods like Minimum Weight Perfect Matching (MWPM) exhibit variable performance across noise models and suffer from polynomial complexity, while tensor network decoders achieve high accuracy but at prohibitively high computational cost. Recent neural decoders reduce complexity but lack the accuracy needed to compete with computationally expensive classical methods. We introduce SAQ-Decoder, a unified framework combining transformer-based learning with constraint aware post-processing that achieves both near Maximum Likelihood (ML) accuracy and linear computational scalability with respect to the syndrome size. Our approach combines a dual-stream transformer architecture that processes syndromes and logical information with asymmetric attention patterns, and a novel differentiable logical loss that directly optimizes Logical Error Rates (LER) through smooth approximations over finite fields. 
SAQ-Decoder achieves high accuracy decoding, with error thresholds of 10.99\% (independent noise) and 18.6\% (depolarizing noise) on toric codes that closely approach the theoretical ML bounds of 11.0\% and 18.9\% while outperforming existing neural and classical baselines in accuracy, complexity, and parameter efficiency. Our findings establish that learned decoders can simultaneously achieve competitive decoding accuracy and computational efficiency, addressing key requirements for practical fault-tolerant quantum computing systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
I would like to note that I am not familiar with the quantum computing or quantum machine learning domain, and therefore I do not have the necessary background to properly evaluate this submission. While I can assess general aspects of clarity and structure, I am not qualified to judge the technical novelty, correctness, or significance of the paper within the context of quantum research.

I respectfully suggest that this paper be reassigned to a reviewer with expertise in quantum algorithms or quantum information, to ensure a fair and accurate evaluation.

### Strengths
See Summary

### Weaknesses
See Summary

### Questions
See Summary

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SAQ-Decoder. It is a neural network approach for QEC that addresses the fundamental accuracy efficiency trade off in QEC decoding. The method combines a dual-stream transformer architecture with a novel differentiable logical loss function and a post-processing algorithm, constraint projected null space descent. The approach achieves near optimal error thresholds while maintaining linear computational complexity.

### Strengths
1.  Strong empirical performance. The decoder achieves very good results on toric codes, also approaching maximum bounds while outperforming both classical methods and recent neural approaches.
2. Novel architectural design. The dual-stream transformer with asymmetric attention patterns is well-motivated for QEC.
3. Differentiable logical loss. Paper provides a rigorous mathematical derivation of a differentiable approximation to the discrete GF(2) constraints, enabling end to end training that directly optimizes logical error rates rather than bit error rates.

### Weaknesses
1. The experiment limited to code distances up to 10. For practical fault-tolerant quantum computing should be larger.
2. The paper lacks comparison with recent strong baselines. 
3. While the method clams applicability to any stabilizer code family, but experiments are limited to surface codes.

### Questions
1. The ablation studies don’t clearly show CPND’s contribution versus simpler projection. Can you provide a direct comparison of logical error rates with/without CPND?
2. Does decoder require retraining for different physical error models?
3. What are the convergence guarantees for CPND?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a transformer-based quantum decoder, called the SAQ-decoder, with several innovative features: dual-stream transformers, a differentiable logical loss, and constraint-aware post-processing. The proposed decoder achieves near-optimal accuracy for topological codes.

### Strengths
The dual-stream transformer architecture is intriguing, and the experimental results appear very promising.

### Weaknesses
1.	It’s unclear what the final output of the proposed decoder actually is. Is it \hat{l}, \hat{e}, or the output of the CPND? Based on the appendix, it seems the final output used for evaluating LER is from the CPND, but this should be clarified more clearly in the main text. Also, e^{pred} above Eq. (13) appears to represent the final output, but this is not explicitly stated.
2.	To my knowledge, other transformer-based decoders for QEC already exist, such as alphaQubit. These should be cited, and the differences explained. Additionally, the dual-stream transformer architecture seems similar to CrossMPT for classical codes, which also uses cross-attention. That work is worth referencing as well.
3.	The MLP ϕb_\phi seems technically identical to the FFN in Kai (2022, PRL), as it takes the syndrome vector and outputs logical information. This similarity should be acknowledged.
4.	The asymmetric structure of the dual streams is not intuitive. Why does the syndrome stream (which contains local information) influence the logical stream (which contains global information)? The opposite direction seems more meaningful. Without empirical results, can the authors illustrate the reasoning behind this design? Moreover, since the syndrome stream is not affected by the logical stream, does this mean the error vector is derived directly from the input syndrome? If so, how does this differ from QECCT?
5.	Is the gain over QECCT mainly due to the novel transformer architecture or the post-processing via CPND? To make a fair comparison, the SAQ-decoder should also be compared to QECCT without CPND.
6.	The non-differentiability of the loss function L(e_{\text{true}} + e_{pred}) has already been addressed by Liu (2019, PRL). Could you explain the difference?
7.	The authors use MWPM as the standard classical decoder. However, BP+OSD is also a well-known and widely used decoder for quantum codes and should be included in the comparison.
8.	The computational complexity is only discussed using Big-O notation. It would be more informative to also provide numerical comparisons, such as FLOPs or inference time.
9.	Regarding parameter efficiency, the SAQ-decoder still relies on syndrome information, which scales quadratically with the lattice size LLL. Therefore, I think it cannot avoid quadratic scaling.

### Questions
In Fig. 6, are the authors testing the effect of both masking and the global token? Does the "no mask" label mean only the global token is used, and "mask only" means only masking is applied? Is SAQ-decoder the version with both features? The labeling is somewhat confusing. Also, it would be helpful to include results where neither masking nor the global token is used.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
SAQ-Decoder proposes a stabilizer-aware decoder for Quantum error codes, that combines existing transformer-based decoding approaches with a constraint-inducing post-processing. The key idea is to use a dual-stream representation - one for syndromes and one for logical information. These streams are then processed by a transformer, which is followed by a lightweight constraint-preserving post-processing stage that enforces exact syndrome consistency.

The architecture attains near-ML thresholds on toric codes, while maintaining linear scaling in syndrome size and strong parameter efficiency. The method outperforms MWPM and prior neural baseline QECCT in both accuracy and scalability. Similar gains hold on rotated surface codes, indicating generalization across codes.

### Strengths
1. The method is well-motivated and seems technically sound, combining learned decoding with constraint-preserving post-processing is a clear and sensible idea.

2. The results shown are strong: the decoder achieves near-ML thresholds on toric and rotated surface codes while maintaining linear scaling and good parameter efficiency.

3. The approach seems to generalize well across different noise models and code families.

### Weaknesses
I do not have sufficient background in quantum error correction to fully assess the novelty of the proposed method relative to prior decoders.
While the approach appears reasonable and the empirical performance is impressive, I am unable to evaluate the theoretical aspects of stabilizer enforcement or the loss formulation.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3
