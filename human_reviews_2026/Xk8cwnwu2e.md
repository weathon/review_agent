# Efficient Message-Passing Transformer for Error Correcting Codes

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Error correcting codes (ECCs) are a fundamental technique for ensuring reliable communication over noisy channels. Recent advances in deep learning have enabled transformer-based decoders to achieve state-of-the-art performance on short codes; however, their computational complexity remains significantly higher than that of classical decoders due to the attention mechanism. To address this challenge, we propose EfficientMPT, an efficient message-passing transformer that significantly reduces computational complexity while preserving decoding performance. A key feature of EfficientMPT is the Efficient Error Correcting (EEC) attention mechanism, which replaces expensive matrix multiplications with lightweight vector-based element-wise operations. Unlike standard attention, EEC attention relies only on query-key interaction using global query vector, efficiently encode global contextual information for ECC decoding. Furthermore, EfficientMPT can serve as a foundation model, capable of decoding various code classes and long codes by fine-tuning. In particular, EfficientMPT achieves 85% and 91% of significant memory reduction and 47% and 57% of FLOPs reduction compared to ECCT for $(648,540)$ and $(1056,880)$ standard LDPC code, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an efficient variant of a transformer-based ECC decoder. The main difference from the previous methods is an efficient attention module, which is based on query and key components without a value component.

### Strengths
The paper conducted extensive experiments and shows improvement in memory size and computational complexity compared to previous works.

### Weaknesses
The contribution of the present paper, in comparison with the recent works on Transformer-based ECC decoding by Choukroun and Park, seems incremental, as it largely reproduces the same elegant algorithm with  a minor technical adjustment.

The paper is not clearly written. It is focused on the many low-level technical details and doesn't explain the intuition and motivation 
of the proposed method.

### Questions
Can the proposed attention module architecture be useful to transformer applications other than ECC?

In line 181 it is said that standard methods have complexity o(n^2).  What is the complexity of the proposed method?

### Soundness
3

### Presentation
2

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
This paper proposes an improved transformer-based decoder for error correction codes. The key novelty is the replacement of matrix multiplications in the attention mechanism with a vector-wise operations using a global query vector. EfficientMPT maintains decoding performance comparable to CrossMPT, while significantly reducing the parameter-count, FLOPs, and GPU usage. This is especially evident at large blocklengths - the efficiency improvements allow training at these lengths, without super-large memory usage.

### Strengths
1. EEC attention is well-motivated, and is a natural extension of the ideas in CrossMPT: the global query vector effectively condenses magnitude vector into one vector, rather than relying on a query matrix. The resulting updates resembles message-passing decoding between syndrome info and magnitude info, with simple updates using information from the other modality.

2. The empirical results are very strong. While maintaining the same performance as CrossMPT, efficientMPT achieves significant reductions in memory and computational complexity with a simple architectural change.

3. The main practical contribution is that it allows scaling transformer-based decoding to long blocklengths, which was infeasible via previous methods. FLOPs scale linearly with n, unlike other methods (quadratic dependence?)

4. EfficientMPT can act as a foundation model - unseen codes can be handled via lightweight finetuning.

5. I like the experiment in Figure 7, supports the understanding that a very string inductive-bias based on the PCM has been imposed on the attention mechanism.

6. The evaluation is thorough - compares with good decoders for canonical codes (SCL for polar rather than BP, etc)

### Weaknesses
No major weaknesses.
While transformer-based decoders (including this paper) are still sub-optimal to classical codes/decoders used in practice, this paper improves efficiency of ECCT - which is a major step in the right direction.

### Questions
1. How does training from scratch compare to finetuning the foundation model? Specifically, can you add FEfficient-MPT results for LDPC(1824,1520) in Figure 8a? (and/or train-from-scratch in 6c)

2. The BER performance for 5G-LDPC codes is missing. Can you please add this for completeness. I'd expect a similar performance as CrossMPT? What is the gap in performance/complexity to the decoders currently employed in 5GNR?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposed a transformer based decoder for error correcting codes. The proposed decoder use a lightweight attention module that replace the costly attention mechanism, in the Masked self attention module and masked cross-attention module.
The proposed EEC decoder run over the magnitude and syndrome separately with the proposed simpler attention module. 

They get very good results that reduce the number of parameters, memory usage, and FLOPs, without compromising the decoding performance achieved by CrossMPT.

### Strengths
1. The proposed decoder is very efficient with respect to previous methods in terms of memory usage and FLOPs

2. The proposed EfficientMPT decoder get competitive results compare to previous methods such as ECCT and CrossMPT

3. The EfficientMPT algorithm can also serve as foundation model, a single model can be generalizes to unseen codes with fine-tuning, and get better results than BP on long LDPC codes.

4. The proposed lightweight attention in the decoder using a global query vector and embedding the parity-check matrix, yields a simpler decoder.

### Weaknesses
1. The proposed algorithm evaluate only on simple AWGN channel, and not on non-Gaussian channels such as Rayleigh channel

2. The proposed foundation needs fine-tuning in order to operate well on larger codes.

3. For Polar codes, the classic SCL decoder still get better results than the proposed EfficientMPT decoder,

### Questions
1. Can you check the results of the proposed decoder on non-AWGN channel?

2. Can you suggest ways to reduce the gap to the SCL results? maybe some changes in the architectural or at loss level?

3. Please run the simulation on all codes at it appears in the paper of [Choukroun & Wolf (2022)], for example in Table 1 you only test two BCH code, instead of 4 codes and larger ones

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces EfficientMPT, a transformer-based decoder for error-correcting codes that targets computational and memory bottlenecks in existing transformer-based decoders. The method introduces an efficient attention mechanism, replacing standard matrix multiplications with vector-based element-wise operations and integrating the parity-check matrix (PCM) directly into the attention module. EfficientMPT is designed to be position- and length-invariant, enabling it to serve as a foundation model for ECC decoding. Experimental results show significant reductions in GPU memory, FLOPs, and parameter count, while maintaining bit error rate (BER) performance comparable to existing transformer-based decoders.

### Strengths
- Effective handling of longer codes, overcoming a key constraint of earlier transformer-based ECC decoders.
- Position and length invariance, supporting its use as a foundation model.
- Thorough assessment across multiple code types.

### Weaknesses
- Limited interpretability analysis; it remains unclear how decoding decisions are made compared to traditional belief propagation.
- Missing comparisons with neural Tanner graph-based decoders.
- The analysis of polar codes would be strengthened by a more prominent and direct comparison with SCL decoders, particularly systematic SCL decoders when evaluating BER.
Further Suggestions
- Please verify the dimensions of magnitude and syndrome embedding matrices after multiplication by H and Hᵀ in Figure 2(c) (green multi-head patches).
- The sentence “The lifting process generates LDPC codes of size (52×Z,10×Z)” appears to describe the PCM dimensions rather than the code parameters (n, k). Please clarify.

### Questions
- The sentence “The magnitude embedding is then added to the resized syndrome embedding, which is resized from (n−k)×d to n×d by multiplying the PCM H” is unclear upon first reading—specifically, whether H is in binary or BPSK form. While later sections resolve this, earlier clarification would be helpful.
- Is any sparsity-inducing regularization applied to the trainable PCM in Figure 7(b)?

### Soundness
3

### Presentation
3

### Contribution
3
