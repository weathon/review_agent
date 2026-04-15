# SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression

- Decision: Accept (Poster)
- Scores: 8, 5, 6, 6

## Abstract
The advancements in Large Language Models (LLMs) have been hindered by
their substantial sizes, which necessitates LLM compression methods for practical
deployment. Singular Value Decomposition (SVD) offers a promising solution for
LLM compression. However, state-of-the-art SVD-based LLM compression meth-
ods have two key limitations: truncating smaller singular values may lead to higher
compression loss, and the lack of update on the compressed weights after SVD
truncation. In this work, we propose SVD-LLM, a SVD-based post-training LLM
compression method that addresses the limitations of existing methods. SVD-LLM
incorporates a truncation-aware data whitening technique to ensure a direct map-
ping between singular values and compression loss. Moreover, SVD-LLM adopts
a parameter update with sequential low-rank approximation to compensate for
the accuracy degradation after SVD compression. We evaluate SVD-LLM on 10
datasets and seven models from three different LLM families at three different
scales. Our results demonstrate the superiority of SVD-LLM over state-of-the-arts,
especially at high model compression ratios.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes SVD-LLM, a novel SVD-based LLM compression method that addresses efficiency issues in previous methods. It establishes a direct mapping between singular values and compression loss with theoretic proof. Extensive experiments demonstrate the effectiveness of the proposed method and support the author’s claims.

### Strengths
1. SVD-LLM can achieve state-of-the-art in terms of compression efficiency and performance on downstream tasks.
2. The inference speedup that SVD-LLM can achieve is inspiring given SVD-LLM’s good performance on PPL and downstream tasks.
3. The research is well presented and it is easy to follow.

### Weaknesses
(a) “SVD for LLM Compression” part in “Related works” is insufficient, and it can be improved by discussing more papers related to SVD-based LLM compression and papers that relate to the Atomic Feature Mimicking (AFM) technique, where the latter is also dedicated to model decomposition for LLM compression. Papers that authors should be aware of are listed as follows.

1. Compressing Transformers: Features Are Low-Rank, but Weights Are Not!, AAAI’23

2. LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation, ICML’23

3. Adaptive Rank Selections for Low-Rank Approximation of Language Models, NAACL’24

(b) The experiments on the latest model are missing. Authors should test their proposed method, i.e., SVD-LLM, and compare it with baselines on the latest model such as LLaMA-3-8B.

### Questions
The concept of “SVD-LLM (U)” in S4.3 - "Modular Sensitivity Study" needs to be clarified. From what I understand, the core part of SVD-LLM is the truncation-aware data whitening, the concept of “SVD-LLM (U)” thus needs to be clarified since SVD-LLM will be degraded to normal truncated-SVD when "truncation-aware data whitening" is disabled.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a LLM compression approach based on SVD called SVD-LLM. SVD-LLM incorporates a whitening process to maintain the orthogonality of activations and align the number of retained singular values and the compression loss. After compressed via SVD, SVD-LLM further finetune the model for better performance.

### Strengths
1. The activation whitening is supported by theoretical analysis, which is convincing.
2. The experiments are extensive, demonstrating improvements comparing to selected baselines.
3. SVD-LLM delivers practical benefits in terms of inference speedup and memory reduction on hardware.

### Weaknesses
1. The comparisons in the main experiments might be \textbf{unfair}, as SVD-LLM adds extra finetuning additional finetuning that is not applied to other SVD-based baselines, which could drastically improve performance under high compression ratios. Extra finetuning is not a novel technique and can be seamlessly integrated into the baselines.
2. Theorem 3.2 and Corollary 3.3 only hold when $S$ is calculated on the fly, but in practice SVD-LLM precomputes $S$ using a calibration set, and it is unclear if the orthogonality of activations  is preserved with this approach. Further experiments are needed here to address this gap, for example, the orthogonality of whitened activations using the precomputed $S$.
3. The paper lacks validation of the method on larger models, such as those at the 70B scale, which is crucial for evaluating the feasibility of model compression approaches.

### Questions
1. The paper claims to use LoRA finetuning in Section 3.2 but it is not clear how it is done. Does this introduces additional parameters, or fused afterward? Or the finetuning is directly applied on $W_u$ and $W_v$?
2. Section 4.2, the authors discuss memory reduction for the KV cache at the expense of recomputation overhead. How does this impact throughput, and by how much?
3. In Section 4.4 and Table 8, SVD-LLM is combined with QuIP#. More details are needed regarding how this combination is executed.
4. In Algorithm 3,  there is a layer-wise closed-form update procedure that was not mentioned in the paper.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
With the remarkable capability of large language models (LLMs) in the field of natural language processing, their large model size becomes a major obstacle for practical deployment. To this end, the paper proposes a Singular Value Decomposition (SVD)-based LLM compression method, which aims to address two key limitations faced by existing SVD methods in compression: the truncation of small singular values that may lead to higher compression loss and the lack of model parameter updating after SVD truncation.

### Strengths
- The paper introduces SVD-LLM, a novel SVD-based compression method that incorporates a truncation-aware data whitening strategy to directly map singular values to compression loss, allowing for more informed truncation decisions.
- The paper provides a rigorous mathematical foundation for the proposed truncation-aware data whitening technique, supported by theoretical proofs and lemmas.
- It proposes a parameter update mechanism with sequential low-rank approximation to compensate for accuracy degradation after compression.
- The authors evaluate SVD-LLM across various datasets and models from three different LLM families at various scales, demonstrating its superiority over existing SVD-based methods, especially at high compression ratios.

### Weaknesses
- The authors should evaluate SVD-LLM on the latest models in LLaMA family, such LLaMA-3-8B.
- Authors are recommended to analyze the computation costs of SVD-LLM and provide the evaluation of its time expense.
- No comparison with the latest SOTA methods, such as FLAP: Fluctuation-based Adaptive Structured Pruning for Large Language Models. Yongqi An, et al. AAAI 2024, available on arXiv since Dec. 2023.

### Questions
Please refer to the weaknesses.

### Soundness
3

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
The authors investigate the compression of large language models through low rank approximation. They point out that the existing works in this direction have the following two problems. First, existing works approximate the weight matrix without considering the input activation matrix. Second, existing works do not finetune the model after the truncation. Consequently, they propose SVD-LLM, which addresses both issues by data whitening and LoRA finetuning. Their experiments show that their method not only outperforms other SVD approximation methods, but also outperforms other compression methods like structured pruning or model quantization. Furthermore, real inference speedup and memory saving can be observed without specialized hardware.

### Strengths
1. The writing is clear and easy to follow.
2. The empirical comparison is extensive. The authors not only compared with the SVD-based baselines, but also compared with other compression methods like structured pruning and quantization to demonstrate the superiority of the proposed method.
3. Real inference speedup and memory saving can be observed without specialized hardware. Unlike some unstructured pruning and/or quantization methods, which require specialized hardware for speedup, this makes the proposed method more practical and applicable.

### Weaknesses
1. As mentioned by SAC, the concept of data aware compression is not new and already exists in [1]. Additionally, in [1] they claim optimality for the problem considered, while this paper seems to offer weaker guarantees. (From my understanding, this paper shows that given the basis/decomposition of their choice, pruning the smallest entries of the middle diagonal matrix is optimal. On the other hand, [1] claim that both their choice of the basis and the truncation is optimal)
2. The LoRA training part of this paper also seems not to be novel as existing in LLM-pruner. In addition, the claim for the necessity of sequential finetuning for W_u, W_v seems odd. For lora finetuning of transformers, similar structure exists for key and query attention matrix W_k, W_q, but existing literature does not report any issues for fine tuning.


[1] Chen, P., Yu, H. F., Dhillon, I., & Hsieh, C. J. (2021). Drone: Data-aware low-rank compression for large nlp models. Advances in neural information processing systems, 34, 29321-29334.

### Questions
1. Can you clarify the difference/advantage compared to Drone? Specifically, are the methods equivalent (It seems not so as I do not see a simple connection between Cholesky and eigendecomposition)? If not, can you show that your method is also optimal?
I acknowledge that the experiment is done at a different scale (Bert vs LLM), so verifying this direction works at a larger scale itself is a contribution, but seems not enough for a full paper.  
2. Can you show the performance or LoRA fine tuning without the sequential finetuning process proposed by the paper? This can be one of the novelties of the paper but requires careful verification. The failure of original finetuning might be because of bad choice of hyperparameters for training.

### Soundness
3

### Presentation
3

### Contribution
2
