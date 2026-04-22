# MOAI: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 2, 6

## Abstract
Privacy concerns have been raised in Large Language Models (LLM) inference when models are deployed in Cloud Service Providers (CSP). Homomorphic encryption (HE) offers a promising solution by enabling secure inference directly over encrypted inputs. However, the high computational overhead of HE remains a major bottleneck. To address this challenge, we propose MOAI, an efficient HE-based, non-interactive framework for secure transformer inference. MOAI gains significant efficiency improvement from:  (1) a novel evaluation flow that combines column and diagonal packing with consistent strategies across all layers, eliminating expensive format conversions. (2) rotation-free algorithms for Softmax and LayerNorm that significantly reduce the number of costly HE rotations, removing 2448 HE rotations in BERT-base inference.    (3) Column packing removes rotations in plaintext–ciphertext matrix multiplications and interleaved batching further reduces the rotations in ciphertext–ciphertext matrix multiplications. MOAI uses at least 1.7x fewer HE rotations compared to the state-of-the-art works across all matrix multiplications of BERT-base. As a result, We achieve a 52.8\% reduction in evaluation time compared to the state-of-the-art HE-based non-interactive secure transformer inference, THOR (Moon et al., CCS'25). We then apply MOAI on the Powerformer's framework and achieve a 55.7\% reduction in evaluation time compared to Powerformer (Park et al., ACL'25), which approximates Softmax and LayerNorm with simpler functions in transformer and proposes HE-based non-interactive transformer inference. We report an amortized time of 2.36 minutes per input on a single GPU environment. We show the extendibility by applying MOAI in LLaMA-3-8B. Our implementation is publicly available as open source.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new framework named MOAI, designed to address privacy concerns associated with LLM inference when deployed on Cloud Service Providers. The framework is based on Homomorphic Encryption, enabling non-interactive secure inference directly on encrypted data. MOAI introduces three optimizations: an evaluation flow combining column and diagonal packing; "rotation-free" algorithms for the non-linear functions Softmax and LayerNorm; and column packing and interleaved batching techniques. Experimental results claim that MOAI achieves a 52.8% reduction in inference time on the BERT-base model compared to THOR, the current SOTA non-retraining method. Furthermore, its techniques are shown to accelerate the retraining-based SOTA method Powerformer by 55.7%. The paper reports an amortized inference time of 2.36 minutes per input on BERT-base and demonstrates the feasibility of extending the framework to the LLaMA-3-8B model.

### Strengths
1.  The paper introduces targeted optimizations for several critical bottlenecks in HE-based Transformer inference, including format conversions, rotations, and matrix multiplications. These optimizations collectively contribute to a significant improvement in the efficiency of secure HE-based inference.
2.  Unlike approaches such as Powerformer, which necessitate model architecture modifications and retraining, MOAI is designed as a "plug-and-play" framework. It does not modify Transformer components, enabling its direct application to existing, pre-trained models and offering superior generality and practical utility.

### Weaknesses
Please refer to my questions below.

### Questions
While this work presents significant achievements, several key aspects require further clarification from the authors, particularly concerning the method's scalability and its reliability in real-world scenarios.

1.  Scalability Regarding Sequence Length: The paper's introduction suggests application scenarios (e.g., protecting "drafts" or "sensitive records") that may involve long texts. However, all end-to-end experiments in the paper appear to be confined to fixed and relatively short sequence lengths (128 for BERT-base, and only 8 for LLaMA-3-8B).
    
       Question: The complexity and error propagation of HE operations are typically highly correlated with the sequence length ($m$). Could the authors provide experimental data or analysis illustrating how the MOAI framework's performance (including inference time) and accuracy (cumulative error) would change when processing longer sequences (e.g., $m=1024$ or $m=4096$)? It is currently unclear whether the performance advantages observed on short sequences can be sustained in long-document scenarios.
2.  Accuracy and Task Scope of the LLaMA-3-8B Experiment: Extending the method to LLaMA-3-8B is a significant claim of this paper. However, two major questions surround this experiment:
    
       Question (Task Scope): The experiment described in Appendix H.2 appears to be a "micro-benchmark" of a single forward pass for an 8-token input, rather than a complete autoregressive generation task. Considering the non-interactive nature of the framework and the 224-second execution time for this single step, the feasibility of this method for practical generative chat applications is questionable.
       Question (Lack of Accuracy Validation): More critically, the paper provides no accuracy evaluation for the LLaMA-3-8B experiment (e.g., Table 9 only reports execution time). All precision and error analyses (e.g., Figure 3 and Table 6) are limited to the 12-layer BERT-base model on classification tasks. Given that LLaMA-3-8B (32 layers) is much deeper than BERT-base (12 layers) and that errors can accumulate layer-by-layer (as shown in Figure 3), how can the authors ensure that the cumulative errors from 32 layers of approximate computations (e.g., for RMSNorm, SiLU, etc.) will not severely degrade the final output quality? We believe the effectiveness of the LLaMA experiment remains ambiguous without this accuracy validation.
3.  Clarity of Application Scenarios (Amortization vs. Latency): The paper's key performance metric (2.36 minutes) is an "amortized time" based on a batch of 256 inputs. This is reasonable for offline analysis tasks.
    
       Question: Many privacy-sensitive applications (such as secure search) are interactive and more sensitive to "latency" than "throughput." Could the authors provide the latency data for MOAI processing a single input? This would help reviewers more clearly define the practical application scope of the framework.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents MOAI, a non-interactive secure Transformer inference framework based on homomorphic encryption (HE). The authors propose a new packing strategy aimed at reducing the number of rotations and format conversions, thereby improving efficiency. To eliminate the overhead introduced by format conversion across layers, they propose a consistent matrix packing strategy. Furthermore, the paper presents a rotation-free evaluation scheme for SoftMax and LayerNorm, as well as a rotation-reduced approach for matrix multiplication. Experimental results demonstrate that MOAI improves evaluation time by 52.8% without sacrificing accuracy.

### Strengths
1. The topic is relevant and timely. Non-interactive secure Transformer inference is important for privacy-preserving applications in communication-constrained scenarios.

2. The paper is well-structured and easy to follow. Prior techniques such as packing strategies and interleaved batching are explained clearly.

3. The evaluation is thorough, including a detailed breakdown of overheads for individual components. The release of the implementation further improves reproducibility and practical impact.

### Weaknesses
1. The technical novelty requires further clarification. The primary contribution appears to lie in the composition and refinement of existing packing strategies. However, several techniques adopted in this paper are not entirely new. For example, reducing the number of rotations via diagonal packing is a well-established approach (e.g., Section 4 in [1]). Applying such a strategy to SoftMax appears to be a straightforward extension.

2. Certain methodological details lack clarity. For instance, in Algorithm 1, the authors describe the evaluation of the SoftMax function, but do not specify the degree of the polynomial used (line 301). Additionally, the authors apply the domain extension technique to approximate the GELU function, but it is unclear why this approach is not similarly applied to the sign function, which would validate the approximation at line 1095. The limitations of previous piecewise approximations are not clearly explained.

3. Some notation and formulations require better consistency. For example, the asymptotic complexities of NEXUS and THOR are essentially the same, yet are presented in different forms. This inconsistency may confuse readers.

### Questions
1. Are there experimental results to support the authors' claim about GELU evaluation in Appendix F.2? Specifically, is there quantitative evidence showing accuracy degradation due to the piecewise low-degree polynomial approximation?

2. Given that MOAI is completely non-interactive, could functional bootstrapping [2] be leveraged to reduce the overhead of non-linear functions?

3. Since the Phantom library currently does not support CKKS bootstrapping, and MOAI's GPU implementation is based on Phantom, how is bootstrapping performed in practice?

4. While this work focuses on optimizing FHE evaluation protocols, could techniques from the machine learning side be integrated as well? For example, work elimination [1], token pruning [3], and KV-cache eviction [4] have shown effectiveness in secure Transformer inference. A discussion on the potential integration of such techniques could be beneficial.

[1] Pang, Qi, et al. "Bolt: Privacy-preserving, accurate and efficient inference for transformers.", IEEE S&P 2024.

[2] Alexandru, et al. "General functional bootstrapping using CKKS.", Crypto, 2025.

[3] Zhang, Yancheng, et al. "Cipherprune: Efficient and scalable private transformer inference.", ICLR 2025.

[4] Zeng, Wenxuan, et al. "MPCache: MPC-Friendly KV Cache Eviction for Efficient Private Large Language Model Inference.", NeurIPS 2025.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work focuses on the secure inference of Transformers when FHE is adopted. The author provides a new flow that includes matrix multiplication and nonlinear functions, mainly to reduce the number of rotations. Results demonstrate the effectiveness compared with THOR and Powerformer.

### Strengths
1. The proposed method that allows the output of QKT in diagonal packing is meaningful; this further makes the rotation in softmax removable. Its correctness is true after careful examination and is supposed to guarantee security.

2. The paper has a clear definition and algorithm.

### Weaknesses
1. The assumption of batching in Section 3.2 is strange to me. Prior works seldom do this, and I only know the batching occurs during the offline stage. Since batching is not a trivial solution in HE, as one client usually queries only one sentence once and queries from different clients cannot be batched.

2. The proposed method seems to rely on the input with a large sequence length. However, in the widely used GPT, the decoding only uses a sequence length of 1.

3. As reported in prior works (Nexus), the time spent on bootstrapping takes around 2/3 of the total time. But this work does not focus on such a major bottleneck.

### Questions
1. The authors use a 23-degree polynomial to approximate the GELU function but do not provide detailed coefficients of the polynomial. The high-degree terms usually have tiny coefficients, such as 10^{-20}. In this case, the fixed-point precision is not enough to express these coefficients and would set them as zero. Given that the high-degree terms are usually extremely large, this will cause a very large error.

2. Could the authors provide the comparison of the #rotation with Nexus when only the QK^T and softmax are included? I believe this can better indicate the optimization of section 3.1. I am also curious about the comparison when batching is not applied.

3. The performance on inputs with small input sequences is more interesting in the GPT model, since the decoding usually comes with an extremely small input length (only 1). Providing a comparison in such cases is important to demonstrate the effectiveness of the proposed method.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper implements BERT inference using a GPU-accelerated implementation of CKKS. The main contributions of this paper are improved analysis of the data packing for both the linear and non-linear operations in encrypted transformer inference. The main benefit of this improved analysis is a reduction in the number of rotations needed for various operations in the transformer inference .

### Strengths
I like that they provide their implementation. In general, the experimental results are well-documented.

### Weaknesses
The contributions all seem very minor/incremental. There is also no consideration of significant optimizations in the analysis of the rotation counts. These hoisting optimizations, where one ciphertext can be rotated many times at a cost similar to rotating the ciphertext once, can dramatically reduce the runtimes of FHE linear evaluation.

### Questions
Not all rotations are created equal. When counting rotations, the number of unique ciphertexts to be rotated often matters much more than the number of output rotations. Did you consider hoisting optimizations when counting rotations?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper address the latency overhead of  homomorphic encryption (HE) in private LLM inference, in non-interactive setting (HE-only inference, instead of HE+MPC hybrid settings). The Authors have developed MOAI framework to reduce the rotation operation (the most-expensive operation in HE) in key  nonlinearities (e.g., Softmax and LayerNorms) and plaintext-ciphertext matrix multiplication, which reduce the end-to-end private inference latency. 

The reduction in number of rotation operations is **substantial**, and also the methodology is very-well explained and it does not required any fine-tuning, and can be adapted to the existing open-sourced LLM.

### Strengths
1. Cryptographically secure private LLM inference is an emerging research area but suffers from high latency (in HE-only inference) and communication (in HE+MPC inference) overheads. In HE-only inference latency stems from rotation  and bootstrapping  operations, and the proposed frame work reduces the former to a significant extend. 


2. The paper has presented a very-detailed latency breakdown (Table 3) and operation-wise comparison with prior work (Table 1 and Table 2; and Table 4 and Table 5), which is **impressive and very useful** to understand the savings coming from reducing the rotations in linear as well as nonlinear operations). The authors also discuss the pitfalls and other shortcomings in the end-to-end implementation, and the assumptions made, in prior works such as NEXUS and THOR.  

3. A detailed discussion and empirical analysis for Softmax approximation is provided in the Appendix G and H.3 (Figure 5 and Figure 6) 


4. The approach is simple and relatively straightforward to implement, in a plug-and-play manner. Also, the authors have provided the open-source implementation

### Weaknesses
The main limitation of HE-only inference lies in **accurately approximating nonlinear operations**, particularly when compared with hybrid HE+MPC approaches. Polynomial approximations in HE are highly sensitive to the data range over which they are defined. Within a narrow predefined range, the approximation error can remain low, but this restricts the *representational flexibility* of the model. Expanding the range to better capture the activation dynamics requires using *higher-degree polynomials*, which in turn **increases the number of bootstrapping operations** and thus end-to-end latency.

If we want to enforce a tighter data range for constraining activations to the region where the polynomial approximation remains valid, we need to add the regularization terms during training or we need to fine-tuning the model. For example, the polynomial approximation of LayerNorm in HE-only inference [1]


More importantly, this approximating with polynomial approach may not scale well to LLMs, due to the presence of well-known massive activation [2,3] which are critical to maintaining model performance. Consequently, achieving an accurate approximation would require an extremely wide range, which again necessitates a high-degree polynomial---defeating the efficiency goal of HE-only inference.


[1]  Zimerman et al., Converting Transformers to Polynomial Form for Secure Inference Over Homomorphic Encryption, ICML 2024

[2] Sun et al., Massive Activations in Large Language Models, COLM 2024

[3] He at al., Understanding and Minimising Outlier Features in Neural Network Training, NeurIPS 2024

### Questions
1. Why GELU latency shown in Table 4 is significantly lower than GELU in THOR, even when a 23-degree polynomial is used for their approximation?  

2. If we substitute the GELU with ReLU in FFN, which is shown a promising solution in HE+MPC private inference on LayerNorm-free LLMs [1], then how it would impact the end-to-end performance in HE-only inference ?



[1] Jha et al., AERO: Entropy-Guided Framework for Private LLM Inference

### Soundness
3

### Presentation
4

### Contribution
3
