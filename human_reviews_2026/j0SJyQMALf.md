# HeLutNet: Extremely Fast Privacy-preserving Inference in Milliseconds via LUT-based Machine Learning Models

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Fully Homomorphic Encryption (FHE) holds immense promise for enabling privacy-preserving machine learning (PPML) inference, allowing computations on encrypted data without needing to expose sensitive information to untrusted third parties, such as cloud providers. Despite this potential, a major obstacle to FHE's widespread adoption is its prohibitively high computational overhead and slow inference speeds. For instance, the current state-of-the-art implementation requires 0.23 seconds for a single MNIST inference on one CPU core, a significant slowdown that renders FHE impractical for real-world applications. To overcome this, we introduce HeLutNet, an ultra-low-latency framework that accelerates FHE-based PPML inference by pioneering the use of lookup table (LUT)-based ML models. LUT-based models are uniquely suited for FHE, as they can represent complex non-linear functions without deep network architectures, enabling high accuracy on diverse datasets with shallow networks. Our approach converts a PyTorch-trained, LUT-based model into an efficient FHE program, implementing LUTs using the BGV scheme's SIMD capabilities and incorporating novel packing and LUT reordering strategies to minimize the number of homomorphic operations performed. Our evaluation across 20 vision, speech, health, and tabular datasets demonstrates consistent speedup and lower memory and communication overhead compared to state-of-the-art FHE-based PPML methods. Notably, we reduce inference time on various datasets to just 8.3 ms on a single-core CPU, achieving a 27.71x speedup over Orion (ASPLOS'25), thereby demonstrating the potential for a practical, real-time FHE-based PPML system.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
* This paper introduces HeLutNet, a framework designed to significantly accelerate Fully Homomorphic Encryption (FHE)-based machine learning inference. The core idea is to leverage lookup table (LUT)-based machine learning models (specifically, Differentiable Weightless Networks or DWNs) instead of traditional deep neural networks. The authors argue that LUT-based models are inherently FHE-friendly because they can represent complex non-linear functions using shallow architectures, avoiding the need for costly polynomial approximations of activation functions or deep computational graphs that necessitate expensive bootstrapping operations.
* HeLutNet converts a PyTorch-trained LUT model into an FHE program using the BGV scheme. It proposes a novel method to implement LUTs efficiently using SIMD integer arithmetic within BGV, aiming to maximize throughput and minimize expensive rotation operations . The paper claims that this approach achieves ultra-low latency, reducing inference times for datasets like MNIST and Fashion-MNIST to just 13 milliseconds on a single CPU core, representing a significant speedup (e.g., 17.7x over state-of-the-art methods) while maintaining competitive accuracy.

### Strengths
* The paper proposes the first (to my knowledge) end-to-end LUT-based network architecture specifically designed for FHE inference. While LUT evaluation in FHE exists [1, 2, 3], applying it to a complete network architecture is a novel direction.
* The reported inference times, particularly the 13 ms for MNIST and Fashion-MNIST on a single core, are exceptionally fast for FHE-based inference and represent a substantial improvement (e.g., 17.7x over Orion) over many existing FHE frameworks operating on similar datasets.
* The strategy of implementing LUTs via multiplexer trees mapped to BGV's SIMD integer arithmetic seems well-reasoned. Choosing a leveled scheme like BGV is appropriate for the inherently shallow nature of the LUT-based models used.
* The evaluation spans a commendable range of 16 datasets across vision, speech, healthcare, and tabular domains, demonstrating the approach's applicability beyond standard image classification benchmarks.
* The authors explicitly discuss the current limitations of LUT-based models, acknowledging they are not yet suitable for larger more complex datasets.

### Weaknesses
* The paper argues that TFHE's fast bootstrapping isn't beneficial for shallow LUT networks and pivots to BGV for its SIMD capabilities. However, TFHE also supports programmable bootstrapping (PBS), which can directly evaluate LUTs. A more detailed ablation study comparing HeLutNet's BGV-SIMD approach against a TFHE-PBS implementation of the same LUT network would be valuable. This comparison should ideally include latency, total homomorphic operations, and memory overhead.

* Given the focus on SIMD efficiency, why was BGV chosen over CKKS? CKKS is often favored for ML due to its approximate arithmetic and strong SIMD capabilities (e.g., used in EVA [4]). Could the authors elaborate on the rationale for selecting BGV?

* Fairness of comparisons to state-of-the-art methods:
    * The 1023x speedup claim over DSHE on MNIST appears to be an unfair comparison. DSHE uses a deep, complex architecture (akin to ResNet-18) with millions of parameters, whereas HeLutNet uses a very lightweight LUT-based model. Comparing these vastly different model types primarily on latency isn't comparing like-for-like in terms of model complexity or representational power.
     * The Appendix (Table 6) includes DT results for Concrete-ML. It also notes Concrete-ML uses quantization (4-bit/8-bit). Does HeLutNet involve quantization? If comparing to Concrete-ML, clarity on whether quantization-aware training or post-training quantization was used for the Concrete-ML baselines is needed, as this choice significantly impacts final accuracy.
    * In Table 3, HeLutNet (a LUT-based model, conceptually similar to decision trees) is primarily compared against XGBoost models from Concrete-ML. While XGBoost is a strong baseline, I feel it is a bit misleading from a latency perspective to only compare HeLutNet and XGBoost without the Decision Tree (DT) as well. I would suggest combining Table 3 and Table 6 in the main text as this provides a more holistic comparison. Furthermore, are there BGV or CKKS implementations of DTs or similar lightweight models in the literature that could serve as more direct baselines than TFHE-based Concrete-ML?

* A few more performance metrics would be useful to contextualize HeLutNet compared to other methods:
    * What are the memory requirements (e.g., key sizes, ciphertext expansion) for HeLutNet compared to baselines?
    * Could the authors report the total number of homomorphic operations (or perhaps "homomorphic LUT evaluations") performed per inference? This provides a hardware-independent measure of computational cost.
    * Since BGV SIMD operations process multiple data points in parallel, reporting amortized latency per data point (if batching is used/possible) alongside single-instance latency would be informative.

* The paper mentions the challenge of FHE on datasets like CIFAR-10 in the introduction  but does not include CIFAR-10 in its own benchmarks. Given that numerous FHE inference papers (including some cited baselines like EVA) report results on CIFAR-10, its omission is noticeable. Could the authors comment on why CIFAR-10 was excluded and whether HeLutNet was attempted on it?

* Minor point: The statement in the abstract "For the first time, we propose accelerating FHE-based ML by using lookup table (LUT) based ML models" could be slightly refined. While this appears to be the first end-to-end network based on LUTs for FHE, research on using LUTs for evaluating parts of computations or specific functions within FHE does exist [1, 2, 3],. Clarifying the novelty as the first complete network architecture based on LUTs for FHE might be more precise.

Reference: \
[1] Kim, Jaeyun et al. “Privacy-Preserving Embedding via Look-up Table Evaluation with Fully Homomorphic Encryption.” *International Conference on Machine Learning*. 2024 \
[2] Chung et al. “Amortized Large Look-up Table Evaluation with Multivariate Polynomials for Homomorphic Encryption.” *Cryptology ePrint Archive*. 2024 \
[3] Hee Cheon et al. “Tree-based Lookup Table on Batched Encrypted Queries using Homomorphic Encryption.” *Cryptology ePrint Archive*. 2024 \
[4] Dathathri et al. "EVA: An encrypted vector arithmetic language and compiler for efficient homomorphic computation." *Proceedings of the 41st ACM SIGPLAN conference on programming language design and implementation*. 2020

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes to HELUTNET framework for decreasing the latency of FHE-based PPML by adapting Look Up Table-based ML models to FHE. This approach, based on the BGV scheme, converts LUT ML models to low-latency FHE programs while utilizing SIMD operations in BGV. The paper evaluates the proposed approach against various baselines w.r.t latency and accuracy on lightweight vision, speech, and tabular datasets.

### Strengths
1)The paper presents a practical approach for FHE-based privacy-preserving applications on lightweight datasets, showing good improvement over the baselines in terms of the latency on the MNIST and Fashion MNIST datasets.
2)The approach is validated on a wide variety of lightweight datasets, showing its ability to generalize across modalities. 
3)The latency problems of FHE are well motivated, and the proposed approach and FHE concepts are clearly presented.

### Weaknesses
1) The related work section is not thorough; the context of the prior work on improving the latency of FHE-based PPML is not established. 
2)Prior work has used LUTs in an FHE setting for PPML; they have not been discussed or cited. Some examples are given below:
	a) Privacy-Preserving Embedding via Look-up Table Evaluation with Fully Homomorphic Encryption
	b) Homomorphic WiSARDs: Efficient Weightless Neural Network training over encrypted data
	c) TT-TFHE: a Torus Fully Homomorphic Encryption-Friendly Neural Network Architecture
3)The paper claims it has better latency than the SOTA MPC protocols without any comparison results presented in the paper.
4)The limitations of expressivity of LUT-based ML models could prohibit scaling to larger datasets.

### Questions
For the datasets other than MNIST and fashion MNIST, the approach is compared only against  Concrete-ML. Is it the SOTA FHE-based PPML method for these datasets?. If not, what were the reasons for not using other methods as baselines for these datasets?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents an implementation of LUT-based FHE for PPML inference, which is technically sound but suffers from significant limitations in novelty and practical utility. While the authors claim to be the first to propose LUT-based models for FHE-accelerated inference, lookup table operations have been a standard technique in TFHE/FHEW schemes (of course, this paper adopted BGV, instead of TFHE), making the technical contribution somewhat incremental and less groundbreaking than presented. The claimed speedup on MNIST is also questionable when amortized fairly against recent PPML baselines, and without adequately accounting for model size differences, the reported 17.7× improvement appears to be primarily an artifact of choosing a straightforward network architecture. Most critically, the complete absence of evaluation on CIFAR-10—which is arguably still a modest dataset in today's PPML research—reveals a fundamental scalability barrier that the authors sidestep rather than address, undermining claims of practical applicability. The core trade-off of using shallow 2-layer models severely sacrifices representational capacity, rendering complex real-world inference tasks infeasible, yet this is presented as a feature rather than honestly discussed as a severe limitation. Compared to concurrent work and other recent FHE frameworks that handle deeper networks and larger datasets, this paper addresses a problem that feels increasingly dated and somewhat orthogonal to the field's current direction. The experimental scope is too narrow, the model architectures are too simplistic, and the practical impact is too limited to warrant acceptance at a top-tier venue, as the work essentially demonstrates that LUTs can be fast in FHE at the cost of the expressiveness needed for non-trivial applications.

### Strengths
The paper presents a technical solution for efficiently implementing LUTs in FHE using SIMD operations within the BGV scheme, enabling faster inference on MNIST without requiring expensive bootstrapping.

### Weaknesses
The paper's evaluation is severely limited by its inability to scale beyond MNIST, with no results on CIFAR-10 and an acknowledged impracticality for larger datasets, thereby fundamentally undermining claims of real-world applicability. The core trade-off of using ultra-shallow 2-layer networks to achieve speed is sacrificing the representational power needed for non-trivial tasks, and the amortized performance gains over recent SOTA baselines become questionable once you account for the vast differences in model complexity and input dimensionality.

### Questions
Does the paper provide any results on scaling this approach to larger models and datasets? What are the core problems and limitations that arise when trying to do so?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To build a privacy-preserving AI model using homomorphic encryption, one must handle numerous non-linear operations, which significantly increase computational overhead. Consequently, homomorphic encryption–based models typically require longer inference times. In this paper, the authors propose a model that leverages lookup tables (LUTs) to enable efficient computation under the BGV scheme.
As a result, the proposed method achieves tens of times faster performance compared to previous approaches, while maintaining competitive accuracy on lightweight datasets such as MNIST.

### Strengths
First of all, I believe the DWN, or more precisely, the efficient construction of the LUT, is a meaningful contribution. There have not been many studies that implemented LUTs efficiently. Typically, LUT-based methods rely on interpolation polynomials or quantization followed by comparison-based approaches. In contrast, this paper presents an optimization specifically tailored to the DWN structure, which I consider to be a clear and notable contribution.

### Weaknesses
The implementation of LUT branching in this paper is rather simple. Such a design heavily depends on the packing scheme, which in turn affects the communication cost. Since the encrypted data is transmitted after thermometer encoding, the overall communication overhead is inevitably large. Although it may still be lower than that of MPC-based approaches, the paper does not provide any discussion or quantitative analysis comparing the communication cost with other FHE-based methods.

Moreover, this paper is not the first to implement WNNs under homomorphic encryption, which further weakens the novelty. Another potential issue is that, due to the dependency on the packing scheme, the model structure could be partially inferred from the packing pattern itself. While the reordering strategy to reduce the number of rotations is indeed a strong contribution, it could also expose relative positional information that may leak insights into the network’s architecture.

Finally, the most critical limitation is the simplicity of the dataset used. The proposed method’s computational cost will likely explode as the model architecture or dataset complexity increases, making it difficult to argue that the approach is efficient based on experiments limited to MNIST. Since this is not the first work of its kind, the paper should at least demonstrate applicability on a more complex dataset or architecture—for example, CIFAR-10—and evaluate the method with a larger LUT size (greater than 3) to establish its scalability and effectiveness.

### Questions
I believe a comparison with other FHE studies implementing LUT or comparison operations is necessary. From my perspective, Decision Tree (DT) models share strong similarities with the proposed approach, as both derive outputs based on predefined, learned rules.
In particular, the branching mechanism in DTs appears conceptually very similar to that of this paper (e.g., Level-up style branching). Therefore, I would like to ask whether the proposed method could be extended or adapted to Decision Tree models, and if so, what modifications would be required to make such an extension feasible.

### Soundness
2

### Presentation
3

### Contribution
2
