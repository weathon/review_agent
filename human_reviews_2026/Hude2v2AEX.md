# Enhancing Communication Compression via Discrepancy-aware Calibration for Federated Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Federated Learning (FL) offers a privacy-preserving paradigm for distributed model training by enabling clients to collaboratively learn a shared model without exchanging their raw data. However, the communication overhead associated with exchanging model updates remains a critical challenge, particularly for devices with limited bandwidth and battery resources. 
Existing communication compression methods largely rely on simple heuristics based on magnitude or randomness. 
For example, Top-k drops the elements with small magnitude, while low-rank methods such as ATOMO and PowerSGD truncate singular values with small magnitude. 
However, these rules do not account for the discrepancy between the compressed and the original outputs, which can lead to the loss of important information. 
To address this issue, we propose a novel discrepancy-aware communication compression method that enhances performance under severely constrained communication conditions.
Each client uses a small subset of its local data as calibration data to directly measure the output discrepancy induced by dropping candidate compression units and uses it as a compression metric to guide the selection. 
By integrating this strategy, we can enhance existing mainstream compression schemes, enabling more efficient communication.
Empirical results across multiple datasets and models show that our method achieves a significant improvement in accuracy under stringent communication constraints, notably an $18.9\\%$ relative accuracy improvement at a compression ratio of $0.1$, validating its efficacy for scalable and communication-efficient FL.
Our code is available at https://github.com/wzy1026wzy/Discrepancy-aware-Compression-for-FL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a new way to do compression in distributed optimization, especially for federated learning. To that aim, the clients use a small subset of their local data as calibration data to measure the output discrepancy induced by compression.

### Strengths
Reducing the communication cost in distributed optimization, in particular for federated learning, is crucial. The approach of taking into account the discrepancy induced by compression is interesting. The experiments are convincing.

### Weaknesses
My main concern is that you use the word "heuristic" multiple times to characterize existing compression methods, in a dismissive and incorrect way. In fact, this is quite the opposite: existing approaches come with thorough theoretical guarantees, which is the contrary of being heuristic, but these guarantees are based on worst-case analysis, which is often too conservative. For instance, top-k and rand-k have the same guarantees, because the worst case vector to which they are applied is a vector will all elements equal, and in that case top-k and rand-k are equivalent. But top-k performs better in practice, and this is intuitive: it is better to keep the coordinates with largest amplitude, which capture the most information, than random coordinates chosen blindly. You should rephrase your statements and cite some papers, such as Beznosikov et al. "On biased compression for distributed learning," 2023; Condat et al. "EF-BV: A Unified Theory of Error Feedback and Variance Reduction Mechanisms for Biased and Unbiased Compression in Distributed Optimization," 2022; Yi et al. "FedComLoc: Communication-Efficient Distributed Training of Sparse and Quantized Models," 2025.

### Questions
No

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
4

### Summary
This paper tackles the key issue of communication costs in federated learning. Existing methods like Top-k sparsification and low-rank decomposition such as ATOMO depend on magnitude heuristics. These approaches fail in federated learning. Clients send accumulated parameter updates over local steps, not single gradients. This breaks the link between magnitude and real importance. The paper introduces a discrepancy-aware compression strategy. It avoids magnitude as an importance measure. Clients use a small local data subset for calibration. This directly assesses output changes from dropping each compression unit, like an element or singular triplet. The method keeps units with the largest output impact. It uses the limited bandwidth more wisely. The experiments demonstrate its effectiveness by augmenting both Top-k and ATOMO , achieving significant accuracy improvements, especially under high compression ratios and non-IID data distributions.

### Strengths
1.The paper's primary strength is its core insight. It correctly diagnoses why existing heuristics fail in the FL paradigm (parameter updates vs. gradients) and proposes a solution that directly exploits a key characteristic of FL (infrequent communication allows for more local computation). This is a fundamental contribution.

2.The experimental results are comprehensive and convincing. The method shows consistent, significant gains over baselines across multiple models, datasets, and compression types (element-wise and low-rank). The gains are most pronounced in the most challenging regimes: high compression and high data heterogeneity.

3.The paper is written with clarity. The motivation, method, and results are all easy to follow.

### Weaknesses
1.The overhead analysis in Table 4 is transparent about the per-round computational cost, showing a non-trivial increase (e.g., ~12-18% total time increase per round in some cases). The authors argue this is an acceptable trade-off. This argument would be much stronger if supported by a "time-to-accuracy" plot. Given the large accuracy gains, it is highly likely that the discrepancy-aware method converges in fewer rounds, potentially leading to a faster total wall-clock time to reach a target accuracy, even with the higher per-round cost. This analysis is currently missing.

2.The paper mentions PowerSGD in the introduction and states the framework is compatible with it. However, the low-rank experiments are conducted only with ATOMO. Given that PowerSGD is a more practical and widely used low-rank approximation than ATOMO (which requires a full SVD), demonstrating the "plug-in" capability and gains on PowerSGD would have significantly strengthened the paper's practical claims.

3.The paper's experiments are rightly focused on comparing the augmented methods to their original magnitude-based versions. However, the related work section mentions other advanced compression techniques, such as adaptive quantization (e.g., FedFQ, FedAQ). While the proposed method addresses selection and not quantization, a comparison or discussion of how it performs against other SOTA communication-efficiency methods (not just magnitude-based ones) would help situate its contribution in the broader landscape.

### Questions
1.The core idea is to find a better importance metric than magnitude. Could this discrepancy-aware metric $L_{comp}(u)$ also be used to guide other compression types, such as quantization?  Could it be used to allocate more bits to elements or singular triplets that have a high output discrepancy score?

2.The paper claims compatibility with PowerSGD. Were there any specific technical challenges that prevented its inclusion in the experiments? Would the application of the discrepancy metric be as straightforward as it is for ATOMO, given that PowerSGD approximates the SVD using iterative methods?

3.There is a subject-verb agreement error in lines 723–724: "the content being compressed (parameter updates) carry little gradient." The subject "the content" is singular, but the verb "carry" is plural. It should be corrected to "carries little gradient" to agree with the singular noun "content. You may also  check similar constructions elsewhere in the paper.

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
3

### Summary
This paper proposes a discrepancy-aware compression method for Federated Learning. It uses a small local calibration dataset to measure the output impact of dropping compression units, replacing conventional magnitude-based rules. This approach serves as a plug-in to enhance existing methods like Top-k and ATOMO, significantly boosting accuracy under tight communication budgets.

### Strengths
1. Results showing significant accuracy gains under high compression ratios.	
2. The method is designed as a plug-in, making it compatible with existing compression schemes.
3. Provides theoretical motivation and intuitive examples to illustrate the limitations of magnitude-based heuristics.

### Weaknesses
1.While the proposed method introduces non-negligible computational overhead due to the calibration process—a significant concern for resource-constrained devices—the empirical validation of its performance gain is not sufficiently comprehensive. The scope of tested compression methods is limited to Top-k and ATOMO. To fully justify the incurred overhead, it is crucial to demonstrate the method's effectiveness across a broader spectrum of techniques, such as SignSGD/z-SignSGD and other quantization-based methods.

2. Lack of comparison with FedAvg or other uncompressed baselines to illustrate the absolute performance gap.

3. No discussion on whether the method can be combined with compensation mechanisms (e.g., error feedback) to further reduce compression loss.

4.The experiments are confined to computer vision tasks. The effectiveness on NLP datasets and models remains unverified, limiting the generalizability of the claims.

### Questions
Please see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Federated Learning (FL) enables privacy-preserving collaborative training by allowing clients to jointly learn a shared model without exchanging raw data. However, the communication overhead from model updates remains a key bottleneck, especially for resource-constrained devices. Existing compression methods largely rely on magnitude-based (e.g., Top-k) or randomness-based heuristics (e.g., ATOMO, PowerSGD), which ignore the discrepancy between compressed and original outputs, often leading to critical information loss. They propose a discrepancy-aware compression method that significantly boosts performance under extreme communication constraints. Each client uses a small subset of its local data for calibration to directly measure the output discrepancy caused by dropping candidate compression units and uses this as a selection criterion. This strategy can be integrated into mainstream compression schemes to enhance communication efficiency. Experiments show a 18.9% relative accuracy improvement at a compression ratio of 0.1, demonstrating its effectiveness for scalable and communication-efficient FL.

### Strengths
1. The paper addresses an important problem with insightful observation.
2. The paper is well-written and clearly organized.

### Weaknesses
1.	Traditional compression algorithms come with theoretical convergence guarantees. Does this algorithm also ensure convergence? 
2.	The experiments suffer from significant shortcomings. In fact, the efficiency of conventional distributed communication compression methods largely stems from their use of error compensation mechanisms, yet this paper does not compare against such approaches in its baselines. 
3.	In traditional distributed communication settings, compression ratios are typically above 1%; in contrast, the highest compression ratio evaluated in this paper is only 10%.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
