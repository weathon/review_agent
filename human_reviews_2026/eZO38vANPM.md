# IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Test-time adaptation (TTA) has been widely explored to prevent performance degradation when test data differ from the training distribution.
However, fully leveraging the rich representations of large pretrained models with minimal parameter updates remains underexplored.
In this paper, we propose Intrinsic Mixture of Spectral Experts (IMSE) that leverages the spectral experts inherently embedded in Vision Transformers. 
We decompose each linear layer via singular value decomposition (SVD) and adapt only the singular values, while keeping the singular vectors fixed.
We further identify a key limitation of entropy minimization in TTA: it often induces feature-collapse, causing the model to rely on domain-specific features rather than class-discriminative features.
To address this, we propose a diversity maximization loss based on expert–input alignment, which encourages diverse utilization of spectral experts during adaptation.
In the continual test-time adaptation (CTTA) scenario, beyond preserving pretrained knowledge, it is crucial to retain and reuse knowledge from previously observed domains. We introduce Domain-Aware Spectral Code Retrieval, which estimates input distributions to detect domain shifts, and retrieves adapted singular values for rapid adaptation.
Consequently, our method achieves state-of-the-art performance on various distribution-shift benchmarks under the TTA setting.
In CTTA and Gradual CTTA, it further improves accuracy by 3.4 percentage point (pp) and 2.4 pp, respectively, while requiring 385 times fewer trainable parameters. 
Our code is available in https://github.com/baek85/IMSE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces IMSE (Intrinsic Mixture of Supervised Experts), a test-time adaptation method that decomposes linear layers in a pretrained model using singular value decomposition (SVD), treating each rank-1 component as a spectral expert. During adaptation, only the singular values are updated while the bases remain fixed, allowing parameter-efficient updates. The method combines entropy minimization with confidence-based filtering, a diversity maximization loss to promote balanced expert activation, and a domain descriptor mechanism for detecting distribution shifts and retrieving previously adapted spectral parameters from a memory bank.

### Strengths
The use of spectral decomposition to form a mixture-of-experts representation is an interesting approach to test-time adaptation.

### Weaknesses
1. The assumption that spectral experts can generalize across domains by only adapting singular values lacks theoretical justification and is not thoroughly evaluated in challenging domain-shift scenarios.
2. The interplay between the different components (e.g., diversity loss, entropy minimization, and domain memory) is complex, but the paper lacks sufficient ablation studies to isolate their individual contributions.
3. The scalability and stability of the domain memory mechanism under long-term continual adaptation are not fully explored, leaving questions about its robustness in practice.

### Questions
1. he framework assumes that adapting only the singular values of pretrained linear layers is sufficient for capturing domain shifts. Could the authors provide theoretical or empirical evidence supporting this assumption, especially for shifts that may require new basis directions?
2. The method combines entropy minimization, diversity regularization, and domain memory. Have the authors performed detailed ablations to assess the individual and joint impact of each component? For example, how does performance change when the diversity loss is removed?
3. The KL-divergence-based domain descriptor mechanism is used to trigger new adaptations and retrieve stored parameters. How robust is this approach under gradual or noisy distribution shifts? What mechanisms are in place to prevent over-segmentation or memory bloat when many small shifts occur?\
4. The diversity regularization loss penalizes widespread activation of individual spectral experts. Could this unintentionally suppress task-relevant features that happen to be common in a new domain? How does the method balance between promoting diversity and retaining important features?
5. Could the authors discuss whether the IMSE framework or its components (e.g., diversity loss, spectral adaptation) can be extended to multi-source domain generalization or continual learning settings where test-time updates are not allowed?
6. While the method is evaluated on common TTA benchmarks, it would be helpful to include results on tasks involving more gradual or subtle domain shifts, or with more fine-grained domain boundaries. Could the authors include such evaluations in a revision?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the test-time adaptation (TTA) problem by leveraging the knowledge in pretrained networks. Specifically, it decomposes each linear layer via singular value decomposition (SVD) and adapts only the singular values. For continual test-time adaptation (CTTA), the method estimates input distributions to detect domain shifts and retrieves previously adapted singular values for rapid adaptation. Extensive experiments demonstrate that the proposed approach achieves state-of-the-art performance.

### Strengths
1. Using $\text{Std}_{i}^{(l)}$ to measure whether each expert captures domain-specific patterns is an interesting design.

2. Detecting new domains by comparing the current input-level descriptor $ϕ$ with the accumulated descriptor $\phi_{(t)}$ is an effective way.

3. This paper analyzes, in an unsupervised manner, that entropy minimization tends to capture domain-related rather than class-discriminative information.

### Weaknesses
1.  In lines 161–163, since $\mathbf{v}_i^{(l)}$ fixed, it is unclear how it can define each spectral expert’s response. Could the authors provide a more detailed explanation of this point.

2. The paper conducts experiments only on three OOD datasets. Could authors also evaluated the method on classical domain adaptation benchmarks such as Office-Home[1] or DomainNet[2]?

3. The paper introduces two loss components, entropy minimization $L_{entmin}$ and diversity maximization $L_{dm}$, but lacks an ablation study isolating their individual contributions. Could the author show the effect of using only entropy minimization, only diversity maximization, and their combination to better understand each term’s impact on adaptation performance.

4. The paper does not include experiments on time analysis (e.g., runtime efficiency) or memory analysis (e.g., computational cost and storage of spectral codes). Could authors provide such experiments to better demonstrate the efficiency of the proposed method?

[1] Venkateswara, Hemanth, et al. "Deep hashing network for unsupervised domain adaptation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[2] Peng, Xingchao, et al. "Moment matching for multi-source domain adaptation." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

### Questions
Please see the above weaknesses. If you can conduct additional experiments to further evaluate your method, I would be willing to raise my score.

### Soundness
2

### Presentation
3

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
This paper proposes Intrinsic Mixture of Spectral Experts, a novel framework for test-time adaptation. IMSE updates only the singular values while keeping the singular vectors fixed, thus enabling parameter-efficient adaptation. To prevent the entropy minimization objective from collapsing feature diversity, the authors introduce a diversity maximization loss based on spectral vector–input alignment. For CTTA, they propose a Domain-Aware Spectral Code Retrieval mechanism, which stores domain-specific singular values and retrieves them according to domain similarity to mitigate forgetting. Experiments on ImageNet-C  and CLIP  backbones show that IMSE achieves state-of-the-art results while using fewer trainable parameters.

### Strengths
1. The paper correctly identifies two practical problems in TTA: the overfitting tendency of entropy minimization and catastrophic forgetting in CTTA.
2. The idea of reinterpreting linear layers as mixtures of rank-1 “spectral experts” is interesting. It provides a compact and interpretable way to perform fine-grained adaptation.
3. Updating only singular values makes IMSE computationally efficient and easily applicable to existing pretrained models such as ViT or CLIP.
4. The reported improvements on ImageNet-C and the low parameter count are convincing, and the experiments are well organized.

### Weaknesses
1. How robust is the retrieval process when encountering a completely new domain that differs substantially from all stored domains?
2. Updating only singular values implicitly constrains parameter updates to a diagonal submanifold of the low-rank space. Does this constraint reduce optimization expressivity when confronted with severe domain shifts?
3. How much extra computational cost is introduced by SVD?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces IMSE (Intrinsic Mixture of Spectral Experts), a new framework for continual test-time adaptation (CTTA). IMSE decomposes each linear layer in Vision Transformers using singular value decomposition (SVD), interpreting the orthogonal basis matrices as a mixture of spectral experts and the diagonal matrix of singular values as spectral weights. During adaptation, IMSE fine-tunes only these spectral weights while keeping the orthogonal bases fixed, enabling highly parameter-efficient adaptation.

To guide unsupervised adaptation, the authors propose a combined loss function that integrates entropy minimization with a diversity maximization term to prevent feature collapse. IMSE achieves state-of-the-art performance on ImageNet-C, -R, and -A benchmarks, outperforming strong TTA baselines such as TENT, SAR, and DPAL while requiring significantly fewer trainable parameters.

For continual TTA, the paper further introduces a Domain Bank that stores domain-specific spectral codes and descriptors, enabling retrieval and reuse of prior adaptations to mitigate catastrophic forgetting. Comprehensive ablation studies demonstrate the contribution of each proposed component.

### Strengths
Originality: IMSE offers a novel interpretation of test-time adaptation through the lens of spectral experts, extending SVD-based parameter-efficient fine-tuning (e.g., LoRA, SVDiff) to the unsupervised TTA setting. The idea of freezing orthogonal bases and adapting only singular values is conceptually simple yet highly effective.

Efficiency and Practicality: By updating only singular values, IMSE achieves orders-of-magnitude parameter reduction (~2000× fewer trainable parameters) while maintaining or exceeding SOTA performance.

Continual Adaptation Innovation: The introduction of the Domain Bank provides a simple but powerful mechanism for mitigating catastrophic forgetting across sequential domains.

Comprehensive Ablation Analysis: The paper includes well-designed ablation studies that isolate the impact of each component.

### Weaknesses
Domain Bank scalability:
Using simple KL divergence over mean–variance descriptors might struggle to discriminate fine-grained domain shifts, potentially leading to incorrect retrievals.

No direct measure of forgetting:
Although the Domain Bank is designed to mitigate catastrophic forgetting, the paper does not include quantitative evidence, e.g. Backward Transfer (BWT), to verify that performance on previously adapted domains remains stable after subsequent adaptations.

### Questions
Could the authors quantify the computational cost of performing full SVDs across all linear layers in large-scale ViTs (e.g., CLIP or MAE)? Have they considered using truncated or randomized SVD to reduce preprocessing overhead, and if so, how would this affect adaptation performance?

### Soundness
4

### Presentation
3

### Contribution
3
