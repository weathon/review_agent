# Domain-Aware Gradient Reuse for Anomaly Detection

- Decision: Reject
- Scores: 4, 4, 2, 8

## Abstract
Anomaly detection relies on recognizing patterns that diverge from normal behavior, yet practical deployment is hampered by the inherent scarcity and heterogeneity of anomalous instances.
These challenges prevent the training set from faithfully characterizing the underlying anomaly distribution, thereby fundamentally constraining the development of effective discriminative models for anomaly detection. 
Inspired by the observed consistency of gradient distributions across related domains during training, Domain-Aware Gradient Reuse (DAGR) is introduced as a transfer‑learning framework that leverages this property.
DAGR first learns an adaptive transformation
by aligning source and target normal gradients, thereby neutralizing domain‑specific effects.  
The same map then pushes forward the source anomalous gradients to computing estimated target anomalous gradients, which are combined with the true target normal gradients to guide the target‑domain detector without labeled anomalies.
This paper establishes a rigorous convergence proof that reinforces the framework’s theoretical foundation. Comprehensive experiments on image and audio datasets demonstrate that the proposed method achieves state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a transfer learning framework called Domain-Aware Gradient Reuse (DAGR), specifically designed to address the problem of anomaly scarcity in anomaly detection. Its core idea is that proximal domains share a similar geometric structure in their gradient distributions for both normal and anomalous data. The method attempts to extract an anomaly gradient signal from a labeled source domain and transfer it to a target domain containing only normal data. The paper provides a theoretical convergence analysis and demonstrates state-of-the-art performance on several image and audio anomaly detection benchmarks.

### Strengths
1. The core idea of performing transfer learning directly in the gradient space, rather than in the feature or output space, is novel. The concept of reusing and transforming gradients from a source domain to guide learning in a target domain is a creative and effective approach.

2. The method is well-structured, combining a channel-selection mechanism (CCCS) to identify domain-invariant components with an adaptive inner-loop optimization (ADPR) to learn the complex gradient transformation. The inclusion of a formal convergence analysis under specific assumptions adds significant theoretical rigor.

3. The paper is generally well-written. The ablation studies are particularly strong, effectively quantifying the contribution of each core component (CCCS and ADPR) and showing that ADPR is the primary driver of performance.

### Weaknesses
1. The convergence proof relies on strong assumptions (e.g., the existence of an invertible map $T$ for label-independent shift, bounded domain-specific perturbations). The paper does not sufficiently discuss how valid these assumptions are in practice, especially for the failed Valve case, which weakens the practical guarantee of the theory.

2. The method's effectiveness relies heavily on several strong assumptions (e.g., gradient consistency, label-independent shift). The empirical validation, while strong on proximal domains, is insufficient to prove its robustness and generalizability under more diverse or challenging domain shifts, making the practical applicability seem narrow.

3. The proposed ADPR module requires an inner-loop optimization for many training step(s), which likely incurs a significant computational overhead compared to standard methods. This practical cost is not analyzed or discussed, which is an important consideration for potential adoption.

### Questions
1. Regarding the practical validity of the theoretical assumptions, could the authors more clearly define the operational boundaries of DAGR? Providing a more comprehensive list or analysis of scenarios where the method is effective versus those where it fails (beyond the Valve case) would greatly help assess its generalizability.

2. The adaptive inner-loop (ADPR) is a key component that likely introduces significant computational overhead. Could the authors provide an analysis of the training time and resource consumption compared to baseline methods? A discussion on this practical cost is essential for evaluating the method's overall utility.

### Soundness
3

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
The paper introduces Domain-Aware Gradient Reuse (DAGR), a transfer-learning framework designed to overcome the challenges of scarcity and heterogeneity of anomalous instances. DAGR is based on the empirical observation that gradient distributions exhibit consistency across related domains during training. The core idea is to estimate the missing anomalous gradient component in the target domain by transferring and denoising the anomalous gradients from a labeled source domain. The framework operates via two key components: (1) the Cross-Domain Consistent Component Selection (CCCS), which filters out domain-specific noise; and (2) the Adaptive Domain-Specific Perturbation Removal (ADPR), which implements the cross-domain gradient distiller. The paper performs experiments on acoustic (DCASE) and visual (DAGM) datasets and achieves state-of-the-art performance.

### Strengths
1. **Re-interpretates Domain Adaptation:** DAGR introduces a gradient-centric perspective to anomaly detection transfer learning, arguing effectively that gradients, rather than features or logits, serve as a potent conduit for knowledge transfer. This reinterprets domain adaptation as the selective reuse of source-domain gradients.

2. **Convergence Proof:** The paper provides a crucial convergence proof showing that the algorithm converges to an optimum neighborhood without error accumulation. Furthermore, the paper rigorously justifies the core hypothesis through the Cross-Class Generalization Theorem, proving that the mapping F learned solely from normal gradients generalizes to align abnormal gradients with the same small error bound $\epsilon$.

3. **Efficiency and Performance:** DAGR achieved significant gains over strong baseline transfer learning methods (PDA, DA, DG) and unsupervised methods. It ranks first in several benchmarks in two different domains.

4. **Benefits of targeted optimization:** The method directly addresses the critical shortcoming of existing transfer-based methods, where source anomalous supervision gradients are optimized for the source distribution and are not guaranteed to benefit the target optimization landscape under domain shift.

### Weaknesses
1. **Sensitivity to Domain Proximity/Compatibility:** The performance of DAGR is highly dependent on the "adjacent-domain scope". The paper explicitly notes that performance suffers substantially on non-proximal transfers (e.g., Fan → Valve) where the gradient-consistency premise is violated. This resulted in significantly depressed AUPRC and Rec@K scores on the Valve target, artificially lowering the overall average.

2. **Implementation Complexity of ADPR:** The Adaptive Domain-Specific Perturbation Removal (ADPR) requires an inner optimization loop where gradients are computed through a "fast weight" construction and back-propagated. The ablation study shows that substituting ADPR with simpler methods like linear gradient mapping leads to a significant performance drop, suggesting the complexity is necessary, but this nested optimization structure may introduce substantial computational overhead that is not fully analyzed in the main body. 

3. **Reliance on Hyperparameter Tuning for CCCS:** The Cross-Domain Consistent Component Selection (CCCS) relies on a channel-masking ratio ($\alpha$). The study determined that $\alpha=$5% provided the best trade-off between noise removal and information retention based on performance peaks observed across four representative benchmarks. The necessity of tuning $\alpha$ to this narrow range suggests sensitivity to this hyperparameter, and it is unclear if a fixed $\alpha$ is universally optimal across all domain pairs (e.g., image vs. audio data) without further tuning.

### Questions
1. **Proximity Diagnosis and Gating Implementation:** The discussion section suggests future work should include a "proximity screen on normal-gradient geometry" to down-weight or disable the mapped anomalous component when domain proximity falls below a threshold, thereby avoiding negative transfer. How do the authors propose to quantify this proximity threshold based on the normal gradient geometry in practice, and what mechanism (e.g., dynamic weight on $\Omega$) would be used for proximity-aware gating in deployment?
2. **Computational Cost Analysis of ADPR:** Given that ADPR employs a meta-learning-style inner optimization loop (Equations 9 and 10), which involves N refinement steps per global training step m, could the authors provide a comparative analysis of the training time or computational complexity (e.g., relative FLOPs) of the full DAGR model versus the ablated variants? Understanding the practical cost of this sophisticated adaptation could play an important role in understanding the overhead of the proposed method in practical scenarios. 
3. **Generalizability of Optimal Masking Ratio $\alpha$:** The optimal channel masking ratio $\alpha$ for CCCS was found to be approximately 5% on the DCASE benchmarks. Did the authors investigate the necessary range of $\alpha$ for the DAGM (visual texture) benchmarks? Is the 5% setting robustly optimal across different sensing modalities and generative mechanisms, or is $\alpha$ a domain-dependent hyperparameter? 
4. **Relationship between Feature Alignment and Gradient Reuse:** Prior work often relies on feature alignment (e.g., Partial Domain Adaptation methods). Since DAGR operates purely in gradient space, have the authors explored combining a feature alignment step with the gradient reuse strategy? Specifically, would incorporating a conventional feature alignment objective mitigate the domain incompatibility issues observed in cases like Fan → Valve, where the gradient consistency premise is not satisfied?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed to learn an adaptive transformation by aligning source and target normal gradients to neutralize the domain-specific effects. Then, the same map pushes forward the source anomalous gradients to computing estimated target anomalous gradients, which are combined with the true target normal gradients to guide the target-domain detector without labeled anomalies. A rigorous convergence proof to justify the proposed method was given.

### Strengths
1.	This proposed transfer-learning framework remains effective even when the target domain contains no anomalous samples.
2.	The gradient-centric perspective to anomaly detection is promising for adaptation strategies grounded in gradient compatibility.

### Weaknesses
1.	The motivation is not solid.
2.	The related work is too distracting. The authors should focus on the transfer-learning based anomaly detection methods.
3.	Some of the equations are not numbered.
4.	The motivation in section 3.1 is very essential to this paper, however, the authors failed to justify it properly.
5.	Table 1 is not on the correct page.
6.	The experimental results are not given with the deviations.
7.	There is no limitation discussion.

### Questions
1.	How do you calculate the anomalous gradients $\nabla\Psi^-$ on the target domain when these data are missing.
2.	What is the gradient distribution alignment principle and how can you confirm the correctness of this principle?
3.	How to confirm the gradient decomposition hypothesis?
4.	How to confirm the claim: The key insight is that gradients—rather than features or logits—exhibit strong cross-domain regularities?
5.	How do you explain the failed case in Table 1, i.e., DCASE target Valve? What do you mean by non-stationary and cross-mechanism?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper “Domain-Aware Gradient Reuse for Anomaly Detection (DAGR)” introduces a novel transfer-learning framework that enables anomaly detection in target domains where no anomalous samples are available. Traditional methods struggle with anomaly scarcity and domain heterogeneity, which limit generalization across datasets. DAGR addresses this by exploiting the observed cross-domain consistency of gradient distributions during training. Instead of aligning features or logits, it operates directly in gradient space, learning an adaptive mapping between source and target gradients. This mapping, trained using only normal data, is then reused to estimate target-domain anomalous gradients from source-domain ones—allowing the target detector to receive informative updates without labeled anomalies. The framework consists of two key modules: Cross-Domain Consistent Component Selection (CCCS), which filters domain-invariant gradient channels, and Adaptive Domain-Specific Perturbation Removal (ADPR), which refines gradients through an inner optimization loop. The method provides theoretical convergence guarantees and achieves state-of-the-art results on both image and audio benchmarks.

### Strengths
Strengths:
1. Proposes a novel gradient transfer perspective.

    Traditional cross-domain anomaly detection primarily focuses on feature alignment or latent distribution matching. As far as I know， your work DAGR, for the first time, elevates cross-domain knowledge transfer to the gradient space. By analyzing the consistency of gradient distributions across different domains, it discovers that alignment can be directly achieved at the gradient level. This is a new theoretical and practical paradigm, distinct from feature alignment or adversarial alignment. This is a good angle to approach the question, introducing gradient-level transfer provides a completely new research direction for anomaly detection and domain adaptation.

2. Applicable to real-world scenarios with no anomaly samples in the target domain.

    In most industrial and medical scenarios, the target domain only contains normal samples and lacks anomaly annotations. DAGR can generate pseudo-anomaly gradients through cross-domain gradient mapping, achieving supervised updates even with no anomaly samples. This effectively solves the scarcity–diversity dilemma of anomaly detection. It can be directly applied in real-world production scenarios without the need to collect anomaly samples in the target domain.

3. The theoretical foundation is comprehensive.
    The authors provide proofs of convergence and cross-class generalization in the appendix, even when the mapping F is trained using only normal gradients, it can generalize to anomalous gradients. And gradient errors do not accumulate during iteration, and the optimization process monotonically converges. The feasibility and stability of "gradient reuse" are theoretically verified. Combining the generalization error bounds of gradient alignment and domain adaptation to form a complete convergence analysis framework.

### Weaknesses
Weaknesses:
1. Limited applicability, I think your work relies on neighboring domain assumption. The core premise of DAGR is that the gradient distributions of the source and target domains have near-isometric geometry. And in Sec. 4.3 Discussion, you mentioned performance degrades significantly in the "Valve" task, because Valve and Fan differ greatly in their physical mechanisms, gradient geometrical geometry does not hold. Therefore, DAGR is only effective in adjacent domains. It cannot handle strongly distributed offsets or heterogeneous domains (such as audio→image, speech→mechanical vibration), but I don’t think this is a big problem. Traditional methods also cannot solve such large cross-domain problems.
2. The DAGR operation operates in gradient space, which leads to higher memory overhead and sensitivity to gradient noise and batch size. Higher implementation complexity and training cost compared to traditional feature alignment methods. I want to know if there are issues with training instability or computational bottlenecks in large-scale models.
3. While your idea of ​​learning on gradients is excellent, I believe it lacks modeling of the semantics of anomalous patterns. Although DAGR transfers the direction of anomalous gradients, this process is a pure gradient-level approximation and does not explicitly model the semantic or structural features of anomalous samples. If the anomalous types are completely different in the target domain, gradient mapping alone may fail to capture meaningful decision boundaries. Therefore, I think this method, focusing only on gradient geometric consistency and neglecting semantic consistency, may limit the model's optimal performance.

### Questions
1. How sensitive is DAGR to the degree of domain similarity, and can it be extended or regularized to handle stronger distribution shifts or heterogeneous modalities?
2. Could you elaborate on whether you observed any training instability or scalability bottlenecks when applying DAGR to large-scale or transformer-based detectors, and how such issues might be mitigated?
3. How does your framework perform when the anomalous categories in the target domain are semantically dissimilar to those in the source domain?

### Soundness
4

### Presentation
3

### Contribution
3
