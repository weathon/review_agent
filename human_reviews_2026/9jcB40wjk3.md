# SAFA-SNN: Sparsity-Aware On-Device Few-Shot Class-Incremental Learning with Fast-Adaptive Structure of Spiking Neural Network

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6, 4

## Abstract
Continuous learning of novel classes is crucial for edge devices to preserve data privacy and maintain reliable performance in dynamic environments. However, the scenario becomes particularly challenging when data samples are insufficient, requiring on-device few-shot class-incremental learning (FSCIL). Although existing work has explored parameter-efficient FSCIL frameworks based on artificial neural networks (ANNs), their deployment is still fundamentally constrained by limited device resources. Spiking neural networks (SNNs) process spatiotemporal information efficiently, offering lower energy consumption, greater biological plausibility, and compatibility with neuromorphic hardware than ANNs. In this work, we propose an SNN-based method containing Sparsity-Aware neuronal dynamics and Fast Adaptive structure (SAFA-SNN) for on-device FSCIL. By threshold regulation, most neurons exhibit stable spikes and others exhibit adaptive spikes. As a result, synaptic traces that encode base-class knowledge are naturally preserved, thereby alleviating catastrophic forgetting. To cope with spike non-differentiability in backpropagation, we employ a gradient-free technique, i.e., zeroth-order optimization. Moreover, class prototypes can limit overfitting on few-shot data but introduce bias. We enhance prototype discriminability by orthogonal subspace projection. Extensive experiments conducted on two standard benchmark datasets (CIFAR-100 and Mini-ImageNet) and three neuromorphic datasets (CIFAR10-DVS, DVS128 Gesture, and N-Caltech101) demonstrate that SAFA-SNN outperforms baselines, specifically achieving at least 4.01\% improvement at the last incremental session on Mini-ImageNet and 20\% lower energy cost on CIFAR-100 over baselines with practical implementation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study focuses on the Few-Shot Class-Incremental Learning (FSCIL) problem, aiming to enable edge devices to continuously learn novel classes under realistic constraints of limited data and energy resources. To achieve this, the authors propose the SAFA-SNN framework, which comprises three main components: Sparsity-Aware Neuron Dynamics, Zeroth-Order Optimization, and Subspace Projection for prototype alignment. Notably, all experiments are implemented and rigorously evaluated on real edge devices.

### Strengths
1. The computation energy measurement methodology employed in this study is notably fair and well-justified.

2. This work pioneers the exploration of on-device SNN-based FSCIL, establishing a solid empirical baseline for future research.

### Weaknesses
1. Some equations lack proper punctuation (e.g., Eqs. 14 and 15), and the third section of Table 3 could be better centered for readability.

2. Although the Zeroth-Order Optimization (ZOO) section provides an upper bound on convergence, it lacks quantitative comparison with surrogate backpropagation and detailed analysis of potential error sources.

3. The paper mentions the degradation issue in deeper networks (e.g., ResNet-34) but does not provide a systematic explanation or proposed solution.

### Questions
Overall, the paper is comprehensive and well-structured; however, I would appreciate clarification on a few points:

1. Does the subspace projection introduce prototype bias toward base classes, potentially impairing performance on novel classes?

2. Is the division between active and stable neurons fixed or adaptively updated during training?

3. Why is the initial mask retained during incremental learning phases? Could this lead to over-saturation of active neurons over time?

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
4

### Summary
This paper applies spiking neural networks (SNNs) to the few-shot class-incremental learning (FSCIL) scenario and proposes SAFT-SNN, which integrates sparsity-aware neuronal dynamics, zeroth-order optimization, and fast adaptive prototype subspace projection. The author deployed the SNN on a mobile platform and conducted experiments on static and neuromorphic datasets. The results demonstrated that their method outperformed other comparative methods.

### Strengths
The experimental results comparing the proposed method with other approaches demonstrate its superior performance and efficiency.

### Weaknesses
1. Why do sparsity-aware neural dynamics allow most neurons to remain stable while activating only a few? The authors should explain the method they propose in more detail.
2. Although the author deployed the proposed method on a mobile platform, they evaluated theoretical power consumption rather than actual runtime power consumption. Moreover, the results in Figure 5 indicate that its speed does not significantly outperform other methods. This casts doubt on the true efficiency of the proposed method.
3. The proposed SAFA-SNN is nearly impossible to deploy on event-driven neuromorphic chips because Eq. (14) involves computationally intensive matrix multiplication. Therefore, it appears that the efficiency and power consumption advantages of SNNs cannot be realized.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes SAFA-SNN (Sparsity-Aware Feature Alignment), a novel framework designed to improve the efficiency and performance of Spiking Neural Networks (SNNs). The core idea is to leverage sparsity-aware alignment between membrane potentials and firing spikes, thereby improving representation stability under sparse activations. The method integrates feature alignment loss with adaptive threshold tuning to balance accuracy and energy efficiency. Extensive experiments on datasets such as CIFAR10-DVS, DVS-Gesture, and ImageNet-DVS are provided.

### Strengths
The paper effectively identifies a key issue in SNN training — misalignment between spike-based representations and underlying continuous features due to sparse and discrete firing dynamics. The motivation is clearly articulated and well supported by empirical visualization. The introduction of a sparsity-aware term that aligns latent representations between spike and potential spaces is interesting. It extends prior surrogate gradient ideas by explicitly modeling alignment errors as optimization objectives.

### Weaknesses
1. Although the “sparsity-aware alignment” terminology is new, conceptually it overlaps with prior ideas such as temporal consistency regularization and membrane potential loss. The authors should clarify in what way SAFA differs fundamentally rather than being another variant of existing potential-based regularization.
2. The sparsity-aware loss is introduced heuristically without formal derivation. It would be beneficial to provide an analysis connecting the proposed alignment loss to information preservation or stability bounds under sparse activations.
3. The ablation in Table 5 (p. 8) only evaluates the removal of individual terms, but does not show sensitivity to sparsity thresholds or alignment weights. Additionally, no visual explanation of the “feature alignment” effect is provided beyond one layer.
4. Since the method claims to be “efficiency-oriented,” a more detailed discussion on energy or FLOPs reduction would strengthen the claim. Currently, results are purely accuracy-based, with no measured power or latency comparison on neuromorphic hardware.
5. It remains unclear whether SAFA-SNN generalizes well under noise, temporal jitter, or unseen sensor dynamics. These aspects are critical for practical deployment of SNNs in event-based vision systems.
6. The paper describes the feature alignment mechanism intuitively but does not provide a clear mechanistic link between sparsity-aware alignment and improved gradient flow or spike stability. It remains unclear why aligning spike and potential features under sparsity constraints leads to consistent accuracy gains. A more detailed theoretical or empirical explanation (e.g., gradient norm analysis) would be valuable.
7. Given that the proposed loss introduces additional terms involving sparsity and alignment, it is important to analyze how this affects training convergence and gradient stability. However, there is no discussion or empirical observation (e.g., training curves or variance analysis) to ensure that the added loss does not destabilize optimization.

### Questions
1. How is the “sparsity-aware weighting” computed dynamically — is it global, per-layer, or per-neuron?
2. What is the relationship between SAFA and temporal consistency loss?
3. Does SAFA introduce any inference-time overhead? If not, please clarify whether alignment is training-only.
4. Can SAFA be combined with spike transformer architectures such as QKFormer or SpikingResformer?
5. Have the authors evaluated the stability of training curves (loss oscillation, firing rate variance)?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Addressing the challenge of on-device Few-Shot Class-Incremental Learning (FSCIL), this paper proposes SAFA-SNN, a method based on Spiking Neural Networks (SNNs) which are more energy-efficient than traditional ANNs.  The framework introduces sparsity-conditioned neuronal dynamics, where most neurons remain stable to mitigate catastrophic forgetting. To overcome training difficulties with SNNs, it employs zeroth-order optimization for gradient estimation. Furthermore, SAFA-SNN uses subspace projection during incremental learning to enhance the learning of new classes and avoid overfitting. Experiments demonstrate that SAFA-SNN outperforms baselines, achieving at least 4.01% improvement on Mini-ImageNet and 20% lower energy costs.

### Strengths
(+) **Pioneering SNN-Based Solution**: It proposes SAFA-SNN, which is noted as the first SNN-based framework designed to solve the general on-device FSCIL problem, offering a novel, energy-efficient alternative to traditional ANNs.

(+) The SAFA is built around Spiking Neural Networks (SNNs), making it inherently suitable for low-power, hardware-friendly deployment on neuromorphic chips or devices like the Jetson series.

(+) The SAFA-SNN framework effectively tackles three core challenges with specific components:

 - Sparsity-Aware Dynamics: Helps mitigate catastrophic forgetting by design.

 - Zeroth-Order Optimization: Provides a practical solution to train SNNs despite the non-differentiable nature of spikes.

 - Subspace Projection: Enhances the model's ability to learn new classes from few examples without overfitting.

### Weaknesses
**Insufficient Justification for Subspace Projection**: The paper fails to demonstrate the specific advantages of its Subspace Projection method. To prove its effectiveness, a comparative analysis against other projection techniques or feature space regularization methods commonly used in FSCIL is necessary.

**Lack of Comparison to Related Methods**: The novelty of the proposed components is unclear due to missing baseline comparisons:

 - Sparsity-Aware Dynamics: This mechanism appears conceptually similar to the subnet [1] or the "soft-subnetwork" approaches presented in SoftNet [2, 3]. The paper should include a direct comparison to highlight the unique contributions and performance benefits of its approach.

 - Zeroth-Order Optimization: The effectiveness of zeroth-order optimization could be contrasted with other methods.

**References**:

[1] Few-shot lifelong learning

[2] On the Soft-Subnetwork for Few-shot Class Incremental Learning

[3] Continual Learning: Forget-Free Winning Subnetworks for Video Representations

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a novel approach, SAFA-SNN, for on-device few-shot class-incremental learning (FSCIL) utilizing Spiking Neural Networks (SNNs). The main contribution of the work is a sparsity-aware SNN architecture designed to mitigate catastrophic forgetting and improve model adaptation to new classes with very few samples. SAFA-SNN integrates several innovative components: sparsity-aware neuronal dynamics, zeroth-order optimization to handle non-differentiability in SNNs, and subspace projection to enhance discriminability of prototypes in the learning process.

### Strengths
1. Tackling FSCIL in an on-device, rehearsal-free setting with SNNs is a timely and important research direction, aligning with the needs of edge computing and energy efficiency.
2. Reporting actual energy consumption measurements from a Jetson Orin AGX device, rather than just theoretical calculations, adds practical credibility to the energy efficiency claims.

### Weaknesses
1. The abstract and introduction are overly high-level and vague. Key terms like "sparsity-conditioned neuronal dynamics," "zeroth-order optimization," and "subspace projection" are mentioned but not intuitively explained.
2. The paper positions itself as the "first SNN-based solution towards general on-device FSCIL." However, it does not adequately address why an SNN-based approach is fundamentally superior for this problem compared to a highly optimized, parameter-efficient ANN-based FSCIL method. The comparison in Table 1 is against other SNN-converted FSCIL methods. To claim a true advancement, comparisons against state-of-the-art non-spiking (ANN) on-device FSCIL methodsare essential. 
3. The abstract claims the method "alleviates overfitting to novel classes." This is a core challenge in FSCIL, but the mechanism (subspace projection) is not sufficiently motivated or explained in the abstract to justify this claim.
4. The claim of "20% lower energy cost" is presented without immediate context. Is this compared to an ANN baseline or another SNN baseline? This crucial detail is missing from the abstract, making the claim difficult to interpret.

### Questions
1. What is the primary novelty: a new SNN neuron/dynamics model, or the application of existing FSCIL strategies (like prototype/projection methods) within an SNN framework? The abstract currently conflates these.
2. Given the widespread success and efficiency of surrogate gradient methods for training SNNs, what is the specific disadvantage of surrogates in the FSCIL setting that necessitates the use of ZOO? The computational overhead of ZOO is typically higher; does this not negate some of the on-device efficiency gains?
3. How does SAFA-SNN compare, in terms of final accuracy and energy consumption, to a strong, non-spiking, parameter-efficient FSCIL method (e.g., based on prompt tuning or adapter modules) running on the same hardware? This is the most critical comparison to establish the value of using SNNs.
4. Can you provide a more detailed, intuitive explanation of the "sparsity-aware dynamic threshold" mechanism? How is sparsity measured and enforced? How does this directly relate to stabilizing old knowledge and integrating new knowledge?

### Soundness
3

### Presentation
2

### Contribution
2
