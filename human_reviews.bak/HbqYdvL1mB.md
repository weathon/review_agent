# Point-Calibrated Spectral Neural Operators

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
Two typical neural models have been extensively studied for operator learning, learning in spatial space via attention mechanism or learning in spectral space via spectral analysis technique such as Fourier Transform. Spatial learning enables point-level flexibility but lacks global continuity constraint, while spectral learning enforces spectral continuity prior but lacks point-wise adaptivity. This work innovatively combines the continuity prior and the point-level flexibility, with the introduced Point-Calibrated Spectral Transform. It achieves this by calibrating the preset spectral eigenfunctions with the predicted point-wise frequency preference via neural gate mechanism. Beyond this, we introduce Point-Calibrated Spectral Neural Operators, which learn operator mappings by approximating functions with the point-level adaptive spectral basis, thereby not only preserving the benefits of spectral prior but also boasting the superior adaptability comparable to the attention mechanis. Comprehensive experiments demonstrate its consistent performance enhancement in extensive PDE solving scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a new neural architecture termed point-calibrated spectral neural operators for learning-based PDE solving. 
The key idea is to adaptively compute a spectral gate to modulate/calibrate the fixed eigenfunctions (spectral basis) of the Laplace-Beltrami transform. Experiments on multiple PDE problems (with structured or unstructured meshes) collectively demonstrate the proposed methods outperform the state-of-the-art method (e.g., the Transolver) by a substantial margin, in terms of not only the approximation accuracy but also the sample efficiency and cross-resolution generalizability.

### Strengths
As I stated in the [Summary], the proposed method shows a clear improvement over the state-of-the-art in PDE fast solving. Also, the paper is in general clearly written and easy to digest.

### Weaknesses
However, the proposed method borrows significantly from the existing closely related works. Specifically, the spectral eigenfunctions (of the Laplace-Beltrami transform) are borrowed directly from an unpublished work [A]. And the basis form of the adaptive Gate functions (one of the key contributions claimed by authors) is widely employed in neural network literature (see, for instance, the squeeze-and-excitation network [B] proposed six years ago). For me, the main message claimed in this paper is that replacing the attention module by these aforementioned components in a classic transformer backbone leads to sizeable performance gain for neural operator learning. This is interesting, but may not be novel enough for ICLR publication. 

Beside, I would not buy the explanation of "Pointwise Frequency Preference Learning" claimed by the authors. Indeed, the gate values are computed in a point-wise manner. However, it doesn't suggest it acts like a frequency selector. If one would implement an adaptive frequency selector, only one single scalar should be multiplied with each eigenfunction instead of the elementwise multiplication. IMO, the gate calibrates/modulates the eigenfunctions to make them more suitable to characterize the given data, but not in the form of frequency preference. After all, frequency can only be evaluated by a set of points instead of a single point (the uncertanty principle)

[A] Gengxiang Chen, Xu Liu, Qinglu Meng, Lu Chen, Changqing Liu, and Yingguang Li. Learning neural operators on riemannian manifolds. arXiv preprint arXiv:2302.08166, 2023.  
[B] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper proposes a Point-Calibrated Laplace-Beltrami Transform, which utilizes spatial information to assist the spectral selection of the spectral neural operator, effectively adapting to spatial variations in partial differential equation systems.

### Strengths
This method is very effective at adapting to spatial variations in systems of partial differential equations, and the organization and writing of the paper are quite good.

### Weaknesses
Minor typos:
The ‘Denotd’ in Figure 1.
Questions:
1. The Laplace-Beltrami Transform is an innovation inspired by the reference "Learning Neural Operators on Riemannian Manifolds." This paper merely introduces Pointwise Frequency Preference Learning based on that work. The introduction of auxiliary modules inevitably increases model parameters and computational complexity, but the paper does not provide ablation studies on parameters. Therefore, it is unclear whether the performance improvement is due to the increased parameters or the proposed algorithm.
2. The theoretical foundation is insufficient. The paper does not provide a unified kernel integral form as presented in [1]. Additionally, it does not prove that PCSM is equivalent to a learnable integral on Ω as demonstrated in [2].
3. In the Zero-shot Resolution Generalization experiment presented in the paper, the training resolution of 211 × 51 is greater than that of the test resolution. So, how would the model perform when generalizing to even larger resolutions?

[1] Neural operator: Learning maps between function spaces with applications to pdes. 
[2] Transolver: A Fast Transformer Solver for PDEs on General Geometries.

### Questions
Minor typos:
The ‘Denotd’ in Figure 1.
Questions:
1. The Laplace-Beltrami Transform is an innovation inspired by the reference "Learning Neural Operators on Riemannian Manifolds." This paper merely introduces Pointwise Frequency Preference Learning based on that work. The introduction of auxiliary modules inevitably increases model parameters and computational complexity, but the paper does not provide ablation studies on parameters. Therefore, it is unclear whether the performance improvement is due to the increased parameters or the proposed algorithm.
2. The theoretical foundation is insufficient. The paper does not provide a unified kernel integral form as presented in [1]. Additionally, it does not prove that PCSM is equivalent to a learnable integral on Ω as demonstrated in [2].
3. In the Zero-shot Resolution Generalization experiment presented in the paper, the training resolution of 211 × 51 is greater than that of the test resolution. So, how would the model perform when generalizing to even larger resolutions?

[1] Neural operator: Learning maps between function spaces with applications to pdes. 
[2] Transolver: A Fast Transformer Solver for PDEs on General Geometries.

### Soundness
3

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
3

### Summary
The authors have addressed some of my concerns, but I still have questions regarding the low-frequency and high-frequency aspects. I am willing to consider increasing my score.

### Strengths
Good organization and structure of the paper, along with solid experimental analysis.

### Weaknesses
The authors have addressed some of my concerns, but I still have questions regarding the low-frequency and high-frequency aspects.

### Questions
The authors have addressed some of my concerns, but I still have questions regarding the low-frequency and high-frequency aspects.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Point-Calibrated Spectral Neural Operator. This new approach combines the flexibility of attention-based methods and the spectral continuity constraints from spectral-based neural operators.

### Strengths
This paper introduces a novel hybrid approach by integrating spectral methods with point-wise adaptability, which is an advancement in the field of neural PDE solvers. It bridges the gap between spectral continuity and spatial flexibility, potentially positioning it as a unique contribution to both operator learning and spectral-based neural networks.

The paper demonstrates a deep understanding of spectral analysis and neural operator design, incorporating the Laplace-Beltrami Transform to handle irregular domains and non-uniform meshes. The theoretical basis for using point-wise frequency preferences and neural gates to calibrate spectral features seems well-founded and addresses specific challenges of both spectral and attention-based methods.

The experiments cover a range of structured and unstructured PDE problems, offering quantitative and qualitative results that consistently show the advantages of PCSM over baselines. The paper explores zero-shot resolution generalization and limited training data scenarios, demonstrating PCSM’s robustness, adaptability, and reduced dependence on extensive data or frequency inputs.

### Weaknesses
1. The methodology, while innovative, may appear complex to readers unfamiliar with spectral methods, neural operators, or Laplace-Beltrami transforms. This could limit accessibility for a broader audience within ICLR, where general machine learning applications dominate. Some sections might benefit from clearer explanations or illustrative examples, especially in the methodology, to help general readers understand the motivation behind certain design choices.

2. Although the PCSM model integrates spatial adaptability, its reliance on spectral transforms might make it challenging to apply in domains where calculating or interpreting spectral features (e.g., LBO eigenfunctions) is difficult or computationally expensive. This approach may also limit the model’s use in non-spectral applications unless the paper provides guidelines on extending or adapting the point-calibrated spectral mechanism to more generalized feature spaces.

3. The experiments utilize computationally demanding resources (like the A100 GPU), which could suggest that PCSM is resource-intensive, particularly for calculating point-wise frequency preferences and training on large irregular domains. Although results show reduced dependence on high-frequency inputs, the model still involves complex frequency selection and spectral processing steps, which might require optimization for real-world applications.

4.  While the paper’s experiments are extensive and well-rounded, they are conducted on benchmark problems. Including a real-world PDE application, or additional industry-related scenarios, could strengthen the claim of broader applicability. Demonstrating PCSM’s utility outside synthetic PDE problems, in genuinely complex physical systems or industrial settings, would bolster its practical significance.

### Questions
1. Comparative Analysis: The paper presents a strong case for the PCSM approach over fixed spectral methods. Could further analysis be provided to illustrate specific PDE scenarios where point-calibration particularly excels compared to the fixed approach?

2. Computational Efficiency: Given the reported need for A100 GPUs, are there any considerations for deploying PCSM in resource-constrained environments? Could simplified versions of PCSM be implemented without significant accuracy loss?

3. Real-World Application Scenarios: Has PCSM been tested on any real-world data or practical engineering problems? Including results from real-world applications (e.g., fluid dynamics in engineering contexts) could add to the paper’s practical significance.

4. Sensitivity to Frequency Parameters: How sensitive is PCSM’s performance to changes in the number of frequency modes (Nk)? Are there guidelines for choosing this parameter in cases where computational resources are limited?

5. Ablation on Multi-Head Mechanism: The multi-head mechanism is mentioned as an enhancement to the spectral mixer. Have the authors conducted any ablation studies to understand how varying the number of heads affects performance?

6. Explainability of Point-Wise Frequency Preferences: Have the authors considered using interpretability techniques to explain the model’s learned point-wise frequency preferences? For instance, do specific frequencies correspond to particular physical phenomena in the PDE domains?

7. Future Extensions of Point-Calibrated Spectral Transform: The paper suggests broader applicability for time-series and computer vision tasks. Could the authors provide some insights into how the Point-Calibrated Spectral Transform might be adapted for non-spectral data, such as images?

Minor comment:
1. Denotd should be Denoted in Figure 1: “Denotd as 'PCSM (w/o Cali)'”​

### Soundness
3

### Presentation
3

### Contribution
3
