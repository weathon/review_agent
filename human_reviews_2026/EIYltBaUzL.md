# Robustify Spiking Neural Networks via Dominant Singular Deflation under Heterogeneous Training Vulnerability

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Spiking Neural Networks (SNNs) process information via discrete spikes, enabling them to operate at remarkably low energy levels. However, our experimental observations reveal a striking vulnerability when SNNs are trained using the mainstream method—direct encoding combined with backpropagation through time (BPTT): even a single backward pass on data drawn from a slightly different distribution can lead to catastrophic network collapse. We refer to this phenomenon as the heterogeneous training vulnerability of SNNs. Our theoretical analysis attributes this vulnerability to the repeated inputs inherent in direct encoding and the gradient accumulation characteristic of BPTT, which together produce an exceptional large Hessian spectral radius. To address this challenge, we develop a hyperparameter-free method called Dominant Singular Deflation (DSD). By orthogonally projecting the dominant singular components of gradients, DSD effectively reduces the Hessian spectral radius, thereby preventing SNNs from settling into sharp minima. Extensive experiments demonstrate that DSD not only mitigates the vulnerability of SNNs under heterogeneous training, but also significantly enhances overall robustness compared to key baselines, providing strong support for safer SNNs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
1. Research Question
- The paper addresses the instability and vulnerability of Spiking Neural Networks (SNNs) under heterogeneous training conditions and adversarial perturbations. 
- Key problem: Why do SNNs collapse under small distribution shifts, and can this be mitigated by stabilizing their spectral dynamics during training?

2. Method Proposed: Dominant Singular Deflation (DSD)
- DSD is a spectral regularization technique applied during backpropagation. 
- Instead of modifying weights or data, DSD removes the dominant singular component from the gradient matrix (Δ W) at each update step. 
- DSD suppresses the gradient’s most unstable direction, aiming to reduce spectral explosion and improve training stability and robustness without introducing new hyperparameters or architectural changes.

3. Theoretical Part
- The authors conduct theoretical analysis based on the spectral properties of the Jacobian and Hessian during BPTT.
- The paper formalizes *heterogeneous training vulnerability* as a consequence of the exponential growth of the Hessian’s spectral radius over time steps.
- It shows (under simplifying assumptions) that the dominant singular direction of (Δ W) aligns across time, amplifying instability.
- Removing the dominant singular direction theoretically reduces the Hessian’s largest eigenvalue, ensuring descent and smoother loss curvature.

4. Experimental Part
- Experiments on CIFAR-10/100, TinyImageNet, and event-based datasets show: DSD stabilizes training (smoother loss, bounded gradient norms).
- DSD Improves robustness under FGSM, PGD without adversarial training, maintains or slightly improves clean accuracy.
- Spectral diagnostics (Hessian radius, singular value histograms) support the theoretical mechanism.

### Strengths
This work reminds me of a relevant paper: SNN-RAT (https://openreview.net/forum?id=xwBdjfKt7_W). The two works do have very similar views—both aim to improve robustness by controlling the spectral properties of SNNs (singular values/Lipschitz constants). However, they have subtle but important differences in their conceptual approach, theoretical approach, and technical implementation. Therefore, I will use SNN-RAT as a comparison to discuss this paper.


1. Theoretical Part

- Presents a novel *optimization-space* spectral stabilization method, contrasting prior *parameter-space* approaches such as SNN-RAT, which constrain the largest singular value of (W). DSD instead regularizes the gradient matrix Δ W, introducing a fresh angle on robustness grounded in training dynamics.
- Provides a clear and mathematically coherent link between Hessian spectral growth, gradient alignment, and heterogeneous training instability — a valuable conceptual contribution to understanding SNN optimization behavior.
- Demonstrates theoretical descent guarantees via spectral deflation, suggesting that DSD maintains convergence while suppressing dominant curvature directions.
- DSO is elegant, minimal, and free of additional hyperparameters, making it an analytically interpretable robustness mechanism.

2. Experimental Part

- Empirical results show that DSD improves both training stability and adversarial robustness without adversarial training, which is a notable distinction from previous works relying on adversarial data augmentation like SNN-RAT/HoSNN.
- Consistent improvements across CIFAR-10/100, TinyImageNet, and DVS datasets, while preserving or slightly improving clean accuracy.
- Spectral diagnostics (gradient singular spectra, Hessian radius) align closely with the theory, strengthening internal validity.
- The evaluation includes white-box, black-box scenarios, reducing concerns about gradient obfuscation.

3. Method Scalability

- The DSD operation is simple and parameter-free, requiring only rank-1 spectral deflation per layer, theoretically lightweight compared to full spectral regularization in SNN-RAT.
- The method integrates seamlessly into standard backpropagation and does not alter network architecture, making it easy to adopt in existing SNN frameworks.
- DSO can reduce the reliance on adversarial samples in SNN adversarial training, which greatly reduces the amount of computation while providing adversarial robustness.

### Weaknesses
1. Theoretical Part
Overall, the paper's proof follows the path of "spectral analysis of the Jacobian product → increasing the radius of the Hessian spectrum → shrinking the Deflation spectrum → improving stability." This is intuitively reasonable.

- It would be helpful for the rigor and clarity of the paper if the author could clearly state the key assumptions in each theorem. In particular, it would deepen our understanding of the problem by indicating when the core theorems proposed fail.

2. Experimental Part
Overall, I think the experimental part is clear, comprehensive, and rigorous.
- More attack methods like Gaussion noise, APGD (https://arxiv.org/pdf/2003.01690) could be tested. Only FGSM and PGD are tested in the current paper. 


3. Method Scalability
- The proposed DSD step requires layer-wise SVD or power iteration, which is computationally demanding for large-scale or convolutional SNNs; the paper should quantify this cost in the main paper.
- DSD depends on full gradient access and is thus incompatible with neuromorphic or local learning implementations, limiting its deployability on real spiking hardware.
- The direct modification of the principal components of the gradient is a concern. Even though the paper provides good results overall, unpredictable results may occur when interacting with other adaptive optimizer such as Adam and RMSProp.

### Questions
1.Could the authors explicitly state the key assumptions/conditions in each theorem?   For example:
  * Under what conditions do the contraction and boundedness assumptions on the Jacobian fail?
  * Are there cases where the spectral radius of the Hessian would not scale linearly with time steps (T)?
  * What happens if the “approximate time-invariance” assumption of the Jacobian is violated (e.g., due to noise, dropout, or adaptive thresholds)?

2. Could the authors consider evaluating the robustness under additional perturbations such as Gaussian noise or stronger adversarial attacks e.g., APGD?

3. Could the authors quantify DSD's computational cost (e.g., time per iteration, FLOPs, or scaling with network size)

4. Could the authors discuss potential adaptations or approximations that make DSD compatible with neuromorphic chips?

5.  Have the authors observed any instabilities or performance inconsistencies during DSD training process or when DSO is combined with different optimizers?

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
In this work, the authors present a novel method, Dominant Singular Deflation (DSD), to address the vulnerability of Spiking Neural Networks (SNNs) under heterogeneous training conditions. The authors provide both theoretical and empirical evidence to support their claims, demonstrating significant improvements in robustness across multiple datasets and attack scenarios. The work is timely and addresses an interesting issue in the safe deployment of SNNs.

### Strengths
1. This manuscript identifies and systematically analyzes the phenomenon of model collapse under heterogeneous training—a realistic yet understudied scenario. The theoretical analysis linking BPTT and direct encoding to the growth of the Hessian spectral radius is rigorous and insightful.
2. In this work, the proposed DSD method is hyperparameter-free. It effectively reduces the spectral radius of the Hessian and preserves the descent property, ensuring stable training without introducing significant overhead.
3. The authors conduct extensive experiments across multiple static and neuromorphic datasets, under both homogeneous and heterogeneous training settings, and against a variety of white-box and black-box attacks. The results consistently show that DSD outperforms existing SOTA methods in robustness.

### Weaknesses
1. As reported in Table 1, DSD leads to a noticeable decrease in clean accuracy compared to vanilla SNNs. This trade-off between robustness and clean performance may limit its applicability in certain real-world scenarios where high clean accuracy is required. However, the reviewer's previous research also discovered similar phenomena. So, what thoughts does the author have regarding the improvement of this issue?
2. While the paper identifies the combination of BPTT and direct encoding as the main culprit for vulnerability, it does not extensively explore how DSD performs with other training paradigms (e.g., SLTT) or encoding methods beyond direct and rate encoding.
3. The author seems to have overlooked some studies on the robustness of SNNs [1-3] in the related work.

[1] "Enhancing the robustness of spiking neural networks with stochastic gating mechanisms." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 1. 2024.

[2] "Towards effective training of robust spiking recurrent neural networks under general input noise via provable analysis." 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD). IEEE, 2023.

[3]"RSC-SNN: Exploring the Trade-off Between Adversarial Robustness and Accuracy in Spiking Neural Networks via Randomized Smoothing Coding." Proceedings of the 32nd ACM International Conference on Multimedia. 2024.

### Questions
Please see weaknesses!

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper indicates that spiking neural networks (SNNs) trained using the backpropagation through time (BPTT) algorithm are inherently susceptible to perturbations from heterogeneous training data (clean and corrupted). The paper analyzes this susceptibility and concludes that it stems from an excessively high Hessian spectral radius. To address this issue, the authors propose Dominant Singular Deflation (DSD), a method that explicitly removes the dominant rank-one singular component from the gradient during training. The author conducted experiments on multiple datasets and demonstrated that their method significantly improves the robustness of SNNs.

### Strengths
1. From a novel perspective, this paper identifies a source of robustness vulnerability in SNNs.
2. A thorough theoretical analysis supports the proposed method.
3. The experimental results demonstrate the performance advantages of the proposed method.

### Weaknesses
1. Whether the analysis presented in this paper holds true for other training methods, such as other parallel training [1] approaches or single-step SNNs that propagate firing rates [2], and encoding schemes, such as temporal encoding, remains to be seen.
2. The proposed method produced SNNs that performed significantly worse than other robust methods on clean data—a clear drawback.
3. As shown in Table 7, the proposed method significantly increases training time. Training time nearly doubles on the static CIFAR10 and CIFAR100 datasets.

```
[1] Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability. NeurIPS. 2023.
[2] Scaling Spike-Driven Transformer With Efficient Spike Firing Approximation Training. IEEE TPAMI. 2025.
```

### Questions
See the weakness.

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
4

### Summary
This paper investigates the unstable training issue in heterogeneous training of SNNs. To address this, they propose a hyperparameter-free method named Dominant Singular Deflation (DSD). This reduces the Hessian spectral radius, prevents convergence to sharp minima, and enhances robustness under different training conditions. Extensive experiments across multiple datasets (CIFAR, TinyImageNet, ImageNet) demonstrate that DSD improves both robustness and stability without incurring inference overhead.

### Strengths
- The paper is well-structured and easy to follow.
- The proposed method is elegant, hyperparameter-free, and mathematically grounded, making it practical for real-world SNN training.
- The method is validated across multiple datasets and architectures, including both static (CIFAR, ImageNet) and event-driven (DVS) data, consistently outperforming state-of-the-art baselines.

### Weaknesses
- The experimental validation does not clearly substantiate the theoretical motivation of the proposed method. While the Dominant Singular Deflation (DSD) algorithm is designed to suppress the unbounded growth of the Hessian’s spectral radius, it is unclear how this mechanism directly translates into enhanced robustness of the SNN models?
- The paper states that DSD is designed to mitigate model collapse during heterogeneous training, but most experimental evaluations emphasize adversarial robustness under homogeneous training. Furthermore, while DSD improves robustness, it also yields the lowest clean-data accuracy in Table 1, which contradicts the paper’s stated goal of achieving stable and reliable training.
- The theoretical analysis in Theorems 1 and 2 relies on the ****Gauss–Newton (GN) Hessian approximation rather than the true Hessian. Since the GN Hessian is always positive semidefinite, this assumption may limit the generality of the theoretical results.

### Questions
Could the authors please clarify what specific training scenario hetero-training refers to in this paper and what is the difference between hetero and homo-training?

### Soundness
2

### Presentation
3

### Contribution
3
