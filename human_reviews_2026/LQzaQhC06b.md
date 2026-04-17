# Coded-Smoothing Module: Coding Theory Helps Generalization

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
We introduce the Coded-Smoothing module, which can be seamlessly integrated into standard training pipelines, both supervised and unsupervised, to regularize learning and improve generalization with minimal computational overhead. In addition, it can be incorporated into the inference pipeline to randomize the model and enhance robustness against adversarial perturbations.
The design of coded-smoothing is inspired by general coded computing, a paradigm originally developed to mitigate straggler and adversarial failures in distributed computing by processing linear combinations of the data rather than the raw inputs. Building on this principle, we adapt coded computing to machine learning by designing an efficient and effective regularization mechanism that encourages smoother representations and more generalizable solutions. Extensive experiments on both supervised and unsupervised tasks demonstrate that coded-smoothing consistently improves generalization and achieves state-of-the-art robustness against gradient-based adversarial attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel training method for improving model generalization, drawing on the concept of coded computing. The approach involves recoding the input data and using it as an auxiliary branch for regularization during training, thereby enhancing the model's generalization capability.

### Strengths
1. The proposed regularization method is both novel and supported by a comprehensive theoretical analysis.
2.The proposed Coded Smoothing module is designed for flexible integration into both supervised and unsupervised training frameworks, and can be readily reproduced due to its straightforward implementation.

### Weaknesses
1. The method description in this paper is thorough, but the experiment appears relatively weak. The classification experiments should be supplemented with results on real-world datasets. Even under computational constraints, validation on a subset such as ImageNet-100[1] should be considered as a minimum requirement.
2. The generalization performance of the proposed method on OOD data remains insufficiently validated. It would be more convincing to include additional evaluations on specialized OOD benchmarks such as ImageNet-R and ImageNet-S.
3. The comparison with baseline methods remains relatively limited, as only classical approaches including ERM and Mixup are included. It would be beneficial to incorporate more recent and advanced baselines to better demonstrate the superiority of the proposed method.
4. The experimental evaluation of the generative model is currently insufficient, as quantitative metrics alone cannot fully capture the perceptual quality of generated samples. It is essential to include qualitative visualizations of the generated outputs. Furthermore, the analysis should at minimum demonstrate the improvement achieved over Mixup when applied in conjunction with WGAN.
5. The data encoding and decoding process introduces dimensionality expansion to the input. Has there been any analysis on the computational efficiency of this approach, particularly regarding the regularization overhead during training and the additional operations required during inference?

[1] Tian Y, Krishnan D, Isola P. Contrastive multiview coding[C]//European conference on computer vision. Cham: Springer International Publishing, 2020: 776-794.

### Questions
I will consider adjusting my score based on the authors' response to these weakness points.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a coded-smoothing module to enhance generalization in image classification.
Inspired by coded computing (Moradi et al., 2024), the authors design a regularization mechanism that promotes smoother feature representations.
Specifically, the module uses spline-based encoder and decoder functions: (1) the spline encoder is fitted to a batch of inputs x; (2) the encoded batch is processed by the main network f; and (3) the decoded outputs are compared with the originals to enforce consistency.
This process imposes a higher-order smoothness constraint on f, leading to better generalization.
Experiments are conducted on supervised image classification under in-domain, adversarial, and covariate-shift settings, as well as an unsupervised image-generation task.

### Strengths
- The paper introduces a novel perspective by adapting coded computing ideas, originally from distributed and information theory, into the context of ML regularization.

- The proposed coded-smoothing module is simple, lightweight, and applicable to both supervised and unsupervised settings.

- The spline-based encoder/decoder enforces higher-order smoothness, providing an interesting theoretical link between coding theory and representation regularization.

### Weaknesses
- Adversarial robustness results are weaker than MixUp. Under covariate shift, performance is also not better than Mixup.

- Missing comparisons and discussions with Manifold Mixup and related methods that already smooth latent representations. Table 1 and 3 should include full results of Manifold Mixup.

- Conceptual novelty is limited: the idea closely resembles Latent/Manifold Mixup and other interpolation-based regularizers.

- The inference method uses batch-based encoding/decoding (rather than single-sample). This reduces practicality in scenarios where only one input comes at a time.

- Notation and clarity issues: several functions (e.g., g1, g2) are undefined, making parts of the paper difficult to follow.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **Coded-Smoothing**, a new regularization module inspired by **coded computing theory**, which traditionally mitigates straggler and adversarial failures in distributed computing by operating on coded (linearly combined) data. The authors adapt this principle to deep learning, proposing a module that encourages **local smoothness** and **better generalization**, while also improving **adversarial robustness**.

The proposed module operates in three steps:  
1. **Encoding** — combines a batch of inputs into coded samples via spline interpolation.  
2. **Computation** — evaluates the model on the coded samples.  
3. **Decoding** — reconstructs approximate outputs of the original samples and penalizes discrepancies to enforce smoothness.

During inference, the authors propose **Randomized Coded Inference (RCI)**, which randomizes the input order before encoding to disrupt gradient-based adversarial attacks (e.g., FGSM, PGD).  

Experiments on CIFAR-10/100, TinyImageNet, and GANs demonstrate that coded-smoothing improves generalization compared to ERM and Mixup, enhances adversarial robustness, and applies effectively in both supervised and unsupervised settings, all with minimal computational overhead.

### Strengths
- **Novel conceptual link:** The paper establishes a creative and original connection between **coding theory** and **regularization in deep learning**, introducing a theoretically motivated way to enforce smoothness.

- **Unified applicability:** The proposed coded-smoothing module works seamlessly for **both supervised and unsupervised** learning settings, unlike Mixup-style approaches that rely on label information.

- **Strong empirical results:** Demonstrates consistent improvements in test accuracy and adversarial robustness across multiple datasets (CIFAR-10/100, TinyImageNet) and architectures.

- **Minimal computational overhead:** The spline-based implementation adds negligible cost to training and inference, making the method practical for large-scale applications.

- **Adversarial robustness:** The **Randomized Coded Inference (RCI)** strategy offers a simple yet effective defense against gradient-based adversarial attacks without requiring adversarial training.

- **Theoretical grounding:** Provides analytical justification (Lemma 1) that links the number of coded samples and function smoothness to the approximation error, offering theoretical intuition for why the method works.

### Weaknesses
- **Narrow comparative evaluation:** Experiments mainly compare against ERM and Mixup. Other strong baselines such as CutMix, Manifold Mixup, consistency regularization, or adversarial training are missing.

- **Hyperparameter interpretability:** The effects of key hyperparameters (e.g., the weighting factor µ and the ratio of coded samples N/K) are not clearly analyzed or justified, and practical tuning guidance is lacking.

- **Ablation study depth:** Although appendices mention ablation analyses, the main paper does not clearly quantify trade-offs or sensitivity regarding batch size, coded sample count, or the balance between smoothness and accuracy.

- **Assumption of smoothness transfer:** It is assumed that spline-induced smoothness in the input domain translates to smoother representations in feature space, but this relationship is not empirically validated.

- **Robustness evaluation limitations:** The reported adversarial robustness results do not appear to test **adaptive attacks** (e.g., expectation-over-transformation), which may overestimate the protection offered by RCI.

### Questions
1. **Gradient flow and differentiability:**  
   The paper provides a pseudo-code implementation of the coded-smoothing module but does not clarify how gradients flow through the encoder and decoder (spline fitting) steps.  
   Are these spline operations treated as differentiable with respect to the network parameters, or are they non-trainable transformations through which gradients do not propagate?  
   Have you observed any gradient stability issues due to the decoding approximation?

2. **Fixed versus random coding points:**  
   The encoding and decoding points (Chebyshev nodes) appear to be fixed throughout training.  
   Have you experimented with randomizing or re-sampling these points during training, or toward the end of training, to act as an additional stochastic regularizer?  
   Could dynamic or randomized coding points improve generalization or robustness?

3. **Theoretical connection:**  
   Lemma 1 provides intuition about the approximation error, but it remains unclear how the proposed regularization quantitatively affects model smoothness or generalization.  
   Can you provide a more formal connection between coded-smoothing and a Lipschitz or higher-order smoothness bound on the function \( f(\cdot) \)?

4. **Adversarial robustness evaluation:**  
   The reported results demonstrate strong performance under FGSM and PGD attacks.  
   Have you tested the method against **adaptive attacks** that account for inference-time randomness (e.g., expectation-over-transformation)?  
   How does the robustness change when such attacks are considered?

5. **Hyperparameter sensitivity:**  
   How sensitive is performance to the choice of key parameters such as the weighting factor \( \mu \), the ratio \( N/K \), and batch size?  
   Can you offer empirical or theoretical guidance on tuning these parameters?

6. **Selective application:**  
   The paper applies coded-smoothing to the full network.  
   Have you explored applying it only to specific layers or blocks to trade off computational cost and regularization strength?

7. **Comparison to other smoothness-based regularizers:**  
   How does coded-smoothing relate to other smoothness-enforcing techniques like Jacobian regularization, spectral normalization, or consistency regularization?  
   Could it be complementary to these methods?

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
The paper proposes a regularization method: coded-smoothing module to regularize deep networks by enforcing local smoothness: a batch is encoded into multiple coded samples via spline/Chebyshev points, passed through a chosen network block, and then decoded to reconstruct the original block outputs; a reconstruction loss is added to the task loss, and at inference a randomized coded inference (RCI) variant permutes batches to disrupt gradient-based attacks; experiments on CIFAR-10/100, TinyImageNet, WGAN-GP (CIFAR-10, CelebA), and distribution shift (CIFAR-10.1/10.2/10C) show modest in-distribution gains over ERM/mixup, improved GAN IS/FID, and notable robustness boosts under FGSM/PGD (though not competitive with strong adversarial training), supported by a lemma that bounds decoding error improving with function smoothness and larger code size.

### Strengths
* The idea of reframing coded computing as a data/function-space regularizer is novel and goes beyond pairwise linearity such as mixup and SMOTE.
* The presentation is clear, provides a clean algorithmic description, a formal error bound for the coded reconstruction, and ablations/sensitivity in appendices.
* Clarity: the training objective is simple (one mixing coefficient μ) and the module integrates at arbitrary layers without label dependence.
* Leads to generalization improvements and inference-time robustness via RCI without retraining or adversarial training.

### Weaknesses
* The evaluation is limited to vision at CIFAR/TinyImageNet scale, with no ImageNet-1k or transformer/non-vision results, so cross-domain applicability remains uncertain.
* The run time impact is not reported. The proposed module contains mixed component both within module, in loss function and in data space. Therefore it is important to understand the impact on module latency and complexity. 
* Visual messages in figures are often not clear. For example figure 1 (a) misses the core idea of code computing that enforcing closeness between decoded estimates and true outputs; and in figure 1(b) the decision boundary in both panels appear very similar.

### Questions
* Scaling/generalization: how does the method perform on ImageNet-1k and on larger transformers (e.g., ViT/BERT) where sequence batching and attention may interact with coded batches?
* Hyperparameter guidance: the method introduces various new hyperparameters, what robust default choices of N/K, μ, and spline/Chebyshev order work across datasets, or any recommended tuning strategy?
* Compute overhead: what are the training/inference time and memory overheads versus ERM/mixup across N and batch sizes, and how does RCI affect latency?
* Baseline breadth: how does the method compare with CutMix, AugMix, consistency regularization, and post-hoc smoothing/ensembling; any negative interactions?
* Dropout comparison: when combined with (or compared to) dropout/DropBlock/stochastic depth, are the gains additive, redundant, or conflicting, and at which layers should each be applied?
* Figure clarity: can you quantify the two-spirals boundary difference (e.g., curvature/total variation metrics, margin maps) to substantiate the qualitative claim?

### Soundness
2

### Presentation
3

### Contribution
2
