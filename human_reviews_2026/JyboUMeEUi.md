# CardioComposer: Leveraging Differentiable Geometry for Compositional Control of Anatomical Diffusion Models

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 6

## Abstract
Generative models of 3D cardiovascular anatomy can synthesize informative structures for clinical research and medical device evaluation, but face a trade-off between geometric controllability and realism. We propose CardioComposer: a programmable, inference time framework for generating multi-class anatomical label maps from interpretable ellipsoidal primitives. These primitives represent geometric attributes such as the size, shape, and position of discrete substructures. We specifically develop differentiable measurement functions based on voxel-wise geometric moments, enabling loss-based gradient guidance during diffusion model sampling. We demonstrate that these losses can constrain individual geometric attributes in a disentangled manner and provide compositional control over multiple substructures. Finally, we show that our method is compatible with a broad range of anatomical systems containing non-convex substructures, spanning cardiac, vascular, and skeletal organs. We release our code at https://github.com/kkadry/CardioComposer.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel framework for imposing geometric constraints models without model retraining. This is achieved by introducing a differentiable geometric loss during inference. The loss aims to match the geometric moments (size, position, and shape) between the predictions and targets. The framework was evaluated on the Totalsegmentator dataset and compared to several baseline methods that integrate the geometric information either explicitly or implicitly.

### Strengths
1. The paper presents a novel framework for anatomical structure generation that enables flexible control over geometric attributes during inference without requiring retraining, a notable advantage over conventional conditional generative approaches.
2. The proposed geometric loss formulation is both interpretable and modular, leveraging moment-based constraints to disentangle control over size, shape, and position. The method further supports compositional multi-part control and demonstrates utility in simulation-based downstream tasks.
3. The authors have shown model's robustness against hyperparameters and non-convex shapes.

### Weaknesses
The method introduces a novel formulation in Eq. (4) to define and normalize geometric moments, offering a clear way to separate size, shape, and orientation characteristics. However, the overall idea of combining a Latent Diffusion Model with test-time constraints has been explored in prior works, so the main novelty lies in the specific geometric moment formulation rather than the general framework.

### Questions
Good paper with clear practical relevance and a well-motivated clinical application. While it does not introduce major technical innovations, it effectively addresses a real-world problem and demonstrates solid empirical value in the medical field.

### Soundness
4

### Presentation
4

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
The paper proposes CardioComposer, a 3D method enabling differentiable, geometry-aware control of anatomical diffusion models. The main contribution is to introduce geometric moment-based losses that are differentiable, which can guide sampling to produce anatomically realistic and geometrically controllable multi-class label maps of the heart.

### Strengths
1. Using geometric moments as differentiable constraints for guiding diffusion models of anatomy is novel.
2. The paper presents clear mathematical formulation of 0th, 1st and 2nd order geometric moments.
3. The paper includes baselines such as explicit and implicit concatenation, cross-attention.
4. The paper evaluates different combinations of proposed loss as the ablation study.

### Weaknesses
1. The experiment is focusing on cardiac structures with limited samples (less than 600 in total). The generalizability and scalability remain concerns.
2. Are the unconditional latent diffusion models (3D VAE and 3D UNet) trained from scratch?
3. In the Figure 7, for unconditional generation, why does the implicit fail but the proposed one seems to be able to generate a much more realistic map? Further explanation or more visualization would strengthen this observation.

### Questions
Will the code be released to the public?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces "CardioComposer," a novel inference-time guidance framework designed to control unconditional diffusion models for the generation of 3D multi-class anatomical segmentations. The primary problem it addresses is the trade-off between geometric controllability and realism in generative models for cardiovascular anatomy, which is a critical component for clinical research and in silico medical device evaluation.

The core contribution is the development of differentiable measurement functions based on voxel-wise geometric moments. These functions compute interpretable geometric attributes—specifically size (from zeroth-order moments), position (from first-order moments), and shape (from scale-normalized second-order moments)—from the generated label maps. The gradients from a loss function comparing these measured moments to target moments are then used to guide the diffusion sampling process at inference time.

### Strengths
1) The core contribution—a flexible, inference-time guidance mechanism based on differentiable geometric moments for 3D multi-class segmentations—is somewhat novel. This method directly tackles a significant bottleneck in the use of generative models for scientific simulations, where interpretable control over specific geometric properties is paramount. The ability to generate "digital siblings" for counterfactual simulations is interesting.

2) The paper is well-structured and clearly written. Figures are informative (outstanding), though some are dense. The related work section is comprehensive. The writing is precise.

3) Extensive evaluation: 200 samples per method, 5+ baselines, 3 metric families (morphological, pointcloud, conditional fidelity). Strong ablations: disentanglement, compositional control, guidance weight tuning. Real-world validation: Used to edit patient anatomy for biophysical simulation (RV volume vs. wall displacement), showing causal form-function modeling.

4) The method is technically sound, with a clear formulation of the geometric moment losses (size, position, and scale-normalized shape). The experimental validation is rigorous.

### Weaknesses
The authors are commendable for their transparent limitations section. Some weaknesses:

1) The framework relies on weights $\lambda_i$ for the aggregate loss and a guidance weight $w$. The paper notes $\lambda_i$ require "experimental tuning" and provides the final values in the appendix37. However, the sensitivity to these weights is not explored. How robust is the method to changes in these hyperparameters?

2) The authors rightly admit that the underlying diffusion model can generate topologically incorrect anatomies, which is a critical failure for biophysical simulations. The proposed solution is post-hoc filtering. It is unclear if the proposed geometric guidance affects the rate of these topological errors

3) While framed as general, all experiments are on cardiac data.

4) In Figure 4, while the proposed method  generally maintains high-quality metrics, there appears to be a slight degradation in realism at high guidance weights ($w > 1.5$)

### Questions
1) How sensitive is the method to VAE reconstruction error?

2) Can you control orientation independently of shape?

3) What happens under extreme geometric targets (e.g., 10× mass)?

### Soundness
4

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
CardioComposer adds a geometric control mechanism to an unconditional diffusion model at inference time, using differentiable moments—volume (0th), centroid (1st), and scale-normalized covariance (2nd)—to control the size, position, and shape of cardiac parts.
No retraining is required: each denoising step decodes a segmentation, measures geometry, computes a loss to target values, and backpropagates its gradient to adjust the latent, enabling disentangled and composable control.
Trained on 596 multi-label cardiac segmentations, the method achieves stable single- and multi-part control, geometric inpainting, and biomechanical simulation compatibility.

### Strengths
1. **Clinical Significance**
The paper addresses a clinically relevant issue—how to create anatomically realistic yet controllable 3D cardiac models. 

2. **Flexible Post-training Optimization**
The proposed inference-time optimization framework is elegant and practical. By introducing differentiable geometric moments as guidance losses, the method allows users to customize geometric targets on demand without retraining or modifying network weights. This design provides an interpretable and lightweight mechanism for controllable shape generation.

3. **Comprehensive Experimental Validation**
The experiments are thorough and convincing. They demonstrate geometric controllability across multiple substructures, evaluate both single- and multi-part guidance, and further validate the realism of generated shapes using distributional metrics and biomechanical simulations. Together, these results strongly support the effectiveness and robustness of the proposed approach.

4. **Clarity and Completeness of Presentation**
The paper is clearly written and technically detailed.

### Weaknesses
1. **Lack of comparison with existing conditional shape generation methods.**

While the paper clearly articulates the benefits of inference-time guidance, it does not compare against established conditional generative models trained for geometric control (e.g., de Wilde et al., 2025 (https://arxiv.org/abs/2504.03313); Kadry et al., 2025 (https://www.nature.com/articles/s41746-024-01332-0), A-SDF (https://arxiv.org/abs/2104.07645). Including such baselines would better quantify the extent to which performance or controllability is improved by the proposed guidance relative to conditional training.

2. **Absence of simulated validation data.**

Although the authors compute geometric attributes (mass, centroid, covariance) from real cardiac segmentations, the evaluation relies solely on real anatomies. A complementary experiment on simulated or procedurally generated geometries—with known ground-truth attributes—would allow denser and more diverse target sampling, providing clearer evidence of control accuracy, disentanglement, and robustness.

3. **Limited explanation for Figures 6 and 8.**

The pair-plot visualizations are intriguing but under-explained: it remains unclear which visual trends (e.g., sharper peaks, denser clusters) indicate successful guidance.


4. **Code is not available**

### Questions
1. The generation quality appears to depend on the amount of training data available. How large a dataset is required for the diffusion model to achieve stable and realistic shape synthesis?

2. Apart from the 0th- and 1st-order moments, are there other geometric features that could serve as control signals?

### Soundness
3

### Presentation
4

### Contribution
3
