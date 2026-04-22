# SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Deep neural networks achieve impressive performance but remain difficult to interpret and control. We present \textbf{SALVE} (Sparse Autoencoder-Latent Vector Editing), a unified "discover, validate, and control" framework that bridges mechanistic interpretability and model editing. Using an $\ell_1$-regularized autoencoder, we learn a sparse, model-native feature basis without supervision. We validate these features with Grad-FAM, a feature-level saliency mapping method that visually grounds latent features in input data. Leveraging the autoencoder's structure, we perform precise and permanent weight-space interventions, enabling continuous modulation of both class-defining and cross-class features. We further derive a critical suppression threshold, $\alpha_{\text{crit}}$, quantifying each class’s reliance on its dominant feature, supporting fine-grained robustness diagnostics. Our approach is validated on both convolutional (ResNet-18) and transformer-based (ViT-B/16) models, demonstrating consistent, interpretable control over their behavior. This work contributes a principled methodology for turning feature discovery into actionable model edits, advancing the development of transparent and controllable AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The current work proposes a three-step process meant to introduce a permanent change in the weights of a given model aiming to control its behavior:

**Step 1 (Discover):** First, a Sparse Autoencoder (SAE) is trained to decompose activations. According to the description provided in Appendix 4, it is a set of two linear layers (an encoder and a decoder), featuring no activation functions,  trained to minimize the reconstruction loss with an L1 penalty applied on the latent SAE features.

**Step 2 (Validate):** The authors apply Grad-CAM on activations from intermediate representation provided by the SAE to determine whether or not a certain feature is semantically relevant for the given task.

**Step 3 (Control):** In the final step the weights of the original model are adjusted by means of adding or subtracting the weights (associated to the selected SAE features) of the SAE decoder layer from the original model’s weights.

Experiments are performed on a Resnet-18 and a ViT-B/16 using the Imagenette benchmark showing that the method is able to identify features that are relevant for certain classes and to steer the decisions of the model.

### Strengths
- The paper is well written, clearly structured and presented.  
- The evaluation is performed both on a CNN and a Transformer model.  
- The results show that the method is able to steer the decisions of the models under the circumstances of the considered setup.

### Weaknesses
**Step 1 (Discover):** It is unclear why the authors have chosen to employ an autoencoder that differs significantly both in terms of structure and in terms of the training objective from the current state-of-the-art SAEs \[1, 2\]. The chosen design and training objective is not motivated in the paper and these decisions do not seem to properly model the goals of an SAE.

**Step 2 (Validate):** The proposed validation step requires human intervention which is not feasible in the context of large models and it is not an improvement over current automatic feature selection methods \[3, 4\].

**Step 3 (Control):** It is unclear why the standard approach of hooking an SAE module to an existing transformer is any less permanent than a weight change. The authors could argue that it is more computationally efficient to change the weights directly rather than passing the activations through an SAE, but not that it is more permanent. Furthermore, the procedure proposed in Eq. 1 seems to be similar to what SAEs regularly do. In a regular setup, increasing or decreasing an SAE latent results in adding or subtracting a decoder weight from the initial activations. Similarly, in Eq. 1, the procedure simply adds or subtracts a decoder weight. While changing the weights rather than the activations is more efficient, it comes with the downside that the addition/subtration proposed in Eq. 1 is always performed with a fixed coefficient, while in a regular setup that coefficient is adjusted for each sample based on the latent SAE representation.

**Evaluation:** The current evaluation procedure is (i) restricted to two small models, (ii) restricted to a classification setup, (iii) restricted to a small benchmark and (iv) lacking a proper comparison with respect to existing approaches. While the current work cites a sum of methods aimed at model control, it does not provide a proper comparison between the proposed approach and the existing ones. In order to prove the efficiency of the proposed approach one or more of the following could have been employed: (i) showing that the approach can steer the prediction of a model towards a gender-neutral one without affecting hair color prediction on CelebA \[5\], (ii) showing that the proposed approach can effectively steer language models \[6\] towards exhibiting desired behaviors, (iii) showing that the proposed approach can effectively steer diffusion models \[7\].

\[1\] Rajamanoharan et al., “Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders”  
\[2\] https://transformer-circuits.pub/2025/january-update/index.html  
\[3\] https://transformer-circuits.pub/2024/scaling-monosemanticity/  
\[4\] Cywinski et al., “SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders”  
\[5\] Gerych et al., “BendVLM: Test-Time Debiasing of Vision-Language Embeddings”  
\[6\] Kim et al., “Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)”  
\[7\] Surkov and Wendler et al., “One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models”

### Questions
[Step 1] Why was the current SAE implementation chosen and how does it compare to existing approaches?

[Step 2] Can the authors provide a comparison between their selection mechanism and existing ones, highlighting the improvements brought in this regard?

[Step 3] Can the authors provide an in depth analysis between their weight space approach and the existing approaches to using SAE for steering?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes using a sparse autoencoder to discover interpretable concepts within a classification model, identifying class-related concepts. Based on these concepts, it edits the final layer to alter model outputs. The method is validated on both ResNet and ViT architectures using the Imagenette dataset.

### Strengths
1. The paper introduces an interpretable model editing method to directly adapt model behavior.
2. Intervention sensitivity analysis establishes a quantitative suppression threshold to calibrate model edits.

### Weaknesses
1. Using SAEs to identify concepts and highlight relevant image regions has been established, such as [1]. Similarly, suppressing model components to adapt model behavior has been explored in works like [2]. Therefore, the technical novelty of this work seems limited.

2. More extensive comparisons with recent model editing or concept-based intervention methods (e.g., [3]) are needed to better demonstrate potential advantages.

3. Experiments are conducted on the relatively small-scale Imagenette dataset, raising concerns about scalability to larger datasets like ImageNet.

[1] Sparse Autoencoders Reveal Selective Remapping of Visual Concepts during Adaptation, ICLR 2025

[2] SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders, ICML 2025

[3] Decomposing and Editing Predictions by Modeling Model Computation, ICML 2024

### Questions
1. Why focus only on editing the final layer? Could exploring SAE concepts and editing in earlier layers enable adapting the model's behavior more effectively?

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
4

### Summary
The paper introduces SALVE (Sparse Autoencoder-Latent Vector Editing), a three-stage pipeline to discover, validate, and control model-native features. A linear SAE with an L1 penalty is trained on late-layer activations to learn a sparse basis; feature semantics are visualized via activation maximization and a proposed Grad-FAM saliency that targets latent features. The same SAE decoder provides directions for permanent weight-space edits that suppress or enhance specific features, and the paper defines $\alpha_\mathrm{crit}$ to quantify how strongly a class relies on a feature. Experiments target ResNet-18 and ViT-B/16 fine-tuned on Imagenette, with class-specific and cross-class interventions and small ablations against a ROME-style baseline.

### Strengths
* Originality: Clear, cohesive pipeline that connects unsupervised SAE features to weight-space edits, not only inference-time steering. The $\alpha_\mathrm{crit}$ metric offers a concrete knob to quantify reliance per sample, which is useful for diagnostics
* Quality: Solid derivation for the analytic approximation of $\alpha_\mathrm{crit}$ with a numerical check; careful discussion about when the linear approximation is reasonable for ResNet versus ViT
* Clarity: Method is easy to follow, with a simple linear SAE, explicit loss, and pseudocode for Grad-FAM and the numerical root finding
* Significance: If scaled and benchmarked broadly, the approach could offer a practical route to permanent, auditable edits tied to interpretable features, which is attractive for safety and compliance scenarios

### Weaknesses
* Only Imagenette is used and only two backbones are tested. There is no evaluation on harder datasets, no distribution shift stress tests, no adversarial or corruption benchmarks, and no human studies for interpretability quality. The ROME comparison is minimal and customized to the final layer rather than the standard internal-layer setting
* The discovery component is a standard linear SAE with L1 sparsity. Grad-FAM adapts Grad-CAM to a latent feature target, which is straightforward. The control step scales weights along decoder directions, which closely resembles linear readout manipulation or feature-direction ablations in the logit space. The contribution is mainly the packaging and the diagnostic $\alpha$ metric, not a fundamentally new mechanism
* Table 1 is helpful but misses several current concept-level and editing baselines in vision and language, such as more recent SAE-steering works and concept-bottleneck variants with post-hoc heads. The empirical section does not compare against activation-space steering on equal footing, INLP-style projections on the same backbone, or pruning-based continuous ablations that preserve accuracy.
* All edits appear to target the classifier head through penultimate activations; it is not shown how deeper-layer edits behave, how much performance drift occurs on unrelated classes in larger settings, or whether repeated edits compose cleanly. The paper acknowledges this as future work.

### Questions
* Can you expand Table 1 to include recent SAE-steering for ViTs and CLIP, editing methods beyond ROME and MEMIT, and causal feature interventions that operate in hidden layers, then compare empirically on shared metrics (e.g.: https://arxiv.org/abs/2504.08729, https://proceedings.mlr.press/v267/zaigrajew25a.html)?
* Your edit multiplies final-layer weights by a function of the decoder column for a feature. Please clarify whether SALVE always acts on the last layer and, if so, how this differs materially from directly editing a linear readout vector or from low-dimensional logit steering; ideally show deep-layer edits with the same decoder direction mapped backward.
* Please benchmark against activation-space steering with identical features and an equivalent compute budget, and report accuracy, off-target effects, and edit permanence. This will isolate the value of permanent weight edits versus temporary steering.
* Can you repeat the study on CIFAR-100 or ImageNet subsets, and include distribution shift tests and corruption robustness, reporting accuracy drops and off-target drift under increasing alpha? alpha should also be connected to adversarial susceptibility
* Your ROME adaptation targets the classifier head. Can you also evaluate a mid-layer ROME or MEMIT edit to match the original setting and show confusion matrices for both, plus your method, on the same task?
* Beyond visuals, can you measure concept-localization quality numerically, for example with Network Dissection-style overlaps, TCAV sensitivity for user-named concepts, or segmentation IoU where labels exist?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SALVE, a framework for discovering, validating, and controlling interpretable features in neural networks through SAEs. The authors train ℓ1-regularized autoencoders on internal activations of vision models to extract sparse feature representations. They validate these features using activation maximization and Grad-FAM visualization technique. The core contribution is a method for performing permanent weight-space edits guided by the learned feature directions, enabling continuous modulation of both class-specific and cross-class features. Experiments demonstrate that the approach enables precise, targeted control over model predictions.

### Strengths
- The paper presents a methodologically sound framework with experimental validation across different model architectures.
- The proposed method effectively enables permanent weight-space interventions by directly modifying model weights guided by discovered latent features.

### Weaknesses
- While the paper proposes a generic framework integrating sparse autoencoders with weight-space interventions, the individual technical components are largely adaptations of existing methods. 
- The comparison with activation steering methods is mentioned but not thoroughly explored empirically. 

Please see questions below for details

### Questions
- The core techniques—sparse autoencoders for feature extraction, weight-space interventions with minimal edits, and visualization methods—have been explored independently in prior work. The novelty lies primarily in their integration rather than in fundamental technical innovation. I personally think that the paper would benefit from more explicit discussion of how the combined framework differs from and improves upon existing approaches beyond simply integrating known techniques, particularly in comparison to prior work on model editing (e.g., MEMIT) and concept-based interventions that have explored similar ideas of using learned representations to guide weight modifications.
- The paper claims advantages such as "continuous modulation", and "nuanced cross-class edits," but lacks rigorous and comprehensive experimental evidence demonstrating concrete scenarios where permanent weight edits provide clear practical benefits over the more commonly used inference-time steering paradigm. A more systematic comparison showing specific use cases where permanent edits excel—such as computational costs during deployment, capabilities that steering cannot achieve, or finer-grained control compared to recent SAE-based steering approaches—would strengthen the claims about the method's distinct value proposition.

### Soundness
2

### Presentation
2

### Contribution
2
