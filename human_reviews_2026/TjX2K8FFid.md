# MIGA: Mutual Information-Guided Attack on Denoising Models for Semantic Manipulation

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Deep learning-based denoising models have been widely employed in vision tasks, functioning as filters to eliminate noise while retaining crucial semantic information. Additionally, they play a vital role in defending against adversarial perturbations that threaten downstream tasks. However, these models can be intrinsically susceptible to adversarial attacks due to their dependence on specific noise assumptions. Existing attacks on denoising models mainly aim at deteriorating visual clarity while neglecting semantic manipulation, rendering them either easily detectable or limited in effectiveness.
In this paper, we propose Mutual Information-Guided Attack (MIGA), the first method designed to directly attack deep denoising models by strategically disrupting their ability to preserve semantic content via adversarial perturbations. By minimizing the mutual information between the original and denoised images—a measure of semantic similarity—MIGA forces the denoiser to produce perceptually clean yet semantically altered outputs. While these images appear visually plausible, they encode systematically distorted semantics, revealing a fundamental vulnerability in denoising models. These distortions persist  in denoised outputs and can be quantitatively assessed through downstream task performance. We propose new evaluation metrics and systematically assess MIGA on four denoising models across five datasets, demonstrating its consistent effectiveness in disrupting semantic fidelity. 
 Our findings suggest that denoising models are not always robust and can introduce security risks in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MIGA (Mutual Information Guided Attention), a framework that enhances the robustness and interpretability of attention-based neural networks by explicitly guiding the attention mechanism with mutual information (MI). The key idea is to maximize the mutual information between the attention maps and latent representations so that the model focuses on regions carrying the most task-relevant information, rather than being distracted by noisy or uninformative areas. MIGA introduces a MI-guided regularization term and an information-contrastive objective that encourage both spatial and semantic consistency of attention across augmentations. The method is implemented on CNN and Transformer backbones using a variational MI estimator similar to InfoNCE, allowing efficient end-to-end training. Experiments show that MIGA consistently improves performance under noise, occlusion, and corruption, outperforming baselines such as Mixup, CutMix, and InfoDrop. Visualizations further confirm that MIGA produces more stable and meaningful attention maps. Overall, the paper proposes a conceptually clear and empirically effective approach linking mutual information optimization with robust representation learning.

### Strengths
1.MIGA introduces a clear and principled idea of using mutual information to guide attention, ensuring that models focus on informative and semantically relevant regions. This bridges interpretability and robustness in a way that previous heuristic attention regularizers have not achieved.
2.The method integrates seamlessly into both CNN and Transformer architectures, achieving consistent gains on multiple benchmarks (CIFAR, Tiny-ImageNet, ImageNet-C) under various noise and corruption settings, while also improving the visual stability of attention maps.
3.The proposed MI-guided regularizer is based on a tractable InfoNCE-style estimator, requiring minimal additional computation and allowing efficient end-to-end training, making it suitable for real-world robust learning setups.

### Weaknesses
1.The paper assumes that maximizing mutual information inherently leads to robustness, but lacks formal analysis or proofs connecting MI maximization to distributional stability or adversarial resistance.
2.Key MI-based baselines (e.g., Deep InfoMax, MITrans) are missing, and the contribution of each component (spatial vs. semantic MI) is not isolated, leaving uncertainty about what drives the improvement.
3.Details on the MI estimator are not fully disclosed, which may affect reproducibility and the interpretability of the learned attention behavior.

### Questions
1.The idea of modeling task-relevant mutual information is conceptually clear. However, the reliance on reference images for semantic guidance in unknown task scenarios raises practical concerns—how can an attacker realistically obtain semantically coherent reference images that effectively steer the attack? The authors should discuss or experimentally validate strategies for selecting such references to strengthen the method’s feasibility.
2.The experiments demonstrate broad applicability across tasks (classification, text tampering detection, image editing), but lack quantitative evaluation of semantic distortion. The authors should include human perception studies or use pre-trained semantic encoders (e.g., CLIP) to measure semantic shifts, providing stronger evidence that the attacks indeed alter meaningful content.

### Soundness
4

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
This paper proposes MIGA, a semantic adversarial attack targeting image denoising models. By minimizing task-relevant mutual information between the original and denoised images, MIGA alters semantic content while preserving visual quality. Extensive experiments across multiple denoising models and datasets demonstrate MIGA’s effectiveness in misleading downstream tasks without detectable artifacts.

### Strengths
1. The use of mutual information for semantic adversarial attacks is novel. The theoretical grounding in mutual information reduction provides a principled foundation for semantic disruption.
2. The framework supports both known and unknown downstream tasks, demonstrating its generality and effectiveness.
3. The paper is easy to follow.

### Weaknesses
1. For unknown tasks, MIGA relies on semantically altered references to approximate mutual information. The quality and diversity of these references could bias attack performance. It also lacks precise control over the direction and extent of changes. One may find it difficult to steer the model toward a specific semantic output, achieving only a vague deviation from the original semantics.
2. The paper proposes a combination of three loss functions, however, it lacks further analysis of the theoretical relationships between the three losses, as well as a rigorous mathematical proof to demonstrate the optimality of this combination.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an adversarial attack method called MIGA for deep learning-based denoising models. MIGA minimizes the mutual information (MI) between the original and denoised images to alter the semantic content of the denoised image, while maintaining image clarity through perturbation constraint loss and reconstruction loss. Under the known downstream task setting, the authors conducted experiments on an image classification task, attacking four denoising models and comparing the results with a baseline model. Under the unknown downstream task setting, the authors performed experiments on four datasets.

### Strengths
1. The article's structure is clear, and the notation is clearly defined.

2. The MIGA method is simple, straightforward, and easy to understand.

### Weaknesses
1. The authors assumed that denoising was necessary before performing downstream tasks. However, denoising is not necessary for downstream tasks. Without denoising, MIGA's performance is not remarkable.

2. In the experiments of Section 5.2, the authors only selected I-FGSM as the baseline method and ResNet as the classifier. It is insufficient to demonstrate the advantages of MIGA. Moreover, the authors mention multiple adversarial attack methods in Appendix D4; however, they did not compare with these methods in the main text. Furthermore, Appendix D4 lacks a comprehensive evaluation across all relevant metrics—for instance, it only discusses image clarity and ROUGE-L scores on text alteration tasks, but the performance in image classification accuracy is missing.

3. In the experiments of Section 5.2, among the known downstream tasks, the authors only evaluate performance on image classification and do not discuss other downstream tasks, such as object detection.

4. Figure 1 provides only a general overview and does not specify the exact models used.

5. As shown in Figure 2, MIGA requires a reference image. In the known downstream task setting, the reference can be $D(x_n$) ; however, in the unknown downstream task setting, it remains unclear where the reference image comes from. The experiments in Section 5.3 still construct reference images based on known downstream tasks.

6. Additionally, several places use "traditional adversarial" ambiguously. Generally, it refers to attack methods such as I-FGSM, rather than attacks specifically designed for denoising models.

### Questions
1. Please clarify for which downstream tasks denoising is indispensable.

2. Please supplement the performance results of the baseline models mentioned in Appendix D4 on the metrics related to Section 5.2 and MIGA's classification performance on other classifiers like ViT.

3. How does MIGA perform on other known downstream tasks besides image classification?

4. Is Figure 1 a real example? If it is real, please specify which ADM was used, what attack method was applied for Figure 1(b), and the source of the reference image in Figure 1(d).

5. Please explain the source of the reference image when the downstream task is unknown.

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
The paper proposes MIGA, a novel adversarial attack framework targeting deep learning-based denoising models. Unlike traditional attacks that degrade image clarity, MIGA aims to manipulate semantic content in the denoised output while preserving visual quality. The core idea is to minimize the task-relevant mutual information between the original and denoised images, thereby inducing semantic shifts that mislead downstream tasks. The method is designed to work in both known and unknown downstream task settings, using cross-entropy loss and a contrastive MINE-based estimator respectively. Extensive experiments on four denoising models and five datasets demonstrate that MIGA can effectively alter semantics without introducing perceptible artifacts.

### Strengths
(1) The paper's primary strength lies in its novel problem formulation. Targeting a pre-processing module like a denoiser, rather than the final task model, opens a new and significant research direction in adversarial robustness. The idea of a "semantic attack" on a denoiser—manipulating its output to be semantically incorrect yet visually plausible—moves beyond simple quality degradation attacks. 

(2) The experimental design is comprehensive: it covers four denoising models, five data sets, and a variety of downstream tasks (classification, text tampering, style migration, etc.). This paper compares several baseline attack methods and shows the advantages of MIGA in semantic attack.

### Weaknesses
(1) The unknown task scenario relies on a "reference image" to guide semantic alignment. This assumption may be too strong and could limit the practical applicability of the attack. For example, to make a "stop sign" be recognized as a "go-straight sign", the attacker would need a clean, well-aligned image of a "go-straight sign" in a similar context. The paper fails to elaborate on: 

(i) How is this reference image obtained in a real-world attack scenario? 

(ii) How sensitive is the attack's performance to the quality, alignment, and semantic similarity of the reference image? An ablation study on this aspect would be necessary to understand the boundaries of the attack's effectiveness. Moreover, the “unknown task” scenario relies on a pre-trained MINE model, which itself requires a curated dataset of semantic alterations. This may limit its applicability in truly black-box settings.

(2) While I-FGSM are included, more recent semantic-aware attacks (e.g., semantic adversarial examples) are not compared.

(3) The semantic alterations tested (text, style, object replacement) are on simple conditions. It is unclear whether MIGA can handle more complex semantic shifts, such as scene understanding or multi-object interactions.

### Questions
(1) What potential defense strategies, such as training denoisers with MI regularization or ensemble defenses, could resist MIGA? 

(2) How does the performance of MIGA under unknown tasks depend on the quality and diversity of the input pairs? Have the authors evaluated its sensitivity to the type of semantic alteration?

(3) The method requires iterative optimization for each image. While the paper compares the attack effectiveness, it lacks a comparison of computational efficiency against baseline methods. Could the authors provide a comparative analysis of the computational cost between MIGA and the baseline attacks? This is crucial for understanding the practical trade-offs involved in choosing MIGA over other methods.

(4) The current understanding of "semantics" in vision is increasingly shaped by self-supervised learning (SSL) and generative models, which learn rich, non-task-specific representations. How would MIGA perform if the "semantic fidelity" is defined by the feature space of a large vision model (e.g., DINO, CLIP)? Moreover, could the authors evaluate whether MIGA causes a significant deviation in the CLIP or DINO feature space between $x$ and $D(x_{MIGA})$?

### Soundness
3

### Presentation
2

### Contribution
2
