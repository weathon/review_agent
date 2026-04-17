# Exploring Specular Reflection Inconsistency for Generalizable Face Forgery Detection

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Detecting deepfakes has become increasingly challenging as forgery faces synthesized by AI-generated methods, particularly diffusion models, achieve unprecedented quality and resolution. Existing forgery detection approaches relying on spatial and frequency features demonstrate limited efficacy against high-quality, entirely synthesized forgeries. In this paper, we propose a novel detection method grounded in the observation that facial attributes governed by complex physical laws and multiple parameters are inherently difficult to replicate. Specifically, we focus on illumination, particularly the specular reflection component in the Phong illumination model, which poses the greatest replication challenge due to its parametric complexity and nonlinear formulation. We introduce a fast and accurate face texture estimation method based on Retinex theory to enable precise specular reflection separation. Furthermore, drawing from the mathematical formulation of specular reflection, we posit that forgery evidence manifests not only in the specular reflection itself but also in its relationship with corresponding face texture and direct light. To address this issue, we design the Specular-Reflection-Inconsistency-Network (SRI-Net), incorporating a two-stage cross-attention mechanism to capture these correlations and integrate specular reflection related features with image features for robust forgery detection. Experimental results demonstrate that our method achieves superior performance on both traditional deepfake datasets and generative deepfake datasets, particularly those containing diffusion-generated forgery faces.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose the hypothesis that specular reflection, a phenomenon governed by complex physical laws, is difficult for AI models to simulate, and therefore can serve as a key cue for generalizable deepfake detection. Specifically, they first employ Retinex for fast and accurate extraction of facial textures, then achieve improved specular reflection separation under the Phong illumination model. Building on these steps, they introduce the Specular-Reflection-Inconsistency Network (SRI-Net), which leverages a two-stage cross-attention mechanism to integrate relationships among specular reflection, facial texture, and direct lighting. Extensive experiments on both traditional and newly released generative deepfake datasets demonstrate that the proposed method outperforms existing approaches.

### Strengths
1. The paper focuses on the physical properties of light–surface interaction, the proposed approach demonstrates improved robustness and universality.

2. The authors replace the traditional BFM PCA-based texture estimation with multi-scale texture extraction based on Retinex theory, and design SRI-Net to effectively model the relationships among direct illumination, texture, and specular reflection.

3. The method is thoroughly evaluated on multiple classic and state-of-the-art datasets, achieving significant performance gains on both frame-level and video-level metrics.

### Weaknesses
1. The central claim, i.e., “Specular reflection is hard to replicate”, is intuitive but not particularly novel. The paper does not provide a new theoretical explanation or quantitative physical analysis to substantiate why specular reflection is inherently more discriminative than other illumination features.

2. Although the authors aim to capture the inconsistencies among specular reflection, texture, and direct illumination, they do not provide any mathematical formulation or quantitative metric defining how such inconsistencies are measured.

3. Although the authors claim that specular reflection is difficult to counterfeit, their method depends on 3D shape estimation (via 3DDFA) and Retinex-based texture extraction, both of which may fail under unconstrained facial poses, occlusions, or complex lighting conditions. This limitation is only briefly mentioned in the appendix and lacks in-depth discussion or empirical validation under “in-the-wild” scenarios.

4. The rationale behind the two-stage structure of SRI-Net and the choice of attention mechanism as the optimal modeling strategy is insufficiently explained and would benefit from clearer justification.

### Questions
Please see weaknesses.

Overall, the reviewer thinks that this paper presents an interesting and promising perspective on leveraging physical properties of specular reflection for deepfake detection. However, the work would benefit from more comprehensive theoretical analysis to substantiate its central hypothesis and further strengthen the proposed framework.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel approach for face forgery detection by leveraging the inherent difficulty of replicating specular reflection under the Phong illumination model. The authors introduce a faster and more accurate Retinex-based texture extraction method to enhance the separation of illumination components, particularly specular reflection. They design the Specular-Reflection-Inconsistency-Network (SRI-Net), which uses a two-stage cross-attention mechanism to capture correlations between specular reflection, face texture, and direct light. Experimental results demonstrate that the method achieves state-of-the-art performance on both traditional deepfake datasets and generative datasets, showcasing its robustness and generalizability.

### Strengths
- Novelty

The paper introduces a unique perspective by focusing on specular reflection as a forgery indicator, which is grounded in well-established physical principles. This approach provides a fresh direction in deepfake detection research.
The use of the Phong illumination model and Retinex theory for texture extraction is innovative and enhances the accuracy of specular reflection separation.


- Strong Experimental Results

The method achieves superior frame-level and video-level AUC scores across traditional and generative datasets, outperforming existing state-of-the-art methods.
The robustness of the approach on high-quality generative datasets highlights its generalizability and resilience against subtle forgery evidence.

- Efficiency

The Retinex-based texture extraction method significantly reduces processing time compared to traditional iterative optimization methods, making it suitable for large-scale data processing.

### Weaknesses
- Limited Analysis of Failure Cases

While the paper briefly discusses misclassification due to extreme facial poses or occlusion, it does not provide detailed insights into how these issues could be mitigated or addressed in future work.

- Dependence on 3D Shape Fitting

The method relies heavily on accurate 3D shape fitting to extract specular reflection. In scenarios with severe occlusion or extreme poses, the performance may degrade significantly. The paper does not explore alternative solutions or enhancements for these situations.

- Complexity of Implementation

Although the method is efficient in terms of processing time, the overall pipeline, including Retinex-based texture extraction, spherical harmonic modeling, and SRI-Net architecture, might be challenging to implement and deploy in real-world applications.

- Limited Real-World Validation

The datasets used for evaluation, while diverse, may not fully represent the complexities of real-world deepfake scenarios. Additional experiments on real-world datasets or adversarially generated deepfakes could strengthen the paper's claims.

### Questions
see above

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
4

### Summary
This manuscript presents a novel methodology for generalizable face forgery detection, underpinned by the observation that facial attributes governed by complex physical laws, particularly the specular reflection component, are inherently challenging for contemporary synthesis models to accurately replicate. The proposed approach systematically decomposes the facial image into fundamental physical components, including 3D shape, texture, and illumination, with the primary objective of isolating the specular reflection component for subsequent inconsistency analysis and detection. The core decomposition strategy employed is based on Retinex theory.

### Strengths
The manuscript is clearly written and easy to follow, with a concise method formulation and a framework that is simple yet demonstrably effective. The method achieves consistently strong results across a broad spectrum of face-swapping techniques, including both diffusion-based and GAN-based pipelines, suggesting good generalization beyond a single generator family. The ablation studies provide compelling evidence that each branch, as well as the method’s design/strategy choices, contributes positively to the overall performance.

### Weaknesses
1.In Section 3.2, the choice to employ Retinex theory and Multi-Scale Retinex (MSR) to achieve a 'smaller to the real texture' and enhance robustness is a major component of the methodology. However, these Retinex-based enhancement techniques are well-established, and their application alone does not appear to be a novel contribution. More critically, previous work [1], has utilized similar Retinex-based methods for image enhancement, yet this paper is not cited.

2.The work is motivated by the claim that the decomposed space is more complex and thereby facilitates distinguishing real from fake; however, no statistical or theoretical evidence is provided, and the visual comparisons in Figs. 3 and 4 fail to substantiate this.

[1] Chen, Han, Yuzhen Lin, and Bin Li. "Exposing face forgery clues via retinex-based image enhancement." Proceedings of the Asian Conference on Computer Vision. 2022.

### Questions
1.Could you elaborate on what is meant by '3D global environmental conditions' (line 176, Section 3.1). It is vague.

2.In Section 3.1 (line 185), there is a potential source of confusion regarding the notation in Formula (1) and Formula (2). Specifically, Formula (1) treats $C$ and $S$ as distinct variables, yet in Formula (2), $S$ is defined as being composed of elements $C_i$. The variable $C_i$ could easily be misinterpreted as a component of $C$, leading to ambiguity and obscuring the intended relationship between $C$, $S$, and $C_i$.

3.For the title 'Illumination and texture separation' in Section 3.2 (line 205), is the intention to decouple the texture and illumination components that were implicitly combined within the variable $C$ as introduced in Formula(1)?

4.The text spanning lines 236 to 247 in Section 3.2 should be broken down into smaller, logical paragraphs. The current dense formatting is difficult to read.
Regarding Formula (6), is there a specific model or methodology suggested for obtaining the identity transformation matrix $T_{id}$? For example, could a pre-trained model such as ArcFace be employed for this purpose, or is an alternative approach required?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a physics-based face forgery detection method that exploits the difficulty of reproducing specular reflection in generative models. The authors propose a Retinex-based texture estimation for clean illumination decomposition, a residual spherical-harmonics model for efficient specular separation, and an SRI-Net that learns physical correlations among reflection, light, and texture.

### Strengths
- This manuscript is well-structured and easy to follow.
- This work provides a clear physical motivation by identifying specular reflection as the most complex component in generated faces, which is a reasonable choice grounded in physical principles.

### Weaknesses
- Retinex-based reflectance estimation has already been used in face forensics and related vision tasks[1,2] for exposing subtle forgery cues, so the usage of Multi-Scale Retinex is not entirely novel.
- This work does not sufficiently validate robustness to common post-processing, such as Gaussian Blur and JPEG compression.

[1] Attention-based Two-stream Convolutional Networks for Face Spoofing Detection. IEEE TIFS 2020.

[2] Exposing Face Forgery Clues via Retinex-based Image Enhancement. ACCV 2022.

### Questions
Specular reflection depends on surface properties. How stable is this method on faces with different skin tones (e.g., using AI-Face [3])?


[3] AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark. CVPR 2025.

### Soundness
3

### Presentation
3

### Contribution
2
