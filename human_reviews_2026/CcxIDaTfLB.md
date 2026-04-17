# All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Recent advances in 3D Gaussian Splatting (3DGS) have revolutionized scene reconstruction, opening new possibilities for 3D steganography by hiding 3D secrets within 3D covers. The key challenge in steganography is ensuring imperceptibility while maintaining high-fidelity reconstruction. However, existing methods often suffer from detectability risks and utilize only suboptimal 3DGS attributes, limiting their full potential. We propose a novel end-to-end key-secured 3D steganography framework (KeySS) that jointly optimizes a 3DGS model and a key-secured decoder for secret reconstruction. Our approach reveals that Gaussian attributes contribute unequally to secret hiding. The framework incorporates a key-controllable mechanism enabling multi-secret hiding and unauthorized access prevention, while systematically exploring optimal attribute update to balance fidelity and security. To rigorously evaluate steganographic imperceptibility beyond conventional 2D metrics, we introduce 3D-Sinkhorn distance analysis, which quantifies distributional differences between original and steganographic Gaussian parameters in the representation space. Extensive experiments show that our method achieves state-of-the-art performance in 3D reconstruction while ensuring high levels of steganographic security. The framework is highly efficient and readily extensible to multi-GPU training. Our code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes KeySS, an end-to-end 3D steganography framework that hides multiple secret 3D scenes inside a cover scene using 3D Gaussian Splatting, while maintaining compatibility with standard rendering pipelines. It introduces a key-controlled decoder to ensure that only the correct key can reveal the hidden secrets, and develops a 3D-Sinkhorn metric to evaluate imperceptibility in the 3D Gaussian space. Experiments show that KeySS achieves high reconstruction fidelity, strong security against unauthorized access, and flexibility for multi-secret hiding.

### Strengths
The paper proposed a novel end-to-end framework for 3D steganography, especially for 3D Gaussian Splatting.
- Key-secured decoding architecture is utilized to transfer the original Gaussians to a set of secret Gaussians, which also supports multiple-secret hiding.
- A new metric, the 3D Sinkhorn distance, to measure the distributional disparities between the original and steganographic Gaussians.
- Extensive Experiments on Mip-NeRF360 and Deep Blending, and ablation studies have shown the effectiveness of the proposed method
- The paper is well written, and the method clarification is clear.

### Weaknesses
1. **Concept confusion**: The 'features', utilized in Line 015, Line 100, Line 107, etc, should be the 'attributes' of Gaussians. Using the term "feature" may cause confusion with the "feature" is used in GS-Hider.

2. **Motivation should be further clarified**: 
- Mentioned in Line 48, the rendering strategy in GS-Hider introduces deviations from the standard GS pipeline. It’s unclear what these deviations refer to, and whether any experiments or visualizations support them. The Keyss method also adds extra MLPs; does this not also deviate from the standard GS pipeline?
- The main motivation of the proposed method is that "Gaussian Attributes
contribute unequally to steganographic effectiveness", while all attributes are equally learned, a transformation from the original Gaussian via a set of MLPs. However, all attributes are transformed equally via a shared set of MLPs. Moreover, Tab. 4 shows that all attributes except 'SH' are used, which seems inconsistent with the stated motivation. Since the method essentially learns a transformation of the secret scene, it is natural that all attributes are required for rendering in a new scene. Therefore, further clarification is needed on how the method design aligns with the proposed motivation.


3. **Clarification of the used metrics**:
- The proposed 3D Sinkhorn distance is less motivated: The Sinkhorn distance differs from the Wasserstein distance (commonly used Optimal Transport) mainly by the addition of an entropy regularization term. A theoretical justification for choosing Sinkhorn over standard OT is needed, as the motivation behind this choice is not clearly explained.
- Main results in Tab. 1 show the effectiveness of the proposed method only by PSNR, while the component selection for the proposed are mainly depends on the proposed Score, shown in Tab. 4. This inconsistent use of the two metrics is confusing. Moreover, the ranking of different settings by Score in Table 4 does not align with their PSNR rankings, which calls for more visual evidence to justify that Score is a more effective metric than PSNR.


4. **Experimental Details**:
- The comparison with related work in the experiments is insufficient (only compared with GS-Hider), though clarified in Line 322 WaterGS and SecureGS are both not open-sourced.
- The input key is embedded with a Clip text encoder. When the input wrong key is semantically similar but not identical to the correct key (e.g., synonyms or slight word order changes), can the model still successfully reconstruct the cover scene? Additionally, the ratio of correct and wrong keys used during training should be clearly described.
- The scales of the scenes in Mip-NeRF360 and DeepBlending differ, resulting in different camera intrinsics and extrinsics. How these camera details are handled should be clearly explained.

5. **Visualization should be refined**: 
- Fig.1 and Fig.2 are hard to follow due to low-contrast colors and unclear fonts. The color scheme makes it difficult to distinguish key elements, and the font choice affects readability. Improving visual clarity would make the diagram easier to understand. 
- The Visualization in Fig.4, Fig.5 are blurred. The specific resolution settings for MIP-NeRF360 and DeepBlender used in the experiments should be explicitly reported for reproducibility and fair comparison.

### Questions
Refer to the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces KeySS (Key-Secured 3D Steganography), a novel end-to-end framework for hiding 3D secret scenes within a 3D Gaussian Splatting (3DGS) cover scene. The core contributions includes (1) a joint optimization framework that learns the 3DGS cover representation and a key-secured decoder simultaneously, (2) a novel security-enforcing loss function ($\mathcal{L}_{incorrect}$) 1 that trains the decoder to reconstruct the original cover scene when an incorrect key is provided, thus preventing unauthorized access, and (3) a new security evaluation metric, 3D-Sinkhorn distance , which measures the statistical imperceptibility of the steganography directly in the 3D Gaussian parameter space, offering a more robust analysis than traditional 2D render-based metrics.

### Strengths
- This work represents a major step forward for 3D steganography. It provides a solution that is not only effective but also practical: it is secure (multi-key, wrong-key defense), high-fidelity, and computationally efficient (maintains 130 FPS rendering). The ability to hide entire 3D scenes (not just watermarks) opens new application possibilities. Furthermore, the 3D-Sinkhon metric is a significant contribution to the community and will likely become a standard for evaluating future work in 3D representation steganography.

-  The KeySS framework is well-designed, end-to-end, and thoughtfully engineered (e.g., using $1 \times 1$ convolutions inspired by PointNet in the decoder). The experimental evaluation is comprehensive and convincing.

### Weaknesses
The framework is designed to hide secret 3D scenes within a cover 3D scene ($\mathcal{G}_{cover} \rightarrow \mathcal{G}_{secret}^s$). This is a powerful capability. However, traditional steganography often involves hiding other data types, like text documents, bitstreams, or audio. It is unclear how KeySS would be adapted for this. Would the text file first need to be represented as a 3DGS scene? This seems inefficient. A brief discussion on the framework's adaptability to more conventional steganographic payloads would be beneficial.

### Questions
Please refer to Weakness part.

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces KeySS, a novel key-secured 3D steganography framework built upon 3D Gaussian Splatting (3DGS) for hiding 3D secrets within 3D scenes. Unlike prior 3D steganographic methods that rely on suboptimal Gaussian features and risk detectability, KeySS jointly optimizes a 3DGS model and a key-protected decoder to balance reconstruction fidelity and steganographic security. The framework enables multi-secret hiding and prevents unauthorized access via a controllable key mechanism. It also proposes 3D-Sinkhorn distance analysis, a new metric to quantitatively evaluate imperceptibility by comparing Gaussian distributions between normal and steganographic scenes. Extensive experiments demonstrate that KeySS achieves state-of-the-art reconstruction quality and strong security guarantees, while remaining efficient and scalable for multi-GPU training.

### Strengths
1. End-to-End 3D Steganography Framework: The paper proposes KeySS, an end-to-end 3D steganographic learning framework that jointly optimizes 3D Gaussian representations and a key-secured decoder, ensuring both high-fidelity rendering and strong steganographic compatibility with the standard 3DGS pipeline.

2. Key-Secured Decoder for Multi-Secret Recovery: A key-controllable decoding mechanism is introduced to enable secure and accurate reconstruction of multiple hidden 3D secrets, effectively preventing unauthorized access.

3. Fidelity–Security Tradeoff Analysis and New Metric: The authors systematically explore how different 3D Gaussian feature combinations affect fidelity and imperceptibility, and propose 3D-Sinkhorn distance, a novel quantitative metric for evaluating steganographic security beyond traditional 2D measures.

### Weaknesses
1. Although KeySS enhances security, its rendering speed is slower than standard 3DGS, which may limit its applicability in real-time scenarios.

2. The authors should provide more illustrative examples and application scenarios to better explain the necessity of introducing key-control, thereby enhancing the motivation, significance, and practical relevance of the paper.

3. The improvement in rendering quality over GS-Hider is relatively limited, while the performance degradation compared to standard 3DGS is significant.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents KeySS, a 3D scene steganography framework built upon 3D Gaussian Splatting (3DGS). The method embeds complete secret scenes within a cover representation while maintaining high visual fidelity and compatibility with standard 3DGS rendering. A key-conditioned decoder, guided by CLIP-based semantic embeddings, modulates selected Gaussian attributes to securely encode and decode hidden content. To evaluate imperceptibility, the authors introduce a 3D-Sinkhorn distance that quantifies subtle distributional deviations in Gaussian parameter space.The paper’s main strength lies in its innovative integration of key-controlled steganography into 3D Gaussian Splatting, achieving secure multi-scene hiding with high fidelity and full rendering compatibility.

### Strengths
The paper’s main strength lies in its innovative integration of key-controlled steganography into 3D Gaussian Splatting, achieving secure multi-scene hiding with high fidelity and full rendering compatibility.

### Weaknesses
Please see questions.

### Questions
Technically, the paper adapts fundamental ideas from information security and steganography to the 3DGS framework. It incorporates a key-controlled access mechanism into a neural decoder, where CLIP text embeddings act as learnable semantic keys that determine whether the model reconstructs the hidden scene or reproduces the visible cover. The classical concept of secure hiding, which reveals the secret only with the correct key and restores the cover otherwise, is reinterpreted in the 3D Gaussian parameter space by modulating attributes such as opacity, rotation, and position instead of image pixels. In addition, the paper extends the notion of distributional detectability from traditional steganalysis to 3D by introducing the 3D-Sinkhorn distance, a metric designed to measure subtle distributional variations among Gaussian parameters. Together, these components represent a careful transfer of established security mechanisms into modern 3D scene representation learning.

The method jointly optimizes the cover and multiple secret scenes using a shared decoder, which raises potential concerns about overfitting among different supervision signals. It is not explicitly stated whether the reported quantitative results are obtained on training views or unseen novel views. Since the model operates on 3D Gaussian representations, it would be important to clarify this evaluation setting and, if possible, include visualizations of the reconstructed geometry (for example, depth maps or mesh) to verify consistency. 

In addition, The paper demonstrates a two-secret configuration under a single cover but does not discuss scalability beyond this case. It would be valuable for the authors to clarify potential challenges in extending the framework to N > 2 secrets, such as interference between keys, degradation of reconstruction fidelity, or training instability.

### Soundness
3

### Presentation
3

### Contribution
2
