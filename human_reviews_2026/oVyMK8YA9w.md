# THE JPEG BLIND SPOT: EXPOSING A CRITICAL VULNERABILITY IN DOCUMENT TAMPERING DETECTION

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Deep models for document tampering detection increasingly rely on multimodal RGB+DCT architectures, implicitly assuming that JPEG block artifact grids (BAG) provide stable cross forgery cues. In this paper, we show that this assumption embeds a strong inductive bias that fails under minimal, adversarially constructed perturbations. Unlike natural images, where JPEG alignment is largely stochastic, document images contain sharply bounded glyph structures, making grid-aligned manipulations trivial for an adversary.
We formalize this phenomenon through two complementary attacks. Grid-Aligned Forgery (GAF) preserves local JPEG block statistics by aligning copy move, splicing, or generative manipulations to the underlying 8×8 grid, removing the inconsistencies current models depend on. Pad–Recompress–Crop (PRC) globally shifts the JPEG grid while leaving RGB content unchanged, probing whether detectors meaningfully fuse RGB and DCT features or merely memorize position dependent frequency cues.
To quantify these effects, we use two evaluation metrics, Attack Success Rate (ASR) for missing forged regions and False Positive Area (FPA) for unintended detections, which capture failure modes not measured by prior work.
Evaluations on the DocTamper benchmark show that both attacks substantially degrade performance across a range of state-of-the-art and robustness-oriented (including adversarially robust) detectors, such as CAT-Net, DTD, FFDN, DocForgeNet, and ADCD-Net. Our findings indicate that many existing models exhibit a strong bias toward JPEG-grid statistics and highlight this as an opportunity for developing more robust multimodal architectures for real world, security critical document forensics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper exposes a major vulnerability in current document tampering detection models, such as DTD and DocForgeNet, which depend on frequency-domain cues like the JPEG compression–induced 8×8 block artifact grid (BAG). To exploit this weakness, the authors introduce two adversarial attack methods: (1) Grid-Aligned Copy-Move (GACM): This attack aligns tampered regions with existing 8×8 grids to preserve BAG consistency and evade detection. (2) Padding-Recompression-Cropping (PRC): This method intentionally offsets JPEG grids to produce artificial BAG inconsistencies, thereby triggering false positives. Extensive experiments on the DocTamper dataset reveal severe performance degradation under these attacks. For instance, the F1-score on the DocTamper-FCD test set drops dramatically from 76.2% to 3.1%. These results confirm that current detection systems largely rely on superficial frequency-domain artifacts rather than learning global semantic representations of tampering. The paper emphasizes the need for semantic-level detection models capable of robust and meaningful tampering analysis. It also highlights the security risks of the identified weakness in high-stakes applications and suggests that GACM and PRC can serve as benchmark attacks for evaluating the robustness of future detection models.

### Strengths
The experimental design demonstrates strong reliability and adaptability. Using DocTamper to assess performance in both standard and cross-domain contexts. By combining pixel-level Precision, Recall, and F1 metrics with qualitative visualizations, it ensures objective and intuitive evaluation. This framework offers a robust, reusable paradigm for future research.
The methodology is highly targeted and practical. The GACM and PRC attacks exploit the 8×8 JPEG block structure, exposing detection models’ overreliance on block-level artifacts instead of semantic features. Closely reflecting real-world tampering, these methods reveal key vulnerabilities and provide clear, reproducible insights for improving document forgery detection systems.

### Weaknesses
The core idea of this work is to exploit JPEG’s 8×8 block alignment to conceal tampering, a strategy that has already been investigated in prior natural image forensic research. The authors do not uncover a new vulnerability mechanism or make any fundamental theoretical contributions. As a result, the novelty lies more in practical instantiation or engineering implementation rather than in revealing the underlying essence of the problem or achieving a principled breakthrough.
The proposed attack is evaluated only on detectors that utilize a single DCT mode, the FPH module in DTD, within a two-stream architecture designed to exploit JPEG block artifacts. However, it does not assess detectors that employ alternative DCT configurations (such as those using dilated convolutions [1]) or multimodal architectures. Moreover, recent multi-branch approaches, such as Luo et al. [2], which integrate DCT, ELA, and noise features, are excluded from the evaluation. Consequently, the paper’s findings highlight vulnerabilities in a limited class of detectors under specific conditions, rather than demonstrating a general weakness across all DCT-based detection methods.
The authors do not discuss how training data or training strategies influence vulnerability to attacks. In our experiments with DTD, we observed that the DCT branch was susceptible to overfitting. However, applying data augmentations such as random image shifts effectively alleviated this problem while maintaining strong performance. If appropriate data augmentations or training strategies can mitigate attack impact, the practical novelty and risk of the proposed method would be reduced. The paper should therefore include experiments or at least a discussion addressing these factors.
Finally, the experimental design and visualization are inadequate. It only reports pre- and post-attack metrics without broader comparisons, ablations, or visual analyses to explain the attack mechanism or concealment of tampering traces. Moreover, it does not propose potential improvements to detection methods. For instance, whether DCT remains an effective feature for document-image forensics or how the DCT branch could be enhanced to resist such attacks. These are important aspects that should be discussed as directions for future research.

[1] Kwon M J, Yu I J, Nam S H, et al. CAT-Net: Compression artifact tracing network for detection and localization of image splicing[C]//Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021: 375-384.
[2] Luo D, Liu Y, Yang R, et al. Toward real text manipulation detection: New dataset and new solution[J]. Pattern Recognition, 2025, 157: 110828.

### Questions
Please clarify the novel contributions of this work beyond existing methods that leverage JPEG 8×8 alignment. Evaluate the proposed attack across a wider range of models, including those using DCT configurations with dilated convolutions, CAT-Net approaches, and multi-branch detectors such as Luo et al. Additionally, conduct experiments on training strategies and data augmentations (e.g., random pixel shifts) to determine whether these techniques can mitigate the attack. Include ablation studies and qualitative visualizations to illustrate the concealment mechanism, and propose or assess potential defenses or methodological improvements.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper exposes a critical vulnerability in state-of-the-art document tampering detection models, which the authors argue rely excessively on low-level JPEG compression artifacts—specifically Block Artifact Grids (BAG)—rather than learning semantically meaningful representations of forgery. To demonstrate this, the authors introduce two novel adversarial attacks: a Grid-Aligned Copy-Move (GACM) attack that preserves local JPEG block statistics during tampering, and a Pad-Recompress-Crop (PRC) attack that deliberately shifts the JPEG grid to induce false positives. When evaluated on the DocTamper benchmark, these attacks catastrophically reduce the performance of leading detectors—sometimes to near-random levels—revealing their fundamental reliance on superficial artifact cues and underscoring the need for more robust, semantically-aware forensic systems.

### Strengths
1. Originality:
While the existence of JPEG grid artifacts and their use in forensics is a well-established field, the core insight—that state-of-the-art (SotA) deep learning models have a critical, over-reliant, and exploitable "blind spot" centered on these artifacts—is novel. The paper models expose a fundamental flaw of the DCT-based forensic models. 
2. Quality:
The methodology is sound and well-grounded in the JPEG compression standard. The experimental setup is thorough: It tests against multiple, recent SotA models (DTD, DocForgeNet, FFDN). It employs a realistic attack implementation using OCR for text box selection. It provides both quantitative results (showing catastrophic drops in F1-score to as low as 1-3%) and qualitative visualizations that make the failure modes immediately understandable. The fact that one model (FFDN) shows greater robustness is noted and plausibly explained (its use of a Visual Enhancement Module), which actually strengthens the paper's argument by showing that reducing reliance on frequency-domain features improves resilience.
3. Clarity:
The paper is well-written and structured. It efficiently establishes the problem context, clearly states its central hypothesis, and provides an intuitive explanation of the JPEG artifact vulnerability before delving into technical details. The attacks (GACM and PRC) are described with precise mathematical formulations and accompanied by clear pseudocode and illustrative figures. The results are presented in a comprehensive table and supported by qualitative figures that effectively demonstrate both the "false negative" (GACM) and "false positive" (PRC) failure modes.

### Weaknesses
1. Scope and Generality of the Attacks:
The paper's central claim is that it exposes a "critical vulnerability in document tampering detection." However, the attacks are demonstrated primarily against a specific type of forgery: copy-move. Document tampering also includes splicing (inserting elements from a different image) and generative forgeries (using AI to alter text in-place).
2. Depth of Evaluation Metrics:
The evaluation relies exclusively on pixel-level Precision, Recall, and F1-score. While standard, these metrics do not fully capture the "catastrophic failure" in a security context. Suggestions: For GACM (False Negatives): Report the Attack Success Rate (ASR), i.e., the percentage of forged instances where the model's detection score in the tampered region falls below a operational threshold (e.g., where the F1-score drops to near zero). The current F1-score is an aggregate; an ASR would more starkly show how often the attack completely bypasses detection. For PRC (False Positives): Quantify the "unreliability" by reporting the False Positive Area (FPA) or the percentage of the original, authentic image that is now flagged as tampered after the PRC attack. A large FPA would powerfully demonstrate how the model becomes practically unusable. Segment-level Evaluation: Pixel-level F1 can be harsh. A segment-level metric (e.g., IoU of connected components) could show that the model not only performs poorly pixel-wise, but fails to localize the entire forged text segment, which is often the operational goal.
3. Analysis of Defenses and Robust Representations:
The paper effectively critiques existing models but offers limited guidance on how to build robust ones. The observation that FFDN is more robust is a starting point, but the analysis is superficial.

### Questions
1. On the Scope and Generality of the Vulnerability:
1.1 The demonstrated attacks focus exclusively on copy-move forgeries. Is the identified vulnerability fundamental to the models' architecture, and therefore also applicable to other common document forgery types, such as splicing (inserting text from a different image) or **generative in-painting**? Could you demonstrate a "Grid-Aligned Splice" attack or discuss why the same principles would/would not apply?
1.2 The attacks exploit the JPEG 8x8 grid. How does the vulnerability manifest with other common document image formats, such as PNG (lossless) or JPEGs with different block sizes? If a document is stored in a lossless format, do these models fail completely, or do they fall back to other (potentially more semantic) features?

2. On the Evaluation and Metrics:
2.1 The current pixel-wise F1-score, while standard, does not fully capture the security implications of a failed detection. We suggest reporting additional metrics: Attack Success Rate (ASR): For GACM, what percentage of forged instances are completely missed (e.g., IoU = 0 or F1 < 0.1)? This directly measures how often the attack bypasses the detector entirely. False Positive Area (FPA): For PRC, what percentage of the original, authentic image is falsely flagged as tampered? This quantifies the "systematic false positives" and the resulting unreliability.
2.2 The PRC attack induces false positives, but are these false positives random noise, or do they correlate with specific semantic structures in the document (e.g., text glyph edges, line intersections)? A qualitative analysis of where the false positives occur could provide deeper insight into the model's faulty reasoning process.

3. On the Architectural Analysis and Pathways to Robustness:
3.1 You hypothesize that FFDN's relative robustness is due to its Visual Enhancement Module (VFM) reducing reliance on frequency features. Did you perform any ablation studies to confirm this? For instance, if you remove the VFM from FFDN, does its performance on GACM drop to the level of DTD? Conversely, if you add a similar RGB-fusion module to DTD, does its robustness improve?
3.2 Have you explored simple defensive strategies or data augmentations based on your findings? For example, does training a model on a dataset that includes examples with random PRC-style grid shifts improve its robustness to both GACM and PRC attacks without harming its performance on standard forgeries?

### Soundness
3

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
This paper identifies a critical vulnerability in current state-of-the-art document tampering detection models, which primarily rely on JPEG compression artifacts, specifically Block Artifact Grids (BAGs), for localizing forged regions. The authors propose two adversarial attacks: Grid-Aligned Copy–Move (GACM) and Pad–Recompress–Crop (PRC). GACM preserves local JPEG block statistics, making forged regions consistent with the 8x8 grid, while PRC deliberately shifts the JPEG grid to induce widespread false positives. Evaluating these attacks on the DocTamper benchmark against leading detection methods (DTD, DocForgeNet, FFDN), the study demonstrates a catastrophic failure, reducing detection rates to near-random chance or triggering significant false positives. This work highlights that existing models often fail to learn genuine semantic tampering representations, instead relying on superficial compression artifacts, thus underscoring a need for more robust forensic systems.

### Strengths
The paper introduces PRC, specifically designed to exploit the underlying mechanism of JPEG compression artifacts. This directly challenges the current paradigm of frequency-domain feature reliance. PRC deliberately misaligns the JPEG grid without visible changes to the document, effectively turning benign images into "tampered" ones from the perspective of current detectors. This highlights a severe blind spot in existing systems.

The study benchmarks against three leading state-of-the-art models (DTD, DocForgeNet, FFDN), all of which explicitly leverage frequency-domain DCT features. This direct comparison against strong baselines validates the effectiveness of the proposed attacks.

### Weaknesses
On Grid-Aligned Copy-Move (GACM) and Novelty:
The observation that Grid-Aligned Copy-Move (GACM) can avoid Block Artifact Grids (BAGs) is a well-established concept in the field of JPEG image forensics. Prior literature has extensively documented this phenomenon, and it's recognized that even random copy-move operations have a 1/64 chance of aligning with grid boundaries. Consequently, the discussion of GACM in this paper does not represent a novel contribution.

On Grid Shift via Pad Recompress Crop (PRC) and Comparative Strength:
While Grid Shift via Pad Recompress Crop (PRC) is a recognized form of image distortion, other common image distortions such as iteratively resize and compress can also destroy the BAGs and leads to performance degradation. It is essential to explicitly articulate the unique strengths or advantages of PRC in degrading forensic performance when compared to these alternative, potentially equally effective, common distortion techniques.

Limited Scope of the Adversarial Attack:
The adversarial attack presented appears to have a significantly limited scope. Firstly, it is stated to apply primarily to frequency-based forensic models, with its impact on non-frequency-based forensic models being negligible. Secondly, its applicability is restricted solely to copy-move forgeries. There is no discussion or demonstration of its extensibility to other prevalent forgery methods, such as splicing or those involving AI-generated content.

Lack of Evaluation Against Robust Models:
A significant omission in the current work is the lack of evaluation against contemporary forensic models specifically designed for robustness against adversarial attacks, such as ADCD-Net [1]. Without experiments or analysis demonstrating the proposed method's efficacy against such specialized defenses, its overall impact and practical relevance in a robust forensic landscape remain unquantified.

[1] Wong, Kahim et al. ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement. ICCV2025.

### Questions
No other questions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper exposes the vulnerability inherent in BAG-based document tampering localization. In details, they propose two attack methods to fool existing methods; and evaluate the proposed attacks on benchmark datasets.

### Strengths
1. The authors have a thorough understanding of existing BAG-based vanilla document tamper detection methods, which is beneficial and necessary for attacking.

2. The authors clarify the mechanism of attacks clearly.

### Weaknesses
1. The stated conclusion for the failed defense against the proposed attacks are flawed. For hand-crafted tampering, such as copy-move, the tampered regions are not required to be generated. For the original reference image, the regions labeled as tampering are intrinsically authentic, but their designation stems solely from their anomalous location. Therefore, forensic models to solve hand-crafted forgeries can not extract genuine semantic representations of tampering patterns, which is different from DeepFake detection; what they can make use of is the inconsistencies between the labeled authentic and tampered regions in low-level patterns, such as BAG. If the statistics of such low-level patterns change, it is reasonable and natural that the performance declines. Additionally, if you change the format of targeted images for tampering localization, such as, “.png”, maybe all BAG-based methods can not work. There is no need to extensively design attack methods.

2. The GACM has limitations. The precise grid distribution must be known. For example, if the target image has been cropped and its origin of DCT grid is not the top-left corner, how to determine the attack area's position is a major challenge.

3. Experimental results do not always reflect the effectiveness of the attack methods. For example, the PRC attack is not significant for the forensic model, FFDN.

### Questions
none

### Soundness
2

### Presentation
2

### Contribution
2
