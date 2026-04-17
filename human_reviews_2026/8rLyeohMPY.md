# Real-Captured Paired Dataset for Nighttime Flare Removal

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Flare removal methods eliminate reflective and scattering flares within images and commonly adopt synthetic data for training. However, they fail to achieve robustness for real-world flare-corrupted images as the synthetic data remains gaps with real-world data. In this paper, we propose a real-captured paired dataset named FlareReal600, which contains both real-captured image pairs and pure flare images. Compared with the existing flare removal dataset Flare7k++, our dataset is particularly effective for real-world scenarios as our data contains the faithful mapping between real flare-corrupted images and real flare-free images. Additionally, previous methods either lack sufficient receptive fields or achieve them with huge computational costs, which leads to flares being partly removed or hardly processing high-resolution images. Therefore, we propose a novel flare removal network named \textbf{M}utual re\textbf{C}eption f\textbf{LA}re \textbf{RE}moval \textbf{N}etwork (McLaren), which utilizes convolutions with diverse kernel sizes and fuses them from the perspective of both spatial and channel dimensions to achieve a sufficient receptive field. Furthermore, we employ a re-parameterization mechanism to avoid occupying excessive computational resources. We conduct extensive experiments to demonstrate the functions of our FlareReal600 dataset and our McLaren network.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new dataset, FlareReal, which is designed to enhance the robustness of flare removal models for real-world scenarios. The authors introduce a McLaren network to process these images efficiently, combining large and small convolution kernels to achieve sufficient receptive fields without high computational costs. Extensive experiments demonstrate that the FlareReal dataset improves existing methods' performance and that McLaren outperforms other state-of-the-art models with fewer parameters.

### Strengths
The gap between synthetic data and real-world images is a critical and persistent challenge in low-level vision. This paper provides a direct and practical solution for the task of flare removal, which is an important problem in real-world scenarios.

### Weaknesses
(1)The FlareReal validation set is small, consisting of only 50 image pairs. A larger validation set would provide greater statistical confidence.
(2)The baseline methods tested are limited to UNet, HINet, and Uformer. The authors can explore additional baseline methods (such as MPRNet and Restormer) and select one or two recent state-of-the-art methods for dataset validation.
(3)To advance progress in this field, it is recommended to include a dataset download link for community access once the paper is accepted.

### Questions
(1)Will this dataset be open-sourced?
(2)Would McLaren’s computational efficiency hold up in more resource-constrained environments, like edge devices or mobile applications? This is an issue that needs to be considered in practical applications.
(3)Are there any notable failure cases or types of flares where the method still struggles? For example, extremely severe flares that occlude most of the image, or complex, colorful artifacts?
(4)The design of MRConv is similar to the paper “Reparameterized Multi-Resolution Convolutions for Long Sequence Modelling”. Please explain the differences between the two designs.

Paper: Cunningham, Jake, et al. "Reparameterized Multi-Resolution Convolutions for Long Sequence Modelling." Advances in Neural Information Processing Systems 37 (2024): 27121-27152.

### Soundness
4

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
This paper tackles the task of flare removal in images, noting that most existing methods rely on synthetic datasets like Flare7k, which limits real-world generalization. To address this, the authors introduce FlareReal, a new dataset containing 3,000 real-captured paired images and 500 standalone flare samples. They also propose McLaren, a UNet-based network incorporating a new Mutual Reception Convolution (MRConv) module that combines multi-kernel depthwise convolutions with re-parameterization for efficient receptive field expansion. Experiments on Flare7k++ and FlareReal show modest improvements over existing baselines such as UNet, HINet, and Uformer.

### Strengths
1. Introduction of a new real-captured dataset:
The paper introduces FlareReal, a real paired dataset for flare removal. This is a meaningful addition within this niche, as most prior datasets such as Flare7k++ rely heavily on synthetic images. Creating a real-captured dataset  under various lighting conditions helps bridge the realism gap and provides a valuable resource for future research in flare removal and related low-level vision tasks.
2. Network design – clean and efficient integration:
The proposed Mutual Reception Convolution (MRConv) introduces a multi-kernel convolutional block that combines diverse kernel sizes and re-parameterization for efficiency. Although the idea draws on established components (multi-scale convolutions, depthwise separability, re-parameterization), the integration is clean and computationally practical for high-resolution flare removal. The network shows performance gains over baseline models.

### Weaknesses
1. Limited contribution of the dataset:
Although the dataset introduces real-captured flare/clean image pairs, its scale (around 3,000 images) may be too small to substantially improve generalization beyond Flare7k++, which already includes 962 real-captured flare images. The diversity and acquisition conditions are not analyzed in depth, raising questions about whether FlareReal is large and varied enough to justify claims of “real-world generalization.” Moreover, dataset collection alone is not typically considered a significant contribution for ICLR unless it yields new insights or methodological advances.
2. Limited novelty in network design:
The proposed Mutual Reception Convolution (MRConv) appears to be a reassembly of existing architectural concepts rather than a novel convolutional formulation. The design combines well-known ideas such as:
Multi-kernel receptive field fusion (Inception-style)
Depthwise separability (MobileNet-style)
Channel reweighting (Squeeze-and-Excitation)
Residual connections (ResNet)
Structural re-parameterization (RepVGG)
While these components are effectively integrated, the paper does not present a new mathematical operation, learning objective, or architecture paradigm elements typically expected in ICLR-level contributions.
3. Lack of theoretical insight and limited analysis:
The paper does not provide theoretical justification or in-depth analysis to support the claim of achieving a “sufficient receptive field without heavy computational burden.” While empirical improvements are reported, there is no explanation of why the proposed design leads to better performance. Analyses such as effective receptive field visualization or feature activation analysis would have offered valuable insight into the model’s behavior and substantiated the design choices. The absence of such analysis limits the work’s interpretability and research depth.
4. Insufficient experimental exploration:
The experiments on McLaren are relatively limited and do not fully validate the claimed advantages. More comprehensive ablation studies, sensitivity analyses, and evaluations under varying noise or lighting conditions are needed to establish robustness. Furthermore, the paper does not report any computational or speed comparisons (e.g., FLOPs, runtime, memory usage) to substantiate its claims of efficiency and scalability. Without such analysis, it is difficult to assess whether McLaren truly achieves a better trade-off between performance and computational cost.
5. Narrow scope and limited impact:
The paper focuses narrowly on flare removal, a specialized low-level vision task. It does not connect the problem to broader computer vision or machine learning themes such as representation learning, domain adaptation, or self-supervision. As a result, the contribution feels more incremental and applied rather than conceptually advancing understanding in computer vision.
6. Poor presentation and citation quality:
The writing suffers from inconsistent referencing of figures and tables (e.g., “fig:Method(b)”, “tab:com”, “fig:mclaren”), which appear as placeholder labels rather than proper references. These issues, along with some grammatical errors, reduce clarity and professionalism.

### Questions
1. The paper’s main contribution is the introduction of the FlareReal dataset, yet there is no clear mention of whether it will be publicly released. Could the authors clarify if there are concrete plans to make the dataset, along with its collection protocol and metadata, publicly available? If not, the contribution would have very limited practical and scientific value, as reproducibility and community impact would be severely constrained.
2. The proposed Mutual Reception Convolution (MRConv) seems closely aligned with existing designs such as Inception, MobileNet, and RepVGG, which already combine multi-kernel processing and re-parameterization. Could the authors clearly articulate the substantive differences between MRConv and these architectures beyond naming or minor implementation variations?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on addressing limitations in existing nighttime flare removal research. It proposes two core contributions: 1) FlareReal, a real-captured paired dataset consisting of 3,000 training image pairs, 500 pure flare images, and 50 validation image pairs. 2) McLaren, a flare removal network built on a U-Net backbone with a custom Mutual Reception Convolution (MRConv). Experiments show FlareReal enhances model performance on real-world data, and McLaren outperforms baselines (Unet, HINet, Uformer) with fewer parameters (13.2M) and FLOPs (33.1G) .

### Strengths
The main strength of this paper comes from the construction of flare dataset. While synthetic datasets (e.g., Flare7k++) dominate existing flare removal research, FlareReal fills a critical gap by providing real-captured paired data. Its focus on real scenarios makes it a valuable resource for the community—experiments confirm models trained on FlareReal (or its mix with Flare7k++) outperform those trained solely on synthetic data on real-captured test sets.

### Weaknesses
1.	Limited Methodological Novelty: McLaren’s core components are assembled from existing techniques rather than introducing fundamental innovations. MRConv’s spatial mixing relies on multi-kernel convolutions and re-parameterization, while its channel mixing uses Squeeze-and-Excitation. The network’s overall design is based on U-Net backbone plus custom convolution; it’s just incremental, not transformative.

2.	Incomplete Dataset Documentation: While the paper outlines high-level data collection steps, critical details for reproducibility are missing: 1) No specification of scene diversity (e.g., percentage of urban vs. suburban scenes, indoor vs. outdoor samples) or light source statistics (e.g., distribution of point vs. extended sources). 2) No standardization of lens contamination (e.g., how much dust/oil to apply, consistency across samples)—variability here could introduce noise into the paired data. 3) No details on the OpenCV registration parameters (e.g., feature detector type, matching thresholds) etc.

3.	Incomprehensive Comparative Experiments: While there exist multiple flare removal methods in the field, this paper only compares the proposed McLaren network against three baseline models (Unet, HINet [2021], Uformer [2022]). Such a limited set of comparisons makes the claimed performance of McLaren less convincing.

### Questions
1.	Dataset Construction Details: Could you provide: 1) a detailed scene breakdown (e.g., number of samples from urban streets, residential areas, etc.) and light source categorization (point, linear, area sources)? 2) a standardized protocol for lens contamination to ensure consistency? 3) Specific OpenCV registration parameters and criteria for discarding unaligned images? 

2.	Methodological Innovation: The design of the flare removal method lacks sufficient novelty, and more innovative elements should be incorporated to distinguish it from existing techniques in the field.

3.	Completeness of Comparative Experiments: Given that there are multiple existing flare removal methods in the field, why does the paper only compare McLaren against three baselines (Unet, HINet[2021], Uformer[2022]) instead of including more SOTA alternatives?

### Soundness
3

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
4

### Summary
The paper presents FlareReal, a real-captured paired dataset for nighttime flare removal, alongside a new model called McLaren that enhances receptive field coverage without heavy computation. The dataset addresses the synthetic–real gap in existing benchmarks, while the proposed Mutual Reception Convolution (MRConv) leverages multiple kernel sizes and re-parameterization for efficient flare suppression.

### Strengths
It fills an important gap in real-world flare data collection, providing 3,000 image pairs with realistic lighting and exposure variations. The MRConv design is elegant and practical, improving both accuracy and efficiency compared to prior methods. Experimental results are solid and include detailed ablations on kernel size and dataset robustness.

### Weaknesses
1. The paper doesn’t clarify how well the FlareReal dataset generalizes beyond smartphone sensors to professional cameras or different optics.
2. Figure 5 comparisons are visually convincing but lack quantitative measures for artifact suppression or perceptual fidelity.
3. The ablation study focuses mainly on MRConv kernel sizes—other design choices like SE blocks or ConvMLP are not analyzed.
4. The dataset construction section omits detailed metadata such as exposure times or ISO settings, which limits reproducibility.
5. The discussion on real-scene corner cases (e.g., rain, multiple reflective surfaces) is brief and could be better connected to real deployment scenarios.

### Questions
See weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
3
