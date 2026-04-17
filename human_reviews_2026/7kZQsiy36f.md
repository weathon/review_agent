# Rethinking Unsupervised Cross-modal Flow Estimation: Learning from Decoupled Optimization and Consistency Constraint

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
This work presents DCFlow, a novel self-supervised cross-modal flow estimation framework that integrates a decoupled optimization strategy and a cross-modal consistency constraint. Unlike previous unsupervised approaches that implicitly learn flow estimation solely from appearance similarity, we introduce a decoupled optimization strategy with task-specific supervision to address modality discrepancy and geometric misalignment distinctly. This is achieved by collaboratively training a modality transfer network and a flow estimation network. To enable reliable motion supervision without ground-truth flow, we propose a geometry-aware data synthesis pipeline combined with an outlier-robust loss. Additionally, we introduce a cross-modal consistency constraint to jointly optimize both networks, significantly improving flow prediction accuracy. For evaluation, we construct a comprehensive cross-modal flow benchmark by repurposing public datasets. Experimental results demonstrate that DCFlow can be integrated with various flow estimation networks and achieves state-of-the-art performance among unsupervised approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DCFlow, a decoupled cross-modal flow estimation framework that can jointly train its two components, a modality transfer network and a single-modality flow estimation, without requiring direct ground-truth supervision. Specifically, a geometry-aware data synthesis pipeline is used to synthesize new views as well as their flow to self-supervise the flow estimation network. Affine transformations are also applied to help refine both networks jointly for better cross-modal consistency. Experiment results show promising results for this method.

### Strengths
1. The paper is well-structured. It is great to have an overview of the proposed method before diving deep into each component.
2. The proposed method is simple and easy to understand. Each method design is explained in the text.
3. The presented results look good, although it may be better if the authors can show more results for more insights (detailed below in weaknesses and questions).

### Weaknesses
1. Academic writing can be better polished. 
- Many references in the paper are cited using `\citet`, disrupting the flow of the text, which should be replaced with `\citep` for smoother readability. 
- Some text also lacks clearance. For example, Line 083-085 says decoupled optimization reduces EPE by 72%, and Line 089-090 says the cross-modal consistency constraint results in a 28% improvement in EPE. This is confusing. Does it mean these two strategies jointly reduce EPE by 100%, so you basically have 0 EPE?
2. Eq (12) is not formatted correctly. "Argmin" stands for the argument values that can minimize the following expression, and adding these argument values does not make sense. Maybe the authors should put just one $\arg\min_{\phi\theta}$ outside the addition of these three loss components.
3. Table 2(c) is a bit weak because it lacks a baseline to compare. It is hard to tell whether the four network results are consistently improved compared to their original results without the proposed strategies in DCFlow.

### Questions
1. Since the two modalities are symmetric. Have the authors tried to switch the two modalities and see how the results may change? For example, if the current framework trains a modality transfer network from modality A to B and a flow estimation network that estimates flow in modality B. It is also possible to switch modality A and B (modality transfer from B to A and a flow estimation in A). It would be interesting to see how this will change the results, or at least the authors should justify why the current modality choices of A and B should be the best one.
2. Since the motion estimation network and the modality transfer network are optimized alternately, I assume the optimization should start with two decently working networks instead of training both from scratch. Could you discuss more on how to start this training?

### Soundness
3

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
This paper proposes a cross-modal flow estimation framework (e.g., between RGB and thermal images) that does not require ground-truth optical flow across modalities. The key idea of the framework is to disentangle modality discrepancy and geometric misalignment. The architecture consists of a flow estimation network and a modality transfer network, which are trained both independently and jointly through a cross-modal consistency constraint that enhances cross-modal flow estimation. The proposed method is evaluated on a newly introduced dataset constructed from MS², VTD, and RNS, achieving superior performance compared to previous unsupervised cross-modal methods.

### Strengths
1. The proposed architecture effectively disentangles modality discrepancy and geometric misalignment, while the cross-modal consistency constraint strengthens the connection between modalities. This design is well-suited for scenarios where cross-modal ground-truth data is scarce and demonstrates effective results. 

2. A comprehensive ablation study is conducted for each component, showing that every part of the proposed framework contributes to performance improvement on the evaluation dataset, thereby supporting the design choices.

### Weaknesses
1. Ambiguity in the use of the term unsupervised
- Although the term “unsupervised” is used throughout the paper, in my opinion its meaning differs from how it is conventionally used in the optical flow field. In this paper, the method is described as unsupervised because it does not require cross-modal optical flow annotations between modalities. However, each individual modality is trained using synthetic data with ground-truth optical flow annotations. Just as a optical flow model (e.g., RAFT)  trained on the FlyingChairs synthetic dataset is typically categorized as a supervised method, it is questionable whether the proposed pipeline, which is trained on synthetic data with GT flow, should be classified as unsupervised. It would be better to use a more precise term that explicitly conveys the idea of training without cross-modal ground-truth flow, rather than referring to the method as completely unsupervised.

2.  Novelty of outlier-robust loss and geometry-aware data synthesis pipeline
- In Lines 81–82, the paper states “we further design an outlier-robust loss”, and in Line 253, it mentions “we propose a geometry-aware data synthesis pipeline with an outlier-robust loss.” However, both ideas appear to overlap with previous works. For the outlier-robust loss, several prior studies—such as WaterMono [a1], MapAnything [a2], and Regularizing Nighttime Weirdness [a3] already adopt a similar strategy by excluding the top percentile of high-error pixels during loss computation. Likewise, the idea of a geometry-aware data synthesis pipeline has been previously explored in [a4] and [a5]. It would be better for the authors to properly cite the relevant prior works to clarify the originality and positioning of these components.

[a1] Ding et al., “WaterMono: Teacher-Guided Anomaly Masking and Enhancement Boosting for Robust Underwater Self-Supervised Monocular Depth Estimation”, IEEE Transactions on Instrumentation and Measurement 2025.

[a2] Keetha et al., “MapAnything: Universal Feed-Forward Metric 3D Reconstruction”, arXiv 2025.

[a3] Wang et al, “Regularizing nighttime weirdness: Efficient self-supervised monocular depth estimation in the dark.”, ICCV 2021.

[a4] Aleotti et al., “Learning optical flow from still images”, CVPR 2021.

[a5] Han, Yunhui, et al., “RealFlow: EM-based Realistic Optical Flow Dataset Generation from Videos”, ECCV 2022.

3. The proposed experiments are conducted on MS², VTD, and RNS datasets, using 80% of frames per video for training and 20% for testing. However, this evaluation setup may be insufficient to assess the model’s generalization capability, since both training and testing samples originate from the same video distribution. It would be more convincing if the authors could report results under a cross-dataset evaluation setting, where the training and evaluation datasets differ, to better demonstrate generalization performance.

4. It would be beneficial if the paper provided a visualization of the training process for the proposed flow estimation network and modality transfer module.

### Questions
1. In Section 3.4, could the authors include more details on how the transformed flow is obtained? In my opinion, while this operation seems feasible, it likely involves non-trivial preprocessing or transformation steps. It would be helpful to include a brief explanation or illustration of this process in the supplementary material.

### Soundness
3

### Presentation
4

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
The paper proposes DCFlow, an unsupervised framework for cross-modal flow that trains a modality-transfer network and a flow network with a decoupled scheme plus a cross-modal consistency objective. The flow net learns from explicit mono-modal supervision generated by a geometry-aware single-image synthesis pipeline with an outlier-robust loss; the transfer net learns via a perceptual loss while flow is frozen; and a consistency loss enforces equivariance under known spatial transforms to tie the two together.

### Strengths
(1) A decoupled optimization recipe that stabilizes training by supervising the flow net on synthetic mono-modal pairs while separately optimizing the transfer net, addressing modality discrepancy and geometric misalignment distinctly. 

(2) A geometry-aware single-image synthesis pipeline + top-τ% trimming that yields robust, dense motion supervision without cross-modal labels. 

(3) A cross-modal consistency constraint that jointly optimizes the two nets under known spatial transforms. 

(4) A repurposed benchmark over MS2/VTD/RNS with RGB/NIR/Thermal pairs and LiDAR-derived sparse flow.

### Weaknesses
1. The paper jumps quickly from identifying limitations of existing appearance-based methods to presenting its full “decoupled + consistency” solution. The reasoning that connects the problem to this specific design is underdeveloped and feels somewhat ad hoc.

2. It seems that several key components, such as the fixed top-τ% robust loss, small affine ranges in the consistency constraint, and reliance on monocular depth synthesis, are justified only empirically with little or none sensitivity or ablation analysis.

3. While quantitative results are strong, the paper can be refined by introducing an analysis of efficiency, runtime, and generalization to unseen domains or sensor types.

### Questions
(1) How is monocular depth obtained for non-RGB modalities (Thermal, NIR), and how reliable is it for generating supervision?

(2) Why was the top-τ% value fixed at 20%, and how sensitive are results to this choice or to alternative robust loss functions?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work introduces a method for cross-modal optical flow estimation. It claims that different modalities of data have their own applicability. For example, during autonomous driving, different modalities of data may be required for day and night. In such cases, there may be issues such as misalignment of different modalities of data, making cross-modal optical flow estimation relatively difficult. Specifically, it proposes DCFlow, which significantly improves the model performance in an unsupervised manner based on the decoupled optimization strategy and synthetic supervision data.

### Strengths
- DCFlow clearly addresses a key limitation in present unsupervised cross-modal flow methods by introducing direct, geometry-aware supervision and robust optimization.
- By conducting separate training with single-modal optical flow supervision and modal transfer supervision, the problem of task coupling in traditional methods is resolved.
- The experimental section is thorough, systematically demonstrating the effect of each technical choice.
- This method circumvents the need for supervised data by using synthetic data, and it is effective and sound.

### Weaknesses
- The approach relies on pretrained monocular depth (Sec 3.3, Fig. 3) to generate synthetic flows. However, there is minimal discussion of the risk or variability this introduces. How does DCFlow handle inaccurate or biased depth prediction in unseen modalities, e.g., thermal-NIR? What happens if the depth model underperforms? This is especially noteworthy given the visual artifacts in certain synthetic example patches (Fig. 4/9). The ablation on the synthetic pipeline is strong, but further stress-testing (e.g., on failure cases with poor depth) is absent.

### Questions
- How robust is DCFlow to failures in monocular depth estimation for the geometry-aware synthetic pipeline?
- For low-performing modalities/gaps (e.g., MS^2 NIR-T), can you provide qualitative failure analysis or error breakdown by source (e.g., transfer vs. flow estimation vs. data synthesis)?

### Soundness
3

### Presentation
3

### Contribution
3
