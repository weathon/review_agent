# Gaussian Entropy Flow World Model for Streaming 3D Occupancy Predition

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 2

## Abstract
In 3D occupancy prediction, temporal information is crucial. Traditional methods fuse multi-frame features through a pipeline of perception, alignment, and fusion, but they overlook the coherence of static elements and the motion patterns of dynamic elements in 3D scenes. Existing methods reformulate 3D prediction as 4D prediction based on current sensor inputs by modeling the continuous evolution of the scene. However, the discrete refinements of the physical properties of dynamic elements in multiple encoding-decoding processes lead to cumulative errors and poor adaptation to dynamic motion. Inspired by non-equilibrium thermodynamics, we propose an Evolutionary Entropy Flow framework that uses Evolutionary Entropy as a carrier for continuous scene evolution, modeling the motion of dynamic elements as the flow of Evolutionary Entropy. We further introduce the Gaussian Entropy Flow World Model (GaussEFW), which represents Evolutionary Entropy Flow as a single, continuous Gaussian Entropy Flow in latent space, in contrast to the discrete refinements from multiple encoding-decoding processes. By predicting Gaussian Entropy Flow based on current RGB observations, we can accurately predict the motion of dynamic elements and learn continuous scene evolution. Extensive experiments on the nuScenes dataset validate the effectiveness of GaussEFW, demonstrating superior performance in dynamic elements prediction and high overall performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the Gaussian Entropy Flow World Model (GaussEFW) for streaming 3D occupancy prediction in dynamic scenes. Building on principles from non-equilibrium thermodynamics, the authors propose modeling the evolution of dynamic scene elements as a continuous flow of Gaussian entropy within a latent space, instead of relying on discrete refinement across multiple encoding-decoding steps. The framework introduces three processes—Entropy Producing, Entropy Exchanging, and Entropy Flowing—to track and refine dynamic elements in a unified way. Experimental results on nuScenes with SurroundOcc labels demonstrate improved dynamic occupancy prediction and overall mIoU when compared against state-of-the-art methods, including GaussianWorld.

### Strengths
-  The paper’s central idea of representing dynamic scene evolution as a continuous Gaussian entropy flow in latent space (as opposed to repeated discrete refinements) is conceptually interesting and aims to address error accumulation—an important shortcoming of current world models. This is visually well-motivated in Figure 1, where the trajectories and error types associated with existing models (discrete, cumulative refinement) are contrasted against the proposed continuous entropy flow.
- The illustration of the GaussEFW architecture in Figure 2 (Page 4) and the associated algorithmic breakdown provide readers with a clear, stepwise perspective on the denoising network, entropy production, and block interactions. This is supported by detailed equations specifying the entropy production, flow, and denoising processes (Equations on Pages 5-6).
- Multiple ablation studies are presented (Tables 2, 5, 6, 7, 8; Pages 8-15), dissecting the impact of the entropy-producing/exchange mechanisms, block configurations, category-wise noise, streaming length, and timestamp corruption. This demonstrates a commitment to experimental rigor.

### Weaknesses
- The performance of occupancy forecasting is relatively low compared to previous state-of-the-art (SOTA) occupancy prediction works, such as UniScene (mIoU 31.76). The authors are encouraged to include comparisons with these methods to provide a more complete benchmark.

- It would be valuable to investigate longer-term predictions and analyze how errors accumulate over time. Currently, there is limited discussion of failure cases or scenarios where the model underperforms. For example, although Figures 4, 5, and 6 highlight improvements, the paper lacks a systematic error or failure analysis. Moreover, the quantitative limitations of GaussEFW under very long streaming sequences are not thoroughly characterized, aside from a brief mention in Table 5.

- In several ablation studies (e.g., Table 2 and Table 7), results are presented without sufficient interpretation or explanation. For instance, why does applying noise only to dynamic categories lead to sharp performance changes? A deeper investigation into these observed effects would strengthen the paper’s analysis.

- While speed and error accumulation are central motivations of the work, the paper does not include benchmarking results on inference speed, memory usage, or computational efficiency compared to recent fast-splatting models. Including these comparisons would make the contributions more convincing and practically relevant.

### Questions
Please refer to the weakness above.

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
4

### Summary
The paper introduces an Evolutionary Entropy Flow framework for modeling the continuous evolution of dynamic 3D elements in 3D occupancy scenes. The proposed GaussEFW uses Gaussian latent representations and an implicit denoising network to model continuous scene evolution. It reformulates dynamic scene prediction as a Gaussian Entropy Flow process in latent space, contrasting with previous discrete refinement methods. Experiments on the nuScenes dataset with SurroundOcc annotations show stronger performance in dynamic element prediction and higher overall performance.

### Strengths
1. The paper presents an interesting and imaginative approach that builds a conceptual bridge between physics-inspired entropy flow and continuous world modeling. The idea of treating dynamic motion as an entropy flow in latent Gaussian space is conceptually novel.
2. This paper is essentially well-written and easy to understand.

### Weaknesses
1. While the proposed method is novel, its significance is limited to a technical contribution. For example, only nuScenes is used for quantitative results. Tests on other datasets (e.g., SemanticKITTI) would better demonstrate generalization to diverse motion scenarios.
2. Regarding the statement "Through these mechanisms, GaussEFW provides an efficient and robust approach for continuous 3D scene prediction...," the paper lacks a clear analysis of the computational efficiency and scalability of GaussEFW compared to previous works such as GaussianWorld. 
3. Evaluation against non-Gaussian temporal 3D methods (e.g., UniScene) is not discussed.
4. Minor: Several mathematical formulations (e.g., Eq. 7) are introduced without sufficient explanation or intuition, which makes them difficult to understand. The capitalization of some letters (e.g., Per, Trans, and Fuse in Eq. 2) should be formatted consistently.

### Questions
1. I wonder if it is possible to generate the trajectory of the ego vehicle as the future occupancy is generated. What's the planning performance compared to related works such as UniAD, GenAD, etc.?
2. Is it possible to visualize more seconds in Figure 5? I'm curious how far the world model can see.
3. I think in Figure 4, the results of GaussianWorld are smoother and better, especially the foreground, like the road.
4. Does the continuous flow modeling increase inference latency?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes GaussEFW, an approach to 3D occupancy prediction that models continuous scene evolution through an evolutionary entropy flow mechanism inspired by nonequilibrium thermodynamics. GaussEFW represents temporal dynamics as a single continuous Gaussian Entropy Flow in latent space. The method introduces three key processes, Entropy Producing, Entropy Exchanging, and Entropy Flowing to capture the motion of dynamic elements. Experiments on the SurroundOcc dataset demonstrate improvements over GaussianWorld and GaussianFormer variants.

### Strengths
1. The idea of implicit temporal fusion in latent Gaussian space is intersting.

2. The method shows empirical improvement on the SurroundOcc dataset over previous Gaussian-based approaches.

3. The attempt to draw an analogy with nonequilibrium thermodynamics is intellectually creative.

### Weaknesses
1. The reviewer’s main concern is that the use of nonequilibrium thermodynamics as an explanatory framework is unconvincing. It is difficult to see a direct connection between the 3D occupancy prediction task and the notion of entropy. The core technical contribution, i.e, modeling the dynamic evolution of Gaussian kernels in latent space, can be fully developed without invoking thermodynamic principles. Framing the work under this lens makes the conceptual motivation unclear and obscures the underlying technical ideas.

2. Fig. 1 (b) does not clearly illustrate what changes are driven by the proposed entropy flow. The visual explanation fails to make evident the mechanism or benefit of this flow, leaving the reader uncertain about the core message that Fig. 1 is meant to convey.

3. Eqs. (2) and (3) appear to describe similar processes using different notations, making it unclear what distinct roles they play. The use of the term world model to denote the network w is ambiguous, as “world model” can refer to multiple subfields and paradigms. While it seems that the authors intend to describe a temporal modeling process akin to an RNN-based fusion scheme, more precise terminology is needed to eliminate confusion and to clarify what is genuinely novel.

4. The paper does not sufficiently describe the 3D occupancy prediction task itself—particularly the final loss function or optimization objectives. Without these details, the method section remains incomplete, and readers who are not already familiar with this research area will struggle to reproduce or properly interpret the approach.

5. No results are provided on computational efficiency, such as inference FPS or parameter count. Since the proposed temporal fusion mechanism increases model complexity, efficiency is a crucial consideration in 3D occupancy prediction. Without reporting inference efficiency or comparing with baselines, it is unclear whether the performance gain stems from a more effective design or simply from added parameters and computation.

6. The left-hand example in Figure 4 shows noticeable discrepancies between the predicted occupancy and the ground truth. However, it is unclear what the color coding represents or what qualitative aspect is being highlighted. Including the corresponding camera image as reference and annotating which colors correspond to which object categories would make the qualitative analysis more interpretable.

7. The experimental validation is limited to the SurroundOcc dataset. However, the mainstream benchmark for nuScenes-based occupancy prediction is Occ3D, and evaluation on this dataset is necessary for fair comparison with SOTA methods such as COTR [1], FB-Occ [2], and GDFusion [3]. Furthermore, results on additional datasets (e.g., SemanticKITTI, Waymo-Occ) are recommended to demonstrate generalization. Reporting the standard IoU metrics used by prior works would also make the comparison clearer and more credible.

8. GDFusion [3] achieves 25.5 mIoU on the SurroundOcc benchmark, yet it is not included in the comparison table. This omission weakens the empirical claims, as GDFusion represents a strong baseline closely related to the proposed approach.

9. The paper’s use of the term world model seems conceptually narrow. The proposed method focuses on temporal fusion for occupancy prediction, rather than the broader predictive or generative modeling typically associated with world models. Prior works in temporal fusion, such as StreamPETR or GDFusion, also address dynamic scene evolution; hence, the authors should clarify what distinctive aspects of their approach justify the “world model” label.

Minor：

10. L-81: There is a missing space after the period in “process.Collectively.”

11. The subsection title at L-161 lacks a terminating period.

12. The titles of Sections 1, 2, 3, and 5 do not end with a period, whereas Section 4 and several of its subsections do. The punctuation style should be standardized across all section and subsection titles.

13. It is unclear whether the columns in Tab. 5 correspond to the number of blocks or another architectural component. This should be explicitly stated either in the table header or in its caption.

14. Several recent works specifically designed for temporal fusion in occupancy prediction are not discussed, including GDFusion [3] and CVT-Occ [4]. Adding a comparison or a discussion of their relationship to the proposed method would provide better context and strengthen the paper’s positioning.

[1] COTR: Compact Occupancy Transformer for Vision-Based 3D Occupancy Prediction, CVPR 2024

[2] FB-Occ: 3D Occupancy Prediction Based on Forward–Backward View Transformation

[3] Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction, CVPR 2025

[4] CVT-Occ: Cost Volume Temporal Fusion for 3D Occupancy Prediction, ECCV 2024

### Questions
1. In L-210, the phrase “While this method performs well in static scenes,” is unclear. What specific method does “this method” refer to? The preceding section only defines a general temporal modeling paradigm but does not describe any particular implementation. Please clarify which approach is being referenced here.
2. Could the authors elaborate on how E_P in Eq. (5) is implemented in practice, and how it interacts with or extends Eq. (4)? A more explicit description of this module’s design and its computational role within the overall entropy flow framework would greatly improve clarity.

### Soundness
2

### Presentation
2

### Contribution
2
