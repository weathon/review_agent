# SatDreamer360: Multiview-Consistent Generation of Ground-Level Scenes from Satellite Imagery

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 8

## Abstract
Generating multiview-consistent $360^\circ$ ground-level scenes from satellite imagery is a challenging task with broad applications in simulation, autonomous navigation, and digital twin cities. Existing approaches primarily focus on synthesizing individual ground-view panoramas, often relying on auxiliary inputs like height maps or handcrafted projections, and struggle to produce multiview consistent sequences. In this paper, we propose SatDreamer360, a framework that generates geometrically consistent multi-view ground-level panoramas from a single satellite image, given a predefined pose trajectory. To address the large viewpoint discrepancy between ground and satellite images, we adopt a triplane representation to encode scene features and design a ray-based pixel attention mechanism that retrieves view-specific features from the triplane. To maintain multi-frame consistency, we introduce a panoramic epipolar-constrained attention module that aligns features across frames based on known relative poses. 
To support the evaluation, we introduce VIGOR++, a large-scale dataset for generating multi-view ground panoramas from a satellite image, by augmenting the original VIGOR dataset with more ground-view images and their pose annotations. Experiments show that SatDreamer360 outperforms existing methods in both satellite-to-ground alignment and multiview consistency.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a framework for generating multiview-consistent 360° ground-level panoramic sequences from a single satellite image and a predefined camera trajectory. The two key methodological contributions are: 1. ray-guided cross-view feature conditioning (a ray-based pixel attention mechanism samples features along viewing rays, establishing explicit geometric correspondences between satellite and ground views), 2. epipolar-constrained attention for panoramas (an attention mechanism that enforces multiview consistency across frames by restricting attention to geometrically valid correspondences, reducing computational complexity). The paper also introduces the VIGOR++ Dataset. Experiments demonstrate state-of-the-art performance across perceptual, semantic, pixel-level, and multiview consistency metrics, outperforming methods like Sat2Density, ControlS2S, and EscherNet.

### Strengths
Novel Problem Formulation: While satellite-to-ground synthesis exists, the paper uniquely formulates the problem as generating geometrically consistent multiview sequences from a single satellite image with trajectory control. This is more challenging and practical than prior single-view or multi-satellite approaches. Most work targets perspective images; the consistent treatment of equirectangular projections throughout the pipeline is less explored.

Ray-based pixel attention with learnable offsets: Starting with uniform sampling and progressively learning offsets and weights as denoising proceeds is an elegant solution to the "noisy latent" problem in early diffusion steps. This bridges geometric sampling with diffusion's iterative nature.

Extending epipolar geometry from pinhole to equirectangular projections and integrating into attention mechanism requires non-trivial mathematical reformulation.

Triplane + diffusion integration: While triplanes exist, conditioning latent diffusion with ray-sampled triplane features (rather than global or BEV features) is novel. The cross-view hybrid attention for enriching planes with orthogonal projections adds spatial reasoning.

VIGOR++ represents substantial effort: 90K+ pairs, 10 cities, semi-automatic trajectory construction with manual refinement. The methodology for building continuous sequences from Street View (connectivity graphs, DFS path extraction) shows engineering creativity.

### Weaknesses
It is unfortunate that Deng et al. (2024) and Xu & Qin (2025) do not provide source code yet. They both appear quite strongly relevant so further discussion of input requirements (single vs. multi-view satellite), architectural differences, computational costs, and theoretical advantages would improve the paper.

Methods like Stable Video Diffusion or AnimateDiff could be adapted by conditioning on satellite images. These would test whether the geometric reasoning actually provides value over general temporal coherence mechanisms.

EscherNet seems like a weak baseline comparison. It is designed for object-centric multiview synthesis, not cross-view scene generation. The paper states they "adapt it to our setting by treating the satellite image as the source view" but provides no implementation details. This adaptation may be fundamentally disadvantaging EscherNet.

All metrics reported as single numbers without error bars, confidence intervals, or variance. Diffusion models are stochastic - different random seeds produce different results. So it would be good to have some sense of whether results are statistically significantly better.

There are quite a few hyperparameters without strongly justified values:
K (number of ray samples): value never specified in main paper
M (epipolar constraint points): "M ≪ HW" but exact value unclear
Why 2 preceding frames in sparse attention? No ablation on alternatives
Triplane resolution not specified until limitations section
Why 300 epochs for each stage? Was convergence analyzed?
Learning rate 7e-05 - was this tuned or borrowed from SD1.5?
Why batch size 32 → 8 → 4 across stages?

### Questions
Based on my comments above, questions that the authors could respond to in the rebuttal:

1. Statistical significance of results
2. Deeper methodological comparitive discussion to Deng et al. (2024) and Xu & Qin (2025)
3. Justification of hyperparameter choices
4. Sparse interframe attention - why 2 frames?
5. Provide additional EscherNet adaptation details
6. Why ResNet for satellite encoding? Have you tested modern alternatives like Vision Transformer?

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
This paper introduces SatDreamer360, a diffusion-based framework that generates multiview-consistent ground-level panoramas conditioned on a single satellite image.
The model integrates three main components:
Ray-guided cross-view feature conditioning, leveraging a triplane representation to model geometric correspondences between satellite and ground perspectives.
Epipolar-constrained attention, enforcing temporal and geometric consistency across panoramic frames using equirectangular projection geometry.
A new dataset, VIGOR++, extending the VIGOR dataset with continuous ground-level sequences and trajectory annotations for evaluating satellite-to-ground synthesis.

### Strengths
1.	Satellite-to-ground image generation is a challenging and underexplored topic. It has practical value for simulation, urban modeling, and cross-view localization.  
2.	The combination of triplane representation, ray-guided sampling, and epipolar attention is technically sound. The pipeline effectively bridges satellite conditioning and panoramic generation.
3.	The results show noticeable improvements in both image realism and geometric coherence. The qualitative examples look visually convincing.

### Weaknesses
1.	The core architectural ideas—triplane representation (EG3D), ray-based sampling (MVDream, Zero123++), and epipolar-constrained attention (EpiDiff)—are largely from prior work. SatDreamer360 primarily integrates these components within a diffusion framework for a new application. While the system integration is well-executed, it lacks a fundamentally new algorithmic or theoretical contribution.
2.	Ablation and analysis are not deep enough. It’s unclear how much each proposed module contributes. The paper would benefit from more detailed ablation and efficiency studies.
3.	Since the task involves generative image synthesis, some form of human preference or perceptual realism test would strengthen the claims.

### Questions
1. Ablation completeness:
Can you provide quantitative ablation results isolating the effects of (a) the triplane representation, (b) the ray-guided conditioning, and (c) the epipolar attention? How much performance gain does each contribute individually?
2. Did you conduct any small-scale user study or human preference comparison? If not, could you report preliminary results or plan to include one in the final version?
3. Since VIGOR++ uses Google Street View imagery, how is licensing and privacy handled?

### Soundness
3

### Presentation
3

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
This paper introduces a diffusion-based framework capable of generating sequential ground-view images conditioned on a given satellite image and trajectory. The method employs a ray-based pixel attention mechanism to effectively aggregate features from a triplane representation. Furthermore, an Epipolar-Constrained Attention module is incorporated to enforce geometric consistency across multiple camera views. In addition, the authors release a new dataset establishing correspondences between satellite maps and ground-view sequences.

### Strengths
* The paper’s objectives are novel and clearly defined. 
* The proposed approach is logically sound, and the contributions are substantial — including the introduction of a new dataset for a newly defined task. 
* The experiment demonstrates the effectiveness of the proposed method with state-of-the-art performance.

### Weaknesses
* Some parts of the paper are not fully explained and require further clarification from the authors (Details shown in the Question part).
* The epipolar constraint only ensures local consistency between two frames rather than global consistency, which makes the overall constraint relatively weak.
* There aren’t enough experiments to examine how each module affects the consistency metric, which I personally consider a core metric for multi-image generation. The existing ablations—for example, comparing full cross-attention with epipolar attention—some consistency metrics (e.g., FVD), the enhancement does not seem significant. The authors need to include more comprehensive evaluations to substantiate their claim of enhancing consistency performance..

### Questions
1. What is the purpose of using a triplane representation? From the ablation studies (Table 5), it seems that the performance gain is not very significant even with increased computational resources. Moreover, since satellite images only provide 2D information, how can the features along the Z-axis in the triplane representation be effectively extracted without height information?

2. When constructing the dataset, how to determine the sampling density of ground images? Does the sampling density (high or low) affect the quality of the generated images?

3. How does the system deal with the dynamic object issue during the image generation process?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SatDreamer360, a generative framework capable of producing multi-frame geometrically consistent ground-level panoramic image sequences from a single satellite image and a predefined camera trajectory. For single-frame generation, the framework employs a Triplane representation and a ray-based pixel attention mechanism to enhance geometric fidelity and visual realism. To ensure spatial and temporal coherence across generated frames, the authors further propose a Panoramic Epipolar-Constrained Attention module, which explicitly models geometric relationships between different views. To evaluate the proposed one-to-many sequence generation task more comprehensively, the authors introduce the VIGOR++ dataset, which extends the original VIGOR dataset by incorporating additional cities and continuous ground-view trajectories, enabling a more complete assessment of the framework’s performance. Overall, the proposed panoramic sequence generation method significantly improves the geometric consistency and continuity of satellite-to-ground image synthesis, providing higher-quality data for downstream applications.

### Strengths
1. Introducing image sequence generation into the satellite-to-ground view image synthesis task enables geometrically consistent sequential images, which are more suitable for downstream applications such as autonomous driving and 3D reconstruction. This approach further enhances the practical value of cross-view generation.

2. The proposed ray-based pixel attention mechanism and the epipolar-constrained attention leverage the geometric priors inherent in the imaging process, thereby improving the interpretability of both the module and the overall framework.

3. The authors constructed the VIGOR++ dataset, which not only increases the diversity of data types by expanding the number of cities but also provides carefully processed street-view image trajectories. This dataset offers a comprehensive and reliable benchmark for the proposed one-to-many sequence generation task and holds great potential value in other fields such as urban perception, autonomous driving, and 3D reconstruction.

4. The authors conducted extensive and thorough experiments to validate their framework, reflected in the use of diverse evaluation metrics across multiple levels and detailed ablation studies for each module. The supplementary materials provide additional methodological details and experimental analyses, and the evaluation of the generated results in downstream tasks is particularly commendable.

### Weaknesses
1. The proposed sequence generation task relies on a single satellite image and predefined trajectory inputs. However, the authors provide insufficient introduction and discussion regarding the trajectory data, including details such as the number of frames within each trajectory, the spatial intervals between frames, and the relative positioning of the trajectories within the satellite imagery.

2. Although the proposed VIGOR++ dataset expands the number of cities to enhance data diversity, it remains limited to U.S. cities. The absence of data from non-U.S. regions such as Asia and Europe may pose potential limitations to the model’s generalization capability.

### Questions
1. It is recommended that the authors include a more detailed description of the dataset in the supplementary materials, such as the number of satellite and street-view images, the number of trajectories corresponding to each satellite image, and the distribution of the dataset across different cities.

2. Considering that the trajectory information is an essential component of the model input and directly governs the generated sequence outputs, I suggest that the authors further discuss the design of trajectory sequences. Specifically, how long the trajectories can be supported, how the sampling interval influences the generation results (as the current discussion only covers short sequences), and whether the relative spatial relationship between the trajectory and the satellite image affects the generation results.

3. Related to the previous point, the current experimental results mainly involve short and nearly straight trajectories. I am curious whether the proposed method can maintain strong geometric consistency when dealing with longer or curved trajectories, especially at intersections or turning points in urban environments.

4. The current visualizations effectively demonstrate the model’s strong multi-view geometric consistency. However, including some failure cases in the supplementary materials and analyzing the underlying reasons would make the framework more convincing.

5. In Figure 2, could the description of the EPIPOLAR-CONSTRAINED ATTENTION be made clearer? The explanation on the right side of this figure appears somewhat confusing.

6. A concern raised by the second limitation is whether the tri-plane representation can adequately replace traditional 3D representations in capturing more complex vertical spatial structures or non-planar terrains. Considering that the dataset primarily focuses on major U.S. cities with relatively flat terrain and similar urban layouts, it would be beneficial to include inference experiments on cities from Asia, Africa, or Europe, even if ground-truth references are not available.

### Soundness
3

### Presentation
3

### Contribution
4
