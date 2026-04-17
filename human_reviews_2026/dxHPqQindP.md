# SkyEvents: A Large-Scale Event-enhanced UAV Dataset for Robust 3D Scene Reconstruction

- Decision: Accept (Poster)
- Scores: 6, 2, 2, 6

## Abstract
Recent advances in large-scale 3D scene reconstruction using unmanned aerial vehicles (UAVs) have spurred increasing interest in neural rendering techniques. However, existing approaches with conventional cameras struggle to capture consistent multi-view images of scenes, particularly in extremely blurred and low-light environments, due to the inherent limitations in dynamic range caused by long exposure and motion blur resulting from camera motion. As a promising solution, bio-inspired event cameras exhibit robustness in extreme scenarios, due to their high dynamic range and microsecond-level temporal resolution. Nevertheless, dedicated event datasets specifically tailored for large-scale UAV 3D scene reconstruction remain limited. To bridge this gap, we introduce SkyEvents, a pioneering large-scale event-enhanced UAV dataset for 3D scene reconstruction, incorporating RGB, event, and LiDAR data. SkyEvents encompasses 45 sequences, spanning over 8 hours of video, captured across a diverse set of illumination conditions, scenarios, and flight altitudes. To facilitate the event-based 3D scene reconstruction with SkyEvents, we propose the Geometry-constrained Timestamp Alignment (GTA) module to align timestamps between the event and RGB cameras. Furthermore, we introduce a Region-wise Event Rendering (RER) loss for supervising the rendering optimization. With SkyEvents, we aim to motivate and equip researchers to advance large-scale 3D scene reconstruction in challenging environments, harnessing the unique strengths of event cameras. Dataset and code will be available at https://github.com/Anthony-ECPKN/SkyEvent.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
1. **Originality-wise**: the paper proposes a large-scale UAV event-based dataset for robust 3D reconstruction.
2. **Quality-wise**: the proposed dataset has varied large-scale scenes including different flight path, scenarios types, illumination, and height settings.
3. **Clarity-wise**: the manuscript is clearly written, with well-structured methodology, detailed explanations, and intuitive visualizations that enhance understanding.

### Strengths
1. A Pioneering and Highly Valuable Dataset: The most significant contribution of this work is the SkyEvents dataset itself. SkyEvents is not only large in scale (over 8 hours, covering 0.72 km²) but also rich in modalities (RGB, Event, LiDAR) and provides accurate 6-DoF poses, filling a major gap. 
2. Solves a Critical Practical Problem: The proposed GTA module directly addresses a very challenging yet common real-world problem: precise timestamp alignment between multiple sensors, especially in low-cost setups without hardware synchronization

### Weaknesses
1. **Regarding the GTA Module**: While the paper demonstrates the module's effectiveness indirectly through its positive impact on 3D reconstruction and other downstream tasks, it lacks a more direct and intuitive quantitative evaluation. To compellingly validate the module's performance, I suggest the authors conduct experiments on existing datasets with pre-aligned ground-truth timestamps (e.g., DSEC, MVSEC). This would allow for a direct measurement of the method's alignment error and a quantitative comparison of its effectiveness against other approaches.
2. **Regarding Low-Light Image Generation**: The approach of generating synthetic low-light images is practical for settings where scenes cannot be re-captured. However, aligning event data captured directly under bright sunlight with these synthetically darkened images could introduce a significant data mismatch. Event cameras are known to exhibit different noise characteristics, such as leaky events, in bright conditions compared to true dark environments. This raises concerns about the fidelity of the alignment. I am curious if the authors implemented any specific procedures to address this potential distortion.
3. **Regarding Evaluation Metrics**: I agree that using the same metrics as the original 3DGS is valid for the motion deblur task. However, for low-light or over-exposed conditions, these reference-based metrics may become ineffective due to the quality degradation of the ground-truth images themselves. An evaluation based on a flawed reference is not meaningful. I recommend that the authors incorporate no-reference image quality assessment metrics (e.g., BRISQUE, HyperIQA) to ensure the validity and intuitiveness of the evaluation results.
4. The video visualization is lacking and this reduces the credibility of the real effect.

---
I have listed my concerns, and the score will be adjusted based on the author's response.

### Questions
Please refer to Weaknesses part.

### Soundness
3

### Presentation
2

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
This paper introduces SkyEvents, a large-scale 3D scene reconstruction dataset that integrates multi-domain signals, including RGB video, event stream, and LiDAR point clouds. The dataset spans 481 minutes, making it three times longer than the previously largest dataset, EvMAPPER. To handle temporal misalignment between RGB and event signals, this paper proposes a Geometry-constrained Timestamp Alignment (GTA) method. Additionally, it introduces a Region-based Event Rendering (RER) loss to improve event-based rendering optimization. Experiments on SkyEvents include comparisons with and without event inputs in rendering, qualitative analyses of the RER loss, and evaluations against existing reconstruction baselines.

### Strengths
- The large-scale real-world SkyEvents dataset is a valuable asset for the scene reconstruction community, providing diverse multi-domain signals.

- Beyond the dataset itself, the methodology for large-scale aerial data collection and the inclusion of hardware details offer practical guidance for future dataset collection efforts.

- The proposed temporal alignment strategy between RGB and event signals enhances the dataset’s usability and convenience for potential users.

### Weaknesses
**Incremental Contributions**

a. The proposed Geometry-constrained Timestamp Alignment (GTA) is practically useful for potential users of the SkyEvents dataset. However, in terms of novelty, it appears to be an extension of Dynamic Time Warping (DTW) [1] rather than a fundamentally new formulation. A discussion or comparison with DTW would strengthen the paper.

[1] Berndt, Donald J., and James Clifford. “Using Dynamic Time Warping to Find Patterns in Time Series.” In AAAI Workshop on Knowledge Discovery in Databases, 1994.

b. The motivation and formulation of the Region-based Event Rendering (RER) loss are difficult to follow. Equation (5) is unclear since $\mathcal{C} _ {\theta}$ is not defined anywhere in the paper, and the function $C$ in the definition of $E$ is also unspecified. Even assuming that this loss is intended to minimize the displacement difference of certain features between the initial and final frames (i.e., $\log \mathcal{C} _ {\theta} (\hat{I} _ {t_1}) - \log \mathcal{C} _ {\theta} (\hat{I} _ {t_2})$) and the accumulation of event activities ($E(t _ 1, t _ 2)$), it remains unclear how this formulation guides or benefits event rendering optimization.

c. Excluding GTA and RER, the main contribution is limited to the dataset itself. However, the supplementary information supporting the usefulness of SkyEvents is quite limited. The paper should elaborate on the design philosophy behind the dataset—specifically, why certain characteristics were selected and how they differ from existing datasets. Furthermore, a comprehensive comparison with other datasets in terms of these characteristics is necessary to better demonstrate the dataset’s uniqueness and value.

**Immature Presentation**

a. As mentioned earlier, some functions used in the definition of the RER loss (i.e., $C_{\theta}$ and $C$) are not defined. Furthermore, the connection between the mathematical definition of RER and its underlying design intuition is not sufficiently explained, making it difficult to understand the motivation behind this formulation.

b. The related work section primarily focuses on datasets. If GTA and RER loss are intended to be core contributions of the paper, a more comprehensive review of related methods should be included to demonstrate the distinct features of the proposed methods compared to existing ones, and to clarify why these components are crucial for the given task, i.e., multi-modal dataset construction and scene reconstruction.

c. The evaluation structure requires revision.

- It is unclear what the main results of the paper are in Section 4.4 (Results), and how these differ from the results presented in Section 4.5.
The authors should clarify what readers are expected to focus on in the baseline comparison shown in Figure 6.

- In Figure 4, it is not explained why comparisons are made with and without GS, and what specific conclusion can be drawn from this comparison.

**No Benefit in Evaluation**

a. In Figure 4, it is unclear what benefit the RER loss provides; the difference between the results with and without RER is not evident.

b. In Table 2, the use of event data does not consistently improve performance. The purpose or takeaway of this comparison should be clarified—what should the reader conclude about the effectiveness of incorporating event signals?

### Questions
Is it typical to use LiDAR point clouds as ground truth for scene reconstruction? While LiDAR provides higher accuracy than other sensing modalities, it still contains measurement noise and unobserved regions from the camera’s viewpoint, which may lead to inaccurate quantitative evaluation. How is this limitation addressed or considered in the evaluation process?

### Soundness
2

### Presentation
1

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
This paper introduces SkyEvents, a large-scale multimodal UAV dataset incorporating RGB, event, and LiDAR data, with particular emphasis on the role of event cameras for 3D scene reconstruction in urban environments. While the dataset is valuable in its focus on event-based sensing under challenging conditions, the paper's primary contribution remains at the data collection and preliminary fusion level.

### Strengths
The strengths of this work include the creation of the comprehensive SkyEvents dataset, which features diverse challenging conditions and multimodal data (RGB, event, LiDAR). It potentially enables improved 3D reconstruction and perception algorithms, introduces effective synchronization (GTA) and rendering techniques (RER), and demonstrates the beneficial impact of event data in various challenging scenarios like low-light and motion blur.

### Weaknesses
The main limitation is the lack of in-depth task-specific analysis demonstrating the concrete benefits of event data for downstream perception tasks such as depth estimation, semantic segmentation, or real-time scene understanding. The experimental results are mostly qualitative and do not sufficiently quantify the actual improvements brought by event modalities. As a result, the paper falls short of establishing clear performance gains or providing comprehensive evaluations that would justify its impact as a benchmark resource.

Furthermore, insufficient details on calibration, synchronization, and data processing hinder reproducibility and broader adoption. Without extensive task-oriented experiments and rigorous benchmarking, the work remains at an early exploratory stage, limiting its immediate utility for advancing event-based perception research.

Overall, while the dataset is a promising resource, the paper does not demonstrate enough analysis or experimental validation to warrant publication in its current form. Significant additional work is necessary to substantiate the practical advantages of the provided data for perception tasks.

### Questions
1. How did you address synchronization challenges between RGB, event, and LiDAR sensors?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work proposes a large-scale multimodal UAV dataset (i.e., SkyEvents) for 3D scene reconstruction, which includes RGB images, events, and LiDAR data. This newly built dataset provides 45 hybrid sequences considering light changes, scene diversity, and flight altitudes. Besides, the authors present a GTA module to align timestamps between two streams and design an RER loss function for supervising the rendering optimization. I believe the large-scale dataset will provide a challenging benchmark for event-based 3D reconstruction using agile drones.

### Strengths
1. The topic of event-based 3D reconstruction using agile drones is very interesting.

2. The large-scale multimodal UAV dataset, which includes events, frames, and point clouds, will benefit the event-based vision community.

3. The writing is clear and easy to understand.

### Weaknesses
1. Figure 1 in the manuscript is very impressive and nicely illustrates large-scale 3D scene reconstruction. The authors are encouraged to include more examples of large-scale 3D scenes in the visualization results, ideally covering high-speed or low-light scenarios, to highlight the advantages of event-based modalities compared to RGB frames and RGB–event multimodal settings.

2. As a dataset or benchmark paper, it would be beneficial to evaluate more existing event-based 3D reconstruction methods for comparison. At least incorporating some open-source algorithms would make the dataset more useful and accessible to a broader research community.

3. Given that this work mainly focuses on presenting a large-scale dataset, it might not be ideal to introduce new components such as the GTA module and RER loss in the same paper, as this could make the focus of the paper less clear.

4. The inclusion of LiDAR point clouds in the dataset is an excellent idea. However, the manuscript currently lacks sufficient experimental results or demonstrations showing the value or impact of the point cloud data.

5. For a dataset paper, Table 1 is particularly important. The authors are encouraged to further enrich this table and clearly articulate the unique contributions and advantages of their dataset compared to existing ones. This would make the paper more solid and convincing.

6. Minor suggestions in the future work:

a. In the Related Work section, the titles of Sections 2.1 and 2.2 could be revised, as their contents overlap. Section 2.2 could be renamed to Event-based 3D Reconstruction for clarity.

b. For large-scale UAV datasets, it is suggested that future work could employ stereo event cameras, as in [1,2]. The authors might even consider multi-UAV collaboration to collect coordinated datasets for 3D scene reconstruction, potentially leveraging event camera simulators such as [3].

c. The authors could also refer to and cite more recent works on event-based 3D reconstruction, e.g., [4].

[1] Active Event-based Stereo Vision, CVPR 2025.

[2] Enhanced event-based dense stereo via cross-sensor knowledge distillation, ICCV 2025.

[3] Physical-based event camera simulator, ECCV 2024.

[4] EvaGaussians: Event stream assisted Gaussian splatting from blurry images, ICCV 2025.

### Questions
1. How were the event camera (e.g., EKV4) and the frame-based camera (e.g., DJI Action 4) physically integrated and synchronized to ensure spatiotemporal alignment, particularly under the vibrations experienced during UAV flight?

2. The authors suggest that the primary motivation for introducing an event camera in UAV scenarios is to address high-speed motion blur. Under what specific conditions of flight altitude and velocity would an event camera provide a clear advantage over a standard 20 FPS RGB camera, which is susceptible to such motion blur?

### Soundness
3

### Presentation
3

### Contribution
3
