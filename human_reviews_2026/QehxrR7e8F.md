# Cross3DReg: Towards a Large-scale Real-world Cross-source Point Cloud Registration Benchmark

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6

## Abstract
Cross-source point cloud registration, which aims to align point cloud data from different sensors, is a fundamental task in 3D vision. However, compared to the same-source point cloud registration, cross-source registration faces two core challenges: the lack of publicly available large-scale real-world datasets for training the deep registration models, and the inherent differences in point clouds captured by multiple sensors. The diverse patterns induced by sensors pose great challenges in robust and accurate point cloud feature extraction and matching, which negatively influence the registration accuracy. To advance research in this field, we construct Cross3DReg, the currently largest and real-world multi-modal cross-source point cloud registration dataset, which is collected by a rotating mechanical LiDAR and a hybrid semi-solid-state LiDAR, respectively. Moreover, we design an overlap-based cross-source registration framework, which utilizes unaligned images to predict the overlapping region between source and target point clouds, effectively filtering out redundant points in the irrelevant regions and significantly mitigating the interference caused by noise in non-overlapping areas.
Then, a visual-geometric attention guided matching module is proposed to enhance the consistency of cross-source point cloud features by fusing image and geometric information to establish reliable correspondences and ultimately achieve accurate and robust registration.
Extensive experiments show that our method achieves state-of-the-art registration performance. Our framework reduces the relative rotation error (RRE) and relative translation error (RTE) by 63.2% and 40.2%, respectively, and improves the registration recall (RR) by 5.4%, which validates its effectiveness in achieving accurate cross-source point cloud registration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the more specific and challenging problem of cross-source point cloud registration, and proposes a real-world dataset named Cross 3DReg for outdoor scenes, which can be used for training and to some extent fills the gap in the field. At the same time, a cross-source registration framework based on cross-registration is also proposed. It uses unaligned images to predict the overlapping areas of the source point cloud and the target point cloud, effectively filtering out redundant points in the irrelevant areas and significantly reducing the interference of noise in the non-overlapping areas. Finally, cross-source point cloud registration is achieved.

### Strengths
1. In the previous research field, either the cross-source datasets were too small to support training, or they were synthetic datasets. The real-world cross-source datasets of this paper have to some extent filled the gap in this field.
2. The article is well-written and easy to understand.
3. This paper uses image features to enhance point cloud features to solve the data problems between cross-source point clouds, which is an effective idea.

### Weaknesses
1. Although cross-source point cloud registration is novel, the dataset you provided, Cross 3DReg, consists of rotating mechanical lidar and mixed semi-solid lidar from different sources. I believe this is essentially radar scanning, and the distribution differences between the data of the two are not significant. The research significance and value are not great, and it is questionable.
2. The capitalization of the names of the baselines needs to be standardized, such as RoITr.
3. The superscripts in formulas and symbols (point clouds and point cloud features) are somewhat chaotic and not conducive to reading. It is recommended to unify and standardize them.
4. The method in this paper is an improvement based on the Geotrans framework, adding two modules, OMP and VGAM. The innovation at the method level is not significant.
5. The ablation experiment setting is unreasonable. A set of VGAM w/o OMP experiments should be added in Table 3. Only from (a) and (d), it is not clear whether it is the influence of OMP or VGAM. The effect of OMP is not clear.

### Questions
1. Based on experience, it is necessary to re-examine whether the OMP module is necessary. Firstly, in some cases, the overlapping estimation of images and point clouds may introduce incorrect masks. Secondly, the overlapping area between the point clouds selected by the two image overlapping masks is smaller than that between the two original point clouds. This reduction in overlap will definitely affect the registration effect. Thirdly, if the data volume is sufficient, I think directly using visual features to enhance the complete point cloud features instead of applying masks to the point clouds may yield better results.
2. You mentioned in 139-141 that the recent methods Zhang et al. (2022a); Xu et al. (2024; 2025) have introduced other modal information, which may fail in cross-source point cloud registration tasks. However, in your method, you also introduced images (containing color, semantic, and texture information), but you only compared it with the point cloud input methods (such as GeoTrans) in the experiments, which is somewhat unfair. Please add comparative experiments with methods that incorporate multimodal input.

### Soundness
2

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
This paper focuses on the task of cross-source point cloud registration, which aims to align 3D data collected from different types of sensors. Compared with same-source registration, this task is more challenging due to the lack of large-scale real-world datasets and the inherent differences in sensor characteristics. To address these issues, the authors construct Cross3DReg, a large-scale real-world multi-modal dataset collected using a rotating mechanical LiDAR and a hybrid semi-solid-state LiDAR. They further propose an overlap-based registration framework that predicts overlapping regions between unaligned images and point clouds to filter out irrelevant points, and introduce a visual-geometric attention module to enhance feature consistency across modalities.

### Strengths
This paper addresses the problem of cross-source point cloud registration, which is technically challenging. 

The construction of a cross-source registration dataset represents a contribution to the community. 

The manuscript is overall well written and easy to follow.

### Weaknesses
1. The core concern lies in the introduction of additional image information during the point cloud registration process. This raises questions about the fairness of the comparison for the baselines do not use the images, and Table 3 shows that the performance improvement primarily stems from the inclusion of image data.

3. Although the authors claim that the images and point clouds are not geometrically aligned, they still assume a significant spatial overlap between the image and the two point clouds. Moreover, the registration process is conducted only within these “image-pc overlapping regions”, which seems difficult to achieve in practice.

3. Furthermore, first determining the overlapping regions between the image and point clouds appears somewhat counter-intuitive, since the modality gap between images and point clouds is typically larger than that between cross-source point clouds.

### Questions
Please refer to the weaknesses listed above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a cross-source point cloud registration method that aligns point cloud data from different sensors. The authors construct a large-scale multi-modal dataset for cross-source point cloud registration. They introduce an overlap-based registration framework that uses unaligned images to predict overlapping regions between source and target point clouds. Additionally, they propose a visual-geometric attention-guided matching module that enhances cross-source point cloud feature consistency by fusing image and geometric information.

### Strengths
- A large scale cross source point cloud registration dataset is constructed.
- An overlap-based cross-source point cloud registration method is proposed.

### Weaknesses
- The dataset construction uses a hybrid semi-solid-state LiDAR to capture source point clouds and a 64-line rotating mechanical LiDAR for target point clouds. However, the rationale for this choice is unclear. Is this a common practice in real-world applications?
- One key to the proposed method's success is predicting the overlap region between source and target point clouds, as well as between point clouds and images. More experimental results demonstrating this prediction should be added.
- The experiments are insufficient. It would better to also evaluate the proposed method on other cross-source point cloud registration datasets. In addition, more state-of-the-art cross-source point cloud registration methods discussed in related work should be compared.

### Questions
- What is the rationale for constructing the cross-source point cloud registration dataset in such a way?
- How does the proposed method perform on other existing cross-source point cloud registration datasets?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel method for cross-source point cloud registration. The key insight is leveraging images to detect overlap regions between point clouds captured by different sensors, thereby addressing registration failures caused by sensor discrepancies. To facilitate research in this area, the authors introduce Cross3DReg, a new large-scale real-world dataset for cross-source point cloud registration, which was previously lacking in the field. Experimental results demonstrate that the proposed method significantly outperforms existing point cloud registration approaches, achieving substantial improvements across evaluation metrics.

### Strengths
- **Clear presentation**: The paper is well-written and easy to follow, providing sufficient context to understand both the field and the proposed method's contributions.
- **Valuable dataset contribution**: The proposed Cross3DReg dataset addresses a critical gap in the community, providing the large-scale real-world benchmark that researchers have been seeking for cross-source point cloud registration.
- **Strong experimental validation**: The proposed method demonstrates substantial performance gains over existing baselines. The ablation study effectively validates the contribution of each component in the proposed approach.

### Weaknesses
- **Concerns regarding evaluation fairness**: While leveraging auxiliary image data to identify overlap regions and guide point matching through the visual-geometric attention module is a distinctive feature of the proposed method, this additional modality was not available to baseline methods. This asymmetry in input data raises concerns about fair comparison—the performance gains may partially stem from access to richer input rather than solely from algorithmic improvements.

### Questions
- **Performance of VRHCF on Cross3DReg**: VRHCF is the only dedicated cross-source point cloud registration method included in the comparison, yet it appears to fail completely on the Cross3DReg benchmark. Could the authors provide insight into why this occurs? Is this due to specific characteristics of the Cross3DReg dataset (e.g., scale, sensor types, scene complexity), limitations in VRHCF's architecture, or implementation issues?

### Soundness
3

### Presentation
3

### Contribution
3
