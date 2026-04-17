# Depth Anything 3: Recovering the Visual Space from Any Views

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 8

## Abstract
We present Depth Anything 3 (DA3), a model that predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses. 
In pursuit of minimal modeling, DA3 yields two key insights:
a single plain transformer (e.g., vanilla DINOv2 encoder) is sufficient as a backbone without architectural specialization, and a singular depth-ray prediction target obviates the need for complex multi-task learning. Through our teacher-student training paradigm, the model achieves a level of detail and generalization on par with Depth Anything 2 (DA2).
We establish a new visual geometry benchmark covering camera pose estimation, any-view geometry and visual rendering. On this benchmark, DA3 sets a new state-of-the-art across all tasks, surpassing prior SOTA VGGT by an average of 35.7\% in camera pose accuracy and 23.6\% in geometric accuracy. Moreover, it outperforms DA2 in monocular depth estimation. All models are trained exclusively on public academic datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents Depth Anything 3 (DA3), which learns to predict dense depth maps and ray maps for estimating 3D geometry and camera poses. DA3 focuses in simplifying geometric understanding both from its model architecture, and also by using depth ray as representation for prediction. The architecture simply adopts a pre-trained DINOv2, which has been powerful in various 2D and 3D tasks, and rearranges the tokens for certain layers to compute the full attention for image tokens across different frames, allowing the information exchange between views. Furthermore, DA3 predicts camera rays instead of point maps, which consists of the camera origin and the direction for each of the pixels, and thoroughly demonstrates that camera rays serve as better representations compared to point clouds.

### Strengths
1. The paper shares an interesting finding that understanding 3D geometry can be done in a simplistic manner, especially without specialized architecture for incorporating multiple views or enforcing geometric constraints. Despite that it has been a recent trend for applying more and more generic architecture in 3D tasks, it is very intriguing to see that it could be done within a pre-trained DINOv2 by tweaking some of its attention layers.

2. The paper formulates a novel framework for predicting camera rays, which is the ray connecting the camera origin and pixels in the image plane. The results and the ablations thoroughly show that the combination of depth and camera ray suffices for effectively understanding 3D geometry, 

3. The proposed method establishes state-of-the-art performance across various geometry tasks, while having similar parameter counts. Furthermore, the method also shows strength in efficiency thanks to its simplistic architecture with having only few layers for cross-view understanding.

4. The paper proposes a benchmark for evaluating visual geometry, and introduces HiRoom dataset which the authors will release in the future.

### Weaknesses
1. Some of the claims are not well justified, or seems to be overclaiming in some points.
- L.157-158 (While point maps are insufficient to ensure consistency, redundant targets can improve pose accuracy but often introduce entanglement that compromises it.) What does the authors intend with "entanglement"? Does this mean that the results deduced from different heads could be problematic (e.g. pose from pose prediction head v. pose from point maps)? The authors should better elaborate the problems of "redundant" prediction to better establish their motivation for predicting camera rays.
- The camera head, despite being optional, is specified to have 0.48B parameters for the Giant variant. Despite the authors show that the computation is negligible as it only has few tokens, having a camera head that has nearly half of the parameter count of the backbone does not seem "lightweight".

2. It is a bit unclear on why depth+ray is more effective compared to point maps. To the reviewer's understanding, depth and ray actually seems like decoupling the point map prediction from Dust3r into two separate predictions. Why would this be better, asides from the empirical results?

3. Although the teacher model seems crucial, the ablations seems to be missing.

### Questions
1. Considering that all pixels from the same camera should have identical origins, it is interesting to see that all of the pixels are required to predict the origin in a dense manner, in addition to the ray direction. Is this simply a design choice, or does this have impact on training? It would also be interesting to see the variance of the predicted origin across each pixels within a single image, and study whether averaging all of the pixels for the pose estimation strategy is helpful.

2. The idea for the architecture seems to share some ideas with ViTDet[1], as it modifies existing pre-trained ViTs to alternate between local/global attentions. However, it is interesting to see that limiting the blocks for cross-view attention was more effective for DA3, as opposed to the results from ViTDet, apart from the obvious efficiency gains. Could the authors provide further analysis on why Full Alt. performs worse? Could it be from the gap between the pre-training and the downstream task, where drastically modifying all of the layers within DINOv2 to all handle cross-view attention be problematic?

[1] Li, Yanghao, et al. "Exploring plain vision transformer backbones for object detection." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

### Soundness
3

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
4

### Summary
The authors address the problem of 3D geometry estimation. They argue that depth and ray predictions constitute the minimal set of 3D predictions necessary for rendering 3D geometry, demonstrating this is the optimal choice. They extend existing transformer architectures by introducing a cross-view interaction transformer layer to handle multi-view inputs. Their method achieves significantly superior performance compared to existing models in both pose estimation and geometry estimation.

### Strengths
- The paper utilized depth and ray map representations to enable full 3D reconstruction from an arbitrary number of input images.
- Discovered an effective architecture design that outperforms previous methods while requiring minimal modifications to DINOv2.
- The paper demonstrates their model's effectiveness across various experimental settings.

### Weaknesses
- **Unclear advantage of depth+ray over point map:** To my knowledge, point maps can effectively represent various 3D information such as depth and pose, and a point map is essentially a combination of depth and ray maps. However, Table 5 shows that point maps hurt pose accuracy. What is the reason for this performance degradation? This finding appears to contradict the ablation study in VGGT, which argues that point map accuracy increases with multimodal outputs. I would like to see a more comprehensive analysis explaining why the combination of ray and depth maps outperforms point maps.
- **Missing ablation studies with point maps:** I am curious about additional experiments in Table 5 calculate the metrics using point map representations not using ray map and depth map. Specifically, what are the results when training with: (1) point maps only, and (2) point maps combined with ray maps and depth maps?
- **Pixel-wise ray map origin justification:** I understand that the origin of the ray map is identical for each image. Is there a specific reason to set the ray map origin in a pixel-wise manner? What is the benefit of this design choice?
- **Distinction between camera head and ray predictions:** What is the precise difference between the camera head and ray predictions? I am also curious about the performance gap between these two approaches. To my knowledge, camera parameters can generate ray maps, and conversely, ray maps can also be used to estimate camera parameters. Could you clarify the relationship and trade-offs between these representations?

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Depth Anything 3. Different from Depth Anything and Depth Anything v2 that can only work with single image, Depth Anything 3 is able to process any number of images. Different from previous methods such as DUSt3R and VGGT, Depth Anything 3 simplify the architecture, making it more scalable to numerous images. In addition, a teacher-student paradigm is used to provide high-quality data. Depth Anything 3 achieves state-of-the-art performance in various tasks, including pose estimation, 3D reconstruction, and rendering.

### Strengths
1. The architecture of Depth Anything 3 is simpler than previous methods. Depth Anything 3 uses a single vision transformer, while previous methods typically use vision transformer and following self- & cross-attention. Input-adaptive self-attention is used in vision transformer to enable cross-view attention without introducing new attention layers. With a simpler structure, Depth Anything 3 is able to process more images, which is meaningful for the future research. 

2. Extensive and thorough evaluation. Performance of pose estimation, 3D reconstruction, and rendering are thoroughly evaluated, where Depth Anything 3 achieves state-of-the-art performance. 

3. Ablation study of depth-ray representation shows that it explicitly outperforms previous representations, e.g. depth+pcd+cam used by VGGT.

### Weaknesses
1. In Table 1 and Table 2, I recommend adding some state-of-the-art methods that are not feed-forward models. This can help the readers have a better understanding of the performance difference between different methods. For example, classical pipelines generally outperform feed-forward models in 3D reconstruction.

2. If the teacher is not used, would the performance degrade explicitly? Currently, I am not sure if the mainly improvement is from the powerful teacher.

### Questions
1. L142: the equation looks wrong. $P$ denotes the 3D point in world coordinate frame, $D_i(u,v) K_i^{-1} p$ denotes the 3D point in camera local frame. To make the equation correct, $R_i, t_i$ should represent camera pose (transformation from camera to world), instead of extrinsics (world to camera). 

2. Sec. 2.4: Is GS-DPT head the only optimizable module, i.e. backbone is fixed?

3. L967: On ETH3D, is the tolerance 0.25 meter? Could the authors provide individual performance on each scene since the scale of scene vary a lot?

4. Typo:
    * L823: a “identity” camera -> an “identity” camera

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents Depth Anything 3, a single model that unifies geometric understanding across any number of views. The method jointly predicts a depth map and a ray map, using a ViT backbone with input adaptive cross view attention. A Dual DPT head shares reassembly modules and branches only at the final fusion stage to jointly infer depth and rays. Experiments across diverse benchmarks show consistent state of the art results in pose estimation, geometric reconstruction, and feed forward novel view synthesis, demonstrating strong accuracy and efficiency.

### Strengths
1. This paper presents a thoughtful analysis of what modalities are truly necessary for strong vision understanding tasks. It argues that depth together with a ray map is a minimal and sufficient target set. The ablation in Table 5 convincingly supports this claim by outperforming alternatives. Although recent work MapAnything also discusses incorporating ray maps into a unified representation, it is a contemporaneous work and does not needed to be considered here. 

2. The Dual DPT head is well designed. By sharing reassembly modules and branching only at the final fusion stage, the approach enforces pixel level alignment while avoiding redundant representations, which benefits both accuracy and efficiency. 

3. The experimental study is extensive and persuasive. The method is validated across pose estimation, geometric reconstruction, and feed forward novel view synthesis, consistently achieving SOTA results.

### Weaknesses
I did not find any major weaknesses. While I know recent advances in this area, I am not fully confident about all technical nuances and distinctions among closely related methods. I am open to perspectives from other reviewers and will continue to track the discussion.

### Questions
I wonder whether Depth Anything 3 is the most suitable title. The previous work is called Depth Anything v2, so if the intention is to follow the series it would be better to use Depth Anything v3 for consistency. Additionally, both v1 and v2 focus primarily on depth prediction. The current title can easily be read as another improvement targeted at depth prediction. It may be worth considering an alternative title that more clearly conveys the contribution..

### Soundness
3

### Presentation
3

### Contribution
3
