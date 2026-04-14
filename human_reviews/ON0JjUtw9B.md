# WIN: Variable-View Implicit LIDAR Upsampling Network

- Decision: Reject
- Scores: 3, 6, 5, 5

## Abstract
LiDAR upsampling aims to increase the resolution of sparse point sets obtained from low-cost sensors, providing better performance for various downstream tasks. Most existing methods transform LiDAR points into range view and design complex neighborhood point interpolation strategies to improve the resolution of point clouds. However, they overlook that the range image representation is insufficient to describe complex local geometric relationships, which limits the geometric accuracy of upsampled points.
To address this issue, we propose WIN, a Variable-View Implicit Network. 
First, we decouple the range image into two novel virtual view representations to compensate for the missing geometric information during range view-based interpolation. Secondly, to fuse the interpolation results of different views, we model the fusion process as a probability distribution problem instead of a simple binary classification task. We introduce a contrast selection module, which captures the feature differences between two representations and outputs the view confidence score for each upsampled point. The underlying idea is that the complementarity of the information is proportional to the feature difference between the two views. Motivated by this insight, we design a loss function based on probabilistic modeling to supervise the results of the selection module.
As a result, compared with the current state-of-the-art (SOTA) method ILN, WIN introduces a small number of parameters (+0.4M) but achieves a +4.5\%  increase in the MAE metric on the CARLA dataset. Furthermore, our method outperforms all existing methods in a downstream task (Depth Completion). The pre-trained model and code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
They propose a novel Variable-View Implicit LiDAR upsampling Network (WIN), which decouples the 3D representation of Range View (RV) into two novel virtual view representations Horizon Range View (HRV) and Vertical Range View (VRV). 

1. A reason of Variable-View upsampling is discussed. HRV and VRV, as an orthogonal transformation of RV, can provide more perspectives for observation without losing any geometric information. 

2. Implicit function methods are used to interpolate points in different views. It allows us to enjoy the advantages of variable-view representations without introducing unnecessary parameters or changing network architecture.

3. A contrast selection module is designed to help each interpolation point to choose the corresponding best view based on the geometric differences in the upsampled image from different representations.

4. Model the best view selection process as a probability distribution problem. 

5. Extensive experiments shows that, WIN achieved SOTA performance on both virtual and real-world datasets. An improvement of 4.53% and 7.01% is achieved on MAE and IoU, respectively.

### Strengths
Strengths:

1. Method is easy to follow and understand. 
2. Figure 1 provides a clear insight, which makes readers better understand the motivation.

### Weaknesses
Weaknesses:

1. The writting in Introduction is overpacking. In introduction section, it seems that the proposed method is complex with theoretically meaning. But, if we read the content in Method section, we find the proposed method is not complex. For example, the authors claim that they model the best view selection process as a probability distribution problem. In Sec. 3.4, Eqs. (6)-(8) is not impressive. 

2. Few state-of-the-art methods are compared in this manuscipt. And it is recommanded to add more comparision results. 

3. Is it really essential to decompose LiDAR point cloud with HRV and VRV? Range view is enough. In range view, the horizontal and vertical information can be used together, achieving more accurate results. 

4. Lacks of theoretical novelty. ICLR is a conference that focuses more on learning theory and related topics.

### Questions
Questions:

Is it really essential to decompose LiDAR point cloud with HRV and VRV? Range view is enough. In range view, the horizontal and vertical information can be used together, achieving more accurate results. 

Lacks of theoretical novelty. ICLR is a conference that focuses more on learning theory and related topics.

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
This work proposes a Variable-View Implicit lidar upsampling Network (WIN) to overcome the limitation of previous work only using a single perspective. In particular, this work first decouples the range view into two novel virtual view representations: Horizon Range View and Vertical Range View, to compensate for the missing geometric information. Then, a Contrast Selection Module is designed to guide the selection process by capturing the feature differences between the different representations. Experimental results demonstrate the effectiveness of the proposed WIN across different datasets and downstream vision tasks, such as depth completion.

### Strengths
+ The motivation for decoupling the 3D representation of a ranging image into two orthogonal view representations is clear and sound. It well addresses the limitation of previous works which only leverage a single perspective view.
+ The interpolation performance is promising and the proposed method can also facilitate some downstream tasks.

### Weaknesses
- While two orthogonal view representations can compensate for more information than the single perspective view in this paper, such a strategy (jointly considering the Horizon Range View and Vertical Range View) has been widely explored in previous 3D vision works. For example, "Joint 3D Proposal Generation and Object Detection from View Aggregation", "Multi-View 3D Object Detection Network for Autonomous Driving", "Deep Continuous Fusion for Multi-Sensor 3D Object Detection", etc. The authors should highlight their unique contributions and insights compared to these cross-view 3D works.
- In the Introduction, the contents of line-120 to line-127 and line-130 to line-138 are redundant with each other. The authors are suggested to integrate these two parts.
-  The proposed contrast selection module can help each interpolation point to choose the corresponding best view. Would this module be further improved using the feature fusion strategy across different views? The concrete reason for selection rather than fusion applied in this work needs to be clarified.
- Some key works about the LiDAR point cloud interpolation-related tasks are missing. For example, "Plin: A network for pseudo-lidar point cloud interpolation", "Pseudo-lidar point cloud interpolation based on 3d motion representation and spatial supervision", "PointINet: Point Cloud Frame Interpolation Network", "IDEA-Net: Dynamic 3D Point Cloud Interpolation via Deep Embedding Alignment", etc. The reviewer suggests that the authors include these references and discuss them briefly in the related work or introduction sections.
- The experimental results are insufficient. For example, this paper lacks important qualitative evaluations of the ablation study and depth completion comparison. 
- This paper is less polished and shows some poor presentations. For example, in line 210: r1, r2 is the upsampling factors; in line 279: use a MLP; in line 272: module(CSM), etc. Besides, the predicted view confidence g is missing in Figure 2.

### Questions
Can the proposed WIN help more downstream LiDAR-based works? Could you provide more discussions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes WIN, a variable-view implicit LiDAR upsampling network, providing a holistic understanding of the scene geometry. Different from previous work relying on single range view representation, this paper incorporates two decomposed views, Horizon Range View (HRV) and Vertical Range View (VRV), to achieve more accurate interpolation results. Furthermore, the authors employ a Contrast Selection Module (CSM) with the view confidence score to predict the best point from two orthogonal views. Based on this, WIN finally outperforms existing methods on both upsampling and downstream tasks.

### Strengths
- In contrast to previous approaches that rely only on a single range view, the authors of this paper provide a novel idea of decomposing it into two orthogonal views to better solve the LiDAR upsampling problem.

- The interpolation module is based on a local implicit function, which is more flexible to adapt to different views. And the overall model is lightweight and efficient.

### Weaknesses
- This paper actually converts the interpolation of depth on range view to the interpolation of 3D coordinates. However, the interpolation on the two orthogonal views is independent of each other and produces inconsistent results.  Although the paper uses another module to select one of the points, it may not be a particularly plausible way of fusion.

- Interpolation using only a simple MLP may be difficult to process in different regions. Especially when the encoder features are only from the range view, the model may not possess the perception ability in HRV and VRV viewpoints.

### Questions
Please refer to the Weaknesses, and there are additional concerns as follows: 

- As mentioned in L237, "RV is insufficient for describing non-smooth geometric surfaces", but the projection of HRV and VRV does not change its essence and still suffers from this problem.

- When choosing the four nearest neighbors of a query point, how to solve the interpolation problem for blank areas, occlusions, or object edges?

- The argument for predicting the best point through feature differences lacks a theoretical basis or a more in-depth explanation.

- In the selection module, the ground truth is modeled using a probabilistic approach. Is it possible to lead to a random distribution? What do the final distributions and proportions from the HRV and VRV look like? It's better to provide more visualization. Also, as opposed to this discrete selection, why not consider other fusion ways like weighted averaging or learnable fusion?

Other minor issues:

- L208: Actually, the number of points in the LiDAR point cloud is not equal to n = H × W points. 

- L216: typo for $R_{l}$.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a new method for LiDAR upsampling. Specifically, it first decouples the range view into two view representations: the Horizon Range View and the Vertical Range View. In each range view, the method predicts interpolation weights to interpolate the point cloud. Finally, a Contrast Selection Module is introduced to select the final result from the interpolation outputs of the two range views.

### Strengths
1. This paper is well-written, making it clear and easy to understand.  
2. Based on the experimental results provided (though in my opinion, the experiments are not thorough enough), the method outperforms all the baselines listed in the paper.

### Weaknesses
1. The paper mentions that decomposing the Range View into the Horizon Range View and Vertical Range View can reduce shape distortion (L510-513). However, for LiDAR upsampling, the most intuitive approach would be to retain the original 3D representation of the point cloud, then use a backbone like Point Transformer [1] to extract per-point features. Later, for each query point, nearby points could be identified, and interpolation weights could be predicted based on the relative spatial relationships between the query point and its neighbors in 3D space. This method avoids any errors introduced by projection, so why did the authors not adopt this approach?

2. In Section 2.1 on OBJECT-LEVEL POINT CLOUD UPSAMPLING, many recent methods [2-7] are missing. Additionally, [6] can be used for point cloud sampling on the KITTI dataset, so the paper should also compare its method to [6]. Fundamentally, I believe that these methods [2-7] could be applied to large-scale LiDAR point clouds. The paper claims these methods entail "huge computational burdens," but this statement lacks experimental evidence.

3. In Equation 5, the final prediction is selected between HRV and VRV, but HRV only contains the x and y components of the 3D coordinates, while VRV contains the z component, meaning each is incomplete. Wouldn’t a more reasonable approach be to predict weights for HRV and VRV (e.g., using features F_d  and F_z as inputs to an MLP to predict weights) and then apply a weighted sum to obtain the final prediction? Why did the authors not choose this approach?

4. The paper uses depth completion as a downstream task, first downsampling the KITTI point cloud data and then upsampling with different methods. However, the upsampled point clouds here match the original point cloud resolution (which still remains sparse), which I find quite unreasonable. I suggest that object detection on point clouds should be used as the downstream task instead, as object detection performance is more sensitive to point cloud resolution. Additionally, the authors should compare object detection performance using the original KITTI point cloud data and the upsampled point clouds obtained with different methods, which would more effectively demonstrate the value of LiDAR upsampling.

5. Current point cloud generation models based on diffusion models also have potential for upsampling tasks. For example, [8] can achieve text-to-LiDAR generation. If a low-resolution range image is used as input to [8] with the text prompt "upsample 4x," how would the results from [8] compare to the method proposed in this paper?

6. This paper lacks the comparison with the baseline [9].

[1] Wu X, Jiang L, Wang P S, et al. Point Transformer V3: Simpler Faster Stronger[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 4840-4851.

[2] Zhao W, Liu X, Zhong Z, et al. Self-supervised arbitrary-scale point clouds upsampling via implicit neural representation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1999-2007.

[3] Feng W, Li J, Cai H, et al. Neural points: Point cloud representation with neural fields for arbitrary upsampling[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 18633-18642.

[4] Qiu S, Anwar S, Barnes N. Pu-transformer: Point cloud upsampling transformer[C]//Proceedings of the Asian conference on computer vision. 2022: 2475-2493.

[5] He Y, Tang D, Zhang Y, et al. Grad-pu: Arbitrary-scale point cloud upsampling via gradient descent with learned distance functions[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 5354-5363.

[6] Qu W, Shao Y, Meng L, et al. A Conditional Denoising Diffusion Probabilistic Model for Point Cloud Upsampling[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 20786-20795.

[7] Rong Y, Zhou H, Xia K, et al. RepKPU: Point Cloud Upsampling with Kernel Point Representation and Deformation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 21050-21060.

[8] Ran H, Guizilini V, Wang Y. Towards Realistic Scene Generation with LiDAR Diffusion Models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 14738-14748.

[9] Helgesen S E M, Nakashima K, Tørresen J, et al. Fast LiDAR Upsampling using Conditional Diffusion Models[J]. arXiv preprint arXiv:2405.04889, 2024.

### Questions
1. The KITTI dataset lacks pairs of low-resolution and high-resolution point clouds for quantitatively evaluating point cloud upsampling methods. So how were the results in Table 1 generated?

2. In Equation 2, does the choice of the number of neighbors affect performance?

### Soundness
2

### Presentation
3

### Contribution
2
