# RayI2P: Learning Rays for Image-to-Point Cloud Registration

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Image-to-point cloud registration aims to estimate the 6-DoF camera pose of a query image relative to a 3D point cloud map. Existing methods fall into two categories: matching-free methods regress pose directly using geometric priors, but lack fine-grained supervision and struggle with precise alignment; matching-based methods construct dense 2D-3D correspondences for PnP-based pose estimation, but are fundamentally limited by projection ambiguity (where multiple geometrically distinct 3D points project to the same image patch, leading to ambiguous feature representations) and scale inconsistency (where fixed-size image patches correspond to 3D regions of varying physical size, causing misaligned receptive fields across modalities). To address these issues, we propose a novel ray-based registration framework that   first predicts patch-wise 3D ray bundles connecting image patches to the 3D scene and then estimates camera pose via a differentiable ray-guided regression module, bypassing the need for explicit 2D-3D correspondences. This formulation naturally resolves projection ambiguity, provides scale-consistent geometry encoding, and enables fine-grained supervision for accurate pose estimation. Experiments on KITTI and nuScenes show that our approach achieves state-of-the-art registration accuracy, outperforming existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of registration between a 3d point cloud and an image of the same scene. Compared to the problems of 3d registration between two
point clouds or image matching between two images, this problem is significantly more difficult. This difficulty comes from the ambiguities introduced by projecting 3d points 
to images and finding the correct matches in the images. Learning based approaches have been dominating the field. The main novelty in this paper is the use of the rays to 
parameterize the pose of the point cloud to the camera coordinate system. The rays are represented using Plücker coordinates, and this has been shown to be beneficial for 
camera pose estimation and for 6d object pose estimation. This parameterization seems to be easier to learn and brings other benefits, like more precise alignment and 
fewer projection ambiguities, and scale inconsistencies. The paper is well-written and not difficult to follow. The novelty seems to be limited, since all elements of the 
proposed method have been known before and just combined using the ray representation. The overall pipeline has two stages composed of the ray prediction module, which basically does 
3d point cloud and image feature fusion and is fairly standard and doesn't depend on the ray parameterization of the image patches. The second stage is a ray-guided regression module
that uses ray parameterization of the image patches, fused multi-modal features, and regresses the pose in terms of rotation and translation. The pipeline is supervised and applied to autonomous driving scenarios, where alignment between LIDAR scans and RGB images has been performed.

### Strengths
The paper is well-motivated and well-written. It was easy to follow it. Using ray parameterization for estimating the pose of the point cloud in the camera coordinate system
This makes sense, as it has been demonstrated for related problems, such as image pose estimation [1] and object pose estimation [2]. However, both approaches use this ray parameterization 
for underlying diffusion models. Here, this is not the case. What I find interesting is the use of reference rays in the ray-guided pose regression. However, this is not well motivated.

### Weaknesses
The proposed method has limited novelty. The first part of the feature fusion, known as the ray prediction module, is not new and doesn't benefit from the camera ray parameterization; however, it utilizes fused features and predicts image patch rays. The motivation behind the use of the reference rays and predicted rays is not given. The paper 
contains numerous ad-hoc design choices that are not clearly motivated or explained. See questions section for more details.
On the experimental side, I find the evaluation limited. It has been done only on two autonomous driving datasets. Why is this choice made for only this dataset and not some indoor datasets. Better motivation is needed for readers who are not interested in autonomous driving. I suppose the reason is that LIDAR and RGB images are not registered in the autonomous vehicles.

### Questions
Can you give more precise intuition behind using the reference rays? How do the guides and stabilize the regression process?

What will happen if you do not predict rays, but matches between 3d points and patches?


The results from Table 1 show that GraphI2P is actually performing very well and it is on par or better than the proposed method. What's the reason? What is the advantage of the proposed approach compared to GraphI2P?

In the ablation study, the results in Table 3 are a bit confusing. It is unclear what the fused features bring, as the rotation estimates in KITTI are more stable (smaller std) than in NuScenes. How do you explain this? What will happen if you have FPF and CPS only? RR seems to improve rotation estimation. Why?

### Soundness
2

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
2

### Summary
This paper proposes an image-to-point cloud registration method that estimates the 6-DoF camera pose of a query image relative to a 3D point cloud map. The proposed ray-based registration framework first predicts patch-wise 3D ray bundles connecting image patches to the 3D scene, then estimates the camera pose via a differentiable ray-guided regression module.

### Strengths
- The proposed ray-based registration framework for image-to-point cloud registration is interesting.
- The experimental results verified the effectiveness of the proposed method.

### Weaknesses
- Additional discussion comparing the proposed method with other ray-based representation methods [1] should be added.
- The approach appears to be a direct application of ray-based representation for pose estimation to the image-to-point cloud registration task. The specific challenge this addresses should be further clarified.

[1] Jason Y. Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. In *The Twelfth International Conference on Learning Representations*, 2024.

### Questions
What is the key challenge in image-to-point cloud registration compared to ray-based representation for pose estimation?

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
3

### Summary
This paper proposed a method for image-to-point cloud registration. Matching-based approaches is challenged by projection ambiguity and depth-induced scale inconsistency. To adress these problems, they introduce a differentiable ray-guided regression module which regress camera pose from predicted Plu ̈cker rays, thereby naturally resolving projection ambiguity and depth-induced scale ambiguity. The authors conduct experiments on KITTI and nuScenes. Method are evaluated by Relative Translation Error (RTE), average Relative Rotation Error (RRE), and registration accuracy (Acc). Compared to existing state-of-the-art approaches, the proposed method achieves strong accuracy, while remaining computationally efficient.

### Strengths
This paper is clearly motivated and well written. The paper intoduces a novel ray-guided pose regression module which addresses projection ambiguity and depth-induced scale inconsistency. Its technical rationale is supported by findings in the generalized camera models. The experimental setup is comprehensive, covering multiple datasets, metrics and baselines.

### Weaknesses
- The author did not analyze the causes of errors in the predicted rays. The neural regression module seems more like a compensatory measure; however, its use may raise concerns about generalization.
- Some of the illustrative figures are not very clear. In Figure 3, does the color of the points represent the depth error or the actual depth value?

### Questions
- Can the ray errors could be mitigated using diffusion-based methods?
- If the error threshold for Acc is set smaller, can the regression-based method achieve precision comparable to that of matching-based methods?

### Soundness
3

### Presentation
3

### Contribution
3
