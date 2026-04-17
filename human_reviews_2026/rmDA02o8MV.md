# A Scene is Worth a Thousand Features: Feed-Forward Camera Localization from a Collection of Image Features

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 2

## Abstract
Visually localizing an image, i.e., estimating its camera pose, requires building a scene representation that serves as a visual map. The representation we choose has direct consequences towards the practicability of our system. Even when starting from mapping images with known camera poses, state-of-the-art approaches still require hours of mapping time in the worst case, and several minutes in the best. This work raises the question whether we can achieve competitive accuracy much faster. We introduce FastForward, a method that creates a map representation and relocalizes a query image on-the-fly in a single feed-forward pass. At the core, we represent multiple mapping images as a collection of features anchored in 3D space. FastForward utilizes these mapping features to predict image-to-scene correspondences for the query image, enabling the estimation of its camera pose. We couple FastForward with image retrieval and achieve state-of-the-art accuracy when compared to other approaches with minimal map preparation time. Furthermore, FastForward demonstrates robust generalization to unseen domains, including challenging large-scale outdoor environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a method for visual mapping and localization. The development of this method is motivated by two observations: the recent success of vision models, such as DUSt3R, in generating 3D models and depth information from multiple images of a scene, and the comparatively long time current localization methods require to construct a scene representation that facilitates accurate, robust, and generalizable mapping.

The core of the method involves training a model to predict the 3D coordinates for pixels in a query image. During training, the model processes query features and sampled features derived from a subset of posed reference images. Each reference mapping feature is augmented with a ray embedding, which encodes the corresponding camera’s position and viewing direction. These augmented reference features are then matched with the query features using an attention mechanism to regress the 3D coordinates of the 2D query pixels. For inference, a small set of reference images is first retrieved from a database (using either an image retrieval process or random sampling). These retrieved images are processed alongside the query image to regress the 3D coordinates of query pixels. The resulting 2D-3D correspondences are subsequently utilized to estimate the query image's absolute pose via a PnP RANSAC procedure.

The proposed method is trained on a collection of indoor and outdoor datasets. It is evaluated on a separate collection of small- to medium-scale indoor and outdoor localization benchmarks, specifically the Cambridge Landmarks, Wayspots, and Rio10 datasets. The method is compared against existing approaches categorized by how they represent the target scene (the mapping process): (a) methods which use Structure-from-Motion (SfM) to explicitly represent the scene as a 3D model (b) Scene Coordinate Regression (SCR) methods, which encode the scene geometry implicitly within the model weights (require per-scene training) and (c) Relative Pose Regression (RPR) methods, which represent the scene via image encodings and their camera poses followed by direct regression of the relative pose between one or more fetched nearest neighbors (minimal time for mapping).

### Strengths
1. The method presents a novel approach to visual mapping and localization, which enables fast mapping time and the demonstrated ability to generalize to scenes not seen during training.
2. The proposed method outperforms recent state-of-the-art RPR methods (minimal mapping time) and outperform SCR methods on several scenes from the Wayspots dataset
3. The paper provides a clear and well-articulated motivation for the proposed architecture, grounding the work in the limitations of existing methods and the capabilities of modern 3D vision models.
4. The paper is well structured and provides ablation study examining the impact of scale normalization and map size (number of reference images/features), providing insight into the method's core components and hyperparameter sensitivity.

### Weaknesses
1. The evaluation lacks a comprehensive discussion of the runtime and storage tradeoffs relative to other baseline methods. Specifically, the proposed approach relies on local feature matching and extraction, which introduces an overhead either at inference (runtime) or in storage (pre-computed features). This is a critical comparison point against RPR methods, which typically only require global feature matching, and against SCR methods. While SCR may incur a long initial training/mapping time, its inference time is generally very fast, making the accuracy/speed trade-off unclear. The paper also lacks the discussion of the one-off mapping effort versus the repeated query time to fully contextualize the value proposition of the proposed technique across all phases (training/mapping/querying).

2. The evaluation is confined to small- to medium-scale scenes. Consequently, the method's scalability to larger, city-scale environments is not demonstrated. As structure-based methods typically show strong performance and robustness in such large-scale settings this comparative assessment (or at least discussion) is necessary.

### Questions
1. Could the authors please provide a comparative table and discussion detailing the trade-offs across all baselines regarding runtime (mapping and querying), storage requirements, and localization accuracy?

2. What are the authors' specific assumptions and engineering considerations required for scaling this method to significantly larger, kilometer-scale environments?

### Soundness
3

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
5

### Summary
This paper addresses scene coordinate regression for camera relocalization without per-scene model training or reconstruction. To this end, the authors first retrieve a set of mapping images with known poses, then extract and randomly sample ViT features from the mapping images, and finally fuse the mapping features with query features to predict the scene-coordinate map of the query image. The camera pose of the query image is recovered via PnP between the predicted scene-coordinate map and the 2D image coordinates. To demonstrate efficiency, the authors conduct experiments on multiple datasets and achieve results comparable to model-based methods, while being substantially more efficient due to the reconstruction-free design.

### Strengths
* The paper is well-motivated; building and storing maps is expensive, especially for large-scale scenes. Therefore, this work provides an alternative solution that could address this limitation.
* The method maximizes reuse of existing public pretrained weights (DUSt3R), which makes reproduction and extension easier.
* The additional reconstruction head on mapping images is reasonable and offers additional supervision during training.
* The experiments are extensive and cover various scene scales and conditions.

### Weaknesses
* The contribution of this paper is limited:

  * First, in terms of problem formulation, the paper still works on scene coordinate regression. Compared with previous scene-agnostic works, the main change is that only the poses are given for the mapping images, while the scene structure is reconstructed implicitly during inference.
  * Compared with DUSt3R, the paper extends the encoder to handle multiple images with known poses by adding additional ray encodings to the tokens.
  * The scene normalization is hard to claim as a contribution: aligning camera poses to an anchor keyframe is common practice in 3D vision; scale normalization is also fairly trivial since SfM reconstructions are scale-normalized by default, so scenes are effectively normalized—especially for model-based camera relocalization.

* The motivation is to make camera localization more efficient; however, the method is still based on scene coordinate regression + PnP. As a result, per-query PnP still dominates computation. In contrast, 3D model-based methods only need to reconstruct the scene once, and the per-query cost of retrieval and PnP is low. Therefore, it is unclear whether the proposed solution is truly more efficient in terms of marginal cost, especially considering the training time.

* The solution still relies on image retrieval, but the paper does not discuss the limitations and failure cases of retrieval—an area where scene-specific localization models can be advantageous. The authors do investigate uniformly sampling the mapping images; however, this alternative provides little new information since it is generally inferior and arguably invalid for the stated setting.

### Questions
- First, please refer to the weaknesses.
- Second, how the proposed solution is robust to inaccurate camera pose of the mapping images?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This article presents a novel visual localization method called FeedForward, which overcomes some of the difficulties associated with traditional mapping approaches. Unlike conventional methods that build complete 3D models or train scene-specific networks, FeedForward uses a limited set of image features embedded in 3D space to estimate camera positions in a single feed-forward pass. The core idea is to represent multiple mapping images as a collection of features embedded in 3D space and use these mapping features to predict image-to-scene correspondences for the query image. It is show with various experiments that the method offers significant advantages in terms of efficiency, scalability and strong generalization capabilities.

### Strengths
Although the architectural design is inspired by DUSt3R, the idea of ​​taking a random sample of features from  multiple posed mapping images instead of considering a reference image as input to the encoder-decoder frame to predict accurate 3D coordinates of the query directly into the map's coordinate system is original and rather clever.

The authors demonstrate that a scene representation consisting of only a few hundreds mapping features obtained with their FeedForward method  is sufficient for accurate visual localization, where building such map is magnitude faster than using competitive methods (including traditional SfM, SCR methods or even Geometric Foundation models such as MAST3R) without  significantly drop in localization accuracy. 

The authors employ a simple yet effective technique for generalizing to new domains by defining one reference mapping image and transforming all other images such that the scene is centered at the origin of the coordinate system. The method computes the scene scale as the maximum camera translation in any spatial coordinate after normalization, which enables metric recovery from normalized predictions using the computed scale factor. This technique allows, on the one hand,  to exploit both metric and non-metric training data and on the other hand to enhances the network's ability to adapt to different scale ranges. 

The model was tested both on classical localization datasets such as Cambridge Landmark, Indoor6, RIO10, 7-Scenes and the more challenging extreme localization dataset Wayspot. 

The article is well-written, well-structured, and easy to follow. I also appreciate that the author's added a section on the model's limitations.

### Weaknesses
The model was tested on various standard datasets such as Cambridge Landmark, Indoor6, RIO10, and 7-Scenes, which are relatively small-scale compared to the scene size. Although these tests provide valuable insights into the model's performance in a variety of settings, it remains unclear how the model would fare on larger-scale datasets like Aachen Day-Night and InLoc compared to e.g MASt3R (Section 4.4 of the corresponding paper) that present more complex and challenging appearance variations.  Evaluating the model's robustness and adaptability to such scenarios would be crucial to better understanding of the model's strengths and limitations in real-world deployment.


At inference time, the model estimates camera poses using PnP-RANSAC with 2D-3D correspondences, which involves an extra optimization step. Given that the query head generates 3D coordinates for each 2D point, it would be interesting to evaluate the accuracy of the pointmap (either evaluating the corresponding depth-map or the  3D reconstruction) directly, in comparison with other Geometric Foundation Model (GFM) methods, the latter ones also evaluated on a per-image basis. To compare to models relying on a pair of images, FeedForward could consider a single image as map, but a denser feature sets, while models predicting pointmaps jointly for multiple views (e.g VGGT) a leave-one-out approach could be employed to construct the map for FeedForward.

### Questions
In the MAST3R/DUST3R paper, localization results are presented for the validation and test sets of the Map-free dataset. Although Figure 4 (supplementary) includes graphs illustrating accuracy as a function of the number of map images in the validation set, a fair comparison with other methods is difficult due to differences in thresholding compared to those used in Evaluation Leaderboard of the Map-free Visual Relocalization. To better assess the model's performance in this context, it is strongly recommended to also provide results obtained using the same evaluation protocol for single view and e.g. top 20.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper consider the problem of visual localization, i.e., the task of estimating the camera pose of a given query image wrt. some scene representation. State-of-the-art localization approaches are typically based on some form of 3D representation of the scene. Depending on the exact representation, building these 3D representations can take minutes to hours. This work thus investigates a scene representation that can be build in seconds instead: Given a query image, image retrieval is used to find the most similar database images representing the scene. The pose of the query image is then predicted wrt. the retrieved database images in a single forward pass. Experiments on standard benchmark datasets show that the proposed approach outperforms other methods that follow a similar paradigm (not building an expensive 3D scene representation) but can potentially perform significantly worse than methods based on a 3D scene representation (see results for Cambridge Landmarks).

### Strengths
The paper tackles an important practical problem and proposes a sound solution.

The method is evaluated on standard benchmark datasets and shows good performance compared to the baseline methods.

The paper is in general easy to read and it is easy to follow its argumentation.

### Weaknesses
My main concern are missing discussions of and comparisons with prior work:

1. Minor: [Kulhanek et al., ViewFormer: NeRF-free Neural Rendering from Few Images Using Transformers, ECCV 2022] proposed a transformer architecture that takes N images and their known poses as well as either an additional image or a camera pose as input and, in a single forward pass, either predicts the pose of the additional image or a rendered view from the additional camera pose. In the main paper, they focus on the latter case, i.e., novel view synthesis, while their supp. mat. presents localization results. While their method is not competitive with the proposed one, it seems to me that the paper still should be included in the related work section.

2. Relatively minor: [Torii et al., Are Large-Scale 3D Models Really Necessary for Accurate Visual Localization?, TPAMI 2021] (based on [Sattler et al., Are Large-Scale 3D Models Really Necessary for Accurate Visual Localization?, CVPR 2017]) asked a similar question to the one posed in the work under submission: Do we really need pre-computed 3D models for accurate visual localization? Their answer is that such models are not necessarily needed. Given the similarities in questions and answers, it seems to make sense to discuss this prior work.

3. Major: An alternative to estimating the pose of a query image by estimating relative poses wrt. each retrieved database image is to compute the pose of the query wrt. multiple database images (in form of semi-generalized relative pose [Zheng & Wu, Structure from motion using structure-less resection, ICCV 2015] or semi-generalized homography [Bhayani et al., Calibrated and Partially Calibrated Semi-Generalized Homographies, ICCV 2021] estimation). The most simplest to implement approach is to estimate the pose of the query image by first estimating the relative pose to one database image from 5 point correspondences (e.g., by estimating the essential matrix) and then using a single additional point correspondence with another database image to recover the scale of the translation [Zhang & Wu, ICCV 2015]. This E5+1 solver has been implemented for quite some time in the PoseLib library and it is quite simple to integrate into a localization pipeline (essentially: perform image retrieval to identify relevant database images, compute 2D-2D matches between the query and the top-retrieved images, estimate the pose by calling PoseLib's E5+1 solver wrapped in RANSAC (PoseLib has this readily implemented). While not as accurate as methods based on 3D representations, it works quite well, as shown in [Panek et al., A Guide to Structureless Visual Localization, arXiv:2504.17636] (the paper compares multiple existing approaches for localization based on representing the scene by a set of images rather than a 3D model, i.e., methods that offer fast representation-computation times). Here are results on multiple datasets obtained by this approach using RoMa feature matching after retrieving the top-20 most relevant database images using EigenPlaces descriptors for retrieval:

**Cambridge Landmarks**

Reporting median position errors [cm] / median orientation errors [deg] / accuracy at 10cm and 10deg / accuracy at 20cm and 20deg / accuracy at 25cm and 2deg:

| method | Great Court | King’s College | Old Hospital | Shop Facade | St Mary’s Church | 
| --- | --- | --- |  --- | --- | --- |
| FastForward | 62 / 4 / 5.1 / 15.5 / - | 24 / 4 / 15.5 / 42.9 / - | 26 / 0.5 / 17.6 / 41.8 / - | 8 / 4 / 64.1 / 92.2 / - | 14 / 0.5 / 28.3 / 75.7 / - |
| E5+1 | 29 / 0.1 / - / - / 43.2 | 19 / 0.3 / - / - / 60.6 | 24 / 0.5/ - / - / 51.6 | 6 / 0.3 / - / - / 95.1 | 10 / 0.3 / - / - / 86.8|

**Indoor 6**

Reporting median position errors [cm] / median orientation errors [deg] / accuracy at 5cm and 5deg / accuracy at 10cm and 10deg / accuracy at 20cm and 20deg

Results for E5+1 are reported as the average over the scenes.

|method | average | 
| --- | --- |
| FastForward | 4 / 0.7 / - / 91.1 / 98.1 |
| E5+1 |  1 / 0.15 / 94.9 / - / - |

As can be seen, this simple baseline performs similar to FastForward and does not require any specific learning. The main potential advantage that I can see for FastForward is that it might be faster at run-time due to requiring only a single forward pass. Then again, one can always extract only a limited number of features per image (e.g., I have good experience extracting 256 or 512 aliked features per image in the context of visual localization), which will make matching and pose estimation faster. As such, I believe that this would need to be shown through detailed experiments.

### Questions
What is the advantage of using FastForward over the simple E5+1 baseline? Please provide detailed experiments with the E5+1 baseline (using aliked features with the LightGlue matcher) by varying the number of extracted features and reporting the pose accuracy-run-time tradeoff and comparing it to FastForward.

### Soundness
2

### Presentation
3

### Contribution
3
